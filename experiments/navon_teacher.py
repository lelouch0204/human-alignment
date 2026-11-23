"""Compare Navon pair rankings with the published SigLIP So400m/14 teacher."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from analyze_navon_shape_pairs import recover_labels


def encode(checkpoint, images):
    """Run the Big Vision image tower from the official Flax NPZ parameters."""
    import tensorflow as tf

    with np.load(checkpoint, allow_pickle=False) as archive:
        prefix = next(k.removesuffix("embedding/kernel") for k in archive
                      if k.endswith("embedding/kernel") and "/img/" in "/" + k)
        params = {k[len(prefix):]: tf.constant(archive[k], dtype=tf.float32)
                  for k in archive if k.startswith(prefix)}

    def norm(x, key):
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + 1e-6) * params[key + "/scale"] + params[key + "/bias"]

    def dense(x, key):
        return x @ params[key + "/kernel"] + params[key + "/bias"]

    def mlp(x, key):
        x = dense(x, key + "/Dense_0")
        return dense(tf.nn.gelu(x, approximate=True), key + "/Dense_1")

    def attention(x, source, key):
        def project(inp, name):
            root = f"{key}/{name}"
            return tf.einsum("btd,dhk->bthk", inp, params[root + "/kernel"]) + params[root + "/bias"]
        q = project(x, "query")
        k = project(source, "key")
        v = project(source, "value")
        logits = tf.einsum("bthk,bshk->bhts", q * (int(q.shape[-1]) ** -0.5), k)
        y = tf.einsum("bhts,bshk->bthk", tf.nn.softmax(logits, axis=-1), v)
        return tf.einsum("bthk,hkd->btd", y, params[key + "/out/kernel"]) + params[key + "/out/bias"]

    attn = "/MultiHeadDotProductAttention_0"
    mlp_path = "/MlpBlock_0"
    pool = "MAPHead_0" if "MAPHead_0/probe" in params else "MAPHead"
    stem = params["embedding/kernel"]
    pos = params["pos_embedding"]

    def forward(x):
        x = tf.nn.conv2d(x, stem, strides=14, padding="VALID") + params["embedding/bias"]
        x = tf.reshape(x, (-1, 729, 1152)) + pos
        for index in range(27):
            path = f"Transformer/encoderblock_{index}"
            z = norm(x, path + "/LayerNorm_0")
            x += attention(z, z, path + attn)
            x += mlp(norm(x, path + "/LayerNorm_1"), path + mlp_path)
        x = norm(x, "Transformer/encoder_norm")
        probe = tf.repeat(params[pool + "/probe"], tf.shape(x)[0], axis=0)
        x = attention(probe, x, pool + attn)
        x += mlp(norm(x, pool + "/LayerNorm_0"), pool + mlp_path)
        x = x[:, 0]
        if "head/kernel" in params:
            x = dense(x, "head")
        return x

    return np.concatenate([forward(batch).numpy() for batch in np.array_split(images, 13)])


def compare(embeddings, root, output):
    with np.load(root / "data/human.npz", allow_pickle=False) as save:
        pairs = save["pairs"]
        rt = save["subject_rt"]
        human = save["human_distance"]
        labels = recover_labels(pairs, save["pair_details"])
    assert embeddings.shape == (49, 1152) and np.isfinite(embeddings).all()
    embeddings = embeddings.astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    teacher = np.linalg.norm(embeddings[pairs[:, 0]] - embeddings[pairs[:, 1]], axis=1)
    left, right = labels[pairs[:, 0]], labels[pairs[:, 1]]
    outline, elements = (left != right).T
    masks = {"outline_only": outline & ~elements, "element_only": ~outline & elements,
             "both": outline & elements}
    assert masks["outline_only"].sum() == 147
    results = {}
    for representation in ("pre_logits", "triplet_logits"):
        distances = []
        for variant in ("base", "unalignet", "alignet"):
            path = root / f"run-7167305/dino_{variant}_{representation}-distances.npz"
            with np.load(path, allow_pickle=False) as save:
                np.testing.assert_array_equal(save["pairs"], pairs)
                distances.append(save["frozen"])

        def metrics(mask, target):
            rank = rankdata(np.column_stack([teacher[mask], target[mask], *(d[mask] for d in distances)]), axis=0)
            rho = np.corrcoef(rank, rowvar=False)
            teacher_change = np.abs(rank[:, 2] - rank[:, 0]) - np.abs(rank[:, 3] - rank[:, 0])
            human_change = np.abs(rank[:, 3] - rank[:, 1]) - np.abs(rank[:, 2] - rank[:, 1])
            harm = human_change > 0
            result = {
                "pairs": int(mask.sum()),
                "teacher_human_rho": float(rho[0, 1]),
                "checkpoint_rhos": {name: dict(teacher=float(rho[i, 0]), human=float(rho[i, 1]))
                                    for i, name in enumerate(("base", "unalignet", "alignet"), 2)},
                "delta_rho_to_teacher": float(rho[3, 0] - rho[2, 0]),
                "delta_rho_to_human": float(rho[3, 1] - rho[2, 1]),
                "pairs_closer_to_teacher": int((teacher_change > 0).sum()),
                "pairs_farther_from_human": int(harm.sum()),
                "of_those_pairs_closer_to_teacher": int((harm & (teacher_change > 0)).sum()),
                "rho_rank_error_changes": float(spearmanr(teacher_change, human_change).statistic),
            }
            return result

        sensitivity = {}
        for axis, name in enumerate(("outline", "element")):
            for identity in range(1, 8):
                keep = (left[:, axis] != identity) & (right[:, axis] != identity)
                sensitivity[f"omit_{name}_{identity}"] = metrics(masks["outline_only"] & keep, human)
        for person in range(rt.shape[1]):
            sensitivity[f"omit_participant_{person + 1}"] = metrics(
                masks["outline_only"], 1 / np.delete(rt, person, axis=1).mean(axis=1))
        results[representation] = dict(full={key: metrics(mask, human) for key, mask in masks.items()},
                                       outline_sensitivity=sensitivity)
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/navon_shapes/teacher-embeddings.npy"))
    parser.add_argument("--root", type=Path, default=Path("outputs/navon_shapes"))
    parser.add_argument("--output", type=Path, default=Path("outputs/navon_shapes/teacher-report.json"))
    args = parser.parse_args()
    if args.checkpoint:
        import tensorflow as tf
        from PIL import Image

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("Run teacher inference in a Slurm GPU allocation")
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)
        manifest = json.loads((args.root / "data/manifest.json").read_text())
        paths = [args.root / "data" / row["path"] for row in manifest["images"]]
        images = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths])
        images = tf.image.resize(images, [384, 384], method="bilinear", antialias=True) / 127.5 - 1
        np.save(args.embeddings, encode(args.checkpoint, images))
    compare(np.load(args.embeddings, allow_pickle=False), args.root, args.output)


if __name__ == "__main__":
    main()
