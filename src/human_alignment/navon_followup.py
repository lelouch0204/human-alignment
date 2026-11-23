"""CPU-only distance decomposition and grouped linear probes on published embeddings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

DATASET = "lelouch0204/navon-alignet"
REVISION = "d334c56a3f0023e25860e55114eb2e9ec5deb3e8"
MODELS = [f"{f}_{v}" for f in ("dino", "siglip") for v in ("base", "unalignet", "alignet")]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
EMBEDDING_SHA256 = {
    "dino_base": "5b5070fb5a51bb99cb81d185aa3405d3b83378f4b82e716fa1472a9f600aa72b",
    "dino_unalignet": "20657d226d916ac5e38f00917c48e5c5f478a2626824e61aaf161987b181e487",
    "dino_alignet": "e7ee507fbee3975a62366b367bb4fcbc3262b868f0455a88bdedb66d7caabf19",
    "siglip_base": "4aac09995e508d21d3ef4d52b4fd4d2deb3b3b2e7bba54a43a07c5a4343fad21",
    "siglip_unalignet": "b730888a054717b72a5dbbd08774a7cb47a9afb7bfb6187e3c80029224add00e",
    "siglip_alignet": "f529c385b8e87ae78593791182c3519e8c38c4905d4e5b8e501771e173920363",
}


def validate_source_files(data):
    for model, expected in EMBEDDING_SHA256.items():
        if hashlib.sha256((data / f"embeddings-{model}.parquet").read_bytes()).hexdigest() != expected:
            raise ValueError(f"Embedding checksum mismatch: {model}")
    raw = (data / "manifest.json").read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_blob != "3a9e57eabc8158a541625730b0b106081290e28b":
        raise ValueError("Manifest does not match the pinned dataset revision")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download(data_dir):
    """Download only metadata and embeddings from the pinned revision."""
    files = {"manifest.json": "artifacts/manifest.json"}
    files.update({f"embeddings-{m}.parquet": f"data/embeddings-{m}.parquet" for m in MODELS})
    for filename, remote in files.items():
        path = data_dir / filename
        if path.exists():
            continue
        print(f"Downloading {filename}", flush=True)
        temporary = path.with_suffix(path.suffix + ".part")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{remote}"
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(path)


def load_embeddings(path, items, representation):
    table = pq.read_table(path, columns=["image_id", "family", "variant", representation])
    model = Path(path).stem.removeprefix("embeddings-")
    family, variant = model.split("_", 1)
    if set(table["family"].to_pylist()) != {family} or set(table["variant"].to_pylist()) != {variant}:
        raise ValueError(f"Checkpoint labels do not match filename: {path}")
    ids = table["image_id"].to_pylist()
    expected = [item["image_id"] for item in items]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"Duplicate or mismatched image IDs: {path}")
    lookup = {image_id: i for i, image_id in enumerate(ids)}
    x = np.asarray(table[representation].to_pylist(), dtype=np.float64)
    x = x[[lookup[image_id] for image_id in expected]]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    expected_dim = {"pre_logits": 768, "triplet_logits": 1024}[representation]
    if x.shape != (len(items), expected_dim) or not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError(f"Invalid embeddings: {path}, {representation}")
    return x / norms


def substitution_indices(k):
    return np.asarray([(x, y, z) for x in range(k) for y in range(x + 1, k)
                       for z in range(k) if z not in (x, y)], dtype=int).T


def component_distances(cube, xyz):
    x, y, z = xyz
    global_distance = 1 - np.einsum("ij,ij->i", cube[x, z], cube[y, z])
    local_distance = 1 - np.einsum("ij,ij->i", cube[z, x], cube[z, y])
    return np.column_stack((global_distance, local_distance, local_distance - global_distance))


def letter_sensitivity(values, xyz, k):
    estimates = np.asarray([values[np.all(xyz != letter, axis=0)].mean(axis=0)
                            for letter in range(k)])
    se = np.sqrt((k - 1) / k * ((estimates - estimates.mean(axis=0)) ** 2).sum(axis=0))
    return estimates, se


def distance_analysis(unit, items, letters, conditions, positions, model, representation):
    k = len(letters)
    xyz = substitution_indices(k)
    idx = {(v["condition"], v["pos_idx"], v["global_letter"], v["local_letter"]): i
           for i, v in enumerate(items) if v["kind"] == "compound"}
    summaries, pairs, all_values = [], [], []
    for condition in conditions:
        for position in positions:
            indices = [[idx[condition, position, g, l] for l in letters] for g in letters]
            cube = unit[np.asarray(indices)]
            values = component_distances(cube, xyz)
            all_values.append(values)
            flat = cube[~np.eye(k, dtype=bool)]
            # Mean off-diagonal cosine distance, without constructing a dense RSM.
            n = len(flat)
            spread = 1 - ((flat.sum(axis=0) ** 2).sum() - n) / (n * (n - 1))
            means = values.mean(axis=0)
            summaries.append(dict(model=model, representation=representation, condition=condition,
                                  position=position, global_distance=means[0], local_distance=means[1],
                                  local_minus_global=means[2], mean_pair_distance=spread))
            for x in range(k):
                for y in range(x + 1, k):
                    mean = values[(xyz[0] == x) & (xyz[1] == y)].mean(axis=0)
                    pairs.append(dict(model=model, representation=representation, condition=condition,
                                      position=position, x=letters[x], y=letters[y],
                                      global_distance=mean[0], local_distance=mean[1],
                                      local_minus_global=mean[2]))
    values = np.concatenate(all_values)
    roles = np.tile(xyz, (1, len(all_values)))
    means = values.mean(axis=0)
    summaries.append(dict(model=model, representation=representation, condition="ALL", position="ALL",
                          global_distance=means[0], local_distance=means[1], local_minus_global=means[2],
                          mean_pair_distance=np.mean([r["mean_pair_distance"] for r in summaries])))
    return summaries, pairs, values, roles


def outer_folds(groups, seed):
    labels = np.unique(groups).copy()
    np.random.default_rng(seed).shuffle(labels)
    for fold, held in enumerate(np.array_split(labels, 5)):
        test = np.flatnonzero(np.isin(groups, held))
        train = np.flatnonzero(~np.isin(groups, held))
        yield fold, train, test


def probe(unit, metadata, target, seed, alphas=ALPHAS):
    y = np.asarray([r[f"{target}_letter"] for r in metadata])
    nuisance = "local" if target == "global" else "global"
    groups = np.asarray([r[f"{nuisance}_letter"] for r in metadata])
    predictions = np.full(len(y), "", dtype="<U1")
    fold_ids = np.full(len(y), -1, dtype=int)
    selected = np.zeros(len(y))
    folds = []
    for fold, train, test in outer_folds(groups, seed):
        assert not set(groups[train]) & set(groups[test])
        assert set(y[train]) == set(y)
        scores = np.zeros(len(alphas))
        denominator = 0
        for inner_train, inner_valid in GroupKFold(n_splits=3).split(unit[train], y[train], groups[train]):
            fit_idx, valid_idx = train[inner_train], train[inner_valid]
            assert not set(groups[fit_idx]) & set(groups[valid_idx])
            assert set(y[fit_idx]) == set(y)
            for a_idx, alpha in enumerate(alphas):
                estimator = RidgeClassifier(alpha=alpha, solver="cholesky")
                estimator.fit(unit[fit_idx], y[fit_idx])
                scores[a_idx] += (estimator.predict(unit[valid_idx]) == y[valid_idx]).sum()
            denominator += len(valid_idx)
        best = max(range(len(alphas)), key=lambda i: (scores[i], alphas[i]))
        estimator = RidgeClassifier(alpha=alphas[best], solver="cholesky")
        estimator.fit(unit[train], y[train])
        predictions[test] = estimator.predict(unit[test])
        fold_ids[test] = fold
        selected[test] = alphas[best]
        folds.append(dict(target=target, fold=fold, held_out_letters="".join(sorted(set(groups[test]))),
                          n_train=len(train), n_test=len(test), alpha=alphas[best],
                          inner_accuracy=scores[best] / denominator,
                          inner_scores=json.dumps({str(a): float(s / denominator) for a, s in zip(alphas, scores)}),
                          accuracy=accuracy_score(y[test], predictions[test]),
                          balanced_accuracy=balanced_accuracy_score(y[test], predictions[test])))
    assert np.all(fold_ids >= 0) and np.all(predictions != "")
    rows = [dict(image_id=r["image_id"], target=target, truth=y[i], nuisance=groups[i],
                 condition=r["condition"], position=r["pos_idx"], fold=int(fold_ids[i]),
                 alpha=selected[i], prediction=predictions[i], correct=int(y[i] == predictions[i]))
            for i, r in enumerate(metadata)]
    return rows, folds


def make_figures(out, distances, scores, representation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    variants = ("base", "unalignet", "alignet")
    for col, family in enumerate(("dino", "siglip")):
        for offset, field, label in ((-0.18, "global_distance", "Global change"),
                                     (0.18, "local_distance", "Local change")):
            values = [next(r[field] for r in distances if r["model"] == f"{family}_{v}"
                           and r["representation"] == representation and r["condition"] == "ALL")
                      for v in variants]
            axes[0, col].bar(np.arange(3) + offset, values, width=0.36, label=label)
        for target, marker in (("global", "o"), ("local", "s")):
            values = [next(r["accuracy"] for r in scores if r["model"] == f"{family}_{v}"
                           and r["representation"] == representation and r["target"] == target
                           and r["slice"] == "ALL") for v in variants]
            axes[1, col].plot(np.arange(3), np.asarray(values) * 100, marker=marker, label=target.title())
        axes[0, col].set_title(family.upper())
        axes[0, col].set_ylabel("Mean cosine distance")
        axes[1, col].set_ylabel("Held-out nuisance identity accuracy (%)")
        axes[1, col].axhline(100 / 26, color="gray", linestyle="--", label="Uniform random")
        axes[1, col].set_ylim(0, 103)
        for ax in axes[:, col]:
            ax.set_xticks(range(3), ["Base", "UnAligNet", "AligNet"])
            ax.legend(fontsize=8)
    fig.suptitle(f"Navon distance decomposition and grouped ridge probes: {representation}")
    fig.savefig(out / f"followup_{representation}.png", dpi=180)
    plt.close(fig)


def write_report(out, distance_rows, score_rows, contrast_rows, probe_contrasts, representations, alphas):
    lines = ["# Navon follow-up results", "", f"Dataset revision: `{REVISION}`.", "",
             "Exploratory analysis of published frozen embeddings; no model inference.", "",
             f"Nested-validation alpha grid: `{list(alphas)}`.", "",
             "All distances below average matched substitutions over three conditions and two positions.",
             "Positive local-minus-global distance means local changes move embeddings farther.", ""]
    for rep in representations:
        lines += [f"## {rep}", "", "| Model | Global-change distance | Local-change distance | Local − global | Global decoding | Local decoding |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for model in MODELS:
            d = next(r for r in distance_rows if r["model"] == model and r["representation"] == rep and r["condition"] == "ALL")
            acc = {r["target"]: r["accuracy"] for r in score_rows if r["model"] == model and r["representation"] == rep and r["slice"] == "ALL"}
            lines.append(f"| {model} | {d['global_distance']:.4f} | {d['local_distance']:.4f} | {d['local_minus_global']:.4f} | {acc['global']:.2%} | {acc['local']:.2%} |")
        lines += ["", "### Paired distance differences", "",
                  "Letter jackknife SE and leave-one-letter-out ranges describe alphabet sensitivity; no independent-trial CI is claimed.", "",
                  "| Contrast | Component | Difference | Letter jackknife SE | Leave-one-letter-out range |",
                  "| --- | --- | ---: | ---: | --- |"]
        for r in contrast_rows:
            if r["representation"] == rep:
                lines.append(f"| {r['family']} {r['contrast']} | {r['component']} | {r['difference']:+.4f} | {r['jackknife_se']:.4f} | [{r['leave_letter_min']:+.4f}, {r['leave_letter_max']:+.4f}] |")
        lines += ["", "### Paired decoding differences", "",
                  "| Contrast | Target | Accuracy difference (pp) | A correct / B wrong | A wrong / B correct |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for r in probe_contrasts:
            if r["representation"] == rep:
                lines.append(f"| {r['family']} {r['contrast']} | {r['target']} | {100*r['delta_accuracy']:+.2f} | {r['a_only_correct']} | {r['b_only_correct']} |")
        lines += ["", f"![Distance and decoding plot](followup_{rep}.png)", ""]
    lines += ["## Scope and interpretation", "",
              "Probes use 3,900 incongruent images, five outer nuisance-letter folds, and three inner grouped folds for alpha selection. All 26 target classes remain in training. Unit-normalized embeddings and intercept-bearing ridge classifiers are used. Uniform random accuracy is 3.85%. Balanced accuracies and rendering slices are saved separately.", "",
              "This tests linear accessibility across nuisance identities, not unseen fonts, novel target letters, new sizes, or human agreement. Images and folds are dependent. Probe comparisons are descriptive; one training seed/checkpoint per variant is available. The source DINO preprocessing remains unverified; Base triplet heads are reported untrained. The distance analysis decomposes the original balanced comparison and is not independent replication.", "",
              "Inputs, hashes, dependencies, arguments, and checks: `run_record.json`. Per-image predictions: `probe_predictions.csv`. Per-fold tuning: `probe_folds.csv`. Pair distances: `distance_pairs.csv`. Letter omissions: `distance_leave_letter_out.csv`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/navon_followup"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-dir", type=Path, help="Reuse downloaded inputs from another run")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--representations", nargs="+", choices=("pre_logits", "triplet_logits"), default=["pre_logits", "triplet_logits"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(a <= 0 or not np.isfinite(a) for a in args.alphas):
        parser.error("alphas must be finite and positive")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    data = args.data_dir or out / "data"
    data.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(data)
    validate_source_files(data)
    manifest = json.loads((data / "manifest.json").read_text())
    items = manifest["items"]
    letters = manifest["config"]["letters"]
    if len({r["image_id"] for r in items}) != len(items) or len(letters) != 26:
        raise ValueError("Expected unique image IDs and the full 26-letter alphabet")
    selected = [i for i, r in enumerate(items) if r["kind"] == "compound" and r["global_letter"] != r["local_letter"]]
    metadata = [items[i] for i in selected]
    conditions = list(manifest["config"]["conditions"])
    positions = list(range(manifest["config"]["num_positions"]))
    if len(selected) != 3900:
        raise ValueError("Expected 3900 incongruent compounds")
    distances, pairs, predictions, folds, scores = [], [], [], [], []
    contrasts, omissions, probe_contrasts = [], [], []
    for rep in args.representations:
        bundles, pred_bundles = {}, {}
        for model in MODELS:
            print(f"Analyzing {model} {rep}", flush=True)
            unit = load_embeddings(data / f"embeddings-{model}.parquet", items, rep)
            d, p, values, xyz = distance_analysis(unit, items, letters, conditions, positions, model, rep)
            distances.extend(d)
            pairs.extend(p)
            bundles[model] = values
            for target in ("global", "local"):
                with threadpool_limits(limits=2):
                    pred, fold_rows = probe(unit[selected], metadata, target, args.seed, args.alphas)
                for row in pred + fold_rows:
                    row.update(model=model, representation=rep)
                predictions.extend(pred)
                folds.extend(fold_rows)
                pred_bundles[model, target] = pred
                slices = {"ALL": pred}
                slices.update({f"condition={c}": [r for r in pred if r["condition"] == c] for c in conditions})
                slices.update({f"position={p}": [r for r in pred if r["position"] == p] for p in positions})
                for name, rows in slices.items():
                    truth, guessed = [r["truth"] for r in rows], [r["prediction"] for r in rows]
                    scores.append(dict(model=model, representation=rep, target=target, slice=name, n=len(rows),
                                       accuracy=accuracy_score(truth, guessed), balanced_accuracy=balanced_accuracy_score(truth, guessed)))
                print(f"  {target}: {np.mean([r['correct'] for r in pred]):.3%}", flush=True)
            # Preserve completed results if a later model fails.
            write_csv(out / "probe_predictions.csv", predictions)
            write_csv(out / "probe_folds.csv", folds)
            write_csv(out / "probe_scores.csv", scores)
        for family in ("dino", "siglip"):
            for baseline in ("unalignet", "base"):
                contrast = f"alignet_minus_{baseline}"
                delta = bundles[f"{family}_alignet"] - bundles[f"{family}_{baseline}"]
                leave, se = letter_sensitivity(delta, xyz, len(letters))
                for j, component in enumerate(("global_distance", "local_distance", "local_minus_global")):
                    contrasts.append(dict(family=family, representation=rep, contrast=contrast, component=component,
                                          difference=delta[:, j].mean(), jackknife_se=se[j],
                                          leave_letter_min=leave[:, j].min(), leave_letter_max=leave[:, j].max()))
                    for i, letter in enumerate(letters):
                        omissions.append(dict(family=family, representation=rep, contrast=contrast, component=component,
                                              omitted_letter=letter, difference=leave[i, j]))
                for target in ("global", "local"):
                    a, b = pred_bundles[f"{family}_alignet", target], pred_bundles[f"{family}_{baseline}", target]
                    assert [r["image_id"] for r in a] == [r["image_id"] for r in b]
                    ac, bc = np.array([r["correct"] for r in a]), np.array([r["correct"] for r in b])
                    probe_contrasts.append(dict(family=family, representation=rep, contrast=contrast, target=target,
                                                delta_accuracy=float((ac - bc).mean()),
                                                a_only_correct=int(((ac == 1) & (bc == 0)).sum()),
                                                b_only_correct=int(((ac == 0) & (bc == 1)).sum())))
    for name, rows in (("distance_summary", distances), ("distance_pairs", pairs),
                       ("distance_contrasts", contrasts), ("distance_leave_letter_out", omissions),
                       ("probe_contrasts", probe_contrasts)):
        write_csv(out / f"{name}.csv", rows)
    for rep in args.representations:
        make_figures(out, distances, scores, rep)
    write_report(out, distances, scores, contrasts, probe_contrasts, args.representations, args.alphas)
    (out / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    record = dict(dataset=DATASET, revision=REVISION, seed=args.seed, alphas=args.alphas,
                  data_dir=str(data), out_dir=str(out),
                  representations=args.representations, python=platform.python_version(),
                  packages={p: importlib.metadata.version(p) for p in ("numpy", "pyarrow", "scikit-learn", "scipy", "matplotlib")},
                  input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir()) if p.is_file()},
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  lock_sha256=hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
                  checks=["Input checksums match the pinned dataset revision",
                          "Unique manifest and embedding IDs, exact sets matched and reordered",
                          "Finite, nonzero embeddings with expected dimensions",
                          "3900 incongruent compounds; each receives one outer prediction per target",
                          "No nuisance identity overlap in outer or inner splits",
                          "All target classes present in every training fold",
                          "Identical image ordering for paired checkpoint contrasts"],
                  inference_run=False)
    (out / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Report: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
