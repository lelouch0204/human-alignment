"""Exchange the class-token residual state within image batches of an official DINO export."""
import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
from tensorflow.python import pywrap_mlir

from human_alignment.models import VisionModel, load_images
from probe_navon_shapes import labels, root, run


def instrument(model_dir, target, layer):
    proto = SavedModel.FromString((model_dir / "saved_model.pb").read_bytes())
    meta = proto.meta_graphs[0]
    index = int(meta.signature_def["serving_default"].outputs[f"layer_{layer}"].name.split(":")[-1])
    count = 0
    for function in meta.graph_def.library.function:
        for node in function.node_def:
            if node.op != "XlaCallModule":
                continue
            raw = stablehlo.deserialize_portable_artifact_str(node.attr["module"].s)
            code = pywrap_mlir.ExperimentalRunPassPipeline(raw, b"", False)
            returns = re.findall(r"^    return ((?:%\d+, ){14}%\d+) :", code, re.M)
            assert len(returns) == 1
            branch = returns[0].split(", ")[index]
            pattern = rf"^    (%\d+) = stablehlo.add %\d+, {branch} : tensor<\?x1370x768xf32>$"
            (match,) = re.finditer(pattern, code, re.M)
            state = match[1]
            tensor = "tensor<?x1370x768xf32>"
            injected = f"""
    %pair_b = stablehlo.get_dimension_size {state}, dim = 0 : ({tensor}) -> tensor<i32>
    %pair_vec = stablehlo.reshape %pair_b : (tensor<i32>) -> tensor<1xi32>
    %pair_tail = stablehlo.constant dense<[1370, 768]> : tensor<2xi32>
    %pair_shape = stablehlo.concatenate %pair_vec, %pair_tail, dim = 0 : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
    %pair_first = stablehlo.constant dense<true> : tensor<1xi1>
    %pair_rest = stablehlo.constant dense<false> : tensor<1369xi1>
    %pair_indices = stablehlo.concatenate %pair_first, %pair_rest, dim = 0 : (tensor<1xi1>, tensor<1369xi1>) -> tensor<1370xi1>
    %pair_mask = stablehlo.dynamic_broadcast_in_dim %pair_indices, %pair_shape, dims = [1] : (tensor<1370xi1>, tensor<3xi32>) -> tensor<?x1370x768xi1>
    %pair_donor = stablehlo.reverse {state}, dims = [0] : {tensor}
    %pair_patched = stablehlo.select %pair_mask, %pair_donor, {state} : tensor<?x1370x768xi1>, {tensor}
"""
            suffix, rest = code[match.end():].split("\n  }", 1)
            suffix = re.sub(re.escape(state) + r"(?!\d)", "%pair_patched", suffix)
            changed = code[:match.end()] + injected + suffix + "\n  }" + rest
            node.attr["module"].s = stablehlo.serialize_portable_artifact_str(changed, "1.12.1")
            count += 1
    assert count == 2
    target.mkdir()
    (target / "saved_model.pb").write_bytes(proto.SerializeToString())
    (target / "variables").symlink_to((model_dir / "variables").resolve())


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("layer", type=int)
    args = parser.parse_args()
    variant, layer = args.variant, args.layer
    name = "DINOv2-B-" + {"base": "base_model", "unalignet": "untransformed"}[variant]
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    cache = np.load(root / f"embeddings/dino_{variant}.npz")["pre_logits"]
    lookup = {tuple(row): i for i, row in enumerate(labels)}
    results = []
    with tempfile.TemporaryDirectory() as folder:
        instrument(Path("models") / name, Path(folder) / name, layer)
        model = VisionModel("dino", variant, folder)
        height, width = model.input_spec.shape.as_list()[1:3]
        paths = sorted((root / "data/images").glob("*.png"))
        images = load_images(paths, (width, height))
        baseline = np.empty_like(cache)
        # Every outline and local identity occurs once in each position per offset.
        for offset in range(1, 7):
            recipients = np.arange(49)
            donors = np.array([lookup[((g-1+offset)%7+1,l)] for g,l in labels])
            for start in range(0,49,7):
                rs, ds = recipients[start:start+7], donors[start:start+7]
                if offset == 1:
                    same = model.embed(images[np.r_[rs,rs[::-1]]])[:7]
                    np.testing.assert_allclose((unit(same) * unit(cache[rs])).sum(axis=1), 1, atol=2e-6, rtol=0)
                    baseline[rs] = same
                patched = model.embed(images[np.r_[rs,ds[::-1]]])[:7]
                delta = unit(patched) - unit(baseline[rs])
                for i,(r,d) in enumerate(zip(rs,ds)):
                    g,l = labels[r]
                    dg = labels[d,0]
                    # Read out against different local elements. Matched control is
                    # an equally frequent third outline in the same reference images.
                    cg = (g-1-offset)%7+1
                    ref = lambda identity: unit(cache[(labels[:,0]==identity)&(labels[:,1]!=l)]).mean(axis=0)
                    results.append(dict(recipient=int(r), donor=int(d),
                                        full_donor_gain=float((unit(cache[d])-unit(cache[r])) @ (ref(dg)-ref(cg))),
                                        donor_gain=float(delta[i] @ (ref(dg)-ref(g))),
                                        control_gain=float(delta[i] @ (ref(cg)-ref(g)))))
    (run / f"patch-{variant}-{layer}.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
