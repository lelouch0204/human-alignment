"""Probe exported DINO layer readouts on the same seven held-out contexts."""
import json
import sys
from pathlib import Path

import numpy as np
from human_alignment.models import VisionModel, load_images
from probe_navon_shapes import evaluate, labels, root, run

import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

variant = sys.argv[1]
model = VisionModel("dino", variant)
manifest = json.loads((root / "data/manifest.json").read_text())
paths = [root / "data" / row["path"] for row in manifest["images"]]
height, width = model.input_spec.shape.as_list()[1:3]
layers = [f"layer_{i}" for i in range(12)] + ["pre_logits"]
features = {name: [] for name in layers}
for start in range(0, len(paths), 16):
    outputs = model.predict(load_images(paths[start:start+16], (width, height)))
    for name in layers:
        value = outputs[name]
        print(name, value.shape, flush=True) if start == 0 else None
        features[name].append(value[:, 0].copy() if value.ndim == 3 else value)
cache = np.load(root / f"embeddings/dino_{variant}.npz")
results = {}
for name, batches in features.items():
    x = np.concatenate(batches).astype(float)
    if name == "pre_logits":
        np.testing.assert_allclose(x, cache["pre_logits"], atol=1e-4, rtol=1e-4)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    results[name] = {task: evaluate(x, target, context) for task, target, context in
                     (("global", *labels.T), ("local", *labels.T[::-1]))}
(run / f"dino-{variant}-layer-probes.json").write_text(json.dumps(results))
