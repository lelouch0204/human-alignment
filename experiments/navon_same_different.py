"""Score the official checkpoints on the original same–different stimuli."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

from human_alignment.models import VisionModel, load_images

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("family", choices=("dino", "siglip"))
parser.add_argument("variant", choices=("base", "unalignet", "alignet"))
args = parser.parse_args()

fingerprint = hashlib.sha256(args.source.read_bytes()).hexdigest()
assert fingerprint == "b74e9c74111655578f46ac9ededc549945a427fb65e1c8b3183b49a6dc9c704d"
data = loadmat(args.source, simplify_cells=True)["L2_str"]
labels = data["stim_details"]
assert set(map(tuple, labels)) == {(g, l) for g in range(1, 8) for l in range(1, 8)}

import tensorflow as tf
from PIL import Image

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
model = VisionModel(args.family, args.variant)
height, width = model.input_spec.shape.as_list()[1:3]
args.out.mkdir(parents=True, exist_ok=True)
paths = []
for i, pixels in enumerate(data["images"]):
    path = args.out / f"{i:02}.png"
    Image.fromarray(pixels).save(path)
    paths.append(path)

batches = [model.predict(load_images(paths[i:i+16], (width, height))) for i in range(0, 49, 16)]
results = {}
for rep in ("pre_logits", "triplet_logits"):
    x = np.concatenate([batch[rep] for batch in batches]).astype(float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    results[rep] = {}
    for axis, task in enumerate(("global", "local")):
        a, b = (data["imgpairs"][task] - 1).T
        changed = labels[a] != labels[b]
        distances = 1 - (x[a] * x[b]).sum(axis=1)
        human = 1 / data["RT"][task].mean(axis=(1, 2))
        # Restrict to different responses by design, with identical irrelevant cue.
        # Faster correct discrimination is expected at larger distance.
        single = changed[:, axis] & ~changed[:, 1-axis]
        dual = changed.all(axis=1)
        accuracy = data["PC"]["pairwise"][task].mean(axis=1)
        results[rep][task] = {}
        for key, mask in (("relevant_only", single), ("both", dual)):
            assert mask.sum() == 147
            results[rep][task][key] = dict(
                n=int(mask.sum()),
                mean_accuracy=float(accuracy[mask].mean()),
                rt_rho=float(spearmanr(human[mask], distances[mask]).statistic),
            )
(args.out / f"{args.family}-{args.variant}.json").write_text(json.dumps(results, indent=2))
