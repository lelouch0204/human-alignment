"""Nested leave-one-context-out ridge probes on cached Navon embeddings."""
import json
from pathlib import Path

import numpy as np
from analyze_navon_shape_pairs import recover_labels

root = Path("outputs/navon_shapes")
run = root / "run-7167305"
human = np.load(root / "data/human.npz")
labels = recover_labels(human["pairs"], human["pair_details"])
alphas = 10.0 ** np.arange(-6, 5)


def predict(x, target, train, test):
    center = x[train].mean(axis=0)
    features = x[train] - center
    y = np.eye(7)[target[train] - 1]
    values, vectors = np.linalg.eigh(features @ features.T)
    left = (x[test] - center) @ features.T @ vectors
    right = vectors.T @ (y - y.mean(axis=0))
    scores = (left[None] / (values + alphas[:, None])[:, None, :]) @ right
    return (scores + y.mean(axis=0)).argmax(axis=2) + 1


def evaluate(x, target, context):
    folds = []
    for held in np.unique(context):
        train = context != held
        validation = []
        for inner in np.unique(context[train]):
            mask = context == inner
            predictions = predict(x, target, train & ~mask, mask)
            validation.append((predictions == target[mask]).mean(axis=1))
        correct = np.mean(validation, axis=0)
        best = np.flatnonzero(correct == correct.max())[-1]
        predictions = predict(x, target, train, ~train)
        folds.append(dict(context=int(held), alpha=float(alphas[best]),
                          correct=int((predictions[best] == target[~train]).sum()),
                          fixed_alpha_correct=(predictions == target[~train]).sum(axis=1).tolist()))
    return folds


if __name__ == "__main__":
    results = {}
    for path in sorted((root / "embeddings").glob("*.npz")):
        cache = np.load(path)
        np.testing.assert_array_equal(cache["image_ids"], np.arange(49))
        for rep in ("pre_logits", "triplet_logits"):
            x = cache[rep].astype(float)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            for task, target, context in (("global", *labels.T), ("local", *labels.T[::-1])):
                folds = evaluate(x, target, context)
                key = f"{path.stem}_{rep}_{task}"
                results[key] = folds
                if rep == "pre_logits":
                    print(key, sum(f["correct"] for f in folds), np.sum([f["fixed_alpha_correct"] for f in folds],axis=0))
    (run / "shape-probes.json").write_text(json.dumps(dict(alphas=alphas.tolist(), folds=results), indent=2))
