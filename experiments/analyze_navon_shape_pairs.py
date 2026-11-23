"""Break down a saved navon_shapes run by which identity changes."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def recover_labels(pairs, details):
    """Each details row holds the global/local labels of both endpoints."""
    assignments = [set() for _ in range(49)]
    for ids, labels in zip(pairs, details.reshape(-1, 2, 2), strict=True):
        for image_id, label in zip(ids, labels, strict=True):
            assignments[image_id].add(tuple(map(int, label)))
    if any(len(labels) != 1 for labels in assignments):
        raise ValueError("Inconsistent identity metadata")
    labels = np.array([item.pop() for item in assignments])
    if set(map(tuple, labels)) != {(g, l) for g in range(1, 8) for l in range(1, 8)}:
        raise ValueError("Expected one stimulus for every global/local combination")
    return labels


def analyze(data_path, run_path, representation):
    with np.load(data_path, allow_pickle=False) as data:
        pairs, rt, human = (data[key] for key in ("pairs", "subject_rt", "human_distance"))
        labels = recover_labels(pairs, data["pair_details"])
    if len(pairs) != 1176 or {tuple(sorted(pair)) for pair in pairs} != set(combinations(range(49), 2)):
        raise ValueError("Incomplete or duplicated image pairs")
    np.testing.assert_allclose(human, 1 / rt.mean(axis=1))
    left, right = labels[pairs[:, 0]], labels[pairs[:, 1]]
    global_changed, local_changed = (left != right).T
    strata = {
        "all": np.ones(len(pairs), dtype=bool),
        "global_only": global_changed & ~local_changed,
        "local_only": local_changed & ~global_changed,
        "both": global_changed & local_changed,
    }
    variants = ("base", "unalignet", "alignet")
    models = [(family, variant) for family in ("dino", "siglip") for variant in variants]
    distances = []
    for family, variant in models:
        with np.load(run_path / f"{family}_{variant}_{representation}-distances.npz") as saved:
            np.testing.assert_array_equal(saved["pairs"], pairs)
            np.testing.assert_array_equal(saved["human"], human)
            distances.append(saved["frozen"])
    distances = np.column_stack(distances)
    scenarios = {"full": (strata["all"], human)}
    for identity in range(1, 8):
        for axis, name in enumerate(("global", "local")):
            keep = (left[:, axis] != identity) & (right[:, axis] != identity)
            scenarios[f"omit_{name}_{identity}"] = (keep, human)
    for participant in range(rt.shape[1]):
        target = 1 / np.delete(rt, participant, axis=1).mean(axis=1)
        scenarios[f"omit_participant_{participant + 1}"] = (strata["all"], target)

    results = {}
    for name, (keep, target) in scenarios.items():
        results[name] = {}
        for stratum, mask in strata.items():
            mask = mask & keep
            ranks = rankdata(np.column_stack((target[mask], distances[mask])), axis=0)
            correlations = np.corrcoef(ranks, rowvar=False)[0, 1:]
            scores = {f"{f}_{v}": float(r) for (f, v), r in zip(models, correlations)}
            changes = {
                f"{f}_{after}_minus_{before}": scores[f"{f}_{after}"] - scores[f"{f}_{before}"]
                for f in ("dino", "siglip")
                for before, after in zip(variants, variants[1:])
            }
            results[name][stratum] = dict(pairs=int(mask.sum()), spearman=scores, changes=changes)
    return dict(representation=representation, labels=labels.tolist(), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/navon_shapes/data/human.npz"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--representation", choices=("pre_logits", "triplet_logits"), default="pre_logits")
    args = parser.parse_args()
    print(json.dumps(analyze(args.data, args.run, args.representation), indent=2, allow_nan=False))
