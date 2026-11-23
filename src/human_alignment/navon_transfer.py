"""Nested rendering/nuisance holdouts for Navon linear decoding."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_alignment.navon_followup import (
    DATASET, REVISION, MODELS, load_embeddings, outer_folds, validate_source_files, write_csv,
)

ALPHAS = tuple(10.0 ** i for i in range(-6, 5))


def ridge_path_predict(x, y, valid, alphas=ALPHAS):
    """Evaluate a ridge grid using one eigendecomposition, matching RidgeClassifier."""
    classes, labels = np.unique(y, return_inverse=True)
    if len(classes) < 3:
        raise ValueError("This multiclass ridge path expects at least three classes")
    targets = 2.0 * np.eye(len(classes))[labels] - 1.0
    mean_x, mean_y = x.mean(axis=0), targets.mean(axis=0)
    centered = x - mean_x
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
    eigenvalues = np.maximum(eigenvalues, 0)
    rhs = vectors.T @ (centered.T @ (targets - mean_y))
    projected = (valid - mean_x) @ vectors
    return [classes[np.argmax((projected / (eigenvalues + a)) @ rhs + mean_y, axis=1)]
            for a in alphas]


def inner_splits(train, groups, conditions):
    """Hold out a nuisance group and one source condition jointly."""
    sources = np.unique(conditions[train])
    if len(sources) != 2:
        raise ValueError("Expected two source rendering conditions")
    for fit_group, valid_group in GroupKFold(n_splits=3).split(train, groups=groups[train]):
        for held_condition in sources:
            fit = train[fit_group][conditions[train[fit_group]] != held_condition]
            valid = train[valid_group][conditions[train[valid_group]] == held_condition]
            assert len(fit) and len(valid)
            assert not set(groups[fit]) & set(groups[valid])
            assert not set(conditions[fit]) & set(conditions[valid])
            yield fit, valid


def transfer_probe(unit, metadata, target, seed=0, alphas=ALPHAS):
    y = np.asarray([r[f"{target}_letter"] for r in metadata])
    nuisance = "local" if target == "global" else "global"
    groups = np.asarray([r[f"{nuisance}_letter"] for r in metadata])
    conditions = np.asarray([r["condition"] for r in metadata])
    predictions, fold_rows = [], []
    for held_condition in np.unique(conditions):
        for fold, nuisance_train, test in outer_folds(groups, seed):
            train = nuisance_train[conditions[nuisance_train] != held_condition]
            assert held_condition not in set(conditions[train])
            assert not set(groups[train]) & set(groups[test])
            assert set(y[train]) == set(y)
            hits, count = np.zeros(len(alphas)), 0
            for fit, valid in inner_splits(train, groups, conditions):
                assert set(y[fit]) == set(y)
                for i, guessed in enumerate(ridge_path_predict(unit[fit], y[fit], unit[valid], alphas)):
                    hits[i] += np.sum(guessed == y[valid])
                count += len(valid)
            best = max(range(len(alphas)), key=lambda i: (hits[i], alphas[i]))
            estimator = RidgeClassifier(alpha=alphas[best], solver="cholesky")
            estimator.fit(unit[train], y[train])
            guessed = estimator.predict(unit[test])
            for j, idx in enumerate(test):
                r = metadata[idx]
                predictions.append(dict(image_id=r["image_id"], target=target, truth=y[idx],
                                        nuisance=groups[idx], global_letter=r["global_letter"],
                                        local_letter=r["local_letter"], condition=conditions[idx],
                                        position=r["pos_idx"], held_condition=held_condition,
                                        domain="out" if conditions[idx] == held_condition else "in",
                                        fold=fold, alpha=alphas[best], prediction=guessed[j],
                                        correct=int(guessed[j] == y[idx])))
            fold_rows.append(dict(target=target, held_condition=held_condition, fold=fold,
                                  held_nuisance="".join(sorted(set(groups[test]))),
                                  n_train=len(train), n_test_out=int(np.sum(conditions[test] == held_condition)),
                                  n_test_in=int(np.sum(conditions[test] != held_condition)),
                                  alpha=alphas[best], inner_accuracy=hits[best] / count,
                                  inner_scores=json.dumps({str(a): float(h/count) for a,h in zip(alphas,hits)})))
    return predictions, fold_rows


def summarize(predictions):
    rows = []
    for held in ["ALL"] + sorted({r["held_condition"] for r in predictions}):
        for domain in ("in", "out"):
            selected = [r for r in predictions if r["domain"] == domain
                        and (held == "ALL" or r["held_condition"] == held)]
            rows.append(dict(held_condition=held, domain=domain, n=len(selected),
                             accuracy=accuracy_score([r["truth"] for r in selected], [r["prediction"] for r in selected]),
                             balanced_accuracy=balanced_accuracy_score([r["truth"] for r in selected], [r["prediction"] for r in selected])))
    return rows


def matched_gaps(predictions):
    groups = {}
    for r in predictions:
        key = (r["held_condition"], r["global_letter"], r["local_letter"], r["position"])
        groups.setdefault(key, []).append(r)
    result = []
    for (held, g, l, pos), rows in sorted(groups.items()):
        inside = [r["correct"] for r in rows if r["domain"] == "in"]
        outside = [r["correct"] for r in rows if r["domain"] == "out"]
        if len(inside) != 2 or len(outside) != 1 or len({r["fold"] for r in rows}) != 1:
            raise ValueError("Each matched comparison needs two source controls and one target condition")
        result.append(dict(held_condition=held, global_letter=g, local_letter=l, position=pos,
                           out_correct=outside[0], in_mean_correct=float(np.mean(inside)),
                           gap=outside[0] - float(np.mean(inside))))
    return result


def make_report(out, scores, contrasts, representations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = ["# Navon rendering-transfer results", "",
             "Nested nuisance-letter and rendering-condition holdouts; frozen embeddings, no inference.",
             f"Dataset `{DATASET}` at `{REVISION}`. Ridge alpha grid: `{ALPHAS}`.", "",
             "In-domain controls and out-of-domain tests use the same fitted classifier and matched letter pairs/positions. In-domain images occur twice across the three fits; these are dependent observations.", ""]
    for rep in representations:
        lookup = {(r["model"],r["target"],r["held_condition"],r["domain"]): r["accuracy"]
                  for r in scores if r["representation"] == rep}
        lines += [f"## {rep}", "", "| Model | Global in-domain | Global transfer | Local in-domain | Local transfer |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for m in MODELS:
            values = [lookup[m,t,"ALL",d] for t in ("global","local") for d in ("in","out")]
            lines.append(f"| {m} | " + " | ".join(f"{v:.2%}" for v in values) + " |")
        lines += ["", "| Family | Target | Held rendering | AligNet − UnAligNet transfer (pp) | Difference in transfer gap (pp) |",
                  "| --- | --- | --- | ---: | ---: |"]
        for r in contrasts:
            if r["representation"] == rep:
                lines.append(f"| {r['family']} | {r['target']} | {r['held_condition']} | {100*r['delta_out_accuracy']:+.2f} | {100*r['delta_gap']:+.2f} |")
        fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for col,family in enumerate(("dino","siglip")):
            for row,target in enumerate(("global","local")):
                ax=axes[row,col]
                for variant,color in zip(("base","unalignet","alignet"),("#64748b","#d97706","#2563eb")):
                    for domain,style in (("out","-"),("in","--")):
                        ys=[100*lookup[f"{family}_{variant}",target,c,domain] for c in ("s10","s14","s18")]
                        ax.plot(range(3),ys,style,marker="o",color=color,label=f"{variant} {'transfer' if domain=='out' else 'control'}")
                ax.set(title=f"{family.upper()} — {target} letter",ylim=(0,103),ylabel="Accuracy (%)")
                ax.set_xticks(range(3),["hold s10","hold s14","hold s18"])
                ax.axhline(100/26,color="gray",linewidth=0.7)
                ax.legend(fontsize=7,ncol=2)
        fig.suptitle(f"Joint rendering and nuisance holdouts: {rep}")
        fig.savefig(out/f"transfer_{rep}.png",dpi=180)
        plt.close(fig)
        lines += ["",f"![Transfer plot](transfer_{rep}.png)",""]
    lines += ["## Limits", "",
              "Size and spacing are coupled. Global identity is not human agreement. Differences in gaps also reflect intrinsic rendering difficulty. No population confidence intervals are claimed because letters, folds, and images are dependent. DINO preprocessing is inherited from the source dataset; Base triplet heads are reported untrained. Inspect tuning boundaries in probe_folds.csv."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=Path("outputs/navon_followup/data"))
    parser.add_argument("--out-dir",type=Path,default=Path("outputs/navon_transfer"))
    parser.add_argument("--representations",nargs="+",choices=("pre_logits","triplet_logits"),default=["pre_logits"])
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    validate_source_files(args.data_dir)
    manifest=json.loads((args.data_dir/"manifest.json").read_text())
    items=manifest["items"]
    selected=[i for i,r in enumerate(items) if r["kind"]=="compound" and r["global_letter"]!=r["local_letter"]]
    metadata=[items[i] for i in selected]
    if len(metadata)!=3900:
        raise ValueError("Expected 3900 incongruent compounds")
    predictions,folds,scores,gaps=[],[],[],[]
    for rep in args.representations:
        for model in MODELS:
            unit=load_embeddings(args.data_dir/f"embeddings-{model}.parquet",items,rep)[selected]
            for target in ("global","local"):
                print(f"{model} {rep} {target}",flush=True)
                with threadpool_limits(limits=2):
                    p,f=transfer_probe(unit,metadata,target)
                s,g=summarize(p),matched_gaps(p)
                for rows in (p,f,s,g):
                    for r in rows:
                        r.update(model=model,representation=rep,target=target)
                predictions.extend(p); folds.extend(f); scores.extend(s); gaps.extend(g)
                print("  "+", ".join(f"{r['domain']}={r['accuracy']:.3%}" for r in s if r['held_condition']=="ALL"),flush=True)
                for name,rows in (("predictions",predictions),("probe_folds",folds),("scores",scores),("matched_gaps",gaps)):
                    write_csv(args.out_dir/f"{name}.csv",rows)
    lookup={(r['model'],r['representation'],r['target'],r['held_condition'],r['domain']):r['accuracy'] for r in scores}
    contrasts=[]
    for rep in args.representations:
        for family in ("dino","siglip"):
            for target in ("global","local"):
                for held in ("ALL","s10","s14","s18"):
                    a=[lookup[f"{family}_alignet",rep,target,held,d] for d in ("in","out")]
                    b=[lookup[f"{family}_unalignet",rep,target,held,d] for d in ("in","out")]
                    contrasts.append(dict(family=family,representation=rep,target=target,held_condition=held,
                                          delta_out_accuracy=a[1]-b[1],delta_in_accuracy=a[0]-b[0],
                                          delta_gap=(a[1]-a[0])-(b[1]-b[0])))
    write_csv(args.out_dir/"contrasts.csv",contrasts)
    make_report(args.out_dir,scores,contrasts,args.representations)
    sources=[Path(__file__),Path(__file__).with_name("navon_followup.py")]
    for source in sources:
        (args.out_dir/source.name).write_bytes(source.read_bytes())
    record=dict(dataset=DATASET,revision=REVISION,seed=0,alphas=ALPHAS,representations=args.representations,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data_dir.glob("*.parquet")},
                packages={p:importlib.metadata.version(p) for p in ("numpy","scikit-learn","pyarrow")},
                n_predictions=len(predictions),inference_run=False)
    (args.out_dir/"run_record.json").write_text(json.dumps(record,indent=2)+"\n")
    print(f"Report: {args.out_dir/'report.md'}",flush=True)


if __name__=="__main__":
    main()
