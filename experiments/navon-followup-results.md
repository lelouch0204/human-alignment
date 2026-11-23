# Navon follow-up results

Completed on 2026-11-18 using the published frozen embeddings at dataset revision
[`d334c56a3f0023e25860e55114eb2e9ec5deb3e8`](https://huggingface.co/datasets/lelouch0204/navon-alignet/tree/d334c56a3f0023e25860e55114eb2e9ec5deb3e8).
No vision models were downloaded or run. The [analysis plan](navon-followup.md)
and [implementation](../src/human_alignment/navon_followup.py) describe the methods.

## Main findings

AligNet has a larger local-versus-global cosine-distance imbalance than UnAligNet
in both model families. Both component distances increase, but local-change
distance increases substantially more. Global-letter information remains linearly
decodable, although accuracy is lower in AligNet than UnAligNet under the broader
regularization search. Local decoding is nearly perfect.

The initial restricted regularization grid gave the opposite global-decoding
ordering. Consequently the probe result must be reported with its tuning protocol;
the initial run alone would support a misleading conclusion about relative linear
accessibility. All runs are retained separately.

## Distance decomposition (`pre_logits`)

Distances average all distinct X,Y,Z substitutions, three rendering conditions,
and both positions. Global change compares C(X,Z) with C(Y,Z); local change
compares C(Z,X) with C(Z,Y). Distance is one minus cosine similarity.

| Model | Global-change distance | Local-change distance | Local − global |
| --- | ---: | ---: | ---: |
| DINOv2 Base | 0.0290 | 0.0539 | 0.0249 |
| DINOv2 UnAligNet | 0.0032 | 0.0094 | 0.0062 |
| DINOv2 AligNet | 0.0196 | 0.0780 | 0.0584 |
| SigLIP Base | 0.0945 | 0.1973 | 0.1028 |
| SigLIP UnAligNet | 0.0480 | 0.1129 | 0.0650 |
| SigLIP AligNet | 0.0678 | 0.1960 | 0.1281 |

AligNet minus UnAligNet increases global-change distance by 0.0164 (DINOv2) and
0.0199 (SigLIP), and local-change distance by 0.0686 and 0.0830. The resulting
local-minus-global contrast is positive after omitting any single letter in any
of the three roles: ranges [0.0491, 0.0535] and [0.0579, 0.0666], respectively.
These are sensitivity ranges, not population confidence intervals.

Overall geometry differs substantially across checkpoints. Mean distance between
distinct incongruent compounds at matched condition/position is 0.0103 in DINOv2
UnAligNet and 0.0839 in AligNet. Thus larger raw distances alone cannot establish
better identity encoding. The components decompose the original balanced
substitution result; they are not an independent replication.

## Linear decoding with broader tuning

Five outer folds hold out local identities for global decoding, or global
identities for local decoding. All 3,900 incongruent images receive one held-out
prediction per model, representation, and target. The ridge regularization value
is selected using three inner nuisance-group folds, from powers of ten between
1e-6 and 1e4. Embeddings are L2-normalized without featurewise scaling. Test labels
are not used to select regularization. Uniform random accuracy is 3.85%.

| Model | `pre_logits` global | `pre_logits` local | `triplet_logits` global | `triplet_logits` local |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 Base | 95.97% | 100.00% | 95.77% | 100.00% |
| DINOv2 UnAligNet | 79.49% | 100.00% | 78.74% | 100.00% |
| DINOv2 AligNet | 72.49% | 100.00% | 70.44% | 100.00% |
| SigLIP Base | 51.79% | 99.72% | 45.72% | 99.77% |
| SigLIP UnAligNet | 54.36% | 99.92% | 53.95% | 99.85% |
| SigLIP AligNet | 48.08% | 99.97% | 47.85% | 100.00% |

For `pre_logits`, AligNet's global decoding is lower than UnAligNet's at each of
the three sizes and at each of the two positions, as well as in the aggregate.
All sizes and positions occur during training; these breakdowns do not measure
transfer to unseen rendering conditions. Base `triplet_logits` is described by
the publisher as an untrained projection and is secondary context only.

### Why the tuning sensitivity matters

| Global-decoding contrast (AligNet − UnAligNet) | Initial grid [0.1, 1, 10, 100] | Broader grid [1e-6, ..., 1e4] |
| --- | ---: | ---: |
| DINOv2 `pre_logits` | +5.15 pp | −7.00 pp |
| SigLIP `pre_logits` | +6.95 pp | −6.28 pp |
| DINOv2 `triplet_logits` | +1.23 pp | −8.31 pp |
| SigLIP `triplet_logits` | +4.49 pp | −6.10 pp |

Many initial fits selected the smallest allowed alpha. The wider search was added
after noticing that boundary behavior and inspecting initial results, so it is
an exploratory sensitivity analysis. Some SigLIP folds still select the wider
grid's minimum. These accuracies describe this ridge protocol, not optimal linear
capacity. No claim of lost information follows from a decoding decrease.

## Interpretation and limits

The results support retained global identity information alongside a stronger
local bias in cosine geometry. Under the broader ridge protocol, global identity
is less accessible in AligNet than UnAligNet. This is a task-specific finding on
one font and coupled size/spacing conditions. It does not establish a loss of
global recognition ability, a mechanism involving spatial frequency, or reduced
human agreement. There are no human judgments on these images, and DINO export
preprocessing provenance remains unresolved. Only one checkpoint per variant is
available; images, letter pairs, and folds are dependent.

## Checks actually run and saved artifacts

- Five automated tests passed: known synthetic component distances and margin
  identity, omission of each letter in every role, outer-fold separation,
  separable synthetic decoding with held-out nuisance identities, and embedding
  row alignment/duplicate rejection.
- Six embedding SHA-256 hashes and the manifest Git blob hash match the pinned
  dataset. Checkpoint labels, image IDs, vector dimensions, finite values, and
  nonzero norms were checked.
- Forty-eight original compound-margin summaries were reproduced from the
  embeddings, with maximum absolute discrepancy approximately 1.11e-9.
- Saved predictions were audited for uniqueness, expected row counts, identical
  fold assignments across checkpoints, recomputed accuracy, and archived analysis
  source hashes. The three runs contain 187,200 predictions in total.
- No TensorFlow installation or image inference was used. Dependencies were
  resolved with `uv lock`, installed using `uv sync --locked --only-group analysis`,
  and commands executed with `uv run --locked --only-group analysis` on Python 3.12.

Generated files are intentionally ignored by Git:

- [Initial run, both representations](../outputs/navon_followup/report.md)
- [Broader-grid primary report](../outputs/navon_followup/sensitivity/report.md)
- [Broader-grid secondary report](../outputs/navon_followup/sensitivity_triplet/report.md)
- [Primary figure](../outputs/navon_followup/sensitivity/followup_pre_logits.png)
- [Prediction audit](../outputs/navon_followup/verification.json)
- [Published-result reproduction check](../outputs/navon_followup/source_reproduction_check.json)

Each run also saves component-distance tables, pair summaries, letter omissions,
per-image predictions, fold tuning results, condition breakdowns, input hashes,
dependency versions, and an analysis source snapshot. The sensitivity runs record
all inner-validation scores for every tested alpha.
