# Navon rendering-transfer results

Completed 2026-11-18 on all six published checkpoints, using `pre_logits` and
`triplet_logits` separately. No models were downloaded and no image inference was
run. The pinned inputs are the same as in the earlier Navon follow-up.

See the [design](navon-transfer.md), [implementation](../src/human_alignment/navon_transfer.py),
and [full generated report](../outputs/navon_transfer/report.md).

## What was tested

For each of s10, s14, and s18, train a linear ridge decoder on the other two
rendering conditions and test on the held-out condition. Five nuisance-letter
folds additionally hold out local identities for global decoding, or global
identities for local decoding. All 3,900 incongruent compounds are covered.

Select regularization only inside the training set. Inner validation crosses
three nuisance-group folds with both directions of transfer between the two
source conditions, over alpha values from 1e-6 to 1e4. The same outer fitted
classifier also predicts the held-out letter combinations and positions in both
source conditions, supplying matched in-domain controls.

This differs from the earlier follow-up's tuning objective. Evaluate transfer
relative to these matched controls rather than subtracting the earlier run's
accuracies, which came from different training data and tuning.

## Primary results (`pre_logits`)

| Model | Global in-domain | Global transfer | Transfer − in-domain | Local transfer |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 Base | 94.54% | 86.72% | −7.82 pp | 96.41% |
| DINOv2 UnAligNet | 79.92% | 55.44% | −24.49 pp | 95.74% |
| DINOv2 AligNet | 72.29% | 52.97% | −19.32 pp | 96.36% |
| SigLIP Base | 43.41% | 37.08% | −6.33 pp | 98.26% |
| SigLIP UnAligNet | 46.12% | 39.41% | −6.71 pp | 98.64% |
| SigLIP AligNet | 45.87% | 37.26% | −8.62 pp | 99.31% |

**AligNet has lower aggregate global transfer accuracy than UnAligNet in both
families, but the additional transfer cost differs between families.**

- DINOv2: the AligNet–UnAligNet global-decoding gap narrows from −7.63 percentage
  points in-domain to −2.46 points on held-out renderings. AligNet's transfer drop
  is 5.17 points smaller. Its lower absolute transfer accuracy does not establish
  a greater sensitivity to the rendering shift than UnAligNet.
- SigLIP: the gap grows from −0.24 points in-domain to −2.15 points on held-out
  renderings. AligNet's transfer drop is 1.91 points larger.
- Local decoding transfers well in all variants. AligNet has slightly higher
  local transfer accuracy than UnAligNet in both families.

## Rendering-specific global transfer

| Family | Held-out rendering | Base | UnAligNet | AligNet | AligNet − UnAligNet |
| --- | --- | ---: | ---: | ---: | ---: |
| DINOv2 | s10 | 92.77% | 55.38% | 54.62% | −0.77 pp |
| DINOv2 | s14 | 90.54% | 60.62% | 55.85% | −4.77 pp |
| DINOv2 | s18 | 76.85% | 50.31% | 48.46% | −1.85 pp |
| SigLIP | s10 | 56.69% | 60.69% | 53.77% | −6.92 pp |
| SigLIP | s14 | 32.23% | 34.15% | 36.69% | +2.54 pp |
| SigLIP | s18 | 22.31% | 23.38% | 21.31% | −2.08 pp |

The SigLIP result is not uniform: AligNet improves on UnAligNet when the middle
condition s14 is held out. Predicting s18, whose local elements are largest and
sparsest, is difficult for global decoding across all checkpoints. In some s10
tests, transfer accuracy exceeds in-domain control accuracy, showing why these
gaps also reflect differences in rendering difficulty.

![Primary transfer plot](../outputs/navon_transfer/transfer_pre_logits.png)

## Secondary representation

The aggregate global transfer ordering also holds for `triplet_logits`:

| Family | Base | UnAligNet | AligNet |
| --- | ---: | ---: | ---: |
| DINOv2 | 85.41% | 52.38% | 51.44% |
| SigLIP | 36.28% | 40.31% | 38.05% |

As with the primary representation, the matched global transfer drop is smaller
for DINOv2 AligNet than UnAligNet, and larger for SigLIP AligNet. Base's triplet
head is reported untrained and should be treated as secondary context.

## Interpretation limits

These are descriptive comparisons of existing checkpoints on one font. Local
size and spacing change together; the experiment measures rendering transfer,
not a pure scale-invariance mechanism. The held-out condition also changes how
easy global recognition is. A smaller transfer drop can accompany lower absolute
accuracy, so it is not sufficient by itself to claim a better representation.

The two in-domain control renderings and the out-of-domain rendering share the
same letter identities and positions. Images, letters, and folds are dependent;
no independent-trial confidence intervals or population significance claims are
made. All target classes are observed during training. The experiment does not
test new fonts, human agreement, or novel target identities. DINO preprocessing
provenance remains inherited and unresolved.

## Checks and artifacts

- Nine automated tests passed, including four transfer-specific tests: the cached
  ridge path matches scikit-learn on both full-rank and rank-deficient synthetic
  inputs; inner splits separate both rendering and nuisance identities; and a
  synthetic separable task verifies complete transfer predictions and controls.
- Source dataset checksums, embedding metadata, finite vectors, dimensions, and
  row alignment were checked. All outer and inner training folds retain every
  target class, while their held-out axes are disjoint from validation/test data.
- The saved-output audit verifies 280,800 predictions from 360 outer fitted probes,
  93,600 matched domain comparisons, consistent nuisance folds across checkpoints,
  correct domain membership, recomputed accuracies/gaps/contrasts, and archived
  source hashes.
- `git diff --check` passed. Commands used the locked uv analysis environment.

All generated artifacts remain under ignored `outputs/navon_transfer/`:

- [Full report](../outputs/navon_transfer/report.md)
- [Per-image predictions](../outputs/navon_transfer/predictions.csv)
- [Matched domain gaps](../outputs/navon_transfer/matched_gaps.csv)
- [Condition and domain scores](../outputs/navon_transfer/scores.csv)
- [Checkpoint contrasts](../outputs/navon_transfer/contrasts.csv)
- [Tuning scores](../outputs/navon_transfer/probe_folds.csv)
- [Run record](../outputs/navon_transfer/run_record.json)
- [Verification](../outputs/navon_transfer/verification.json)
