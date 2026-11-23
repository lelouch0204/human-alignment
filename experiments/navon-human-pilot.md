# Navon human-response alignment pilot

**Legacy letter-response pilot:** `run_navon_human.sbatch` now launches the
[recovered shape-distance experiment](navon-shapes.md). The direct Python commands
below remain available, but its old SBATCH instructions are superseded.

This pilot tests whether human response supervision improves prediction of unseen
Navon behavior beyond ordinary letter-label supervision. It trains two separate
instruction-specific linear softmax heads on a **frozen official DINOv2-B Base**
checkpoint (`dino`, `base`). `pre_logits` is the primary representation.
SigLIP-B Base and `triplet_logits` are optional separate runs.

The script implements the small adaptation pilot discussed in chat. It does not
fine-tune the backbone or implement AligNet's teacher–student triplet distillation.
The resulting weights are readouts, not new AligNet checkpoints. Human responses
must be linked to recoverable stimuli before the real experiment can run. The
OSF source schema and stimulus availability have not yet been verified, so there
is no automatic OSF importer or invented behavioral data.

## Input contract

Supply a curated JSON manifest; keep it and images under ignored `outputs/`.
Image paths are relative to the manifest. A minimal schema illustration follows
(this one-row example is intentionally not a complete runnable dataset):

```json
{
  "labels": ["H", "S"],
  "provenance": "Study, dataset version, source files, response-key mapping, exclusions and participant aggregation",
  "source_audit": "How stimulus originals and all derived renderings were linked",
  "split_description": "Which sources, rendering conditions or studies were held out and why",
  "stimuli": [
    {
      "image_id": "study1-HS-small",
      "source_id": "canonical-HS-source",
      "study": "study1",
      "condition": "small",
      "split": "train",
      "path": "images/study1-HS-small.png",
      "sha256": "actual SHA256 of image bytes",
      "global_letter": "H",
      "local_letter": "S"
    }
  ],
  "responses": [
    {"image_id": "study1-HS-small", "level": "global", "counts": [80, 20]},
    {"image_id": "study1-HS-small", "level": "local", "counts": [15, 85]}
  ]
}
```

`counts` contains observed response counts in the exact order of `labels`.
Decode response keys to letter identities separately for each participant's key
mapping. Include incorrect valid responses. Do not manufacture distributions
from condition-average accuracy, reaction times, or the intended correct answer.
Predefine handling of omissions, invalid keys and participant exclusions, and
record it in `provenance`. This implementation conditions on valid letter responses;
it does not model omissions or response times. Pool only compatible task designs
and response options; start with one well-documented study.

Provide `train`, `validation`, and `test` splits, each containing responses at
both target levels. Training must cover every response label at both levels.
Each image/instruction pair has one aggregated response-count row. Each stimulus
has a unique `image_id`. All images derived from the same original stimulus
must share a canonical `source_id` and remain in one split. The validator checks
source separation, hashes, exact image duplication across splits, response
counts, label coverage and IDs. It cannot discover undeclared image relationships.

Choose the split **before fitting**. Hold out independent stimulus sources;
where data allow, additionally reserve a rendering condition or a whole study
for test and use a different held-out condition for validation. The script
honors supplied splits; it does not automatically create condition/study holdouts.
Holding out participants while reusing the same images does not test stimulus
transfer. If participants appear across splits, conclusions concern new stimuli
for that participant population, not unseen-participant generalization. Report
both distinct-source counts and participant coverage in the dataset audit.

## Matched training and evaluation

1. Extract frozen Base embeddings once. Use the existing `models.load_images`
   preprocessing: EXIF transpose, RGB, direct OpenCV bicubic resize to the export's
   input dimensions, and float32 scaling to [0,1]. DINO's original training
   preprocessing provenance remains unresolved. The script records the export
   fingerprint, TensorFlow version and preprocessing; it does not apply augmentation.
2. Normalize each embedding to unit length. Append an intercept and place it in
   a global or local feature block. This gives two independent instruction-specific
   linear heads with the same features and response labels.
3. Train a human head on normalized observed response counts using soft-target
   cross-entropy, and a matched control on one-hot correct-letter labels.
   Architecture, initialization, optimizer, regularization and update budget match.
   Every stimulus/instruction row has equal weight, regardless of response count.
   L2 applies to all weights, including intercepts.
4. Use full-batch Adam (learning rate 0.03, L2 0.001, 500 updates) with seeds 0, 1,
   and 2. Select each run's best epoch using validation cross-entropy for its own
   training target. Thus the control uses only correct-letter validation labels;
   the human condition uses human validation distributions. No test targets enter
   the optimizer or epoch selection. These are fixed pilot defaults, not tuned claims.
5. Evaluate both readouts on exactly the same test rows. Primary metric is human
   response cross-entropy (lower is better). Also save human KL divergence,
   correct-letter accuracy, expected agreement with a sampled human response,
   and observed human letter accuracy. Report all/global/local and congruent/
   incongruent strata; absent strata are `null`. Accuracy interference can be
   computed as congruent minus incongruent accuracy separately at each target
   level. All reported aggregates are means over stimulus/instruction rows.

A useful result is improved held-out human agreement beyond the letter-label
control, particularly under rendering or study transfer. Good training fit alone
is insufficient. Seed variation is optimizer variation, not a human-population
confidence interval. This initial script does not estimate participant/source
bootstrap intervals; retain trial-level data for that subsequent analysis.

## Run on the GPU machine

Install dependencies separately, with the existing lockfile. Checkpoints must
already be in `models/` (the SBATCH job never downloads them). For a deliberate
checkpoint download, the standalone `embed` command accepts `--download`.

```bash
uv sync --locked
mkdir -p outputs/navon_human
uv run --locked python -m human_alignment.navon_human validate \
  --manifest outputs/navon_human/data/manifest.json
export NAVON_HUMAN_MANIFEST="$PWD/outputs/navon_human/data/manifest.json"
sbatch run_navon_human.sbatch
```

Create the log directory **before** submission: SLURM opens its output file
before running the shell script. The job is based on `run_navon.sbatch`: `coe-gpu`,
one GPU, four CPUs, 32 GB RAM, 30 minutes, and the same cluster uv paths/XLA flag.
Set `NAVON_UV_ROOT` to override the original `/storage/ice1/4/5/nparikh44` root on
another account. Existing `UV_*` variables take precedence. Adjust SLURM time
and resources if the curated dataset requires it. The job uses
`uv run --locked --no-sync`; it does not install dependencies during inference.

Optional environment settings:

| Variable | Default |
| --- | --- |
| `NAVON_HUMAN_FAMILY` | `dino` |
| `NAVON_HUMAN_REPRESENTATION` | `pre_logits` |
| `NAVON_MODELS` | `models` |
| `NAVON_HUMAN_BATCH_SIZE` | `32` |
| `NAVON_HUMAN_EPOCHS` | `500` |
| `NAVON_HUMAN_EMBEDDINGS` | `outputs/navon_human/<family>_base_<representation>.npz` |
| `NAVON_HUMAN_OUT` | `outputs/navon_human/run-<job ID>` |

The job validates inputs, extracts embeddings when the cache is absent, and trains
both objectives in a separate process. Existing caches must match the complete
manifest hash, image order, Base variant, family and representation; choose a new
cache path if these inputs change. Training refuses nonempty output directories.
Do not submit concurrent jobs that write the same embedding cache.

Equivalent standalone commands:

```bash
uv run --locked python -m human_alignment.navon_human embed \
  --manifest outputs/navon_human/data/manifest.json \
  --family dino --representation pre_logits \
  --embeddings outputs/navon_human/dino_base_pre_logits.npz
uv run --locked python -m human_alignment.navon_human train \
  --manifest outputs/navon_human/data/manifest.json \
  --embeddings outputs/navon_human/dino_base_pre_logits.npz \
  --out outputs/navon_human/pilot --seeds 0 1 2
```

For another family/representation, pass both flags consistently to both commands
and choose separate embedding/output paths. The direct-command cache default is
DINO Base `pre_logits`; the SBATCH script constructs a matching filename.

Outputs include a report with input hashes, configuration, split counts, selected
epochs and test metrics; six readout NPZ files; and per-run JSON training histories
and test response probabilities. No model checkpoint or data files are committed.
The frozen-backbone `train` and `validate` steps need only NumPy and the standard
library, and can run in a NumPy-only Python environment using
`python src/human_alignment/navon_human.py ...`.

## Checks performed locally

Five synthetic software tests cover numerical gradients, source-leakage and
invalid-count rejection, instruction-specific heads, test-target independence,
end-to-end cached-embedding training, metadata validation and metric arithmetic.
`bash -n run_navon_human.sbatch` checks shell syntax. No TensorFlow inference,
checkpoint download, real-human-data training or SLURM submission was performed.

```bash
uv run --locked python -m unittest discover -s tests -p 'test_navon_human.py'
bash -n run_navon_human.sbatch
```
