# Human-distance alignment on hierarchical shapes

This is the active experiment launched by `run_navon_human.sbatch`. It replaces
the job's earlier requirement for a manually curated letter-response manifest.
The older `navon_human.py` remains available for letter-response datasets; this
experiment uses `navon_shapes.py` and the recovered Jacob & Arun data.

## Run on the cluster

From this repository on branch `experiment/navon-human-alignment`:

```bash
uv sync --locked
uv run --locked python -m human_alignment.navon_shapes prepare --download-data
mkdir -p outputs/navon_human
sbatch --export=ALL,NAVON_DOWNLOAD_MODELS=1 run_navon_human.sbatch
```

The preparation command downloads the pinned 547 KB public MATLAB file, checks its
SHA256, extracts 49 original grayscale images losslessly to PNG, and exports
anonymous human-distance arrays. No personal metadata are exported. Prepared
inputs are reused only after hash validation. Images and data remain ignored in
`outputs/navon_shapes/data/`. The raw publication file contains participant
metadata and is not committed.

The job uses the original cluster defaults (`coe-gpu`, one GPU, four CPUs, 32 GB)
with a two-hour limit for six model loads and optional downloads. The log directory
must exist before submission. `NAVON_DOWNLOAD_MODELS=1` allows missing official
checkpoints to be downloaded into `models/`; omit it when checkpoints are already
cached. Dependency installation occurs separately with `uv sync --locked`.
The job uses `uv run --locked --no-sync`.

The job runs each checkpoint in a fresh process: DINOv2-B and SigLIP-B, each with
Base, UnAligNet (`untransformed`) and AligNet. It saves both `pre_logits` (primary)
and `triplet_logits` (secondary) and skips validated existing embedding caches.
Analysis then runs on CPU. Results go to `outputs/navon_shapes/run-<job ID>/`.

Environment overrides: `NAVON_UV_ROOT` changes the original cluster account root
(default `/storage/ice1/4/5/nparikh44`); `NAVON_MODELS` changes the checkpoint cache;
`NAVON_SHAPES_DATA`, `NAVON_SHAPES_CACHE`, and `NAVON_SHAPES_OUT` change paths;
`NAVON_HUMAN_BATCH_SIZE` controls inference batch size (default 16).
Do not run simultaneous jobs writing the same embedding cache.

## Source and human targets

Source: [Jacob & Arun (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7594584/),
[author repository](https://github.com/georginjacob/Multiscale), commit
`eabd49541b10745fd2469e4137acefbc859a5a3d`, `FigureCodes/L2_VSmain.mat`.
Expected SHA256:
`2c4b72f6fe9f37c1848bb1023c67d2b27e0a275171784cd884cd91f7349f86c7`.

There are 49 images (400×400), all 1,176 unordered pairs and eight participants.
The released RT array is 1,176×8×2: the final axis distinguishes the two search
directions. Following the averaging structure in `figure_04.m`, average the two
directions for each participant, then average participants. Define human distance
as the reciprocal of that mean RT in seconds. Do not average inverse RTs instead.
The pinned consolidated data contain finite positive RTs; no additional exclusion
is applied here. Pairwise accuracy is available in the original file but is not
used to invent trial-level exclusion flags. This is a perceptual-distance target,
not human odd-one-out probabilities or letter-response distributions.

## Frozen evaluation

Use `models.load_images` consistently within each family: RGB conversion,
OpenCV bicubic direct resize to the exported input size, float32 scaling to [0,1].
Original DINO training preprocessing provenance remains unresolved. Save checkpoint
fingerprints and TensorFlow version alongside embeddings. Unit-normalize feature
vectors and compute Euclidean distances. Report Pearson and Spearman correlations
with human distances for all six checkpoints and both representations.

## Small adaptation experiment

Fit only Base embeddings, separately by family and representation. This is a
nonnegative diagonal distance metric, not backbone fine-tuning or full AligNet
distillation. For unit embeddings z, pair features are `(z_i - z_j)^2`. Learn
nonnegative weights w predicting normalized human distance as the sum of weighted
pair features. Penalize mean `(w - 1)^2` to shrink toward the frozen metric.
The representation transform is `z * sqrt(w)` without renormalizing afterward;
the predictor is its squared Euclidean distance with recorded scaling constants.

Use seven deterministic image folds (seed 2026), seven images per test fold,
seven different images for validation, and 35 for training. During tuning, only
within-training pairs enter optimization and only within-validation pairs enter
selection. Tune regularization over `[0.01, 0.1, 1, 10]` by normalized validation
MSE, with all scales calculated from training data. Refit on all 42 non-test
images, including cross-pairs among those images, then evaluate only pairs whose
**both** endpoints are held-out images. Save split IDs, selected penalties,
scales, weights and pair predictions.

This evaluates 21 pairs per fold, 147 pairs overall. It does not claim all 1,176
pairs are independent held-out examples. Compare adapted versus frozen Spearman
within each test fold, and report the descriptive mean fold Spearman. The frozen
adaptation control uses squared Euclidean distances; their ranks equal Euclidean
distance ranks. Its Pearson correlation need not equal the main frozen Euclidean
analysis. No confidence intervals or significance claims are computed. Participant
and image dependence require a dedicated uncertainty analysis before strong claims.
No same–different or additional rendering-condition data enter this first run.

## Outputs and validation

`report.json` contains input hashes, all frozen correlations, adaptation folds and
scores. `*-distances.npz` records complete frozen distances; `*-weights.npz` stores
seven fitted metrics; `*-predictions.json` stores matched held-out predictions.
The output directory must be empty to avoid overwriting results.

Checks run locally: source hash and image extraction on real published data;
synthetic tests for image-disjoint splits, metric recovery, feature normalization,
and independence of fitting from held-out targets. Inference compatibility and
real-model results still require the cluster run. Synthetic features are not
benchmark evidence.

```bash
uv run --locked python -m unittest discover -s tests -p 'test_navon_shapes.py'
bash -n run_navon_human.sbatch
```
