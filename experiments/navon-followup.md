# Navon follow-up: distances and linear decoding

This exploratory follow-up uses the existing published embeddings. The original
similarity effects were known when this plan was written; this is not a
preregistered replication. The probe design below was fixed before running probes.

Dataset: `lelouch0204/navon-alignet`, revision
`d334c56a3f0023e25860e55114eb2e9ec5deb3e8`.

## Distance decomposition

For every unordered letter pair X,Y and every third letter Z distinct from both,
measure at matched condition and position:

- Global-change distance: `1 - cos(C(X,Z), C(Y,Z))`.
- Local-change distance: `1 - cos(C(Z,X), C(Z,Y))`.
- Their difference, local minus global, is positive when local changes cause
  larger embedding changes. This equals the negative of the notebook's matched
  substitution margin; it is a decomposition of that result, not new evidence.

Report each component by condition and position, and averaged over all slices.
Keep families and representations separate. For AligNet minus UnAligNet and Base,
report paired differences, leave-one-letter-out ranges, and letter-jackknife
standard errors, excluding a letter in any of X,Y,Z. These describe sensitivity
within this finite alphabet, not generalization to fonts, people, or training seeds.
Save pair-level values averaged over Z and also mean cosine distance over all
distinct incongruent compounds as context for overall embedding spread.

## Controlled linear decoding

Primary output: `pre_logits`. Secondary: `triplet_logits`, whose Base projection
is reported as untrained by the dataset publisher. Do not pool the outputs.

Use all 3,900 incongruent compounds (26 x 25 x 3 conditions x 2 positions).
Global and local letter identities are separate 26-class targets. Excluding
congruent images prevents direct identity agreement from simplifying either task.

- L2-normalize each image embedding; no featurewise scaling or PCA.
- Fit a ridge classifier with an intercept. This measures linear accessibility,
  not optimal decoding capacity or human agreement.
- Five outer folds partition the nuisance alphabet using seed 0. For global
  decoding hold out local letters; for local decoding hold out global letters.
  All conditions and positions for a letter combination stay together.
- Select alpha from `[0.1, 1, 10, 100]` with three inner GroupKFold splits on
  nuisance identities within the outer training set. Optimize pooled validation
  accuracy; break ties in favor of the larger alpha. Fit on the complete outer
  training set, then predict its held-out nuisance identities once.
- Use identical folds for every model and representation. Save every outer
  prediction, selected alpha, accuracy, balanced accuracy, and condition breakdown.
- The same target classes occur in train and test. These are tests of transfer
  across nuisance identities, not unseen target letters or unseen rendering styles.
- Report 1/26 (3.85%) as the uniform random reference. Class counts within folds
  are slightly unequal after excluding congruent images; also report balanced
  accuracy. Do not treat correlated images or the five folds as independent trials
  for confidence intervals.

## Execution

`uv sync --locked --only-group analysis` installs CPU analysis dependencies only.

```bash
uv run --locked --only-group analysis python src/human_alignment/navon_followup.py --download
uv run --locked --only-group analysis python -m unittest discover -s tests -p 'test_navon_followup.py'
```

Downloads and results live under ignored `outputs/navon_followup/`. The run record
stores input hashes, dependency versions, arguments, and checks. No model loading,
image inference, or alteration of the original research documents is needed.

Interpret increased local preference jointly with component distances and decoding.
Good global decoding despite local-biased similarity supports retained, linearly
accessible global information. Lower decoding performance indicates reduced
accessibility under this specific probe protocol, not absence of global information.
DINO export preprocessing provenance and the absence of a matched human baseline
remain limitations of the source data.

## Additional regularization sensitivity check

The first run selected alpha=0.1 (the lower grid boundary) in many folds,
particularly for DINO. This motivates a separate exploratory run, retaining the
original results, with alpha values from 1e-6 to 1e4 in powers of ten. Selection
still uses only the inner training folds. This checks whether the original range
over-regularizes models with compressed cosine geometry; no outer test accuracy
is used to select alpha. This follow-up was added after viewing first-run results.

```bash
uv run --locked --only-group analysis python src/human_alignment/navon_followup.py \
  --data-dir outputs/navon_followup/data \
  --out-dir outputs/navon_followup/sensitivity \
  --representations pre_logits \
  --alphas 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1 10 100 1000 10000
```

The same broad-grid check is also run with `--representations triplet_logits`
and `--out-dir outputs/navon_followup/sensitivity_triplet`. The original
restricted-grid run, both sensitivity runs, and source snapshots are kept
separately. Some SigLIP inner folds select the lowest value even on this wider
grid; decoding results remain protocol-dependent estimates, not optimal-capacity
claims. Read the restricted and broad results together.
