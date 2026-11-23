# LLM handoff: human-alignment / Navon research


> **Purpose:** Self-contained context for a new chat or coding agent. Read this before proposing work or interpreting results. This is a research and repository state snapshot, not an instruction to launch training.
>
> **Updated:** 2026-11-20, America/New_York.
> **Workspace:** `/home/himi/bin/human-alignment`.
> **Current branch:** `experiment/navon-human-alignment`.
> **Current HEAD:** `218e1a1457dca5a8f3774357a591126440a2e69f`.
> **Latest user request:** Document the entire conversation/work in an LLM-first Markdown file for another chat. No additional experiment, commit, or push was requested in this documentation turn.


## 1. Read this first


- We compare **official Base, UnAligNet, and AligNet TensorFlow SavedModels**, for **DINOv2-B** and **SigLIP-B**.
- Two CPU-only experiments on **published letter-Navon embeddings** have actually completed: distance/decoding follow-up and rendering transfer. Their results are below.
- A new pipeline using **49 hierarchical shape images and real human visual-search data** has been implemented, tested, committed and pushed. **No real-model inference or results from this experiment have been supplied in this conversation.** Local `outputs/navon_shapes/` currently contains prepared data, not model-result reports.
- The current SLURM entry point is **`run_navon_human.sbatch`**, but it now invokes **`human_alignment.navon_shapes`**. Its name survived a change of dataset and method.
- **`navon_human.py` is the older letter-response pilot.** It expects a curated JSON manifest. It does not read LoGlo XLSX/CSV files directly and is not the script used by the current SLURM job.
- The shape pipeline evaluates the six existing checkpoints unchanged. It separately learns **nonnegative diagonal distance weights on frozen Base embeddings**. It does **not** fine-tune a backbone or reproduce the full AligNet teacher–student recipe.
- **Do not assert that the downloaded AligNet fails the 49-shape human-distance task. That evaluation is pending.** Earlier letter-decoding findings do not establish that conclusion.
- The local workspace contains substantial **uncommitted earlier work**. A fresh clone of the pushed fork does not contain all those files. Preserve them.


## 2. User intent and project rules


### Research intent


The user initially requested analysis of `experiments/navon_analysis.ipynb`, comparison with the AligNet repository and the Nature paper, and ideas using the published Navon embedding dataset. They then authorized the next experiments, explored human Navon datasets, requested a training pilot on a new branch, requested a fork and push, and later requested that the scripts use the recovered human shape dataset.


The user explicitly clarified that “the third experiment” meant **clean-image recognition from proposal.md**, not a new human study. Later discussion of human alignment is a separate research extension.


The user's latest conceptual question concerned **the actual pretrained AligNet checkpoint downloaded for their project**, not an abstract or newly trained model. The answer must distinguish its original natural-object similarity supervision from spatial global/local tasks.


### Repository instructions (`AGENTS.md`)


- Use official TensorFlow exports for **all** variants, including Base.
- `dino` = DINOv2-B; `siglip` = SigLIP-B.
- UnAligNet = official **`untransformed`** export.
- Keep preprocessing consistent within each family and document provenance.
- Treat `pre_logits` and `triplet_logits` as different representations.
- Keep checkpoints in ignored `models/` and generated data/results in ignored `outputs/`.
- Generated smoke images or synthetic software fixtures are not benchmark evidence.
- Use **uv**, Python **3.12**, the existing `pyproject.toml`, and generated `uv.lock`.
- Use `uv sync --locked` and `uv run --locked` once the lock exists. Do not handwrite lockfiles.
- **Do not install heavy dependencies, download models, or run inference locally unless explicitly requested.** The user runs GPU code elsewhere. Prior script/run-command requests did not authorize local GPU inference.
- State checks actually run; do not claim runtime compatibility without a smoke run.
- Keep implementation small and preserve research documents unless asked to revise them.
- Do not commit credentials, checkpoints or downloaded datasets.


### Collaboration preferences


Work autonomously on authorized tasks; avoid repetitive confirmation requests. Give concise progress updates. No subagents were authorized for this work. Do not send emails or messages to dataset authors without explicit authorization. Downloading public data and examining code has been within scope.


## 3. Git state and what was published


| Item             | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| Upstream/origin  | `https://github.com/lelouch0204/human-alignment`                                                           |
| Fork remote      | `https://github.com/hizv/human-alignment.git`                                                              |
| Branch           | `experiment/navon-human-alignment`                                                                         |
| Starting commit  | `666ea03` — existing Navon generation/inference/notebook pipeline                                          |
| First new commit | `879e3c4` — frozen-backbone human letter-response pilot, tests, SLURM script, documentation                |
| Latest commit    | `218e1a1` — recovered shape-distance dataset pipeline, SciPy dependency, tests, updated SLURM job and docs |
| Pushed branch    | https://github.com/hizv/human-alignment/tree/experiment/navon-human-alignment                              |


Both new commits were pushed and the remote HEAD was verified via GitHub API in the prior implementation turn. No PR was created. No fresh remote verification was necessary for this documentation snapshot.


### Uncommitted state verified on 2026-11-20, before writing this handoff


Modified tracked files:


- `README.md`: earlier experiment sections beyond the committed shape-pilot section.
- `pyproject.toml`: earlier `analysis` dependency group.
- `uv.lock`: corresponding analysis-group resolution.


Untracked files:


- `experiments/clean-recognition.md`
- `experiments/navon-followup.md`
- `experiments/navon-followup-results.md`
- `experiments/navon-transfer.md`
- `experiments/navon-transfer-results.md`
- `src/human_alignment/clean_recognition.py`
- `src/human_alignment/navon_followup.py`
- `src/human_alignment/navon_transfer.py`
- `tests/test_clean_recognition.py`
- `tests/test_navon_followup.py`
- `tests/test_navon_transfer.py`
- `s41586-025-09631-6.pdf` — existing user-supplied paper; do not delete or casually commit.


This handoff is an additional new file. Only stage relevant changes if a future user requests a commit. Do not use `git add .` or reset the workspace to clean it.


**Dependency distinction:** the pushed commit includes SciPy as a normal project dependency. The additional CPU `analysis` group exists only in the current uncommitted working tree. Commands using `--only-group analysis` therefore apply to this workspace, not necessarily a fresh fork checkout.


## 4. Model and conceptual conventions


### Model loading


`src/human_alignment/models.py` handles official export discovery, caching, loading and preprocessing.


- Families: `dino -> DINOv2-B`, `siglip -> SigLIP-B`.
- Variants: `base -> base_model`, `alignet -> alignet`, `unalignet -> untransformed`.
- Publisher index: https://storage.googleapis.com/alignet/models/index.html
- Serving input: `images`; float32 RGB NHWC in [0,1].
- `load_images`: PIL EXIF transpose, RGB conversion, OpenCV `INTER_CUBIC` direct resize, divide by 255.
- SigLIP released configuration supports the preprocessing choice. **Original DINO training preprocessing provenance remains unresolved.** Avoid explaining family differences as if preprocessing were fully settled.
- `pre_logits` is primary. `triplet_logits` is secondary; the publisher describes Base's triplet projection as untrained.


### “Global” has two meanings here


- **Navon spatial global:** the overall large letter/shape, e.g. a large H made from small Ts.
- **Navon spatial local:** the constituent small Ts.
- **AligNet semantic global organization:** relationships across natural-object concepts and broader semantic categories.


AligNet's original recipe aligns a teacher with human natural-object similarity judgments, then distils teacher similarity targets into student representations using sampled image triplets. Human alignment to those judgments does not imply human-like behavior on every attention or spatial grouping task.


Authoritative references:


- https://github.com/google-deepmind/alignet
- https://www.nature.com/articles/s41586-025-09631-6
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12611773/
- Local paper: `s41586-025-09631-6.pdf`.


Possible explanations for earlier local-heavy geometry include task/supervision mismatch and readout dependence. These are hypotheses, not demonstrated causal mechanisms. Lower decoding accuracy does not establish lost information.


## 5. Completed letter-Navon analyses


### Inputs and reproducibility


Public dataset: https://huggingface.co/datasets/lelouch0204/navon-alignet


Pinned revision: **`d334c56a3f0023e25860e55114eb2e9ec5deb3e8`**.


No Hugging Face login was needed. Public metadata and six embedding parquet files were downloaded; no checkpoints were downloaded or run locally.


- 4,264 stimuli total: 4,056 compound images and 208 isolated-letter images.
- Compound design: 26×26 letter combinations × three coupled size/spacing conditions × two positions.
- 3,900 incongruent compounds used for decoding.
- Conditions: s10, s14, s18; size and spacing change together, so these do not isolate a pure scale mechanism.
- Each of six checkpoint files includes `pre_logits` (768 dimensions) and `triplet_logits` (1,024 dimensions).
- Inputs: `outputs/navon_followup/data/manifest.json`, `embeddings-{family}_{variant}.parquet`, `source_summary_metrics.json`.
- 48 original compound-margin summaries reproduced with maximum absolute discrepancy approximately **1.11e-9**.


### A. Distance decomposition and nuisance-held-out ridge decoding


Files:


- `src/human_alignment/navon_followup.py`
- `experiments/navon-followup.md`
- `experiments/navon-followup-results.md`
- `tests/test_navon_followup.py`


Global-change distance compares C(X,Z) versus C(Y,Z); local-change distance compares C(Z,X) versus C(Z,Y), for distinct X,Y,Z. Metric: one minus cosine similarity.


Primary representation results:


| Checkpoint       | Global-change distance | Local-change distance | Global decoding, broad alpha grid |
| ---------------- | ---------------------: | --------------------: | --------------------------------: |
| DINO Base        |                 0.0290 |                0.0539 |                            95.97% |
| DINO UnAligNet   |                 0.0032 |                0.0094 |                            79.49% |
| DINO AligNet     |                 0.0196 |                0.0780 |                            72.49% |
| SigLIP Base      |                 0.0945 |                0.1973 |                            51.79% |
| SigLIP UnAligNet |                 0.0480 |                0.1129 |                            54.36% |
| SigLIP AligNet   |                 0.0678 |                0.1960 |                            48.08% |


Primary local decoding: 99.72–100%. Secondary global decoding, in Base/UnAligNet/AligNet order: DINO 95.77/78.74/70.44%; SigLIP 45.72/53.95/47.85%.


Protocol: unit-normalized embeddings, RidgeClassifier with intercept, five outer nuisance-letter folds, three inner nuisance-group folds. Global decoding holds out local identities; local decoding holds out global identities. All target identities remain represented during training. All rendering conditions occur in training in this experiment.


**Critical correction: tuning changed the apparent result.** Initial alphas `[0.1,1,10,100]` gave AligNet-minus-UnAligNet global accuracy gains of +5.15 pp (DINO) and +6.95 pp (SigLIP) for `pre_logits`. Many fits hit the smallest alpha. The broader exploratory grid, powers of ten from 1e-6 to 1e4, gave **−7.00 pp and −6.28 pp**, respectively. Some SigLIP folds still hit the broader lower boundary. Report tuning sensitivity; do not selectively cite the initial result or claim optimal linear capacity.


Interpretation: both global- and local-change distances increase from UnAligNet to AligNet, but local distances increase substantially more. Global identity remains decodable. Geometry, accessibility and human agreement are distinct claims.


Saved runs:


- `outputs/navon_followup/`: initial grid, both representations.
- `outputs/navon_followup/sensitivity/`: broader grid, primary representation.
- `outputs/navon_followup/sensitivity_triplet/`: broader grid, secondary representation.
- Audits: `verification.json`, `source_reproduction_check.json`; 187,200 predictions across three runs audited.


### B. Rendering transfer


Files:


- `src/human_alignment/navon_transfer.py`
- `experiments/navon-transfer.md`
- `experiments/navon-transfer-results.md`
- `tests/test_navon_transfer.py`


Train on two rendering conditions and test the third, additionally holding out nuisance-letter groups. Inner validation crosses three nuisance folds with both transfer directions between source conditions; broad alpha grid. The same fitted classifier predicts matched in-domain controls on both source renderings.


| Checkpoint       | Global in-domain | Global transfer | Local transfer |
| ---------------- | ---------------: | --------------: | -------------: |
| DINO Base        |           94.54% |          86.72% |         96.41% |
| DINO UnAligNet   |           79.92% |          55.44% |         95.74% |
| DINO AligNet     |           72.29% |          52.97% |         96.36% |
| SigLIP Base      |           43.41% |          37.08% |         98.26% |
| SigLIP UnAligNet |           46.12% |          39.41% |         98.64% |
| SigLIP AligNet   |           45.87% |          37.26% |         99.31% |


AligNet has lower aggregate global transfer accuracy than UnAligNet, but:


- DINO AligNet's transfer drop is **5.17 pp smaller**; lower absolute accuracy does not mean greater rendering sensitivity.
- SigLIP AligNet's transfer drop is **1.91 pp larger**.
- SigLIP improves relative to UnAligNet for held-out s14 (+2.54 pp), so the pattern is not uniform.
- Secondary global transfer: DINO 85.41/52.38/51.44%; SigLIP 36.28/40.31/38.05%.


Outputs: `outputs/navon_transfer/`, including `report.md`, prediction/tuning/contrast CSVs, figures, `run_record.json`, `verification.json`, and `audit.py`.


Audit covered 280,800 predictions, 360 outer fits and 93,600 matched domain comparisons. No human judgments accompany these generated letter stimuli. No novel fonts or unseen target identities were tested. Images and folds are dependent; no population significance claims were made.


### Local CPU reproduction commands


These need the **local uncommitted analysis group and scripts**:


```bash
uv sync --locked --only-group analysis
uv run --locked --only-group analysis python src/human_alignment/navon_followup.py --download
uv run --locked --only-group analysis python src/human_alignment/navon_followup.py \
 --data-dir outputs/navon_followup/data \
 --out-dir outputs/navon_followup/sensitivity \
 --representations pre_logits \
 --alphas 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1 10 100 1000 10000
uv run --locked --only-group analysis python src/human_alignment/navon_transfer.py \
 --representations pre_logits triplet_logits
```


For secondary broad-grid decoding, use `--representations triplet_logits` with `--out-dir outputs/navon_followup/sensitivity_triplet`. Preserve previous outputs rather than blindly rerunning into completed directories.


## 6. Clean-image recognition: implemented but blocked on inputs


Proposal question 3 asks whether UnAligNet improves clean object recognition over Base. AligNet-versus-UnAligNet is a secondary alignment contrast.


Files, currently uncommitted: `src/human_alignment/clean_recognition.py`, `experiments/clean-recognition.md`, `tests/test_clean_recognition.py`.


Method:


- Same 16 categories as the shape–texture benchmark: airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog, elephant, keyboard, knife, oven, truck.
- Average **raw** embeddings of separate reference images per category, normalize prototypes, and classify unit-normalized test embeddings by cosine similarity.
- Reference and test images must be disjoint. Exclude clean/conflict benchmark source images and their derivatives from references, including shape and texture sources.
- Same images/labels/splits for all checkpoints; representations analyzed separately.
- Macro category accuracy and paired category-stratified source bootstrap, 5,000 draws, seed 2026. References fixed; no reference-selection uncertainty or multiplicity correction.
- Manifest validation requires source IDs, provenance, hashes and both splits for every category. It cannot infer undeclared transformed duplicates from hashes alone.


Candidate benchmark: https://github.com/bethgelab/model-vs-human/releases/tag/v0.1


Category mapping pinned at repository revision `79ed7cd1d9ac3a114e71b6638f0a98bdd627316c`, file `modelvshuman/helper/human_categories.py`.


The prototype readout differs from the benchmark's original classifier-probability aggregation. Do not directly equate scores across readouts.


**User confirmed they do not have existing clean embeddings or an ImageNet reference subset.** ImageNet specifically is not mandatory, but independent correctly labelled references and clean test images are required. No recognition benchmark was run, and no clean-image result exists. Navon embeddings cannot answer this question. This task has not been canceled, but the subsequent active work shifted to the human shape dataset.


## 7. Human dataset search and corrections


### LoGlo


Repository: https://osf.io/xtrhe/


Associated papers:


- https://pmc.ncbi.nlm.nih.gov/articles/PMC12627199/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13364934/


OSF webpages initially returned 403 in the web tool. Public API requests worked via network-enabled curl. No sign-in was needed. Do not interpret a failed webpage fetch as unavailable data.


The top-level **`dataset descriptions.xlsx` is a study catalogue**, not participant trial responses. Dataset components must be inspected separately.


FXLE component: https://osf.io/z39j2/


- **FXLEa:** `FXLEa_trials.csv`, 15,600 rows, including 1,200 practice and 14,400 test rows. Columns include local/global letter, `level` (`glo`/`loc`), response, correct response, ACC, RT, participant/session and exclusion flags.
- Four H/T compound image assets are available with original browser experiment code. Filename order is **local first, global second**: `navon_h_t.png` means small Hs forming a large T.
- Observed exclusion flags include `include`, `exclude`, and `consider`; decisions need an explicit documented policy.
- Four distinct images do not support our intended stimulus-disjoint train/validation/test study. Splitting repeated trials or participants while retaining the same images does not demonstrate new-image transfer.
- **FXLEb1:** Kimchi–Palmer similarity/preference task. Downloaded CSV has 4,800 rows and maps reference/left/right local/global identities plus local/global key mappings. Actual values in the inspected CSV use square and triangle identities. Image assets are listed. This is a different response objective, and limited variety remains a constraint.


No verified XLSX/CSV-to-training importer was implemented for these LoGlo datasets. Do not claim the legacy training script is connected to LoGlo.


### Other candidates


- **COSNaT:** https://osf.io/ankzv/ ; paper https://pmc.ncbi.nlm.nih.gov/articles/PMC12431921/ . Verified experiment ZIP and behavioral workbook listings. Six incongruent E/H/S combinations; does not resolve stimulus diversity. Workbook trial-level schema was not established.
- **THINGS:** https://things-initiative.org/ ; https://osf.io/f5rn6/ . Approximately 4.7 million human triplet judgments over 1,854 object images. Appropriate to original AligNet supervision, not a spatial Navon dataset; AligNet already uses this signal.
- Experiment demos such as PsyToolkit/Testable are stimulus/experiment tools, not automatically published human training datasets.


### Number-based Navon / pupil study: still incomplete


Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12417611/
Data: https://osf.io/5dqcn/


Paper reports 36 distinct digit/brightness stimuli and 30 participants. Public data include EyeLink ASC recordings, Python analysis code, and JASP summaries. The native storage, providers, child components and wiki were checked. No exact original image assets or generation code were located.


One downloaded recording has 145 trial event groups, including block 99. It contains condition, global/local colour and number, response, RT, gaze validity, and catch-trial metadata. Digits observed: 2,3,5,6,8,9. **`cor_resp == 0` means correct** in the authors' code. Do not invert this mapping.


Local copies: `outputs/navon_data_audit/pupil/navon_sbj_1.asc`, `osf_main.py`. Do not invent exact pixels from digit identities and present them as original stimuli. This candidate is secondary now that the shape dataset is recovered.


## 8. Recovered Jacob & Arun dataset: current active pipeline


### Availability resolved


- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7594584/
- OSF: https://osf.io/zt8k3/
- Manuscript Codes component: https://osf.io/4b2rp/
- **Author GitHub:** https://github.com/georginjacob/Multiscale
- Pinned commit: **`eabd49541b10745fd2469e4137acefbc859a5a3d`**.


**Correction to earlier search:** native OSF storage was empty, but the code component exposes a separate **GitHub provider**. Following `/v2/nodes/4b2rp/files/` revealed it. Enumerating `/files/github/` and all pagination recovered the data. Both downloaded MAT files matched the pinned GitHub blob hashes.


| File                        | Verified contents                                                                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FigureCodes/L2_VSmain.mat` | 49 uint8 grayscale images, each 400×400; all 1,176 unordered pairs; eight participants; RT array 1,176×8×2; pairwise accuracy and task metadata                       |
| `FigureCodes/L2_SD.mat`     | 49 uint8 grayscale images, each 150×150; 16 participants; global and local same–different RT arrays each 588×16×2; pairwise accuracy, image-pair and shape identities |


MAT root variable: `L2_str`.


Visual-search fields include `images`, `img_pairs` (MATLAB one-based), `Image_Pair_Details`, `RT.trial_wise`, `RT.subject_wise.Mean`, `PC.image_pairs`, and `task`.


Same–different fields include `images`, `stim_details` (global shape, local shape), `imgpairs.global/local`, `RT.global/local`, `trialDetails`, and `PC.pairwise`.


Visual-search pair IDs cover 1–49, with 1,176 unique unordered pairs. All 18,816 RT entries are finite and positive, approximately 0.386–9.940 seconds. This integrity check does not reconstruct original trial-exclusion decisions. The files contain participant metadata; avoid exposing personal fields. Export anonymous axes only.


Additional listed but not audited/downloaded experiment files: `L2_GLIE.mat`, `L2_IEgrouping.mat`, `L2_IEpos2.mat`, `L2_IEpossizenum.mat`. They may support broader tests involving position, size, count, grouping or different shape arrangements; do not assume they are interchangeable with the 49-image task.


### Download fingerprints


- VS MAT SHA256: `2c4b72f6fe9f37c1848bb1023c67d2b27e0a275171784cd884cd91f7349f86c7`.
- SD MAT SHA256: `b74e9c74111655578f46ac9ededc549945a427fb65e1c8b3183b49a6dc9c704d`.
- Canonical VS download: `https://raw.githubusercontent.com/georginjacob/Multiscale/eabd49541b10745fd2469e4137acefbc859a5a3d/FigureCodes/L2_VSmain.mat`.
- Audit copies: `outputs/navon_data_audit/shapes/`.
- Source/hash record: `outputs/navon_data_audit/sources.json`.
- Audit document: `experiments/navon-dataset-audit.md`.


### Implemented behavior (`navon_shapes.py`)


Commands: `prepare`, `embed`, `analyze`.


1. **Prepare:** fetch pinned VS MAT only with `--download-data`; verify SHA256; extract all 49 images to PNG; write `human.npz` and a hash manifest. Existing prepared data are validated and reused so NPZ regeneration does not invalidate embedding caches.
2. **Human target:** average the two search directions per participant, then average eight participants, then invert: **`1 / mean(RT)`**. Do not use `mean(1/RT)`. Direction averaging was checked against the author's `figure_04.m`. No new trial exclusions are applied to the released finite consolidated RTs.
3. **Embed:** use official checkpoints and shared preprocessing, one checkpoint per process. Cache both representations, image order, family, variant and manifest hash. Reject mismatched caches. Record export fingerprint, TF version and preprocessing.
4. **Frozen evaluation:** unit-normalized embedding Euclidean distances versus human distances; Pearson and Spearman for six checkpoints × two representations = 12 comparisons.
5. **Adaptation:** Base only, each family and representation separately. Fit nonnegative diagonal metric weights on squared differences of unit embeddings; shrink weights toward one. This gives four adapted settings.


### Exact adaptation protocol


For pair features `x_ij = (z_i - z_j)^2` with unit z:


- Training-only scale `xs = mean(sum(x_ij))` and `ys = mean(human_distance)`.
- Minimize `mean((x @ w / xs - y / ys)^2) + lambda * mean((w - 1)^2)` with `w >= 0` using SciPy L-BFGS-B.
- Transform interpretation: `z * sqrt(w)`, without renormalization; squared distance predictor scaled by `ys/xs`.
- Seven deterministic image folds, random seed 2026: seven test images, seven separate validation images, 35 train images.
- Only within-training pairs fit candidates; only within-validation pairs select lambda from `[0.01,0.1,1,10]` by normalized MSE.
- Refit on all 42 non-test images including cross-pairs between the previous train/validation sets.
- Evaluate only pairs with **both endpoints in the seven test images**: 21 per fold, 147 total.
- Save weights, scale constants, split IDs, selected lambda, validation scores and matched frozen/adapted predictions.
- Frozen control inside adaptation uses squared Euclidean distances. Its Spearman equals Euclidean-distance Spearman; Pearson need not equal the main frozen evaluation.
- Report descriptive mean fold Spearman, not an independent-pair significance result.


### Scientific limits


Only 49 underlying images; 1,176 pairs are dependent. No inferential confidence intervals currently implemented. Generalization across participants is not tested by the image split; the same participant population supplies train/test distance targets. Same–different images have different resolution and task demands; do not assume pixel-identical cross-task data. Visual-search single-item images omit full search-array context. RT-derived dissimilarities are not measured triplet-choice probabilities. Generating triplets would introduce an explicit modeling assumption.


## 9. Current cluster commands


For a fresh checkout on the user's GPU machine:


```bash
git clone -b experiment/navon-human-alignment \
 https://github.com/hizv/human-alignment.git navon-human
cd navon-human
uv sync --locked
uv run --locked python -m human_alignment.navon_shapes prepare --download-data
mkdir -p outputs/navon_human
sbatch --export=ALL,NAVON_DOWNLOAD_MODELS=1 run_navon_human.sbatch
```


If already on that branch with the fork tracking remote, update with an appropriate non-destructive pull/fetch after checking local changes. Do not blindly reset a user's checkout.


`NAVON_DOWNLOAD_MODELS=1` permits downloads of missing checkpoints on the cluster. Omit it if models are already cached. This is not permission to run those downloads locally in the chat workspace.


SLURM configuration, based on original `run_navon.sbatch`:


- partition `coe-gpu`; one GPU; four CPUs; 32 GB RAM; **two-hour limit**.
- logs `outputs/navon_human/slurm-%j.out`; create directory **before submission**.
- original uv account-root default `/storage/ice1/4/5/nparikh44`, overridable with `NAVON_UV_ROOT`.
- existing `UV_*` variables take precedence; uv executables prepended to PATH.
- default XLA flag `--xla_gpu_strict_conv_algorithm_picker=false`.
- job uses `uv run --locked --no-sync`; dependencies installed separately.
- each model embedded in fresh process; then CPU analysis.


| Environment variable     | Default                             |
| ------------------------ | ----------------------------------- |
| `NAVON_MODELS`           | `models`                            |
| `NAVON_SHAPES_DATA`      | `outputs/navon_shapes/data`         |
| `NAVON_SHAPES_CACHE`     | `outputs/navon_shapes/embeddings`   |
| `NAVON_SHAPES_OUT`       | `outputs/navon_shapes/run-<job ID>` |
| `NAVON_HUMAN_BATCH_SIZE` | `16`                                |
| `NAVON_DOWNLOAD_MODELS`  | off unless `1`                      |


Outputs: `report.json`, 12 frozen `*-distances.npz`, four `*-weights.npz` (seven fold metrics each), and four `*-predictions.json` files. Output directory must be empty. Avoid concurrent jobs writing the same embedding cache.


Guide: `experiments/navon-shapes.md`.


## 10. Legacy human letter-response pilot


Committed in `879e3c4` and still available: `src/human_alignment/navon_human.py`, `tests/test_navon_human.py`, `experiments/navon-human-pilot.md`.


- Frozen official Base backbone, default DINO, primary `pre_logits`.
- JSON manifest with source-disjoint train/validation/test image splits, paths, hashes, global/local identities and response counts by instruction.
- Two instruction-specific linear softmax heads on unit embeddings, with intercepts.
- Human objective: observed response-frequency cross-entropy. Control: correct-letter one-hot cross-entropy.
- Equal stimulus/instruction weighting; matched seeds, architecture, optimizer, budget.
- Full-batch Adam, learning rate .03, L2 .001, 500 updates, seeds 0/1/2.
- Best epoch selected on validation targets for each objective.
- Metrics: human cross-entropy/KL, true-letter accuracy, expected human agreement; global/local and congruent/incongruent strata.
- Tested only with synthetic software fixtures; no real human-data training run.


Its original SLURM instructions are superseded; the document now has a notice pointing to the shape experiment. Do not pass `NAVON_HUMAN_MANIFEST` expecting the current SLURM job to use it.


## 11. Validation and environment history


Previous implementation turn actually ran:


- 23 local unit tests total: follow-up 5, transfer 4, clean recognition 5, legacy human pilot 5, shape pipeline 4.
- Shape tests: image-disjoint folds, invariance to positive image rescaling after normalization, anisotropic metric recovery, and unchanged outer-fold fit when its test targets are altered.
- Real-source MAT hash checks and lossless image extraction.
- Prepare idempotency / stable manifest fingerprint check.
- Synthetic six-cache end-to-end analysis: 12 frozen comparisons, four adaptations, 147 held-out pairs each. Those synthetic outputs were temporary and are **not model results**.
- `bash -n run_navon_human.sbatch` and `git diff --check`.
- Pushed commit verified via GitHub API.


This documentation turn read repository state and existing result documents; it did not rerun inference or claim new experiment results.


Local execution environment history:


- No local checkpoints or TensorFlow inference were used during the work.
- Lightweight uv binary was installed temporarily at `/tmp/navon-uv/bin/uv`; cache `/tmp/navon-uv-cache`.
- Local CPU commands used `UV_CACHE_DIR=/tmp/navon-uv-cache /tmp/navon-uv/bin/uv run --locked --only-group analysis ...`.
- Matplotlib cache used `/tmp/navon-matplotlib`.
- `/tmp` helpers may disappear; don't depend on them on another machine.
- `uv sync --only-group analysis` removes packages outside that group; use normal `uv sync --locked` on the inference machine.
- GitHub CLI was absent from PATH; a temporary binary at `/tmp/navon-gh/bin/gh` used existing **keyring** authentication for account `hizv`. Never expose tokens or keyring contents. Network-enabled auth worked; sandboxed DNS failure initially looked like auth failure.
- Git metadata and network operations required sandbox escalation. Branch creation, commits, fork creation and pushes succeeded. No permission rejection remains pending.


## 12. Priorities for the next chat


1. **Determine whether the user has run the SLURM job and obtained results.** No such results are in this conversation. Inspect provided logs/report rather than inventing conclusions.
2. If troubleshooting, verify Python/dependencies, official checkpoint signatures, cluster paths/resources and cache hashes first. Runtime compatibility remains unverified until actual inference.
3. Once results exist, interpret all six frozen checkpoints separately from the four Base adaptations. Compare paired held-out predictions and inspect folds rather than reporting just one aggregate score.
4. Add uncertainty analysis that respects participants and shared images if research claims require it. Do not treat pairs/folds as independent trials.
5. Consider the additional shape/position/grouping files or same–different task only after auditing mappings, task differences and split leakage.
6. Keep the clean-recognition experiment's missing reference/test data visible as unfinished work; do not substitute Navon stimuli for object categories.
7. Do not accidentally omit or overwrite the earlier uncommitted letter-Navon analyses when changing branches or preparing a PR.


## 13. Quick reading order


1. `AGENTS.md` — binding local project instructions.
2. `experiments/navon-shapes.md` — active run instructions and methods.
3. `src/human_alignment/navon_shapes.py` and `run_navon_human.sbatch` — implementation.
4. `experiments/navon-dataset-audit.md` — dataset availability evidence; its proposed next-step wording predates the completed importer.
5. `experiments/navon-followup-results.md` and `experiments/navon-transfer-results.md` — completed letter-Navon results, local uncommitted files.
6. `proposal.md` — research aims, especially clean object recognition and disjoint-reference requirements; preserve it.
7. `experiments/clean-recognition.md` — pending clean recognition pipeline.


## 14. Suggested opening context for another chat


> Continue the human-alignment research project using CHAT_HANDOFF.md. The active pushed branch is experiment/navon-human-alignment at 218e1a1 in hizv/human-alignment. The recovered 49-shape dataset pipeline is implemented and tested, but actual model inference results have not yet been provided. Earlier completed letter-Navon analyses and the clean-recognition pipeline remain uncommitted locally. Preserve that work, obey AGENTS.md, and do not run model downloads or inference locally without explicit authorization. Distinguish the current shape-distance script from the older letter-response script, and do not assume the downloaded AligNet fails the new human-distance task.



