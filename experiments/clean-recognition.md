# Clean-image recognition: proposal question 3

Status: implementation prepared; no clean-image inference or benchmark accuracy
has been run. This workspace has Navon embeddings, but no object-image reference
set, clean test set, or model checkpoints. Navon embeddings cannot answer this
question. The completed rendering-transfer experiment is documented in
[navon-transfer-results.md](navon-transfer-results.md).

## Question and fixed method

Primary contrast: UnAligNet minus Base balanced accuracy, separately for DINOv2-B
and SigLIP-B. AligNet minus UnAligNet provides the alignment contrast.
Use official SavedModels, including Base; UnAligNet is `untransformed`.
`pre_logits` is primary; `triplet_logits` is a separate secondary analysis.

Use the 16 categories in the official
[model-vs-human category mapping](https://github.com/bethgelab/model-vs-human/blob/79ed7cd1d9ac3a114e71b6638f0a98bdd627316c/modelvshuman/helper/human_categories.py):
airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog, elephant,
keyboard, knife, oven, truck. The published clean `original` condition is a
candidate test set; data are available through the project's
[release](https://github.com/bethgelab/model-vs-human/releases/tag/v0.1).
Supply independent clean reference images from these same categories, with
verified category mappings. Fix the reference sample before examining results;
prefer the same number of independent reference sources per category.

For each checkpoint, average **raw** reference embeddings within category, then
normalize each prototype to unit length. Normalize test embeddings and choose
the largest cosine similarity. No fitted classifier, tuning, or test-set selection.
Exact ties use the alphabetic category order. This implements the proposal's
prototype readout; published model-vs-human classifier-head accuracies use a
different readout and should not be presented as directly matched results.

All checkpoints use identical images and splits. Images use the repository's
`models.load_images`: EXIF correction, RGB conversion, direct OpenCV bicubic
resize to export input dimensions, and float32 scaling to [0,1]. This matches our
existing loader. SigLIP's released configuration supports this preprocessing;
DINO's original training preprocessing provenance remains unresolved. This is a
limitation when interpreting family differences.

Report macro-average category accuracy and per-image predictions. Compute paired
95% percentile intervals for each contrast using 5,000 category-stratified
bootstrap draws of test **source IDs**, retaining every derivative of a sampled
source. References remain fixed: intervals do not include uncertainty from
choosing a new reference set. These are exploratory intervals without a
multiple-comparison correction. Inspect category-level errors from predictions.csv.

## Inputs and leakage audit

Create a JSON manifest under ignored `outputs/clean_recognition/`. Paths are
relative to that manifest (absolute paths also work). Required structure:

```json
{
  "dataset_provenance": "Dataset names, versions, split choices and category mapping",
  "source_audit": "How original sources and derivatives were identified and excluded",
  "benchmark_source_ids": ["canonical-source-001"],
  "images": [
    {
      "image_id": "reference-airplane-001",
      "path": "images/reference-airplane-001.jpg",
      "sha256": "actual SHA256 of image bytes",
      "source_id": "independent-source-001",
      "category": "airplane",
      "split": "reference",
      "provenance": "Original dataset identifier and source image identifier"
    }
  ]
}
```

The example is a schema illustration, not a runnable dataset. Include reference
and test images for every category. `benchmark_source_ids` must contain sources
from the clean and cue-conflict benchmark, including **both shape and texture
sources**. Derivatives share their original's canonical source ID. Do not put
any of these sources in the reference set. A random split of clean benchmark
images is insufficient if those same images supplied cue-conflict sources.

Validation checks hashes, duplicate bytes, unique image IDs, all categories in
both splits, source consistency, and reference exclusion. It cannot infer
undeclared relationships from pixels: review the source registry and provenance
before running. Hash checks do not establish absence of transformed duplicates.

## Execution on the inference machine

From repository root, with the manifest and images available:

```bash
uv sync --locked
uv run --locked python -m human_alignment.clean_recognition validate \
  --manifest outputs/clean_recognition/manifest.json
for family in dino siglip; do
  for variant in base unalignet alignet; do
    uv run --locked python -m human_alignment.clean_recognition embed \
      --manifest outputs/clean_recognition/manifest.json \
      --family "$family" --variant "$variant"
  done
done
uv run --locked python -m human_alignment.clean_recognition analyze \
  --manifest outputs/clean_recognition/manifest.json
```

Each command loads one model in a fresh process. Checkpoints must already be in
`models/`; add `--download` to `embed` only when downloads are wanted. Adjust
`--batch-size` for GPU memory. Do not modify the manifest between checkpoints.
Artifacts go to ignored `outputs/clean_recognition/`: six embedding NPZ files,
checkpoint/preprocessing metadata, predictions.csv, and report.json with paired
intervals and input hashes. Each embedding file includes both representations.

Analysis can also run without TensorFlow using the existing analysis group:

```bash
uv run --locked --only-group analysis python src/human_alignment/clean_recognition.py analyze \
  --manifest outputs/clean_recognition/manifest.json
uv run --locked --only-group analysis python -m unittest discover -s tests -p 'test_clean_recognition.py'
```

Local validation covers prototype arithmetic, paired inference on identical
predictions, rejection of leaked source IDs and altered bytes, and zero vectors.
Synthetic tests are software checks, not recognition evidence. TensorFlow runtime
compatibility and real-data performance remain unverified until the GPU run.
