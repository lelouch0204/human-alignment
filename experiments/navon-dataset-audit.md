# Human local–global dataset availability audit

Checked 2026-11-18. No model inference or training was run.

## Recovered: Jacob & Arun multiscale-shape data

The earlier empty-listing result was incomplete: the main OSF project has no
native stored files, but its Manuscript Codes component connects to **GitHub
storage**. Following that provider resolves availability.

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7594584/
- OSF project: https://osf.io/zt8k3/
- Code component: https://osf.io/4b2rp/
- Author repository: https://github.com/georginjacob/Multiscale
- Verified commit: `eabd49541b10745fd2469e4137acefbc859a5a3d`

Downloaded `FigureCodes/L2_VSmain.mat` and `FigureCodes/L2_SD.mat` via the OSF
GitHub provider and verified their Git blob hashes against the pinned repository.
Both files were successfully opened with SciPy. They are under ignored
`outputs/navon_data_audit/shapes/`. Download URLs, SHA256 hashes and file sizes
are recorded in `outputs/navon_data_audit/sources.json`.

### Actual contents verified

| File | Images | Human measurements |
| --- | --- | --- |
| `L2_VSmain.mat` | 49 grayscale uint8 arrays, each 400×400 | Eight participants; all 1,176 unordered image pairs; RT array 1,176×8×2, plus participant-level RT summaries and pairwise accuracy |
| `L2_SD.mat` | 49 grayscale uint8 arrays, each 150×150 | 16 participants; separate global/local same–different task RT arrays, each 588×16×2; pairwise accuracy, image IDs and global/local shape identities |

Visual-search pair IDs cover 1–49 and contain all 1,176 distinct unordered pairs.
All 18,816 RT entries are finite and positive, ranging from approximately 0.386
to 9.940 seconds. This is an array-integrity check, not a validation of the
original authors' trial exclusion rules. The task includes search-array context;
the stored single-item images do not encode the entire displayed search array.
The files also contain participant metadata, which was not printed or exported
in this audit. Preserve the anonymous participant axis for later uncertainty
analysis without exposing personal fields.

The repository additionally lists these consolidated experiment files:
`L2_GLIE.mat`, `L2_IEgrouping.mat`, `L2_IEpos2.mat`, `L2_IEpossizenum.mat`.
The README associates them with additional shape, position, size, element-count
and grouping experiments. Those files were not downloaded or validated here;
they should not be counted as additional matched training stimuli yet. CSV
exports and MATLAB/R analysis code are also present.

### Implications for our experiment

This is usable for human dissimilarity alignment and a baseline evaluation of
our six existing checkpoints. The published visual-search approach uses inverse
mean search time as a perceptual dissimilarity measure. Confirm averaging,
exclusions, and direction handling against the original figure code before
constructing targets. These are task-derived distances, not observed odd-one-out
choice probabilities; a triplet conversion would introduce a modeling assumption.

Start by comparing frozen embedding distances with human pairwise dissimilarities.
Then consider a strongly regularized, small representation transform, selecting
hyperparameters within training data and holding out entire stimulus identities.
A random pair split would share images across training and test. The 1,176 pairs
are not 1,176 independent images. Bootstrap participant/source units appropriately;
treat cross-task evaluation as a separate question, since same–different and
visual-search measurements use different task instructions and image resolutions.

The existing `navon_human.py` predicts letter responses. It cannot consume these
pairwise measurements unchanged: this experiment needs a separate distance-based
objective/evaluator, not fabricated letter-response counts.

## Partially recovered: number-based Navon / pupil study

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12417611/
- Repository: https://osf.io/5dqcn/

Verified native OSF storage, child-component listing, and project wiki. The
release contains EyeLink ASC recordings, a Python analysis script, and JASP
summaries. No alternate storage provider or child component was listed.
Downloaded one recording and the analysis script to ignored
`outputs/navon_data_audit/pupil/`.

The sample recording contains 145 trial event groups, including a block numbered
99. Metadata include `condition`, `global_colour`, `local_colour`,
`global_number`, `local_number`, `key_resp`, `cor_resp`, `reaction_time`,
`valid_gaze`, and catch-trial fields. Observed digits are 2, 3, 5, 6, 8, and 9.
These fields are also explicitly used in the authors' parser. The analysis
script documents `cor_resp == 0` as correct, so blindly treating that field as a
standard Boolean accuracy would invert the labels.

The paper reports 36 distinct stimuli. Exact source image assets or generation
code were not located in the inspected release. Identity metadata do not recover
the exact pixels, contrast balancing, or catch-trial temporal changes. Therefore
this dataset remains unsuitable for a verified image-response training manifest
until those materials are recovered. Do not substitute newly drawn digits and
present them as the original human stimuli.

## Recommendation

Prioritize the recovered 49-shape visual-search release. It now has verified
images, pair identifiers and participant-level human measurements. The next
step is an importer and frozen-checkpoint human-distance evaluation, followed
by a small adaptation experiment if the data support a reliable held-out test.
Keep the pupil study secondary while original image availability is unresolved.
