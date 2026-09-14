# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Identify technosignature signals in cadence snippets taken from a digital spectrometer.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
00034abb3629,0.5
0004be0baf70,0.5
0005be4d0752,0.5
etc.

```

## Dataset
The data is from a digital spectrometer, which takes incoming raw data from the telescope (amounting to hundreds of TB per day) and performs a Fourier Transform to generate a spectrogram. These spectrograms, also referred to as filterbank files, or dynamic spectra, consist of measurements of signal intensity as a function of frequency and time.

Below is an example of an FM radio signal. This is not from the GBT, but from a small antenna attached to a software defined radio dongle (a $20 piece of kit that you can plug into your laptop to pick up signals). The data we get from the GBT are very similar, but split into larger numbers of frequency channels, covering a much broader instantaneous frequency range, and with much better sensitivity.

![frequency-time-plot](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/b59a57f3-11a7-4493-8268-55c3fa632f7e/Untitled.png)

The screenshot above shows frequency on the horizontal axis (running from around 88.2 to 89.8 MHz) and time on the vertical axis. The bright orange feature at 88.5 MHz is the FM signal from KQED, a radio station in the San Francisco Bay Area. The solid yellow blocks on either side (one highlighted by the pointer in the screenshot) are the KQED “HD radio” signal (the same data as the FM signal, but encoded digitally). Additional FM stations are visible at different frequencies, including another obvious FM signal (without the corresponding digital sidebands) at 89.5 MHz.

The spectrometer generates similar spectrograms to the one shown above, but typically spanning several GHz of the radio spectrum (rather than the approx. 2 MHz shown above). The data are stored either as filterbank format or HDF5 format files, but essentially are arrays of intensity as a function of frequency and time, accompanied by headers containing metadata such as the direction the telescope was pointed in, the frequency scale, and so on. We generate over 1 PB of spectrograms per year; individual filterbank files can be tens of GB in size. We have discarded the majority of the metadata and are simply presenting numpy arrays consisting of small regions of the spectrograms that we refer to as “snippets”.

The spectrometer is searching for candidate signatures of extraterrestrial technology - so-called technosignatures. The main obstacle to doing so is that our own human technology (not just radio stations, but wifi routers, cellphones, and even electronics that are not deliberately designed to transmit radio signals) also gives off radio signals. We refer to these human-generated signals as “radio frequency interference”, or RFI.

One method we use to isolate candidate technosignatures from RFI is to look for signals that appear to be coming from particular positions on the sky. Typically we do this by alternating observations of our primary target star with observations of three nearby stars: 5 minutes on star “A”, then 5 minutes on star “B”, then back to star “A” for 5 minutes, then “C”, then back to “A”, then finishing with 5 minutes on star “D”. One set of six observations (ABACAD) is referred to as a “cadence”. Since we're just giving you a small range of frequencies for each cadence, we refer to the datasets you'll be analyzing as “cadence snippets”.

An example of an extraterrestrial signal:

![voyager-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.39.42.png)

As the plot title suggests, this is the Voyager 1 spacecraft. Even though it's 20 billion kilometers from Earth, it's picked up clearly by the GBT. The first, third, and fifth panels are the “A” target (the spacecraft, in this case). The yellow diagonal line is the radio signal coming from Voyager. It's detected when we point at the spacecraft, and it disappears when we point away. It's a diagonal line in this plot because the relative motion of the Earth and the spacecraft imparts a Doppler drift, causing the frequency to change over time. As it happens, that's another possible way to reject RFI, which has a higher tendency to remain at a fixed frequency over time.

While it would be nice to train our algorithms entirely on observations of interplanetary spacecraft, there are not many examples of them, and we also want to be able to find a wider range of signal types. So we've turned to simulating technosignature candidates.

We've taken tens of thousands of cadence snippets, which we're calling the haystack, and we've hidden needles among them. Some of these needles look similar to the Voyager 1 signal above and should be easy to detect, even with classical detection algorithms. Others are hidden in noisy regions of the spectrum and will be harder, even though they might be relatively obvious on visual inspection:

![needle-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.34.06.png)

After we perform the signal injections, we normalize each snippet, so you probably can't identify most of the needles just by looking for excess energy in the corresponding array. You'll likely need a more subtle algorithm that looks for patterns that appear only in the on-target observations.

Not all of the “needle” signals look like diagonal lines, and they may not be present for the entirety of all three “A” observations, but what they do have in common is that they are only present in some or all of the “A” observations (panels 1, 3, and 5 in the cadence snippets). Your challenge is to train an algorithm to find as many needles as you can, while minimizing the number of false positives from the haystack.

- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has dimension `(6, 273, 256)`, with the 1st dimension representing the 6 positions of the cadence, and the 2nd and 3rd dimensions representing the 2D spectrogram.
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
- **old_leaky_data** - full pre-relaunch data, including test labels; you should not assume this data is helpful (it may or may not be).

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        input/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        working/
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
```

-> data/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> data/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.75697

# 6. Current score

0.50014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5002) has done: 'The timeout is dominated by per-file `np.load` overhead plus Python multiprocessing IPC for ~60k files (54k train + 6k test), and by expensive `np.quantile` calls per sample. I keep the exact same feature definitions and CV training logic, but speed up feature extraction by (1) reading `.npy` via memory-mapping to avoid copying and reduce RAM pressure, (2) switching to `ThreadPoolExecutor` so NumPy’s C code can run in parallel without pickling/IPC overhead, and (3) using `np.partition`-based exact quantiles for 0.95/0.99 (equivalent to NumPy’s default linear method but much faster here). I also precompute indices and avoid extra Python overhead in loops while preserving identical outputs except negligible float rounding.'
- What this solution (achieved 0.5002) has done: 'Your score (0.5002 AUC) is essentially random, which strongly suggests the predictions are not aligned to the correct `id` order in the submission (or features are being read from the wrong files) rather than the model being inherently weak. The minimal fix is to rebuild `id -> path` mapping by using the **sample submission / train_labels ids** as the source of truth and searching for each file across the nested `0/..15` subfolders, ensuring every `id` is paired with its correct `.npy`. I also add a quick OOF AUC computation to confirm the model is learning (this doesn’t change training). This keeps your exact feature set and LogisticRegression CV logic unchanged, but should move AUC up toward your target band by fixing the likely misalignment bug.'
- What this solution (achieved 0.5002) has done: 'Your AUC is essentially random, and with this kind of feature+LR pipeline that almost always means the features and labels are getting mispaired (often from silently missing/mismatched `.npy` paths) rather than the model being intrinsically weak. I make the smallest change that forces **exact 1:1 alignment** between `train_labels.csv` / `sample_submission.csv` ids and the corresponding `.npy` files: build a full `id -> path` index once, then assert no missing ids (instead of skipping missing train rows, which can scramble the learned signal). I also add a deterministic sort check for the file index and keep the exact same feature definitions and CV logic, so the score should increase toward your target without changing the modeling approach. Finally, the script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5002) has done: 'Your current public AUC (~0.50) is consistent with a submission/id ordering mismatch rather than a genuinely uninformative model, so the smallest score-moving change is to force **exact id-aligned path indexing** and to fail loudly if anything is duplicated or missing. I also make the per-id file lookup deterministic by building `id -> path` via a stable scan and asserting uniqueness, then reordering features strictly by `train_labels.csv` / `sample_submission.csv` id order (your core feature extractor and LR+CV training remain unchanged). Finally, I add a lightweight sanity check that the submission ids exactly match the sample submission ids in the same order before writing `submission.csv`, which prevents silent misalignment that yields random AUC.'
- What this solution (achieved 0.5002) has done: 'Your current AUC (~0.50) indicates the model’s predictions are effectively uninformative; with this exact feature+LogReg pipeline, the most common cause is not “weak modeling” but a subtle train/test distribution mismatch from reading the wrong folder structure (e.g., `/kaggle/input/train` vs the actual competition dataset folder). I make the smallest score-relevant change by auto-resolving the correct dataset root (preferring `/kaggle/input/seti-breakthrough-listen/{train,test,...}` when present), while keeping the exact same feature definitions, CV, and model hyperparameters. I also add strict checks that train/test ids match the expected counts and that we’re not accidentally indexing the legacy `old_leaky_data` files. These changes should move the score upward toward your target by ensuring you’re training/inferencing on the intended competition data with correct id alignment.'
- What this solution (achieved 0.50121) has done: 'Your current public AUC (~0.50) is consistent with “almost-constant probabilities” rather than an id-order bug (you already assert id alignment), so the smallest score-moving change is to fix the model being underfit due to too-strong regularization on these 19 hand-crafted features. I keep the exact same feature extraction and the same 5-fold StratifiedKFold + LogisticRegression pipeline, but slightly weaken regularization (increase `C`) and add `class_weight="balanced"` to better handle any imbalance—both are minimal, metric-relevant changes that typically move AUC upward without changing the approach. I also add a quick diagnostic print of OOF probability std to confirm we’re no longer near-constant. The script still run end-to-end and write a valid `submission.csv` with the required columns and id order.'
- What this solution (achieved 0.50071) has done: 'Your pipeline is now producing non-constant predictions, so the remaining gap to the target is most likely from a weak linear decision boundary rather than id misalignment. To move AUC upward with minimal disruption, I keep the exact same features and 5-fold CV training loop, but (1) switch LogisticRegression to `solver="saga"` with an L1 penalty to allow sparse feature selection and non-uniform shrinkage, and (2) set `C` via a tiny fixed grid and pick the best by OOF AUC (still logistic regression, same CV semantics, no early stopping). This is a small, metric-aligned change that typically improves linear models on hand-crafted features without changing the approach. The code still asserts strict id/path alignment and writes a valid `submission.csv`.'
- What this solution (achieved 0.50071) has done: 'Your current score is far below the target (gap ≈ -0.256), so we should cautiously improve AUC without changing the overall pipeline. The smallest high-impact fix here is to address a likely feature/label mismatch caused by using the wrong sample/test id list: your `sample_submission.csv` has 6000 rows, which corresponds to the *older* SETI dataset, while the current competition test set is ~35k; this mismatch typically yields near-random public AUC even if everything “runs”. I modify dataset root resolution to prefer the directory that contains the modern `test/` with ~35k files (and its matching sample submission, if present), and I add strict assertions that the sample submission ids exactly match the discovered test ids (count + set) to prevent silent misalignment. The model, features, CV, and training approach remain unchanged; we only ensure we’re training/inferencing on the intended dataset split so the AUC can move up toward the target.'
- What this solution (achieved 0.50071) has done: 'Your score is far below the target (0.50071 vs 0.75697), so we should improve AUC with the smallest changes that keep the same feature extractor and the same LogisticRegression+CV approach. The biggest likely issue is not modeling but **training on the wrong dataset root**: you currently force `sample_submission.csv` to exist and to match test file count, which select the legacy 6k-test root and cap performance near random. I change root selection to prioritize the root whose **test file count matches `old_leaky_data/test_labels_old.csv` (~35847)** (a strong indicator of the modern dataset), while still using the same train labels and keeping strict id alignment by building the test id list directly from the discovered test files (instead of trusting a potentially-mismatched sample submission). Finally, I still write a valid `submission.csv` (`id,target`) sorted by id to be deterministic and valid even if the provided sample submission is stale.'
- What this solution (achieved 0.50014) has done: 'Your current AUC (0.50071) is far below the target (0.75697), so we should make the smallest score-relevant improvements that keep your exact feature extractor and the same LogisticRegression+CV approach. The most likely reason for near-random AUC with “sane” non-constant predictions is that the linear model is still too constrained for these handcrafted features; a minimal way to increase capacity without changing the approach is to keep LogisticRegression but switch from L1 to L2 (still `saga`) and tune `C` on the same OOF AUC selection you already do. I also make one strictly-metric-aligned tweak: disable `class_weight="balanced"` (AUC is rank-based and balancing can sometimes hurt ranking with already informative features), and expand the `C` grid slightly toward weaker regularization. Everything else (data root selection, id/path alignment, feature definitions, CV training loop, submission writing) remains unchanged.'
- What this solution (achieved 0.50014) has done: 'Your score is far below the target, so we should make a small, metric-relevant improvement without changing your feature set or overall LR+CV approach. The most likely cause of ~0.50 AUC despite “working” code is that the current DATA_ROOT selection is accidentally training/inferencing on the legacy 6k-test dataset (because `expected_test_n` is taken from `old_leaky_data/test_labels_old.csv`, which exists in multiple roots and can correspond to a different dataset), so I make root selection strict by requiring `train_labels.csv` to match ~54k rows and preferring the root whose train/test `.npy` counts match that. Additionally, I ensure we never sort/reorder the submission ids unless we can prove the required order (sample_submission) matches the discovered test ids; otherwise we keep the discovered test-id order deterministically (sorted) to avoid silent mismatch. These are minimal changes that preserve your exact feature extractor and model training, but should move AUC upward toward the target by fixing dataset/id alignment.'
- What this solution (achieved 0.50014) has done: 'I fix the root-cause dataset mismatch: your code pulls “expected” test ids from `old_leaky_data/test_labels_old.csv` (~35k) but the actual competition test set here is 6000 files, so the assertions and id mapping fail and no submission is written. The minimal, score-relevant correction is to derive `test_ids_all` from the chosen `test/` folder (and only use `sample_submission.csv` to enforce the required submission id order when it matches). I keep your exact feature extractor and LogisticRegression+CV logic intact, only adjusting the root-selection scoring and the id/order checks so the pipeline runs end-to-end and produces `submission.csv` with `id,target`. This should also move AUC up from ~0.50 by ensuring train/test ids align to the scored dataset rather than a mismatched legacy label file.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
BASE_INPUT = "/kaggle/input"

CANDIDATE_ROOTS = [
    os.path.join(BASE_INPUT, "seti-breakthrough-listen"),
    BASE_INPUT,
]


def _count_npy_under(dir_path: str) -> int:
    return len(list(glob.iglob(os.path.join(dir_path, "*", "*.npy"))))


def _has_min_files(r: str) -> bool:
    return (
        os.path.isdir(os.path.join(r, "train"))
        and os.path.isdir(os.path.join(r, "test"))
        and os.path.exists(os.path.join(r, "train_labels.csv"))
    )


valid_roots = [r for r in CANDIDATE_ROOTS if _has_min_files(r)]
assert valid_roots, (
    "Could not find dataset root containing train/, test/, train_labels.csv "
    f"under candidates: {CANDIDATE_ROOTS}"
)

root_meta = []
for r in valid_roots:
    train_n = _count_npy_under(os.path.join(r, "train"))
    test_n = _count_npy_under(os.path.join(r, "test"))
    lbl_rows = len(pd.read_csv(os.path.join(r, "train_labels.csv")))
    ss_path = os.path.join(r, "sample_submission.csv")
    ss_rows = len(pd.read_csv(ss_path)) if os.path.exists(ss_path) else None
    root_meta.append((r, train_n, test_n, lbl_rows, ss_rows))

print("Candidate roots meta (root, train_npy, test_npy, label_rows, sample_sub_rows):")
for m in root_meta:
    print(m)


def _root_score(m):
    r, train_n, test_n, lbl_rows, ss_rows = m
    score = 0

    score += 10_000_000 if lbl_rows == 54000 else -10_000_000
    score += 5_000_000 if train_n == lbl_rows else -abs(train_n - lbl_rows) * 1000

    if ss_rows is not None:
        score += 2_000_000
        score += 20_000_000 if ss_rows == test_n else -abs(ss_rows - test_n) * 1000
    else:
        score -= 1_000_000

    score += min(test_n, 100_000)

    return score


DATA_ROOT = sorted(root_meta, key=_root_score, reverse=True)[0][0]

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

train_labels = pd.read_csv(LABELS_PATH)
assert train_labels["id"].is_unique, "train_labels.csv has duplicate ids"

sample_sub = None
if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    assert sample_sub["id"].is_unique, "sample_submission.csv has duplicate ids"
    print("Found sample_submission.csv rows:", len(sample_sub))
else:
    print("No sample_submission.csv found under chosen root (will not rely on it).")

train_npy = _count_npy_under(TRAIN_DIR)
test_npy = _count_npy_under(TEST_DIR)
print(
    "Resolved counts -> train npy:",
    train_npy,
    "test npy:",
    test_npy,
    "label rows:",
    len(train_labels),
)

assert len(train_labels) == 54000, (
    "Selected a dataset root whose train_labels.csv is not 54,000 rows. "
    "This often indicates the wrong dataset root."
)
assert train_npy == len(
    train_labels
), "Train .npy count does not match train_labels rows; this can scramble id/label pairing."

if sample_sub is not None:
    assert len(sample_sub) == test_npy, (
        f"sample_submission rows ({len(sample_sub)}) != discovered test file count ({test_npy}). "
        "This indicates a mismatched root."
    )

train_labels.head()




## === cell 2
def _quantile_linear_partition(x1d: np.ndarray, q: float) -> np.float32:
    """
    Exact quantile matching numpy default for method='linear' for 1D array,
    computed via order statistics using np.partition (O(n)).
    """
    x = np.asarray(x1d, dtype=np.float32).ravel()
    n = x.size
    if n == 0:
        return np.float32(np.nan)
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return np.float32(np.partition(x, lo)[lo])
    part = np.partition(x, (lo, hi))
    x_lo = part[lo]
    x_hi = part[hi]
    return np.float32(x_lo + (h - lo) * (x_hi - x_lo))


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    arr shape: (6, 273, 256)
    Returns a 1D feature vector.
    """
    a = arr.astype(np.float32, copy=False)

    A = a[[0, 2, 4]]
    OFF = a[[1, 3, 5]]

    mean_all = a.mean()
    std_all = a.std()

    flat = a.reshape(-1)
    q95_all = _quantile_linear_partition(flat, 0.95)
    q99_all = _quantile_linear_partition(flat, 0.99)

    mean_A = A.mean()
    mean_OFF = OFF.mean()
    std_A = A.std()
    std_OFF = OFF.std()

    abs_mean_A = np.abs(A).mean()
    abs_mean_OFF = np.abs(OFF).mean()

    grad_t_A = np.abs(np.diff(A, axis=1)).mean()
    grad_f_A = np.abs(np.diff(A, axis=2)).mean()
    grad_t_OFF = np.abs(np.diff(OFF, axis=1)).mean()
    grad_f_OFF = np.abs(np.diff(OFF, axis=2)).mean()

    feats = np.array(
        [
            mean_all,
            std_all,
            q95_all,
            q99_all,
            mean_A,
            mean_OFF,
            mean_A - mean_OFF,
            std_A,
            std_OFF,
            std_A - std_OFF,
            abs_mean_A,
            abs_mean_OFF,
            abs_mean_A - abs_mean_OFF,
            grad_t_A,
            grad_f_A,
            grad_t_OFF,
            grad_f_OFF,
            grad_t_A - grad_t_OFF,
            grad_f_A - grad_f_OFF,
        ],
        dtype=np.float32,
    )

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats




## === cell 3
from concurrent.futures import ThreadPoolExecutor


def build_full_id_to_path(root_dir: str) -> dict:
    """
    Build a deterministic id->path map.
    """
    paths = list(glob.iglob(os.path.join(root_dir, "*", "*.npy")))
    paths.sort()
    id_to_path = {}
    dup_ids = []
    for p in paths:
        fid = os.path.splitext(os.path.basename(p))[0]
        if fid in id_to_path:
            dup_ids.append(fid)
        else:
            id_to_path[fid] = p
    assert (
        len(dup_ids) == 0
    ), f"Duplicate .npy ids found under {root_dir}. Example: {dup_ids[0]}"
    return id_to_path


train_id2path_full = build_full_id_to_path(TRAIN_DIR)
test_id2path_full = build_full_id_to_path(TEST_DIR)

for p in list(train_id2path_full.values())[:5]:
    assert (
        "old_leaky_data" not in p
    ), f"Unexpected old_leaky_data path used for training: {p}"
for p in list(test_id2path_full.values())[:5]:
    assert (
        "old_leaky_data" not in p
    ), f"Unexpected old_leaky_data path used for test: {p}"

train_ids_all = train_labels["id"].astype(str).to_numpy()

if sample_sub is not None:
    test_ids_all = sample_sub["id"].astype(str).to_numpy()
else:
    test_ids_all = np.array(sorted(test_id2path_full.keys()), dtype=str)

print("Indexed train files:", len(train_id2path_full), "label ids:", len(train_ids_all))
print(
    "Indexed test files:", len(test_id2path_full), "test ids used:", len(test_ids_all)
)

missing_train_ids = [rid for rid in train_ids_all if rid not in train_id2path_full]
missing_test_ids = [rid for rid in test_ids_all if rid not in test_id2path_full]

assert (
    len(missing_train_ids) == 0
), f"Missing train .npy for {len(missing_train_ids)} ids. Example: {missing_train_ids[0]}"
assert (
    len(missing_test_ids) == 0
), f"Missing test .npy for {len(missing_test_ids)} ids. Example: {missing_test_ids[0]}"


def _load_and_extract(path: str):
    arr = np.load(path, mmap_mode="r")
    return extract_features(arr)


train_targets_all = train_labels["target"].to_numpy(dtype=np.int64)

train_paths = [train_id2path_full[rid] for rid in train_ids_all]
y = train_targets_all.copy()
id_list = train_ids_all.copy()

n_train = len(train_paths)
n_feats = 19
X = np.empty((n_train, n_feats), dtype=np.float32)

max_workers = min(os.cpu_count() or 2, 8)
chunksize = 64

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in enumerate(
        ex.map(_load_and_extract, train_paths, chunksize=chunksize)
    ):
        X[i] = feats

print("Train features:", X.shape, "Targets:", y.shape, "Pos rate:", float(y.mean()))



## === cell 4
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]

best_auc = -1.0
best_C = None
best_oof = None
best_models = None

for C_val in C_GRID:
    oof = np.zeros(len(y), dtype=np.float32)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        penalty="l2",
                        max_iter=400,
                        n_jobs=1,
                        random_state=42,
                        C=C_val,
                        class_weight=None,
                    ),
                ),
            ]
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1].astype(np.float32)
        models.append(model)

    auc = roc_auc_score(y, oof)
    print(
        f"C={C_val} -> OOF AUC={float(auc):.6f} | pred mean={float(oof.mean()):.4f} std={float(oof.std()):.6f}"
    )

    if auc > best_auc:
        best_auc = float(auc)
        best_C = C_val
        best_oof = oof.copy()
        best_models = models

print("Best C:", best_C, "Best OOF AUC:", best_auc)
print(
    "Best OOF prediction summary:",
    float(best_oof.min()),
    float(best_oof.max()),
    float(best_oof.mean()),
    "std:",
    float(best_oof.std()),
)

models = best_models



## === cell 5
test_ids = test_ids_all
n_test = len(test_ids)

test_paths = [test_id2path_full[rid] for rid in test_ids]

X_test = np.empty((n_test, n_feats), dtype=np.float32)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in enumerate(
        ex.map(_load_and_extract, test_paths, chunksize=chunksize)
    ):
        X_test[i] = feats

print("Test features:", X_test.shape)

test_pred = np.zeros(len(test_ids), dtype=np.float32)
for model in models:
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred /= len(models)

test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})

assert submission["id"].is_unique, "Submission contains duplicate ids"
if sample_sub is not None:
    assert np.array_equal(
        submission["id"].to_numpy(), sample_sub["id"].astype(str).to_numpy()
    ), "Submission ids/order must match sample_submission.csv exactly."

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission))
print(submission.head())
