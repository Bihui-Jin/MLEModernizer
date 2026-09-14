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

0.7467085579018485

# 6. Current score

0.51113

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50624) has done: 'The timeout is dominated by Python-level loops repeatedly calling `np.load` and doing multiple full-array reductions per snippet (mean/std/max/min) for ~54k train + 6k test files. I keep the exact same feature set and LogisticRegression training, but speed up feature extraction by (1) using `np.load(..., mmap_mode="r")` to reduce I/O overhead, (2) computing per-panel mean/std/max/min in a single pass via reshaping (equivalent reductions), and (3) avoiding repeated computations like `(A**2).mean()` being evaluated three times. I also parallelize feature extraction with a deterministic, ordered `ThreadPoolExecutor` (NumPy reductions release the GIL, and disk I/O benefits), preserving row alignment and results. Nothing about the model, CV, or features changes—only how we compute the same numbers faster.'
- What this solution (achieved 0.50999) has done: 'We need to move your ROC-AUC up from 0.50624 toward 0.7467 (higher is better), and the gap (~0.24) is >30% of target, so we’re allowed to adjust core modeling as needed. The simplest high-impact fix for SETI is to switch from a linear model to a non-linear tabular model (RandomForest) while keeping your exact same feature extraction and train/test alignment, which should substantially improve AUC without changing the data pipeline. I keep the same 5-fold StratifiedKFold evaluation and still write `submission.csv` in the required `id,target` format. To stay within the 600s budget, I also keep the fast threaded feature building unchanged and set conservative RF hyperparameters to avoid timeouts.'
- What this solution (achieved 0.50198) has done: 'The main timeout drivers here are (1) repeated per-file `np.load` calls across tens of thousands of `.npy` files and (2) extremely expensive RandomForest training (5-fold CV + final fit) with `n_estimators=800` on 54k rows. I keep the exact feature definitions and the exact RandomForest hyperparameters/fit semantics, but reduce wall time by (a) eliminating Python-level overhead in featurization (faster file indexing, deterministic parallel mapping, larger chunks, fewer closures), (b) enabling fully deterministic, persistent on-disk caching of the feature matrices via memory-mapped `.npy` writing (so feature extraction is never recomputed within the run), and (c) avoiding the extra full-data refit by reusing the already-trained fold models to produce test predictions (averaging predicted probabilities), which preserves the model family/training approach and avoids the largest redundant fit. Additionally, I make CV splits lazy (not materialized) and reduce avoidable copies in feature extraction while preserving identical computations (ddof=0, same reductions). These changes target the timeout without changing the core algorithm (same features, same RF parameters, same AUC evaluation semantics).'
- What this solution (achieved 0.49237) has done: 'The timeout is dominated by training 5 very large RandomForest models (1200 trees each) on 54k rows and by repeated feature extraction I/O; both are heavy but only the former is truly explosive. To preserve the exact same modeling logic while reducing wall time, the main safe win is to avoid doing full 5-fold training inside the submission run and instead train a single RandomForest with the same hyperparameters on all training data (same estimator family, same features, same objective, same predict_proba semantics). On the feature side, we keep the exact same feature definitions but speed up file discovery and feature building by using `os.scandir` recursion (less overhead than `os.walk`) and by keeping the existing memmap cache behavior (so reruns don’t recompute). These changes keep accuracy behavior consistent (often slightly better vs CV-averaging) while cutting training time by ~5x and reducing Python/file-system overhead.'
- What this solution (achieved 0.49947) has done: 'You’re far below the target AUC (0.492 vs 0.7467; higher is better), so we should legitimately improve generalization with minimal, low-risk modeling changes while keeping your exact feature extraction and overall pipeline intact. The smallest effective lever here is to reduce overfitting and improve probability ranking by switching the RandomForest to stronger regularization (shallower trees, larger leaf sizes) and using ExtraTrees (same family of tree ensembles, same `predict_proba` semantics, typically better AUC on noisy tabular stats). We keep the same cached feature matrices, same train/test alignment, and still output `submission.csv` with `id,target`. We also keep runtime within budget by using fewer trees (but not via early stopping or approximations) and by retaining `n_jobs=-1`.'
- What this solution (achieved 0.49744) has done: 'We need to raise AUC from 0.49947 toward 0.7467 (higher is better), so we should improve generalization with a minimal, low-risk change to the existing ExtraTrees pipeline. The biggest issue in your current code is that the ExtraTrees output probabilities are often poorly calibrated/ranked; wrapping the trained ExtraTrees in `CalibratedClassifierCV(method="isotonic")` with a small internal CV usually improves ROC-AUC without changing the base feature extraction or model family. To keep runtime reasonable and preserve core logic, we keep the same ExtraTrees hyperparameters and features, but fit a calibrated wrapper on top and use its `predict_proba` for the submission. We also keep everything deterministic via `random_state` and reuse the cached feature matrices unchanged.'
- What this solution (achieved 0.50472) has done: 'Your current AUC (0.49744) is far below the target (0.7467), so we should make a small, legitimate change that tends to improve ranking without changing your feature extraction or the ExtraTrees core model. The minimal high-impact fix here is to switch probability calibration from isotonic (which can overfit on noisy tabular features) to sigmoid/Platt scaling, keeping the same `CalibratedClassifierCV` wrapper and CV=3. To avoid any accidental distribution shift from `class_weight="balanced"` when we already calibrate probabilities, we also remove class weighting (the trees still learn the same way; calibration handles mapping to probabilities). Everything else (paths, cached features, model family, predict_proba submission) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.51208) has done: 'We need to raise ROC-AUC from 0.50472 toward 0.74671 (higher is better), so the smallest reliable gain is to improve feature scaling/conditioning without changing the model family, feature definitions, or training objective. ExtraTrees itself doesn’t need scaling, but your top-level model is actually a calibrated classifier (sigmoid) which *does* fit a logistic regression on the base model scores, and that calibration step benefits from having well-conditioned, standardized input scores and from using a shuffle-aware CV. I keep your exact feature extraction and ExtraTrees hyperparameters, but (1) add a `StandardScaler` on X (and Xt) and (2) switch the calibrator CV to `StratifiedKFold(shuffle=True, random_state=...)` for more stable calibration splits. These are minimal changes that typically improve probability ranking on SETI-like tabular stats while preserving end-to-end behavior and producing the same valid `submission.csv`.'
- What this solution (achieved 0.51256) has done: 'We need to increase ROC-AUC from 0.51208 toward 0.74671 (higher is better), so we make a minimal, legitimate modeling improvement that keeps your exact feature extraction and overall training approach intact. The smallest high-impact lever here is to increase the capacity of the same ExtraTrees+sigmoid calibration pipeline by using more trees and slightly deeper trees, while keeping the same regularization knobs (leaf/split) so we don’t wildly overfit. We also set a deterministic `random_state` for the calibrator’s internal logistic regression (where supported) to reduce run-to-run variance and keep evaluation stable. All I/O paths and the submission schema remain unchanged, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.51113) has done: 'You’re far below the target AUC (0.5126 vs 0.7467; higher is better), so we should improve ranking with the smallest legitimate modeling change while keeping your exact feature extraction and pipeline intact. The most direct low-risk win here is to increase diversity and reduce variance of the same tree-ensemble family by switching from `ExtraTreesClassifier` to `RandomForestClassifier` (same `predict_proba` semantics), keeping your calibration (`sigmoid`) and scaling exactly as-is. This is a core-logic change but allowed because the gap is >30% of the target, and it typically lifts AUC on this kind of tabular-stat SETI baseline. Everything else (paths, caching, threading, submission format) is preserved so it still runs end-to-end and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT = "/kaggle/input"  # Kaggle standard; in this environment, data is mirrored under /kaggle/data too
ALT_DATA_ROOT = "/kaggle/data"


def _pick_existing_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


BASE_PATH = _pick_existing_path(
    os.path.join(DATA_ROOT, "seti-breakthrough-listen"),
    os.path.join(ALT_DATA_ROOT, "seti-breakthrough-listen"),
    DATA_ROOT,
    ALT_DATA_ROOT,
)

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not find Kaggle data root under /kaggle/input or /kaggle/data."
    )

TRAIN_LABELS_PATH = _pick_existing_path(
    os.path.join(BASE_PATH, "train_labels.csv"),
    os.path.join(DATA_ROOT, "train_labels.csv"),
    os.path.join(ALT_DATA_ROOT, "train_labels.csv"),
)
SAMPLE_SUB_PATH = _pick_existing_path(
    os.path.join(BASE_PATH, "sample_submission.csv"),
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    os.path.join(ALT_DATA_ROOT, "sample_submission.csv"),
)

TRAIN_DIR = _pick_existing_path(
    os.path.join(BASE_PATH, "train"),
    os.path.join(DATA_ROOT, "train"),
    os.path.join(ALT_DATA_ROOT, "train"),
)
TEST_DIR = _pick_existing_path(
    os.path.join(BASE_PATH, "test"),
    os.path.join(DATA_ROOT, "test"),
    os.path.join(ALT_DATA_ROOT, "test"),
)

if (
    TRAIN_LABELS_PATH is None
    or SAMPLE_SUB_PATH is None
    or TRAIN_DIR is None
    or TEST_DIR is None
):
    raise FileNotFoundError(
        f"Missing expected files/dirs. Found BASE_PATH={BASE_PATH}, "
        f"TRAIN_LABELS_PATH={TRAIN_LABELS_PATH}, SAMPLE_SUB_PATH={SAMPLE_SUB_PATH}, "
        f"TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}"
    )

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train_labels:", train_labels.shape, "sample_sub:", sample_sub.shape)
print(train_labels.head())
print(sample_sub.head())




## === cell 1
def list_npy_files(root_dir):
    id_to_path = {}

    def _recurse(d):
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_dir():
                    _recurse(entry.path)
                elif entry.is_file() and entry.name.endswith(".npy"):
                    sid = entry.name[:-4]
                    id_to_path[sid] = entry.path

    _recurse(root_dir)
    return id_to_path


train_id_to_path = list_npy_files(TRAIN_DIR)
test_id_to_path = list_npy_files(TEST_DIR)

missing_train = (
    train_labels.loc[~train_labels["id"].isin(train_id_to_path.keys()), "id"]
    .head(5)
    .tolist()
)
missing_test = (
    sample_sub.loc[~sample_sub["id"].isin(test_id_to_path.keys()), "id"]
    .head(5)
    .tolist()
)

print(
    "Train files:",
    len(train_id_to_path),
    " Train labels:",
    len(train_labels),
    " Missing labeled train examples (sample):",
    missing_train,
)
print(
    "Test files:",
    len(test_id_to_path),
    " Sample submission ids:",
    len(sample_sub),
    " Missing test examples (sample):",
    missing_test,
)




## === cell 2
def extract_features_from_snippet(x):
    """
    x: (6, 273, 256) float16/float32
    Return a 1D feature vector.
    Core-logic: simple statistics and A-vs-BCD contrast features.
    """
    x = x.astype(np.float32, copy=False)

    x2d = x.reshape(6, -1)  # (6, 273*256)
    panel_mean = x2d.mean(axis=1)
    panel_var = x2d.var(axis=1)  # ddof=0 default
    panel_std = np.sqrt(panel_var, dtype=np.float32)
    panel_max = x2d.max(axis=1)
    panel_min = x2d.min(axis=1)

    A = x[[0, 2, 4]]
    BCD = x[[1, 3, 5]]

    mean_A = A.mean()
    mean_BCD = BCD.mean()
    std_A = np.sqrt(A.var(), dtype=np.float32)
    std_BCD = np.sqrt(BCD.var(), dtype=np.float32)

    A_freq_prof = A.mean(axis=(0, 1))  # (256,)
    B_freq_prof = BCD.mean(axis=(0, 1))  # (256,)
    A_time_prof = A.mean(axis=(0, 2))  # (273,)
    B_time_prof = BCD.mean(axis=(0, 2))  # (273,)

    freq_diff = A_freq_prof - B_freq_prof
    time_diff = A_time_prof - B_time_prof

    A2_mean = np.square(A, dtype=np.float32).mean()
    BCD2_mean = np.square(BCD, dtype=np.float32).mean()

    proj_feats = np.array(
        [
            freq_diff.mean(),
            freq_diff.std(),  # ddof=0 default
            freq_diff.max(),
            freq_diff.min(),
            time_diff.mean(),
            time_diff.std(),  # ddof=0 default
            time_diff.max(),
            time_diff.min(),
            A2_mean,
            BCD2_mean,
            A2_mean - BCD2_mean,
            mean_A - mean_BCD,
            std_A - std_BCD,
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [panel_mean, panel_std, panel_max, panel_min, proj_feats]
    ).astype(np.float32, copy=False)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


some_id = train_labels["id"].iloc[0]
x0 = np.load(train_id_to_path[some_id], mmap_mode="r")
f0 = extract_features_from_snippet(x0)
print("Snippet shape:", x0.shape, "Feature shape:", f0.shape)



## === cell 3
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _default_featurize_workers():
    cpu = os.cpu_count() or 4
    return min(8, max(1, cpu // 2))


def _featurize_id_path(path):
    x = np.load(path, mmap_mode="r")
    return extract_features_from_snippet(x)


def build_feature_matrix(
    ids, id_to_path, n_features, cache_path=None, max_workers=None
):
    if cache_path is not None and os.path.exists(cache_path):
        X_cached = np.load(cache_path, mmap_mode="r")
        if X_cached.shape == (len(ids), n_features) and X_cached.dtype == np.float32:
            return np.asarray(X_cached)

    if max_workers is None:
        max_workers = _default_featurize_workers()

    paths = [id_to_path[sid] for sid in ids]

    if cache_path is not None:
        Xout = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32, shape=(len(ids), n_features)
        )
    else:
        Xout = np.empty((len(ids), n_features), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in enumerate(ex.map(_featurize_id_path, paths, chunksize=512)):
            Xout[i] = feats

    if cache_path is not None:
        del Xout
        Xout = np.load(cache_path, mmap_mode="r")
        return np.asarray(Xout)

    return Xout


train_ids = train_labels["id"].values
y = train_labels["target"].values.astype(np.int32)

X = build_feature_matrix(
    train_ids,
    train_id_to_path,
    n_features=f0.shape[0],
    cache_path="/kaggle/working/X_train_feats.npy",
    max_workers=None,
)
print("X shape:", X.shape, "y mean:", y.mean())

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)

base_clf = RandomForestClassifier(
    n_estimators=1200,
    max_depth=20,
    min_samples_leaf=8,
    min_samples_split=16,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight=None,
    bootstrap=True,
)

cal_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

try:
    clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=cal_cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
except TypeError:
    clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=cal_cv,
        n_jobs=-1,
    )

clf.fit(X_scaled, y)
print("Trained Calibrated RandomForest (sigmoid, cv=3) on full training set.")



## === cell 4
test_ids = sample_sub["id"].values
Xt = build_feature_matrix(
    test_ids,
    test_id_to_path,
    n_features=X.shape[1],
    cache_path="/kaggle/working/X_test_feats.npy",
    max_workers=None,
)

Xt_scaled = scaler.transform(Xt).astype(np.float32, copy=False)

test_pred = clf.predict_proba(Xt_scaled)[:, 1].astype(np.float32, copy=False)

sub = pd.DataFrame({"id": sample_sub["id"].values, "target": test_pred})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
