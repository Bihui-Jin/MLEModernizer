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

0.7565577900934172

# 6. Current score

0.51091

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50888) has done: 'The current notebook fails because it tries to read multiple external “../input/.../submission.csv” files that do not exist in this environment; since no model is trained here, we must replace that broken ensemble-input with an on-the-fly baseline that uses only the provided competition data. To keep changes minimal and ensure a valid `.csv` submission is always produced, I load `sample_submission.csv` and fill it with a deterministic score derived from each test `.npy` snippet using simple “A vs (B,C,D)” energy-difference features (no external dependencies beyond numpy/pandas). I also ensure IDs align exactly to `sample_submission.csv` order and clip predictions to `[0,1]`. This run end-to-end and write `submission.csv` in the working directory.'
- What this solution (achieved 0.49367) has done: 'Your current score (0.50888) is far below the target (0.75656), so we need a real improvement while still keeping the same “no-training, per-snippet feature -> sigmoid probability” core approach. I keep your A-vs-OFF contrast idea but make the features more signal-specific by adding (1) a row/column “line strength” measure that boosts narrowband/diagonal-like structure and (2) a robust tail-energy contrast using quantiles on the positive A-OFF difference. Then I replace the single hand-tuned logit mapping with a tiny, deterministic logistic regression calibrated on the official `train_labels.csv` using these same features (still simple, no deep model, and preserves evaluation semantics as probability outputs). This should move AUC upward substantially with minimal conceptual changes, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.49385) has done: 'We keep your exact “extract simple A-vs-OFF features + StandardScaler + LogisticRegression + predict_proba” pipeline, but fix the main issue hurting AUC: training AUC printed on the full training set is misleading and indicates overfitting/miscalibration without any validation. I add out-of-fold (OOF) predictions using StratifiedKFold and then refit the same LogisticRegression on the full data using a regularization strength selected from a tiny grid based on mean OOF AUC (this stays the same model/approach, just better-chosen C). To better match the ROC-AUC metric without changing semantics, I also switch the classifier to `penalty="l2"` explicitly and use `class_weight="balanced"` (still logistic regression) to reduce bias under class imbalance. These are minimal changes, should remain well within the 600s budget, and should move your score upward toward the target.'
- What this solution (achieved 0.49485) has done: 'Your current score (0.49385) is far below the target AUC (0.75656), so we should improve generalization while keeping the same core pipeline: “A vs OFF feature extraction + StandardScaler + LogisticRegression + predict_proba”. The biggest low-risk gain here is to stop using `class_weight="balanced"` (it can distort probability ranking under AUC for this dataset) and to select regularization strength `C` more robustly (use more folds + a slightly wider `C` grid) while keeping the same model. I also switch to `LogisticRegressionCV` with the same solver/penalty to reduce variance in the CV selection with minimal code changes, and keep everything deterministic. The submission writing, ID alignment, and feature extraction remain unchanged.'
- What this solution (achieved 0.49749) has done: 'Your current AUC is far below the target, so we should improve generalization while keeping the same pipeline: the exact same 9 handcrafted “A vs OFF” features + StandardScaler + LogisticRegression probability output. The biggest issue is likely mismatch between how the scaler is fit (currently on full data) and how the model is cross-validated/selected; we make the CV selection strictly “proper” by fitting the scaler inside each fold via an sklearn Pipeline, which typically boosts AUC ranking stability without changing the model class. We also switch to a slightly more robust CV setup (more folds) and a slightly wider C grid to reduce under/over-regularization risk, while staying well within runtime. Submission writing, ID alignment, and prediction semantics remain unchanged.'
- What this solution (achieved 0.50684) has done: 'Most of the timeout is coming from slow per-file `np.load` calls over ~60k snippets plus the very expensive 10-fold `LogisticRegressionCV` over 11 Cs (110 fits) and then another 10-fold OOF loop (10 more fits). To preserve the exact algorithm/semantics, we (1) speed up file discovery and ID→path mapping, (2) parallelize feature extraction across CPU cores while keeping deterministic ordering, and (3) remove the redundant second CV training loop by using `cross_val_predict` with a fixed estimator at the selected `C` (still 10 fits, same CV splits, same OOF predictions). These changes do not alter feature definitions, the model, the CV strategy, or the final fitting/prediction logic; they only eliminate repeated work and reduce Python overhead/IO wall-time. We also keep paths unchanged and keep determinism via fixed `random_state` and stable ordering.'
- What this solution (achieved 0.49434) has done: 'Your current AUC is far below the target, so the most likely “minimal but real” gain is to make the linear model slightly more expressive without changing the overall pipeline (same 9 handcrafted features → scaling → logistic-family classifier → predict_proba). I keep the exact feature extraction and prediction semantics, but switch from plain LogisticRegression to LogisticRegression with a small polynomial expansion (degree=2, interactions only) inside the sklearn Pipeline, which often helps these handcrafted features capture non-linear contrasts and improve ROC-AUC. I still select regularization strength via CV (same folds, same scoring) to avoid over/under-regularization, and keep deterministic settings and the same submission formatting and ID alignment. This is a focused change aimed at increasing ranking quality (AUC) while preserving the core “feature+logistic” approach and staying within runtime.'
- What this solution (achieved 0.5075) has done: 'We keep your exact feature extraction and “scaled features → (optional poly interactions) → logistic-family classifier → predict_proba” semantics, but remove the polynomial expansion because it’s the most likely cause of the AUC collapse (it can overfit interactions on these noisy handcrafted features and hurt ranking on test). To still improve toward the target with minimal change, we keep the same CV-based C selection (LogisticRegressionCV) and increase robustness slightly by using `solver="liblinear"` (often more stable on small-to-medium feature sets) and a touch stronger regularization grid centered around smaller C values. Everything else (data loading, ID alignment, determinism, submission writing) stays the same, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.51091) has done: 'Your timeout is dominated by reading tens of thousands of small `.npy` files and extracting features in a process pool, plus doing nested cross-validation twice (LogisticRegressionCV and then cross_val_predict) with additional parallelism overhead. I keep the exact feature definitions and model choices, but make the pipeline faster by (1) switching to a thread pool for I/O-heavy `.npy` loading (avoids process-spawn/pickling overhead), (2) ensuring the feature extraction runs purely in-place with minimal temporary arrays, (3) removing the redundant second CV pass by reusing the CV predictions already produced inside `LogisticRegressionCV` (same folds/metric; preserves evaluation semantics), and (4) reducing oversubscription by setting BLAS thread env vars and not double-parallelizing. These changes are runtime-focused and keep the same algorithm/core logic and predictions (up to negligible floating-point differences).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

BASE_INPUT = "/kaggle/data"

TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_LABELS_PATH), f"Missing {TRAIN_LABELS_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "id",
    "target",
], "Unexpected sample_submission.csv format"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
assert list(train_labels.columns) == [
    "id",
    "target",
], "Unexpected train_labels.csv format"

print("sample_submission rows:", len(sample_sub))
print("train_labels rows:", len(train_labels))




## === cell 1
def _collect_npy_paths(root_dir: str):
    out = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            with os.scandir(entry.path) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        out.append(f.path)
    out.sort()
    return out


test_files = _collect_npy_paths(TEST_DIR)
train_files = _collect_npy_paths(TRAIN_DIR)
assert len(test_files) > 0, "No test .npy files found."
assert len(train_files) > 0, "No train .npy files found."

id_to_test_path = {os.path.splitext(os.path.basename(fp))[0]: fp for fp in test_files}
id_to_train_path = {os.path.splitext(os.path.basename(fp))[0]: fp for fp in train_files}

missing_test = [fid for fid in sample_sub["id"].values if fid not in id_to_test_path]
assert (
    len(missing_test) == 0
), f"Missing {len(missing_test)} test files. Example: {missing_test[:5]}"

missing_train = [
    fid for fid in train_labels["id"].values if fid not in id_to_train_path
]
assert (
    len(missing_train) == 0
), f"Missing {len(missing_train)} train files. Example: {missing_train[:5]}"

print("Found train files:", len(train_files))
print("Found test files:", len(test_files))




## === cell 2
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def extract_features_from_snippet(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    med = np.median(x).astype(np.float32)
    mad = np.median(np.abs(x - med)).astype(np.float32) + np.float32(1e-6)
    x = (x - med) / (np.float32(1.4826) * mad)

    A = (x[0] + x[2] + x[4]) * np.float32(1.0 / 3.0)
    OFF = (x[1] + x[3] + x[5]) * np.float32(1.0 / 3.0)
    diff = A - OFF

    mad_diff = float(np.mean(np.abs(diff), dtype=np.float32))
    std_diff = float(np.std(diff, dtype=np.float32))

    qa = float(np.quantile(A, 0.99))
    qo = float(np.quantile(OFF, 0.99))
    qdiff995 = qa - qo

    pos = diff.copy()
    np.maximum(pos, np.float32(0.0), out=pos)
    qpos99 = float(np.quantile(pos, 0.99))
    qpos995 = float(np.quantile(pos, 0.995))
    mean_pos = float(np.mean(pos, dtype=np.float32))

    row_means = diff.mean(axis=1, dtype=np.float32)
    col_means = diff.mean(axis=0, dtype=np.float32)
    line_row = float(np.max(row_means))
    line_col = float(np.max(col_means))

    varA = float(np.var(A, dtype=np.float32))
    varOFF = float(np.var(OFF, dtype=np.float32))
    var_ratio = varA / (varOFF + 1e-6)

    return np.array(
        [
            mad_diff,
            std_diff,
            qdiff995,
            mean_pos,
            qpos99,
            qpos995,
            line_row,
            line_col,
            var_ratio,
        ],
        dtype=np.float32,
    )




## === cell 3
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def _features_for_id_from_path(fid_and_path):
    fid, path = fid_and_path
    arr = np.load(path, mmap_mode="r")
    return fid, extract_features_from_snippet(arr)


train_ids = train_labels["id"].values
y_train = train_labels["target"].values.astype(np.int32, copy=False)
n_train = len(train_ids)

train_id_path = [(fid, id_to_train_path[fid]) for fid in train_ids]
X_train = np.empty((n_train, 9), dtype=np.float32)

cpu = os.cpu_count() or 4
max_workers = min(32, cpu * 4)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, (_fid, feats) in enumerate(
        ex.map(_features_for_id_from_path, train_id_path, chunksize=256)
    ):
        X_train[i] = feats

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
C_grid = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], dtype=float)

pipe_cv = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegressionCV(
                Cs=C_grid,
                cv=cv,
                scoring="roc_auc",
                solver="liblinear",
                penalty="l2",
                max_iter=800,
                n_jobs=1,
                refit=True,
                random_state=0,
            ),
        ),
    ]
)
pipe_cv.fit(X_train, y_train)

best_C = float(pipe_cv.named_steps["clf"].C_[0])
print("Selected C:", best_C)

lr_cv = pipe_cv.named_steps["clf"]
from sklearn.base import clone

pipe_fixed = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="liblinear",
                penalty="l2",
                max_iter=800,
                C=best_C,
                random_state=0,
            ),
        ),
    ]
)

oof = np.empty(n_train, dtype=np.float32)
for tr_idx, va_idx in cv.split(X_train, y_train):
    m = clone(pipe_fixed)
    m.fit(X_train[tr_idx], y_train[tr_idx])
    oof[va_idx] = m.predict_proba(X_train[va_idx])[:, 1].astype(np.float32)

print("OOF AUC @ selected C:", roc_auc_score(y_train, oof))

clf = pipe_cv
train_pred = clf.predict_proba(X_train)[:, 1]
print("Train AUC (in-sample, not CV):", roc_auc_score(y_train, train_pred))




## === cell 4
test_ids = sample_sub["id"].values
n_test = len(test_ids)

test_id_path = [(fid, id_to_test_path[fid]) for fid in test_ids]
X_test = np.empty((n_test, 9), dtype=np.float32)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, (_fid, feats) in enumerate(
        ex.map(_features_for_id_from_path, test_id_path, chunksize=256)
    ):
        X_test[i] = feats

preds = clf.predict_proba(X_test)[:, 1].astype(np.float32)
preds = np.clip(preds, 0.0, 1.0)
sample_sub["target"] = preds

print(sample_sub.head())
print(sample_sub["target"].describe())




## === cell 5
out_path = "submission.csv"
sample_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sample_sub))
print("Columns:", list(sample_sub.columns))
