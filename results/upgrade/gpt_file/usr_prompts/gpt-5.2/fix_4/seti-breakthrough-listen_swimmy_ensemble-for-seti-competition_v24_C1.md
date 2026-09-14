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

0.757186373534826

# 6. Current score

0.51047

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your current notebook is trying to ensemble several external Kaggle notebook submissions from `../input/...`, but those datasets are not available in this environment, causing the first `read_csv` to fail and preventing any `submission.csv` from being written. I replace that broken external-ensemble step with a minimal, fully self-contained baseline that reads the provided `sample_submission.csv` and outputs a valid `submission.csv` with constant probabilities (score be ~0.5 AUC, but it run end-to-end and yield a valid file). I also keep the original cell order/intent by making the “data1..data6” frames exist, so later cells don’t crash. Once you confirm it submits successfully, we can iterate toward the target score with a real model (would require adding ML packages not currently installed, or implementing a lightweight feature-based model with sklearn if available).'
- What this solution (achieved 0.51047) has done: 'The timeout is dominated by per-file `np.load` plus many repeated `np.percentile` calls inside Python loops over ~54k train + 6k test files, and by repeatedly computing `np.abs(D)` and percentiles on the full 2D arrays. I keep the exact feature set and model logic, but speed it up by (1) computing percentiles via a single `np.percentile(..., q=[...])` call per array instead of many calls, (2) reusing precomputed `absD`, (3) avoiding repeated `os.path.exists` checks by directly trying `np.load` and catching `FileNotFoundError`, and (4) parallelizing feature extraction across CPU cores with deterministic ordering. I also remove the redundant submission DataFrame copies (they don’t change predictions) to cut overhead without changing outputs. All paths, features, folds, model parameters, and prediction semantics remain the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/data",
    "/kaggle/input",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE_DIR = _first_existing(BASE_CANDIDATES)
if BASE_DIR is None:
    raise FileNotFoundError(
        f"Could not find competition data directory in any of: {BASE_CANDIDATES}"
    )

SAMPLE_SUB_PATHS = [
    os.path.join(BASE_DIR, "sample_submission.csv"),
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
SAMPLE_SUB = _first_existing(SAMPLE_SUB_PATHS)
if SAMPLE_SUB is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {SAMPLE_SUB_PATHS}"
    )

sample = pd.read_csv(SAMPLE_SUB)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must have columns id,target; got columns={list(sample.columns)}"
    )

print("Using BASE_DIR:", BASE_DIR)
print("Loaded sample_submission:", sample.shape)



## === cell 2
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TRAIN_LABELS_PATHS = [
    os.path.join(BASE_DIR, "train_labels.csv"),
    "/kaggle/data/train_labels.csv",
    "/kaggle/input/train_labels.csv",
]
TRAIN_LABELS = _first_existing(TRAIN_LABELS_PATHS)
if TRAIN_LABELS is None:
    raise FileNotFoundError(
        f"Could not find train_labels.csv in any of: {TRAIN_LABELS_PATHS}"
    )

train_labels = pd.read_csv(TRAIN_LABELS)
if not {"id", "target"}.issubset(train_labels.columns):
    raise ValueError(
        f"train_labels.csv must have columns id,target; got columns={list(train_labels.columns)}"
    )


def _find_split_dir(base_dir, split):
    cands = [
        os.path.join(base_dir, split),
        os.path.join("/kaggle/data", split),
        os.path.join("/kaggle/input", split),
        os.path.join("/kaggle/data/seti-breakthrough-listen", split),
        os.path.join("/kaggle/input/seti-breakthrough-listen", split),
    ]
    return _first_existing(cands)


TRAIN_DIR = _find_split_dir(BASE_DIR, "train")
TEST_DIR = _find_split_dir(BASE_DIR, "test")
if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not find train/ or test/ directories. TRAIN_DIR={TRAIN_DIR}, TEST_DIR={TEST_DIR}"
    )

print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)


def _npy_path(root, id_):
    return os.path.join(root, str(id_)[0], f"{id_}.npy")


_PCTS_MAIN = np.array([5, 25, 50, 75, 95], dtype=np.float32)
_PCTS_ROWCOL = np.array([90], dtype=np.float32)


def extract_features_from_npy(path):
    x = np.load(path)  # expected (6,273,256), float16
    x = x.astype(np.float32, copy=False)

    A = (x[0] + x[2] + x[4]) / 3.0
    O = (x[1] + x[3] + x[5]) / 3.0
    D = A - O
    absD = np.abs(D)

    feats = []

    for arr in (A, O, D, absD):
        p = np.percentile(arr, _PCTS_MAIN)
        feats.extend(
            [
                float(arr.mean()),
                float(arr.std()),
                float(p[0]),
                float(p[1]),
                float(p[2]),
                float(p[3]),
                float(p[4]),
                float(arr.min()),
                float(arr.max()),
            ]
        )

    for arr in (A, D, absD):
        row = arr.mean(axis=1)  # (273,)
        col = arr.mean(axis=0)  # (256,)
        pr = np.percentile(row, _PCTS_ROWCOL)[0]
        pc = np.percentile(col, _PCTS_ROWCOL)[0]
        feats.extend(
            [
                float(row.mean()),
                float(row.std()),
                float(pr),
                float(col.mean()),
                float(col.std()),
                float(pc),
            ]
        )

    thr = np.percentile(D, 99.5)
    feats.append(float((D > thr).mean()))
    thr2 = np.percentile(absD, 99.5)
    feats.append(float((absD > thr2).mean()))

    return np.asarray(feats, dtype=np.float32)


from concurrent.futures import ThreadPoolExecutor

_MAX_WORKERS = min(32, (os.cpu_count() or 4))


def _build_feature_matrix(ids, root_dir, n_features):
    X_out = np.zeros((len(ids), n_features), dtype=np.float32)

    def _one(i_id):
        i, id_ = i_id
        p = _npy_path(root_dir, id_)
        try:
            feats = extract_features_from_npy(p)
            return i, feats, False
        except FileNotFoundError:
            return i, None, True

    missing_local = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for i, feats, is_missing in ex.map(_one, enumerate(ids), chunksize=64):
            if is_missing:
                missing_local += 1
            else:
                X_out[i] = feats
    return X_out, missing_local


N_FEATURES = 9 * 4 + 6 * 3 + 2
y = train_labels["target"].astype(int).to_numpy()

X, missing = _build_feature_matrix(train_labels["id"].values, TRAIN_DIR, N_FEATURES)

print(
    "Built train features:",
    X.shape,
    "missing files:",
    missing,
    "pos_rate:",
    y.mean().round(4),
)



## === cell 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

oof = np.zeros(len(y), dtype=np.float32)
for tr_idx, va_idx in skf.split(X, y):
    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=200,
        random_state=RANDOM_SEED,
    )
    model.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = model.predict_proba(X[va_idx])[:, 1].astype(np.float32)

cv_auc = roc_auc_score(y, oof)
print("CV AUC (sanity check):", cv_auc)

final_model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=200,
    random_state=RANDOM_SEED,
)
final_model.fit(X, y)



## === cell 4
test_ids = sample["id"].values
X_test, missing_test = _build_feature_matrix(test_ids, TEST_DIR, X.shape[1])

print("Built test features:", X_test.shape, "missing files:", missing_test)

test_pred = final_model.predict_proba(X_test)[:, 1].astype(np.float64)
test_pred = np.clip(test_pred, 0.0, 1.0)

data6 = sample.copy()
data6["target"] = test_pred



## === cell 5
out_path = "submission.csv"
data6[["id", "target"]].to_csv(out_path, index=False)

print("Wrote", out_path, "with shape", data6.shape)
print(data6.head())
print(
    "Prediction stats:",
    float(np.min(data6["target"])),
    float(np.mean(data6["target"])),
    float(np.max(data6["target"])),
)
