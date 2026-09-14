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

0.7571820127408645

# 6. Current score

0.50597

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49798) has done: 'The timeout is dominated by Python-level loops that `np.load` 60k+ `.npy` files one-by-one and compute features per-sample, plus an expensive `glob.glob` over all files up-front. I keep the exact same features and model, but speed it up by (1) building id→path maps with `os.scandir` instead of `glob`, (2) vectorizing `extract_features_from_array` to eliminate the inner `for i in range(6)` loop, (3) parallelizing feature extraction across CPU cores with deterministic ordering, and (4) preallocating output arrays to avoid large list growth and `vstack` overhead. These changes are provably equivalent (same computations, just reordered/parallelized) and preserve evaluation semantics; only negligible floating-point summation order differences may occur.'
- What this solution (achieved 0.49795) has done: 'Your current score (0.49798) is far below the target (0.75718), so we should make a small, legitimate change that materially improves generalization without changing the feature set or model family. The biggest issue is that the current LogisticRegression is effectively using a too-strong default regularization (C=1) for this 34-feature problem; increasing C (weaker regularization) typically improves ROC-AUC here while keeping the exact same model/approach. I add a tiny cross-validated selection of `C` (and `class_weight` as an option) using your existing StratifiedKFold and still train a single LogisticRegression pipeline, then fit once on all training data with the chosen hyperparameters. This preserves core logic (same features, same scaler+logreg, same CV loop semantics) and should move the score upward toward the target.'
- What this solution (achieved 0.49805) has done: 'Your current score is far below the target, so we should aim for a meaningful but still minimal/generalization-oriented improvement without changing the model family or feature definitions. The biggest likely issue is that the CV-selected hyperparameters are chosen using OOF AUC, but you then refit a single model on all data; instead, we can keep the exact same pipeline and folds but generate test predictions as an average of per-fold models (a standard “CV ensemble”), which typically increases ROC-AUC for linear models with standardized features. This preserves the same features, the same scaler+logistic regression, and the same training approach (StratifiedKFold fits); it only changes how we aggregate predictions for test. We also keep your CV selection of `C`/`class_weight` unchanged, and we still write a valid `submission.csv` with the required columns and id order.'
- What this solution (achieved 0.49916) has done: 'I fix the immediate runtime error by making the declared feature count match the actual number of features produced (44), which unblocks both train/test featurization and downstream model training. Then I ensure the CV-selection/training cells run in order by keeping variables defined (once featurization succeeds, the NameErrors disappear). I also add a defensive assertion that the feature extractor output length matches `N_FEATS` so this can’t silently break again. These changes are execution/correctness fixes and should not change modeling semantics beyond allowing the pipeline to run and produce a valid `submission.csv`.'
- What this solution (achieved 0.50597) has done: 'Main runtime is spent (1) scanning the filesystem with `iglob` over tens of thousands of files and (2) per-file feature extraction calling `np.quantile` multiple times. I replace directory scanning with direct path construction from the provided `id` lists (same files, far fewer syscalls) and I replace `np.quantile(..., method="linear")` with an exactly-equivalent linear-quantile implementation using `np.partition` (avoids full sort; same definition, negligible FP diffs only). I also remove redundant quantile calls by computing shared quantiles in one pass, and I keep multithreaded loading but tune chunksize upward to reduce scheduling overhead. Core model (scaler, CV, logistic regression, folds, and prediction semantics) stays identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(42)

BASE_DIR = "/kaggle/input/seti-breakthrough-listen"
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert set(train_labels.columns) >= {"id", "target"}
assert list(sample_sub.columns)[:2] == ["id", "target"]
train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("train_labels:", train_labels.shape, "sample_submission:", sample_sub.shape)




## === cell 1
def id_to_path(root_dir: str, fid: str) -> str:
    return os.path.join(root_dir, fid[0], f"{fid}.npy")


def build_id_to_path_map_from_ids(root_dir: str, ids):
    mp = {}
    for fid in ids:
        mp[fid] = id_to_path(root_dir, fid)
    return mp


train_ids = train_labels["id"].tolist()
test_ids = sample_sub["id"].tolist()

train_id_to_path = build_id_to_path_map_from_ids(TRAIN_DIR, train_ids)
test_id_to_path = build_id_to_path_map_from_ids(TEST_DIR, test_ids)

missing_train = [fid for fid, p in train_id_to_path.items() if not os.path.exists(p)]
missing_test = [fid for fid, p in test_id_to_path.items() if not os.path.exists(p)]
print("Train ids:", len(train_ids), "Test ids:", len(test_ids))
print("Missing train files for labels:", len(missing_train))
print("Missing test files:", len(missing_test))
assert len(missing_train) == 0, f"Missing train files. Example: {missing_train[:3]}"
assert len(missing_test) == 0, f"Missing test files. Example: {missing_test[:3]}"



## === cell 2
import math
from concurrent.futures import ThreadPoolExecutor


def _quantile_linear_1d(a: np.ndarray, q: float) -> np.float32:
    a = np.asarray(a)
    n = a.size
    if n == 0:
        return np.float32(np.nan)
    if q <= 0.0:
        return np.float32(np.min(a))
    if q >= 1.0:
        return np.float32(np.max(a))

    i = q * (n - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        v = np.partition(a, lo)[lo]
        return np.float32(v)

    aa = a  # may be view
    part = np.partition(aa, (lo, hi))
    v_lo = part[lo]
    v_hi = part[hi]
    w = i - lo
    return np.float32(v_lo + (v_hi - v_lo) * w)


def _quantile_linear_axis1(a2d: np.ndarray, q: float) -> np.ndarray:
    m = a2d.shape[0]
    out = np.empty(m, dtype=np.float32)
    for r in range(m):
        out[r] = _quantile_linear_1d(a2d[r], q)
    return out


def _quantiles_linear_1d(a: np.ndarray, qs) -> np.ndarray:
    a = np.asarray(a)
    n = a.size
    qs = np.asarray(qs, dtype=np.float64)
    out = np.empty(qs.shape[0], dtype=np.float32)
    if n == 0:
        out.fill(np.nan)
        return out

    mins = None
    maxs = None
    idxs = []
    params = []
    for j, q in enumerate(qs):
        if q <= 0.0:
            if mins is None:
                mins = np.min(a)
            out[j] = np.float32(mins)
            continue
        if q >= 1.0:
            if maxs is None:
                maxs = np.max(a)
            out[j] = np.float32(maxs)
            continue
        i = q * (n - 1)
        lo = int(math.floor(i))
        hi = int(math.ceil(i))
        idxs.append(lo)
        idxs.append(hi)
        params.append((j, lo, hi, i - lo))

    if params:
        uniq = np.unique(np.array(idxs, dtype=np.int64))
        part = np.partition(a, uniq)
        for j, lo, hi, w in params:
            if lo == hi:
                out[j] = np.float32(part[lo])
            else:
                v_lo = part[lo]
                v_hi = part[hi]
                out[j] = np.float32(v_lo + (v_hi - v_lo) * w)
    return out


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))

    x_flat = x.reshape(6, -1)
    panel_p95 = _quantile_linear_axis1(x_flat, 0.95)

    A = x[[0, 2, 4]].mean(axis=0)
    B = x[[1, 3, 5]].mean(axis=0)
    D = A - B

    D_flat = D.ravel()
    D_abs_flat = np.abs(D_flat)

    feats = np.empty(24 + 8 + 2 + 3 + 4 + 1 + 2 + 5 + 2 + 2 + 1, dtype=np.float32)
    k = 0

    pm = np.stack([panel_mean, panel_std, panel_max, panel_p95], axis=1).ravel()
    feats[k : k + pm.size] = pm
    k += pm.size  # 24

    feats[k] = A.mean()
    k += 1
    feats[k] = A.std()
    k += 1
    feats[k] = B.mean()
    k += 1
    feats[k] = B.std()
    k += 1
    feats[k] = D.mean()
    k += 1
    feats[k] = D.std()
    k += 1
    feats[k] = D.max()
    k += 1
    feats[k] = _quantile_linear_1d(D_flat, 0.95)
    k += 1

    feats[k] = np.abs(np.diff(D, axis=0)).mean()
    k += 1
    feats[k] = np.abs(np.diff(D, axis=1)).mean()
    k += 1

    d2_mean = (D_flat * D_flat).mean()
    feats[k] = d2_mean
    k += 1
    feats[k] = np.sqrt(d2_mean + 1e-8)
    k += 1
    feats[k] = _quantile_linear_1d(D_abs_flat, 0.99)
    k += 1

    row_prof = D.mean(axis=1)
    col_prof = D.mean(axis=0)
    feats[k] = row_prof.std()
    k += 1
    feats[k] = col_prof.std()
    k += 1
    feats[k] = row_prof.max() - row_prof.min()
    k += 1
    feats[k] = col_prof.max() - col_prof.min()
    k += 1

    a_flat = A.ravel()
    b_flat = B.ravel()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
    feats[k] = float(np.dot(a_flat, b_flat) / denom)
    k += 1

    feats[k] = float(np.maximum(D_flat, 0).mean())
    k += 1
    feats[k] = float(np.maximum(-D_flat, 0).mean())
    k += 1

    feats[k] = float(D_abs_flat.mean())
    k += 1
    feats[k] = float(D_abs_flat.std())
    k += 1

    q_abs = _quantiles_linear_1d(D_abs_flat, [0.90, 0.95, 0.99])
    feats[k] = float(q_abs[0])
    k += 1
    feats[k] = float(q_abs[1])
    k += 1
    feats[k] = float(q_abs[2])
    k += 1

    mu = float(D_flat.mean())
    sigma = float(D_flat.std() + 1e-6)
    z = (D_flat - mu) / sigma
    feats[k] = float((z**3).mean())
    k += 1
    feats[k] = float((z**4).mean())
    k += 1

    D_abs_2d = np.abs(D)
    feats[k] = float(D_abs_2d.sum(axis=0).max())
    k += 1
    feats[k] = float(D_abs_2d.sum(axis=1).max())
    k += 1

    feats[k] = float(np.linalg.norm(D_flat) / (np.linalg.norm(b_flat) + 1e-6))
    k += 1

    if k != feats.shape[0]:
        raise RuntimeError(
            f"Feature packing mismatch: wrote k={k}, allocated={feats.shape[0]}"
        )

    return feats


N_FEATS = int(
    extract_features_from_array(np.zeros((6, 273, 256), dtype=np.float32)).shape[0]
)
print("N_FEATS =", N_FEATS)


def _featurize_path(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")  # read-only mmap
    return extract_features_from_array(arr)


def load_and_featurize(ids, id_to_path_map, max_workers=None, chunksize=256):
    n = len(ids)
    X_out = np.empty((n, N_FEATS), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    paths = (id_to_path_map[fid] for fid in ids)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in enumerate(ex.map(_featurize_path, paths, chunksize=chunksize)):
            if feat.shape[0] != N_FEATS:
                raise ValueError(
                    f"Feature length mismatch for index={i}: got {feat.shape[0]}, expected {N_FEATS}"
                )
            X_out[i] = feat

    return X_out


y = train_labels["target"].astype(np.int8).values

print("Featurizing train...")
X = load_and_featurize(train_ids, train_id_to_path, chunksize=256)
print("X shape:", X.shape, "y shape:", y.shape)
assert X.shape[1] == N_FEATS



## === cell 3
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

C_GRID = [0.3, 1.0, 3.0, 10.0]
CW_GRID = [None, "balanced"]

splits = list(skf.split(X, y))

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)

_LR_NJOBS = max(1, (os.cpu_count() or 4))


def make_lr(C=1.0, class_weight=None):
    return LogisticRegression(
        max_iter=400,
        solver="lbfgs",
        n_jobs=_LR_NJOBS,
        C=C,
        class_weight=class_weight,
    )


best_params = None
best_cv_auc = -1.0
best_oof = None

print("Selecting C/class_weight via CV (same folds)...")
for C in C_GRID:
    for cw in CW_GRID:
        oof_tmp = np.zeros(len(y), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            lr_tmp = make_lr(C=C, class_weight=cw)
            lr_tmp.fit(X_scaled[tr_idx], y[tr_idx])
            oof_tmp[va_idx] = lr_tmp.predict_proba(X_scaled[va_idx])[:, 1].astype(
                np.float32
            )
        cv_auc_tmp = roc_auc_score(y, oof_tmp)
        print(f"  C={C:<4} class_weight={str(cw):<8} CV AUC={cv_auc_tmp:.6f}")
        if cv_auc_tmp > best_cv_auc:
            best_cv_auc = cv_auc_tmp
            best_params = (C, cw)
            best_oof = oof_tmp.copy()

print(
    "Best params:",
    {"C": best_params[0], "class_weight": best_params[1]},
    "CV AUC:",
    f"{best_cv_auc:.6f}",
)

C_BEST, CW_BEST = best_params

oof = np.zeros(len(y), dtype=np.float32)
fold_models = []
print("Cross-validating with best params...")
for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    lr = make_lr(C=C_BEST, class_weight=CW_BEST)
    lr.fit(X_scaled[tr_idx], y[tr_idx])
    fold_models.append(lr)
    oof[va_idx] = lr.predict_proba(X_scaled[va_idx])[:, 1].astype(np.float32)
    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Overall CV AUC: {cv_auc:.6f}")

print("Preparing fold-ensemble (we will still also fit on full data for reference)...")
print("Fitting on full training data...")
lr_full = make_lr(C=C_BEST, class_weight=CW_BEST)
lr_full.fit(X_scaled, y)



## === cell 4
print("Featurizing test...")
X_test = load_and_featurize(test_ids, test_id_to_path, chunksize=256)
print("X_test shape:", X_test.shape)
assert X_test.shape[1] == N_FEATS

X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)

test_pred_cv = np.zeros(X_test_scaled.shape[0], dtype=np.float32)
for lr_fold in fold_models:
    test_pred_cv += lr_fold.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
test_pred_cv /= len(fold_models)

test_pred_full = lr_full.predict_proba(X_test_scaled)[:, 1].astype(np.float32)

test_pred = np.clip(test_pred_cv, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
assert submission.shape[0] == sample_sub.shape[0]
assert (submission["id"].values == sample_sub["id"].values).all()

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
