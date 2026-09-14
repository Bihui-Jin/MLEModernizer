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

0.5002

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5002) has done: 'The timeout is dominated by per-file `np.load` overhead plus Python multiprocessing IPC for ~60k files (54k train + 6k test), and by expensive `np.quantile` calls per sample. I keep the exact same feature definitions and CV training logic, but speed up feature extraction by (1) reading `.npy` via memory-mapping to avoid copying and reduce RAM pressure, (2) switching to `ThreadPoolExecutor` so NumPy’s C code can run in parallel without pickling/IPC overhead, and (3) using `np.partition`-based exact quantiles for 0.95/0.99 (equivalent to NumPy’s default linear method but much faster here). I also precompute indices and avoid extra Python overhead in loops while preserving identical outputs except negligible float rounding.'

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

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
BASE_INPUT = "/kaggle/input"
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(LABELS_PATH), f"Missing: {LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def build_id_to_path(root_dir: str):
    id_to_path = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        fid = f.name[:-4]
                        id_to_path[fid] = f.path
    return id_to_path


train_id2path = build_id_to_path(TRAIN_DIR)
test_id2path = build_id_to_path(TEST_DIR)

n_train_files = len(train_id2path)
n_test_files = len(test_id2path)
n_labels = len(train_labels)
n_sample = len(sample_sub)

print("Train files:", n_train_files, "Train labels:", n_labels)
print("Test files:", n_test_files, "Sample submission rows:", n_sample)

train_id_set = set(train_id2path.keys())
test_id_set = set(test_id2path.keys())

missing_train = train_labels.loc[~train_labels["id"].isin(train_id_set), "id"]
if len(missing_train) > 0:
    print(
        "Warning: missing train .npy for",
        len(missing_train),
        "label rows. Example:",
        missing_train.iloc[0],
    )

missing_test = sample_sub.loc[~sample_sub["id"].isin(test_id_set), "id"]
if len(missing_test) > 0:
    print(
        "Warning: missing test .npy for",
        len(missing_test),
        "sample rows. Example:",
        missing_test.iloc[0],
    )



## === cell 3


def _quantile_linear_partition(x1d: np.ndarray, q: float) -> np.float32:
    """
    Exact quantile matching numpy default for method='linear' (historically 'interpolation'='linear')
    for 1D array, computed via order statistics using np.partition (O(n)).
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




## === cell 4
from concurrent.futures import ThreadPoolExecutor


def _load_and_extract(path: str):
    arr = np.load(path, mmap_mode="r")  # float16 stored; memory-mapped read
    return extract_features(arr)


train_ids_all = train_labels["id"].to_numpy()
train_targets_all = train_labels["target"].to_numpy(dtype=np.int64)

train_paths = []
train_y = []
train_used_ids = []
for rid, t in zip(train_ids_all, train_targets_all):
    p = train_id2path.get(rid)
    if p is None:
        continue
    train_paths.append(p)
    train_y.append(int(t))
    train_used_ids.append(rid)

train_y = np.asarray(train_y, dtype=np.int64)
n_train = len(train_paths)

n_feats = 19
X = np.empty((n_train, n_feats), dtype=np.float32)

max_workers = min(os.cpu_count() or 2, 8)
chunksize = 64  # larger chunks reduce scheduling overhead for threads

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in enumerate(
        ex.map(_load_and_extract, train_paths, chunksize=chunksize)
    ):
        X[i] = feats

y = train_y
id_list = train_used_ids

print("Train features:", X.shape, "Targets:", y.shape, "Pos rate:", y.mean())



## === cell 5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(y), dtype=np.float32)
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=200,
                    n_jobs=None,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = model.predict_proba(X[va_idx])[:, 1].astype(np.float32)
    models.append(model)
    print(f"Fold {fold} done. Val pred mean={oof[va_idx].mean():.4f}")

print("OOF prediction summary:", float(oof.min()), float(oof.max()), float(oof.mean()))



## === cell 6
test_ids = sample_sub["id"].to_numpy()
n_test = len(test_ids)

test_paths = [test_id2path.get(rid) for rid in test_ids]
missing_mask = np.fromiter((p is None for p in test_paths), dtype=bool, count=n_test)
missing_test_ids = test_ids[missing_mask].tolist()

X_test = np.empty((n_test, n_feats), dtype=np.float32)
X_test[missing_mask] = 0.0

idx_present = np.flatnonzero(~missing_mask)
paths_present = [test_paths[i] for i in idx_present]

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for out_i, feats in zip(
        idx_present, ex.map(_load_and_extract, paths_present, chunksize=chunksize)
    ):
        X_test[out_i] = feats

print("Test features:", X_test.shape)
if missing_test_ids:
    print(
        "Warning: missing test files for",
        len(missing_test_ids),
        "ids. Example:",
        missing_test_ids[0],
    )



## === cell 7
test_pred = np.zeros(len(test_ids), dtype=np.float32)
for model in models:
    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred /= len(models)

test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.head(), submission["target"].describe()



## === cell 8
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission))
print(submission.head())
