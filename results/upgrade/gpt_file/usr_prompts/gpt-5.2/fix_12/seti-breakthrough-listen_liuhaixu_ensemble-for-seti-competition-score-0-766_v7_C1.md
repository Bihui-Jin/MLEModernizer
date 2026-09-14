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

0.7626671920980538

# 6. Current score

0.50555

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50562) has done: 'Your code likely fails to “yield” a Kaggle score because it can crash during feature extraction: `_p99_exact_via_partition` calls `np.argpartition(flat, (i, i + 1))` which is out-of-bounds when `i == n-1` (common due to floating rounding), preventing `submission.csv` from being written. I make a minimal, semantics-preserving fix by clamping the second partition index to `n-1` and handling the edge case safely, so the pipeline runs end-to-end and produces a valid submission. I also force `n_jobs=1` to avoid `multiprocessing` “fork/spawn” issues in constrained notebook environments (this doesn’t change logic, just stability). No model/feature logic is changed beyond the small p99 indexing bugfix, so the score should improve from “not yielded” to a valid AUC submission and move toward your target.'
- What this solution (achieved 0.5055) has done: 'The timeout is dominated by per-file `.npy` loading and feature extraction over ~54k train + 6k test files, with additional overhead from multiprocessing pickling/IPC and repeated allocations inside the feature function. I keep the exact same features and models, but speed up by (1) using a faster path map builder (glob), (2) switching to a thread pool (I/O-bound mmap loads) to avoid process overhead, (3) preallocating output arrays and returning raw feature vectors without extra copies, and (4) micro-optimizing the percentile computation to avoid unnecessary dtype conversions/copies while preserving the exact interpolation semantics. All file paths, features, CV/training logic, and blend weights stay the same; changes only remove overhead and redundant work.'
- What this solution (achieved 0.50555) has done: 'Your current public score (~0.5055) is far below the target (~0.7627), so we should make a small, legitimate improvement that better matches the AUC metric without changing your feature set or model families. The safest high-impact tweak here is calibration: keep the exact same CV/training loop and two linear models, but replace the fixed 0.6/0.4 blend with an out-of-fold–fit blend weight chosen to maximize OOF AUC (still fully leakage-safe). Then train the full models on all data and apply that single learned blend weight to test predictions; this typically improves ranking/AUC with minimal code change. I also keep deterministic settings and submission alignment exactly as you have.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input",  # common Kaggle location
    "/kaggle/data",  # as provided in this environment listing
]


def find_existing_path(rel_path: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {rel_path} under any of {DATA_ROOT_CANDIDATES}"
    )


TRAIN_LABELS_PATH = find_existing_path("train_labels.csv")
SAMPLE_SUB_PATH = find_existing_path("sample_submission.csv")
TRAIN_DIR = find_existing_path("train")
TEST_DIR = find_existing_path("test")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("train_labels:", train_labels.shape, "sample_submission:", sample_sub.shape)
print(
    "TRAIN_DIR exists:",
    os.path.isdir(TRAIN_DIR),
    "TEST_DIR exists:",
    os.path.isdir(TEST_DIR),
)




## === cell 2
def build_id_to_path_map(root_dir: str) -> dict:
    out = {}
    pattern = os.path.join(root_dir, "*", "*.npy")
    for p in glob.iglob(pattern):
        fn = os.path.basename(p)
        out[fn[:-4]] = p
    return out


train_path_by_id = build_id_to_path_map(TRAIN_DIR)
test_path_by_id = build_id_to_path_map(TEST_DIR)

print(
    "Found train .npy:", len(train_path_by_id), "Found test .npy:", len(test_path_by_id)
)

train_labels = train_labels[train_labels["id"].isin(train_path_by_id)].reset_index(
    drop=True
)
print("Usable labeled train rows:", train_labels.shape)




## === cell 3
def _p99_exact_via_partition_float32(flat_f32_1d: np.ndarray) -> np.float32:
    n = flat_f32_1d.size
    if n == 0:
        return np.float32(np.nan)

    h = (n - 1) * 0.99
    i = int(h)
    g = np.float32(h - i)

    if i < 0:
        i = 0
        g = np.float32(0.0)
    if i >= n - 1:
        i = n - 1
        g = np.float32(0.0)

    j = i + 1 if i + 1 < n else i
    part = np.partition(flat_f32_1d, j)
    x_i = part[i]
    if j == i:
        return np.float32(x_i)
    x_j = part[j]
    return np.float32(x_i + (x_j - x_i) * g)


def extract_features_from_snippet(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # on-target
    BCD = x[[1, 3, 5]]  # off-target

    A_mean = A.mean()
    O_mean = BCD.mean()
    A_std = A.std()
    O_std = BCD.std()

    diff_mean = A_mean - O_mean
    ratio_std = (A_std + 1e-6) / (O_std + 1e-6)

    A_p99 = _p99_exact_via_partition_float32(A.reshape(-1))
    O_p99 = _p99_exact_via_partition_float32(BCD.reshape(-1))
    p99_diff = A_p99 - O_p99

    A_max = A.max()
    O_max = BCD.max()
    max_diff = A_max - O_max

    A_dt = np.abs(np.diff(A, axis=1)).mean()
    O_dt = np.abs(np.diff(BCD, axis=1)).mean()
    dt_ratio = (A_dt + 1e-6) / (O_dt + 1e-6)

    A_df = np.abs(np.diff(A, axis=2)).mean()
    O_df = np.abs(np.diff(BCD, axis=2)).mean()
    df_ratio = (A_df + 1e-6) / (O_df + 1e-6)

    return np.array(
        [
            A_mean,
            O_mean,
            A_std,
            O_std,
            diff_mean,
            ratio_std,
            A_p99,
            O_p99,
            p99_diff,
            A_max,
            O_max,
            max_diff,
            A_dt,
            O_dt,
            dt_ratio,
            A_df,
            O_df,
            df_ratio,
        ],
        dtype=np.float32,
    )


from concurrent.futures import ThreadPoolExecutor


def _featurize_path(p: str) -> np.ndarray:
    x = np.load(p, mmap_mode="r")
    return extract_features_from_snippet(x)


def build_feature_df(
    ids, path_by_id, label_df=None, max_rows=None, n_jobs=None, chunksize=None
):
    if max_rows is not None:
        ids = ids[:max_rows]

    paths = []
    ids_ok = []
    paths_append = paths.append
    ids_ok_append = ids_ok.append
    for _id in ids:
        p = path_by_id.get(_id)
        if p is not None:
            ids_ok_append(_id)
            paths_append(p)

    n = len(paths)
    n_feat = 18
    feats_mat = np.empty((n, n_feat), dtype=np.float32)

    if n_jobs is None:
        cpu = os.cpu_count() or 2
        n_jobs = min(16, max(1, cpu))  # cap to avoid oversubscription

    if n_jobs <= 1:
        for i, p in enumerate(paths):
            feats_mat[i] = _featurize_path(p)
            if (i + 1) % 5000 == 0:
                print(f"Processed {i+1}/{n}")
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for i, f in enumerate(ex.map(_featurize_path, paths, chunksize=256)):
                feats_mat[i] = f
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i+1}/{n} (threads)")

    df = pd.DataFrame(feats_mat, columns=[f"f{j:02d}" for j in range(n_feat)])
    df.insert(0, "id", np.array(ids_ok, dtype=object))

    if label_df is not None:
        df = df.merge(label_df[["id", "target"]], on="id", how="left")
    return df




## === cell 4
train_ids = train_labels["id"].tolist()
Xy = build_feature_df(train_ids, train_path_by_id, label_df=train_labels)
print("Train feature df:", Xy.shape)
print(Xy.head())

assert Xy["target"].notna().all(), "Some training targets are missing after merge."



## === cell 5
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

feature_cols = [c for c in Xy.columns if c.startswith("f")]
X = Xy[feature_cols].to_numpy(dtype=np.float32)
y = Xy["target"].to_numpy(dtype=np.int64)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lr = np.zeros(len(X), dtype=np.float32)
oof_ridge = np.zeros(len(X), dtype=np.float32)


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
        ]
    )
    lr.fit(X_tr, y_tr)
    oof_lr[va] = lr.predict_proba(X_va)[:, 1].astype(np.float32)

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=1.0, random_state=42)),
        ]
    )
    ridge.fit(X_tr, y_tr)
    oof_ridge[va] = sigmoid(ridge.decision_function(X_va)).astype(np.float32)

    print(f"Finished fold {fold}")

auc_lr = roc_auc_score(y, oof_lr)
auc_ridge = roc_auc_score(y, oof_ridge)

grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
best_w = 0.6
best_auc = -1.0
for w in grid:
    auc = roc_auc_score(y, w * oof_lr + (1.0 - w) * oof_ridge)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

print(
    "OOF AUC lr:",
    auc_lr,
    "ridge:",
    auc_ridge,
    "best_blend_auc:",
    best_auc,
    "best_w(lr):",
    best_w,
)



## === cell 6
test_ids = sample_sub["id"].tolist()  # ensure we predict exactly required ids/order
existing_test_ids = [_id for _id in test_ids if _id in test_path_by_id]

test_feat_df = build_feature_df(existing_test_ids, test_path_by_id, label_df=None)
print("Test feature df:", test_feat_df.shape)

aligned = sample_sub[["id"]].merge(
    test_feat_df[["id"] + feature_cols], on="id", how="left"
)
X_test = aligned[feature_cols].to_numpy(dtype=np.float32)

lr_full = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
    ]
)
lr_full.fit(X, y)

ridge_full = Pipeline(
    [("scaler", StandardScaler()), ("clf", RidgeClassifier(alpha=1.0, random_state=42))]
)
ridge_full.fit(X, y)

pred_lr = np.full(len(sample_sub), 0.5, dtype=np.float32)
pred_ridge = np.full(len(sample_sub), 0.5, dtype=np.float32)

mask = ~np.isnan(X_test).any(axis=1)
if mask.any():
    pred_lr[mask] = lr_full.predict_proba(X_test[mask])[:, 1].astype(np.float32)
    pred_ridge[mask] = sigmoid(ridge_full.decision_function(X_test[mask])).astype(
        np.float32
    )

pred = best_w * pred_lr + (1.0 - best_w) * pred_ridge
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_sub["id"], "target": pred.astype(np.float32)})

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["id", "target"]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
print(submission.head())
