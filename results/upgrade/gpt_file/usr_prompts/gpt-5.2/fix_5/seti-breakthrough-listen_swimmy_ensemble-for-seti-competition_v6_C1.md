# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE = "/kaggle/data"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
TRAIN_LABELS_PATH = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

if not os.path.exists(TRAIN_LABELS_PATH):
    BASE = "/kaggle/input"
    TRAIN_DIR = os.path.join(BASE, "train")
    TEST_DIR = os.path.join(BASE, "test")
    TRAIN_LABELS_PATH = os.path.join(BASE, "train_labels.csv")
    SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(
    TRAIN_LABELS_PATH
), f"Missing train_labels.csv at {TRAIN_LABELS_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert {"id", "target"}.issubset(train_labels.columns)
assert {"id", "target"}.issubset(sample_sub.columns)
print("train_labels:", train_labels.shape, "sample_submission:", sample_sub.shape)




## === cell 2
def _gather_npy_paths(root_dir: str):
    out = []
    stack = [root_dir]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        if e.name.endswith(".npy"):
                            out.append(e.path)
        except FileNotFoundError:
            pass
    out.sort()
    return out


train_paths = _gather_npy_paths(TRAIN_DIR)
test_paths = _gather_npy_paths(TEST_DIR)

assert len(train_paths) > 0, f"No .npy files found under {TRAIN_DIR}"
assert len(test_paths) > 0, f"No .npy files found under {TEST_DIR}"


def _id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


train_ids = [_id_from_path(p) for p in train_paths]
test_ids = [_id_from_path(p) for p in test_paths]

train_path_by_id = {i: p for i, p in zip(train_ids, train_paths)}
test_path_by_id = {i: p for i, p in zip(test_ids, test_paths)}

train_labels = train_labels[train_labels["id"].isin(train_path_by_id)].reset_index(
    drop=True
)
print(
    "Train ids with files:", train_labels.shape[0], "Test files:", len(test_path_by_id)
)




## === cell 3
def extract_features_from_snippet(x: np.ndarray) -> np.ndarray:
    """
    x: np.ndarray shape (6, 273, 256)
    returns: 1D float32 feature vector

    Core logic unchanged: same hand-crafted feature set -> logistic regression.
    """
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # on-target
    B = x[[1, 3, 5]]  # off-target

    mean_all = float(x.mean())
    std_all = float(x.std()) + 1e-6
    max_all = float(x.max())
    min_all = float(x.min())

    mean_A = float(A.mean())
    mean_B = float(B.mean())
    std_A = float(A.std()) + 1e-6
    std_B = float(B.std()) + 1e-6

    d = A.mean(axis=0) - B.mean(axis=0)  # (273,256)
    abs_d = np.abs(d)

    l1 = float(abs_d.mean())
    l2 = float(np.sqrt((d * d).mean()))
    d_max = float(d.max())
    d_min = float(d.min())

    row_mean = d.mean(axis=1)  # (273,)
    col_mean = d.mean(axis=0)  # (256,)
    row_std = float(row_mean.std())
    col_std = float(col_mean.std())

    gy = np.diff(d, axis=0)
    gx = np.diff(d, axis=1)
    grad_energy = float((gx * gx).mean() + (gy * gy).mean())

    q90, q95, q99 = np.percentile(abs_d, [90, 95, 99], method="linear")
    q90 = float(q90)
    q95 = float(q95)
    q99 = float(q99)

    frac_q90 = float((abs_d > q90).mean())
    frac_q95 = float((abs_d > q95).mean())
    frac_q99 = float((abs_d > q99).mean())

    abs_row = abs_d.mean(axis=1)
    abs_col = abs_d.mean(axis=0)
    abs_row_max = float(abs_row.max())
    abs_col_max = float(abs_col.max())

    abs_row_p95 = float(np.percentile(abs_row, 95, method="linear"))
    abs_col_p95 = float(np.percentile(abs_col, 95, method="linear"))

    A_panel_mean = A.mean(axis=(1, 2))
    B_panel_mean = B.mean(axis=(1, 2))
    A_panel_std = A.std(axis=(1, 2))
    B_panel_std = B.std(axis=(1, 2))

    A_mean_std = float(A_panel_mean.std())
    B_mean_std = float(B_panel_mean.std())
    A_std_mean = float(A_panel_std.mean())
    B_std_mean = float(B_panel_std.mean())

    A0, A1, A2 = A[0], A[1], A[2]
    B0, B1, B2 = B[0], B[1], B[2]

    def corr(u, v):
        u = u.ravel()
        v = v.ravel()
        u = u - u.mean()
        v = v - v.mean()
        denom = np.sqrt((u * u).mean()) * np.sqrt((v * v).mean()) + 1e-6
        return float((u * v).mean() / denom)

    corr_A01 = corr(A0, A1)
    corr_A12 = corr(A1, A2)
    corr_A02 = corr(A0, A2)

    corr_B01 = corr(B0, B1)
    corr_B12 = corr(B1, B2)
    corr_B02 = corr(B0, B2)

    feats = np.array(
        [
            mean_all,
            std_all,
            max_all,
            min_all,
            mean_A,
            mean_B,
            (mean_A - mean_B),
            std_A,
            std_B,
            (std_A - std_B),
            l1,
            l2,
            d_max,
            d_min,
            row_std,
            col_std,
            grad_energy,
            q90,
            q95,
            q99,
            frac_q90,
            frac_q95,
            frac_q99,
            abs_row_max,
            abs_col_max,
            abs_row_p95,
            abs_col_p95,
            A_mean_std,
            B_mean_std,
            A_std_mean,
            B_std_mean,
            corr_A01,
            corr_A12,
            corr_A02,
            corr_B01,
            corr_B12,
            corr_B02,
        ],
        dtype=np.float32,
    )

    feats[~np.isfinite(feats)] = 0.0
    return feats


def _features_from_path(p: str) -> np.ndarray:
    arr = np.load(p, mmap_mode="r")
    return extract_features_from_snippet(arr)


def build_feature_matrix(
    ids, path_by_id, verbose_every=5000, max_workers=None, chunksize=32
):
    from concurrent.futures import ProcessPoolExecutor

    n = len(ids)
    paths = [path_by_id[_id] for _id in ids]

    first = np.load(paths[0], mmap_mode="r")
    f0 = extract_features_from_snippet(first)
    m = f0.shape[0]

    X = np.empty((n, m), dtype=np.float32)
    X[0] = f0

    if n == 1:
        return X

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(cpu, 8)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_features_from_path, paths[1:], chunksize=chunksize)
        for j, feat in enumerate(it, start=1):
            X[j] = feat
            if verbose_every and (j + 1) % verbose_every == 0:
                print(f"Processed {j+1}/{n} files")

    return X




## === cell 4
train_ids_ordered = train_labels["id"].tolist()
y = train_labels["target"].astype(int).values

X_train = build_feature_matrix(train_ids_ordered, train_path_by_id, verbose_every=10000)

test_ids_ordered = sample_sub["id"].tolist()
missing_test = [i for i in test_ids_ordered if i not in test_path_by_id]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Some sample_submission ids not found on disk (showing up to 5): {missing_test[:5]}"
    )

X_test = build_feature_matrix(test_ids_ordered, test_path_by_id, verbose_every=2000)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)



## === cell 5
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X_train), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=800,
                    n_jobs=None,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    clf.fit(X_tr, y_tr)
    oof[va_idx] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
    test_pred += clf.predict_proba(X_test)[:, 1].astype(np.float32) / n_splits

    print(f"Fold {fold}/{n_splits} done")



## === cell 6
sub = pd.DataFrame({"id": test_ids_ordered, "target": np.clip(test_pred, 0.0, 1.0)})

assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["id", "target"]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())
