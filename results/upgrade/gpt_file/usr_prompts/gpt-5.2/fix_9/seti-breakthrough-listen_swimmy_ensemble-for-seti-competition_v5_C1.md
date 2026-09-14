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
import glob
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_INPUT = "/kaggle/input"
DATA_ROOT_CANDIDATES = [
    os.path.join(BASE_INPUT, "seti-breakthrough-listen"),
    os.path.join(BASE_INPUT, "seti-breakthrough-listen", "seti-breakthrough-listen"),
    os.path.join(BASE_INPUT, "data"),
    os.path.join(BASE_INPUT, "input"),
]


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


DATA_ROOT = first_existing(*DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset root among: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_DIR = first_existing(os.path.join(DATA_ROOT, "train"))
TEST_DIR = first_existing(os.path.join(DATA_ROOT, "test"))
LABELS_PATH = first_existing(os.path.join(DATA_ROOT, "train_labels.csv"))
SAMPLE_SUB_PATH = first_existing(os.path.join(DATA_ROOT, "sample_submission.csv"))

for pth, nm in [
    (TRAIN_DIR, "TRAIN_DIR"),
    (TEST_DIR, "TEST_DIR"),
    (LABELS_PATH, "LABELS_PATH"),
    (SAMPLE_SUB_PATH, "SAMPLE_SUB_PATH"),
]:
    if pth is None:
        raise FileNotFoundError(
            f"Missing required path for {nm} under DATA_ROOT={DATA_ROOT}"
        )

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("LABELS_PATH:", LABELS_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 2
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import multiprocessing as mp


def list_npy_files(root_dir):
    files = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    files.sort()
    return files


def ids_from_files(files):
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

train_ids = ids_from_files(train_files)
test_ids = ids_from_files(test_files)

train_labels = pd.read_csv(LABELS_PATH)

train_labels_idx = train_labels.set_index("id")
missing_train_labels = [i for i in train_ids if i not in train_labels_idx.index]
if missing_train_labels:
    raise ValueError(
        f"Some train ids have no labels (count={len(missing_train_labels)}). "
        f"Example: {missing_train_labels[:5]}"
    )
train_labels_aligned = train_labels_idx.loc[train_ids].reset_index()

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

id_to_test_file = {os.path.splitext(os.path.basename(f))[0]: f for f in test_files}
test_ids_ordered = sample_sub["id"].tolist()

missing_test_files = [i for i in test_ids_ordered if i not in id_to_test_file]
if missing_test_files:
    raise ValueError(
        f"Some sample_submission ids are missing from test files (count={len(missing_test_files)}). "
        f"Example: {missing_test_files[:5]}"
    )

test_files_ordered = [id_to_test_file[i] for i in test_ids_ordered]

_Q_95 = 0.95
_Q_99 = 0.99
_Q_05 = 0.05
_Q_01 = 0.01


def _quantiles_linear_partition_1d(a1d, qs):
    """
    Compute multiple quantiles (linear interpolation between order stats) using ONE np.partition.
    This preserves the exact semantics of _quantile_linear_partition_1d for each q.
    """
    n = a1d.size
    if n == 0:
        return [np.nan] * len(qs)

    pos = (n - 1) * np.asarray(qs, dtype=np.float64)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)

    kth = np.unique(np.concatenate([lo, hi]))
    part = np.partition(a1d, kth)

    out = []
    for p, l, h in zip(pos, lo, hi):
        if l == h:
            out.append(float(part[l]))
        else:
            v_lo = float(part[l])
            v_hi = float(part[h])
            w = float(p - l)
            out.append(v_lo + (v_hi - v_lo) * w)
    return out


def extract_features_from_array(x):
    """
    x shape: (6, 273, 256)
    Cadence panels: A,B,A,C,A,D -> indices 0,1,2,3,4,5
    """
    x = x.astype(np.float32, copy=False)

    A0, B, A1, C, A2, Dp = x[0], x[1], x[2], x[3], x[4], x[5]
    A = (A0 + A1 + A2) * (1.0 / 3.0)
    O = (B + C + Dp) * (1.0 / 3.0)

    Diff = A - O
    AD = np.abs(Diff)

    diff_flat = Diff.ravel()
    ad_flat = AD.ravel()

    diff_mean = float(Diff.mean())
    diff_std = float(Diff.std())
    ad_mean = float(AD.mean())
    ad_std = float(AD.std())
    ad_max = float(AD.max())

    ad_q95, ad_q99 = _quantiles_linear_partition_1d(ad_flat, (_Q_95, _Q_99))
    diff_q05, diff_q95, diff_q01, diff_q99 = _quantiles_linear_partition_1d(
        diff_flat, (_Q_05, _Q_95, _Q_01, _Q_99)
    )

    feats = [
        diff_mean,
        diff_std,
        ad_mean,
        ad_std,
        ad_max,
        ad_q95,
        ad_q99,
    ]

    row = AD.mean(axis=1)  # (273,)
    col = AD.mean(axis=0)  # (256,)

    row_mean = float(row.mean())
    row_std = float(row.std())
    row_max = float(row.max())
    (row_q95,) = _quantiles_linear_partition_1d(row, (_Q_95,))

    col_mean = float(col.mean())
    col_std = float(col.std())
    col_max = float(col.max())
    (col_q95,) = _quantiles_linear_partition_1d(col, (_Q_95,))

    feats += [row_mean, row_std, row_max, row_q95]
    feats += [col_mean, col_std, col_max, col_q95]

    eps = 1e-6
    absA_mean = float(np.abs(A).mean())
    absO_mean = float(np.abs(O).mean())
    sqA_mean = float(np.square(A).mean())
    sqO_mean = float(np.square(O).mean())
    feats += [
        (absA_mean + eps) / (absO_mean + eps),
        (sqA_mean + eps) / (sqO_mean + eps),
    ]

    feats += [
        diff_q05,
        diff_q95,
        diff_q01,
        diff_q99,
    ]

    absA_pan = (np.abs(A0).mean() + np.abs(A1).mean() + np.abs(A2).mean()) / 3.0
    absO_pan = (np.abs(B).mean() + np.abs(C).mean() + np.abs(Dp).mean()) / 3.0
    sqA_pan = (np.square(A0).mean() + np.square(A1).mean() + np.square(A2).mean()) / 3.0
    sqO_pan = (np.square(B).mean() + np.square(C).mean() + np.square(Dp).mean()) / 3.0
    feats += [
        float(absA_pan - absO_pan),
        float(sqA_pan - sqO_pan),
    ]

    A_t = A.max(axis=1)  # (273,)
    O_t = O.max(axis=1)
    d = A_t - O_t
    feats += [
        float(d.mean()),
        float(np.abs(d).mean()),
        float(A_t.max()),
        float(O_t.max()),
    ]

    return np.asarray(feats, dtype=np.float32)


def _features_from_file(fpath):
    arr = np.load(fpath, mmap_mode="r")
    return extract_features_from_array(arr)


def build_feature_matrix(files):
    n = len(files)
    X = np.empty((n, 27), dtype=np.float32)

    cpu = os.cpu_count() or 2
    workers = min(cpu, 8)  # cap to reduce process overhead/IO contention
    chunksize = 64

    if workers <= 1:
        for i, f in enumerate(files):
            X[i] = _features_from_file(f)
        return X

    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for i, feat in enumerate(
            pool.imap(_features_from_file, files, chunksize=chunksize)
        ):
            X[i] = feat
    return X


X_train = build_feature_matrix(train_files)
y_train = train_labels_aligned["target"].values.astype(np.int64)

X_test = build_feature_matrix(test_files_ordered)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "pos_rate:", y_train.mean())
print("X_test:", X_test.shape, "sample_sub:", sample_sub.shape)



## === cell 3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X_train), dtype=np.float32)
test_logit_sum = np.zeros(len(X_test), dtype=np.float64)


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[tr_idx])
    X_va = scaler.transform(X_train[va_idx])
    X_te = scaler.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_tr, y_train[tr_idx])

    oof[va_idx] = model.predict_proba(X_va)[:, 1].astype(np.float32)
    test_logit_sum += model.decision_function(X_te).astype(np.float64) / skf.n_splits

test_pred = _sigmoid(test_logit_sum).astype(np.float32)

print("OOF pred summary:", float(oof.min()), float(oof.mean()), float(oof.max()))
print(
    "Test pred summary:",
    float(test_pred.min()),
    float(test_pred.mean()),
    float(test_pred.max()),
)
print("OOF AUC:", float(roc_auc_score(y_train, oof)))



## === cell 4
sub = sample_sub.copy()

sub["target"] = test_pred.astype(np.float32)
sub["target"] = sub["target"].clip(0.0, 1.0)

if len(sub) != len(sample_sub):
    raise ValueError("Submission row count mismatch.")
if sub["target"].isna().any():
    raise ValueError("Submission contains NaN predictions.")

print(sub.head())



## === cell 5
data6 = sub



## === cell 6
data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
