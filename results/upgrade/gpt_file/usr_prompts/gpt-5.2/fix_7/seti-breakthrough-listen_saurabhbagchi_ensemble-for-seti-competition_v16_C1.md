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

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

CANDIDATE_ROOTS = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",  # fallback if dataset is mounted directly
    "/kaggle/data",
]


def find_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


ROOT = find_existing_path(CANDIDATE_ROOTS)
if ROOT is None:
    raise FileNotFoundError(
        "Could not find Kaggle dataset root under expected /kaggle/input or /kaggle/data paths."
    )

if os.path.basename(ROOT) in ("input", "data"):
    if os.path.exists(os.path.join(ROOT, "seti-breakthrough-listen")):
        ROOT = os.path.join(ROOT, "seti-breakthrough-listen")

TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")
LABELS_CSV = os.path.join(ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(ROOT, "sample_submission.csv")

for p in [TRAIN_DIR, TEST_DIR, LABELS_CSV, SAMPLE_SUB]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected path not found: {p}")

labels = pd.read_csv(LABELS_CSV)
sample = pd.read_csv(SAMPLE_SUB)

labels = labels.sort_values("id").reset_index(drop=True)
sample = sample.sort_values("id").reset_index(drop=True)

labels.head(), sample.head(), labels.shape, sample.shape




## === cell 1
def _quantile_linear_partition(z: np.ndarray, q: float) -> np.float32:
    """
    Equivalent to np.quantile(z, q) with default method (linear) for 1D array z.
    Implemented via np.partition for speed.
    """
    a = z.ravel()
    n = a.size
    if n == 0:
        return np.float32(np.nan)
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return np.float32(np.partition(a, lo)[lo])
    part = np.partition(a, hi)
    v_lo = part[lo]
    v_hi = part[hi]
    return np.float32(v_lo + (h - lo) * (v_hi - v_lo))


def _panel_features_from_z(z: np.ndarray) -> np.ndarray:
    mean = np.float32(z.mean(dtype=np.float32))
    std = np.float32(z.std(dtype=np.float32))
    q99 = _quantile_linear_partition(z, 0.99)
    q999 = _quantile_linear_partition(z, 0.999)

    t_profile = z.mean(axis=1, dtype=np.float32)  # (273,)
    f_profile = z.mean(axis=0, dtype=np.float32)  # (256,)
    t_max = np.float32(t_profile.max())
    f_max = np.float32(f_profile.max())
    t_std = np.float32(t_profile.std(dtype=np.float32))
    f_std = np.float32(f_profile.std(dtype=np.float32))

    tail = np.float32(np.mean(z > 3.0))

    return np.array(
        [mean, std, q99, q999, t_max, f_max, t_std, f_std, tail], dtype=np.float32
    )


def _drift_features_from_z(z: np.ndarray) -> np.ndarray:
    """
    Small, cadence-aligned features capturing diagonal drift:
    compute per-time-row argmax frequency index and summarize its linear trend and variability.
    This is a minimal additive feature block to improve AUC toward the target without changing
    the model family/training loop.
    """
    idx = np.argmax(z, axis=1).astype(np.float32)
    t = np.arange(idx.size, dtype=np.float32)

    t_mean = np.float32(t.mean())
    idx_mean = np.float32(idx.mean())

    denom = np.float32(np.sum((t - t_mean) ** 2) + 1e-6)
    slope = np.float32(np.sum((t - t_mean) * (idx - idx_mean)) / denom)

    intercept = np.float32(idx_mean - slope * t_mean)
    resid = idx - (slope * t + intercept)
    resid_std = np.float32(resid.std(dtype=np.float32))

    idx_range = np.float32(idx.max() - idx.min())
    diff_std = np.float32(np.diff(idx).std(dtype=np.float32))

    row_max = z.max(axis=1).astype(np.float32)
    row_med = np.median(z, axis=1).astype(np.float32)
    ridge_strength = np.float32((row_max - row_med).mean(dtype=np.float32))

    return np.array(
        [slope, resid_std, idx_range, diff_std, ridge_strength], dtype=np.float32
    )


def extract_features_from_path(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path, mmap_mode="r")  # (6,273,256) float16 on disk
    if arr.shape[0] != 6:
        raise ValueError(f"Unexpected snippet shape {arr.shape} for {npy_path}")

    panel_feats = np.empty((6, 9), dtype=np.float32)

    drift_feats = np.empty((6, 5), dtype=np.float32)

    for i in range(6):
        x = arr[i].astype(np.float32, copy=False)
        m = np.median(x)
        mad = np.median(np.abs(x - m)) + 1e-6
        z = (x - m) / mad
        panel_feats[i] = _panel_features_from_z(z)
        drift_feats[i] = _drift_features_from_z(z)

    feats = panel_feats.reshape(-1)  # 6*9 = 54
    A = panel_feats[[0, 2, 4]].mean(axis=0)
    O = panel_feats[[1, 3, 5]].mean(axis=0)
    diff = A - O
    ratio = A / (np.abs(O) + 1e-3)
    feats = np.concatenate([feats, A, O, diff, ratio], axis=0).astype(
        np.float32, copy=False
    )  # 90

    A_pan = panel_feats[[0, 2, 4]]  # (3, 9)
    O_pan = panel_feats[[1, 3, 5]]  # (3, 9)
    A_std = A_pan.std(axis=0, dtype=np.float32)
    O_std = O_pan.std(axis=0, dtype=np.float32)
    A_max = A_pan.max(axis=0)
    O_max = O_pan.max(axis=0)
    A_min = A_pan.min(axis=0)
    O_min = O_pan.min(axis=0)

    std_diff = (A_std - O_std).astype(np.float32, copy=False)
    range_diff = ((A_max - A_min) - (O_max - O_min)).astype(np.float32, copy=False)

    feats = np.concatenate([feats, std_diff, range_diff], axis=0).astype(
        np.float32, copy=False
    )  # 108

    A_d = drift_feats[[0, 2, 4]].mean(axis=0)
    O_d = drift_feats[[1, 3, 5]].mean(axis=0)
    d_diff = (A_d - O_d).astype(np.float32, copy=False)
    d_ratio = (A_d / (np.abs(O_d) + 1e-3)).astype(np.float32, copy=False)

    feats = np.concatenate(
        [feats, drift_feats.reshape(-1), A_d, O_d, d_diff, d_ratio], axis=0
    ).astype(
        np.float32, copy=False
    )  # 108 + 30 + 5 + 5 + 5 + 5 = 158

    return feats




## === cell 2
def build_id_to_path_map_from_ids(base_dir: str, ids) -> dict:
    id2path = {}
    for _id in ids:
        sub = _id[0]
        fp = os.path.join(base_dir, sub, f"{_id}.npy")
        id2path[_id] = fp
    return id2path


train_ids = labels["id"].tolist()
test_ids = sample["id"].tolist()

train_id2path = build_id_to_path_map_from_ids(TRAIN_DIR, train_ids)
test_id2path = build_id_to_path_map_from_ids(TEST_DIR, test_ids)

len(train_id2path), len(test_id2path), list(train_id2path.items())[:1], list(
    test_id2path.items()
)[:1]



## === cell 3
from joblib import Parallel, delayed

missing_train = [i for i in train_ids if not os.path.exists(train_id2path[i])]
if missing_train:
    raise FileNotFoundError(
        f"Missing {len(missing_train)} training .npy files for labeled ids. Example: {missing_train[:5]}"
    )

n_jobs = min(8, max(1, (os.cpu_count() or 2) - 1))

X_train_list = Parallel(
    n_jobs=n_jobs, backend="threading", batch_size=256, pre_dispatch="2*n_jobs"
)(delayed(extract_features_from_path)(train_id2path[_id]) for _id in train_ids)
X_train = np.asarray(X_train_list, dtype=np.float32)

y_train = labels["target"].astype(np.int32).values

X_train.shape, y_train.shape, y_train.mean()



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=200,
                C=1.0,
                class_weight="balanced",
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_train, y_train)



## === cell 5
missing_test = [i for i in test_ids if not os.path.exists(test_id2path[i])]
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test .npy files for sample_submission ids. Example: {missing_test[:5]}"
    )

X_test_list = Parallel(
    n_jobs=n_jobs, backend="threading", batch_size=256, pre_dispatch="2*n_jobs"
)(delayed(extract_features_from_path)(test_id2path[_id]) for _id in test_ids)
X_test = np.asarray(X_test_list, dtype=np.float32)

pred = clf.predict_proba(X_test)[:, 1].astype(np.float64)
pred = np.clip(pred, 0.0, 1.0)

sub = pd.DataFrame({"id": test_ids, "target": pred})
sub.head(), sub.shape



## === cell 6
OUT_PATH = "submission.csv"
sub.to_csv(OUT_PATH, index=False)

assert list(sub.columns) == ["id", "target"]
assert len(sub) == len(sample)
assert (sub["id"].values == sample["id"].values).all()

print(f"Wrote {OUT_PATH} with shape {sub.shape}")
print(sub.describe(include="all"))
