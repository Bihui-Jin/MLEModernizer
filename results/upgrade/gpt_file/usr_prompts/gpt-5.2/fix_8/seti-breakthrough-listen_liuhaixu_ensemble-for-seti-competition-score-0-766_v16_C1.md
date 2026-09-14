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
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
DATA_ROOT_CANDIDATES = [
    Path("../input/seti-breakthrough-listen"),
    Path("/kaggle/input/seti-breakthrough-listen"),
    Path("../input"),  # fallback if data is directly under ../input
    Path("/kaggle/input"),
]


def find_data_root():
    for root in DATA_ROOT_CANDIDATES:
        if (
            (root / "train_labels.csv").exists()
            and (root / "train").exists()
            and (root / "test").exists()
        ):
            return root
        if (root / "seti-breakthrough-listen" / "train_labels.csv").exists():
            return root / "seti-breakthrough-listen"
    raise FileNotFoundError(
        "Could not locate dataset root containing train_labels.csv, train/, test/"
    )


DATA_ROOT = find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
LABELS_PATH = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR exists:", TRAIN_DIR.exists(), "TEST_DIR exists:", TEST_DIR.exists())
print(
    "LABELS_PATH exists:",
    LABELS_PATH.exists(),
    "SAMPLE_SUB_PATH exists:",
    SAMPLE_SUB_PATH.exists(),
)

labels = pd.read_csv(LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(labels.head())
print(sample_sub.head())




## === cell 2
def id_to_path(folder: Path, id_: str) -> Path:
    s = str(id_)
    return folder / s[0] / f"{s}.npy"


def _percentiles_nearest_rank_flat(a_flat: np.ndarray, qs) -> np.ndarray:
    n = a_flat.size
    if n == 0:
        return np.full(len(qs), np.nan, dtype=np.float32)
    ks = np.floor((np.asarray(qs, dtype=np.float32) / 100.0) * (n - 1)).astype(np.int64)
    part = np.partition(a_flat, ks)
    return part[ks].astype(np.float32, copy=False)


def _mean_pixelwise_std3(A0, A1, A2) -> np.float32:
    A0 = A0.astype(np.float32, copy=False)
    A1 = A1.astype(np.float32, copy=False)
    A2 = A2.astype(np.float32, copy=False)
    m = (A0 + A1 + A2) * (1.0 / 3.0)
    m2 = (A0 * A0 + A1 * A1 + A2 * A2) * (1.0 / 3.0)
    var = m2 - m * m
    np.maximum(var, 0.0, out=var)
    return np.sqrt(var, dtype=np.float32).mean(dtype=np.float32).astype(np.float32)


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A0, B, A1, C, A2, D = x[0], x[1], x[2], x[3], x[4], x[5]

    A_mean = (A0 + A1 + A2) * (1.0 / 3.0)
    off_mean = (B + C + D) * (1.0 / 3.0)

    diff = np.empty_like(A_mean, dtype=np.float32)
    np.subtract(A_mean, off_mean, out=diff)

    adiff = np.empty_like(diff, dtype=np.float32)
    np.abs(diff, out=adiff)

    diff_flat = diff.reshape(-1)
    adiff_flat = adiff.reshape(-1)

    diff_mean = diff_flat.mean(dtype=np.float32)
    diff_std = diff_flat.std(dtype=np.float32)

    diff_med, diff_p90, diff_p10 = _percentiles_nearest_rank_flat(
        diff_flat, [50.0, 90.0, 10.0]
    )

    adiff_mean = adiff_flat.mean(dtype=np.float32)
    adiff_std = adiff_flat.std(dtype=np.float32)

    adiff_med, adiff_p95, adiff_p99 = _percentiles_nearest_rank_flat(
        adiff_flat, [50.0, 95.0, 99.0]
    )

    feats = [
        diff_mean,
        diff_std,
        diff_med,
        diff_p90,
        diff_p10,
        adiff_mean,
        adiff_std,
        adiff_med,
        adiff_p95,
        adiff_p99,
    ]

    row_mean = diff.mean(axis=1, dtype=np.float32)  # (273,)
    col_mean = diff.mean(axis=0, dtype=np.float32)  # (256,)

    row_std = row_mean.std(dtype=np.float32)
    row_p95, row_p5 = _percentiles_nearest_rank_flat(row_mean, [95.0, 5.0])

    col_std = col_mean.std(dtype=np.float32)
    col_p95, col_p5 = _percentiles_nearest_rank_flat(col_mean, [95.0, 5.0])

    feats += [
        row_std,
        (row_p95 - row_p5).astype(np.float32),
        col_std,
        (col_p95 - col_p5).astype(np.float32),
    ]

    A_stack_mean = (
        (
            A0.mean(dtype=np.float32)
            + A1.mean(dtype=np.float32)
            + A2.mean(dtype=np.float32)
        )
        * (1.0 / 3.0)
    ).astype(np.float32)
    off_stack_mean = (
        (B.mean(dtype=np.float32) + C.mean(dtype=np.float32) + D.mean(dtype=np.float32))
        * (1.0 / 3.0)
    ).astype(np.float32)

    def _pooled_std3(X1, X2, X3) -> np.float32:
        m1 = X1.mean(dtype=np.float32)
        m2 = X2.mean(dtype=np.float32)
        m3 = X3.mean(dtype=np.float32)
        v1 = X1.var(dtype=np.float32)
        v2 = X2.var(dtype=np.float32)
        v3 = X3.var(dtype=np.float32)
        n1 = np.float32(X1.size)
        n2 = np.float32(X2.size)
        n3 = np.float32(X3.size)
        N = n1 + n2 + n3
        m = (n1 * m1 + n2 * m2 + n3 * m3) / N
        var = (
            n1 * (v1 + (m1 - m) * (m1 - m))
            + n2 * (v2 + (m2 - m) * (m2 - m))
            + n3 * (v3 + (m3 - m) * (m3 - m))
        ) / N
        return np.sqrt(var, dtype=np.float32).astype(np.float32)

    A_stack_std = _pooled_std3(A0, A1, A2)
    off_stack_std = _pooled_std3(B, C, D)

    A_temporal_var = _mean_pixelwise_std3(A0, A1, A2)
    off_temporal_var = _mean_pixelwise_std3(B, C, D)

    feats += [
        A_stack_mean,
        A_stack_std,
        off_stack_mean,
        off_stack_std,
        A_temporal_var.astype(np.float32),
        off_temporal_var.astype(np.float32),
    ]

    return np.asarray(feats, dtype=np.float32)


def load_npy_by_path(p: Path) -> np.ndarray:
    return np.load(p, mmap_mode="r")


tmp_id = labels["id"].iloc[0]
tmp_x = load_npy_by_path(id_to_path(TRAIN_DIR, str(tmp_id)))
tmp_f = extract_features_from_array(tmp_x)
print("Array shape:", tmp_x.shape, "Feature dim:", tmp_f.shape)



## === cell 3
import zipfile

WORK_ROOT = Path("/kaggle/working")
LOCAL_DATA_ROOT = WORK_ROOT / "_seti_local"
LOCAL_TRAIN_DIR = LOCAL_DATA_ROOT / "train"
LOCAL_TEST_DIR = LOCAL_DATA_ROOT / "test"


def _is_valid_zip(path: Path) -> bool:
    try:
        return path.exists() and zipfile.is_zipfile(path)
    except OSError:
        return False


def _maybe_extract_split(split: str, local_dir: Path) -> Path:
    if local_dir.exists():
        return local_dir

    split_dir = DATA_ROOT / split
    if split_dir.exists():
        return split_dir

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    zip_candidates = [
        DATA_ROOT / f"{split}.zip",
        DATA_ROOT.parent / f"{split}.zip",
        Path("/kaggle/input") / f"{split}.zip",
        Path("../input") / f"{split}.zip",
    ]
    zip_path = next((p for p in zip_candidates if _is_valid_zip(p)), None)
    if zip_path is None:
        return split_dir

    print(f"Extracting {zip_path} -> {LOCAL_DATA_ROOT} (one-time, for faster I/O)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{split}/")]
        zf.extractall(LOCAL_DATA_ROOT, members)
    return local_dir


TRAIN_DIR_FAST = _maybe_extract_split("train", LOCAL_TRAIN_DIR)
TEST_DIR_FAST = _maybe_extract_split("test", LOCAL_TEST_DIR)

print("TRAIN_DIR_FAST:", TRAIN_DIR_FAST, "exists:", TRAIN_DIR_FAST.exists())
print("TEST_DIR_FAST :", TEST_DIR_FAST, "exists:", TEST_DIR_FAST.exists())



## === cell 4
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

feat_dim = tmp_f.shape[0]
n_train = len(labels)
y = labels["target"].astype(int).values
ids_train = labels["id"].astype(str).values

_FEAT_BASE = None


def _init_featurizer(folder_str: str):
    global _FEAT_BASE
    _FEAT_BASE = folder_str


def _featurize_id_batch(id_list):
    out = np.empty((len(id_list), feat_dim), dtype=np.float32)
    base = _FEAT_BASE
    for i, id_str in enumerate(id_list):
        p = f"{base}/{id_str[0]}/{id_str}.npy"
        x = np.load(p, mmap_mode="r")
        out[i] = extract_features_from_array(x)
    return out


def _iter_batches(arr, batch_size: int):
    n = len(arr)
    for i in range(0, n, batch_size):
        yield arr[i : i + batch_size]


X = np.empty((n_train, feat_dim), dtype=np.float32)

cpu = os.cpu_count() or 2
max_workers = min(6, cpu)
batch_size = 1024

try:
    ctx = mp.get_context("fork")
except ValueError:
    ctx = mp.get_context("spawn")

with ProcessPoolExecutor(
    mp_context=ctx,
    max_workers=max_workers,
    initializer=_init_featurizer,
    initargs=(str(TRAIN_DIR_FAST),),
) as ex:
    write_pos = 0
    for feats in ex.map(
        _featurize_id_batch, _iter_batches(ids_train, batch_size), chunksize=1
    ):
        n_b = feats.shape[0]
        X[write_pos : write_pos + n_b] = feats
        write_pos += n_b
        if write_pos % 5000 == 0 or write_pos == n_train:
            print(
                f"Extracted {write_pos}/{n_train} train samples (workers={max_workers}, batch={batch_size})"
            )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(n_train, dtype=np.float32)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None)),
    ]
)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    model.fit(X[trn_idx], y[trn_idx])
    oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    auc = roc_auc_score(y[val_idx], oof[val_idx])
    print(f"Fold {fold} AUC: {auc:.5f}")

print("OOF AUC:", roc_auc_score(y, oof))

model.fit(X, y)



## === cell 5
test_ids = sample_sub["id"].astype(str).values
n_test = len(test_ids)
Xt = np.empty((n_test, X.shape[1]), dtype=np.float32)


def _featurize_test_id_batch(id_list):
    out = np.empty((len(id_list), feat_dim), dtype=np.float32)
    base = _FEAT_BASE
    for i, id_str in enumerate(id_list):
        p = f"{base}/{id_str[0]}/{id_str}.npy"
        if not os.path.exists(p):
            raise FileNotFoundError(f"Test id {id_str} not found under {base}")
        x = np.load(p, mmap_mode="r")
        out[i] = extract_features_from_array(x)
    return out


with ProcessPoolExecutor(
    mp_context=ctx,
    max_workers=max_workers,
    initializer=_init_featurizer,
    initargs=(str(TEST_DIR_FAST),),
) as ex:
    write_pos = 0
    for feats in ex.map(
        _featurize_test_id_batch, _iter_batches(test_ids, batch_size), chunksize=1
    ):
        n_b = feats.shape[0]
        Xt[write_pos : write_pos + n_b] = feats
        write_pos += n_b
        if write_pos % 1000 == 0 or write_pos == n_test:
            print(
                f"Extracted {write_pos}/{n_test} test samples (workers={max_workers}, batch={batch_size})"
            )

pred = model.predict_proba(Xt)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "target": pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("target min/mean/max:", float(pred.min()), float(pred.mean()), float(pred.max()))
