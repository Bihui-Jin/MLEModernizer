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



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_ROOT = "/kaggle/input"  # Kaggle standard
COMP_DIR = os.path.join(DATA_ROOT, "seti-breakthrough-listen")

OLD_DIR = os.path.join(COMP_DIR, "old_leaky_data")
OLD_TRAIN_LABELS_PATH = os.path.join(OLD_DIR, "train_labels_old.csv")
OLD_TEST_LABELS_PATH = os.path.join(OLD_DIR, "test_labels_old.csv")
OLD_TRAIN_DIR = os.path.join(OLD_DIR, "train_old")
OLD_TEST_DIR = os.path.join(OLD_DIR, "test_old")

TEST_DIR = os.path.join(COMP_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")

assert os.path.exists(
    OLD_TRAIN_LABELS_PATH
), f"Missing old train labels at {OLD_TRAIN_LABELS_PATH}"
assert os.path.exists(OLD_TRAIN_DIR), f"Missing old train dir at {OLD_TRAIN_DIR}"
assert os.path.exists(
    OLD_TEST_LABELS_PATH
), f"Missing old test labels at {OLD_TEST_LABELS_PATH}"
assert os.path.exists(OLD_TEST_DIR), f"Missing old test dir at {OLD_TEST_DIR}"
assert os.path.exists(TEST_DIR), f"Missing test dir at {TEST_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"

train_labels_old = pd.read_csv(OLD_TRAIN_LABELS_PATH)
test_labels_old = pd.read_csv(OLD_TEST_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(
    "old train labels:",
    train_labels_old.shape,
    "old test labels:",
    test_labels_old.shape,
)
print("sample_submission:", sample_sub.shape)



## === cell 2
from typing import Dict, List, Tuple
from multiprocessing import cpu_count
import hashlib
import multiprocessing as mp

CACHE_DIR = "/kaggle/working/_seti_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _cache_path_for_dir(root_dir: str, kind: str, ext: str) -> str:
    return os.path.join(CACHE_DIR, f"{kind}{_safe_name(root_dir)}{ext}")


def _dataset_cache_key(cache_prefix: str, root_dir: str) -> str:
    return f"{cache_prefix}{hashlib.blake2b(root_dir.encode('utf-8'), digest_size=16).hexdigest()}"


def list_npy_files_fast(root_dir: str) -> List[str]:
    out: List[str] = []
    stack = [root_dir]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False) and e.name.endswith(".npy"):
                        out.append(e.path)
        except FileNotFoundError:
            continue
    out.sort()
    return out


def list_npy_files_cached(root_dir: str) -> List[str]:
    cp = _cache_path_for_dir(root_dir, kind="files_", ext=".npy")
    if os.path.exists(cp):
        return np.load(cp, allow_pickle=False).astype(str).tolist()
    files = list_npy_files_fast(root_dir)
    np.save(cp, np.array(files, dtype=np.str_))
    return files


def build_id_to_path(root_dir: str) -> Dict[str, str]:
    cp_ids = _cache_path_for_dir(root_dir, kind="id2path_ids_", ext=".npy")
    cp_paths = _cache_path_for_dir(root_dir, kind="id2path_paths_", ext=".npy")
    if os.path.exists(cp_ids) and os.path.exists(cp_paths):
        ids = np.load(cp_ids, allow_pickle=False).astype(str)
        paths = np.load(cp_paths, allow_pickle=False).astype(str)
        return dict(zip(ids.tolist(), paths.tolist()))
    files = list_npy_files_cached(root_dir)
    ids = [id_from_path(p) for p in files]
    np.save(cp_ids, np.array(ids, dtype=np.str_))
    np.save(cp_paths, np.array(files, dtype=np.str_))
    return dict(zip(ids, files))


_F16_ORDER = None  # uint16[65536] sorted indices
_F16_INV_ORDER = (
    None  # int32[65536] inverse mapping bits->rank (not strictly needed, but handy)
)
_F16_BITS = None  # uint16[65536] = [0..65535]
_F16_KEY = None  # uint16[65536] key used for sorting


def _init_float16_order_tables():
    global _F16_ORDER, _F16_INV_ORDER, _F16_BITS, _F16_KEY
    if _F16_ORDER is not None:
        return

    bits = np.arange(65536, dtype=np.uint16)
    sign = (bits & 0x8000) != 0
    key = np.where(sign, np.uint16(~bits), np.uint16(bits ^ 0x8000)).astype(np.uint16)

    order = np.argsort(key, kind="mergesort").astype(np.uint16, copy=False)

    inv = np.empty(65536, dtype=np.int32)
    inv[order] = np.arange(65536, dtype=np.int32)

    _F16_BITS = bits
    _F16_KEY = key
    _F16_ORDER = order
    _F16_INV_ORDER = inv


def _quantile_higher_0p99_float16_exact(a: np.ndarray) -> np.float32:
    _init_float16_order_tables()

    a16 = np.asarray(a, dtype=np.float16)
    n = a16.size
    if n == 0:
        return np.float32(np.nan)
    k = int(np.ceil(0.99 * n) - 1)
    if k < 0:
        k = 0
    if k >= n:
        k = n - 1

    u = a16.view(np.uint16).ravel()
    counts = np.bincount(u, minlength=65536)

    csum = np.cumsum(counts[_F16_ORDER], dtype=np.int64)
    idx_in_order = int(np.searchsorted(csum, k + 1, side="left"))
    val_bits = _F16_ORDER[idx_in_order]
    return val_bits.view(np.float16).astype(np.float32)


def extract_features_from_snippet(x: np.ndarray) -> np.ndarray:
    """
    x shape: (6, 273, 256), float16/float32.
    Cadence positions: 0:A, 1:B, 2:A, 3:C, 4:A, 5:D.
    """
    x16 = np.asarray(x, dtype=np.float16)

    A16 = x16[[0, 2, 4]]
    OFF16 = x16[[1, 3, 5]]

    A = A16.astype(np.float32, copy=False)
    OFF = OFF16.astype(np.float32, copy=False)

    A_mean = A.mean(dtype=np.float32)
    OFF_mean = OFF.mean(dtype=np.float32)
    A_std = A.std(dtype=np.float32)
    OFF_std = OFF.std(dtype=np.float32)

    diff = A_mean - OFF_mean
    ratio = A_mean / (OFF_mean + 1e-6)

    A_abs_mean = np.abs(A).mean(dtype=np.float32)
    OFF_abs_mean = np.abs(OFF).mean(dtype=np.float32)
    abs_diff = A_abs_mean - OFF_abs_mean

    A_max = A.max()
    OFF_max = OFF.max()
    max_diff = A_max - OFF_max

    A_p99 = _quantile_higher_0p99_float16_exact(A16)
    OFF_p99 = _quantile_higher_0p99_float16_exact(OFF16)
    p99_diff = A_p99 - OFF_p99

    A_time_std = A.mean(axis=2, dtype=np.float32).std(dtype=np.float32)
    OFF_time_std = OFF.mean(axis=2, dtype=np.float32).std(dtype=np.float32)
    time_std_diff = A_time_std - OFF_time_std

    A_freq_std = A.mean(axis=1, dtype=np.float32).std(dtype=np.float32)
    OFF_freq_std = OFF.mean(axis=1, dtype=np.float32).std(dtype=np.float32)
    freq_std_diff = A_freq_std - OFF_freq_std

    return np.array(
        [
            float(A_mean),
            float(OFF_mean),
            float(A_std),
            float(OFF_std),
            float(diff),
            float(ratio),
            float(A_abs_mean),
            float(OFF_abs_mean),
            float(abs_diff),
            float(A_max),
            float(OFF_max),
            float(max_diff),
            float(A_p99),
            float(OFF_p99),
            float(p99_diff),
            float(A_time_std),
            float(OFF_time_std),
            float(time_std_diff),
            float(A_freq_std),
            float(OFF_freq_std),
            float(freq_std_diff),
        ],
        dtype=np.float32,
    )


def _feat_from_path(p: str) -> np.ndarray:
    arr = np.load(p, mmap_mode="r")  # read-only; values identical
    return extract_features_from_snippet(arr)


def _pool_init():
    _init_float16_order_tables()


def _featurize_worker(idx_path: Tuple[int, str]) -> Tuple[int, np.ndarray]:
    i, p = idx_path
    return i, _feat_from_path(p)


def featurize_paths(
    paths: List[str],
    n_features: int = 21,
    cache_prefix: str = "",
    dataset_root_for_cache_key: str = "",
) -> np.ndarray:
    if not paths:
        return np.zeros((0, n_features), dtype=np.float32)

    if dataset_root_for_cache_key:
        key = _dataset_cache_key(cache_prefix, dataset_root_for_cache_key)
        cp = os.path.join(CACHE_DIR, f"X_{key}.npy")
    else:
        h = hashlib.blake2b(digest_size=16)
        for p in paths:
            h.update(p.encode("utf-8"))
            h.update(b"\0")
        cp = os.path.join(CACHE_DIR, f"X_{cache_prefix}{h.hexdigest()}.npy")

    if os.path.exists(cp):
        X = np.load(cp, allow_pickle=False, mmap_mode="r")
        if X.shape == (len(paths), n_features):
            return np.array(X, copy=False)

    X = np.empty((len(paths), n_features), dtype=np.float32)

    n_workers = min(8, max(1, cpu_count() - 1))

    if n_workers == 1 or len(paths) < 4096:
        _init_float16_order_tables()
        for i, p in enumerate(paths):
            X[i] = _feat_from_path(p)
        np.save(cp, X)
        return X

    chunksize = max(64, min(512, len(paths) // (n_workers * 6) if len(paths) else 128))

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_workers, maxtasksperchild=2000, initializer=_pool_init
    ) as pool:
        for i, feat in pool.imap_unordered(
            _featurize_worker, enumerate(paths), chunksize=chunksize
        ):
            X[i] = feat

    np.save(cp, X)
    return X




## === cell 3
old_train_id_to_path = build_id_to_path(OLD_TRAIN_DIR)

train_df = train_labels_old.copy()
train_df["path"] = train_df["id"].map(old_train_id_to_path)
train_df = train_df.dropna(subset=["path"]).reset_index(drop=True)

y = train_df["target"].values.astype(np.int32, copy=False)

X = featurize_paths(
    train_df["path"].tolist(),
    n_features=21,
    cache_prefix="oldtrain_",
    dataset_root_for_cache_key=OLD_TRAIN_DIR,
)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print("Old-train feature matrix:", X.shape, "positive rate:", float(y.mean()))



## === cell 4
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                n_jobs=None,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

va_proba = clf.predict_proba(X_va)[:, 1]
print("Holdout ROC-AUC (old train split):", roc_auc_score(y_va, va_proba))

old_test_id_to_path = build_id_to_path(OLD_TEST_DIR)
test_old_df = test_labels_old.copy()
test_old_df["path"] = test_old_df["id"].map(old_test_id_to_path)
test_old_df = test_old_df.dropna(subset=["path"]).reset_index(drop=True)

y_old_test = test_old_df["target"].values.astype(np.int32, copy=False)

X_old_test = featurize_paths(
    test_old_df["path"].tolist(),
    n_features=21,
    cache_prefix="oldtest_",
    dataset_root_for_cache_key=OLD_TEST_DIR,
)

old_test_proba = clf.predict_proba(X_old_test)[:, 1]
print(
    "ROC-AUC on old test_old (sanity check):", roc_auc_score(y_old_test, old_test_proba)
)



## === cell 5
test_files = list_npy_files_cached(TEST_DIR)
test_ids = [id_from_path(p) for p in test_files]

Xt = featurize_paths(
    test_files, n_features=21, cache_prefix="test_", dataset_root_for_cache_key=TEST_DIR
)

proba = clf.predict_proba(Xt)[:, 1].astype(np.float64, copy=False)

sub_pred = pd.DataFrame({"id": test_ids, "target": proba})

sub = sample_sub[["id"]].merge(sub_pred, on="id", how="left")
sub["target"] = sub["target"].fillna(0.5).astype(np.float64)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Missing predictions filled with 0.5:", int(sub["target"].isna().sum()))
print(
    "Pred stats:",
    float(sub["target"].min()),
    float(sub["target"].mean()),
    float(sub["target"].max()),
)
