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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_ROOT_CANDIDATES = [
    "/kaggle/input",  # typical Kaggle
    "/kaggle/data",  # as provided in this environment listing
    "../input",  # fallback
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)

DATA_ROOT = _first_existing(DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(f"None of the data roots exist: {DATA_ROOT_CANDIDATES}")

COMP_ROOT = os.path.join(DATA_ROOT, "seti-breakthrough-listen")
if not os.path.exists(COMP_ROOT):
    COMP_ROOT = DATA_ROOT

train_labels_path = os.path.join(COMP_ROOT, "train_labels.csv")
sample_sub_path = os.path.join(COMP_ROOT, "sample_submission.csv")
train_dir = os.path.join(COMP_ROOT, "train")
test_dir = os.path.join(COMP_ROOT, "test")

for p in [train_labels_path, sample_sub_path, train_dir, test_dir]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_labels = pd.read_csv(train_labels_path)
sample_sub = pd.read_csv(sample_sub_path)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)



## === cell 1
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def id_to_path(base_dir: str, id_str: str) -> str:
    shard = id_str[0]
    return os.path.join(base_dir, shard, f"{id_str}.npy")


def verify_ids_exist_fast(base_dir: str, ids, sample_n: int = 64) -> None:
    if not ids:
        raise ValueError("Empty id list")
    n = len(ids)
    idxs = np.linspace(0, n - 1, num=min(sample_n, n), dtype=int)
    missing = []
    for i in idxs:
        id_str = ids[i]
        p = id_to_path(base_dir, id_str)
        if not os.path.exists(p):
            missing.append(id_str)
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError(
            f"Missing at least {len(missing)} sampled files under {base_dir}. "
            f"Examples: {missing[:10]}"
        )


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    arr: (6, 273, 256)
    Panels: 0(A1),1(B),2(A2),3(C),4(A3),5(D)
    Features are simple summary statistics focusing on A vs off-target differences.
    """
    x = arr.astype(np.float32, copy=False)

    A_mean = (x[0] + x[2] + x[4]) * (1.0 / 3.0)
    O_mean = (x[1] + x[3] + x[5]) * (1.0 / 3.0)

    diff = A_mean - O_mean
    absdiff = np.abs(diff)

    feats = [
        float(A_mean.mean()),
        float(A_mean.std()),
        float(O_mean.mean()),
        float(O_mean.std()),
        float(diff.mean()),
        float(diff.std()),
        float(absdiff.mean()),
        float(absdiff.std()),
    ]

    diff_t = diff.mean(axis=1)  # (273,)
    diff_f = diff.mean(axis=0)  # (256,)
    absdiff_t = absdiff.mean(axis=1)
    absdiff_f = absdiff.mean(axis=0)

    feats += [
        float(diff_t.mean()),
        float(diff_t.std()),
        float(diff_t.max()),
        float(diff_t.min()),
        float(diff_f.mean()),
        float(diff_f.std()),
        float(diff_f.max()),
        float(diff_f.min()),
        float(absdiff_t.mean()),
        float(absdiff_t.std()),
        float(absdiff_t.max()),
        float(absdiff_f.mean()),
        float(absdiff_f.std()),
        float(absdiff_f.max()),
    ]

    eps = 1e-6
    A_abs_mean = float(np.abs(A_mean).mean())
    O_abs_mean = float(np.abs(O_mean).mean())
    feats += [
        A_abs_mean,
        O_abs_mean,
        float(A_abs_mean / (O_abs_mean + eps)),
    ]

    return np.asarray(feats, dtype=np.float32)


_one_id = train_labels["id"].iloc[0]
_one_path = id_to_path(train_dir, _one_id)
if not os.path.exists(_one_path):
    raise FileNotFoundError(
        f"Could not locate training npy for id={_one_id} at {_one_path}"
    )

_one_arr = np.load(_one_path, mmap_mode="r")
if _one_arr.shape != (6, 273, 256):
    raise ValueError(f"Unexpected array shape: {_one_arr.shape}, expected (6,273,256)")
_feat_len = extract_features(_one_arr).shape[0]
_feat_len



## === cell 2
from sklearn.preprocessing import PolynomialFeatures


def _load_and_extract_path(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return extract_features(arr)


def _featurize_chunk(args: Tuple[int, List[str]]) -> Tuple[int, np.ndarray]:
    start, paths = args
    out = np.empty((len(paths), _feat_len), dtype=np.float32)
    for i, p in enumerate(paths):
        out[i] = _load_and_extract_path(p)
    return start, out


def _build_paths(base_dir: str, ids: List[str]) -> List[str]:
    return [id_to_path(base_dir, s) for s in ids]


def _save_memmap(path: str, arr: np.ndarray) -> None:
    mm = np.memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    del mm


def _load_memmap(path: str, shape, dtype=np.float32):
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


train_ids = train_labels["id"].tolist()
y = train_labels["target"].astype(int).to_numpy()

verify_ids_exist_fast(train_dir, train_ids)

_cpu = os.cpu_count() or 4
max_workers = min(16, _cpu)
sk_n_jobs = max(1, min(_cpu, 8))

feat_cache_dir = "/kaggle/working/feat_cache_v2"
os.makedirs(feat_cache_dir, exist_ok=True)

train_cache_path = os.path.join(feat_cache_dir, "X_train_feats_float32.mmp")
train_cache_meta = os.path.join(feat_cache_dir, "X_train_ids.txt")

X = None
if os.path.exists(train_cache_path) and os.path.exists(train_cache_meta):
    with open(train_cache_meta, "r") as f:
        cached_ids = [line.strip() for line in f if line.strip()]
    if cached_ids == train_ids:
        X = _load_memmap(
            train_cache_path, shape=(len(train_ids), _feat_len), dtype=np.float32
        )

if X is None:
    train_paths = _build_paths(train_dir, train_ids)

    chunk_size = 2048
    tasks = [
        (i, train_paths[i : i + chunk_size])
        for i in range(0, len(train_paths), chunk_size)
    ]

    X_build = np.empty((len(train_ids), _feat_len), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for start, block in ex.map(_featurize_chunk, tasks, chunksize=1):
            X_build[start : start + block.shape[0]] = block

    _save_memmap(train_cache_path, X_build)
    with open(train_cache_meta, "w") as f:
        f.write("\n".join(train_ids))
    X = _load_memmap(
        train_cache_path, shape=(len(train_ids), _feat_len), dtype=np.float32
    )

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)

C_grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
l1_ratio_grid = [0.0]

best_params = None
best_auc = -np.inf

fold_splits = list(cv.split(np.zeros(len(y)), y))

X_base = np.asarray(X, dtype=np.float32, order="C")
poly_fit = PolynomialFeatures(degree=2, include_bias=False)
poly_fit.fit(X_base)
n_poly = poly_fit.transform(X_base[:1]).shape[1]

Xp_cache_path = os.path.join(feat_cache_dir, "X_train_poly_deg2_float32.mmp")
Xp_cache_meta = os.path.join(feat_cache_dir, "X_train_poly_ids.txt")
Xp = None
if os.path.exists(Xp_cache_path) and os.path.exists(Xp_cache_meta):
    with open(Xp_cache_meta, "r") as f:
        cached_ids = [line.strip() for line in f if line.strip()]
    if cached_ids == train_ids:
        Xp = _load_memmap(
            Xp_cache_path, shape=(len(train_ids), n_poly), dtype=np.float32
        )

if Xp is None:
    Xp_build = poly_fit.transform(X_base).astype(np.float32, copy=False)
    _save_memmap(Xp_cache_path, Xp_build)
    with open(Xp_cache_meta, "w") as f:
        f.write("\n".join(train_ids))
    Xp = _load_memmap(Xp_cache_path, shape=(len(train_ids), n_poly), dtype=np.float32)

fold_buffers = []
for tr_idx, va_idx in fold_splits:
    fold_buffers.append(
        (
            np.empty((len(tr_idx), n_poly), dtype=np.float32),
            np.empty((len(va_idx), n_poly), dtype=np.float32),
        )
    )


def _scale_inplace(
    mat: np.ndarray, mean_: np.ndarray, inv_scale_: np.ndarray
) -> np.ndarray:
    mat -= mean_
    mat *= inv_scale_
    return mat


for C in C_grid:
    for l1_ratio in l1_ratio_grid:
        oof = np.zeros(len(y), dtype=np.float64)

        for (tr_idx, va_idx), (buf_tr, buf_va) in zip(fold_splits, fold_buffers):
            buf_tr[:] = Xp[tr_idx]
            buf_va[:] = Xp[va_idx]

            mean_ = buf_tr.mean(axis=0, dtype=np.float64).astype(np.float32)
            var_ = buf_tr.var(axis=0, dtype=np.float64).astype(np.float32)
            scale_ = np.sqrt(var_, dtype=np.float32)
            scale_[scale_ == 0.0] = 1.0
            inv_scale_ = (1.0 / scale_).astype(np.float32, copy=False)

            _scale_inplace(buf_tr, mean_, inv_scale_)
            _scale_inplace(buf_va, mean_, inv_scale_)

            lr = LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                max_iter=500,
                C=C,
                class_weight=None,
                n_jobs=sk_n_jobs,
                random_state=42,
            )
            lr.fit(buf_tr, y[tr_idx])
            oof[va_idx] = lr.predict_proba(buf_va)[:, 1]

        auc = roc_auc_score(y, oof)
        if auc > best_auc:
            best_auc = auc
            best_params = (C, l1_ratio)

best_C, best_l1_ratio = best_params

model = Pipeline(
    steps=[
        ("poly", poly),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                l1_ratio=best_l1_ratio,
                max_iter=500,
                C=best_C,
                class_weight=None,
                n_jobs=sk_n_jobs,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_base, y)



## === cell 3
test_ids = sample_sub["id"].tolist()

verify_ids_exist_fast(test_dir, test_ids)

test_cache_path = os.path.join(feat_cache_dir, "X_test_feats_float32.mmp")
test_cache_meta = os.path.join(feat_cache_dir, "X_test_ids.txt")

X_test = None
if os.path.exists(test_cache_path) and os.path.exists(test_cache_meta):
    with open(test_cache_meta, "r") as f:
        cached_ids = [line.strip() for line in f if line.strip()]
    if cached_ids == test_ids:
        X_test = _load_memmap(
            test_cache_path, shape=(len(test_ids), _feat_len), dtype=np.float32
        )

if X_test is None:
    test_paths = _build_paths(test_dir, test_ids)

    chunk_size = 2048
    tasks = [
        (i, test_paths[i : i + chunk_size])
        for i in range(0, len(test_paths), chunk_size)
    ]

    X_test_build = np.empty((len(test_ids), _feat_len), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for start, block in ex.map(_featurize_chunk, tasks, chunksize=1):
            X_test_build[start : start + block.shape[0]] = block

    _save_memmap(test_cache_path, X_test_build)
    with open(test_cache_meta, "w") as f:
        f.write("\n".join(test_ids))
    X_test = _load_memmap(
        test_cache_path, shape=(len(test_ids), _feat_len), dtype=np.float32
    )

X_test_base = np.asarray(X_test, dtype=np.float32, order="C")
pred = model.predict_proba(X_test_base)[:, 1].astype(np.float64)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": pred})
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["id", "target"]
submission.head()
