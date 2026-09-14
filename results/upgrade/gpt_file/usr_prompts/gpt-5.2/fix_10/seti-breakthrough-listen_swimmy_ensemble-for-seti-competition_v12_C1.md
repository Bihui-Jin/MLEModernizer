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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

DATA_ROOT = "/kaggle/input"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS), f"Missing: {TRAIN_LABELS}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR)):
    alt_root = os.path.join(DATA_ROOT, "seti-breakthrough-listen")
    alt_train = os.path.join(alt_root, "train")
    alt_test = os.path.join(alt_root, "test")
    alt_labels = os.path.join(alt_root, "train_labels.csv")
    alt_sample = os.path.join(alt_root, "sample_submission.csv")
    if (
        os.path.isdir(alt_train)
        and os.path.isdir(alt_test)
        and os.path.exists(alt_labels)
        and os.path.exists(alt_sample)
    ):
        TRAIN_DIR, TEST_DIR, TRAIN_LABELS, SAMPLE_SUB = (
            alt_train,
            alt_test,
            alt_labels,
            alt_sample,
        )

assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS)
sample_submission = pd.read_csv(SAMPLE_SUB)

train_labels["id"] = train_labels["id"].astype(str)
sample_submission["id"] = sample_submission["id"].astype(str)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _iter_npy_files_fast(root_dir: str):
    """Iterate .npy files under root_dir, supporting both flat and nested (class-subdir) layouts."""
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".npy"):
                yield entry.path
            elif entry.is_dir():
                with os.scandir(entry.path) as it2:
                    for f in it2:
                        if f.is_file() and f.name.endswith(".npy"):
                            yield f.path


def _id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _quantiles_linear_multi_via_partition(flat: np.ndarray, qs) -> np.ndarray:
    n = flat.size
    if n == 0:
        raise ValueError("Empty array for quantile")
    qs = np.asarray(qs, dtype=np.float64)
    pos = (n - 1) * qs
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)

    idx = np.unique(np.concatenate([lo, hi]))
    part = np.partition(flat, idx)

    out = np.empty(qs.shape[0], dtype=np.float32)
    for i, (p, l, h) in enumerate(zip(pos, lo, hi)):
        if l == h:
            out[i] = np.float32(part[l])
        else:
            xlo = float(part[l])
            xhi = float(part[h])
            w = float(p - l)
            out[i] = np.float32(xlo + (xhi - xlo) * w)
    return out


def compute_features_from_npy(arr: np.ndarray) -> np.ndarray:
    """
    Core logic preserved: same deterministic statistical feature extraction from (6,273,256).
    Speedups come from:
      - computing all needed quantiles using ONE np.partition call per flat buffer (and one each for A/B),
      - avoiding extra temporaries/copies, and keeping reductions in float32 where safe.
    """
    x = arr.astype(np.float32, copy=False)

    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(f"Unexpected snippet shape {x.shape}, expected (6,273,256)")

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    A_mean = A.mean()
    B_mean = B.mean()
    A_std = A.std()
    B_std = B.std()
    A_max = A.max()
    B_max = B.max()

    contrast = np.array(
        [
            A_mean - B_mean,
            A_std - B_std,
            A_max - B_max,
            (A_max + 1e-6) / (B_max + 1e-6),
        ],
        dtype=np.float32,
    )

    x_flat = x.reshape(-1)

    q05, q25, q50, q75, q95 = _quantiles_linear_multi_via_partition(
        x_flat, [0.05, 0.25, 0.50, 0.75, 0.95]
    )
    x_max = np.float32(x_flat.max())

    overall = np.array([x_flat.mean(), x_flat.std(), x_max, q50], dtype=np.float32)
    spread = np.array([q95 - q05, q75 - q25], dtype=np.float32)

    A_panel_mean = panel_mean[[0, 2, 4]]
    B_panel_mean = panel_mean[[1, 3, 5]]
    A_panel_max = panel_max[[0, 2, 4]]
    B_panel_max = panel_max[[1, 3, 5]]
    delta_panel_mean = (A_panel_mean - B_panel_mean).astype(np.float32)
    delta_panel_max = (A_panel_max - B_panel_max).astype(np.float32)

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    Aq95, Aq50 = _quantiles_linear_multi_via_partition(A_flat, [0.95, 0.50])
    Bq95, Bq50 = _quantiles_linear_multi_via_partition(B_flat, [0.95, 0.50])

    robust_contrast = np.array(
        [
            Aq95 - Bq95,
            (Aq95 + 1e-6) / (Bq95 + 1e-6),
            Aq50 - Bq50,
        ],
        dtype=np.float32,
    )

    pair_mean_diff = (A.mean(axis=(1, 2)) - B.mean(axis=(1, 2))).astype(np.float32)
    pair_max_diff = (A.max(axis=(1, 2)) - B.max(axis=(1, 2))).astype(np.float32)
    pair_std_diff = (A.std(axis=(1, 2)) - B.std(axis=(1, 2))).astype(np.float32)

    pair_stats = np.array(
        [
            pair_mean_diff.mean(),
            pair_mean_diff.std(),
            pair_max_diff.mean(),
            pair_max_diff.std(),
            pair_std_diff.mean(),
            pair_std_diff.std(),
        ],
        dtype=np.float32,
    )

    diff = (A - B).astype(np.float32, copy=False)
    abs_diff = np.abs(diff)
    diff_energy = np.array(
        [
            abs_diff.mean(),
            (diff * diff).mean(),
            abs_diff.max(),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [
            panel_mean,  # 6
            panel_std,  # 6
            panel_max,  # 6
            contrast,  # 4
            overall,  # 4
            np.array([q05, q25, q75, q95], dtype=np.float32),  # 4
            spread,  # 2
            delta_panel_mean,  # 3
            delta_panel_max,  # 3
            robust_contrast,  # 3
            pair_mean_diff,  # 3
            pair_max_diff,  # 3
            pair_std_diff,  # 3
            pair_stats,  # 6
            diff_energy,  # 3
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    return feats


def build_id_to_path_map_for_ids(root_dir: str, ids) -> dict:
    ids = list(ids)
    m = {}
    missing = []
    for sid in ids:
        sub = sid[:2]
        p = os.path.join(root_dir, sub, f"{sid}.npy")
        if os.path.exists(p):
            m[sid] = p
        else:
            missing.append(sid)

    if not missing:
        return m

    need = set(missing)
    for p in _iter_npy_files_fast(root_dir):
        sid = _id_from_path(p)
        if sid in need:
            m[sid] = p
            if len(m) == len(ids):
                break

    if len(m) != len(ids):
        missing2 = sorted(set(ids).difference(m.keys()))
        raise FileNotFoundError(
            f"Missing {len(missing2)} .npy files under {root_dir}, e.g. {missing2[:5]}"
        )
    return m


_FEATURE_CACHE_DIR = os.path.join("/kaggle/working", "feature_cache_v1")
os.makedirs(_FEATURE_CACHE_DIR, exist_ok=True)


def _feat_cache_path_for_id(sid: str) -> str:
    return os.path.join(_FEATURE_CACHE_DIR, f"{sid}.npy")


def build_features_for_ids_parallel(
    ids, id_to_path: dict, max_workers: int = None
) -> np.ndarray:
    ids = list(ids)
    if len(ids) == 0:
        return np.empty((0, 0), dtype=np.float32)

    p0 = _feat_cache_path_for_id(ids[0])
    if os.path.exists(p0):
        first_feats = np.load(p0)
    else:
        first_arr = np.load(id_to_path[ids[0]], mmap_mode="r")
        first_feats = compute_features_from_npy(first_arr)
        np.save(p0, first_feats)
    n_feats = int(first_feats.shape[0])

    X = np.empty((len(ids), n_feats), dtype=np.float32)
    X[0] = first_feats

    def _one(i: int):
        sid = ids[i]
        cp = _feat_cache_path_for_id(sid)
        if os.path.exists(cp):
            return i, np.load(cp)
        arr = np.load(id_to_path[sid], mmap_mode="r")
        feats = compute_features_from_npy(arr)
        np.save(cp, feats)
        return i, feats

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_one, range(1, len(ids)), chunksize=512):
            X[i] = feats
    return X




## === cell 2
train_ids = train_labels["id"].tolist()
test_ids = sample_submission["id"].tolist()

train_id2path = build_id_to_path_map_for_ids(TRAIN_DIR, train_ids)
test_id2path = build_id_to_path_map_for_ids(TEST_DIR, test_ids)

X_train = build_features_for_ids_parallel(train_ids, train_id2path)
y_train = train_labels["target"].values.astype(int)

X_test = build_features_for_ids_parallel(test_ids, test_id2path)

print("X_train:", X_train.shape, "X_test:", X_test.shape)




## === cell 3
from concurrent.futures import ThreadPoolExecutor


def oof_and_test_predictions_manyC(
    X, y, X_test, Cs, n_splits=5, random_state=RANDOM_STATE
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_preds = [np.zeros(X_test.shape[0], dtype=np.float64) for _ in Cs]

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(X_test)

        def _fit_one(jC):
            j, C = jC
            lr = LogisticRegression(
                C=C,
                max_iter=500,
                solver="lbfgs",
                n_jobs=None,
                random_state=RANDOM_STATE,
            )
            lr.fit(Xtr_s, ytr)
            return j, lr.predict_proba(Xte_s)[:, 1]

        max_workers = min(len(Cs), (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for j, te_pred in ex.map(_fit_one, list(enumerate(Cs))):
                test_preds[j] += te_pred / n_splits

    return test_preds


Cs = [0.3, 0.6, 1.0, 1.5, 2.5, 4.0]
test_preds = oof_and_test_predictions_manyC(X_train, y_train, X_test, Cs, n_splits=5)

data1 = pd.DataFrame({"id": test_ids, "target": test_preds[0]})
data2 = pd.DataFrame({"id": test_ids, "target": test_preds[1]})
data3 = pd.DataFrame({"id": test_ids, "target": test_preds[2]})
data4 = pd.DataFrame({"id": test_ids, "target": test_preds[3]})
data5 = pd.DataFrame({"id": test_ids, "target": test_preds[4]})
data6 = pd.DataFrame({"id": test_ids, "target": test_preds[5]})




## === cell 4
data1.head()




## === cell 5
data2.head()




## === cell 6
avg_pred = np.mean(np.vstack(test_preds), axis=0)
data_ens = pd.DataFrame({"id": test_ids, "target": np.clip(avg_pred, 0.0, 1.0)})

data6 = data_ens




## === cell 7
submission = data6[["id", "target"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert (
    submission.shape[0] == sample_submission.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(submission.columns) == [
    "id",
    "target",
], "Submission columns must be ['id','target']"
assert os.path.exists("submission.csv"), "submission.csv was not created"
