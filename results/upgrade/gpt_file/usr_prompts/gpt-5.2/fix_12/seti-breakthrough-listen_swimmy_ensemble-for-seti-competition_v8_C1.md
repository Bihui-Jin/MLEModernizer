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

0.7571724744809541

# 6. Current score

0.50485

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the broken dependency on external Kaggle notebook outputs (the `../input/.../submission.csv` files don’t exist in your environment), and replace it with a self-contained inference pipeline that reads the provided `train_labels.csv` and the `.npy` cadence snippets under `/kaggle/input/train/` and `/kaggle/input/test/`. To keep the “core logic” spirit of your current approach (a simple score-level ensembling), I implement a lightweight ensemble of multiple simple, complementary feature-based models (logistic regressions) and average their probabilities. I also add strict ID alignment to the sample submission to guarantee a valid `id,target` CSV with exactly 6000 rows. This should run end-to-end under the listed packages and produce `submission.csv`.'
- What this solution (achieved 0.50596) has done: 'The main timeout drivers are repeated filesystem probing in `find_npy_by_id` (worst-case 16 `exists()` calls per sample) and slow per-file `np.load` I/O done serially for ~60k arrays. I make path resolution O(1) by constructing file paths directly from the known folder layout, and I batch feature extraction using a multiprocessing pool so file reads + numpy reductions run in parallel. I also remove redundant pipeline construction inside the CV loop (same logic, less overhead) and keep all numeric operations and model settings identical so results remain the same up to negligible FP noise.'
- What this solution (achieved 0.51332) has done: 'Your current score (0.50596) is far below the target (0.75717), so we should make a small, legitimate change that increases AUC without changing the overall approach (feature extraction → logistic regression ensemble → average probabilities). The biggest likely issue is that the features are extremely weak for this competition; we can strengthen them minimally by adding a few simple “structure” features that capture the diagonal/drift patterns (via lightweight gradients and A-vs-B residual energy) while keeping the same model family and training loop. These added features are cheap to compute (just means/stds of finite differences) and keep the evaluation semantics identical (still probabilistic AUC). Everything else (paths, CV, ensemble averaging, submission alignment) is preserved.'
- What this solution (achieved 0.51176) has done: 'Your current AUC (0.513) is far below the target (0.757), so we need a small but meaningful lift while keeping the same overall pipeline (handcrafted features → StandardScaler+LogisticRegression ensemble → averaged probabilities). The biggest low-risk gain here is to make the features a bit more “signal-shape aware” without changing the model family: add a few cheap features capturing diagonal/drift structure (simple time-frequency diagonal gradients) and robust “A vs B” contrast via per-pixel positive residual energy. I also add `class_weight="balanced"` in LogisticRegression, which often improves ROC AUC in imbalanced problems without changing the core training loop. Everything else (paths, CV, ensembling, submission alignment, no leaky data) is kept the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.50772) has done: 'The timeout is dominated by feature extraction doing heavy per-file numpy work plus repeated disk I/O across 60k files, compounded by Python-loop overhead and repeated quantile computations. I keep the exact same features and model training logic, but make feature extraction faster by (1) loading with `mmap_mode='r'` to reduce memory pressure, (2) eliminating repeated full-array passes where we can compute equivalent statistics from already-derived arrays, (3) replacing `np.quantile` calls (which sort) with partition-based quantiles (exact order statistic, matching the quantile definition used), and (4) removing the ineffective `lru_cache` (it doesn’t help because paths are unique) to avoid overhead. I also switch the threadpool pipeline to use `imap` (ordered) and write directly into preallocated matrices to cut tuple shuffling/assignment overhead while preserving determinism and identical semantics.'
- What this solution (achieved 0.50485) has done: 'The timeout is overwhelmingly dominated by feature extraction: you repeatedly load ~60k `.npy` files and compute many statistics, and the current implementation does extra passes over the same arrays plus expensive `np.partition` calls that re-partition multiple times per sample. I keep the exact same features and model logic, but speed up by (1) using `os.scandir`-based path indexing, (2) precomputing reusable means/stds/max/min and derived arrays once per sample, (3) avoiding repeated flatten/astype/ravel work, (4) computing multiple quantiles/top-k from a single partition where it’s provably equivalent, and (5) tuning the threadpool chunking to reduce Python overhead while staying deterministic. These changes preserve evaluation semantics and only introduce negligible floating-point differences from reordering equivalent operations.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
INPUT_ROOT = Path("/kaggle/input")
DATASET_ROOT = INPUT_ROOT / "seti-breakthrough-listen"

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
TRAIN_LABELS_PATH = DATASET_ROOT / "train_labels.csv"
SAMPLE_SUB_PATH = DATASET_ROOT / "sample_submission.csv"

assert TRAIN_LABELS_PATH.exists(), f"Missing: {TRAIN_LABELS_PATH}"
assert SAMPLE_SUB_PATH.exists(), f"Missing: {SAMPLE_SUB_PATH}"
assert TRAIN_DIR.exists(), f"Missing: {TRAIN_DIR}"
assert TEST_DIR.exists(), f"Missing: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert list(sample_sub.columns) == [
    "id",
    "target",
], "sample_submission.csv must have columns: id,target"
assert train_labels.columns.tolist() == [
    "id",
    "target",
], "train_labels.csv must have columns: id,target"

train_ids = train_labels["id"].astype(str).str.lower().values
y = train_labels["target"].astype(np.int8).values
test_ids = sample_sub["id"].astype(str).str.lower().values

print("Train rows:", len(train_labels), " Test rows:", len(sample_sub))
print("Train positive rate:", y.mean())




## === cell 2
def _build_id_to_path(root_dir: Path) -> dict:
    id2p = {}
    root = str(root_dir)
    for sub in os.scandir(root):
        if not sub.is_dir():
            continue
        for f in os.scandir(sub.path):
            if f.is_file() and f.name.endswith(".npy"):
                stem = f.name[:-4].lower()
                id2p[stem] = Path(f.path)
    return id2p


train_id2path = _build_id_to_path(TRAIN_DIR)
test_id2path = _build_id_to_path(TEST_DIR)

missing_train = [i for i in train_ids if i not in train_id2path]
missing_test = [i for i in test_ids if i not in test_id2path]
assert (
    not missing_train
), f"Missing {len(missing_train)} train npy files (showing first 5): {missing_train[:5]}"
assert (
    not missing_test
), f"Missing {len(missing_test)} test npy files (showing first 5): {missing_test[:5]}"

train_paths = [str(train_id2path[_id]) for _id in train_ids]
test_paths = [str(test_id2path[_id]) for _id in test_ids]

A_IDX = np.array([0, 2, 4], dtype=np.int64)
B_IDX = np.array([1, 3, 5], dtype=np.int64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> np.float32:
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.mean(a * a) * np.mean(b * b)) + 1e-8)
    return np.float32(float(np.mean(a * b) / denom))


def _topk_mean_flat32(flat32: np.ndarray, k: int) -> np.float32:
    n = flat32.size
    k = int(max(1, min(int(k), n)))
    tmp = flat32.copy()
    topk = np.partition(tmp, n - k)[-k:]
    return np.float32(float(topk.mean()))


def _quantile_partition_linear_flat32(flat32: np.ndarray, q: float) -> np.float32:
    n = flat32.size
    if n == 0:
        return np.float32(np.nan)
    h = (n - 1) * float(q)
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        tmp = flat32.copy()
        return np.float32(np.partition(tmp, lo)[lo])
    tmp1 = flat32.copy()
    vlo = float(np.partition(tmp1, lo)[lo])
    tmp2 = flat32.copy()
    vhi = float(np.partition(tmp2, hi)[hi])
    return np.float32(vlo + (h - lo) * (vhi - vlo))


def extract_features(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)  # shape (6,273,256)

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))
    panel_min = x.min(axis=(1, 2))

    A = x[A_IDX]  # (3,273,256)
    B = x[B_IDX]  # (3,273,256)

    A_mean = A.mean()
    B_mean = B.mean()
    A_std = A.std()
    B_std = B.std()
    A_max = A.max()
    B_max = B.max()

    mean_diff = A_mean - B_mean
    std_diff = A_std - B_std
    max_diff = A_max - B_max

    A_spec = A.mean(axis=1)  # (3,256)
    B_spec = B.mean(axis=1)
    A_time = A.mean(axis=2)  # (3,273)
    B_time = B.mean(axis=2)

    spec_diff = A_spec - B_spec
    time_diff = A_time - B_time

    base_feat = np.concatenate(
        [
            panel_mean,
            panel_std,
            panel_max,
            panel_min,
            np.array(
                [
                    A_mean,
                    B_mean,
                    A_std,
                    B_std,
                    A_max,
                    B_max,
                    mean_diff,
                    std_diff,
                    max_diff,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    A_spec.mean(),
                    A_spec.std(),
                    B_spec.mean(),
                    B_spec.std(),
                    spec_diff.mean(),
                    spec_diff.std(),
                    A_time.mean(),
                    A_time.std(),
                    B_time.mean(),
                    B_time.std(),
                    time_diff.mean(),
                    time_diff.std(),
                ],
                dtype=np.float32,
            ),
        ]
    )

    A_flat = A.ravel()
    B_flat = B.ravel()

    nA = A_flat.size
    nB = B_flat.size
    kA = int(np.floor(0.99 * (nA - 1)))
    kB = int(np.floor(0.99 * (nB - 1)))
    tmpA = A_flat.copy()
    tmpB = B_flat.copy()
    Ath = np.float32(np.partition(tmpA, kA)[kA])
    Bth = np.float32(np.partition(tmpB, kB)[kB])

    quant_feat = np.array(
        [(A_flat > Ath).mean(), (B_flat > Bth).mean(), Ath, Bth, Ath - Bth],
        dtype=np.float32,
    )

    d_t_A = np.diff(A, axis=1)  # (3,272,256)
    d_f_A = np.diff(A, axis=2)  # (3,273,255)
    d_t_B = np.diff(B, axis=1)
    d_f_B = np.diff(B, axis=2)

    d_t_A_mean = d_t_A.mean()
    d_t_A_std = d_t_A.std()
    d_f_A_mean = d_f_A.mean()
    d_f_A_std = d_f_A.std()

    d_t_B_mean = d_t_B.mean()
    d_t_B_std = d_t_B.std()
    d_f_B_mean = d_f_B.mean()
    d_f_B_std = d_f_B.std()

    grad_feat = np.array(
        [
            d_t_A_mean,
            d_t_A_std,
            d_f_A_mean,
            d_f_A_std,
            d_t_B_mean,
            d_t_B_std,
            d_f_B_mean,
            d_f_B_std,
            (d_t_A_std - d_t_B_std),
            (d_f_A_std - d_f_B_std),
        ],
        dtype=np.float32,
    )

    resid = A - B  # (3,273,256)
    resid_abs = np.abs(resid)
    resid_sq_mean = np.mean(resid * resid)

    resid_feat = np.array(
        [
            resid.mean(),
            resid.std(),
            np.mean(resid_abs),
            np.sqrt(resid_sq_mean),
        ],
        dtype=np.float32,
    )

    diag_p_A = A[:, 1:, 1:] - A[:, :-1, :-1]
    diag_p_B = B[:, 1:, 1:] - B[:, :-1, :-1]
    diag_m_A = A[:, 1:, :-1] - A[:, :-1, 1:]
    diag_m_B = B[:, 1:, :-1] - B[:, :-1, 1:]

    diag_feat = np.array(
        [
            diag_p_A.mean(),
            diag_p_A.std(),
            diag_m_A.mean(),
            diag_m_A.std(),
            diag_p_B.mean(),
            diag_p_B.std(),
            diag_m_B.mean(),
            diag_m_B.std(),
            (diag_p_A.std() - diag_p_B.std()),
            (diag_m_A.std() - diag_m_B.std()),
        ],
        dtype=np.float32,
    )

    resid_pos = np.maximum(resid, 0.0)
    resid_neg = np.maximum(-resid, 0.0)

    resid_pos_sq_mean = np.mean(resid_pos * resid_pos)
    resid_neg_sq_mean = np.mean(resid_neg * resid_neg)

    resid_rect_feat = np.array(
        [
            resid_pos.mean(),
            resid_pos.std(),
            np.sqrt(resid_pos_sq_mean),
            resid_neg.mean(),
            resid_neg.std(),
            np.sqrt(resid_neg_sq_mean),
        ],
        dtype=np.float32,
    )

    A_avg = A.mean(axis=0)  # (273,256)
    B_avg = B.mean(axis=0)
    resid_avg_pos = np.maximum(A_avg - B_avg, 0.0)

    resid_flat = resid_avg_pos.ravel().astype(np.float32, copy=False)

    contrast_feat = np.array(
        [
            _quantile_partition_linear_flat32(resid_flat, 0.90),
            _quantile_partition_linear_flat32(resid_flat, 0.95),
            _quantile_partition_linear_flat32(resid_flat, 0.99),
            _topk_mean_flat32(resid_flat, k=256),
            _topk_mean_flat32(resid_flat, k=1024),
            _safe_corr(A_avg, B_avg),
            _safe_corr(A_avg.mean(axis=1), B_avg.mean(axis=1)),  # time-series corr
            _safe_corr(A_avg.mean(axis=0), B_avg.mean(axis=0)),  # freq-series corr
        ],
        dtype=np.float32,
    )

    A_fmax = A.max(axis=2)  # (3,273) max over frequency per time
    B_fmax = B.max(axis=2)
    A_tmax = A.max(axis=1)  # (3,256) max over time per frequency
    B_tmax = B.max(axis=1)

    line_feat = np.array(
        [
            A_fmax.mean(),
            A_fmax.std(),
            A_fmax.max(),
            B_fmax.mean(),
            B_fmax.std(),
            B_fmax.max(),
            (A_fmax.mean() - B_fmax.mean()),
            (A_fmax.max() - B_fmax.max()),
            A_tmax.mean(),
            A_tmax.std(),
            A_tmax.max(),
            B_tmax.mean(),
            B_tmax.std(),
            B_tmax.max(),
            (A_tmax.mean() - B_tmax.mean()),
            (A_tmax.max() - B_tmax.max()),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate(
        [
            base_feat,
            quant_feat,
            grad_feat,
            resid_feat,
            diag_feat,
            resid_rect_feat,
            contrast_feat,
            line_feat,
        ]
    )
    return feat




## === cell 3
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


def _load_and_extract(path_str: str) -> np.ndarray:
    arr = np.load(path_str, mmap_mode="r")  # float16 (6,273,256)
    return extract_features(arr)


def _load_and_extract_indexed(args):
    i, p = args
    return i, _load_and_extract(p)


N_WORKERS = min(16, max(1, cpu_count()))
CHUNK = 1024

n_features = (6 * 4) + 9 + 12 + 5 + 10 + 4 + 10 + 6 + 8 + 16
X_train = np.zeros((len(train_ids), n_features), dtype=np.float32)

with ThreadPool(processes=N_WORKERS) as pool:
    for i, feat in pool.imap(
        _load_and_extract_indexed, enumerate(train_paths), chunksize=CHUNK
    ):
        X_train[i] = feat

print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)



## === cell 4
Cs = [0.5, 1.0, 2.0]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(X_train, y))

oof_preds = np.zeros(len(train_ids), dtype=np.float32)
models = []


def _standardize_train_valid(Xtr: np.ndarray, Xva: np.ndarray):
    mean_ = Xtr.mean(axis=0, dtype=np.float64)
    var_ = Xtr.var(axis=0, dtype=np.float64)
    scale_ = np.sqrt(var_, dtype=np.float64)
    scale_[scale_ == 0.0] = 1.0
    Xtr_s = (Xtr - mean_) / scale_
    Xva_s = (Xva - mean_) / scale_
    return (
        Xtr_s.astype(np.float32, copy=False),
        Xva_s.astype(np.float32, copy=False),
        mean_.astype(np.float32),
        scale_.astype(np.float32),
    )


for C in Cs:
    oof_c = np.zeros(len(train_ids), dtype=np.float32)

    for tr_idx, va_idx in splits:
        Xtr = X_train[tr_idx]
        Xva = X_train[va_idx]
        Xtr_s, Xva_s, _, _ = _standardize_train_valid(Xtr, Xva)

        lr = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=300,
            n_jobs=None,
            class_weight="balanced",
        )
        lr.fit(Xtr_s, y[tr_idx])
        oof_c[va_idx] = lr.predict_proba(Xva_s)[:, 1].astype(np.float32)

    oof_preds += oof_c / len(Cs)

    pipe_full = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=300,
            n_jobs=None,
            class_weight="balanced",
        ),
    )
    pipe_full.fit(X_train, y)
    models.append(pipe_full)

try:
    from sklearn.metrics import roc_auc_score

    print("OOF AUC (ensemble):", roc_auc_score(y, oof_preds))
except Exception as e:
    print("Could not compute OOF AUC (non-fatal):", repr(e))



## === cell 5
X_test = np.zeros((len(test_ids), X_train.shape[1]), dtype=np.float32)

with ThreadPool(processes=N_WORKERS) as pool:
    for i, feat in pool.imap(
        _load_and_extract_indexed, enumerate(test_paths), chunksize=CHUNK
    ):
        X_test[i] = feat

print("X_test shape:", X_test.shape)



## === cell 6
test_pred = np.zeros(len(test_ids), dtype=np.float32)
for m in models:
    test_pred += m.predict_proba(X_test)[:, 1].astype(np.float32) / len(models)

test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})

submission = (
    sample_sub[["id"]]
    .assign(id=sample_sub["id"].astype(str).str.lower())
    .merge(submission, on="id", how="left")
)
assert (
    submission["target"].notna().all()
), "Some test IDs did not get predictions (ID/path mismatch)."

submission["id"] = sample_sub["id"].astype(str).values
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
