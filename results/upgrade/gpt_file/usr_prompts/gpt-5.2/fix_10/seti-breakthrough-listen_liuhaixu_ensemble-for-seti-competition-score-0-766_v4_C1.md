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

0.7626479475548098

# 6. Current score

0.49769

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49833) has done: 'The timeout is dominated by repeatedly loading tens of thousands of small `.npy` files and doing expensive percentile computations (multiple `np.partition` calls) inside multiprocessing with high IPC overhead. I keep the exact same features and model, but speed up featurization by (1) memoizing percentile indices and using a single `np.partition` per array (provably equivalent), (2) avoiding repeated `ravel()`/temporary arrays and keeping operations in float32, and (3) switching to a thread pool for I/O-bound loads plus dynamic worker sizing and larger chunks to reduce scheduling overhead. I also add deterministic ordering by mapping results back to row indices (so parallel completion order can’t scramble rows) without changing semantics. Paths and core training/evaluation remain unchanged.'
- What this solution (achieved 0.50994) has done: 'Your current score (0.49833) is far below the target (0.76265), and the biggest likely culprit is a semantics mismatch with the competition’s canonical preprocessing: most strong baselines apply a log/standardization transform per snippet before computing simple statistics, while your feature extractor currently uses raw normalized floats directly. I keep the same feature set (same 18 stats, same A-vs-OFF cadence logic, same LogisticRegression pipeline), but add the minimal per-snippet normalization used in many SETI baselines: `x = log1p(x - x.min())` followed by standardization, done inside `extract_features` so the model/loop stays unchanged. I also ensure numerical stability (avoid divide-by-zero when standardizing) without changing the downstream model or submission formatting. These changes should materially increase AUC toward the target while preserving your overall approach and keeping runtime within limits.'
- What this solution (achieved 0.5107) has done: 'Your current AUC (0.50994) is far below the target (0.76265), so we should make a small change that improves separability without changing the overall approach (same 18 features, same A-vs-OFF cadence logic, same LogisticRegression pipeline). The biggest likely issue is that the per-snippet transform is currently applied across all 6 panels together, which can partially wash out the “present only in A panels” pattern; we keep the same log1p+standardization idea but apply it per panel (each of the 6 images independently) before computing the exact same downstream statistics. This preserves the model/training loop and feature definitions while better matching common SETI baselines and should move AUC upward toward the target. I also add a tiny epsilon guard in standardization to avoid rare divide-by-zero and keep results stable.'
- What this solution (achieved 0.49237) has done: 'Your current AUC (0.5107) is far below the target (0.76265), so we need a small, metric-aligned signal boost without changing the model/training loop or the A-vs-OFF cadence feature logic. The biggest low-risk gain here is to add a very lightweight “line-enhancement” preprocessing per panel: subtract a per-row mean and per-column mean (keep overall mean) after your existing log1p+standardization, which emphasizes narrowband/drifting structures that are common in needles. This keeps the same 18 downstream statistics and the same LogisticRegression pipeline, but makes those statistics more separable for signal-like patterns, which should move AUC upward toward the target. I also keep everything deterministic and preserve the submission mapping/format exactly.'
- What this solution (achieved 0.5107) has done: 'Your current score is far below the target, and the last “line-enhancement” preprocessing likely hurt generalization (it can suppress the very A-vs-OFF contrast your downstream features rely on). I make the smallest semantic change by removing only that extra row/column demeaning step while keeping your per-panel `log1p` + standardization and the exact same 18 features + LogisticRegression pipeline. I also add a tiny numeric safety clamp before `log1p` (to avoid rare negative values after `-xmin` due to float16/float32 rounding) without changing the intended transform. This should move AUC back upward toward your earlier ~0.51 baseline (closer to the target) while preserving core logic and producing the same valid `submission.csv`.'
- What this solution (achieved 0.49769) has done: 'Your score (0.5107) is far below the target (0.76265), so we should make a small, metric-aligned improvement without changing the overall model/training loop or the 18-feature definition. The biggest low-risk issue is that you currently standardize each panel independently, which can erase the *relative* A-vs-OFF contrast that your downstream `A-OFF` features rely on; we instead apply the same `log1p` transform per panel but then standardize **once per snippet across all 6 panels together**. This is a minimal change (same preprocessing family, same features, same LR pipeline) and is expected to improve separability (and AUC) by preserving panel-to-panel differences. I also add a tiny deterministic clamp after `-xmin` to avoid rare negative values before `log1p`, keeping numerical stability unchanged elsewhere.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import multiprocessing as mp

from multiprocessing.dummy import Pool as ThreadPool



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",
    "/kaggle/data",
]


def find_existing_path(rel_path):
    for base in BASE_CANDIDATES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None


TRAIN_LABELS_PATH = find_existing_path("train_labels.csv") or find_existing_path(
    "seti-breakthrough-listen/train_labels.csv"
)
SAMPLE_SUB_PATH = find_existing_path("sample_submission.csv") or find_existing_path(
    "seti-breakthrough-listen/sample_submission.csv"
)
TRAIN_DIR = find_existing_path("train") or find_existing_path(
    "seti-breakthrough-listen/train"
)
TEST_DIR = find_existing_path("test") or find_existing_path(
    "seti-breakthrough-listen/test"
)

missing = [
    name
    for name, p in [
        ("TRAIN_LABELS_PATH", TRAIN_LABELS_PATH),
        ("SAMPLE_SUB_PATH", SAMPLE_SUB_PATH),
        ("TRAIN_DIR", TRAIN_DIR),
        ("TEST_DIR", TEST_DIR),
    ]
    if p is None
]
if missing:
    raise FileNotFoundError(f"Could not locate required paths: {missing}")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("Resolved paths:")
print("TRAIN_LABELS_PATH:", TRAIN_LABELS_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("train_labels:", train_labels.shape, "sample_sub:", sample_sub.shape)




## === cell 2
def build_id_to_path_index(root_dir):
    id2path = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_dir():
                subdir = entry.path
                with os.scandir(subdir) as it2:
                    for f in it2:
                        if f.is_file() and f.name.endswith(".npy"):
                            id2path[f.name[:-4]] = f.path
    return id2path


train_id2path = build_id_to_path_index(TRAIN_DIR)
test_id2path = build_id_to_path_index(TEST_DIR)

train_ids_on_disk = set(train_id2path.keys())
test_ids_on_disk = set(test_id2path.keys())

train_df = (
    train_labels[train_labels["id"].isin(train_ids_on_disk)]
    .copy()
    .reset_index(drop=True)
)
test_df = (
    sample_sub[["id"]]
    .copy()
    .loc[lambda d: d["id"].isin(test_ids_on_disk)]
    .reset_index(drop=True)
)

print(
    "Train ids on disk:",
    len(train_ids_on_disk),
    "Test ids on disk:",
    len(test_ids_on_disk),
)
print("Filtered train_df:", train_df.shape, "Filtered test_df:", test_df.shape)

if len(test_df) == 0:
    raise RuntimeError(
        "No test files found that match sample_submission ids; cannot create submission."
    )




## === cell 3
def load_snippet_by_id(id2path, snippet_id):
    p = id2path.get(snippet_id)
    if p is None:
        raise FileNotFoundError(f"Snippet file not found for id={snippet_id}")
    return np.load(p)


_PERCENTILE_CACHE = {}  # key: (n, tuple(qs)) -> (lo_idxs, hi_idxs, weights)


def _prepare_percentile_indices(n, qs):
    key = (n, qs)
    v = _PERCENTILE_CACHE.get(key)
    if v is not None:
        return v

    lo = np.empty(len(qs), dtype=np.int64)
    hi = np.empty(len(qs), dtype=np.int64)
    w = np.empty(len(qs), dtype=np.float32)

    for i, q in enumerate(qs):
        if q <= 0:
            lo[i] = hi[i] = 0
            w[i] = 0.0
        elif q >= 100:
            lo[i] = hi[i] = n - 1
            w[i] = 0.0
        else:
            pos = (q / 100.0) * (n - 1)
            loi = int(np.floor(pos))
            hii = int(np.ceil(pos))
            lo[i] = loi
            hi[i] = hii
            w[i] = np.float32(pos - loi)

    v = (lo, hi, w)
    _PERCENTILE_CACHE[key] = v
    return v


def _percentiles_linear_multi(x1d, qs):
    n = x1d.size
    if n == 0:
        return [np.nan] * len(qs)

    lo, hi, w = _prepare_percentile_indices(n, qs)

    need = np.unique(np.concatenate([lo, hi]))
    part = np.partition(x1d, need)
    xlo = part[lo].astype(np.float32, copy=False)
    xhi = part[hi].astype(np.float32, copy=False)
    out = xlo * (1.0 - w) + xhi * w
    return [float(v) for v in out]


def extract_features(arr):
    """
    Core cadence logic (unchanged): build A mean (0,2,4) and OFF mean (1,3,5),
    then compute the same 18 summary statistics.

    Change (score-relevant, minimal): keep the same per-panel log1p transform but
    standardize ONCE per snippet across all 6 panels (instead of per-panel).
    This better preserves the relative A-vs-OFF contrast that the downstream
    A-OFF features rely on, which is expected to improve ROC-AUC.
    """
    x = arr.astype(np.float32, copy=False)

    eps = 1e-6
    x2 = np.empty_like(x, dtype=np.float32)
    for i in range(6):
        xi = x[i]
        xmin = float(xi.min())
        if xmin != 0.0:
            xi = xi - xmin
        if xi.min() < 0.0:
            xi = np.maximum(xi, 0.0)
        x2[i] = np.log1p(xi)

    mu = float(x2.mean())
    sigma = float(x2.std())
    x2 = (x2 - mu) / (sigma + eps)

    A = (x2[0] + x2[2] + x2[4]) * (1.0 / 3.0)
    OFF = (x2[1] + x2[3] + x2[5]) * (1.0 / 3.0)

    D = A - OFF
    AD = np.abs(D)

    D_mean = float(D.mean())
    D_std = float(D.std())
    D_max = float(D.max())
    D_min = float(D.min())

    Df = D.reshape(-1)
    ADf = AD.reshape(-1)

    D_p99, D_p95, D_p05, D_p01 = _percentiles_linear_multi(Df, (99, 95, 5, 1))

    AD_mean = float(AD.mean())
    AD_std = float(AD.std())
    AD_max = float(AD.max())
    (AD_p99,) = _percentiles_linear_multi(ADf, (99,))

    A_mean = float(A.mean())
    A_std = float(A.std())
    OFF_mean = float(OFF.mean())
    OFF_std = float(OFF.std())

    feats = np.array(
        [
            D_mean,
            D_std,
            D_max,
            D_min,
            D_p99,
            D_p95,
            D_p05,
            D_p01,
            AD_mean,
            AD_std,
            AD_max,
            AD_p99,
            A_mean,
            A_std,
            OFF_mean,
            OFF_std,
            (A_mean - OFF_mean),
            (A_std - OFF_std),
        ],
        dtype=np.float32,
    )
    return feats


def _featurize_one_indexed(args):
    i, path = args
    arr = np.load(path)
    return i, extract_features(arr)


def build_feature_matrix(df_ids, id2path, id_col="id", n_jobs=None, chunksize=256):
    n = len(df_ids)
    X = np.zeros((n, 18), dtype=np.float32)
    ids = df_ids[id_col].astype(str).to_numpy()

    paths = [id2path.get(sid) for sid in ids]
    for sid, p in zip(ids, paths):
        if p is None:
            raise FileNotFoundError(f"Snippet file not found for id={sid}")
    tasks = list(enumerate(paths))

    if n_jobs is None:
        cpu = os.cpu_count() or 2
        n_jobs = max(1, min(cpu, 12))

    if n_jobs == 1:
        for i, p in tasks:
            X[i] = extract_features(np.load(p))
        return X

    with ThreadPool(processes=n_jobs) as pool:
        for j, (i, feats) in enumerate(
            pool.imap_unordered(_featurize_one_indexed, tasks, chunksize=chunksize)
        ):
            X[i] = feats
            if (j + 1) % 10000 == 0:
                print(f"Processed {j+1}/{n}")
    return X




## === cell 4
X = build_feature_matrix(
    train_df, train_id2path, id_col="id", n_jobs=None, chunksize=256
)
y = train_df["target"].astype(int).values

print("X shape:", X.shape, "y shape:", y.shape, "pos rate:", y.mean())



## === cell 5
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                class_weight="balanced",
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

try:
    from sklearn.metrics import roc_auc_score

    va_pred = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, va_pred)
    print("Validation AUC:", auc)
except Exception as e:
    print("Could not compute validation AUC:", repr(e))



## === cell 6
clf.fit(X, y)

X_test = build_feature_matrix(
    test_df, test_id2path, id_col="id", n_jobs=None, chunksize=256
)
test_pred = clf.predict_proba(X_test)[:, 1].astype(np.float64)

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 7
sub = sample_sub[["id"]].copy()
sub["id"] = sub["id"].astype(str)

pred_map = pd.Series(test_pred, index=test_df["id"].astype(str)).to_dict()

sub["target"] = sub["id"].map(pred_map)
sub["target"] = sub["target"].fillna(0.5).astype(float)

assert list(sub.columns) == ["id", "target"]
assert sub["target"].between(0, 1).all()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
