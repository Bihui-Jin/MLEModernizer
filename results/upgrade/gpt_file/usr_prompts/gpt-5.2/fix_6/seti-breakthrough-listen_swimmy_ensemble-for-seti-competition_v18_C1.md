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

0.7571827590774655

# 6. Current score

0.49861

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read multiple external “../input/...” submissions that don’t exist in this environment, so nothing downstream is defined and no `submission.csv` is written. To keep the “ensemble of other submissions” core idea but make it runnable, I change it to dynamically search for any available `submission.csv` files under `/kaggle/input/` and blend those if found. If none are available (likely here), it fall back to a valid baseline: generate a submission using the competition’s `sample_submission.csv` with a constant probability (0.5), ensuring correct columns and row alignment. I also add a small safety step to align by `id` and fill missing predictions, so the output is always valid.'
- What this solution (achieved 0.51061) has done: 'The timeout is dominated by per-id filesystem globbing in `_find_npy_path` (called ~60k times) and repeated small Python-loop overhead; both are far more expensive than the actual feature math. I replace glob-per-file with a one-time recursive index of all `.npy` files in train/test and then do O(1) dictionary lookups, which is provably equivalent in terms of which file gets loaded (we keep the same “sorted-first-match” semantics). I also batch feature extraction into preallocated NumPy arrays (no `vstack` growth) and avoid per-sample temporary allocations where possible, keeping the exact same feature definitions and LogisticRegression fit/predict logic. These changes reduce asymptotic filesystem work and Python overhead while preserving identical model/feature/evaluation behavior (up to negligible FP order differences).'
- What this solution (achieved 0.49861) has done: 'The timeout is dominated by per-file `np.load` over ~54k train + 6k test files, plus the expensive recursive glob index build across nested directories; both add heavy Python overhead and random disk I/O. I keep the exact same feature extraction and LogisticRegression training, but speed up data access by (1) avoiding the full recursive index build and instead using a deterministic “direct path then one-time glob fallback” resolver, and (2) parallelizing feature extraction with a thread pool (NumPy releases the GIL during load/compute), while preserving order and determinism. I also avoid redundant work (duplicate imports, repeated list/tuple conversions) and reduce unnecessary allocations, without changing any math or model semantics. The resulting script should stay within 600s on typical Kaggle CPU by cutting directory scanning cost and utilizing parallel I/O/compute.'

# 9. Code solution

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



## === cell 1
BASE_INPUT = "/kaggle/input"
DATA_DIR = "/kaggle/data"

sample_path_candidates = [
    os.path.join(BASE_INPUT, "sample_submission.csv"),
    os.path.join(DATA_DIR, "sample_submission.csv"),
]
SAMPLE_PATH = next((p for p in sample_path_candidates if os.path.exists(p)), None)
if SAMPLE_PATH is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(sample_path_candidates)
    )

sample = pd.read_csv(SAMPLE_PATH)
if list(sample.columns) != ["id", "target"]:
    sample = sample.rename(
        columns={sample.columns[0]: "id", sample.columns[1]: "target"}
    )
sample["id"] = sample["id"].astype(str)



## === cell 2
pass



## === cell 3
candidate_paths = sorted(
    set(
        glob.glob(os.path.join(BASE_INPUT, "submission.csv"))
        + glob.glob(os.path.join(BASE_INPUT, "*submission*.csv"))
    )
)
candidate_paths = [
    p for p in candidate_paths if os.path.basename(p) != "sample_submission.csv"
]

subs = []
for p in candidate_paths:
    try:
        df = pd.read_csv(p)
        if "id" in df.columns and "target" in df.columns:
            df = df[["id", "target"]].copy()
            df["id"] = df["id"].astype(str)
            df["target"] = pd.to_numeric(df["target"], errors="coerce")
            subs.append((p, df))
    except Exception:
        continue

loaded_submission_paths = [p for p, _ in subs]
loaded_submission_paths[:10], len(loaded_submission_paths)



## === cell 4
if subs:
    data1 = subs[0][1].copy()
    data1.head()
else:
    data1 = None
    pd.DataFrame(
        {
            "info": [
                "No external submission files found under /kaggle/input; will not use ensemble for scoring."
            ]
        }
    )



## === cell 5
if len(subs) >= 2:
    data2 = subs[1][1].copy()
    data2.head()
else:
    data2 = None
    pd.DataFrame({"info": ["No second external submission available."]})



## === cell 6
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

TRAIN_LABELS_CANDIDATES = [
    os.path.join(BASE_INPUT, "train_labels.csv"),
    os.path.join(DATA_DIR, "train_labels.csv"),
]
TRAIN_LABELS_PATH = next(
    (p for p in TRAIN_LABELS_CANDIDATES if os.path.exists(p)), None
)
if TRAIN_LABELS_PATH is None:
    raise FileNotFoundError(
        "Could not find train_labels.csv in expected locations: "
        + ", ".join(TRAIN_LABELS_CANDIDATES)
    )

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
train_labels["id"] = train_labels["id"].astype(str)
train_labels["target"] = train_labels["target"].astype(int)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

if not os.path.isdir(TRAIN_DIR):
    TRAIN_DIR = os.path.join(BASE_INPUT, "train")
if not os.path.isdir(TEST_DIR):
    TEST_DIR = os.path.join(BASE_INPUT, "test")

if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
    TRAIN_DIR2 = os.path.join(DATA_DIR, "seti-breakthrough-listen", "train")
    TEST_DIR2 = os.path.join(DATA_DIR, "seti-breakthrough-listen", "test")
    if os.path.isdir(TRAIN_DIR2):
        TRAIN_DIR = TRAIN_DIR2
    if os.path.isdir(TEST_DIR2):
        TEST_DIR = TEST_DIR2

_fallback_cache = {}


def _fallback_build_map(root_dir: str) -> dict:
    m = _fallback_cache.get(root_dir)
    if m is not None:
        return m
    patt = os.path.join(root_dir, "**", "*.npy")
    paths = glob.glob(patt, recursive=True)
    mm = {}
    for p in paths:
        base = os.path.basename(p)
        if base.endswith(".npy"):
            mm[base[:-4]] = p
    _fallback_cache[root_dir] = mm
    return mm


def _find_npy_path(root_dir: str, id_str: str) -> str:
    p0 = os.path.join(root_dir, f"{id_str}.npy")
    if os.path.exists(p0):
        return p0
    sub = id_str[0] if id_str else ""
    if sub:
        p1 = os.path.join(root_dir, sub, f"{id_str}.npy")
        if os.path.exists(p1):
            return p1
    mm = _fallback_build_map(root_dir)
    return mm.get(id_str, "")


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    arr: (6, 273, 256) float16/float32
    Minimal, domain-consistent features:
      - per-panel mean/std/max
      - A-panels (0,2,4) vs non-A panels (1,3,5) differences
      - simple "line-likeness": mean absolute gradient along time/freq
    """
    x = arr.astype(np.float32, copy=False)
    m = float(x.mean())
    s = float(x.std())
    if s > 0:
        x = (x - m) / (s + 1e-6)

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))

    A_idx = (0, 2, 4)
    B_idx = (1, 3, 5)
    A = x[A_idx]
    B = x[B_idx]

    A_mean = float(A.mean())
    B_mean = float(B.mean())
    A_std = float(A.std())
    B_std = float(B.std())
    A_max = float(A.max())
    B_max = float(B.max())

    gt = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2))  # 6
    gf = np.abs(np.diff(x, axis=2)).mean(axis=(1, 2))  # 6
    gt_A = float(gt[0] + gt[2] + gt[4]) / 3.0
    gt_B = float(gt[1] + gt[3] + gt[5]) / 3.0
    gf_A = float(gf[0] + gf[2] + gf[4]) / 3.0
    gf_B = float(gf[1] + gf[3] + gf[5]) / 3.0

    feats = np.concatenate(
        [
            panel_mean,
            panel_std,
            panel_max,
            np.array(
                [
                    A_mean,
                    B_mean,
                    A_mean - B_mean,
                    A_std,
                    B_std,
                    A_std - B_std,
                    A_max,
                    B_max,
                    A_max - B_max,
                    gt_A,
                    gt_B,
                    gt_A - gt_B,
                    gf_A,
                    gf_B,
                    gf_A - gf_B,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    return feats


train_ids = train_labels["id"].tolist()
y_all = train_labels["target"].to_numpy()

n_train = len(train_ids)
n_feat = 6 * 3 + 15


def _load_and_extract_train(i_id):
    i, id_str = i_id
    p = _find_npy_path(TRAIN_DIR, id_str)
    if not p:
        return i, None
    arr = np.load(p, mmap_mode=None)  # keep exact load semantics (no mmap)
    return i, extract_features(arr)


X_all = np.zeros((n_train, n_feat), dtype=np.float32)
present_mask = np.zeros(n_train, dtype=bool)

max_workers = min(8, (mp.cpu_count() or 2))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in ex.map(
        _load_and_extract_train, enumerate(train_ids), chunksize=128
    ):
        if feats is None:
            continue
        X_all[i] = feats
        present_mask[i] = True

missing_train = int((~present_mask).sum())
X = X_all[present_mask]
y = y_all[present_mask]

scaler = StandardScaler(with_mean=True, with_std=True)
X = scaler.fit_transform(X).astype(np.float32, copy=False)

clf = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=200,
    random_state=42,
)
clf.fit(X, y)

print(
    f"Trained LogisticRegression on X.shape={X.shape}, dropped_missing_train={missing_train}, pos_rate={y.mean():.4f}"
)



## === cell 7
test_ids = sample["id"].tolist()
n_test = len(test_ids)

Xte = np.zeros((n_test, n_feat), dtype=np.float32)
present_test = np.zeros(n_test, dtype=bool)


def _load_and_extract_test(i_id):
    i, id_str = i_id
    p = _find_npy_path(TEST_DIR, id_str)
    if not p:
        return i, None
    arr = np.load(p, mmap_mode=None)
    return i, extract_features(arr)


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feats in ex.map(_load_and_extract_test, enumerate(test_ids), chunksize=128):
        if feats is None:
            continue
        Xte[i] = feats
        present_test[i] = True

missing_test = int((~present_test).sum())

Xte_scaled = scaler.transform(Xte).astype(np.float32, copy=False)
preds = clf.predict_proba(Xte_scaled)[:, 1].astype(np.float32)

if missing_test > 0:
    default_p = float(y.mean())
    preds[~present_test] = default_p

preds = np.clip(preds, 0.0, 1.0)

data6 = sample.copy()
data6["target"] = preds

print(
    f"Predicted on Xte.shape={Xte.shape}, missing_test={missing_test}, pred_range=({preds.min():.6f},{preds.max():.6f})"
)
data6.head()



## === cell 8
out_path = "submission.csv"
data6.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert list(data6.columns) == ["id", "target"]
assert len(data6) == len(sample)
print(
    f"Wrote {out_path} with shape={data6.shape} and target range=({data6['target'].min()}, {data6['target'].max()})"
)
