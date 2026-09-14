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

0.7589987562359158

# 6. Current score

0.49825

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your current notebook is an ensemble that depends on 7 external Kaggle Dataset/Notebook submission files that are not present in this environment, causing the FileNotFoundError and preventing any submission from being written. To keep the “core logic” (a weighted average ensemble) but make it runnable end-to-end, I (1) automatically discover any available `submission.csv` files under `/kaggle/input` and use them in the same weighted-ensemble pattern when possible, and (2) fall back to a safe baseline submission (all 0.5) using the provided `sample_submission.csv` when those inputs are unavailable. I also ensure the output has exactly `id,target` with 6000 rows and writes `submission.csv` in the working directory. This is the smallest change that guarantees a valid CSV and, when any ensemble components exist, use them to improve score toward your target.'
- What this solution (achieved 0.49833) has done: 'The timeout is almost entirely driven by loading tens of thousands of `.npy` files and doing feature extraction in Python per file; the current threadpool still pays heavy per-file overhead and also rebuilds ID/path lists repeatedly. I keep the exact same feature definitions and the same 5-fold LogisticRegression pipeline, but speed up data access by (1) using `os.scandir`-based directory walking instead of `glob`, (2) vectorizing the per-array feature extraction to reduce temporary arrays and redundant passes, and (3) using a bounded threadpool with a larger chunk size plus `np.load(..., mmap_mode="r")` to reduce memory pressure and improve I/O behavior. I also make the feature caching validation cheaper and ensure ids are pre-converted to numpy arrays once to avoid repeated Python overhead. All changes are runtime-focused and preserve the algorithm and outputs up to negligible floating-point differences.'
- What this solution (achieved 0.49834) has done: 'Your current score is far below the target, so we should improve AUC while keeping the same overall pipeline (handcrafted features → StandardScaler → 5-fold LogisticRegression → mean test proba). The smallest high-impact fix is to prevent cross-validation leakage: right now you fit the scaler+model on the full training fold but never validate/tune, and the split is random (which can mismatch the data’s grouped structure); switching to a deterministic group-aware split using the directory label (the 0–15 subfolder) as a proxy “group” typically improves generalization for this dataset without changing the model/feature logic. Additionally, we should ensure train/test ids are aligned to actual files (avoid silent zero-feature rows when an id is missing a file), because those zeros push predictions toward ~0.5 and hurt AUC. These are minimal, semantics-preserving changes that usually move the score upward toward your target.'
- What this solution (achieved 0.49825) has done: 'Your current AUC is far below the target, so we should make a small, semantics-preserving change that tends to improve generalization without changing the model or features. The main issue is that your `groups` are derived from the directory label (0–15), which is a very coarse proxy and can over-constrain folds; switching to deterministic stratified CV (no groups) usually improves AUC for this handcrafted-feature + LogisticRegression baseline on this dataset. I keep the exact same feature extraction, scaler, LogisticRegression settings, and 5-fold averaging, only changing the CV splitter and adding a stable ID sort so predictions align deterministically. The script still run end-to-end and write a valid `submission.csv` with `id,target` and 6000 rows.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


BASE = first_existing(BASE_CANDIDATES)
if BASE is None:
    raise FileNotFoundError(f"None of the base paths exist: {BASE_CANDIDATES}")

COMP_ROOT = (
    BASE
    if os.path.exists(os.path.join(BASE, "train"))
    else os.path.join(BASE, "seti-breakthrough-listen")
)
if not os.path.exists(COMP_ROOT):
    COMP_ROOT = BASE

TRAIN_DIR = os.path.join(COMP_ROOT, "train")
TEST_DIR = os.path.join(COMP_ROOT, "test")

LABELS_PATHS = [
    os.path.join(COMP_ROOT, "train_labels.csv"),
    os.path.join(BASE, "train_labels.csv"),
]
SAMPLE_PATHS = [
    os.path.join(COMP_ROOT, "sample_submission.csv"),
    os.path.join(BASE, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
]


def read_first_csv(paths):
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"CSV not found in any of: {paths}")


train_labels = read_first_csv(LABELS_PATHS)
sample_sub = read_first_csv(SAMPLE_PATHS)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("COMP_ROOT:", COMP_ROOT)
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print("train_labels:", train_labels.shape, "sample_sub:", sample_sub.shape)




## === cell 2
def extract_features_from_array(x):
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]
    O = x[[1, 3, 5]]

    A_mean = A.mean()
    O_mean = O.mean()

    A_std = A.std()
    O_std = O.std()

    diff = A - O
    diff_mean = diff.mean()
    diff_std = diff.std()

    A_absmean = np.abs(A).mean()
    O_absmean = np.abs(O).mean()
    diff_absmean = np.abs(diff).mean()

    A_max = A.max()
    O_max = O.max()
    max_diff = A_max - O_max

    A_freq = A.mean(axis=1)  # (3, 256)
    O_freq = O.mean(axis=1)  # (3, 256)
    A_time = A.mean(axis=2)  # (3, 273)
    O_time = O.mean(axis=2)  # (3, 273)

    freq_diff = A_freq - O_freq
    time_diff = A_time - O_time

    freq_diff_mean = freq_diff.mean()
    freq_diff_std = freq_diff.std()
    time_diff_mean = time_diff.mean()
    time_diff_std = time_diff.std()

    A_center = A - A_mean
    O_center = O - O_mean
    A_c2 = A_center * A_center
    O_c2 = O_center * O_center
    A_kurt_proxy = (A_c2 * A_c2).mean() / (A_std * A_std * A_std * A_std + 1e-6)
    O_kurt_proxy = (O_c2 * O_c2).mean() / (O_std * O_std * O_std * O_std + 1e-6)

    return np.array(
        [
            A_mean,
            O_mean,
            A_std,
            O_std,
            diff_mean,
            diff_std,
            A_absmean,
            O_absmean,
            diff_absmean,
            A_max,
            O_max,
            max_diff,
            freq_diff_mean,
            freq_diff_std,
            time_diff_mean,
            time_diff_std,
            A_kurt_proxy,
            O_kurt_proxy,
        ],
        dtype=np.float32,
    )




## === cell 3
def build_id_path_group_map(root_dir):
    mp = {}
    grp = {}
    if not os.path.exists(root_dir):
        return mp, grp
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            group_name = os.path.basename(entry.path)
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        id_ = f.name[:-4]
                        mp[id_] = f.path
                        grp[id_] = group_name
    return mp, grp


train_map, train_group_map = build_id_path_group_map(TRAIN_DIR)
test_map, _ = build_id_path_group_map(TEST_DIR)

print("Train .npy files found:", len(train_map))
print("Test  .npy files found:", len(test_map))

missing_train = train_labels.loc[~train_labels["id"].isin(train_map), "id"]
missing_test = sample_sub.loc[~sample_sub["id"].isin(test_map), "id"]
print("Missing train files for labels:", len(missing_train))
print("Missing test files for sample ids:", len(missing_test))

train_df = train_labels[train_labels["id"].isin(train_map)].reset_index(drop=True)

train_df = train_df.sort_values("id").reset_index(drop=True)

test_ids_existing = [i for i in sample_sub["id"].tolist() if i in test_map]
if len(test_ids_existing) != len(sample_sub):
    print(
        f"Warning: {len(sample_sub) - len(test_ids_existing)} test ids missing files; will fill with 0.5 later."
    )
test_ids = sorted(test_ids_existing)




## === cell 4
FEATURE_CACHE_TRAIN = "train_features.npy"
FEATURE_CACHE_TEST = "test_features.npy"
FEATURE_CACHE_TRAIN_IDS = "train_features_ids.npy"
FEATURE_CACHE_TEST_IDS = "test_features_ids.npy"


def compute_features_for_ids(ids, id_path_map, cache_path, cache_ids_path):
    ids_arr = np.asarray(ids, dtype=str)

    if os.path.exists(cache_path) and os.path.exists(cache_ids_path):
        try:
            cached_ids = np.load(cache_ids_path, allow_pickle=True)
            if cached_ids.shape[0] == ids_arr.shape[0]:
                cached_ids_str = np.asarray(cached_ids, dtype=str)
                if np.array_equal(cached_ids_str, ids_arr):
                    arr = np.load(cache_path, mmap_mode="r")
                    if arr.shape == (len(ids_arr), 18) and arr.dtype == np.float32:
                        return np.asarray(arr)
        except Exception:
            pass

    paths = [id_path_map.get(id_) for id_ in ids_arr]
    n = len(ids_arr)
    feats = np.zeros((n, 18), dtype=np.float32)

    def worker(i):
        p = paths[i]
        if p is None:
            return i, None
        x = np.load(p, mmap_mode="r")
        return i, extract_features_from_array(x)

    max_workers = min(8, (os.cpu_count() or 2))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, fvec in ex.map(worker, range(n), chunksize=256):
            if fvec is not None:
                feats[i, :] = fvec

    np.save(cache_path, feats)
    np.save(cache_ids_path, np.asarray(ids_arr, dtype=object))
    return feats


X_train = compute_features_for_ids(
    train_df["id"].tolist(), train_map, FEATURE_CACHE_TRAIN, FEATURE_CACHE_TRAIN_IDS
)
y_train = train_df["target"].to_numpy(dtype=np.int32)

X_test = compute_features_for_ids(
    test_ids, test_map, FEATURE_CACHE_TEST, FEATURE_CACHE_TEST_IDS
)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)




## === cell 5
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=200,
                C=1.0,
                n_jobs=None,
                class_weight=None,
                random_state=42,
            ),
        ),
    ]
)

test_pred = np.zeros(len(test_ids), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    model.fit(X_tr, y_tr)
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    print(f"Fold {fold} done.")

test_pred = np.clip(test_pred, 0.0, 1.0)

sub_out = sample_sub.copy()
sub_out["target"] = 0.5
pred_map = dict(zip(test_ids, test_pred.astype(np.float32)))
sub_out["target"] = sub_out["id"].map(pred_map).fillna(0.5).astype(np.float32)

sub_out.to_csv("submission.csv", index=False)

sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == [
    "id",
    "target",
], "Submission must have columns: id,target"
assert len(sub_check) == len(sample_sub), f"Submission must have {len(sample_sub)} rows"
assert sub_check["target"].between(0, 1).all(), "All targets must be in [0,1]"

print("Wrote submission.csv")
print(sub_check.head())
print(sub_check["target"].describe())
