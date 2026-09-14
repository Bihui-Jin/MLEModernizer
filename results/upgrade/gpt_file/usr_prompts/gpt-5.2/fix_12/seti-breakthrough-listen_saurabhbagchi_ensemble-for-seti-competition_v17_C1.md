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

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)

BASE = Path("/kaggle/input")
COMP_BASE = BASE / "seti-breakthrough-listen"

if (COMP_BASE / "train").exists() and (COMP_BASE / "test").exists():
    DATA_BASE = COMP_BASE
else:
    DATA_BASE = BASE

TRAIN_DIR = DATA_BASE / "train"
TEST_DIR = DATA_BASE / "test"
TRAIN_LABELS_PATH = DATA_BASE / "train_labels.csv"
SAMPLE_SUB_PATH = DATA_BASE / "sample_submission.csv"

assert TRAIN_DIR.exists(), f"Missing train dir: {TRAIN_DIR}"
assert TEST_DIR.exists(), f"Missing test dir: {TEST_DIR}"
assert TRAIN_LABELS_PATH.exists(), f"Missing train labels: {TRAIN_LABELS_PATH}"
assert SAMPLE_SUB_PATH.exists(), f"Missing sample submission: {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

train_labels.head(), sample_sub.head()




## === cell 1
def list_npy_files(root: Path):
    """
    Bug fix: the dataset is sharded, but shard folder names are not guaranteed
    to be exactly '0'..'15' in every copied layout. The previous implementation
    missed many files -> most predictions defaulted to 0.5 (AUC ~ 0.5).
    We now:
      - scan all immediate subdirectories for *.npy
      - also include any *.npy directly under root (flat layout fallback)
    """
    files = []

    files.extend(root.glob("*.npy"))

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        files.extend(d.glob("*.npy"))

    return sorted(files)


def id_from_path(p: Path) -> str:
    name = p.name
    while name.endswith(".npy"):
        name = name[: -len(".npy")]
    return name


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

train_map = {id_from_path(p): p for p in train_files}
test_map = {id_from_path(p): p for p in test_files}

train_keys = set(train_map.keys())
train_df = train_labels[train_labels["id"].isin(train_keys)].copy()
train_df["path"] = train_df["id"].map(train_map)
train_df = train_df.sort_values("id").reset_index(drop=True)

test_ids = sample_sub["id"].tolist()
test_paths = [test_map.get(i, None) for i in test_ids]
missing_test = [i for i, p in zip(test_ids, test_paths) if p is None]

print("Using DATA_BASE:", DATA_BASE)
print("Train files found:", len(train_files))
print("Test files found:", len(test_files))
print("Unique train ids from files:", len(train_map))
print("Unique test ids from files:", len(test_map))
print("Train rows with files:", len(train_df), "of", len(train_labels))
print("Test rows missing files:", len(missing_test), "of", len(test_ids))
if missing_test:
    print("First missing test ids:", missing_test[:5])

train_cov = len(train_df) / len(train_labels)
test_miss = len(missing_test) / len(test_ids)
print(f"Train coverage: {train_cov:.4f} ; Test missing rate: {test_miss:.4f}")

assert train_cov > 0.95, "Too many train files missing; ID mapping likely broken."
assert test_miss < 0.05, "Too many test files missing; ID mapping likely broken."

len(train_df), len(test_paths), (
    len(missing_test),
    missing_test[:5] if missing_test else [],
)


## === cell 2
A_IDX = np.array([0, 2, 4])
B_IDX = np.array([1, 3, 5])


def _quantile90_flat_lastaxis(x2d: np.ndarray) -> np.ndarray:
    """
    Compute q=0.90 for each row of a 2D array using partial selection.
    Returns shape (N,). Works on float32. Equivalent to np.quantile(row, 0.90)
    with default method ('linear') for discrete data.
    """
    M = x2d.shape[1]
    if M == 1:
        return x2d[:, 0]

    pos = 0.90 * (M - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        kth = lo
        part = np.partition(x2d, kth, axis=1)
        return part[:, kth]

    w = np.float32(pos - lo)
    part = np.partition(x2d, (lo, hi), axis=1)
    vlo = part[:, lo]
    vhi = part[:, hi]
    return vlo + (vhi - vlo) * w


def extract_features_from_snippet(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))
    panel_min = x.min(axis=(1, 2))

    A = x[A_IDX]
    B = x[B_IDX]

    A_mean = A.mean()
    B_mean = B.mean()
    A_std = A.std()
    B_std = B.std()
    A_max = A.max()
    B_max = B.max()

    diff_mean = A_mean - B_mean
    diff_std = A_std - B_std
    diff_max = A_max - B_max

    time_profile = x.mean(axis=2)  # (6,273)
    freq_profile = x.mean(axis=1)  # (6,256)
    time_var = time_profile.var(axis=1)  # (6,)
    freq_var = freq_profile.var(axis=1)  # (6,)

    pair_dmean = panel_mean[A_IDX] - panel_mean[B_IDX]  # (3,)
    pair_dstd = panel_std[A_IDX] - panel_std[B_IDX]  # (3,)
    pair_dmax = panel_max[A_IDX] - panel_max[B_IDX]  # (3,)

    time_var_diff = time_var[A_IDX].mean() - time_var[B_IDX].mean()
    freq_var_diff = freq_var[A_IDX].mean() - freq_var[B_IDX].mean()

    fA = A.mean(axis=1)  # (3,256) mean over time
    fB = B.mean(axis=1)
    tA = A.mean(axis=2)  # (3,273) mean over freq
    tB = B.mean(axis=2)

    fA_mean = fA.mean()
    fB_mean = fB.mean()
    fA_max = fA.max()
    fB_max = fB.max()
    tA_mean = tA.mean()
    tB_mean = tB.mean()
    tA_max = tA.max()
    tB_max = tB.max()

    fA_q90 = _quantile90_flat_lastaxis(fA.reshape(1, -1))[0]
    fB_q90 = _quantile90_flat_lastaxis(fB.reshape(1, -1))[0]
    tA_q90 = _quantile90_flat_lastaxis(tA.reshape(1, -1))[0]
    tB_q90 = _quantile90_flat_lastaxis(tB.reshape(1, -1))[0]

    med = np.median(x)
    pos = np.maximum(x - med, 0.0)
    posA = pos[A_IDX]
    posB = pos[B_IDX]
    posA_mean = posA.mean()
    posB_mean = posB.mean()
    posA_max = posA.max()
    posB_max = posB.max()

    feats = np.concatenate(
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
                    diff_mean,
                    diff_std,
                    diff_max,
                ],
                dtype=np.float32,
            ),
            time_var.astype(np.float32),
            freq_var.astype(np.float32),
            pair_dmean.astype(np.float32),
            pair_dstd.astype(np.float32),
            pair_dmax.astype(np.float32),
            np.array([time_var_diff, freq_var_diff], dtype=np.float32),
            np.array(
                [
                    fA_mean,
                    fB_mean,
                    fA_max,
                    fB_max,
                    fA_q90,
                    fB_q90,
                    tA_mean,
                    tB_mean,
                    tA_max,
                    tB_max,
                    tA_q90,
                    tB_q90,
                    (fA_mean - fB_mean),
                    (fA_max - fB_max),
                    (fA_q90 - fB_q90),
                    (tA_mean - tB_mean),
                    (tA_max - tB_max),
                    (tA_q90 - tB_q90),
                    posA_mean,
                    posB_mean,
                    posA_max,
                    posB_max,
                    (posA_mean - posB_mean),
                    (posA_max - posB_max),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return feats


def extract_features_batch(xb: np.ndarray) -> np.ndarray:
    x = xb  # (N,6,273,256) float32

    panel_mean = x.mean(axis=(2, 3))  # (N,6)
    panel_std = x.std(axis=(2, 3))  # (N,6)
    panel_max = x.max(axis=(2, 3))  # (N,6)
    panel_min = x.min(axis=(2, 3))  # (N,6)

    A = x[:, A_IDX, :, :]  # (N,3,273,256)
    B = x[:, B_IDX, :, :]

    A_mean = A.mean(axis=(1, 2, 3))  # (N,)
    B_mean = B.mean(axis=(1, 2, 3))
    A_std = A.std(axis=(1, 2, 3))
    B_std = B.std(axis=(1, 2, 3))
    A_max = A.max(axis=(1, 2, 3))
    B_max = B.max(axis=(1, 2, 3))

    diff_mean = A_mean - B_mean
    diff_std = A_std - B_std
    diff_max = A_max - B_max

    time_profile = x.mean(axis=3)  # (N,6,273)
    freq_profile = x.mean(axis=2)  # (N,6,256)
    time_var = time_profile.var(axis=2)  # (N,6)
    freq_var = freq_profile.var(axis=2)  # (N,6)

    pair_dmean = panel_mean[:, A_IDX] - panel_mean[:, B_IDX]  # (N,3)
    pair_dstd = panel_std[:, A_IDX] - panel_std[:, B_IDX]  # (N,3)
    pair_dmax = panel_max[:, A_IDX] - panel_max[:, B_IDX]  # (N,3)

    time_var_diff = time_var[:, A_IDX].mean(axis=1) - time_var[:, B_IDX].mean(axis=1)
    freq_var_diff = freq_var[:, A_IDX].mean(axis=1) - freq_var[:, B_IDX].mean(axis=1)

    group_feats = np.stack(
        [A_mean, B_mean, A_std, B_std, A_max, B_max, diff_mean, diff_std, diff_max],
        axis=1,
    ).astype(
        np.float32, copy=False
    )  # (N,9)

    extra_feats = np.concatenate(
        [
            pair_dmean.astype(np.float32, copy=False),
            pair_dstd.astype(np.float32, copy=False),
            pair_dmax.astype(np.float32, copy=False),
            np.stack([time_var_diff, freq_var_diff], axis=1).astype(
                np.float32, copy=False
            ),
        ],
        axis=1,
    )  # (N,11)

    fA = A.mean(axis=2)  # (N,3,256) mean over time
    fB = B.mean(axis=2)
    tA = A.mean(axis=3)  # (N,3,273) mean over freq
    tB = B.mean(axis=3)

    fA_mean = fA.mean(axis=(1, 2))
    fB_mean = fB.mean(axis=(1, 2))
    fA_max = fA.max(axis=(1, 2))
    fB_max = fB.max(axis=(1, 2))
    tA_mean = tA.mean(axis=(1, 2))
    tB_mean = tB.mean(axis=(1, 2))
    tA_max = tA.max(axis=(1, 2))
    tB_max = tB.max(axis=(1, 2))

    fA_flat = fA.reshape(fA.shape[0], -1)
    fB_flat = fB.reshape(fB.shape[0], -1)
    tA_flat = tA.reshape(tA.shape[0], -1)
    tB_flat = tB.reshape(tB.shape[0], -1)
    fA_q90 = _quantile90_flat_lastaxis(fA_flat)
    fB_q90 = _quantile90_flat_lastaxis(fB_flat)
    tA_q90 = _quantile90_flat_lastaxis(tA_flat)
    tB_q90 = _quantile90_flat_lastaxis(tB_flat)

    med = np.median(x, axis=(1, 2, 3))  # (N,)
    pos = np.maximum(x - med[:, None, None, None], 0.0)
    posA = pos[:, A_IDX, :, :]
    posB = pos[:, B_IDX, :, :]
    posA_mean = posA.mean(axis=(1, 2, 3))
    posB_mean = posB.mean(axis=(1, 2, 3))
    posA_max = posA.max(axis=(1, 2, 3))
    posB_max = posB.max(axis=(1, 2, 3))

    axis_contrast = np.stack(
        [
            fA_mean,
            fB_mean,
            fA_max,
            fB_max,
            fA_q90,
            fB_q90,
            tA_mean,
            tB_mean,
            tA_max,
            tB_max,
            tA_q90,
            tB_q90,
            (fA_mean - fB_mean),
            (fA_max - fB_max),
            (fA_q90 - fB_q90),
            (tA_mean - tB_mean),
            (tA_max - tB_max),
            (tA_q90 - tB_q90),
            posA_mean,
            posB_mean,
            posA_max,
            posB_max,
            (posA_mean - posB_mean),
            (posA_max - posB_max),
        ],
        axis=1,
    ).astype(
        np.float32, copy=False
    )  # (N,24)

    feats = np.concatenate(
        [
            panel_mean.astype(np.float32, copy=False),
            panel_std.astype(np.float32, copy=False),
            panel_max.astype(np.float32, copy=False),
            panel_min.astype(np.float32, copy=False),
            group_feats,
            time_var.astype(np.float32, copy=False),
            freq_var.astype(np.float32, copy=False),
            extra_feats,
            axis_contrast,
        ],
        axis=1,
    )
    return feats


p0 = train_df["path"].iloc[0]
arr0 = np.load(p0)
feat0 = extract_features_from_snippet(arr0)
arr0.shape, feat0.shape, feat0[:10]


## === cell 3
from concurrent.futures import ThreadPoolExecutor

_GLOBAL_EX = None
_GLOBAL_EX_WORKERS = None


def _load_npy_f32(path_str: str) -> np.ndarray:
    arr = np.load(path_str, mmap_mode="r")
    return arr.astype(np.float32, copy=False)


def _get_executor(max_workers: int):
    global _GLOBAL_EX, _GLOBAL_EX_WORKERS
    if _GLOBAL_EX is None or _GLOBAL_EX_WORKERS != max_workers:
        if _GLOBAL_EX is not None:
            _GLOBAL_EX.shutdown(wait=True, cancel_futures=False)
        _GLOBAL_EX = ThreadPoolExecutor(max_workers=max_workers)
        _GLOBAL_EX_WORKERS = max_workers
    return _GLOBAL_EX


def build_feature_matrix(
    paths, batch_size: int = 256, max_workers: int = None
) -> np.ndarray:
    n = len(paths)
    n_features = (6 * 4) + 9 + 6 + 6 + 11 + 24  # 80
    X_out = np.empty((n, n_features), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    ex = _get_executor(max_workers)

    xb_buf = np.empty((batch_size, 6, 273, 256), dtype=np.float32)

    path_strs = [str(p) for p in paths]

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        bsz = end - start
        batch_paths = path_strs[start:end]

        for i, arr in enumerate(ex.map(_load_npy_f32, batch_paths)):
            xb_buf[i, ...] = arr  # (6,273,256)

        X_out[start:end] = extract_features_batch(xb_buf[:bsz])
    return X_out


train_paths = train_df["path"].tolist()
X = build_feature_matrix(train_paths, batch_size=256)
y = train_df["target"].to_numpy(dtype=np.int32)

X.shape, y.shape, (y.mean(), y.min(), y.max())


## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []
for tr_idx, va_idx in skf.split(X, y):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    C=1.0,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X[tr_idx], y[tr_idx])
    models.append(model)

len(models)


## === cell 5
test_pred = np.full((len(test_ids),), 0.5, dtype=np.float32)

valid_idx = [i for i, p in enumerate(test_paths) if p is not None]
if valid_idx:
    valid_paths = [test_paths[i] for i in valid_idx]
    X_test_valid = build_feature_matrix(valid_paths, batch_size=256)

    proba_sum = np.zeros((len(valid_idx),), dtype=np.float64)
    for m in models:
        proba_sum += m.predict_proba(X_test_valid)[:, 1]
    proba_avg = (proba_sum / len(models)).astype(np.float32)

    test_pred[np.array(valid_idx, dtype=np.int64)] = proba_avg

test_pred = np.clip(test_pred, 0.0, 1.0)

sub = pd.DataFrame({"id": test_ids, "target": test_pred})
assert list(sub.columns) == ["id", "target"]
assert len(sub) == len(sample_sub)

sub.head(), sub["target"].describe()


## === cell 6
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv preview:")
print(sub.head(10).to_string(index=False))

if _GLOBAL_EX is not None:
    _GLOBAL_EX.shutdown(wait=True, cancel_futures=False)
