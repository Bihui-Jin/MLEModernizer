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

0.75713

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50484) has done: 'The timeout is dominated by per-file disk I/O and Python-loop overhead when loading ~60k `.npy` files and computing features one at a time. I keep the exact same 17-feature extraction logic and the same LogisticRegression pipeline, but make feature extraction faster by (1) avoiding `glob` + full sorting, (2) using a faster path lookup via `os.walk`, and (3) parallelizing `.npy` loading + feature computation with a thread pool (safe here because NumPy releases the GIL during heavy array ops and file reads). I also micro-optimize the feature function to reduce repeated `mean(axis=...).mean(axis=...)` passes while preserving identical results. All paths and model semantics remain unchanged.'
- What this solution (achieved 0.50744) has done: 'Your current score (0.50484) is far below the target (0.75713), so we should make a small, metric-aligned improvement without changing the overall approach (still: hand-crafted features + LogisticRegression). The main issue is that the 17 global summary stats are too weak for AUC here; we can add a few additional, still “same core logic” statistical features that capture the SETI cadence structure (A vs OFF) and diagonal/drift-like energy patterns more directly. I keep the same pipeline (StandardScaler + LogisticRegression) and training procedure, but expand the feature vector in a deterministic, lightweight way and keep the submission schema/ordering identical. This should move AUC upward toward your target while staying well within runtime constraints.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH = "/kaggle/input/seti-breakthrough-listen"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def list_npy_files(root_dir: str):
    return glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

len(train_files), len(test_files), train_files[:2], test_files[:2]




## === cell 3
def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]
    OFF = x[[1, 3, 5]]

    g_mean = x.mean()
    g_std = x.std()
    g_max = x.max()
    g_min = x.min()

    a_mean = A.mean()
    off_mean = OFF.mean()
    a_std = A.std()
    off_std = OFF.std()
    a_max = A.max()
    off_max = OFF.max()

    mean_diff = a_mean - off_mean
    std_diff = a_std - off_std
    max_diff = a_max - off_max

    a_slice_means = A.mean(axis=(1, 2))  # (3,)
    off_slice_means = OFF.mean(axis=(1, 2))  # (3,)

    a_time_mean = a_slice_means
    off_time_mean = off_slice_means

    a_freq_mean = A.mean(axis=(1, 2))
    off_freq_mean = OFF.mean(axis=(1, 2))

    a_time_var = a_time_mean.std()
    off_time_var = off_time_mean.std()
    a_freq_var = a_freq_mean.std()
    off_freq_var = off_freq_mean.std()

    base = np.array(
        [
            g_mean,
            g_std,
            g_max,
            g_min,
            a_mean,
            off_mean,
            a_std,
            off_std,
            a_max,
            off_max,
            mean_diff,
            std_diff,
            max_diff,
            a_time_var,
            off_time_var,
            a_freq_var,
            off_freq_var,
        ],
        dtype=np.float32,
    )

    abs_mean_diff = np.abs(mean_diff)
    abs_std_diff = np.abs(std_diff)

    a_slice_std = a_slice_means.std()
    off_slice_std = off_slice_means.std()

    a_time_structure = A.std(axis=1).mean(
        axis=(0, 1)
    )  # equivalent to .mean(axis=1).mean()
    off_time_structure = OFF.std(axis=1).mean(axis=(0, 1))
    a_freq_structure = A.std(axis=2).mean(axis=(0, 1))
    off_freq_structure = OFF.std(axis=2).mean(axis=(0, 1))

    k = np.float32(2.0)
    a_thr = a_mean + k * a_std
    off_thr = off_mean + k * off_std
    a_tail_frac = (A > a_thr).mean()
    off_tail_frac = (OFF > off_thr).mean()
    tail_frac_diff = a_tail_frac - off_tail_frac

    extra = np.array(
        [
            abs_mean_diff,
            abs_std_diff,
            a_slice_std,
            off_slice_std,
            a_time_structure,
            off_time_structure,
            a_freq_structure,
            off_freq_structure,
            a_tail_frac,
            off_tail_frac,
            tail_frac_diff,
        ],
        dtype=np.float32,
    )

    x_flat = x.reshape(-1)
    a_flat = A.reshape(-1)
    off_flat = OFF.reshape(-1)

    q = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float32)
    x_q = np.quantile(x_flat, q).astype(np.float32)
    a_q = np.quantile(a_flat, q).astype(np.float32)
    off_q = np.quantile(off_flat, q).astype(np.float32)

    x_iqr = np.float32(x_q[3] - x_q[1])
    a_iqr = np.float32(a_q[3] - a_q[1])
    off_iqr = np.float32(off_q[3] - off_q[1])

    A_mean_t = A.mean(axis=2)  # (3,273)
    OFF_mean_t = OFF.mean(axis=2)
    A_row_max = A_mean_t.max(axis=1).mean()
    OFF_row_max = OFF_mean_t.max(axis=1).mean()

    A_mean_f = A.mean(axis=1)  # (3,256)
    OFF_mean_f = OFF.mean(axis=1)
    A_col_max = A_mean_f.max(axis=1).mean()
    OFF_col_max = OFF_mean_f.max(axis=1).mean()

    row_max_diff = np.float32(A_row_max - OFF_row_max)
    col_max_diff = np.float32(A_col_max - OFF_col_max)

    a_panel_range = np.float32(a_slice_means.max() - a_slice_means.min())
    off_panel_range = np.float32(off_slice_means.max() - off_slice_means.min())
    panel_range_diff = np.float32(a_panel_range - off_panel_range)

    A_time_maxmean = A_row_max
    OFF_time_maxmean = OFF_row_max
    A_freq_maxmean = A_col_max
    OFF_freq_maxmean = OFF_col_max

    time_maxmean_diff = np.float32(A_time_maxmean - OFF_time_maxmean)
    freq_maxmean_diff = np.float32(A_freq_maxmean - OFF_freq_maxmean)

    added = np.array(
        [
            x_q[0],
            x_q[1],
            x_q[2],
            x_q[3],
            x_q[4],
            x_iqr,
            a_q[0],
            a_q[1],
            a_q[2],
            a_q[3],
            a_q[4],
            a_iqr,
            off_q[0],
            off_q[1],
            off_q[2],
            off_q[3],
            off_q[4],
            off_iqr,
            A_row_max,
            OFF_row_max,
            row_max_diff,
            A_col_max,
            OFF_col_max,
            col_max_diff,
            a_panel_range,
            off_panel_range,
            panel_range_diff,
            A_time_maxmean,
            OFF_time_maxmean,
            time_maxmean_diff,
            A_freq_maxmean,
            OFF_freq_maxmean,
            freq_maxmean_diff,
        ],
        dtype=np.float32,
    )

    return np.concatenate([base, extra, added], axis=0)


def id_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]




## === cell 4
train_path_by_id = {id_from_path(p): p for p in train_files}
test_path_by_id = {id_from_path(p): p for p in test_files}

missing_train = train_labels.loc[
    ~train_labels["id"].isin(train_path_by_id.keys()), "id"
]
missing_test = sample_sub.loc[~sample_sub["id"].isin(test_path_by_id.keys()), "id"]

len(missing_train), len(missing_test)



## === cell 5
from concurrent.futures import ThreadPoolExecutor


def _featurize_one(p: str):
    if p is None:
        return None
    try:
        arr = np.load(p)  # keep default mmap_mode=None for identical behavior
        return extract_features_from_array(arr)
    except Exception:
        return None


_CPU = os.cpu_count() or 4
_MAX_WORKERS = min(
    16, max(4, _CPU)
)  # IO-bound workload; 4-16 is typically optimal on Kaggle
_CHUNKSIZE = 256  # larger chunk reduces Python scheduling overhead

N_FEATS = 17 + 11 + 33  # base + extra + added (33 values)

X_train = np.zeros((len(train_labels), N_FEATS), dtype=np.float32)
y_train = train_labels["target"].astype(np.int64).values

train_ids = train_labels["id"].values
train_paths = [train_path_by_id.get(_id) for _id in train_ids]

with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
    for i, feats in enumerate(
        ex.map(_featurize_one, train_paths, chunksize=_CHUNKSIZE)
    ):
        X_train[i] = 0.0 if feats is None else feats

X_train.shape, y_train.shape, X_train[:2]



## === cell 6
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs", max_iter=1000, n_jobs=None, C=1.0, class_weight=None
            ),
        ),
    ]
)

model.fit(X_train, y_train)



## === cell 7
X_test = np.zeros((len(sample_sub), N_FEATS), dtype=np.float32)
test_ids = sample_sub["id"].values
test_paths = [test_path_by_id.get(_id) for _id in test_ids]

with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
    for i, feats in enumerate(ex.map(_featurize_one, test_paths, chunksize=_CHUNKSIZE)):
        X_test[i] = 0.0 if feats is None else feats

pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

pred.min(), pred.max(), pred.mean()



## === cell 8
submission = pd.DataFrame({"id": test_ids, "target": pred})
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape
