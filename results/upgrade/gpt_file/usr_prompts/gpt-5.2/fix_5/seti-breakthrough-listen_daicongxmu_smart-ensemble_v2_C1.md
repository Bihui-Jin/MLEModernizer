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

0.7563403755987789

# 6. Current score

0.49875

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49858) has done: 'The timeout is dominated by per-file disk I/O and Python-loop overhead while extracting features for ~54k train + 6k test `.npy` files one-by-one. I keep the exact same feature definitions and model training, but make feature extraction faster by (1) using `os.scandir` (much faster than recursive `glob`) to build `id->path`, (2) using a `ThreadPoolExecutor` to overlap `.npy` decompression/I/O across multiple workers while preserving output order, and (3) reducing per-sample overhead inside feature extraction (fewer temporary arrays, `ddof=0` explicitly, and preallocated output). This preserves identical core logic and should cut wall time substantially without changing the algorithm or training semantics.'
- What this solution (achieved 0.49835) has done: 'Your current AUC (~0.50) indicates the model is effectively guessing; the most likely issue is that the engineered features are missing the key “A vs off” structure (needles appear in A panels only), so the classifier can’t separate classes. I keep the same pipeline (feature extraction → scaler → logistic regression with CV → average probabilities) but add a few very cheap, directly relevant “A-minus-off” summary features (mean/std/max over (A0+A1+A2)/3 minus (O0+O1+O2)/3) while preserving all existing features. This is a minimal semantic change that should move AUC upward toward your target without changing the model family or training approach. I also add an internal out-of-fold AUC print to verify the direction locally (doesn’t affect submission).'
- What this solution (achieved 0.49875) has done: 'Your AUC is still near-random, so the smallest score-relevant change is to add a couple of very cheap “A vs off” *structure* features that better match how needles are injected (present in A panels 0/2/4 but not in off panels 1/3/5), while keeping your existing feature set and the exact same model/training loop. Concretely, we keep all current global summary features and add (1) correlation of A-mean time series across the three A panels vs across off panels, and (2) a normalized “excess energy in A relative to off” using mean absolute deviation around each panel mean. These features are fast, deterministic, and should move AUC upward toward your target without changing architecture or training semantics. Everything else (paths, CV, scaler+logreg, submission format) remains the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/data"
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def build_id2path(root_dir: str) -> dict:
    id2path = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        _id = f.name[:-4]
                        id2path[_id] = f.path
    return id2path


train_id2path = build_id2path(TRAIN_DIR)
test_id2path = build_id2path(TEST_DIR)

train_df = train_labels[train_labels["id"].isin(train_id2path)].reset_index(drop=True)

test_ids = sample_sub["id"].tolist()
missing_test = [i for i in test_ids if i not in test_id2path]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test .npy files; example: {missing_test[0]}"
    )

print(f"Train rows with files: {len(train_df)} / {len(train_labels)}")
print(f"Test rows: {len(test_ids)}")



## === cell 1
from concurrent.futures import ThreadPoolExecutor


def _corr_1d(a: np.ndarray, b: np.ndarray) -> np.float32:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    am = a.mean()
    bm = b.mean()
    da = a - am
    db = b - bm
    denom = np.sqrt((da * da).mean() * (db * db).mean()) + 1e-6
    return np.float32((da * db).mean() / denom)


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2), ddof=0)
    maxs = x.max(axis=(1, 2))
    mins = x.min(axis=(1, 2))

    A0 = x[0]
    A1 = x[2]
    A2 = x[4]
    O0 = x[1]
    O1 = x[3]
    O2 = x[5]

    A_mean = (A0.mean() + A1.mean() + A2.mean()) / 3.0
    off_mean = (O0.mean() + O1.mean() + O2.mean()) / 3.0

    A_var = (A0.var(ddof=0) + A1.var(ddof=0) + A2.var(ddof=0)) / 3.0 + (
        (
            (A0.mean() - A_mean) ** 2
            + (A1.mean() - A_mean) ** 2
            + (A2.mean() - A_mean) ** 2
        )
        / 3.0
    )
    off_var = (O0.var(ddof=0) + O1.var(ddof=0) + O2.var(ddof=0)) / 3.0 + (
        (
            (O0.mean() - off_mean) ** 2
            + (O1.mean() - off_mean) ** 2
            + (O2.mean() - off_mean) ** 2
        )
        / 3.0
    )

    A_std = np.sqrt(A_var, dtype=np.float32)
    off_std = np.sqrt(off_var, dtype=np.float32)

    diff_mean = A_mean - off_mean
    ratio_std = (A_std + 1e-6) / (off_std + 1e-6)

    A0_time = A0.mean(axis=1)
    A1_time = A1.mean(axis=1)
    A2_time = A2.mean(axis=1)
    O0_time = O0.mean(axis=1)
    O1_time = O1.mean(axis=1)
    O2_time = O2.mean(axis=1)

    A0_freq = A0.mean(axis=0)
    A1_freq = A1.mean(axis=0)
    A2_freq = A2.mean(axis=0)
    O0_freq = O0.mean(axis=0)
    O1_freq = O1.mean(axis=0)
    O2_freq = O2.mean(axis=0)

    time_var_diff = (
        A0_time.var(ddof=0) + A1_time.var(ddof=0) + A2_time.var(ddof=0)
    ) / 3.0 - (O0_time.var(ddof=0) + O1_time.var(ddof=0) + O2_time.var(ddof=0)) / 3.0
    freq_var_diff = (
        A0_freq.var(ddof=0) + A1_freq.var(ddof=0) + A2_freq.var(ddof=0)
    ) / 3.0 - (O0_freq.var(ddof=0) + O1_freq.var(ddof=0) + O2_freq.var(ddof=0)) / 3.0

    A_stack = (A0 + A1 + A2) / 3.0
    O_stack = (O0 + O1 + O2) / 3.0
    D = A_stack - O_stack
    d_mean = D.mean()
    d_std = D.std(ddof=0)
    d_max = D.max()
    d_min = D.min()

    corr_A = (
        _corr_1d(A0_time, A1_time)
        + _corr_1d(A0_time, A2_time)
        + _corr_1d(A1_time, A2_time)
    ) / 3.0
    corr_O = (
        _corr_1d(O0_time, O1_time)
        + _corr_1d(O0_time, O2_time)
        + _corr_1d(O1_time, O2_time)
    ) / 3.0
    corr_diff = np.float32(corr_A - corr_O)

    A_mad = (
        np.mean(np.abs(A0 - A0.mean()))
        + np.mean(np.abs(A1 - A1.mean()))
        + np.mean(np.abs(A2 - A2.mean()))
    ) / 3.0
    O_mad = (
        np.mean(np.abs(O0 - O0.mean()))
        + np.mean(np.abs(O1 - O1.mean()))
        + np.mean(np.abs(O2 - O2.mean()))
    ) / 3.0
    mad_ratio = np.float32((A_mad + 1e-6) / (O_mad + 1e-6))
    mad_diff = np.float32(A_mad - O_mad)

    feats = np.empty(6 * 4 + 8 + 4 + 5, dtype=np.float32)
    feats[0:6] = means
    feats[6:12] = stds
    feats[12:18] = maxs
    feats[18:24] = mins
    feats[24:32] = np.array(
        [
            A_mean,
            off_mean,
            diff_mean,
            A_std,
            off_std,
            ratio_std,
            time_var_diff,
            freq_var_diff,
        ],
        dtype=np.float32,
    )
    feats[32:36] = np.array([d_mean, d_std, d_max, d_min], dtype=np.float32)
    feats[36:41] = np.array(
        [corr_A, corr_O, corr_diff, mad_ratio, mad_diff], dtype=np.float32
    )
    return feats


def extract_features_for_ids(ids, id2path, batch_print_every=5000, max_workers=None):
    ids = list(ids)
    X = np.zeros((len(ids), 6 * 4 + 8 + 4 + 5), dtype=np.float32)

    def _load_and_extract(idx_id):
        idx, _id = idx_id
        arr = np.load(id2path[_id])  # (6,273,256) float16 on disk
        return idx, extract_features_from_array(arr)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(16, cpu)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, feats in ex.map(_load_and_extract, enumerate(ids), chunksize=64):
            X[idx] = feats
            done += 1
            if batch_print_every and done % batch_print_every == 0:
                print(f"Processed {done}/{len(ids)}")
    return X


X_train = extract_features_for_ids(
    train_df["id"].tolist(), train_id2path, batch_print_every=10000
)
y_train = train_df["target"].astype(np.int8).values

print("X_train shape:", X_train.shape, "y_train mean:", y_train.mean())



## === cell 2
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

models = []
oof = np.zeros(len(y_train), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    n_jobs=None,
                    class_weight=None,
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train[tr_idx], y_train[tr_idx])
    oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    models.append(model)
    print(f"Trained fold {fold}/{n_splits}")

print("OOF AUC:", roc_auc_score(y_train, oof))



## === cell 3
X_test = extract_features_for_ids(test_ids, test_id2path, batch_print_every=2000)
print("X_test shape:", X_test.shape)

preds = np.zeros(len(test_ids), dtype=np.float64)
for m in models:
    preds += m.predict_proba(X_test)[:, 1]
preds /= len(models)

preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": preds})



## === cell 4
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
