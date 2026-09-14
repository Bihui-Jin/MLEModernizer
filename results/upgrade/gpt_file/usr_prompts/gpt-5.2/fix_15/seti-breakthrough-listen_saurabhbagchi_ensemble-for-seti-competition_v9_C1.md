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
import hashlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_DIR = "/kaggle/data"
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def find_npy_path_fast(file_id: str, root_dir: str) -> str:
    shard = str(file_id)[0]
    p = os.path.join(root_dir, shard, f"{file_id}.npy")
    if os.path.exists(p):
        return p
    pattern = os.path.join(root_dir, "**", f"{file_id}.npy")
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise FileNotFoundError(f"Could not find {file_id}.npy under {root_dir}")
    return min(hits, key=len)


def _diag_score_vectorized(M: np.ndarray, rates=(-2, -1, 0, 1, 2)) -> np.ndarray:
    T, F = M.shape
    rows = np.arange(T, dtype=np.int32)[:, None]  # (T,1)
    f0 = np.arange(F, dtype=np.int32)[None, :]  # (1,F)
    scores = np.empty(len(rates), dtype=np.float32)

    for j, r in enumerate(rates):
        idx = f0 + (r * rows)  # (T,F)
        valid = (idx >= 0) & (idx < F)
        if not valid.any():
            scores[j] = 0.0
            continue

        all_rows_valid = valid.all(axis=0)  # (F,)
        if not all_rows_valid.any():
            scores[j] = 0.0
            continue

        idx_valid = idx[:, all_rows_valid]  # (T, F_valid)
        gathered = M[rows, idx_valid]
        sums = gathered.sum(axis=0)
        scores[j] = float(sums.max()) if sums.size else 0.0

    return scores


def _quantile_linear_partition(a: np.ndarray, q: float) -> np.float32:
    """
    Exact equivalent of numpy.quantile(a, q, method='linear') without full sort,
    by selecting the two surrounding order statistics and linearly interpolating.
    """
    a = np.asarray(a)
    n = a.size
    if n == 0:
        return np.float32(np.nan)
    if q <= 0.0:
        return np.float32(a.min())
    if q >= 1.0:
        return np.float32(a.max())

    h = (n - 1) * q
    i = int(np.floor(h))
    j = int(np.ceil(h))
    if i == j:
        v = np.partition(a, i)[i]
        return np.float32(v)

    ai = np.partition(a, i)[i]
    aj = np.partition(a, j)[j]
    gamma = h - i
    return np.float32(ai + (aj - ai) * gamma)


def extract_features_from_arr(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    eps = np.float32(1e-6)

    panel_mean = x.mean(axis=(1, 2))
    panel_std = x.std(axis=(1, 2))
    panel_max = x.max(axis=(1, 2))
    panel_min = x.min(axis=(1, 2))

    A = x[[0, 2, 4]]  # on-target
    B = x[[1, 3, 5]]  # off-target

    A_mean = A.mean()
    B_mean = B.mean()
    A_std = A.std()
    B_std = B.std()

    A_abs_mean = np.abs(A).mean()
    B_abs_mean = np.abs(B).mean()

    freq_profile_A = A.mean(axis=(0, 1))
    freq_profile_B = B.mean(axis=(0, 1))
    time_profile_A = A.mean(axis=(0, 2))
    time_profile_B = B.mean(axis=(0, 2))

    fp_diff = freq_profile_A - freq_profile_B
    tp_diff = time_profile_A - time_profile_B

    mean_abs_AB_diff = np.mean(np.abs(A - B))
    pooled = np.sqrt(0.5 * (A_std * A_std + B_std * B_std) + eps)
    standardized_mean_diff = (A_mean - B_mean) / pooled

    base_feats = np.concatenate(
        [
            panel_mean,
            panel_std,
            panel_max,
            panel_min,
            np.array(
                [
                    A_mean,
                    B_mean,
                    A_mean - B_mean,
                    A_std,
                    B_std,
                    A_std - B_std,
                    A_abs_mean,
                    B_abs_mean,
                    A_abs_mean - B_abs_mean,
                    fp_diff.mean(),
                    fp_diff.std(),
                    fp_diff.max(),
                    fp_diff.min(),
                    tp_diff.mean(),
                    tp_diff.std(),
                    tp_diff.max(),
                    tp_diff.min(),
                    mean_abs_AB_diff,
                    standardized_mean_diff,
                ],
                dtype=np.float32,
            ),
        ]
    )

    A_max_t_f = A.max(axis=2)
    B_max_t_f = B.max(axis=2)
    A_time_maxfreq_mean = A_max_t_f.mean()
    B_time_maxfreq_mean = B_max_t_f.mean()
    A_time_maxfreq_std = A_max_t_f.std()
    B_time_maxfreq_std = B_max_t_f.std()

    A_freq_maxtime_mean = A.max(axis=1).mean()
    B_freq_maxtime_mean = B.max(axis=1).mean()

    A_fp_max = freq_profile_A.max()
    B_fp_max = freq_profile_B.max()
    A_fp_peakiness = A_fp_max / (freq_profile_A.std() + eps)
    B_fp_peakiness = B_fp_max / (freq_profile_B.std() + eps)

    abs_fp_diff = np.abs(fp_diff)
    k = 8
    topk_fp = np.partition(abs_fp_diff, -k)[-k:].mean()

    abs_tp_diff = np.abs(tp_diff)
    topk_tp = np.partition(abs_tp_diff, -k)[-k:].mean()

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)

    q50_A = _quantile_linear_partition(A_flat, 0.5)
    q90_A = _quantile_linear_partition(A_flat, 0.9)
    q99_A = _quantile_linear_partition(A_flat, 0.99)
    q995_A = _quantile_linear_partition(A_flat, 0.995)

    q50_B = _quantile_linear_partition(B_flat, 0.5)
    q90_B = _quantile_linear_partition(B_flat, 0.9)
    q99_B = _quantile_linear_partition(B_flat, 0.99)
    q995_B = _quantile_linear_partition(B_flat, 0.995)

    k_tail = max(1, int(0.01 * A_flat.size))
    A_top_tail_mean = np.partition(A_flat, -k_tail)[-k_tail:].mean()
    B_top_tail_mean = np.partition(B_flat, -k_tail)[-k_tail:].mean()

    A_med = q50_A
    B_med = q50_B
    A_peak_minus_med = np.float32(A_flat.max() - A_med)
    B_peak_minus_med = np.float32(B_flat.max() - B_med)
    peak_contrast = np.float32(A_peak_minus_med - B_peak_minus_med)
    q995_contrast = np.float32(q995_A - q995_B)

    extra_feats = np.array(
        [
            A_time_maxfreq_mean,
            B_time_maxfreq_mean,
            A_time_maxfreq_mean - B_time_maxfreq_mean,
            A_time_maxfreq_std,
            B_time_maxfreq_std,
            A_time_maxfreq_std - B_time_maxfreq_std,
            A_freq_maxtime_mean,
            B_freq_maxtime_mean,
            A_freq_maxtime_mean - B_freq_maxtime_mean,
            A_fp_max,
            B_fp_max,
            A_fp_max - B_fp_max,
            A_fp_peakiness,
            B_fp_peakiness,
            A_fp_peakiness - B_fp_peakiness,
            topk_fp,
            topk_tp,
            q50_A,
            q50_B,
            q50_A - q50_B,
            q90_A,
            q90_B,
            q90_A - q90_B,
            q99_A,
            q99_B,
            q99_A - q99_B,
            A_top_tail_mean,
            B_top_tail_mean,
            A_top_tail_mean - B_top_tail_mean,
            q995_A,
            q995_B,
            q995_contrast,
            A_peak_minus_med,
            B_peak_minus_med,
            peak_contrast,
        ],
        dtype=np.float32,
    )

    A_mean_panel = A.mean(axis=0)  # (273, 256)
    B_mean_panel = B.mean(axis=0)

    A_argmax_f = np.argmax(A_mean_panel, axis=1).astype(np.float32)
    B_argmax_f = np.argmax(B_mean_panel, axis=1).astype(np.float32)

    t = np.arange(A_argmax_f.size, dtype=np.float32)
    t0 = t - t.mean()
    denom = np.dot(t0, t0) + eps
    A_slope = np.dot(t0, A_argmax_f - A_argmax_f.mean()) / denom
    B_slope = np.dot(t0, B_argmax_f - B_argmax_f.mean()) / denom

    A_arg_std = A_argmax_f.std()
    B_arg_std = B_argmax_f.std()

    A_diag = _diag_score_vectorized(A_mean_panel)
    B_diag = _diag_score_vectorized(B_mean_panel)

    drift_feats = np.concatenate(
        [
            np.array(
                [
                    A_slope,
                    B_slope,
                    A_slope - B_slope,
                    A_arg_std,
                    B_arg_std,
                    A_arg_std - B_arg_std,
                ],
                dtype=np.float32,
            ),
            A_diag,
            B_diag,
            (A_diag - B_diag),
        ]
    )

    return np.concatenate([base_feats, extra_feats, drift_feats])


def _ids_cache_key(root_dir: str, ids: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(root_dir.encode("utf-8"))
    joined = "\n".join(ids.tolist()).encode("utf-8")
    h.update(joined)
    return h.hexdigest()


import multiprocessing as mp


def _feat_worker_idx_path(args):
    i, path = args
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return i, extract_features_from_arr(arr)


def load_features_for_ids(root_dir: str, ids: np.ndarray) -> np.ndarray:
    n = len(ids)
    cache_dir = os.path.join("/kaggle/working", "feat_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"X_v3_{_ids_cache_key(root_dir, ids)}.npy")

    if os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        return np.asarray(X, dtype=np.float32)

    ids_list = ids.tolist()
    paths = [find_npy_path_fast(fid, root_dir) for fid in ids_list]

    first_arr = np.load(paths[0], mmap_mode="r", allow_pickle=False)
    first_feat = extract_features_from_arr(first_arr)
    d = first_feat.shape[0]

    X = np.empty((n, d), dtype=np.float32)
    X[0] = first_feat

    if n > 1:
        cpu = os.cpu_count() or 2
        workers = max(1, min(6, cpu))
        chunksize = 128 if n >= 8192 else (64 if n >= 2048 else 16)

        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        with ctx.Pool(processes=workers) as pool:
            for i, feat in pool.imap_unordered(
                _feat_worker_idx_path,
                ((i, paths[i]) for i in range(1, n)),
                chunksize=chunksize,
            ):
                X[i] = feat

    np.save(cache_path, X)
    return X


one_id = train_labels["id"].iloc[0]
one_path = find_npy_path_fast(one_id, TRAIN_DIR)
one_arr = np.load(one_path, mmap_mode="r", allow_pickle=False)
(one_id, one_arr.shape, one_arr.dtype, extract_features_from_arr(one_arr).shape)




## === cell 2
from sklearn.metrics import roc_auc_score

train_ids = train_labels["id"].values
y = train_labels["target"].values.astype(np.int64)

X = load_features_for_ids(TRAIN_DIR, train_ids)


def _stable_split_mask(ids: np.ndarray, val_frac: float = 0.2) -> np.ndarray:
    thr = int(val_frac * (2**32 - 1))
    out = np.empty(ids.shape[0], dtype=bool)
    md5 = hashlib.md5
    for i, s in enumerate(ids.tolist()):
        v = int.from_bytes(
            md5(str(s).encode("utf-8")).digest()[:4], "little", signed=False
        )
        out[i] = v <= thr
    return out


val_mask = _stable_split_mask(train_ids, val_frac=0.2)
min_class = min(
    y[val_mask].sum(),
    (1 - y[val_mask]).sum(),
    y[~val_mask].sum(),
    (1 - y[~val_mask]).sum(),
)

if val_mask.sum() < 100 or (~val_mask).sum() < 100 or min_class < 50:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
else:
    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]

clf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced",
                C=1.0,
                solver="saga",
                penalty="l2",
                max_iter=6000,
                n_jobs=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

clf.fit(X_train, y_train)

val_proba = clf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_proba)
(val_proba.min(), val_proba.max(), float(val_proba.mean()), float(val_auc))




## === cell 3
test_ids = sample_sub["id"].values
X_test = load_features_for_ids(TEST_DIR, test_ids)

test_proba = clf.predict_proba(X_test)[:, 1].astype(np.float64)
test_proba = np.clip(test_proba, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": test_proba})
submission.head(), submission.shape




## === cell 4
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

assert os.path.exists(out_path), "submission.csv was not written"
assert list(submission.columns) == ["id", "target"], "Wrong submission columns"
assert submission["id"].is_unique, "Duplicate ids in submission"
assert len(submission) == len(
    sample_sub
), "Submission row count mismatch vs sample_submission"

print(f"Wrote {out_path} with shape {submission.shape}")
print(submission.head())
