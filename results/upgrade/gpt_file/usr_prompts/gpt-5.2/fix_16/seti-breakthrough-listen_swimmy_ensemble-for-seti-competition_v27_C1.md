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
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

BASE = "/kaggle/data"
TEST_DIR = os.path.join(BASE, "test")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

if not os.path.exists(SAMPLE_SUB_PATH):
    BASE = "/kaggle/input"
    TEST_DIR = os.path.join(BASE, "test")
    SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

sample = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample["id"].astype(str).tolist()

id_to_path = {}
for entry in os.scandir(TEST_DIR):
    if not entry.is_dir():
        continue
    with os.scandir(entry.path) as it:
        for f in it:
            if f.is_file() and f.name.endswith(".npy"):
                sid = f.name[:-4]
                id_to_path[sid] = f.path

missing = [i for i in sample_ids if i not in id_to_path]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test .npy files referenced by sample_submission. Example: {missing[:5]}"
    )




## === cell 1
def _winsorize(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    flat = x.reshape(-1)

    n = flat.size
    k_lo = int(np.floor((lo / 100.0) * (n - 1)))
    k_hi = int(np.floor((hi / 100.0) * (n - 1)))

    a = float(np.partition(flat, k_lo)[k_lo])
    b = float(np.partition(flat, k_hi)[k_hi])
    return np.clip(x, a, b)


def _standardize_from_winsorized(xw: np.ndarray) -> np.ndarray:
    mu = float(xw.mean())
    sd = float(xw.std())
    return (xw - mu) / (sd + 1e-6)


def _mean_abs_grad(x: np.ndarray) -> float:
    dt = np.abs(x[1:, :] - x[:-1, :]).mean()
    df = np.abs(x[:, 1:] - x[:, :-1]).mean()
    return float(0.5 * (dt + df))


def _maxproj_freq_grad(x: np.ndarray) -> float:
    v = x.max(axis=0)
    return float(np.mean(np.abs(v[1:] - v[:-1])))


def _tv_after_freq_smooth(x: np.ndarray) -> float:
    xs = (x[:, :-2] + 2.0 * x[:, 1:-1] + x[:, 2:]) * 0.25
    dt = np.abs(xs[1:, :] - xs[:-1, :]).mean()
    df = np.abs(xs[:, 1:] - xs[:, :-1]).mean()
    return float(0.5 * (dt + df))


def _row_median_removed_freq_grad(x: np.ndarray) -> float:
    xr = x - np.median(x, axis=1, keepdims=True)
    xs = (xr[:, :-2] + 2.0 * xr[:, 1:-1] + xr[:, 2:]) * 0.25
    df = np.abs(xs[:, 1:] - xs[:, :-1]).mean()
    return float(df)


def _time_smooth_rowmed_freq_grad(x: np.ndarray) -> float:
    xt = (x[:-2, :] + 2.0 * x[1:-1, :] + x[2:, :]) * 0.25
    xr = xt - np.median(xt, axis=1, keepdims=True)
    xs = (xr[:, :-2] + 2.0 * xr[:, 1:-1] + xr[:, 2:]) * 0.25
    df = np.abs(xs[:, 1:] - xs[:, :-1]).mean()
    return float(df)


def _sliding_max_1d_along_freq(x2d: np.ndarray, k: int = 5) -> np.ndarray:
    T, F = x2d.shape
    if k <= 1:
        return x2d.copy()
    outF = F - k + 1
    if outF <= 0:
        return x2d.max(axis=1, keepdims=True)
    m = x2d[:, :outF].copy()
    for i in range(1, k):
        m = np.maximum(m, x2d[:, i : i + outF])
    return m


def _drift_template_score_from_panel(
    x: np.ndarray,
    slopes: np.ndarray,
    assume_centered: bool = True,
    apply_relu: bool = True,
    take_abs: bool = False,
    f0_step: int = 5,
) -> float:
    x = x.astype(np.float32, copy=False)

    if not assume_centered:
        x = x - np.median(x, axis=1, keepdims=True)

    if take_abs:
        x = np.abs(x)
        apply_relu = False

    if apply_relu:
        x = np.maximum(x, 0.0)

    xs = (x[:, :-2] + 2.0 * x[:, 1:-1] + x[:, 2:]) * 0.25  # (T, 254)
    m = _sliding_max_1d_along_freq(xs, k=5)  # (T, 250)

    T, F = m.shape
    t = np.arange(T, dtype=np.int32)

    best = 0.0
    f0_step = int(max(1, f0_step))
    f0_grid = np.arange(0, F, f0_step, dtype=np.int32)

    row_idx = np.arange(T, dtype=np.int32)[:, None]

    for s in slopes:
        if s == 0:
            val = float(m[:, f0_grid].sum(axis=0).max())
            if val > best:
                best = val
            continue

        if s > 0:
            max_f0 = F - 1 - s * (T - 1)
            if max_f0 < 0:
                continue
            f0s = f0_grid[f0_grid <= max_f0]
        else:
            min_f0 = -s * (T - 1)
            if min_f0 >= F:
                continue
            f0s = f0_grid[f0_grid >= min_f0]

        if f0s.size == 0:
            continue

        idx = (f0s[None, :] + s * t[:, None]).astype(np.int32)
        diag_sums = m[row_idx, idx].sum(axis=0)
        val = float(diag_sums.max())
        if val > best:
            best = val

    return float(best / (T + 1e-6))


def _drift_template_score(
    x_std: np.ndarray, x_wins: np.ndarray, slopes: np.ndarray = None
) -> float:
    if slopes is None:
        slopes = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)

    s_relu = _drift_template_score_from_panel(
        x_std, slopes, assume_centered=True, apply_relu=True, take_abs=False, f0_step=3
    )
    s_abs = _drift_template_score_from_panel(
        x_std, slopes, assume_centered=True, apply_relu=False, take_abs=True, f0_step=3
    )
    s_rm_relu = _drift_template_score_from_panel(
        x_std, slopes, assume_centered=False, apply_relu=True, take_abs=False, f0_step=3
    )
    s_rm_abs = _drift_template_score_from_panel(
        x_std, slopes, assume_centered=False, apply_relu=False, take_abs=True, f0_step=3
    )
    s_center = 0.5 * (s_relu + s_abs)
    s_rowmed = 0.5 * (s_rm_relu + s_rm_abs)
    s_std_score = float(0.6 * s_center + 0.4 * s_rowmed)

    sw_relu = _drift_template_score_from_panel(
        x_wins, slopes, assume_centered=True, apply_relu=True, take_abs=False, f0_step=3
    )
    sw_abs = _drift_template_score_from_panel(
        x_wins, slopes, assume_centered=True, apply_relu=False, take_abs=True, f0_step=3
    )
    sw_rm_relu = _drift_template_score_from_panel(
        x_wins,
        slopes,
        assume_centered=False,
        apply_relu=True,
        take_abs=False,
        f0_step=3,
    )
    sw_rm_abs = _drift_template_score_from_panel(
        x_wins,
        slopes,
        assume_centered=False,
        apply_relu=False,
        take_abs=True,
        f0_step=3,
    )
    sw_center = 0.5 * (sw_relu + sw_abs)
    sw_rowmed = 0.5 * (sw_rm_relu + sw_rm_abs)
    s_raw_score = float(0.6 * sw_center + 0.4 * sw_rowmed)

    return float(0.5 * s_std_score + 0.5 * s_raw_score)


def score_snippet(arr: np.ndarray) -> float:
    if arr.ndim != 3 or arr.shape[0] != 6:
        raise ValueError(f"Unexpected snippet shape {arr.shape}; expected (6,273,256)")

    a_idx = [0, 2, 4]
    b_idx = [1, 3, 5]

    f1 = np.empty(6, dtype=np.float32)
    f2 = np.empty(6, dtype=np.float32)
    f3 = np.empty(6, dtype=np.float32)
    f4 = np.empty(6, dtype=np.float32)
    f5 = np.empty(6, dtype=np.float32)
    f6 = np.empty(6, dtype=np.float32)

    for k in range(6):
        xw = _winsorize(arr[k], 1.0, 99.0)
        x = _standardize_from_winsorized(xw)

        f1[k] = _mean_abs_grad(x)
        f2[k] = _maxproj_freq_grad(x)
        f3[k] = _tv_after_freq_smooth(x)
        f4[k] = _row_median_removed_freq_grad(x)
        f5[k] = _time_smooth_rowmed_freq_grad(x)
        f6[k] = _drift_template_score(x, xw)

    a1, b1 = float(f1[a_idx].mean()), float(f1[b_idx].mean())
    a2, b2 = float(f2[a_idx].mean()), float(f2[b_idx].mean())
    a3, b3 = float(f3[a_idx].mean()), float(f3[b_idx].mean())
    a4, b4 = float(f4[a_idx].mean()), float(f4[b_idx].mean())
    a5, b5 = float(f5[a_idx].mean()), float(f5[b_idx].mean())
    a6, b6 = float(f6[a_idx].mean()), float(f6[b_idx].mean())

    a_disp = float(np.std(f4[a_idx]))
    b_disp = float(np.std(f4[b_idx]))
    persist_contrast = b_disp - a_disp

    return (
        (a1 - b1)
        + 0.30 * (a2 - b2)
        + 0.25 * (a3 - b3)
        + 0.20 * (a4 - b4)
        + 0.08 * (a5 - b5)
        + 0.10 * (a6 - b6)
        + 0.05 * persist_contrast
    )




## === cell 2
scores = np.empty(len(sample_ids), dtype=np.float32)

for j, sid in enumerate(sample_ids):
    path = id_to_path[sid]
    arr = np.load(path, mmap_mode="r")  # stored float16
    scores[j] = score_snippet(arr)

order = np.argsort(scores, kind="mergesort")
ranks = np.empty_like(order, dtype=np.float32)
ranks[order] = np.arange(len(scores), dtype=np.float32)

eps = 1.0 / (len(scores) + 2.0)
probs = (ranks + 1.0) / (len(scores) + 1.0)
probs = np.clip(probs, eps, 1.0 - eps).astype(np.float32)




## === cell 3
sub = pd.DataFrame({"id": sample_ids, "target": probs})
sub = sample[["id"]].merge(sub, on="id", how="left")
if sub["target"].isna().any():
    raise RuntimeError("Some predictions are missing after merge; check id alignment.")

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
