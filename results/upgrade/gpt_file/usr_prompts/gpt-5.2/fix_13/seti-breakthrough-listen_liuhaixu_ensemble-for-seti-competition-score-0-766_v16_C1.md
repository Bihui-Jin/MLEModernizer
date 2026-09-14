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

0.7627427987309934

# 6. Current score

0.51227

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50095) has done: 'The timeout is dominated by per-file `.npy` loading and feature extraction over ~54k train + 6k test files; the current multiprocessing setup adds overhead and the ZIP extraction step can explode runtime. I (1) avoid any zip extraction and always read from the already-extracted folder structure, (2) speed up feature extraction by eliminating repeated allocations and by using NumPy’s built-in percentile on the already-flattened views (same quantiles, negligible FP diffs), and (3) reduce multiprocessing overhead by switching to a `ThreadPoolExecutor` (NumPy releases the GIL in heavy ops + I/O-bound loads) with larger batches and fewer cross-process serialization costs. The model/training (StandardScaler + LogisticRegression, CV loop, same data/targets) stays identical.'
- What this solution (achieved 0.5074) has done: 'Your current AUC (0.50095) is far below the target (0.76274), so we need a real signal improvement with minimal logic changes. The biggest likely issue is that the feature set is “too symmetric” (A-minus-off using means) and can collapse separation, so I’m adding a tiny, core-consistent extension: also compute the same summary stats for each individual A panel vs the off-mean and then aggregate (mean/std/max) across the three A panels—this preserves the original approach (handcrafted stats + LogisticRegression) but gives the linear model more discriminative information. I’m also switching `LogisticRegression` to `class_weight="balanced"` (still the same model) because this competition is imbalanced and it usually lifts ROC-AUC without changing evaluation semantics. Everything else (data loading, CV loop, scaler+LR pipeline, submission writing) stays the same.'
- What this solution (achieved 0.51093) has done: 'Your current AUC (0.5074) is far below the target (0.76274), so we need a genuine lift without changing the overall “handcrafted stats → scaler → logistic regression” approach. The most likely issue is that the features are still too “global” and miss the key competition cue: signals that are present in A panels but absent in B/C/D, often as narrowband or drifting tracks. I add a minimal, core-consistent set of additional summary features that capture *where* energy concentrates (frequency-band vs time-track) and *how* it differs between A and off panels, while keeping the same training loop and model type. I also fix a subtle threading scheduling issue (collecting futures in submission order rather than completion order) to ensure deterministic alignment and reduce the risk of accidental mis-ordering if changed later.'
- What this solution (achieved 0.51227) has done: 'Your current script likely didn’t yield a Kaggle score because it’s too slow for the 600s limit (full train featurization over 54k files) even though it writes `submission.csv`. To move toward the target AUC without changing your feature/model logic, I (1) stop doing any full-train OOF CV computation (it doesn’t affect the submission) and instead fit the same scaler+logreg on a capped subset of training files for speed, (2) keep the exact same feature extraction and LogisticRegression pipeline, and (3) ensure deterministic, correctly-aligned test predictions written to `submission.csv`. This should produce a valid submission within time and usually improves over near-random scores by enabling actual model training to complete.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
DATA_ROOT_CANDIDATES = [
    Path("../input/seti-breakthrough-listen"),
    Path("/kaggle/input/seti-breakthrough-listen"),
    Path("../input"),  # fallback if data is directly under ../input
    Path("/kaggle/input"),
]


def find_data_root():
    for root in DATA_ROOT_CANDIDATES:
        if (
            (root / "train_labels.csv").exists()
            and (root / "train").exists()
            and (root / "test").exists()
        ):
            return root
        if (root / "seti-breakthrough-listen" / "train_labels.csv").exists():
            return root / "seti-breakthrough-listen"
    raise FileNotFoundError(
        "Could not locate dataset root containing train_labels.csv, train/, test/"
    )


DATA_ROOT = find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
LABELS_PATH = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR exists:", TRAIN_DIR.exists(), "TEST_DIR exists:", TEST_DIR.exists())
print(
    "LABELS_PATH exists:",
    LABELS_PATH.exists(),
    "SAMPLE_SUB_PATH exists:",
    SAMPLE_SUB_PATH.exists(),
)

labels = pd.read_csv(LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(labels.head())
print(sample_sub.head())




## === cell 2
def id_to_path(folder: Path, id_: str) -> Path:
    s = str(id_)
    return folder / s[0] / f"{s}.npy"


def _mean_pixelwise_std3(A0, A1, A2) -> np.float32:
    A0 = A0.astype(np.float32, copy=False)
    A1 = A1.astype(np.float32, copy=False)
    A2 = A2.astype(np.float32, copy=False)
    m = (A0 + A1 + A2) * (1.0 / 3.0)
    m2 = (A0 * A0 + A1 * A1 + A2 * A2) * (1.0 / 3.0)
    var = m2 - m * m
    np.maximum(var, 0.0, out=var)
    return np.sqrt(var, dtype=np.float32).mean(dtype=np.float32).astype(np.float32)


def _panel_feats_from_diff(diff2d: np.ndarray) -> np.ndarray:
    """
    Core-consistent summary stats on a single (273,256) diff map.
    """
    diff2d = diff2d.astype(np.float32, copy=False)
    adiff2d = np.abs(diff2d)

    diff_flat = diff2d.reshape(-1)
    adiff_flat = adiff2d.reshape(-1)

    diff_mean = diff_flat.mean(dtype=np.float32)
    diff_std = diff_flat.std(dtype=np.float32)
    diff_med, diff_p90, diff_p10 = np.percentile(diff_flat, [50.0, 90.0, 10.0]).astype(
        np.float32, copy=False
    )

    adiff_mean = adiff_flat.mean(dtype=np.float32)
    adiff_std = adiff_flat.std(dtype=np.float32)
    adiff_med, adiff_p95, adiff_p99 = np.percentile(
        adiff_flat, [50.0, 95.0, 99.0]
    ).astype(np.float32, copy=False)

    row_mean = diff2d.mean(axis=1, dtype=np.float32)  # (273,)
    col_mean = diff2d.mean(axis=0, dtype=np.float32)  # (256,)

    row_std = row_mean.std(dtype=np.float32)
    row_p95, row_p5 = np.percentile(row_mean, [95.0, 5.0]).astype(
        np.float32, copy=False
    )
    col_std = col_mean.std(dtype=np.float32)
    col_p95, col_p5 = np.percentile(col_mean, [95.0, 5.0]).astype(
        np.float32, copy=False
    )

    return np.asarray(
        [
            diff_mean,
            diff_std,
            diff_med,
            diff_p90,
            diff_p10,
            adiff_mean,
            adiff_std,
            adiff_med,
            adiff_p95,
            adiff_p99,
            row_std,
            (row_p95 - row_p5).astype(np.float32),
            col_std,
            (col_p95 - col_p5).astype(np.float32),
        ],
        dtype=np.float32,
    )


def _axis_concentration_feats(panel2d: np.ndarray) -> np.ndarray:
    """
    Summary features for frequency/time concentration (narrowband vs spread).
    """
    panel2d = panel2d.astype(np.float32, copy=False)
    a = np.abs(panel2d)

    freq_prof = a.mean(axis=0, dtype=np.float32)  # (256,)
    time_prof = a.mean(axis=1, dtype=np.float32)  # (273,)

    eps = np.float32(1e-6)

    f_mean = freq_prof.mean(dtype=np.float32)
    t_mean = time_prof.mean(dtype=np.float32)

    f_max = freq_prof.max().astype(np.float32)
    t_max = time_prof.max().astype(np.float32)

    f_p95, f_p50 = np.percentile(freq_prof, [95.0, 50.0]).astype(np.float32, copy=False)
    t_p95, t_p50 = np.percentile(time_prof, [95.0, 50.0]).astype(np.float32, copy=False)

    f_std = freq_prof.std(dtype=np.float32)
    t_std = time_prof.std(dtype=np.float32)

    return np.asarray(
        [
            (f_max / (f_mean + eps)).astype(np.float32),
            (t_max / (t_mean + eps)).astype(np.float32),
            (f_p95 / (f_p50 + eps)).astype(np.float32),
            (t_p95 / (t_p50 + eps)).astype(np.float32),
            (f_std / (f_mean + eps)).astype(np.float32),
            (t_std / (t_mean + eps)).astype(np.float32),
        ],
        dtype=np.float32,
    )


def _track_gradient_feats(panel2d: np.ndarray) -> np.ndarray:
    """
    Add minimal "track-likeness" features via gradient energies.
    """
    p = panel2d.astype(np.float32, copy=False)
    a = np.abs(p)

    dt = np.diff(a, axis=0)  # (272,256)
    df = np.diff(a, axis=1)  # (273,255)

    adt = np.abs(dt)
    adf = np.abs(df)

    eps = np.float32(1e-6)

    dt_mean = adt.mean(dtype=np.float32)
    df_mean = adf.mean(dtype=np.float32)
    dt_p95 = np.percentile(adt.reshape(-1), 95.0).astype(np.float32, copy=False)
    df_p95 = np.percentile(adf.reshape(-1), 95.0).astype(np.float32, copy=False)

    m = min(dt.shape[0], df.shape[0])  # 272
    n = min(dt.shape[1], df.shape[1])  # 255
    diag_energy = (adt[:m, :n] * adf[:m, :n]).mean(dtype=np.float32)

    base = a.mean(dtype=np.float32) + eps

    return np.asarray(
        [
            (dt_mean / base).astype(np.float32),
            (df_mean / base).astype(np.float32),
            (dt_p95 / base).astype(np.float32),
            (df_p95 / base).astype(np.float32),
            (diag_energy / (base * base)).astype(np.float32),
        ],
        dtype=np.float32,
    )


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A0, B, A1, C, A2, D = x[0], x[1], x[2], x[3], x[4], x[5]

    A_mean = (A0 + A1 + A2) * (1.0 / 3.0)
    off_mean = (B + C + D) * (1.0 / 3.0)

    diff = A_mean - off_mean

    base_feats = _panel_feats_from_diff(diff)

    diff_A0 = A0 - off_mean
    diff_A1 = A1 - off_mean
    diff_A2 = A2 - off_mean

    f0 = _panel_feats_from_diff(diff_A0)
    f1 = _panel_feats_from_diff(diff_A1)
    f2 = _panel_feats_from_diff(diff_A2)

    F = np.stack([f0, f1, f2], axis=0)  # (3, 14)
    perA_mean = F.mean(axis=0, dtype=np.float32)
    perA_std = F.std(axis=0, dtype=np.float32)
    perA_max = F.max(axis=0)

    ax_A = _axis_concentration_feats(A_mean)
    ax_off = _axis_concentration_feats(off_mean)
    ax_diff = _axis_concentration_feats(diff)

    ax0 = _axis_concentration_feats(diff_A0)
    ax1 = _axis_concentration_feats(diff_A1)
    ax2 = _axis_concentration_feats(diff_A2)
    AX = np.stack([ax0, ax1, ax2], axis=0)  # (3, 6)
    ax_perA_mean = AX.mean(axis=0, dtype=np.float32)
    ax_perA_std = AX.std(axis=0, dtype=np.float32)
    ax_perA_max = AX.max(axis=0)

    tg_A = _track_gradient_feats(A_mean)
    tg_off = _track_gradient_feats(off_mean)
    tg_diff = _track_gradient_feats(diff)

    tg0 = _track_gradient_feats(diff_A0)
    tg1 = _track_gradient_feats(diff_A1)
    tg2 = _track_gradient_feats(diff_A2)
    TG = np.stack([tg0, tg1, tg2], axis=0)  # (3, 5)
    tg_perA_mean = TG.mean(axis=0, dtype=np.float32)
    tg_perA_std = TG.std(axis=0, dtype=np.float32)
    tg_perA_max = TG.max(axis=0)

    A_stack_mean = (
        (
            A0.mean(dtype=np.float32)
            + A1.mean(dtype=np.float32)
            + A2.mean(dtype=np.float32)
        )
        * (1.0 / 3.0)
    ).astype(np.float32)
    off_stack_mean = (
        (B.mean(dtype=np.float32) + C.mean(dtype=np.float32) + D.mean(dtype=np.float32))
        * (1.0 / 3.0)
    ).astype(np.float32)

    def _pooled_std3(X1, X2, X3) -> np.float32:
        m1 = X1.mean(dtype=np.float32)
        m2 = X2.mean(dtype=np.float32)
        m3 = X3.mean(dtype=np.float32)
        v1 = X1.var(dtype=np.float32)
        v2 = X2.var(dtype=np.float32)
        v3 = X3.var(dtype=np.float32)
        n1 = np.float32(X1.size)
        n2 = np.float32(X2.size)
        n3 = np.float32(X3.size)
        N = n1 + n2 + n3
        m = (n1 * m1 + n2 * m2 + n3 * m3) / N
        var = (
            n1 * (v1 + (m1 - m) * (m1 - m))
            + n2 * (v2 + (m2 - m) * (m2 - m))
            + n3 * (v3 + (m3 - m) * (m3 - m))
        ) / N
        return np.sqrt(var, dtype=np.float32).astype(np.float32)

    A_stack_std = _pooled_std3(A0, A1, A2)
    off_stack_std = _pooled_std3(B, C, D)

    A_temporal_var = _mean_pixelwise_std3(A0, A1, A2)
    off_temporal_var = _mean_pixelwise_std3(B, C, D)

    tail_feats = np.asarray(
        [
            A_stack_mean,
            A_stack_std,
            off_stack_mean,
            off_stack_std,
            A_temporal_var.astype(np.float32),
            off_temporal_var.astype(np.float32),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [
            base_feats,
            perA_mean,
            perA_std,
            perA_max,
            ax_A,
            ax_off,
            ax_diff,
            ax_perA_mean,
            ax_perA_std,
            ax_perA_max,
            tg_A,
            tg_off,
            tg_diff,
            tg_perA_mean,
            tg_perA_std,
            tg_perA_max,
            tail_feats,
        ]
    ).astype(np.float32, copy=False)
    return feats


def load_npy_by_path(p: Path) -> np.ndarray:
    return np.load(p, mmap_mode="r")


tmp_id = labels["id"].iloc[0]
tmp_x = load_npy_by_path(id_to_path(TRAIN_DIR, str(tmp_id)))
tmp_f = extract_features_from_array(tmp_x)
print("Array shape:", tmp_x.shape, "Feature dim:", tmp_f.shape)



## === cell 3
TRAIN_DIR_FAST = TRAIN_DIR
TEST_DIR_FAST = TEST_DIR
print("TRAIN_DIR_FAST:", TRAIN_DIR_FAST, "exists:", TRAIN_DIR_FAST.exists())
print("TEST_DIR_FAST :", TEST_DIR_FAST, "exists:", TEST_DIR_FAST.exists())



## === cell 4
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

feat_dim = tmp_f.shape[0]
ids_all = labels["id"].astype(str).values
y_all = labels["target"].astype(int).values

MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "20000"))
MAX_TRAIN_SAMPLES = min(MAX_TRAIN_SAMPLES, len(ids_all))

rng = np.random.RandomState(42)
pos_idx = np.where(y_all == 1)[0]
neg_idx = np.where(y_all == 0)[0]

n_pos = min(len(pos_idx), max(2000, MAX_TRAIN_SAMPLES // 4))
n_neg = min(len(neg_idx), MAX_TRAIN_SAMPLES - n_pos)

sel_pos = (
    rng.choice(pos_idx, size=n_pos, replace=False) if n_pos < len(pos_idx) else pos_idx
)
sel_neg = (
    rng.choice(neg_idx, size=n_neg, replace=False) if n_neg < len(neg_idx) else neg_idx
)
sel_idx = np.concatenate([sel_pos, sel_neg])
rng.shuffle(sel_idx)

ids_train = ids_all[sel_idx]
y = y_all[sel_idx]

print(
    f"Training on subset: {len(ids_train)}/{len(ids_all)} (pos={int(y.sum())}, neg={int((1-y).sum())})"
)


def _iter_batches_with_pos(arr, batch_size: int):
    n = len(arr)
    for i in range(0, n, batch_size):
        yield i, arr[i : i + batch_size]


def _featurize_id_batch(folder_str: str, id_list, feat_dim_local: int):
    out = np.empty((len(id_list), feat_dim_local), dtype=np.float32)
    for i, id_str in enumerate(id_list):
        p = f"{folder_str}/{id_str[0]}/{id_str}.npy"
        x = np.load(p, mmap_mode="r")
        out[i] = extract_features_from_array(x)
    return out


X = np.empty((len(ids_train), feat_dim), dtype=np.float32)

cpu = os.cpu_count() or 2
max_workers = min(8, cpu)  # moderate cap avoids disk thrash
batch_size = 1024  # smaller batch reduces per-future memory spikes

folder_train_str = str(TRAIN_DIR_FAST)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = []
    for start_pos, batch in _iter_batches_with_pos(ids_train, batch_size):
        fut = ex.submit(_featurize_id_batch, folder_train_str, batch, feat_dim)
        fut.start_pos = start_pos
        futures.append(fut)

    done = 0
    for fut in as_completed(futures):
        feats = fut.result()
        start_pos = fut.start_pos
        X[start_pos : start_pos + feats.shape[0]] = feats
        done += feats.shape[0]
        if done % 5000 == 0 or done == len(ids_train):
            print(
                f"Extracted ~{done}/{len(ids_train)} train samples (threads={max_workers}, batch={batch_size})"
            )

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=2000, solver="lbfgs", n_jobs=None, class_weight="balanced"
            ),
        ),
    ]
)

model.fit(X, y)
print("Finished fitting model on subset.")



## === cell 5
test_ids = sample_sub["id"].astype(str).values
n_test = len(test_ids)
Xt = np.empty((n_test, feat_dim), dtype=np.float32)

folder_test_str = str(TEST_DIR_FAST)


def _featurize_test_id_batch(folder_str: str, id_list, feat_dim_local: int):
    out = np.empty((len(id_list), feat_dim_local), dtype=np.float32)
    for i, id_str in enumerate(id_list):
        p = f"{folder_str}/{id_str[0]}/{id_str}.npy"
        if not os.path.exists(p):
            raise FileNotFoundError(f"Test id {id_str} not found under {folder_str}")
        x = np.load(p, mmap_mode="r")
        out[i] = extract_features_from_array(x)
    return out


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = []
    for start_pos, batch in _iter_batches_with_pos(test_ids, batch_size):
        fut = ex.submit(_featurize_test_id_batch, folder_test_str, batch, feat_dim)
        fut.start_pos = start_pos
        futures.append(fut)

    done = 0
    for fut in as_completed(futures):
        feats = fut.result()
        start_pos = fut.start_pos
        Xt[start_pos : start_pos + feats.shape[0]] = feats
        done += feats.shape[0]
        if done % 1000 == 0 or done == n_test:
            print(
                f"Extracted ~{done}/{n_test} test samples (threads={max_workers}, batch={batch_size})"
            )

pred = model.predict_proba(Xt)[:, 1].astype(np.float32)

pred = np.clip(pred, np.float32(1e-6), np.float32(1.0 - 1e-6))

submission = pd.DataFrame({"id": test_ids, "target": pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("target min/mean/max:", float(pred.min()), float(pred.mean()), float(pred.max()))
