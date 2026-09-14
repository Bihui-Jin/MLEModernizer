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

0.7571658004342819

# 6. Current score

0.4968

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49758) has done: 'The timeout is dominated by per-file disk I/O and Python overhead: you load ~60k `.npy` files and also have an expensive fallback `glob` search for missing paths. I remove the `glob` fallback by deterministically building a fast id→path index once (using `os.scandir`) and reuse it for both train and test, which is equivalent but avoids thousands of filesystem scans. I also speed up feature extraction by eliminating `np.quantile` (slow) in favor of an equivalent percentile computation using `np.partition` on a flattened copy (same quantiles, much faster), and by using `np.einsum` for the second moment without allocating `x*x`. Finally, I keep the same model/training logic but make loading more efficient by reusing the thread pool pattern with an appropriate `chunksize` and avoiding repeated checks.'
- What this solution (achieved 0.49321) has done: 'Your current AUC is far below the target, so we should make a small, legitimate improvement without changing the overall approach (still: handcrafted features → standardized logistic regression). The biggest score lift with minimal risk is to use a stronger but still “same model” regularization setting and class balancing, plus a slightly more robust feature scaling choice for heavy-tailed features. Concretely, I keep your exact feature extractor and pipeline structure, but (1) enable `class_weight="balanced"` to reduce bias from class imbalance, and (2) tune `C` upward moderately to reduce underfitting, and (3) switch scaler to `RobustScaler` (still a linear scaling step) which is often better for these statistics-heavy features and typically improves AUC. These are minimal, metric-aligned changes that usually improve ROC-AUC while preserving the core logic and still producing a valid `submission.csv`.'
- What this solution (achieved 0.50371) has done: 'Your CV AUC is very low for this competition, so the smallest legitimate lift without changing the overall “handcrafted features → scaler → logistic regression” approach is to make the train/validation split respect the dataset’s known grouping by the first character of `id` (the folder name `0..f`), which reduces distribution mismatch and typically improves generalization AUC. I keep your feature extractor and model exactly the same, but replace `train_test_split(..., stratify=y)` with a `StratifiedGroupKFold` single-fold split using `group = id[0]`. I also keep the rest of the pipeline identical and still produce `submission.csv` in the same format. This is a minimal change aimed at improving score toward the target without altering core semantics.'
- What this solution (achieved 0.49784) has done: 'Main bottlenecks are (1) scanning the train/test directory trees with Python loops, (2) repeatedly loading tens of thousands of small `.npy` files with high per-file overhead, and (3) quantile computation doing extra dtype work. The changes below keep the exact same features and model, but reduce overhead by: using `glob` (C-optimized) for path discovery, using a `ProcessPoolExecutor` (NumPy workloads release/contend on GIL and benefit from multi-process parallelism), precomputing quantile indices once (same interpolation semantics), and removing redundant dtype conversions/copies. This preserves evaluation semantics and should bring runtime under 600s by substantially speeding up feature matrix construction (the dominant cost).'
- What this solution (achieved 0.49973) has done: 'Your current score is far below the target, so we should aim for a modest but real lift without changing the “handcrafted features → scaler → logistic regression” core logic. The highest-impact minimal fix here is to remove the distribution mismatch created by using `class_weight="balanced"` (AUC is threshold-free, and this reweighting often hurts probability ranking when the train/test priors match), and to make the logistic regression slightly more regularized to reduce noisy overfitting from heavy-tailed handcrafted stats. I also switch the solver to `lbfgs` (still plain L2 logistic regression) for more stable convergence on standardized dense features, keeping the rest of your pipeline identical. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.49889) has done: 'I make one minimal, metric-aligned change to improve ranking AUC without changing your feature extractor or the “scaler → L2 logistic regression” core approach: switch back from `RobustScaler` to `StandardScaler`, since your features (means/std/energies/quantiles) are already fairly well-behaved and logistic regression typically ranks better with z-scored inputs here. I also set `random_state` in `LogisticRegression` for deterministic behavior (no semantic change) and slightly reduce `C` to curb noisy overfitting that can hurt ROC-AUC on this competition (still the same model). Everything else (data loading, quantiles, multiprocessing, split logic, submission writing) stays identical and still produce a valid `submission.csv`. These are small, safe adjustments aimed at lifting you from ~0.50 AUC toward the 0.757 target band.'
- What this solution (achieved 0.4978) has done: 'Main runtime is dominated by repeatedly scanning the filesystem with `glob` and, more importantly, by loading every `.npy` and running `np.partition` for quantiles three separate times (CV train, CV valid, full train, then test). I keep the exact same feature definitions and model pipeline, but cache extracted features per-id so each file is loaded and processed at most once across the whole run. I also build the id→path map faster using `os.scandir` (same semantics as your `glob("*/*.npy")`), and I reuse the already-built training feature matrix for the “fit on all” step to avoid recomputation. These changes are provably equivalent (pure-function caching + equivalent directory traversal) and should cut total feature-extraction time by ~2–3×, which is typically what causes the 10-minute timeout.'
- What this solution (achieved 0.4968) has done: 'Your score is far below the target, so we make the smallest changes that typically improve ROC-AUC ranking without changing your overall “handcrafted stats → impute/scale → L2 logistic regression” approach. The main issue is underfitting/limited expressiveness from linear LR on these features; switching to `LogisticRegressionCV` (still logistic regression, same loss/architecture family) to pick `C` via internal CV usually gives a meaningful lift while preserving core semantics. We also expand the `C` search range modestly and keep everything else (feature extraction, caching, split-by-group validation, submission writing) identical. This should move you upward toward the 0.757 target without introducing new modeling components.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

BASE = "/kaggle/input"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
TRAIN_LABELS = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS), f"Missing: {TRAIN_LABELS}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS)
sample_sub = pd.read_csv(SAMPLE_SUB)

train_labels.head(), sample_sub.head()



## === cell 1
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


def _build_id_to_path_map(root_dir: str) -> dict:
    mp_ = {}
    with os.scandir(root_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            subdir = entry.path
            with os.scandir(subdir) as it2:
                for f in it2:
                    if f.is_file() and f.name.endswith(".npy"):
                        sid = f.name[:-4]  # strip ".npy"
                        mp_[sid] = f.path
    return mp_


TRAIN_ID2PATH = _build_id_to_path_map(TRAIN_DIR)
TEST_ID2PATH = _build_id_to_path_map(TEST_DIR)

_Q_PROBS = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)

_N_FLAT = 6 * 273 * 256
_h = (_N_FLAT - 1) * _Q_PROBS
_Q_LO = np.floor(_h).astype(np.int64)
_Q_HI = np.ceil(_h).astype(np.int64)
_Q_FRAC = (_h - _Q_LO).astype(np.float32)
_Q_KTH = np.unique(np.concatenate([_Q_LO, _Q_HI]))


def _fast_quantiles_flat_fixedn(x_flat_f32: np.ndarray) -> np.ndarray:
    x_part = np.partition(x_flat_f32, _Q_KTH)
    v_lo = x_part[_Q_LO]
    v_hi = x_part[_Q_HI]
    return (v_lo + (v_hi - v_lo) * _Q_FRAC).astype(np.float32, copy=False)


def extract_features(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)  # (6,273,256)

    ex = x.mean(axis=(1, 2), dtype=np.float32)  # (6,)

    ex2 = (
        np.einsum("ijk,ijk->i", x, x, optimize=True) / (x.shape[1] * x.shape[2])
    ).astype(np.float32, copy=False)
    var = np.maximum(ex2 - ex * ex, 0.0).astype(np.float32, copy=False)
    std = np.sqrt(var, dtype=np.float32)

    a_mean = ex[[0, 2, 4]].mean(dtype=np.float32)
    off_mean = ex[[1, 3, 5]].mean(dtype=np.float32)
    diff_mean = a_mean - off_mean
    ratio_mean = a_mean / (off_mean + 1e-6)

    a_std = std[[0, 2, 4]].mean(dtype=np.float32)
    off_std = std[[1, 3, 5]].mean(dtype=np.float32)
    diff_std = a_std - off_std
    ratio_std = a_std / (off_std + 1e-6)

    energy = ex2
    a_energy = energy[[0, 2, 4]].mean(dtype=np.float32)
    off_energy = energy[[1, 3, 5]].mean(dtype=np.float32)
    diff_energy = a_energy - off_energy
    ratio_energy = a_energy / (off_energy + 1e-6)

    q = _fast_quantiles_flat_fixedn(x.reshape(-1))

    eps = np.float32(1e-6)
    log_ratio_mean = (
        np.log(np.abs(a_mean) + eps) - np.log(np.abs(off_mean) + eps)
    ).astype(np.float32)
    log_ratio_std = (
        np.log(np.abs(a_std) + eps) - np.log(np.abs(off_std) + eps)
    ).astype(np.float32)
    log_ratio_energy = (
        np.log(np.abs(a_energy) + eps) - np.log(np.abs(off_energy) + eps)
    ).astype(np.float32)

    pooled_mean_scale = (np.abs(a_mean) + np.abs(off_mean) + eps).astype(np.float32)
    pooled_std_scale = (a_std + off_std + eps).astype(np.float32)
    pooled_energy_scale = (a_energy + off_energy + eps).astype(np.float32)

    norm_diff_mean = (diff_mean / pooled_mean_scale).astype(np.float32)
    norm_diff_std = (diff_std / pooled_std_scale).astype(np.float32)
    norm_diff_energy = (diff_energy / pooled_energy_scale).astype(np.float32)

    feats = np.concatenate(
        [
            ex.astype(np.float32, copy=False),  # 6
            std.astype(np.float32, copy=False),  # 6
            energy.astype(np.float32, copy=False),  # 6
            np.array(
                [diff_mean, ratio_mean, diff_std, ratio_std, diff_energy, ratio_energy],
                dtype=np.float32,
            ),  # 6
            np.array(
                [
                    log_ratio_mean,
                    log_ratio_std,
                    log_ratio_energy,
                    norm_diff_mean,
                    norm_diff_std,
                    norm_diff_energy,
                ],
                dtype=np.float32,
            ),  # 6
            q,  # 5
        ]
    )

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    return feats


_FEAT_CACHE_TRAIN = {}
_FEAT_CACHE_TEST = {}


def _load_and_extract_one(i_sid_path_cache):
    i, sid, path, cache = i_sid_path_cache
    feats = cache.get(sid)
    if feats is None:
        arr = np.load(path, mmap_mode="r")
        feats = extract_features(arr)
        cache[sid] = feats
    return i, feats


_tmp = np.load(TRAIN_ID2PATH[train_labels["id"].iloc[0]], mmap_mode="r")
_feat_len = int(len(extract_features(_tmp)))
_feat_len


def build_feature_matrix(ids, id2path: dict, cache: dict) -> np.ndarray:
    ids = list(ids)
    n = len(ids)

    X = np.empty((n, _feat_len), dtype=np.float32)

    triples = []
    for i, sid in enumerate(ids):
        p = id2path.get(sid)
        if p is None:
            raise FileNotFoundError(f"Could not find npy for id={sid}")
        triples.append((i, sid, p, cache))

    cpu = os.cpu_count() or 4
    max_workers = min(16, cpu * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_load_and_extract_one, triples, chunksize=128):
            X[i] = feats

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return X


assert _feat_len == (6 + 6 + 6 + 6 + 6 + 5), f"Feature length mismatch: got {_feat_len}"
_feat_len



## === cell 2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

train_ids = train_labels["id"].values
y = train_labels["target"].values.astype(np.int32)

groups = np.array([sid[0] for sid in train_ids], dtype="<U1")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(cv.split(train_ids, y, groups=groups))

tr_ids, va_ids = train_ids[tr_idx], train_ids[va_idx]
y_tr, y_va = y[tr_idx], y[va_idx]

X_tr = build_feature_matrix(tr_ids, TRAIN_ID2PATH, cache=_FEAT_CACHE_TRAIN)
X_va = build_feature_matrix(va_ids, TRAIN_ID2PATH, cache=_FEAT_CACHE_TRAIN)

clf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegressionCV(
                Cs=np.logspace(-2, 2, 9),  # modest range; still minimal and fast
                cv=3,
                scoring="roc_auc",
                solver="lbfgs",
                penalty="l2",
                max_iter=2000,
                class_weight=None,
                n_jobs=1,
                refit=True,
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)
va_pred = clf.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, va_pred)
auc



## === cell 3
X_all = build_feature_matrix(train_ids, TRAIN_ID2PATH, cache=_FEAT_CACHE_TRAIN)
clf.fit(X_all, y)

test_ids = sample_sub["id"].values
X_test = build_feature_matrix(test_ids, TEST_ID2PATH, cache=_FEAT_CACHE_TEST)
test_pred = clf.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": test_ids, "target": test_pred.astype(np.float32)})
submission["target"] = submission["target"].clip(0.0, 1.0)
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape
