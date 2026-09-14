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

0.755708240220353

# 6. Current score

0.51904

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49604) has done: 'The timeout is dominated by Python-level per-file overhead: recursive `glob` over all `.npy` files and then tens of thousands of small `np.load` calls with repeated multi-pass reductions inside `extract_features_from_arr`. I keep the exact same feature set and model pipeline, but make the file lookup O(1) without globbing by deriving paths directly from the known folder structure, and I remove redundant array passes by computing means/stds with `float32` in one go while keeping outputs numerically equivalent. I also add a small, safe on-disk cache of extracted feature matrices (keyed by the ids list) so reruns don’t re-extract, and I use a thread pool for I/O-bound loading while preserving deterministic ordering. These changes reduce wall time substantially without changing the algorithm, features, or training semantics.'
- What this solution (achieved 0.50821) has done: 'Your score is far below the target, so we should improve model signal while keeping your overall pipeline (feature extraction + LogisticRegression) intact. The biggest likely issue is that the current features are too global and don’t encode the key competition clue: “needle” appears in A panels (0/2/4) but not in B panels (1/3/5), often as a drifting narrowband line. I keep your same training loop and model, but add a small set of extra *A-vs-B contrast* and *line-likeness* scalar features (using simple reductions) that align better with the ROC-AUC objective without changing the approach. I also make `build_id_to_path_index` return `None` (so we never accidentally trigger a huge glob index build) and make `np.load` explicitly `allow_pickle=False` for safety/determinism.'
- What this solution (achieved 0.50652) has done: 'You’re currently far below the target AUC, so the smallest likely win is to make the existing LogisticRegression better match the metric by improving generalization and class-imbalance handling without changing the overall pipeline (same features → same LR). I keep your feature extractor and training loop intact, but (1) set `class_weight="balanced"` to reduce bias toward the majority class and usually lift ROC-AUC, and (2) slightly strengthen regularization (`C`) to reduce overfitting to noisy handcrafted features. I also add a local ROC-AUC computation on the held-out split so you can sanity-check directionality before submitting; this doesn’t affect the submission. Everything still runs end-to-end and writes `submission.csv` with the required schema.'
- What this solution (achieved 0.50881) has done: 'Your current AUC (0.5065) is far below the target (0.7557), so we need a modest, safe boost while keeping the same overall pipeline (handcrafted features → impute/scale → LogisticRegression). The smallest high-impact fix is to correct a likely feature bug: `time_profile_*` currently averages over both A panels and frequency, producing a 1D “time profile” of length 273 incorrectly computed (it collapses the wrong axis); fixing it preserves intent but increases signal. I also add two very cheap, metric-aligned features that capture the core competition cue (A vs B) with minimal change: the mean absolute A–B difference and a standardized A–B contrast; these don’t alter the modeling approach and typically lift ROC-AUC. Everything else (paths, caching, training loop, model type) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.51904) has done: 'Your current score (0.50881) is far below the target (0.7557), so we should make a small, low-risk improvement that keeps your pipeline (handcrafted reductions → impute/scale → LogisticRegression) intact but adds a bit more discriminative signal aligned with the competition cue (signal present in A panels but not B panels). The minimal change is to add a few extra A-vs-B *quantile* contrast features (median and upper-tail differences) which are robust to noise and capture “bright line” injections better than means alone, without changing the model type or training approach. I also stratify the train/val split by a stable hash of `id` (still a simple holdout split) to reduce split variance and make the learned calibration more consistent, which typically improves public AUC without altering evaluation semantics. Everything else (paths, caching, threading, submission writing) stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

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
def build_id_to_path_index(root_dir: str):
    return None


def find_npy_path_from_index(id_to_path, file_id: str, root_dir: str) -> str:
    p = os.path.join(root_dir, file_id[0], f"{file_id}.npy")
    if os.path.exists(p):
        return p

    cached = id_to_path.get(file_id) if isinstance(id_to_path, dict) else None
    if cached is not None and os.path.exists(cached):
        return cached

    pattern = os.path.join(root_dir, "**", f"{file_id}.npy")
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise FileNotFoundError(f"Could not find {file_id}.npy under {root_dir}")
    p2 = min(hits, key=len)
    if isinstance(id_to_path, dict):
        id_to_path[file_id] = p2
    return p2


def extract_features_from_arr(x: np.ndarray) -> np.ndarray:
    """
    Feature extractor:
    - Keeps original global summary features (means/std/min/max per panel + global A/B stats).
    - Keeps existing A-vs-B contrast + line-likeness features.

    Minimal score-improving change toward target:
    - Add a few robust A-vs-B *quantile* contrast features (median and upper-tail differences).
      These are simple reductions (no new modeling approach) and often capture narrow bright
      injected signals better than means, improving ROC-AUC while keeping the same pipeline.
    """
    x = x.astype(np.float32, copy=False)

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
    eps = np.float32(1e-6)
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

    A_time_maxfreq_mean = A.max(axis=2).mean()
    B_time_maxfreq_mean = B.max(axis=2).mean()
    A_time_maxfreq_std = A.max(axis=2).std()
    B_time_maxfreq_std = B.max(axis=2).std()

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
    q50_A, q90_A, q99_A = np.quantile(A_flat, [0.5, 0.9, 0.99])
    q50_B, q90_B, q99_B = np.quantile(B_flat, [0.5, 0.9, 0.99])

    k_tail = max(1, int(0.01 * A_flat.size))
    A_top_tail_mean = np.partition(A_flat, -k_tail)[-k_tail:].mean()
    B_top_tail_mean = np.partition(B_flat, -k_tail)[-k_tail:].mean()

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
        ],
        dtype=np.float32,
    )

    return np.concatenate([base_feats, extra_feats])


from concurrent.futures import ThreadPoolExecutor


def _ids_cache_key(root_dir: str, ids: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(root_dir.encode("utf-8"))
    joined = "\n".join(ids.tolist()).encode("utf-8")
    h.update(joined)
    return h.hexdigest()


def load_features_for_ids(root_dir: str, ids: np.ndarray, id_to_path) -> np.ndarray:
    n = len(ids)
    cache_dir = os.path.join("/kaggle/working", "feat_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"X_{_ids_cache_key(root_dir, ids)}.npy")

    if os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        return np.asarray(X, dtype=np.float32)

    first_path = find_npy_path_from_index(id_to_path, ids[0], root_dir)
    first_arr = np.load(first_path, mmap_mode="r", allow_pickle=False)
    first_feat = extract_features_from_arr(first_arr)
    d = first_feat.shape[0]

    X = np.empty((n, d), dtype=np.float32)
    X[0] = first_feat

    def _worker(i: int):
        file_id = ids[i]
        path = find_npy_path_from_index(id_to_path, file_id, root_dir)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return i, extract_features_from_arr(arr)

    max_workers = min(8, (os.cpu_count() or 2))
    if n > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, feat in ex.map(_worker, range(1, n), chunksize=64):
                X[i] = feat

    np.save(cache_path, X)
    return X


train_id_to_path = build_id_to_path_index(TRAIN_DIR)
test_id_to_path = build_id_to_path_index(TEST_DIR)

one_id = train_labels["id"].iloc[0]
one_path = find_npy_path_from_index(train_id_to_path, one_id, TRAIN_DIR)
one_arr = np.load(one_path, mmap_mode="r", allow_pickle=False)
(one_id, one_arr.shape, one_arr.dtype, extract_features_from_arr(one_arr).shape)



## === cell 2
from sklearn.metrics import roc_auc_score

train_ids = train_labels["id"].values
y = train_labels["target"].values.astype(np.int64)

X = load_features_for_ids(TRAIN_DIR, train_ids, train_id_to_path)


def _stable_split_mask(ids: np.ndarray, val_frac: float = 0.2) -> np.ndarray:
    thr = int(val_frac * (2**32 - 1))
    out = np.zeros(len(ids), dtype=bool)
    for i, s in enumerate(ids):
        h = hashlib.md5(str(s).encode("utf-8")).digest()
        v = int.from_bytes(h[:4], "little", signed=False)
        out[i] = v <= thr
    return out


val_mask = _stable_split_mask(train_ids, val_frac=0.2)

if val_mask.sum() < 100 or (~val_mask).sum() < 100:
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
                C=0.5,
                solver="lbfgs",
                max_iter=2000,
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
X_test = load_features_for_ids(TEST_DIR, test_ids, test_id_to_path)

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
