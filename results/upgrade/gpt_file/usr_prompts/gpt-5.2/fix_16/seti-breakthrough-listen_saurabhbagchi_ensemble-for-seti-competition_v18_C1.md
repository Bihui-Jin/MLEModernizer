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

0.7556746902410042

# 6. Current score

0.49659

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49845) has done: 'The timeout is dominated by feature extraction repeatedly spawning process pools and doing many small `.npy` reads, plus training the same model twice. I keep the exact same features and logistic-regression pipeline, but make feature building faster by (1) using a thread pool (NumPy `np.load(..., mmap_mode="r")` is largely I/O-bound here and threads avoid heavy process startup/pickling), (2) using larger contiguous chunks and deterministic in-order collection to reduce overhead, and (3) vectorizing the “contrast” statistics without flattening. I also preserve the training semantics by fitting on the full data once (the second `fit(X,y)` overwrites the first, so removing the first does not change the final model used for predictions), while still computing the split for parity. All paths and core logic (features, model, loss/solver) remain unchanged.'
- What this solution (achieved 0.49321) has done: 'Your score (0.49845) is far below the target (0.75567), so we need a small but meaningful improvement without changing the overall approach (handcrafted statistics + logistic regression). The biggest likely issue is that the current features are too weak (simple per-panel stats + global A-vs-B contrast), so we minimally enrich the same feature family by adding per-cadence A/B-group statistics (means/stds/max/min on the aggregated A and aggregated B), plus a couple of stable “difference-of-stats” terms; this preserves the same feature-extraction paradigm and the same LR pipeline. We also set `class_weight="balanced"` (a single LR hyperparameter) to handle class imbalance, which often materially improves ROC-AUC for this competition while keeping the same model/solver/loss. Everything else (paths, reading, train/test handling, submission formatting) stays the same.'
- What this solution (achieved 0.50285) has done: 'Your current AUC is far below the target, so we should make a small, legitimate feature-strengthening change while keeping the same “handcrafted stats → StandardScaler → LogisticRegression(lbfgs)” core. The biggest low-risk gain is to add a few more stable statistics that capture the *structure* of A vs B panels (not just global mean/std), namely robust quantiles and simple row/column aggregation summaries on the A-mean, B-mean, and (A−B) images. This preserves the same overall approach (no new model family, no new training loop), but gives LR more signal to separate needles from RFI patterns and typically improves ROC-AUC materially on SETI. We keep all paths, submission formatting, and training-on-full-data semantics unchanged, only updating the feature vector size accordingly.'
- What this solution (achieved 0.49876) has done: 'Your current AUC (0.50285) is far below the target (0.75567), so we should improve signal extraction while preserving the same overall “handcrafted stats → StandardScaler → LogisticRegression(lbfgs)” pipeline. The smallest high-impact change here is to add a few more **structure-sensitive** but still simple summary features: (1) per-panel “center-of-mass” along time/frequency (captures drifting/line-like energy), and (2) a low-frequency projection via small fixed-bucket pooling on the A-mean, B-mean, and (A−B) images (captures coarse morphology without changing model family). These are deterministic, cheap to compute, and keep the same training semantics and evaluation behavior, while typically improving separability for SETI. All paths, submission formatting, and the LR solver/loss remain unchanged; only the feature vector grows.'
- What this solution (achieved 0.49659) has done: 'The main timeout driver is feature extraction: it loads ~60k `.npy` files and runs several expensive operations per file (notably `np.quantile`, repeated `linspace`, and a Python loop for pooling). I keep the exact same features and model, but remove redundant per-file allocations, replace the slow pooling loop with an equivalent vectorized `reduceat` mean, compute correlations with a cheaper but mathematically equivalent formula, and use `np.partition` to compute the same quantiles much faster. I also avoid building full chunk lists in memory and tune threading/chunking to reduce overhead while still overlapping disk I/O. These changes preserve evaluation semantics and should cut runtime substantially without changing the algorithm.'
- What this solution (achieved 0.49659) has done: 'Your current AUC (0.49659) is far below the target (0.75567), so we need a meaningful but still “same core logic” improvement: keep the exact same handcrafted feature families and the same StandardScaler→LogisticRegression pipeline, but fix a key bug in the “robust quantiles” computation that currently uses the wrong indices (it treats quantile levels as absolute indices rather than positions within the flattened array). This mistake makes the quantile features almost constant and uninformative, hurting AUC; correcting it should move you substantially toward the target without changing the model family, training procedure, or evaluation semantics. I also add a tiny safety guard for partition indices and keep feature dimensionality identical (still 276) so everything else stays stable. The submission format/paths remain unchanged and the script still runs end-to-end within the same constraints.'
- What this solution (achieved 0.49659) has done: 'Your current score (0.49659) is far below the target (0.75567), so we need a meaningful lift while keeping the same overall “handcrafted stats → StandardScaler → LogisticRegression(lbfgs)” approach. The smallest high-impact fix is to correct the robust-quantile feature computation: `np.partition(x, idxs)` does not guarantee global ordering across multiple indices, so `part[idxs]` is not equal to the true quantiles and becomes noisy/uninformative; switching to a single `np.partition` at `max_k` and then sorting only the first `max_k+1` elements yields exact order-statistics at our desired indices with minimal added cost. This keeps the exact same feature family and dimensionality (still 276), preserves training semantics, and should materially improve ROC-AUC toward the target. Everything else (paths, model, submission format, runtime constraints) remains unchanged.'
- What this solution (achieved 0.49659) has done: 'Your current AUC is far below the target, and the biggest “minimal-change” lever that should materially move you toward 0.755 is correcting the quantile features: the current `_robust_quantiles` still returns incorrect values for multiple quantile indices because sorting only the first `max_k+1` partitioned elements does not yield global order statistics at arbitrary indices. I replace it with an exact, deterministic multi-quantile selection using `np.partition` at all required indices (still fast, but now correct), keeping the same feature family and the same feature dimensionality (276) so the rest of your pipeline stays identical. Everything else (paths, thread-based feature building, StandardScaler→LogisticRegression(lbfgs), submission format) remains unchanged.'
- What this solution (achieved 0.49659) has done: 'Your current AUC (0.49659) is far below the target (0.75567), so we need a real lift while keeping the same overall “handcrafted stats → StandardScaler → LogisticRegression(lbfgs)” pipeline. The largest likely culprit in your current code is that the “robust quantiles” are still not true quantiles: `np.partition(x, uniq)[idxs]` is not guaranteed to return the global k-th order statistic for each k when multiple k’s are requested, so those features become noisy and hurt AUC. I replace `_robust_quantiles` with an exact (but still fast) multi-k selection: partition once at `max_k`, then fully sort only the prefix up to `max_k`, and index into that sorted prefix; this yields correct order-statistics for all requested indices with limited overhead. Everything else (feature families, feature count=276, model, paths, submission formatting) remains unchanged.'
- What this solution (achieved 0.49659) has done: 'Your current AUC (0.49659) is far below the target (0.75567), so we should make a small, legitimate improvement that keeps the exact same overall pipeline: handcrafted stats → StandardScaler → LogisticRegression(lbfgs). The most likely “score-killer” left is the quantile feature extractor: it still does not compute true quantiles reliably, making 15 key features effectively noisy and hurting separability. I replace `_robust_quantiles` with a correct, deterministic multi-quantile computation (still using `np.partition`, but now guaranteed correct), keeping the feature count (276), model, paths, and training semantics unchanged. This should materially increase ROC-AUC toward the target without changing the core approach.'
- What this solution (achieved 0.49659) has done: 'Your AUC is far below the target, so we need a real signal lift while keeping the same “handcrafted stats → StandardScaler → LogisticRegression(lbfgs)” core. The biggest likely score-killer left is that `_robust_quantiles` is still not computing true quantiles (multi-`k` partition doesn’t guarantee each `k` is the global k-th order statistic), which makes 15 important features noisy and near-useless. I replace it with an exact, deterministic multi-quantile implementation using a full sort (fast enough at 69,888 elements) so those features become informative, without changing the model, feature families, or feature dimensionality (still 276). Everything else (paths, threading, submission formatting) stays the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import concurrent.futures as cf

RANDOM_STATE = 42

DATA_ROOT = "/kaggle/data"  # as provided in the prompt file tree
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

OUT_SUB_PATH = "submission.csv"

np.random.seed(RANDOM_STATE)




## === cell 1
def list_npy_ids(base_dir: str):
    out_paths = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".npy"):
                out_paths.append(os.path.join(root, fn))
    out_paths.sort()
    ids = [os.path.splitext(os.path.basename(p))[0] for p in out_paths]
    return ids, out_paths


train_ids_from_disk, train_paths = list_npy_ids(TRAIN_DIR)
test_ids_from_disk, test_paths = list_npy_ids(TEST_DIR)

if len(train_paths) == 0 or len(test_paths) == 0:
    raise FileNotFoundError(
        f"Could not find .npy files. train_paths={len(train_paths)}, test_paths={len(test_paths)}. "
        f"Expected under {TRAIN_DIR} and {TEST_DIR}."
    )

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

train_id_set = set(train_ids_from_disk)
train_labels = train_labels[train_labels["id"].isin(train_id_set)].copy()
train_labels = train_labels.sort_values("id").reset_index(drop=True)

test_id_set = set(test_ids_from_disk)
missing_test = set(sample_sub["id"]) - test_id_set
if len(missing_test) > 0:
    sample_sub = sample_sub[sample_sub["id"].isin(test_id_set)].copy()
sample_sub = sample_sub.sort_values("id").reset_index(drop=True)

train_id_to_path = pd.Series(
    train_paths, index=[os.path.splitext(os.path.basename(p))[0] for p in train_paths]
)
test_id_to_path = pd.Series(
    test_paths, index=[os.path.splitext(os.path.basename(p))[0] for p in test_paths]
)



## === cell 2
A_IDX = np.array([0, 2, 4], dtype=np.int64)
B_IDX = np.array([1, 3, 5], dtype=np.int64)

Q_LEVELS = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float32)

POOL_BINS = (7, 8)  # 7*8=56 pooled features per image

_R_IDX = np.linspace(0.0, 1.0, 273, dtype=np.float32)
_C_IDX = np.linspace(0.0, 1.0, 256, dtype=np.float32)

_R_EDGES = np.linspace(0, 273, POOL_BINS[0] + 1, dtype=np.int64)
_C_EDGES = np.linspace(0, 256, POOL_BINS[1] + 1, dtype=np.int64)

_N_FLAT = 273 * 256

_Q_IDXS = np.rint((_N_FLAT - 1) * Q_LEVELS.astype(np.float64)).astype(np.int64)
_Q_IDXS = np.clip(_Q_IDXS, 0, _N_FLAT - 1)


def _axis_agg_stats(img2d: np.ndarray) -> np.ndarray:
    """
    img2d: (273, 256)
    Returns compact stats of row-wise and col-wise aggregated signals.
    """
    r = img2d.mean(axis=1)  # (273,)
    c = img2d.mean(axis=0)  # (256,)

    out = np.array(
        [
            r.mean(),
            r.std(),
            r.max(),
            r.min(),
            c.mean(),
            c.std(),
            c.max(),
            c.min(),
        ],
        dtype=np.float32,
    )
    return out


def _robust_quantiles(flat: np.ndarray) -> np.ndarray:
    """
    Score-relevant fix (minimal but high impact):
    Make quantile features *exact* and deterministic.

    The previous multi-k np.partition approach does NOT guarantee that each requested k
    is the global k-th order statistic when multiple k's are queried, which makes these
    15 quantile features noisy/uninformative and can collapse AUC.

    Sorting 69,888 float32s is fast enough here and yields true order statistics,
    improving separability while keeping the same feature family + feature count.
    """
    x = flat.astype(np.float32, copy=False)
    x = np.ascontiguousarray(x.reshape(-1))
    xs = np.sort(x)  # exact order statistics
    return xs[_Q_IDXS].astype(np.float32, copy=False)


def _center_of_mass_2d(img2d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Deterministic, light-weight "where is the energy" feature.
    Uses positive-shifted weights to avoid cancellation from normalized data.
    Returns [row_mean, row_std, col_mean, col_std] in normalized [0,1] coordinates.
    """
    img = img2d.astype(np.float32, copy=False)
    w = img - img.min()
    s = float(w.sum()) + eps

    r_w = w.sum(axis=1)  # (273,)
    c_w = w.sum(axis=0)  # (256,)

    r_mean = float((r_w * _R_IDX).sum() / s)
    c_mean = float((c_w * _C_IDX).sum() / s)

    r_var = float((r_w * (_R_IDX - r_mean) ** 2).sum() / s)
    c_var = float((c_w * (_C_IDX - c_mean) ** 2).sum() / s)

    return np.array([r_mean, np.sqrt(r_var), c_mean, np.sqrt(c_var)], dtype=np.float32)


def _pool_mean(img2d: np.ndarray, bins_r: int, bins_c: int) -> np.ndarray:
    r_edges = _R_EDGES
    c_edges = _C_EDGES

    img = img2d.astype(np.float32, copy=False)

    rs = r_edges[:-1]
    cs = c_edges[:-1]
    block_sums_r = np.add.reduceat(img, rs, axis=0)  # (bins_r, W)
    block_sums = np.add.reduceat(block_sums_r, cs, axis=1)  # (bins_r, bins_c)

    r_sizes = (r_edges[1:] - r_edges[:-1]).astype(np.float32)  # (bins_r,)
    c_sizes = (c_edges[1:] - c_edges[:-1]).astype(np.float32)  # (bins_c,)
    denom = (r_sizes[:, None] * c_sizes[None, :]).astype(np.float32)

    out = (block_sums / denom).astype(np.float32, copy=False)
    return out.reshape(-1)


def _pair_corr_flat(
    x_flat: np.ndarray, y_flat: np.ndarray, eps: float = 1e-6
) -> np.float32:
    x = x_flat.astype(np.float32, copy=False)
    y = y_flat.astype(np.float32, copy=False)
    x0 = x - x.mean()
    y0 = y - y.mean()
    num = float(np.dot(x0, y0) / x0.size)
    den = (
        float(np.sqrt(np.dot(x0, x0) / x0.size) * np.sqrt(np.dot(y0, y0) / y0.size))
        + eps
    )
    return np.float32(num / den)


def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)  # preserve original dtype handling

    x6 = x.reshape(6, -1)  # (6, 273*256)
    means6 = x6.mean(axis=1)
    stds6 = x6.std(axis=1)
    mx6 = x6.max(axis=1)
    mn6 = x6.min(axis=1)

    a = x[A_IDX].mean(axis=0)  # (273,256)
    b = x[B_IDX].mean(axis=0)  # (273,256)
    d = a - b

    d_flat = d.reshape(-1)
    contrast = np.array(
        [d_flat.mean(), d_flat.std(), d_flat.max(), d_flat.min()], dtype=np.float32
    )

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)

    a_stats = np.array(
        [a_flat.mean(), a_flat.std(), a_flat.max(), a_flat.min()], dtype=np.float32
    )
    b_stats = np.array(
        [b_flat.mean(), b_flat.std(), b_flat.max(), b_flat.min()], dtype=np.float32
    )
    ab_diff_stats = a_stats - b_stats  # stable difference-of-stats

    a_q = _robust_quantiles(a_flat)
    b_q = _robust_quantiles(b_flat)
    d_q = _robust_quantiles(d_flat)

    a_ax = _axis_agg_stats(a)
    b_ax = _axis_agg_stats(b)
    d_ax = _axis_agg_stats(d)

    com_a = _center_of_mass_2d(a)
    com_b = _center_of_mass_2d(b)
    com_d = _center_of_mass_2d(d)

    pr, pc = POOL_BINS
    pool_a = _pool_mean(a, pr, pc)
    pool_b = _pool_mean(b, pr, pc)
    pool_d = _pool_mean(d, pr, pc)

    a0, a1, a2 = x[A_IDX[0]], x[A_IDX[1]], x[A_IDX[2]]
    b0, b1, b2 = x[B_IDX[0]], x[B_IDX[1]], x[B_IDX[2]]

    a_mean = (a0 + a1 + a2) / 3.0
    b_mean = (b0 + b1 + b2) / 3.0

    a_mad = (np.abs(a0 - a_mean) + np.abs(a1 - a_mean) + np.abs(a2 - a_mean)) / 3.0
    b_mad = (np.abs(b0 - b_mean) + np.abs(b1 - b_mean) + np.abs(b2 - b_mean)) / 3.0

    a_mad_flat = a_mad.reshape(-1)
    b_mad_flat = b_mad.reshape(-1)

    a_mad_stats = np.array(
        [a_mad_flat.mean(), a_mad_flat.std(), a_mad_flat.max(), a_mad_flat.min()],
        dtype=np.float32,
    )
    b_mad_stats = np.array(
        [b_mad_flat.mean(), b_mad_flat.std(), b_mad_flat.max(), b_mad_flat.min()],
        dtype=np.float32,
    )
    mad_diff_stats = a_mad_stats - b_mad_stats

    a0f, a1f, a2f = a0.reshape(-1), a1.reshape(-1), a2.reshape(-1)
    b0f, b1f, b2f = b0.reshape(-1), b1.reshape(-1), b2.reshape(-1)

    a_corrs = np.array(
        [
            _pair_corr_flat(a0f, a1f),
            _pair_corr_flat(a0f, a2f),
            _pair_corr_flat(a1f, a2f),
        ],
        dtype=np.float32,
    )
    b_corrs = np.array(
        [
            _pair_corr_flat(b0f, b1f),
            _pair_corr_flat(b0f, b2f),
            _pair_corr_flat(b1f, b2f),
        ],
        dtype=np.float32,
    )
    corrs_summary = np.array(
        [
            a_corrs.mean(),
            a_corrs.std(),
            b_corrs.mean(),
            b_corrs.std(),
            (a_corrs.mean() - b_corrs.mean()),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [
            means6,
            stds6,
            mx6,
            mn6,  # 24
            contrast,  # 4  -> 28
            a_stats,
            b_stats,
            ab_diff_stats,  # 12 -> 40
            a_q,
            b_q,
            d_q,  # 15 -> 55
            a_ax,
            b_ax,
            d_ax,  # 24 -> 79
            com_a,
            com_b,
            com_d,  # 12 -> 91
            pool_a,
            pool_b,
            pool_d,  # 168 -> 259
            a_mad_stats,
            b_mad_stats,
            mad_diff_stats,  # 12 -> 271
            corrs_summary,  # 5 -> 276
        ]
    ).astype(np.float32, copy=False)
    return feats


def _features_from_path(p: str) -> np.ndarray:
    arr = np.load(p, mmap_mode="r")
    if arr.shape != (6, 273, 256):
        raise ValueError(
            f"Unexpected shape for path={p}: got {arr.shape}, expected (6,273,256)"
        )
    return extract_features_from_array(arr)


def build_feature_matrix(
    ids, id_to_path_series: pd.Series, n_features: int
) -> np.ndarray:
    paths = id_to_path_series.loc[ids].to_list()
    n = len(paths)
    X = np.empty((n, n_features), dtype=np.float32)
    if n == 0:
        return X

    max_workers = min(16, max(2, (os.cpu_count() or 2)))
    chunk_size = 256

    def _process_range(s: int, e: int):
        out = np.empty((e - s, n_features), dtype=np.float32)
        for j, p in enumerate(paths[s:e]):
            out[j] = _features_from_path(p)
        return s, out

    if n <= chunk_size:
        s, out = _process_range(0, n)
        X[s : s + out.shape[0]] = out
        return X

    futures = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            futures.append(ex.submit(_process_range, s, e))
        for fut in cf.as_completed(futures):
            s, out = fut.result()
            X[s : s + out.shape[0]] = out

    return X




## === cell 3
N_FEATURES = 276

train_ids = train_labels["id"].tolist()
y = train_labels["target"].astype(np.int64).values

X = build_feature_matrix(train_ids, train_id_to_path, n_features=N_FEATURES)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)



## === cell 4
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=300,
                C=1.0,
                n_jobs=None,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
    ]
)

clf.fit(X, y)



## === cell 5
test_ids = sample_sub["id"].tolist()
X_test = build_feature_matrix(test_ids, test_id_to_path, n_features=N_FEATURES)

pred = clf.predict_proba(X_test)[:, 1].astype(np.float64)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_ids, "target": pred})
submission.to_csv(OUT_SUB_PATH, index=False)

print(f"Wrote {OUT_SUB_PATH} with shape {submission.shape}")
print(submission.head())



## === cell 6
assert list(submission.columns) == ["id", "target"]
assert submission["id"].isna().sum() == 0
assert submission["target"].isna().sum() == 0
assert submission["target"].between(0, 1).all()
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count must match sample_submission"
assert (
    submission["id"].tolist() == sample_sub["id"].tolist()
), "IDs must be in sample_submission order"
print("Submission format checks passed.")
