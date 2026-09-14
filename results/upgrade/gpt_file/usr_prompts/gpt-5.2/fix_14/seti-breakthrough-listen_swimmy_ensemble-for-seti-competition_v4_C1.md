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

0.7571464972782159

# 6. Current score

0.50251

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50267) has done: 'The timeout is dominated by Python-level loops doing 60k+ individual `np.load` calls plus repeated percentile computations per file. I keep the exact same features and model, but make feature extraction faster by using `np.partition` to compute the 5th/95th percentiles (exact for this use), reduce temporary allocations, and parallelize per-file extraction with a thread pool (I/O + NumPy ops release the GIL enough to help). I also avoid building huge recursive glob lists by resolving file paths directly from IDs (since the dataset is sharded by the first hex character), eliminating expensive directory scans. These changes preserve evaluation semantics and should bring runtime under 600s.'
- What this solution (achieved 0.50267) has done: 'Your current score (0.50267 AUC) is essentially random, which strongly suggests the per-sample feature vector is being corrupted (most commonly by an in-place side effect inside the percentile helper). I make the percentile computation non-mutating by operating on a local copy of the flattened array, preserving the exact feature definitions and the same LogisticRegression setup. This is a minimal change to restore feature correctness and should move AUC upward toward your 0.757 target without changing the model/training approach. I also add a tiny numeric safety clip on predicted probabilities to avoid any extreme values (this does not change ranking in practice, but is harmless for AUC).'
- What this solution (achieved 0.49767) has done: 'Your AUC being ~0.50 suggests the model is training on features that don’t match the test-time feature distribution; the most common cause here is path resolution/splitting logic (sharded folders) silently loading the wrong files or failing over to a different dataset root. I make the dataset root selection deterministic for this competition (prefer the canonical `.../seti-breakthrough-listen/` layout) and add strict sanity checks that each loaded `.npy`’s filename matches the requested `id`, preventing accidental cross-loads that destroy signal and push AUC toward random. I also standardize features with `StandardScaler` (still LogisticRegression, same loss/solver, same features) which typically improves calibration/ranking for linear models on heterogeneous feature scales and should move AUC upward toward your 0.757 target with minimal risk. Everything else (feature definitions, model family, and submission format) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.49767) has done: 'Your current AUC (~0.50) suggests the signal is being lost in feature extraction rather than needing a different model, so I keep your exact feature definitions and LogisticRegression training loop, but fix the percentile computation to avoid repeated full-array copies and reduce numerical noise from the custom implementation. Specifically, I replace the hand-rolled “linear” percentile with `np.percentile(..., method="linear")` on float32 arrays (same definition as intended, but more reliable), and I compute all needed percentiles in one call per array to prevent subtle inconsistencies. This is a minimal change that preserves evaluation semantics (same features conceptually) while typically restoring meaningful ranking and moving AUC toward your 0.757 target. Everything else (paths, sharding, scaling, model, submission schema) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.49767) has done: 'Your current AUC (~0.50) indicates the model isn’t learning useful signal, and the most common cause here is either a mismatch between the `id` order and the feature rows, or silent feature corruption from NaNs/infs or unstable percentile behavior. I keep your exact feature definitions and LogisticRegression approach, but (1) make feature extraction numerically safe (replace any NaN/inf with finite values), (2) compute percentiles on a contiguous float32 view to avoid subtle dtype/stride issues, and (3) add a quick local AUC sanity-check on the held-out split so you can confirm learning before generating the submission. These are minimal changes that preserve evaluation semantics and should move AUC upward toward your 0.757 target without changing the model family or feature set. The script still runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.49767) has done: 'Your AUC being ~0.50 despite reasonable features strongly suggests the extracted feature rows are not aligned with the corresponding `id`s (a subtle but common issue when using `ThreadPoolExecutor.map` with `chunksize`, which can yield out-of-order consumption depending on the backend). I make the feature-matrix build deterministic by collecting futures and writing results back by index, guaranteeing the `i`th feature vector always matches the `i`th `id` while keeping the exact same features and LogisticRegression training. I also add a quick, lightweight checksum-like sanity check that a couple of loaded paths match their expected IDs (no extra data, no metric change), to catch any accidental root/path mismatch early. These minimal changes should move AUC upward toward your 0.757 target while preserving your model and feature definitions and still producing a valid `submission.csv`.'
- What this solution (achieved 0.49767) has done: 'Your AUC being ~0.50 strongly suggests the train/test feature matrices are being built from the wrong files due to an `id_to_npy_path` assumption that doesn’t match this dataset’s actual sharding (it’s by the first hex char of the *full filename*, e.g. `0/000...npy`, not by the first char of the bare `id`). I make path resolution robust by searching for the correct shard with a tiny fixed set of candidates per id (fast, no full directory scans) and add a strict “exactly one match” check to prevent silent misloads that randomize training. This keeps your exact feature definitions, scaling, and LogisticRegression training unchanged, but should restore meaningful signal and move AUC up toward the 0.757 target. The rest of the pipeline remains identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.49792) has done: 'Your AUC being ~0.50 suggests the model is learning on effectively uninformative/corrupted inputs; the most common “minimal-change” fix here is to ensure we are loading the correct `.npy` for every `id` (your current `id_to_npy_path` guesses shards and can miss the true shard, silently breaking signal). I replace shard-guessing with a one-time index of the actual `train/` and `test/` directories (fast enough) to build an exact `id -> path` map, then use that map everywhere to guarantee alignment. I keep the exact same feature definitions and LogisticRegression training setup, but I also remove `class_weight="balanced"` (AUC is rank-based and balancing can distort ranking for this task) to nudge performance upward toward your 0.757 target without changing the modeling approach. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.49787) has done: 'Your AUC is far below the target, so we should make a small, legitimate improvement that preserves the same feature definitions and LogisticRegression approach. The biggest likely gain with minimal risk is to set LogisticRegression’s `C` based on a quick internal CV over a tiny grid (still the same model/solver/loss; just a better regularization strength for ranking/AUC). To keep runtime under 600s, we reuse the already-extracted feature matrix and only do CV on the scaled features (no extra file I/O). We then refit once on the full training set with the chosen `C` and generate the same `submission.csv` format.'
- What this solution (achieved 0.50251) has done: 'Your current AUC (~0.50) is far below the target, so we should make a small change that improves ranking without changing the overall approach (same handcrafted features + LogisticRegression). The most likely issue is underfitting from too-strong regularization and fold-to-fold instability, so I (1) widen the `C` grid modestly and (2) use a slightly more stable CV estimate (5 folds) while keeping the same solver/loss and using the already-extracted feature matrix (no extra I/O). After selecting `C`, I refit on the full scaled training set (instead of only the 80% split) to legitimately improve generalization for the Kaggle test set while preserving evaluation semantics. Everything else (feature extraction, scaling, probability output, and submission format/path) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.50251) has done: 'Your score (~0.50 AUC) indicates the model is effectively not learning signal; with your current pipeline, the most likely cause is a train/test distribution mismatch introduced by fitting the `StandardScaler` twice (first on `X_tr`, then refit on full `X` before scoring test). I keep the exact same features and LogisticRegression setup, but make the scaling consistent by refitting both the scaler and the final classifier on the full training data together (same as you intended, but without reusing a scaler object that was fit on a different distribution). I also make the CV selection operate on a pipeline-like “scale inside each fold” manner to avoid leakage from scaling on the whole `X_tr_s` before CV (still the same model and features; just correct evaluation and typically better generalization). These are minimal, semantics-preserving fixes that should move AUC upward toward your 0.757 target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.50251) has done: 'Your AUC is essentially random relative to the target, so the smallest high-impact fix is to make sure each `.npy` is loaded from the correct competition dataset split (new train/test), not accidentally from the similarly-structured `old_leaky_data` directories that can silently scramble the mapping and destroy signal. I keep your exact feature definitions and LogisticRegression+StandardScaler approach, but harden `build_id_path_index()` to only index `.npy` files whose IDs are actually in the expected `train_labels.csv` / `sample_submission.csv` lists, and I assert the index sizes match exactly to fail fast if the wrong root is selected. This is a minimal, semantics-preserving change aimed specifically at restoring correct train/test alignment, which should move AUC up toward your 0.757 target while keeping runtime within limits. The script still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42


def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


DATA_ROOT = find_existing_path(
    [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/input",
        "/kaggle/data",
    ]
)

TRAIN_DIR = find_existing_path(
    [
        os.path.join(DATA_ROOT, "train"),
        "/kaggle/input/seti-breakthrough-listen/train",
        "/kaggle/data/seti-breakthrough-listen/train",
    ]
)

TEST_DIR = find_existing_path(
    [
        os.path.join(DATA_ROOT, "test"),
        "/kaggle/input/seti-breakthrough-listen/test",
        "/kaggle/data/seti-breakthrough-listen/test",
    ]
)

TRAIN_LABELS_PATH = find_existing_path(
    [
        os.path.join(DATA_ROOT, "train_labels.csv"),
        "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
        "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
    ]
)

SAMPLE_SUB_PATH = find_existing_path(
    [
        os.path.join(DATA_ROOT, "sample_submission.csv"),
        "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
        "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    ]
)

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_ids = train_labels["id"].tolist()
test_ids = sample_sub["id"].tolist()

y = train_labels["target"].to_numpy(dtype=np.int64)

train_id_set = set(train_ids)
test_id_set = set(test_ids)


def build_id_path_index(root_dir, allowed_ids):
    id2path = {}
    for shard in os.listdir(root_dir):
        shard_dir = os.path.join(root_dir, shard)
        if not os.path.isdir(shard_dir):
            continue
        for fn in os.listdir(shard_dir):
            if not fn.endswith(".npy"):
                continue
            _id = fn[:-4]
            if _id not in allowed_ids:
                continue
            if _id not in id2path:
                id2path[_id] = os.path.join(shard_dir, fn)
    return id2path


TRAIN_ID2PATH = build_id_path_index(TRAIN_DIR, train_id_set)
TEST_ID2PATH = build_id_path_index(TEST_DIR, test_id_set)

missing_train = [i for i in train_ids if i not in TRAIN_ID2PATH]
missing_test = [i for i in test_ids if i not in TEST_ID2PATH]
if missing_train:
    raise FileNotFoundError(
        f"Missing {len(missing_train)} train .npy files, e.g. {missing_train[:5]} "
        f"(TRAIN_DIR={TRAIN_DIR}). This usually means the wrong dataset root was indexed."
    )
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test .npy files, e.g. {missing_test[:5]} "
        f"(TEST_DIR={TEST_DIR}). This usually means the wrong dataset root was indexed."
    )

if len(TRAIN_ID2PATH) != len(train_ids):
    raise RuntimeError(
        f"TRAIN_ID2PATH size mismatch: got {len(TRAIN_ID2PATH)} expected {len(train_ids)} "
        f"(TRAIN_DIR={TRAIN_DIR})."
    )
if len(TEST_ID2PATH) != len(test_ids):
    raise RuntimeError(
        f"TEST_ID2PATH size mismatch: got {len(TEST_ID2PATH)} expected {len(test_ids)} "
        f"(TEST_DIR={TEST_DIR})."
    )


def id_to_npy_path(root_dir, _id):
    if root_dir == TRAIN_DIR:
        return TRAIN_ID2PATH[_id]
    if root_dir == TEST_DIR:
        return TEST_ID2PATH[_id]
    raise ValueError(f"Unknown root_dir: {root_dir}")


_ = id_to_npy_path(TRAIN_DIR, train_ids[0])
_ = id_to_npy_path(TEST_DIR, test_ids[0])




## === cell 1
def extract_features_from_path(path, expected_id=None):
    if expected_id is not None:
        base = os.path.basename(path)
        if base != f"{expected_id}.npy":
            raise ValueError(
                f"ID/path mismatch: expected {expected_id}.npy, got {base}"
            )

    x = np.load(path)  # (6, 273, 256), float16
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # on-target
    O = x[[1, 3, 5]]  # off-target

    A_mean = A.mean(axis=0)
    O_mean = O.mean(axis=0)
    D = A_mean - O_mean

    Df = np.ascontiguousarray(D, dtype=np.float32).ravel()
    Af = np.ascontiguousarray(A_mean, dtype=np.float32).ravel()
    Of = np.ascontiguousarray(O_mean, dtype=np.float32).ravel()

    D_p05, D_p95 = np.percentile(Df, [5.0, 95.0], method="linear")
    A_p05, A_p95 = np.percentile(Af, [5.0, 95.0], method="linear")
    O_p05, O_p95 = np.percentile(Of, [5.0, 95.0], method="linear")

    absD = np.abs(D)
    feats = np.array(
        [
            D.mean(),
            D.std(),
            absD.mean(),
            absD.max(),
            D_p95,
            D_p05,
            (A_mean.mean() - O_mean.mean()),
            (A_mean.std() - O_mean.std()),
            (np.abs(A_mean).mean() - np.abs(O_mean).mean()),
            (A_p95 - O_p95),
            (A_p05 - O_p05),
        ],
        dtype=np.float32,
    )

    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats


def build_feature_matrix(ids, root_dir, max_workers=None):
    X = np.zeros((len(ids), 11), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    def _one(i, _id):
        p = id_to_npy_path(root_dir, _id)
        return i, extract_features_from_path(p, expected_id=_id)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, i, _id) for i, _id in enumerate(ids)]
        for fut in as_completed(futures):
            i, feats = fut.result()
            X[i] = feats

    return X


for _check_id in (train_ids[0], train_ids[len(train_ids) // 2], train_ids[-1]):
    _p = id_to_npy_path(TRAIN_DIR, _check_id)
    _ = extract_features_from_path(_p, expected_id=_check_id)

X = build_feature_matrix(train_ids, TRAIN_DIR)



## === cell 2
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


def select_C_by_cv(
    X_raw,
    y,
    C_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    n_splits=5,
    random_state=RANDOM_STATE,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_C = None
    best_auc = -np.inf

    for C in C_grid:
        aucs = []
        for tr_idx, va_idx in skf.split(X_raw, y):
            scaler_cv = StandardScaler()
            X_tr_s = scaler_cv.fit_transform(X_raw[tr_idx])
            X_va_s = scaler_cv.transform(X_raw[va_idx])

            clf_cv = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE,
                C=C,
            )
            clf_cv.fit(X_tr_s, y[tr_idx])
            p = clf_cv.predict_proba(X_va_s)[:, 1]
            aucs.append(roc_auc_score(y[va_idx], p))

        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_C = C

    return best_C, best_auc


best_C, cv_auc = select_C_by_cv(X_tr, y_tr)
print("Selected C:", best_C, "CV AUC:", cv_auc)

scaler_holdout = StandardScaler()
X_tr_s = scaler_holdout.fit_transform(X_tr)
X_va_s = scaler_holdout.transform(X_va)

clf_holdout = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE,
    C=best_C,
)
clf_holdout.fit(X_tr_s, y_tr)

va_pred = clf_holdout.predict_proba(X_va_s)[:, 1]
print("Holdout AUC:", roc_auc_score(y_va, va_pred))

scaler = StandardScaler()
X_all_s = scaler.fit_transform(X)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE,
    C=best_C,
)
clf.fit(X_all_s, y)



## === cell 3
X_test = build_feature_matrix(test_ids, TEST_DIR)
X_test_s = scaler.transform(X_test)

test_pred = clf.predict_proba(X_test_s)[:, 1].astype(np.float64)
test_pred = np.clip(test_pred, 1e-9, 1 - 1e-9)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("target min/max:", submission["target"].min(), submission["target"].max())
