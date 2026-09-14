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

0.7626591269737379

# 6. Current score

0.52045

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to read other Kaggle notebooks’ `../input/.../submission.csv` files that don’t exist in your environment, so nothing downstream is defined and no submission is written. I keep the “ensemble submissions” intent, but make it robust: it attempt to load those files if present and otherwise fall back to creating a valid submission from the provided `sample_submission.csv`. This guarantees an end-to-end run that always produces `submission.csv` with the correct columns and row order. Since you currently have no valid score, this prioritizes correctness and generating a valid submission; if some ensemble sources are available, it still blend them with your original weights.'
- What this solution (achieved 0.51452) has done: 'The timeout is dominated by the fallback path that loads tens of thousands of `.npy` files and computes quantiles using repeated `np.partition` calls (multiple times per sample), plus slow Python-loop I/O. I keep the exact feature definitions and the same logistic-regression CV training logic, but make feature extraction provably equivalent and much faster by: (1) computing all needed quantiles in one `np.quantile(..., method="linear")` call per array, (2) avoiding repeated partitions, (3) speeding up file indexing with `os.scandir`, and (4) parallelizing feature extraction with a deterministic `ThreadPoolExecutor` (I/O-bound) while preserving row order and outputs. These changes reduce both asymptotic constants and wall-clock time without changing the model, folds, or semantics (only negligible float rounding differences).'
- What this solution (achieved 0.51419) has done: 'I keep your exact fallback model (quantile-based handcrafted features + 5-fold LogisticRegression) but make two minimal, score-relevant fixes that typically lift AUC substantially from ~0.5: (1) ensure each sample’s prediction is aligned to the exact `sample_submission.csv` id order (right now `sample_ids.to_frame(index=False)` can misbehave and introduce NaNs/misalignment), and (2) make the classifier class-balance aware (`class_weight="balanced"`) to reduce bias from the dataset’s imbalance without changing the modeling approach. These are small changes that preserve the overall logic and still finish within the time limit. The rest of the pipeline (external submission blending if present; otherwise fallback training; always writing a valid `submission.csv`) remains the same.'
- What this solution (achieved 0.51419) has done: 'We keep your exact feature set and 5-fold LogisticRegression pipeline, but fix a score-critical issue: your current test indexing uses `sample_submission.csv` (6000 ids) while the real test set is ~36000 ids, so your submission is missing most rows and scores near-random. I change the test-id source to be the actual filenames in `/kaggle/.../test/**.npy`, then align predictions back to that full id list in a deterministic order. This preserves the same training logic and semantics, but should move AUC substantially upward toward your target by making the submission valid for the full test set. The external-submission blending path remains intact; it still be used if those files exist, otherwise the fallback model generates full-length predictions.'
- What this solution (achieved 0.51501) has done: 'Your current score (~0.514) suggests the fallback model is running but producing weak signal; we keep the exact feature set and LogisticRegression pipeline, and make only score-relevant tweaks that typically move AUC upward without changing the approach. First, we add two cheap but informative aggregate features that preserve the same “A vs B” logic (difference-of-means and difference-of-stds between A and B) and keep everything else identical. Second, we slightly strengthen the LogisticRegression regularization (increase `C`) to reduce underfitting on these low-dimensional handcrafted features while keeping the same solver/training loop. Finally, we keep the external-submission ensemble path intact and still write a valid full-length `submission.csv`.'
- What this solution (achieved 0.51517) has done: 'We keep your exact pipeline (same handcrafted “A vs B” quantile features + 5-fold LogisticRegression) and only make two score-relevant, low-risk adjustments to move AUC up from ~0.515 toward your 0.763 target. First, we fix a subtle but important mismatch: the external-submission ensemble currently aligns to `sample_submission.csv` (6000 ids), which can silently misalign/NaN-fill if ever used; we align external submissions to the *actual test ids* discovered from the test folder, and only then blend. Second, we add a tiny, semantics-preserving calibration step: after CV, we fit the same model once on all training data and blend its test probabilities 50/50 with the CV-averaged test probabilities to reduce fold variance without changing the approach or metric.'
- What this solution (achieved 0.51504) has done: 'Your current AUC (~0.515) indicates the fallback model is essentially near-random, so we need a small change that adds real signal without changing the overall approach (still handcrafted statistics + 5-fold LogisticRegression). I keep the same feature philosophy (“A vs B” aggregate comparisons) but add a couple of very cheap, strongly-informative features: fraction of “bright” pixels in A vs B at a shared threshold, and the same computed on the A−B difference map; these often correlate with needle presence while staying within the same handcrafted-summary paradigm. I also make the “external submissions” ensemble path safer by only triggering fallback training when *no* external predictions are present (instead of any NaN), preventing unnecessary overwrites. These are minimal, score-relevant, and should move the score upward toward your 0.763 target while still finishing under the time limit.'
- What this solution (achieved 0.51504) has done: 'Your current AUC (~0.515) is far below the target (~0.763), so we need a small but meaningful signal boost without changing the overall “handcrafted stats → 5-fold LogisticRegression” core. The most score-relevant minimal fix is to correct how the A/B cadence panels are aggregated: right now `(a - b)` is computed with mismatched pairing/broadcasting (A0-A1, A0-A2, …) instead of the intended per-cadence pairing (A0-B0, A1-B1, A2-B2), which can destroy the discriminative “on-target vs off-target” signal. I change only the difference computation to be properly paired and keep the same feature set structure and training loop intact, and I also adjust the “bright” feature that uses `d_flat > 0` to reflect the corrected paired difference. Everything else (paths, CV, LogisticRegression, submission writing) remains the same.'
- What this solution (achieved 0.51874) has done: 'The timeout is dominated by the fallback path: it loads and feature-engineers every train and test `.npy` file and repeatedly computes expensive percentiles per sample, plus it uses a 5-fold CV + full refit. To keep identical core logic and predictions, we (1) avoid any fallback execution unless strictly necessary, (2) make feature extraction provably equivalent but faster by using `np.partition`-based exact quantiles for the specific indices used (no approximation), (3) reduce Python overhead and I/O overhead by streaming file existence checks via directory listing (instead of per-id `os.path.exists`) and by writing features as futures complete (avoids executor/map overhead), and (4) speed up LogisticRegression with `n_jobs` where supported while keeping solver/iterations identical. These changes preserve the algorithm, model, training loops, and evaluation semantics while cutting constant factors enough to fit under 600 seconds.'
- What this solution (achieved 0.52045) has done: 'The timeout is dominated by the fallback path: it scans the entire train/test directories to build an `available_ids` set and then spawns tens of thousands of thread tasks (one per file), while also recomputing feature dimension by running feature extraction on a dummy array. The optimized version removes the full-directory scan (paths are deterministically derived from `id`, so we just check `os.path.exists` per id), batches thread work by chunk to reduce executor overhead, and computes feature dimension once as a constant (derived from the fixed feature construction). These changes preserve identical feature values and model training/inference semantics, while substantially reducing Python overhead and filesystem traversal time. External-submission ensemble logic and I/O paths remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

CANDIDATE_SUB_PATHS = [
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    "../input/seti-learned-image-resizing/submission.csv",
    "../input/rerun-seti-e-t-resnet18d-baseline/submission.csv",
    "../input/ensemble-for-seti-competition/submission.csv",
    "../input/fixed-gradual-warmup-custom-head/submission.csv",
]

SAMPLE_SUB_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = next((p for p in SAMPLE_SUB_PATHS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(SAMPLE_SUB_PATHS)
    )

sample = pd.read_csv(sample_path)
if not {"id", "target"}.issubset(sample.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns ['id','target'], got: {list(sample.columns)}"
    )


def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


TRAIN_LABELS_PATHS = [
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
    "../input/train_labels.csv",
]
train_labels_path = _find_first_existing(TRAIN_LABELS_PATHS)

TRAIN_DIR_CANDIDATES = [
    "/kaggle/input/train",
    "/kaggle/data/train",
    "../input/train",
]
TEST_DIR_CANDIDATES = [
    "/kaggle/input/test",
    "/kaggle/data/test",
    "../input/test",
]
train_dir = _find_first_existing(TRAIN_DIR_CANDIDATES)
test_dir = _find_first_existing(TEST_DIR_CANDIDATES)

align_id_df = pd.DataFrame({"id": sample["id"].astype("string")})


def _load_submission_if_exists(
    path: str, align_id_df: pd.DataFrame
) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["id", "target"])
    if not {"id", "target"}.issubset(df.columns):
        return None
    df = df.copy()
    df["id"] = df["id"].astype("string")
    df = df.drop_duplicates("id")
    return align_id_df.merge(df, on="id", how="left")


loaded = [_load_submission_if_exists(p, align_id_df) for p in CANDIDATE_SUB_PATHS]
data1, data2, data3, data4, data5, data6, data7 = loaded

available = [i + 1 for i, d in enumerate(loaded) if d is not None]
print(f"Loaded {len(available)} external submission(s): {available} (out of 7).")
print(f"Using sample_submission.csv from: {sample_path}")
print(f"Alignment id source: sample_submission; rows={len(align_id_df)}")


## === cell 1
data11 = data1.copy() if data1 is not None else align_id_df.assign(target=np.nan).copy()


## === cell 2
weights = {
    "data1": 0.12,
    "data2": 0.10,
    "data3": 0.10,
    "data4": 0.11,
    "data5": 0.11,
    "data6": 0.58,
}
sources = {
    "data1": data1,
    "data2": data2,
    "data3": data3,
    "data4": data4,
    "data5": data5,
    "data6": data6,
}

present = [(name, df, weights[name]) for name, df in sources.items() if df is not None]
if len(present) == 0:
    data11["target"] = np.nan
else:
    wsum = sum(w for _, _, w in present)
    preds = np.zeros(len(align_id_df), dtype=np.float64)
    for name, df, w in present:
        p = pd.to_numeric(df["target"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        p = np.where(np.isfinite(p), p, 0.5)
        preds += (w / wsum) * p
    data11["target"] = np.clip(preds, 0.0, 1.0)

data11 = align_id_df.merge(data11[["id", "target"]], on="id", how="left")


## === cell 3
from concurrent.futures import ThreadPoolExecutor, as_completed


def _exact_percentiles_intpos(x1d: np.ndarray, q_list: list[float]) -> np.ndarray:
    return np.percentile(x1d, q_list).astype(np.float32, copy=False)


def _extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    a = x[[0, 2, 4]]  # A0, A1, A2
    o = x[[1, 3, 5]]  # off-target panels (B, C, D)

    a_flat = a.reshape(-1)
    o_flat = o.reshape(-1)

    d = a - o
    d_flat = d.reshape(-1)

    a_mean = float(a_flat.mean(dtype=np.float64))
    a_std = float(a_flat.std(dtype=np.float64))
    o_mean = float(o_flat.mean(dtype=np.float64))
    o_std = float(o_flat.std(dtype=np.float64))
    d_mean = float(d_flat.mean(dtype=np.float64))
    d_std = float(d_flat.std(dtype=np.float64))

    a_q = _exact_percentiles_intpos(a_flat, [10, 50, 90, 99])
    o_q = _exact_percentiles_intpos(o_flat, [10, 50, 90, 99])
    d_q = _exact_percentiles_intpos(d_flat, [10, 50, 90, 99])

    fa = np.array([a_mean, a_std, a_q[0], a_q[1], a_q[2]], dtype=np.float32)
    fo = np.array([o_mean, o_std, o_q[0], o_q[1], o_q[2]], dtype=np.float32)
    fd = np.array([d_mean, d_std, d_q[0], d_q[1], d_q[2]], dtype=np.float32)

    amax = float(a_flat.max())
    omax = float(o_flat.max())
    amin = float(a_flat.min())
    omin = float(o_flat.min())

    aq99 = float(a_q[3])
    oq99 = float(o_q[3])

    extra = np.array(
        [
            amax,
            omax,
            amax - amin,
            omax - omin,
            float((a_flat > aq99).mean(dtype=np.float64)),
            float((o_flat > oq99).mean(dtype=np.float64)),
        ],
        dtype=np.float32,
    )

    ab_extra = np.array(
        [float(a_mean - o_mean), float(a_std - o_std)], dtype=np.float32
    )

    thr = float(a_q[2])  # A 0.90 quantile as a robust high-intensity threshold
    bright_ab = np.array(
        [
            float(
                (a_flat > thr).mean(dtype=np.float64)
                - (o_flat > thr).mean(dtype=np.float64)
            ),
            float((d_flat > 0.0).mean(dtype=np.float64)),
        ],
        dtype=np.float32,
    )

    a0, a1, a2 = a[0].reshape(-1), a[1].reshape(-1), a[2].reshape(-1)
    o0, o1, o2 = o[0].reshape(-1), o[1].reshape(-1), o[2].reshape(-1)
    a_pair = (
        np.abs(a0 - a1).mean(dtype=np.float64)
        + np.abs(a1 - a2).mean(dtype=np.float64)
        + np.abs(a0 - a2).mean(dtype=np.float64)
    ) / 3.0
    o_pair = (
        np.abs(o0 - o1).mean(dtype=np.float64)
        + np.abs(o1 - o2).mean(dtype=np.float64)
        + np.abs(o0 - o2).mean(dtype=np.float64)
    ) / 3.0
    consistency = np.array(
        [float(a_pair), float(o_pair), float(a_pair - o_pair)], dtype=np.float32
    )

    dmax = float(d_flat.max())
    dq99 = float(d_q[3])
    paired_diff_extra = np.array([dmax, dq99], dtype=np.float32)

    return np.concatenate(
        [fa, fo, fd, extra, ab_extra, bright_ab, consistency, paired_diff_extra],
        axis=0,
    )


def _id_to_npy_path(base_dir: str, id_: str) -> str:
    return os.path.join(base_dir, id_[0], f"{id_}.npy")


def _filter_existing_ids(ids: list[str], base_dir: str) -> tuple[list[str], list[str]]:
    kept_ids = []
    kept_paths = []
    for id_ in ids:
        p = _id_to_npy_path(base_dir, id_)
        if os.path.exists(p):
            kept_ids.append(id_)
            kept_paths.append(p)
    return kept_ids, kept_paths


FEAT_DIM = 30


def _build_feature_matrix_from_ids(
    ids: list[str], base_dir: str
) -> tuple[np.ndarray, list[str]]:
    kept_ids, kept_paths = _filter_existing_ids(ids, base_dir)
    if not kept_ids:
        return np.zeros((0, FEAT_DIM), dtype=np.float32), []

    X = np.empty((len(kept_ids), FEAT_DIM), dtype=np.float32)

    def _process_chunk(start_end: tuple[int, int]):
        s, e = start_end
        out = np.empty((e - s, FEAT_DIM), dtype=np.float32)
        for j, path in enumerate(kept_paths[s:e]):
            arr = np.load(path, mmap_mode=None)
            out[j] = _extract_features_from_array(arr)
        return s, out

    n = len(kept_paths)
    max_workers = min(
        16, (os.cpu_count() or 4)
    )  # avoid oversubscription for numpy heavy ops
    chunk = max(64, n // (max_workers * 4) if n else 64)
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_chunk, r) for r in ranges]
        for fut in as_completed(futures):
            s, out = fut.result()
            e = s + out.shape[0]
            X[s:e] = out
            done += out.shape[0]
            if done % 5000 == 0:
                print(f"  processed {done}/{n}")

    return X, kept_ids


need_fallback = (len(present) == 0) or data11["target"].isna().all()

if need_fallback:
    if train_labels_path is None or train_dir is None or test_dir is None:
        print(
            "Fallback training disabled (missing train/test paths). Writing 0.5 predictions."
        )
        data11["target"] = 0.5
    else:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        train_labels = pd.read_csv(train_labels_path, usecols=["id", "target"])
        train_labels = train_labels.drop_duplicates("id")
        train_labels["id"] = train_labels["id"].astype(str)

        train_ids = train_labels["id"].tolist()
        print(f"Building train features for {len(train_ids)} ids from: {train_dir}")
        X_train, kept_train_ids = _build_feature_matrix_from_ids(train_ids, train_dir)
        y_train = (
            train_labels.set_index("id")
            .loc[kept_train_ids, "target"]
            .to_numpy(dtype=np.int64)
        )
        print(f"Train feature matrix: {X_train.shape}")

        test_ids_full = align_id_df["id"].astype(str).tolist()
        print(f"Building test features for {len(test_ids_full)} ids from: {test_dir}")
        X_test, kept_test_ids = _build_feature_matrix_from_ids(test_ids_full, test_dir)
        print(f"Test feature matrix: {X_test.shape}")

        kept_test_idx = pd.Index(kept_test_ids)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        test_pred_cv = np.zeros(len(kept_test_ids), dtype=np.float64)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=300,
                        solver="lbfgs",
                        class_weight="balanced",
                        C=3.0,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
            model.fit(X_train[tr], y_train[tr])
            test_pred_cv += model.predict_proba(X_test)[:, 1] / skf.n_splits
            print(f"Fold {fold} done.")

        model.fit(X_train, y_train)
        test_pred_full = model.predict_proba(X_test)[:, 1]
        test_pred = 0.5 * test_pred_cv + 0.5 * test_pred_full

        pred_map = pd.Series(test_pred, index=kept_test_idx)

        data11 = pd.DataFrame({"id": test_ids_full})
        data11["target"] = data11["id"].map(pred_map).astype("float64")
        data11["target"] = data11["target"].fillna(0.5).clip(0.0, 1.0)
else:
    data11["target"] = (
        pd.to_numeric(data11["target"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    )


## === cell 4
out_path = "submission.csv"
data11[["id", "target"]].to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == [
    "id",
    "target",
], f"Bad submission columns: {list(check.columns)}"
assert check["id"].astype(str).is_unique, "Submission ids must be unique."
assert check["target"].notna().all(), "Submission has NaN predictions."
print(f"Wrote {out_path} with shape {check.shape}")
print(check.head())
