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

0.7563564198819381

# 6. Current score

0.52456

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49297) has done: 'Main bottleneck is per-file `np.load` + repeated small NumPy reductions executed ~60k times in pure Python loops; this I/O + overhead dominates and causes the timeout. The core logic is preserved, but feature extraction is made provably equivalent and faster by (1) memory-mapping `.npy` loads to avoid extra copies, (2) computing the same statistics with fewer passes (reuse intermediate means/profiles), and (3) parallelizing feature extraction across CPU cores with ordered outputs so labels/ids stay aligned. Training is kept identical (same model, folds, solver, iterations), with small constant-factor reductions (avoid rebuilding identical pipelines). Paths and outputs remain unchanged.'
- What this solution (achieved 0.50634) has done: 'Your current score is far below the target, so we should improve discrimination while keeping the same overall approach (fast per-file feature extraction + logistic regression). The biggest score issue is that the features are mostly global statistics over the whole cube, which weakly capture the competition’s key pattern: signals present in A panels (0,2,4) but absent in B/C/D (1,3,5). I add a small set of *difference-focused* features (A−BCD) based on time/frequency profiles and energy ratios, computed with the same cheap NumPy reductions (no new models, no new training loop). I also fix the sharding bug in `id_to_path` (it currently uses only the first hex digit; the dataset is sharded by the first character `'0'..'15'` as a string, so we should not convert from hex), which can silently point to the wrong folder structure in other environments and hurts robustness.'
- What this solution (achieved 0.51728) has done: 'We keep your feature-extraction + logistic-regression pipeline intact, but add a very small set of additional “A vs not-A” contrast features that are still just cheap reductions (means/stds/maxes) and should improve AUC toward the target because they encode the ABACAD structure more directly. We also add a lightweight nonlinearity by appending squared versions of the same features (via `PolynomialFeatures(degree=2, include_bias=False)`), which keeps the same model family (logistic regression) and training loop but lets it capture interactions without changing your architecture. Finally, we tune only the regularization strength (`C`) mildly and use `class_weight="balanced"` to better handle any imbalance, both of which typically move ROC-AUC upward with minimal risk and no change in evaluation semantics. The submission writing and paths remain unchanged and still produce `submission.csv`.'
- What this solution (achieved 0.49783) has done: 'The timeout is dominated by per-file feature extraction across ~54k train + 6k test `.npy` files, and using a `ProcessPoolExecutor` causes high overhead because each task re-imports modules and repeatedly opens files. I keep the exact same features and the same scikit-learn pipeline/training, but switch feature extraction to a `ThreadPoolExecutor` (NumPy I/O + reductions release the GIL) and preallocate the output arrays to avoid huge Python lists and `vstack` overhead. I also remove repeated `os.path.exists` checks inside `id_to_path` by resolving shard paths once and relying on a single existence mask pass per split, which is correctness-equivalent. These changes reduce overhead substantially while preserving identical algorithmic behavior and predictions up to negligible FP differences.'
- What this solution (achieved 0.51728) has done: 'Your current score is far below the target, so we should improve AUC while keeping the same feature-extraction + logistic-regression pipeline. The biggest likely correctness issue is the `id_to_path` sharding: the dataset folders are named `"0".."15"` by the first hex character as a string, not by converting that hex digit to an integer (so ids starting with `"a"` should go under folder `"a"`, not `"10"`); this can cause many missing files and near-random predictions. I fix `id_to_path` to use the first character directly and add a strict check that train/test files exist to avoid silently training on a broken subset. Everything else (features, model, training loop, submission writing) is kept the same to ensure minimal change and stable behavior.'
- What this solution (achieved 0.51728) has done: 'The runtime error comes from an incorrect sharding rule in `id_to_path`: this dataset stores files under folders named by the first hex character (`"0".."f"`), not by converting that character to an integer (`"0".."15"`), so most files are reported missing and the pipeline aborts. I fix `id_to_path` to use the first character directly and keep the existing strict existence checks so we don’t silently train on a tiny subset. I also make feature extraction robust to `numpy` mmap returning non-writeable arrays by avoiding in-place operations, while preserving the exact same reductions/features and the same sklearn pipeline. Finally, the script run end-to-end and write `submission.csv` with the required `id,target` columns in the sample submission order.'
- What this solution (achieved 0.51728) has done: 'We keep your exact feature set and logistic-regression-with-poly pipeline, but fix two issues that commonly cap AUC here: (1) the model is trained on a subset when files are “missing” due to selecting the wrong dataset root, and (2) without `warm_start`, each CV fold solves from scratch and can converge to slightly different optima under time pressure. I make the input root auto-resolve to the correct competition folder (prefer `/kaggle/input/seti-breakthrough-listen/` when present) while keeping paths otherwise unchanged, and I enforce a strict “no missing files” policy so we don’t silently train on a tiny biased subset. Finally, I keep the same solver/iterations/architecture but enable `warm_start=True` for stability and slightly better convergence consistency, which should move AUC upward toward your target without changing evaluation semantics.'
- What this solution (achieved 0.51734) has done: 'To move your AUC upward toward the 0.756 target without changing the core approach (cheap per-file NumPy reductions + LogisticRegression on engineered features), I’m adding a tiny set of *ABACAD-structure-aligned* features that capture “needle present in A panels but not in B/C/D” more directly. These additions are still simple reductions (means/stds/maxes) and reuse already-computed time/frequency profiles, so runtime stays within budget and the training loop/model family remain identical. I’m also making a very small, safe adjustment to the regularization strength (`C`) to improve fit under the expanded feature space, while keeping the same pipeline, solver, and CV procedure. Submission writing, paths, and semantics remain unchanged.'
- What this solution (achieved 0.52246) has done: 'We keep your fast feature-extraction + (scaler → poly2 → logistic regression) pipeline unchanged, but add a few more *A vs B/C/D drift-aware* contrast features that are still just cheap NumPy reductions and directly reflect the ABACAD “present only in A” pattern. Specifically, we compute per-panel frequency-centroid drift across time and summarize how much stronger the drift is in A than in B/C/D, plus a couple of lightweight “top-k contrast” features that emphasize sparse, line-like differences. These are minimal additions (no new model/training loop) and should improve discrimination to move AUC upward toward your 0.756 target. We only update `FEAT_DIM` accordingly and keep all paths and submission writing identical.'
- What this solution (achieved 0.52633) has done: 'Your current AUC (0.52246) is well below the target (0.75636), so we should improve discrimination while keeping the same “cheap features → scaler → poly2 → logistic regression” core intact. The smallest score-relevant change is to (1) make the drift/centroid features invariant to per-snippet global offsets by using per-panel nonnegative weights (subtract each panel’s own min) instead of a single global min, and (2) add two extremely cheap ABACAD-aligned features: a normalized “A vs B/C/D” cross-correlation peak in time and frequency difference profiles, which helps detect line-like structure without changing the model. Everything else (paths, sharding, CV/training loop, model family, submission writing) remains unchanged, and the new features only add a tiny constant amount of compute per file.'
- What this solution (achieved 0.52996) has done: 'Your current AUC (0.52633) is far below the target (0.75636), so we should add a small amount of additional signal-discriminative information while keeping the same “cheap per-file reductions → scaler → poly2 → logistic regression” pipeline intact. The most score-relevant minimal change is to add a few ABACAD-aligned “A-only line strength” features by measuring how much stronger the A−BCD difference image’s strongest time/frequency lines are compared to B/C/D, using only `max/mean/std` reductions (no new model, no new training loop). I also add a tiny set of robust rank-like features (log1p of top-line ratios) to help the same logistic regression separate heavy-tailed patterns without changing evaluation semantics. Everything else (paths, sharding, parallel extraction, CV setup, model family, submission writing) remains unchanged, and `FEAT_DIM` is updated accordingly.'
- What this solution (achieved 0.52456) has done: 'We keep your exact pipeline (feature extraction → StandardScaler → PolynomialFeatures(2) → LogisticRegression) and only make tiny, score-relevant adjustments that typically improve ROC-AUC without changing the core logic. Specifically, we add a small set of “A-only vs B/C/D” features computed from the already-derived A−BCD difference image (a few robust max/mean/std and soft-thresholded energies), which directly aligns with the ABACAD structure and is still just cheap NumPy reductions. We also make a minimal regularization tweak (slightly stronger regularization) to reduce overfitting risk from the added features while keeping the same solver/training loop. Finally, we update `FEAT_DIM` accordingly and keep submission formatting/ordering unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",  # common Kaggle dataset root for this competition
    "/kaggle/input",  # fallback
]
BASE_PATH = next(
    (p for p in BASE_PATH_CANDIDATES if os.path.exists(p)), "/kaggle/input"
)

TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def id_to_path(root_dir: str, _id: str) -> str:
    shard = _id[0].lower()
    return os.path.join(root_dir, shard, f"{_id}.npy")


def extract_features_from_npy(path: str) -> np.ndarray:
    """
    Minimal, fast feature extraction from (6,273,256) float16 snippet.

    Core logic preserved: cheap NumPy reductions + A vs B/C/D contrasts + poly2+logreg.

    Score-relevant minimal change (vs prior version):
      Add a few extremely cheap and robust A-only vs B/C/D features computed from the
      already-available A-BCD difference image. These are simple reductions (mean/std/max
      and soft-thresholded energies) and help ROC-AUC by encoding the ABACAD pattern more directly.
    """
    x = np.load(path, mmap_mode="r")  # (6, 273, 256)
    x = x.astype(np.float32, copy=False)
    eps = np.float32(1e-6)

    mean = x.mean()
    std = x.std()
    mx = x.max()
    mn = x.min()

    a = x[[0, 2, 4]]
    bcd = x[[1, 3, 5]]
    a_mean = a.mean()
    b_mean = bcd.mean()
    a_std = a.std()
    b_std = bcd.std()

    xc = x - mean
    l2 = np.sqrt((xc * xc).mean())

    tprof = x.mean(axis=2)  # (6,273)
    fprof = x.mean(axis=1)  # (6,256)

    t_mean = tprof.mean()
    t_std = tprof.std()
    t_max = tprof.max()

    f_mean = fprof.mean()
    f_std = fprof.std()
    f_max = fprof.max()

    tA = tprof[[0, 2, 4]].mean(axis=0)  # (273,)
    tB = tprof[[1, 3, 5]].mean(axis=0)  # (273,)
    fA = fprof[[0, 2, 4]].mean(axis=0)  # (256,)
    fB = fprof[[1, 3, 5]].mean(axis=0)  # (256,)

    t_diff = tA - tB
    f_diff = fA - fB

    t_diff_l2 = np.sqrt((t_diff * t_diff).mean())
    f_diff_l2 = np.sqrt((f_diff * f_diff).mean())
    t_diff_max = t_diff.max()
    t_diff_min = t_diff.min()
    f_diff_max = f_diff.max()
    f_diff_min = f_diff.min()

    a_over_b_mean = a_mean / (b_mean + eps)
    a_over_b_std = a_std / (b_std + eps)

    abs_contrast_mean = (a_mean - b_mean) / (std + eps)
    tdiff_pos_mass = np.mean(np.maximum(t_diff, 0.0))
    fdiff_pos_mass = np.mean(np.maximum(f_diff, 0.0))
    tA_peak = tA.max() - tA.mean()
    tB_peak = tB.max() - tB.mean()
    fA_peak = fA.max() - fA.mean()
    fB_peak = fB.max() - fB.mean()
    t_peak_diff = tA_peak - tB_peak
    f_peak_diff = fA_peak - fB_peak

    t_diff_abs_mean = np.mean(np.abs(t_diff))
    f_diff_abs_mean = np.mean(np.abs(f_diff))
    t_diff_pos_frac = np.mean(t_diff > 0.0)
    f_diff_pos_frac = np.mean(f_diff > 0.0)

    t_diff_abs_over_t_std = t_diff_abs_mean / (t_std + eps)
    f_diff_abs_over_f_std = f_diff_abs_mean / (f_std + eps)

    t_pos = np.maximum(t_diff, 0.0)
    f_pos = np.maximum(f_diff, 0.0)
    t_pos_max_over_mean = (t_pos.max()) / (tdiff_pos_mass + eps)
    f_pos_max_over_mean = (f_pos.max()) / (fdiff_pos_mass + eps)

    freq_idx = np.arange(x.shape[2], dtype=np.float32)  # (256,)
    x_min_panel = x.min(axis=(1, 2), keepdims=True)  # (6,1,1)
    xf = x - x_min_panel  # non-negative per panel
    tf = xf.sum(axis=2) + eps  # (6,273)
    cent = (xf * freq_idx[None, None, :]).sum(axis=2) / tf  # (6,273)

    t_idx = np.arange(x.shape[1], dtype=np.float32)  # (273,)
    t0 = t_idx - t_idx.mean()
    tvar = (t0 * t0).mean() + eps
    c0 = cent - cent.mean(axis=1, keepdims=True)
    slope = (c0 * t0[None, :]).mean(axis=1) / tvar  # (6,)
    abs_slope = np.abs(slope)

    a_abs_slope_mean = abs_slope[[0, 2, 4]].mean()
    b_abs_slope_mean = abs_slope[[1, 3, 5]].mean()
    slope_contrast = (a_abs_slope_mean - b_abs_slope_mean) / (abs_slope.mean() + eps)

    cent_std = cent.std(axis=1)  # (6,)
    a_cent_std = cent_std[[0, 2, 4]].mean()
    b_cent_std = cent_std[[1, 3, 5]].mean()
    cent_std_contrast = (a_cent_std - b_cent_std) / (cent_std.mean() + eps)

    k_f = 16
    k_t = 16
    f_sorted = np.sort(np.abs(f_diff))
    t_sorted = np.sort(np.abs(t_diff))
    f_topk_mean = f_sorted[-k_f:].mean()
    t_topk_mean = t_sorted[-k_t:].mean()
    f_topk_over_abs = f_topk_mean / (f_diff_abs_mean + eps)
    t_topk_over_abs = t_topk_mean / (t_diff_abs_mean + eps)

    def _norm_corr_peak(u: np.ndarray, v: np.ndarray) -> np.float32:
        u = u.astype(np.float32, copy=False)
        v = v.astype(np.float32, copy=False)
        u = u - u.mean()
        v = v - v.mean()
        un = np.sqrt((u * u).mean()) + eps
        vn = np.sqrt((v * v).mean()) + eps
        u = u / un
        v = v / vn
        n = u.shape[0]
        m = 1 << int(np.ceil(np.log2(2 * n - 1)))
        U = np.fft.rfft(u, m)
        V = np.fft.rfft(v, m)
        corr = np.fft.irfft(U * np.conj(V), m)[: (2 * n - 1)]
        return np.float32(np.max(np.abs(corr)) / np.float32(n))

    t_corr_peak = _norm_corr_peak(tA, tB)
    f_corr_peak = _norm_corr_peak(fA, fB)

    x_centered = x - x.mean(axis=(1, 2), keepdims=True)  # (6,273,256)
    time_line = x_centered.max(axis=2)  # (6,273)
    freq_line = x_centered.max(axis=1)  # (6,256)

    time_line_max = time_line.max(axis=1)  # (6,)
    freq_line_max = freq_line.max(axis=1)  # (6,)

    a_time_line_max = time_line_max[[0, 2, 4]].mean()
    b_time_line_max = time_line_max[[1, 3, 5]].mean()
    a_freq_line_max = freq_line_max[[0, 2, 4]].mean()
    b_freq_line_max = freq_line_max[[1, 3, 5]].mean()

    time_line_ratio = a_time_line_max / (b_time_line_max + eps)
    freq_line_ratio = a_freq_line_max / (b_freq_line_max + eps)
    time_line_contrast = (a_time_line_max - b_time_line_max) / (std + eps)
    freq_line_contrast = (a_freq_line_max - b_freq_line_max) / (std + eps)

    log_time_line_ratio = np.log1p(np.maximum(time_line_ratio - 1.0, 0.0))
    log_freq_line_ratio = np.log1p(np.maximum(freq_line_ratio - 1.0, 0.0))

    a_img = a.mean(axis=0)  # (273,256)
    b_img = bcd.mean(axis=0)  # (273,256)
    d_img = a_img - b_img  # (273,256)

    d_abs = np.abs(d_img)
    d_pos = np.maximum(d_img, 0.0)

    d_mean = d_img.mean()
    d_std = d_img.std()
    d_abs_mean = d_abs.mean()
    d_abs_max = d_abs.max()
    d_pos_mean = d_pos.mean()
    d_pos_max = d_pos.max()

    thr1 = np.float32(0.5) * (std + eps)
    thr2 = np.float32(1.0) * (std + eps)
    d_pos_thr1 = np.maximum(d_img - thr1, 0.0)
    d_pos_thr2 = np.maximum(d_img - thr2, 0.0)
    d_pos_thr1_mean = d_pos_thr1.mean()
    d_pos_thr2_mean = d_pos_thr2.mean()
    d_pos_thr1_l2 = np.sqrt((d_pos_thr1 * d_pos_thr1).mean())
    d_pos_thr2_l2 = np.sqrt((d_pos_thr2 * d_pos_thr2).mean())

    d_time_line = d_img.max(axis=1)  # (273,)
    d_freq_line = d_img.max(axis=0)  # (256,)
    d_time_line_max = d_time_line.max()
    d_freq_line_max = d_freq_line.max()
    d_time_line_mean = d_time_line.mean()
    d_freq_line_mean = d_freq_line.mean()
    d_time_line_max_over_mean = d_time_line_max / (d_time_line_mean + eps)
    d_freq_line_max_over_mean = d_freq_line_max / (d_freq_line_mean + eps)

    return np.array(
        [
            mean,
            std,
            mx,
            mn,
            a_mean,
            b_mean,
            a_std,
            b_std,
            a_mean - b_mean,
            l2,
            t_mean,
            t_std,
            t_max,
            f_mean,
            f_std,
            f_max,
            t_diff_l2,
            f_diff_l2,
            t_diff_max,
            t_diff_min,
            f_diff_max,
            f_diff_min,
            a_over_b_mean,
            a_over_b_std,
            abs_contrast_mean,
            tdiff_pos_mass,
            fdiff_pos_mass,
            t_peak_diff,
            f_peak_diff,
            t_diff_abs_mean,
            f_diff_abs_mean,
            t_diff_pos_frac,
            f_diff_pos_frac,
            t_diff_abs_over_t_std,
            f_diff_abs_over_f_std,
            t_pos_max_over_mean,
            f_pos_max_over_mean,
            a_abs_slope_mean,
            b_abs_slope_mean,
            slope_contrast,
            cent_std_contrast,
            f_topk_over_abs,
            t_topk_over_abs,
            t_corr_peak,
            f_corr_peak,
            time_line_ratio,
            freq_line_ratio,
            time_line_contrast,
            freq_line_contrast,
            log_time_line_ratio,
            log_freq_line_ratio,
            d_mean,
            d_std,
            d_abs_mean,
            d_abs_max,
            d_pos_mean,
            d_pos_max,
            d_pos_thr1_mean,
            d_pos_thr2_mean,
            d_pos_thr1_l2,
            d_pos_thr2_l2,
            d_time_line_max,
            d_freq_line_max,
            d_time_line_max_over_mean,
            d_freq_line_max_over_mean,
        ],
        dtype=np.float32,
    )




## === cell 3
from concurrent.futures import ThreadPoolExecutor


def _featurize_one(path: str) -> np.ndarray:
    return extract_features_from_npy(path)


train_ids = train_labels["id"].values
train_y = train_labels["target"].values

train_paths = [id_to_path(TRAIN_DIR, _id) for _id in train_ids]
exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_paths), count=len(train_paths), dtype=bool
)

missing = int((~exists_mask).sum())
if missing > 0:
    miss_examples = [p for p, ok in zip(train_paths, exists_mask) if not ok][:5]
    raise FileNotFoundError(
        f"Missing {missing} training files (out of {len(train_paths)}). "
        f"Examples: {miss_examples}. "
        f"Likely shard/path mismatch; fix BASE_PATH/TRAIN_DIR before training."
    )

train_paths_ok = train_paths
y_ok = train_y

N_OK = len(train_paths_ok)
FEAT_DIM = 65
X = np.empty((N_OK, FEAT_DIM), dtype=np.float32)

max_workers = min(32, (os.cpu_count() or 4) * 2)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feat in enumerate(ex.map(_featurize_one, train_paths_ok, chunksize=128)):
        X[i] = feat

y = np.array(y_ok, dtype=np.int64)

X.shape, y.shape, y.mean()



## === cell 4
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

SEED = 42

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

oof = np.zeros(len(y), dtype=np.float32)


def make_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    n_jobs=None,
                    random_state=SEED,
                    C=1.0,
                    class_weight="balanced",
                    warm_start=True,
                ),
            ),
        ]
    )


for tr_idx, va_idx in skf.split(X, y):
    model_fold = make_model()
    model_fold.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = model_fold.predict_proba(X[va_idx])[:, 1].astype(np.float32)

model = make_model()
model.fit(X, y)



## === cell 5
test_ids = sample_sub["id"].values
test_paths = [id_to_path(TEST_DIR, _id) for _id in test_ids]
exists_mask_t = np.fromiter(
    (os.path.exists(p) for p in test_paths), count=len(test_paths), dtype=bool
)

missing_test = int((~exists_mask_t).sum())
if missing_test > 0:
    miss_examples_t = [p for p, ok in zip(test_paths, exists_mask_t) if not ok][:5]
    raise FileNotFoundError(
        f"Missing {missing_test} test files (out of {len(test_paths)}). "
        f"Examples: {miss_examples_t}. "
        f"Likely shard/path mismatch; fix BASE_PATH/TEST_DIR before creating submission."
    )

X_test = np.empty((len(test_ids), X.shape[1]), dtype=np.float32)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, feat in enumerate(ex.map(_featurize_one, test_paths, chunksize=128)):
        X_test[i] = feat

X_test.shape



## === cell 6
test_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "target": test_pred.astype(float)})
assert submission.shape[0] == sample_sub.shape[0]
assert (submission["id"].values == sample_sub["id"].values).all()

submission.to_csv("submission.csv", index=False)
submission.head(), submission.shape
