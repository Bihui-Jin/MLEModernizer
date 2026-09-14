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

0.7571313712626022

# 6. Current score

0.49885

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The runtime errors come from trying to read ensemble submissions from `../input/...` paths that don’t exist in this environment; this prevents `data1..data6` from ever being defined and stops the notebook before writing `submission.csv`. I keep the ensemble-blending core logic intact, but make the code robust by dynamically finding whichever `submission.csv` files are actually available under `/kaggle/input` (and falling back to the provided `sample_submission.csv` if none are found). I also enforce correct `id` alignment across blended files (merge on `id`) so the blend is logically correct and produces a valid `id,target` submission. Finally, it always write `submission.csv` with the correct header and 6000 rows.'
- What this solution (achieved 0.5) has done: 'Your current score of 0.5 indicates the blend is effectively producing near-constant predictions (or repeatedly blending the sample submission), which yields random AUC. I keep your ensemble-blending structure intact, but (1) only blend *distinct* submission files (skip duplicates/near-constant ones), (2) rank candidate submissions by how “non-constant” their predictions are (higher std) so the blend is more informative, and (3) fall back to a simple, deterministic transformation (stretch around 0.5) that can help AUC when predictions are overly squashed, without changing the overall semantics (still probabilities in [0,1]). These changes are minimal, still produce a valid `submission.csv`, and should move the score upward toward the target band if any real model submissions exist in the environment.'
- What this solution (achieved 0.49545) has done: 'Your current 0.5 AUC strongly suggests you are not actually blending any meaningful model outputs (likely only the `sample_submission.csv` with constant-ish 0.5s), so the smallest score-improving change is to replace the “fallback to sample submission” with a legitimate, lightweight model that produces non-constant predictions. I keep the ensemble/blending core logic intact when real `submission.csv` files exist, but add a deterministic fallback that trains a simple logistic regression on fast-to-compute handcrafted features from the `.npy` snippets. This preserves evaluation semantics (still outputs probabilities for `target`) and should move AUC upward toward your ~0.757 target without changing architecture/training loops (there were none before). I also ensure strict `id` alignment with the sample submission and always write a valid `submission.csv` with 6000 rows.'
- What this solution (achieved 0.49731) has done: 'The timeout is dominated by the fallback path: it enumerates all train IDs and then loads tens of thousands of `.npy` files and computes several expensive statistics (especially `np.quantile`) in Python/NumPy per file, plus multiprocessing overhead from per-file task dispatch. To finish within 600s without changing model/feature semantics, I (1) eliminate the expensive submission-file directory scan by directly loading `sample_submission.csv` from known locations, (2) make feature extraction provably equivalent but faster by replacing `np.quantile(..., 0.99)` with `np.partition`-based exact order-statistic selection and by avoiding repeated reductions, and (3) drastically reduce I/O/multiprocessing overhead by using a `ThreadPoolExecutor` (NumPy loads release the GIL) with larger chunking and by avoiding building large Python sets of available IDs via full directory listing when we can just trust labels and check existence on demand. These changes preserve the same features, same logistic regression, and the same training/evaluation semantics, but cut constant factors and overhead enough to meet the 10-minute hard limit.'
- What this solution (achieved 0.49144) has done: 'Your current AUC (~0.497) is far below the target (~0.757), so we should cautiously improve the fallback model rather than tuning the ensemble blend. The smallest legitimate score-lift (without changing the overall “handcrafted features + logistic regression” core) is to fix a likely path bug in `id_to_npy_path` (the shard folders are `0..15`, not the first character of the id), because a wrong path makes many feature rows all-zeros and drives predictions toward ~0.5. I also remove the `nonzero_mask` drop (keep samples with missing files but mark them explicitly) and instead add a simple “missing file” indicator feature so the model can learn to ignore those rows rather than silently biasing the training set. These are minimal, metric-aligned changes that should move AUC upward toward the target while keeping the same model, training approach, and submission semantics.'
- What this solution (achieved 0.5) has done: 'Your current AUC (~0.49) is far below the target (~0.757), so we should improve the fallback model while keeping the same “handcrafted features + logistic regression” core. The biggest likely cause is that `id_to_npy_path` uses a hex-hash shard, but the dataset on disk is sharded by the **first character** of the id (`0..f`), so many files are missed and feature rows become “missing”, pushing predictions toward ~0.5. I fix the shard mapping to use `id_[0].lower()` and add a tiny, deterministic calibration step using the validation split to choose the best linear “stretch around 0.5” (kept within [0,1]) to better match AUC without changing the model/feature semantics. Everything else (features, logistic regression, submission format, paths) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I keep your ensemble/fallback structure intact, but fix two issues that are likely keeping AUC near 0.5: (1) your feature vector is mis-sized (13 real features were being written into a 14-column matrix with an overlap), and (2) the directory sharding for the `.npy` files is `0..15` (not hex `0..f`), so many files may be treated as “missing” and predictions collapse toward 0.5. Concretely, I make `X` have 13 feature columns + 1 missing-indicator (14 total) and write the 13 computed features into `X[i, :13]` correctly. I also change `id_to_npy_path` to resolve the correct shard by checking for existence across `0..15` (with a tiny cache), which is minimal but prevents mass-missing features and should move AUC upward toward your ~0.757 target. Everything else (model, training loop, blending, submission writing) stays the same.'
- What this solution (achieved 0.5) has done: 'Your AUC is stuck near 0.5 because the fallback feature extractor is likely treating most `.npy` files as “missing” due to incorrect shard resolution (it checks `0..15` and `a..f`, but the dataset is sharded by **first character** `0..f`). I make `id_to_npy_path` deterministically map to `base_dir/<first_hex_char>/<id>.npy` (and verify existence), which should dramatically reduce missing rows and move predictions upward toward the 0.757 target without changing the model or features. I also remove the expensive/unused full-directory scan (`available_ids_in_dir`) to save time, and keep the rest (features, logistic regression, blending, submission writing) identical. The output still be a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.49539) has done: 'Your current AUC (0.5) is far below the target (~0.757), so the smallest credible lift is to make the fallback model actually see the correct `.npy` files and produce non-constant predictions. The main issue is `id_to_npy_path`: this competition’s folders are sharded by **0..15** (not hex `0..f`), so mapping by first hex character marks many files missing and collapses predictions toward 0.5. I change `id_to_npy_path` to deterministically map shards using `int(first_hex_char, 16)` (0–15) and keep a small existence-based fallback, leaving the feature set, logistic regression, blending, and submission formatting unchanged. This should materially increase AUC toward the target while keeping core logic intact and runtime within limits.'
- What this solution (achieved 0.5) has done: 'Your fallback model is likely still producing near-random AUC because many `.npy` files are being treated as missing due to an incorrect shard mapping: this dataset is sharded by the *first hex character* (`0..f`), not by `int(first_char,16)` (`0..15`). I make `id_to_npy_path` map deterministically to `base_dir/<first_hex_char>/<id>.npy` (with a tiny cached existence fallback), which should drastically reduce missing features and move predictions upward toward your target AUC. I also add a quick runtime check that reports the missing-rate on a small training subset (no change to training logic), and keep the ensemble/blend logic, features, logistic regression pipeline, and submission writing semantics unchanged.'
- What this solution (achieved 0.49885) has done: 'I keep your overall structure (blend if real submissions exist; otherwise train the same logistic-regression-on-handcrafted-features fallback) but fix a score-critical data loading bug: the `train/` and `test/` folders here are sharded by `0..15` (decimal), not hex characters, so many `.npy` files are currently treated as missing and predictions collapse toward ~0.5 AUC. I update `id_to_npy_path` to deterministically map to `base_dir/<int(first_hex_char,16)>/<id>.npy` and keep a tiny cached fallback existence check, which should sharply reduce missing-rate while preserving your exact feature set and model. I also make the final “stretch” step conditional: keep your existing stretch when blending external submissions, but avoid extra distortion in the fallback path (since you already calibrate `best_s` on a validation split), which should improve AUC stability upward toward the target. The script still run end-to-end within constraints and always write a valid `submission.csv` aligned to `sample_submission.csv`.'

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

INPUT_ROOTS = ["../input", "/kaggle/input", "/kaggle/data"]


def _existing_dirs(paths):
    return [p for p in paths if os.path.isdir(p)]


def find_submission_files(max_files=50):
    """
    Find candidate submission.csv files under Kaggle input mounts.
    Strictly bounded to avoid spending significant time scanning large directory trees.
    """
    subs = []
    for root in _existing_dirs(INPUT_ROOTS):
        patterns = [
            os.path.join(root, "*", "submission.csv"),
            os.path.join(root, "*", "*", "submission.csv"),
        ]
        for pat in patterns:
            for p in glob.iglob(pat):
                subs.append(p)
                if len(subs) >= max_files:
                    break
            if len(subs) >= max_files:
                break
        if len(subs) >= max_files:
            break

    seen = set()
    subs_unique = []
    for p in subs:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            subs_unique.append(p)
    return subs_unique


def load_sample_submission():
    candidates = [
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )


def read_submission(path):
    df = pd.read_csv(path)
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"{path} does not have required columns id,target. Columns: {df.columns.tolist()}"
        )
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        df["target"] = df["target"].fillna(0.5)
    return df


def align_to_sample(df, sample):
    out = sample[["id"]].merge(df, on="id", how="left")
    out["target"] = out["target"].fillna(0.5).astype(float)
    return out


def find_labels_csv():
    candidates = [
        "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
        "/kaggle/data/train_labels.csv",
        "/kaggle/input/train_labels.csv",
        "../input/train_labels.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find train_labels.csv in expected locations.")


def find_data_dir(kind):
    """
    kind in {'train','test'}; return directory containing shard folders.
    """
    candidates = [
        f"/kaggle/input/seti-breakthrough-listen/{kind}",
        f"/kaggle/data/{kind}",
        f"/kaggle/input/{kind}",
        f"../input/{kind}",
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Could not find {kind}/ directory in expected locations.")


_ID_SHARD_CACHE = {}


def id_to_npy_path(base_dir, id_):
    """
    Change (score-critical): in this environment the snippet files are sharded into folders "0".."15"
    (decimal), where shard = int(first_hex_char_of_id, 16). If we instead use hex folders ("a","b"...),
    most files are treated as missing, features become mostly zeros + missing-indicator, and AUC collapses.

    We deterministically map to base_dir/<0..15>/<id>.npy and keep a small cached existence fallback.
    """
    id_ = str(id_)
    if id_ in _ID_SHARD_CACHE:
        shard = _ID_SHARD_CACHE[id_]
        return os.path.join(base_dir, str(shard), f"{id_}.npy")

    first = id_[0].lower()
    try:
        shard = int(first, 16)
    except ValueError:
        shard = 0

    p = os.path.join(base_dir, str(shard), f"{id_}.npy")
    if os.path.exists(p):
        _ID_SHARD_CACHE[id_] = shard
        return p

    for sh in range(16):
        pp = os.path.join(base_dir, str(sh), f"{id_}.npy")
        if os.path.exists(pp):
            _ID_SHARD_CACHE[id_] = sh
            return pp

    _ID_SHARD_CACHE[id_] = shard
    return p


def _p99_axis1_exact_for_256(a_2d):
    idx = 252
    part = np.partition(a_2d, idx, axis=1)
    return part[:, idx]


def _features_from_array(x):
    """
    Minimal handcrafted features for SETI snippets.
    x shape: (6, 273, 256).
    Returns 13 features.
    """
    x = x.astype(np.float32, copy=False)

    mean_all = float(x.mean())
    std_all = float(x.std())

    panel_mean = x.mean(axis=(1, 2))  # (6,)
    panel_std = x.std(axis=(1, 2))  # (6,)

    A = x[[0, 2, 4]]
    B = x[[1, 3, 5]]

    mean_A = float(panel_mean[[0, 2, 4]].mean())
    mean_B = float(panel_mean[[1, 3, 5]].mean())
    std_A = float(A.std())
    std_B = float(B.std())

    prof = x.mean(axis=1)  # (6, 256)
    prof_mean = prof.mean(axis=1)  # (6,)
    prof_max = prof.max(axis=1)  # (6,)

    prof_p99 = _p99_axis1_exact_for_256(prof)  # (6,)

    time_std = x.std(axis=1).mean(axis=1)  # (6,)

    a_means = panel_mean[[0, 2, 4]]
    b_means = panel_mean[[1, 3, 5]]
    a_stds = panel_std[[0, 2, 4]]
    b_stds = panel_std[[1, 3, 5]]

    a_mean_disp = float(a_means.std(ddof=0))
    b_mean_disp = float(b_means.std(ddof=0))
    a_std_disp = float(a_stds.std(ddof=0))
    b_std_disp = float(b_stds.std(ddof=0))

    feats = np.array(
        [
            mean_all,  # 0
            std_all,  # 1
            mean_A - mean_B,  # 2
            std_A - std_B,  # 3
            float(np.abs(mean_A - mean_B)),  # 4
            float(prof_mean.mean()),  # 5
            float(prof_max.mean()),  # 6
            float(prof_p99.mean()),  # 7
            float(time_std.mean()),  # 8
            a_mean_disp,  # 9
            b_mean_disp,  # 10
            a_std_disp,  # 11
            b_std_disp,  # 12
        ],
        dtype=np.float32,
    )
    return feats


def _features_from_id(args):
    id_, base_dir = args
    fp = id_to_npy_path(base_dir, id_)
    try:
        arr = np.load(fp, mmap_mode=None, allow_pickle=False)
    except FileNotFoundError:
        return id_, None
    return id_, _features_from_array(arr)


def make_features(ids, base_dir, n_jobs=None, chunksize=128):
    X = np.zeros((len(ids), 14), dtype=np.float32)
    if len(ids) == 0:
        return X

    if n_jobs is None:
        n_jobs = max(1, min(8, (os.cpu_count() or 2)))

    def _set_row(i, feats, missing):
        if feats is not None:
            X[i, :13] = feats  # 13 computed features
        X[i, 13] = 1.0 if missing else 0.0  # missing indicator

    if n_jobs == 1 or len(ids) < 512:
        for i, id_ in enumerate(ids):
            fp = id_to_npy_path(base_dir, id_)
            if not os.path.exists(fp):
                _set_row(i, None, True)
                continue
            arr = np.load(fp, mmap_mode=None, allow_pickle=False)
            _set_row(i, _features_from_array(arr), False)
        return X

    from concurrent.futures import ThreadPoolExecutor

    id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for start in range(0, len(ids), chunksize):
            batch = ids[start : start + chunksize]
            for id_, feats in ex.map(_features_from_id, [(i, base_dir) for i in batch]):
                i = id_to_idx[id_]
                if feats is None:
                    _set_row(i, None, True)
                else:
                    _set_row(i, feats, False)
    return X




## === cell 1
sample_sub = load_sample_submission()

submission_files = find_submission_files()
loaded = []
for p in submission_files:
    try:
        df = read_submission(p)
        if len(df) == len(sample_sub) and df["id"].is_unique:
            df = align_to_sample(df, sample_sub)
            loaded.append((p, df))
    except Exception:
        continue

uniq = []
seen_fp = set()
for p, df in loaded:
    fp = pd.util.hash_pandas_object(df["target"].round(6), index=False).sum()
    if fp not in seen_fp:
        seen_fp.add(fp)
        uniq.append((p, df))

filtered = []
for p, df in uniq:
    std = float(df["target"].std(ddof=0))
    if np.isfinite(std) and std > 1e-4:
        filtered.append((p, df, std))

filtered.sort(key=lambda x: x[2], reverse=True)

USE_FALLBACK_MODEL = len(filtered) == 0




## === cell 2
if USE_FALLBACK_MODEL:
    labels_path = find_labels_csv()
    train_dir = find_data_dir("train")
    test_dir = find_data_dir("test")

    train_labels = pd.read_csv(labels_path)
    train_labels["id"] = train_labels["id"].astype(str)

    _probe_ids = train_labels["id"].iloc[:512].tolist()
    _probe_missing = sum(
        1 for _id in _probe_ids if not os.path.exists(id_to_npy_path(train_dir, _id))
    )
    print(f"[probe] train missing-rate (first 512): {_probe_missing}/512")

    X = make_features(
        train_labels["id"].tolist(), train_dir, n_jobs=None, chunksize=256
    )
    y = train_labels["target"].astype(int).values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=300, solver="lbfgs", n_jobs=None)),
        ]
    )
    clf.fit(X_tr, y_tr)

    from sklearn.metrics import roc_auc_score

    va_pred = clf.predict_proba(X_va)[:, 1].astype(np.float64)
    best_s, best_auc = 1.0, roc_auc_score(y_va, va_pred)
    for s in (0.8, 1.0, 1.25, 1.5, 1.75):
        p = 0.5 + s * (va_pred - 0.5)
        p = np.clip(p, 0.0, 1.0)
        auc = roc_auc_score(y_va, p)
        if auc > best_auc + 1e-6:
            best_auc, best_s = auc, s

    test_ids = sample_sub["id"].astype(str).tolist()
    X_test = make_features(test_ids, test_dir, n_jobs=None, chunksize=256)
    preds = clf.predict_proba(X_test)[:, 1].astype(np.float64)

    preds = 0.5 + best_s * (preds - 0.5)
    preds = np.clip(preds, 0.0, 1.0)

    data1 = sample_sub.copy()
    data1["target"] = preds
    data2 = data1.copy()
    data3 = data1.copy()
    data4 = data1.copy()
    data5 = data1.copy()
    data6 = data1.copy()
else:
    dfs = [d for _, d, _ in filtered]
    while len(dfs) < 6:
        dfs.append(dfs[-1].copy())
    data1, data2, data3, data4, data5, data6 = dfs[:6]




## === cell 3
data6 = data6.copy()
data6["target"] = (
    0.75 * data5["target"]
    + 0.15 * data4["target"]
    + 0.10 * data6["target"]
    + 0.00 * data2["target"]
    + 0.00 * data3["target"]
)

if not USE_FALLBACK_MODEL:
    stretch = 1.25
    data6["target"] = 0.5 + stretch * (data6["target"] - 0.5)
    data6["target"] = data6["target"].clip(0.0, 1.0)




## === cell 4
data6[["id", "target"]].to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv"), "submission.csv was not created."
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == [
    "id",
    "target",
], f"Bad submission columns: {sub_check.columns.tolist()}"
assert len(sub_check) == len(
    sample_sub
), f"Bad submission row count: {len(sub_check)} != {len(sample_sub)}"
assert (
    sub_check["id"].astype(str).tolist() == sample_sub["id"].astype(str).tolist()
), "ID order misalignment vs sample_submission.csv"
