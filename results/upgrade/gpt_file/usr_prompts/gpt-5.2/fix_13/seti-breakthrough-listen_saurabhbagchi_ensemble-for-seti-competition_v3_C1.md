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

0.7275326111973154

# 6. Current score

0.4927

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50575) has done: 'Main bottlenecks are (1) extremely heavy disk I/O from loading tens of thousands of small `.npy` files, (2) multiprocessing overhead (pickling + process startup) dominating because each task is small, and (3) extra filesystem scans and Python-loop overhead. To fit in 600s without changing the feature logic or model, the script below (a) avoids scanning the filesystem by deriving file paths directly from ids (the directory is the first hex char), (b) uses a thread pool (I/O-bound) instead of process pool to cut overhead while keeping parallelism, (c) keeps `np.load(..., mmap_mode='r')` + `astype(copy=False)` to minimize memory copies, and (d) uses chunked mapping with preallocated output arrays and direct index assignment. These are provably equivalent wrt features/model outputs (up to negligible float differences) and preserve all paths, training split, and LogisticRegression semantics.'
- What this solution (achieved 0.50575) has done: 'Your score is far below the target, so the smallest safe way to move AUC upward (without changing the model or feature definitions) is to fix a subtle but impactful bug: you sort `train_labels` after filtering missing files, but you don’t rebuild `train_paths` in the same sorted order, so features can be misaligned with labels and tank AUC. I rebuild `train_paths` from the sorted `train_labels` and also add a quick alignment check (same id order) to prevent regressions. Additionally, I make the test feature extraction loop index-safe by iterating `zip(present_idx, ex.map(...))` (avoids any potential mismatch if behavior changes), keeping semantics identical. These minimal changes should materially improve AUC while preserving your feature logic, pipeline, and output format.'
- What this solution (achieved 0.50575) has done: 'Your current score is near random, so the most likely cause is a train/test mismatch rather than weak modeling. The dataset on disk is stored under `/kaggle/input/seti-breakthrough-listen/{train,test}/...`, but your code points to `/kaggle/input/{train,test}`, which silently load the wrong/partial files and break the relationship between features and labels, collapsing AUC. I make the smallest possible change: auto-detect the correct competition root directory and keep everything else (feature extraction, LogisticRegression pipeline, and submission format) identical. I also add a strict id↔path alignment assertion to prevent accidental misalignment regressions.'
- What this solution (achieved 0.50575) has done: 'Your AUC is near random, so the highest-impact minimal fix is to ensure the model sees the same “cadence semantics” as the competition expects: the 6 panels are ABACAD, so “A” is indices [0,2,4] and “non‑A” is [1,3,5]; your current code mistakenly treats [0,2,4] vs [1,3,5] as A vs B, which mixes C/D into B and destroys signal. I change only the feature extraction grouping to use A vs non‑A correctly while keeping the same summary-stat features, model, training loop, and submission format. I also keep the existing id↔path alignment safeguards and I/O approach unchanged so runtime stays within limits and behavior remains stable aside from the intended semantic correction. This should move AUC upward toward the target without changing the overall pipeline.'
- What this solution (achieved 0.50421) has done: 'Your current AUC is still near-random, so the remaining high-impact issue is likely a semantic mismatch in feature extraction rather than model strength. We keep the same summary-stat feature set and LogisticRegression pipeline, but fix the A-vs-off cadence comparison: subtracting `A - off` directly is wrong because `A` has 3 panels and `off` has 3 panels, but they are not paired; instead we should compare `A` to the **mean off-target panel** (broadcast), which preserves the intended “on-target vs off-target” signal without changing the overall approach. Additionally, we make the feature shape self-check strict (assert length equals `n_feats`) to prevent silent mis-scoring from any future mismatch, while keeping I/O paths and submission format unchanged.'
- What this solution (achieved 0.50552) has done: 'Your AUC is still near-random vs the target, which strongly suggests the “on-target vs off-target” signal is being washed out by an averaging choice rather than the classifier itself. Keeping the same summary-stat feature set and the same LogisticRegression pipeline, I change only the diff computation to compare the *mean A panel* against the *mean off-target panel* (both are (273,256)), which preserves cadence semantics while reducing variance from mixing three A panels at once. I also add a tiny amount of regularization calibration (`class_weight="balanced"`) to reduce bias from class imbalance without changing the model family/training loop. Everything else (paths, I/O strategy, feature count, submission writing) remains identical and the script still produces `submission.csv`.'
- What this solution (achieved 0.49451) has done: 'Your current AUC is still near-random versus the target, so the most likely remaining issue is that the feature extraction is not capturing the competition’s key cadence structure (ABACAD) strongly enough. Keeping the same overall pipeline (summary-stat features → StandardScaler → LogisticRegression) and the same training loop, I add a minimal set of cadence-consistency features that compare each A panel to the immediately adjacent off-target panels (AB, AC, AD pairs) and summarize those diffs with the same mean/std/max/min style stats. This preserves the core “on-target vs off-target summary” logic while better matching the intended signal behavior (needles should be present in A and absent in neighboring off panels). I keep submission formatting unchanged and update `n_feats` accordingly with strict shape checks so the run remains stable and produces `submission.csv`.'
- What this solution (achieved 0.49451) has done: 'We keep your feature extraction and LogisticRegression pipeline intact, but fix two small performance/quality issues that can keep AUC near-random: (1) ensure that any missing test files don’t silently become “all-zero” feature rows that the model treats as meaningful (we instead fill them with the training-feature mean, a neutral in-distribution imputation), and (2) add a tiny amount of regularization stabilization by increasing `max_iter` and setting a deterministic `tol` (same solver/model, just more reliable convergence). These changes don’t alter the core modeling approach or feature definitions, but they reduce harmful artifacts and should move AUC upward toward your target. The script still runs end-to-end within constraints and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.49267) has done: 'Your current AUC is far below the target, so we should make the smallest change that’s likely to improve generalization without changing your feature logic or model family. The most probable issue is that `class_weight="balanced"` is over-correcting for only mild imbalance in this competition, pushing probabilities toward extremes and hurting ROC ranking; we revert to the default (`class_weight=None`) while keeping the exact same LogisticRegression + StandardScaler pipeline and training flow. To keep convergence stable and deterministic, we leave `max_iter/tol/solver/random_state` as-is. Everything else (data root detection, feature extraction, I/O strategy, and submission writing) remains unchanged.'
- What this solution (achieved 0.4927) has done: 'Your current AUC is far below the target, so we should make the smallest change likely to improve ranking without altering your feature set or model family. The most probable low-risk issue here is solver convergence/suboptimality: `lbfgs` can be sensitive to feature scaling and class imbalance on large-but-low-dim data, and `liblinear` often gives a better-behaved optimum for plain LogisticRegression at this feature count. I switch only the LogisticRegression solver to `liblinear` (still LogisticRegression, same training flow, same loss/semantics) and keep `max_iter/tol/random_state` deterministic. Everything else (data root detection, feature extraction, I/O parallelism, missing-test handling, and submission writing) stays identical so runtime remains within limits and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

BASE_INPUT = "/kaggle/input"
CANDIDATE_ROOTS = [
    os.path.join(BASE_INPUT, "seti-breakthrough-listen"),
    BASE_INPUT,
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if (
        os.path.exists(os.path.join(r, "train_labels.csv"))
        and os.path.isdir(os.path.join(r, "train"))
        and os.path.isdir(os.path.join(r, "test"))
    ):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root with train_labels.csv, train/, and test/ under /kaggle/input."
    )

TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

assert os.path.exists(TRAIN_LABELS_PATH), f"Missing: {TRAIN_LABELS_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(RANDOM_STATE)

DATA_ROOT, TRAIN_LABELS_PATH, TRAIN_DIR, TEST_DIR



## === cell 1
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert set(train_labels.columns) == {"id", "target"}
assert set(sample_sub.columns) == {"id", "target"}
assert train_labels["target"].isin([0, 1]).all()

train_labels.head(), sample_sub.head()




## === cell 2
def path_from_id(root_dir: str, _id: str) -> str:
    return os.path.join(root_dir, _id[0], f"{_id}.npy")


train_ids_all = train_labels["id"].astype(str).values
train_paths_all = [path_from_id(TRAIN_DIR, _id) for _id in train_ids_all]

exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_paths_all), count=len(train_paths_all), dtype=bool
)

train_labels = train_labels.loc[exists_mask].copy()

train_labels = train_labels.sort_values("id").reset_index(drop=True)
train_paths = [
    path_from_id(TRAIN_DIR, _id) for _id in train_labels["id"].astype(str).values
]

assert len(train_paths) == len(train_labels)
assert all(
    os.path.basename(p) == f"{_id}.npy"
    for p, _id in zip(train_paths[:200], train_labels["id"].astype(str).values[:200])
), "Train path/id misalignment detected."

missing_on_disk = 54000 - len(train_labels)
missing_on_disk




## === cell 3
def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    arr shape: (6, 273, 256), float16/float32
    Summary features per cadence.

    Keeps existing global-per-panel stats (mean/std/max/min),
    keeps A-vs-off mean-image diff stats,
    keeps cadence-aligned pairwise diff summaries for AB, AC, AD.
    """
    x = arr.astype(np.float32, copy=False)

    m = x.mean(axis=(1, 2))
    s = x.std(axis=(1, 2))
    mx = x.max(axis=(1, 2))
    mn = x.min(axis=(1, 2))

    A = x[[0, 2, 4]]
    off = x[[1, 3, 5]]

    A_mean = A.mean()
    off_mean = off.mean()
    A_std = A.std()
    off_std = off.std()
    A_max = A.max()
    off_max = off.max()

    A_avg = A.mean(axis=0)  # (273, 256)
    off_avg = off.mean(axis=0)  # (273, 256)
    diff = A_avg - off_avg  # (273, 256)

    diff_mean = diff.mean()
    diff_std = diff.std()
    diff_max = diff.max()
    diff_min = diff.min()

    dAB = x[0] - x[1]
    dAC = x[2] - x[3]
    dAD = x[4] - x[5]

    pair_stats = np.array(
        [
            dAB.mean(),
            dAB.std(),
            dAB.max(),
            dAB.min(),
            dAC.mean(),
            dAC.std(),
            dAC.max(),
            dAC.min(),
            dAD.mean(),
            dAD.std(),
            dAD.max(),
            dAD.min(),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [
            m,
            s,
            mx,
            mn,
            np.array(
                [
                    A_mean,
                    off_mean,
                    A_std,
                    off_std,
                    A_max,
                    off_max,
                    diff_mean,
                    diff_std,
                    diff_max,
                    diff_min,
                ],
                dtype=np.float32,
            ),
            pair_stats,
        ],
        axis=0,
    )
    return feats


tmp = np.load(path_from_id(TRAIN_DIR, train_labels["id"].iloc[0]))
extract_features(tmp).shape



## === cell 4
from concurrent.futures import ThreadPoolExecutor


def _features_from_path(p: str) -> np.ndarray:
    return extract_features(np.load(p, mmap_mode="r"))


n_feats = 6 * 4 + 10 + 12
X = np.zeros((len(train_labels), n_feats), dtype=np.float32)
y = train_labels["target"].astype(np.int8).values

cpu_cnt = os.cpu_count() or 2
n_workers = min(16, max(4, cpu_cnt))


def _fill_features(paths, out_X, max_workers: int):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in enumerate(ex.map(_features_from_path, paths, chunksize=64)):
            if feats.shape[0] != out_X.shape[1]:
                raise ValueError(
                    f"Feature length mismatch at row {i}: got {feats.shape[0]}, expected {out_X.shape[1]}"
                )
            out_X[i] = feats


_fill_features(train_paths, X, n_workers)

X.shape, y.shape, float(y.mean())



## === cell 5
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                class_weight=None,
                max_iter=2000,
                tol=1e-4,
                solver="liblinear",
                n_jobs=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)



## === cell 6
X_test = np.zeros((len(sample_sub), X.shape[1]), dtype=np.float32)

test_ids = sample_sub["id"].astype(str).values
test_paths = [path_from_id(TEST_DIR, _id) for _id in test_ids]

exists_mask_test = np.fromiter(
    (os.path.exists(p) for p in test_paths), count=len(test_paths), dtype=bool
)
missing_test = int((~exists_mask_test).sum())

present_idx = np.flatnonzero(exists_mask_test)
present_paths = [test_paths[i] for i in present_idx.tolist()]

if len(present_paths):
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, feats in zip(
            present_idx.tolist(),
            ex.map(_features_from_path, present_paths, chunksize=64),
        ):
            if feats.shape[0] != X_test.shape[1]:
                raise ValueError(
                    f"Test feature length mismatch at idx {idx}: got {feats.shape[0]}, expected {X_test.shape[1]}"
                )
            X_test[idx] = feats

if missing_test > 0:
    X_mean = X.mean(axis=0, dtype=np.float32)
    miss_idx = np.flatnonzero(~exists_mask_test)
    X_test[miss_idx] = X_mean

missing_test



## === cell 7
test_pred = clf.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_sub["id"].values, "target": test_pred})
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape
