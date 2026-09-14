# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7625199137062958

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.506) has done: 'The timeout is dominated by slow feature extraction over tens of thousands of small `.npy` files, amplified by ProcessPool overhead and repeated `np.partition` calls per sample. I keep the exact same features/model/training loop, but make extraction faster by (1) removing per-sample array copies, (2) computing the 90/95/99 percentiles with a single `np.partition` call using `kth=` (same exact semantics), (3) using a faster thread-based parallel loader (NumPy releases the GIL in these ops; avoids heavy process serialization), and (4) caching features with a dataset fingerprint so train/test caches don’t collide and are safely reused. All paths, folds, LogisticRegression pipeline, and feature definitions remain unchanged.'
- What this solution (achieved 0.50437) has done: 'I keep your feature set and LogisticRegression pipeline unchanged, but fix one subtle issue that likely holds AUC near random: your current feature extraction cancels much of the signal by averaging A panels together and B panels together before differencing. Without changing the model/training loop, I compute the same statistics on the *per-panel* (A1-B1, A2-B2, A3-B3) differences and then aggregate, which matches the cadence structure and typically improves discrimination. I also add an out-of-fold ROC-AUC print to verify the direction of improvement locally without affecting submission semantics. Everything else (paths, caching, CV, final fit, submission format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.51531) has done: 'I keep your feature definitions and LogisticRegression pipeline intact, but fix a likely AUC-killer: the features currently compute percentiles and row/col maxima on the *mean* absolute A-B difference across the 3 A/B pairs, which can wash out pair-specific needles; instead we compute those same stats per pair and then aggregate (mean/std) so the model can “see” inconsistent presence across A panels without changing the model or training loop. I also remove the last remaining avoidable copy in percentile computation by partitioning the input array in-place (safe because we pass in a fresh per-pair view) to speed up extraction without altering semantics. Finally, I bump the cache tag to avoid mixing old and new feature layouts while keeping paths and submission format unchanged.'
- What this solution (achieved 0.51695) has done: 'Your current score (0.51531) is far below the target (0.76252), so we should cautiously improve AUC without changing the overall modeling approach (still: handcrafted features + StandardScaler + LogisticRegression). The smallest likely win is to align the features more directly with the cadence structure by adding a few additional per-pair “coherence” summaries (how similar the three A-B pairs are), which often helps AUC while preserving the same extraction/training semantics. I also switch LogisticRegression to a slightly more appropriate regularization strength via a tiny CV over `C` inside the existing StratifiedKFold loop (same model, same training approach), selecting the best mean-fold AUC and then fitting once on all data. All paths, caching, submission format, and core pipeline remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.51907) has done: 'Your current AUC (0.51695) is far below the target (0.76252), so we should make the smallest change that legitimately improves signal extraction without altering the overall “handcrafted features + StandardScaler + LogisticRegression” approach. The main likely issue is that the features only look at absolute A–B differences, which discards the sign pattern that can help separate true “on-target only” needles from common RFI; we add a parallel signed-difference feature block (same summaries) while keeping the existing absolute-diff block intact. This is a minimal extension to feature extraction (no change to model family, training loop structure, or loss), and it typically yields a meaningful AUC lift for this competition. We bump the cache tag to avoid mixing old/new feature arrays and keep submission writing unchanged.'
- What this solution (achieved 0.51905) has done: 'Your current AUC (0.519) is far below the target (0.7625), so we should improve discrimination with the smallest feature-side change while keeping the same “handcrafted features → StandardScaler → LogisticRegression” pipeline and training loop intact. The biggest low-risk gain here is to add a tiny set of cadence-structure features that directly measure “present in A, absent in B/C/D” by comparing energy in the three A panels vs the three non-A panels, plus per-panel A-vs-nonA contrasts; this stays within the existing feature-extraction approach and preserves evaluation semantics. I also fix a subtle percentile bug: using the “90/95/99 percentile” routine on signed differences is not meaningful as-is because it partitions unsorted signed values (negative-heavy), so we instead compute signed percentiles on `diff_p` but also add robust positive/negative tail mass features (means of positive/negative parts) without changing the model. Cache tags are bumped to avoid mixing old feature arrays, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.51872) has done: 'Your current AUC (0.519) is far below the target (0.7625), so we should improve discrimination with the smallest possible change while keeping the same handcrafted-feature extraction + StandardScaler + LogisticRegression training loop. The most likely issue is that the feature set is missing the competition’s key cadence pattern: signals present in A panels but absent in non‑A (B/C/D) panels. I add a tiny, directly-relevant set of “A-only evidence” features computed from per-pair positive excess `max(A-B, 0)` (and its non‑A counterpart `max(B-A,0)`), plus a simple asymmetry ratio; this preserves your overall feature approach and model, but gives the classifier a clearer needle-vs-RFI cue. I bump the cache tag to avoid mixing old/new feature layouts, and keep the submission writing unchanged.'
- What this solution (achieved 0.52266) has done: 'I fix the feature-dimension bug that causes the `IndexError` by allocating the `feats` array with the correct size inferred from how many values are actually written (score-neutral, just makes it run). Then, because cell 2 currently crashes before defining `load_and_extract`, I ensure cell order starts at 1 and that all definitions execute before they’re used, eliminating the downstream `NameError`. Finally, I keep the model/training logic unchanged and make sure we always write a valid `submission.csv` with `id,target` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

np.random.seed(42)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",  # if competition dataset is mounted directly here
    "/kaggle/data",
]


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


base = _first_existing(*BASE_CANDIDATES)
if base is None:
    raise FileNotFoundError(
        "Could not locate Kaggle dataset directory under known /kaggle/* paths."
    )

if os.path.exists(os.path.join(base, "train")) and os.path.exists(
    os.path.join(base, "test")
):
    DATA_DIR = base
else:
    sub = _first_existing(os.path.join(base, "seti-breakthrough-listen"))
    if sub is None:
        raise FileNotFoundError(
            f"Could not locate train/test directories under: {base}"
        )
    DATA_DIR = sub

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_PATH = _first_existing(
    os.path.join(DATA_DIR, "train_labels.csv"),
    "/kaggle/input/train_labels.csv",
    "/kaggle/data/train_labels.csv",
)
SAMPLE_SUB_PATH = _first_existing(
    os.path.join(DATA_DIR, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)

if LABELS_PATH is None:
    raise FileNotFoundError("train_labels.csv not found.")
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError("sample_submission.csv not found.")

train_labels = pd.read_csv(LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

for df, name in [(train_labels, "train_labels"), (sample_sub, "sample_submission")]:
    if not {"id", "target"}.issubset(df.columns):
        raise ValueError(
            f"{name} must contain columns ['id','target'], got: {df.columns.tolist()}"
        )

train_labels["id"] = train_labels["id"].astype(str)
sample_sub["id"] = sample_sub["id"].astype(str)

print("DATA_DIR:", DATA_DIR)
print("Train labels:", train_labels.shape, "Sample submission:", sample_sub.shape)




## === cell 1
def list_npy_files(root_dir):
    files = []
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_dir():
                with os.scandir(entry.path) as it2:
                    for f in it2:
                        if f.is_file() and f.name.endswith(".npy"):
                            files.append(f.path)
    return files


train_files = list_npy_files(TRAIN_DIR)
test_files = list_npy_files(TEST_DIR)

if len(train_files) == 0:
    raise FileNotFoundError(f"No train .npy files found under {TRAIN_DIR}")
if len(test_files) == 0:
    raise FileNotFoundError(f"No test .npy files found under {TEST_DIR}")


def id_from_path(p):
    return os.path.splitext(os.path.basename(p))[0]


train_path_by_id = {id_from_path(p): p for p in train_files}
test_path_by_id = {id_from_path(p): p for p in test_files}

train_id_set = set(train_path_by_id.keys())
test_id_set = set(test_path_by_id.keys())

missing_train = train_labels.loc[~train_labels["id"].isin(train_id_set), "id"]
missing_test = sample_sub.loc[~sample_sub["id"].isin(test_id_set), "id"]

if len(missing_train) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_train)} train ids in train folder. Example: {missing_train.iloc[:5].tolist()}"
    )
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test ids in test folder. Example: {missing_test.iloc[:5].tolist()}"
    )

print("Train npy files:", len(train_files), "Test npy files:", len(test_files))



## === cell 2
from concurrent.futures import ThreadPoolExecutor


def _percentiles_90_95_99_exact_inplace(flat_f32):
    n = flat_f32.size
    qs = np.array([90.0, 95.0, 99.0], dtype=np.float32)
    idx = (qs / 100.0) * (n - 1)
    lo = np.floor(idx).astype(np.int64)
    hi = np.ceil(idx).astype(np.int64)

    uniq = np.unique(np.concatenate([lo, hi]))
    np.partition(flat_f32, kth=uniq, axis=0)  # in-place partition
    vals = {int(k): flat_f32[int(k)] for k in uniq}

    out = np.empty(3, dtype=np.float32)
    for i in range(3):
        if lo[i] == hi[i]:
            out[i] = vals[int(lo[i])]
        else:
            v0 = vals[int(lo[i])]
            v1 = vals[int(hi[i])]
            out[i] = v0 + (idx[i] - lo[i]) * (v1 - v0)
    return out


def extract_features_from_array(x):
    x = np.asarray(x).astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # (3,H,W)
    B = x[[1, 3, 5]]  # (3,H,W)

    D = A - B  # signed diff (3,H,W)
    AD = np.abs(D)  # abs diff (3,H,W)

    nonA = x[[1, 3, 5]]  # same as B here, but kept explicit for clarity/semantics
    A_energy = A.reshape(3, -1).mean(axis=1)
    nonA_energy = nonA.reshape(3, -1).mean(axis=1)
    A_nonA_contrast = A_energy - nonA_energy  # (3,)

    A_frame_means = A_energy
    B_frame_means = B.reshape(3, -1).mean(axis=1)

    pair_pct_abs = np.empty((3, 3), dtype=np.float32)  # (pair, q)
    pair_rowmax_abs = np.empty(3, dtype=np.float32)
    pair_colmax_abs = np.empty(3, dtype=np.float32)
    pair_rowstd_abs = np.empty(3, dtype=np.float32)
    pair_colstd_abs = np.empty(3, dtype=np.float32)
    pair_energy_abs = np.empty(3, dtype=np.float32)

    pair_pct_signed = np.empty((3, 3), dtype=np.float32)  # percentiles of signed D
    pair_energy_signed = np.empty(3, dtype=np.float32)  # mean signed D

    pair_posmean = np.empty(3, dtype=np.float32)
    pair_negmean = np.empty(3, dtype=np.float32)

    pair_Aonly_posmean = np.empty(3, dtype=np.float32)  # mean(max(A-B,0))
    pair_nonAonly_posmean = np.empty(3, dtype=np.float32)  # mean(max(B-A,0))

    pair_pct_pos = np.empty((3, 3), dtype=np.float32)  # percentiles of pos=max(D,0)
    pair_pct_negmag = np.empty((3, 3), dtype=np.float32)  # percentiles of -min(D,0)
    pair_rowmax_pos = np.empty(3, dtype=np.float32)
    pair_colmax_pos = np.empty(3, dtype=np.float32)
    pair_rowmax_negmag = np.empty(3, dtype=np.float32)
    pair_colmax_negmag = np.empty(3, dtype=np.float32)

    for p in range(3):
        absdiff_p = AD[p]  # (H,W)
        diff_p = D[p]  # (H,W)

        flat_abs = np.asarray(absdiff_p, dtype=np.float32, order="C").ravel()
        pair_pct_abs[p] = _percentiles_90_95_99_exact_inplace(flat_abs)

        row_mean_abs = absdiff_p.mean(axis=1)
        col_mean_abs = absdiff_p.mean(axis=0)
        pair_rowmax_abs[p] = row_mean_abs.max()
        pair_colmax_abs[p] = col_mean_abs.max()
        pair_rowstd_abs[p] = row_mean_abs.std()
        pair_colstd_abs[p] = col_mean_abs.std()
        pair_energy_abs[p] = absdiff_p.mean()

        flat_signed = np.asarray(diff_p, dtype=np.float32, order="C").ravel()
        pair_pct_signed[p] = _percentiles_90_95_99_exact_inplace(flat_signed)
        pair_energy_signed[p] = diff_p.mean()

        pos = np.maximum(diff_p, 0.0)
        neg = np.minimum(diff_p, 0.0)
        pair_posmean[p] = pos.mean()
        pair_negmean[p] = neg.mean()

        pair_Aonly_posmean[p] = pos.mean()  # max(A-B,0)
        pair_nonAonly_posmean[p] = (-neg).mean()  # max(B-A,0) = max(-(A-B),0)

        flat_pos = np.asarray(pos, dtype=np.float32, order="C").ravel()
        flat_negmag = np.asarray(-neg, dtype=np.float32, order="C").ravel()
        pair_pct_pos[p] = _percentiles_90_95_99_exact_inplace(flat_pos)
        pair_pct_negmag[p] = _percentiles_90_95_99_exact_inplace(flat_negmag)

        row_mean_pos = pos.mean(axis=1)
        col_mean_pos = pos.mean(axis=0)
        pair_rowmax_pos[p] = row_mean_pos.max()
        pair_colmax_pos[p] = col_mean_pos.max()

        row_mean_negmag = (-neg).mean(axis=1)
        col_mean_negmag = (-neg).mean(axis=0)
        pair_rowmax_negmag[p] = row_mean_negmag.max()
        pair_colmax_negmag[p] = col_mean_negmag.max()

    eps = np.float32(1e-6)
    energy_cv_abs = pair_energy_abs.std() / (pair_energy_abs.mean() + eps)
    pct99_cv_abs = pair_pct_abs[:, 2].std() / (pair_pct_abs[:, 2].mean() + eps)

    energy_cv_signed = pair_energy_signed.std() / (
        np.abs(pair_energy_signed.mean()) + eps
    )
    pct99_cv_signed = pair_pct_signed[:, 2].std() / (
        np.abs(pair_pct_signed[:, 2].mean()) + eps
    )

    pct99_cv_pos = pair_pct_pos[:, 2].std() / (pair_pct_pos[:, 2].mean() + eps)
    pct99_cv_negmag = pair_pct_negmag[:, 2].std() / (pair_pct_negmag[:, 2].mean() + eps)

    A0, A1, A2 = A[0], A[1], A[2]
    A_abs01 = np.abs(A0 - A1).mean()
    A_abs02 = np.abs(A0 - A2).mean()
    A_abs12 = np.abs(A1 - A2).mean()
    A_abs_pair_mean = (A_abs01 + A_abs02 + A_abs12) / np.float32(3.0)
    A_abs_pair_std = np.std([A_abs01, A_abs02, A_abs12]).astype(np.float32)

    A_stack = A.reshape(3, -1)
    A_pix_std_mean = A_stack.std(axis=0).mean()
    A_pix_mean_std = A_stack.mean(axis=0).std()

    A_mean_img = A.mean(axis=0)
    nonA_mean_img = nonA.mean(axis=0)
    Amean_minus_nonAmean = A_mean_img - nonA_mean_img
    Amean_minus_nonA_absmean = np.abs(Amean_minus_nonAmean).mean()
    Amean_minus_nonA_posmean = np.maximum(Amean_minus_nonAmean, 0.0).mean()
    Amean_minus_nonA_negmag_mean = np.maximum(-Amean_minus_nonAmean, 0.0).mean()

    feats = np.empty(72, dtype=np.float32)
    k = 0

    feats[k] = A_frame_means.mean()
    k += 1
    feats[k] = B_frame_means.mean()
    k += 1
    feats[k] = D.mean()
    k += 1
    feats[k] = AD.mean()
    k += 1

    feats[k] = A.std()
    k += 1
    feats[k] = B.std()
    k += 1
    feats[k] = D.std()
    k += 1
    feats[k] = AD.std()
    k += 1

    feats[k : k + 3] = pair_pct_abs.mean(axis=0)
    k += 3
    feats[k : k + 3] = pair_pct_abs.std(axis=0)
    k += 3

    feats[k] = pair_rowmax_abs.mean()
    k += 1
    feats[k] = pair_colmax_abs.mean()
    k += 1
    feats[k] = pair_rowstd_abs.mean()
    k += 1
    feats[k] = pair_colstd_abs.mean()
    k += 1
    feats[k] = pair_rowmax_abs.std()
    k += 1
    feats[k] = pair_colmax_abs.std()
    k += 1
    feats[k] = pair_rowstd_abs.std()
    k += 1
    feats[k] = pair_colstd_abs.std()
    k += 1

    feats[k] = (np.mean(A * A) + eps) / (np.mean(B * B) + eps)
    k += 1

    feats[k] = A_frame_means.std()
    k += 1
    feats[k] = B_frame_means.std()
    k += 1

    feats[k] = pair_energy_abs.mean()
    k += 1
    feats[k] = pair_energy_abs.std()
    k += 1
    feats[k] = energy_cv_abs
    k += 1
    feats[k] = pct99_cv_abs
    k += 1

    feats[k : k + 3] = pair_pct_signed.mean(axis=0)
    k += 3
    feats[k : k + 3] = pair_pct_signed.std(axis=0)
    k += 3
    feats[k] = pair_energy_signed.mean()
    k += 1
    feats[k] = pair_energy_signed.std()
    k += 1
    feats[k] = energy_cv_signed
    k += 1
    feats[k] = pct99_cv_signed
    k += 1

    feats[k] = A_energy.mean()
    k += 1
    feats[k] = nonA_energy.mean()
    k += 1
    feats[k] = A_nonA_contrast.mean()
    k += 1
    feats[k] = A_nonA_contrast.std()
    k += 1

    feats[k] = pair_posmean.mean()
    k += 1
    feats[k] = pair_posmean.std()
    k += 1
    feats[k] = pair_negmean.mean()
    k += 1
    feats[k] = pair_negmean.std()
    k += 1

    Aonly_mean = pair_Aonly_posmean.mean()
    nonAonly_mean = pair_nonAonly_posmean.mean()
    feats[k] = Aonly_mean
    k += 1
    feats[k] = pair_Aonly_posmean.std()
    k += 1
    feats[k] = nonAonly_mean
    k += 1
    feats[k] = (Aonly_mean + eps) / (nonAonly_mean + eps)
    k += 1  # asymmetry ratio

    feats[k : k + 3] = pair_pct_pos.mean(axis=0)
    k += 3
    feats[k : k + 3] = pair_pct_negmag.mean(axis=0)
    k += 3

    feats[k] = pair_rowmax_pos.mean()
    k += 1
    feats[k] = pair_colmax_pos.mean()
    k += 1
    feats[k] = pair_rowmax_negmag.mean()
    k += 1
    feats[k] = pair_colmax_negmag.mean()
    k += 1

    feats[k] = pct99_cv_pos
    k += 1
    feats[k] = pct99_cv_negmag
    k += 1

    feats[k] = A_abs_pair_mean
    k += 1
    feats[k] = A_abs_pair_std
    k += 1
    feats[k] = A_pix_std_mean
    k += 1
    feats[k] = A_pix_mean_std
    k += 1
    feats[k] = Amean_minus_nonA_absmean
    k += 1
    feats[k] = Amean_minus_nonA_posmean
    k += 1
    feats[k] = Amean_minus_nonA_negmag_mean
    k += 1

    assert k == feats.shape[0], (k, feats.shape[0])
    return feats


_tmp = np.load(train_path_by_id[train_labels["id"].iloc[0]], mmap_mode="r")
_feat = extract_features_from_array(_tmp)
N_FEATS = int(_feat.shape[0])
assert N_FEATS > 0, "Feature dimension must be positive."
print("Feature dimension:", N_FEATS)


def _load_one_indexed(args):
    i, path = args
    arr = np.load(path, mmap_mode="r")
    return i, extract_features_from_array(arr)


def _dataset_fingerprint(path_by_id, ids, max_items=256):
    if not ids:
        return "empty"
    take = min(max_items // 2, len(ids))
    sample_ids = ids[:take] + ids[-take:]
    acc = 0
    for _id in sample_ids:
        p = path_by_id[_id]
        try:
            st = os.stat(p)
            acc = (acc * 1315423911 + st.st_size + int(st.st_mtime)) & 0xFFFFFFFF
        except FileNotFoundError:
            acc = (acc * 2654435761) & 0xFFFFFFFF
    return f"{acc:08x}"


def load_and_extract(ids, path_by_id, max_workers=None, chunksize=128, cache_tag="v10"):
    cache_dir = "/kaggle/working/feature_cache"
    os.makedirs(cache_dir, exist_ok=True)

    fp = _dataset_fingerprint(path_by_id, ids)
    cache_path = os.path.join(
        cache_dir, f"feats_{cache_tag}_{fp}_{len(ids)}_{N_FEATS}.npy"
    )

    if os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode="r")
        X = np.asarray(X, dtype=np.float32, order="C")
        return X

    paths = [path_by_id[_id] for _id in ids]
    X = np.empty((len(paths), N_FEATS), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in ex.map(_load_one_indexed, enumerate(paths), chunksize=chunksize):
            X[i] = feat

    np.save(cache_path, X)
    return X




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1580040093.py in <cell line: 0>()
    283 
    284 _tmp = np.load(train_path_by_id[train_labels["id"].iloc[0]], mmap_mode="r")
--> 285 _feat = extract_features_from_array(_tmp)
    286 N_FEATS = int(_feat.shape[0])
    287 assert N_FEATS > 0, "Feature dimension must be positive."

/tmp/ipykernel_11/1580040093.py in extract_features_from_array(x)
    278     k += 1
    279 
--> 280     assert k == feats.shape[0], (k, feats.shape[0])
    281     return feats
    282 

AssertionError: (70, 72)

## === cell 3
from sklearn.metrics import roc_auc_score

train_ids = train_labels["id"].tolist()
y = train_labels["target"].astype(np.int8).values

X = load_and_extract(train_ids, train_path_by_id, cache_tag="train_v10")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

C_GRID = [0.3, 1.0, 3.0]

best_C = None
best_auc = -1.0

for C in C_GRID:
    oof = np.zeros(len(train_ids), dtype=np.float32)

    base_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    C=C,
                    class_weight=None,
                    n_jobs=None,
                    random_state=42,
                ),
            ),
        ]
    )

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = clone(base_model)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1].astype(np.float32)

    auc = roc_auc_score(y, oof)
    print(f"C={C} -> OOF ROC-AUC: {auc:.6f}")
    if auc > best_auc:
        best_auc = auc
        best_C = C

print("Selected C:", best_C, "with OOF AUC:", best_auc)

final_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                C=best_C,
                class_weight=None,
                n_jobs=None,
                random_state=42,
            ),
        ),
    ]
)
final_model.fit(X, y)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4212804839.py in <cell line: 0>()
      4 y = train_labels["target"].astype(np.int8).values
      5 
----> 6 X = load_and_extract(train_ids, train_path_by_id, cache_tag="train_v10")
      7 
      8 skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

NameError: name 'load_and_extract' is not defined

## === cell 4
test_ids = sample_sub["id"].tolist()

X_test = load_and_extract(test_ids, test_path_by_id, cache_tag="test_v10")

test_pred = final_model.predict_proba(X_test)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "target": test_pred})
if submission["target"].isna().any():
    raise ValueError("NaNs detected in predictions.")

submission = submission[["id", "target"]]
if submission.shape[0] != sample_sub.shape[0]:
    raise ValueError(
        f"Submission row count {submission.shape[0]} != sample_submission {sample_sub.shape[0]}"
    )

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/841517975.py in <cell line: 0>()
      1 test_ids = sample_sub["id"].tolist()
      2 
----> 3 X_test = load_and_extract(test_ids, test_path_by_id, cache_tag="test_v10")
      4 
      5 test_pred = final_model.predict_proba(X_test)[:, 1].astype(np.float32)

NameError: name 'load_and_extract' is not defined
