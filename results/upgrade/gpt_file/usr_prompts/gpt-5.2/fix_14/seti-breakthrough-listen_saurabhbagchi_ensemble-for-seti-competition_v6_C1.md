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

0.7486250604200507

# 6. Current score

0.50988

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50536) has done: 'The timeout is almost certainly dominated by per-file `np.load` overhead across ~60k snippets plus expensive repeated `np.partition` calls that copy large flattened arrays multiple times per snippet. I keep the exact same feature set and model, but speed up feature extraction by computing all needed quantiles with a single `np.partition` per array (and per A/B subset) and by eliminating unnecessary array copies/partitions. I also reduce multiprocessing overhead by using a threads-based pool (NumPy releases the GIL in heavy ops and threads avoid fork/spawn + pickle costs) and by batching results more efficiently; correctness is preserved because each file is still featurized identically and deterministically. Finally, I avoid re-reading/processing paths repeatedly and keep all I/O paths unchanged.'
- What this solution (achieved 0.50536) has done: 'Your current score (0.50536) is far below the target (0.7486), so we should improve (not just speed up) while keeping the same feature set and LogisticRegression pipeline. The biggest likely issue is that the train/test file ordering via `id_to_path` is wrong for this dataset layout: files are stored under folders `0..15` (first hex char), not just `0/1`, so many ids be mapped to non-existent paths or (worse) silently mismatched in other environments, collapsing AUC toward random. I change path resolution to robustly find the correct subfolder for each id (while keeping paths within the provided train/test directories) and keep everything else (features, model, CV, submission) identical. This is a minimal, high-impact fix expected to move AUC substantially toward your target without altering core modeling logic.'
- What this solution (achieved 0.50536) has done: 'Your current AUC (0.505) is close to random, so the most likely issue is not the classifier but a subtle train/test misalignment or silent feature corruption. I keep the exact same feature definitions and LogisticRegression pipeline, but (1) make file resolution deterministic and validated by building an `id -> path` index once per split (train/test) instead of per-id globbing, and (2) add a hard check that every loaded `.npy` actually matches the expected `(6, 273, 256)` shape to avoid accidental wrong-file/format reads that can collapse AUC. These are minimal changes that preserve core logic while strongly reducing the chance of path mismatches and bad inputs, which should move AUC materially toward your target. The submission writing remains identical (`submission.csv` with `id,target`).'
- What this solution (achieved 0.50536) has done: 'Your current AUC is near-random, so the most likely issue is a train/label misalignment coming from `build_id_to_path_index` silently overwriting duplicate IDs: the dataset is sharded and contains the *same filename* under multiple subfolders, and your dict keeps only the last one encountered. I keep the exact same features and LogisticRegression pipeline, but rebuild the id→path index deterministically while validating uniqueness and ensuring every `id` maps to exactly one `.npy` (fail-fast if not). This minimal fix should move performance substantially upward toward your target without changing modeling logic or the metric semantics. I also keep test ordering aligned to `sample_submission.csv` exactly as before and still write `submission.csv` with `id,target`.'
- What this solution (achieved 0.50536) has done: 'Your current AUC (0.505) is near-random versus the 0.7486 target, so the most likely score issue is not the LogisticRegression itself but a data/label mismatch caused by incorrect `id -> .npy path` mapping. I make the smallest high-impact fix by building the mapping directly from the *requested ids* (using the known shard folder = first hex character of `id`), with a one-time fallback scan only if needed, and I hard-validate that each loaded file’s basename matches the requested id. This preserves your exact feature set, model, CV, and submission semantics while preventing silent wrong-file reads that can collapse AUC. The rest of the pipeline (features, scaling, LR, CV, submission format) stays unchanged.'
- What this solution (achieved 0.50536) has done: 'Your AUC is near-random versus the 0.7486 target, which strongly suggests a train/label feature-row misalignment or silently wrong file reads rather than the LogisticRegression itself. I keep the exact same feature set and model, but harden `ids_to_paths()` so it deterministically resolves each id to exactly one existing `.npy` (including handling shard folders robustly) and fail-fast on any duplicate-id ambiguity instead of silently picking the first hit. I also add a basename check in the fallback branch and a quick existence/uniqueness verification pass so we never train on features extracted from the wrong snippets. These are minimal, score-relevant correctness fixes that should move AUC substantially upward toward your target without changing modeling semantics.'
- What this solution (achieved 0.50988) has done: 'Your AUC is near-random versus the target, so the most likely cause is still a subtle train/test misalignment or silently wrong file reads; I add a strict, cheap verification that every resolved path’s basename matches the requested id and that every id resolves via the expected shard folder (first hex char), failing fast if not. To move score upward without changing core modeling/feature logic, I also fix one feature bug in `_linear_quantiles_1d_via_single_partition`: using a single `np.partition(..., k=max(his))` does not guarantee correctness for smaller order statistics, so I switch to a correct multi-`k` partition call that remains the same quantile definition but avoids corrupted quantile features that can collapse AUC. Everything else (same 15 features, same StandardScaler+LogisticRegression, same CV loop, same submission format/path) stays identical.'
- What this solution (achieved 0.50988) has done: 'Your current AUC is still near-random (0.51 vs target 0.7486), which usually indicates a train/test feature-row misalignment rather than a weak model. I keep the exact same 15 features and the same StandardScaler+LogisticRegression training approach, but fix the most likely alignment bug: your `sample_submission.csv` here has 6000 rows, while the provided `test/` contains far more files—so you are likely predicting for the wrong/partial test set. I switch test ID sourcing to be derived from the actual `test/` filenames (and then build `submission.csv` from those ids), which preserves modeling semantics but ensures the submission rows match the true test set. I also add a strict check that `len(test_ids)` matches the number of `.npy` files found to prevent silent partial submissions.'
- What this solution (achieved 0.50988) has done: 'Your AUC is still near-random versus the 0.7486 target, so the most likely remaining issue is that your training features are not being extracted from the same physical files as their labels (silent duplicate-id ambiguity across shard folders), which would destroy learnable signal without throwing an error. I make the smallest score-relevant change: resolve each train/test id to a unique `.npy` by directly using the expected shard folder (first hex char) and, if duplicates exist elsewhere, *refuse* to proceed unless the shard-resolved file exists—this prevents accidental “wrong copy” selection from fallback scans. I also add a strict uniqueness check when scanning directories (if used) so we never silently overwrite duplicates in an index. Core feature extraction, model, CV, and submission semantics remain identical.'
- What this solution (achieved 0.50988) has done: 'Your AUC being ~0.51 (near-random) versus a 0.7486 target most strongly suggests the model is training on features that do not correspond to the intended `id` labels (or the submission is not aligned to the right test ids), not that LogisticRegression is too weak. I make the smallest correctness-focused change: derive `test_ids` strictly from `sample_submission.csv` when available (so the submission rows match Kaggle’s expected test set), and resolve both train/test paths from those explicit id lists. I also add strict, cheap integrity checks that each resolved file’s basename matches the requested id and that we never proceed if any id can’t be resolved, preventing silent misalignment that collapses AUC. Core feature extraction (15 features), model (StandardScaler + LogisticRegression), and CV training loop remain unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BASE_INPUT = "/kaggle/input"
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

if not os.path.exists(TRAIN_LABELS_PATH):
    TRAIN_LABELS_PATH = os.path.join(
        BASE_INPUT, "seti-breakthrough-listen", "train_labels.csv"
    )
if not os.path.exists(TRAIN_DIR):
    TRAIN_DIR = os.path.join(BASE_INPUT, "seti-breakthrough-listen", "train")
if not os.path.exists(TEST_DIR):
    TEST_DIR = os.path.join(BASE_INPUT, "seti-breakthrough-listen", "test")
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = os.path.join(
        BASE_INPUT, "seti-breakthrough-listen", "sample_submission.csv"
    )

assert os.path.exists(
    TRAIN_LABELS_PATH
), f"Missing train_labels.csv at {TRAIN_LABELS_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing train dir at {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir at {TEST_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 1
def _fallback_build_full_id_to_paths_index(root_dir: str) -> dict:
    """
    Change rationale (score-relevant): never allow silent overwriting of duplicate ids when
    scanning shards; duplicates can cause label/feature mismatch and collapse AUC.
    Build id -> [paths...] so we can detect ambiguity explicitly.
    """
    paths = glob.glob(os.path.join(root_dir, "*", "*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found under {root_dir}/*/*.npy")
    idx = {}
    for p in paths:
        _id = os.path.splitext(os.path.basename(p))[0]
        idx.setdefault(_id, []).append(p)
    return idx


def ids_to_paths(root_dir: str, ids, require_shard_hit: bool = True) -> list:
    """
    Change rationale (score-relevant): enforce deterministic id->path mapping that matches
    the competition's shard layout (folder == first hex char of id). If require_shard_hit=True,
    we *do not* fall back to a global scan for missing ids, because fallback can pick the wrong
    duplicate copy in some environments and destroy train/label alignment.
    Additionally, we hard-check that resolved path basename == requested id to prevent silent mismatch.
    """
    out = [None] * len(ids)
    missing_ids = []

    for i, _id in enumerate(ids):
        _id = str(_id)
        shard = _id[0].lower()
        p = os.path.join(root_dir, shard, f"{_id}.npy")
        if os.path.exists(p):
            if os.path.splitext(os.path.basename(p))[0] != _id:
                raise ValueError(
                    f"Basename/id mismatch for resolved path: {_id} -> {p}"
                )
            parent = os.path.basename(os.path.dirname(p)).lower()
            if parent != shard:
                raise ValueError(
                    f"Shard folder mismatch for id={_id}: expected {shard}, got {parent}"
                )
            out[i] = p
        else:
            missing_ids.append(_id)

    if missing_ids and require_shard_hit:
        raise FileNotFoundError(
            f"{len(missing_ids)} ids did not resolve via shard folder under {root_dir}. "
            f"First few missing: {missing_ids[:10]}"
        )

    if not missing_ids:
        return out

    idx = _fallback_build_full_id_to_paths_index(root_dir)
    still_missing = []
    for i, _id in enumerate(ids):
        if out[i] is not None:
            continue
        _id = str(_id)
        candidates = idx.get(_id, [])
        if len(candidates) == 0:
            still_missing.append(_id)
            continue
        if len(candidates) > 1:
            raise RuntimeError(
                f"Ambiguous mapping for id={_id}: found {len(candidates)} files. "
                f"First few: {candidates[:5]}"
            )
        p2 = candidates[0]
        if os.path.splitext(os.path.basename(p2))[0] != _id:
            raise ValueError(f"Basename/id mismatch from index: {_id} -> {p2}")
        out[i] = p2

    if still_missing:
        raise FileNotFoundError(
            f"Could not locate {len(still_missing)} ids under {root_dir}. "
            f"First few missing: {still_missing[:10]}"
        )

    return out


train_ids = train_labels["id"].astype(str).tolist()
test_ids = sample_sub["id"].astype(str).tolist()

ordered_train_files = ids_to_paths(TRAIN_DIR, train_ids, require_shard_hit=True)
ordered_test_files = ids_to_paths(TEST_DIR, test_ids, require_shard_hit=True)

if len(set(train_ids)) != len(train_ids):
    raise RuntimeError(
        "Duplicate ids found in train_labels.csv; this would break alignment."
    )
if len(set(test_ids)) != len(test_ids):
    raise RuntimeError(
        "Duplicate ids found in sample_submission.csv; this would break alignment."
    )

assert all(
    p is not None for p in ordered_train_files
), "Some train ids did not resolve to files"
assert all(
    p is not None for p in ordered_test_files
), "Some test ids did not resolve to files"
assert len(ordered_train_files) == len(train_labels), "Train file list length mismatch"
assert len(ordered_test_files) == len(sample_sub), "Test file list length mismatch"

len(ordered_train_files), len(ordered_test_files), ordered_train_files[
    :2
], ordered_test_files[:2]



## === cell 2
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


def _linear_quantiles_1d_via_single_partition(x: np.ndarray, qs) -> dict:
    """
    Correct multi-k partition for exact order-statistic based linear interpolation quantiles.
    """
    n = x.size
    if n == 0:
        return {float(q): np.nan for q in qs}

    qs = [float(q) for q in qs]
    hs = [(n - 1) * q for q in qs]
    los = [int(np.floor(h)) for h in hs]
    his = [int(np.ceil(h)) for h in hs]

    ks = np.unique(np.asarray(los + his, dtype=np.int64))
    xp = np.partition(x.copy(), ks)

    out = {}
    for q, h, lo, hi in zip(qs, hs, los, his):
        if lo == hi:
            out[q] = float(xp[lo])
        else:
            x_lo = float(xp[lo])
            x_hi = float(xp[hi])
            out[q] = x_lo + (h - lo) * (x_hi - x_lo)
    return out


def extract_features(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)

    mean_all = a.mean()
    std_all = a.std()

    flat_all = a.ravel()
    q_all = _linear_quantiles_1d_via_single_partition(flat_all, (0.5, 0.9, 0.1))
    med_all = q_all[0.5]
    q90_all = q_all[0.9]
    q10_all = q_all[0.1]

    A = a[[0, 2, 4]]
    B = a[[1, 3, 5]]

    mean_A = A.mean()
    mean_B = B.mean()
    std_A = A.std()
    std_B = B.std()

    flat_A = A.ravel()
    flat_B = B.ravel()
    qA = _linear_quantiles_1d_via_single_partition(flat_A, (0.99, 0.95))
    qB = _linear_quantiles_1d_via_single_partition(flat_B, (0.99, 0.95))
    qA99, qA95 = qA[0.99], qA[0.95]
    qB99, qB95 = qB[0.99], qB[0.95]

    var_axis2_mean = a.var(axis=2).mean()
    var_axis1_mean = a.var(axis=1).mean()

    dx = np.abs(np.diff(a, axis=2)).mean()
    dy = np.abs(np.diff(a, axis=1)).mean()

    feat = np.asarray(
        [
            mean_all,
            std_all,
            med_all,
            q90_all,
            q10_all,
            (mean_A - mean_B),
            (std_A - std_B),
            (mean_A / (mean_B + 1e-6)),
            var_axis2_mean,
            var_axis1_mean,
            (qA99 - qB99),
            (qA95 - qB95),
            dx,
            dy,
            (dx - dy),
        ],
        dtype=np.float32,
    )
    return feat


_EXPECTED_SHAPE = (6, 273, 256)


def _featurize_one(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.shape != _EXPECTED_SHAPE:
        raise ValueError(f"Unexpected array shape {arr.shape} for file: {path}")
    return extract_features(arr)


def build_feature_matrix(files, max_files=None, n_jobs=None, chunksize=256):
    if max_files is None:
        max_files = len(files)
    files = files[:max_files]

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    X = np.zeros((len(files), 15), dtype=np.float32)

    if n_jobs == 1:
        for i, p in enumerate(files):
            X[i] = _featurize_one(p)
        return X

    with ThreadPool(processes=n_jobs) as pool:
        for i, feat in enumerate(pool.imap(_featurize_one, files, chunksize=chunksize)):
            X[i] = feat
    return X


X_train = build_feature_matrix(ordered_train_files)
y_train = train_labels["target"].values.astype(np.int32)

X_train.shape, y_train.shape, X_train[:1]



## === cell 3
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        random_state=RANDOM_STATE,
        n_jobs=None,
    ),
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(y_train), dtype=np.float32)

for tr_idx, va_idx in skf.split(X_train, y_train):
    model_fold = clone(model)
    model_fold.fit(X_train[tr_idx], y_train[tr_idx])
    oof[va_idx] = model_fold.predict_proba(X_train[va_idx])[:, 1].astype(np.float32)

auc = roc_auc_score(y_train, oof)
auc



## === cell 4
model.fit(X_train, y_train)

X_test = build_feature_matrix(ordered_test_files)

pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

sub = pd.DataFrame({"id": np.asarray(test_ids, dtype=str), "target": pred})
sub.to_csv("submission.csv", index=False)

sub.head(), sub.shape
