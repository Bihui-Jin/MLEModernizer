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

0.7480686684378159

# 6. Current score

0.50668

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49949) has done: 'The timeout is dominated by Python-level file discovery (`glob` over tens of thousands of files) and single-threaded per-file feature extraction/loading. I keep the exact same feature calculations and the same scikit-learn pipeline, but replace `glob` with a faster `os.scandir`-based walk and eliminate redundant rescans. I also parallelize the `.npy` loading + feature extraction with a thread pool (I/O-bound; preserves identical math per file) and add a small in-process cache for repeated loads. These changes reduce overhead and wall time without changing model logic or feature semantics (only negligible float-order differences possible).'
- What this solution (achieved 0.49949) has done: 'Your current 0.499 AUC is close to random, which strongly suggests an alignment/ordering bug rather than a modeling weakness. The core issue is that `ThreadPoolExecutor.map()` preserves input order, but you’re filling `X[i]` using `enumerate(...)`, which assigns features in completion order if anything changes (and can also silently misalign if any future changes occur); we make the row assignment explicitly keyed by `id` to guarantee correct alignment. Additionally, we ensure the train/test id lists are strictly ordered to match the CSVs and use deterministic caching behavior. These minimal changes preserve the exact feature definitions and the same sklearn pipeline, but should move AUC substantially upward toward your target.'
- What this solution (achieved 0.49651) has done: 'Your current AUC (~0.499) is consistent with a near-random predictor, and with this exact setup the most common cause is still an `id`↔row misalignment (either from duplicate filenames in the folder structure overriding earlier entries, or from mismatched ordering between the feature matrix and the label/submission `id` lists). I make the smallest score-relevant fixes by (1) building the id→path map with deterministic collision handling (and asserting uniqueness to avoid silent overwrites), and (2) aligning `train_df` exactly to the feature matrix order (and verifying it) before splitting/training. This preserves the same feature extraction, the same model/pipeline, and the same training loop, but removes the main failure mode that drives AUC toward 0.5.'
- What this solution (achieved 0.49651) has done: 'Your AUC being ~0.5 strongly suggests the model is training on features that don’t correspond to the correct labels (still an alignment issue), or the train set is being inadvertently filtered/mismatched vs labels. I make two minimal, score-relevant fixes: (1) align `train_df` to `train_ids` *exactly* (no `.isin` filtering that can silently drop/reshuffle rows) by building it from the intersection and then reindexing to the feature order, and (2) add strict assertions that every `train_id` has a file and that the `y` you use matches that exact ordered id list. These changes preserve identical feature extraction and the same sklearn pipeline/training, but remove the most common remaining source of near-random AUC. If your environment is CPU-only, I also cap thread workers conservatively to reduce contention without changing computed features.'
- What this solution (achieved 0.49651) has done: 'Your AUC is still near-random, so the most likely remaining issue is that the test `id` list is not aligned to the actual available test files (sample_submission order can be trusted, but only if every id exists and maps uniquely), and/or that `np.load` caching is returning arrays that can be mutated downstream (rare, but can silently corrupt features across threads). I keep the exact same feature math and the same sklearn pipeline, but (1) make the id→path map strictly unique while also verifying that the *exact* `sample_submission` ids are present (and only once) in `test_map`, and (2) make cached loads read-only and return a fresh view to prevent any accidental in-place modifications from contaminating subsequent feature extraction. Finally, I add a lightweight in-notebook validation AUC computation to confirm we’re no longer training on misaligned labels (this does not change training). These are minimal, score-relevant correctness fixes that should move you substantially upward toward the 0.748 target if misalignment/corruption was the cause.'
- What this solution (achieved 0.51106) has done: 'Your AUC being ~0.5 despite a reasonable pipeline strongly indicates the model is still not seeing consistent, label-aligned signals—here the most likely remaining cause is that your current features average away the discriminative structure (diagonal/line-like energy) and can collapse to near-random, even when perfectly aligned. To move score upward toward your 0.748 target while preserving the same core approach (handcrafted features + LogisticRegression), I add a tiny set of additional, still-very-cheap summary features that capture line-ness without changing the model class or training loop: (1) robust “top-k energy” statistics on D, (2) simple gradient-energy measures, and (3) a 1D drift proxy via max-over-freq per time. These are deterministic, fast, and keep evaluation semantics identical, while typically lifting AUC materially compared to pure global moments. I also keep the existing alignment assertions and submission writing unchanged.'
- What this solution (achieved 0.50774) has done: 'We keep your exact model/pipeline and the same feature families, but make two score-relevant, minimal adjustments that typically lift AUC for this SETI task without changing the approach: (1) add a tiny set of “A-only vs Off-only” contrast features (same summary stats you already compute, but on `Amean - Omean` is sometimes not enough; separate A and O time/freq projections help), and (2) make the logistic regression slightly less underfit by increasing `C` modestly while keeping solver/iterations the same. These changes should move performance upward from ~0.51 toward your 0.748 target while preserving evaluation semantics and producing the same valid `submission.csv`. Runtime remains dominated by I/O; feature additions are O(n) and negligible compared to loading. All paths and submission formatting remain unchanged.'
- What this solution (achieved 0.50668) has done: 'Your current AUC (~0.51) is far below the 0.748 target, so we should improve (not degrade) while keeping the same overall approach (handcrafted features + LogisticRegression). The smallest score-relevant change is to add a few more discriminative-but-cheap summary features that capture “line-like / concentrated energy” separately in A and Off panels (e.g., max-per-time and max-per-freq stats, and simple concentration ratios), without changing the model class, training loop, or loss. To keep everything consistent and stable, we only extend `extract_features_from_array()` and update `n_feats` accordingly, leaving all paths, alignment checks, and submission writing unchanged. This should move AUC upward toward the target while staying well within the same core logic and runtime budget.'
- What this solution (achieved 0.50668) has done: 'We keep your exact overall approach (handcrafted features + LogisticRegression) but fix a subtle, score-critical bug: `ThreadPoolExecutor.map()` in the stdlib does not accept `chunksize`, so your current featurization can silently fail or behave unexpectedly depending on environment. We also make `n_feats` automatically derived from `extract_features_from_array()` to prevent any accidental feature-length mismatch that can degrade training. Finally, we add a very small, metric-aligned calibration tweak at prediction time (rank-preserving power transform) that often improves ROC-AUC slightly without changing the model or labels, helping move you upward toward the 0.748 target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/seti-breakthrough-listen",
    "/kaggle/data/seti-breakthrough-listen",
    "/kaggle/input",
    "/kaggle/data",
]
BASE = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        if os.path.exists(os.path.join(b, "train_labels.csv")) or os.path.exists(
            os.path.join(b, "seti-breakthrough-listen", "train_labels.csv")
        ):
            BASE = b
            break
if BASE is None:
    for b in BASE_CANDIDATES:
        if os.path.exists(b):
            BASE = b
            break


def resolve_path(*parts):
    return os.path.join(BASE, *parts)


if os.path.exists(resolve_path("train_labels.csv")):
    DATA_ROOT = BASE
else:
    DATA_ROOT = resolve_path("seti-breakthrough-listen")

TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

assert os.path.exists(
    TRAIN_LABELS_PATH
), f"Missing train_labels.csv at {TRAIN_LABELS_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing train/ dir at {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test/ dir at {TEST_DIR}"

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_labels.head(), sample_sub.head()




## === cell 2
def build_id_to_path_map(folder: str):
    """
    Score-critical correctness: enforce unique id->path mapping (no silent overwrites),
    and keep deterministic traversal.
    """
    id_to_path = {}
    collisions = {}

    with os.scandir(folder) as it:
        subdirs = [e for e in it if e.is_dir()]
    subdirs.sort(key=lambda e: e.name)

    for entry in subdirs:
        subdir = entry.path
        with os.scandir(subdir) as it2:
            files = [f for f in it2 if f.is_file() and f.name.endswith(".npy")]
        files.sort(key=lambda f: f.name)

        for f in files:
            fid = f.name[:-4]
            prev = id_to_path.get(fid)
            if prev is not None and prev != f.path:
                collisions.setdefault(fid, set()).update([prev, f.path])
                continue
            id_to_path[fid] = f.path

    if collisions:
        example = next(iter(collisions.items()))
        raise RuntimeError(
            f"Found {len(collisions)} duplicate ids with multiple file paths. "
            f"Example id={example[0]} paths={sorted(list(example[1]))[:3]}"
        )

    return id_to_path


train_map = build_id_to_path_map(TRAIN_DIR)
test_map = build_id_to_path_map(TEST_DIR)

train_labels = train_labels.drop_duplicates(subset=["id"]).copy()
train_labels = train_labels.set_index("id")

train_ids = sorted(set(train_labels.index).intersection(train_map.keys()))
if len(train_ids) == 0:
    raise RuntimeError("No overlap between train_labels ids and train .npy files.")

train_df = train_labels.loc[train_ids].reset_index()

missing_train = len(train_labels) - len(train_ids)
if missing_train > 0:
    print(
        f"Warning: {missing_train} train ids have no corresponding .npy files; they will be skipped."
    )

print("Train rows used:", len(train_df))
print("Train files found:", len(train_map))
print("Test files found:", len(test_map))
print("Sample submission rows:", len(sample_sub))

assert (
    train_df["id"].tolist() == train_ids
), "train_df order must exactly match train_ids."
assert all(fid in train_map for fid in train_ids), "Every train id must have a file."

assert sample_sub[
    "id"
].is_unique, "sample_submission contains duplicate ids unexpectedly."




## === cell 3
def extract_features_from_array(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]  # (3,273,256)
    O = x[[1, 3, 5]]  # (3,273,256)

    Amean = A.mean(axis=0)
    Omean = O.mean(axis=0)
    D = Amean - Omean

    feats = []
    for arr in (Amean, Omean, D):
        feats.extend(
            [
                float(arr.mean()),
                float(arr.std()),
                float(np.median(arr)),
                float(np.percentile(arr, 10)),
                float(np.percentile(arr, 90)),
                float(arr.max()),
                float(arr.min()),
            ]
        )

    dt = D.mean(axis=1)  # average over freq -> per-time (273,)
    df = D.mean(axis=0)  # average over time -> per-freq (256,)
    feats.extend(
        [
            float(dt.mean()),
            float(dt.std()),
            float(dt.max()),
            float(dt.min()),
            float(df.mean()),
            float(df.std()),
            float(df.max()),
            float(df.min()),
        ]
    )

    feats.extend(
        [
            float((D * D).mean()),
            float(np.abs(D).mean()),
        ]
    )

    absD = np.abs(D)

    flat = absD.reshape(-1)
    k1 = 256
    k2 = 1024
    top1 = np.partition(flat, flat.size - k1)[-k1:]
    top2 = np.partition(flat, flat.size - k2)[-k2:]
    feats.extend(
        [
            float(top1.mean()),
            float(top1.max()),
            float(top2.mean()),
        ]
    )

    d_t = np.diff(D, axis=0)
    d_f = np.diff(D, axis=1)
    feats.extend(
        [
            float(np.mean(d_t * d_t)),
            float(np.mean(d_f * d_f)),
            float(np.mean(np.abs(d_t))),
            float(np.mean(np.abs(d_f))),
        ]
    )

    tmax = D.max(axis=1)
    feats.extend(
        [
            float(tmax.mean()),
            float(tmax.std()),
            float(np.max(np.diff(tmax))),  # upward jumpiness
            float(np.min(np.diff(tmax))),  # downward jumpiness
        ]
    )

    At = Amean.mean(axis=1)  # (273,)
    Ot = Omean.mean(axis=1)  # (273,)
    Af = Amean.mean(axis=0)  # (256,)
    Of = Omean.mean(axis=0)  # (256,)

    feats.extend(
        [
            float(At.std()),
            float(Ot.std()),
            float(At.max() - At.min()),
            float(Ot.max() - Ot.min()),
            float(Af.std()),
            float(Of.std()),
            float(Af.max() - Af.min()),
            float(Of.max() - Of.min()),
        ]
    )

    def _conc_feats(arr2d: np.ndarray) -> list:
        absx = np.abs(arr2d)
        s = float(absx.sum()) + 1e-8

        m_t = absx.max(axis=1)  # (273,)
        m_f = absx.max(axis=0)  # (256,)

        flatx = absx.reshape(-1)
        k_small = 256
        top_small = np.partition(flatx, flatx.size - k_small)[-k_small:]
        top_small_sum = float(top_small.sum())

        return [
            float(m_t.mean()),
            float(m_t.std()),
            float(m_t.max()),
            float(m_f.mean()),
            float(m_f.std()),
            float(m_f.max()),
            float(top_small_sum / s),
            float(float(flatx.max()) / s),
        ]

    feats.extend(_conc_feats(Amean))
    feats.extend(_conc_feats(Omean))
    feats.extend(_conc_feats(D))

    return np.array(feats, dtype=np.float32)


@lru_cache(maxsize=8192)
def _load_npy_cached(path: str) -> np.ndarray:
    """
    Score-stability/correctness: make cached arrays read-only to prevent any accidental
    in-place mutation from contaminating later feature extraction across threads.
    """
    arr = np.load(path)
    try:
        arr.setflags(write=False)
    except Exception:
        pass
    return arr


def featurize_ids(ids, id_to_path):
    """
    Keep identical feature math; ensure row assignment is keyed by id and deterministic.

    Change (score-critical correctness): remove unsupported `chunksize` argument from
    ThreadPoolExecutor.map() (stdlib does not support it). This prevents runtime/ordering
    issues that can lead to effectively-random AUC.
    """
    ids = list(ids)  # preserve caller order exactly

    _probe_id = ids[0]
    _probe_arr = _load_npy_cached(id_to_path[_probe_id])
    n_feats = int(extract_features_from_array(_probe_arr.view()).shape[0])

    X = np.zeros((len(ids), n_feats), dtype=np.float32)
    id_to_row = {fid: i for i, fid in enumerate(ids)}

    def _one(fid):
        arr = _load_npy_cached(id_to_path[fid])
        return fid, extract_features_from_array(arr.view())

    max_workers = min(16, (os.cpu_count() or 4) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fid, feats in ex.map(_one, ids):
            X[id_to_row[fid]] = feats

    return X


some_id = train_df.loc[0, "id"]
tmpX = featurize_ids([some_id], train_map)
tmpX.shape, tmpX[0, :5]



## === cell 4
X = featurize_ids(train_ids, train_map)
y = train_df["target"].astype(int).values

assert len(train_ids) == len(y) == X.shape[0]
assert (
    train_df["id"].tolist() == train_ids
), "Final alignment check failed (id order mismatch)."

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                n_jobs=None,
                C=3.0,
            ),
        ),
    ]
)

model.fit(X_tr, y_tr)

va_pred = model.predict_proba(X_va)[:, 1]
print("Holdout AUC:", roc_auc_score(y_va, va_pred))



## === cell 5
test_ids = sample_sub["id"].tolist()

missing = [fid for fid in test_ids if fid not in test_map]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test .npy files, e.g. {missing[:5]}"
    )

X_test = featurize_ids(test_ids, test_map)
pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

eps = 1e-12
pred = np.clip(pred, eps, 1 - eps)
pred = pred**1.05  # small adjustment; monotonic in [0,1]

sub = pd.DataFrame({"id": test_ids, "target": pred})
sub.to_csv("submission.csv", index=False)

sub.head(), sub.shape
