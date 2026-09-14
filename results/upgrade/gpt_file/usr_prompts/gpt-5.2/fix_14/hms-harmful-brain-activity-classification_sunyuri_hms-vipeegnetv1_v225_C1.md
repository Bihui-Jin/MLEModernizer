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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.3686512599091318

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the environment crash caused by TensorFlow/protobuf incompatibility by deferring TensorFlow import until it’s actually needed, and by providing a safe fallback that still produces a valid submission when pretrained weights are unavailable. Then I fix the model-weights loading path issue by automatically searching for the fold weight files under `/kaggle/input` and, if none are found, switching to a score-safe baseline using the normalized mean class distribution from `train.csv`. Finally, I guarantee the submission has the exact required columns, row count, and that probabilities are clipped and renormalized to sum to 1 (preventing submission rejection).'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far from the target (0.36865), and the biggest likely reason is that you’re not actually running the pretrained TF inference (so you fall back to the global prior). I make minimal changes to reliably enable TF inference by (1) broadening the weight-file search to match common naming patterns, and (2) fixing a test-time crash in `DataGenerator` where `TARGETS_RAW` is referenced even during test mode (causing inference to silently not happen in some environments). I also ensure the submission probabilities are always strictly valid (clipped + renormalized), without changing the model or preprocessing logic. These changes should move you substantially toward the target by using the intended trained weights instead of the weak prior baseline.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far from the target (0.36865), and the most likely cause is that the code is still falling back to the global prior because the pretrained fold weight directory isn’t being detected (so TF inference never runs). I make a minimal change to the weight-directory detection so it also recognizes common cases where fold weights are stored in a subfolder (e.g., `.../models/...`) and where filenames contain `fold0` etc. but don’t start with `fold`. I also make the test-mode `DataGenerator` robust by removing a test-time reference to `TARGETS_RAW` (which can silently break inference depending on execution path). These changes keep the model and preprocessing identical, but should enable the intended multi-fold TF inference and move the score substantially toward the target.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779; lower is better) is far from the target (0.36865), so we should make minimal fixes that increase the chance the pretrained multi-fold inference actually runs and produces well-calibrated probabilities. The largest correctness bug in your inference path is that the test `DataGenerator` always returns `(x, y, sample_weights)` even in `mode="test"`, which can break/alter Keras `predict` behavior; we return just `x` in test mode and avoid allocating/filling `y` and `sample_weights`. Also, `sign_id` is read via attribute access and can be missing depending on pandas behavior; we use `row["sign_id"]` safely. Finally, we harden weight discovery to search deeper and accept more common filename patterns so you don’t silently fall back to the weak global prior.'
- What this solution (achieved 1.39779) has done: 'Your score is far from the target (1.39779 vs 0.36865, lower-is-better), so the most impactful minimal fix is to ensure you actually run the intended pretrained multi-fold TF inference rather than silently falling back to the global prior. I (1) make weight discovery slightly more permissive so it finds common fold filenames (including 1-based fold numbering), and (2) fix a real inference-time bug in `DataGenerator`: it references `TARGETS_RAW` to compute `sample_weight` even though those columns are not present for test and shouldn’t be touched at all. These changes preserve the model, preprocessing, and prediction semantics, but greatly increase the chance that inference runs end-to-end with your trained weights and yields a substantially better KL score. Submission validity (columns, row order, probability normalization) is kept strict.'
- What this solution (achieved 1.39779) has done: 'Your current score is much worse than the target (1.39779 vs 0.36865, lower-is-better), so the smallest likely improvement is to ensure the pretrained multi-fold inference actually runs (instead of silently failing or producing misaligned batches). I fix a real bug in `DataGenerator` where the train/valid branch references `*_raw` vote columns that are never created in this script (this can crash or prevent proper behavior depending on execution path), by creating those `_raw` columns once from the existing vote columns. I also make weight discovery slightly more robust by accepting `.weights.h5` and `.hdf5` files (common Keras save formats) so the code doesn’t fall back to the weak prior due to missing filename patterns. These changes preserve the model, preprocessing, and prediction semantics, while increasing the chance you actually use the intended weights and thereby move the KL score toward the target.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is far above the target (0.36865), so the smallest high-impact improvement is to make sure the pretrained TF inference actually executes and loads the correct fold weights rather than silently falling back or loading nothing useful. I keep the model and preprocessing identical, but (1) broaden weight-file resolution to correctly pick up common “best/epoch/model” filenames per fold, and (2) make weight discovery return the directory with the *most* fold-like files (instead of the first match), which reduces the chance of accidentally loading an unrelated .h5. Finally, I enforce an explicit, strict probability normalization step (already present) and ensure the submission always matches `sample_submission.csv` ordering/columns to avoid hidden alignment penalties.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779; lower-is-better) is far above the target (0.36865), so we should make the smallest changes that most likely improve score by ensuring your intended pretrained inference runs correctly and produces aligned, complete predictions. The biggest correctness risk in your pipeline is that the test generator can fail when a required EEG lead column is missing (notably `EKG`), which can silently break inference or cause NaNs/misalignment; I add a minimal, safe fallback that substitutes zeros for missing channels so inference always completes. I also fix the `sub` construction to avoid redundant merges and ensure strict 1:1 alignment with `sample_submission.csv` ordering, preventing any hidden row/order penalties. Finally, I add a tiny safety net to guarantee `preds_all` length matches `test` length (otherwise fallback prior fills), improving robustness without changing the model, preprocessing, or loss.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779; lower-is-better) is far above the target (0.36865), so the smallest likely improvement is to ensure the pretrained multi-fold inference actually runs and loads weights, instead of silently falling back to the global prior. I make weight discovery more permissive by also accepting common “best/epoch/model” filenames even when they don’t contain the substring `fold`/`split`, and I make fold-to-file resolution robust by extracting fold indices from filenames via regex. I also add a tiny safety fix for spectrogram mode: in test mode your generator was hardcoding `r_spe=0`, which can cause a key error or incorrect indexing if spectrograms are enabled; we instead center-crop the available 10-minute test spectrogram without changing the model. These changes preserve your model/preprocessing/training semantics and focus only on enabling the intended inference path to move KL toward the target.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is far from the target (0.36865), and the most likely reason is that the pretrained inference still isn’t actually running (or is running with wrong/missing weights) so you effectively submit a weak global prior. I make two minimal, score-relevant fixes: (1) broaden and harden weight discovery so it can find fold weights even when filenames don’t contain “fold/split” (common for Kaggle datasets) while still requiring one file per fold, and (2) fix a real selection bug in `DataGenerator` where the `rows = df.loc[(...) * (...) * ...]` expression can behave incorrectly due to boolean multiplication; switching to `&` preserves identical intent but makes the row matching reliable. These preserve the model, preprocessing, and prediction semantics, but should materially improve score by ensuring correct weights are loaded and (if ever used) the train/valid row selection is correct. Submission validity (columns/order/sum-to-1) remains strictly enforced.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779; lower-is-better) is far above the target (0.36865), so the smallest score-relevant improvement is to reduce inference-time distribution shift and ensure the exact intended pretrained pipeline is used at test time. The biggest such issue in your script is that you enable mixed precision for inference, which can noticeably change logits/probabilities vs the weights’ training precision and hurt KL; I disable mixed precision in inference-only mode while keeping training behavior unchanged. I also add a strict fallback for missing EEG columns (notably `EKG`) and for short/edge-length slices so the generator always returns correctly-shaped tensors (avoiding silent bad batches that degrade predictions). Finally, I keep your submission alignment/normalization logic intact, only strengthening numeric stability.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower-is-better) is far from the target (0.36865), and the most likely remaining reason is that test-time inference is still not actually using the intended fold weights (so you’re effectively submitting a weak prior-like distribution). I make two minimal, score-relevant fixes: (1) make the weight discovery accept “plain” `.h5/.keras/.hdf5` files even when filenames don’t include keywords like “fold/best”, and (2) improve fold-to-file resolution to reliably map each fold to the correct file (including the common case where files are named like `model_0.h5`, `0.h5`, or live deeper under the model dataset). These changes preserve your model, preprocessing, and prediction normalization, but should enable real multi-fold TF inference and move KL substantially toward the target.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241105b"  # the path of trained model weights for testing

import os, io, time, gc, warnings, re

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

LOAD_DATA_FROM = None
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 100  # resampled EEG sampling rate
EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_MULTIPLY = 4

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / 0.4)  # the width of the STFT (eeg spectrogram)  50 * 5

filter_range = [0.5, 45]  # eeg filtering range
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed
BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg
READ_SPE_FILES = False  # preprocess spectrogram

spectrograms = {}  # preprocessed spectrograms for training
eegs = {}  # preprocessed eegs for training
stfts = {}  # preprocessed short-time fourier transform plots for training
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

TEST_BATCHSIZE = 128

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

for c in TARGETS:
    raw_c = f"{c}_raw"
    if raw_c not in df.columns:
        df[raw_c] = df[c].astype(np.float32)




## === cell 1
def find_weight_dir(preferred_dir: str) -> str | None:
    def is_weight_file(fn: str) -> bool:
        lfn = fn.lower()
        return lfn.endswith((".h5", ".keras", ".hdf5", ".weights.h5"))

    def count_weight_files(d: str) -> int:
        if not os.path.isdir(d):
            return 0
        return sum(is_weight_file(fn) for fn in os.listdir(d))

    def walk_limited(root: str, max_depth: int = 6):
        root = os.path.abspath(root)
        for cur, dirs, files in os.walk(root):
            rel = os.path.relpath(cur, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue
            yield cur, files

    best_dir = None
    best_ct = 0

    for cand in [preferred_dir]:
        ct = count_weight_files(cand)
        if ct > best_ct:
            best_ct = ct
            best_dir = cand

    if os.path.isdir(preferred_dir):
        for dd in sorted(os.listdir(preferred_dir)):
            cand = os.path.join(preferred_dir, dd)
            ct = count_weight_files(cand)
            if ct > best_ct:
                best_ct = ct
                best_dir = cand

    for root in ["/kaggle/input", preferred_dir]:
        if not root or (not os.path.isdir(root)):
            continue
        for cur, files in walk_limited(root, max_depth=6):
            if not any(is_weight_file(fn) for fn in files):
                continue
            ct = count_weight_files(cur)
            if ct > best_ct:
                best_ct = ct
                best_dir = cur

    return best_dir


WEIGHT_DIR = find_weight_dir(LOAD_MODELS_FROM)
print("Preferred weight dir:", LOAD_MODELS_FROM)
print("Resolved weight dir:", WEIGHT_DIR)


def _extract_fold_index_from_name(fn: str) -> int | None:
    lfn = fn.lower()
    m = re.search(r"(fold|split)[\s_\-]*([0-9]{1,2})", lfn)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    m2 = re.search(r"(?:^|[^a-z0-9])f(?:old)?[\s_\-]*([0-9]{1,2})(?:[^a-z0-9]|$)", lfn)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    m3 = re.match(r"^([0-9]{1,2})\.(h5|keras|hdf5)$", lfn)
    if m3:
        try:
            return int(m3.group(1))
        except Exception:
            return None
    m4 = re.search(r"(?:^|[^0-9])([0-9]{1,2})(?:[^0-9]|$)", lfn)
    if m4:
        try:
            return int(m4.group(1))
        except Exception:
            return None
    return None


def resolve_fold_weight_path(weight_dir: str, fi: int) -> str | None:
    if not weight_dir or (not os.path.isdir(weight_dir)):
        return None

    fi1 = fi + 1

    candidates = [
        os.path.join(weight_dir, f"fold{fi}_stage2.h5"),
        os.path.join(weight_dir, f"fold{fi}.h5"),
        os.path.join(weight_dir, f"fold{fi}_best.h5"),
        os.path.join(weight_dir, f"fold{fi}_model.h5"),
        os.path.join(weight_dir, f"fold{fi}_weights.h5"),
        os.path.join(weight_dir, f"fold{fi}_stage2.keras"),
        os.path.join(weight_dir, f"fold{fi}.keras"),
        os.path.join(weight_dir, f"fold{fi}_best.keras"),
        os.path.join(weight_dir, f"Fold{fi}.h5"),
        os.path.join(weight_dir, f"Fold{fi}.keras"),
        os.path.join(weight_dir, f"split{fi}.h5"),
        os.path.join(weight_dir, f"split{fi}.keras"),
        os.path.join(weight_dir, f"fold{fi}.hdf5"),
        os.path.join(weight_dir, f"split{fi}.hdf5"),
        os.path.join(weight_dir, f"fold{fi}.weights.h5"),
        os.path.join(weight_dir, f"split{fi}.weights.h5"),
        os.path.join(weight_dir, f"{fi}.h5"),
        os.path.join(weight_dir, f"{fi}.keras"),
        os.path.join(weight_dir, f"{fi}.hdf5"),
        os.path.join(weight_dir, f"model_{fi}.h5"),
        os.path.join(weight_dir, f"model_{fi}.keras"),
        os.path.join(weight_dir, f"model_{fi}.hdf5"),
        os.path.join(weight_dir, f"best_{fi}.h5"),
        os.path.join(weight_dir, f"best_{fi}.keras"),
        os.path.join(weight_dir, f"best_{fi}.hdf5"),
        os.path.join(weight_dir, f"fold{fi1}_stage2.h5"),
        os.path.join(weight_dir, f"fold{fi1}.h5"),
        os.path.join(weight_dir, f"fold{fi1}_best.h5"),
        os.path.join(weight_dir, f"fold{fi1}_model.h5"),
        os.path.join(weight_dir, f"fold{fi1}_weights.h5"),
        os.path.join(weight_dir, f"fold{fi1}_stage2.keras"),
        os.path.join(weight_dir, f"fold{fi1}.keras"),
        os.path.join(weight_dir, f"fold{fi1}_best.keras"),
        os.path.join(weight_dir, f"Fold{fi1}.h5"),
        os.path.join(weight_dir, f"Fold{fi1}.keras"),
        os.path.join(weight_dir, f"split{fi1}.h5"),
        os.path.join(weight_dir, f"split{fi1}.keras"),
        os.path.join(weight_dir, f"fold{fi1}.hdf5"),
        os.path.join(weight_dir, f"split{fi1}.hdf5"),
        os.path.join(weight_dir, f"fold{fi1}.weights.h5"),
        os.path.join(weight_dir, f"split{fi1}.weights.h5"),
        os.path.join(weight_dir, f"{fi1}.h5"),
        os.path.join(weight_dir, f"{fi1}.keras"),
        os.path.join(weight_dir, f"{fi1}.hdf5"),
        os.path.join(weight_dir, f"model_{fi1}.h5"),
        os.path.join(weight_dir, f"model_{fi1}.keras"),
        os.path.join(weight_dir, f"model_{fi1}.hdf5"),
        os.path.join(weight_dir, f"best_{fi1}.h5"),
        os.path.join(weight_dir, f"best_{fi1}.keras"),
        os.path.join(weight_dir, f"best_{fi1}.hdf5"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    wanted = {fi, fi1}
    scored = []
    for fn in os.listdir(weight_dir):
        lfn = fn.lower()
        if not lfn.endswith((".h5", ".keras", ".hdf5", ".weights.h5")):
            continue

        fidx = _extract_fold_index_from_name(fn)
        if fidx is None or (fidx not in wanted):
            continue

        score = 0
        if "best" in lfn:
            score += 5
        if "stage2" in lfn:
            score += 3
        if "final" in lfn:
            score += 2
        if "weights" in lfn:
            score += 1
        score += 2 if fidx == fi else 0
        scored.append((score, fn))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(weight_dir, scored[0][1])

    return None


def have_all_folds(weight_dir: str | None, n_folds: int) -> bool:
    if not weight_dir:
        return False
    for fi in range(n_folds):
        if resolve_fold_weight_path(weight_dir, fi) is None:
            return False
    return True


CAN_USE_TF_INFERENCE = (not NEEDTRAIN) and have_all_folds(WEIGHT_DIR, SPLITS)
print("Can use TF inference:", CAN_USE_TF_INFERENCE)




## === cell 2
def make_prior_submission(
    train_df: pd.DataFrame, test_path: str, out_path: str = "submission.csv"
) -> pd.DataFrame:
    test_df = pd.read_csv(test_path)
    y = train_df[TARGETS].to_numpy(dtype=np.float64)
    row_sum = y.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    y = y / row_sum
    prior = y.mean(axis=0)
    prior = np.clip(prior, 1e-8, 1.0)
    prior = prior / prior.sum()

    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    for k, col in enumerate(TARGETS):
        sub[col] = prior[k]

    sub = sub[["eeg_id"] + list(TARGETS)]
    probs = sub[TARGETS].to_numpy(dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub.loc[:, TARGETS] = probs
    sub.to_csv(out_path, index=False)
    print("Wrote fallback prior submission:", out_path, "shape:", sub.shape)
    print(sub.head())
    return sub


if (not NEEDTRAIN) and (not CAN_USE_TF_INFERENCE):
    _ = make_prior_submission(
        df, os.path.join(LOAD_DATA_FROM, "test.csv"), "submission.csv"
    )



## === cell 3
if CAN_USE_TF_INFERENCE or NEEDTRAIN:
    from PIL import Image
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy import signal
    from sklearn.metrics import confusion_matrix

    import tensorflow as tf
    from tensorflow.keras import optimizers

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    MIX = True
    if not NEEDTRAIN:
        MIX = False

    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision could not be enabled; continuing.")
    else:
        print("Using full precision")



## === cell 4
if CAN_USE_TF_INFERENCE or NEEDTRAIN:

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            dataframe,
            batch_size=32,
            shuffle=False,
            sample_weights=False,
            mode="train",
            eegs=None,
            stfts=None,
            specs=None,
            imgs=None,
        ):

            self.dataframe = dataframe
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sample_weights = sample_weights
            self.mode = mode
            self.eegs = eegs
            self.stfts = stfts
            self.specs = specs
            self.imgs = imgs
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.dataframe) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)

            if self.mode == "test":
                return x
            return x, y, sample_weights

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        (4 * 4 + 2) * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros(
                    (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            if self.mode != "test":
                y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
                sample_weights = np.zeros((len(indexes), 1), dtype="float32")
            else:
                y = np.zeros((0, 0), dtype="float32")
                sample_weights = np.zeros((0, 0), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]

                sign_id = (
                    row["sign_id"]
                    if "sign_id" in row.index
                    else getattr(row, "sign_id", i)
                )

                if self.mode != "test":
                    sample_weight = float(np.sum(row[TARGETS].values)) / 20.0

                if self.mode == "test":
                    r_spe = None
                    r_eeg = 0
                    r_stft = 0
                else:
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        & (df.seizure_vote == row.seizure_vote_raw)
                        & (df.lpd_vote == row.lpd_vote_raw)
                        & (df.gpd_vote == row.gpd_vote_raw)
                        & (df.lrda_vote == row.lrda_vote_raw)
                        & (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)

                    if self.mode == "train":
                        rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                            drop=True
                        )
                        row = rows.loc[0, :]
                    elif self.mode == "valid":
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds

                if "spe" in DATATYPE:
                    spe_full = self.specs[row.spectrogram_id]
                    if self.mode == "test":
                        start_t = max((spe_full.shape[0] - 300) // 2, 0)
                    else:
                        start_t = int(r_spe) if r_spe is not None else 0
                        start_t = max(min(start_t, max(spe_full.shape[0] - 300, 0)), 0)

                    spe = []  # LL RL LP RP
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                spe_full[
                                    start_t : (start_t + 300), k * 100 : (k + 1) * 100
                                ].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
                    eeg_full = self.eegs[row.eeg_id]
                    s0 = int(round(r_eeg * RSFREQ))
                    s1 = int(round((r_eeg + 50) * RSFREQ))
                    if s0 < 0:
                        s0 = 0
                    if s1 > eeg_full.shape[1]:
                        s1 = eeg_full.shape[1]
                    eeg = eeg_full[:, s0:s1]
                    need = int(round(50 * RSFREQ))
                    if eeg.shape[1] < need:
                        pad = need - eeg.shape[1]
                        eeg = np.pad(eeg, ((0, 0), (0, pad)), mode="constant")

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                        stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    spe = np.clip(spe, a_min=1e-6, a_max=1e6)
                    spe = np.log2(spe)

                    spe = spe[
                        :,
                        :,
                        round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]
                    spe = spe[[0, 2, 3, 1], :, :]

                    if self.mode == "train":
                        spe[0:2, :] = spe[0:2, :][np.random.permutation(2), :]
                        spe[2:4, :] = spe[2:4, :][np.random.permutation(2), :]
                        if np.random.rand() > 0.5:
                            spe = spe[::-1, :, :]

                        if np.random.rand() > 0.5:
                            for ii in range(spe.shape[0]):
                                m1 = round(np.random.rand() * spe.shape[2] / 2)
                                m2 = round(np.random.rand() * spe.shape[2] / 2)
                                if np.random.rand() > 0.5:
                                    m1 = spe.shape[2] - m1
                                    m2 = spe.shape[2] - m2
                                m_min = min(m1, m2)
                                m_max = min(
                                    max(m1, m2), m_min + round(spe.shape[2] * 0.1)
                                )
                                spe[ii, :, m_min:m_max] = 0

                    spe = (spe - np.mean(spe, keepdims=True)) / (
                        np.std(spe, keepdims=True) + 1e-6
                    )
                    x_spe[j] = spe

                if "eeg" in DATATYPE:
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]

                    if self.mode == "train":
                        eeg[0:8, :] = eeg[0:8, :][np.random.permutation(8), :]
                        eeg[10:18, :] = eeg[10:18, :][np.random.permutation(8), :]
                        if np.random.rand() > 0.5:
                            eeg = eeg[::-1, :]

                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )
                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                    eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                        np.std(eeg_save, keepdims=True) + 1e-6
                    )
                    x_eeg[j] = eeg

                if "stft" in DATATYPE:
                    stft = np.clip(stft, a_min=1e-6, a_max=1e6)
                    stft = np.log2(stft)

                    if self.mode == "train":
                        stft[0:8, :, :] = stft[0:8, :, :][
                            np.random.permutation(8), :, :
                        ]
                        stft[10:18, :, :] = stft[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :, :]

                    stft_save = np.zeros(
                        (round(stft.shape[0] / 2 * stft.shape[1]), stft.shape[2] * 2),
                        dtype=np.float32,
                    )
                    for ii in range(stft.shape[0]):
                        stft_save[
                            ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                            (ii % 2) * stft.shape[2] : (ii % 2 + 1) * stft.shape[2],
                        ] = stft[ii, :, :]

                    stft = (stft_save - np.mean(stft_save, keepdims=True)) / (
                        np.std(stft_save, keepdims=True) + 1e-6
                    )
                    x_stft[j] = stft

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)

                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                    x_img[j] = img

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1

            x = []
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "stft" in DATATYPE:
                x.append(x_stft)
            if "img" in DATATYPE:
                x.append(x_img)

            return x, y, sample_weights




## === cell 5
if CAN_USE_TF_INFERENCE or NEEDTRAIN:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step

            if warmth_rate == 0:
                self.warm_step = 1
            else:
                self.warm_step = int(self.total_step * warmth_rate)

            self.lr_max = lr_max
            self.lr_min = lr_min

        @tf.function
        def __call__(self, step):
            step = step + 1
            if step < self.warm_step:
                lr = self.lr_max / self.warm_step * step
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0
                    + tf.cos(
                        (step - self.warm_step)
                        / (self.total_step - self.warm_step)
                        * np.pi
                    )
                )
            return lr




## === cell 6
if CAN_USE_TF_INFERENCE or NEEDTRAIN:

    def temporal_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(1, 3), strides=(1, 1), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(1, 3), strides=(1, 2), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def external_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1,
            dilation_rate=(4, 1),
            kernel_size=(4, 1),
            strides=(1, 1),
            padding="valid",
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def internal_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(4, 1), strides=(4, 1), padding="valid"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def build_model():
        inp = []
        y = None

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_spe[:, 0, :, :],
                    inp_spe[:, 1, :, :],
                    inp_spe[:, 2, :, :],
                    inp_spe[:, 3, :, :],
                ]
            )
            x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
            x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    (4 * 4 + 2) * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                )
            )
            x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
                inp_eeg
            )
            x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg[:, :, :, :])

            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

            inp.append(inp_eeg)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            else:
                y = x_eeg

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

            base_model_stft = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

            inp.append(inp_stft)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            else:
                y = x_stft

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(inp_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

            inp.append(inp_img)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
            else:
                y = x_img

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 7
if (not NEEDTRAIN) and CAN_USE_TF_INFERENCE:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

    models = []
    for model_i in range(SPLITS):
        print(f"Fold {model_i + 1}")
        with strategy.scope():
            m = build_model()

        wpath = resolve_fold_weight_path(WEIGHT_DIR, model_i)
        if wpath is None:
            raise FileNotFoundError(
                f"Missing weights for fold {model_i} in {WEIGHT_DIR}"
            )
        print("Loading weights:", wpath)
        m.load_weights(wpath)
        models.append(m)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    if "spe" in DATATYPE:
        PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
        files_test = os.listdir(PATH_test)
        print(f"There are {len(files_test)} test spectrogram parquets")

        for i, f in enumerate(files_test):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH_test}{f}")
            name = int(f.split(".")[0])
            spectrograms_test[name] = tmp.iloc[:, 1:].values

    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

    preds_all = []

    if ("eeg" in DATATYPE) or ("stft" in DATATYPE) or ("img" in DATATYPE):
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")

        batch_start = 0
        for i, eeg_id in enumerate(test.eeg_id):
            if i % 100 == 0:
                print(i, ", ", end="")

            eeg_default = pd.read_parquet(
                os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
            )

            n_samples = len(eeg_default)
            eeg = []
            for channel in BRAIN:
                a_ch, b_ch = channel.split("-")
                if (a_ch in eeg_default.columns) and (b_ch in eeg_default.columns):
                    eeg_temp = (
                        eeg_default.loc[:, a_ch] - eeg_default.loc[:, b_ch]
                    ).values
                else:
                    eeg_temp = np.zeros((n_samples,), dtype=np.float32)
                eeg_temp = eeg_temp.astype(np.float32, copy=False)
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                ff, tt, ss = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                test_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(test_plot)):
                    eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
                    eeg_plot = eeg_plot[
                        :,
                        round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                            (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                        ),
                    ]

                    img_save = np.zeros(
                        (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                    )
                    for ii in range(eeg_plot.shape[0]):
                        fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                        fig.patch.set_facecolor("black")

                        plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)
                        plt.xlim(-5, eeg_plot.shape[1] + 5)
                        plt.ylim(0, 200)
                        plt.axis("off")

                        byte_stream = io.BytesIO()
                        plt.savefig(
                            byte_stream, format="png", bbox_inches="tight", dpi=100
                        )
                        byte_stream.seek(0)
                        img = Image.open(byte_stream)
                        img = np.array(img)[:, :, :1]
                        img = img / 255
                        img = np.array(img, dtype=np.float32)
                        byte_stream.truncate()
                        plt.close("all")

                        if img.shape != (36, IMG_WIDE, 1):
                            img = np.concatenate((img, img, img), 2)
                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]

                        img_save[ii, :, :] = img

                    imgs_test[test_plot.sign_id[j]] = img_save

            eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if "eeg" in DATATYPE:
                eegs_test[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts_test[eeg_id] = ss
                stfts_test[-eeg_id] = tt

            is_batch_end = ((i + 1) % TEST_BATCHSIZE == 0) or (
                (i + 1) == len(test.eeg_id)
            )
            if is_batch_end:
                batch_end = i + 1
                df_batch = test.iloc[batch_start:batch_end, :].reset_index(drop=True)

                test_gen = DataGenerator(
                    df_batch,
                    shuffle=False,
                    sample_weights=False,
                    batch_size=TEST_BATCHSIZE,
                    mode="test",
                    specs=spectrograms_test,
                    eegs=eegs_test,
                    stfts=stfts_test,
                    imgs=imgs_test,
                )

                preds = []
                for model_i in range(SPLITS):
                    pred = models[model_i].predict(test_gen, verbose=0)
                    preds.append(pred)
                pred = np.mean(preds, axis=0)

                preds_all.append(pred)

                eegs_test = {}
                stfts_test = {}
                imgs_test = {}
                gc.collect()

                batch_start = batch_end

    if len(preds_all) > 0:
        preds_all = np.concatenate(preds_all, axis=0)
    else:
        preds_all = np.zeros((0, len(TARGETS)), dtype=np.float32)

    if preds_all.shape[0] != len(test):
        ytr = df[TARGETS].to_numpy(dtype=np.float64)
        rs = ytr.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        ytr = ytr / rs
        prior = ytr.mean(axis=0)
        prior = np.clip(prior, 1e-8, 1.0)
        prior = prior / prior.sum()
        fixed = np.tile(prior.reshape(1, -1), (len(test), 1))
        ncopy = min(preds_all.shape[0], len(test))
        if ncopy > 0:
            fixed[:ncopy, :] = preds_all[:ncopy, :]
        preds_all = fixed.astype(np.float32)
        print("Warning: preds_all length mismatch; filled to match test length.")

    preds_all = np.clip(preds_all, 1e-8, 1.0)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    pred_map = pd.DataFrame(preds_all, columns=list(TARGETS))
    pred_map["eeg_id"] = test.eeg_id.values
    sub = sample_sub[["eeg_id"]].merge(pred_map, on="eeg_id", how="left")

    if sub[TARGETS].isna().any().any():
        y = df[TARGETS].to_numpy(dtype=np.float64)
        row_sum = y.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        y = y / row_sum
        prior = y.mean(axis=0)
        prior = np.clip(prior, 1e-8, 1.0)
        prior = prior / prior.sum()
        sub[TARGETS] = sub[TARGETS].fillna(pd.Series(prior, index=TARGETS))

    probs = sub[TARGETS].to_numpy(dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub.loc[:, TARGETS] = probs

    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
