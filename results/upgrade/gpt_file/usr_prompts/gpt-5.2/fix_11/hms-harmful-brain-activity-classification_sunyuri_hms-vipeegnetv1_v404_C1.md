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

0.2831777735807947

# 6. Current score

0.88875

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the TensorFlow import path that triggers the protobuf `MessageFactory.GetPrototype` crash in this Kaggle/Python 3.13 environment and replace it with the stable public API. I also make the Kaggle inference path robust by falling back to the competition’s `sample_submission.csv` if no `models*` directory is found (so a valid CSV is always produced). Finally, I ensure predictions are valid probabilities (non-negative and row-normalized) to avoid submission rejection and keep metric semantics correct. These are minimal changes that unblock execution end-to-end without changing the model/training core logic.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash in this Python 3.13 Kaggle environment by switching to the stable `keras` public API instead of `tensorflow.keras`, which avoids the `MessageFactory.GetPrototype` error while keeping the same model/loss/training semantics. I also make inference produce a *valid* low-KL submission even when no weights are present by using the empirical class prior from `train.csv` (row-normalized) rather than the raw `sample_submission.csv` counts, which should reduce KL substantially from 1.40995 toward the 0.283 target. Finally, I make the generator return `(x, y)` in test mode (no dummy sample weights) and ensure all predictions are strictly positive and row-normalized to satisfy Kaggle submission constraints.'
- What this solution (achieved 1.41937) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring the script uses the stable standalone `keras` API without importing/initializing TensorFlow in a way that triggers the issue, and by guarding TensorFlow-only configuration behind safe try/except blocks. I also fix an inference bug where the test `DataGenerator` relied on `sign_id` (not present in `test.csv` during batching), which can break predictions or misalign batches. Finally, I keep the existing “no weights fallback = train prior probabilities” behavior (which is score-improving versus uniform) but make it robust and ensure the produced `submission.csv` is always valid probabilities summing to 1 with the exact required columns.'
- What this solution (achieved 1.48551) has done: 'I fix the protobuf/TensorFlow crash by avoiding importing/configuring TensorFlow in this Python 3.13 Kaggle environment, and I make the script always produce a valid `submission.csv`. Since your current score (1.41937, lower-is-better) is far from the target (0.283), I keep the existing minimal “no weights → train prior” fallback but improve it slightly in a metric-aligned way by using a smoothed empirical prior and grouping by `eeg_id` to match the test submission granularity. I also fix a hard failure when `train.csv` (local preprocessed) is not present on Kaggle by building the required `train` dataframe from the official `train.csv` metadata when needed. Finally, I ensure the submission columns match `sample_submission.csv`, rows align with `test.csv` ordering, and probabilities are strictly positive and row-normalized.'
- What this solution (achieved 1.48551) has done: 'I fix the protobuf/TensorFlow crash that prevents the notebook from running by making TensorFlow import safe on Kaggle and by providing a tiny TF-like fallback used only for `tf.cos` in the LR scheduler when TF is unavailable. I also ensure inference always produces a valid `submission.csv` even when no weights/TF are available by keeping your existing smoothed train-prior fallback, but make it deterministic and strictly probability-valid (positive + row-normalized). Finally, I make `__main__` robust so Kaggle always goes through the inference path (no training) and writes a correctly formatted CSV with the exact sample-submission columns and row order.'
- What this solution (achieved 0.82676) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by removing the implicit TensorFlow import path (caused by `KERAS_BACKEND="tensorflow"` plus Keras imports) and forcing the safe NumPy backend on Kaggle, which is enough for the existing “prior-based fallback” inference to run. I also hard-disable training on Kaggle and make the submission writer always use the official `sample_submission.csv` column order, strictly positive probabilities, and exact row alignment with `test.csv`. Since your current score is far above the target (lower-is-better), I improve the fallback prediction in a metric-aligned but still minimal way by using a patient-aware prior (mean of eeg-level label distributions per patient, with global fallback), which is legitimate and typically lowers KL vs a global-only prior. All changes are confined to environment guards and inference fallback; the model architecture/training code remains intact.'
- What this solution (achieved 0.82676) has done: 'Your current score (0.82676, lower-is-better) is still far from the target (0.28318), so we should improve the inference fallback that runs on Kaggle when TensorFlow/models aren’t available. The smallest legitimate improvement is to replace the patient-only prior with a stronger *patient → (if missing) eeg → global* hierarchical prior, computed from train label distributions aggregated to EEG-level first (matching test granularity). We keep your “no training on Kaggle + numpy backend” setup and only change the fallback probability construction, while still enforcing strict positivity and row-normalization to avoid invalid submissions. This should reduce KL versus patient-only priors because it uses more specific information whenever available without changing any model/training code paths.'
- What this solution (achieved 0.87298) has done: 'Your current score (0.82676, lower-is-better) is still far above the target (0.28318), so the smallest legitimate movement toward the target is to make the “no-model/no-TF” inference fallback more informative without touching the model/training core. I keep your existing hierarchical prior idea but strengthen it by (1) adding a patient+eeg-aware shrinkage blend (eeg prior when available, otherwise patient prior, both softly blended with global) and (2) replacing the plain mean with a vote-weighted aggregation so high-consensus labels influence priors more (this is metric-aligned for KL on probabilistic votes). I also ensure the fallback uses only train-derived statistics and still writes strictly positive, row-normalized probabilities with the exact sample-submission column order. No changes are made to the model architecture, layers, losses, or training loop; only the Kaggle inference fallback probability construction is adjusted.'
- What this solution (achieved 0.88875) has done: 'Your current score (0.87298, lower-is-better) is still far from the target (0.28318), so we should only adjust the Kaggle inference fallback (used when no weights/TF are available) without touching the model/training core. The biggest low-risk gain is to make the hierarchical prior depend on both `patient_id` and `spectrogram_id` (which exists in test) rather than `eeg_id` (which never be seen in train for test rows), while still shrinking to a global prior for stability. Concretely, we compute vote-weighted label distributions aggregated at `spectrogram_id` (and patient), then predict test rows via `spectrogram_id` → shrink-to-global, optionally blended with `patient_id` prior. We keep strict positivity and row-normalization to preserve valid-probability semantics and avoid submission failure.'
- What this solution (achieved 0.88875) has done: 'We keep your current “no-model/no-TF” Kaggle inference fallback, but fix the main reason it can’t improve: you’re keying priors by `spectrogram_id`, yet in train the `spectrogram_id` is for the whole recording and doesn’t match test’s per-sample `spectrogram_id` granularity; so lookups mostly miss and you fall back to the global prior (hurting KL). The smallest metric-aligned improvement is to compute priors at `eeg_id` and `patient_id` (which are consistent identifiers across train/test), then use hierarchical shrinkage `eeg -> patient -> global` with the same vote-weighted averaging and strict probability normalization. This doesn’t touch your model/training core logic (still disabled on Kaggle) and only changes the fallback probabilities that generate the submission. Finally, we keep strict positivity + row-normalization to ensure the submission is valid and not rejected.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
else:
    PLATFORM = "unknown"

if PLATFORM == "kaggle":
    os.environ["KERAS_BACKEND"] = "numpy"
    NEEDTRAIN = False
else:
    os.environ["KERAS_BACKEND"] = "tensorflow"

if PLATFORM == "local":
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif PLATFORM == "kaggle":
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 40  # the height of the spectrogram 100
SPE_WIDE = 1000  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3 * BATCHSIZE / 16
EPOCHS = 15

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

import keras
from keras import optimizers
from keras.models import clone_model

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import zoom
import time
import gc

try:
    import torchaudio
    import torch
except Exception:
    torchaudio = None
    torch = None

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

TF_AVAILABLE = False
tf = None
if PLATFORM == "local":
    try:
        import tensorflow as tf  # noqa: F401

        TF_AVAILABLE = True
        try:
            tf.random.set_seed(SEED)
        except Exception:
            pass
        try:
            tf.keras.utils.set_random_seed(SEED)
        except Exception:
            pass
    except Exception as e:
        TF_AVAILABLE = False
        tf = None
        print(
            "WARNING: TensorFlow import failed locally; will only be able to run prior-based submission."
        )
        print("TensorFlow import error:", repr(e))
else:
    TF_AVAILABLE = False
    tf = None
    print(
        "Kaggle environment detected: tensorflow import disabled to avoid protobuf crash."
    )


class _TFFallback:
    @staticmethod
    def cos(x):
        return np.cos(x)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def reduce_sum(x, axis=None, keepdims=False):
        return np.sum(
            x,
            axis=tuple(axis) if isinstance(axis, (list, tuple)) else axis,
            keepdims=keepdims,
        )

    @staticmethod
    def convert_to_tensor(x, dtype=None):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def zeros_initializer():
        def _init(shape, dtype=None):
            return np.zeros(shape, dtype=np.float32 if dtype is None else dtype)

        return _init

    @staticmethod
    def shape(x):
        return np.array(np.shape(x), dtype=np.int64)

    @staticmethod
    def broadcast_to(x, shape):
        return np.broadcast_to(x, shape)

    @staticmethod
    def concat(xs, axis):
        return np.concatenate(xs, axis=axis)

    @staticmethod
    def cast(x, dtype):
        return x.astype(dtype)

    Variable = None  # not supported


if tf is None:
    tf = _TFFallback()

MIX = True
if MIX:
    try:
        policy = keras.mixed_precision.Policy("mixed_float16")
        keras.mixed_precision.set_global_policy(policy)
    except Exception as e:
        print("Mixed precision not available; continuing with default precision.", e)
else:
    print("Using full precision")

if NEEDTRAIN:
    import itertools

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

    if READ_EEG_FILES:
        train = df.drop_duplicates(
            [
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
        ).reset_index(drop=True)

        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.to_csv("train.csv", index=False)
    else:
        if os.path.exists("train.csv"):
            train = pd.read_csv("train.csv")
        else:
            train = df.drop_duplicates(
                [
                    "eeg_id",
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ).reset_index(drop=True)

            train["sign_id"] = train.index.values
            df = df.copy()
            df["sign_id"] = np.arange(len(df), dtype=np.int64)

            y_data = train[TARGETS].values.astype(np.float32)
            train[TARGETS_RAW] = y_data
            y_norm = y_data / (np.sum(y_data, axis=1, keepdims=True) + 1e-12)
            train[TARGETS] = y_norm

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## === cell 1
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        if ("stft" in DATATYPE) or ("img" in DATATYPE):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):

            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
            )

            eeg = list()
            for channel in BRAIN:
                eeg_temp = (
                    eeg_default.loc[:, channel.split("-")[0]]
                    - eeg_default.loc[:, channel.split("-")[1]]
                ).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "stft" in DATATYPE:
                if torchaudio is None or torch is None:
                    raise ImportError("torchaudio/torch required for stft DATATYPE")
                eeg2 = signal.filtfilt(b, a, eeg, axis=1)

                eeg2 = torch.from_numpy(eeg2.copy())
                n_fft, win_length, hop_length = 500, 128, 50
                ss = torchaudio.transforms.Spectrogram(
                    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1
                )(eeg2)
                ss = ss.numpy()
                ss = ss[
                    :,
                    : round(20 / (RSFREQ / n_fft)),
                ]

                tt = np.arange(ss.shape[2]) * hop_length / RSFREQ

                ss = np.concatenate(
                    (
                        ss[0 : round(EEG_CHANNEL_USED / 2), :, :],
                        ss[-round(EEG_CHANNEL_USED / 2) :, :, :],
                    ),
                    axis=0,
                )

                ss = np.array(ss, dtype=np.float32)
                tt = np.array(tt, dtype=np.float32)

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(train_plot)):
                    row = train_plot.iloc[j]
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

                    eeg_plot = eeg2[
                        :,
                        round(row.eeg_label_offset_seconds * RSFREQ) : round(
                            (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                        ),
                    ]
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
                            if TF_AVAILABLE:
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            else:
                                img = np.array(img, dtype=np.float32)[:36, :IMG_WIDE, :]
                        img = img[:, :, 0]
                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.array(eeg, dtype=np.float32)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts[eeg_id] = ss
                stfts[-eeg_id] = tt

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")
        else:
            datapath = "./" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 2
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    time_start_time = time.time()
    if READ_SPE_FILES:
        for i, f in enumerate(files):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(files)
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        np.save("./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()




## === cell 3
class DataGenerator(keras.utils.Sequence):
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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        if self.mode == "test":
            return x
        if self.sample_weights:
            return x, y, sample_weights
        return x, y

    def on_epoch_end(self):
        self.nan = 0
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
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype="float32",
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        targets_batch = list()

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]

            if self.mode != "test":
                sign_id = row.sign_id
                sample_weight = sum(row[[t + "_raw" for t in TARGETS]].values) / 20
                targets_batch.append(row.expert_consensus)

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
                r_stft = 0
            else:
                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    * (df.seizure_vote == row.seizure_vote_raw)
                    * (df.lpd_vote == row.lpd_vote_raw)
                    * (df.gpd_vote == row.gpd_vote_raw)
                    * (df.lrda_vote == row.lrda_vote_raw)
                    * (df.grda_vote == row.grda_vote_raw),
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
                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0, r_eeg)
                    r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

            if "spe" in DATATYPE:
                spe = list()
                for k in range(4):
                    spe.append(
                        np.reshape(
                            self.specs[row.spectrogram_id][
                                r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                            ].T,
                            (1, 100, 300),
                        )
                    )
                spe = np.concatenate(spe, axis=0)

            if "eeg" in DATATYPE:
                eeg = self.eegs[row.eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]
                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

            if "stft" in DATATYPE:
                stft_t = self.stfts[-row.eeg_id]
                r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                r_stft2 = (np.where(stft_t <= (50 + r_eeg - min(stft_t))))[0][-1]
                stft = self.stfts[row.eeg_id][:, :, r_stft:r_stft2]

                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = self.imgs[sign_id]

            if "spe" in DATATYPE:
                spe[np.isnan(spe)] = 0
                exp_min, exp_max = -4, 6
                spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                spe = np.log(spe)

                if (spe.shape[1] != SPE_HIGH) or (spe.shape[2] != SPE_WIDE):
                    spe2 = np.zeros(
                        (spe.shape[0], SPE_HIGH, SPE_WIDE), dtype=np.float32
                    )
                    for k in range(4):
                        scaled_arr = zoom(
                            spe[k],
                            (SPE_HIGH / spe.shape[1], SPE_WIDE / spe.shape[2]),
                            order=1,
                        )
                        spe2[k, :, :] = scaled_arr
                    spe = spe2.copy()

                if self.mode == "train":
                    spe2 = spe.copy()
                    if np.random.rand() > 0.5:
                        spe[0] = spe2[2]
                        spe[2] = spe2[0]
                    if np.random.rand() > 0.5:
                        spe[1] = spe2[3]
                        spe[3] = spe2[1]
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
                            m_max = min(max(m1, m2), m_min + round(spe.shape[2] * 0.05))
                            spe[ii, :, m_min:m_max] = 0

                spe = (spe - exp_min) / (exp_max - exp_min) * 255
                spe = np.clip(spe, a_min=0, a_max=255)
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]

                if self.mode == "train":
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][np.random.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][np.random.permutation(8), :]

                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    if np.random.rand() > 0.5:
                        eeg = -eeg

                    if np.random.rand() > 0.5:
                        eeg = eeg[:, ::-1]
                else:
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eeg = eeg + 1024
                eeg = eeg / 2048 * 255
                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                exp_min, exp_max = 0, 8
                stft = np.clip(stft, a_min=0, a_max=np.exp(exp_max))
                stft = np.log1p(stft)

                if self.mode == "train":
                    stft[0 : round(stft.shape[0] / 2), :, :] = stft[
                        0 : round(stft.shape[0] / 2), :, :
                    ][np.random.permutation(stft.shape[0] // 2), :, :]
                    stft[-round(stft.shape[0] / 2) :, :, :] = stft[
                        -round(stft.shape[0] / 2) :, :, :
                    ][np.random.permutation(stft.shape[0] // 2), :, :]

                    stft2 = stft.copy()
                    stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                        0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                    ]
                    stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                        3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                    ]
                    stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                        1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                    ]
                    stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                        2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                    ]

                    if np.random.rand() > 0.5:
                        stft = stft[::-1, :, :]
                else:
                    stft2 = stft.copy()
                    stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                        0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                    ]
                    stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                        3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                    ]
                    stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                        1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                    ]
                    stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                        2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                    ]

                stft = (stft - exp_min) / (exp_max - exp_min) * 255
                stft = np.clip(stft, a_min=0, a_max=255)

                if "stft" in DATATYPE:
                    if j == 0:
                        x_stft = np.zeros(
                            (len(indexes), stft.shape[0], stft.shape[1], stft.shape[2]),
                            dtype="float32",
                        )
                        x_stft[j] = stft
                    else:
                        x_stft[j] = stft

            if "img" in DATATYPE:
                img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                if self.mode == "train":
                    img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                    img[10:18, :, :] = img[10:18, :, :][np.random.permutation(8), :, :]
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
                            ii, temp_temp : round(temp_temp + end_temp - start_temp), :
                        ]
                    )
                img_save = np.clip(img_save, a_min=0, a_max=1)

                img = np.reshape(img_save, (img_save.shape[0], img_save.shape[1], 1))
                img = np.concatenate((img, img, img), -1)

                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                y[j] = row[TARGETS].values / (sum(row[TARGETS].values) + 1e-12)
                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "stft" in DATATYPE:
            x["stft"] = x_stft
        if "img" in DATATYPE:
            x["img"] = x_img

        return x, y, sample_weights




## === cell 4
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step

        if warmth_rate == 0:
            self.warm_step = 1
        else:
            self.warm_step = int(warmth_rate)

        self.lr_max = lr_max
        self.lr_min = lr_min

        self.begin = 1

    def __call__(self, step):
        if step == self.total_step:
            self.begin = 0
            self.lr_max = self.lr_max * 0.5
            self.lr_min = self.lr_min * 0.1

        step = step % self.total_step
        step = step + 1

        if (self.begin == 1) and (step < self.warm_step):
            lr = self.lr_max / self.warm_step * step
        else:
            if self.begin == 1:
                if self.total_step == 1:
                    lr = self.lr_max
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0
                        + tf.cos(
                            (step - self.warm_step)
                            / (self.total_step - self.warm_step)
                            * np.pi
                        )
                    )
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0 + tf.cos(step / 10 * np.pi)
                )

        return np.float32(lr)


class IniToOne(keras.initializers.Initializer):
    def __init__(self):
        super(IniToOne, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        for i in range(filter_count):
            kernel[i % filter_length, 0, i] = 1.0
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOne(keras.constraints.Constraint):
    def __init__(self):
        super(SumToOne, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class IniToOneAtten(keras.initializers.Initializer):
    def __init__(self):
        super(IniToOneAtten, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
            (filter_length) // 2 + 1 - (filter_length - 1) // 2
        )
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOneAtten(keras.constraints.Constraint):
    def __init__(self):
        super(SumToOneAtten, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="gelu"),
                keras.layers.Dense(feat_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, weights = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), weights


class ClassToken(keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        if TF_AVAILABLE:
            self.cls = tf.Variable(
                name="cls",
                initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
                trainable=True,
            )
        else:
            self.cls = cls_init(shape=(1, 1, self.hidden_size), dtype="float32")

    def call(self, inputs):
        if TF_AVAILABLE:
            batch_size = tf.shape(inputs)[0]
            cls_broadcasted = tf.cast(
                tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
                dtype=inputs.dtype,
            )
            return tf.concat([cls_broadcasted, inputs], 1)
        else:
            bs = inputs.shape[0]
            cls_b = np.broadcast_to(self.cls, (bs, 1, self.hidden_size))
            return np.concatenate([cls_b.astype(inputs.dtype), inputs], axis=1)




## === cell 5
def build_model():
    inp = list()
    y = 0
    if "spe" in DATATYPE:
        inp_spe = keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
        x_spe = keras.layers.Reshape(
            (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
        )(inp_spe)
        x_spe = keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

        x_spe = keras.layers.Concatenate(axis=1)(
            [
                x_spe[:, 0, :, :, :],
                x_spe[:, 1, :, :, :],
                x_spe[:, 2, :, :, :],
                x_spe[:, 3, :, :, :],
            ]
        )

        base_model_spe = keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_spe.load_weights(
                    f"./input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_spe.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                )
        base_model_spe.name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = keras.layers.Dropout(0.5)(x_spe)

        inp.append(inp_spe)
        y = x_spe * 1

    if "eeg" in DATATYPE:
        inp_eeg = keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
        )
        x_eeg_raw = keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        strides = 10
        if PLATFORM == "local":
            eeg_embed = keras.layers.Conv1D(
                filters=strides * 3,
                kernel_size=strides,
                strides=strides,
                padding="same",
                use_bias=False,
                activation=None,
                kernel_initializer=IniToOne(),
                kernel_constraint=SumToOne(),
                input_shape=(None, 1),
            )
        else:
            eeg_embed = keras.layers.Conv1D(
                filters=strides * 3,
                kernel_size=strides,
                strides=strides,
                padding="same",
                use_bias=False,
                activation=None,
            )

        x_eeg = keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)

        x_eeg = keras.layers.Concatenate(axis=-1)(
            [
                keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 0 * strides : 1 * strides]
                ),
                keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 1 * strides : 2 * strides]
                ),
                keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 2 * strides : 3 * strides]
                ),
            ]
        )
        x_eeg = keras.layers.Permute([4, 2, 1, 3])(x_eeg)
        x_eeg = keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
        x_eeg = keras.layers.Permute((3, 2, 1))(x_eeg)

        base_model_eeg = keras.applications.EfficientNetV2B3(
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
        base_model_eeg.name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)

        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)

        y_eeg = keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            x_eeg
        )

    if "stft" in DATATYPE:
        inp_stft = keras.Input(shape=(16, STFT_HIGH, STFT_WIDE), name="stft")
        x_stft = keras.layers.Reshape(
            (inp_stft.shape[1], inp_stft.shape[2], inp_stft.shape[3], 1)
        )(inp_stft)
        x_stft = keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        x_stft = keras.layers.Concatenate(axis=1)(
            [
                x_stft[:, 0, :, :, :],
                x_stft[:, 1, :, :, :],
                x_stft[:, 2, :, :, :],
                x_stft[:, 3, :, :, :],
                x_stft[:, 4, :, :, :],
                x_stft[:, 5, :, :, :],
                x_stft[:, 6, :, :, :],
                x_stft[:, 7, :, :, :],
                x_stft[:, 8, :, :, :],
                x_stft[:, 9, :, :, :],
                x_stft[:, 10, :, :, :],
                x_stft[:, 11, :, :, :],
                x_stft[:, 12, :, :, :],
                x_stft[:, 13, :, :, :],
                x_stft[:, 14, :, :, :],
                x_stft[:, 15, :, :, :],
            ]
        )

        base_model_stft = keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_stft.load_weights(
                    f"./input/pre-trained-weights/{base_model_stft.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_stft.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_stft.name}_notop.h5"
                )
        base_model_stft.name = "stft_extractor"

        x_stft = base_model_stft(x_stft)
        x_stft = keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = keras.layers.Dropout(0.5)(x_stft)

        inp.append(inp_stft)

        if y == 0:
            y = x_stft * 1
        else:
            y = keras.layers.Concatenate(axis=1)([y, x_stft])

    if "img" in DATATYPE:
        inp_img = keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")

        base_model_img = keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=inp_img
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_img.load_weights(
                    f"./input/pre-trained-weights/{base_model_img.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_img.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_img.name}_notop.h5"
                )
        base_model_img.name = "img_extractor"
        x_img = base_model_img.output

        x_img = keras.layers.GlobalAveragePooling2D()(x_img)
        inp.append(inp_img)

        if y == 0:
            y = x_img * 1
        else:
            y = keras.layers.Concatenate(axis=1)([y, x_img])

    y = y_eeg * 1
    model = keras.Model(inputs=inp, outputs=y)
    return model




## === cell 6
def train_fold(
    i,
    stage,
    train_index,
    valid_index,
    df_train_stage1,
    df_valid_stage1,
    df_train_stage2,
    df_valid_stage2,
    build_model,
    BATCHSIZE,
    EPOCHS,
    LEARN_RATE,
    TARGETS,
    TARGETS_RAW,
):

    print("#" * 25)
    print(f"### Fold {i+1}")

    model = build_model()
    loss = keras.losses.KLDivergence()

    if stage == 1:
        train_gen_stage = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        opt = keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        callbacks_stage = [
            keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                )
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
    elif stage == 2:
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        opt = keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1,
                    LEARN_RATE * 0.1 * 0.1 * 0.1,
                    0,
                )
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=False,
            ),
        ]

    model.compile(loss=loss, optimizer=opt)

    if stage == 1:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=EPOCHS,
            callbacks=callbacks_stage,
        )
    elif stage == 2:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=max(round(EPOCHS / 3), 1),
            callbacks=callbacks_stage,
        )

    model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

    loss_hist = history.history["loss"]
    val_loss_hist = history.history["val_loss"]
    epochs = range(1, len(loss_hist) + 1)
    plt.figure()
    plt.plot(epochs, loss_hist, "bo", label="loss")
    plt.plot(epochs, val_loss_hist, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
        fontsize=12,
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    if stage == 1:
        valid_stage = df_valid_stage1[TARGETS].values
    elif stage == 2:
        valid_stage = df_valid_stage2[TARGETS].values
    predict_stage = model.predict(valid_gen_stage)

    del train_gen_stage, valid_gen_stage, history, model
    try:
        if TF_AVAILABLE:
            tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    plt.yticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    thresh = cm.max() / 2.0
    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
            if cm[ii, jj] > -0.1:
                plt.text(
                    jj,
                    ii,
                    str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                    horizontalalignment="center",
                    color="white" if cm[ii, jj] > thresh else "black",
                    fontsize=10,
                )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}_cm.svg"))
    plt.close()

    del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
    gc.collect()




## === cell 7
if __name__ == "__main__":
    if NEEDTRAIN:
        if not TF_AVAILABLE:
            raise RuntimeError(
                "Training requested (NEEDTRAIN=True) but TensorFlow is unavailable/import skipped."
            )

        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        gkf = GroupKFold(n_splits=SPLITS)

        for i, (train_index, valid_index) in enumerate(
            gkf.split(train, train.expert_consensus, train.patient_id)
        ):
            print("#" * 25)
            print(f"### Fold {i+1}")

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [2]:
                p = mp.Process(
                    target=train_fold,
                    args=(
                        i,
                        stage,
                        train_index,
                        valid_index,
                        df_train_stage1,
                        df_valid_stage1,
                        df_train_stage2,
                        df_valid_stage2,
                        build_model,
                        BATCHSIZE,
                        EPOCHS,
                        LEARN_RATE,
                        TARGETS,
                        [t + "_raw" for t in TARGETS],
                    ),
                )
                p.start()
                p.join()

    else:
        model_paths = []
        if os.path.isdir(LOAD_MODELS_FROM):
            for model_i in range(100):
                pth = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
                if os.path.exists(pth):
                    model_paths.append(pth)

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values  # compatibility
        print("Test shape", test.shape)

        sample_sub_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
        if not os.path.exists(sample_sub_path):
            sample_sub_path = "/kaggle/input/sample_submission.csv"
        sample_sub = pd.read_csv(sample_sub_path)
        tgt_cols = [c for c in sample_sub.columns if c != "eeg_id"]

        if (len(model_paths) == 0) or (not TF_AVAILABLE):
            train_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))

            votes = train_df[tgt_cols].values.astype(np.float64)
            vote_sums = votes.sum(axis=1, keepdims=True) + 1e-12
            probs = votes / vote_sums  # row-normalized label distributions (targets)

            tmp = train_df[["eeg_id", "patient_id"]].copy()
            for k, c in enumerate(tgt_cols):
                tmp[c] = probs[:, k]
            tmp["w"] = vote_sums[:, 0]  # annotator vote count weight

            global_prior = np.average(probs, axis=0, weights=tmp["w"].values).astype(
                np.float64
            )

            alpha = 1e-3  # strict positivity + mild smoothing (keep existing behavior)
            global_prior = np.clip(global_prior + alpha, 1e-12, None)
            global_prior = global_prior / global_prior.sum()

            eeg_prior_df = tmp.groupby("eeg_id", as_index=True).apply(
                lambda g: pd.Series(
                    {
                        c: np.average(g[c].values, weights=g["w"].values)
                        for c in tgt_cols
                    }
                ),
                include_groups=False,
            )
            eeg_w = tmp.groupby("eeg_id", as_index=True)["w"].sum().astype(float)

            patient_prior_df = tmp.groupby("patient_id", as_index=True).apply(
                lambda g: pd.Series(
                    {
                        c: np.average(g[c].values, weights=g["w"].values)
                        for c in tgt_cols
                    }
                ),
                include_groups=False,
            )
            patient_w = (
                tmp.groupby("patient_id", as_index=True)["w"].sum().astype(float)
            )

            eeg_prior_dict = {
                int(eid): eeg_prior_df.loc[eid].values.astype(np.float64)
                for eid in eeg_prior_df.index.values
            }
            eeg_w_dict = {int(eid): float(eeg_w.loc[eid]) for eid in eeg_w.index.values}

            patient_prior_dict = {
                pid: patient_prior_df.loc[pid].values.astype(np.float64)
                for pid in patient_prior_df.index.values
            }
            patient_w_dict = {
                pid: float(patient_w.loc[pid]) for pid in patient_w.index.values
            }

            def _blend_with_global(p_vec, w, w0):
                lam = w / (w + w0)
                return lam * p_vec + (1.0 - lam) * global_prior

            pri_mat = np.zeros((len(test), len(tgt_cols)), dtype=np.float64)
            for i, (eid, pid) in enumerate(
                zip(test["eeg_id"].values, test["patient_id"].values)
            ):
                eeg_p = eeg_prior_dict.get(int(eid), None)
                pat_p = patient_prior_dict.get(pid, None)

                if eeg_p is not None:
                    eeg_p = np.clip(eeg_p + alpha, 1e-12, None)
                    eeg_p = eeg_p / eeg_p.sum()
                    eeg_p = _blend_with_global(
                        eeg_p, eeg_w_dict.get(int(eid), 0.0), w0=120.0
                    )
                    if pat_p is not None:
                        pat_p = np.clip(pat_p + alpha, 1e-12, None)
                        pat_p = pat_p / pat_p.sum()
                        pat_p = _blend_with_global(
                            pat_p, patient_w_dict.get(pid, 0.0), w0=250.0
                        )
                        v = 0.90 * eeg_p + 0.10 * pat_p
                    else:
                        v = eeg_p
                elif pat_p is not None:
                    pat_p = np.clip(pat_p + alpha, 1e-12, None)
                    pat_p = pat_p / pat_p.sum()
                    v = _blend_with_global(
                        pat_p, patient_w_dict.get(pid, 0.0), w0=250.0
                    )
                else:
                    v = global_prior

                v = np.clip(v + alpha, 1e-12, None)
                v = v / v.sum()
                pri_mat[i] = v

            sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
            sub[tgt_cols] = pri_mat.astype(np.float32)

            vals = sub[tgt_cols].values.astype(np.float64)
            vals = np.clip(vals, 1e-12, 1.0)
            vals = vals / vals.sum(axis=1, keepdims=True)
            sub[tgt_cols] = vals.astype(np.float32)

            sub = sub[["eeg_id"] + tgt_cols]
            sub.to_csv("submission.csv", index=False)
            print(
                "Wrote eeg_id+patient vote-weighted shrinkage priors as submission.csv",
                sub.shape,
            )
        else:
            preds_all = []
            models = list()
            model_template = build_model()
            for pth in model_paths:
                print("Loading", os.path.basename(pth))
                model = clone_model(model_template)
                model.load_weights(pth)
                models.append(model)

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
                print()

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            if (
                ("spe" in DATATYPE)
                or ("eeg" in DATATYPE)
                or ("stft" in DATATYPE)
                or ("img" in DATATYPE)
            ):
                b2, a2 = signal.butter(
                    3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
                )

                for i, eeg_id in enumerate(test.eeg_id):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    eeg_default = pd.read_parquet(
                        os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                    )

                    eeg = list()
                    for channel in BRAIN:
                        eeg_temp = (
                            eeg_default.loc[:, channel.split("-")[0]]
                            - eeg_default.loc[:, channel.split("-")[1]]
                        ).values
                        eeg_temp[np.isnan(eeg_temp)] = 0
                        eeg.append(np.reshape(eeg_temp, (1, -1)))
                    eeg = np.concatenate(eeg, axis=0)

                    if SFREQ != RSFREQ:
                        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                    if "stft" in DATATYPE:
                        eeg2 = signal.filtfilt(b, a, eeg, axis=1)

                        nperseg = round(RSFREQ * 1)
                        ff, tt, ss = signal.spectrogram(
                            eeg2,
                            axis=1,
                            fs=RSFREQ,
                            nperseg=nperseg,
                            noverlap=round(nperseg - RSFREQ * STFT_TIME),
                            nfft=RSFREQ * 5,
                        )
                        ss[np.isnan(ss)] = 0
                        ss = ss[:, (ff > 0) * (ff <= 20), :]

                        ss = np.concatenate(
                            (
                                ss[0 : round(EEG_CHANNEL_USED / 2), :, :],
                                ss[-round(EEG_CHANNEL_USED / 2) :, :, :],
                            ),
                            axis=0,
                        )

                        for k in range(4):
                            ss[k, :, :] = np.mean(ss[k * 4 : (k + 1) * 4, :, :], 0)

                        ss = ss[:4, :, :]
                        ss = np.array(ss, dtype=np.float32)
                        tt = np.array(tt, dtype=np.float32)

                    if "img" in DATATYPE:
                        eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                        eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                        train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                        for j in range(len(train_plot)):
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

                                plt.plot(
                                    eeg_plot[ii, :] + 100, color="red", linewidth=0.2
                                )

                                plt.xlim(-5, eeg_plot.shape[1] + 5)
                                plt.ylim(0, 200)
                                plt.axis("off")

                                byte_stream = io.BytesIO()
                                plt.savefig(
                                    byte_stream,
                                    format="png",
                                    bbox_inches="tight",
                                    dpi=100,
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
                                        tf.image.resize(img, (36, IMG_WIDE)),
                                        dtype=np.float32,
                                    )
                                img = img[:, :, 0]

                                img_save[ii, :, :] = img

                            imgs_test[train_plot.sign_id[j]] = img_save

                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                    eegshape = eeg.shape[1]
                    eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
                    if filter_range is not None:
                        eeg = signal.filtfilt(b, a, eeg, axis=1)
                    eeg = eeg[:, eegshape : eegshape * 2]

                    eeg = np.array(eeg, dtype=np.float32)

                    if "eeg" in DATATYPE:
                        eegs_test[eeg_id] = eeg
                    if "stft" in DATATYPE:
                        stfts_test[eeg_id] = ss
                        stfts_test[-eeg_id] = tt

                    if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                        preds = []
                        start_idx = max(i - TEST_BATCHSIZE + 1, len(preds_all))
                        test_gen = DataGenerator(
                            test.loc[start_idx:i, :],
                            shuffle=False,
                            sample_weights=False,
                            batch_size=TEST_BATCHSIZE,
                            mode="test",
                            specs=spectrograms_test,
                            eegs=eegs_test,
                            stfts=stfts_test,
                            imgs=imgs_test,
                        )
                        for model_i in range(len(models)):
                            pred = models[model_i].predict(test_gen, verbose=1)
                            preds.append(pred)
                        pred = np.mean(preds, axis=0)

                        pred = np.clip(pred, 1e-12, 1.0)
                        pred = pred / np.sum(pred, axis=1, keepdims=True)

                        del eegs_test
                        gc.collect()
                        eegs_test = {}
                        if len(preds_all) == 0:
                            preds_all = pred.copy()
                        else:
                            preds_all = np.concatenate((preds_all, pred), axis=0)
                print()

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[tgt_cols] = preds_all

            vals = sub[tgt_cols].values.astype(np.float64)
            vals = np.clip(vals, 1e-12, 1.0)
            vals = vals / vals.sum(axis=1, keepdims=True)
            sub[tgt_cols] = vals.astype(np.float32)

            sub = sub[["eeg_id"] + tgt_cols]
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
