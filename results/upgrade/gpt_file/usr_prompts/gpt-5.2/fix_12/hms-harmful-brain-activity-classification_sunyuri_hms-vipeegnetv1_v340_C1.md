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

0.299238872720528

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash by pinning TensorFlow’s protobuf backend to the pure-Python implementation (this avoids the `MessageFactory.GetPrototype` incompatibility that can happen in newer Python/protobuf combos) before TensorFlow is imported. I also make the Kaggle inference path robust by ensuring `train.csv` is not required when `NEEDTRAIN=False`, and by making the test batching slice correct (the current code mixes up `len(preds_all)` with row indices). Finally, I add a safe fallback that produces a valid `submission.csv` (uniform probabilities) if no model weights are found, so you always get a submit-ready CSV; if weights exist, the core model/prediction logic remains unchanged.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf backend before any TensorFlow import and by removing the unused `reset_default_graph` import that can break on TF2/Keras3. I also make inference more robust by compiling the cloned models (so `predict` is stable across TF/Keras versions) and by hardening the test batching logic to never produce a misaligned `preds_all` length. Finally, I add a safe post-processing step that ensures probabilities are finite, clipped, and sum to 1 for every `eeg_id`, which directly protects the KL metric and avoids invalid submissions without changing the core model. These are minimal, execution-unblocking and score-improving calibration fixes (your current 1.40995 is far from the 0.299 target, so improvement is needed).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *before any TensorFlow/protobuf import* and by adding a safe fallback that skips TF-based inference if TF still fails to import (so you always get a valid `submission.csv`). I also correct a key inference bug in `DataGenerator`: in `mode="test"` it currently uses the per-batch `sign_id` (0..batch-1) to index `imgs_test`, which was stored using the global `sign_id` (from `test.index`), causing wrong/empty inputs and severely harming the score; the fix is to use `row.eeg_id` for test mode (which is stable and available). Finally, I ensure submission probabilities are finite, clipped, and row-normalized (score-safe for KL) without changing the model/training logic.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf import crash by forcing a compatible pure-Python protobuf backend early and (critically) pinning protobuf to the “python” implementation before any TF-related modules are touched, plus adding a safe fallback path that still produces a valid `submission.csv` if TF cannot be imported on this Python 3.13 runtime. Then we address a key inference-quality bug: during test-time generation, the code must index `imgs_test`/cached items by `eeg_id` consistently (not per-batch indices), and we also ensure cached dictionaries are cleared correctly between batches to avoid mixing old/new samples. Finally, we keep the model/training core unchanged, but add a small, metric-safe probability post-processing (finite + clipping + renormalization) to reduce KL blow-ups, which should improve the score from the current very poor 1.40995 toward the target band.'
- What this solution (achieved 1.40995) has done: 'You’re currently crashing at TensorFlow import with the protobuf `MessageFactory.GetPrototype` incompatibility (common on newer Python/protobuf combos), so the first fix is to force the pure-Python protobuf implementation *and* ensure it’s set before anything that could transitively import protobuf/TensorFlow. Next, because your score is far worse than the target (lower-is-better), we should avoid the uniform-fallback path and instead make sure inference actually uses the intended EEG inputs by keeping the test generator keyed by `eeg_id` (and ensuring the per-batch cache dictionaries don’t mix samples). Finally, we keep your model/training core intact and only add strictly metric-safe postprocessing (finite/clipped/row-normalized probabilities) and guaranteed CSV writing to produce a valid submission.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` via environment before any TensorFlow/protobuf-related import; if TF still cannot import on this Python 3.13 runtime, we fall back to a valid (but lower-score) uniform submission to guarantee a CSV is produced. To improve the score toward the target (lower is better) when TF works, we ensure inference uses the intended cached test inputs by consistently keying `imgs_test`/etc. by `eeg_id` (not per-batch indices) and keep the test batching concatenation aligned to `test` order. Finally, we keep the model/training core unchanged but harden probability post-processing (finite, clipped, renormalized) to avoid KL blow-ups and invalid rows.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash on Kaggle’s Python 3.13 by forcing the C++ protobuf backend (the “python” backend is what triggers the `MessageFactory.GetPrototype` error in this environment) before any TensorFlow/protobuf import happens. I also make inference robust by ensuring the test DataGenerator always keys cached test items by `eeg_id` (not batch-local indices) and by guaranteeing the final prediction array is aligned to `test.csv` row order and row-normalized for KL safety. These changes keep your model/training core intact while unblocking real model inference (so you don’t fall back to uniform predictions) and should substantially improve the score from 1.40995 toward the 0.299 target. Finally, the script always write a valid `submission.csv` with correct columns and probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the C++ protobuf backend (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`) before TensorFlow is imported, which is the most common stable setting on Kaggle/Python 3.13 for this error. I also ensure inference always builds/loads models inside the active distribution strategy scope, so weight loading and predict execution are consistent across TF/Keras versions and devices (this is score-positive because it prevents silent misbehavior). Finally, I keep your existing model/data logic intact, but harden the submission post-processing to guarantee strictly valid probabilities (finite, clipped, row-normalized), which is KL-safe and should reduce the current very poor score toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.299), so we should make the smallest fixes that restore correct inference rather than changing the model. The biggest score-killer is that the test generator uses `r_eeg=0` for all test rows, so you always feed only the first 50s window regardless of the “central” convention used in training; we align test-time cropping to match the training/valid convention by using `r_eeg = (EEG_LENGTH - 50)/2 = 0` generally, but more importantly we make it robust if `EEG_LENGTH_USED` changes and keep the same centering logic. Next, we ensure inference uses the stage2 weights from the actual folder reliably by setting `SPLITS` from discovered weight files (so you don’t silently average missing folds or crash), without changing any training logic. Finally, we keep your existing KL-safe postprocessing, but also apply it to the per-fold predictions before averaging to prevent any fold producing near-zeros that can blow up KL.'
- What this solution (achieved 1.40995) has done: 'Your current gap to the target is large (1.40995 vs 0.299, lower-is-better), so the biggest likely score-killer is incorrect test-time EEG preprocessing compared to what your trained model expects. I make the test-time EEG path match the train/valid generator exactly by applying the same channel selection/reordering, centering/cropping to `EEG_LENGTH_USED`, and scaling to the model’s 0..255 range before caching `eegs_test`. This is a minimal change that preserves your model and inference loop while fixing a train/test preprocessing mismatch that can severely inflate KL. I also keep the existing KL-safe probability postprocessing and submission formatting unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest likely score-positive fix is to remove a train/test preprocessing mismatch in the test path: right now you preprocess EEG twice (once in `_prep_eeg_like_generator`, then again inside `DataGenerator`), which distorts the input distribution and can severely hurt KL. I keep your architecture/training/inference flow unchanged, but make test-time EEG caching store the *raw* 18-channel differential EEG (like training), and let `DataGenerator` apply the exact same scaling/cropping/channel selection as it does for train/valid. Additionally, I make a tiny robustness tweak so `DataGenerator` uses `self.dataframe` (not the global `df`) to look up rows when training isn’t present, without changing semantics. Submission writing and KL-safe probability normalization are kept as-is.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # *** local kaggle *** local training or online testing
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
    NEEDTRAIN = False
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

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 45
STFT_TIME = 0.15
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / STFT_TIME)  # the width of the STFT (eeg spectrogram)

filter_range = None  # eeg filtering range   [0.5, 45] None
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

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    tf = None

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

if NEEDTRAIN:
    df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
    TARGETS = df.columns[-6:]
else:
    _ss = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    TARGETS = _ss.columns[1:]

print("Targets", list(TARGETS))

if (not TF_AVAILABLE) and NEEDTRAIN:
    raise RuntimeError(
        f"TensorFlow failed to import; cannot train. Error was: {TF_IMPORT_ERROR}"
    )



## === cell 1
if TF_AVAILABLE:
    print(tf.version.VERSION)
    print(tf.config.list_physical_devices("GPU"))
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    MIX = True
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("Using full precision")



## === cell 2
if NEEDTRAIN:
    df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
    print("Train shape:", df.shape)

    TARGETS_RAW = [c + "_raw" for c in TARGETS]

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
        train = pd.read_csv("train.csv")



## === cell 3
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        if filter_range is not None:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
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

            eeg = []
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
                nperseg = round(RSFREQ * 0.2)
                ff, tt, ss = signal.spectrogram(
                    eeg2,
                    axis=1,
                    fs=RSFREQ,
                    nperseg=nperseg,
                    noverlap=round(nperseg - RSFREQ * STFT_TIME),
                    nfft=320,
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

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
                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]
                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

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

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 4
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



## === cell 5
if TF_AVAILABLE:

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
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]

                if self.mode == "test":
                    sign_key = row.eeg_id
                else:
                    sign_key = row.sign_id

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = max(0.0, (EEG_LENGTH - 50.0) / 2.0)
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

                if "spe" in DATATYPE:
                    spe = []
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

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    r_stft2 = (np.where(stft_t <= (50 + r_eeg - min(stft_t))))[0][-1]
                    stft = self.stfts[row.eeg_id][:, :, r_stft:r_stft2]
                    stft = stft[
                        :,
                        :,
                        round((50 - STFT_LENGTH) / 2 / STFT_TIME) : round(
                            (50 + STFT_LENGTH) / 2 / STFT_TIME
                        ),
                    ]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                        stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_key]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    exp_min, exp_max = -4, 6
                    spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    spe = np.log(spe)
                    spe = spe[
                        :,
                        :,
                        round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]
                    if self.mode == "train":
                        spe2 = spe.copy()
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[2]
                            spe[2] = spe2[0]
                        if np.random.rand() > 0.5:
                            spe[1] = spe2[3]
                            spe[3] = spe2[1]
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[1]
                            spe[2] = spe2[3]
                            spe[1] = spe2[0]
                            spe[3] = spe[2]
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
                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )
                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

                    if self.mode == "train":
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
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

                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]
                    else:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                    eeg = np.clip(eeg_save, a_min=-255, a_max=255)
                    eeg = eeg + 255
                    eeg = eeg / 2
                    x_eeg[j] = eeg

                if "stft" in DATATYPE:
                    exp_min, exp_max = -6, 6
                    stft = np.clip(stft, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    stft = np.log(stft)
                    stft = np.concatenate(
                        (
                            stft[0 : round(EEG_CHANNEL_USED / 2), :, :],
                            stft[-round(EEG_CHANNEL_USED / 2) :, :, :],
                        ),
                        axis=0,
                    )
                    if self.mode == "train":
                        stft[0 : round(EEG_CHANNEL_USED / 2), :] = stft[
                            0 : round(EEG_CHANNEL_USED / 2), :
                        ][np.random.permutation(8), :]
                        stft[-round(EEG_CHANNEL_USED / 2) :, :] = stft[
                            -round(EEG_CHANNEL_USED / 2) :, :
                        ][np.random.permutation(8), :]
                        stft2 = stft.copy()
                        stft[4:8, :] = stft2[12:16, :]
                        stft[8:12, :] = stft2[4:8, :]
                        stft[12:16, :] = stft2[8:12, :]
                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :]
                    else:
                        stft2 = stft.copy()
                        stft[4:8, :] = stft2[12:16, :]
                        stft[8:12, :] = stft2[4:8, :]
                        stft[12:16, :] = stft2[8:12, :]

                    stft_save = np.zeros(
                        (stft.shape[0] * stft.shape[1] // 2, stft.shape[2] * 2),
                        dtype=np.float32,
                    )
                    for ii in range(stft.shape[0]):
                        stft_save[
                            ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                            ii % 2 :: 2,
                        ] = stft[ii, :, :]

                    stft = (stft_save - exp_min) / (exp_max - exp_min) * 255
                    stft = np.clip(stft, a_min=0, a_max=255)

                    if self.mode == "train":
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * stft.shape[1])
                            stft[
                                :,
                                mask : round(
                                    mask + np.random.rand() * stft.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * stft.shape[1])
                            stft[
                                :,
                                mask : round(
                                    mask + np.random.rand() * stft.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * stft.shape[1])
                            stft[
                                :,
                                mask : round(
                                    mask + np.random.rand() * stft.shape[1] * 0.02
                                ),
                            ] = 0

                    if j == 0:
                        x_stft = np.zeros(
                            (len(indexes), stft.shape[0], stft.shape[1]),
                            dtype="float32",
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
                        sample_weights[j] = (
                            sum(row[[t + "_raw" for t in TARGETS]].values) / 20
                        )
                    else:
                        sample_weights[j] = 1.0
                else:
                    sample_weights[j] = 1.0

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




## === cell 6
if TF_AVAILABLE:

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
            return np.float32(lr)

    class IniToOne(tf.keras.initializers.Initializer):
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

    class SumToOne(tf.keras.constraints.Constraint):
        def __init__(self):
            super(SumToOne, self).__init__()

        def __call__(self, w):
            w = tf.abs(w)
            w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
            return w_normed

        def get_config(self):
            return {}

    class IniToOneAtten(tf.keras.initializers.Initializer):
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

    class SumToOneAtten(tf.keras.constraints.Constraint):
        def __init__(self):
            super(SumToOneAtten, self).__init__()

        def __call__(self, w):
            w = tf.abs(w)
            w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
            return w_normed

        def get_config(self):
            return {}

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(ff_dim, activation="gelu"),
                    tf.keras.layers.Dense(feat_dim),
                ]
            )
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output, weights = self.att(
                inputs, inputs, return_attention_scores=True
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output), weights

    class ClassToken(tf.keras.layers.Layer):
        def build(self, input_shape):
            cls_init = tf.zeros_initializer()
            self.hidden_size = input_shape[-1]
            self.cls = tf.Variable(
                name="cls",
                initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
                trainable=True,
            )

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            cls_broadcasted = tf.cast(
                tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
                dtype=inputs.dtype,
            )
            return tf.concat([cls_broadcasted, inputs], 1)




## === cell 7
if TF_AVAILABLE:

    def build_model():
        inp = []

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
            x_spe = tf.keras.layers.Reshape(
                (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
            )(inp_spe)
            x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

            base_model_spe = tf.keras.applications.EfficientNetV2B0(
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
            base_model_spe._name = "spe_extractor"

            base_model_spe_pre = tf.keras.Model(
                base_model_spe.input, base_model_spe.get_layer("block3b_add").output
            )
            base_model_spe_pre._name = "spe_extractor_pre"
            x_spe1 = base_model_spe_pre(x_spe[:, 0, :, :, :])
            x_spe2 = base_model_spe_pre(x_spe[:, 1, :, :, :])
            x_spe3 = base_model_spe_pre(x_spe[:, 2, :, :, :])
            x_spe4 = base_model_spe_pre(x_spe[:, 3, :, :, :])

            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )
            base_model_spe_after = tf.keras.Model(
                base_model_spe_pre.output, base_model_spe.output
            )
            base_model_spe_after._name = "spe_extractor_after"
            x_spe = base_model_spe_after(x_spe)

            x_spe = x_spe[
                :, :, (x_spe.shape[2] - 1) // 2 : (x_spe.shape[2]) // 2 + 1, :
            ]
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = tf.keras.layers.Dropout(0.9)(x_spe)
            inp.append(inp_spe)
            y_spe = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_spe)

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                name="eeg",
            )
            x_eeg_raw = tf.keras.layers.Reshape(
                (inp_eeg.shape[1], inp_eeg.shape[2], 1)
            )(inp_eeg)

            strides = 10
            if PLATFORM == "local":
                eeg_embed = tf.keras.layers.Conv1D(
                    filters=strides * 3,
                    kernel_size=strides,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                    activation=None,
                    kernel_initializer=None if PLATFORM != "local" else None,
                    input_shape=(None, 1),
                )
            else:
                eeg_embed = tf.keras.layers.Conv1D(
                    filters=strides * 3,
                    kernel_size=strides,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                    activation=None,
                )

            x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)

            x_eeg = tf.keras.layers.Concatenate(axis=-1)(
                [
                    tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                        x_eeg[:, :, :, 0 * strides : 1 * strides]
                    ),
                    tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                        x_eeg[:, :, :, 1 * strides : 2 * strides]
                    ),
                    tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                        x_eeg[:, :, :, 2 * strides : 3 * strides]
                    ),
                ]
            )
            x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
            x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
            x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

            base_model_eeg = tf.keras.applications.EfficientNetV2B0(
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
            base_model_eeg._name = "eeg_extractor"

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = x_eeg[
                :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
            ]
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            y_eeg = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_eeg)

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(
                shape=(STFT_HIGH * EEG_CHANNEL_USED // 2, STFT_WIDE * 2), name="stft"
            )
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

            base_model_stft = tf.keras.applications.EfficientNetV2B0(
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
            base_model_stft._name = "stft_extractor"

            base_model_stft_pre = tf.keras.Model(
                base_model_stft.input, base_model_stft.get_layer("block3b_add").output
            )
            base_model_stft_pre._name = "stft_extractor_pre"
            x_stft1 = base_model_stft_pre(x_stft[:, 0 * 32 : 1 * 32, :, :])
            x_stft2 = base_model_stft_pre(x_stft[:, 1 * 32 : 2 * 32, :, :])
            x_stft3 = base_model_stft_pre(x_stft[:, 2 * 32 : 3 * 32, :, :])
            x_stft4 = base_model_stft_pre(x_stft[:, 3 * 32 : 4 * 32, :, :])
            x_stft5 = base_model_stft_pre(x_stft[:, 4 * 32 : 5 * 32, :, :])
            x_stft6 = base_model_stft_pre(x_stft[:, 5 * 32 : 6 * 32, :, :])
            x_stft7 = base_model_stft_pre(x_stft[:, 6 * 32 : 7 * 32, :, :])
            x_stft8 = base_model_stft_pre(x_stft[:, 7 * 32 : 8 * 32, :, :])
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [x_stft1, x_stft2, x_stft3, x_stft4, x_stft5, x_stft6, x_stft7, x_stft8]
            )
            base_model_stft_after = tf.keras.Model(
                base_model_stft_pre.output, base_model_stft.output
            )
            base_model_stft_after._name = "stft_extractor_after"
            x_stft = base_model_stft_after(x_stft)

            x_stft = x_stft[
                :, :, (x_stft.shape[2] - 1) // 2 : (x_stft.shape[2]) // 2 + 1, :
            ]
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = tf.keras.layers.Dropout(0.2)(x_stft)
            inp.append(inp_stft)
            y_stft = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_stft)

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")
            base_model_img = tf.keras.applications.EfficientNetB0(
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
            base_model_img._name = "img_extractor"
            x_img = base_model_img.output
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            inp.append(inp_img)
            y_img = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_img)

        y = y_eeg * 1
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 8
if NEEDTRAIN and TF_AVAILABLE:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K, gc
    import itertools

    gkf = GroupKFold(n_splits=SPLITS)

    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.expert_consensus, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
        df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

        df_train_stage2 = df_train_stage1[
            np.sum(df_train_stage1[[c + "_raw" for c in TARGETS]].values, 1) >= 10
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[[c + "_raw" for c in TARGETS]].values, 1) >= 10
        ].reset_index(drop=True)

        train_gen = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
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

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.01, 5)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{1}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )
        model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        del model, history, train_gen, valid_gen
        K.clear_session()
        gc.collect()

        train_gen = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
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

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    round(EPOCHS / 3), LEARN_RATE * 0.1, LEARN_RATE * 0.1 * 0.1, 0
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{2}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=round(EPOCHS / 3),
            callbacks=callbacks_list,
        )
        model.load_weights(os.path.join("models", f"fold{i}_stage2.h5"))

        del model, history, train_gen, valid_gen
        K.clear_session()
        gc.collect()



## === cell 9
if not NEEDTRAIN:
    import glob

    def _kl_safe_probs(arr, n_classes: int):
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=1.0 / n_classes, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 1e-6, 1.0)
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr.astype(np.float32)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    if not TF_AVAILABLE:
        print(
            f"WARNING: TensorFlow failed to import ({TF_IMPORT_ERROR}). Writing uniform submission."
        )
        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        uniform = np.full(
            (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
        sub[TARGETS] = uniform
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
    else:
        stage2_files = sorted(
            glob.glob(os.path.join(LOAD_MODELS_FROM, "fold*_stage2.h5"))
        )
        if len(stage2_files) == 0:
            print(
                f"WARNING: No model weights found in {LOAD_MODELS_FROM}. Writing uniform submission."
            )
            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            uniform = np.full(
                (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
            )
            sub[TARGETS] = uniform
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
        else:
            fold_ids = []
            for p in stage2_files:
                base = os.path.basename(p)
                try:
                    fid = int(base.split("_")[0].replace("fold", ""))
                    fold_ids.append(fid)
                except Exception:
                    pass
            fold_ids = sorted(set(fold_ids))
            if len(fold_ids) == 0:
                raise RuntimeError(f"Could not parse fold ids from: {stage2_files[:3]}")

            print(f"Found folds: {fold_ids}")

            with strategy.scope():
                model_template = build_model()
                models = []
                for fid in fold_ids:
                    wpath = os.path.join(LOAD_MODELS_FROM, f"fold{fid}_stage2.h5")
                    print(f"Load fold {fid}")
                    model = clone_model(model_template)
                    model.load_weights(wpath)
                    model.compile(
                        loss=tf.keras.losses.KLDivergence(),
                        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE),
                    )
                    models.append(model)

            if "spe" in DATATYPE:
                PATH_test_spe = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
                files_test = os.listdir(PATH_test_spe)
                print(f"There are {len(files_test)} test spectrogram parquets")
                for i, f in enumerate(files_test):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    tmp = pd.read_parquet(f"{PATH_test_spe}{f}")
                    name = int(f.split(".")[0])
                    spectrograms_test[name] = tmp.iloc[:, 1:].values
                print()

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            preds_all = []

            if (
                ("spe" in DATATYPE)
                or ("eeg" in DATATYPE)
                or ("stft" in DATATYPE)
                or ("img" in DATATYPE)
            ):
                if filter_range is not None:
                    b, a = signal.butter(
                        3, np.float32(filter_range) * 2 / RSFREQ, "bandpass"
                    )
                b2, a2 = signal.butter(
                    3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
                )

                for i, eeg_id in enumerate(test.eeg_id):
                    if i % 100 == 0:
                        print(i, ", ", end="")

                    eeg_default = pd.read_parquet(
                        os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                    )

                    eeg = []
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
                        nperseg = round(RSFREQ * 0.2)
                        ff, tt, ss = signal.spectrogram(
                            eeg2,
                            axis=1,
                            fs=RSFREQ,
                            nperseg=nperseg,
                            noverlap=round(nperseg - RSFREQ * STFT_TIME),
                            nfft=320,
                        )
                        ss[np.isnan(ss)] = 0
                        ss = ss[:, (ff > 0) * (ff <= 20), :]

                    if "img" in DATATYPE:
                        eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                        eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

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
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]
                            img_save[ii, :, :] = img

                        imgs_test[eeg_id] = img_save

                    if filter_range is not None:
                        eeg = signal.filtfilt(b, a, eeg, axis=1)

                    eeg = np.array(eeg, dtype=np.float32)

                    if "eeg" in DATATYPE:
                        eegs_test[eeg_id] = eeg
                    if "stft" in DATATYPE:
                        stfts_test[eeg_id] = ss
                        stfts_test[-eeg_id] = tt

                    if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                        batch_end = i + 1
                        batch_start = max(0, batch_end - TEST_BATCHSIZE)

                        test_gen = DataGenerator(
                            test.iloc[batch_start:batch_end].reset_index(drop=True),
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
                        for model_i in range(len(models)):
                            pred_i = models[model_i].predict(test_gen, verbose=0)
                            pred_i = _kl_safe_probs(pred_i, len(TARGETS))
                            preds.append(pred_i)
                        pred = np.mean(preds, axis=0)
                        pred = _kl_safe_probs(pred, len(TARGETS))

                        eegs_test = {}
                        stfts_test = {}
                        imgs_test = {}
                        gc.collect()

                        if len(preds_all) == 0:
                            preds_all = pred.copy()
                        else:
                            preds_all = np.concatenate((preds_all, pred), axis=0)

                print()

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})

            preds_all = np.asarray(preds_all)
            if preds_all.shape[0] != len(test):
                raise RuntimeError(
                    f"preds_all length mismatch: got {preds_all.shape[0]} rows, expected {len(test)}"
                )

            sub[TARGETS] = _kl_safe_probs(preds_all, len(TARGETS))
            sub = sub[["eeg_id"] + list(TARGETS)]
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
