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

0.2939802417846457

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash caused by an incompatible `protobuf` version (TensorFlow sometimes breaks with `MessageFactory.GetPrototype` in this Kaggle/Python setup) by forcing the pure-Python protobuf implementation before importing TensorFlow. I also fix the notebook-to-script cell numbering (starting at 1) and make the inference path robust on Kaggle by defaulting to uniform probabilities if model weights are not found, ensuring a valid `submission.csv` is always written. Finally, I ensure the submission columns match `sample_submission.csv` and each row sums to 1 (with a tiny renormalization safeguard), which is score-safe and prevents submission rejection.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* ensuring the related environment variables are set before any protobuf/TensorFlow import, plus falling back to `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` only when needed. I also make sure inference actually uses trained weights if they exist (instead of silently producing near-uniform predictions), by robustly locating a valid `models*` directory and by building/cloning the model under the same distribution scope used for TF determinism. Finally, I keep the submission formatting safeguards (clip + renormalize) so every row sums to 1, and ensure a `submission.csv` is always written end-to-end in Kaggle.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by switching the protobuf implementation to the Python backend *before* TensorFlow is imported, and by making the import path robust if Kaggle’s bundled protobuf still triggers `MessageFactory.GetPrototype`. Then I ensure `NEEDTRAIN` correctly stays `False` on Kaggle (so the notebook runs fast and uses provided weights), and improve inference scoring by (1) loading weights from the best available path and (2) adding a tiny probability floor + renormalization (prevents KL blow-ups from exact zeros without changing core model logic). Finally, I guarantee the submission matches `sample_submission.csv` column order and row count and always writes `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation environment variables *before any* TensorFlow-related import and by adding a safe fallback that restarts the import path if the known `MessageFactory.GetPrototype` error occurs. Then I make sure inference actually finds and loads the provided fold weights on Kaggle by robustly locating a `models*` directory under `/kaggle/input/` (including nested competition folders), because falling back to uniform predictions is the main reason the score is currently very poor. Finally, I keep your model/training logic unchanged, but add a small KL-safe probability floor + renormalization (already present) and ensure submission column order/row alignment exactly matches `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend *before any TensorFlow/protobuf-related import* and by adding a robust fallback that retries the import in a clean way if the known `MessageFactory.GetPrototype` error occurs. Then, to move the score down toward the target (lower is better), I make sure inference actually finds and loads the fold weight files from Kaggle inputs (your current 1.40995 strongly suggests it is still often falling back to near-uniform predictions due to not finding weights). Finally, I keep your model/data logic unchanged but add two score-safe correctness guards: ensure `TARGETS` comes from `sample_submission.csv` (correct column order) and enforce a tiny probability floor + renormalization to avoid KL divergence blow-ups from zeros.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend earlier and more robustly (including setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` unconditionally before any TF import), because the current error shows the C++ protobuf path is still being used. Then I correct the Kaggle inference path so it actually finds and loads the fold weight files by searching inside the competition dataset directory (not just `/kaggle/input/*/models*`), which is the most likely reason the score is stuck at ~1.41 (near-uniform fallback). Finally, I keep your model and preprocessing logic unchanged, but make submission generation safer by ensuring prediction rows align to `test.csv` order and by applying a tiny probability floor + renormalization (score-stable, prevents KL blowups/rejection).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf backend *before any TensorFlow import* and by avoiding the brittle retry logic that still ends up hitting the failing C++ protobuf path in this environment. Then I make the Kaggle inference reliably find the provided fold weight files by searching common nested input locations (the current 1.40995 strongly indicates it is still often falling back to uniform predictions due to missing weights). Finally, I keep your model and preprocessing logic unchanged, but add a tiny KL-safe probability floor + renormalization (already present) and ensure submission row/column alignment matches `sample_submission.csv` exactly.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf backend earlier and more robustly, and by applying a safe monkey-patch for the missing `MessageFactory.GetPrototype` method before importing TensorFlow. Then I make model-weight discovery on Kaggle reliably locate the provided fold weight files (the current score strongly suggests it’s still often falling back to uniform predictions due to not finding weights). Finally, I keep your model and inference logic unchanged, but ensure submission alignment/column order matches `sample_submission.csv` and keep the tiny probability floor + renormalization to avoid KL blow-ups from zeros.'
- What this solution (achieved 1.40995) has done: 'I fix the crash happening before TensorFlow imports by removing the brittle `MessageFactory.GetPrototype` monkey-patch and instead forcing the pure-Python protobuf backend early (which avoids the C++ protobuf path that triggers this error in this environment). Then I make the Kaggle weight-directory discovery a bit more robust so it’s much less likely to fall back to uniform predictions (which is the main reason your score is stuck very high at ~1.41). Finally, I keep your model/training/inference logic unchanged, but add a safe probability floor + renormalization (already present) and ensure submission columns/order exactly match `sample_submission.csv` so the file is always valid.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash by forcing the pure-Python protobuf backend before any TensorFlow/protobuf import and by adding a small pre-import patch for the specific `MessageFactory.GetPrototype` AttributeError seen in this Kaggle/Python setup. Then I fix the main score issue (1.40995 is consistent with uniform-fallback predictions) by making model-weight discovery more robust, specifically searching for fold `.h5` files under common `/kaggle/input/**` nested locations and using the best match. Finally, I keep your model/inference logic unchanged but ensure the submission always uses the exact `sample_submission.csv` column order and applies a tiny probability floor + renormalization to avoid KL blow-ups from zeros.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.29398), and the most likely cause is that inference is still not loading the intended fold weights and/or is running the model in training mode (Dropout active), both of which can severely hurt KL. I make minimal changes to (1) deterministically locate weights by also accepting `.keras`/SavedModel-style files and by preferring a directory that contains the most complete set, and (2) force inference mode by calling the model with `training=False` (so Dropout is disabled) while keeping the same architecture and preprocessing. I also make the submission safer for KL by using a slightly larger probability floor (still tiny) before renormalization to avoid extreme penalties from near-zero probabilities, without changing the core modeling logic. The script still run end-to-end and always write a valid `submission.csv` with the correct columns and rows summing to 1.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

cwd_parts = os.getcwd().split(os.sep)
if len(cwd_parts) > 1 and cwd_parts[1] == "home":
    PLATFORM = "local"  # local kaggle
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif len(cwd_parts) > 1 and cwd_parts[1] == "kaggle":
    PLATFORM = "kaggle"  # kaggle
    NEEDTRAIN = False
    if os.path.isdir("/kaggle/input/"):
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "kaggle"
    NEEDTRAIN = False

DATATYPE = ["eeg"]  # spe, eeg, stft, img
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100
SPE_WIDE = 256

STFT_LENGTH = 45
STFT_TIME = 0.15
STFT_HIGH = 32
STFT_WIDE = round(STFT_LENGTH / STFT_TIME)

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024

BATCHSIZE = 16

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False
READ_SPE_FILES = False

spectrograms = {}
eegs = {}
stfts = {}
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype  # type: ignore[attr-defined]
except Exception:
    pass

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf

print(tf.version.VERSION)
print(tf.config.list_physical_devices("GPU"))
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

try:
    from tensorflow.python.framework.ops import reset_default_graph
except Exception:

    def reset_default_graph():
        return


import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

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

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))

sample_sub_for_targets = pd.read_csv(
    os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
)
TARGETS = sample_sub_for_targets.columns[1:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

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
            df["sign_id"] = df.index.values
            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## === cell 2
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

        if "eeg" in DATATYPE:
            if os.path.exists(os.path.join(datapath, "eegs.npy")):
                eegs = np.load(
                    os.path.join(datapath, "eegs.npy"), allow_pickle=True
                ).item()
        if "stft" in DATATYPE:
            if os.path.exists(os.path.join(datapath, "stfts.npy")):
                stfts = np.load(
                    os.path.join(datapath, "stfts.npy"), allow_pickle=True
                ).item()
        if "img" in DATATYPE:
            if os.path.exists(os.path.join(datapath, "imgs.npy")):
                imgs = np.load(
                    os.path.join(datapath, "imgs.npy"), allow_pickle=True
                ).item()



## === cell 3
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




## === cell 4
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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
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
            sign_id = row.sign_id
            if self.mode != "test":
                sample_weight = sum(row[TARGETS_RAW].values) / 20

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
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

            if "img" in DATATYPE:
                img = self.imgs[sign_id]

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
                eeg_save = np.zeros((x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32)

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
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "img" in DATATYPE:
            x["img"] = x_img

        return x, y, sample_weights




## === cell 5
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
                    (step - self.warm_step) / (self.total_step - self.warm_step) * np.pi
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
        attn_output, weights = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), weights


class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

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




## === cell 6
def build_model():
    inp = list()

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
        )
        x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        x_eeg = tf.keras.layers.Conv2D(
            filters=8, kernel_size=(1, 4), strides=(1, 2), padding="same"
        )(x_eeg_raw)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        x_eeg = tf.keras.layers.Conv2D(
            filters=10, kernel_size=(1, 4), strides=(1, 2), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        x_eeg = tf.keras.layers.Conv2D(
            filters=12,
            kernel_size=(1, 6),
            strides=(1, 3),
            padding="same",
            activation="sigmoid",
        )(x_eeg)
        x_eeg = x_eeg * 255

        x_eeg = tf.keras.layers.Permute([2, 1, 3])(x_eeg)
        x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], -1))(x_eeg)
        x_eeg = tf.keras.layers.Permute([2, 1])(x_eeg)
        x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], 1))(x_eeg)
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

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
        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)
        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

    y = y_eeg * 1
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 7
if NEEDTRAIN:
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
            np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
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

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="loss")
        plt.plot(epochs, val_loss, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage1.svg"))
        plt.close()

        valid_stage1 = df_valid_stage1[TARGETS].values
        predict_stage1 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage1, 1), np.argmax(predict_stage1, 1))
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
        plt.savefig(os.path.join("models", f"fold{i}_stage1_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
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

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="loss")
        plt.plot(epochs, val_loss, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage2.svg"))
        plt.close()

        valid_stage2 = df_valid_stage2[TARGETS].values
        predict_stage2 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage2, 1), np.argmax(predict_stage2, 1))
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
        plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
        gc.collect()



## === cell 8
if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    def _find_models_dir_kaggle():
        candidates = []

        if LOAD_MODELS_FROM and os.path.isdir(LOAD_MODELS_FROM):
            candidates.append(LOAD_MODELS_FROM)

        bases = [
            "/kaggle/input",
            "/kaggle/input/hms-harmful-brain-activity-classification",
            "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
            "/kaggle/working",
        ]
        for base in bases:
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                rel = os.path.relpath(root, base)
                if rel.count(os.sep) >= 12:
                    dirs[:] = []
                if any(
                    (
                        f.startswith("fold")
                        and (f.endswith(".h5") or f.endswith(".keras"))
                    )
                    for f in files
                ):
                    candidates.append(root)

        def _has(d, fold, stage):
            for ext in (".h5", ".keras"):
                if os.path.isfile(os.path.join(d, f"fold{fold}_stage{stage}{ext}")):
                    return True
            if os.path.isdir(os.path.join(d, f"fold{fold}_stage{stage}")):
                return True
            return False

        def score_dir(d):
            s2 = sum(_has(d, k, 2) for k in range(SPLITS))
            s1 = sum(_has(d, k, 1) for k in range(SPLITS))
            return (s2, s1, -len(d))  # prefer more complete + shorter paths

        best = None
        best_score = (-1, -1, -(10**9))
        for d in candidates:
            sc = score_dir(d)
            if sc > best_score:
                best_score = sc
                best = d
        return best if best is not None else LOAD_MODELS_FROM

    if PLATFORM == "kaggle":
        LOAD_MODELS_FROM = _find_models_dir_kaggle()

    print("LOAD_MODELS_FROM =", LOAD_MODELS_FROM)

    def _resolve_weight_path(models_dir, fold, stage):
        for ext in (".h5", ".keras"):
            p = os.path.join(models_dir, f"fold{fold}_stage{stage}{ext}")
            if os.path.exists(p):
                return p
        p_dir = os.path.join(models_dir, f"fold{fold}_stage{stage}")
        if os.path.isdir(p_dir):
            return p_dir
        return os.path.join(models_dir, f"fold{fold}_stage{stage}.h5")

    stage = 2
    if LOAD_MODELS_FROM and os.path.isdir(LOAD_MODELS_FROM):
        have_stage2 = all(
            os.path.exists(_resolve_weight_path(LOAD_MODELS_FROM, model_i, 2))
            for model_i in range(SPLITS)
        )
        if not have_stage2:
            have_stage1 = all(
                os.path.exists(_resolve_weight_path(LOAD_MODELS_FROM, model_i, 1))
                for model_i in range(SPLITS)
            )
            if have_stage1:
                stage = 1

    weight_ok = True
    if not (LOAD_MODELS_FROM and os.path.isdir(LOAD_MODELS_FROM)):
        weight_ok = False

    for model_i in range(SPLITS):
        wpath = _resolve_weight_path(LOAD_MODELS_FROM, model_i, stage)
        if not os.path.exists(wpath):
            print(f"WARNING: missing weights: {wpath}")
            weight_ok = False

    if not weight_ok:
        preds_all = np.full(
            (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
    else:
        preds_all = []
        models = list()

        with strategy.scope():
            model_template = build_model()

        for model_i in range(SPLITS):
            print(f"Fold {model_i + 1}")
            with strategy.scope():
                model = clone_model(model_template)
            model.load_weights(_resolve_weight_path(LOAD_MODELS_FROM, model_i, stage))
            models.append(model)

        if (
            ("spe" in DATATYPE)
            or ("eeg" in DATATYPE)
            or ("stft" in DATATYPE)
            or ("img" in DATATYPE)
        ):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
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

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eegshape = eeg.shape[1]
                eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
                if filter_range is not None:
                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = eeg[:, eegshape : eegshape * 2]
                eeg = np.array(eeg, dtype=np.float32)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    preds = []
                    start = max(0, i - TEST_BATCHSIZE + 1)
                    batch_df = test.iloc[start : i + 1].reset_index(drop=True)
                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )

                    for mi in range(SPLITS):
                        pred = []
                        for bidx in range(len(test_gen)):
                            xb, _, _ = test_gen[bidx]
                            yb = models[mi](xb, training=False).numpy()
                            pred.append(yb)
                        pred = np.concatenate(pred, axis=0)
                        pred = pred[: len(batch_df)]
                        preds.append(pred)

                    pred = np.mean(preds, axis=0).astype(np.float32)

                    eegs_test = {}
                    gc.collect()

                    if len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

        preds_all = np.asarray(preds_all, dtype=np.float32)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = preds_all

    eps = 1e-4
    sub[TARGETS] = sub[TARGETS].clip(lower=eps)
    row_sum = sub[TARGETS].sum(axis=1).values.reshape(-1, 1)
    row_sum[row_sum == 0] = 1.0
    sub[TARGETS] = sub[TARGETS].values / row_sum

    sub = sub[sample_sub.columns]
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
