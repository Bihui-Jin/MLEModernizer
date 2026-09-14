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

0.2806487863933449

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the crash caused by protobuf/TensorFlow incompatibility by forcing the pure-Python protobuf implementation before importing TensorFlow (instead of popping those env vars). I also fix the “Invalid submission: Submission and answers must have the same length” issue by ensuring we always build predictions in the exact `sample_submission.csv` row order and by writing the submission from `sample_submission` directly (not a potentially mismatched copy). Finally, I make inference robust to missing/empty model folders by keeping the existing uniform fallback, while guaranteeing the output file is `submission.csv` with correct columns and row count.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash that happens at import time by switching to the C++ protobuf implementation (the current forced pure-Python protobuf is what triggers the `MessageFactory.GetPrototype` error with this TF build). I keep the rest of the pipeline unchanged, but make the environment detection/training flags more robust so Kaggle always skips training and goes straight to inference. To move the score down toward the target (lower is better) with minimal semantic change, I also add a very small, metric-safe prior-mixing calibration at inference time (blend model probs with a data-driven class prior from train), then renormalize to ensure row sums equal 1. The submission writing remains in `sample_submission.csv` order and always produces `submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by switching away from the forced pure-Python protobuf runtime (which triggers the `MessageFactory.GetPrototype` error in this Kaggle TF build) and keeping the environment variables consistent before importing TensorFlow. I also fix the “Invalid submission: Submission and answers must have the same length” issue by ensuring predictions are generated strictly in `sample_submission.csv` row order and by making the batch concatenation logic shape-safe even if a last batch has a different size. Finally, I keep the core model/inference logic unchanged, but add a tiny amount of defensive post-processing (clipping + renormalization) and always write `submission.csv` with the exact required columns and row count.'

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

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"

cwd_parts = os.getcwd().split(os.sep)
if len(cwd_parts) > 1 and cwd_parts[1] == "home":
    PLATFORM = "local"
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "kaggle" if os.path.isdir("/kaggle") else "local"
    if PLATFORM == "kaggle":
        NEEDTRAIN = False
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg", "stft"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
    if not os.path.exists(os.path.join(LOAD_DATA_FROM, "train.csv")):
        alt = "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification"
        if os.path.exists(os.path.join(alt, "train.csv")):
            LOAD_DATA_FROM = alt

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
LEARN_RATE = 1e-3
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

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import zoom
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

print(tf.version.VERSION)
print(tf.config.list_physical_devices("GPU"))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enable skipped:", e)

MIX = True
if MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision")


def load_weights_if_exists(model, path: str):
    if path is None:
        return False
    if os.path.exists(path):
        model.load_weights(path)
        return True
    print(f"[WARN] Pretrained weights not found, skipping: {path}")
    return False


sample_sub_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
sample_sub_tmp = pd.read_csv(sample_sub_path, nrows=5)
TARGETS = [c for c in sample_sub_tmp.columns if c != "eeg_id"]
print("Targets", list(TARGETS))

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
print("Train shape:", df.shape)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2193850705.py in <cell line: 0>()
    126 from sklearn.metrics import confusion_matrix
    127 
--> 128 import tensorflow as tf
    129 from tensorflow.keras import optimizers
    130 from tensorflow.keras.models import clone_model

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
def compute_stft_tf(
    eeg_2d: np.ndarray, rsfreq: int, n_fft=500, win_length=128, hop_length=50
):
    """
    eeg_2d: float32 array [channels, time]
    returns:
      ss: float32 array [channels, freq_bins_cropped, time_frames]
      tt: float32 array [time_frames]
    """
    x = tf.convert_to_tensor(eeg_2d, dtype=tf.float32)  # [C, T]
    frames = tf.signal.stft(
        x,
        frame_length=win_length,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # [C, frames, freq]
    mag = tf.abs(frames)  # [C, frames, freq]
    mag = tf.transpose(mag, perm=[0, 2, 1])  # [C, freq, frames]

    f_keep = int(round(20 / (rsfreq / n_fft)))
    mag = mag[:, :f_keep, :]

    n_frames = tf.shape(mag)[2]
    tt = tf.cast(tf.range(n_frames), tf.float32) * (hop_length / float(rsfreq))

    return mag.numpy().astype(np.float32), tt.numpy().astype(np.float32)


TARGETS_RAW = [c + "_raw" for c in TARGETS]

if (not READ_EEG_FILES) and (not os.path.exists("train.csv")):
    print(
        'Working "train.csv" not found; building it from original train.csv metadata.'
    )
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
    y_prob = y_data / y_data.sum(axis=1, keepdims=True)
    train[TARGETS] = y_prob

    train.to_csv("train.csv", index=False)
else:
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

if "sign_id" not in df.columns:
    df["sign_id"] = np.arange(len(df), dtype=np.int64)

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4210308910.py in <cell line: 0>()
     29 
     30 
---> 31 TARGETS_RAW = [c + "_raw" for c in TARGETS]
     32 
     33 if (not READ_EEG_FILES) and (not os.path.exists("train.csv")):

NameError: name 'TARGETS' is not defined

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
                ss, tt = compute_stft_tf(
                    eeg2.astype(np.float32),
                    RSFREQ,
                    n_fft=500,
                    win_length=128,
                    hop_length=50,
                )

                ss = np.concatenate(
                    (
                        ss[0 : round(EEG_CHANNEL_USED / 2), :, :],
                        ss[-round(EEG_CHANNEL_USED / 2) :, :, :],
                    ),
                    axis=0,
                )

                ss = np.array(ss, dtype=np.float32)
                tt = np.array(tt, dtype=np.float32)

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
    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        eeg_path = os.path.join(datapath, "eegs.npy")
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(
                f"Preprocessed eegs.npy not found at {eeg_path}. "
                f"Set READ_EEG_FILES=True to build preprocessing (slow) or provide preprocess in input."
            )
        eegs = np.load(eeg_path, allow_pickle=True).item()
        if "stft" in DATATYPE:
            stft_path = os.path.join(datapath, "stfts.npy")
            if not os.path.exists(stft_path):
                raise FileNotFoundError(
                    f"Preprocessed stfts.npy not found at {stft_path}. "
                    f"Set READ_EEG_FILES=True to build preprocessing (slow) or provide preprocess in input."
                )
            stfts = np.load(stft_path, allow_pickle=True).item()



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
        stage=2,
    ):

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs if eegs is not None else {}
        self.stfts = stfts if stfts is not None else {}
        self.specs = specs if specs is not None else {}
        self.imgs = imgs if imgs is not None else {}
        self.stage = stage
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

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

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = getattr(row, "sign_id", i)

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
                    * (df.grda_vote == row.grda_vote_raw)
                    * (df.other_vote == row.other_vote_raw),
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
                    if row.eeg_id in self.eegs:
                        r_eeg = r_eeg + np.random.random() * 10 - 5
                        r_eeg = max(0, r_eeg)
                        r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)
                    else:
                        r_eeg = max(0, min(r_eeg, 0))

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
                if row.eeg_id in self.eegs:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]
                else:
                    eeg = np.zeros((len(BRAIN), EEG_LENGTH * RSFREQ), dtype=np.float32)

                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

            if "stft" in DATATYPE:
                if (-row.eeg_id in self.stfts) and (row.eeg_id in self.stfts):
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    r_stft2 = (np.where(stft_t <= (50 + r_eeg - min(stft_t))))[0][-1]
                    stft = self.stfts[row.eeg_id][:, :, r_stft:r_stft2]
                else:
                    stft = np.zeros(
                        (EEG_CHANNEL_USED, STFT_HIGH, STFT_WIDE), dtype=np.float32
                    )

                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = self.imgs.get(
                    sign_id,
                    np.zeros((EEG_CHANNEL_USED, 36, IMG_WIDE), dtype=np.float32),
                )

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
                    if self.stage == 2:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                    else:
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
                    if self.stage == 2:
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
                    else:
                        stft[0 : round(stft.shape[0] / 2), :, :] = stft[
                            0 : round(stft.shape[0] / 2), :, :
                        ][np.random.permutation(stft.shape[0] // 2), :, :]
                        stft[-round(stft.shape[0] / 2) :, :, :] = stft[
                            -round(stft.shape[0] / 2) :, :
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
                        if np.random.rand() > 0.5:
                            stft = stft[:, ::-1, :]
                        if np.random.rand() > 0.5:
                            stft = stft[:, :, ::-1]
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

                if j == 0:
                    x_stft = np.zeros(
                        (len(indexes), stft.shape[0], stft.shape[1], stft.shape[2]),
                        dtype="float32",
                    )
                x_stft[j] = stft

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                if self.sample_weights:
                    sample_weights[j] = sum(row[TARGETS_RAW].values) / 20
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

        return x, {"output": y, "output_eeg": y, "output_stft": y}, sample_weights




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1640957549.py in <cell line: 0>()
----> 1 class DataGenerator(tf.keras.utils.Sequence):
      2     def __init__(
      3         self,
      4         dataframe,
      5         batch_size=32,

NameError: name 'tf' is not defined

## === cell 5
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step
        self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
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


class IniToOne(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape
        kernel = np.zeros(shape, dtype=np.float32)
        for i in range(filter_count):
            kernel[i % filter_length, 0, i] = 1.0
        return tf.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self):
        return {}


class SumToOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class IniToOneAtten(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape
        kernel = np.zeros(shape, dtype=np.float32)
        kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
            (filter_length) // 2 + 1 - (filter_length - 1) // 2
        )
        return tf.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self):
        return {}


class SumToOneAtten(tf.keras.constraints.Constraint):
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




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3073212647.py in <cell line: 0>()
----> 1 class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
      2     def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
      3         super(CosineAnnealingLRScheduler, self).__init__()
      4         self.total_step = total_step
      5         self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)

NameError: name 'optimizers' is not defined

## === cell 6
def build_model():
    inp = []
    y = 0

    y_eeg = None
    y_stft = None

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
        x_spe = tf.keras.layers.Reshape(
            (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
        )(inp_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [
                x_spe[:, 0, :, :, :],
                x_spe[:, 1, :, :, :],
                x_spe[:, 2, :, :, :],
                x_spe[:, 3, :, :, :],
            ]
        )

        base_model_spe = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                load_weights_if_exists(
                    base_model_spe,
                    f"./input/pre-trained-weights/{base_model_spe.name}_notop.h5",
                )
            if PLATFORM == "kaggle":
                load_weights_if_exists(
                    base_model_spe,
                    f"/kaggle/input/pre-trained-weights/{base_model_spe.name}_notop.h5",
                )
        base_model_spe.name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.5)(x_spe)

        inp.append(inp_spe)
        y = x_spe * 1

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

        strides = 10
        if PLATFORM == "local":
            eeg_embed = tf.keras.layers.Conv1D(
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

        base_model_eeg = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                load_weights_if_exists(
                    base_model_eeg,
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5",
                )
            if PLATFORM == "kaggle":
                load_weights_if_exists(
                    base_model_eeg,
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5",
                )
        base_model_eeg.name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)

        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32", name="output_eeg"
        )(x_eeg)
        x_eeg = tf.keras.layers.Dense(128)(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)

        if y == 0:
            y = x_eeg * 1
        else:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(16, STFT_HIGH, STFT_WIDE), name="stft")
        x_stft = tf.keras.layers.Reshape(
            (inp_stft.shape[1], inp_stft.shape[2], inp_stft.shape[3], 1)
        )(inp_stft)
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        x_stft = tf.keras.layers.Concatenate(axis=1)(
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

        base_model_stft = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                load_weights_if_exists(
                    base_model_stft,
                    f"./input/pre-trained-weights/{base_model_stft.name}_notop.h5",
                )
            if PLATFORM == "kaggle":
                load_weights_if_exists(
                    base_model_stft,
                    f"/kaggle/input/pre-trained-weights/{base_model_stft.name}_notop.h5",
                )
        base_model_stft.name = "stft_extractor"

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = tf.keras.layers.Dropout(0.5)(x_stft)

        inp.append(inp_stft)

        y_stft = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32", name="output_stft"
        )(x_stft)
        x_stft = tf.keras.layers.Dense(128)(x_stft)
        x_stft = tf.keras.layers.LeakyReLU()(x_stft)

        if y == 0:
            y = x_stft * 1
        else:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")
        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=inp_img
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                load_weights_if_exists(
                    base_model_img,
                    f"./input/pre-trained-weights/{base_model_img.name}_notop.h5",
                )
            if PLATFORM == "kaggle":
                load_weights_if_exists(
                    base_model_img,
                    f"/kaggle/input/pre-trained-weights/{base_model_img.name}_notop.h5",
                )
        base_model_img.name = "img_extractor"
        x_img = base_model_img.output
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

        inp.append(inp_img)
        if y == 0:
            y = x_img * 1
        else:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])

    y = tf.keras.layers.Dense(
        len(TARGETS), activation="softmax", dtype="float32", name="output"
    )(y)

    if y_eeg is None:
        y_eeg = y
    if y_stft is None:
        y_stft = y

    model = tf.keras.Model(inputs=inp, outputs=[y, y_eeg, y_stft])
    return model




## === cell 7
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
    print(f"### Fold {i + 1}")

    model = build_model()

    if stage == 1:
        loss_weights = {"output": 1 / 3, "output_eeg": 1 / 3, "output_stft": 1 / 3}
        train_gen_stage = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            stage=stage,
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
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
    else:
        loss_weights = {"output": 1 / 3, "output_eeg": 1 / 3, "output_stft": 1 / 3}
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            stage=stage,
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
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1,
                    LEARN_RATE * 0.1 * 0.1 * 0.1,
                    0,
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

    model.compile(
        loss={
            "output": tf.keras.losses.KLDivergence(),
            "output_eeg": tf.keras.losses.KLDivergence(),
            "output_stft": tf.keras.losses.KLDivergence(),
        },
        loss_weights=loss_weights,
        optimizer=opt,
    )

    if stage == 1:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=EPOCHS,
            callbacks=callbacks_stage,
        )
    else:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=max(round(EPOCHS / 3), 1),
            callbacks=callbacks_stage,
        )

    model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    val_output_loss = history.history["val_output_loss"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="loss")
    plt.plot(epochs, val_loss, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}", fontsize=12
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    if stage == 1:
        valid_stage = df_valid_stage1[TARGETS].values
    else:
        valid_stage = df_valid_stage2[TARGETS].values
    predict_stage = np.array(model.predict(valid_gen_stage, verbose=0))[0]

    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
    gc.collect()

    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(
        "Confusion Matrix "
        + str(round(val_output_loss[np.where(val_loss == min(val_loss))[0][0]], 4))
    )
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    plt.yticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    thresh = cm.max() / 2.0
    import itertools

    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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




## === cell 8
if __name__ == "__main__":
    if PLATFORM == "kaggle":
        NEEDTRAIN = False

    if (not NEEDTRAIN) and (not os.path.isdir(LOAD_MODELS_FROM)):
        print(
            f"LOAD_MODELS_FROM not found: {LOAD_MODELS_FROM}. Will skip training (Kaggle) and use uniform predictions."
        )

    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold

        if PLATFORM == "kaggle":
            SPLITS_RUN = 2
            EPOCHS_RUN = 1
        else:
            SPLITS_RUN = SPLITS
            EPOCHS_RUN = EPOCHS

        gkf = GroupKFold(n_splits=SPLITS_RUN)

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

            for stage in [1, 2]:
                train_fold(
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
                    EPOCHS_RUN,
                    LEARN_RATE,
                    TARGETS,
                    TARGETS_RAW,
                )

        LOAD_MODELS_FROM = "models"
        NEEDTRAIN = False

    models = []
    model_template = build_model()

    if os.path.isdir(LOAD_MODELS_FROM):
        for model_i in range(100):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(wpath):
                print(f"Fold {model_i + 1}")
                model = clone_model(model_template)
                model.load_weights(wpath)
                models.append(model)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    print("Test shape", test.shape)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sub = sample_sub.copy()
    sub_eeg_ids = sub["eeg_id"].astype(np.int64).values

    prior_counts = df[TARGETS].to_numpy(dtype=np.float64).sum(axis=0)
    prior = prior_counts / np.maximum(prior_counts.sum(), 1.0)
    prior = prior.astype(np.float32)
    PRIOR_MIX_ALPHA = 0.20

    if len(models) == 0:
        print(
            f"[WARN] No model weights found under {LOAD_MODELS_FROM}. Writing prior-mixed uniform predictions."
        )
        preds_all = np.full(
            (len(sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
        preds_all = (1.0 - PRIOR_MIX_ALPHA) * preds_all + PRIOR_MIX_ALPHA * prior[
            None, :
        ]
        preds_all = np.clip(preds_all, 1e-8, 1.0)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)
        sub.loc[:, TARGETS] = preds_all
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
    else:
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

        preds_all_batches = []
        batch_start = 0

        test_map = test.set_index("eeg_id")

        for i, eeg_id in enumerate(sub_eeg_ids):
            if i % 100 == 0:
                print(i, ", ", end="")

            eeg_file = os.path.join(PATH_test, f"{int(eeg_id)}.parquet")
            if (eeg_id not in test_map.index) or (not os.path.exists(eeg_file)):
                eeg = np.zeros((len(BRAIN), EEG_LENGTH * RSFREQ), dtype=np.float32)
                eegs_test[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts_test[eeg_id] = np.zeros(
                        (EEG_CHANNEL_USED, STFT_HIGH, STFT_WIDE), dtype=np.float32
                    )
                    stfts_test[-eeg_id] = np.arange(STFT_WIDE, dtype=np.float32)
            else:
                eeg_default = pd.read_parquet(eeg_file)

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
                    ss, tt = compute_stft_tf(
                        eeg2.astype(np.float32),
                        RSFREQ,
                        n_fft=500,
                        win_length=128,
                        hop_length=50,
                    )
                    ss = np.concatenate(
                        (
                            ss[0 : round(EEG_CHANNEL_USED / 2), :, :],
                            ss[-round(EEG_CHANNEL_USED / 2) :, :, :],
                        ),
                        axis=0,
                    )
                    ss = np.array(ss, dtype=np.float32)
                    tt = np.array(tt, dtype=np.float32)

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

            is_batch_end = ((i + 1) % TEST_BATCHSIZE == 0) or (
                (i + 1) == len(sub_eeg_ids)
            )
            if is_batch_end:
                batch_end = i + 1

                batch_df = pd.DataFrame({"eeg_id": sub_eeg_ids[batch_start:batch_end]})
                if "spe" in DATATYPE:
                    batch_df = batch_df.merge(
                        test[["eeg_id", "spectrogram_id"]], on="eeg_id", how="left"
                    )
                else:
                    batch_df["spectrogram_id"] = 0

                test_gen = DataGenerator(
                    batch_df.reset_index(drop=True),
                    shuffle=False,
                    sample_weights=False,
                    batch_size=TEST_BATCHSIZE,
                    mode="test",
                    specs=spectrograms_test,
                    eegs=eegs_test,
                    stfts=stfts_test,
                    imgs=imgs_test,
                    stage=2,
                )

                preds = []
                for m in models:
                    pred = np.array(m.predict(test_gen, verbose=0))[0]
                    preds.append(pred)
                pred = np.mean(preds, axis=0)

                expected = batch_end - batch_start
                if pred.shape[0] != expected:
                    pred = pred[:expected]

                preds_all_batches.append(pred)

                eegs_test = {}
                stfts_test = {}
                imgs_test = {}
                gc.collect()

                batch_start = batch_end

        print()
        preds_all = np.concatenate(preds_all_batches, axis=0)

        if preds_all.shape[0] != len(sub):
            preds_all = preds_all[: len(sub)]
        if preds_all.shape[0] != len(sub):
            raise ValueError(
                f"Prediction rows {preds_all.shape[0]} != submission rows {len(sub)}."
            )

        preds_all = np.asarray(preds_all, dtype=np.float32)

        preds_all = np.clip(preds_all, 1e-8, 1.0)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

        preds_all = (1.0 - PRIOR_MIX_ALPHA) * preds_all + PRIOR_MIX_ALPHA * prior[
            None, :
        ]
        preds_all = np.clip(preds_all, 1e-8, 1.0)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

        sub.loc[:, TARGETS] = preds_all
        sub.loc[:, TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(
            axis=1, keepdims=True
        )
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1760089072.py in <cell line: 0>()
     61 
     62     models = []
---> 63     model_template = build_model()
     64 
     65     if os.path.isdir(LOAD_MODELS_FROM):

/tmp/ipykernel_55/3715759456.py in build_model()
     46 
     47     if "eeg" in DATATYPE:
---> 48         inp_eeg = tf.keras.Input(
     49             shape=(
     50                 EEG_CHANNEL_USED * EEG_MULTIPLY,

NameError: name 'tf' is not defined
