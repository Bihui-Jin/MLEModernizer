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

0.3158216294907964

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash by removing the unused `torchaudio/torch` imports that trigger a protobuf incompatibility in the Kaggle image, and I also remove the unused `reset_default_graph` import. Since Kaggle inference mode is used (`NEEDTRAIN=False`), I make the test-time prediction loop robust: ensure at least one fold model is found (otherwise fall back to uniform probabilities) and fix the slicing bug that can produce empty/incorrect batches. Finally, I enforce valid probability rows (clip + renormalize) before writing `submission.csv` with the exact required columns.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by forcing the pure-Python protobuf implementation early (before importing TensorFlow) and by guarding determinism/mixed-precision settings for compatibility with the Kaggle runtime. Then I enable inference-time weight loading and ensure the script always produces a valid `submission.csv` with the exact required columns and per-row probabilities summing to 1. To improve score toward the target (lower is better) from the current very poor 1.40995, I also set `NEEDTRAIN=False` on Kaggle only when no trained weights directory is present; otherwise, it use the available fold weights for real predictions (uniform fallback remains only as a last resort). These changes keep the model, training, and feature logic intact and only address runtime stability and correct/usable inference.'
- What this solution (achieved 1.40995) has done: 'I fix the import-time protobuf/TensorFlow crash by ensuring the environment variables are set before any TensorFlow/Keras import and by adding a safe fallback that disables TF determinism/mixed precision if the runtime can’t support them. Then I fix inference to reliably find and load fold weights from the Kaggle `/kaggle/input/...` mount, and also make the test batching/prediction accumulation robust so it always produces exactly 9850 rows aligned to `test.csv`. Finally, I enforce strict probability validity (clip + renormalize) and write `submission.csv` with the exact required columns, which should move the score down from the uniform/fallback-like behavior toward the target by ensuring real model predictions are used when weights exist.'
- What this solution (achieved 1.40995) has done: 'I fix the import-time TensorFlow/protobuf crash by switching protobuf to the pure-Python implementation *before* any TensorFlow/Keras import, which is the root cause of the current runtime failure. Then I make `NEEDTRAIN` auto-disable on Kaggle unless a valid fold-weights directory is found, so the notebook reliably produces a real prediction submission when weights exist, otherwise it still produces a valid uniform-probability fallback. Finally, I ensure the script always writes `submission.csv` with the exact required columns and strictly normalized probabilities summing to 1 for every `eeg_id`, preventing submission-format failures.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation even when the Kaggle runtime pre-sets it to “cpp”, and by applying the env var before any TensorFlow import. Then I fix an inference-time bug where fold models are cloned but never compiled, which can cause Keras predict/load to behave inconsistently across versions. Finally, I make weight discovery deterministic and ensure the submission always has exactly the required columns with strictly normalized probabilities (this is score-neutral by itself, but ensures real model predictions are used instead of uniform fallback, improving score from 1.40995 toward the 0.3158 target when weights exist).'
- What this solution (achieved 1.40995) has done: 'We fix the import-time crash (`MessageFactory` / protobuf / TensorFlow incompatibility) by forcing the pure-Python protobuf backend **before** TensorFlow is imported and by adding a safe fallback that disables determinism/mixed-precision if the runtime can’t support them. Then we make Kaggle inference reliably load fold weights by auto-discovering a real weights folder and by correctly building/compiling cloned models before loading weights. Finally, we keep your exact modeling/training semantics but ensure the test prediction loop always generates exactly 9850 probability rows aligned to `test.csv`, with strict clipping+renormalization so each row sums to 1 and the submission is always valid (and should improve score vs. the current near-uniform behavior).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend *and* a compatible protobuf version before TensorFlow loads (the current `MessageFactory.GetPrototype` error typically comes from an incompatible protobuf runtime). Then I make Kaggle inference actually use pretrained fold weights if they exist by auto-discovering a valid `models*/fold*_stage2.weights.h5` directory and enabling `NEEDTRAIN` only when no usable weights are found, which should reduce the KL score substantially from the current near-fallback behavior. Finally, I keep your model/training logic unchanged but make submission writing strict (clip + renormalize) and ensure the prediction loop always yields exactly 9850 rows aligned to `test.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash (`MessageFactory` / protobuf incompatibility) by ensuring TensorFlow is imported only after forcing the pure-Python protobuf backend and by avoiding importing `google.protobuf` directly (which can trigger the problematic C++ implementation). Then I make Kaggle inference reliably load fold weights by (a) searching both the auto-discovered models directory and the default `LOAD_MODELS_FROM`, and (b) using `model_template.get_config()/from_config` instead of `clone_model` to avoid serialization edge cases under mixed precision. Finally, I keep your modeling/training logic unchanged but make the test prediction loop always produce exactly 9850 rows aligned to `test.csv`, with strict clip+renormalize so each row sums to 1 (improves score vs. the current near-uniform/failed-inference behavior).'
- What this solution (achieved 1.40995) has done: 'We fix the immediate runtime crash caused by an incompatible protobuf runtime by forcing TensorFlow to use the pure-Python protobuf implementation *and* monkey-patching the missing `MessageFactory.GetPrototype` API before importing TensorFlow. Then we make Kaggle inference actually use the provided pretrained fold weights by auto-discovering a `models*` directory under `/kaggle/input` and only falling back to uniform predictions if no weights are found (this should materially reduce the KL score from 1.40995 toward the 0.3158 target). Finally, we keep your model/data logic intact but ensure the test-loop always outputs exactly 9850 rows aligned to `test.csv`, with strict clipping+renormalization so every row sums to 1 and the submission format is always valid.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

cwd_parts = os.getcwd().split(os.sep)
root_marker = cwd_parts[1] if len(cwd_parts) > 1 else ""
if root_marker == "home":
    PLATFORM = "local"  # local training or local kaggle
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif root_marker == "kaggle":
    PLATFORM = "kaggle"  # kaggle notebook
    NEEDTRAIN = False
else:
    PLATFORM = "kaggle" if os.path.isdir("/kaggle/input") else "local"

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

import warnings

warnings.filterwarnings("ignore")

try:
    from google.protobuf import message_factory as _message_factory

    if (
        hasattr(_message_factory, "MessageFactory")
        and not hasattr(_message_factory.MessageFactory, "GetPrototype")
        and hasattr(_message_factory.MessageFactory, "GetMessageClass")
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype  # type: ignore[attr-defined]
except Exception as _e:
    pass

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import optimizers

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import zoom
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

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
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", repr(e))

MIX = True
if MIX:
    try:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    except Exception as e:
        print("Mixed precision not enabled:", repr(e))
else:
    print("Using full precision")

if NEEDTRAIN:
    import itertools

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

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
        train = pd.read_csv("train.csv")

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
                raise RuntimeError(
                    "Training with 'stft' requires torch/torchaudio; keep DATATYPE=['eeg'] in this environment."
                )

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

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
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
            sign_id = row.sign_id
            if self.mode != "test":
                sample_weight = sum(row[TARGETS_RAW].values) / 20
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
                spe = list()  # LL RL LP RP
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




## === cell 5
def build_model():
    inp = list()
    y = 0
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
                base_model_spe.load_weights(
                    f"./input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_spe.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_spe.name}_notop.h5"
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

        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)

        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

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
        base_model_stft.name = "stft_extractor"

        x_stft = base_model_stft(x_stft)

        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

        x_stft = tf.keras.layers.Dropout(0.5)(x_stft)

        inp.append(inp_stft)

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
                base_model_img.load_weights(
                    f"./input/pre-trained-weights/{base_model_img.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_img.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_img.name}_notop.h5"
                )
        base_model_img.name = "img_extractor"
        x_img = base_model_img.output

        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

        inp.append(inp_img)

        if y == 0:
            y = x_img * 1
        else:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])

    y = y_eeg * 1
    model = tf.keras.Model(inputs=inp, outputs=y)
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
    print(f"### Fold {i + 1}")

    model = build_model()
    loss = tf.keras.losses.KLDivergence()

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
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1,
                    LEARN_RATE * 0.1 * 0.1,
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

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
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
    elif stage == 2:
        valid_stage = df_valid_stage2[TARGETS].values
    predict_stage = model.predict(valid_gen_stage)

    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
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

    import itertools

    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
                        TARGETS_RAW,
                    ),
                )
                p.start()
                p.join()

    else:
        def _discover_models_roots():
            roots = []
            if os.path.isdir("/kaggle/input"):
                top = "/kaggle/input"
                for d in os.listdir(top):
                    p = os.path.join(top, d)
                    if os.path.isdir(p) and d.startswith("models"):
                        roots.append(p)
                nested = "/kaggle/input/hms-harmful-brain-activity-classification"
                if os.path.isdir(nested):
                    for d in os.listdir(nested):
                        p = os.path.join(nested, d)
                        if os.path.isdir(p) and d.startswith("models"):
                            roots.append(p)

                if len(roots) == 0:
                    for root, dirs, files in os.walk("/kaggle/input"):
                        base = os.path.basename(root)
                        if base.startswith("models"):
                            roots.append(root)
                            if len(roots) >= 5:
                                break

            if LOAD_MODELS_FROM:
                roots.append(LOAD_MODELS_FROM)
            seen = set()
            out = []
            for r in roots:
                if r and r not in seen:
                    seen.add(r)
                    out.append(r)
            return out

        model_roots = _discover_models_roots()
        print("Candidate model roots:")
        for r in model_roots[:50]:
            print(" -", r)

        preds_all = []
        models = list()

        model_template = build_model()
        infer_loss = tf.keras.losses.KLDivergence()
        infer_opt = tf.keras.optimizers.AdamW(learning_rate=1e-4)

        def _new_model_from_template():
            cfg = model_template.get_config()
            m = tf.keras.Model.from_config(cfg, custom_objects=globals())
            return m

        found_any = False
        for model_i in range(100):
            wpath = None
            for root in model_roots:
                candidate = os.path.join(root, f"fold{model_i}_stage2.weights.h5")
                if os.path.exists(candidate):
                    wpath = candidate
                    break
            if wpath is not None:
                found_any = True
                print(f"Fold {model_i + 1} weights found:", wpath)
                model = _new_model_from_template()
                model.compile(loss=infer_loss, optimizer=infer_opt)
                model.load_weights(wpath)
                models.append(model)

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        def _write_submission(test_df, probs, out_path="submission.csv"):
            probs = np.asarray(probs, dtype=np.float32)
            probs = np.clip(probs, 1e-7, 1.0)
            denom = np.sum(probs, axis=1, keepdims=True)
            denom = np.where(denom <= 0, 1.0, denom)
            probs = probs / denom
            sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
            sub[TARGETS] = probs
            sub.to_csv(out_path, index=False)
            print("Submission shape", sub.shape)
            print(sub.head())

        if len(models) == 0:
            print(
                "WARNING: No model weights found. Writing uniform-probability submission."
            )
            uniform = np.ones((len(test), len(TARGETS)), dtype=np.float32) / len(
                TARGETS
            )
            _write_submission(test, uniform, "submission.csv")
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
                        end = i + 1
                        start = end - (
                            TEST_BATCHSIZE
                            if (end % TEST_BATCHSIZE == 0)
                            else (end % TEST_BATCHSIZE)
                        )
                        start = max(0, start)
                        batch_df = test.iloc[start:end].reset_index(drop=True)

                        test_gen = DataGenerator(
                            batch_df,
                            shuffle=False,
                            sample_weights=False,
                            batch_size=len(batch_df),
                            mode="test",
                            specs=spectrograms_test,
                            eegs=eegs_test,
                            stfts=stfts_test,
                            imgs=imgs_test,
                        )

                        preds = []
                        for m in models:
                            preds.append(m.predict(test_gen, verbose=0))
                        pred = np.mean(preds, axis=0)

                        eegs_test = {}
                        stfts_test = {}
                        imgs_test = {}
                        gc.collect()

                        if len(preds_all) == 0:
                            preds_all = pred.copy()
                        else:
                            preds_all = np.concatenate((preds_all, pred), axis=0)

            preds_all = np.asarray(preds_all, dtype=np.float32)

            if preds_all.shape[0] != len(test):
                print(
                    "WARNING: preds_all rows != test rows. Truncating/padding with uniform to match."
                )
                if preds_all.shape[0] > len(test):
                    preds_all = preds_all[: len(test)]
                else:
                    pad = np.ones(
                        (len(test) - preds_all.shape[0], len(TARGETS)), dtype=np.float32
                    ) / len(TARGETS)
                    preds_all = np.concatenate([preds_all, pad], axis=0)

            _write_submission(test, preds_all, "submission.csv")
