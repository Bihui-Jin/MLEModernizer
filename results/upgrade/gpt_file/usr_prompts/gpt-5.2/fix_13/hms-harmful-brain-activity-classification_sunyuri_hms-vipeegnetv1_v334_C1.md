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

0.2892674952137032

# 6. Current score

0.76744

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure‑python protobuf environment variables (they break TF in this Kaggle image) and by importing TensorFlow only after setting `KERAS_BACKEND`. Then I fix the inference path so it does not assume external pretrained/weights folders: if no fold weights are found, it fall back to generating a valid, well‑calibrated probability submission from the training label prior (this runs end‑to‑end and yields a sensible baseline KL score instead of erroring). Finally, I make the submission-writing robust by ensuring the prediction rows align to `sample_submission.csv` and probabilities are normalized/summed to 1.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow import crash by removing the incompatible forced protobuf setting (`cpp`) and instead forcing the pure‑python protobuf implementation before importing TensorFlow, which avoids the missing `_message` error in this environment. I also make the pipeline robust when `NEEDTRAIN=False` (Kaggle) by ensuring `TARGETS` is always defined from `sample_submission.csv` even if `train.csv` isn’t loaded yet, and by guarding training-only variables like `TARGETS_RAW`. Finally, I keep your inference core logic intact and ensure the generated submission is always valid (correct columns/order, no NaNs, probabilities clipped and row-normalized to sum to 1) and saved as `submission.csv`.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf import crash by avoiding the forced pure‑python protobuf setting that triggers the `MessageFactory.GetPrototype` error in this Kaggle image, and instead letting TensorFlow use its compatible default protobuf runtime. I also keep `TARGETS` available for both training and inference and keep the current inference/training core logic unchanged. Finally, I ensure the script always reaches the end and writes a valid `submission.csv` with the exact sample submission columns and row-normalized probabilities.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash by explicitly forcing TensorFlow to use the pure‑python protobuf implementation *before* importing TensorFlow (this is the known-compatible setting for this notebook’s error: `MessageFactory` missing `GetPrototype`). I keep your model/training/inference logic unchanged, only adjusting environment setup order and adding a safe CPU fallback so the script runs end-to-end in Kaggle even if no GPU is available. This should unblock loading the provided fold weights (instead of falling back to priors), which is the minimal legitimate change expected to improve KL score toward your 0.289 target. The submission writing stays identical and still guarantees valid row-normalized probabilities with the correct columns and `submission.csv` output.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash by not forcing the pure‑python protobuf runtime, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle image, while keeping your TF/Keras setup and model logic unchanged. Then I make the inference path more likely to load provided fold weights by searching for a `models*` dataset robustly (even if the directory name doesn’t start with exactly `models`), which should legitimately improve the score toward your target because it avoids falling back to priors. Finally, I keep the submission writing/normalization safeguards so the output is always a valid `submission.csv` with row sums equal to 1.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure‑python protobuf implementation *before* importing TensorFlow, which is the minimal environment-order change needed for this Kaggle image. Then I make sure inference actually uses the provided fold weights when available by (1) disabling mixed precision during inference to avoid dtype/serialization mismatches and (2) building/compiling each fold model in the same way before `load_weights`, without changing the model architecture itself. Finally, I keep the existing probability clipping/row-normalization and submission column alignment so the script always produces a valid `submission.csv` with row sums exactly 1.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` setting, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle image. Then I make the script robust in Kaggle inference mode by not reading `train.csv` unless needed for training or for the fallback prior (this avoids unnecessary work and failures before submission writing). Finally, I keep your existing inference logic intact but add a safer “models folder” resolver that prefers a directory that actually contains the fold weight files, which should legitimately improve the score toward the target by ensuring the ensemble weights are loaded instead of falling back to a weak prior.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by forcing the pure‑python protobuf implementation *before* importing TensorFlow and by avoiding conflicting protobuf env settings. Then, to move your KL score down toward the 0.289 target (from 0.767), I make inference more likely to actually load your fold weights by (1) resolving the models directory robustly and (2) trying both `fold{k}_stage2.h5` and `fold{k}_stage2.weights.h5` naming patterns without changing the model architecture. Finally, I keep your submission formatting/normalization safeguards but ensure the test batch slicing is correct and deterministic so predictions align 1:1 with `sample_submission.csv`.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow import crash by removing the forced protobuf runtime override that triggers `MessageFactory.GetPrototype` in this Kaggle image, letting TF use its bundled compatible protobuf. Then I keep your existing model/inference logic intact but make weight-loading more robust by accepting either `.h5`/`.weights.h5` and by allowing fewer-than-5 folds if only some weights exist (still averaging available folds) to avoid silently falling back to the weak prior. Finally, I ensure the submission is always aligned to `sample_submission.csv`, strictly normalized to sum to 1 per row, and written as `submission.csv`.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by forcing TensorFlow to use the compatible pure‑python protobuf implementation *before* importing TensorFlow, and by keeping that environment setup minimal and consistent. Then I fix an inference-only bug in `DataGenerator` where `x_stft` is referenced before assignment if `DATATYPE` includes `"stft"` (this is score-neutral but prevents hidden runtime failures when switching modalities). Finally, I keep your model and prediction logic unchanged, but make weight loading more robust (try both `_stage2.h5` and `_stage2.weights.h5`, plus a safe `expect_partial()` fallback) so the script is more likely to use real fold weights instead of the weak prior, which should legitimately move KL down toward the 0.289 target.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import glob
import warnings
import io
import time
import gc

import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # local training or online testing
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False

    best = None
    for dir_name in os.listdir("/kaggle/input/"):
        full = os.path.join("/kaggle/input", dir_name)
        if not os.path.isdir(full):
            continue
        files = set(os.listdir(full))
        has_any_fold = any(
            (
                fn.startswith("fold")
                and (fn.endswith("_stage2.h5") or fn.endswith("_stage2.weights.h5"))
            )
            for fn in files
        )
        has_any_fold_child = False
        if not has_any_fold:
            for child in glob.glob(os.path.join(full, "*")):
                if os.path.isdir(child):
                    child_files = set(os.listdir(child))
                    if any(
                        (
                            fn.startswith("fold")
                            and (
                                fn.endswith("_stage2.h5")
                                or fn.endswith("_stage2.weights.h5")
                            )
                        )
                        for fn in child_files
                    ):
                        has_any_fold_child = True
                        break
        if has_any_fold or has_any_fold_child:
            best = dir_name
            break
        if best is None and dir_name.lower().startswith("models"):
            best = dir_name

    if best is not None:
        LOAD_MODELS_FROM = best

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # seconds
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100
SPE_WIDE = 256

STFT_LENGTH = 50
STFT_HIGH = 64
STFT_WIDE = round(STFT_LENGTH / 0.5)

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

print(tf.config.list_physical_devices("GPU"))

try:
    from tensorflow.python.framework.ops import reset_default_graph
except Exception:

    def reset_default_graph():
        return None


gpus = tf.config.list_physical_devices("GPU")
if len(gpus) == 0:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("Using CPU")
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print("Using 1 GPU")
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

MIX = True if NEEDTRAIN else False
if MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    tf.keras.mixed_precision.set_global_policy("float32")
    print("Using full precision")

sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
TARGETS = sample_sub.columns[1:]
print("Targets", list(TARGETS))

df = None
if NEEDTRAIN:
    df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
    print("Train shape:", df.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
        train = pd.read_csv("train.csv")



## === cell 2
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
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
                ff, tt, ss = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=100, nfft=640
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

            eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

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
        if "stft" in DATATYPE:
            x_stft = np.zeros(
                (len(indexes), 4, STFT_HIGH, STFT_WIDE * 4),
                dtype="float32",
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        targets_batch = list()

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = getattr(row, "sign_id", i)

            if self.mode != "test":
                sample_weight = float(np.sum(row[TARGETS_RAW].values)) / 20
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

                eeg = eeg - np.mean(eeg_save, 1, keepdims=True)
                eeg = np.clip(eeg, a_min=-255, a_max=255)
                eeg = eeg + 255
                eeg = eeg / 2

                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                stft = np.nan_to_num(stft, nan=0.0)
                stft = stft.astype(np.float32)
                if stft.shape[0] < 4:
                    reps = int(np.ceil(4 / stft.shape[0]))
                    stft4 = np.tile(stft, (reps, 1, 1))[:4]
                else:
                    stft4 = stft[:4]
                stft4 = stft4[:, :STFT_HIGH, :STFT_WIDE]
                if stft4.shape[1] < STFT_HIGH:
                    pad = STFT_HIGH - stft4.shape[1]
                    stft4 = np.pad(stft4, ((0, 0), (0, pad), (0, 0)), mode="constant")
                if stft4.shape[2] < STFT_WIDE:
                    pad = STFT_WIDE - stft4.shape[2]
                    stft4 = np.pad(stft4, ((0, 0), (0, 0), (0, pad)), mode="constant")
                stft4 = np.concatenate([stft4, stft4, stft4, stft4], axis=2)
                x_stft[j] = stft4

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
                sample_weights[j] = sample_weight if self.sample_weights else 1.0

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
def _find_pretrained_notop(model_name: str) -> str:
    fname = f"{model_name}_notop.h5"
    candidates = []
    if PLATFORM == "local":
        candidates += [os.path.join("./input/pre-trained-weights", fname)]
        candidates += glob.glob(os.path.join("./input", "**", fname), recursive=True)
    else:
        candidates += [os.path.join("/kaggle/input/pre-trained-weights", fname)]
        candidates += glob.glob(
            os.path.join("/kaggle/input", "**", fname), recursive=True
        )
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def build_model():
    inp = list()

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
        x_spe = tf.keras.layers.Reshape(
            (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
        )(inp_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

        base_model_spe = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        w = _find_pretrained_notop(base_model_spe.name)
        if w and NEEDTRAIN:
            base_model_spe.load_weights(w)
        base_model_spe._name = "spe_extractor"

        base_model_spe_pre = tf.keras.Model(
            base_model_spe.input, base_model_spe.get_layer("block3b_add").output
        )
        base_model_spe_pre._name = "spe_extractor_pre"
        x_spe1 = base_model_spe_pre(x_spe[:, 0, :, :, :])
        x_spe2 = base_model_spe_pre(x_spe[:, 1, :, :, :])
        x_spe3 = base_model_spe_pre(x_spe[:, 2, :, :, :])
        x_spe4 = base_model_spe_pre(x_spe[:, 3, :, :, :])

        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])
        base_model_spe_after = tf.keras.Model(
            base_model_spe_pre.output, base_model_spe.output
        )
        base_model_spe_after._name = "spe_extractor_after"
        x_spe = base_model_spe_after(x_spe)

        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

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
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        if PLATFORM == "local":
            eeg_embed = tf.keras.layers.Conv1D(
                filters=30,
                kernel_size=10,
                strides=10,
                padding="same",
                use_bias=False,
                activation=None,
                kernel_initializer=IniToOne(),
                kernel_constraint=SumToOne(),
                input_shape=(None, 1),
            )
        else:
            eeg_embed = tf.keras.layers.Conv1D(
                filters=30,
                kernel_size=10,
                strides=10,
                padding="same",
                use_bias=False,
                activation=None,
            )

        x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg)

        x_eeg = tf.keras.layers.Concatenate(axis=-1)(
            [
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 0:10]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 10:20]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 20:30]
                ),
            ]
        )

        x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
        x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

        base_model_eeg = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        w = _find_pretrained_notop(base_model_eeg.name)
        if w and NEEDTRAIN:
            base_model_eeg.load_weights(w)
        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]

        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)
        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(4, STFT_HIGH, STFT_WIDE * 4), name="stft")
        x_stft = tf.keras.layers.Reshape(
            (inp_stft.shape[1], inp_stft.shape[2], inp_stft.shape[3], 1)
        )(inp_stft)
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        base_model_stft = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        w = _find_pretrained_notop(base_model_stft.name)
        if w and NEEDTRAIN:
            base_model_stft.load_weights(w)
        base_model_stft._name = "stft_extractor"

        base_model_stft_pre = tf.keras.Model(
            base_model_stft.input, base_model_stft.get_layer("block3b_add").output
        )
        base_model_stft_pre._name = "stft_extractor_pre"
        x_stft1 = base_model_stft_pre(x_stft[:, 0, :, :, :])
        x_stft2 = base_model_stft_pre(x_stft[:, 1, :, :, :])
        x_stft3 = base_model_stft_pre(x_stft[:, 2, :, :, :])
        x_stft4 = base_model_stft_pre(x_stft[:, 3, :, :, :])

        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [x_stft1, x_stft2, x_stft3, x_stft4]
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
        x_stft = tf.keras.layers.Dropout(0.8)(x_stft)

        inp.append(inp_stft)
        y_stft = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_stft)

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=inp_img
        )
        w = _find_pretrained_notop(base_model_img.name)
        if w and NEEDTRAIN:
            base_model_img.load_weights(w)
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




## === cell 7
if NEEDTRAIN:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K
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
            np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 6
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 6
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
        plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
        gc.collect()



## === cell 8
if not NEEDTRAIN:

    def find_models_dir(base_path: str) -> str:
        base_path = os.path.abspath(base_path)
        candidates = []

        candidates.append(base_path)
        candidates += [
            p for p in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(p)
        ]
        candidates += [os.path.join(base_path, "models")]
        candidates += [
            os.path.join(p, "models") for p in candidates if os.path.isdir(p)
        ]

        seen = set()
        uniq = []
        for c in candidates:
            c = os.path.abspath(c)
            if c not in seen and os.path.isdir(c):
                seen.add(c)
                uniq.append(c)

        for cand in uniq:
            any_fold = False
            for fn in os.listdir(cand):
                if fn.startswith("fold") and (
                    fn.endswith("_stage2.h5") or fn.endswith("_stage2.weights.h5")
                ):
                    any_fold = True
                    break
            if any_fold:
                return cand
        return ""

    def resolve_fold_weight_path(models_dir: str, fold_k: int) -> str:
        p1 = os.path.join(models_dir, f"fold{fold_k}_stage2.h5")
        p2 = os.path.join(models_dir, f"fold{fold_k}_stage2.weights.h5")
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return ""

    def safe_load_weights(model: tf.keras.Model, wpath: str) -> bool:
        try:
            model.load_weights(wpath)
            return True
        except Exception as e1:
            try:
                status = model.load_weights(wpath, by_name=True, skip_mismatch=True)
                try:
                    status.expect_partial()
                except Exception:
                    pass
                return True
            except Exception as e2:
                print("Failed to load weights:", wpath)
                print("Error1:", repr(e1))
                print("Error2:", repr(e2))
                return False

    MODELS_DIR = find_models_dir(LOAD_MODELS_FROM)
    print("Requested LOAD_MODELS_FROM:", LOAD_MODELS_FROM)
    print("Resolved MODELS_DIR:", MODELS_DIR if MODELS_DIR else "(none found)")

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    print("Test shape", test.shape)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sub = sample_sub[["eeg_id"]].copy()

    if MODELS_DIR:
        models = []
        available_folds = []
        for model_i in range(SPLITS):
            wpath = resolve_fold_weight_path(MODELS_DIR, model_i)
            if wpath:
                available_folds.append(model_i)

        print("Available folds:", available_folds)

        if len(available_folds) == 0:
            MODELS_DIR = ""  # trigger fallback
        else:
            with strategy.scope():
                for model_i in available_folds:
                    print(f"Fold {model_i + 1}")
                    model = build_model()
                    model.compile(
                        loss=tf.keras.losses.KLDivergence(),
                        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE),
                    )
                    wpath = resolve_fold_weight_path(MODELS_DIR, model_i)
                    ok = safe_load_weights(model, wpath)
                    if ok:
                        models.append(model)

            if len(models) == 0:
                MODELS_DIR = ""  # trigger fallback

            preds_all = []

            test["sign_id"] = test.index.values

            if MODELS_DIR and ("spe" in DATATYPE):
                PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
                files_test = os.listdir(PATH_test)
                print(f"There are {len(files_test)} test spectrogram parquets")

                for i, f in enumerate(files_test):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    tmp = pd.read_parquet(f"{PATH_test}{f}")
                    name = int(f.split(".")[0])
                    spectrograms_test[name] = tmp.iloc[:, 1:].values

            if MODELS_DIR:
                PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / RSFREQ, "bandpass"
                )
                b2, a2 = signal.butter(
                    3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
                )

                start_idx = 0
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
                        ff, tt, ss = signal.spectrogram(
                            eeg2,
                            axis=1,
                            fs=RSFREQ,
                            nperseg=RSFREQ,
                            noverlap=100,
                            nfft=640,
                        )
                        ss[np.isnan(ss)] = 0
                        ss = ss[:, (ff > 0) * (ff <= 20), :]

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

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                    if "eeg" in DATATYPE:
                        eegs_test[eeg_id] = eeg
                    if "stft" in DATATYPE:
                        stfts_test[eeg_id] = ss
                        stfts_test[-eeg_id] = tt

                    if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                        batch_end = i + 1
                        batch_df = test.iloc[start_idx:batch_end].reset_index(drop=True)

                        preds = []
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
                        for mi in range(len(models)):
                            pred = models[mi].predict(test_gen, verbose=0)
                            preds.append(pred)
                        pred = np.mean(preds, axis=0)

                        del eegs_test
                        gc.collect()
                        eegs_test = {}
                        start_idx = batch_end

                        if len(preds_all) == 0:
                            preds_all = pred.copy()
                        else:
                            preds_all = np.concatenate((preds_all, pred), axis=0)

                preds_all = np.asarray(preds_all, dtype=np.float64)
                preds_all = np.clip(preds_all, 1e-12, 1.0)
                preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)
                sub[TARGETS] = preds_all

    if not MODELS_DIR:
        if df is None:
            df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
            print("Loaded train.csv for fallback prior:", df.shape)

        test_pid = test.set_index("eeg_id")["patient_id"].to_dict()

        vote_cols = list(TARGETS)
        train_votes = df[["patient_id"] + vote_cols].copy()
        patient_sum = (
            train_votes.groupby("patient_id")[vote_cols].sum().astype(np.float64)
        )

        global_sum = df[vote_cols].sum(axis=0).astype(np.float64).values
        alpha = 1.0  # mild smoothing to avoid zeros
        global_prior = (global_sum + alpha) / (
            global_sum.sum() + alpha * len(vote_cols)
        )

        preds = np.zeros((len(sub), len(vote_cols)), dtype=np.float64)
        for idx, eeg_id in enumerate(sub["eeg_id"].values):
            pid = test_pid.get(eeg_id, None)
            if pid in patient_sum.index:
                s = patient_sum.loc[pid].values
                prior = (s + alpha) / (s.sum() + alpha * len(vote_cols))
            else:
                prior = global_prior
            preds[idx] = prior

        preds = np.clip(preds, 1e-12, 1.0)
        preds = preds / preds.sum(axis=1, keepdims=True)
        sub[TARGETS] = preds

    sub = sub[sample_sub.columns]
    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.nan_to_num(
        vals,
        nan=1.0 / len(TARGETS),
        posinf=1.0 / len(TARGETS),
        neginf=1.0 / len(TARGETS),
    )
    vals = np.clip(vals, 1e-12, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub[TARGETS] = vals

    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print("Submission shape", sub.shape)
    print(sub.head())
