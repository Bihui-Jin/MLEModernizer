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

0.2746703339616783

# 6. Current score

1.47425

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the protobuf implementation override that is forcing the pure-Python protobuf runtime (which is incompatible with TensorFlow in this Kaggle image and triggers the `MessageFactory.GetPrototype` error). I also make the platform detection robust for Kaggle Notebook/Code Runner paths and ensure inference still runs when no pretrained weights directory is present (Kaggle test mode should only load your fold weights). Finally, I keep the core training/inference logic unchanged while guaranteeing the script always writes a valid `submission.csv` with the required columns and row-wise probability normalization.'
- What this solution (achieved 1.40995) has done: 'I remove the protobuf-runtime forcing (which triggers the `MessageFactory.GetPrototype` crash with TensorFlow in this environment) and make the script robust to Kaggle paths while preserving the exact modeling/training logic. I also ensure the code always finds the competition dataset directory (whether it’s under `/kaggle/input/...` or `/kaggle/data/...`) and always writes a valid `submission.csv` with the required columns and row-wise probability normalization. Finally, I keep inference identical but add a safe fallback to the uniform submission only when no fold weights are present, so you always get a valid CSV end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by removing the forced protobuf “cpp” runtime setting, which is incompatible in this Kaggle image and triggers the `google.protobuf.pyext._message` import error. I also make the platform detection and dataset path selection robust so the script reliably finds `/kaggle/input/hms-harmful-brain-activity-classification` (or `/kaggle/data/...`) without changing your training/inference logic. Finally, I keep the model/training semantics intact but guarantee that inference always writes a valid `submission.csv` with the exact required columns and row-wise probability normalization (no more “Not yielded”).'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring we don’t force an incompatible protobuf runtime and by importing TensorFlow only after the environment is sanitized; this is a runtime blocker. I also make inference robust by (a) always finding the dataset directory, (b) ensuring `build_model()` can run in Kaggle inference without requiring missing pre-trained weights, and (c) writing a valid `submission.csv` with correct columns and per-row probability normalization. Because your current score (1.40995, lower-is-better) is far from the target (~0.2747), I remove the “uniform submission” fallback that severely hurts score and instead fall back to a simple, legitimate prior based on the training label distribution only when no fold weights are found (still minimal change, but strongly improves KL over uniform). All modeling/training architecture and loss remain unchanged.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by ensuring we don’t force an incompatible protobuf runtime and by importing TensorFlow only after environment variables are sanitized (this is the root cause of the `MessageFactory.GetPrototype` failure). I also make dataset path detection deterministic for Kaggle’s `/kaggle/input/...` layout so train/test CSVs are always found without manual edits. To move the KL score down toward your target (lower is better) while keeping your core model/training logic intact, I remove the “train-prior fallback then exit” behavior during inference and instead ensure the script always loads the available fold weights from the correct location (and only uses the train-prior fallback if truly no weights exist). Finally, I keep submission formatting strict (correct columns, correct order, per-row normalization) and always write `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring protobuf is forced to the pure-Python implementation *before* importing TensorFlow (this is the only reliable way to avoid the known incompatibility in this environment). I also make the dataset directory selection deterministic for Kaggle by preferring the explicit competition folder, preventing accidental reads from the wrong level. Finally, to move the KL score down toward your target (lower is better) without changing the model/training core, I remove the inference-time mismatch where Kaggle builds a different Conv1D embedding layer than training; instead, both training and inference use the exact same initialization/constraints path, improving weight compatibility and predictions.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf runtime (it’s what triggers `MessageFactory.GetPrototype` in this Kaggle image) and keeping TensorFlow import order safe. I also make the Kaggle platform detection choose the correct dataset directory deterministically and ensure inference always writes a valid `submission.csv` with the required column order and row-wise normalization. To move the KL score down toward your target (lower is better) without changing model/training semantics, I keep your existing model inference path intact and only improve the “no weights found” fallback to use a properly normalized train-label prior (already present) and ensure it uses the same TARGETS order as the sample submission.'
- What this solution (achieved 1.47425) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"` setting (it causes the missing `_message` error in this environment) and letting TensorFlow use the compatible runtime. Then I make the inference-only path robust by ensuring `df` (train metadata) is always loaded even when `NEEDTRAIN=False`, so the patient/global prior fallback can run without a `NameError`. Finally, I keep your model/training/inference logic unchanged while guaranteeing that a valid `submission.csv` is always written with the exact required columns and per-row probability normalization.'
- What this solution (achieved 1.47425) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow, which is the most stable workaround in Kaggle’s current image. I also make inference score move toward your target by ensuring the script actually finds and loads fold weights from common Kaggle working/input locations instead of silently falling back to the weak patient/global-prior submission (which is what yields ~1.47 KL). Finally, I keep your model/training/inference logic intact while ensuring the submission is always written as `submission.csv` with the exact required columns and per-row normalization to sum to 1.'
- What this solution (achieved 1.47425) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by removing the forced pure-Python protobuf override and ensuring we don’t set an incompatible protobuf implementation before importing TensorFlow. Then I make inference reliably locate fold weight files by expanding the search to common Kaggle locations (including `/kaggle/working`, `/kaggle/input`, and the current directory), preventing the code from silently falling back to the weak patient/global prior that yields ~1.47 KL. Finally, I keep your model/training/inference logic unchanged while guaranteeing the submission is written as `submission.csv` with the exact required columns and per-row probability normalization.'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri

email: syuri@tju.edu.cn
"""

import os

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
]:
    if k in os.environ:
        os.environ.pop(k, None)

os.environ["KERAS_BACKEND"] = "tensorflow"

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.path.exists("/kaggle") and (
    os.path.exists("/kaggle/input") or os.path.exists("/kaggle/data")
):
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    try:
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
    except FileNotFoundError:
        pass
else:
    PLATFORM = "local"
    try:
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
    except FileNotFoundError:
        pass

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)


def _find_dataset_dir():
    if PLATFORM == "local":
        candidates = [
            "./input/hms-harmful-brain-activity-classification",
            "./data/hms-harmful-brain-activity-classification",
            "./input",
            "./data",
        ]
    else:
        candidates = [
            "/kaggle/input/hms-harmful-brain-activity-classification",
            "/kaggle/data/hms-harmful-brain-activity-classification",
            "/kaggle/input",
            "/kaggle/data",
        ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            return p
    return candidates[0]


if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = _find_dataset_dir()
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = _find_dataset_dir()

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg

eegs = {}  # preprocessed eegs for training
eegs_test = {}

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

import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow.keras import optimizers

print(tf.version.VERSION)
print(tf.config.list_physical_devices("GPU"))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 按需分配显存
    except RuntimeError as e:
        print(e)

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

MIX = True
if MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    import itertools

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
        if not os.path.exists("train.csv"):
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
            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data
            train.to_csv("train.csv", index=False)
        train = pd.read_csv("train.csv")
else:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):

            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet")), engine="pyarrow"
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

            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)

            eeg = np.array(eeg, dtype=np.float32)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        sample_weights=False,
        mode="train",
        eegs=None,
        stage=2,
    ):

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stage = stage
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
        x_eeg = np.zeros(
            (len(indexes), EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)),
            dtype="float32",
        )
        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            if self.mode != "test":
                sample_weight = sum(row[TARGETS_RAW].values) / 20

            if self.mode == "test":
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
                r_eeg = row.eeg_label_offset_seconds
                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0, r_eeg)
                    r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

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

            eeg = eeg[
                :,
                round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                    (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                ),
            ]

            if self.mode == "train":
                if (self.stage == 2) and (np.random.rand() > 0):
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                else:
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

            eeg = np.clip(eeg, a_min=-255, a_max=255)
            eeg = eeg + 255
            eeg = eeg / 2

            x_eeg[j] = eeg

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1

        return x_eeg, y, sample_weights




## === cell 2
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




## === cell 3
def build_model():
    inp_eeg = tf.keras.Input(
        shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)), name="eeg"
    )
    x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
        inp_eeg
    )

    strides = 10

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
        wpath = None
        if PLATFORM == "local":
            wpath = f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
        if PLATFORM == "kaggle":
            wpath = f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
        if (wpath is not None) and os.path.exists(wpath):
            base_model_eeg.load_weights(wpath)
        else:
            print(
                "WARNING: pre-trained weights not found; continuing with random init."
            )

    base_model_eeg.name = "eeg_extractor"

    x_eeg = base_model_eeg(x_eeg)

    x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
        x_eeg
    )

    model = tf.keras.Model(inputs=inp_eeg, outputs=y)
    return model




## === cell 4
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
            eegs=eegs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            eegs=eegs,
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
    elif stage == 2:
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            eegs=eegs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2 * 2,
            mode="valid",
            eegs=eegs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1 * 3)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1 * 3,
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
    tf.keras.backend.clear_session()
    gc.collect()

    import itertools

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




## === cell 5
if __name__ == "__main__":
    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        mp.set_start_method("spawn")  # 设置进程启动方式为 spawn

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
        preds_all = []
        models = list()

        candidate_model_dirs = []

        for p in [
            "/kaggle/working/models",
            "/kaggle/working",
            "/kaggle/input",
            "/kaggle/data",
        ]:
            if os.path.exists(p):
                candidate_model_dirs.append(p)

        if LOAD_MODELS_FROM:
            candidate_model_dirs.append(LOAD_MODELS_FROM)

        if os.path.exists("./models"):
            candidate_model_dirs.append("./models")
        if os.path.exists("."):
            candidate_model_dirs.append(".")

        if LOAD_MODELS_FROM and os.path.isdir(LOAD_MODELS_FROM):
            candidate_model_dirs.append(os.path.join(LOAD_MODELS_FROM, "models"))

        if os.path.exists("/kaggle/input"):
            try:
                for dir_name in os.listdir("/kaggle/input"):
                    if dir_name.startswith("models"):
                        candidate_model_dirs.append(
                            os.path.join("/kaggle/input", dir_name)
                        )
                        candidate_model_dirs.append(
                            os.path.join("/kaggle/input", dir_name, "models")
                        )
            except Exception:
                pass

        seen = set()
        candidate_model_dirs = [
            d for d in candidate_model_dirs if d and (not (d in seen or seen.add(d)))
        ]

        found_dir = None
        for d in candidate_model_dirs:
            if d and os.path.exists(d):
                for model_i in range(999):
                    if os.path.exists(
                        os.path.join(d, f"fold{model_i}_stage2.weights.h5")
                    ):
                        found_dir = d
                        break
            if found_dir is not None:
                break

        if found_dir is None:
            found_dir = LOAD_MODELS_FROM

        print("Using model directory:", found_dir)

        for model_i in range(999):
            w = os.path.join(found_dir, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(w):
                print(f"Fold {model_i + 1}")
                model = build_model()
                model.load_weights(w)
                models.append(model)

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        if len(models) == 0:
            print("WARNING: No fold weights found. Writing patient-prior submission.")
            sample_sub = pd.read_csv(
                os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
            )

            global_pri = df[TARGETS].sum(axis=0).values.astype(np.float64)
            global_pri = np.clip(global_pri, 1e-12, None)
            global_pri = global_pri / global_pri.sum()

            df_votes = df[["patient_id"] + list(TARGETS)].copy()
            patient_sum = df_votes.groupby("patient_id")[list(TARGETS)].sum()

            pri_mat = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
            for idx, pid in enumerate(test["patient_id"].values):
                if pid in patient_sum.index:
                    v = patient_sum.loc[pid].values.astype(np.float64)
                    v = np.clip(v, 1e-12, None)
                    v = v / v.sum()
                    pri_mat[idx] = v
                else:
                    pri_mat[idx] = global_pri

            sub = pd.DataFrame({"eeg_id": sample_sub["eeg_id"].values})
            for k, c in enumerate(TARGETS):
                sub[c] = pri_mat[:, k]

            sub[TARGETS] = (
                sub[TARGETS]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0 / len(TARGETS))
            )
            sub[TARGETS] = sub[TARGETS].clip(lower=1e-12)
            sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].sum(
                axis=1
            ).values.reshape(-1, 1)

            sub = sub[sample_sub.columns.tolist()]
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
        else:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")
                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet")),
                    engine="pyarrow",
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

                eegs_test[eeg_id] = eeg

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    batch_end = i + 1
                    batch_start = batch_end - len(eegs_test)
                    batch_df = test.iloc[batch_start:batch_end].reset_index(drop=True)

                    preds = []
                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        eegs=eegs_test,
                        stage=2,
                    )
                    for model_i in range(len(models)):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0)

                    del eegs_test
                    gc.collect()
                    eegs_test = {}

                    if len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

            sample_sub = pd.read_csv(
                os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
            )
            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = preds_all.astype(np.float64)

            sub[TARGETS] = (
                sub[TARGETS]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0 / len(TARGETS))
            )
            sub[TARGETS] = sub[TARGETS].clip(lower=1e-12)
            row_sums = sub[TARGETS].sum(axis=1).values.reshape(-1, 1)
            sub[TARGETS] = sub[TARGETS].values / row_sums

            sub = sub[sample_sub.columns.tolist()]
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
