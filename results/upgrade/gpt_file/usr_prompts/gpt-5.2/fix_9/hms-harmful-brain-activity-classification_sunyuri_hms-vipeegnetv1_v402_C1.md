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

0.2837218611159194

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting, which is incompatible with the Kaggle TensorFlow build and triggers the `MessageFactory.GetPrototype` error. On Kaggle, your code also tries to train but has no preprocessed EEG dict loaded (so the generator ends up effectively empty / invalid); I change Kaggle behavior to strictly do inference-only using provided model weights when available, and otherwise fall back to a safe uniform-probability submission so you always get a valid CSV. I also fix a submission column alignment bug (you were writing `sub[TARGETS]` but then selecting sample_submission columns) and enforce probability normalization + row-sum=1. Finally, I make the code robust to missing models/preprocess paths while keeping the model and generator logic intact.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing a compatible protobuf implementation/version before importing TensorFlow. Then I ensure Kaggle inference uses the provided official ImageNet EfficientNetV2B3 weights (instead of random weights) which should substantially reduce KL divergence versus the current weak baseline, while keeping your model architecture and training/inference logic unchanged. Finally, I harden test-time generation against missing/short EEG segments and enforce strict probability normalization so the submission is always valid and sums to 1 per row.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override and keeping the environment stable for Kaggle’s TensorFlow build. Then I fix a subtle training-data row selection bug in the generator (it accidentally filters with `*` instead of `&` and omits `other_vote`, which can silently select wrong rows and hurt KL), which should improve score toward your target without changing the model architecture or training semantics. Finally, I make inference more robust to missing/failed parquet reads by safely falling back to zeros for that EEG, and I keep strict probability clipping+normalization to guarantee a valid submission CSV.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by removing the environment settings that force deterministic ops and mixed precision, which are known to trigger this incompatibility on Kaggle’s TF/protobuf build. Then I ensure inference can actually run by loading the required test EEG signals even when `NEEDTRAIN=False` (right now the training-side EEG cache stays empty on Kaggle). Finally, I keep your model and generator logic intact but harden the submission pipeline: always build predictions in the exact `sample_submission.csv` column order, clip+renormalize probabilities to sum to 1, and always write `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing a compatible pure-Python protobuf implementation *before* importing TensorFlow (this is the root runtime blocker in your trace). Then I make Kaggle inference actually use trained fold weights if they exist, otherwise still produce a valid uniform-probability submission, while keeping your model/generator logic unchanged. Finally, I harden a couple of edge cases that can silently break inference quality (missing EEG in dict and empty row matches) and ensure the submission columns exactly match `sample_submission.csv` and every row sums to 1.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

os.environ["KERAS_BACKEND"] = "tensorflow"

NEEDTRAIN = True  # train the model (may be overridden on Kaggle)
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # local training/testing
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img ***
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 40
SPE_WIDE = 1000

STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3 * BATCHSIZE / 16
EPOCHS = 15
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

import tensorflow as tf

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

print("Using full precision (stability-first)")

if NEEDTRAIN:
    import itertools

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

TARGETS_RAW = [c + "_raw" for c in TARGETS]

if NEEDTRAIN:
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

            y_data = train[TARGETS].values.astype(np.float32)
            train[TARGETS_RAW] = y_data
            y_norm = y_data / np.clip(y_data.sum(axis=1, keepdims=True), 1e-6, None)
            train[TARGETS] = y_norm
            train.to_csv("train.csv", index=False)
else:
    train = df.head(1).copy()
    train["sign_id"] = 0
    for c in TARGETS_RAW:
        train[c] = train[c.replace("_raw", "")]

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
            else:
                eegs = {}
        if "stft" in DATATYPE:
            if os.path.exists(os.path.join(datapath, "stfts.npy")):
                stfts = np.load(
                    os.path.join(datapath, "stfts.npy"), allow_pickle=True
                ).item()
            else:
                stfts = {}
        if "img" in DATATYPE:
            if os.path.exists(os.path.join(datapath, "imgs.npy")):
                imgs = np.load(
                    os.path.join(datapath, "imgs.npy"), allow_pickle=True
                ).item()
            else:
                imgs = {}



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
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        if self.mode == "test":
            return x
        if self.sample_weights:
            return x, y, sample_weights
        else:
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
            _ = row.sign_id  # compatibility

            if self.mode == "test":
                r_eeg = 0
            else:
                sample_weight = float(sum(row[TARGETS_RAW].values)) / 20.0

                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    & (df.seizure_vote == row.seizure_vote_raw)
                    & (df.lpd_vote == row.lpd_vote_raw)
                    & (df.gpd_vote == row.gpd_vote_raw)
                    & (df.lrda_vote == row.lrda_vote_raw)
                    & (df.grda_vote == row.grda_vote_raw)
                    & (df.other_vote == row.other_vote_raw),
                    :,
                ].reset_index(drop=True)
                if len(rows) == 0:
                    rows = pd.DataFrame([row])

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

                r_eeg = (
                    row.eeg_label_offset_seconds
                    if "eeg_label_offset_seconds" in row
                    else 0
                )
                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0, r_eeg)
                    if (self.eegs is not None) and (row.eeg_id in self.eegs):
                        r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)
                    else:
                        r_eeg = 0

            if "eeg" in DATATYPE:
                if (self.eegs is None) or (row.eeg_id not in self.eegs):
                    full = np.zeros(
                        (len(BRAIN), int(EEG_LENGTH * RSFREQ)), dtype=np.float32
                    )
                else:
                    full = self.eegs[row.eeg_id]

                start = int(round(r_eeg * RSFREQ))
                end = int(round((r_eeg + 50) * RSFREQ))
                if end > full.shape[1]:
                    pad = end - full.shape[1]
                    full = np.pad(full, ((0, 0), (0, pad)), mode="constant")
                eeg = full[:, start:end]

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

            if self.mode != "test":
                yv = row[TARGETS].values.astype(np.float32)
                y[j] = yv / np.clip(np.sum(yv), 1e-6, None)
                sample_weights[j] = sample_weight if self.sample_weights else 1.0

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "img" in DATATYPE:
            x["img"] = x_img

        return x, y, sample_weights




## === cell 4
class CosineAnnealingLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
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
        return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

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




## === cell 5
def build_model():
    inp = []

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
            include_top=False, weights="imagenet", include_preprocessing=True
        )

        if NEEDTRAIN:
            w_local = f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
            w_kaggle = (
                f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
            )
            w_path = w_local if PLATFORM == "local" else w_kaggle
            if os.path.exists(w_path):
                base_model_eeg.load_weights(w_path)
            else:
                print(
                    f"Note: custom pretrained weights not found at {w_path}. Using ImageNet weights."
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

    y = y_eeg * 1
    return tf.keras.Model(inputs=inp, outputs=y)




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
    else:
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

    if len(train_gen_stage) == 0 or len(valid_gen_stage) == 0:
        raise ValueError(
            f"Empty generator: train_len={len(train_gen_stage)}, valid_len={len(valid_gen_stage)}. "
            "Check preprocessing / eegs dict population."
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

    loss_hist = history.history["loss"]
    val_loss_hist = history.history["val_loss"]
    epochs_range = range(1, len(loss_hist) + 1)
    plt.figure()
    plt.plot(epochs_range, loss_hist, "bo", label="loss")
    plt.plot(epochs_range, val_loss_hist, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
        fontsize=12,
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    valid_stage = (df_valid_stage1 if stage == 1 else df_valid_stage2)[TARGETS].values
    predict_stage = model.predict(valid_gen_stage, verbose=0)

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
    plt.xticks(tick_marks, [f"{TARGETS[i][:-5]}" for i in range(6)], fontsize=10)
    plt.yticks(tick_marks, [f"{TARGETS[i][:-5]}" for i in range(6)], fontsize=10)
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




## === cell 7
if __name__ == "__main__":
    models_found = False
    if not NEEDTRAIN:
        for model_i in range(100):
            if os.path.exists(
                os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
            ):
                models_found = True
                break

        if not models_found:
            print(
                f"Warning: No fold weights found in {LOAD_MODELS_FROM}. Will write uniform submission."
            )
    else:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        use_mp = PLATFORM == "local"
        if use_mp:
            mp.set_start_method("spawn", force=True)

        train_run = train
        SPLITS_RUN = SPLITS

        gkf = GroupKFold(n_splits=max(SPLITS_RUN, 2))

        splits_iter = gkf.split(
            train_run, train_run.expert_consensus, train_run.patient_id
        )

        for i, (train_index, valid_index) in enumerate(splits_iter):
            print("#" * 25)
            print(f"### Fold {i + 1}")

            df_train_stage1 = train_run.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train_run.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [1, 2]:
                if use_mp:
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
                        EPOCHS,
                        LEARN_RATE,
                        TARGETS,
                        TARGETS_RAW,
                    )

        LOAD_MODELS_FROM = "models"
        models_found = True

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    print("Test shape", test.shape)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sub_cols = list(sample_sub.columns[1:])  # required order/columns

    if not models_found:
        preds_all = np.full(
            (len(test), len(sub_cols)), 1.0 / len(sub_cols), dtype=np.float32
        )
    else:
        model_template = build_model()

        models = []
        for model_i in range(100):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(wpath):
                print(f"Fold {model_i + 1}")
                model = tf.keras.models.clone_model(model_template)
                model.load_weights(wpath)
                models.append(model)

        if len(models) == 0:
            print(
                f"Warning: No fold weights found in {LOAD_MODELS_FROM}. Will write uniform submission."
            )
            preds_all = np.full(
                (len(test), len(sub_cols)), 1.0 / len(sub_cols), dtype=np.float32
            )
        else:
            preds_all = None
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            n_test = len(test)

            for start in range(0, n_test, TEST_BATCHSIZE):
                end = min(start + TEST_BATCHSIZE, n_test)
                batch_df = test.iloc[start:end].reset_index(drop=True)

                eegs_test = {}
                for eeg_id in batch_df.eeg_id.values:
                    try:
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
                    except Exception as e:
                        print(f"Warning: failed to read eeg_id={eeg_id}: {repr(e)}")
                        eeg = np.zeros(
                            (len(BRAIN), int(EEG_LENGTH * RSFREQ)), dtype=np.float32
                        )

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

                batch_df = batch_df.copy()
                batch_df["sign_id"] = batch_df.index.values

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

                preds = []
                for m in models:
                    pred_m = m.predict(test_gen, verbose=0)
                    preds.append(pred_m)

                pred = np.mean(np.stack(preds, axis=0), axis=0)

                if preds_all is None:
                    preds_all = pred.astype(np.float32, copy=True)
                else:
                    preds_all = np.concatenate(
                        (preds_all, pred.astype(np.float32)), axis=0
                    )

                del eegs_test, test_gen, preds, pred
                gc.collect()

            preds_all = np.asarray(preds_all, dtype=np.float32)

    eps = 1e-7
    preds_all = np.clip(preds_all, eps, 1.0)
    preds_all = preds_all / np.clip(preds_all.sum(axis=1, keepdims=True), eps, None)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[sub_cols] = preds_all

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sum check:",
        float(np.max(np.abs(sub.iloc[:, 1:].sum(axis=1).values - 1.0))),
    )
