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

3.12

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

0.352176304303861

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the two runtime blockers: the protobuf environment override that breaks TensorFlow import on Kaggle, and the Keras model-building error caused by multiple EfficientNetB0 instances sharing the same default name. I remove the protobuf forcing lines (they’re not needed here) and explicitly give each EfficientNetB0 a unique `name` so Keras can build a multi-branch model. I also add a safe fallback for `TARGETS` when `NEEDTRAIN=False` (since `df` may not be loaded), ensuring the submission columns are always correct and ordered like `sample_submission.csv`. These changes are execution/stability fixes and should enable generating a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf version in the Kaggle Python 3.12 environment by forcing the pure-Python protobuf implementation before importing TensorFlow. I keep the model and data logic unchanged, but ensure the environment variables are set early and correctly so the script runs end-to-end. I also add a tiny numeric safety to guarantee the submission probabilities are valid (finite, clipped, and renormalized) without changing the intended evaluation semantics. This should both unblock execution and prevent invalid submissions while keeping the same modeling approach so your score can move back toward the target by actually using the trained weights instead of failing/falling back.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is required in the Kaggle Py3.12 image for some TF builds). Next, I remove the internet dependency that causes the EfficientNet ImageNet-weight download (403) by switching the EfficientNet backbones to `weights=None`, which preserves the architecture and allows offline execution. Finally, I keep the existing submission logic but add a small safety check to ensure all predictions are finite, clipped, and row-normalized so the CSV is always valid and won’t fail submission.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by removing the protobuf environment overrides that are incompatible with the Kaggle Py3.12 image (they currently trigger the `MessageFactory.GetPrototype` error). Then I load EfficientNetB0 with `weights="imagenet"` when available locally (Kaggle has these cached) and automatically fall back to `weights=None` if it can’t be loaded, which should improve score versus training-from-scratch while keeping the same architecture. Finally, I make the weight-loading path robust by searching common Kaggle locations for the `models2024030301` directory and, if no weights are found, still produce a valid probability submission. These changes preserve the model/training semantics and mainly restore the intended pretrained backbone + proper inference to move KL divergence down toward the target.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is required in some Kaggle Py3.12 images and directly addresses the `MessageFactory.GetPrototype` error). I keep the model/data pipeline identical, but make the EfficientNet backbone weight-loading deterministic and offline-safe by using `weights="imagenet"` when available and otherwise falling back to `weights=None` (no architecture change). I also ensure we actually find and load the provided fold weight files by searching common Kaggle input locations recursively for the expected `f*_stage*.h5` files; failing that, it still produces a valid uniform-probability submission. Finally, I keep the probability safety normalization so the submission always sums to 1 and matches `sample_submission.csv` column order.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blocker (`MessageFactory.GetPrototype` from protobuf/TensorFlow incompatibility) by removing the forced pure-Python protobuf environment variables that trigger this error in the Kaggle Py3.12 image. Then I restore the intended “use pretrained EfficientNet when possible” behavior while keeping it offline-safe (try `imagenet`, otherwise fall back to `None`) and ensure weight files are found reliably so you don’t silently fall back to uniform predictions (which is a major reason the score is far from the target). Finally, I keep your exact model/data logic intact but add a tiny, score-positive calibration (very light label-smoothing at inference) plus strict probability sanitization/normalization to ensure a valid submission that improves KL divergence without changing the architecture or training semantics.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation to pure-Python *before* importing TensorFlow (this resolves the `MessageFactory.GetPrototype` AttributeError in Kaggle Py3.12). Next, I make sure the script actually loads the provided fold weight files by broadening the weight-file search logic (the previous filter could accidentally exclude correct `/kaggle/input/...` paths, leading to uniform predictions and a much worse KL score). Finally, I keep the same model and inference logic but keep a small probability-safety normalization so the submission is always valid (rows sum to 1, finite, clipped), which is score-neutral aside from preventing invalid outputs.'

# 9. Code solution

## === cell 0
import os
import io
from PIL import Image

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

print("TensorFlow version =", tf.__version__)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024030301"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024
BATCHSIZE = 16

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

spectrograms = {}
eegs = {}
imgs = {}
stfts = {}

spectrograms2 = {}
eegs2 = {}
imgs2 = {}
stfts2 = {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enabling skipped:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision enabling skipped:", repr(e))
else:
    print("Using full precision")

SAMPLE_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
if PLATFORM == "local":
    SAMPLE_PATH = (
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )

sample_cols = pd.read_csv(SAMPLE_PATH, nrows=1).columns.tolist()
TARGETS = [c for c in sample_cols if c != "eeg_id"]
print("Targets", TARGETS)

if NEEDTRAIN:
    if PLATFORM == "local":
        df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
    elif PLATFORM == "kaggle":
        df = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
        )
    print("Train shape:", df.shape)
    df.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

    if READ_SPEC_FILES * READ_EEG_FILES * READ_IMG_FILES * READ_STFT_FILES:
        train_max = df.groupby("eeg_id")[["eeg_label_offset_seconds"]].agg(
            {"eeg_label_offset_seconds": "max"}
        )
        train_min = df.groupby("eeg_id")[["eeg_label_offset_seconds"]].agg(
            {"eeg_label_offset_seconds": "min"}
        )

        train_max.columns = ["eeg_label_offset_seconds_max"]
        train_min.columns = ["eeg_label_offset_seconds_min"]

        df2 = df.merge(train_max, on="eeg_id")
        df2 = df2.merge(train_min, on="eeg_id")

        df2["s_max"] = abs(
            df2.eeg_label_offset_seconds - df2.eeg_label_offset_seconds_max
        )
        df2["s_min"] = abs(
            df2.eeg_label_offset_seconds - df2.eeg_label_offset_seconds_min
        )

        xx = df2.loc[:, ["s_max", "s_min"]].min(1)
        df2 = df2.iloc[:, :15]
        df2["selected"] = xx

        df2 = df2.sort_values("selected", ascending=False).reset_index(drop=True)
        df3 = df2.drop_duplicates("eeg_id").reset_index(drop=True)

        num_all = 0
        for i in range(len(TARGETS)):
            num_all = max(num_all, sum(np.argmax(df3[TARGETS].values, 1) == i))

        train = pd.DataFrame()
        for i in range(len(TARGETS)):
            train_temp = df2.iloc[np.argmax(df2[TARGETS].values, 1) == i].reset_index(
                drop=True
            )
            train_temp = train_temp.iloc[
                np.random.permutation(len(train_temp))
            ].reset_index(drop=True)
            ii = 1
            while len(train_temp.groupby("eeg_id").head(ii)) < num_all:
                ii = ii + 1
            print(ii)
            if ii > 1:
                if len(train_temp.groupby("eeg_id").head(ii)) > num_all:
                    train_temp1 = train_temp.groupby("eeg_id").head(ii - 1)
                    train_temp2 = train_temp.groupby("eeg_id").head(ii)
                    train_temp2 = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.drop_duplicates(keep=False).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.iloc[
                        np.random.permutation(len(train_temp2))
                    ].reset_index(drop=True)
                    train_temp2 = train_temp2[
                        : (num_all - len(train_temp.groupby("eeg_id").head(ii - 1)))
                    ]
                    train_temp = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
            else:
                train_temp = train_temp.groupby("eeg_id").head(ii)
            print(len(train_temp))
            train = pd.concat([train, train_temp]).reset_index(drop=True)

        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.head()

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 2
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        data,
        batch_size=32,
        shuffle=False,
        augment=False,
        mode="train",
        specs=None,
        eegs=None,
        imgs=None,
        stfts=None,
    ):

        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.mode = mode
        self.specs = specs
        self.eegs = eegs
        self.imgs = imgs
        self.stfts = stfts
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            elif self.mode == "valid":
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = round(row.eeg_label_offset_seconds * SFREQ)
            else:
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = round(row.eeg_label_offset_seconds * SFREQ)

            if self.mode == "train":
                x1 = np.random.rand() * (LENGTH / 2 - 20)
                x2 = np.random.rand() * (LENGTH / 2 - 20)
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                else:
                    x1 = x1 + 20
                    x2 = x2 + 20
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))
                x_spe_max = min(x_spe_max, round(x_spe_min + LENGTH * 0.1))

                x1 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                x2 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                if np.random.rand() < 0.5:
                    x1 = x1 + EEG_LENGTH * SFREQ / 2
                    x2 = x2 + EEG_LENGTH * SFREQ / 2
                else:
                    x1 = x1 + 10 * SFREQ / 2
                    x2 = x2 + 10 * SFREQ / 2
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(max(x1, x2))

                x1 = np.random.rand() * (LENGTH / 2 - 42)
                x2 = np.random.rand() * (LENGTH / 2 - 42)
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                else:
                    x1 = x1 + 42
                    x2 = x2 + 42
                x_img_min = round(min(x1, x2))
                x_img_max = round(max(x1, x2))

            for k in range(4):
                if "spe" in DATATYPE:
                    spe = self.specs[row.spectrogram_id][
                        r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                    ].T
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
                    spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                    spe = np.array(spe, dtype=np.int16)

                    spe = self.cmaps[spe]

                    spe = np.reshape(spe, (100, 300, 3))
                    spe = spe[
                        :,
                        max(round((600 / 2 - 256) / 2), 0) : min(
                            (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
                        ),
                        :,
                    ]
                    spe = np.array(
                        tf.image.resize(spe, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

                    if self.mode == "train":
                        spe[:, x_spe_min:x_spe_max, :] = 0

                    x_spe[
                        j,
                        round((HIGH - spe.shape[0]) / 2) : round(
                            (HIGH + spe.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = spe
                    x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (0.225**2)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :,
                        r_eeg
                        + round((50 - EEG_LENGTH) / 2 * SFREQ) : r_eeg
                        + round((50 + EEG_LENGTH) / 2 * SFREQ),
                        k,
                    ]

                    x_eeg[j, 1:5, :, k] = eeg
                    x_eeg[j, :, :, k] = (
                        x_eeg[j, :, :, k] - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                    ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    if self.mode == "test":
                        img = self.imgs[row.eeg_id][:, :, k]
                    else:
                        img = self.imgs[row.sign_id][:, :, k]

                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                    x_img[j, :, :, k] = img

                if "stft" in DATATYPE:
                    if self.mode == "test":
                        stft = self.stfts[row.eeg_id][:, :, k]
                    else:
                        stft = self.stfts[row.sign_id][:, :, k]

                    stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                    stft = np.log(stft)
                    stft = np.nan_to_num(stft, nan=0.0)
                    stft = np.round((stft - self.cmin) / (self.cmax - self.cmin) * 255)
                    shape0, shape1 = stft.shape[0], stft.shape[1]
                    stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                    stft = np.array(stft, dtype=np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (shape0, shape1, 3))
                    stft = np.array(
                        tf.image.resize(stft, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

                    x_stft[
                        j,
                        round((HIGH - stft.shape[0]) / 2) : round(
                            (HIGH + stft.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = stft
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test":
                label = row[TARGETS].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0

        x = list()
        if "spe" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xx) + x_spe[::-1, :, :, :, :] * xx
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xx) + x_eeg[::-1, :, :, :] * xx
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)

        if "stft" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_stft.shape[0], 1, 1, 1, 1))
                x_stft = x_stft * (1 - xx) + x_stft[::-1, :, :, :, :] * xx
            x.append(x_stft)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 3
def _effnet_b0_backbone(name: str):
    try:
        return tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=None,
            name=name,
        )
    except Exception as e:
        print(f"EfficientNetB0 imagenet weights unavailable for {name}: {repr(e)}")
        return tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=None,
            name=name,
        )


def build_model():
    l2n = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1, name="cat_spe_chans")(
            [x_spe1, x_spe2, x_spe3, x_spe4]
        )

        base_model_spe = _effnet_b0_backbone(name="efficientnetb0_spe")
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)[:, :128]
        x_spe = l2n(x_spe)

        inp.append(inp_spe)

        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(6, round(EEG_LENGTH * SFREQ), 4), name="inp_eeg"
        )
        x_eeg1 = inp_eeg[:, :, :, 0:1]
        x_eeg2 = inp_eeg[:, :, :, 1:2]
        x_eeg3 = inp_eeg[:, :, :, 2:3]
        x_eeg4 = inp_eeg[:, :, :, 3:4]

        x_eeg11 = tf.keras.layers.Concatenate(axis=1, name="cat_eeg_11")(
            [x_eeg1, x_eeg2 * 0, x_eeg3, x_eeg4 * 0]
        )
        x_eeg22 = tf.keras.layers.Concatenate(axis=1, name="cat_eeg_22")(
            [x_eeg1 * 0, x_eeg2, x_eeg3 * 0, x_eeg4]
        )
        x_eeg33 = (
            tf.keras.layers.Concatenate(axis=1, name="cat_eeg_33")(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )
            * 0
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3, name="cat_eeg_axis3")(
            [x_eeg11, x_eeg22, x_eeg33]
        )

        base_model_eeg = _effnet_b0_backbone(name="efficientnetb0_eeg")
        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)[:, :128]
        x_eeg = l2n(x_eeg)

        inp.append(inp_eeg)

        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1, name="cat_spe_eeg")([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="inp_img")
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1, name="cat_img_chans")(
            [x_img1, x_img2, x_img3, x_img4]
        )
        x_img = tf.keras.layers.Concatenate(axis=3, name="cat_img_rgb")(
            [x_img, x_img, x_img]
        )

        base_model_img = _effnet_b0_backbone(name="efficientnetb0_img")
        base_model_img._name = "img_extractor"

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)[:, :128]
        x_img = l2n(x_img)

        inp.append(inp_img)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="cat_prev_img")([y, x_img])
        else:
            y = x_img

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
        x_stft1 = inp_stft[:, :, :, :, 0]
        x_stft2 = inp_stft[:, :, :, :, 1]
        x_stft3 = inp_stft[:, :, :, :, 2]
        x_stft4 = inp_stft[:, :, :, :, 3]
        x_stft = tf.keras.layers.Concatenate(axis=1, name="cat_stft_chans")(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )

        base_model_stft = _effnet_b0_backbone(name="efficientnetb0_stft")
        base_model_stft._name = "stft_extractor"

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)[
            :, :128
        ]
        x_stft = l2n(x_stft)

        inp.append(inp_stft)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="cat_prev_stft")([y, x_stft])
        else:
            y = x_stft

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32", name="head")(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_multimodal")
    return model




## === cell 4
if not NEEDTRAIN:
    if "spe" in DATATYPE:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
        elif PLATFORM == "kaggle":
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )
        print("Test shape", test.shape)
        test.head()

        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = list()
            list_img = list()
            list_stft = list()
            for region in BRAIN.keys():
                eeg = np.zeros(
                    (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                )
                for chan_i, chan in enumerate(BRAIN[region]):
                    eeg[chan_i, :] = (
                        eeg_default.loc[:, chan.split("-")[0]]
                        - eeg_default.loc[:, chan.split("-")[1]]
                    ).values

                eeg[np.isnan(eeg)] = 0

                if 200 != SFREQ:
                    eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                eeg = signal.filtfilt(b, a, eeg, axis=1)

                time_temp = 0
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE:
                    ff, tt, pp = signal.spectrogram(
                        eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        fs=SFREQ,
                        nperseg=232,
                        noverlap=194,
                    )
                    pp = pp[:, ff <= 40, :]
                    pp = np.mean(pp, 0)
                    list_stft.append(np.reshape(pp, (pp.shape[0], pp.shape[1], 1)))

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if "stft" in DATATYPE:
                list_stft = np.concatenate(list_stft, 2)
                stfts2[name] = list_stft

            if "eeg" in DATATYPE:
                eegs2[name] = list_eeg

            if "img" in DATATYPE:
                eeg_all_region = np.concatenate(list_img, 0)
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 200
                for ii in range(eeg_all_region.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-10, eeg_all_region.shape[1] + 10)
                plt.ylim(-amp / 2, eeg_all_region.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img = Image.open(byte_stream)
                img = np.array(img)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img = np.concatenate((img, img, img), 2)
                img = np.array(
                    tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img = img[:, :, 0:1]

                img = np.concatenate(
                    [
                        img[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img[:, :, 0] = -img[:, :, 0]
                img[:, :, 2] = -img[:, :, 2]

                imgs2[name] = img

    preds = []
    model = build_model()
    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
        stfts=stfts2,
    )

    def _find_weight_files(root_hint: str, stage: int):
        roots = []
        if root_hint:
            roots.append(root_hint)
        roots += ["/kaggle/input", "/kaggle/working"]

        candidates = []
        for r in roots:
            if r and os.path.isdir(r):
                candidates.append(r)

        found = []
        for base in candidates:
            for dirpath, _, filenames in os.walk(base):
                for i in range(5):
                    fname = f"f{i}_stage{stage}.h5"
                    if fname in filenames:
                        found.append((i, os.path.join(dirpath, fname)))

        found = sorted(found, key=lambda t: (t[0], t[1]))
        uniq = {}
        for fold, path in found:
            if fold not in uniq:
                uniq[fold] = path
        return [(f, uniq[f]) for f in sorted(uniq.keys())]

    available_folds = _find_weight_files(LOAD_MODELS_FROM, STAGETEST)
    if len(available_folds) == 0:
        print(f"\nWARNING: No weight files found (hint={LOAD_MODELS_FROM}).")
        print("Writing fallback uniform predictions to submission.csv.")
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        print("Found weight files:")
        for fold, path in available_folds:
            print(f"  fold={fold} path={path}")

        for fold, wpath in available_folds:
            print(f"\nFold {fold+1}")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

    pred = np.asarray(pred, dtype=np.float32)

    eps_smooth = 0.01
    pred = (1.0 - eps_smooth) * pred + eps_smooth * (1.0 / 6.0)

    pred = np.nan_to_num(pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred

    sample = pd.read_csv(SAMPLE_PATH, nrows=1)
    ordered_cols = ["eeg_id"] + [c for c in sample.columns if c != "eeg_id"]
    sub = sub[ordered_cols]

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row prob sum stats:",
        sub.iloc[:, -6:].sum(axis=1).min(),
        sub.iloc[:, -6:].sum(axis=1).max(),
    )
