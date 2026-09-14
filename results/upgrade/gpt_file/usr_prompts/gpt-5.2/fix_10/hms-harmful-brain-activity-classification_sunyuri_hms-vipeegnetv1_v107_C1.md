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

0.3573726374966366

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that breaks TF in this environment, while keeping determinism/mixed-precision behavior intact. Then I make the inference robust to missing external weight files by falling back to a safe uniform-probability submission when weights are not found, ensuring a valid `submission.csv` is always produced. I also fix the off-by-one cell numbering to match the required format and keep paths unchanged for Kaggle. These changes are score-neutral when weights exist (normal inference runs), and they at least yield a valid submission when they don’t.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by ensuring protobuf uses the C++ implementation (and not a broken python fallback) before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I keep your inference logic intact, but add a safe probability calibration step (simple blend with uniform) that nudges predictions away from overconfident spikes; this typically improves KL-divergence without changing the model or training. Finally, I make sure the script always writes a valid `submission.csv` with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by removing the forced protobuf “cpp” override (it breaks in this environment) and instead forcing the safe pure-Python protobuf implementation before importing TensorFlow. This unblocks the downstream `tf` usage so `DataGenerator`, image resizing, model build, and prediction run end-to-end. I also fix the off-by-one cell numbering (your script starts at cell 0) so it matches the required format, while keeping all core model/inference logic unchanged. Finally, the script always write a valid `submission.csv` with correct columns and row-wise probabilities summing to 1 (uniform fallback remains if weights are absent).'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"



## === cell 1
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024030501"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 256  # 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 42

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



## === cell 2
import io
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

print("TensorFlow version =", tf.__version__)

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
    print("enable_op_determinism not available:", repr(e))

MIX = True
if MIX:
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision policy set to mixed_float16")
    except Exception as e:
        print("Could not set mixed precision policy:", repr(e))
        print("Using full precision")
else:
    print("Using full precision")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## === cell 4
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



## === cell 5
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
                x1 = np.random.rand() * LENGTH / 2
                x2 = np.random.rand() * LENGTH / 2
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))

                spe_remove = list()
                if np.random.rand() < 0.5:
                    spe_remove.append(0)
                else:
                    spe_remove.append(2)
                if np.random.rand() < 0.5:
                    spe_remove.append(1)
                else:
                    spe_remove.append(3)

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
                        if np.random.rand() > 0.5:
                            if k in spe_remove:
                                spe = spe * 0

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

        x = list()
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "img" in DATATYPE:
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)
        if "stft" in DATATYPE:
            x.append(x_stft)

        return x, y




## === cell 6
def build_model():
    def l2norm_layer():
        return tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))

    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])

        base_model_spe = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_spe"
        )
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = l2norm_layer()(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))
        x_eeg1 = inp_eeg[:, :, :, 0:1]
        x_eeg2 = inp_eeg[:, :, :, 1:2]
        x_eeg3 = inp_eeg[:, :, :, 2:3]
        x_eeg4 = inp_eeg[:, :, :, 3:4]

        x_eeg = tf.keras.layers.Concatenate(axis=1)([x_eeg1, x_eeg2, x_eeg3, x_eeg4])
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_eeg"
        )
        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = l2norm_layer()(x_eeg)

        inp.append(inp_eeg)

        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1)([x_img1, x_img2, x_img3, x_img4])
        x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_img"
        )
        base_model_img._name = "img_extractor"

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = l2norm_layer()(x_img)

        inp.append(inp_img)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
        else:
            y = x_img

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_stft1 = inp_stft[:, :, :, :, 0]
        x_stft2 = inp_stft[:, :, :, :, 1]
        x_stft3 = inp_stft[:, :, :, :, 2]
        x_stft4 = inp_stft[:, :, :, :, 3]
        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )

        base_model_stft = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=None,
            name="efficientnetb0_stft",
        )
        base_model_stft._name = "stft_extractor"

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = l2norm_layer()(x_stft)

        inp.append(inp_stft)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
        else:
            y = x_stft

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 7
try:
    from scipy import signal as sp_signal  # type: ignore

    HAVE_SCIPY = True
    print("Using SciPy signal backend")
except Exception as e:
    HAVE_SCIPY = False
    sp_signal = None
    print("SciPy not available, using NumPy fallbacks:", repr(e))


def _resample_poly_fallback(x, up, down, axis=1):
    if up == down:
        return x
    n = x.shape[axis]
    new_n = int(round(n * up / down))
    old_idx = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, new_n, endpoint=False, dtype=np.float32)
    x_move = np.moveaxis(x, axis, -1)
    y_move = np.empty(x_move.shape[:-1] + (new_n,), dtype=np.float32)
    flat = x_move.reshape(-1, n)
    out_flat = y_move.reshape(-1, new_n)
    for i in range(flat.shape[0]):
        out_flat[i] = np.interp(new_idx, old_idx, flat[i]).astype(
            np.float32, copy=False
        )
    y_move = out_flat.reshape(y_move.shape)
    return np.moveaxis(y_move, -1, axis)


def _bandpass_fallback(x, low, high, fs, axis=1):
    x_move = np.moveaxis(x, axis, -1)
    n = x_move.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x_move, axis=-1)
    mask = (freqs >= low) & (freqs <= high)
    X *= mask.astype(X.dtype)
    y = np.fft.irfft(X, n=n, axis=-1).astype(np.float32, copy=False)
    return np.moveaxis(y, -1, axis)


def _spectrogram_fallback(x, fs, nperseg, noverlap):
    step = nperseg - noverlap
    n = x.shape[-1]
    if n < nperseg:
        pad = nperseg - n
        x = np.pad(x, [(0, 0), (0, pad)], mode="constant")
        n = x.shape[-1]
    nwin = 1 + (n - nperseg) // step
    win = np.hanning(nperseg).astype(np.float32)
    ff = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    tt = (np.arange(nwin) * step + nperseg / 2) / fs
    pp = np.empty((x.shape[0], ff.shape[0], nwin), dtype=np.float32)
    for i in range(nwin):
        seg = x[:, i * step : i * step + nperseg] * win[None, :]
        S = np.fft.rfft(seg, axis=-1)
        pp[:, :, i] = (np.abs(S) ** 2).astype(np.float32, copy=False)
    return ff, tt, pp


def butter_bandpass_coeffs(order, low, high, fs):
    if HAVE_SCIPY:
        b, a = sp_signal.butter(order, np.float32([low, high]) * 2 / fs, "bandpass")
        return b, a
    return None, None


def apply_bandpass(x, b, a, low, high, fs, axis=1):
    if HAVE_SCIPY:
        return sp_signal.filtfilt(b, a, x, axis=axis).astype(np.float32, copy=False)
    return _bandpass_fallback(x, low, high, fs, axis=axis)


def resample_if_needed(x, target_fs, orig_fs=200, axis=1):
    if orig_fs == target_fs:
        return x
    if HAVE_SCIPY:
        return sp_signal.resample_poly(x, target_fs, orig_fs, axis=axis).astype(
            np.float32, copy=False
        )
    return _resample_poly_fallback(x, target_fs, orig_fs, axis=axis)


def compute_spectrogram(x, fs, nperseg=232, noverlap=194):
    if HAVE_SCIPY:
        return sp_signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return _spectrogram_fallback(x, fs=fs, nperseg=nperseg, noverlap=noverlap)




## === cell 8
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

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}

    b, a = butter_bandpass_coeffs(3, filter_range[0], filter_range[1], SFREQ)

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
                    eeg = resample_if_needed(eeg, SFREQ, orig_fs=200, axis=1)

                eeg = apply_bandpass(
                    eeg, b, a, filter_range[0], filter_range[1], SFREQ, axis=1
                )

                time_temp = 0
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE:
                    ff, tt, pp = compute_spectrogram(
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

    expected_any_weights = False
    for i in range(5):
        w1 = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
        w2 = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.weights.h5")
        if os.path.exists(w1) or os.path.exists(w2):
            expected_any_weights = True
            break

    if not expected_any_weights:
        print(f"\nWARNING: No weights found under {LOAD_MODELS_FROM}.")
        print(
            "Falling back to uniform probabilities to produce a valid submission.csv."
        )
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
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

        for i in range(5):
            print(f"\nFold {i+1}")
            w1 = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            w2 = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.weights.h5")
            if os.path.exists(w1):
                model.load_weights(w1)
            elif os.path.exists(w2):
                model.load_weights(w2)
            else:
                raise FileNotFoundError(f"Could not find weights: {w1} or {w2}")

            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print()
        print("Test preds shape", pred.shape)

    eps = 1e-7
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    alpha = 0.20
    pred = (1.0 - alpha) * pred + alpha * (1.0 / 6.0)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    if PLATFORM == "local":
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    sub = sample_sub.copy()
    pred_df = pd.DataFrame(pred, columns=list(TARGETS))
    pred_df["eeg_id"] = test["eeg_id"].values
    sub = sub.merge(pred_df, on="eeg_id", how="left", suffixes=("", "_pred"))

    for c in TARGETS:
        if c not in sub.columns:
            raise RuntimeError(f"Missing target column in submission: {c}")
        sub[c] = sub[c].fillna(1.0 / 6.0).astype(np.float32)

    pred_arr = sub[list(TARGETS)].to_numpy(np.float32)
    pred_arr = np.clip(pred_arr, eps, 1.0)
    pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)
    sub[list(TARGETS)] = pred_arr

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-sum stats:",
        sub.iloc[:, -6:].sum(axis=1).min(),
        sub.iloc[:, -6:].sum(axis=1).max(),
    )

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
