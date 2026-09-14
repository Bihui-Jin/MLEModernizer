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

0.3500537969034325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow startup crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it triggers the `MessageFactory.GetPrototype` error in Kaggle’s TF stack). Then I make model weight loading robust by auto-detecting the actual weights directory and filenames available under `/kaggle/input` (your configured `models2024022903` path doesn’t exist), while keeping the same model and 5-fold averaging logic. Finally, if no weights are found at all, the script still produce a valid `submission.csv` (uniform probabilities) so you can submit and get a score instead of failing; when weights are found, it use them to move the score down toward the target.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation **before** importing TensorFlow (this is the compatible workaround on Kaggle’s current TF stack). Then we keep your existing model/5-fold ensembling logic intact, but make the weight discovery a bit more robust (also checking nested directories for `.h5` files) so it’s more likely to actually load your trained folds and improve the KL score toward the target. Finally, we ensure the script always writes a valid `submission.csv` with the exact required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it is the direct cause of the `MessageFactory.GetPrototype` error in this environment). Then, to improve the score toward your target (lower is better), I ensure the model actually averages predictions across folds by rebuilding a fresh model per fold before loading weights (otherwise batchnorm/statemomentum and leftover weights can contaminate later folds and hurt KL). Finally, I keep your architecture/data pipeline intact, but make the submission robust: exact required columns, float probabilities, and row-wise normalization to sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf startup crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and `..._VERSION=3`) before importing TensorFlow, which is the stable workaround for Kaggle’s current TF/protobuf mismatch causing `MessageFactory.GetPrototype` errors. I also fix a hard runtime error where `scipy` is imported but not guaranteed installed (per your environment), by replacing the SciPy-based filtering/resampling with a minimal NumPy-only path that still produces correctly-shaped EEG tensors (core model unchanged). Finally, I keep your exact model/5-fold averaging logic and submission formatting, ensuring predictions are finite and row-normalized to sum to 1 so the submission is always valid.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import io
import re
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

PLATFORM = "kaggle"  # local / kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img'
STAGE = 3
LOAD_MODELS_FROM = "models2024022903"

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

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

print("TensorFlow version =", tf.__version__)

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
except Exception as e:
    print("Determinism enabling failed (non-fatal):", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision enabling failed (non-fatal):", repr(e))
else:
    print("Using full precision")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

    if READ_SPEC_FILES * READ_EEG_FILES * READ_IMG_FILES:
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
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

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
                x1 = np.random.rand() * (256 / 2 - 20)
                x2 = np.random.rand() * (256 / 2 - 20)
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_spe_min = x_spe_min + 128
                    x_spe_max = x_spe_max + 128

                x1 = np.random.rand() * (2048 / 2 - 500)
                x2 = np.random.rand() * (2048 / 2 - 500)
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(min(x1, x2))  # keep original logic
                if np.random.rand() < 0.5:
                    x_eeg_min = x_eeg_min + 1024
                    x_eeg_max = x_eeg_max + 1024

                x1 = np.random.rand() * (256 / 2 - 64)
                x2 = np.random.rand() * (256 / 2 - 64)
                x_img_min = round(min(x1, x2))
                x_img_max = round(min(x1, x2))  # keep original logic
                if np.random.rand() < 0.5:
                    x_img_min = x_img_min + 128
                    x_img_max = x_img_max + 128

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
                        max(round((600 / 2 - LENGTH) / 2), 0) : min(
                            (round((600 / 2 - LENGTH) / 2) + LENGTH), spe.shape[1]
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
                    if self.mode == "train":
                        eeg[:, x_eeg_min:x_eeg_max] = 0

                    x_eeg[j, 1:5, :, k] = eeg
                    x_eeg[j, :, :, k] = (
                        x_eeg[j, :, :, k] - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                    ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    if self.mode == "test":
                        img = self.imgs[row.eeg_id][:, :, k]
                    else:
                        img = self.imgs[row.sign_id][:, :, k]
                    if self.mode == "train":
                        if np.random.randn() > 0:
                            img = -img
                        img[:, x_img_min:x_img_max] = 0
                    x_img[j, :, :, k] = img

            if self.mode != "test":
                label = row[TARGETS].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0
        if (self.mode == "train") and (np.random.random() > 0.5):
            alpha = 1.0
            xx = [np.random.beta(alpha, alpha) for _ in range(y.shape[0])]

        x = []
        if "spe" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx_ = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xx_) + x_spe[::-1, :, :, :, :] * xx_
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx_ = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xx_) + x_eeg[::-1, :, :, :] * xx_
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx_ = np.reshape(xx, (x_img.shape[0], 1, 1, 1))
                x_img = x_img * (1 - xx_) + x_img[::-1, :, :, :] * xx_
            x.append(x_img)

        if (self.mode == "train") and (alpha > 0):
            xx_ = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx_) + y[::-1, :] * xx_

        return x, y




## === cell 3
def _effnet_b0_backbone(name: str):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        name=name,
    )
    return base


def build_model():
    l2norm = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    inp = []

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1, name="spe_concat4")(
            [x_spe1, x_spe2, x_spe3, x_spe4]
        )

        base_model_spe = _effnet_b0_backbone("spe_extractor")
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="spe_gap")(x_spe)
        x_spe = l2norm(x_spe)

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
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="eeg_concat4")(
            [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3, name="eeg_to3ch")(
            [x_eeg, x_eeg, x_eeg]
        )

        base_model_eeg = _effnet_b0_backbone("eeg_extractor")
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="eeg_gap")(x_eeg)
        x_eeg = l2norm(x_eeg)

        inp.append(inp_eeg)
        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1, name="spe_eeg_concat")([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="inp_img")
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1, name="img_concat4")(
            [x_img1, x_img2, x_img3, x_img4]
        )
        x_img = tf.keras.layers.Concatenate(axis=3, name="img_to3ch")(
            [x_img, x_img, x_img]
        )

        base_model_img = _effnet_b0_backbone("img_extractor")
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="img_gap")(x_img)
        x_img = l2norm(x_img)

        inp.append(inp_img)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="all_modal_concat")([y, x_img])
        else:
            y = x_img

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32", name="head")(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    return model




## === cell 4
def _find_weights_for_folds(load_root: str, stage: int, n_folds: int = 5):
    """
    Keep same fold-averaging logic, but accept common variants and search
    a few nested levels under /kaggle/input.
    """
    candidates = []

    def _add_dir(p):
        if p and os.path.isdir(p) and p not in candidates:
            candidates.append(p)

    _add_dir(load_root)

    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for d in os.listdir(kaggle_input):
            p = os.path.join(kaggle_input, d)
            _add_dir(p)
            if os.path.isdir(p):
                for dd in os.listdir(p):
                    pp = os.path.join(p, dd)
                    _add_dir(pp)
                    if os.path.isdir(pp):
                        for ddd in os.listdir(pp):
                            ppp = os.path.join(pp, ddd)
                            _add_dir(ppp)

    pattern = re.compile(rf"^f(\d+)_stage{stage}.*\.h5$")
    fold_to_path = {}

    for base in candidates:
        try:
            for fn in os.listdir(base):
                m = pattern.match(fn)
                if m:
                    fi = int(m.group(1))
                    if 0 <= fi < n_folds and fi not in fold_to_path:
                        fold_to_path[fi] = os.path.join(base, fn)
        except Exception:
            continue

    return [fold_to_path.get(i) for i in range(n_folds)]




## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_path = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
        PATH_SPE = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_path = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        PATH_SPE = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    sample = pd.read_csv(sample_path)

    test_unique = test.drop_duplicates("eeg_id").copy()
    test_unique = sample[["eeg_id"]].merge(test_unique, on="eeg_id", how="left")
    if test_unique.isna().any().any():
        missing = test_unique[test_unique["spectrogram_id"].isna()]["eeg_id"].nunique()
        print(
            f"Warning: {missing} eeg_id(s) in sample_submission not found in test.csv"
        )

    print("Original test shape", test.shape)
    print("Sample_submission shape", sample.shape)
    print("Unique-aligned test shape (used for prediction)", test_unique.shape)

    spectrograms2 = None
    if "spe" in DATATYPE:
        files2 = os.listdir(PATH_SPE)
        print(f"There are {len(files2)} test spectrogram parquets")
        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH_SPE}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()

    eegs2 = None
    imgs2 = None
    if ("eeg" in DATATYPE) or ("img" in DATATYPE):
        files2 = os.listdir(PATH_EEG)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {} if "eeg" in DATATYPE else None
        imgs2 = {} if "img" in DATATYPE else None

        aligned_eeg_ids = set(test_unique.eeg_id.values.tolist())

        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in aligned_eeg_ids:
                continue

            eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")

            list_eeg = []
            list_img = []
            for region in BRAIN.keys():
                eeg = np.zeros(
                    (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                )
                for chan_i, chan in enumerate(BRAIN[region]):
                    a, b = chan.split("-")[0], chan.split("-")[1]
                    eeg[chan_i, :] = (
                        eeg_default.loc[:, a] - eeg_default.loc[:, b]
                    ).values

                eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

                if 200 != SFREQ:
                    old_n = eeg.shape[1]
                    new_n = int(round(old_n * (SFREQ / 200.0)))
                    if new_n > 1:
                        x_old = np.linspace(0.0, 1.0, old_n, dtype=np.float32)
                        x_new = np.linspace(0.0, 1.0, new_n, dtype=np.float32)
                        eeg_rs = np.empty((eeg.shape[0], new_n), dtype=np.float32)
                        for ch in range(eeg.shape[0]):
                            eeg_rs[ch] = np.interp(x_new, x_old, eeg[ch]).astype(
                                np.float32
                            )
                        eeg = eeg_rs

                time_temp = 0
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                time_start = max(0, time_start)
                time_stop = min(eeg.shape[1], time_stop)

                if "img" in DATATYPE:
                    list_img.append(eeg[:, time_start:time_stop])
                if "eeg" in DATATYPE:
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            if "eeg" in DATATYPE:
                list_eeg = np.concatenate(list_eeg, 2)
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
        print()

    test_gen = DataGenerator(
        test_unique,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
    )

    weight_paths = _find_weights_for_folds(LOAD_MODELS_FROM, stage=STAGE, n_folds=5)
    print("Requested LOAD_MODELS_FROM:", LOAD_MODELS_FROM)
    print("Detected weight paths per fold:", weight_paths)

    preds = []
    loaded_any = False
    for i in range(5):
        wpath = weight_paths[i]
        if wpath is None or (not os.path.isfile(wpath)):
            print(f"Fold {i+1}: weights not found; skipping.")
            continue

        with strategy.scope():
            model = build_model()

        print(f"Fold {i+1}: loading weights from {wpath}")
        model.load_weights(wpath)

        pred_i = model.predict(test_gen, verbose=1)
        pred_i = np.asarray(pred_i)
        if pred_i.ndim != 2 or pred_i.shape[1] != 6:
            raise ValueError(
                f"Unexpected pred shape from model.predict: {pred_i.shape}"
            )
        if pred_i.shape[0] != len(test_unique):
            raise ValueError(
                f"Prediction length mismatch: got {pred_i.shape[0]} but expected {len(test_unique)}"
            )

        preds.append(pred_i.astype(np.float64))
        loaded_any = True

        del model
        tf.keras.backend.clear_session()

    if loaded_any:
        pred_unique = np.mean(np.stack(preds, axis=0), axis=0)
    else:
        pred_unique = np.full((len(test_unique), 6), 1.0 / 6.0, dtype=np.float64)

    print("Unique preds shape", pred_unique.shape)

    pred_df = pd.DataFrame(pred_unique, columns=list(TARGETS))
    pred_df["eeg_id"] = test_unique["eeg_id"].values
    pred_df = sample[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

    pred = pred_df[list(TARGETS)].to_numpy(dtype=np.float64)

    pred = np.nan_to_num(pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    eps = 1e-12
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = sample.copy()
    sub.loc[:, TARGETS] = pred.astype(np.float32)

    probs = sub[TARGETS].to_numpy(dtype=np.float64)
    probs = np.nan_to_num(probs, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub.loc[:, TARGETS] = probs.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    print("Submission shape", sub.shape)
    print(
        "Row-sum check (min/max):",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).max(),
    )
    print(sub.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_single_column(self, loc, value, plane_indexer)
   2132             try:
-> 2133                 self.obj._mgr.column_setitem(
   2134                     loc, plane_indexer, value, inplace_only=True

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in column_setitem(self, loc, idx, value, inplace_only)
   1334         if inplace_only:
-> 1335             col_mgr.setitem_inplace(idx, value)
   1336         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in setitem_inplace(self, indexer, value, warn)
   2043 
-> 2044         super().setitem_inplace(indexer, value)
   2045 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in setitem_inplace(self, indexer, value, warn)
    362 
--> 363         arr[indexer] = value
    364 

ValueError: could not broadcast input array from shape (778942,) into shape (9850,)

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/547824692.py in <cell line: 0>()
    217 
    218     sub = sample.copy()
--> 219     sub.loc[:, TARGETS] = pred.astype(np.float32)
    220 
    221     probs = sub[TARGETS].to_numpy(dtype=np.float64)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __setitem__(self, key, value)
    909 
    910         iloc = self if self.name == "iloc" else self.obj.iloc
--> 911         iloc._setitem_with_indexer(indexer, value, self.name)
    912 
    913     def _validate_key(self, key, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer(self, indexer, value, name)
   1940         if take_split_path:
   1941             # We have to operate column-wise
-> 1942             self._setitem_with_indexer_split_path(indexer, value, name)
   1943         else:
   1944             self._setitem_single_block(indexer, value, name)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer_split_path(self, indexer, value, name)
   1980                 # TODO: avoid np.ndim call in case it isn't an ndarray, since
   1981                 #  that will construct an ndarray, which will be wasteful
-> 1982                 self._setitem_with_indexer_2d_value(indexer, value)
   1983 
   1984             elif len(ilocs) == 1 and lplane_indexer == len(value) and not is_scalar(pi):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer_2d_value(self, indexer, value)
   2055                 # casting to list so that we do type inference in setitem_single_column
   2056                 value_col = value_col.tolist()
-> 2057             self._setitem_single_column(loc, value_col, pi)
   2058 
   2059     def _setitem_with_indexer_frame_value(self, indexer, value: DataFrame, name: str):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_single_column(self, loc, value, plane_indexer)
   2158                         stacklevel=find_stack_level(),
   2159                     )
-> 2160                 self.obj.isetitem(loc, value)
   2161         else:
   2162             # set value into the column (first attempting to operate inplace, then

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in isetitem(self, loc, value)
   4266             return
   4267 
-> 4268         arraylike, refs = self._sanitize_column(value)
   4269         self._iset_item_mgr(loc, arraylike, inplace=False, refs=refs)
   4270 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (778942) does not match length of index (9850)
