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

0.468439133027326

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (and stop removing it), which resolves the `MessageFactory.GetPrototype` error on Kaggle’s Py3.12 environments. Then I fix the missing-weights runtime failure by making inference robust: if the expected `models20240220/*.h5` files are not present, the code fall back to a valid probability baseline (uniform over classes) so it still produces a correct `submission.csv`. I also fix a logic bug in test mode of `DataGenerator` (it incorrectly reads `row.spec_id` before renaming/ensuring the column exists) by consistently using the renamed `spec_id` field. These changes are minimal, keep the model code intact, and guarantee a valid submission CSV with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting both `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` *before* importing TensorFlow, which is the reliable workaround on Kaggle Py3.12 for the `MessageFactory.GetPrototype` error. Then, because your current score (1.40995, lower-is-better) indicates you’re submitting the uniform fallback, I make the code actually find and load the provided 5 fold weight files by searching common Kaggle input/working locations (while keeping the same model, data pipeline, and inference logic). I keep the uniform fallback as a safety net, but only use it if weights truly can’t be found. Finally, I keep the existing probability clipping/renormalization to ensure valid KL-divergence submissions (rows sum to 1, no zeros).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf environment variables are set before any TensorFlow-related import and by adding a safe fallback that retries TensorFlow import with those settings if Kaggle’s runtime still hits the `MessageFactory.GetPrototype` issue. Then I keep your model/data logic intact but make weight discovery more robust so it actually loads the provided fold `.h5` files instead of falling back to uniform predictions (which is what produced the poor 1.40995 score). Finally, I keep the existing probability clipping/renormalization so the submission is always valid for KL divergence (no zeros and each row sums to 1), and always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except AttributeError as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    import importlib

    tf = importlib.import_module("tensorflow")

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetB0

from scipy import signal
from scipy.signal import spectrogram

print("TensorFlow version =", tf.__version__)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240220"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    cand1 = f"/kaggle/input/{LOAD_MODELS_FROM}"
    cand2 = f"/kaggle/working/{LOAD_MODELS_FROM}"
    LOAD_MODELS_FROM = cand1 if os.path.exists(cand1) else cand2

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 64
LENGTH = 256

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision option not available:", repr(e))
else:
    print("Using full precision")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")

    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()

if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                list_eeg = []
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
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs[name] = list_eeg

        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()



## === cell 2
try:
    import albumentations as albu
except Exception:
    albu = None

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
        df=None,
    ):
        if mode != "test":
            self.df = df.merge(data.iloc[:, :6], on="eeg_id", how="inner").reset_index(
                drop=True
            )

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
        self.b, self.a = signal.butter(
            3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_spe, X_eeg, y = self.__data_generation(indexes)
        return [X, X_spe, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        X_spe = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            if self.mode == "test":
                row = self.data.iloc[i]
                r_img = 0
                r_eeg = 0
                spec_id = int(row["spec_id"])
            elif self.mode == "valid":
                row = self.data.iloc[i]
                rows = self.df[
                    (self.df.eeg_id == row.eeg_id)
                    * (self.df.spectrogram_id == row.spec_id)
                ].reset_index(drop=True)
                r_img = round((max(rows["max"]) + min(rows["min"])) // 4)
                r_eeg = round(row.eeg_median)
                label = row[TARGETS].values
                spec_id = row.spec_id
                label = label / sum(label)
                if sum(label == 1):
                    label[label == 0] = 1e-2
                    label[label == 1] = 1 - 5 * 1e-2
            else:
                row = self.data.iloc[i]
                rows = self.df[
                    (self.df.eeg_id == row.eeg_id)
                    * (self.df.spectrogram_id == row.spec_id)
                ].reset_index(drop=True)
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row = rows.iloc[0]
                r_img = round(row.spectrogram_label_offset_seconds // 2)
                r_eeg = round(row.eeg_label_offset_seconds)
                label = row[TARGETS].values
                spec_id = row.spectrogram_id
                label = label / sum(label)
                if sum(label == 1):
                    label[label == 0] = 1e-2
                    label[label == 1] = 1 - 5 * 1e-2

            for k in range(4):
                img = self.specs[spec_id][
                    r_img : r_img + 300, k * 100 : (k + 1) * 100
                ].T
                img = np.nan_to_num(img, nan=0.0)
                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)

                img_eeg = self.eegs[int(row.eeg_id)][
                    :, round(r_eeg * SFREQ) : round(r_eeg * SFREQ + 50 * SFREQ), k
                ]
                img_eeg = np.nan_to_num(img_eeg, nan=0.0)
                img_eeg = signal.filtfilt(self.b, self.a, img_eeg, axis=1)
                img_eeg = img_eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)
                img_map = self.cmaps[img - 1]
                img_map = np.reshape(img_map, (100, 300, 3))

                img_map = img_map[
                    :,
                    max(round((300 - LENGTH) / 2), 0) : min(
                        (round((300 - LENGTH) / 2) + LENGTH), img_map.shape[1]
                    ),
                    :,
                ]

                if self.mode == "train":
                    img_map = img_map[
                        :,
                        round(np.random.random() * img_map.shape[1] / 3) : -max(
                            1, round(np.random.random() * img_map.shape[1] / 3)
                        ),
                        :,
                    ]

                img_map = np.array(
                    tf.image.resize(img_map, ((HIGH - 16), LENGTH)), dtype=np.float32
                )
                img_map[:, :, 0] = (img_map[:, :, 0] - 0.485) / (0.229**2)
                img_map[:, :, 1] = (img_map[:, :, 1] - 0.456) / (0.224**2)
                img_map[:, :, 2] = (img_map[:, :, 2] - 0.406) / (0.225**2)

                X[
                    j,
                    round((HIGH - img_map.shape[0]) / 2) : round(
                        (HIGH + img_map.shape[0]) / 2
                    ),
                    :,
                    :,
                    k,
                ] = img_map

                for eeg_i in range(img_eeg.shape[0]):
                    freqs, t, spectrum = spectrogram(
                        img_eeg[eeg_i, :], SFREQ, nfft=256, nperseg=128, noverlap=112
                    )
                    spectrum = spectrum[freqs <= 40, :]
                    if eeg_i == 0:
                        spec = spectrum.copy()
                    else:
                        spec = spec + spectrum
                spec = spec / img_eeg.shape[0]
                cmin = -10
                cmax = 10
                spec = np.clip(spec, np.exp(cmin), np.exp(cmax))
                spec = np.log(spec)

                spec = np.round((spec - cmin) / (cmax - cmin) * 256)
                spec = np.reshape(spec, (spec.shape[0] * spec.shape[1]))
                spec = np.array(spec, dtype=np.int16)
                spec_map = self.cmaps[spec - 1]
                spec_map = np.reshape(spec_map, (sum(freqs <= 40), len(t), 3))

                if self.mode == "train":
                    spec_map = spec_map[
                        :,
                        round(np.random.random() * spec_map.shape[1] / 3) : -max(
                            1, round(np.random.random() * spec_map.shape[1] / 3)
                        ),
                        :,
                    ]

                spec_map = np.array(
                    tf.image.resize(spec_map, ((HIGH - 16), LENGTH)), dtype=np.float32
                )
                spec_map[:, :, 0] = (spec_map[:, :, 0] - 0.485) / (0.229**2)
                spec_map[:, :, 1] = (spec_map[:, :, 1] - 0.456) / (0.224**2)
                spec_map[:, :, 2] = (spec_map[:, :, 2] - 0.406) / (0.225**2)

                X_spe[
                    j,
                    round((HIGH - spec_map.shape[0]) / 2) : round(
                        (HIGH + spec_map.shape[0]) / 2
                    ),
                    :,
                    :,
                    k,
                ] = spec_map

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = label

        if self.mode == "train":
            for per_i in range(X.shape[0]):
                if np.random.random() > 0.5:
                    xx = X[per_i, :, :, :, :].copy()
                    X[per_i, :, :, :, 0] = xx[:, :, :, 2]
                    X[per_i, :, :, :, 2] = xx[:, :, :, 0]

                if np.random.random() > 0.5:
                    xx = X[per_i, :, :, :, :].copy()
                    X[per_i, :, :, :, 1] = xx[:, :, :, 3]
                    X[per_i, :, :, :, 3] = xx[:, :, :, 1]

                if np.random.random() > 0.5:
                    xx = X_spe[per_i, :, :, :, :].copy()
                    X_spe[per_i, :, :, :, 1] = xx[:, :, :, 3]
                    X_spe[per_i, :, :, :, 3] = xx[:, :, :, 1]

                if np.random.random() > 0.5:
                    xx = X_spe[per_i, :, :, :, :].copy()
                    X_spe[per_i, :, :, :, 1] = xx[:, :, :, 3]
                    X_spe[per_i, :, :, :, 3] = xx[:, :, :, 1]

                if np.random.random() > 0.5:
                    xx = X_eeg[per_i, :, :, :].copy()
                    X_eeg[per_i, :, :, 0] = xx[:, :, 2]
                    X_eeg[per_i, :, :, 2] = xx[:, :, 0]

                if np.random.random() > 0.5:
                    xx = X_eeg[per_i, :, :, :].copy()
                    X_eeg[per_i, :, :, 1] = xx[:, :, 3]
                    X_eeg[per_i, :, :, 3] = xx[:, :, 1]

                for zo_i in range(X.shape[-1]):
                    if np.random.random() > 0.5:
                        X[per_i, :, :, :, zo_i] = X[per_i, :, ::-1, :, zo_i]

                    if np.random.random() > 0.5:
                        X_spe[per_i, :, :, :, zo_i] = X_spe[per_i, :, ::-1, :, zo_i]

                    if np.random.random() > 0.5:
                        X_eeg[per_i, :, :, zo_i] = X_eeg[per_i, :, ::-1, zo_i]
                    if np.random.random() > 0.5:
                        X_eeg[per_i, :, :, zo_i] = X_eeg[per_i, ::-1, :, zo_i]

        return X, X_spe, X_eeg, y




## === cell 3
def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="tanh",
            dilation_rate=dilation_rate,
        )(x)
        sigm_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            dilation_rate=dilation_rate,
        )(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="spectrogram_extractor"
    )

    x_inp = tf.keras.layers.Concatenate(axis=2)([inp, inp_spe])
    x0 = x_inp[:, :, :, :, 0]
    x1 = x_inp[:, :, :, :, 1]
    x2 = x_inp[:, :, :, :, 2]
    x3 = x_inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="eeg_extractor"
    )

    x0_eeg = inp_eeg[:, :, :, :1]
    x1_eeg = inp_eeg[:, :, :, 1:2]
    x2_eeg = inp_eeg[:, :, :, 2:3]
    x3_eeg = inp_eeg[:, :, :, 3:4]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])
    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_spe, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 4
def _find_weight_paths(load_models_from: str, ver: int):
    """
    Robustness + score improvement: find the provided fold weight files so we don't fall back to uniform.
    Searches common Kaggle input/working locations and nested dataset directories.
    """
    expected_files = [f"EB2_v{ver}_f{i}.h5" for i in range(5)]

    roots = []
    if load_models_from:
        roots.append(load_models_from)
    roots.extend(["/kaggle/input", "/kaggle/working"])

    for root in roots:
        if root and os.path.isdir(root):
            direct = [os.path.join(root, f) for f in expected_files]
            if all(os.path.exists(p) for p in direct):
                return direct

    for root in roots:
        if not (root and os.path.isdir(root)):
            continue
        try:
            for d in os.listdir(root):
                sub = os.path.join(root, d)
                if not os.path.isdir(sub):
                    continue

                if load_models_from:
                    sub2 = os.path.join(sub, os.path.basename(load_models_from))
                    if os.path.isdir(sub2):
                        cand = [os.path.join(sub2, f) for f in expected_files]
                        if all(os.path.exists(p) for p in cand):
                            return cand

                cand = [os.path.join(sub, f) for f in expected_files]
                if all(os.path.exists(p) for p in cand):
                    return cand
        except Exception:
            pass

    return None


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = []
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
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            eegs2[name] = list_eeg

    weight_paths = _find_weight_paths(LOAD_MODELS_FROM, VER)
    have_all_weights = weight_paths is not None

    if not have_all_weights:
        print("\nWARNING: Expected model weights not found in common Kaggle locations.")
        print(
            "LOAD_MODELS_FROM =",
            LOAD_MODELS_FROM,
            "exists =",
            os.path.exists(LOAD_MODELS_FROM),
        )
        print("Falling back to uniform probabilities to create a valid submission.csv")
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        print("\nFound weights:")
        for p in weight_paths:
            print(" -", p)

        preds = []
        with strategy.scope():
            model = build_model()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        for i in range(5):
            print(f"\nFold {i+1}")
            wpath = weight_paths[i]
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row sums min/max:",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).max(),
    )
    print(sub.head())
