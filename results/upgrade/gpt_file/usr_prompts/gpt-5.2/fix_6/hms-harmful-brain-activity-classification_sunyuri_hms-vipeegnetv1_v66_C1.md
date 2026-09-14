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

0.5383913144990755

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'We fix two execution blockers while keeping the modeling/prediction logic intact: (1) the protobuf `MessageFactory.GetPrototype` crash is triggered by forcing the pure-Python protobuf implementation, so we remove that environment override; (2) the pretrained fold weight paths don’t exist in this Kaggle dataset, so we automatically fall back to generating a valid submission using a safe uniform probability distribution when weights are missing (rather than crashing). We also make the code robust to either `/kaggle/input/...` or `/kaggle/data/...` dataset roots, and keep the submission format/row-sum constraints enforced. These changes are score-neutral in the “weights missing” case (the only way to produce a submission) and preserve the original ensemble inference when weights are available.'

# 9. Code solution

## === cell 0
import os
import glob

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from scipy import signal

print("TensorFlow version =", tf.__version__)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402182"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 128
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
        print("Mixed precision option not applied:", repr(e))
else:
    print("Using full precision")


def _first_existing_path(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


TRAIN_CSV = _first_existing_path(
    [
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
        "/kaggle/data/hms-harmful-brain-activity-classification/train.csv",
        "./input/hms-harmful-brain-activity-classification/train.csv",
    ]
)
if TRAIN_CSV is None:
    raise FileNotFoundError("Could not locate train.csv in expected Kaggle paths.")

df = pd.read_csv(TRAIN_CSV)

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

_global_prior = df[TARGETS].sum(axis=0).astype(np.float64).values
_global_prior = _global_prior / _global_prior.sum()
_global_prior = np.clip(_global_prior, 1e-12, 1.0)
_global_prior = (_global_prior / _global_prior.sum()).astype(np.float32)
print("Global class prior:", dict(zip(TARGETS, _global_prior.round(6))))




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
                list_eeg = list()
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
    import albumentations as albu  # noqa: F401
except Exception as e:
    albu = None
    print("albumentations not available (not required):", repr(e))

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

        self.data = data.reset_index(drop=True)
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.specs = specs
        self.eegs = eegs

        if mode != "test":
            self.df = df.merge(
                self.data.iloc[:, :6], on="eeg_id", how="inner"
            ).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]

        self.b, self.a = signal.butter(
            3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            if self.mode == "test":
                row1 = self.data.iloc[i]
                rows = self.df[self.df.patient_id == row1.patient_id].reset_index(
                    drop=True
                )
                if len(rows) == 0:
                    rows = self.df.sample(n=1, random_state=0).reset_index(drop=True)
                else:
                    rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )
                row2 = rows.iloc[0]
            else:
                if j < (self.batch_size / 6 * 1):
                    target = "Seizure"
                elif j < (self.batch_size / 6 * 2):
                    target = "GPD"
                elif j < (self.batch_size / 6 * 3):
                    target = "LRDA"
                elif j < (self.batch_size / 6 * 4):
                    target = "Other"
                elif j < (self.batch_size / 6 * 5):
                    target = "GRDA"
                else:
                    target = "LPD"

                rows = self.df[self.df.expert_consensus == target].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row1 = rows.iloc[0]

                rows = self.df[self.df.patient_id_x == row1.patient_id_x].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row2 = rows.iloc[0]

                mixup1 = 0.5
                mixup2 = 1 - mixup1
                label = (
                    row1[TARGETS].values / sum(row1[TARGETS].values) * mixup1
                    + row2[TARGETS].values / sum(row2[TARGETS].values) * mixup2
                )

            for k in range(4):
                if self.mode == "test":
                    img1 = self.specs[row1.spec_id][
                        0 : 0 + 300, k * 100 : (k + 1) * 100
                    ].T
                    img1 = np.nan_to_num(img1, nan=0.0)
                    img1 = np.clip(img1, np.exp(self.cmin), np.exp(self.cmax))
                    img1 = np.log(img1)

                    img2 = self.specs[row2.spectrogram_id][
                        round(row2.spectrogram_label_offset_seconds / 2) : round(
                            row2.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img2 = np.nan_to_num(img2, nan=0.0)
                    img2 = np.clip(img2, np.exp(self.cmin), np.exp(self.cmax))
                    img2 = np.log(img2)

                    img_eeg1 = self.eegs[row1.eeg_id][:, :, k]
                    img_eeg1 = np.nan_to_num(img_eeg1, nan=0.0)

                    img_eeg2 = self.eegs[row2.eeg_id][:, :, k]
                    img_eeg2 = np.nan_to_num(img_eeg2, nan=0.0)

                    img = img1 * 0.5 + img2 * 0.5
                    img_eeg = img_eeg1 * 0.5 + img_eeg2 * 0.5
                else:
                    img1 = self.specs[row1.spectrogram_id][
                        round(row1.spectrogram_label_offset_seconds / 2) : round(
                            row1.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img2 = self.specs[row2.spectrogram_id][
                        round(row2.spectrogram_label_offset_seconds / 2) : round(
                            row2.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img_eeg1 = self.eegs[row1.eeg_id][
                        :,
                        round(row1.eeg_label_offset_seconds * SFREQ) : round(
                            row1.eeg_label_offset_seconds * SFREQ + 50 * SFREQ
                        ),
                        k,
                    ]
                    img_eeg2 = self.eegs[row2.eeg_id][
                        :,
                        round(row2.eeg_label_offset_seconds * SFREQ) : round(
                            row2.eeg_label_offset_seconds * SFREQ + 50 * SFREQ
                        ),
                        k,
                    ]

                    img1 = np.nan_to_num(img1, nan=0.0)
                    img1 = np.clip(img1, np.exp(self.cmin), np.exp(self.cmax))
                    img1 = np.log(img1)
                    img2 = np.nan_to_num(img2, nan=0.0)
                    img2 = np.clip(img2, np.exp(self.cmin), np.exp(self.cmax))
                    img2 = np.log(img2)
                    img = img1 * mixup1 + img2 * mixup2

                    img_eeg1 = np.nan_to_num(img_eeg1, nan=0.0)
                    img_eeg2 = np.nan_to_num(img_eeg2, nan=0.0)
                    img_eeg = img_eeg1 * mixup1 + img_eeg2 * mixup2

                img = img[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img.shape[1]
                    ),
                ]
                if HIGH != 100:
                    img_r = np.array(
                        tf.image.resize(
                            np.reshape(img, (img.shape[0], img.shape[1], 1)),
                            ((HIGH - 32), LENGTH),
                        ),
                        dtype=np.float32,
                    )
                    img_r = img_r[:, :, 0]
                    X[
                        j,
                        round((HIGH - img_r.shape[0]) / 2) : round(
                            (HIGH + img_r.shape[0]) / 2
                        ),
                        :,
                        k,
                    ] = img_r
                else:
                    X[j, :, :, k] = img

                X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                    np.std(X[j, :, :, k]) + 1e-6
                )

                img_eeg = signal.filtfilt(self.b, self.a, img_eeg, axis=1)
                img_eeg = img_eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = label

                r = np.random.permutation(X.shape[0])
                X = X[r]
                X_eeg = X_eeg[r]
                y = y[r]

        return X, X_eeg, y




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
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=None,
        name="spectrogram_efficientnetb0",
    )
    base_model._name = "spectrogram_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0 = inp[:, :, :, :1]
    x1 = inp[:, :, :, 1:2]
    x2 = inp[:, :, :, 2:3]
    x3 = inp[:, :, :, 3:4]
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])
    x = tf.keras.layers.Concatenate(axis=3)([x, x, x])
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="eeg_efficientnetb0"
    )
    base_model_eeg._name = "eeg_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model_eeg.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model_eeg.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
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

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 4
if not NEEDTRAIN:
    TEST_CSV = _first_existing_path(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
            "/kaggle/data/hms-harmful-brain-activity-classification/test.csv",
            "./input/hms-harmful-brain-activity-classification/test.csv",
        ]
    )
    if TEST_CSV is None:
        raise FileNotFoundError("Could not locate test.csv in expected Kaggle paths.")

    test = pd.read_csv(TEST_CSV)

    train_df_for_mix = df.copy()

    print("Test shape", test.shape)

    SPEC_DIR = _first_existing_path(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/",
            "/kaggle/data/hms-harmful-brain-activity-classification/test_spectrograms/",
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/",
        ]
    )
    EEG_DIR = _first_existing_path(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/",
            "/kaggle/data/hms-harmful-brain-activity-classification/test_eegs/",
            "./input/hms-harmful-brain-activity-classification/test_eegs/",
        ]
    )
    if SPEC_DIR is None or EEG_DIR is None:
        raise FileNotFoundError(
            "Could not locate test_spectrograms/ or test_eegs/ dirs."
        )

    files2 = sorted(os.listdir(SPEC_DIR))
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(os.path.join(SPEC_DIR, f))
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    files2 = sorted(os.listdir(EEG_DIR))
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(os.path.join(EEG_DIR, f))
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = list()
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
    print()

    candidate_weight_dirs = [
        LOAD_MODELS_FROM,
        f"/kaggle/input/{os.path.basename(LOAD_MODELS_FROM)}",
        f"/kaggle/data/{os.path.basename(LOAD_MODELS_FROM)}",
    ]
    LOAD_MODELS_FROM_RESOLVED = (
        _first_existing_path(candidate_weight_dirs) or LOAD_MODELS_FROM
    )
    if LOAD_MODELS_FROM_RESOLVED != LOAD_MODELS_FROM:
        print("Resolved model dir:", LOAD_MODELS_FROM_RESOLVED)

    weight_files = [
        os.path.join(LOAD_MODELS_FROM_RESOLVED, f"EB2_v{VER}_f{i}.h5") for i in range(5)
    ]
    available = [wf for wf in weight_files if os.path.exists(wf)]
    if len(available) == 0:
        print(f"WARNING: No fold weights found under: {LOAD_MODELS_FROM_RESOLVED}")
        print(
            "Falling back to global class-prior probabilities (improves over uniform)."
        )
        pred = np.tile(_global_prior[None, :], (len(test), 1)).astype(np.float32)
    else:
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
            df=train_df_for_mix,
        )

        for wf in weight_files:
            if not os.path.exists(wf):
                print(f"Missing fold weights (skipping): {wf}")
                continue
            print(f"Loading weights: {wf}")
            model.load_weights(wf)
            fold_pred = model.predict(test_gen, verbose=1)
            preds.append(fold_pred)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-12, 1.0)
    sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].sum(axis=1, keepdims=True)
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row-sum check (min/mean/max):",
        float(sub[TARGETS].sum(axis=1).min()),
        float(sub[TARGETS].sum(axis=1).mean()),
        float(sub[TARGETS].sum(axis=1).max()),
    )
    print("Wrote: submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/552629598.py in <cell line: 0>()
    137     sub[TARGETS] = pred
    138     sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-12, 1.0)
--> 139     sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].sum(axis=1, keepdims=True)
    140     sub.to_csv("submission.csv", index=False)
    141     print("Submission shape", sub.shape)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in sum(self, axis, skipna, numeric_only, min_count, **kwargs)
  11668         **kwargs,
  11669     ):
> 11670         result = super().sum(axis, skipna, numeric_only, min_count, **kwargs)
  11671         return result.__finalize__(self, method="sum")
  11672 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in sum(self, axis, skipna, numeric_only, min_count, **kwargs)
  12504         **kwargs,
  12505     ):
> 12506         return self._min_count_stat_function(
  12507             "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
  12508         )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _min_count_stat_function(self, name, func, axis, skipna, numeric_only, min_count, **kwargs)
  12469     ):
  12470         assert name in ["sum", "prod"], name
> 12471         nv.validate_func(name, (), kwargs)
  12472 
  12473         validate_bool_kwarg(skipna, "skipna", none_allowed=False)

/usr/local/lib/python3.11/dist-packages/pandas/compat/numpy/function.py in validate_func(fname, args, kwargs)
    416 
    417     validation_func = _validation_funcs[fname]
--> 418     return validation_func(args, kwargs)

/usr/local/lib/python3.11/dist-packages/pandas/compat/numpy/function.py in __call__(self, args, kwargs, fname, max_fname_arg_count, method)
     86             validate_kwargs(fname, kwargs, self.defaults)
     87         elif method == "both":
---> 88             validate_args_and_kwargs(
     89                 fname, args, kwargs, max_fname_arg_count, self.defaults
     90             )

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, compat_args)
    221 
    222     kwargs.update(args_dict)
--> 223     validate_kwargs(fname, kwargs, compat_args)
    224 
    225 

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in validate_kwargs(fname, kwargs, compat_args)
    163     kwds = kwargs.copy()
    164     _check_for_invalid_keys(fname, kwargs, compat_args)
--> 165     _check_for_default_values(fname, kwds, compat_args)
    166 
    167 

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in _check_for_default_values(fname, arg_val_dict, compat_args)
     79 
     80         if not match:
---> 81             raise ValueError(
     82                 f"the '{key}' parameter is not supported in "
     83                 f"the pandas implementation of {fname}()"

ValueError: the 'keepdims' parameter is not supported in the pandas implementation of sum()
