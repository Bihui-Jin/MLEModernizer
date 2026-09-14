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

0.4542823045725072

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.48867) has done: 'I fix the immediate runtime failure by removing the hard dependency on an external `/kaggle/input/models20240120` dataset and instead fall back to a safe, valid-probability baseline when weights are unavailable. This ensures the notebook always runs end-to-end and writes a correctly formatted `submission.csv` with rows summing to 1 (required by the metric). I also make the data-path handling more robust by automatically selecting the existing competition root directory in this environment, without changing any model/training core logic. If weight files are present, the script behave as before and use them for inference.'

# 9. Code solution

## === cell 0
import os, sys, subprocess, warnings

warnings.filterwarnings("ignore")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"],
    check=False,
)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240120"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

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

import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph

print("TensorFlow version =", tf.__version__)

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
        print("Mixed precision could not be enabled:", repr(e))
else:
    print("Using full precision")


def _pick_existing_root(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


if PLATFORM == "kaggle":
    COMP_ROOT = _pick_existing_root(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification",
            "/kaggle/data/hms-harmful-brain-activity-classification",
        ]
    )
else:
    COMP_ROOT = "./input/hms-harmful-brain-activity-classification"

print("Using COMP_ROOT:", COMP_ROOT)



## === cell 1
df = pd.read_csv(f"{COMP_ROOT}/train.csv")

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



## === cell 2
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = f"{COMP_ROOT}/train_spectrograms/"
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



## === cell 3
if NEEDTRAIN:
    from scipy import signal

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = f"{COMP_ROOT}/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                time_temp = train[train.eeg_id == name].eeg_median.iloc[-1]
                time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

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

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
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



## === cell 4
import albumentations as albu

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
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eegs = eegs
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 4, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == "test":
                r = 0
            elif self.mode == "valid":
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = np.random.randint(row["min"], row["max"] + 1) // 2

            for k in range(4):
                img = self.specs[row.spec_id][r : r + 300, k * 100 : (k + 1) * 100].T
                img_eeg = self.eegs[row.eeg_id][:, :, k]

                img = np.clip(img, np.exp(-6), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = img[
                    :,
                    round((600 / 2 - LENGTH) / 2) : (
                        round((600 / 2 - LENGTH) / 2) + LENGTH
                    ),
                ]
                if HIGH < 100:
                    img = np.resize(img, (HIGH, LENGTH))
                    X[j, :, :, k] = img
                elif HIGH > 100:
                    X[
                        j,
                        round((HIGH - img.shape[0]) / 2) : -round(
                            (HIGH - img.shape[0]) / 2
                        ),
                        :,
                        k,
                    ] = img
                else:
                    X[j, :, :, k] = img

                X_eeg[j, :, :, k] = img_eeg

            X[j, :, :, :] = (X[j, :, :, :] - np.mean(X[j, :, :, :])) / (
                np.std(X[j, :, :, :]) + 1e-6
            )
            X_eeg[j, :, :, :] = (X_eeg[j, :, :, :]) - np.mean(X_eeg[j, :, :, :])
            X_eeg[j, :, :, :] = X_eeg[j, :, :, :] / (np.std(X_eeg[j, :, :, :]) + 1e-6)

            if self.mode != "test":
                y[j] = row[TARGETS].values

        return X, X_eeg, y

    def __random_transform(self, img):
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
            ]
        )
        return composition(image=img)["image"]

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
        return img_batch




## === cell 5
if NEEDTRAIN:
    import math

    LR_START = 1e-6
    LR_MAX = 1e-3
    LR_MIN = 1e-6
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 0
    EPOCHS2 = 10

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
            decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.5 * (1 + math.cos(phase))
            lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
        return lr

    rng = [i for i in range(EPOCHS2)]
    lr_y = [lrfn(x) for x in rng]
    plt.figure(figsize=(10, 4))
    plt.plot(rng, lr_y, "-o")
    plt.xlabel("epoch", size=14)
    plt.ylabel("learning rate", size=14)
    plt.title("Cosine Training Schedule", size=16)
    plt.show()

    LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

    LR_START = 1e-4
    LR_MAX = 1e-3
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 0
    LR_STEP_DECAY = 0.1
    EVERY = 2
    EPOCHS = 5

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY ** (
                (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) // EVERY
            )
        return lr

    rng = [i for i in range(EPOCHS)]
    y = [lrfn(x) for x in rng]
    plt.figure(figsize=(10, 4))
    plt.plot(rng, y, "o-")
    plt.xlabel("epoch", size=14)
    plt.ylabel("learning rate", size=14)
    plt.title("Step Training Schedule", size=16)
    plt.show()

    LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)




## === cell 6
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
    inp_eeg = tf.keras.Input(shape=(4, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False, weights="imagenet"
    )

    x0 = inp[:, :, :, :1]
    x1 = inp[:, :, :, 1:2]
    x2 = inp[:, :, :, 2:3]
    x3 = inp[:, :, :, 3:4]
    x01 = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

    x = tf.keras.layers.Concatenate(axis=3)([x01, x01, x01])
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    base_model_eeg = tf.keras.applications.EfficientNetB1(
        include_top=False, weights="imagenet"
    )

    x0_eeg = inp_eeg[:, :, :, :1]
    x1_eeg = inp_eeg[:, :, :, 1:2]
    x2_eeg = inp_eeg[:, :, :, 2:3]
    x3_eeg = inp_eeg[:, :, :, 3:4]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)

    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)
    x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 7
if NEEDTRAIN:
    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K, gc

    all_oof = []
    all_true = []

    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        train_gen = DataGenerator(
            train.iloc[train_index],
            shuffle=True,
            batch_size=16,
            specs=spectrograms,
            eegs=eegs,
        )
        valid_gen = DataGenerator(
            train.iloc[valid_index],
            shuffle=False,
            batch_size=32,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
        )

        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        K.clear_session()

        callbacks_list = [
            LR,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"EB2_v{VER}_f{i}.h5",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5, monitor="val_loss", mode="min"
            ),
        ]

        with strategy.scope():
            model = build_model()
        model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )

        oof = model.predict(valid_gen, verbose=1)
        all_oof.append(oof)
        all_true.append(train.iloc[valid_index][TARGETS].values)

        del model, oof
        K.clear_session()
        reset_default_graph()
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)



## === cell 8
if NEEDTRAIN:
    import sys

    if PLATFORM == "local":
        sys.path.append("./input/kaggle-kl-div")
    elif PLATFORM == "kaggle":
        sys.path.append("/kaggle/input/kaggle-kl-div")
    from kaggle_kl_div import score

    oof = pd.DataFrame(all_oof.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true["id"] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name="id")
    print("CV Score KL-Div for EfficientNetB2 =", cv)



## === cell 9
if not NEEDTRAIN:
    test = pd.read_csv(f"{COMP_ROOT}/test.csv")
    print("Test shape", test.shape)

    sample = pd.read_csv(f"{COMP_ROOT}/sample_submission.csv")

    def _find_weight_dir(ver, n_folds=5):
        candidates = []
        candidates.append(LOAD_MODELS_FROM)
        candidates.append("/kaggle/working")
        candidates.append("/kaggle/working/models")
        candidates.append(os.getcwd())
        candidates.append(COMP_ROOT)
        candidates.append(os.path.join(COMP_ROOT, "models"))
        candidates.append(os.path.join(COMP_ROOT, "models20240120"))

        for d in candidates:
            if not isinstance(d, str) or len(d) == 0:
                continue
            if not os.path.isdir(d):
                continue
            ok = True
            for i in range(n_folds):
                if not os.path.isfile(os.path.join(d, f"EB2_v{ver}_f{i}.h5")):
                    ok = False
                    break
            if ok:
                return d
        return None

    weight_dir = _find_weight_dir(VER, n_folds=5)
    use_model = weight_dir is not None
    if use_model:
        expected_weight_paths = [
            os.path.join(weight_dir, f"EB2_v{VER}_f{i}.h5") for i in range(5)
        ]
        print("Found weights in:", weight_dir)
    else:
        print(
            "WARNING: Could not find all required fold weights. Falling back to a stronger baseline."
        )

    if use_model:
        PATH_SPEC = f"{COMP_ROOT}/test_spectrograms/"
        files2 = os.listdir(PATH_SPEC)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values

        test2 = test.rename({"spectrogram_id": "spec_id"}, axis=1)

        from scipy import signal

        PATH_EEG = f"{COMP_ROOT}/test_eegs/"
        files2 = os.listdir(PATH_EEG)
        print(f"\nThere are {len(files2)} test eeg parquets")

        eegs2 = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")
            name = int(f.split(".")[0])

            if len(test2[test2.eeg_id == name]) > 0:
                time_temp = 0
                time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

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

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs2[name] = list_eeg

        test_gen = DataGenerator(
            test2,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        preds = []
        with strategy.scope():
            model = build_model()

        for i in range(5):
            wpath = expected_weight_paths[i]
            print(f"Fold {i + 1} loading: {wpath}")
            model.load_weights(wpath)
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0).astype(np.float32)
        pred = np.clip(pred, 1e-7, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test2.eeg_id.values})
        sub[TARGETS] = pred
        sub = sample[["eeg_id"]].merge(sub, on="eeg_id", how="left")
        sub[TARGETS] = sub[TARGETS].astype(np.float32)
        sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-7, 1.0)
        sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(
            axis=1, keepdims=True
        )
    else:
        global_prior = train[TARGETS].mean(axis=0).values.astype(np.float32)
        global_prior = np.clip(global_prior, 1e-7, 1.0)
        global_prior = global_prior / global_prior.sum()

        patient_priors = train.groupby("patient_id")[TARGETS].mean().astype(np.float32)
        pp = patient_priors.values
        pp = np.clip(pp, 1e-7, 1.0)
        pp = pp / pp.sum(axis=1, keepdims=True)
        patient_priors.loc[:, TARGETS] = pp

        test_pat = test[["eeg_id", "patient_id"]].copy()
        test_pat = test_pat.merge(
            patient_priors, left_on="patient_id", right_index=True, how="left"
        )

        for idx, c in enumerate(TARGETS):
            test_pat[c] = test_pat[c].astype("float32")
            test_pat[c] = test_pat[c].fillna(float(global_prior[idx]))

        sub = sample.copy()
        test_pat = sample[["eeg_id"]].merge(test_pat, on="eeg_id", how="left")

        sub[TARGETS] = test_pat[TARGETS].values.astype(np.float32)
        sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-7, 1.0)
        sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(
            axis=1, keepdims=True
        )

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row sums (min/mean/max):",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).mean(),
        sub[TARGETS].sum(axis=1).max(),
    )
    print("Wrote: submission.csv")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2835030924.py in <cell line: 0>()
    170         test_pat = sample[["eeg_id"]].merge(test_pat, on="eeg_id", how="left")
    171 
--> 172         sub[TARGETS] = test_pat[TARGETS].values.astype(np.float32)
    173         sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-7, 1.0)
    174         sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4348 
   4349             elif isinstance(value, np.ndarray) and value.ndim == 2:
-> 4350                 self._iset_not_inplace(key, value)
   4351 
   4352             elif np.ndim(value) > 1:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _iset_not_inplace(self, key, value)
   4378 
   4379             for i, col in enumerate(key):
-> 4380                 self[col] = igetitem(value, i)
   4381 
   4382         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

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
