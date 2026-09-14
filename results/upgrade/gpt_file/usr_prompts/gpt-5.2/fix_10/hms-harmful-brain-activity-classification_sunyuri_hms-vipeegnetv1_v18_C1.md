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

0.4683979233755924

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/Protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle’s Python 3.12 runtime. Then I fix the invalid submission by ensuring `eeg_id` is unique and aligned to `sample_submission.csv` (your generator currently emits one prediction per row in `test.csv`, but Kaggle expects one per unique `eeg_id`, and duplicates lead to row-sum validation failure). Finally, I keep your model/inference logic intact but add a safe aggregation step (mean over duplicate `eeg_id` if any remain) and a final renormalization to guarantee each row sums to 1 within float tolerance.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash in Python 3.12 by setting the protobuf env vars before any TensorFlow-related import and also forcing the pure-Python protobuf module to be imported first (this avoids the `MessageFactory.GetPrototype` attribute error). Then I ensure the inference uses the intended pre-trained weights from `/kaggle/input/...` by correcting the model directory variable so it points to an existing Kaggle dataset folder (otherwise you silently fall back to random weights, which explains the poor 1.40995 KL score). Finally, I keep your model and generator logic intact but make the submission alignment more robust by always merging onto `sample_submission.csv`, filling any missing predictions with uniform probabilities, and renormalizing to guarantee rows sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/Protobuf crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow import and also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` unconditionally early, which addresses the `MessageFactory.GetPrototype` error in Kaggle’s Py3.12 runtime. Then I correct the model-weights path fallback by auto-discovering the correct `/kaggle/input/...` directory that actually contains the `EB2_v{VER}_f{i}.h5` files (your current `LOAD_MODELS_FROM` often points to a non-existent dataset, leading to random-weight inference and a very poor KL score). Finally, I keep your model, generator, and inference logic intact while making submission alignment robust (merge onto `sample_submission.csv`, fill missing with uniform, and renormalize) to guarantee a valid CSV with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash in Python 3.12 by forcing protobuf’s pure-Python backend *and* pinning a compatible protobuf version before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error). Then I make the model-weights discovery more robust so your code actually finds and loads the provided fold `.h5` files (your current path often misses, leading to random-weight inference and the very poor KL score). Finally, I keep your generator/model logic unchanged but harden inference/submission alignment (handle any missing spec/eeg IDs safely, and always renormalize) to ensure a valid `submission.csv` with per-row probabilities summing to 1.'

# 9. Code solution

## === cell 0
import os, sys, gc, math, warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False

LOAD_MODELS_FROM = "models202401302"
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

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

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
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (global policy)")
    except Exception:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled (experimental)")
        except Exception as e:
            print("Mixed precision could not be enabled:", str(e))
else:
    print("Using full precision")

print(
    "LOAD_MODELS_FROM =",
    LOAD_MODELS_FROM,
    "| exists =",
    os.path.exists(LOAD_MODELS_FROM),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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



## === cell 2
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
    from scipy import signal

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
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
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

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



## === cell 3
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
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = bool(augment) and (albu is not None)
        self.mode = mode
        self.specs = specs
        self.eegs = eegs
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
        return (X, X_eeg), y

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
        img = np.ones((HIGH, LENGTH), dtype="float32")
        img_eeg = np.ones((4, round(EEG_LENGTH * SFREQ)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == "test":
                r = 0
            elif self.mode == "valid":
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = int((row["min"] + row["max"]) // 4)

            for k in range(4):
                if (self.specs is not None) and (row.spec_id in self.specs):
                    img = self.specs[row.spec_id][
                        r : r + 300, k * 100 : (k + 1) * 100
                    ].T
                else:
                    img = np.zeros((100, 300), dtype="float32")

                if (self.eegs is not None) and (row.eeg_id in self.eegs):
                    img_eeg = self.eegs[row.eeg_id][:, :, k]
                else:
                    img_eeg = np.zeros((4, round(EEG_LENGTH * SFREQ)), dtype="float32")

                img = np.clip(img, np.exp(-6), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = img[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img.shape[1]
                    ),
                ]
                if HIGH != 100:
                    img_rs = tf.image.resize(
                        np.reshape(img, (img.shape[0], img.shape[1], 1)),
                        ((HIGH - 64), LENGTH),
                    )
                    img_rs = np.array(img_rs, dtype=np.float32)[:, :, 0]
                    X[
                        j,
                        round((HIGH - img_rs.shape[0]) / 2) : round(
                            (HIGH + img_rs.shape[0]) / 2
                        ),
                        :,
                        k,
                    ] = img_rs
                else:
                    X[j, :, :, k] = img

                X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                    np.std(X[j, :, :, k]) + 1e-6
                )

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = row[TARGETS].values

        return X, X_eeg, y

    def __random_transform(self, img):
        if albu is None:
            return img
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
            img_batch[i,] = self.__random_transform(img_batch[i,])
        return img_batch




## === cell 4
def _load_efficientnet_backbone(name: str):
    if name == "B2":
        return tf.keras.applications.EfficientNetB2(
            include_top=False, weights="imagenet", input_shape=None
        )
    if name == "B1":
        return tf.keras.applications.EfficientNetB1(
            include_top=False, weights="imagenet", input_shape=None
        )
    raise ValueError("Unknown backbone")


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    try:
        base_model = _load_efficientnet_backbone("B2")
        base_model._name = "spectrogram_extractor"
    except Exception as e:
        print(
            "Warning: could not load imagenet weights for EfficientNetB2. Using random init. Error:",
            str(e),
        )
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False, weights=None, input_shape=None
        )
        base_model._name = "spectrogram_extractor"

    x0 = inp[:, :, :, :1]
    x1 = inp[:, :, :, 1:2]
    x2 = inp[:, :, :, 2:3]
    x3 = inp[:, :, :, 3:4]
    x01 = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])
    x = tf.keras.layers.Concatenate(axis=3)([x01, x01, x01])  # to 3 channels

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    try:
        base_model_eeg = _load_efficientnet_backbone("B1")
        base_model_eeg._name = "eeg_extractor"
    except Exception as e:
        print(
            "Warning: could not load imagenet weights for EfficientNetB1. Using random init. Error:",
            str(e),
        )
        base_model_eeg = tf.keras.applications.EfficientNetB1(
            include_top=False, weights=None, input_shape=None
        )
        base_model_eeg._name = "eeg_extractor"

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




## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    print("Raw test shape", test.shape)

    test = test.drop_duplicates(subset=["eeg_id"]).reset_index(drop=True)
    print("Unique eeg_id test shape", test.shape)

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
    test["min"] = 0
    test["max"] = 0

    files2 = os.listdir(PATH_SPEC)
    print(f"There are {len(files2)} test spectrogram parquets")
    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    from scipy import signal

    files2 = os.listdir(PATH_EEG)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            time_temp = 0
            time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
            time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

            eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                drop=True
            )

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

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            eegs2[name] = list_eeg
    print()

    if PLATFORM == "kaggle" and (not os.path.exists(LOAD_MODELS_FROM)):
        target_fnames = [f"EB2_v{VER}_f{i}.h5" for i in range(5)]
        found_dir = None
        base = "/kaggle/input"
        try:
            for d in os.listdir(base):
                cand = os.path.join(base, d)
                if not os.path.isdir(cand):
                    continue
                if all(os.path.exists(os.path.join(cand, fn)) for fn in target_fnames):
                    found_dir = cand
                    break
                try:
                    for d2 in os.listdir(cand):
                        cand2 = os.path.join(cand, d2)
                        if os.path.isdir(cand2) and all(
                            os.path.exists(os.path.join(cand2, fn))
                            for fn in target_fnames
                        ):
                            found_dir = cand2
                            break
                    if found_dir is not None:
                        break
                except Exception:
                    pass
        except Exception:
            found_dir = None

        if found_dir is not None:
            LOAD_MODELS_FROM = found_dir
            print("Auto-found model directory:", LOAD_MODELS_FROM)
        else:
            print(
                "Could not auto-find model directory in /kaggle/input; will fall back to current path."
            )

    preds = []
    with strategy.scope():
        model = build_model()

    test_gen = DataGenerator(
        test, shuffle=False, batch_size=32, mode="test", specs=spectrograms2, eegs=eegs2
    )

    missing = []
    for i in range(5):
        wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
        if not os.path.exists(wpath):
            missing.append(wpath)

    if len(missing) > 0:
        print(
            "WARNING: Missing model weights; will run inference with randomly initialized model."
        )
        for p in missing[:5]:
            print(" -", p)
        pred = model.predict(test_gen, verbose=1)
    else:
        for i in range(5):
            print(f"Fold {i+1}")
            wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
        pred = np.mean(preds, axis=0)

    print("Test preds shape", pred.shape)

    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    pred_df = pd.DataFrame(pred, columns=list(TARGETS))
    pred_df.insert(0, "eeg_id", test.eeg_id.values)

    pred_df = pred_df.groupby("eeg_id", as_index=False)[list(TARGETS)].mean()

    sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

    miss_mask = sub[list(TARGETS)].isna().any(axis=1)
    if miss_mask.any():
        sub.loc[miss_mask, list(TARGETS)] = 1.0 / len(TARGETS)

    arr = sub[list(TARGETS)].to_numpy(dtype=np.float64)
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    sub[list(TARGETS)] = arr.astype(np.float32)

    sub = sub[sample_sub.columns]  # required order
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    rs = sub[list(TARGETS)].sum(axis=1)
    print("Row-sum min/max:", float(rs.min()), float(rs.max()))
    print(sub.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3489926833.py in <cell line: 0>()
    150         for p in missing[:5]:
    151             print(" -", p)
--> 152         pred = model.predict(test_gen, verbose=1)
    153     else:
    154         for i in range(5):

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling DepthwiseConv2D.call().

Negative dimension size caused by subtracting 5 from 4 for '{{node functional_1/efficientnetb1_1/block6a_dwconv_1/depthwise}} = DepthwiseConv2dNative[T=DT_HALF, data_format="NHWC", dilations=[1, 1, 1, 1], explicit_paddings=[], padding="VALID", strides=[1, 2, 2, 1]](functional_1/efficientnetb1_1/block6a_dwconv_pad_1/Pad, functional_1/efficientnetb1_1/block6a_dwconv_1/depthwise/Cast)' with input shapes: [32,4,259,672], [5,5,672,1].

Arguments received by DepthwiseConv2D.call():
  • inputs=tf.Tensor(shape=(32, 4, 259, 672), dtype=float16)
