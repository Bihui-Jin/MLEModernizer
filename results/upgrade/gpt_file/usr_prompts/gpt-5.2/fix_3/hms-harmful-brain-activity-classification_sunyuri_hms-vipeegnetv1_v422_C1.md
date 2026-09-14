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

0.2756066448364537

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash caused by TensorFlow/Protobuf incompatibility in this Kaggle Python 3.13 environment by removing the hard TensorFlow dependency from the inference path and generating a safe fallback submission when no compatible trained weights can be loaded. I also fix a logic bug in test batching (`len(preds_all)` used as an index) and ensure predictions are valid probabilities that sum to 1 for every row (required for submission validity and KL metric). The core model/training code is left intact, but in Kaggle mode (where training is disabled anyway) the script bypass TensorFlow if it cannot be imported, and still write `submission.csv`. This produces a valid submission end-to-end (score won’t be competitive without models, but it run and yield a submission as requested).'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow import crash in Python 3.13 (protobuf `MessageFactory.GetPrototype` issue) by making TensorFlow strictly optional on Kaggle and routing inference/training to a robust non-TF fallback when TF cannot be imported. To improve the score from the current uniform-probability fallback (1.40995) toward the target (0.2756), the fallback produce a strong prior based on normalized mean class probabilities computed from `train.csv` vote distributions (score-improving but still leakage-safe). I also ensure the submission always matches `sample_submission.csv` ordering/columns and that each row sums to 1 with safe clipping. Core model/training code and semantics remain unchanged; we only adjust the execution path and submission construction when TensorFlow is unavailable.'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri

email: syuri@tju.edu.cn
"""

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

import os
import warnings

warnings.filterwarnings("ignore")

os.environ["KERAS_BACKEND"] = "tensorflow"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # *** local kaggle *** local training or online testing
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

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

import numpy as np
import pandas as pd
from scipy import signal
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print(tf.version.VERSION)
        print(tf.config.list_physical_devices("GPU"))
    except Exception as e:
        print("TF GPU config skipped:", repr(e))

    MIX = True
    if MIX:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        except Exception as e:
            print("Mixed precision skipped:", repr(e))
    else:
        print("Using full precision")

except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    print("TensorFlow unavailable, will use fallback inference. Error:", repr(e))

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is required for training, but failed to import. Error: "
            + repr(TF_IMPORT_ERROR)
        )

    import itertools  # original behavior

    TARGETS_RAW = [t + "_raw" for t in TARGETS]

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
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
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



## === cell 2
if TF_AVAILABLE:
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

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
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
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
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0

                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0

                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
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
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1

            return x_eeg, y, sample_weights

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
            attn_output, weights = self.att(
                inputs, inputs, return_attention_scores=True
            )
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

    def build_model():
        inp_eeg = tf.keras.Input(
            shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)), name="eeg"
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
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )

        base_model_eeg.name = "eeg_extractor"
        x_eeg = base_model_eeg(x_eeg)

        x_eeg = tf.keras.layers.AveragePooling2D((x_eeg.shape[1], 1))(x_eeg)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)
        strides = 1
        if PLATFORM == "local":
            eeg_embed = tf.keras.layers.Conv1D(
                filters=strides,
                kernel_size=x_eeg.shape[2],
                strides=x_eeg.shape[2],
                padding="valid",
                use_bias=False,
                activation=None,
                kernel_initializer=IniToOneAtten(),
                kernel_constraint=SumToOneAtten(),
                input_shape=(None, 1),
            )
        else:
            eeg_embed = tf.keras.layers.Conv1D(
                filters=strides,
                kernel_size=x_eeg.shape[2],
                strides=x_eeg.shape[2],
                padding="valid",
                use_bias=False,
                activation=None,
            )
        x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            x_eeg
        )

        model = tf.keras.Model(inputs=inp_eeg, outputs=y)
        return model




## === cell 3
if TF_AVAILABLE:

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
        import matplotlib.pyplot as plt

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

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        import itertools

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




## === cell 4
def _normalize_probs(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, eps, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s).astype(np.float32)
    return p / s


def _write_fallback_submission(strategy: str = "train_prior"):
    sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
    test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
    train_path = os.path.join(LOAD_DATA_FROM, "train.csv")

    sample = pd.read_csv(sample_path)
    test = pd.read_csv(test_path)

    sub = sample[["eeg_id"]].copy()
    if not np.array_equal(sub["eeg_id"].values, test["eeg_id"].values):
        test_ids = set(test["eeg_id"].values.tolist())
        sub = sub[sub["eeg_id"].isin(test_ids)].reset_index(drop=True)

    if strategy == "train_prior":
        dft = pd.read_csv(train_path, usecols=list(TARGETS))
        y = dft[list(TARGETS)].values.astype(np.float32)
        y = y / np.clip(y.sum(axis=1, keepdims=True), 1e-7, None)
        prior = y.mean(axis=0, keepdims=True)  # shape (1,6)
        probs = np.repeat(prior, len(sub), axis=0)
        probs = _normalize_probs(probs)
    else:
        probs = np.full((len(sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)

    sub[list(TARGETS)] = probs
    sub.to_csv("submission.csv", index=False)
    print(f"Wrote fallback submission.csv using strategy={strategy}, shape={sub.shape}")
    print(sub.head())


if __name__ == "__main__":
    if NEEDTRAIN:
        if not TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow is required for training, but failed to import. Error: "
                + repr(TF_IMPORT_ERROR)
            )

        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        mp.set_start_method("spawn")

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
        if not TF_AVAILABLE:
            _write_fallback_submission(strategy="train_prior")
        else:
            preds_all = []
            models = []

            model_template = build_model()
            for model_i in range(999):
                if os.path.exists(
                    os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
                ):
                    print(f"Fold {model_i + 1}")
                    model = clone_model(model_template)
                    model.load_weights(
                        os.path.join(
                            LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5"
                        )
                    )
                    models.append(model)

            test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
            test["sign_id"] = test.index.values
            print("Test shape", test.shape)

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

            if len(models) == 0:
                print("No model weights found in:", LOAD_MODELS_FROM)
                _write_fallback_submission(strategy="train_prior")
            else:
                start_idx = 0
                for i, eeg_id in enumerate(test.eeg_id):
                    if i % 100 == 0:
                        print(i, ", ", end="")

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
                        end_idx = i + 1
                        batch_df = test.iloc[start_idx:end_idx].reset_index(drop=True)

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
                        start_idx = end_idx

                        if len(preds_all) == 0:
                            preds_all = pred.copy()
                        else:
                            preds_all = np.concatenate((preds_all, pred), axis=0)

                preds_all = np.asarray(preds_all, dtype=np.float32)
                preds_all = _normalize_probs(preds_all, eps=1e-7)

                sample = pd.read_csv(
                    os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
                )
                sub = sample[["eeg_id"]].copy()
                pred_df = pd.DataFrame({"eeg_id": test.eeg_id.values})
                pred_df[list(TARGETS)] = preds_all
                sub = sub.merge(pred_df, on="eeg_id", how="left")
                if sub[list(TARGETS)].isna().any().any():
                    dft = pd.read_csv(
                        os.path.join(LOAD_DATA_FROM, "train.csv"), usecols=list(TARGETS)
                    )
                    y = dft[list(TARGETS)].values.astype(np.float32)
                    y = y / np.clip(y.sum(axis=1, keepdims=True), 1e-7, None)
                    prior = y.mean(axis=0, keepdims=True)
                    fill = np.repeat(prior, len(sub), axis=0)
                    fill = _normalize_probs(fill)
                    sub[list(TARGETS)] = sub[list(TARGETS)].fillna(
                        pd.DataFrame(fill, columns=list(TARGETS))
                    )

                sub[list(TARGETS)] = _normalize_probs(
                    sub[list(TARGETS)].values.astype(np.float32)
                )
                sub.to_csv("submission.csv", index=False)
                print("Submission shape", sub.shape)
                print(sub.head())
