# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3158216294907964

# 6. Current score

0.71051

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I modify the script to avoid the TensorFlow import errors by safely handling a failed import, skip model construction when TensorFlow isn’t available, and provide a simple baseline prediction (the average class distribution from the training set) when no pretrained models are found. This ensures a valid `submission.csv` is written without altering the core training logic, and the baseline prediction is expected to bring the KL‑divergence closer to the target score.'
- What this solution (achieved 1.39779) has done: 'The script failed because TensorFlow could not be imported (causing an exception) and later code still attempted to use TensorFlow‑related objects and undefined utilities. I added safe imports for `scipy.signal` and `zoom`, forced TensorFlow to be treated as unavailable after the import attempt (so the model loading block is skipped), and provided a minimal dummy `build_model` function to prevent a NameError if TensorFlow ever becomes available. These changes keep the original logic intact while guaranteeing that the fallback baseline‑mean prediction path runs and a proper `submission.csv` is written.'
- What this solution (achieved 1.39779) has done: 'I fixed the import error by correctly importing NumPy, added a safe import for Pillow (Image) to avoid crashes if it’s missing, and kept the rest of the logic unchanged so the script runs end‑to‑end and writes a valid `submission.csv` using the fallback mean‑probability baseline when no models are found.'
- What this solution (achieved 1.64506) has done: 'I replace the simple global‑mean fallback with a patient‑aware baseline: compute per‑patient average vote distributions (normalised) from the training data and use them for test rows that share the same patient_id, falling back to the overall mean otherwise. This small change keeps the original logic intact while providing a more tailored prediction, which should lower the KL‑divergence toward the target value.'
- What this solution (achieved 0.71539) has done: 'We keep the original pipeline but replace the pure patient‑wise mean fallback with a smoothed version: each patient’s mean vote distribution is blended with the global class mean, weighted by how many training samples that patient has. This reduces over‑fitting on patients with few examples and is expected to lower the KL‑divergence, bringing the score closer to the target while leaving all other logic untouched.'
- What this solution (achieved 0.75645) has done: 'We raise the smoothing factor used when blending patient‑specific vote distributions with the global mean. A larger `SMOOTHING` value gives more weight to the globally‑averaged probabilities, which reduces over‑fitting to patients with few training samples and is expected to lower the KL‑divergence toward the target. The change is limited to the constant definition and does not alter any core logic or model architecture.'
- What this solution (achieved 0.71423) has done: 'We lower the smoothing factor used when blending patient‑specific vote distributions with the global mean. A smaller `SMOOTHING` puts more weight on the patient‑wise averages, which generally yields probabilities closer to the true distribution for patients seen in the training set and should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 0.71051) has done: 'I fix the pandas `sum` call that used the unsupported `keepdims` argument. By computing the row sums separately and then dividing each row by its sum, the script correctly normalize the probabilities and write a valid `submission.csv`. This change restores end‑to‑end execution without altering the core modeling logic.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

NEEDTRAIN = False  # set to False to run inference only and avoid heavy training
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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

EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 40  # the height of the spectrogram 100
SPE_WIDE = 1000  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3 * BATCHSIZE / 16
EPOCHS = 15

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg
READ_SPE_FILES = False  # preprocess spectrogram

spectrograms = {}  # preprocessed spectrograms for training
eegs = {}  # preprocessed eegs for training
stfts = {}  # preprocessed short-time fourier transform plots for training
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
import io

try:
    from PIL import Image
except Exception:
    Image = None
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

try:
    from scipy import signal
    from scipy.ndimage import zoom
except Exception:
    signal = None
    zoom = None

tf = None


class Sequence:
    pass


class optimizers:
    class schedules:
        class LearningRateSchedule:
            pass


def clone_model(x):
    return x


print("TensorFlow import disabled – using fallback mode.")

if tf is not None:
    tf = None  # forced disable (kept for compatibility with original script)

MIX = True
if tf is not None and MIX:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision or fallback without TF")

if NEEDTRAIN:
    import itertools

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
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
        train = pd.read_csv("train.csv")

if filter_range is not None and signal is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

if tf is not None:

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
            ct = int(np.ceil(len(self.dataframe) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[index * self.batch_size : (index + self.batch_size)]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

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

            targets_batch = list()

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                sign_id = row.sign_id
                if self.mode != "test":
                    sample_weight = sum(row[TARGETS_RAW].values) / 20
                    targets_batch.append(row.expert_consensus)

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                    r_stft = 0
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
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds
                    if self.mode == "train":
                        r_eeg = r_eeg + np.random.random() * 10 - 5
                        r_eeg = max(0, r_eeg)
                        r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

                if "spe" in DATATYPE:
                    spe = list()  # LL RL LP RP
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                self.specs[row.spectrogram_id][
                                    r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                ].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
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

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    r_stft2 = (np.where(stft_t <= (50 + r_eeg - min(stft_t))))[0][-1]
                    stft = self.stfts[row.eeg_id][:, :, r_stft:r_stft2]

                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                    stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    exp_min, exp_max = -4, 6
                    spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    spe = np.log(spe)

                    if (spe.shape[1] != SPE_HIGH) or (spe.shape[2] != SPE_WIDE):
                        spe2 = np.zeros(
                            (spe.shape[0], SPE_HIGH, SPE_WIDE), dtype=np.float32
                        )
                        for k in range(4):
                            scaled_arr = zoom(
                                spe[k],
                                (SPE_HIGH / spe.shape[1], SPE_WIDE / spe.shape[2]),
                                order=1,
                            )
                            spe2[k, :, :] = scaled_arr
                        spe = spe2.copy()

                    if self.mode == "train":
                        spe2 = spe.copy()
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[2]
                            spe[2] = spe2[0]
                        if np.random.rand() > 0.5:
                            spe[1] = spe2[3]
                            spe[3] = spe2[1]
                        if np.random.rand() > 0.5:
                            spe = spe[::-1, :, :]

                        if np.random.rand() > 0.5:
                            for ii in range(spe.shape[0]):
                                m1 = round(np.random.rand() * spe.shape[2] / 2)
                                m2 = round(np.random.rand() * spe.shape[2] / 2)
                                if np.random.rand() > 0.5:
                                    m1 = spe.shape[2] - m1
                                    m2 = spe.shape[2] - m2
                                m_min = min(m1, m2)
                                m_max = min(
                                    max(m1, m2), m_min + round(spe.shape[2] * 0.05)
                                )
                                spe[ii, :, m_min:m_max] = 0

                    spe = (spe - exp_min) / (exp_max - exp_min) * 255
                    spe = np.clip(spe, a_min=0, a_max=255)

                    x_spe[j] = spe

                if "eeg" in DATATYPE:
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

                if "stft" in DATATYPE:
                    exp_min, exp_max = 0, 8
                    stft = np.clip(stft, a_min=0, a_max=np.exp(exp_max))
                    stft = np.log1p(stft)

                    if self.mode == "train":
                        stft[0 : round(stft.shape[0] / 2), :, :] = stft[
                            0 : round(stft.shape[0] / 2), :, :
                        ][np.random.permutation(stft.shape[0] // 2), :, :]
                        stft[-round(stft.shape[0] / 2) :, :, :] = stft[
                            -round(stft.shape[0] / 2) :, :, :
                        ][np.random.permutation(stft.shape[0] // 2), :, :]

                        stft2 = stft.copy()
                        stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                            0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                        ]
                        stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                            3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                        ]
                        stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                            1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                        ]
                        stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                            2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                        ]

                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :, :]

                    else:
                        stft2 = stft.copy()
                        stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                            0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                        ]
                        stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                            3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                        ]
                        stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                            1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                        ]
                        stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                            2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                        ]

                    stft = (stft - exp_min) / (exp_max - exp_min) * 255
                    stft = np.clip(stft, a_min=0, a_max=255)

                    if "stft" in DATATYPE:
                        if j == 0:
                            x_stft = np.zeros(
                                (
                                    len(indexes),
                                    stft.shape[0],
                                    stft.shape[1],
                                    stft.shape[2],
                                ),
                                dtype="float32",
                            )
                            x_stft[j] = stft
                        else:
                            x_stft[j] = stft

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)

                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)

                    x_img[j] = img

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1

            x = {}
            if "spe" in DATATYPE:
                x["spe"] = x_spe
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            if "stft" in DATATYPE:
                x["stft"] = x_stft
            if "img" in DATATYPE:
                x["img"] = x_img

            return x, y, sample_weights

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

else:

    class DataGenerator:
        pass


if __name__ == "__main__":
    if NEEDTRAIN:
        pass
    else:
        preds_all = []
        models = []

        if tf is not None:
            model_template = build_model()
            for model_i in range(100):
                weight_path = os.path.join(
                    LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5"
                )
                if os.path.exists(weight_path):
                    print(f"Fold {model_i+1}")
                    model = clone_model(model_template)
                    model.load_weights(weight_path)
                    models.append(model)

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        if not models:
            print("No pretrained models found – using improved patient‑aware baseline.")
            train_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
            target_cols = train_df.columns[-6:]

            train_probs = train_df[target_cols].values.astype(np.float32)
            train_probs = train_probs / train_probs.sum(axis=1, keepdims=True)

            global_mean = train_probs.mean(axis=0)

            patient_means = {}
            patient_counts = {}
            for pid, prob in zip(train_df["patient_id"].values, train_probs):
                if pid not in patient_means:
                    patient_means[pid] = prob
                    patient_counts[pid] = 1
                else:
                    patient_means[pid] += prob
                    patient_counts[pid] += 1
            for pid in patient_means:
                patient_means[pid] = patient_means[pid] / patient_counts[pid]

            SMOOTHING = 1.0  # smaller than before to keep more patient specificity
            patient_weights = {pid: np.sqrt(cnt) for pid, cnt in patient_counts.items()}

            preds = []
            for pid in test["patient_id"].values:
                if pid in patient_means:
                    w_patient = patient_weights[pid]
                    blended = (
                        patient_means[pid] * w_patient + global_mean * SMOOTHING
                    ) / (w_patient + SMOOTHING)
                    preds.append(blended)
                else:
                    preds.append(global_mean)
            preds_all = np.stack(preds, axis=0)
        else:
            pass

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        row_sums = sub[TARGETS].sum(axis=1).replace(0, 1e-12)
        sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        sub.head()
