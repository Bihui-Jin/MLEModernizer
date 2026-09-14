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

0.3582291977273557

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.39779) has done: 'The fix removes the unused TensorFlow import that caused a protobuf error and adds a safe fallback when pretrained model files are missing. Instead of trying to load non‑existent weights, the script now computes the average class distribution from the training data and uses it as a baseline prediction for every test sample, guaranteeing a valid `submission.csv` with probabilities that sum to 1.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
"""

import os, warnings, io, gc, time, itertools
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models20241107b"  # the path of trained model weights for testing

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
EEG_MULTIPLY = 9
IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324
SPE_HIGH = 100
SPE_WIDE = 256
STFT_LENGTH = 50
STFT_HIGH = 32
STFT_WIDE = round(STFT_LENGTH / 0.4)
filter_range = [0.5, 45]
filter_range2 = [0.1, 35]
SEED = 2024
BATCHSIZE = 16
TEST_BATCHSIZE = 128
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5
BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]

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
tf.config.experimental.enable_op_determinism()

MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
else:
    print("Using full precision")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))




## === cell 2
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
        return x, y, sample_weights

    def on_epoch_end(self):
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
                    (4 * 4 + 2) * EEG_MULTIPLY // 3,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    3,
                ),
                dtype="float32",
            )
        if "stft" in DATATYPE:
            x_stft = np.zeros(
                (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = row.get("sign_id", None)

            if self.mode == "test":
                r_spe = r_eeg = 0
            else:
                rows = df.loc[(df.eeg_id == row.eeg_id)].reset_index(drop=True)
                row_match = rows.iloc[0]
                r_spe = (
                    round(row_match.spectrogram_label_offset_seconds / 2)
                    if "spe" in DATATYPE
                    else 0
                )
                r_eeg = row_match.eeg_label_offset_seconds if "eeg" in DATATYPE else 0

            if "spe" in DATATYPE:
                spe = []
                for k in range(4):
                    seg = self.specs[row.spectrogram_id][
                        r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                    ].T
                    spe.append(np.reshape(seg, (1, 100, 300)))
                spe = np.concatenate(spe, axis=0)
                spe[np.isnan(spe)] = 0
                spe = np.clip(spe, 1e-6, 1e6)
                spe = np.log2(spe)
                spe = spe[
                    :,
                    :,
                    round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                        (spe.shape[2] - SPE_WIDE) / 2
                    ),
                ]
                spe = spe[[0, 2, 3, 1], :, :]
                if self.mode == "train":
                    if np.random.rand() > 0.5:
                        spe = spe[::-1, :, :]
                spe = (spe - np.mean(spe, keepdims=True)) / (
                    np.std(spe, keepdims=True) + 1e-6
                )
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = self.eegs[row.eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]
                eeg_save = np.zeros(
                    (x_eeg.shape[1], x_eeg.shape[2], x_eeg.shape[3]), dtype=np.float32
                )
                if self.mode == "train":
                    eeg[0:8] = eeg[0:8][np.random.permutation(8)]
                    eeg[10:18] = eeg[10:18][np.random.permutation(8)]
                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]
                for ii in range(eeg_save.shape[0]):
                    eeg_save[ii, :, 0] = eeg[
                        ii // (EEG_MULTIPLY // 3),
                        (
                            ii % (EEG_MULTIPLY // 3) + 0 * EEG_MULTIPLY // 3
                        ) :: EEG_MULTIPLY,
                    ][: eeg_save.shape[1]]
                    eeg_save[ii, :, 1] = eeg[
                        ii // (EEG_MULTIPLY // 3),
                        (
                            ii % (EEG_MULTIPLY // 3) + 1 * EEG_MULTIPLY // 3
                        ) :: EEG_MULTIPLY,
                    ]
                    eeg_save[ii, :, 2] = eeg[
                        ii // (EEG_MULTIPLY // 3),
                        (
                            ii % (EEG_MULTIPLY // 3) + 2 * EEG_MULTIPLY // 3
                        ) :: EEG_MULTIPLY,
                    ]
                eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                    np.std(eeg_save, keepdims=True) + 1e-6
                )
                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                stft = self.stfts[row.eeg_id][
                    :, :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]
                stft = np.clip(stft, 1e-6, 1e6)
                stft = np.log2(stft)
                if self.mode == "train" and np.random.rand() > 0.5:
                    stft = stft[::-1, :, :]
                stft_save = np.zeros(
                    (round(stft.shape[0] / 2 * stft.shape[1]), stft.shape[2] * 2),
                    dtype=np.float32,
                )
                for ii in range(stft.shape[0]):
                    stft_save[
                        ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                        (ii % 2) * stft.shape[2] : (ii % 2 + 1) * stft.shape[2],
                    ] = stft[ii]
                stft = (stft_save - np.mean(stft_save, keepdims=True)) / (
                    np.std(stft_save, keepdims=True) + 1e-6
                )
                x_stft[j] = stft

            if "img" in DATATYPE and sign_id is not None:
                img = self.imgs[sign_id]
                if self.mode == "train":
                    img[0:8] = img[0:8][np.random.permutation(8)]
                    img[10:18] = img[10:18][np.random.permutation(8)]
                    if np.random.rand() > 0.5:
                        img = img[::-1, :, :]
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                y[j] = row[TARGETS].values / row[TARGETS].values.sum()
                sample_weights[j] = 1.0

        x = []
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "stft" in DATATYPE:
            x.append(x_stft)
        if "img" in DATATYPE:
            x.append(x_img)

        return x, y, sample_weights




## === cell 3
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super().__init__()
        self.total_step = total_step
        self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
        self.lr_max = lr_max
        self.lr_min = lr_min

    @tf.function
    def __call__(self, step):
        step = step + 1
        if step < self.warm_step:
            lr = self.lr_max / self.warm_step * step
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0
                + tf.cos(
                    (step - self.warm_step) / (self.total_step - self.warm_step) * np.pi
                )
            )
        return lr




## === cell 4
def build_model():
    inputs = []
    representations = []

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
        x = tf.keras.layers.Concatenate(axis=1)(
            [
                inp_spe[:, 0, :, :],
                inp_spe[:, 1, :, :],
                inp_spe[:, 2, :, :],
                inp_spe[:, 3, :, :],
            ]
        )
        x = tf.keras.layers.Reshape((x.shape[1], x.shape[2], 1))(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, x, x])
        base_spe = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=x
        )
        x = base_spe.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        inputs.append(inp_spe)
        representations.append(x)

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                (4 * 4 + 2) * EEG_MULTIPLY // 3,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                3,
            )
        )
        base_eeg = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights=None,
            input_tensor=inp_eeg,
            include_preprocessing=False,
        )
        x = base_eeg.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        inputs.append(inp_eeg)
        representations.append(x)

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
        x = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(inp_stft)
        x = tf.keras.layers.Concatenate(axis=-1)([x, x, x])
        base_stft = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=x
        )
        x = base_stft.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        inputs.append(inp_stft)
        representations.append(x)

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
        base_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=inp_img
        )
        x = base_img.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        inputs.append(inp_img)
        representations.append(x)

    if len(representations) > 1:
        y = tf.keras.layers.Concatenate(axis=1)(representations)
    else:
        y = representations[0]

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model




## === cell 5
if not NEEDTRAIN:
    model_template = build_model()
    models = []
    missing_weights = False
    for fold_idx in range(SPLITS):
        print(f"Loading fold {fold_idx + 1}")
        model = clone_model(model_template)
        weight_path = os.path.join(LOAD_MODELS_FROM, f"fold{fold_idx}_stage2.h5")
        try:
            model.load_weights(weight_path)
            models.append(model)
        except FileNotFoundError:
            print(f"Weight file not found: {weight_path}")
            missing_weights = True
            break

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    if missing_weights:
        train_probs = df[TARGETS].div(df[TARGETS].sum(axis=1), axis=0)
        mean_probs = train_probs.mean(axis=0).values
        preds_avg = np.tile(mean_probs, (len(test), 1))
    else:
        spectrograms_test = {}
        eegs_test = {}
        stfts_test = {}
        imgs_test = {}

        if "spe" in DATATYPE:
            path_spe = os.path.join(LOAD_DATA_FROM, "test_spectrograms")
            for f in os.listdir(path_spe):
                tmp = pd.read_parquet(os.path.join(path_spe, f))
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values

        path_eeg = os.path.join(LOAD_DATA_FROM, "test_eegs")
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")

        for i, eeg_id in enumerate(test.eeg_id):
            eeg_default = pd.read_parquet(os.path.join(path_eeg, f"{eeg_id}.parquet"))

            eeg = []
            for ch in BRAIN:
                left, right = ch.split("-")
                diff = (eeg_default[left] - eeg_default[right]).values
                diff[np.isnan(diff)] = 0
                eeg.append(diff.reshape(1, -1))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            eeg_filt = signal.filtfilt(b, a, eeg, axis=1)
            eeg_filt = np.clip(eeg_filt, -1024, 1024)
            eegs_test[eeg_id] = eeg_filt

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                f_, t_, s_ = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                )
                s_[np.isnan(s_)] = 0
                s_ = s_[:, (f_ > 0) & (f_ <= 20), :]
                stfts_test[eeg_id] = s_
                stfts_test[-eeg_id] = t_

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, -1024, 1024)
                img_save = np.zeros((eeg2.shape[0], 36, IMG_WIDE), dtype=np.float32)
                for ch_idx in range(eeg2.shape[0]):
                    fig = plt.figure(figsize=(3.93, 2 / 18 * 2))
                    plt.plot(eeg2[ch_idx, :] + 100, color="red", linewidth=0.2)
                    plt.axis("off")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    plt.close()
                    buf.seek(0)
                    img = Image.open(buf)
                    img = np.array(img)[:, :, 0] / 255.0
                    img = np.resize(img, (36, IMG_WIDE))
                    img_save[ch_idx] = img
                imgs_test[test.loc[i, "sign_id"]] = img_save

        test_gen = DataGenerator(
            test,
            batch_size=TEST_BATCHSIZE,
            shuffle=False,
            sample_weights=False,
            mode="test",
            specs=spectrograms_test,
            eegs=eegs_test,
            stfts=stfts_test,
            imgs=imgs_test,
        )

        all_preds = []
        for fold_idx, model in enumerate(models):
            print(f"Predicting with fold {fold_idx + 1}")
            preds = model.predict(test_gen, verbose=1)
            all_preds.append(preds)

        preds_avg = np.mean(np.stack(all_preds, axis=0), axis=0)
        preds_avg = preds_avg / preds_avg.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = preds_avg
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
