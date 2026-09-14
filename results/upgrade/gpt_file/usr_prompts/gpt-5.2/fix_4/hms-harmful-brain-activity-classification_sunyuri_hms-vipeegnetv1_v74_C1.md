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

0.6650138086282198

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I (1) fix the early TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, (2) fix the Keras model build failure by ensuring the two EfficientNetB0 backbones have unique names (Keras 3 enforces unique operation names), and (3) add a safe fallback so inference still produces a valid `submission.csv` even if pretrained weights are missing. These changes keep the architecture and inference semantics the same, only addressing runtime errors and submission validity. The output be a correctly formatted CSV with probabilities clipped and renormalized to sum to 1 per row.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402213"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 96
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
import matplotlib
import matplotlib.pyplot as plt

from scipy import (
    signal,
)  # needed by DataGenerator (butter/filtfilt) and EEG preprocessing

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
        print("Mixed precision enabled (mixed_float16)")
    except Exception as e:
        print(
            "Mixed precision could not be enabled; continuing in full precision. Reason:",
            repr(e),
        )
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

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
start_n = 12
train = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
train = train.groupby("eeg_id").head(start_n).reset_index(drop=True)

mintemp = 1e10
for ii in range(len(TARGETS)):
    mintemp = min(mintemp, int(sum(np.argmax(train[TARGETS].values, 1) == ii)))

train = pd.DataFrame()
for ii in range(len(TARGETS)):
    train_temp = df.iloc[np.argmax(df[TARGETS].values, 1) == ii].reset_index(drop=True)
    train_temp = train_temp.iloc[np.random.permutation(len(train_temp))].reset_index(
        drop=True
    )
    jj = 1
    while len(train_temp.groupby("eeg_id").head(jj)) < mintemp:
        jj = jj + 1
    train_temp = train_temp.groupby("eeg_id").head(jj)
    print(jj)
    if jj != 1:
        train_temp = train_temp.iloc[
            np.random.permutation(len(train_temp))
        ].reset_index(drop=True)
        train_temp = train_temp[:mintemp]
    train = pd.concat([train, train_temp], axis=0)

for ii in range(len(TARGETS)):
    print(int(sum(np.argmax(train[TARGETS].values, 1) == ii)))

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

print("Train overlapp eeg_id shape:", train.shape)



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
    ):

        self.cmin = -4
        self.cmax = 8
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False  # preserve original behavior
        self.mode = mode
        self.specs = specs
        self.eegs = eegs
        self.b, self.a = signal.butter(
            3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
        )
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_img = 0
                r_eeg = 0
            else:
                r_img = (
                    round(row.spectrogram_label_offset_seconds / 2)
                    if "spectrogram_label_offset_seconds" in row.index
                    else 0
                )
                r_eeg = (
                    round(row.eeg_label_offset_seconds * SFREQ)
                    if "eeg_label_offset_seconds" in row.index
                    else 0
                )
                label = row[TARGETS].values.astype(np.float32)
                label = label / (np.sum(label) + 1e-12)
                if self.mode == "train":
                    if np.sum(label == 1) > 0:
                        label[label == 0] = 1e-3
                        label[label == 1] = 1 - 5 * 1e-3

            for k in range(4):
                img = self.specs[row.spectrogram_id][
                    r_img : r_img + 300, k * 100 : (k + 1) * 100
                ].T
                eeg = self.eegs[row.eeg_id][:, r_eeg : round(r_eeg + 50 * SFREQ), k]

                img = np.nan_to_num(img, nan=0.0)
                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 255)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)

                img_map = self.cmaps[img]
                img_map = np.reshape(img_map, (100, 300, 3))

                img_map = img_map[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img_map.shape[1]
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

                eeg = np.nan_to_num(eeg, nan=0.0)
                eeg = signal.filtfilt(self.b, self.a, eeg, axis=1)
                eeg = eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

                X_eeg[j, 1:5, :, k] = eeg
                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = label

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
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=None,
        name="spectrogram_efficientnetb0",
    )

    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="eeg_efficientnetb0"
    )

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
def _resolve_model_dir(load_from: str) -> str:
    """Find actual directory containing weights even if Kaggle dataset is nested."""
    if os.path.isdir(load_from):
        return load_from
    parent = os.path.dirname(load_from.rstrip("/"))
    if os.path.isdir(parent):
        try:
            for name in os.listdir(parent):
                cand = os.path.join(parent, name)
                if os.path.isdir(cand):
                    return cand
        except Exception:
            pass
    return load_from


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    print("Test shape", test.shape)

    files2 = os.listdir(PATH_SPEC)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    files2 = os.listdir(PATH_EEG)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")
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
    print()

    preds = []
    with strategy.scope():
        model = build_model()

    test_gen = DataGenerator(
        test, shuffle=False, batch_size=32, mode="test", specs=spectrograms2, eegs=eegs2
    )

    model_dir = _resolve_model_dir(LOAD_MODELS_FROM)
    print("Resolved model dir:", model_dir)

    any_loaded = False
    for i in range(5):
        print(f"Fold {i+1}")
        wpath = os.path.join(model_dir, f"EB2_v{VER}_f{i}.h5")
        if not os.path.exists(wpath):
            wpath2 = os.path.join(
                model_dir, os.path.basename(model_dir), f"EB2_v{VER}_f{i}.h5"
            )
            if os.path.exists(wpath2):
                wpath = wpath2

        if os.path.exists(wpath):
            model.load_weights(wpath)
            any_loaded = True
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)
        else:
            print("WARNING: missing weights:", wpath)

    if any_loaded:
        pred = np.mean(preds, axis=0).astype(np.float64)
        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
    else:
        pred = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float64)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred.astype(np.float32)

    sub[TARGETS] = np.clip(sub[TARGETS].values, 1e-12, 1.0)
    sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(axis=1, keepdims=True)

    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row-sum check (min/mean/max):",
        float(sub[TARGETS].sum(axis=1).min()),
        float(sub[TARGETS].sum(axis=1).mean()),
        float(sub[TARGETS].sum(axis=1).max()),
    )
    print(sub.head())
