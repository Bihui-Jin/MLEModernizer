# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024032603"
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
NSPLIT = 5
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



## === cell 1
import sys
import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd, numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix  # noqa: F401

try:
    import librosa  # noqa: F401
except Exception as e:
    print("Warning: librosa import failed (not needed for inference):", repr(e))

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
    print("Determinism not fully enabled:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision option not available:", repr(e))
else:
    print("Using full precision")



## === cell 2
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



## === cell 3
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
        train = pd.concat([train, df3]).reset_index(drop=True)
        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.head()
        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 4
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
        targets=None,
    ):

        self.targets = targets
        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["cividis"](np.linspace(0, 1, 256))[:, :3]

        self.data = data.reset_index(drop=True)
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
            x_eeg = np.zeros((len(indexes), 4, round(50 * SFREQ), 4), dtype="float32")
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            elif self.mode == "valid":
                r_spe = (
                    round(row.spectrogram_label_offset_seconds / 2)
                    if "spectrogram_label_offset_seconds" in row
                    else 0
                )
                r_eeg = (
                    round(row.eeg_label_offset_seconds * SFREQ)
                    if "eeg_label_offset_seconds" in row
                    else 0
                )
            else:
                r_spe = (
                    round(row.spectrogram_label_offset_seconds / 2)
                    if "spectrogram_label_offset_seconds" in row
                    else 0
                )
                r_eeg = (
                    round(row.eeg_label_offset_seconds * SFREQ)
                    if "eeg_label_offset_seconds" in row
                    else 0
                )

            if self.mode == "train":
                x1 = np.random.rand() * (LENGTH / 2 - 20)
                x2 = np.random.rand() * (LENGTH / 2 - 20)
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))

                x1 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                x2 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                if np.random.rand() < 0.5:
                    x1 = x1 + EEG_LENGTH * SFREQ / 2
                    x2 = x2 + EEG_LENGTH * SFREQ / 2
                else:
                    x1 = x1 + 10 * SFREQ / 2
                    x2 = x2 + 10 * SFREQ / 2
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(max(x1, x2))

                x1 = np.random.rand() * (LENGTH / 2 - 42)
                x2 = np.random.rand() * (LENGTH / 2 - 42)
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                else:
                    x1 = x1 + 42
                    x2 = x2 + 42
                x_img_min = round(min(x1, x2))
                x_img_max = round(max(x1, x2))

            for k in range(4):
                if "img" in DATATYPE:
                    key = row.eeg_id if self.mode == "test" else row.sign_id
                    if (self.imgs is not None) and (key in self.imgs):
                        img = self.imgs[key][:, :, k, :]
                    else:
                        img = np.zeros((IMG_HIGH, IMG_WIDE, 3), dtype=np.float32)
                    x_img[j, :, :, :, k] = img

            if self.mode != "test":
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0

        x = list()
        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 5
_USE_EFN = False
print(
    "Using tf.keras.applications EfficientNetB0 fallback (efficientnet.tfkeras disabled)."
)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


def build_model(TARGETS_PRETRAIN):
    inp = list()
    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
        x_img1 = inp_img[:, :, :, :, 0]
        x_img2 = inp_img[:, :, :, :, 1]
        x_img3 = inp_img[:, :, :, :, 2]
        x_img4 = inp_img[:, :, :, :, 3]
        x_img = tf.keras.layers.Concatenate(axis=1)([x_img1, x_img2, x_img3, x_img4])

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_img._name = "img_extractor"

        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_img.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_img.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

        y = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )(x_img)

        inp.append(inp_img)

    y = tf.keras.layers.Dense(
        len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
    )(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model


def my_loss(y_ture, y_pred):
    y_pred1 = y_pred[:, 5:6]
    y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
    y_pred2 = y_pred[:, 0:5]
    y_pred = tf.concat((y_pred2, y_pred1), axis=1)
    return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 6
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    print("Test shape", test.shape, "Sample submission shape", sample_sub.shape)

    test = sample_sub[["eeg_id"]].merge(test, on="eeg_id", how="left")
    print("Aligned test shape (to sample_submission):", test.shape)

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    needed_ids = test["eeg_id"].astype(int).to_numpy()
    needed_set = set(needed_ids.tolist())
    print(f"Need to process {len(needed_set)} EEGs")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}

    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    _pairs = []
    for region in ("LL", "RL", "LP", "RP"):
        for pair in BRAIN[region]:
            c0, c1 = pair.split("-")
            _pairs.append((c0, c1))
    _unique_cols = sorted(set([c for p in _pairs for c in p]))
    _col_to_idx = {c: i for i, c in enumerate(_unique_cols)}
    _idx0 = np.array([_col_to_idx[c0] for (c0, c1) in _pairs], dtype=np.int32)
    _idx1 = np.array([_col_to_idx[c1] for (c0, c1) in _pairs], dtype=np.int32)

    def _resize_u8_to_float01(
        img_u8_3c: np.ndarray, out_h: int, out_w: int
    ) -> np.ndarray:
        im = Image.fromarray(img_u8_3c, mode="RGB")
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr

    def _fast_render_eeg_to_img(
        eeg_all_region: np.ndarray, amp: float, xpad: int, ypad_factor: float
    ) -> np.ndarray:
        n_chan, n_t = eeg_all_region.shape
        height = int(n_chan * amp + amp / 2 * ypad_factor)
        width = int(n_t + 2 * xpad)
        canvas = np.zeros((height, width), dtype=np.uint8)

        x = (np.arange(n_t, dtype=np.int32) + xpad)[None, :]  # (1, T)
        offs = (
            np.arange(n_chan, dtype=np.float32) * amp
            + (np.arange(n_chan, dtype=np.float32) // 4) * amp
        )[:, None]
        y = eeg_all_region.astype(np.float32, copy=False) + offs  # (C, T)
        y = np.rint(y).astype(np.int32, copy=False)
        y = np.clip(y, 0, height - 1)
        x = np.clip(x, 0, width - 1)
        canvas[y, x] = 255
        return canvas

    def _postprocess_canvas_to_stack(canvas: np.ndarray) -> np.ndarray:
        img_u8 = canvas[:, :, None]
        img_u8 = np.repeat(img_u8, 3, axis=2)  # RGB-like for resize
        img = _resize_u8_to_float01(img_u8, IMG_HIGH * 4, IMG_WIDE)  # float01, 3c
        img = img[:, :, 0:1]  # keep one channel like original

        img = np.concatenate(
            [
                img[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                img[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                img[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                img[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        img[:, :, 0] = -img[:, :, 0]
        img[:, :, 2] = -img[:, :, 2]
        img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))
        return img

    for i, eeg_id in enumerate(needed_ids):
        if i % 200 == 0:
            print(i, ", ", end="")

        fpath = os.path.join(PATH2, f"{int(eeg_id)}.parquet")
        if not os.path.exists(fpath):
            continue

        eeg_df = pd.read_parquet(fpath, columns=_unique_cols)
        x = eeg_df.to_numpy(dtype=np.float32, copy=False).T  # (n_cols, T)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        diffs = x[_idx0] - x[_idx1]  # (16, T)

        if 200 != SFREQ:
            diffs = signal.resample_poly(diffs, SFREQ, 200, axis=1)
        diffs = signal.filtfilt(b, a, diffs, axis=1)

        time_temp = 0
        time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
        time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)
        eeg_all_region = diffs[:, time_start:time_stop]

        if "img" in DATATYPE:
            canvas1 = _fast_render_eeg_to_img(
                eeg_all_region, amp=200, xpad=10, ypad_factor=5.0
            )
            img1 = _postprocess_canvas_to_stack(canvas1)

            eeg_all_region2 = eeg_all_region[
                :,
                round(eeg_all_region.shape[1] * 1 / 4) : round(
                    eeg_all_region.shape[1] * 3 / 4
                ),
            ]
            canvas2 = _fast_render_eeg_to_img(
                eeg_all_region2, amp=150, xpad=5, ypad_factor=5.0
            )
            img2 = _postprocess_canvas_to_stack(canvas2)

            eeg_all_region3 = eeg_all_region[
                :,
                round(eeg_all_region.shape[1] * 2 / 5) : round(
                    eeg_all_region.shape[1] * 3 / 5
                ),
            ]
            canvas3 = _fast_render_eeg_to_img(
                eeg_all_region3, amp=100, xpad=2, ypad_factor=5.0
            )
            img3 = _postprocess_canvas_to_stack(canvas3)

            img = np.concatenate([img1, img2, img3], axis=-1)
            imgs2[int(eeg_id)] = img

    def _find_model_dir(requested: str) -> str | None:
        if (requested is not None) and os.path.isdir(requested):
            return requested

        base = "/kaggle/input"
        token = os.path.basename(str(requested).rstrip("/"))
        try:
            ds = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        except Exception:
            return None

        candidates = []
        for d in ds:
            root = os.path.join(base, d)
            if token.lower() in d.lower():
                candidates.append(root)
            nested = os.path.join(root, token)
            if os.path.isdir(nested):
                candidates.append(nested)

        for c in candidates:
            ok = True
            for i_fold in range(NSPLIT):
                wpath = os.path.join(c, f"f{i_fold}_stage{STAGETEST}.h5")
                if not os.path.exists(wpath):
                    ok = False
                    break
            if ok:
                return c

        return candidates[0] if candidates else None

    if PLATFORM == "kaggle":
        found = _find_model_dir(LOAD_MODELS_FROM)
        if found is None:
            print(
                f"Warning: Model directory not found for '{LOAD_MODELS_FROM}'. Will write uniform submission."
            )
            LOAD_MODELS_FROM = None
        else:
            print("Using model directory:", found)
            LOAD_MODELS_FROM = found

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})

    if LOAD_MODELS_FROM is None:
        uniform = np.full(
            (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
        sub[TARGETS] = uniform
    else:
        preds = []
        with strategy.scope():
            model = build_model(TARGETS)

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            imgs=imgs2,
            stfts=stfts2,
            targets=TARGETS,
        )

        for i_fold in range(NSPLIT):
            print(f"Fold {i_fold + 1}")
            wpath = os.path.join(LOAD_MODELS_FROM, f"f{i_fold}_stage{STAGETEST}.h5")
            if not os.path.exists(wpath):
                raise FileNotFoundError(f"Missing weight file: {wpath}")
            model.load_weights(wpath)
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        sub[TARGETS] = pred

    sub = sample_sub[["eeg_id"]].merge(sub, on="eeg_id", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(1.0 / len(TARGETS)).astype(np.float32)
    sub[TARGETS] = sub[TARGETS].clip(1e-8, 1.0)
    sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row-sum stats:",
        float(sub.iloc[:, -6:].sum(axis=1).min()),
        float(sub.iloc[:, -6:].sum(axis=1).max()),
    )
    sub.head()
