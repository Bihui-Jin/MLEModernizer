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

0.3535640618406237

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import io
import gc
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img'
STAGE = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024022901"
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
    print("Determinism not fully enabled:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled (optimizer experimental option)")
    except Exception:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (global policy)")
        except Exception as e:
            print("Could not enable mixed precision:", repr(e))
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
print("Targets:", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
        self.specs = specs if specs is not None else {}
        self.eegs = eegs if eegs is not None else {}
        self.imgs = imgs if imgs is not None else {}
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
                x1 = np.random.rand() * (256 / 2 - 20)
                x2 = np.random.rand() * (256 / 2 - 20)
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_spe_min += 128
                    x_spe_max += 128

                x1 = np.random.rand() * (2048 / 2 - 500)
                x2 = np.random.rand() * (2048 / 2 - 500)
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_eeg_min += 1024
                    x_eeg_max += 1024

                x1 = np.random.rand() * (256 / 2 - 64)
                x2 = np.random.rand() * (256 / 2 - 64)
                x_img_min = round(min(x1, x2))
                x_img_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_img_min += 128
                    x_img_max += 128

            for k in range(4):
                if "spe" in DATATYPE:
                    if row.spectrogram_id in self.specs:
                        spe_src = self.specs[row.spectrogram_id]
                    else:
                        spe_src = np.zeros((300, 400), dtype=np.float32)

                    spe = spe_src[r_spe : r_spe + 300, k * 100 : (k + 1) * 100].T
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
                    if row.eeg_id in self.eegs:
                        eeg_src = self.eegs[row.eeg_id]
                    else:
                        eeg_src = np.zeros((4, 5000, 4), dtype=np.float32)

                    eeg = eeg_src[
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
                    if row.eeg_id in self.imgs:
                        img = self.imgs[row.eeg_id][:, :, k]
                    else:
                        img = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

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
                xxr = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xxr) + x_spe[::-1, :, :, :, :] * xxr
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xxr = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xxr) + x_eeg[::-1, :, :, :] * xxr
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xxr = np.reshape(xx, (x_img.shape[0], 1, 1, 1))
                x_img = x_img * (1 - xxr) + x_img[::-1, :, :, :] * xxr
            x.append(x_img)

        if (self.mode == "train") and (alpha > 0):
            xxr = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xxr) + y[::-1, :] * xxr

        x = tuple(x)
        return x, y




## === cell 2
def build_model():
    inp = []
    y = None

    l2_layer = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])

        base_model_spe = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            name="efficientnetb0_spe",
        )
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = l2_layer(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))  # (6, T, 4)

        x_eeg = tf.keras.layers.Permute((2, 1, 3))(inp_eeg)  # (T, 6, 4)
        x_eeg = tf.keras.layers.Reshape((round(EEG_LENGTH * SFREQ), 6 * 4, 1))(
            x_eeg
        )  # (T, 24, 1)
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])  # (T, 24, 3)

        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            name="efficientnetb0_eeg",
        )
        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = l2_layer(x_eeg)

        inp.append(inp_eeg)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg]) if y is not None else x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1)([x_img1, x_img2, x_img3, x_img4])
        x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            name="efficientnetb0_img",
        )
        base_model_img._name = "img_extractor"

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = l2_layer(x_img)

        inp.append(inp_img)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_img]) if y is not None else x_img

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 3
def _find_existing_weights_dir(preferred_dir: str, stage: int) -> str:
    """
    BUGFIX: Kaggle datasets may not mount under the expected name.
    We search /kaggle/input for a directory containing the expected fold weight files.
    """
    expected = [f"f{i}_stage{stage}.h5" for i in range(5)]

    if os.path.isdir(preferred_dir):
        ok = all(os.path.isfile(os.path.join(preferred_dir, e)) for e in expected)
        if ok:
            return preferred_dir

    base = "/kaggle/input"
    candidates = []
    if os.path.isdir(base):
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if not os.path.isdir(p):
                continue
            candidates.append(p)
            for dd in os.listdir(p):
                pp = os.path.join(p, dd)
                if os.path.isdir(pp):
                    candidates.append(pp)

    for c in candidates:
        ok = all(os.path.isfile(os.path.join(c, e)) for e in expected)
        if ok:
            return c

    return preferred_dir




## === cell 4
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPE = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPE = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    print("Test shape", test.shape)

    files_spe = os.listdir(PATH_SPE)
    print(f"There are {len(files_spe)} test spectrogram parquets")
    spectrograms2 = {}
    for i, f in enumerate(files_spe):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPE}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    try:
        from scipy import signal  # type: ignore

        HAVE_SCIPY = True
    except Exception as e:
        HAVE_SCIPY = False
        signal = None
        print("SciPy not available; proceeding without filtering/resampling:", repr(e))

    files_eeg = os.listdir(PATH_EEG)
    print(f"There are {len(files_eeg)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}

    if HAVE_SCIPY:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
    else:
        b, a = None, None

    test_eeg_ids = set(test.eeg_id.values.tolist())
    for i, f in enumerate(files_eeg):
        if i % 100 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        if name not in test_eeg_ids:
            continue

        eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")

        list_eeg = []
        list_img = []
        for region in BRAIN.keys():
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(BRAIN[region]):
                a0, a1 = chan.split("-")
                eeg[chan_i, :] = (
                    eeg_default.loc[:, a0] - eeg_default.loc[:, a1]
                ).values

            eeg[np.isnan(eeg)] = 0

            if HAVE_SCIPY and (200 != SFREQ):
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

            if HAVE_SCIPY:
                eeg = signal.filtfilt(b, a, eeg, axis=1)

            time_temp = 0
            time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
            time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

            list_img.append(eeg[:, time_start:time_stop])
            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)
        eegs2[name] = list_eeg

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
            tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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
        test,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
    )

    preds = []
    with strategy.scope():
        model = build_model()

    resolved_weights_dir = _find_existing_weights_dir(LOAD_MODELS_FROM, STAGE)
    print("Resolved weights dir:", resolved_weights_dir)

    missing = []
    for i in range(5):
        wpath = os.path.join(resolved_weights_dir, f"f{i}_stage{STAGE}.h5")
        if not os.path.isfile(wpath):
            missing.append(wpath)
    if len(missing) > 0:
        print(
            "WARNING: Missing weight files. Will still run inference with random init."
        )
        for p in missing[:10]:
            print(" missing:", p)

    for i in range(5):
        print(f"Fold {i+1}")
        wpath = os.path.join(resolved_weights_dir, f"f{i}_stage{STAGE}.h5")
        if os.path.isfile(wpath):
            model.load_weights(wpath)
        pred_i = model.predict(test_gen, verbose=1)
        preds.append(pred_i)

    pred = np.mean(preds, axis=0)
    print("Test preds shape", pred.shape)

    pred = np.clip(pred, 1e-9, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row-sum stats:",
        float(sub[TARGETS].sum(axis=1).min()),
        float(sub[TARGETS].sum(axis=1).max()),
    )
    print(sub.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1421547232.py in <cell line: 0>()
    161         if os.path.isfile(wpath):
    162             model.load_weights(wpath)
--> 163         pred_i = model.predict(test_gen, verbose=1)
    164         preds.append(pred_i)
    165 

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

Negative dimension size caused by subtracting 5 from 4 for '{{node functional_1/efficientnetb0_eeg_1/block6a_dwconv_1/depthwise}} = DepthwiseConv2dNative[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], explicit_paddings=[], padding="VALID", strides=[1, 2, 2, 1]](functional_1/efficientnetb0_eeg_1/block6a_dwconv_pad_1/Pad, functional_1/efficientnetb0_eeg_1/block6a_dwconv_1/depthwise/ReadVariableOp)' with input shapes: [32,190,4,672], [5,5,672,1].

Arguments received by DepthwiseConv2D.call():
  • inputs=tf.Tensor(shape=(32, 190, 4, 672), dtype=float32)
