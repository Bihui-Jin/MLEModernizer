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

0.3381727920590047

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "model2024040401"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

threshold = 0.2

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 128  # 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024

NSPLIT = 5
BATCHSIZE = 16

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

READ_EXTRA_SPEC_FILES = True
READ_EXTRA_EEG_FILES = True
READ_EXTRA_IMG_FILES = True
READ_EXTRA_STFT_FILES = True

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

import os
import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.metrics import confusion_matrix

try:
    import cupy as cp
except Exception:
    cp = None
    print("cupy not available; proceeding without it.")

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
tf.config.experimental.enable_op_determinism()

MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
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
df.head()

try:
    import efficientnet.tfkeras as efn
except Exception:
    from tensorflow.keras.applications import EfficientNetB0 as _EffNetB0

    class _EfnNamespace:
        EfficientNetB0 = _EffNetB0

    efn = _EfnNamespace()
    print(
        "efficientnet.tfkeras not found; using tf.keras.applications EfficientNetB0 instead."
    )


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
        self.data = data
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
                (len(indexes), 6, round(20 * SFREQ), 3, 4), dtype="float32"
            )
            x_eeg2 = np.zeros((len(indexes), 4, round(50 * SFREQ), 4), dtype="float32")
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            row_spe = row
            row_eeg = row
            row_img = row
            row_stft = row

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            else:
                r_spe = round(row_spe.spectrogram_label_offset_seconds / 2)
                r_eeg = round(row_eeg.eeg_label_offset_seconds * SFREQ)

            if self.mode == "train":
                x1 = np.random.rand() * LENGTH / 2
                x2 = np.random.rand() * LENGTH / 2
                x_spe_min1 = round(min(x1, x2))
                x_spe_max1 = round(max(x1, x2))
                x1 = np.random.rand() * LENGTH / 2 + LENGTH / 2
                x2 = np.random.rand() * LENGTH / 2 + LENGTH / 2
                x_spe_min2 = round(min(x1, x2))
                x_spe_max2 = round(max(x1, x2))

            for k in range(4):
                if "spe" in DATATYPE:
                    spe = self.specs[row_spe.spectrogram_id][
                        r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                    ].T
                    if (spe.shape[0] != 100) or (spe.shape[1] != 300):
                        spe2 = np.zeros((100, 300))
                        spe2[: spe.shape[0], : spe.shape[1]] = spe
                        spe = spe2
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)
                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
                    spe = np.reshape(spe, (-1,))
                    spe = np.array(spe, dtype=np.int16)
                    spe = self.cmaps[spe]
                    spe = np.reshape(spe, (100, 300, 3))
                    spe = spe[
                        :,
                        round((spe.shape[1] - LENGTH) / 2) : -round(
                            (spe.shape[1] - LENGTH) / 2
                        ),
                        :,
                    ]
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
                    eeg = self.eegs[row_eeg.eeg_id][
                        :, r_eeg : r_eeg + round(50 * SFREQ), k
                    ]
                    if eeg.shape[1] < 50 * SFREQ:
                        eeg = np.concatenate((eeg, eeg), 1)[:, : 50 * SFREQ]
                    eeg1 = eeg[:, round(10 * SFREQ) : round(30 * SFREQ)]
                    eeg2 = eeg[:, round(15 * SFREQ) : round(35 * SFREQ)]
                    eeg3 = eeg[:, round(20 * SFREQ) : round(40 * SFREQ)]
                    x_eeg[j, 1:5, :, 0, k] = eeg1
                    x_eeg[j, 1:5, :, 1, k] = eeg2
                    x_eeg[j, 1:5, :, 2, k] = eeg3
                    x_eeg[j, :, :, :, k] = (
                        x_eeg[j, :, :, :, k]
                        - np.mean(x_eeg[j, :, :, :, k], 1, keepdims=True)
                    ) / (np.std(x_eeg[j, :, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    if self.mode == "test":
                        img = self.imgs[row_img.eeg_id][:, :, k, :]
                    else:
                        img = self.imgs[row_img.sign_id][:, :, k, :]
                    x_img[j, :, :, :, k] = img

                if "stft" in DATATYPE:
                    if self.mode == "test":
                        stft = self.stfts[row_stft.eeg_id][:, :, :, k]
                    else:
                        stft = self.stfts[row_stft.sign_id][:, :, :, k]
                    if (stft.shape[1] != 64) or (stft.shape[2] != 256):
                        stft2 = np.zeros((4, 64, 128))
                        stft2[:, : stft.shape[1], : stft.shape[2]] = stft
                        stft = stft2
                    stft = np.concatenate([stft[0], stft[1], stft[2], stft[3]], 1)
                    stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                    stft = np.log(stft)
                    stft = np.nan_to_num(stft, nan=0.0)
                    stft = np.round((stft - self.cmin) / (self.cmax - self.cmin) * 255)
                    stft = np.reshape(stft, (-1,))
                    stft = np.array(stft, dtype=np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (64, 128 * 4, 3))
                    x_stft[j, :, :, :, k] = stft
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test":
                label = 0
                for src in ["spe", "eeg", "img", "stft"]:
                    if src in DATATYPE:
                        label = label + row_spe[self.targets].values
                label = label / len(DATATYPE)
                if self.mode == "train" and np.sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        if "eeg" in DATATYPE:
            for i_eeg in range(x_eeg2.shape[0]):
                xx = np.std(x_eeg2[i_eeg], 1, keepdims=True)
                xx = np.mean(xx)
                x_eeg2[i_eeg] = (
                    x_eeg2[i_eeg] - np.mean(x_eeg2[i_eeg], 1, keepdims=True)
                ) / (xx + 1e-6)

        x = []
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "img" in DATATYPE:
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)
        if "stft" in DATATYPE:
            x.append(x_stft)
        return x, y


def build_model(TARGETS_PRETRAIN):
    inp = []
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [inp_spe[:, :, :, :, i] for i in range(4)]
        )
        base_model_spe = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/"
                "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if PLATFORM == "local" and os.path.exists(weight_path):
                base_model_spe.load_weights(weight_path)
            elif PLATFORM == "kaggle":
                weight_path = (
                    "/kaggle/input/tf-efficientnet-imagenet-weights/"
                    "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
                if os.path.exists(weight_path):
                    base_model_spe.load_weights(weight_path)
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.nn.l2_normalize(x_spe, -1)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
        x_eeg = tf.keras.layers.Concatenate(axis=1)(
            [inp_eeg[:, :, :, :, i] for i in range(4)]
        )
        base_model_eeg = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/"
                "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if PLATFORM == "local" and os.path.exists(weight_path):
                base_model_eeg.load_weights(weight_path)
            elif PLATFORM == "kaggle":
                weight_path = (
                    "/kaggle/input/tf-efficientnet-imagenet-weights/"
                    "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
                if os.path.exists(weight_path):
                    base_model_eeg.load_weights(weight_path)
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.nn.l2_normalize(x_eeg, -1)
        inp.append(inp_eeg)
        y = (
            tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            if "spe" in DATATYPE
            else x_eeg
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
        x_img = tf.keras.layers.Concatenate(axis=1)(
            [
                inp_img[:, :, :, :, 0],
                inp_img[:, :, :, :, 2],
                inp_img[:, :, :, :, 3],
                inp_img[:, :, :, :, 1],
            ]
        )
        base_model_img = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/"
                "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if PLATFORM == "local" and os.path.exists(weight_path):
                base_model_img.load_weights(weight_path)
            elif PLATFORM == "kaggle":
                weight_path = (
                    "/kaggle/input/tf-efficientnet-imagenet-weights/"
                    "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
                if os.path.exists(weight_path):
                    base_model_img.load_weights(weight_path)
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = tf.nn.l2_normalize(x_img, -1)
        inp.append(inp_img)
        y = (
            tf.keras.layers.Concatenate(axis=1)([y, x_img])
            if ("spe" in DATATYPE or "eeg" in DATATYPE)
            else x_img
        )

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4))
        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [inp_stft[:, :, :, :, i] for i in range(4)]
        )
        base_model_stft = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        if NEEDTRAIN:
            weight_path = (
                "./input/tf-efficientnet-imagenet-weights/"
                "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
            if PLATFORM == "local" and os.path.exists(weight_path):
                base_model_stft.load_weights(weight_path)
            elif PLATFORM == "kaggle":
                weight_path = (
                    "/kaggle/input/tf-efficientnet-imagenet-weights/"
                    "efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
                if os.path.exists(weight_path):
                    base_model_stft.load_weights(weight_path)
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = tf.nn.l2_normalize(x_stft, -1)
        inp.append(inp_stft)
        y = (
            tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            if any(t in DATATYPE for t in ["spe", "eeg", "img"])
            else x_stft
        )

    y = tf.keras.layers.Dense(
        len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
    )(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    if "spe" in DATATYPE:
        path2 = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            if PLATFORM == "local"
            else "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        files2 = os.listdir(path2)
        print(f"There are {len(files2)} test spectrogram parquets")
        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(os.path.join(path2, f))
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values

    from scipy import signal

    path2 = (
        "./input/hms-harmful-brain-activity-classification/test_eegs/"
        if PLATFORM == "local"
        else "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    )
    files2 = os.listdir(path2)
    print(f"There are {len(files2)} test eeg parquets")
    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(os.path.join(path2, f))
        name = int(f.split(".")[0])
        if len(test[test.eeg_id == name]) == 0:
            continue

        list_eeg, list_img, list_stft = [], [], []
        for region in BRAIN.keys():
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(BRAIN[region]):
                eeg[chan_i, :] = (
                    eeg_default.loc[:, chan.split("-")[0]]
                    - eeg_default.loc[:, chan.split("-")[1]]
                ).values
            eeg[np.isnan(eeg)] = 0
            if 200 != SFREQ:
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
            eeg = signal.filtfilt(b, a, eeg, axis=1)

            time_start = round((50 - EEG_LENGTH) / 2 * SFREQ)
            time_stop = round((50 + EEG_LENGTH) / 2 * SFREQ)

            list_img.append(eeg[:, time_start:time_stop])

            if "stft" in DATATYPE:
                frequencies, times, Sxx = signal.spectrogram(
                    eeg, SFREQ, nperseg=256, noverlap=219, nfft=320
                )
                valid_freq = (frequencies > 0.0) & (frequencies <= 20)
                Sxx_filtered = Sxx[:, valid_freq, :-1]
                Sxx_filtered = np.reshape(
                    Sxx_filtered,
                    (
                        Sxx_filtered.shape[0],
                        Sxx_filtered.shape[1],
                        Sxx_filtered.shape[2],
                        1,
                    ),
                )
                list_stft.append(Sxx_filtered)

            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)
        if "stft" in DATATYPE:
            stfts2[name] = np.concatenate(list_stft, -1)
        if "eeg" in DATATYPE:
            eegs2[name] = list_eeg
        if "img" in DATATYPE:
            eeg_all_region = np.concatenate(list_img, 0)

            def make_img(region_eeg, amp, crop_frac=None):
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                for ii in range(region_eeg.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(region_eeg[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-10, region_eeg.shape[1] + 10)
                plt.ylim(-amp / 2, region_eeg.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                img = np.array(img)[:, :, :1]
                buf.truncate()
                plt.close("all")
                img = np.concatenate((img, img, img), 2)
                img = np.array(
                    tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img = img[:, :, 0:1]
                if crop_frac:
                    start, end = crop_frac
                    img = img[:, :, start:end]
                return img

            img = make_img(eeg_all_region, amp=200)
            img2 = make_img(
                eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 1 / 4) : round(
                        eeg_all_region.shape[1] * 3 / 4
                    ),
                ],
                amp=150,
            )
            img3 = make_img(
                eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 2 / 5) : round(
                        eeg_all_region.shape[1] * 3 / 5
                    ),
                ],
                amp=100,
            )
            img = np.concatenate([img, img2, img3], -1)
            imgs2[name] = img

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

    preds = []
    for i in range(NSPLIT):
        print(f"Fold {i + 1}")
        weight_file = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
        if os.path.exists(weight_file):
            model.load_weights(weight_file)
        else:
            print(f"Weight file {weight_file} not found – using random init.")
        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    print("\nTest preds shape", pred.shape)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print("Row probability sums (first 5 rows):")
    print(sub.iloc[:, -6:].sum(axis=1).head())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
