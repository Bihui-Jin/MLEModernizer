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

0.3430339504409042

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import os, io
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

tf.random.set_seed(2024)
np.random.seed(2024)
os.environ["PYTHONHASHSEED"] = "2024"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

try:
    import cupy as cp
except Exception:
    cp = None  # fallback to None; code will use numpy/scipy where needed

try:
    import efficientnet.tfkeras as efn
except Exception:
    efn = None


PLATFORM = "kaggle"  # local/kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [1, 2, 3]
STAGETEST = 3
threshold = 0.2

EEG_LENGTH = 30  # seconds
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
READ_EXTRA_SPEC_FILES = True
READ_EXTRA_EEG_FILES = True
READ_EXTRA_IMG_FILES = True
READ_EXTRA_STFT_FILES = True

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

if PLATFORM == "local":
    DATA_ROOT = "./input/hms-harmful-brain-activity-classification"
    LOAD_MODELS_FROM = "./input/models2024040301"
else:
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
    LOAD_MODELS_FROM = "/kaggle/input/models2024040301"

train_path = os.path.join(DATA_ROOT, "train.csv")
df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]  # last 6 columns are the vote targets
print("Targets:", list(TARGETS))


class DataGenerator(tf.keras.utils.Sequence):
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
        return self.__data_generation(indexes)

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
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            else:
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = round(row.eeg_label_offset_seconds * SFREQ)

            if "spe" in DATATYPE:
                for k in range(4):
                    spe = self.specs[row.spectrogram_id][
                        r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                    ].T
                    if spe.shape != (100, 300):
                        tmp = np.zeros((100, 300))
                        tmp[: spe.shape[0], : spe.shape[1]] = spe
                        spe = tmp
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)
                    spe = np.round(
                        (spe - self.cmin) / (self.cmax - self.cmin) * 255
                    ).astype(np.int16)
                    spe = self.cmaps[spe]
                    spe = spe[
                        :,
                        round((spe.shape[1] - LENGTH) / 2) : -round(
                            (spe.shape[1] - LENGTH) / 2
                        ),
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
                for k in range(4):
                    eeg = self.eegs[row.eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]
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
                        - np.mean(x_eeg[j, :, :, :, k], axis=1, keepdims=True)
                    ) / (np.std(x_eeg[j, :, :, :, k], axis=1, keepdims=True) + 1e-6)

            if "img" in DATATYPE:
                for k in range(4):
                    if self.mode == "test":
                        img = self.imgs[row.eeg_id][:, :, k, :]
                    else:
                        img = self.imgs[row.sign_id][:, :, k, :]
                    x_img[j, :, :, :, k] = img

            if "stft" in DATATYPE:
                for k in range(4):
                    if self.mode == "test":
                        stft = self.stfts[row.eeg_id][:, :, :, k]
                    else:
                        stft = self.stfts[row.sign_id][:, :, :, k]
                    if stft.shape[1] != 64 or stft.shape[2] != 256:
                        tmp = np.zeros((4, 64, 128))
                        tmp[:, : stft.shape[1], : stft.shape[2]] = stft
                        stft = tmp
                    stft = np.concatenate([stft[i] for i in range(4)], 1)
                    stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                    stft = np.log(stft)
                    stft = np.nan_to_num(stft, nan=0.0)
                    stft = np.round(
                        (stft - self.cmin) / (self.cmax - self.cmin) * 255
                    ).astype(np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (64, 128 * 4, 3))
                    x_stft[j, :, :, :, k] = stft
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test":
                label = row[self.targets].values.astype("float32")
                label = label / len(DATATYPE)  # simple averaging across modalities
                y[j] = label

        inputs = []
        if "spe" in DATATYPE:
            inputs.append(x_spe)
        if "eeg" in DATATYPE:
            inputs.append(x_eeg)
        if "img" in DATATYPE:
            inputs.append(x_img)
        if "stft" in DATATYPE:
            inputs.append(x_stft)
        return inputs, y


def build_model(target_names):
    inputs = []
    representations = []

    def get_efficientnet():
        if efn is not None:
            return efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
        else:
            return tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=None
            )

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        splits = [inp_spe[:, :, :, :, i] for i in range(4)]
        x_spe = tf.keras.layers.Concatenate(axis=1)(splits)
        backbone = get_efficientnet()
        try:
            weight_path = os.path.join(LOAD_MODELS_FROM, "efficientnet_spe_weights.h5")
            backbone.load_weights(weight_path)
        except Exception:
            pass
        x_spe = backbone(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.nn.l2_normalize(x_spe, -1)
        inputs.append(inp_spe)
        representations.append(x_spe)

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
        splits = [inp_eeg[:, :, :, :, i] for i in range(4)]
        x_eeg = tf.keras.layers.Concatenate(axis=1)(splits)
        backbone = get_efficientnet()
        try:
            weight_path = os.path.join(LOAD_MODELS_FROM, "efficientnet_eeg_weights.h5")
            backbone.load_weights(weight_path)
        except Exception:
            pass
        x_eeg = backbone(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.nn.l2_normalize(x_eeg, -1)
        inputs.append(inp_eeg)
        representations.append(x_eeg)

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
        splits = [inp_img[:, :, :, :, i] for i in range(4)]
        x_img = tf.keras.layers.Concatenate(axis=1)(splits)
        backbone = get_efficientnet()
        try:
            weight_path = os.path.join(LOAD_MODELS_FROM, "efficientnet_img_weights.h5")
            backbone.load_weights(weight_path)
        except Exception:
            pass
        x_img = backbone(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = tf.nn.l2_normalize(x_img, -1)
        inputs.append(inp_img)
        representations.append(x_img)

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4))
        splits = [inp_stft[:, :, :, :, i] for i in range(4)]
        x_stft = tf.keras.layers.Concatenate(axis=1)(splits)
        backbone = get_efficientnet()
        try:
            weight_path = os.path.join(LOAD_MODELS_FROM, "efficientnet_stft_weights.h5")
            backbone.load_weights(weight_path)
        except Exception:
            pass
        x_stft = backbone(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = tf.nn.l2_normalize(x_stft, -1)
        inputs.append(inp_stft)
        representations.append(x_stft)

    if len(representations) > 1:
        combined = tf.keras.layers.Concatenate(axis=1)(representations)
    else:
        combined = representations[0]

    outputs = tf.keras.layers.Dense(
        len(target_names), activation="softmax", dtype="float32"
    )(combined)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if not NEEDTRAIN:
    test_path = os.path.join(DATA_ROOT, "test.csv")
    test = pd.read_csv(test_path)
    print("Test shape:", test.shape)

    spectrograms2 = {}
    if "spe" in DATATYPE:
        spec_dir = os.path.join(DATA_ROOT, "test_spectrograms")
        for f in os.listdir(spec_dir):
            if f.endswith(".parquet"):
                tmp = pd.read_parquet(os.path.join(spec_dir, f))
                spectrograms2[int(f.split(".")[0])] = tmp.iloc[:, 1:].values

    from scipy import signal

    eegs2, imgs2, stfts2 = {}, {}, {}
    eeg_dir = os.path.join(DATA_ROOT, "test_eegs")
    for f in os.listdir(eeg_dir):
        if not f.endswith(".parquet"):
            continue
        name = int(f.split(".")[0])
        if test[test.eeg_id == name].empty:
            continue
        eeg_default = pd.read_parquet(os.path.join(eeg_dir, f))
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        list_eeg, list_img, list_stft = [], [], []
        for region in BRAIN.values():
            eeg = np.zeros((len(region), eeg_default.shape[0]), dtype=np.float32)
            for i, ch in enumerate(region):
                left, right = ch.split("-")
                eeg[i, :] = (eeg_default[left] - eeg_default[right]).values
            eeg[np.isnan(eeg)] = 0
            if 200 != SFREQ:
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
            eeg = signal.filtfilt(b, a, eeg, axis=1)

            start = round((50 - EEG_LENGTH) / 2 * SFREQ)
            stop = round((50 + EEG_LENGTH) / 2 * SFREQ)
            list_img.append(eeg[:, start:stop])

            if "stft" in DATATYPE:
                freqs, times, Sxx = signal.spectrogram(
                    eeg, SFREQ, nperseg=256, noverlap=219, nfft=320
                )
                valid = (freqs > 0) & (freqs <= 20)
                Sxx = Sxx[:, valid, :-1]
                Sxx = Sxx[..., np.newaxis]
                list_stft.append(Sxx)

            list_eeg.append(eeg[..., np.newaxis])

        eegs2[name] = np.concatenate(list_eeg, axis=2)
        if "img" in DATATYPE:
            img = np.stack(list_img, axis=0).astype(
                np.float32
            )  # shape: (regions, time)
            img = np.mean(img, axis=0, keepdims=True)  # collapse regions
            img = np.tile(img[..., np.newaxis], (1, 1, 1, 1))  # mock 4‑channel
            imgs2[name] = img
        if "stft" in DATATYPE and list_stft:
            stfts2[name] = np.concatenate(list_stft, axis=-1)

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

    preds = model.predict(test_gen, verbose=1)

    preds = preds / preds.sum(axis=1, keepdims=True)

    submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    submission[TARGETS] = preds
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print("Submission saved to", submission_path)
    print("First rows of submission:")
    print(submission.head())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
