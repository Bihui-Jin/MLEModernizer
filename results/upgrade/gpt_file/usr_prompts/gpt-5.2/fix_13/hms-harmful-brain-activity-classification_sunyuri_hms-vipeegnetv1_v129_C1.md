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

0.3741418941078759

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting, which is incompatible with the protobuf version in this Kaggle image and triggers the `MessageFactory.GetPrototype` error. Then I fix the missing weights issue by auto-discovering the actual `.h5` files in the provided input directory (including nested folders) and mapping them by fold/stage so the existing fold-ensemble inference logic can run without changing the model. Finally, I add a safe fallback to output a valid, normalized submission (using the sample submission priors) if weights still cannot be found, ensuring a `.csv` is always produced end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the known safe workaround for the `MessageFactory.GetPrototype` issue in some Kaggle images). I also make the weight discovery more robust by falling back to any available `.h5` file if the expected `f{fold}_stage{STAGETEST}.h5` naming isn’t found, so the model ensemble actually loads weights instead of silently using priors (which is likely why the score is far from the target). Finally, I keep the exact model/data logic intact and only ensure predictions are valid probabilities and aligned to `sample_submission.csv` order to avoid submission-format/ordering penalties.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting (it’s what triggers the `MessageFactory.GetPrototype` AttributeError in this environment) and keeping imports otherwise unchanged. Then I make sure the correct dataset paths are used consistently and that `test` is always loaded (even if `spe` is not in `DATATYPE`) so the inference pipeline can run end-to-end. Finally, I keep the existing model and inference logic intact but add a tiny, metric-safe probability “floor” before normalization to avoid overconfident zeros that can inflate KL divergence, which should move the score downward toward your target without changing the core approach. The script always write a valid `submission.csv` with the required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation **before** importing TensorFlow (the current code imports TF after unsetting these env vars, which triggers the `MessageFactory.GetPrototype` error in this Kaggle image). I also ensure inference actually runs for `spe/eeg/img` by setting the read flags automatically based on `DATATYPE`; otherwise the generator key-error because the dicts are empty. Finally, I keep your model and preprocessing logic intact but add a tiny probability floor (already present) and enforce strict row-wise normalization for a valid KL submission; this should also nudge the score downward from the current overly-bad output that likely came from missing/empty inputs.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by not forcing the pure-Python protobuf runtime (this env var combination triggers the `MessageFactory.GetPrototype` error in this Kaggle image) and by importing TensorFlow only after setting only safe TF env vars. Then I make the inference actually use the model outputs that match the competition’s 6 target columns: the current `my_loss` (and likely the saved heads) imply the model outputs 7 classes and must be merged back to 6 for submission, but the current code submits raw outputs directly, which badly hurts KL. Finally, I keep the model/data pipeline intact and only add a small probability floor + strict renormalization after the 7→6 merge to avoid zeros and ensure valid probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* importing TensorFlow (this specific `MessageFactory.GetPrototype` error is triggered by the compiled protobuf runtime in some Kaggle images). Then I keep your model and inference pipeline intact but ensure predictions are calibrated for the KL metric by always merging any 7-class output down to the required 6 classes and applying a tiny probability floor + strict renormalization (this is score-improving and submission-safe). Finally, I keep the existing weight auto-discovery but make it robust to nested directories and ensure we always write a valid `submission.csv` with the correct columns and order matching `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf environment variables that are triggering `MessageFactory.GetPrototype` in this Kaggle image, while keeping the rest of the pipeline unchanged. Then I ensure the model is compiled with the existing `my_loss` before calling `predict`, because some TF/Keras builds can error or behave inconsistently on weight-loading/predict without a compiled graph. Finally, to move the KL score down toward the target without changing the model’s core logic, I apply a very small, metric-safe probability floor (smaller than the current 1e-4) and strict renormalization so no class gets near-zero probability (which can strongly penalize KL).'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import io
import re
import warnings
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

PLATFORM = "kaggle"
NEEDTRAIN = False

DATATYPE = ["eeg", "spe", "img"]  # keep core logic unchanged
STAGETRAIN = [2, 3]
STAGETEST = 3

LOAD_MODELS_FROM = "models2024032401"
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
BATCHSIZE = 16

READ_SPEC_FILES = "spe" in DATATYPE
READ_EEG_FILES = "eeg" in DATATYPE
READ_IMG_FILES = "img" in DATATYPE
READ_STFT_FILES = "stft" in DATATYPE

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
except Exception:
    pass

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception:
        print("Mixed precision option not available; continuing")
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
print("Targets", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import librosa
from scipy import signal

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

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False  # preserve original behavior
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
                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]
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

                    if self.mode == "train":
                        x_eeg[j, :, x_eeg_min:x_eeg_max, :, k] = 0

                if "spe" in DATATYPE:
                    spe = self.specs[row.spectrogram_id][
                        r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                    ].T
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)

                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
                    spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                    spe = np.array(spe, dtype=np.int16)
                    spe = self.cmaps[spe]
                    spe = np.reshape(spe, (100, 300, 3))
                    spe = spe[
                        :,
                        max(round((600 / 2 - 256) / 2), 0) : min(
                            (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
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
                    for ch in range(3):
                        x_spe[j, :, :, ch, k] = (
                            x_spe[j, :, :, ch, k] - np.mean(x_spe[j, :, :, ch, k])
                        ) / ((np.std(x_spe[j, :, :, ch, k]) + 1e-6) ** 1)

                if "img" in DATATYPE:
                    img = self.imgs[row.eeg_id][:, :, k, :]
                    if self.mode == "train":
                        img[:, x_img_min:x_img_max, :] = 0
                    x_img[j, :, :, :, k] = img

                if "stft" in DATATYPE:
                    stft = self.stfts[row.eeg_id][:, :, k]
                    stft = np.nan_to_num(stft, nan=0.0)
                    stft = (stft - np.mean(stft)) / (np.std(stft) + 1e-6)
                    cmin = np.min(stft)
                    cmax = np.max(stft)
                    stft = np.clip(stft, cmin, cmax)
                    stft = np.round((stft - cmin) / (cmax - cmin + 1e-6) * 255)

                    shape0, shape1 = stft.shape[0], stft.shape[1]
                    stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                    stft = np.array(stft, dtype=np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (shape0, shape1, 3))
                    stft = np.array(
                        tf.image.resize(stft, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

                    x_stft[
                        j,
                        round((HIGH - stft.shape[0]) / 2) : round(
                            (HIGH + stft.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = stft
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test":
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0
        x = []

        if "spe" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xx) + x_spe[::-1, :, :, :, :] * xx
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xx) + x_eeg[::-1, :, :, :] * xx
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)

        if "stft" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_stft.shape[0], 1, 1, 1, 1))
                x_stft = x_stft * (1 - xx) + x_stft[::-1, :, :, :, :] * xx
            x.append(x_stft)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 2
def _l2norm_layer(axis=-1, name=None):
    return tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=axis), name=name)


def _efficientnet_b0_backbone(name: str):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # keep consistent with original inference which loads fold weights afterward
        input_shape=None,
        name=name,
    )
    return base


def build_model(TARGETS_PRETRAIN):
    inp = []

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe = tf.keras.layers.Concatenate(axis=1, name="spe_concat")(
            [inp_spe[:, :, :, :, i] for i in range(4)]
        )
        base_model_spe = _efficientnet_b0_backbone("spe_efficientnetb0")
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="spe_gap")(x_spe)
        x_spe = _l2norm_layer(axis=-1, name="spe_l2norm")(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4), name="inp_eeg")
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="eeg_concat")(
            [inp_eeg[:, :, :, :, i] for i in range(4)]
        )
        base_model_eeg = _efficientnet_b0_backbone("eeg_efficientnetb0")
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="eeg_gap")(x_eeg)
        x_eeg = _l2norm_layer(axis=-1, name="eeg_l2norm")(x_eeg)
        inp.append(inp_eeg)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="spe_eeg_merge")([y, x_eeg])
            if ("spe" in DATATYPE)
            else x_eeg
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
        x_img = tf.keras.layers.Concatenate(axis=1, name="img_concat")(
            [inp_img[:, :, :, :, i] for i in range(4)]
        )
        base_model_img = _efficientnet_b0_backbone("img_efficientnetb0")
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="img_gap")(x_img)
        x_img = _l2norm_layer(axis=-1, name="img_l2norm")(x_img)
        inp.append(inp_img)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="prev_img_merge")([y, x_img])
        else:
            y = x_img

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
        x_stft = tf.keras.layers.Concatenate(axis=1, name="stft_concat")(
            [inp_stft[:, :, :, :, i] for i in range(4)]
        )
        base_model_stft = _efficientnet_b0_backbone("stft_efficientnetb0")
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="stft_gap")(x_stft)
        x_stft = _l2norm_layer(axis=-1, name="stft_l2norm")(x_stft)
        inp.append(inp_stft)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="prev_stft_merge")([y, x_stft])
        else:
            y = x_stft

    y = tf.keras.layers.Dense(
        len(TARGETS_PRETRAIN),
        activation="softmax",
        dtype="float32",
        name="head_softmax",
    )(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    return model


def my_loss(y_ture, y_pred):
    y_pred1 = y_pred[:, 5:6]
    y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
    y_pred2 = y_pred[:, 0:5]
    y_pred = tf.concat((y_pred2, y_pred1), axis=1)
    return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 3
if PLATFORM == "local":
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    PATH2_SPE = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    PATH2_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    PATH2_SPE = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    PATH2_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print("Test shape", test.shape)

if (not NEEDTRAIN) and ("spe" in DATATYPE) and READ_SPEC_FILES:
    files2 = os.listdir(PATH2_SPE)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2_SPE}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

if (not NEEDTRAIN) and (READ_EEG_FILES or READ_IMG_FILES or READ_STFT_FILES):
    files2 = os.listdir(PATH2_EEG)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}

    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2_EEG}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = []
            list_img = []
            list_stft = []

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

                time_temp = 0
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE and READ_STFT_FILES:
                    mel_spec = librosa.feature.melspectrogram(
                        y=eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        sr=SFREQ,
                        hop_length=round(50 * SFREQ / 256),
                        n_fft=512,
                        n_mels=128,
                        fmin=0,
                        fmax=40,
                        win_length=128,
                    )
                    mel_spec = np.mean(mel_spec, 0)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.min).astype(
                        np.float32
                    )
                    list_stft.append(
                        np.reshape(
                            mel_spec_db, (mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                        )
                    )

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if "stft" in DATATYPE and READ_STFT_FILES:
                list_stft = np.concatenate(list_stft, 2)
                stfts2[name] = list_stft

            if "eeg" in DATATYPE and READ_EEG_FILES:
                eegs2[name] = list_eeg

            if "img" in DATATYPE and READ_IMG_FILES:
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
                    tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
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
                img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))

                eeg_all_region2 = eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 1 / 4) : round(
                        eeg_all_region.shape[1] * 3 / 4
                    ),
                ]
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 150
                for ii in range(eeg_all_region2.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-5, eeg_all_region2.shape[1] + 5)
                plt.ylim(-amp / 2, eeg_all_region2.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img2 = Image.open(byte_stream)
                img2 = np.array(img2)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img2 = np.concatenate((img2, img2, img2), 2)
                img2 = np.array(
                    tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img2 = img2[:, :, 0:1]
                img2 = np.concatenate(
                    [
                        img2[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img2[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img2[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img2[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )
                img2[:, :, 0] = -img2[:, :, 0]
                img2[:, :, 2] = -img2[:, :, 2]
                img2 = np.reshape(
                    img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1)
                )

                eeg_all_region3 = eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 2 / 5) : round(
                        eeg_all_region.shape[1] * 3 / 5
                    ),
                ]
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 100
                for ii in range(eeg_all_region3.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-2, eeg_all_region3.shape[1] + 2)
                plt.ylim(-amp / 2, eeg_all_region3.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img3 = Image.open(byte_stream)
                img3 = np.array(img3)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img3 = np.concatenate((img3, img3, img3), 2)
                img3 = np.array(
                    tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img3 = img3[:, :, 0:1]
                img3 = np.concatenate(
                    [
                        img3[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img3[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img3[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img3[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )
                img3[:, :, 0] = -img3[:, :, 0]
                img3[:, :, 2] = -img3[:, :, 2]
                img3 = np.reshape(
                    img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1)
                )

                img = np.concatenate([img, img2, img3], -1)
                imgs2[name] = img

    print()




## === cell 4
def _resolve_weights_dir(base_path: str) -> str:
    if not os.path.isdir(base_path):
        return base_path
    try:
        base_files = os.listdir(base_path)
        if any(fn.endswith(".h5") for fn in base_files):
            return base_path
    except Exception:
        return base_path

    best = None
    for root, dirs, files in os.walk(base_path):
        if any(fn.endswith(".h5") for fn in files):
            best = root
            break
        rel = os.path.relpath(root, base_path)
        if rel != "." and rel.count(os.sep) >= 2:
            dirs[:] = []
    return best if best is not None else base_path


def _discover_weight_files(base_path: str):
    found = []
    if not os.path.exists(base_path):
        return found
    for root, _, files in os.walk(base_path):
        for fn in files:
            if fn.endswith(".h5"):
                found.append(os.path.join(root, fn))
    return sorted(found)


def _parse_fold_stage(path: str):
    bn = os.path.basename(path)
    m = re.search(r"f(\d+)_stage(\d+)\.h5$", bn)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _fallback_weights_by_fold(all_h5_paths):
    fb = {}
    for p in all_h5_paths:
        bn = os.path.basename(p)
        m = re.search(r"f(\d+)_stage(\d+)\.h5$", bn)
        if m:
            fold = int(m.group(1))
            stage = int(m.group(2))
            fb.setdefault(fold, []).append((stage, p))
    pick = {}
    for fold, lst in fb.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        pick[fold] = lst_sorted[-1][1]
    return pick


def merge_pred_to_6cols(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    if pred.ndim != 2:
        raise ValueError(f"Unexpected pred ndim: {pred.ndim}")
    if pred.shape[1] == 6:
        return pred
    if pred.shape[1] == 7:
        other = np.sum(pred[:, 5:7], axis=1, keepdims=True)
        return np.concatenate([pred[:, 0:5], other], axis=1)
    raise ValueError(
        f"Unexpected number of model outputs: {pred.shape[1]} (expected 6 or 7)"
    )


preds = []

weights_dir = _resolve_weights_dir(LOAD_MODELS_FROM)
all_h5 = _discover_weight_files(LOAD_MODELS_FROM)
print("Weights base:", LOAD_MODELS_FROM)
print("Resolved weights dir (first h5 location heuristic):", weights_dir)
print("Total discovered .h5 files under base:", len(all_h5))
if len(all_h5) > 0:
    print("Example weight file:", all_h5[0])

weight_map = {}
for p in all_h5:
    fs = _parse_fold_stage(p)
    if fs is not None:
        weight_map[fs] = p

fallback_by_fold = _fallback_weights_by_fold(all_h5)

with strategy.scope():
    model = build_model(TARGETS)
    model.compile(optimizer="adam", loss=my_loss)

test_gen = DataGenerator(
    test,
    shuffle=False,
    batch_size=BATCHSIZE * 2,
    mode="test",
    specs=spectrograms2 if "spe" in DATATYPE else None,
    eegs=eegs2 if "eeg" in DATATYPE else None,
    imgs=imgs2 if "img" in DATATYPE else None,
    stfts=stfts2 if "stft" in DATATYPE else None,
    targets=TARGETS,
)

ok_folds = 0
for i in range(5):
    key = (i, STAGETEST)
    wpath = weight_map.get(key, None)
    if wpath is None:
        cand = os.path.join(weights_dir, f"f{i}_stage{STAGETEST}.h5")
        if os.path.exists(cand):
            wpath = cand
        else:
            wpath = fallback_by_fold.get(i, None)

    if (wpath is None) or (not os.path.exists(wpath)):
        print(f"WARNING: Missing weights for fold {i} (wanted stage {STAGETEST}).")
        continue

    print(f"Fold {i+1} loading: {wpath}")
    model.load_weights(wpath)
    pred_i = model.predict(test_gen, verbose=1)
    pred_i = merge_pred_to_6cols(pred_i)
    preds.append(pred_i)
    ok_folds += 1

if PLATFORM == "local":
    sample_sub = pd.read_csv(
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
else:
    sample_sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )

if ok_folds == 0:
    print(
        "ERROR: No weight files were loaded. Falling back to sample_submission priors."
    )
    prior = sample_sub[list(TARGETS)].values.astype(np.float64)
    prior = np.clip(prior, 1e-8, 1.0)
    pred = prior / prior.sum(axis=1, keepdims=True)
else:
    pred = np.mean(preds, axis=0)
    print("\nTest preds shape (after merge if needed)", pred.shape)

    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, 0.0, 1.0)

    eps_floor = 1e-6
    pred = pred + eps_floor
    pred = pred / pred.sum(axis=1, keepdims=True)

sub = sample_sub[["eeg_id"]].copy()
sub[list(TARGETS)] = pred.astype(np.float32)
sub = sub[["eeg_id"] + list(TARGETS)]

row_sums = sub[list(TARGETS)].sum(axis=1).values
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
