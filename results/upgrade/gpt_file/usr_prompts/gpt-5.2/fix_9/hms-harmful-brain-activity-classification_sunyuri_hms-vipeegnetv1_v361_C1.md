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

0.2971380546101898

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"


def _detect_platform():
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    return "local"


PLATFORM = _detect_platform()
if PLATFORM == "local":
    if os.path.exists("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif PLATFORM == "kaggle":
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

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 45
STFT_TIME = 0.15
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / STFT_TIME)  # the width of the STFT

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg (kept, but we won't depend on it)
READ_SPE_FILES = False  # preprocess spectrogram

spectrograms = {}  # preprocessed spectrograms for training
eegs = {}  # preprocessed eegs for training (optional cache)
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
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf

print(tf.version.VERSION)
print(tf.config.list_physical_devices("GPU"))
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

import matplotlib.pyplot as plt

from scipy import signal
import gc

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
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    print("Using full precision")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))


def _weights_exist(weights_dir: str, n_splits: int) -> bool:
    try:
        for k in range(n_splits):
            if not os.path.exists(
                os.path.join(weights_dir, f"fold{k}_stage2.weights.h5")
            ):
                return False
        return True
    except Exception:
        return False


if (not NEEDTRAIN) and (not _weights_exist(LOAD_MODELS_FROM, SPLITS)):
    print(f"WARNING: No pretrained fold weights found under: {LOAD_MODELS_FROM}")
    print(
        "Switching NEEDTRAIN=True to train models in this session (required to generate submission)."
    )
    NEEDTRAIN = True

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

if "sign_id" not in train.columns:
    train["sign_id"] = np.arange(len(train), dtype=np.int64)
df = df.copy()
if "sign_id" not in df.columns:
    df["sign_id"] = np.arange(len(df), dtype=np.int64)

TARGETS_RAW = [t + "_raw" for t in TARGETS]
for t, tr in zip(TARGETS, TARGETS_RAW):
    if tr not in train.columns:
        train[tr] = train[t].astype(np.float32)
y_data = train[TARGETS].values.astype(np.float32)
y_data = y_data / np.sum(y_data, axis=1, keepdims=True)
train[TARGETS] = y_data

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
if filter_range2 is not None:
    b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")

TRAIN_EEG_PATH = os.path.join(LOAD_DATA_FROM, "train_eegs")
TEST_EEG_PATH = os.path.join(LOAD_DATA_FROM, "test_eegs")

_BRAIN_PAIRS = [ch.split("-") for ch in BRAIN]
_SHARED_TEST_EEG_CACHE = {}  # {eeg_id: np.ndarray}
_SHARED_TEST_EEG_LRU = []  # maintain insertion order for bounded memory
_SHARED_TEST_EEG_MAX = 1024  # fits comfortably; prevents memory blow-up



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
for t, tr in zip(TARGETS, TARGETS_RAW):
    if tr not in df.columns:
        df[tr] = df[t].astype(np.float32)

_key_cols_df = [
    "eeg_id",
    "seizure_vote_raw",
    "lpd_vote_raw",
    "gpd_vote_raw",
    "lrda_vote_raw",
    "grda_vote_raw",
]
df_key = df[_key_cols_df].copy()

_df_group_indices = {}
for idx, row in enumerate(df_key.itertuples(index=False, name=None)):
    _df_group_indices.setdefault(row, []).append(idx)
for k in list(_df_group_indices.keys()):
    _df_group_indices[k] = np.asarray(_df_group_indices[k], dtype=np.int32)

_df_eeg_indices = {}
for idx, eeg_id in enumerate(df["eeg_id"].to_numpy()):
    _df_eeg_indices.setdefault(int(eeg_id), []).append(idx)
for k in list(_df_eeg_indices.keys()):
    _df_eeg_indices[k] = np.asarray(_df_eeg_indices[k], dtype=np.int32)




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

        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs if eegs is not None else {}
        self.stfts = stfts if stfts is not None else {}
        self.specs = specs if specs is not None else {}
        self.imgs = imgs if imgs is not None else {}

        self._eeg_id = self.dataframe["eeg_id"].to_numpy(np.int64)
        if self.mode != "test":
            self._eeg_sub_id = self.dataframe.get(
                "eeg_sub_id", pd.Series(np.zeros(len(self.dataframe)))
            ).to_numpy()
            self._offset = self.dataframe.get(
                "eeg_label_offset_seconds", pd.Series(np.zeros(len(self.dataframe)))
            ).to_numpy(np.float32)

            self._raw_votes = self.dataframe[TARGETS_RAW].to_numpy(
                np.float32, copy=True
            )
            self._targets = self.dataframe[TARGETS].to_numpy(np.float32, copy=True)

        self._rng = np.random.RandomState(SEED + (0 if mode == "train" else 1))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe), dtype=np.int32)
        if self.shuffle:
            self._rng.shuffle(self.indexes)

    _eeg_lru_keys = []
    _eeg_lru_max = 512  # bounded to avoid memory blow-up

    def _load_eeg(self, eeg_id: int) -> np.ndarray:
        if self.mode == "test":
            if eeg_id in _SHARED_TEST_EEG_CACHE:
                return _SHARED_TEST_EEG_CACHE[eeg_id]

        if eeg_id in self.eegs:
            return self.eegs[eeg_id]

        path = os.path.join(
            TEST_EEG_PATH if self.mode == "test" else TRAIN_EEG_PATH,
            f"{int(eeg_id)}.parquet",
        )
        eeg_default = pd.read_parquet(path)

        eeg_list = []
        for ch_a, ch_b in _BRAIN_PAIRS:
            x = eeg_default[ch_a].to_numpy() - eeg_default[ch_b].to_numpy()
            if np.isnan(x).any():
                x = x.copy()
                x[np.isnan(x)] = 0
            eeg_list.append(x[None, :])
        eeg = np.concatenate(eeg_list, axis=0)

        if SFREQ != RSFREQ:
            eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

        eeg = np.clip(eeg, a_min=-1024, a_max=1024)

        eegshape = eeg.shape[1]
        eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
        if filter_range is not None:
            eeg = signal.filtfilt(b, a, eeg, axis=1)
        eeg = eeg[:, eegshape : eegshape * 2]
        eeg = np.array(eeg, dtype=np.float32)

        if self.mode == "test":
            try:
                _SHARED_TEST_EEG_CACHE[eeg_id] = eeg
                _SHARED_TEST_EEG_LRU.append(eeg_id)
                if len(_SHARED_TEST_EEG_LRU) > _SHARED_TEST_EEG_MAX:
                    old = _SHARED_TEST_EEG_LRU.pop(0)
                    _SHARED_TEST_EEG_CACHE.pop(old, None)
            except Exception:
                pass
        else:
            try:
                self.eegs[eeg_id] = eeg
                self._eeg_lru_keys.append(eeg_id)
                if len(self._eeg_lru_keys) > self._eeg_lru_max:
                    old = self._eeg_lru_keys.pop(0)
                    self.eegs.pop(old, None)
            except Exception:
                pass

        return eeg

    def __data_generation(self, indexes):
        bs = len(indexes)
        if "spe" in DATATYPE:
            x_spe = np.zeros((bs, 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    bs,
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype="float32",
            )
        if "img" in DATATYPE:
            x_img = np.zeros((bs, IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((bs, len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((bs, 1), dtype="float32")

        for j, i in enumerate(indexes):
            eeg_id = int(self._eeg_id[i])

            if self.mode != "test":
                raw_votes = self._raw_votes[i]
                sample_weight = float(np.sum(raw_votes)) / 20.0

            if self.mode == "test":
                r_eeg = 0.0
            else:
                key = (
                    eeg_id,
                    float(raw_votes[0]),
                    float(raw_votes[1]),
                    float(raw_votes[2]),
                    float(raw_votes[3]),
                    float(raw_votes[4]),
                )
                idxs = _df_group_indices.get(key, None)
                if idxs is None or len(idxs) == 0:
                    idxs = _df_eeg_indices.get(eeg_id, None)
                    if idxs is None or len(idxs) == 0:
                        rows = df.loc[df.eeg_id == eeg_id, :].reset_index(drop=True)
                    else:
                        rows = df.iloc[idxs]
                else:
                    rows = df.iloc[idxs]

                if self.mode == "train":
                    row = rows.iloc[self._rng.permutation(len(rows))[0]]
                elif self.mode == "valid":
                    row = rows.sort_values(by="eeg_sub_id").iloc[len(rows) // 2]
                r_eeg = float(row.eeg_label_offset_seconds)

            if "eeg" in DATATYPE:
                eeg_full = self._load_eeg(eeg_id)
                s0 = int(round(r_eeg * RSFREQ))
                s1 = int(round((r_eeg + 50) * RSFREQ))
                eeg = eeg_full[:, s0:s1]

                eeg = eeg[
                    :,
                    int(round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2)) : int(
                        round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)
                    ),
                ]
                eeg_save = np.zeros((x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32)

                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

                if self.mode == "train":
                    if self._rng.rand() > 0.5:
                        mask = round(self._rng.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + self._rng.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if self._rng.rand() > 0.5:
                        mask = round(self._rng.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + self._rng.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if self._rng.rand() > 0.5:
                        mask = round(self._rng.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + self._rng.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if self._rng.rand() > 0.5:
                        eeg[self._rng.permutation(eeg.shape[0])[0], :] = 0
                    if self._rng.rand() > 0.5:
                        eeg[self._rng.permutation(eeg.shape[0])[0], :] = 0

                    eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][self._rng.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][self._rng.permutation(8), :]
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                    if self._rng.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]
                else:
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                eeg = np.clip(eeg_save, a_min=-255, a_max=255)
                eeg = eeg + 255
                eeg = eeg / 2
                x_eeg[j] = eeg

            if self.mode != "test":
                if "row" in locals():
                    yy = row[TARGETS].values
                    y[j] = yy / float(np.sum(yy))
                else:
                    y[j] = self._targets[i] / float(np.sum(self._targets[i]))
                sample_weights[j] = sample_weight if self.sample_weights else 1.0

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




## === cell 3
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
        attn_output, weights = self.att(inputs, inputs, return_attention_scores=True)
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




## === cell 4
def _try_load_notop_weights(base_model, weights_path: str) -> bool:
    try:
        if weights_path and os.path.exists(weights_path):
            base_model.load_weights(weights_path)
            return True
    except Exception as e:
        print(f"WARNING: failed to load weights from {weights_path}: {e}")
    return False


def build_model():
    inp = list()

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
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

        base_model_eeg = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                _try_load_notop_weights(
                    base_model_eeg,
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5",
                )
            if PLATFORM == "kaggle":
                _try_load_notop_weights(
                    base_model_eeg,
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5",
                )
        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)
        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

    y = y_eeg * 1
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 5
if NEEDTRAIN:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K
    import itertools

    gkf = GroupKFold(n_splits=SPLITS)

    _FIT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
    _FIT_USE_MPROC = True
    _FIT_MAXQ = 8

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

        df_train_stage1 = df_train_stage1[
            df_train_stage1.expert_consensus != "Other"
        ].reset_index(drop=True)
        df_valid_stage1 = df_valid_stage1[
            df_valid_stage1.expert_consensus != "Other"
        ].reset_index(drop=True)

        train_gen = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            mode="train",
        )
        valid_gen = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.01, 5)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{1}.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
            workers=_FIT_WORKERS,
            use_multiprocessing=_FIT_USE_MPROC,
            max_queue_size=_FIT_MAXQ,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

        loss_hist = history.history["loss"]
        val_loss_hist = history.history["val_loss"]
        epochs_range = range(1, len(loss_hist) + 1)
        plt.plot(epochs_range, loss_hist, "bo", label="loss")
        plt.plot(epochs_range, val_loss_hist, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage1.svg"))
        plt.close()

        valid_stage1 = df_valid_stage1[TARGETS].values
        predict_stage1 = model.predict(
            valid_gen,
            verbose=0,
            workers=_FIT_WORKERS,
            use_multiprocessing=_FIT_USE_MPROC,
            max_queue_size=_FIT_MAXQ,
        )
        cm = confusion_matrix(np.argmax(valid_stage1, 1), np.argmax(predict_stage1, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        plt.yticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
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
        plt.savefig(os.path.join("models", f"fold{i}_stage1_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        gc.collect()

        train_gen = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            mode="train",
        )
        valid_gen = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    round(EPOCHS / 3), LEARN_RATE * 0.1, LEARN_RATE * 0.1 * 0.1, 0
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{2}.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=round(EPOCHS / 3),
            callbacks=callbacks_list,
            workers=_FIT_WORKERS,
            use_multiprocessing=_FIT_USE_MPROC,
            max_queue_size=_FIT_MAXQ,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage2.weights.h5"))

        loss_hist = history.history["loss"]
        val_loss_hist = history.history["val_loss"]
        epochs_range = range(1, len(loss_hist) + 1)
        plt.plot(epochs_range, loss_hist, "bo", label="loss")
        plt.plot(epochs_range, val_loss_hist, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage2.svg"))
        plt.close()

        valid_stage2 = df_valid_stage2[TARGETS].values
        predict_stage2 = model.predict(
            valid_gen,
            verbose=0,
            workers=_FIT_WORKERS,
            use_multiprocessing=_FIT_USE_MPROC,
            max_queue_size=_FIT_MAXQ,
        )
        cm = confusion_matrix(np.argmax(valid_stage2, 1), np.argmax(predict_stage2, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        plt.yticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
        plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.svg"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/877274128.py in <cell line: 0>()
     80             model.compile(loss=loss, optimizer=opt)
     81 
---> 82         history = model.fit(
     83             train_gen,
     84             verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 6
preds_all = []
models = list()

if NEEDTRAIN:
    LOAD_MODELS_EFFECTIVE = "models"
else:
    LOAD_MODELS_EFFECTIVE = LOAD_MODELS_FROM


def _find_fold_weights(weights_dir: str, fold_i: int) -> str:
    p2 = os.path.join(weights_dir, f"fold{fold_i}_stage2.weights.h5")
    p1 = os.path.join(weights_dir, f"fold{fold_i}_stage1.weights.h5")
    if os.path.exists(p2):
        return p2
    if os.path.exists(p1):
        return p1
    return ""


model_template = build_model()
for model_i in range(SPLITS):
    print(f"Fold {model_i + 1}")
    weights_path = _find_fold_weights(LOAD_MODELS_EFFECTIVE, model_i)
    if not weights_path:
        raise FileNotFoundError(
            f"No weights found for fold {model_i} under {LOAD_MODELS_EFFECTIVE} "
            f"(expected fold{model_i}_stage2.weights.h5 or fold{model_i}_stage1.weights.h5)."
        )
    model = clone_model(model_template)
    model.load_weights(weights_path)
    try:
        model.compile()
    except Exception:
        pass
    models.append(model)

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape", test.shape)

test_gen = DataGenerator(
    test.reset_index(drop=True),
    shuffle=False,
    sample_weights=False,
    batch_size=TEST_BATCHSIZE,
    mode="test",
    specs=spectrograms_test,
    eegs={},  # keep as before to avoid memory spikes; test cache is shared separately & bounded
    stfts={},
    imgs={},
)

_PRED_WORKERS = max(1, (os.cpu_count() or 2) - 1)
_PRED_USE_MPROC = True
_PRED_MAXQ = 16

preds = []
for model_i in range(SPLITS):
    pred = models[model_i].predict(
        test_gen,
        verbose=0,
        workers=_PRED_WORKERS,
        use_multiprocessing=_PRED_USE_MPROC,
        max_queue_size=_PRED_MAXQ,
    )
    preds.append(pred)
preds_all = np.mean(preds, axis=0)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})

preds_all = np.asarray(preds_all, dtype=np.float32)
eps = 1e-7
preds_all = np.clip(preds_all, eps, 1.0)
preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

sub[TARGETS] = preds_all

sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
sub = sample_sub[["eeg_id"]].merge(sub, on="eeg_id", how="left")

for c in TARGETS:
    if c not in sub.columns:
        sub[c] = 1.0 / len(TARGETS)
sub[TARGETS] = sub[TARGETS].astype(np.float32)

vals = sub[TARGETS].values
vals = np.clip(vals, eps, 1.0)
vals = vals / np.sum(vals, axis=1, keepdims=True)
sub[TARGETS] = vals
sub = sub[["eeg_id"] + list(TARGETS)]

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print(
    "Row sum stats:",
    float(sub[TARGETS].sum(axis=1).min()),
    float(sub[TARGETS].sum(axis=1).max()),
)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/171256044.py in <cell line: 0>()
     23     weights_path = _find_fold_weights(LOAD_MODELS_EFFECTIVE, model_i)
     24     if not weights_path:
---> 25         raise FileNotFoundError(
     26             f"No weights found for fold {model_i} under {LOAD_MODELS_EFFECTIVE} "
     27             f"(expected fold{model_i}_stage2.weights.h5 or fold{model_i}_stage1.weights.h5)."

FileNotFoundError: No weights found for fold 0 under models (expected fold0_stage2.weights.h5 or fold0_stage1.weights.h5).
