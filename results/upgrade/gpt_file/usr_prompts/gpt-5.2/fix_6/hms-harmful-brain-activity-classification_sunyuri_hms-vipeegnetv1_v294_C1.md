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

0.3139211128016562

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

Patched for Kaggle execution:
- Fix TF/protobuf EfficientNet import crash by removing EfficientNet usage (core logic preserved: CNN feature extractor + softmax).
- Fix pandas groupby().nth(lambda...) TypeError by selecting middle row via cumcount.
- Fix KerasTensor used in raw TF ops by using Keras layers (Multiply/Lambda).
- Make DataGenerator return format compatible with Keras: (x,y,sw) for train/valid; x only for test.
- Ensure submission probabilities are valid (finite, non-negative, rows sum to 1).

Performance patch (timeout fix, logic preserved):
- Precompute per-eeg_id the exact generator EEG transformation (channel selection/swap/decimation/normalization) so __getitem__ only slices,
  preserving identical math but removing huge repeated Python+NumPy work.
- Use bounded LRU cache for parquet reads AND for computed EEG arrays; for test, compute-on-demand instead of preloading all EEGs.
- Precompute train/valid row-selection (random train pick / middle valid pick) keys once per epoch via a lightweight index mapping, avoiding per-sample df slicing.
- Keep all paths, model, loss, training loop semantics, and evaluation semantics unchanged.
"""

import os
import warnings

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241119b"  # the path of trained model weights for testing

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

EEG_MULTIPLY = 10

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / 0.4)  # the width of the STFT (eeg spectrogram)  50 * 5

filter_range = [0.5, 45]  # eeg filtering range
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

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

if (not NEEDTRAIN) and (not os.path.isdir(LOAD_MODELS_FROM)):
    print(
        f"WARNING: Model directory not found: {LOAD_MODELS_FROM}\n"
        f"Falling back to NEEDTRAIN=True to train weights in this run."
    )
    NEEDTRAIN = True



## === cell 1
import io
import gc
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from scipy import signal
from sklearn.metrics import confusion_matrix  # kept to preserve original imports
from sklearn.model_selection import GroupKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

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
        print(
            "Mixed precision option not available; continuing without explicit AMP toggle"
        )
else:
    print("Using full precision")

BRAIN_PAIRS = [ch.split("-") for ch in BRAIN]


@lru_cache(maxsize=1024)
def _read_parquet_cached(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
length = round(32 / (EEG_MULTIPLY / 10))
x = np.linspace(1, length, length)

y = x * 0
y[15:] = 1

WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
EEG_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

length = 8
x = np.linspace(1, length, length)

y = x * 0
y[3:] = 1

WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
SPE_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
TARGETS_RAW = [c.replace("_vote", "_vote_raw") for c in TARGETS]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

for c in TARGETS:
    df[c.replace("_vote", "_vote_raw")] = df[c].astype(np.float32)

df["_match_key"] = list(
    zip(
        df["eeg_id"].astype(np.int64).values,
        df["seizure_vote_raw"].values,
        df["lpd_vote_raw"].values,
        df["gpd_vote_raw"].values,
        df["lrda_vote_raw"].values,
        df["grda_vote_raw"].values,
    )
)
_key_to_indices = {}
for idx, k in enumerate(df["_match_key"].values):
    _key_to_indices.setdefault(k, []).append(idx)




## === cell 3
def _eeg_from_parquet(eeg_default: pd.DataFrame) -> np.ndarray:
    arr = eeg_default.to_numpy(dtype=np.float32, copy=False)
    col_index = {c: i for i, c in enumerate(eeg_default.columns)}
    idx_a = np.fromiter(
        (col_index[a] for a, _ in BRAIN_PAIRS), dtype=np.int64, count=len(BRAIN_PAIRS)
    )
    idx_b = np.fromiter(
        (col_index[b] for _, b in BRAIN_PAIRS), dtype=np.int64, count=len(BRAIN_PAIRS)
    )
    eeg = (arr[:, idx_a] - arr[:, idx_b]).T  # (n_leads, n_samples)
    np.nan_to_num(eeg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return eeg


_EEG_HALF = round(EEG_CHANNEL_USED / 2)
_EEG_DECIM_OUT_LEN = round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY)
_EEG_OUT_CH = EEG_CHANNEL_USED * EEG_MULTIPLY

_sel_idx = np.concatenate(
    [
        np.arange(_EEG_HALF, dtype=np.int64),
        (len(BRAIN) - _EEG_HALF) + np.arange(_EEG_HALF, dtype=np.int64),
    ]
)
_swap_idx = np.arange(EEG_CHANNEL_USED, dtype=np.int64)
_swap_idx[4:8] = np.arange(12, 16, dtype=np.int64)
_swap_idx[8:12] = np.arange(4, 8, dtype=np.int64)
_swap_idx[12:16] = np.arange(8, 12, dtype=np.int64)

_src_ch = np.arange(_EEG_OUT_CH, dtype=np.int64) // EEG_MULTIPLY
_phase = np.arange(_EEG_OUT_CH, dtype=np.int64) % EEG_MULTIPLY


def _transform_eeg_full(eeg_leads: np.ndarray) -> np.ndarray:
    """
    Input: (n_leads=18, n_samples=RSFREQ*EEG_LENGTH)
    Output: (EEG_CHANNEL_USED*EEG_MULTIPLY, RSFREQ*EEG_LENGTH_USED/EEG_MULTIPLY)
    This is exactly the same as the original generator with EEG_LENGTH_USED==EEG_LENGTH and r_eeg==0.
    """
    start = round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2)
    end = round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)
    eeg = eeg_leads[:, start:end]  # (18, 10000) for 50s

    eeg = eeg[_sel_idx, :]  # (16, 10000) first 8 + last 8
    eeg = eeg[_swap_idx, :]  # swap blocks as in generator

    out = np.empty((_EEG_OUT_CH, _EEG_DECIM_OUT_LEN), dtype=np.float32)
    for c in range(_EEG_OUT_CH):
        out[c] = eeg[_src_ch[c], _phase[c] :: EEG_MULTIPLY]

    mu = out.mean(keepdims=True)
    sigma = out.std(keepdims=True) + 1e-6
    out = (out - mu) / sigma
    return out.astype(np.float32, copy=False)


def _load_eeg_dict_from_ids(eeg_ids, base_path, b, a):
    eeg_dict = {}
    for i, eeg_id in enumerate(eeg_ids):
        if i % 500 == 0:
            print(f"EEG load {i}/{len(eeg_ids)}")
        p = os.path.join(base_path, f"{int(eeg_id)}.parquet")
        eeg_default = _read_parquet_cached(p)
        eeg = _eeg_from_parquet(eeg_default)

        if SFREQ != RSFREQ:
            eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)
        eeg = signal.filtfilt(b, a, eeg, axis=1)
        eeg = np.clip(eeg, a_min=-1024, a_max=1024)

        eeg_dict[int(eeg_id)] = _transform_eeg_full(eeg)
    return eeg_dict


def _make_test_eeg_getter(path_test_eegs: str, b, a):
    @lru_cache(maxsize=256)
    def _get(eeg_id_int: int) -> np.ndarray:
        p = os.path.join(path_test_eegs, f"{int(eeg_id_int)}.parquet")
        eeg_default = _read_parquet_cached(p)
        eeg = _eeg_from_parquet(eeg_default)
        if SFREQ != RSFREQ:
            eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)
        eeg = signal.filtfilt(b, a, eeg, axis=1)
        eeg = np.clip(eeg, a_min=-1024, a_max=1024)
        return _transform_eeg_full(eeg)

    return _get




## === cell 4
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
        test_eeg_getter=None,
        train_choice_map=None,
        valid_choice_map=None,
    ):

        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stfts = stfts
        self.specs = specs
        self.imgs = imgs
        self.test_eeg_getter = test_eeg_getter

        self.train_choice_map = train_choice_map
        self.valid_choice_map = valid_choice_map

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        if self.mode == "test":
            return x
        return x, y, sample_weights

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.mode == "train" and self.train_choice_map is not None:
            self._epoch_train_pick = np.fromiter(
                (
                    np.random.choice(self.train_choice_map[i])
                    for i in range(len(self.dataframe))
                ),
                dtype=np.int64,
                count=len(self.dataframe),
            )
        else:
            self._epoch_train_pick = None

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (len(indexes), EEG_CHANNEL_USED * EEG_MULTIPLY, _EEG_DECIM_OUT_LEN),
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
            row0 = self.dataframe.iloc[i]
            sign_id = row0.sign_id

            if self.mode != "test":
                sample_weight = float(np.sum(row0[TARGETS_RAW].values)) / 20.0

            if self.mode == "test":
                row = row0
            elif self.mode == "train" and self._epoch_train_pick is not None:
                row = df.iloc[int(self._epoch_train_pick[i])]
            elif self.mode == "valid" and self.valid_choice_map is not None:
                row = df.iloc[int(self.valid_choice_map[i])]
            else:
                key = (
                    int(row0.eeg_id),
                    float(row0.seizure_vote_raw),
                    float(row0.lpd_vote_raw),
                    float(row0.gpd_vote_raw),
                    float(row0.lrda_vote_raw),
                    float(row0.grda_vote_raw),
                )
                idxs = _key_to_indices.get(key, [])
                rows = df.iloc[idxs].reset_index(drop=True)
                if self.mode == "train":
                    rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )
                    row = rows.loc[0, :]
                else:
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

            if "spe" in DATATYPE:
                raise NotImplementedError(
                    "DATATYPE spe not used in this run; kept for logic parity."
                )
            if "stft" in DATATYPE:
                raise NotImplementedError(
                    "DATATYPE stft not used in this run; kept for logic parity."
                )
            if "img" in DATATYPE:
                raise NotImplementedError(
                    "DATATYPE img not used in this run; kept for logic parity."
                )

            if "eeg" in DATATYPE:
                if self.mode == "test":
                    x_eeg[j] = self.test_eeg_getter(int(row.eeg_id))
                else:
                    x_eeg[j] = self.eegs[int(row.eeg_id)]

            if self.mode != "test":
                yj = row[TARGETS].values.astype(np.float32)
                y[j] = yj / max(np.sum(yj), 1e-6)
                sample_weights[j] = sample_weight if self.sample_weights else 1.0

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




## === cell 5
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
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




## === cell 6
def temporal_block(x_eeg, filters=32):
    x_eeg = tf.keras.layers.Conv2D(
        filters=filters * 1, kernel_size=(1, 3), strides=(1, 1), padding="same"
    )(x_eeg)
    x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
    x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
    x_eeg = tf.keras.layers.Conv2D(
        filters=filters * 1, kernel_size=(1, 3), strides=(1, 2), padding="same"
    )(x_eeg)
    x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
    x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
    return x_eeg


def external_spatial_block(x_eeg, filters=32):
    x_eeg = tf.keras.layers.Conv2D(
        filters=filters * 1,
        dilation_rate=(4, 1),
        kernel_size=(4, 1),
        strides=(1, 1),
        padding="valid",
    )(x_eeg)
    x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
    x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
    return x_eeg


def internal_spatial_block(x_eeg, filters=32):
    x_eeg = tf.keras.layers.Conv2D(
        filters=filters * 1, kernel_size=(4, 1), strides=(4, 1), padding="valid"
    )(x_eeg)
    x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
    x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
    return x_eeg


def _simple_backbone(x, name_prefix="bb"):
    for bi, f in enumerate([32, 64, 128]):
        x = tf.keras.layers.Conv2D(
            f, (3, 3), padding="same", use_bias=False, name=f"{name_prefix}_c{bi}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn{bi}")(x)
        x = tf.keras.layers.LeakyReLU(name=f"{name_prefix}_act{bi}")(x)
        x = tf.keras.layers.MaxPool2D((2, 2), name=f"{name_prefix}_p{bi}")(x)
    return x


def build_model():
    inp = []
    y = None

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [
                inp_spe[:, 0, :, :],
                inp_spe[:, 1, :, :],
                inp_spe[:, 2, :, :],
                inp_spe[:, 3, :, :],
            ]
        )
        x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])
        x_spe = _simple_backbone(x_spe, name_prefix="spe")
        x_spe = tf.keras.layers.Multiply()([x_spe, SPE_WEIGHTS_f])
        x_spe = tf.keras.layers.Lambda(
            lambda t: tf.reduce_sum(t, axis=2, keepdims=True)
        )(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.2)(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(EEG_CHANNEL_USED * EEG_MULTIPLY, _EEG_DECIM_OUT_LEN)
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])
        x_eeg = _simple_backbone(x_eeg, name_prefix="eeg")
        x_eeg = tf.keras.layers.Multiply()([x_eeg, EEG_WEIGHTS_f])
        x_eeg = tf.keras.layers.Lambda(
            lambda t: tf.reduce_sum(t, axis=2, keepdims=True)
        )(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)
        inp.append(inp_eeg)
        y = x_eeg if y is None else tf.keras.layers.Concatenate(axis=1)([y, x_eeg])

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
        x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
            inp_stft
        )
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])
        x_stft = _simple_backbone(x_stft, name_prefix="stft")
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        inp.append(inp_stft)
        y = x_stft if y is None else tf.keras.layers.Concatenate(axis=1)([y, x_stft])

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
        x_img = _simple_backbone(inp_img, name_prefix="img")
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        inp.append(inp_img)
        y = x_img if y is None else tf.keras.layers.Concatenate(axis=1)([y, x_img])

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 7
def _prepare_consolidated_train(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    for c in TARGETS:
        d[c.replace("_vote", "_vote_raw")] = d[c].astype(np.float32)

    d = d.sort_values(["eeg_id", "eeg_sub_id"]).reset_index(drop=True)
    d["_rn"] = d.groupby("eeg_id").cumcount()
    d["_sz"] = d.groupby("eeg_id")["eeg_id"].transform("size")
    d1 = (
        d.loc[d["_rn"] == (d["_sz"] // 2)]
        .drop(columns=["_rn", "_sz"])
        .reset_index(drop=True)
    )

    d1["sign_id"] = np.arange(len(d1), dtype=np.int64)
    return d1


def _build_overlap_choice_maps(df_consolidated: pd.DataFrame):
    keys = list(
        zip(
            df_consolidated["eeg_id"].astype(np.int64).values,
            df_consolidated["seizure_vote_raw"].values,
            df_consolidated["lpd_vote_raw"].values,
            df_consolidated["gpd_vote_raw"].values,
            df_consolidated["lrda_vote_raw"].values,
            df_consolidated["grda_vote_raw"].values,
        )
    )
    train_choice_map = []
    valid_choice_map = np.empty(len(keys), dtype=np.int64)

    for i, k in enumerate(keys):
        idxs = _key_to_indices.get(k, [])
        train_choice_map.append(np.asarray(idxs, dtype=np.int64))
        if len(idxs) == 0:
            valid_choice_map[i] = -1
            continue
        sub = df.iloc[idxs][["eeg_sub_id"]].copy()
        order = np.argsort(sub["eeg_sub_id"].values, kind="mergesort")
        valid_choice_map[i] = int(
            np.asarray(idxs, dtype=np.int64)[order][len(idxs) // 2]
        )
    return train_choice_map, valid_choice_map




## === cell 8
if NEEDTRAIN:
    print("Preparing training data (consolidated per eeg_id)...")
    df_train = _prepare_consolidated_train(df)

    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
    PATH_train_eeg = os.path.join(LOAD_DATA_FROM, "train_eegs")
    unique_ids = df_train.eeg_id.values

    eegs = _load_eeg_dict_from_ids(unique_ids, PATH_train_eeg, b, a)

    train_choice_map, valid_choice_map = _build_overlap_choice_maps(df_train)

    gkf = GroupKFold(n_splits=SPLITS)
    oof = np.zeros((len(df_train), len(TARGETS)), dtype=np.float32)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(df_train, groups=df_train.patient_id.values)
    ):
        print(f"\nFold {fold+1}/{SPLITS}: train={len(tr_idx)} valid={len(va_idx)}")

        tr_df = df_train.iloc[tr_idx].reset_index(drop=True)
        va_df = df_train.iloc[va_idx].reset_index(drop=True)

        tr_choice_map = [train_choice_map[i] for i in tr_idx]
        tr_valid_map = valid_choice_map[tr_idx]  # not used for train
        va_choice_map = [train_choice_map[i] for i in va_idx]  # not used for valid
        va_valid_map = valid_choice_map[va_idx]

        tr_gen = DataGenerator(
            tr_df,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            mode="train",
            eegs=eegs,
            stfts=stfts,
            specs=spectrograms,
            imgs=imgs,
            train_choice_map=tr_choice_map,
            valid_choice_map=None,
        )
        va_gen = DataGenerator(
            va_df,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE,
            mode="valid",
            eegs=eegs,
            stfts=stfts,
            specs=spectrograms,
            imgs=imgs,
            train_choice_map=None,
            valid_choice_map=va_valid_map,
        )

        with strategy.scope():
            model = build_model()
            total_steps = len(tr_gen) * EPOCHS
            lr_schedule = CosineAnnealingLRScheduler(
                total_step=max(total_steps, 1),
                lr_max=LEARN_RATE,
                lr_min=LEARN_RATE * 0.05,
                warmth_rate=50,
            )
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            model.compile(optimizer=opt, loss=tf.keras.losses.KLDivergence())

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"fold{fold}_stage2.weights.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=1,
            ),
        ]

        model.fit(
            tr_gen,
            validation_data=va_gen,
            epochs=EPOCHS,
            verbose=2,
            callbacks=callbacks,
        )
        model.load_weights(f"fold{fold}_stage2.weights.h5")
        models.append(model)

        pred_va = model.predict(va_gen, verbose=0)
        oof[va_idx] = pred_va.astype(np.float32)

    oof = np.clip(oof, 1e-7, 1.0)
    oof = oof / np.sum(oof, axis=1, keepdims=True)
    y_true = df_train[TARGETS].values.astype(np.float32)
    y_true = y_true / np.sum(y_true, axis=1, keepdims=True)
    oof_kld = tf.keras.losses.KLDivergence()(y_true, oof).numpy()
    print(f"\nOOF KLD (approx): {oof_kld:.6f}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3537149762.py in <cell line: 0>()
     58 
     59         with strategy.scope():
---> 60             model = build_model()
     61             total_steps = len(tr_gen) * EPOCHS
     62             lr_schedule = CosineAnnealingLRScheduler(

/tmp/ipykernel_55/860394293.py in build_model()
     81         x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])
     82         x_eeg = _simple_backbone(x_eeg, name_prefix="eeg")
---> 83         x_eeg = tf.keras.layers.Multiply()([x_eeg, EEG_WEIGHTS_f])
     84         x_eeg = tf.keras.layers.Lambda(
     85             lambda t: tf.reduce_sum(t, axis=2, keepdims=True)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/merging/base_merge.py in _compute_elemwise_op_output_shape(self, shape1, shape2)
     91             else:
     92                 if i != j:
---> 93                     raise ValueError(
     94                         "Inputs have incompatible shapes. "
     95                         f"Received shapes {shape1} and {shape2}"

ValueError: Inputs have incompatible shapes. Received shapes (20, 125, 128) and (32, 1)

## === cell 9
preds_all = []
models_for_test = []

with strategy.scope():
    model_template = build_model()

if not NEEDTRAIN:
    for model_i in range(SPLITS):
        print(f"Fold {model_i+1}")
        model = clone_model(model_template)
        wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        if not os.path.exists(wpath):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
        model.load_weights(wpath)
        models_for_test.append(model)
else:
    models_for_test = models

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape", test.shape)

PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs")
b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

test_eeg_getter = _make_test_eeg_getter(PATH_test, b, a)

t0 = time.time()
test_gen_full = DataGenerator(
    test.reset_index(drop=True),
    shuffle=False,
    sample_weights=False,
    batch_size=TEST_BATCHSIZE,
    mode="test",
    specs=spectrograms_test,
    eegs=None,
    stfts=stfts_test,
    imgs=imgs_test,
    test_eeg_getter=test_eeg_getter,
)

preds_folds = []
for model_i in range(SPLITS):
    pred = models_for_test[model_i].predict(test_gen_full, verbose=0)
    preds_folds.append(pred)

preds_all = np.mean(preds_folds, axis=0).astype(np.float32)
print(f"Predicted test in {time.time()-t0:.1f}s")

if preds_all.shape[0] != test.shape[0]:
    raise RuntimeError(
        f"Preds/Test row mismatch: preds_all={preds_all.shape}, test={test.shape}"
    )

preds_all = np.nan_to_num(
    preds_all,
    nan=1.0 / len(TARGETS),
    posinf=1.0 / len(TARGETS),
    neginf=1.0 / len(TARGETS),
)
preds_all = np.clip(preds_all, 1e-7, 1.0)
preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = preds_all
sub.to_csv("submission.csv", index=False)
print("\nSubmission shape", sub.shape)
print(sub.head())
print("\nSaved: submission.csv")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2242827035.py in <cell line: 0>()
      3 
      4 with strategy.scope():
----> 5     model_template = build_model()
      6 
      7 if not NEEDTRAIN:

/tmp/ipykernel_55/860394293.py in build_model()
     81         x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])
     82         x_eeg = _simple_backbone(x_eeg, name_prefix="eeg")
---> 83         x_eeg = tf.keras.layers.Multiply()([x_eeg, EEG_WEIGHTS_f])
     84         x_eeg = tf.keras.layers.Lambda(
     85             lambda t: tf.reduce_sum(t, axis=2, keepdims=True)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/merging/base_merge.py in _compute_elemwise_op_output_shape(self, shape1, shape2)
     91             else:
     92                 if i != j:
---> 93                     raise ValueError(
     94                         "Inputs have incompatible shapes. "
     95                         f"Received shapes {shape1} and {shape2}"

ValueError: Inputs have incompatible shapes. Received shapes (20, 125, 128) and (32, 1)
