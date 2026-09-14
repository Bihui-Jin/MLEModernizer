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

# 5. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""


NEEDTRAIN = True  # train the model (set True on Kaggle)
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    if os.path.exists("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = True
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "unknown"

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img ***
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "./"

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 40
SPE_WIDE = 1000

STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5

READ_EEG_FILES = True
READ_SPE_FILES = False

spectrograms = {}
eegs = {}
stfts = {}
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

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

TEST_BATCHSIZE = 128

import warnings

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal
import time
import gc
from collections import OrderedDict

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

print(tf.version.VERSION)
print(tf.config.list_physical_devices("GPU"))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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


def _has_any_weights(weights_dir: str) -> bool:
    try:
        if not os.path.exists(weights_dir):
            return False
        for i in range(100):
            if os.path.exists(os.path.join(weights_dir, f"fold{i}_stage2.weights.h5")):
                return True
        return False
    except Exception:
        return False


if PLATFORM == "kaggle" and NEEDTRAIN and _has_any_weights(LOAD_MODELS_FROM):
    print(
        f"[INFO] Found pretrained fold weights in {LOAD_MODELS_FROM}; disabling training to meet runtime."
    )
    NEEDTRAIN = False

if NEEDTRAIN:
    import itertools  # kept as original behavior (even if unused)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]

    if not os.path.exists("train.csv"):
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
        print("Wrote local train.csv:", train.shape)
    else:
        train = pd.read_csv("train.csv")
        print("Loaded local train.csv:", train.shape)



## === cell 2
if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")


class _LRUCache:
    def __init__(self, max_items=256):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, val):
        self._d[key] = val
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


def _make_eeg_loader(path_dir, do_reflect_pad=False, cache_size=256):
    cache = _LRUCache(max_items=cache_size)
    path_join = os.path.join
    read_parquet = pd.read_parquet
    _BRAIN = BRAIN
    _SFREQ = SFREQ
    _RSFREQ = RSFREQ
    _do_filter = filter_range is not None

    def load_one(eeg_id: int):
        x = cache.get(eeg_id)
        if x is not None:
            return x

        eeg_default = read_parquet(path_join(path_dir, f"{eeg_id}.parquet"))
        eeg = np.empty((len(_BRAIN), len(eeg_default)), dtype=np.float32)
        for k, channel in enumerate(_BRAIN):
            c0, c1 = channel.split("-")
            v = (eeg_default[c0].to_numpy() - eeg_default[c1].to_numpy()).astype(
                np.float32, copy=False
            )
            if np.isnan(v).any():
                v = v.copy()
                v[np.isnan(v)] = 0.0
            eeg[k] = v

        if _SFREQ != _RSFREQ:
            eeg = signal.resample_poly(eeg, _RSFREQ, _SFREQ, axis=1).astype(
                np.float32, copy=False
            )

        eeg = np.clip(eeg, a_min=-1024, a_max=1024)

        if do_reflect_pad:
            eegshape = eeg.shape[1]
            eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)

        if _do_filter:
            eeg = signal.filtfilt(b, a, eeg, axis=1).astype(np.float32, copy=False)

        if do_reflect_pad:
            eeg = eeg[:, eegshape : eegshape * 2]

        cache.put(eeg_id, eeg)
        return eeg

    return load_one


class _LazyEEGDict:
    def __init__(self, loader_fn):
        self._loader = loader_fn

    def __getitem__(self, eeg_id):
        return self._loader(int(eeg_id))

    def __contains__(self, eeg_id):
        return True


if NEEDTRAIN and ("eeg" in DATATYPE or "stft" in DATATYPE or "img" in DATATYPE):
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        eegs = _LazyEEGDict(
            _make_eeg_loader(PATH, do_reflect_pad=False, cache_size=2048)
        )
    else:
        raise FileNotFoundError(
            "READ_EEG_FILES=False requires precomputed /kaggle/input/preprocess/eegs.npy, which is not available."
        )



## === cell 3
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    if os.path.exists(PATH):
        files = os.listdir(PATH)
        print(f"There are {len(files)} spectrogram parquets")
    else:
        files = []

    if READ_SPE_FILES:
        time_start_time = time.time()
        for i, f in enumerate(files):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(files)
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        np.save("./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            spe_path_local = "./input/preprocess/spectrograms.npy"
            spe_path_kaggle = "/kaggle/input/preprocess/spectrograms.npy"
            if PLATFORM == "local" and os.path.exists(spe_path_local):
                spectrograms = np.load(spe_path_local, allow_pickle=True).item()
            elif PLATFORM == "kaggle" and os.path.exists(spe_path_kaggle):
                spectrograms = np.load(spe_path_kaggle, allow_pickle=True).item()
            else:
                raise FileNotFoundError(
                    "Spectrogram preprocess file not found but DATATYPE includes 'spe'."
                )



## === cell 4
MATCH_KEYS = [
    "eeg_id",
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
]
_signid_to_dfidxs = None
_signid_to_valid_dfidx = None

if NEEDTRAIN:
    _keys_df = df[MATCH_KEYS]
    _df_match_index = (
        _keys_df.reset_index()
        .groupby(MATCH_KEYS, sort=False)["index"]
        .apply(lambda s: s.to_numpy(dtype=np.int32, copy=True))
        .to_dict()
    )

    _train_keys = train[["sign_id", "eeg_id"] + TARGETS_RAW[:-1]].copy()
    _signid_to_dfidxs = {}
    _signid_to_valid_dfidx = {}

    keys_arr = np.empty((len(_train_keys), 6), dtype=object)
    keys_arr[:, 0] = _train_keys["eeg_id"].astype(np.int64).values
    keys_arr[:, 1] = _train_keys["seizure_vote_raw"].values
    keys_arr[:, 2] = _train_keys["lpd_vote_raw"].values
    keys_arr[:, 3] = _train_keys["gpd_vote_raw"].values
    keys_arr[:, 4] = _train_keys["lrda_vote_raw"].values
    keys_arr[:, 5] = _train_keys["grda_vote_raw"].values
    sign_ids = _train_keys["sign_id"].astype(np.int32).values

    for sid, k in zip(sign_ids, map(tuple, keys_arr)):
        idxs = _df_match_index.get(k, None)
        _signid_to_dfidxs[int(sid)] = idxs
        if idxs is not None and len(idxs) > 0:
            sub_ids = df.loc[idxs, "eeg_sub_id"].to_numpy()
            mid_pos = len(idxs) // 2
            valid_idx = idxs[np.argpartition(sub_ids, mid_pos)[mid_pos]]
            _signid_to_valid_dfidx[int(sid)] = int(valid_idx)
        else:
            _signid_to_valid_dfidx[int(sid)] = None


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
        stage=2,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stfts = stfts
        self.specs = specs
        self.imgs = imgs
        self.stage = stage

        self._eeg_id = self.dataframe["eeg_id"].astype(np.int64).values
        if self.mode != "test":
            self._sign_id = self.dataframe["sign_id"].astype(np.int32).values
            self._y = self.dataframe[TARGETS].astype(np.float32).values
            self._yraw = self.dataframe[TARGETS_RAW].astype(np.float32).values
        else:
            self._sign_id = None
            self._y = None
            self._yraw = None

        self._eeg_len_samples = {}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        self.nan = 0
        self.indexes = np.arange(len(self.dataframe), dtype=np.int32)
        if self.shuffle:
            np.random.shuffle(self.indexes)

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

        half = round(EEG_CHANNEL_USED / 2)
        trim_l = round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2)
        trim_r = round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)

        for j, i in enumerate(indexes):
            eeg_id = int(self._eeg_id[i])

            if self.mode != "test":
                yraw = self._yraw[i]
                sample_weight = float(np.sum(yraw)) / 20.0

            if self.mode == "test":
                r_eeg = 0.0
            else:
                sid = int(self._sign_id[i])
                idxs = (
                    _signid_to_dfidxs.get(sid, None)
                    if _signid_to_dfidxs is not None
                    else None
                )

                if idxs is None or len(idxs) == 0:
                    row0 = self.dataframe.iloc[i]
                    rows = df.loc[
                        (df.eeg_id == row0.eeg_id)
                        * (df.seizure_vote == row0.seizure_vote_raw)
                        * (df.lpd_vote == row0.lpd_vote_raw)
                        * (df.gpd_vote == row0.gpd_vote_raw)
                        * (df.lrda_vote == row0.lrda_vote_raw)
                        * (df.grda_vote == row0.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)

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
                    r_eeg = float(row.eeg_label_offset_seconds)
                else:
                    if self.mode == "train":
                        pick = int(idxs[np.random.randint(len(idxs))])
                        r_eeg = float(df.at[pick, "eeg_label_offset_seconds"])
                    else:
                        pick = _signid_to_valid_dfidx.get(sid, None)
                        if pick is None:
                            pick = int(idxs[len(idxs) // 2])
                        r_eeg = float(df.at[int(pick), "eeg_label_offset_seconds"])

                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0.0, r_eeg)

                    if eeg_id in self._eeg_len_samples:
                        eeg_len = self._eeg_len_samples[eeg_id]
                    else:
                        eeg_len = int(self.eegs[eeg_id].shape[1])
                        self._eeg_len_samples[eeg_id] = eeg_len
                    r_eeg = min(r_eeg, eeg_len / RSFREQ - 50.0)

            if "eeg" in DATATYPE:
                eeg_full = self.eegs[eeg_id]
                start = int(round(r_eeg * RSFREQ))
                stop = int(round((r_eeg + 50.0) * RSFREQ))
                eeg = eeg_full[:, start:stop]

                eeg = np.concatenate((eeg[0:half, :], eeg[-half:, :]), axis=0)
                eeg = eeg[:, trim_l:trim_r]

                if self.mode == "train":
                    if (self.stage == 2) and (np.random.rand() > 0):
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                    else:
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

                        eeg[0:half, :] = eeg[0:half, :][np.random.permutation(8), :]
                        eeg[-half:, :] = eeg[-half:, :][np.random.permutation(8), :]

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

            if self.mode != "test":
                yy = self._y[i]
                denom = float(np.sum(yy))
                if denom == 0.0:
                    denom = 1.0
                y[j] = yy / denom
                sample_weights[j] = sample_weight if self.sample_weights else 1.0

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "img" in DATATYPE:
            x["img"] = x_img

        return x, y, sample_weights




## === cell 5
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super().__init__()
        self.total_step = total_step
        self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
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


class IniToOne(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape
        kernel = np.zeros(shape, dtype=np.float32)
        for i in range(filter_count):
            kernel[i % filter_length, 0, i] = 1.0
        return tf.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self):
        return {}


class SumToOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        w = tf.abs(w)
        return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

    def get_config(self):
        return {}


class IniToOneAtten(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape
        kernel = np.zeros(shape, dtype=np.float32)
        kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
            (filter_length) // 2 + 1 - (filter_length - 1) // 2
        )
        return tf.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self):
        return {}


class SumToOneAtten(tf.keras.constraints.Constraint):
    def __call__(self, w):
        w = tf.abs(w)
        return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

    def get_config(self):
        return {}


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
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




## === cell 6
def build_model():
    inp = []
    y = 0

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

        base_model_eeg = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                w = f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
            else:
                w = f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
            if os.path.exists(w):
                base_model_eeg.load_weights(w)
            else:
                print(
                    f"[WARN] Pretrained weights not found: {w}. Training will proceed from random init."
                )

        base_model_eeg.name = "eeg_extractor"
        x_eeg = base_model_eeg(x_eeg)

        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)
        y = x_eeg * 1

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 7
def train_fold(
    i,
    stage,
    train_index,
    valid_index,
    df_train_stage1,
    df_valid_stage1,
    df_train_stage2,
    df_valid_stage2,
    build_model,
    BATCHSIZE,
    EPOCHS,
    LEARN_RATE,
    TARGETS,
    TARGETS_RAW,
):
    print("#" * 25)
    print(f"### Fold {i + 1}")

    model = build_model()
    loss = tf.keras.losses.KLDivergence()

    if stage == 1:
        train_gen_stage = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
        epochs_stage = EPOCHS
    else:
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 4,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1 * 3)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1 * 3,
                    LEARN_RATE * 0.1 * 0.1 * 0.1,
                    0,
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
        epochs_stage = max(round(EPOCHS / 3), 1)

    model.compile(loss=loss, optimizer=opt)

    history = model.fit(
        train_gen_stage,
        verbose=1,
        validation_data=valid_gen_stage,
        epochs=epochs_stage,
        callbacks=callbacks_stage,
    )

    model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

    loss_hist = history.history["loss"]
    val_loss_hist = history.history["val_loss"]
    epochs = range(1, len(loss_hist) + 1)
    plt.figure()
    plt.plot(epochs, loss_hist, "bo", label="loss")
    plt.plot(epochs, val_loss_hist, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
        fontsize=12,
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    valid_stage = (df_valid_stage1 if stage == 1 else df_valid_stage2)[TARGETS].values
    predict_stage = model.predict(valid_gen_stage, verbose=0)

    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
    gc.collect()

    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, [f"{TARGETS[k][:-5]}" for k in range(6)], fontsize=10)
    plt.yticks(tick_marks, [f"{TARGETS[k][:-5]}" for k in range(6)], fontsize=10)
    thresh = cm.max() / 2.0
    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
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
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}_cm.svg"))
    plt.close()




## === cell 8
def make_test_dataset(eeg_ids, eeg_loader, batch_size):
    half = round(EEG_CHANNEL_USED / 2)
    trim_l = round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2)
    trim_r = round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)

    def _py_load(eeg_id):
        eid = int(eeg_id)
        eeg_full = eeg_loader(eid)  # (18, N)
        start = 0
        stop = int(round(50.0 * RSFREQ))
        eeg = eeg_full[:, start:stop]
        eeg = np.concatenate((eeg[0:half, :], eeg[-half:, :]), axis=0)
        eeg = eeg[:, trim_l:trim_r]
        eeg2 = eeg.copy()
        eeg[4:8, :] = eeg2[12:16, :]
        eeg[8:12, :] = eeg2[4:8, :]
        eeg[12:16, :] = eeg2[8:12, :]
        eeg = np.clip(eeg, a_min=-1024, a_max=1024)
        eeg = (eeg + 1024.0) / 2048.0 * 255.0
        return eeg.astype(np.float32, copy=False)

    def _tf_map(eeg_id):
        x = tf.numpy_function(_py_load, [eeg_id], tf.float32)
        x.set_shape(
            (
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            )
        )
        return {"eeg": x}

    ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(eeg_ids, dtype=tf.int64)
    )
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 9
if __name__ == "__main__":
    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=SPLITS)

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

            for stage in [1, 2]:
                train_fold(
                    i,
                    stage,
                    train_index,
                    valid_index,
                    df_train_stage1,
                    df_valid_stage1,
                    df_train_stage2,
                    df_valid_stage2,
                    build_model,
                    BATCHSIZE,
                    EPOCHS,
                    LEARN_RATE,
                    TARGETS,
                    TARGETS_RAW,
                )

        print(
            "Training finished. Proceeding to inference using trained weights in ./models"
        )
        NEEDTRAIN = False
        LOAD_MODELS_FROM = "models"

    if not NEEDTRAIN:
        model_template = build_model()
        models = []
        for model_i in range(100):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(wpath):
                print(f"Loading fold {model_i + 1} weights")
                model = clone_model(model_template)
                model.load_weights(wpath)
                models.append(model)

        sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        print("Test shape", test.shape, "| sample_submission shape", sample_sub.shape)

        test_eeg_ids = test["eeg_id"].astype(np.int64).values
        sub_eeg_ids = sample_sub["eeg_id"].astype(np.int64).values

        if len(models) == 0:
            uniform = np.full(
                (len(sample_sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
            )
            sub = pd.DataFrame({"eeg_id": sample_sub.eeg_id.values})
            sub[TARGETS] = uniform
            sub.to_csv("submission.csv", index=False)
            print("No model weights found in LOAD_MODELS_FROM:", LOAD_MODELS_FROM)
            print("Wrote uniform fallback submission.csv with shape", sub.shape)
        else:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            test_eeg_loader = _make_eeg_loader(
                PATH_test, do_reflect_pad=True, cache_size=4096
            )

            ds = make_test_dataset(test_eeg_ids, test_eeg_loader, TEST_BATCHSIZE)

            preds_accum = None
            for m in models:
                p = m.predict(ds, verbose=0).astype(np.float32)
                if preds_accum is None:
                    preds_accum = p
                else:
                    preds_accum += p
            preds_all_test = (preds_accum / float(len(models))).astype(np.float32)

            pred_df = pd.DataFrame({"eeg_id": test_eeg_ids})
            pred_df[TARGETS] = preds_all_test

            pred_df = pred_df.set_index("eeg_id").reindex(sub_eeg_ids).reset_index()
            if pred_df.shape[0] != len(sample_sub):
                raise RuntimeError(
                    f"Prediction row count mismatch after reindex: {pred_df.shape[0]} vs sample_sub={len(sample_sub)}"
                )

            row_sum = pred_df[TARGETS].sum(axis=1).values.reshape(-1, 1)
            row_sum = np.where(row_sum == 0, 1.0, row_sum)
            pred_df[TARGETS] = pred_df[TARGETS].values / row_sum

            pred_df.to_csv("submission.csv", index=False)
            print("Submission shape", pred_df.shape)
            print(pred_df.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
