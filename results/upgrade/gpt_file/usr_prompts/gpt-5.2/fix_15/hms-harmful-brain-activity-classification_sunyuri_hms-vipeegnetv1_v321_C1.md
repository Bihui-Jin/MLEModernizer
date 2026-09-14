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

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # local training or local testing
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # kaggle training or online testing
    NEEDTRAIN = False
    found_models = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
            found_models = True
    if not found_models:
        if os.path.isdir("/kaggle/working/models"):
            found_models = True
            LOAD_MODELS_FROM = "/kaggle/working/models"
        else:
            NEEDTRAIN = True
else:
    PLATFORM = "kaggle"

DATATYPE = ["spe"]  # spe, eeg, stft, img
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # seconds
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16

EEG_MULTIPLY = 10

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100
SPE_WIDE = 256  # cropped width from 300 -> 256

STFT_LENGTH = 50
STFT_HIGH = 32
STFT_WIDE = round(STFT_LENGTH / 0.4)

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024

BATCHSIZE = 16

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False
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
import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

try:
    from tensorflow.python.framework.ops import reset_default_graph  # type: ignore
except Exception:
    reset_default_graph = None

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
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

try:
    _cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception:
        print("Mixed precision requested but not available; continuing.")
else:
    print("Using full precision")

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
print("Train shape:", df.shape)
print("Targets", list(TARGETS))


if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

    if READ_EEG_FILES:
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
    else:
        if os.path.exists("train.csv"):
            train = pd.read_csv("train.csv")
        else:
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
            train[[t + "_raw" for t in TARGETS]] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data



## === cell 1
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        if ("stft" in DATATYPE) or ("img" in DATATYPE):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
            )

            eeg = list()
            for channel in BRAIN:
                eeg_temp = (
                    eeg_default.loc[:, channel.split("-")[0]]
                    - eeg_default.loc[:, channel.split("-")[1]]
                ).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                ff, tt, ss = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)

                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(train_plot)):
                    row = train_plot.iloc[j]
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

                    eeg_plot = eeg2[
                        :,
                        round(row.eeg_label_offset_seconds * RSFREQ) : round(
                            (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                        ),
                    ]
                    eeg_plot = eeg_plot[
                        :,
                        round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                            (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                        ),
                    ]

                    img_save = np.zeros(
                        (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                    )
                    for ii in range(eeg_plot.shape[0]):
                        fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                        fig.patch.set_facecolor("black")

                        plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)

                        plt.xlim(-5, eeg_plot.shape[1] + 5)
                        plt.ylim(0, 200)
                        plt.axis("off")

                        byte_stream = io.BytesIO()
                        plt.savefig(
                            byte_stream, format="png", bbox_inches="tight", dpi=100
                        )
                        byte_stream.seek(0)
                        img = Image.open(byte_stream)
                        img = np.array(img)[:, :, :1]
                        img = img / 255
                        img = np.array(img, dtype=np.float32)
                        byte_stream.truncate()
                        plt.close("all")

                        if img.shape != (36, IMG_WIDE, 1):
                            img = np.concatenate((img, img, img), 2)
                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]

                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

            eeg = signal.filtfilt(b, a, eeg, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts[eeg_id] = ss
                stfts[-eeg_id] = tt

        if PLATFORM == "kaggle":
            os.makedirs("/kaggle/working/preprocess", exist_ok=True)
            out_dir = "/kaggle/working/preprocess"
        else:
            os.makedirs("./input/preprocess", exist_ok=True)
            out_dir = "./input/preprocess"

        if "eeg" in DATATYPE:
            np.save(os.path.join(out_dir, "eegs.npy"), eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save(os.path.join(out_dir, "stfts.npy"), stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save(os.path.join(out_dir, "imgs.npy"), imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/working/preprocess"
            if not os.path.exists(datapath):
                datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 2
if NEEDTRAIN:

    class LazySpectrogramStore:
        def __init__(self, base_path: str, cache_size: int = 64):
            self.base_path = base_path
            self.cache_size = int(cache_size)
            self._cache = {}  # spectrogram_id -> np.ndarray (T,400)
            self._lru = []  # keys in recency order

        def _touch(self, key: int):
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            if len(self._lru) > self.cache_size:
                old = self._lru.pop(0)
                self._cache.pop(old, None)

        def get_raw(self, spectrogram_id: int) -> np.ndarray:
            sid = int(spectrogram_id)
            arr = self._cache.get(sid)
            if arr is None:
                fn = os.path.join(self.base_path, f"{sid}.parquet")
                tmp = pd.read_parquet(fn)
                arr = tmp.iloc[:, 1:].to_numpy()
                self._cache[sid] = arr
            self._touch(sid)
            return arr

    spectrograms = {}
    spe_store_train = LazySpectrogramStore(
        os.path.join(LOAD_DATA_FROM, "train_spectrograms"), cache_size=128
    )



## === cell 3
if NEEDTRAIN:
    _keys_df = list(TARGETS)  # raw columns in original df
    _df_key_tuples = list(
        zip(
            df["eeg_id"].to_numpy(),
            df[_keys_df[0]].to_numpy(),
            df[_keys_df[1]].to_numpy(),
            df[_keys_df[2]].to_numpy(),
            df[_keys_df[3]].to_numpy(),
            df[_keys_df[4]].to_numpy(),
            df[_keys_df[5]].to_numpy(),
        )
    )
    df_group_map = {}
    for idx, k in enumerate(_df_key_tuples):
        df_group_map.setdefault(k, []).append(idx)
    df_group_median_idx = {}
    for k, idxs in df_group_map.items():
        sub_ids = df.loc[idxs, "eeg_sub_id"].to_numpy()
        order = np.argsort(sub_ids, kind="mergesort")
        df_group_median_idx[k] = idxs[order[len(order) // 2]]




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
        spe_store=None,  # LazySpectrogramStore
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
        self.spe_store = spe_store

        self._np = {}
        cols = ["sign_id", "eeg_id", "spectrogram_id"]
        for c in cols:
            if c in dataframe.columns:
                self._np[c] = dataframe[c].to_numpy()
        if self.mode != "test":
            self._np["targets"] = dataframe[TARGETS].to_numpy(
                dtype=np.float32, copy=False
            )
            self._np["targets_raw"] = dataframe[[t + "_raw" for t in TARGETS]].to_numpy(
                dtype=np.float32, copy=False
            )
            self._np["votes_raw_cols"] = {
                "seizure_vote_raw": dataframe["seizure_vote_raw"].to_numpy(),
                "lpd_vote_raw": dataframe["lpd_vote_raw"].to_numpy(),
                "gpd_vote_raw": dataframe["gpd_vote_raw"].to_numpy(),
                "lrda_vote_raw": dataframe["lrda_vote_raw"].to_numpy(),
                "grda_vote_raw": dataframe["grda_vote_raw"].to_numpy(),
                "other_vote_raw": dataframe["other_vote_raw"].to_numpy(),
            }
            self._np["expert_consensus"] = dataframe["expert_consensus"].to_numpy()

        self._exp_min, self._exp_max = -4.0, 6.0
        self._amin = float(np.exp(self._exp_min))
        self._amax = float(np.exp(self._exp_max))
        self._crop_l = int(round((300 - SPE_WIDE) / 2))
        self._crop_r = int(300 - self._crop_l)

        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.dataframe) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)

        if isinstance(x, list):
            x = tuple(x)

        if self.mode == "test":
            return x
        return x, y, sample_weights

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _prep_spe_window(self, base: np.ndarray, r_spe: int, mode: str):
        if base.ndim == 4:
            spe = base[r_spe]
        else:
            seg = base[r_spe : (r_spe + 300), :]
            spe = seg.reshape(seg.shape[0], 4, 100).transpose(1, 2, 0)  # (4,100,300)
            np.nan_to_num(spe, nan=0.0, copy=False)
            spe = np.clip(spe, self._amin, self._amax)
            spe = np.log(spe)
            spe = spe[:, :, self._crop_l : self._crop_r]  # (4,100,256)

            if mode == "train":
                spe2 = spe.copy()
                if np.random.rand() > 0.5:
                    spe[0] = spe2[2]
                    spe[2] = spe2[0]
                if np.random.rand() > 0.5:
                    spe[1] = spe2[3]
                    spe[3] = spe2[1]
                if np.random.rand() > 0.5:
                    spe[0] = spe2[1]
                    spe[2] = spe2[3]
                    spe[1] = spe2[0]
                    spe[3] = spe[2]

            spe = (spe - self._exp_min) / (self._exp_max - self._exp_min) * 255.0
            spe = np.clip(spe, 0.0, 255.0).astype(np.float32, copy=False)
        return spe

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    len(indexes),
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
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

        mode = self.mode
        specs = self.specs
        eegs = self.eegs
        stfts = self.stfts
        imgs = self.imgs

        for j, i in enumerate(indexes):
            if mode == "test":
                sign_id = self._np["sign_id"][i] if "sign_id" in self._np else i
                eeg_id = self._np["eeg_id"][i]
                spectrogram_id = self._np["spectrogram_id"][i]
                r_spe = 0
                r_eeg = 0.0
            else:
                sign_id = self._np["sign_id"][i]
                eeg_id = self._np["eeg_id"][i]
                sample_weight = float(self._np["targets_raw"][i].sum()) / 20.0

                k = (
                    int(eeg_id),
                    float(self._np["votes_raw_cols"]["seizure_vote_raw"][i]),
                    float(self._np["votes_raw_cols"]["lpd_vote_raw"][i]),
                    float(self._np["votes_raw_cols"]["gpd_vote_raw"][i]),
                    float(self._np["votes_raw_cols"]["lrda_vote_raw"][i]),
                    float(self._np["votes_raw_cols"]["grda_vote_raw"][i]),
                    float(self._np["votes_raw_cols"]["other_vote_raw"][i]),
                )
                if mode == "train":
                    idxs = df_group_map[k]
                    picked = idxs[np.random.randint(0, len(idxs))]
                    row2 = df.iloc[picked]
                elif mode == "valid":
                    picked = df_group_median_idx[k]
                    row2 = df.iloc[picked]
                else:
                    row2 = self.dataframe.iloc[i]  # fallback

                spectrogram_id = int(row2.spectrogram_id)
                r_spe = round(float(row2.spectrogram_label_offset_seconds) / 2)
                r_eeg = float(row2.eeg_label_offset_seconds)

            if "spe" in DATATYPE:
                if self.spe_store is not None:
                    base = self.spe_store.get_raw(int(spectrogram_id))
                else:
                    base = specs[int(spectrogram_id)]
                spe = self._prep_spe_window(base, int(r_spe), mode)
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eegs[eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]

            if "stft" in DATATYPE:
                stft_t = stfts[-eeg_id]
                r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                stft = stfts[eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                    stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = imgs[sign_id]

            if mode != "test":
                row_targets = self._np["targets"][i]
                y[j] = row_targets / float(row_targets.sum())

                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
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




## === cell 5
class CosineAnnealingLRScheduler:
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        self.total_step = int(total_step)
        if warmth_rate == 0:
            self.warm_step = 1
        else:
            self.warm_step = int(warmth_rate)
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)

    def __call__(self, epoch, lr=None):
        step = int(epoch) + 1
        if step < self.warm_step:
            lr_out = self.lr_max / float(self.warm_step) * float(step)
        else:
            denom = float(max(self.total_step - self.warm_step, 1))
            t = (float(step - self.warm_step) / denom) * np.pi
            lr_out = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + np.cos(t))
        return float(lr_out)




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


def build_model():
    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
        x_spe = tf.keras.layers.Reshape(
            (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
        )(inp_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

        base_model_spe = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                wpath = f"./input/pre-trained-weights/{base_model_spe.name}_notop.h5"
            else:
                wpath = (
                    f"/kaggle/input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                )
            if os.path.exists(wpath):
                base_model_spe.load_weights(wpath)
            else:
                print(
                    f"[WARN] Pretrained weights not found at {wpath}. Training from scratch."
                )

        base_model_spe._name = "spe_extractor"

        base_model_spe_pre = tf.keras.Model(
            base_model_spe.input, base_model_spe.get_layer("block3b_add").output
        )
        base_model_spe_pre._name = "spe_extractor_pre"
        x_spe1 = base_model_spe_pre(x_spe[:, 0, :, :, :])
        x_spe2 = base_model_spe_pre(x_spe[:, 1, :, :, :])
        x_spe3 = base_model_spe_pre(x_spe[:, 2, :, :, :])
        x_spe4 = base_model_spe_pre(x_spe[:, 3, :, :, :])

        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])
        base_model_spe_after = tf.keras.Model(
            base_model_spe_pre.output, base_model_spe.output
        )
        base_model_spe_after._name = "spe_extractor_after"
        x_spe = base_model_spe_after(x_spe)

        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

        inp.append(inp_spe)
        y = x_spe

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 7
if NEEDTRAIN:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K
    import itertools

    gkf = GroupKFold(n_splits=SPLITS)

    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.expert_consensus, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
        df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

        df_train_stage2 = df_train_stage1[
            np.sum(df_train_stage1[[t + "_raw" for t in TARGETS]].values, 1) >= 6
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[[t + "_raw" for t in TARGETS]].values, 1) >= 6
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
            spe_store=spe_store_train if "spe" in DATATYPE else None,
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
            spe_store=spe_store_train if "spe" in DATATYPE else None,
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
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

        del model, history, train_gen, valid_gen
        K.clear_session()
        if reset_default_graph is not None:
            try:
                reset_default_graph()
            except Exception:
                pass
        gc.collect()




## === cell 8
def _find_models_dir(preferred_dir: str) -> str | None:
    candidates = []
    if preferred_dir and os.path.isdir(preferred_dir):
        candidates.append(preferred_dir)
    for p in [
        "./models",
        "/kaggle/working/models",
        "/kaggle/input",
        "/kaggle/working",
        ".",
    ]:
        if os.path.isdir(p):
            candidates.append(p)
    if os.path.isdir("/kaggle/input"):
        for dn in os.listdir("/kaggle/input"):
            full = os.path.join("/kaggle/input", dn)
            if os.path.isdir(full) and dn.startswith("models"):
                candidates.append(full)

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)

    for c in out:
        if (
            os.path.exists(os.path.join(c, "fold0_stage2.weights.h5"))
            or os.path.exists(os.path.join(c, "fold0_stage1.weights.h5"))
            or os.path.exists(os.path.join(c, "fold0_stage2.h5"))
            or os.path.exists(os.path.join(c, "fold0_stage1.h5"))
        ):
            return c
    return None


models_dir = _find_models_dir("models" if NEEDTRAIN else LOAD_MODELS_FROM)
print("Resolved models_dir:", models_dir)

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape", test.shape)

spe_store_test = None
if "spe" in DATATYPE:
    PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"

    class LazySpectrogramStore:
        def __init__(self, base_path: str, cache_size: int = 64):
            self.base_path = base_path
            self.cache_size = int(cache_size)
            self._cache = {}
            self._lru = []

        def _touch(self, key: int):
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            if len(self._lru) > self.cache_size:
                old = self._lru.pop(0)
                self._cache.pop(old, None)

        def get_raw(self, spectrogram_id: int) -> np.ndarray:
            sid = int(spectrogram_id)
            arr = self._cache.get(sid)
            if arr is None:
                fn = os.path.join(self.base_path, f"{sid}.parquet")
                tmp = pd.read_parquet(fn)
                arr = tmp.iloc[:, 1:].to_numpy()
                self._cache[sid] = arr
            self._touch(sid)
            return arr

    spe_store_test = LazySpectrogramStore(PATH_test, cache_size=256)


def make_test_dataset(
    test_df: pd.DataFrame, spe_store: LazySpectrogramStore, batch_size: int
):
    exp_min, exp_max = -4.0, 6.0
    amin, amax = float(np.exp(exp_min)), float(np.exp(exp_max))
    crop_l = int(round((300 - SPE_WIDE) / 2))
    crop_r = int(300 - crop_l)

    sids = test_df["spectrogram_id"].to_numpy(dtype=np.int64, copy=False)

    def _py_load_one(sid_np):
        sid = int(sid_np)
        base = spe_store.get_raw(sid)  # (T,400)
        seg = base[0:300, :]
        spe = seg.reshape(seg.shape[0], 4, 100).transpose(1, 2, 0)  # (4,100,300)
        np.nan_to_num(spe, nan=0.0, copy=False)
        spe = np.clip(spe, amin, amax)
        spe = np.log(spe)
        spe = spe[:, :, crop_l:crop_r]  # (4,100,256)
        spe = (spe - exp_min) / (exp_max - exp_min) * 255.0
        spe = np.clip(spe, 0.0, 255.0).astype(np.float32, copy=False)
        return spe

    def _tf_map(sid):
        out = tf.numpy_function(_py_load_one, [sid], Tout=tf.float32)
        out.set_shape((4, SPE_HIGH, SPE_WIDE))
        return (out,)

    ds = tf.data.Dataset.from_tensor_slices(sids)
    ds = ds.map(
        _tf_map,
        num_parallel_calls=min(8, os.cpu_count() or 4),
        deterministic=True,
    )
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))

if models_dir is None:
    sub = sample_sub.copy()
    probs = np.full((len(sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
    sub[TARGETS] = probs
    sub.to_csv("submission.csv", index=False)
    print("No model weights found. Wrote uniform submission.csv with shape", sub.shape)
else:
    with strategy.scope():
        model = build_model()
        model.compile()

    test_ds = make_test_dataset(test, spe_store_test, TEST_BATCHSIZE)

    preds_sum = None
    used_folds = 0
    for model_i in range(SPLITS):
        print(f"Fold {model_i + 1}")

        w2_new = os.path.join(models_dir, f"fold{model_i}_stage2.weights.h5")
        w1_new = os.path.join(models_dir, f"fold{model_i}_stage1.weights.h5")
        w2_old = os.path.join(models_dir, f"fold{model_i}_stage2.h5")
        w1_old = os.path.join(models_dir, f"fold{model_i}_stage1.h5")

        wpath = None
        if os.path.exists(w2_new):
            wpath = w2_new
        elif os.path.exists(w1_new):
            wpath = w1_new
        elif os.path.exists(w2_old):
            wpath = w2_old
        elif os.path.exists(w1_old):
            wpath = w1_old

        if wpath is None:
            print(
                f"[WARN] Missing weights for fold {model_i}: {w2_new} / {w1_new} / {w2_old} / {w1_old}. Skipping."
            )
            continue

        model.load_weights(wpath)
        pred = model.predict(test_ds, verbose=0)
        if preds_sum is None:
            preds_sum = pred.astype(np.float64)
        else:
            preds_sum += pred.astype(np.float64)
        used_folds += 1

    if used_folds == 0:
        sub = sample_sub.copy()
        probs = np.full((len(sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
        sub[TARGETS] = probs
        sub.to_csv("submission.csv", index=False)
        print(
            "No usable fold weights found inside models_dir. Wrote uniform submission.csv with shape",
            sub.shape,
        )
    else:
        preds_all = (preds_sum / float(used_folds)).astype(np.float32)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub = sample_sub[["eeg_id"]].merge(sub, on="eeg_id", how="left")

        probs = sub[TARGETS].to_numpy(dtype=np.float64)
        nan_rows = ~np.isfinite(probs).all(axis=1)
        if nan_rows.any():
            probs[nan_rows] = 1.0 / len(TARGETS)
        probs = np.clip(probs, 1e-9, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        sub[TARGETS] = probs.astype(np.float32)

        sub.to_csv("submission.csv", index=False)
        print("Used folds:", used_folds)
        print("Submission shape", sub.shape)
        print(sub.head())
