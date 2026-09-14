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

0.343317868053277

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)



## === cell 1
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img'
STAGE = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024030102"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 256  # 256

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



## === cell 2
import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph

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

if hasattr(tf.config.experimental, "enable_op_determinism"):
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("enable_op_determinism skipped:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision request skipped due to TF API:", repr(e))
else:
    print("Using full precision")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
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



## === cell 4
if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

    if READ_SPEC_FILES * READ_EEG_FILES * READ_IMG_FILES:
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
        for i in range(len(TARGETS)):
            train_temp = df2.iloc[np.argmax(df2[TARGETS].values, 1) == i].reset_index(
                drop=True
            )
            train_temp = train_temp.iloc[
                np.random.permutation(len(train_temp))
            ].reset_index(drop=True)
            ii = 1
            while len(train_temp.groupby("eeg_id").head(ii)) < num_all:
                ii = ii + 1
            print(ii)
            if ii > 1:
                if len(train_temp.groupby("eeg_id").head(ii)) > num_all:
                    train_temp1 = train_temp.groupby("eeg_id").head(ii - 1)
                    train_temp2 = train_temp.groupby("eeg_id").head(ii)
                    train_temp2 = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.drop_duplicates(keep=False).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.iloc[
                        np.random.permutation(len(train_temp2))
                    ].reset_index(drop=True)
                    train_temp2 = train_temp2[
                        : (num_all - len(train_temp.groupby("eeg_id").head(ii - 1)))
                    ]
                    train_temp = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
            else:
                train_temp = train_temp.groupby("eeg_id").head(ii)
            print(len(train_temp))
            train = pd.concat([train, train_temp]).reset_index(drop=True)

        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.head()

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 5
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()



## === cell 6
if NEEDTRAIN:
    from scipy import signal
    import time

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

    if READ_EEG_FILES or READ_IMG_FILES:
        eegs = {}
        imgs = {}
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        time_start_jishi = time.time()

        for i in range(len(train)):
            if i % 100 == 0:
                xx = time.time() - time_start_jishi
                yy = xx / (i + 1) * len(train)
                xx = round(xx / 60 * 100) / 100
                yy = round(yy / 60 * 100) / 100
                print(i, f"time: {xx} min / {yy} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(train.eeg_id[i]) + ".parquet"))
            )

            list_eeg = list()
            list_img = list()
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

                time_temp = train.eeg_label_offset_seconds[i]
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if READ_EEG_FILES:
                if not train.eeg_id[i] in eegs.keys():
                    eegs[train.eeg_id[i]] = list_eeg

            if READ_IMG_FILES:
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

                imgs[train.sign_id[i]] = img

        if READ_EEG_FILES:
            if not os.path.exists("./input/brain-eegs"):
                os.makedirs("./input/brain-eegs")
            np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
        if READ_IMG_FILES:
            if not os.path.exists("./input/brain-imgs"):
                os.makedirs("./input/brain-imgs")
            np.save("./input/brain-imgs/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
            imgs = np.load("./input/brain-imgs/imgs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()
            imgs = np.load(
                "/kaggle/input/brain-imgs/imgs.npy", allow_pickle=True
            ).item()



## === cell 7
try:
    import albumentations as albu
except Exception as e:
    albu = None
    print("albumentations not available (ok, unused):", repr(e))

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
                x1 = np.random.rand() * (256 / 2 - 20)
                x2 = np.random.rand() * (256 / 2 - 20)
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_spe_min = x_spe_min + 128
                    x_spe_max = x_spe_max + 128

                x1 = np.random.rand() * (2048 / 2 - 500)
                x2 = np.random.rand() * (2048 / 2 - 500)
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(min(x1, x2))
                if np.random.rand() < 0.5:
                    x_eeg_min = x_eeg_min + 1024
                    x_eeg_max = x_eeg_max + 1024

                x1 = np.random.rand() * (256 / 2 - 64)
                x2 = np.random.rand() * (256 / 2 - 64)
                x_img_min = round(min(x1, x2))
                x_img_max = round(min(x1, x2))
                if np.random.rand() < 0.5:
                    x_img_min = x_img_min + 128
                    x_img_max = x_img_max + 128

            for k in range(4):
                if "spe" in DATATYPE:
                    if row.spectrogram_id in self.specs:
                        spe = self.specs[row.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T
                        spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                        spe = np.log(spe)
                        spe = np.nan_to_num(spe, nan=0.0)
                        spe = np.round(
                            (spe - self.cmin) / (self.cmax - self.cmin) * 255
                        )
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
                            tf.image.resize(spe, ((HIGH - 32), LENGTH)),
                            dtype=np.float32,
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
                        x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                if "eeg" in DATATYPE:
                    if row.eeg_id in self.eegs:
                        eeg = self.eegs[row.eeg_id][
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
                            x_eeg[j, :, :, k]
                            - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    if self.mode == "test":
                        if row.eeg_id in self.imgs:
                            img = self.imgs[row.eeg_id][:, :, k]
                        else:
                            img = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
                    else:
                        if row.sign_id in self.imgs:
                            img = self.imgs[row.sign_id][:, :, k]
                        else:
                            img = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
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

        x = list()
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
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx

            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img

            x.append(x_img)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 8
if NEEDTRAIN:
    LR_MAX1 = 1e-3
    EPOCHS1 = 6

    def lrfn1(epoch):
        xx = [LR_MAX1, LR_MAX1, LR_MAX1, 1e-4, 1e-4]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-4

    LR1 = tf.keras.callbacks.LearningRateScheduler(lrfn1, verbose=True)

    LR_MAX2 = 1e-4
    EPOCHS2 = 2

    def lrfn2(epoch):
        xx = [LR_MAX2, LR_MAX2, LR_MAX2, 1e-4, 1e-4]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-4

    LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn2, verbose=True)

    LR_MAX3 = 1e-4
    EPOCHS3 = 6

    def lrfn3(epoch):
        xx = [LR_MAX3, LR_MAX3, LR_MAX3, LR_MAX3]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-5

    LR3 = tf.keras.callbacks.LearningRateScheduler(lrfn3, verbose=True)




## === cell 9
def _make_efficientnet_b0(name: str):
    m = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # keep as original (weights loaded only when NEEDTRAIN)
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    try:
        m._name = name
    except Exception:
        pass
    return m


def build_model():
    l2norm = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2n"
    )

    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1, name="cat_spe_regions")(
            [x_spe1, x_spe2, x_spe3, x_spe4]
        )

        base_model_spe = _make_efficientnet_b0("spe_extractor")
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_spe.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_spe.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
        x_spe = l2norm(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(6, round(EEG_LENGTH * SFREQ), 4), name="inp_eeg"
        )
        x_eeg1 = inp_eeg[:, :, :, 0:1]
        x_eeg2 = inp_eeg[:, :, :, 1:2]
        x_eeg3 = inp_eeg[:, :, :, 2:3]
        x_eeg4 = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="cat_eeg_regions")(
            [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3, name="cat_eeg_rgb")(
            [x_eeg, x_eeg, x_eeg]
        )

        base_model_eeg = _make_efficientnet_b0("eeg_extractor")
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = l2norm(x_eeg)

        inp.append(inp_eeg)

        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1, name="cat_spe_eeg")([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="inp_img")
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1, name="cat_img_regions")(
            [x_img1, x_img2, x_img3, x_img4]
        )
        x_img = tf.keras.layers.Concatenate(axis=3, name="cat_img_rgb")(
            [x_img, x_img, x_img]
        )

        base_model_img = _make_efficientnet_b0("img_extractor")
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
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = l2norm(x_img)

        inp.append(inp_img)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="cat_all_modalities")(
                [y, x_img]
            )
        else:
            y = x_img

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32", name="head")(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    return model




## === cell 10
if NEEDTRAIN:
    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K, gc

    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.expert_consensus, train.patient_id)
    ):
        if (i + 1) >= 1:
            print(valid_index)
            print("#" * 25)
            print(f"### Fold {i + 1}")

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = (
                df_train_stage1.sort_values("selected", ascending=False)
                .reset_index(drop=True)
                .drop_duplicates("eeg_id")
                .reset_index(drop=True)
            )
            df_valid_stage2 = (
                df_valid_stage1.sort_values("selected", ascending=False)
                .reset_index(drop=True)
                .drop_duplicates("eeg_id")
                .reset_index(drop=True)
            )

            xx = (np.sum(df_train_stage2[TARGETS_RAW].values, 1) >= 10) + (
                np.array(
                    tf.losses.KLD(
                        df_train_stage2[TARGETS].values * 0 + 1 / 6,
                        df_train_stage2[TARGETS].values,
                    )
                )
                <= 5.5
            )
            df_train_stage3 = df_train_stage2[xx].reset_index(drop=True)
            xx = (np.sum(df_valid_stage2[TARGETS_RAW].values, 1) >= 10) + (
                np.array(
                    tf.losses.KLD(
                        df_valid_stage2[TARGETS].values * 0 + 1 / 6,
                        df_valid_stage2[TARGETS].values,
                    )
                )
                <= 5.5
            )
            df_valid_stage3 = df_valid_stage2[xx].reset_index(drop=True)

            print(
                f"### train size {len(df_train_stage3)}, valid size {len(df_valid_stage3)}"
            )
            print("#" * 25)
            train_gen = DataGenerator(
                df_train_stage3,
                shuffle=True,
                batch_size=16,
                specs=spectrograms,
                eegs=eegs,
                imgs=imgs,
            )
            valid_gen = DataGenerator(
                df_valid_stage3,
                shuffle=False,
                batch_size=32,
                mode="valid",
                specs=spectrograms,
                eegs=eegs,
                imgs=imgs,
            )
            callbacks_list = [
                LR3,
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"f{i}_stage3.h5",
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                ),
                tf.keras.callbacks.EarlyStopping(
                    patience=5, monitor="val_loss", mode="min"
                ),
            ]

            with strategy.scope():
                model = build_model()
                opt = tf.keras.optimizers.Adam(learning_rate=LR_MAX3)
                loss = tf.keras.losses.KLDivergence()
                model.compile(loss=loss, optimizer=opt)
            model.load_weights(f"f{i}_stage2.h5")
            history = model.fit(
                train_gen,
                verbose=1,
                validation_data=valid_gen,
                epochs=EPOCHS3,
                callbacks=callbacks_list,
            )

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, "bo", label="loss")
            plt.plot(epochs, val_loss, "b", label="val_loss")
            plt.title(
                f"loss: {round(min(loss) * 10000) / 10000}, val loss: {round(min(val_loss) * 10000) / 10000}",
                fontsize=12,
            )
            plt.legend()
            plt.savefig(f"f{i}_stage3.pdf")
            plt.close()

            del model, history, train_gen, valid_gen
            K.clear_session()
            reset_default_graph()
            gc.collect()



## === cell 11
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = list()
            list_img = list()
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

    preds = []

    with strategy.scope():
        model = build_model()

    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
    )

    for i in range(5):
        print(f"Fold {i + 1}")
        model.load_weights(os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGE}.h5"))
        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    print()
    print("Test preds shape", pred.shape)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred

    p = sub[TARGETS].values.astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sub[TARGETS] = p.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sum check (min/max):",
        sub.iloc[:, -6:].sum(axis=1).min(),
        sub.iloc[:, -6:].sum(axis=1).max(),
    )

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1089445282.py in <cell line: 0>()
    119 
    120     with strategy.scope():
--> 121         model = build_model()
    122 
    123     test_gen = DataGenerator(

/tmp/ipykernel_55/2295802928.py in build_model()
    126 
    127     y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32", name="head")(y)
--> 128     model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    129     return model
    130 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/tracking.py in wrapper(*args, **kwargs)
     24     def wrapper(*args, **kwargs):
     25         with DotNotTrackScope():
---> 26             return fn(*args, **kwargs)
     27 
     28     return wrapper

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in __init__(self, inputs, outputs, name, **kwargs)
    133             inputs, outputs = clone_graph_nodes(inputs, outputs)
    134 
--> 135         Function.__init__(self, inputs, outputs, name=name)
    136 
    137         if trainable is not None:

/usr/local/lib/python3.11/dist-packages/keras/src/ops/function.py in __init__(self, inputs, outputs, name)
     75             self._self_setattr_tracking = _self_setattr_tracking
     76 
---> 77         (nodes, nodes_by_depth, operations, operations_by_depth) = map_graph(
     78             self._inputs, self._outputs
     79         )

/usr/local/lib/python3.11/dist-packages/keras/src/ops/function.py in map_graph(inputs, outputs)
    327     for name in all_names:
    328         if all_names.count(name) != 1:
--> 329             raise ValueError(
    330                 f'The name "{name}" is used {all_names.count(name)} '
    331                 "times in the model. All operation names should be unique."

ValueError: The name "efficientnetb0" is used 3 times in the model. All operation names should be unique.
