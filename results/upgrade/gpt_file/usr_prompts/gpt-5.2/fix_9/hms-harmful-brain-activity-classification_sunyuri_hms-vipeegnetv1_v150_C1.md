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

0.3288345313888168

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` incompatibility seen in this Kaggle image. Then I make model weight loading robust: if the external `/kaggle/input/models2024040102` directory or expected `.h5` files are missing, the script fall back to producing a valid probabilistic submission using the provided `sample_submission.csv` priors (guaranteed to sum to 1), so you always get a `submission.csv`. These changes are minimal and keep the core model/inference logic intact when weights are available; they only add stability and ensure end-to-end execution. Finally, I keep the output columns/order exactly as required and validate row-wise probability sums.'
- What this solution (achieved 1.41937) has done: 'We fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf backend is set before *any* TensorFlow-related imports, and we also add a safe fallback path: if TensorFlow still cannot import in this Kaggle image, the script generate a valid probabilistic submission using the sample submission priors. This keeps your core model/inference logic unchanged when TF + weights are available, but guarantees end-to-end execution and a valid `submission.csv` otherwise. To move the score down (better) vs the current 1.40995 toward the 0.3288 target, we also improve the fallback distribution by using class priors computed from `train.csv` (normalized vote proportions), which is a legitimate calibration step and typically scores much better than the raw sample submission. All changes are minimal and directly address the crash + score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img", "stft"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024040102"
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

NSPLIT = 5
BATCHSIZE = 12

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

READ_EXTRA_SPEC_FILES = False
READ_EXTRA_EEG_FILES = False
READ_EXTRA_IMG_FILES = False
READ_EXTRA_STFT_FILES = False

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
import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # noqa: F401

TF_AVAILABLE = True
try:
    import tensorflow as tf

    print("TensorFlow version =", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print("WARNING: TensorFlow failed to import; will use fallback submission.")
    print("TF import error:", repr(e))

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if TF_AVAILABLE:
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
        except Exception as e:
            print(
                "Mixed precision not enabled (not supported in this TF build):", repr(e)
            )
    else:
        print("Using full precision")

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

VER = 1



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 4
train_votes = df[list(TARGETS)].astype(np.float64).sum(axis=0).values
train_priors = train_votes / np.sum(train_votes)
train_priors = np.clip(train_priors, 1e-12, 1.0)
train_priors = train_priors / train_priors.sum()
print("Train priors:", dict(zip(TARGETS, train_priors.round(6))))



## === cell 5
try:
    import albumentations as albu  # noqa: F401
except Exception:
    albu = None

if TF_AVAILABLE:
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
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
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
                x_eeg2 = np.zeros(
                    (len(indexes), 4, round(50 * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros(
                    (len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32"
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")
            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                elif self.mode == "valid":
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row.eeg_label_offset_seconds * SFREQ)
                else:
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row.eeg_label_offset_seconds * SFREQ)

                if self.mode == "train":
                    x1 = np.random.rand() * LENGTH / 2
                    x2 = np.random.rand() * LENGTH / 2
                    if np.random.rand() < 0.5:
                        x1 = x1 + LENGTH / 2
                        x2 = x2 + LENGTH / 2
                    x_spe_min = round(min(x1, x2))
                    x_spe_max = round(max(x1, x2))

                for k in range(4):
                    if "spe" in DATATYPE:
                        spe = self.specs[row.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T

                        if (spe.shape[0] != 100) or (spe.shape[1] != 300):
                            spe2 = np.zeros((100, 300))
                            spe2[: spe.shape[0], : spe.shape[1]] = spe
                            spe = spe2

                        spe = np.nan_to_num(spe, nan=0.0)
                        spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                        spe = np.log(spe)

                        spe = np.round(
                            (spe - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
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
                        eeg = self.eegs[row.eeg_id][
                            :, r_eeg : r_eeg + round(50 * SFREQ), k
                        ]

                        if eeg.shape[1] < 50 * SFREQ:
                            eeg = np.concatenate((eeg, eeg), 1)
                            eeg = eeg[:, : 50 * SFREQ]

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
                            img = self.imgs[row.eeg_id][:, :, k, :]
                        else:
                            img = self.imgs[row.sign_id][:, :, k, :]
                        x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        if self.mode == "test":
                            stft = self.stfts[row.eeg_id][:, :, :, k]
                        else:
                            stft = self.stfts[row.sign_id][:, :, :, k]

                        stft = np.concatenate(
                            [
                                stft[0, :, :],
                                stft[1, :, :],
                                stft[2, :, :],
                                stft[3, :, :],
                            ],
                            1,
                        )
                        stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                        stft = np.log(stft)
                        stft = np.nan_to_num(stft, nan=0.0)

                        stft = np.round(
                            (stft - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)

                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (64, 128 * 4, 3))

                        x_stft[j, :, :, :, k] = stft
                        x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                if self.mode != "test":
                    label = row[self.targets].values
                    if self.mode == "train" and sum(label == 1):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label == 1] = 1 - 5 * xx
                    y[j] = label

            if "eeg" in DATATYPE:
                for i_eeg in range(x_eeg2.shape[0]):
                    xx = np.std(x_eeg2[i_eeg, :, :, :], 1, keepdims=True)
                    xx = np.mean(xx)
                    x_eeg2[i_eeg, :, :, :] = (
                        x_eeg2[i_eeg, :, :, :]
                        - np.mean(x_eeg2[i_eeg, :, :, :], 1, keepdims=True)
                    ) / (xx + 1e-6)

            x = list()
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "img" in DATATYPE:
                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img
                x.append(x_img)
            if "stft" in DATATYPE:
                x.append(x_stft)

            return x, y




## === cell 6
if TF_AVAILABLE:
    _effnet_counter = 0

    def _effnetb0_no_top(name: str):
        global _effnet_counter
        _effnet_counter += 1
        unique_name = f"{name}_effnetb0_{_effnet_counter}"
        model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=None,
            name=unique_name,
        )
        return model

    def build_model(TARGETS_PRETRAIN):
        l2norm = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )

        inp = list()

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1, name="concat_spe_tiles")(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = _effnetb0_no_top("spe_extractor")
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
            x_spe = l2norm(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4), name="inp_eeg")
            x_eeg1 = inp_eeg[:, :, :, :, 0]
            x_eeg2 = inp_eeg[:, :, :, :, 1]
            x_eeg3 = inp_eeg[:, :, :, :, 2]
            x_eeg4 = inp_eeg[:, :, :, :, 3]

            x_eeg = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_tiles")(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )

            base_model_eeg = _effnetb0_no_top("eeg_extractor")
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
            x_eeg = l2norm(x_eeg)

            inp.append(inp_eeg)

            if "spe" in DATATYPE:
                y = tf.keras.layers.Concatenate(axis=1, name="concat_spe_eeg")(
                    [y, x_eeg]
                )
            else:
                y = x_eeg

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
            x_img1 = inp_img[:, :, :, :, 0]
            x_img2 = inp_img[:, :, :, :, 1]
            x_img3 = inp_img[:, :, :, :, 2]
            x_img4 = inp_img[:, :, :, :, 3]
            x_img = tf.keras.layers.Concatenate(axis=1, name="concat_img_tiles")(
                [x_img1, x_img2, x_img3, x_img4]
            )

            base_model_img = _effnetb0_no_top("img_extractor")
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
            x_img = l2norm(x_img)

            inp.append(inp_img)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="concat_prev_img")(
                    [y, x_img]
                )
            else:
                y = x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4), name="inp_stft")
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1, name="concat_stft_tiles")(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = _effnetb0_no_top("stft_extractor")
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
            x_stft = l2norm(x_stft)

            inp.append(inp_stft)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="concat_prev_stft")(
                    [y, x_stft]
                )
            else:
                y = x_stft

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN),
            activation="softmax",
            dtype="float32",
            name="pred_head",
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y, name="hms_multimodal_effnet")
        return model

    def my_loss(y_ture, y_pred):
        y_pred1 = y_pred[:, 5:6]
        y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 7
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

    print("Test shape", test.shape)

    if not TF_AVAILABLE:
        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        pri = np.tile(train_priors.reshape(1, -1), (len(sub), 1))
        pri = np.clip(pri, 1e-12, 1.0)
        pri = pri / pri.sum(axis=1, keepdims=True)
        sub[list(TARGETS)] = pri
        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)
        print("Wrote fallback submission.csv (TF unavailable) with shape", sub.shape)
        print(
            "Row-sum min/max:",
            sub[TARGETS].sum(axis=1).min(),
            sub[TARGETS].sum(axis=1).max(),
        )
    else:
        weight_paths = [
            os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            for i in range(NSPLIT)
        ]
        have_all_weights = os.path.isdir(LOAD_MODELS_FROM) and all(
            os.path.exists(p) for p in weight_paths
        )

        if not have_all_weights:
            print("WARNING: Model weights not found. Expected paths (first few):")
            for p in weight_paths[:3]:
                print(" -", p, "exists=", os.path.exists(p))

            sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
            pri = np.tile(train_priors.reshape(1, -1), (len(sub), 1))
            pri = np.clip(pri, 1e-12, 1.0)
            pri = pri / pri.sum(axis=1, keepdims=True)
            sub[list(TARGETS)] = pri
            sub = sub[["eeg_id"] + list(TARGETS)]
            sub.to_csv("submission.csv", index=False)
            print(
                "Wrote fallback submission.csv (missing weights) with shape", sub.shape
            )
            print(
                "Row-sum min/max:",
                sub[TARGETS].sum(axis=1).min(),
                sub[TARGETS].sum(axis=1).max(),
            )
        else:
            if "spe" in DATATYPE:
                if PLATFORM == "local":
                    PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
                elif PLATFORM == "kaggle":
                    PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

                files2 = os.listdir(PATH2)
                print(f"There are {len(files2)} test spectrogram parquets")

                spectrograms2 = {}
                for i, f in enumerate(files2):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    tmp = pd.read_parquet(f"{PATH2}{f}")
                    name = int(f.split(".")[0])
                    spectrograms2[name] = tmp.iloc[:, 1:].values
                print()

            from scipy import signal

            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
            elif PLATFORM == "kaggle":
                PATH2 = (
                    "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
                )

            files2 = os.listdir(PATH2)
            print(f"\nThere are {len(files2)} test eeg parquets")

            eegs2 = {}
            imgs2 = {}
            stfts2 = {}
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
            for i, f in enumerate(files2):
                if i % 100 == 0:
                    print(i, ", ", end="")
                eeg_default = pd.read_parquet(f"{PATH2}{f}")
                name = int(f.split(".")[0])

                if len(test[test.eeg_id == name]) > 0:
                    list_eeg = list()
                    list_img = list()
                    list_stft = list()
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
                        time_start = round(
                            time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ
                        )
                        time_stop = round(
                            time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ
                        )

                        list_img.append(eeg[:, time_start:time_stop])

                        if "stft" in DATATYPE:
                            frequencies, times, Sxx = signal.spectrogram(
                                eeg[
                                    :,
                                    round(time_temp * SFREQ) : round(
                                        (time_temp + 50) * SFREQ
                                    ),
                                ],
                                SFREQ,
                                nperseg=256,
                                noverlap=219,
                                nfft=320,
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

                        list_eeg.append(
                            np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1))
                        )

                    list_eeg = np.concatenate(list_eeg, 2)

                    if "stft" in DATATYPE:
                        list_stft = np.concatenate(list_stft, -1)
                        stfts2[name] = list_stft

                    if "eeg" in DATATYPE:
                        eegs2[name] = list_eeg

                    if "img" in DATATYPE:
                        eeg_all_region = np.concatenate(list_img, 0)

                        fig = plt.figure(clear=True)
                        fig.patch.set_facecolor("black")
                        amp = 200
                        for ii in range(eeg_all_region.shape[0]):
                            jj = ii * amp + (ii // 4) * amp
                            plt.plot(
                                eeg_all_region[ii, :] + jj, color="red", linewidth=0.5
                            )
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
                        img = np.reshape(
                            img, (img.shape[0], img.shape[1], img.shape[2], 1)
                        )

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
                            plt.plot(
                                eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5
                            )
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
                            plt.plot(
                                eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5
                            )
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

            for i in range(NSPLIT):
                print(f"\nFold {i + 1}")
                model.load_weights(
                    os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
                )
                pred_i = model.predict(test_gen, verbose=1)
                preds.append(pred_i)

            pred = np.mean(preds, axis=0)
            print("\nTest preds shape", pred.shape)

            pred = np.asarray(pred, dtype=np.float64)
            if pred.ndim != 2:
                raise ValueError(
                    f"Unexpected prediction ndim={pred.ndim}, shape={pred.shape}"
                )
            if pred.shape[1] != len(TARGETS):
                raise ValueError(
                    f"Unexpected number of columns in prediction: got {pred.shape[1]}, expected {len(TARGETS)}"
                )

            pred = np.clip(pred, 1e-12, 1.0)
            pred = pred / pred.sum(axis=1, keepdims=True)

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = pred
            sub = sub[["eeg_id"] + list(TARGETS)]
            sub.to_csv("submission.csv", index=False)

            print("Submission shape", sub.shape)
            print(
                "Row-sum min/max:",
                sub[TARGETS].sum(axis=1).min(),
                sub[TARGETS].sum(axis=1).max(),
            )
            sub.head()
