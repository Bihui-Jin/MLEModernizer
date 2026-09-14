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

0.3512232189003345

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the Python protobuf implementation before importing TensorFlow (this is the root cause of the `MessageFactory GetPrototype` error). Then I make the model-weights loading robust by checking whether the provided `/kaggle/input/models2024040502` directory exists and contains the expected `.h5` files; if not, the code fall back to a valid, well-formed uniform-probability submission (so you always get a `submission.csv`). Finally, I keep the prediction post-processing (clipping + renormalization) to guarantee each row sums to 1 and avoid submission failures under the KL-divergence metric.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash reliably by setting the protobuf environment variables before *any* TensorFlow-related import and by falling back safely if TF still can’t import in this runtime. Then, to move the KL score down (better) toward your target, I remove the “uniform fallback” behavior and instead use a strong, competition-safe fallback: class-prior probabilities computed from `train.csv` (normalized vote proportions), which is a minimal logic change and typically far better than uniform when models/weights are unavailable. I also make submission column ordering and row-sum constraints strictly match `sample_submission.csv`, with clipping + renormalization to avoid invalid submissions. Core model logic remains unchanged; only robustness and prediction fallback/calibration are adjusted.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash that currently happens before your `try/except` can catch it by moving the protobuf environment variables to the very top of the script (before any TF-related import path is touched). I also make GPU selection safe in Kaggle’s typical single-GPU environment (setting `CUDA_VISIBLE_DEVICES="0,1"` can cause unexpected behavior), without changing any model logic. Finally, I keep your existing prior-based fallback submission behavior (since current score is far from the target and TF/weights may be missing) while ensuring the submission is always well-formed, correctly ordered, and row-normalized for KL divergence.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash that currently happens before your `try/except` by forcing a compatible protobuf runtime *before any TF-related import* and (if needed) pinning protobuf to the pure-Python API via environment variables in a way that works on Kaggle/Python 3.12. I also make the script resilient so it always completes and writes a valid `submission.csv`: if TensorFlow still can’t import or model weights are missing, it produce the train-derived class-prior probabilities (better than uniform for KL). Finally, I keep your model/data logic unchanged, but add strict submission validation (column order + row-sum normalization) to avoid invalid submissions and accidental score blow-ups.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash that happens before your `try/except` by forcing a compatible protobuf runtime *before any TensorFlow import occurs* (including the indirect imports TF triggers). Then I make the pipeline always complete by cleanly falling back to a strong prior-based submission when TF can’t import or model weights aren’t present, while keeping your model/prediction logic unchanged when weights exist. Finally, I harden submission validity for the KL metric by enforcing column order to match `sample_submission.csv` and by clipping + renormalizing probabilities so every row sums to 1.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 2
print(DATATYPE)

LOAD_MODELS_FROM = "models2024040502"
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

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

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

import io
from PIL import Image

import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf = None
try:
    import tensorflow as tf
except Exception as e:
    print("WARNING: TensorFlow failed to import; will run fallback submission only.")
    print("TF import error:", repr(e))

try:
    from sklearn.metrics import confusion_matrix
except Exception:
    confusion_matrix = None

librosa = None
if "stft" in DATATYPE:
    try:
        import librosa as _librosa

        librosa = _librosa
    except Exception:
        librosa = None

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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if tf is not None:
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
            print("Mixed precision request ignored (not supported in this TF build)")
    else:
        print("Using full precision")



## === cell 2
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



## === cell 3
if tf is not None:
    try:
        import albumentations as albu  # kept as in original (even if unused)
    except Exception:
        albu = None
else:
    albu = None

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if tf is not None:

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
            if "img" in DATATYPE:
                x_img = np.zeros(
                    (len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32"
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
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
                    if "spe" in DATATYPE:
                        spe = self.specs[row.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T
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
                            max(round((600 / 2 - 256) / 2), 0) : min(
                                (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
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
                            stft = self.stfts[row.eeg_id][:, :, k]
                        else:
                            stft = self.stfts[row.sign_id][:, :, k]

                        stft = np.nan_to_num(stft, nan=0.0)
                        cmin = 0
                        cmax = 60
                        stft = np.clip(stft, cmin, cmax)
                        stft = np.round((stft - cmin) / (cmax - cmin) * 255)

                        shape0, shape1 = stft.shape[0], stft.shape[1]
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)
                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (shape0, shape1, 3))

                        x_stft[
                            j,
                            round((HIGH - stft.shape[0]) / 2) : round(
                                (HIGH + stft.shape[0]) / 2
                            ),
                            :,
                            :,
                            k,
                        ] = stft
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
                    xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                    x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
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




## === cell 4
if tf is not None:
    from tensorflow.keras.applications import EfficientNetB0

    def build_model(TARGETS_PRETRAIN):
        inp = list()
        y = None

        l2_layer = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_spe[:, :, :, :, 0],
                    inp_spe[:, :, :, :, 1],
                    inp_spe[:, :, :, :, 2],
                    inp_spe[:, :, :, :, 3],
                ]
            )

            base_model_spe = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_spe",
            )
            base_model_spe._name = "spe_extractor"
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
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = l2_layer(x_spe)
            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_eeg[:, :, :, :, 0],
                    inp_eeg[:, :, :, :, 1],
                    inp_eeg[:, :, :, :, 2],
                    inp_eeg[:, :, :, :, 3],
                ]
            )

            base_model_eeg = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_eeg",
            )
            base_model_eeg._name = "eeg_extractor"
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
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2_layer(x_eeg)
            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_img[:, :, :, :, 0],
                    inp_img[:, :, :, :, 1],
                    inp_img[:, :, :, :, 2],
                    inp_img[:, :, :, :, 3],
                ]
            )

            base_model_img = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_img",
            )
            base_model_img._name = "img_extractor"
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
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = l2_layer(x_img)
            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_stft[:, :, :, :, 0],
                    inp_stft[:, :, :, :, 1],
                    inp_stft[:, :, :, :, 2],
                    inp_stft[:, :, :, :, 3],
                ]
            )

            base_model_stft = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_stft",
            )
            base_model_stft._name = "stft_extractor"
            if NEEDTRAIN:
                if PLATFORM == "local":
                    base_model_stft.load_weights(
                        "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_stft.load_weights(
                        "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                    )

            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = l2_layer(x_stft)
            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                if y is not None
                else x_stft
            )

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

    def my_loss(y_ture, y_pred):
        y_pred1 = y_pred[:, 5:6]
        y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    print("Test shape", test.shape)

    def _class_prior_from_train(train_df, targets):
        votes = train_df[targets].astype("float64").values
        totals = votes.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        probs = votes / totals
        prior = probs.mean(axis=0)
        prior = np.clip(prior, 1e-7, 1.0)
        prior = prior / prior.sum()
        return prior.astype("float32")

    class_prior = _class_prior_from_train(df, list(TARGETS))
    print("Class prior fallback:", dict(zip(TARGETS, class_prior.round(6))))

    def _ensure_valid_probs(pred, eps=1e-7):
        pred = np.asarray(pred, dtype=np.float64)
        pred = np.clip(pred, eps, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        return pred.astype(np.float32)

    def _write_prior_submission(out_path="submission.csv"):
        sub_p = pd.DataFrame({"eeg_id": test.eeg_id.values})
        pred = np.tile(class_prior.reshape(1, -1), (len(test), 1))
        pred = _ensure_valid_probs(pred)
        sub_p[list(sample_sub.columns[1:])] = pred
        sub_p = sub_p[["eeg_id"] + list(sample_sub.columns[1:])]
        sub_p.to_csv(out_path, index=False)
        print(f"Wrote prior fallback submission to {out_path} with shape {sub_p.shape}")
        return sub_p

    expected_weights = [
        os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5") for i in range(5)
    ]
    have_all_weights = os.path.isdir(LOAD_MODELS_FROM) and all(
        os.path.exists(p) for p in expected_weights
    )

    if (tf is None) or (not have_all_weights):
        print(
            "WARNING: Using fallback submission (TF unavailable and/or model weights missing)."
        )
        if not os.path.isdir(LOAD_MODELS_FROM):
            print("Looked for directory:", LOAD_MODELS_FROM, "(not found)")
        else:
            missing = [p for p in expected_weights if not os.path.exists(p)]
            if missing:
                print("Missing weight files (showing up to 5):", missing[:5])
        sub = _write_prior_submission("submission.csv")
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            else:
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
        else:
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        imgs2 = {}
        stfts2 = {}
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        test_eeg_ids = set(test.eeg_id.values.tolist())
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids:
                continue

            eeg_default = pd.read_parquet(f"{PATH2}{f}")

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

                if "stft" in DATATYPE:
                    if librosa is None:
                        raise ImportError(
                            "librosa is required when DATATYPE includes 'stft'"
                        )
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

            if "stft" in DATATYPE:
                list_stft = np.concatenate(list_stft, 2)
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

        preds = []
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

        for i in range(5):
            print(f"Fold {i + 1}")
            wpath = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = _ensure_valid_probs(pred, eps=1e-7)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[list(sample_sub.columns[1:])] = pred
        sub = sub[["eeg_id"] + list(sample_sub.columns[1:])]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row sum check (min/max):",
            sub.iloc[:, 1:].sum(axis=1).min(),
            sub.iloc[:, 1:].sum(axis=1).max(),
        )
        sub.head()
