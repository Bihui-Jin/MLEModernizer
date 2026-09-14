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

0.3515837175108384

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by removing the forced pure-Python protobuf setting that’s incompatible with the Kaggle runtime, while keeping determinism settings intact. Then I make weight loading robust: automatically fall back to Kaggle’s dataset root for any available `.h5/.weights.h5` files (or skip missing folds) so the notebook always produces a `submission.csv` instead of crashing. If no pretrained weights are found at all, the code still generate a valid, normalized submission using a safe uniform distribution (score be poor, but it “yield” a submission). Finally, I fix the `row.sign_id` bug in the generator (not present in this dataset) to prevent latent runtime errors if `img/stft` are used outside test mode.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash that prevents the pipeline from running by explicitly forcing the Python protobuf implementation before importing TensorFlow (this addresses the `MessageFactory.GetPrototype` error in this runtime). I also make spectrogram loading conditional on `READ_SPEC_FILES` (currently it loads spectrograms whenever `"spe"` is in `DATATYPE`, causing massive I/O and potential key errors when `spectrograms2` is empty). Finally, I make the test generator robust when a requested modality dict is missing a key by safely substituting zeros, ensuring the script always completes and writes a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the Python protobuf implementation (the C++ one is unavailable in this runtime), which unblocks `tf` so the later cells (generator/model/inference) can run. Then I keep the existing inference/weight-loading logic but make execution robust by ensuring all required globals are defined in the same run (cells start at 1) and by compiling the model before `predict()` to avoid occasional TF/Keras runtime complaints. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns and row-normalized probabilities (even if no weights are found), so you get a “yielded” submission end-to-end.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment setting and instead using a safe import fallback (and if TF still cannot import in this runtime, the script fall back to a statistically reasonable label-prior submission rather than crashing). I also correct a logic issue that currently makes the code skip loading spectrograms even though `DATATYPE` includes `"spe"`, which leads to near-zero inputs and a much worse KL score; this change preserves the intended core inference logic and should move the score substantially toward your target. Finally, I ensure the submission always matches the sample’s required columns/order and that probabilities are strictly normalized and clipped.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this runtime otherwise raises `MessageFactory.GetPrototype` on TF import). Then I correct a data-loading logic bug so EEG/image inputs aren’t silently all-zeros when `DATATYPE` includes `"eeg"`/`"img"` (previously those flags were hardcoded False), which should move the KL score materially toward your target without changing the model architecture or inference approach. Finally, I keep the existing weight-discovery/ensemble logic but make it robust to missing model folders and ensure the submission is always written with correct columns and normalized probabilities.'

# 9. Code solution

## === cell 0
import os
import io
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        "WARNING: TensorFlow failed to import; will fall back to prior-based submission."
    )
    print("TF import error:", repr(e))

try:
    from sklearn.metrics import confusion_matrix  # noqa: F401
except Exception as e:
    print("sklearn import skipped:", repr(e))

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024031201"
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
READ_EEG_FILES = "eeg" in DATATYPE or "img" in DATATYPE or "stft" in DATATYPE
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

if TF_AVAILABLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
else:
    strategy = None

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
if TF_AVAILABLE:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism enable skipped:", repr(e))

MIX = True
if TF_AVAILABLE:
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision option not available, continuing:", repr(e))
    else:
        print("Using full precision")


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


TRAIN_CSV = _first_existing(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    "/kaggle/data/hms-harmful-brain-activity-classification/train.csv",
    "./input/hms-harmful-brain-activity-classification/train.csv",
)
TEST_CSV = _first_existing(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "/kaggle/data/hms-harmful-brain-activity-classification/test.csv",
    "./input/hms-harmful-brain-activity-classification/test.csv",
)
TEST_SPE_DIR = _first_existing(
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/",
    "/kaggle/data/hms-harmful-brain-activity-classification/test_spectrograms/",
    "./input/hms-harmful-brain-activity-classification/test_spectrograms/",
)
TEST_EEG_DIR = _first_existing(
    "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/",
    "/kaggle/data/hms-harmful-brain-activity-classification/test_eegs/",
    "./input/hms-harmful-brain-activity-classification/test_eegs/",
)

df = pd.read_csv(TRAIN_CSV)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

train_votes = df[TARGETS].astype(np.float64).values
train_prior = train_votes.sum(axis=0)
train_prior = train_prior / np.maximum(train_prior.sum(), 1.0)
print("Train prior:", dict(zip(TARGETS, train_prior.round(6))))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    import albumentations as albu  # noqa: F401
except Exception as e:
    print("albumentations import skipped (not required):", repr(e))

if TF_AVAILABLE:
    TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
    TARS2 = {x: y for y, x in TARS.items()}

if TF_AVAILABLE:

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
            self.stfts = stfts if stfts is not None else {}
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
                    (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            y = np.zeros((len(indexes), 6), dtype="float32")

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
                                max(round((600 / 2 - 256) / 2), 0) : min(
                                    (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
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
                            x_eeg[j, 1:5, :, k] = eeg
                        x_eeg[j, :, :, k] = (
                            x_eeg[j, :, :, k]
                            - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                    if "img" in DATATYPE:
                        if row.eeg_id in self.imgs:
                            img = self.imgs[row.eeg_id][:, :, k]
                            x_img[j, :, :, k] = img

                    if "stft" in DATATYPE:
                        if row.eeg_id in self.stfts:
                            stft = self.stfts[row.eeg_id][:, :, k]

                            stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                            stft = np.log(stft)
                            stft = np.nan_to_num(stft, nan=0.0)
                            stft = np.round(
                                (stft - self.cmin) / (self.cmax - self.cmin) * 255
                            )
                            shape0, shape1 = stft.shape[0], stft.shape[1]
                            stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                            stft = np.array(stft, dtype=np.int16)
                            stft = self.cmaps[stft]
                            stft = np.reshape(stft, (shape0, shape1, 3))
                            stft = np.array(
                                tf.image.resize(stft, ((HIGH - 32), LENGTH)),
                                dtype=np.float32,
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
                    label = row[TARGETS].values.astype(np.float32)

                    if self.mode == "train" and np.any(label == 0):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label != xx] = label[label != xx]
                        imax = int(np.argmax(label))
                        label[imax] = max(label[imax] - 5 * xx, 1e-6)

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
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5
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




## === cell 2
if TF_AVAILABLE:

    def _efficientnet_b0_backbone(name: str):
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            name=name,
        )
        return base

    def build_model():
        l2norm = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2"
        )

        inp = list()

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1, name="spe_concat_regions")(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = _efficientnet_b0_backbone("efficientnetb0_spe")
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D(name="spe_gap")(x_spe)
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

            x_eeg = tf.keras.layers.Concatenate(axis=1, name="eeg_concat_regions")(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )
            x_eeg = tf.keras.layers.Concatenate(axis=3, name="eeg_to_rgb")(
                [x_eeg, x_eeg, x_eeg]
            )

            base_model_eeg = _efficientnet_b0_backbone("efficientnetb0_eeg")
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="eeg_gap")(x_eeg)
            x_eeg = l2norm(x_eeg)

            inp.append(inp_eeg)

            if "spe" in DATATYPE:
                y = tf.keras.layers.Concatenate(axis=1, name="concat_spe_eeg")(
                    [y, x_eeg]
                )
            else:
                y = x_eeg

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="inp_img")
            x_img1 = inp_img[:, :, :, 0:1]
            x_img2 = inp_img[:, :, :, 1:2]
            x_img3 = inp_img[:, :, :, 2:3]
            x_img4 = inp_img[:, :, :, 3:4]
            x_img = tf.keras.layers.Concatenate(axis=1, name="img_concat_regions")(
                [x_img1, x_img2, x_img3, x_img4]
            )
            x_img = tf.keras.layers.Concatenate(axis=3, name="img_to_rgb")(
                [x_img, x_img, x_img]
            )

            base_model_img = _efficientnet_b0_backbone("efficientnetb0_img")
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D(name="img_gap")(x_img)
            x_img = l2norm(x_img)

            inp.append(inp_img)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="concat_prev_img")(
                    [y, x_img]
                )
            else:
                y = x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1, name="stft_concat_regions")(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = _efficientnet_b0_backbone("efficientnetb0_stft")
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D(name="stft_gap")(x_stft)
            x_stft = l2norm(x_stft)

            inp.append(inp_stft)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="concat_prev_stft")(
                    [y, x_stft]
                )
            else:
                y = x_stft

        y = tf.keras.layers.Dense(
            6, activation="softmax", dtype="float32", name="head"
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
        return model




## === cell 3
from scipy import signal


def _list_weight_candidates(base_dir: str, stage: int):
    cands = []
    if base_dir and os.path.isdir(base_dir):
        for i in range(5):
            cands.append((i, os.path.join(base_dir, f"f{i}_stage{stage}.h5")))
            cands.append((i, os.path.join(base_dir, f"f{i}_stage{stage}.weights.h5")))
        for root, _, files in os.walk(base_dir):
            for fn in files:
                if (
                    fn.endswith(".h5")
                    and f"stage{stage}" in fn
                    and (
                        "f0" in fn
                        or "f1" in fn
                        or "f2" in fn
                        or "f3" in fn
                        or "f4" in fn
                    )
                ):
                    fold = None
                    for i in range(5):
                        if f"f{i}" in fn:
                            fold = i
                            break
                    if fold is not None:
                        cands.append((fold, os.path.join(root, fn)))
    seen = set()
    out = []
    for fold, p in cands:
        if p not in seen:
            seen.add(p)
            out.append((fold, p))
    return out


test_df = pd.read_csv(TEST_CSV)
print("Test shape", test_df.shape)
print(test_df.head())

if not TF_AVAILABLE:
    pred = np.tile(train_prior.reshape(1, -1), (len(test_df), 1)).astype(np.float64)
    pred = np.clip(pred, 1e-8, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    sub[list(TARGETS)] = pred.astype(np.float32)
    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)
    print("Wrote fallback submission.csv (TF unavailable). Shape:", sub.shape)
else:
    if ("spe" in DATATYPE) and READ_SPEC_FILES:
        PATH2 = TEST_SPE_DIR
        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(os.path.join(PATH2, f))
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()
    else:
        spectrograms2 = {}

    PATH2 = TEST_EEG_DIR
    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    test_eeg_ids = set(test_df.eeg_id.values.tolist())

    if READ_EEG_FILES:
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(os.path.join(PATH2, f))
            name = int(f.split(".")[0])

            if name in test_eeg_ids:
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
                    time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                    list_img.append(eeg[:, time_start:time_stop])

                    if "stft" in DATATYPE:
                        ff, tt, pp = signal.spectrogram(
                            eeg[
                                :,
                                round(time_temp * SFREQ) : round(
                                    (time_temp + 50) * SFREQ
                                ),
                            ],
                            fs=SFREQ,
                            nperseg=232,
                            noverlap=194,
                        )
                        pp = pp[:, ff <= 40, :]
                        pp = np.mean(pp, 0)
                        list_stft.append(np.reshape(pp, (pp.shape[0], pp.shape[1], 1)))

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
                    amp = 100
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

                    imgs2[name] = img
        print()
    else:
        print("Skipping EEG parquet processing (no EEG-derived modalities enabled).")

    preds = []
    with strategy.scope():
        model = build_model()
        model.compile(optimizer="adam", loss="kullback_leibler_divergence")

    test_gen = DataGenerator(
        test_df,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
        stfts=stfts2,
    )

    base_dirs = []
    base_dirs.append(LOAD_MODELS_FROM)
    base_dirs.append("/kaggle/input")
    base_dirs.append("/kaggle/working")

    loaded_any = False
    used_folds = set()

    for base_dir in base_dirs:
        candidates = _list_weight_candidates(base_dir, STAGETEST)
        for fold, wpath in candidates:
            if fold in used_folds:
                continue
            if os.path.exists(wpath):
                print(f"Loading weights for fold {fold} from: {wpath}")
                model.load_weights(wpath)
                pred_i = model.predict(test_gen, verbose=1)
                preds.append(pred_i)
                used_folds.add(fold)
                loaded_any = True
        if len(used_folds) >= 5:
            break

    if not loaded_any:
        print(
            "WARNING: No model weights found. Using train-prior predictions for a valid submission."
        )
        pred = np.tile(train_prior.reshape(1, -1), (len(test_df), 1)).astype(np.float64)
    else:
        pred = np.mean(preds, axis=0)

    print("Test preds shape", np.asarray(pred).shape)

    pred = np.asarray(pred, dtype=np.float64)
    pred = np.nan_to_num(pred, nan=1.0 / 6, posinf=1.0 / 6, neginf=1.0 / 6)
    pred = np.clip(pred, 1e-8, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    sub[list(TARGETS)] = pred.astype(np.float32)

    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sums (min/max):",
        sub.iloc[:, -6:].sum(axis=1).min(),
        sub.iloc[:, -6:].sum(axis=1).max(),
    )
