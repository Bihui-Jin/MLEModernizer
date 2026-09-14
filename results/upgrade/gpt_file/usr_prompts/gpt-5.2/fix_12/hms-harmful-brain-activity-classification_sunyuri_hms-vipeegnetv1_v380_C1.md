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

0.2948451937752093

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf runtime crash by disabling C++ protobuf implementations before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment. Then I fix inference so it can always produce a valid `submission.csv`: if no external `models*` directory with weights exists, it fall back to a safe “no-model” baseline that predicts the global normalized vote prior from `train.csv` (valid probabilities summing to 1). Finally, I make the data path detection robust (prefer `/kaggle/input/hms-harmful-brain-activity-classification`) and keep your core modeling/training code unchanged, only adding guardrails so it runs end-to-end within the notebook and writes a valid CSV.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top (before any TensorFlow/Keras import path is touched) and by adding a safe fallback that skips TF entirely if TF import still fails. Then, to improve the score from the current prior-only baseline (1.41937) toward the target (0.2948), I enable inference-time loading of the provided EfficientNetV2B0 “notop” weights (if present in `/kaggle/input/pre-trained-weights/`) even when `NEEDTRAIN=False`, which is score-improving but keeps the same architecture and training semantics. Finally, I make submission generation always valid by enforcing probability normalization, correct column order, and writing `submission.csv` in all branches.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import is touched, and I also disable mixed precision by default to avoid additional TF edge-case failures on Kaggle. Then, to improve score from the current prior-only baseline (1.41937) toward the target (0.2948), I keep your architecture and inference semantics intact but make the code actually find and load existing fold weights from typical Kaggle input locations (instead of almost always falling back to the prior). Finally, I harden submission generation by guaranteeing correct column order, probability normalization, and always writing `submission.csv` even if weights or TF are unavailable.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation **before any TensorFlow/Keras import** and by explicitly disabling the C++ protobuf path, which is the root cause in Kaggle’s TF build. Then I make inference actually use available trained fold weights by searching common Kaggle input locations more robustly (including nested directories), instead of silently falling back to the weak prior baseline that produced 1.41937. Finally, I harden submission generation: enforce correct column order, safe clipping, and per-row normalization so the CSV is always valid and accepted by Kaggle.'
- What this solution (achieved 1.41937) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation *before anything else* and also setting `TF_USE_LEGACY_KERAS=1`, which is a known stabilizer in Kaggle’s TF environment for this competition. Next, I switch Kaggle inference to actually run the model by default (`NEEDTRAIN=False`) and make the weights search robust to both `stage1` and `stage2` checkpoints so it doesn’t silently fall back to the weak prior baseline that caused the 1.419 score. Finally, I harden inference to handle missing outputs (e.g., if only one head exists) and always normalize/clamp probabilities and write a valid `submission.csv` with the correct columns and row alignment.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow import, and I also make the TF import failure path reliably fall back to a valid prior-based submission. Next, I correct a few inference-time bugs that prevent the model path from working even when weights exist (missing `x_stft` allocation when only `stft` is used; potential `output_eeg/output_stft` undefined in `build_model`; and a bad reshape in `_extract_pred_output`). Finally, I ensure the submission is always written with correct column order and row-wise probability normalization, which is score-safe and prevents Kaggle submission rejection; when weights are found, this should improve score substantially versus the prior-only 1.41937 baseline.'
- What this solution (achieved 1.41937) has done: 'The timeout is driven mainly by two things: (1) training 5 folds × (stage1 15 epochs + stage2 ~5 epochs) with heavy EfficientNetV2B0 backbones, and (2) extremely slow per-batch Python work in `DataGenerator` (pandas row selection, repeated `df` filtering/group lookups, and per-sample STFT indexing). To finish under 600s without changing the core model/training logic, the fastest safe fix is to avoid training entirely in the Kaggle runtime and run inference only (loading existing fold weights when present, else falling back to the already-implemented prior baseline). Additionally, inference is sped up by eliminating redundant model cloning/building, forcing `use_multiprocessing=False` for predict (reduces IPC overhead), precomputing STFT time indices for test batches, and using faster numpy access paths while keeping identical preprocessing and prediction semantics.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by setting protobuf/TensorFlow environment toggles before any TensorFlow-related imports and by adding a robust fallback that still writes a valid `submission.csv` if TF cannot import. Then I fix a Keras `Sequence` compatibility issue where `__getitem__` always returns `(x,y,sample_weight)` even in test mode (Keras `predict` expects only `x`), which can silently break inference or produce wrong outputs. Finally, I harden inference by ensuring `DataGenerator` uses the correct global `df_group_indices` lookup (or none) without slow pandas filtering when not needed, and I always enforce per-row probability normalization and correct column order in the submission (score-neutral but prevents submission rejection).'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

NEEDTRAIN = False  # keep original intent (inference-focused)
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

cwd = os.getcwd()
parts = cwd.split(os.sep)
if len(parts) > 1 and parts[1] == "home":
    PLATFORM = "local"
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif len(parts) > 1 and parts[1] == "kaggle":
    PLATFORM = "kaggle"
    if os.path.isdir("/kaggle/input/"):
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "kaggle"

DATATYPE = ["eeg", "stft"]  # *** spe, eeg, stft, img ***
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    if os.path.exists(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    ):
        LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
    else:
        LOAD_DATA_FROM = "/kaggle/input"
else:
    LOAD_DATA_FROM = "/kaggle/input"

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import zoom
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    optimizers = None
    clone_model = None
    print(
        "WARNING: TensorFlow import failed; will write a prior-based submission.csv. Error:",
        repr(e),
    )

if TF_AVAILABLE:
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
    except Exception as e:
        print("WARNING: enable_op_determinism failed:", repr(e))

    MIX = False
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("Using full precision")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TARGETS_RAW = [c + "_raw" for c in TARGETS]

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
y_data = train[TARGETS].values.astype(np.float32)
train[TARGETS_RAW] = y_data
y_norm = y_data / np.clip(y_data.sum(axis=1, keepdims=True), 1e-6, None)
train[TARGETS] = y_norm

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

DF_GROUP_INDICES = None



## === cell 2
pass



## === cell 3
pass



## === cell 4
if TF_AVAILABLE:

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
            df_group_indices=None,
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
            self.df_group_indices = df_group_indices

            self._eeg_id = self.dataframe["eeg_id"].to_numpy()
            if "spectrogram_id" in self.dataframe.columns:
                self._spectrogram_id = self.dataframe["spectrogram_id"].to_numpy()
            else:
                self._spectrogram_id = None
            if "sign_id" in self.dataframe.columns:
                self._sign_id = self.dataframe["sign_id"].to_numpy()
            else:
                self._sign_id = None

            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.dataframe) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            if self.mode == "test":
                return x
            if self.sample_weights:
                return x, y, sample_weights
            return x, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

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
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")
            if "stft" in DATATYPE:
                x_stft = np.zeros(
                    (len(indexes), 16, STFT_HIGH, STFT_WIDE), dtype="float32"
                )

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                    eeg_id = int(self._eeg_id[i])
                    spectrogram_id = (
                        int(self._spectrogram_id[i])
                        if self._spectrogram_id is not None
                        else None
                    )
                    sign_id = (
                        int(self._sign_id[i]) if self._sign_id is not None else None
                    )
                    sample_weight = 1.0
                    row = None
                else:
                    row = self.dataframe.iloc[i]
                    sign_id = row.sign_id
                    sample_weight = float(np.sum(row[TARGETS_RAW].values)) / 20.0

                    key = (
                        int(row.eeg_id),
                        float(row.seizure_vote_raw),
                        float(row.lpd_vote_raw),
                        float(row.gpd_vote_raw),
                        float(row.lrda_vote_raw),
                        float(row.grda_vote_raw),
                    )
                    idxs = (
                        self.df_group_indices.get(key, None)
                        if self.df_group_indices
                        else None
                    )
                    if idxs is None:
                        rows = df.loc[
                            (df.eeg_id == row.eeg_id)
                            * (df.seizure_vote == row.seizure_vote_raw)
                            * (df.lpd_vote == row.lpd_vote_raw)
                            * (df.gpd_vote == row.gpd_vote_raw)
                            * (df.lrda_vote == row.lrda_vote_raw)
                            * (df.grda_vote == row.grda_vote_raw),
                            :,
                        ].reset_index(drop=True)
                    else:
                        rows = df.iloc[np.fromiter(idxs, dtype=np.int64)].reset_index(
                            drop=True
                        )

                    if self.mode == "train":
                        rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                            drop=True
                        )
                        row = rows.loc[0, :]
                    elif self.mode == "valid":
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )

                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds
                    eeg_id = int(row.eeg_id)
                    spectrogram_id = int(row.spectrogram_id)

                if "spe" in DATATYPE:
                    spe = []
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                self.specs[spectrogram_id][
                                    r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                ].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]
                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-eeg_id]
                    start_val = r_eeg - float(np.min(stft_t))
                    end_val = 50 + r_eeg - float(np.min(stft_t))
                    r_stft = int(np.searchsorted(stft_t, start_val, side="left"))
                    r_stft2 = int(np.searchsorted(stft_t, end_val, side="right") - 1)
                    r_stft = max(r_stft, 0)
                    r_stft2 = min(r_stft2, len(stft_t) - 1)
                    stft = self.stfts[eeg_id][:, :, r_stft : (r_stft2 + 1)]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                    stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    exp_min, exp_max = -4, 6
                    spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    spe = np.log(spe)

                    if (spe.shape[1] != SPE_HIGH) or (spe.shape[2] != SPE_WIDE):
                        spe2 = np.zeros(
                            (spe.shape[0], SPE_HIGH, SPE_WIDE), dtype=np.float32
                        )
                        for k in range(4):
                            scaled_arr = zoom(
                                spe[k],
                                (SPE_HIGH / spe.shape[1], SPE_WIDE / spe.shape[2]),
                                order=1,
                            )
                            spe2[k, :, :] = scaled_arr
                        spe = spe2.copy()

                    if self.mode == "train":
                        spe2 = spe.copy()
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[2]
                            spe[2] = spe2[0]
                        if np.random.rand() > 0.5:
                            spe[1] = spe2[3]
                            spe[3] = spe2[1]
                        if np.random.rand() > 0.5:
                            spe = spe[::-1, :, :]

                        if np.random.rand() > 0.5:
                            for ii in range(spe.shape[0]):
                                m1 = round(np.random.rand() * spe.shape[2] / 2)
                                m2 = round(np.random.rand() * spe.shape[2] / 2)
                                if np.random.rand() > 0.5:
                                    m1 = spe.shape[2] - m1
                                    m2 = spe.shape[2] - m2
                                m_min = min(m1, m2)
                                m_max = min(
                                    max(m1, m2), m_min + round(spe.shape[2] * 0.05)
                                )
                                spe[ii, :, m_min:m_max] = 0

                    spe = (spe - exp_min) / (exp_max - exp_min) * 255
                    spe = np.clip(spe, a_min=0, a_max=255)
                    x_spe[j] = spe

                if "eeg" in DATATYPE:
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]

                    if self.mode == "train":
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

                        eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                            0 : round(EEG_CHANNEL_USED / 2), :
                        ][np.random.permutation(8), :]
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                            -round(EEG_CHANNEL_USED / 2) :, :
                        ][np.random.permutation(8), :]
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                        if np.random.rand() > 0.5:
                            eeg = eeg[::-1, :]
                    else:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                    eeg = np.clip(eeg, a_min=-255, a_max=255)
                    eeg = eeg + 255
                    eeg = eeg / 2
                    x_eeg[j] = eeg

                if "stft" in DATATYPE:
                    exp_min, exp_max = 0, 8
                    stft = np.clip(stft, a_min=0, a_max=np.exp(exp_max))
                    stft = np.log1p(stft)

                    if self.mode == "train":
                        stft[0 : round(stft.shape[0] / 2), :, :] = stft[
                            0 : round(stft.shape[0] / 2), :, :
                        ][np.random.permutation(stft.shape[0] // 2), :, :]
                        stft[-round(stft.shape[0] / 2) :, :, :] = stft[
                            -round(stft.shape[0] / 2) :, :, :
                        ][np.random.permutation(stft.shape[0] // 2), :, :]
                        stft2 = stft.copy()
                        stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                            0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                        ]
                        stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                            3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                        ]
                        stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                            1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                        ]
                        stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                            2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                        ]
                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :, :]
                    else:
                        stft2 = stft.copy()
                        stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                            0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                        ]
                        stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                            3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                        ]
                        stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                            1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                        ]
                        stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                            2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                        ]

                    stft = (stft - exp_min) / (exp_max - exp_min) * 255
                    stft = np.clip(stft, a_min=0, a_max=255)
                    x_stft[j] = stft

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)
                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                    x_img[j] = img

                if self.mode != "test":
                    yj = row[TARGETS].values.astype(np.float32)
                    y[j] = yj / np.clip(np.sum(yj), 1e-6, None)
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

            return x, {"output": y, "output_eeg": y, "output_stft": y}, sample_weights




## === cell 5
if TF_AVAILABLE:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
            self.lr_max = lr_max
            self.lr_min = lr_min

        def __call__(self, step):
            step = step + 1
            if step < self.warm_step:
                lr = self.lr_max / self.warm_step * step
            else:
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
            attn_output, weights = self.att(
                inputs, inputs, return_attention_scores=True
            )
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
if TF_AVAILABLE:

    def _maybe_load_notop_weights(base_model):
        if PLATFORM == "local":
            wpath = f"./input/pre-trained-weights/{base_model.name}_notop.h5"
        else:
            wpath = f"/kaggle/input/pre-trained-weights/{base_model.name}_notop.h5"
        if os.path.exists(wpath):
            try:
                base_model.load_weights(wpath)
                return True
            except Exception as e:
                print(
                    "WARNING: failed to load pretrained weights:",
                    wpath,
                    "error:",
                    repr(e),
                )
        return False

    def build_model():
        inp = []
        y = 0

        output_eeg = None
        output_stft = None

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                name="eeg",
            )
            x_eeg_raw = tf.keras.layers.Reshape(
                (inp_eeg.shape[1], inp_eeg.shape[2], 1)
            )(inp_eeg)

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
            base_model_eeg.name = "eeg_extractor"
            _maybe_load_notop_weights(base_model_eeg)

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = x_eeg[
                :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
            ]
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            output_eeg = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32", name="output_eeg"
            )(x_eeg)

            x_eeg = tf.keras.layers.Dense(32)(x_eeg)
            x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)

            if y == 0:
                y = x_eeg * 1
            else:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(16, STFT_HIGH, STFT_WIDE), name="stft")
            x_stft = tf.keras.layers.Reshape(
                (inp_stft.shape[1], inp_stft.shape[2], inp_stft.shape[3], 1)
            )(inp_stft)
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [x_stft[:, k, :, :, :] for k in range(16)]
            )

            base_model_stft = tf.keras.applications.EfficientNetV2B0(
                include_top=False, weights=None, include_preprocessing=True
            )
            base_model_stft.name = "stft_extractor"
            _maybe_load_notop_weights(base_model_stft)

            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = tf.keras.layers.Dropout(0.5)(x_stft)

            inp.append(inp_stft)
            output_stft = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32", name="output_stft"
            )(x_stft)

            x_stft = tf.keras.layers.Dense(32)(x_stft)
            x_stft = tf.keras.layers.LeakyReLU()(x_stft)

            if y == 0:
                y = x_stft * 1
            else:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])

        output = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32", name="output"
        )(y)

        if output_eeg is None:
            output_eeg = output
        if output_stft is None:
            output_stft = output

        model = tf.keras.Model(inputs=inp, outputs=[output, output_eeg, output_stft])
        return model




## === cell 7
def _write_prior_submission(test_df: pd.DataFrame, out_path: str = "submission.csv"):
    prior = df[TARGETS].sum(axis=0).values.astype(np.float64)
    prior = prior / np.clip(prior.sum(), 1e-12, None)
    preds_all = np.tile(prior.reshape(1, -1), (len(test_df), 1))
    preds_all = np.nan_to_num(preds_all, nan=1.0 / len(TARGETS))
    preds_all = np.clip(preds_all, 1e-12, 1.0)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    sub[TARGETS] = preds_all
    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv(out_path, index=False)
    print("Wrote prior-based submission.csv with shape", sub.shape)
    print(sub.head())


def _find_models_dir(expected_splits: int) -> tuple[str | None, int | None]:
    candidates = []
    if LOAD_MODELS_FROM:
        candidates.append(LOAD_MODELS_FROM)

    candidates += ["./models", "/kaggle/working/models"]

    if PLATFORM == "kaggle":
        candidates += [
            "/kaggle/input/models",
            "/kaggle/input/model",
            "/kaggle/input/hms-harmful-brain-activity-classification/models",
            "/kaggle/input/hms-harmful-brain-activity-classification/model",
        ]
        try:
            for d in os.listdir("/kaggle/input"):
                candidates.append(os.path.join("/kaggle/input", d))
                candidates.append(os.path.join("/kaggle/input", d, "models"))
                candidates.append(os.path.join("/kaggle/input", d, "model"))
        except Exception:
            pass

    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)

    def has_all_fold_files(d: str, stage: int) -> bool:
        if not os.path.isdir(d):
            return False
        for i in range(expected_splits):
            if not os.path.exists(os.path.join(d, f"fold{i}_stage{stage}.weights.h5")):
                return False
        return True

    for stage in (2, 1):
        for d in uniq:
            if has_all_fold_files(d, stage):
                return d, stage
        for d in uniq:
            if os.path.isdir(d):
                try:
                    for sub in os.listdir(d):
                        dd = os.path.join(d, sub)
                        if has_all_fold_files(dd, stage):
                            return dd, stage
                except Exception:
                    pass

    return None, None


def _extract_pred_output(pred_obj, batch_size: int, n_classes: int) -> np.ndarray:
    if isinstance(pred_obj, (list, tuple)):
        pred = np.asarray(pred_obj[0])
    else:
        pred = np.asarray(pred_obj)
    if pred.ndim == 1:
        pred = pred.reshape(batch_size, n_classes)
    return pred


if __name__ == "__main__":
    if not TF_AVAILABLE:
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        _write_prior_submission(test, "submission.csv")
    else:
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        models_dir, stage_use = _find_models_dir(SPLITS)
        if models_dir is None:
            print("No fold weights found; using prior baseline.")
            _write_prior_submission(test, "submission.csv")
        else:
            print("Using model weights from:", models_dir, "stage:", stage_use)

            preds_all = []
            models = []

            model_template = build_model()

            for model_i in range(SPLITS):
                print(f"Fold {model_i + 1}")
                model = clone_model(model_template)
                model.load_weights(
                    os.path.join(
                        models_dir, f"fold{model_i}_stage{stage_use}.weights.h5"
                    )
                )
                models.append(model)

            if "spe" in DATATYPE:
                PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
                files_test = os.listdir(PATH_test)
                print(f"There are {len(files_test)} test spectrogram parquets")
                for i, f in enumerate(files_test):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    tmp = pd.read_parquet(f"{PATH_test}{f}")
                    name = int(f.split(".")[0])
                    spectrograms_test[name] = tmp.iloc[:, 1:].values
                print()

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            if (
                ("spe" in DATATYPE)
                or ("eeg" in DATATYPE)
                or ("stft" in DATATYPE)
                or ("img" in DATATYPE)
            ):
                b2, a2 = signal.butter(
                    3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
                )

                batch_start = 0
                for i, eeg_id in enumerate(test.eeg_id.values):
                    if i % 100 == 0:
                        print(i, ", ", end="")

                    eeg_default = pd.read_parquet(
                        os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                    )
                    arr = eeg_default  # pandas df

                    eeg = []
                    for channel in BRAIN:
                        a_ch, b_ch = channel.split("-")
                        eeg_temp = arr[a_ch].to_numpy() - arr[b_ch].to_numpy()
                        eeg_temp = np.nan_to_num(eeg_temp, nan=0.0)
                        eeg.append(eeg_temp[None, :])
                    eeg = np.concatenate(eeg, axis=0)

                    if SFREQ != RSFREQ:
                        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                    if "stft" in DATATYPE:
                        eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                        nperseg, noverlap, nfft = 128, 128 - 50, 500
                        stft_list = []
                        for ch in range(eeg2.shape[0]):
                            f, t, Z = signal.stft(
                                eeg2[ch].astype(np.float32),
                                fs=RSFREQ,
                                window="hann",
                                nperseg=nperseg,
                                noverlap=noverlap,
                                nfft=nfft,
                                boundary=None,
                                padded=False,
                            )
                            stft_list.append(np.abs(Z).astype(np.float32))
                        ss = np.stack(stft_list, axis=0)
                        ss = ss[:, : round(20 / (RSFREQ / nfft)), :]
                        tt = t.astype(np.float32)
                        ss = np.concatenate(
                            (
                                ss[0 : round(EEG_CHANNEL_USED / 2), :, :],
                                ss[-round(EEG_CHANNEL_USED / 2) :, :, :],
                            ),
                            axis=0,
                        )
                        ss = np.array(ss, dtype=np.float32)
                        tt = np.array(tt, dtype=np.float32)

                    if "img" in DATATYPE:
                        eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                        eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                        test_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                        for j in range(len(test_plot)):
                            eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
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
                                plt.plot(
                                    eeg_plot[ii, :] + 100, color="red", linewidth=0.2
                                )
                                plt.xlim(-5, eeg_plot.shape[1] + 5)
                                plt.ylim(0, 200)
                                plt.axis("off")

                                byte_stream = io.BytesIO()
                                plt.savefig(
                                    byte_stream,
                                    format="png",
                                    bbox_inches="tight",
                                    dpi=100,
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
                                        tf.image.resize(img, (36, IMG_WIDE)),
                                        dtype=np.float32,
                                    )
                                img = img[:, :, 0]
                                img_save[ii, :, :] = img

                            imgs_test[int(test_plot.sign_id[j])] = img_save

                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                    eegshape = eeg.shape[1]
                    eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
                    if filter_range is not None:
                        eeg = signal.filtfilt(b, a, eeg, axis=1)
                    eeg = eeg[:, eegshape : eegshape * 2]
                    eeg = np.array(eeg, dtype=np.float32)

                    if "eeg" in DATATYPE:
                        eegs_test[int(eeg_id)] = eeg
                    if "stft" in DATATYPE:
                        stfts_test[int(eeg_id)] = ss
                        stfts_test[int(-eeg_id)] = tt

                    if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                        batch_end = i + 1
                        batch_df = test.iloc[batch_start:batch_end, :].reset_index(
                            drop=True
                        )

                        test_gen = DataGenerator(
                            batch_df,
                            shuffle=False,
                            sample_weights=False,
                            batch_size=TEST_BATCHSIZE,
                            mode="test",
                            specs=spectrograms_test,
                            eegs=eegs_test,
                            stfts=stfts_test,
                            imgs=imgs_test,
                            df_group_indices=None,
                        )

                        preds = []
                        for model_i in range(SPLITS):
                            pred_obj = models[model_i].predict(
                                test_gen,
                                verbose=0,
                                workers=1,
                                use_multiprocessing=False,
                                max_queue_size=8,
                            )
                            pred = _extract_pred_output(
                                pred_obj, len(batch_df), len(TARGETS)
                            )
                            preds.append(pred)
                        pred = np.mean(preds, axis=0)

                        eegs_test = {}
                        stfts_test = {}
                        imgs_test = {}
                        gc.collect()

                        if len(preds_all) == 0:
                            preds_all = pred.copy()
                        else:
                            preds_all = np.concatenate((preds_all, pred), axis=0)

                        batch_start = batch_end

                print()

            preds_all = np.asarray(preds_all, dtype=np.float64)
            preds_all = np.nan_to_num(
                preds_all,
                nan=1.0 / len(TARGETS),
                posinf=1.0 / len(TARGETS),
                neginf=1.0 / len(TARGETS),
            )
            preds_all = np.clip(preds_all, 1e-12, 1.0)
            preds_all = preds_all / np.clip(
                preds_all.sum(axis=1, keepdims=True), 1e-12, None
            )

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = preds_all
            sub = sub[["eeg_id"] + list(TARGETS)]
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
