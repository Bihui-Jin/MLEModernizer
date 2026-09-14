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

0.3478364216127923

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the incompatible on-the-fly pip installs that trigger the protobuf `MessageFactory.GetPrototype` crash in this Kaggle/Python 3.13 environment, and instead use the preinstalled `tensorflow.keras.applications.EfficientNetB0` to keep the exact same model role (EfficientNetB0 feature extractor) without the missing `efficientnet` package. I also fix the cell numbering and ensure `build_model()` is always defined before inference. Finally, I make the inference pipeline robust by (1) defining `TARGETS` from `sample_submission.csv` to guarantee correct column order, and (2) handling the case where pretrained fold weight files are not present by falling back to a safe uniform-probability submission (so a valid `.csv` is always produced).'
- What this solution (achieved 1.40995) has done: 'I remove the protobuf-related crash by eliminating the problematic mixed-precision/protobuf interaction and ensuring TensorFlow runs deterministically without triggering that code path. Then I fix the core score issue: your model is currently built with `weights=None` for EfficientNetB0, which makes inference essentially random and explains the very poor KL; switching to ImageNet pretrained weights (same EfficientNetB0 feature extractor role) is a minimal change that should substantially lower the KL toward your target. Finally, I make inference robust in this Kaggle layout by resolving the correct dataset root path automatically and always writing a valid `submission.csv` with correct column order and row-normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the crash happening before any training/inference by resolving the protobuf/TensorFlow incompatibility that causes `MessageFactory.GetPrototype` to be missing under this Kaggle Python 3.13 image. The minimal safe way is to force TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which avoids that code path and restores end-to-end execution. I keep your model/data logic intact (same generator, same EfficientNetB0 feature extractor usage, same inference loop) and only adjust the import/runtime setup plus add a small safeguard to always write a valid `submission.csv` even if something unexpected happens. This should also allow the ImageNet-pretrained EfficientNet inference to run (instead of falling back to uniform), which is the necessary step to move KL down toward your target.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by forcing TensorFlow to use the pure-Python protobuf implementation *and* the pure-Python parser implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` code path that breaks under this Kaggle Py3.13 image. I also make GPU selection safe (don’t force `/gpu:0` when no GPU is available) so the script runs on both CPU-only and GPU runtimes. Finally, I keep your model/inference logic intact but ensure Keras `Sequence` returns the right structure in test mode (no dummy `y/sample_weights` needed) to avoid shape/adapter issues during `.predict()` and ensure a valid `submission.csv` is always written with normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'You’re hitting the protobuf/TensorFlow crash before any training/inference runs, so the first minimal fix is to force the pure‑Python protobuf implementation *and* disable C++ parsing before importing TensorFlow; this reliably avoids `MessageFactory.GetPrototype` on Kaggle’s Py3.13 image. Next, I keep your model/data pipeline intact but make test-mode `DataGenerator` return only `x` (not dummy `y/sample_weight`) to avoid Keras adapter edge-cases during `.predict()`. Finally, I ensure the prediction rows always align to the test dataframe order and that the output probabilities are clipped and re-normalized so the submission is valid (row sums to 1), which should move KL down substantially vs. the current broken run/uniform fallback.'
- What this solution (achieved 1.40995) has done: 'The crash happens immediately when importing TensorFlow because this Kaggle Python 3.13 image has an incompatible protobuf runtime where `MessageFactory.GetPrototype` is missing; the pure‑Python fallback env vars you set are not sufficient on their own. I add a small, targeted workaround that monkey-patches `google.protobuf.message_factory.MessageFactory` to provide `GetPrototype`/`GetMessageClass` before importing TensorFlow, which unblocks the rest of your pipeline without changing your model/data logic. I also keep your existing robust submission writing (including row-wise normalization) intact so a valid `submission.csv` is always produced. With TensorFlow actually running, your ImageNet‑pretrained EfficientNet inference execute (instead of failing early), which should reduce KL substantially toward your target.'
- What this solution (achieved 1.40995) has done: 'You’re failing before any training/inference because TensorFlow import triggers a protobuf API mismatch: TF expects `MessageFactory.GetPrototype`, but the runtime protobuf in this Kaggle Py3.13 image doesn’t provide it. The current monkey-patch checks the *class* but the error is on the *instance*, so it never actually fixes what TF calls; I patch both the class and the instance method names safely before importing TensorFlow. This is score-improving because it unblocks the real ImageNet-pretrained EfficientNet inference instead of crashing/falling back to uniform predictions (which yields very poor KL). I keep the model/data pipeline intact and still retain the uniform-submission fallback for safety, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying a safe monkey-patch that adds `GetPrototype`/`GetMessageClass` to both the protobuf `MessageFactory` class and its instances via `__getattr__`, which matches how TensorFlow accesses it in this Kaggle Py3.13 image. I keep your model/data/inference logic intact, but make the patch run before importing TensorFlow so the rest of the pipeline can execute instead of erroring/falling back. This should materially improve the score (lower KL) because it unblocks real EfficientNet(ImageNet) inference instead of the uniform-probability fallback. I also keep the existing safety fallback and submission normalization to guarantee a valid `submission.csv` is always written.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by simplifying the protobuf workaround: instead of a brittle `__getattr__` hook that still raises for `GetPrototype`, we add safe aliases only when possible and never raise during the patch step. This unblocks the rest of the pipeline so the ImageNet-pretrained EfficientNetB0 inference can run (instead of failing early and/or falling back), which should materially reduce KL from the current 1.40995 toward your 0.3478 target. I also keep submission writing unchanged but add a final guard to always row-normalize and match the sample submission column order exactly. No changes are made to the model architecture, feature extraction, loss, or inference semantics beyond the crash fix and safety normalization.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime error that stops execution: TensorFlow import triggers a protobuf API mismatch where an *instance* of `MessageFactory` lacks `GetPrototype`. The minimal robust fix is to monkey‑patch `google.protobuf.message_factory.MessageFactory` so both the class and instances provide `GetPrototype`/`GetMessageClass` via safe aliases and a fallback `__getattr__`, before importing TensorFlow. This is score-improving (lower KL) because it allows the real EfficientNet(ImageNet) inference path to run instead of crashing (which otherwise forces the uniform fallback). I keep your model/data/inference logic unchanged and still preserve the uniform-submission fallback + row-normalization to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the current hard crash (a recursion error in the protobuf monkey-patch) by removing the fragile `__getattr__` hook and instead adding simple, non-recursive aliases for `GetPrototype`/`GetMessageClass` only when missing. This keeps the original intent (work around the TF/protobuf mismatch on Kaggle Py3.13) but prevents infinite recursion and allows TensorFlow to import so real EfficientNet(ImageNet) inference can run, which should substantially reduce KL from the current 1.40995 toward your 0.3478 target. I keep your model/data/inference core logic unchanged, and preserve the existing “uniform fallback” + probability renormalization to guarantee a valid `submission.csv` is always produced. No training behavior or architecture be changed.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os, warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_PARSER_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON_PARSER", "python")

try:
    from google.protobuf import message_factory as _message_factory

    MF = _message_factory.MessageFactory

    if hasattr(MF, "GetMessageClass") and not hasattr(MF, "GetPrototype"):
        try:
            MF.GetPrototype = MF.GetMessageClass
        except Exception:
            pass
    if hasattr(MF, "GetPrototype") and not hasattr(MF, "GetMessageClass"):
        try:
            MF.GetMessageClass = MF.GetPrototype
        except Exception:
            pass
except Exception:
    pass

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241108d"  # the path of trained model weights for testing


def _resolve_data_root(base="/kaggle/input/hms-harmful-brain-activity-classification"):
    cand1 = base
    cand2 = os.path.join(base, "hms-harmful-brain-activity-classification")
    if os.path.exists(os.path.join(cand2, "train.csv")):
        return cand2
    return cand1


if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = _resolve_data_root(
        "/kaggle/input/hms-harmful-brain-activity-classification"
    )

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate
EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_MULTIPLY = 5

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import io, time, gc
from PIL import Image
import pandas as pd, numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) == 0:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("Using CPU")
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print("Using 1 GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

MIX = False
print("Using full precision (mixed precision disabled)")

sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]
print("Targets", TARGETS)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
print("Train shape:", df.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
        train = pd.read_csv("train.csv")




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

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stfts = stfts
        self.specs = specs
        self.imgs = imgs
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.dataframe) / self.batch_size))
        return ct

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

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    len(indexes),
                    (4 * 4 + 2) * EEG_MULTIPLY,
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

        if self.mode != "test":
            TARGETS_RAW = [t + "_raw" for t in TARGETS]
            targets_batch = list()

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = row.sign_id
            if self.mode != "test":
                sample_weight = float(np.sum(row[TARGETS_RAW].values)) / 20.0
                targets_batch.append(row.expert_consensus)

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
                r_stft = 0
            else:
                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    * (df.seizure_vote == row.seizure_vote_raw)
                    * (df.lpd_vote == row.lpd_vote_raw)
                    * (df.gpd_vote == row.gpd_vote_raw)
                    * (df.lrda_vote == row.lrda_vote_raw)
                    * (df.grda_vote == row.grda_vote_raw),
                    :,
                ].reset_index(drop=True)
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

            if "spe" in DATATYPE:
                spe = list()
                for k in range(4):
                    spe.append(
                        np.reshape(
                            self.specs[row.spectrogram_id][
                                r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                            ].T,
                            (1, 100, 300),
                        )
                    )
                spe = np.concatenate(spe, axis=0)

            if "eeg" in DATATYPE:
                eeg = self.eegs[row.eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]

            if "stft" in DATATYPE:
                stft_t = self.stfts[-row.eeg_id]
                r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                    stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = self.imgs[sign_id]

            if "spe" in DATATYPE:
                spe[np.isnan(spe)] = 0
                spe = np.clip(spe, a_min=1e-6, a_max=1e6)
                spe = np.log2(spe)

                spe = spe[
                    :,
                    :,
                    round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                        (spe.shape[2] - SPE_WIDE) / 2
                    ),
                ]
                spe = spe[[0, 2, 3, 1], :, :]

                if self.mode == "train":
                    spe[0:2, :] = spe[0:2, :][np.random.permutation(2), :]
                    spe[2:4, :] = spe[2:4, :][np.random.permutation(2), :]
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
                            m_max = min(max(m1, m2), m_min + round(spe.shape[2] * 0.1))
                            spe[ii, :, m_min:m_max] = 0

                spe = (spe - np.mean(spe, keepdims=True)) / (
                    np.std(spe, keepdims=True) + 1e-6
                )
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]
                eeg_save = np.zeros((x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32)

                if self.mode == "train":
                    eeg[0:8, :] = eeg[0:8, :][np.random.permutation(8), :]
                    eeg[10:18, :] = eeg[10:18, :][np.random.permutation(8), :]
                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]
                else:
                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                    np.std(eeg_save, keepdims=True) + 1e-6
                )
                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                stft = np.clip(stft, a_min=1e-6, a_max=1e6)
                stft = np.log2(stft)

                if self.mode == "train":
                    stft[0:8, :, :] = stft[0:8, :, :][np.random.permutation(8), :, :]
                    stft[10:18, :, :] = stft[10:18, :, :][
                        np.random.permutation(8), :, :
                    ]
                    if np.random.rand() > 0.5:
                        stft = stft[::-1, :, :]

                stft_save = np.zeros(
                    (round(stft.shape[0] / 2 * stft.shape[1]), stft.shape[2] * 2),
                    dtype=np.float32,
                )
                for ii in range(stft.shape[0]):
                    stft_save[
                        ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                        (ii % 2) * stft.shape[2] : (ii % 2 + 1) * stft.shape[2],
                    ] = stft[ii, :, :]

                stft = (stft_save - np.mean(stft_save, keepdims=True)) / (
                    np.std(stft_save, keepdims=True) + 1e-6
                )
                x_stft[j] = stft

            if "img" in DATATYPE:
                img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                if self.mode == "train":
                    img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                    img[10:18, :, :] = img[10:18, :, :][np.random.permutation(8), :, :]
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
                            ii, temp_temp : round(temp_temp + end_temp - start_temp), :
                        ]
                    )
                img_save = np.clip(img_save, a_min=0, a_max=1)

                img = np.reshape(
                    img_save, (img_save.shape[0], img_save.shape[[1]][0], 1)
                )
                img = np.concatenate((img, img, img), -1)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                y_row = row[TARGETS].values.astype(np.float32)
                y[j] = y_row / np.sum(y_row)
                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1.0

        x = list()
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "stft" in DATATYPE:
            x.append(x_stft)
        if "img" in DATATYPE:
            x.append(x_img)

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




## === cell 4
from tensorflow.keras.applications import EfficientNetB0


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

        base_model_spe = EfficientNetB0(include_top=False, weights="imagenet")
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                (4 * 4 + 2) * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            )
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

        base_model_eeg = EfficientNetB0(include_top=False, weights="imagenet")
        base_model_eeg._name = "eeg_extractor"
        x_eeg = base_model_eeg(x_eeg)

        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

        inp.append(inp_eeg)
        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
        else:
            y = x_eeg

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
        x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
            inp_stft
        )
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        base_model_stft = EfficientNetB0(include_top=False, weights="imagenet")
        base_model_stft._name = "stft_extractor"

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

        inp.append(inp_stft)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
        else:
            y = x_stft

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))

        base_model_img = EfficientNetB0(include_top=False, weights="imagenet")
        base_model_img._name = "img_extractor"

        x_img = base_model_img(inp_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

        inp.append(inp_img)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("stft" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
        else:
            y = x_img

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 5
def _write_uniform_submission(test_df, out_path="submission.csv"):
    uniform = np.full(
        (len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
    )
    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    sub[TARGETS] = uniform
    sub.to_csv(out_path, index=False)
    return sub


if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    weight_paths = [
        os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        for model_i in range(SPLITS)
    ]
    missing = [p for p in weight_paths if not os.path.exists(p)]
    if len(missing) > 0:
        print(
            "WARNING: Missing model weight files, writing safe uniform submission instead."
        )
        sub = _write_uniform_submission(test, "submission.csv")
        print("Submission shape", sub.shape)
        print(sub.head())
    else:
        try:
            preds_all = []
            models = list()
            with strategy.scope():
                model_template = build_model()

            for model_i in range(SPLITS):
                print(f"Fold {model_i+1}")
                with strategy.scope():
                    model = clone_model(model_template)
                model.load_weights(weight_paths[model_i])
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

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            if ("eeg" in DATATYPE) or ("stft" in DATATYPE) or ("img" in DATATYPE):
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / RSFREQ, "bandpass"
                )
                b2, a2 = signal.butter(
                    3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
                )

                batch_start = 0

                for i, eeg_id in enumerate(test.eeg_id):
                    if i % 100 == 0:
                        print(i, ", ", end="")

                    eeg_default = pd.read_parquet(
                        os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
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
                            eeg2,
                            axis=1,
                            fs=RSFREQ,
                            nperseg=RSFREQ,
                            noverlap=60,
                            nfft=160,
                        )
                        ss[np.isnan(ss)] = 0
                        ss = ss[:, (ff > 0) * (ff <= 20), :]

                    if "img" in DATATYPE:
                        eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                        eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                        train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                        for j in range(len(train_plot)):
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

                            imgs_test[train_plot.sign_id[j]] = img_save

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                    if "eeg" in DATATYPE:
                        eegs_test[eeg_id] = eeg
                    if "stft" in DATATYPE:
                        stfts_test[eeg_id] = ss
                        stfts_test[-eeg_id] = tt

                    if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                        batch_end = i + 1
                        df_batch = test.iloc[batch_start:batch_end].reset_index(
                            drop=True
                        )

                        preds = []
                        test_gen = DataGenerator(
                            df_batch,
                            shuffle=False,
                            sample_weights=False,
                            batch_size=TEST_BATCHSIZE,
                            mode="test",
                            specs=spectrograms_test,
                            eegs=eegs_test,
                            stfts=stfts_test,
                            imgs=imgs_test,
                        )

                        for model_i in range(SPLITS):
                            pred = models[model_i].predict(test_gen, verbose=0)
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

            preds_all = np.asarray(preds_all, dtype=np.float32)

            if preds_all.shape[0] != len(test):
                raise RuntimeError(
                    f"Pred rows ({preds_all.shape[0]}) != test rows ({len(test)}). Check batching."
                )

            preds_all = np.clip(preds_all, 1e-9, 1.0)
            preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = preds_all
            sub = sub[["eeg_id"] + TARGETS]
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
        except Exception as e:
            print(
                "ERROR during inference, falling back to uniform submission. Error:",
                repr(e),
            )
            sub = _write_uniform_submission(test, "submission.csv")
            print("Submission shape", sub.shape)
            print(sub.head())
