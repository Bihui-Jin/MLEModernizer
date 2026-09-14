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

0.3322175357581101

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The timeout is dominated by per-file parquet reads and repeated Pandas/DataFrame overhead inside the test loop, plus building a huge `preds_all` via repeated `np.concatenate`. I keep the exact preprocessing + model forward-pass logic, but speed it up by (1) precomputing parquet file paths and using a thread pool to overlap I/O with CPU filtering, (2) replacing repeated concatenations with preallocated output storage, and (3) enabling `tf.function`-compiled inference (no change to architecture or weights) while keeping determinism seeds. I also fix the “Invalid submission length” error by ensuring we always output exactly `len(sample_submission)` rows in the same order. All paths and model logic remain unchanged; only inference plumbing and data loading are optimized.'
- What this solution (achieved 1.40995) has done: 'I fix the startup crash caused by an incompatible protobuf implementation by forcing TensorFlow to use the pure-Python protobuf backend before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` error). I also fix a major inference bug where all `tf.function` inference wrappers accidentally used the last fold’s model due to Python closure capture, which severely hurts score; each fold now have its own correctly bound inference function. Finally, I keep the rest of the preprocessing/model logic unchanged and ensure the submission is generated in exactly `sample_submission.csv` order with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *before any TensorFlow-related import occurs*, and I add a safe fallback to disable C++ protobuf if the attribute error still appears. I also ensure the script always writes a valid `submission.csv` with exactly the same `eeg_id` order/length as `sample_submission.csv`, and that each row sums to 1 (with clipping + renormalization). To move the KL score down toward the target (lower is better) without changing the core model, I add a minimal test-time probability smoothing step that blends the model predictions with the global training prior (a common KL-calibration trick) while keeping the same model/weights and inference semantics. All paths, model architecture, and preprocessing remain unchanged; only crash-proofing and a tiny post-processing calibration are added.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf startup crash by forcing the pure-Python protobuf implementation *before* any `google.protobuf`/`tensorflow` import and by adding a safe fallback that fully purges protobuf/tensorflow modules if the first import fails. Then I keep your existing model/data logic intact, but add a small guard so the GPU strategy doesn’t crash on CPU-only environments (common on Kaggle notebooks). Finally, I keep the existing submission formatting/renormalization and the light prior-blend calibration (alpha) to move the KL score down toward the target while ensuring the output file is always a valid `submission.csv` with the exact sample order and row-sum=1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf startup crash by forcing the pure-Python protobuf implementation earlier and adding a robust import fallback that also forces the C++ implementation off (this directly addresses the `MessageFactory.GetPrototype` error). Then I keep your exact model + preprocessing logic intact, but ensure the optimized inference path always initializes/returns the expected input arrays for the active `DATATYPE` (avoiding undefined `x_stft/x_img` if those modes are ever enabled). Finally, I keep the existing prior-blend calibration (alpha) and enforce strict submission alignment to `sample_submission.csv` order with per-row normalization so the CSV is always valid and the KL score should move down from the current 1.40995 toward the target.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by strengthening the “safe import” routine: it force the pure-Python protobuf backend, fully purge conflicting `google.protobuf` modules, and retry the TF import in a more robust way so the notebook can start reliably on Kaggle. Then I keep your existing model architecture, preprocessing, and inference logic intact, but I add a small, score-improving KL-calibration tweak by tuning the prior-blend `alpha` down from 0.25 to a safer 0.10 (still minimal post-processing, but less over-smoothing which can cause high KL). Finally, I ensure the submission is always written as `submission.csv` with exactly the same row order/length as `sample_submission.csv` and with per-row probabilities clipped and normalized to sum to 1.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf import crash that prevents the notebook from running by setting the correct protobuf environment variables early and, if needed, patching the missing `MessageFactory.GetPrototype` attribute before importing TensorFlow. This is a pure runtime-stability fix that does not change your model, preprocessing, or training/inference logic. After that, we keep your existing inference pipeline and submission formatting intact so it still writes a valid `submission.csv` in the `sample_submission.csv` order with per-row probabilities summing to 1. No score-tuning changes are introduced beyond ensuring the code executes end-to-end.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241113a"  # the path of trained model weights for testing

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

EEG_MULTIPLY = 25

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import io
from PIL import Image
import pandas as pd, numpy as np


def _import_tensorflow_safely():
    """
    Fix: robust TF import under Kaggle python 3.13 where protobuf can raise:
      AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

    We (1) force pure-Python protobuf via env vars (already set above),
    (2) pre-patch google.protobuf.message_factory.MessageFactory to include GetPrototype
        as an alias to GetMessageClass when needed,
    (3) purge tensorflow/protobuf modules and retry.
    """
    import sys
    import importlib

    def _purge(prefixes):
        for k in list(sys.modules.keys()):
            if any(k == p or k.startswith(p + ".") for p in prefixes):
                sys.modules.pop(k, None)

    def _patch_protobuf_message_factory():
        try:
            from google.protobuf import message_factory as _mf  # type: ignore

            if hasattr(_mf, "MessageFactory"):
                MF = _mf.MessageFactory
                if not hasattr(MF, "GetPrototype") and hasattr(MF, "GetMessageClass"):
                    setattr(MF, "GetPrototype", MF.GetMessageClass)
        except Exception:
            pass

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    os.environ["PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION"] = "1"

    _patch_protobuf_message_factory()

    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception:
        _purge(["tensorflow", "google.protobuf"])
        importlib.invalidate_caches()
        _patch_protobuf_message_factory()
        import tensorflow as tf  # noqa: F401

        return tf


tf = _import_tensorflow_safely()

from tensorflow.keras.models import clone_model

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
import gc
from concurrent.futures import ThreadPoolExecutor

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
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Using full precision (mixed precision disabled for stability)")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train_votes = df[list(TARGETS)].to_numpy(dtype=np.float64)
train_prior = train_votes.sum(axis=0)
train_prior = train_prior / np.clip(train_prior.sum(), 1e-12, None)
train_prior = train_prior.astype(np.float32)
print("Train prior:", dict(zip(TARGETS, train_prior.tolist())))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TARGETS_RAW = [c + "_raw" for c in TARGETS]




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

        targets_batch = list()

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = int(row["sign_id"]) if "sign_id" in row.index else None

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
                r_stft = 0
            else:
                sample_weight = sum(row[TARGETS_RAW].values) / 20
                targets_batch.append(row.expert_consensus)

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
                spe = list()  # LL RL LP RP
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

                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

                if self.mode == "train":
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

                img = np.reshape(img_save, (img_save.shape[0], img_save.shape[1], 1))
                img = np.concatenate((img, img, img), -1)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1

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

        base_model_spe = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=None
        )
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            )
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=None
        )
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

        base_model_stft = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=None
        )
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

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=None
        )
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




## === cell 4
def _resolve_models_dir(requested_dir: str) -> str:
    if os.path.isdir(requested_dir):
        return requested_dir
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return requested_dir

    name = os.path.basename(requested_dir.rstrip("/"))
    cand = os.path.join(base, name)
    if os.path.isdir(cand):
        return cand

    try:
        for d in os.listdir(base):
            dpath = os.path.join(base, d)
            if not os.path.isdir(dpath):
                continue
            for root, _, files in os.walk(dpath):
                if any(
                    fn.startswith("fold0") and fn.endswith("_stage2.h5") for fn in files
                ):
                    return root
    except Exception:
        pass
    return requested_dir


LOAD_MODELS_FROM = _resolve_models_dir(LOAD_MODELS_FROM)
print("Resolved LOAD_MODELS_FROM:", LOAD_MODELS_FROM)



## === cell 5
if not NEEDTRAIN:
    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))

    test = test.set_index("eeg_id").reindex(sample_sub["eeg_id"].values).reset_index()
    test["sign_id"] = np.arange(len(test), dtype=np.int64)
    print("Test shape", test.shape, "Sample_sub shape", sample_sub.shape)

    weights_ok = True
    expected = [
        os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        for model_i in range(SPLITS)
    ]
    missing = [p for p in expected if not os.path.isfile(p)]
    if len(missing) > 0:
        weights_ok = False
        print(
            "WARNING: Missing model weights; will write uniform-probability submission."
        )
        print("Missing examples:", missing[:3])

    if weights_ok:
        model_template = build_model()
        models = []
        infer_fns = []

        def _make_infer_fn(m):
            @tf.function(reduce_retracing=True)
            def _infer(x):
                return m(x, training=False)

            return _infer

        for model_i in range(SPLITS):
            print(f"Fold {model_i + 1}")
            model = clone_model(model_template)
            model.load_weights(
                os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
            )
            models.append(model)
            infer_fns.append(_make_infer_fn(model))

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

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs")

        if ("eeg" in DATATYPE) or ("stft" in DATATYPE) or ("img" in DATATYPE):
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            brain_pairs = [ch.split("-") for ch in BRAIN]
            brain_a = [p[0] for p in brain_pairs]
            brain_b = [p[1] for p in brain_pairs]

            eeg_ids = test["eeg_id"].to_numpy()
            eeg_paths = [
                os.path.join(PATH_test, f"{int(eid)}.parquet") for eid in eeg_ids
            ]

            n_test = len(test)
            preds_all = np.empty((n_test, len(TARGETS)), dtype=np.float32)

            max_workers = min(8, (os.cpu_count() or 2))
            executor = ThreadPoolExecutor(max_workers=max_workers)

            def _load_parquet(path):
                return pd.read_parquet(path)

            for start in range(0, n_test, TEST_BATCHSIZE):
                end = min(start + TEST_BATCHSIZE, n_test)
                bs = end - start

                if "eeg" in DATATYPE:
                    x_eeg = np.empty(
                        (
                            bs,
                            EEG_CHANNEL_USED * EEG_MULTIPLY,
                            round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                        ),
                        dtype=np.float32,
                    )
                if "stft" in DATATYPE:
                    x_stft = np.empty(
                        (bs, STFT_HIGH * 9, STFT_WIDE * 2), dtype=np.float32
                    )
                if "img" in DATATYPE:
                    x_img = np.empty((bs, IMG_HIGH, IMG_WIDE, 3), dtype=np.float32)

                futs = [
                    executor.submit(_load_parquet, eeg_paths[idx])
                    for idx in range(start, end)
                ]
                eeg_dfs = [f.result() for f in futs]

                for bi, eeg_default in enumerate(eeg_dfs):
                    gi = start + bi
                    if gi % 200 == 0:
                        print(gi, ", ", end="")

                    A = eeg_default[brain_a].to_numpy(dtype=np.float32, copy=False).T
                    B = eeg_default[brain_b].to_numpy(dtype=np.float32, copy=False).T
                    eeg = A - B
                    np.nan_to_num(eeg, copy=False, nan=0.0)

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
                        np.nan_to_num(ss, copy=False, nan=0.0)
                        ss = ss[:, (ff > 0) * (ff <= 20), :]

                    if "img" in DATATYPE:
                        eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                        eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                        train_plot = test[test.eeg_id == eeg_ids[gi]].reset_index(
                            drop=True
                        )
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

                            imgs_test[int(train_plot.sign_id[j])] = img_save

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                    if "eeg" in DATATYPE:
                        eeg = eeg[:, 0 : round((0 + 50) * RSFREQ)]
                        eeg = eeg[
                            :,
                            round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                                (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                            ),
                        ]
                        eeg = np.concatenate(
                            (
                                eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                                eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                            ),
                            axis=0,
                        )
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                        eeg_save = np.empty(
                            (
                                EEG_CHANNEL_USED * EEG_MULTIPLY,
                                eeg.shape[1] // EEG_MULTIPLY,
                            ),
                            dtype=np.float32,
                        )
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                        mu = eeg_save.mean(keepdims=True)
                        sd = eeg_save.std(keepdims=True) + 1e-6
                        x_eeg[bi] = (eeg_save - mu) / sd

                x_list = []
                if "spe" in DATATYPE:
                    raise RuntimeError(
                        "This optimized inference path currently expects DATATYPE=['eeg'] only."
                    )
                if "eeg" in DATATYPE:
                    x_list.append(x_eeg)
                if "stft" in DATATYPE:
                    x_list.append(x_stft)
                if "img" in DATATYPE:
                    x_list.append(x_img)

                fold_sum = np.zeros((bs, len(TARGETS)), dtype=np.float32)
                for model_i in range(SPLITS):
                    pred = (
                        infer_fns[model_i](x_list)
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    fold_sum += pred
                pred_mean = fold_sum / float(SPLITS)

                preds_all[start:end] = pred_mean

                del x_list, eeg_dfs, futs, fold_sum, pred_mean
                if "eeg" in DATATYPE:
                    del x_eeg
                if "stft" in DATATYPE:
                    del x_stft
                if "img" in DATATYPE:
                    del x_img
                gc.collect()
            print()

            executor.shutdown(wait=True)

        preds_all = np.asarray(preds_all, dtype=np.float32)
        preds_all = np.clip(preds_all, 1e-6, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        alpha = 0.10  # 0 -> original preds, 1 -> pure prior
        preds_all = (1.0 - alpha) * preds_all + alpha * train_prior[None, :]
        preds_all = np.clip(preds_all, 1e-6, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
    else:
        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = 1.0 / len(TARGETS)

    sub = sub.set_index("eeg_id").reindex(sample_sub["eeg_id"].values).reset_index()
    sub[TARGETS] = sub[TARGETS].fillna(1.0 / len(TARGETS))

    arr = sub[TARGETS].to_numpy(dtype=np.float32)
    arr = np.clip(arr, 1e-6, 1.0)
    arr = arr / np.sum(arr, axis=1, keepdims=True)
    sub[TARGETS] = arr

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print("Saved submission.csv")
