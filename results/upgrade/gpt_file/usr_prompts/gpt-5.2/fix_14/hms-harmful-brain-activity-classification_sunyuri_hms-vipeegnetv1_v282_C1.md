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

0.4376204139770831

# 6. Current score

1.19172

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` incompatibility seen in some Kaggle Python 3.13 images. I also fix the submission length mismatch by ensuring predictions are aligned to `sample_submission.csv` order (not via a merge that can introduce NaNs/duplicates) and by generating predictions for every row in the provided test file exactly once. Finally, I keep your model/data logic unchanged, but make the inference path robust: if model weights aren’t present, it still produce a valid, correctly-sized submission using the train label prior (score be worse than the target but at least yield a valid submission).'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is forced *before any TensorFlow-related import occurs*, and by importing `google.protobuf` early so the environment variable takes effect in the current process. Then I make the inference path actually use the provided fold weights when available by compiling/loading models under the selected distribution strategy (to avoid occasional device/variable placement issues) and keeping the submission alignment exactly as `sample_submission.csv`. These changes are execution/stability fixes that should also improve the score substantially versus the current “label prior fallback” behavior (1.39779) by enabling real model inference. I keep the model architecture, preprocessing, and training/inference semantics unchanged.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash that currently prevents the notebook from running by switching to the compatible TensorFlow build for the Kaggle Python 3.13 image (Keras 3 + TF) and ensuring protobuf is configured before any TF import. Then I keep your existing model/inference logic intact, but make the runtime more robust by forcing CPU if no GPU exists and by guarding mixed precision flags that can error on newer TF builds. Finally, I ensure the script always produces a valid `submission.csv` with probabilities summing to 1 and aligned exactly to `sample_submission.csv`, so the score can improve from the prior-fallback baseline by actually loading and running the fold weights when available.'
- What this solution (achieved 1.39779) has done: 'I fix the protobuf/TensorFlow crash by ensuring the pure-Python protobuf runtime is selected *before* any TensorFlow import and by forcing protobuf to use the pure-Python backend at runtime when possible. Then I keep your model and preprocessing logic unchanged, but make inference reliably use the provided fold weights by loading under the distribution strategy and avoiding failures caused by missing/incorrect weight paths. Finally, I keep the submission alignment to `sample_submission.csv`, enforce valid probability normalization, and always write a `submission.csv` with the required columns and row count to improve the score from the current label-prior-like behavior toward your target.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring protobuf uses the pure-Python implementation *before any TensorFlow import*, and by importing protobuf first in a controlled way. Then I keep your model/inference logic the same but make the test-time spectrogram loading safe for Kaggle’s 600s limit by only loading the spectrogram parquet files actually referenced by `test.csv` (instead of all files), which is a correctness/performance fix that also enables real model inference (improving score vs the current prior-like fallback). Finally, I keep the submission alignment exactly to `sample_submission.csv`, enforce probability normalization, and always write a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by defensively importing TensorFlow only after forcing the pure-Python protobuf runtime and, if that still fails, automatically falling back to a score-safe non-TF inference path (so the script always runs end-to-end). Then, to move the score down toward your target (lower is better) without changing your core model/training logic, I ensure we actually use the provided fold weights when they exist and avoid the current “label prior fallback” unless TF truly cannot start. Finally, I keep the submission alignment exactly to `sample_submission.csv` order and enforce strict probability normalization so the submission is always valid.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution by forcing the pure-Python protobuf runtime earlier and more strictly, and by guarding the TensorFlow import with a safe fallback path if the environment still can’t initialize TF. Then I make sure inference uses real fold weights when they exist (instead of silently falling back to label priors), because your current score (1.39779, lower-is-better) indicates you’re likely not actually running the trained model. Finally, I keep the model/data logic intact but harden the submission alignment and probability normalization so the output is always a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype` missing) by forcing the pure-Python protobuf backend *and* disabling the C++ protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a safe import ordering before any TensorFlow import. Then I make the TensorFlow import robust: if it still fails, the script fall back to label-prior predictions (valid but higher KL), otherwise it proceed to load fold weights and run real inference (which should lower your score toward the target). Finally, I keep your model/data logic intact while ensuring the submission is always aligned to `sample_submission.csv`, has correct columns, and each row sums to 1.'
- What this solution (achieved 1.19172) has done: 'The timeout is dominated by per-row DataGenerator work (expensive `df.loc[...]` filtering inside every `__getitem__`) and heavy Python overhead from repeatedly building small arrays. I keep the same model, weights, preprocessing math, and prediction loop, but precompute an O(1) lookup table that maps each consolidated training row to its chosen “valid center” row and a list of candidate rows for “train random pick”, eliminating repeated dataframe scans. I also precompute channel index pairs and speed up EEG differencing with vectorized NumPy, and switch test-time prediction to `predict_on_batch` with direct NumPy batch construction (same tensors, less Keras Sequence overhead). These changes are provably equivalent in semantics (same row selection logic and same feature transforms), just much faster.'

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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

try:
    import google.protobuf.internal.api_implementation as _pb_api_impl  # type: ignore

    try:
        _pb_api_impl._SetType("python")  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    pass

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["spe"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241117c"  # the path of trained model weights for testing

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

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

TF_AVAILABLE = True
TF_IMPORT_ERROR = None

try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    tf = None
    optimizers = None
    clone_model = None
    print("WARNING: TensorFlow import failed; will use non-TF fallback if needed.")
    print("TF import error:", repr(e))

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

if TF_AVAILABLE:
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
if TF_AVAILABLE:
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

MIX = True
if TF_AVAILABLE and MIX:
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (mixed_float16 policy)")
    except Exception:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled (experimental option)")
        except Exception:
            print("Mixed precision not available; continuing")
elif TF_AVAILABLE:
    print("Using full precision")

length = round(32 / (EEG_MULTIPLY / 10))
x = np.linspace(1, length, length)
y = x / length * 2
WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
WEIGHTS_f = None
if TF_AVAILABLE:
    WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

BRAIN_PAIR = [(c.split("-")[0], c.split("-")[1]) for c in BRAIN]


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = []
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

            eeg_cols = eeg_default.columns
            eeg = []
            for a_ch, b_ch in BRAIN_PAIR:
                eeg_temp = (
                    eeg_default[a_ch].to_numpy() - eeg_default[b_ch].to_numpy()
                ).astype(np.float32, copy=False)
                np.nan_to_num(eeg_temp, copy=False, nan=0.0)
                eeg.append(eeg_temp[None, :])
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

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()


## === cell 3
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    time_start_time = time.time()
    if READ_SPE_FILES:
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
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()


## === cell 4
if TF_AVAILABLE:
    _HAS_TRAIN_RAW = "TARGETS_RAW" in globals()
    if _HAS_TRAIN_RAW:
        _grp_cols = [
            "eeg_id",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
        ]
        _df_groups = df[
            _grp_cols
            + [
                "eeg_sub_id",
                "eeg_label_offset_seconds",
                "spectrogram_label_offset_seconds",
                "spectrogram_id",
            ]
        ].copy()
        _df_groups["_row_idx"] = np.arange(len(df), dtype=np.int32)

        _groups = {}
        for key, g in _df_groups.groupby(_grp_cols, sort=False):
            g_sorted = g.sort_values("eeg_sub_id", kind="mergesort")
            idxs = g_sorted["_row_idx"].to_numpy(dtype=np.int32, copy=False)
            mid_idx = int(idxs[len(idxs) // 2])
            _groups[key] = (idxs, mid_idx)

        def _make_group_key_from_train_row(row) -> tuple:
            return (
                int(row.eeg_id),
                float(row.seizure_vote_raw),
                float(row.lpd_vote_raw),
                float(row.gpd_vote_raw),
                float(row.lrda_vote_raw),
                float(row.grda_vote_raw),
            )

        def _get_group_info_for_row(row):
            return _groups[_make_group_key_from_train_row(row)]

    else:
        _groups = None

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
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
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

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                sign_id = row.sign_id
                if self.mode != "test":
                    sample_weight = 1.0
                    if _HAS_TRAIN_RAW:
                        sample_weight = float(np.sum(row[TARGETS_RAW].values) / 20)

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                    r_stft = 0
                    row_use = row
                else:
                    if _HAS_TRAIN_RAW:
                        idxs, mid_idx = _get_group_info_for_row(row)
                        if self.mode == "train":
                            pick = int(idxs[np.random.randint(0, len(idxs))])
                        else:  # "valid"
                            pick = int(mid_idx)
                        row_use = df.iloc[pick]
                    else:
                        row_use = row

                    r_spe = round(float(row_use.spectrogram_label_offset_seconds) / 2)
                    r_eeg = float(row_use.eeg_label_offset_seconds)

                if "spe" in DATATYPE:
                    spe = []  # LL RL LP RP
                    spec_arr = self.specs[int(row_use.spectrogram_id)]
                    rs = int(r_spe)
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                spec_arr[rs : (rs + 300), k * 100 : (k + 1) * 100].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[int(row_use.eeg_id)][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-int(row_use.eeg_id)]
                    r_stft = (np.where(stft_t >= (r_eeg - float(np.min(stft_t)))))[0][0]
                    stft = self.stfts[int(row_use.eeg_id)][
                        :, :, r_stft : (r_stft + STFT_WIDE)
                    ]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                        stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    spe = np.clip(spe, a_min=np.exp(-4), a_max=np.exp(6))
                    spe = np.log(spe)
                    spe = spe[
                        :,
                        :,
                        round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]

                    if self.mode == "train":
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

                    spe = (spe - np.mean(spe, keepdims=True)) / (
                        np.std(spe, keepdims=True) + 1e-6
                    )
                    x_spe[j] = spe

                if self.mode != "test":
                    yj = row[TARGETS].values.astype("float32")
                    yj = yj / max(float(np.sum(yj)), 1e-12)
                    y[j] = yj

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
if TF_AVAILABLE:

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
                        (step - self.warm_step)
                        / (self.total_step - self.warm_step)
                        * np.pi
                    )
                )
            return lr




## === cell 6
if TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB0

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

            base_model_spe = EfficientNetB0(include_top=False, weights=None)
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

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

            base_model_eeg = EfficientNetB0(include_top=False, weights=None)
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)

            x_eeg = tf.multiply(x_eeg, WEIGHTS_f)
            x_eeg = tf.reduce_sum(x_eeg, 2, keepdims=True)

            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

            base_model_stft = EfficientNetB0(include_top=False, weights=None)
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                if y is not None
                else x_stft
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
            base_model_img = EfficientNetB0(include_top=False, weights=None)
            base_model_img._name = "img_extractor"
            x_img = base_model_img(inp_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 7
if NEEDTRAIN:
    if not TF_AVAILABLE:
        raise RuntimeError(
            f"NEEDTRAIN=True but TensorFlow is not available: {repr(TF_IMPORT_ERROR)}"
        )

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
        print(f"### Fold {i+1}")

        df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
        df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

        df_train_stage2 = df_train_stage1[
            np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 6
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 6
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
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, 1e-5, 5)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{1}.h5"),
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

        model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        loss_hist = history.history["loss"]
        val_loss_hist = history.history["val_loss"]
        epochs_range = range(1, len(loss_hist) + 1)
        plt.plot(epochs_range, loss_hist, "bo", label="loss")
        plt.plot(epochs_range, val_loss_hist, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage1.pdf"))
        plt.close()

        valid_stage1 = df_valid_stage1[TARGETS].values
        predict_stage1 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage1, 1), np.argmax(predict_stage1, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        plt.yticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
        plt.savefig(os.path.join("models", f"fold{i}_stage1_cm.pdf"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        gc.collect()

        train_gen = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(round(EPOCHS / 3), LEARN_RATE * 0.1, 1e-5, 0)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{2}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=round(EPOCHS / 3),
            callbacks=callbacks_list,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage2.h5"))

        loss_hist = history.history["loss"]
        val_loss_hist = history.history["val_loss"]
        epochs_range = range(1, len(loss_hist) + 1)
        plt.plot(epochs_range, loss_hist, "bo", label="loss")
        plt.plot(epochs_range, val_loss_hist, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage2.pdf"))
        plt.close()

        valid_stage2 = df_valid_stage2[TARGETS].values
        predict_stage2 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage2, 1), np.argmax(predict_stage2, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        plt.yticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
        plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.pdf"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        gc.collect()




## === cell 8
def _resolve_weights_dir(base_dir: str) -> str:
    if os.path.isdir(base_dir):
        candidates = [base_dir]
        for root, dirs, files in os.walk(base_dir):
            if any(fn.startswith("fold0_") and fn.endswith(".h5") for fn in files):
                candidates.append(root)
                break
        for c in candidates:
            if os.path.isdir(c):
                fs = os.listdir(c)
                if any(fn.startswith("fold0_") and fn.endswith(".h5") for fn in fs):
                    return c
        return base_dir
    return base_dir


def _pick_weight_file(weights_dir: str, fold_idx: int) -> str:
    f2 = os.path.join(weights_dir, f"fold{fold_idx}_stage2.h5")
    f1 = os.path.join(weights_dir, f"fold{fold_idx}_stage1.h5")
    if os.path.exists(f2):
        return f2
    if os.path.exists(f1):
        return f1
    if os.path.isdir(weights_dir):
        for fn in os.listdir(weights_dir):
            if fn.startswith(f"fold{fold_idx}_") and fn.endswith(".h5"):
                return os.path.join(weights_dir, fn)
    return f2


def _make_label_prior_predictions(
    train_df: pd.DataFrame, test_df: pd.DataFrame, targets
) -> np.ndarray:
    y = train_df[list(targets)].values.astype(np.float64)
    y = y / np.clip(y.sum(axis=1, keepdims=True), 1e-12, None)
    prior = y.mean(axis=0)
    prior = np.clip(prior, 1e-8, 1.0)
    prior = prior / prior.sum()
    preds = np.tile(prior.reshape(1, -1), (len(test_df), 1)).astype(np.float32)
    return preds


def _extract_spectrogram_features_from_array(arr_600x400: np.ndarray) -> np.ndarray:
    x = np.asarray(arr_600x400, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, a_min=np.exp(-4), a_max=np.exp(6))
    x = np.log(x + 1e-12)

    feats = []
    for k in range(4):
        region = x[:, k * 100 : (k + 1) * 100]  # (time=600, freq=100)
        m = float(region.mean())
        s = float(region.std())
        q25 = float(np.quantile(region, 0.25))
        q50 = float(np.quantile(region, 0.50))
        q75 = float(np.quantile(region, 0.75))

        low = region[:, :50]
        high = region[:, 50:]
        m_low = float(low.mean())
        m_high = float(high.mean())
        e = float((region * region).mean())
        feats.extend([m, s, q25, q50, q75, m_low, m_high, e])

    feats.extend([float(x.mean()), float(x.std()), float((x * x).mean())])
    return np.asarray(feats, dtype=np.float32)


def _train_fallback_sklearn_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, targets
) -> np.ndarray:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    y_prob = train_df[list(targets)].values.astype(np.float32)
    y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)
    y_cls = np.argmax(y_prob, axis=1).astype(np.int32)

    PATH_train_spe = os.path.join(LOAD_DATA_FROM, "train_spectrograms")
    PATH_test_spe = os.path.join(LOAD_DATA_FROM, "test_spectrograms")

    spec_cache = {}

    def load_spec(sid: int, is_test: bool) -> np.ndarray:
        if sid in spec_cache:
            return spec_cache[sid]
        base = PATH_test_spe if is_test else PATH_train_spe
        fpath = os.path.join(base, f"{int(sid)}.parquet")
        tmp = pd.read_parquet(fpath)
        arr = tmp.iloc[:, 1:].values  # (600, 400)
        spec_cache[sid] = arr
        return arr

    unique_train_spec = train_df["spectrogram_id"].astype(int).unique()
    train_feat_by_sid = {}
    t0 = time.time()
    for i, sid in enumerate(unique_train_spec):
        if (i % 500) == 0:
            gc.collect()
        arr = load_spec(int(sid), is_test=False)
        train_feat_by_sid[int(sid)] = _extract_spectrogram_features_from_array(arr)
    print(
        "Fallback: extracted train spectrogram features:",
        len(train_feat_by_sid),
        "in",
        round(time.time() - t0, 2),
        "sec",
    )

    X_train = np.vstack(
        [
            train_feat_by_sid[int(sid)]
            for sid in train_df["spectrogram_id"].astype(int).values
        ]
    ).astype(np.float32)

    gkf = GroupKFold(n_splits=5)
    groups = train_df["patient_id"].values

    oof = np.zeros((len(train_df), len(targets)), dtype=np.float32)
    test_pred_accum = np.zeros((len(test_df), len(targets)), dtype=np.float32)

    unique_test_spec = test_df["spectrogram_id"].astype(int).unique()
    test_feat_by_sid = {}
    t0 = time.time()
    for i, sid in enumerate(unique_test_spec):
        if (i % 500) == 0:
            gc.collect()
        arr = load_spec(int(sid), is_test=True)
        test_feat_by_sid[int(sid)] = _extract_spectrogram_features_from_array(arr)
    print(
        "Fallback: extracted test spectrogram features:",
        len(test_feat_by_sid),
        "in",
        round(time.time() - t0, 2),
        "sec",
    )

    X_test = np.vstack(
        [
            test_feat_by_sid[int(sid)]
            for sid in test_df["spectrogram_id"].astype(int).values
        ]
    ).astype(np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_cls, groups)):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegression(
                        multi_class="multinomial",
                        solver="lbfgs",
                        C=1.0,
                        max_iter=200,
                        n_jobs=1,
                        random_state=SEED + fold,
                    ),
                ),
            ]
        )
        clf.fit(X_train[tr_idx], y_cls[tr_idx])
        oof[va_idx] = clf.predict_proba(X_train[va_idx]).astype(np.float32)
        test_pred_accum += clf.predict_proba(X_test).astype(np.float32)

    test_pred = test_pred_accum / 5.0

    prior = y_prob.mean(axis=0).astype(np.float32)
    prior = np.clip(prior, 1e-8, 1.0)
    prior = prior / prior.sum()
    alpha = 0.10
    test_pred = (1 - alpha) * test_pred + alpha * prior.reshape(1, -1)

    test_pred = np.clip(test_pred, 1e-8, 1.0)
    test_pred = test_pred / np.sum(test_pred, axis=1, keepdims=True)
    return test_pred.astype(np.float32)


def _build_test_batch_inputs(batch_df: pd.DataFrame, spectrograms_test: dict) -> list:
    x = []
    if "spe" in DATATYPE:
        bs = len(batch_df)
        x_spe = np.zeros((bs, 4, SPE_HIGH, SPE_WIDE), dtype=np.float32)
        for j, sid in enumerate(batch_df["spectrogram_id"].astype(int).to_numpy()):
            spec_arr = spectrograms_test[int(sid)]
            rs = 0
            spe = np.empty((4, 100, 300), dtype=np.float32)
            for k in range(4):
                spe[k] = spec_arr[rs : (rs + 300), k * 100 : (k + 1) * 100].T
            np.nan_to_num(spe, copy=False, nan=0.0)
            spe = np.clip(spe, a_min=np.exp(-4), a_max=np.exp(6))
            spe = np.log(spe)
            left = round((spe.shape[2] - SPE_WIDE) / 2)
            right = -left if left != 0 else None
            spe = spe[:, :, left:right]
            spe = (spe - spe.mean(keepdims=True)) / (spe.std(keepdims=True) + 1e-6)
            x_spe[j] = spe
        x.append(x_spe)
    if "eeg" in DATATYPE:
        raise NotImplementedError(
            "Direct test batch builder is only used for spe-only mode in this optimized path."
        )
    if "stft" in DATATYPE:
        raise NotImplementedError(
            "Direct test batch builder is only used for spe-only mode in this optimized path."
        )
    if "img" in DATATYPE:
        raise NotImplementedError(
            "Direct test batch builder is only used for spe-only mode in this optimized path."
        )
    return x


if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    weights_dir = _resolve_weights_dir(LOAD_MODELS_FROM)
    print("Resolved weights_dir:", weights_dir)

    has_weights = os.path.isdir(weights_dir) and any(
        (fn.startswith("fold0_") and fn.endswith(".h5"))
        for fn in os.listdir(weights_dir)
    )

    if (not TF_AVAILABLE) or (not has_weights):
        if not TF_AVAILABLE:
            print("WARNING: TensorFlow unavailable; using non-TF fallback model.")
            print("TF import error was:", repr(TF_IMPORT_ERROR))
        else:
            print(
                f"WARNING: weights_dir not found or contains no fold weights: {weights_dir}"
            )
            print("Using non-TF fallback model.")
        try:
            preds_all = _train_fallback_sklearn_and_predict(df, test, TARGETS)
        except Exception as e:
            print(
                "WARNING: sklearn fallback failed; reverting to label prior. Error:",
                repr(e),
            )
            preds_all = _make_label_prior_predictions(df, test, TARGETS)
    else:
        preds_all = []
        models = []

        with strategy.scope():
            model_template = build_model()

        for model_i in range(SPLITS):
            print(f"Fold {model_i+1}")
            with strategy.scope():
                model = clone_model(model_template)
                model.compile(
                    optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE),
                    loss=tf.keras.losses.KLDivergence(),
                )
                wfile = _pick_weight_file(weights_dir, model_i)
                if not os.path.exists(wfile):
                    raise FileNotFoundError(
                        f"Missing weights for fold {model_i}: expected {wfile}. "
                        f"Available files: {sorted(os.listdir(weights_dir))[:50]}"
                    )
                model.load_weights(wfile)
            models.append(model)

        if "spe" in DATATYPE:
            PATH_test_spe = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            needed_spec_ids = set(test["spectrogram_id"].astype(int).unique().tolist())
            print(f"Need {len(needed_spec_ids)} unique test spectrogram_ids")

            time_start = time.time()
            for k, sid in enumerate(sorted(needed_spec_ids)):
                if (k % 200) == 0:
                    print(k, ", ", end="")
                fpath = os.path.join(PATH_test_spe, f"{sid}.parquet")
                tmp = pd.read_parquet(fpath)
                spectrograms_test[int(sid)] = tmp.iloc[:, 1:].values
            print()
            print(
                "Loaded needed test spectrograms in",
                round(time.time() - time_start, 2),
                "sec",
            )

        if ("eeg" in DATATYPE) or ("stft" in DATATYPE) or ("img" in DATATYPE):
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")

                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for a_ch, b_ch in BRAIN_PAIR:
                    eeg_temp = (
                        eeg_default[a_ch].to_numpy() - eeg_default[b_ch].to_numpy()
                    ).astype(np.float32, copy=False)
                    np.nan_to_num(eeg_temp, copy=False, nan=0.0)
                    eeg.append(eeg_temp[None, :])
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
                    start_idx = i - ((i % TEST_BATCHSIZE))  # start of current chunk
                    end_idx = i  # inclusive
                    batch_df = test.iloc[start_idx : end_idx + 1].reset_index(drop=True)

                    preds = []
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

            print()
        else:
            n = len(test)
            preds_all = np.zeros((n, len(TARGETS)), dtype=np.float32)
            for start in range(0, n, TEST_BATCHSIZE):
                end = min(n, start + TEST_BATCHSIZE)
                batch_df = test.iloc[start:end]
                x_batch = _build_test_batch_inputs(batch_df, spectrograms_test)

                fold_preds = []
                for model_i in range(SPLITS):
                    fold_preds.append(
                        models[model_i].predict_on_batch(x_batch).astype(np.float32)
                    )
                preds_all[start:end] = np.mean(fold_preds, axis=0)

            preds_all = preds_all.astype(np.float32)

        preds_all = np.asarray(preds_all, dtype=np.float32)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    assert len(sample_sub) == len(
        test
    ), f"test.csv len {len(test)} != sample_submission.csv len {len(sample_sub)}"

    preds_all = np.asarray(preds_all, dtype=np.float32)
    if preds_all.shape[0] != len(test):
        raise ValueError(
            f"Prediction length mismatch: got {preds_all.shape[0]} rows, expected {len(test)}"
        )

    if not np.array_equal(sample_sub["eeg_id"].values, test["eeg_id"].values):
        pred_df = pd.DataFrame(preds_all, columns=list(TARGETS))
        pred_df["eeg_id"] = test["eeg_id"].values
        pred_df = pred_df.set_index("eeg_id").reindex(sample_sub["eeg_id"].values)
        if pred_df.isna().any().any():
            raise ValueError(
                "Reindexing to sample_submission introduced NaNs (eeg_id mismatch)."
            )
        preds_all = pred_df[list(TARGETS)].values.astype(np.float32)

    preds_all = np.clip(preds_all, 1e-8, 1.0)
    preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

    sub = sample_sub[["eeg_id"]].copy()
    sub[list(TARGETS)] = preds_all.astype(np.float32)

    sub[list(TARGETS)] = np.clip(sub[list(TARGETS)].values, 1e-8, 1.0)
    sub[list(TARGETS)] = sub[list(TARGETS)].values / np.sum(
        sub[list(TARGETS)].values, axis=1, keepdims=True
    )

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-sum check: min/max =",
        float(sub[TARGETS].sum(1).min()),
        float(sub[TARGETS].sum(1).max()),
    )
    print("eeg_id uniqueness:", sub["eeg_id"].is_unique)
    print("Expected rows:", len(sample_sub), "Written rows:", len(sub))
