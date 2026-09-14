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

0.3463126379953835

# 6. Current score

1.44107

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing the C++ protobuf implementation (the current env var settings trigger the known `MessageFactory.GetPrototype` failure in some Kaggle TF builds). Then I make the model-weight loading robust: if the expected `/kaggle/input/models2024040403/*.h5` files are missing, the script fall back to a valid, well-formed probability submission (row-normalized class priors from train), ensuring a `submission.csv` is always produced. I also guard optional spectrogram loading so it won’t crash if `"spe"` isn’t in `DATATYPE`, and I keep the existing core model/data logic untouched when weights are available.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` failure in this Kaggle TF build. I keep the existing model/data logic intact, and ensure the script always reaches submission writing. Additionally, I make the GPU strategy selection robust so it won’t crash on CPU-only environments. These changes are execution-stability fixes and are score-neutral (they simply allow your intended inference path to run when weights are present, otherwise keeping the same prior fallback).'
- What this solution (achieved 1.41937) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime *and* ensuring TensorFlow is imported only after that environment change (the current error indicates the C++ protobuf path is still being used). I also make the submission path robust by falling back to a well-formed prior-probability submission if TensorFlow still fails to import at runtime, so a valid `submission.csv` is always produced. These changes are execution/stability-focused and keep your model/data core logic unchanged when TensorFlow + weights are available, while improving score versus a broken/no-model path by guaranteeing reasonable probabilities. Finally, I keep the required row-normalization to ensure every row sums to 1 (submission validity).'
- What this solution (achieved 1.41937) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf runtime *and* ensuring TensorFlow is imported only after those environment variables are set, plus disabling TF’s use of the C++ protobuf implementation when possible. This is necessary to let your intended inference path run; otherwise you always fall back to priors, which explains the poor score versus the target. I also add a safe fallback to the prior-based submission if TensorFlow still fails at runtime, and keep the model/data core logic unchanged. Finally, I keep the submission strictly valid by row-normalizing probabilities and writing `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash that’s stopping the intended inference path by setting the protobuf environment variables *before* any TensorFlow-related import and by importing `google.protobuf` first to lock the pure-Python implementation. I also remove the GPU visibility env change that currently happens too late to be reliable, while keeping your strategy selection and model/data logic unchanged. Finally, I keep (and slightly harden) the prior-based fallback so a valid `submission.csv` is always written even if TF still fails or model weights are missing, and ensure predictions are always properly normalized.'
- What this solution (achieved 1.41937) has done: 'I fix the protobuf/TensorFlow crash causing `MessageFactory.GetPrototype` by forcing the pure-Python protobuf implementation *and* blocking the incompatible C++/upb path before any TensorFlow import (this currently still leaks through in Kaggle Py3.12 builds). Then I add a safe “TF import retry” that sets the remaining env knobs and imports protobuf earlier, so the intended model inference path can run instead of always falling back to priors (your current 1.419 score strongly suggests the fallback path is being used). Finally, I keep the exact modeling/data logic unchanged, but harden submission writing and normalization so a valid `submission.csv` is always produced even if TF still cannot load or weights are missing.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash that is currently stopping the script at import time by setting protobuf env vars earlier and (critically) forcing the Python protobuf backend via `google.protobuf.internal.api_implementation` before importing TensorFlow. Then, to move the score toward your target (lower is better), I ensure the intended model-inference path actually runs when weights are present; otherwise you are stuck on the prior fallback which explains the very poor 1.419 score. Finally, I harden the submission-writing path so it always produces a valid `submission.csv` with rows summing to 1, even if TF still fails or the model files are missing.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by preventing TensorFlow from importing the incompatible `upb`/C++ protobuf backend in this Kaggle Py3.12 environment and falling back cleanly if TF still cannot be imported. This is an execution-unblocking change: right now you never reach the intended model inference path (hence the very poor 1.419 score), so fixing TF import is the smallest legitimate change expected to move the score toward the 0.346 target. I also remove the hard CPU-forcing `CUDA_VISIBLE_DEVICES=""` (it can hurt inference stability/perf and is unnecessary once import is fixed), while keeping the model/data logic unchanged. Finally, I keep the prior-based submission fallback and strict row-normalization to always produce a valid `submission.csv`.'
- What this solution (achieved 1.44107) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution by removing the brittle protobuf “force python” hack and instead cleanly handling TF availability in this Py3.12 Kaggle environment (where TF may simply be unusable). To move the score down toward your target (lower is better) without changing your model core logic, I replace the very-weak global prior fallback with a stronger but still label-safe fallback: patient-conditioned class priors computed from train.csv (and global prior for unseen patients). I also fix a logic bug in the DataGenerator where EEG labels accidentally use `row_spe` instead of `row_eeg`, which can hurt calibration when the TF path is actually used. Finally, I keep strict row-normalization and always write a valid `submission.csv`.'
- What this solution (achieved 1.44107) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime *before* importing TensorFlow, which unblocks your intended inference path (and should substantially lower the KL score vs the weak prior fallback). I keep your model/data logic unchanged, but make the TF import handling robust: if TF still can’t import, we fall back to the existing patient-prior submission so a valid `submission.csv` is always produced. I also keep the GPU/strategy selection safe for CPU-only environments, and preserve strict probability row-normalization for submission validity. These changes are directly targeted at the runtime error and getting back to the model-based predictions to move score toward the 0.346 target.'
- What this solution (achieved 1.44107) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution at import time by preventing TF from loading the incompatible C++/upb protobuf backend in this Kaggle Py3.12 environment (the `MessageFactory.GetPrototype` error). This should unblock the intended model inference path (when weights exist) and is the smallest legitimate change expected to reduce KL toward your 0.346 target instead of always falling back to priors (which yields ~1.44). I also keep a robust fallback: if TF still can’t import or weights are missing, the script still write a valid `submission.csv` using patient-conditioned priors. Core model/data logic is preserved; changes are limited to import/runtime stability and submission robustness.'
- What this solution (achieved 1.44107) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *before* anything can import TensorFlow, and by hard-disabling the C++/upb protobuf path (the current setup still leaks through in this Kaggle Py3.12 environment). This is execution-unblocking and should move the score down toward your target because it allows the intended model inference path to run instead of always falling back to priors (~1.44 KL). I also keep the existing patient-prior fallback path intact so a valid `submission.csv` is always produced if TensorFlow or model weights are unavailable. Core model/data logic, architecture, and prediction semantics are otherwise unchanged.'

# 9. Code solution

## === cell 0
import os
import io
import warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C", "1")

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
tf = None


def _import_tensorflow_safely():
    """
    Import TensorFlow while forcing python protobuf implementation.
    If TF import still fails, caller will fall back to prior-based submission.
    """
    try:
        import google.protobuf  # noqa: F401

        try:
            from google.protobuf.internal import api_implementation  # type: ignore

            try:
                api_implementation._SetImplementationType("python")
            except Exception:
                pass
        except Exception:
            pass

        import tensorflow as _tf  # noqa: F401

        return _tf, None
    except Exception as e:
        return None, repr(e)


tf, TF_IMPORT_ERROR = _import_tensorflow_safely()
TF_AVAILABLE = tf is not None

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False

DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3

LOAD_MODELS_FROM = "models2024040403"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

threshold = 0.2

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 128

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024
NSPLIT = 5
BATCHSIZE = 16

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

READ_EXTRA_SPEC_FILES = True
READ_EXTRA_EEG_FILES = True
READ_EXTRA_IMG_FILES = True
READ_EXTRA_STFT_FILES = True

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
    print("TensorFlow version =", tf.__version__)

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

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision requested but not enabled on this TF build")
    else:
        print("Using full precision")
else:
    print("WARNING: TensorFlow import failed; will generate a prior-based submission.")
    print("TF import error:", TF_IMPORT_ERROR)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    import albumentations as albu  # type: ignore
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

                row_spe = row
                row_eeg = row
                row_img = row
                row_stft = row

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    r_spe = round(row_spe.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row_eeg.eeg_label_offset_seconds * SFREQ)

                if self.mode == "train":
                    x1 = np.random.rand() * LENGTH / 2
                    x2 = np.random.rand() * LENGTH / 2
                    x_spe_min1 = round(min(x1, x2))
                    x_spe_max1 = round(max(x1, x2))
                    x1 = np.random.rand() * LENGTH / 2 + LENGTH / 2
                    x2 = np.random.rand() * LENGTH / 2 + LENGTH / 2
                    x_spe_min2 = round(min(x1, x2))
                    x_spe_max2 = round(max(x1, x2))

                    x1 = np.random.rand() * IMG_WIDE / 2
                    x2 = np.random.rand() * IMG_WIDE / 2
                    x_img_min1 = round(min(x1, x2))
                    x_img_max1 = round(max(x1, x2))
                    x1 = np.random.rand() * IMG_WIDE / 2 + IMG_WIDE / 2
                    x2 = np.random.rand() * IMG_WIDE / 2 + IMG_WIDE / 2
                    x_img_min2 = round(min(x1, x2))
                    x_img_max2 = round(max(x1, x2))

                for k in range(4):
                    if "spe" in DATATYPE:
                        spe = self.specs[row_spe.spectrogram_id][
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

                        if (self.mode == "train") and (np.random.random() > 0.5):
                            spe[:, x_spe_min1:x_spe_max1, :] = 0
                        if (self.mode == "train") and (np.random.random() > 0.5):
                            spe[:, x_spe_min2:x_spe_max2, :] = 0

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
                        eeg = self.eegs[row_eeg.eeg_id][
                            :, r_eeg : r_eeg + round(50 * SFREQ), k
                        ]

                        if eeg.shape[1] < 50 * SFREQ:
                            eeg = np.concatenate((eeg, eeg), 1)
                            eeg = eeg[:, : round(50 * SFREQ)]

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
                        img = self.imgs[row_img.eeg_id][:, :, k, :]
                        x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        stft = self.stfts[row_stft.eeg_id][:, :, :, k]

                        if (stft.shape[1] != 64) or (stft.shape[2] != 256):
                            stft2 = np.zeros((4, 64, 128))
                            stft2[:, : stft.shape[1], : stft.shape[2]] = stft
                            stft = stft2

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
                    label = 0
                    if "spe" in DATATYPE:
                        label = label + row_spe[self.targets].values
                    if "eeg" in DATATYPE:
                        label = label + row_eeg[self.targets].values
                    if "img" in DATATYPE:
                        label = label + row_img[self.targets].values
                    if "stft" in DATATYPE:
                        label = label + row_stft[self.targets].values
                    label = label / len(DATATYPE)

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




## === cell 2
if TF_AVAILABLE:
    try:
        import efficientnet.tfkeras as efn  # type: ignore

        EfficientNetB0 = efn.EfficientNetB0
        print("Using efficientnet.tfkeras EfficientNetB0")
    except Exception:
        EfficientNetB0 = tf.keras.applications.EfficientNetB0
        print(
            "efficientnet.tfkeras not available; using tf.keras.applications.EfficientNetB0"
        )

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
            x_spe = tf.keras.layers.Concatenate(axis=1, name="cat_spe")(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="spe_extractor"
            )
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
            x_eeg = tf.keras.layers.Concatenate(axis=1, name="cat_eeg")(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )

            base_model_eeg = EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="eeg_extractor"
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
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
            x_img1 = inp_img[:, :, :, :, 0]
            x_img2 = inp_img[:, :, :, :, 1]
            x_img3 = inp_img[:, :, :, :, 2]
            x_img4 = inp_img[:, :, :, :, 3]
            x_img = tf.keras.layers.Concatenate(axis=1, name="cat_img")(
                [x_img1, x_img3, x_img4, x_img2]
            )

            base_model_img = EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="img_extractor"
            )
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
            x_img = l2norm(x_img)

            inp.append(inp_img)
            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="cat_prev_img")([y, x_img])
            else:
                y = x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4), name="inp_stft")
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1, name="cat_stft")(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="stft_extractor"
            )
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
            x_stft = l2norm(x_stft)

            inp.append(inp_stft)
            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="cat_prev_stft")(
                    [y, x_stft]
                )
            else:
                y = x_stft

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN),
            activation="softmax",
            dtype="float32",
            name="head_softmax",
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
        return model

    def my_loss(y_ture, y_pred):
        y_pred1 = y_pred[:, 5:6]
        y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 3
def make_patient_prior_submission(
    train_df: pd.DataFrame, test_df: pd.DataFrame, targets
) -> np.ndarray:
    td = train_df.copy()
    votes = td[list(targets)].astype(np.float64)
    td_sum = votes.sum(axis=1).values
    td_sum = np.clip(td_sum, 1e-12, None)
    probs = votes.values / td_sum[:, None]
    td_probs = pd.DataFrame(probs, columns=list(targets))
    td_probs["patient_id"] = td["patient_id"].values

    pat_mean = td_probs.groupby("patient_id")[list(targets)].mean()

    global_votes = train_df[list(targets)].sum(axis=0).values.astype(np.float64)
    global_votes = np.clip(global_votes, 1e-12, None)
    global_prior = global_votes / global_votes.sum()

    out = np.zeros((len(test_df), len(targets)), dtype=np.float32)
    for i, pid in enumerate(test_df["patient_id"].values):
        if pid in pat_mean.index:
            p = pat_mean.loc[pid].values.astype(np.float64)
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
        else:
            p = global_prior
        out[i] = p.astype(np.float32)

    out = np.clip(out, 1e-12, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


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

if not TF_AVAILABLE:
    pred = make_patient_prior_submission(df, test, TARGETS)
    sub = sample_sub.copy()
    sub[TARGETS] = pred
    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-12, None)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub[TARGETS] = vals.astype(np.float32)
    sub.to_csv("submission.csv", index=False)
    print("Wrote patient-prior-based submission.csv due to TF import failure.")
    print("TF import error:", TF_IMPORT_ERROR)
    print("Submission shape", sub.shape)
    print(sub.head())
else:
    if "spe" in DATATYPE:
        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()
    else:
        spectrograms2 = {}

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
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
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
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
                frequencies, times, Sxx = signal.spectrogram(
                    eeg[:, round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ)],
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

            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

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
                tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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
            img2 = np.reshape(img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1))

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
                tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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
            img3 = np.reshape(img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1))

            img = np.concatenate([img, img2, img3], -1)
            imgs2[name] = img

    print()

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

    pred = None
    weights_missing = False
    missing_paths = []

    try:
        with strategy.scope():
            model = build_model(TARGETS)
    except Exception as e:
        print("WARNING: Model build failed; creating patient-prior submission instead.")
        print("Model build error:", repr(e))
        pred = make_patient_prior_submission(df, test, TARGETS)
        weights_missing = True  # skip weight loading

    preds = []
    if not weights_missing:
        for i in range(NSPLIT):
            weight_path = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            if not os.path.exists(weight_path):
                weights_missing = True
                missing_paths.append(weight_path)
                break

    if not weights_missing:
        for i in range(NSPLIT):
            print(f"Fold {i+1}")
            weight_path = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            model.load_weights(weight_path)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred = np.clip(pred, 1e-12, None)
        pred = pred / pred.sum(axis=1, keepdims=True)
    else:
        if pred is None:
            print(
                "WARNING: Model weights not found; creating patient-prior submission instead."
            )
            for p in missing_paths[:3]:
                print("Missing:", p)
            pred = make_patient_prior_submission(df, test, TARGETS)
        print("Patient-prior preds shape", pred.shape)

    sub = sample_sub.copy()
    sub[TARGETS] = pred

    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-12, None)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub[TARGETS] = vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sums (min/mean/max):",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).mean(),
        sub[TARGETS].sum(axis=1).max(),
    )
