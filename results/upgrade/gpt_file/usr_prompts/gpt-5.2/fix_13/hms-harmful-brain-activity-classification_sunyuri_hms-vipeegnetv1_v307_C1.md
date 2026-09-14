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

0.3265909548437108

# 6. Current score

0.87497

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the two blockers that prevent an end-to-end run and a valid `submission.csv`: (1) the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure‑Python protobuf implementation before importing TensorFlow, and (2) the missing external weights directory by adding a safe fallback that generates calibrated, valid probabilities directly from the training vote distribution when weights are unavailable. This keeps the core model/data pipeline intact (we only bypass weight-loading when files don’t exist), guarantees rows sum to 1, and ensures a `.csv` submission is always written. The fallback uses the mean normalized label distribution (a sensible baseline for KL divergence) to move score from “not yielded” to a valid, reasonable score.'
- What this solution (achieved 0.87497) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling C-descriptor parsing before importing TensorFlow, which is the reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle-style images. I also add a safe import-time fallback: if TensorFlow still cannot be imported, the code automatically switch to the existing train-prior baseline so an end-to-end `submission.csv` is always produced. To improve the current KL score toward the target (lower is better) without changing the core model, I make the fallback baseline stronger but still legitimate by using a patient-level (leave-one-patient-out safe) smoothed prior from `train.csv` when available, otherwise falling back to the global prior. All probability outputs be clipped and renormalized to guarantee each row sums to 1 and matches the required submission columns.'
- What this solution (achieved 0.87497) has done: 'I fix the TensorFlow/protobuf crash earlier in the script by forcing the pure-Python protobuf runtime and monkey-patching the missing `MessageFactory.GetPrototype` method before importing TensorFlow (this is the direct cause of your current runtime error). I also make the TensorFlow import fully optional and guarantee an end-to-end run by keeping the existing patient/global-prior fallback when TF or weights are unavailable. Finally, I keep the submission strictly valid for the KL metric by clipping and renormalizing probabilities to sum to 1 and writing `submission.csv` with the exact required columns. These changes are runtime/stability focused and keep your model/training logic intact.'
- What this solution (achieved 0.87497) has done: 'I fix the protobuf monkey-patch so it targets the correct `google.protobuf.message_factory.MessageFactory` class (the current patch hits a different symbol and the import still crashes). I also ensure the patch is applied before any TensorFlow import and add a defensive second patch against the instance method lookup that triggers `'MessageFactory' object has no attribute 'GetPrototype'`. These changes are runtime/stability focused and keep your model/inference logic intact; once TensorFlow imports, your existing weight-loading/inference path can run (otherwise the existing smoothed patient/global prior fallback still produce a valid KL-safe submission). Finally, I keep the probability clipping/renormalization as-is to guarantee rows sum to 1 and the CSV matches the required columns.'
- What this solution (achieved 0.87497) has done: 'I fix the TensorFlow/protobuf import crash by applying a more robust monkey-patch directly to `google.protobuf.message_factory.MessageFactory` (and also the module-level symbol) *before* any TensorFlow import, ensuring `GetPrototype` exists and preventing the current `AttributeError`. This is a runtime blocker fix and is score-neutral by itself, but it enables your existing multi-fold EfficientNet inference path to run when weights are present (which should improve KL substantially versus the prior-only fallback and move you toward the 0.3266 target). I also make the TensorFlow import guard stricter: if TF import still fails for any reason, the code continues to produce a valid submission using the existing smoothed patient/global prior baseline. Finally, I keep the probability clipping/renormalization to guarantee each row sums to 1 and the CSV schema matches the required submission columns.'
- What this solution (achieved 0.87497) has done: 'I fix the protobuf/TensorFlow import crash by applying a robust `GetPrototype` monkey-patch to the exact protobuf class TensorFlow calls, and I do it before any TensorFlow import happens. This is a runtime blocker fix and should be score-improving because it allows your intended multi-fold EfficientNet inference path to run when weights are available, instead of always falling back to the weaker prior baseline. I also make the patch defensive across protobuf versions by patching both `google.protobuf.message_factory` and `google.protobuf.message_factory.MessageFactory` symbols, and I keep your existing prior fallback + probability clipping/renormalization unchanged to guarantee a valid KL-safe submission CSV even if TF/weights are unavailable.'
- What this solution (achieved 0.87497) has done: 'I fix the protobuf/TensorFlow import crash by applying a more robust `MessageFactory.GetPrototype` patch that also handles cases where TensorFlow calls the method on an instance or where the class symbol differs across protobuf versions. This is a runtime blocker fix and is score-improving because it enables your intended multi-fold EfficientNet inference path to run when the Kaggle dataset includes the weight files; otherwise your existing smoothed patient/global prior fallback remains unchanged. I also make the TensorFlow import guard stricter so any remaining TF/protobuf issues cleanly trigger the fallback without aborting before writing `submission.csv`. Finally, I keep the submission probabilities clipped/renormalized to ensure every row sums to 1 and the CSV matches the required columns.'
- What this solution (achieved 0.87497) has done: 'I fix the protobuf monkey-patch that currently raises `AttributeError: GetPrototype` by removing the unsafe `__getattr__` injection and instead adding a simple `GetPrototype` method when it’s missing. This patch be applied before any TensorFlow import, preserving your intended TensorFlow inference path when weights exist, while keeping the existing safe fallback to the smoothed patient/global prior if TF or weights are unavailable. I also keep the submission probability clipping/renormalization so every row sums to 1 and the CSV matches the required columns. These changes are runtime/stability focused and should enable your higher-performing model path to run (improving KL toward the target) when the weights dataset is present.'
- What this solution (achieved 0.87497) has done: 'I fix the TensorFlow/protobuf crash by making the protobuf monkey-patch robust to the exact call pattern TensorFlow uses (it can call `GetPrototype` on a *default* `MessageFactory` instance inside the module, not just on the class). Then I ensure TensorFlow is only imported after the patch is applied, and if it still fails, the script continue to the existing patient/global-prior fallback so it always writes a valid `submission.csv`. These changes are runtime/stability focused (score-neutral by themselves) but should allow your intended multi-fold model inference path to run when weights are present, which is the legitimate way to improve KL toward the target. Finally, I keep the probability clipping/renormalization to guarantee each row sums to 1 and matches the required columns.'
- What this solution (achieved 0.87497) has done: 'I fix the remaining TensorFlow/protobuf import crash by strengthening the protobuf `MessageFactory.GetPrototype` monkey-patch so it also patches the actual default factory instance used inside `google.protobuf.message_factory` (commonly `_DEFAULT_FACTORY`), and by applying it before any TensorFlow import. This is a runtime/stability fix and should be score-improving because it allows your intended multi-fold model inference path to run when weights are present, instead of always falling back to the weaker prior baseline. I also make the TensorFlow import guard fully safe (any import failure triggers the existing patient/global-prior fallback without aborting) and keep the submission probability clipping/renormalization so each row sums to 1 and matches the required columns. No model/training logic, architecture, or evaluation semantics are changed.'
- What this solution (achieved 0.87497) has done: 'I fix the runtime blocker by forcing protobuf’s pure-Python implementation and applying a more robust `MessageFactory.GetPrototype` patch that also targets the actual default factory instance used by protobuf/TensorFlow. I also ensure the patch runs before any TensorFlow import, and wrap the TensorFlow import in a safe guard so the script always continues to the existing patient/global-prior fallback and writes `submission.csv`. These changes keep your model/inference logic intact but should allow the intended TF path to run when available (improving KL toward the target), while remaining score-neutral if TF still cannot load. Finally, I keep the probability clipping/renormalization to guarantee each row sums to 1 and the submission schema is valid.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C_DESCRIPTORS", "1"
)


def _patch_protobuf_messagefactory_getprototype():
    """
    TensorFlow (via protobuf internal helpers) may call MessageFactory.GetPrototype.
    Some protobuf builds have MessageFactory without GetPrototype, causing:
      AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

    Robust patch strategy:
      - Patch MessageFactory class if missing GetPrototype and GetMessageClass exists.
      - Patch module-level default factory instances that TF/protobuf may call directly:
        _DEFAULT_FACTORY, _default_factory, message_factory, default_factory.
      - Patch the actual object returned by module-level "Default"/"GetFactory" if present.
    """

    def _add_getprototype_to_class(MF):
        if MF is None:
            return
        if hasattr(MF, "GetPrototype"):
            return
        if not hasattr(MF, "GetMessageClass"):
            return

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        try:
            setattr(MF, "GetPrototype", _GetPrototype)
        except Exception:
            pass

    def _add_getprototype_to_instance(inst):
        if inst is None:
            return
        if hasattr(inst, "GetPrototype"):
            return
        if not hasattr(inst, "GetMessageClass"):
            return

        def _GetPrototype(descriptor, _inst=inst):
            return _inst.GetMessageClass(descriptor)

        try:
            setattr(inst, "GetPrototype", _GetPrototype)
        except Exception:
            pass

    def _apply(mf_module):
        MF = getattr(mf_module, "MessageFactory", None)
        _add_getprototype_to_class(MF)

        for name in (
            "_DEFAULT_FACTORY",
            "_default_factory",
            "default_factory",
            "message_factory",
        ):
            _add_getprototype_to_instance(getattr(mf_module, name, None))

        for getter in ("Default", "GetFactory"):
            fn = getattr(mf_module, getter, None)
            if callable(fn):
                try:
                    _add_getprototype_to_instance(fn())
                except Exception:
                    pass

    try:
        import google.protobuf.message_factory as mf_mod

        _apply(mf_mod)
    except Exception:
        pass

    try:
        from google.protobuf import message_factory as mf_mod2

        _apply(mf_mod2)
    except Exception:
        pass


_patch_protobuf_messagefactory_getprototype()

import warnings

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241126a"  # the path of trained model weights for testing

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

LEARN_RATE = 1e-4
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import io
from PIL import Image
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    tf = None
    optimizers = None
    clone_model = None
    print("WARNING: TensorFlow failed to import; will use baseline submission.")
    print("TF import error:", TF_IMPORT_ERROR)

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
import time
import gc

if TF_AVAILABLE:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(
            device="/gpu:0" if len(gpus) == 1 else "/cpu:0"
        )
        print(f"Using {len(gpus)} GPU")
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

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision requested but not available; continuing.")
    else:
        print("Using full precision")

    length = round(32 / (EEG_MULTIPLY / 10))
    x = np.linspace(1, length, length)
    y = x * 0
    y[15:] = 1
    WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
    WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
    WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
    EEG_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [i + "_raw" for i in TARGETS]

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
        ):

            self.dataframe = dataframe.reset_index(drop=True)
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
            return int(np.ceil(len(self.dataframe) / self.batch_size))

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
                    spe = []
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
                    spe = (spe - np.mean(spe, keepdims=True)) / (
                        np.std(spe, keepdims=True) + 1e-6
                    )
                    x_spe[j] = spe

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]
                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )

                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

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
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                        stft = stft[:, :, :STFT_WIDE]

                    stft = np.clip(stft, a_min=1e-6, a_max=1e6)
                    stft = np.log2(stft)

                    if self.mode == "train":
                        stft[0:8, :, :] = stft[0:8, :, :][
                            np.random.permutation(8), :, :
                        ]
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
                    sign_id = row.sign_id
                    img = self.imgs[sign_id]
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
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                    if self.sample_weights:
                        sample_weights[j] = (
                            sum(row[[t + "_raw" for t in TARGETS]].values) / 20
                        )
                    else:
                        sample_weights[j] = 1.0
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




## === cell 3
if TF_AVAILABLE:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super().__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
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




## === cell 4
if TF_AVAILABLE:

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

            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None
            )
            base_model_spe = tf.keras.Model(
                inputs=base_model_spe.input,
                outputs=base_model_spe.get_layer("block5c_add").output,
            )
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

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_tensor=x_eeg
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg.output
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

            inp.append(inp_eeg)
            if y is not None:
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
                include_top=False, weights=None
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

            inp.append(inp_stft)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            else:
                y = x_stft

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(inp_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

            inp.append(inp_img)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
            else:
                y = x_img

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 5
def make_smoothed_patient_prior(train_df: pd.DataFrame, targets, alpha: float = 1.0):
    y = train_df[list(targets)].values.astype(np.float64)
    y = y / np.maximum(y.sum(axis=1, keepdims=True), 1e-12)

    global_prior = y.mean(axis=0)
    global_prior = np.clip(global_prior, 1e-12, 1.0)
    global_prior = global_prior / global_prior.sum()

    by_patient = {}
    if "patient_id" in train_df.columns:
        for pid, g in train_df.groupby("patient_id", sort=False):
            yg = g[list(targets)].values.astype(np.float64)
            yg = yg / np.maximum(yg.sum(axis=1, keepdims=True), 1e-12)
            patient_mean = yg.mean(axis=0)

            smoothed = patient_mean + alpha * global_prior
            smoothed = np.clip(smoothed, 1e-12, 1.0)
            smoothed = smoothed / smoothed.sum()
            by_patient[int(pid)] = smoothed

    return global_prior, by_patient




## === cell 6
if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    global_prior, patient_prior_map = make_smoothed_patient_prior(
        df, TARGETS, alpha=1.0
    )

    if not TF_AVAILABLE:
        print("TensorFlow unavailable; writing baseline submission.")
        preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
        for i, pid in enumerate(test["patient_id"].astype(int).values):
            preds_all[i] = patient_prior_map.get(pid, global_prior)
        preds_all = np.clip(preds_all, 1e-12, 1.0)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)
    else:
        model_template = build_model()

        models = []
        weight_paths = []
        for model_i in range(SPLITS):
            weight_paths.append(
                os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
            )
        have_all_weights = all(os.path.exists(p) for p in weight_paths)

        if have_all_weights:
            print(f"Found weights in {LOAD_MODELS_FROM}. Running model inference...")
            for model_i in range(SPLITS):
                print(f"Fold {model_i + 1}")
                model = clone_model(model_template)
                model.load_weights(weight_paths[model_i])
                models.append(model)

            if "spe" in DATATYPE:
                PATH_test_spe = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
                files_test = os.listdir(PATH_test_spe)
                print(f"There are {len(files_test)} test spectrogram parquets")
                for i, f in enumerate(files_test):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    tmp = pd.read_parquet(f"{PATH_test_spe}{f}")
                    name = int(f.split(".")[0])
                    spectrograms_test[name] = tmp.iloc[:, 1:].values

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            preds_all_list = []
            start_idx = 0

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")

                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
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
                            img = np.array(img)[:, :, :1] / 255.0
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

                        imgs_test[test_plot.sign_id[j]] = img_save

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts_test[eeg_id] = ss
                    stfts_test[-eeg_id] = tt

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    end_idx = i + 1
                    batch_df = test.iloc[start_idx:end_idx].reset_index(drop=True)

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
                    preds_all_list.append(pred)

                    start_idx = end_idx

                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

            preds_all = np.concatenate(preds_all_list, axis=0)

            preds_all = np.asarray(preds_all, dtype=np.float64)
            preds_all = np.clip(preds_all, 1e-12, 1.0)
            preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

        else:
            print(f"WARNING: Model weights not found in '{LOAD_MODELS_FROM}'.")
            print(
                "Falling back to smoothed patient/global prior from train vote distribution."
            )

            preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
            for i, pid in enumerate(test["patient_id"].astype(int).values):
                preds_all[i] = patient_prior_map.get(pid, global_prior)
            preds_all = np.clip(preds_all, 1e-12, 1.0)
            preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = preds_all.astype(np.float32)

    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-12, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub[TARGETS] = vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print("Saved to submission.csv")
