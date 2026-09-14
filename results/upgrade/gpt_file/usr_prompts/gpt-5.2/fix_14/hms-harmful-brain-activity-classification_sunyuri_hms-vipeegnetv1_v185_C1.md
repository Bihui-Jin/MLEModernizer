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

0.312954467843227

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` AttributeError that occurs in some Kaggle images. Then I fix the Keras Functional model build error by giving each EfficientNet backbone a unique `name` (instead of reusing the default `efficientnetb0` three times), which unblocks inference without changing the architecture. Finally, I ensure the script always writes a valid `submission.csv` with correct columns and per-row probabilities summing to 1, even if expected weight files are missing.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable forcing to the very top of the script (before any TensorFlow-related import can happen), which resolves the `MessageFactory.GetPrototype` error in Kaggle’s Python 3.12 images. Then I correct the per-channel normalization bug where the code divides by variance instead of standard deviation for spectrogram/STFT images, which is a score-impacting logic issue but does not change the model architecture or training/inference flow. Finally, I keep the existing unique EfficientNet names and submission-sanity checks so inference runs end-to-end and always writes a valid `submission.csv` with row probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation at the very top of the script (before any TF-related import) and also forcing the legacy API to avoid `MessageFactory.GetPrototype` issues in some Kaggle Python 3.12 images. Then I add a safe fallback to `tf.keras.applications.efficientnet.EfficientNetB0` in case `tf.keras.applications.EfficientNetB0` is missing in the installed TF build, which prevents runtime errors without changing the model architecture. Finally, to move the score down toward the target (since lower is better and current is far worse), I ensure the model actually loads weights from a valid Kaggle dataset path by adding an auto-detection fallback to common input folders if the configured `models2024040802` folder is absent—this preserves inference semantics and should substantially improve from “uniform preds / missing weights” behavior.'
- What this solution (achieved 1.40995) has done: 'I fix the current runtime crash coming from the protobuf/TensorFlow incompatibility by forcing a compatible protobuf setting even earlier and by pinning the Python-side protobuf implementation before any TensorFlow import happens. Then I make the script robust to the common case where the Kaggle image still raises `MessageFactory.GetPrototype` during TensorFlow import by falling back to the bundled `tensorflow-cpu` import path if available (without changing model logic). Finally, since your current score is far worse than target (lower is better), I ensure the model actually finds and loads the intended fold weight files by improving the weights-folder auto-detection (still only searching under `/kaggle/input`), because missing weights leading to uniform predictions is the primary reason for a very bad KL score.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by forcing the safe protobuf Python implementation before any TensorFlow import and, if needed, importing TensorFlow in a way that avoids the `MessageFactory.GetPrototype` failure. Then I make the weight-folder auto-detection stricter so it actually finds fold `.h5` files under `/kaggle/input` (your current score strongly suggests weights are not being loaded and the script is effectively near-uniform). Finally, I keep the model/data pipeline unchanged and ensure we still always write a valid `submission.csv` with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime failure caused by TensorFlow importing protobuf in a way that triggers `MessageFactory.GetPrototype` on this Kaggle Python 3.12 image by forcefully setting the protobuf env vars (not just `setdefault`) and importing `google.protobuf` before TensorFlow. Then I make the weight-folder auto-detection more reliable by searching recursively under `/kaggle/input` for the expected `f*_stage{STAGETEST}.h5` files so the model actually loads learned weights instead of falling back to uniform predictions (which is the main driver of the very poor KL score). Finally, I keep the model/data pipeline unchanged and only harden submission writing to always match `sample_submission.csv` column order and ensure row-wise probabilities sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow-related import and by proactively patching `google.protobuf.message_factory.MessageFactory.GetPrototype` to call `GetMessageClass` when the former is missing (the exact AttributeError you’re hitting). This is a minimal, execution-unblocking compatibility shim and does not change your model, preprocessing, or inference logic. I also keep your existing weight auto-detection and submission sanity checks so the pipeline reliably produces `submission.csv`, and (since your current score is far from target) this should allow actual weight loading/inference instead of failing early, which is the main lever to move KL down toward the target.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by applying a compatibility shim that patches both `google.protobuf.message_factory.MessageFactory.GetPrototype` and the newer C++/upb-backed factory classes that TensorFlow may use in Python 3.12 (your current patch only covers one path, so the `GetPrototype` AttributeError still occurs). This is an execution-unblocking change and does not alter your model, preprocessing, or inference semantics. Once TF imports, the existing weight auto-detection and fold-loading should start working, which should move the KL score substantially down from the near-uniform/missing-weights behavior toward your target. I keep the submission writing and probability normalization intact to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf patch so it handles the *instance* `MessageFactory` objects that TensorFlow/protobuf create (your current patch only touches the class and misses the object path, which is why `GetPrototype` is still missing). I apply the shim before importing TensorFlow and cover both Python and C++/upb-backed factories, unblocking TF import without changing your model/data logic. Then I keep the rest of the pipeline intact so it can actually load fold weights and run inference end-to-end; this should materially improve KL from the current near-uniform/missing-weights behavior toward your target. Submission writing and probability normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf patch so it robustly adds a `GetPrototype` method to the *actual factory instances* TensorFlow/protobuf use in this Kaggle Python 3.12 image (your current patch misses some object paths, causing the `MessageFactory object has no attribute GetPrototype` crash). This is execution-unblocking and score-neutral by itself, but it enables TensorFlow to import and your trained fold weights to load; that should materially improve KL from the current near-uniform/failed-inference behavior toward your target. I also fix a small but fatal inference-time bug in `DataGenerator` where it references `row.sign_id` (not present in test/train), which can break image/STFT generation; switching to `row.eeg_id` preserves semantics for consolidated data and unblocks end-to-end inference. Finally, I keep your model architecture/training logic unchanged and still write a valid `submission.csv` with correct columns and row-wise probabilities summing to 1.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_LEGACY_API"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024040802"
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

try:
    import google.protobuf  # noqa: F401
    import google.protobuf.message_factory as _py_mf

    def _safe_setattr(obj, name, value):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False

    def _patch_factory_instance(factory_obj):
        """Add GetPrototype to a factory instance if missing and GetMessageClass exists."""
        if factory_obj is None:
            return
        if hasattr(factory_obj, "GetPrototype"):
            return
        if not hasattr(factory_obj, "GetMessageClass"):
            return

        def _GetPrototype(descriptor):
            return factory_obj.GetMessageClass(descriptor)

        _safe_setattr(factory_obj, "GetPrototype", _GetPrototype)

    def _patch_factory_class(factory_cls):
        """Add GetPrototype to a factory class if missing and GetMessageClass exists."""
        if factory_cls is None:
            return
        if hasattr(factory_cls, "GetPrototype"):
            return
        if not hasattr(factory_cls, "GetMessageClass"):
            return

        def _GetPrototype(self, descriptor):  # type: ignore
            return self.GetMessageClass(descriptor)

        _safe_setattr(factory_cls, "GetPrototype", _GetPrototype)

    _patch_factory_class(getattr(_py_mf, "MessageFactory", None))
    for _nm in ("_DEFAULT_FACTORY", "default_factory", "DefaultFactory"):
        _patch_factory_instance(getattr(_py_mf, _nm, None))
    try:
        _patch_factory_instance(_py_mf.message_factory._DEFAULT_FACTORY)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _patch_factory_instance(_py_mf.MessageFactory())
    except Exception:
        pass

    try:
        import google.protobuf.pyext._message as _cpp_message  # type: ignore

        _patch_factory_class(getattr(_cpp_message, "MessageFactory", None))
        _patch_factory_class(getattr(_cpp_message, "DefaultFactory", None))
        for _nm in ("default_factory", "DefaultFactory", "_DEFAULT_FACTORY"):
            _patch_factory_instance(getattr(_cpp_message, _nm, None))
        try:
            _patch_factory_instance(_cpp_message.MessageFactory())  # type: ignore
        except Exception:
            pass
    except Exception:
        pass

    try:
        _tmp_fac = _py_mf.MessageFactory()
        if not hasattr(_tmp_fac, "GetPrototype") and hasattr(
            _tmp_fac, "GetMessageClass"
        ):
            _patch_factory_instance(_tmp_fac)
    except Exception:
        pass

except Exception as e:
    print("Warning: protobuf patch not applied:", repr(e))

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed (likely protobuf incompatibility in this Kaggle image). "
        "Tried forcing pure-Python protobuf via env vars and patching MessageFactory (class+instance)."
    ) from e

import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

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
except Exception as e:
    print("Warning: could not enable op determinism:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Warning: mixed precision experimental option not available:", repr(e))
else:
    print("Using full precision")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if PLATFORM == "kaggle":

    def _folder_has_stage_weights(p: str, stage: int) -> bool:
        try:
            files = os.listdir(p)
        except Exception:
            return False
        needed = [f"f{i}_stage{stage}.h5" for i in range(5)]
        return any(fn in files for fn in needed) or any(
            fn.startswith("f") and fn.endswith(f"_stage{stage}.h5") for fn in files
        )

    def _find_any_stage_weight_file(base: str, stage: int) -> str | None:
        """Return directory containing at least one stage weight file, else None."""
        if not os.path.isdir(base):
            return None
        try:
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if os.path.isdir(p) and _folder_has_stage_weights(p, stage):
                    return p
        except Exception:
            pass
        for root, dirs, files in os.walk(base):
            for fn in files:
                if fn.startswith("f") and fn.endswith(f"_stage{stage}.h5"):
                    return root
        return None

    if not os.path.exists(LOAD_MODELS_FROM) or not _folder_has_stage_weights(
        LOAD_MODELS_FROM, STAGETEST
    ):
        base = "/kaggle/input"
        preferred = _find_any_stage_weight_file(LOAD_MODELS_FROM, STAGETEST)
        if preferred is None:
            preferred = _find_any_stage_weight_file(base, STAGETEST)

        if preferred is not None:
            print(
                f"LOAD_MODELS_FROM not usable; auto-selected weights folder: {preferred}"
            )
            LOAD_MODELS_FROM = preferred
        else:
            print(f"Warning: could not find any f*_stage{STAGETEST}.h5 under {base}.")
            print("Will proceed; if no weights load, submission will be uniform.")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
                (len(indexes), 6, round(20 * SFREQ), 3, 4), dtype="float32"
            )
            x_eeg2 = np.zeros((len(indexes), 4, round(50 * SFREQ), 4), dtype="float32")
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

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
                rows = df.loc[df.eeg_id == row.eeg_id, :].reset_index(drop=True)
                for lk in TARGETS:
                    rows = rows.loc[rows[lk] == row[lk + "_raw"], :].reset_index(
                        drop=True
                    )
                rows_eeg = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )

                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = round(rows_eeg.eeg_label_offset_seconds[0] * SFREQ)

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

                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
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

                    x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / 0.229
                    x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / 0.224
                    x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / 0.225

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]

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

                    eeg = (eeg - np.mean(eeg, 1, keepdims=True)) / (
                        np.std(eeg, 1, keepdims=True) + 1e-6
                    )
                    x_eeg2[j, :, :, k] = eeg

                if "img" in DATATYPE:
                    img_key = row.eeg_id
                    img = self.imgs[img_key][:, :, k, :]
                    x_img[j, :, :, :, k] = img

                if "stft" in DATATYPE:
                    stft_key = row.eeg_id
                    stft = self.stfts[stft_key][:, :, k]
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

                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / 0.229
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / 0.224
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / 0.225

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
            x.append(x_eeg2)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
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
def _EfficientNetB0(include_top=False, weights=None, input_shape=None, name=None):
    try:
        fn = tf.keras.applications.EfficientNetB0
    except AttributeError:
        fn = tf.keras.applications.efficientnet.EfficientNetB0
    return fn(
        include_top=include_top, weights=weights, input_shape=input_shape, name=name
    )


def build_model(TARGETS_PRETRAIN):
    l2norm = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2_norm"
    )

    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1, name="concat_spe_quadrants")(
            [x_spe1, x_spe2, x_spe3, x_spe4]
        )

        base_model_spe = _EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_spe"
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

        x_eeg1 = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_pair12")(
            [x_eeg1, x_eeg2]
        )
        x_eeg2 = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_pair34")(
            [x_eeg3, x_eeg4]
        )
        x_eeg = tf.keras.layers.Concatenate(axis=2, name="concat_eeg_pairs")(
            [x_eeg1, x_eeg2]
        )

        base_model_eeg = _EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_eeg"
        )

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = l2norm(x_eeg)

        inp.append(inp_eeg)

        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_spe_eeg")([y, x_eeg])
        else:
            y = x_eeg

    if "eeg" in DATATYPE:
        inp_eeg2 = tf.keras.Input(shape=(4, round(50 * SFREQ), 4), name="inp_eeg2")
        inp.append(inp_eeg2)

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
        x_img1 = inp_img[:, :, :, :, 0]
        x_img2 = inp_img[:, :, :, :, 1]
        x_img3 = inp_img[:, :, :, :, 2]
        x_img4 = inp_img[:, :, :, :, 3]
        x_img = tf.keras.layers.Concatenate(axis=1, name="concat_img_quadrants")(
            [x_img1, x_img2, x_img3, x_img4]
        )

        base_model_img = _EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_img"
        )

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = l2norm(x_img)

        inp.append(inp_img)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="concat_modalities")(
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
        x_stft = tf.keras.layers.Concatenate(axis=1, name="concat_stft_quadrants")(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )

        base_model_stft = _EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=None,
            name="efficientnetb0_stft",
        )

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
        x_stft = l2norm(x_stft)

        inp.append(inp_stft)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1, name="concat_modalities_with_stft")(
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
if not NEEDTRAIN:
    if "spe" in DATATYPE:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
        elif PLATFORM == "kaggle":
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )
        print("Test shape", test.shape)

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
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test eeg parquets")

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
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE:
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
    with strategy.scope():
        model = build_model(TARGETS)

    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2 if "spe" in DATATYPE else None,
        eegs=eegs2 if "eeg" in DATATYPE else None,
        imgs=imgs2 if "img" in DATATYPE else None,
        stfts=stfts2 if "stft" in DATATYPE else None,
        targets=TARGETS,
    )

    for i in range(5):
        print(f"Fold {i + 1}")
        wpath = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
        if os.path.exists(wpath):
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
        else:
            print(f"Warning: missing weights: {wpath}")

    if len(preds) == 0:
        print(
            "Warning: no weights loaded; using uniform probabilities to create a valid submission."
        )
        pred = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float64)
    else:
        pred = np.mean(preds, axis=0)

    print("\nTest preds shape", pred.shape)

    pred = np.asarray(pred, dtype=np.float64)
    pred = np.nan_to_num(
        pred,
        nan=1.0 / pred.shape[1],
        posinf=1.0 / pred.shape[1],
        neginf=1.0 / pred.shape[1],
    )
    pred = np.clip(pred, 1e-9, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sample_path = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
    if os.path.exists(sample_path):
        sample_sub = pd.read_csv(sample_path)
        out_cols = list(sample_sub.columns)
    else:
        out_cols = ["eeg_id"] + list(TARGETS)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub = sub[out_cols]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-sum check (min/max):",
        sub.iloc[:, -6:].sum(axis=1).min(),
        sub.iloc[:, -6:].sum(axis=1).max(),
    )
    assert sub.columns[0] == "eeg_id"
    assert list(sub.columns[1:]) == list(TARGETS)
    assert np.allclose(sub.iloc[:, 1:].sum(axis=1).values, 1.0, atol=1e-6)
    print("Wrote submission.csv")
