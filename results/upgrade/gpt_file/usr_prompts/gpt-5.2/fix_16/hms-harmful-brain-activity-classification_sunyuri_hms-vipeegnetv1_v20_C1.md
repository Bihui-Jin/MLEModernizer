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

0.5048960753931401

# 6. Current score

1.39794

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation early and making mixed precision optional so the notebook runs in the Kaggle Python 3.12 environment. Then I fix the submission-length mismatch by ensuring we always build predictions for every `eeg_id` in `sample_submission.csv` (not `test.csv`) and by aligning/ordering rows exactly to the sample submission before writing `submission.csv`. Finally, I make inference robust to missing/partial EEG/spec files by safely falling back to uniform probabilities per missing item while still producing a valid, correctly-sized submission with row sums equal to 1.'
- What this solution (achieved 1.39794) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf to the pure-Python implementation earlier and (most importantly) forcing a compatible protobuf runtime configuration before importing TensorFlow, plus disabling mixed precision by default in this environment (it doesn’t change core logic and avoids common TF 2.15+ + py312 edge crashes). Then I make sure inference always produces predictions for every `eeg_id` in `sample_submission.csv` and writes a correctly-shaped `submission.csv` with probabilities summing to 1. Finally, to move the KL score down toward your target (lower is better) without changing the model, I add a minimal probability calibration step that blends model predictions with the global class prior estimated from train labels (a standard KL-safe smoothing).'
- What this solution (achieved 1.39794) has done: 'We fix the TensorFlow/protobuf runtime crash by pinning protobuf to the pure-Python implementation *before* importing TensorFlow and by applying a small compatibility shim that restores `MessageFactory.GetPrototype` when the installed protobuf version removed it. This is a correctness/stability fix only and keeps your model/training/inference logic unchanged. Then we keep your existing prior-blend calibration (already minimal) and ensure we always write `submission.csv` aligned to `sample_submission.csv` with probabilities summing to 1. Finally, we add a deterministic seed to reduce fold-to-fold noise without changing the core approach.'
- What this solution (achieved 1.39794) has done: 'We fix the TensorFlow/protobuf crash by applying the `MessageFactory.GetPrototype` compatibility shim *before* importing TensorFlow, and we make it robust across protobuf versions by patching both `google.protobuf.message_factory` and `google.protobuf.symbol_database` if needed. This is a stability-only change that preserves your model/training/inference logic. Then we keep your existing inference + prior-blend calibration unchanged so the score should improve only via actually running the model (instead of falling back to uniform), and we continue to guarantee a valid `submission.csv` aligned to `sample_submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.39794) has done: 'I fix the TensorFlow/protobuf crash by applying a protobuf compatibility shim that patches the *symbol_database default* MessageFactory instance (the one TensorFlow actually uses) before importing TensorFlow. This keeps your model/training/inference logic unchanged but unblocks execution in the Kaggle Py3.12 runtime. Then I ensure inference runs (instead of falling back to uniform) by keeping the model-loading path check and generator alignment intact, and still always write `submission.csv` aligned to `sample_submission.csv` with row-wise probabilities summing to 1. No architectural/training changes are introduced; the only behavior change is that the model can now actually run, which should move KL (lower is better) down toward your target.'
- What this solution (achieved 1.39794) has done: 'We fix the protobuf/TensorFlow crash by applying a safer compatibility shim that patches the *actual* default protobuf `MessageFactory` instance TensorFlow uses, covering both `MessageFactory` and the symbol database factory/pool cases before importing TensorFlow. This change is stability-only and keeps your model/training/inference logic intact. Then we keep your existing prior-blend calibration and submission alignment exactly as-is, only adding a tiny guard so the submission columns always match the sample submission order and sums stay normalized. The rest of the pipeline is unchanged, and it run end-to-end and write `submission.csv`.'
- What this solution (achieved 1.39794) has done: 'We fix the TensorFlow/protobuf crash by strengthening the protobuf `MessageFactory.GetPrototype` shim so it patches the *actual* factory instance(s) TensorFlow uses (including the symbol_database default factory), and we do it before importing TensorFlow. This is a stability-only change that unblocks execution and allows your model inference to run instead of falling back to uniform/prior-only predictions (which should reduce KL toward your target). We also make the GPU strategy selection robust when no GPU is available to avoid runtime errors on CPU-only sessions. The rest of your model, data pipeline, calibration blend, and submission formatting are kept unchanged.'
- What this solution (achieved 1.39794) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by strengthening the protobuf compatibility shim so it reliably patches the *actual* default factory instance TensorFlow uses in this Kaggle Py3.12 runtime, before importing TensorFlow. This is a stability-only change and keeps your model, data pipeline, and calibration logic intact. Then I ensure we always build a correctly-sized submission aligned to `sample_submission.csv`, with row-wise probabilities summing to 1 (unchanged behavior, but with extra safety normalization). No training/architecture changes are introduced; the main effect is that inference can run instead of crashing/falling back, which should reduce KL toward your target.'
- What this solution (achieved 1.39794) has done: 'We fix the TensorFlow import crash by strengthening the protobuf `MessageFactory.GetPrototype` compatibility shim so it patches both the class and the *actual default factory instance* used by `symbol_database.Default()` before importing TensorFlow. This is a stability-only change (no model/training logic changes) and unblocks real inference so you’re no longer falling back to uniform/prior-only predictions, which should move KL down toward your target. We also add a small safety fallback to switch `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `python` (pure-Python) if TensorFlow import still fails on this runtime. Finally, we keep your existing prior-blend calibration and submission alignment intact, ensuring a valid `submission.csv` with correct row count/ordering and row sums = 1.'
- What this solution (achieved 1.39794) has done: 'I fix the TensorFlow import crash by applying a stronger protobuf compatibility shim *before* importing TensorFlow, including patching the symbol_database default factory and forcing the pure-Python protobuf runtime when needed. This is a runtime/stability fix only and doesn’t change your model architecture or training/inference logic. Then I keep your existing inference + prior-blend calibration, but ensure the generator always has the columns it expects and that predictions are always produced for every `eeg_id` in `sample_submission.csv` in the exact order required. The submission be written as `submission.csv` with the correct header, row count, and row-wise probability sums of 1.'
- What this solution (achieved 1.39794) has done: 'Your run is currently blocked at TensorFlow import due to a protobuf API mismatch (`MessageFactory.GetPrototype` missing), so the model never reaches inference reliably and the submission falls back toward uniform/prior-only predictions (high KL). I strengthen the protobuf compatibility shim so it patches both the `MessageFactory` class and the *actual default factory instances* (including those created later), and as a safety net force pure-Python protobuf and retry TF import once. Then I keep your existing model/training/inference logic unchanged, only ensuring the pipeline proceeds to inference and always writes a valid `submission.csv` aligned to `sample_submission.csv` with probabilities summing to 1. This should legitimately reduce KL toward your target by enabling real model predictions instead of fallback behavior.'
- What this solution (achieved 1.39794) has done: 'The immediate blocker is the TensorFlow import crash caused by protobuf’s missing `MessageFactory.GetPrototype`; your shim doesn’t reliably patch the *actual* factory instance used during TF import in this environment. I make the protobuf shim stronger by patching both `MessageFactory.GetPrototype` and `GetMessageClass` across the class and default instances (including the symbol_database default factory), and I apply it *and* set the pure-Python protobuf env vars before any TensorFlow-related import. This is a runtime-only fix (no model/training changes) and should allow real model inference to run instead of falling back, which should reduce KL toward your target. I also keep your prior-blend calibration intact and keep submission alignment to `sample_submission.csv` unchanged to ensure a valid `submission.csv`.'
- What this solution (achieved 1.39794) has done: 'I fix the TensorFlow/protobuf crash by applying a more robust protobuf shim *before* importing TensorFlow, patching both the `MessageFactory` class and the actual default factory instance used by `symbol_database.Default()`. This is a runtime/stability fix only and keeps your model/training/inference logic unchanged, but it should allow real inference to run instead of falling back to uniform/prior-only predictions (which is likely why KL is stuck high). I also keep your existing prior-blend calibration and submission alignment, only adding a tiny safety normalization/NaN guard to guarantee valid probabilities. The script then run end-to-end and write `submission.csv` with the exact required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

PLATFORM = "kaggle"  # local 平台 或 kaggle 平台
NEEDTRAIN = False  # 是否需要训练，如果线上 infer 则不需要
READ_SPEC_FILES = False  # 是否需要预处理谱图
READ_EEG_FILES = False  # 是否需要预处理脑电数据
LOAD_MODELS_FROM = "models20240131"  # 训练好的模型保存位置，调用直接 infer
if PLATFORM == "local":  # 模型加载路径
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # 脑电样本使用的时间长度
SFREQ = 100  # 脑电样本重采样率
HIGH = 512  # 谱图频率长度
LENGTH = 256  # 谱图时间长度

CONVERTIMAGE = False
CONVERTIMAGE_EEG = False
if CONVERTIMAGE or CONVERTIMAGE_EEG:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvas

filter_range = [0.5, 40]  # 脑电滤波范围
BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],  # 重参考
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

VER = 1  # 版本号


def _add_getprototype_to_factory(factory_obj):
    """Patch a protobuf factory-like instance to provide GetPrototype if missing."""
    if factory_obj is None:
        return
    try:
        if (not hasattr(factory_obj, "GetPrototype")) and hasattr(
            factory_obj, "GetMessageClass"
        ):

            def _GetPrototype(descriptor, _f=factory_obj):
                return _f.GetMessageClass(descriptor)

            setattr(factory_obj, "GetPrototype", _GetPrototype)
    except Exception:
        pass


def _add_getmessageclass_to_factory(factory_obj):
    """Patch a protobuf factory-like instance to provide GetMessageClass if missing."""
    if factory_obj is None:
        return
    try:
        if (not hasattr(factory_obj, "GetMessageClass")) and hasattr(
            factory_obj, "GetPrototype"
        ):

            def _GetMessageClass(descriptor, _f=factory_obj):
                return _f.GetPrototype(descriptor)

            setattr(factory_obj, "GetMessageClass", _GetMessageClass)
    except Exception:
        pass


def _patch_protobuf_for_tf():
    """
    Patch protobuf to restore MessageFactory.GetPrototype expected by some TF builds.
    We patch both:
      - the MessageFactory class methods
      - and the actual default factory instance used by symbol_database.Default()
    """
    try:
        from google.protobuf import message_factory as _message_factory
        from google.protobuf import symbol_database as _symbol_database
        from google.protobuf import descriptor_pool as _descriptor_pool

        MF = getattr(_message_factory, "MessageFactory", None)
        if MF is not None:
            try:
                if (not hasattr(MF, "GetPrototype")) and hasattr(MF, "GetMessageClass"):

                    def _GetPrototype_cls(self, descriptor):
                        return self.GetMessageClass(descriptor)

                    setattr(MF, "GetPrototype", _GetPrototype_cls)
            except Exception:
                pass

            try:
                if (not hasattr(MF, "GetMessageClass")) and hasattr(MF, "GetPrototype"):

                    def _GetMessageClass_cls(self, descriptor):
                        return self.GetPrototype(descriptor)

                    setattr(MF, "GetMessageClass", _GetMessageClass_cls)
            except Exception:
                pass

            try:
                _orig_init = MF.__init__

                def _patched_init(self, *args, **kwargs):
                    _orig_init(self, *args, **kwargs)
                    _add_getprototype_to_factory(self)
                    _add_getmessageclass_to_factory(self)

                if getattr(MF.__init__, "_tfshim_patched", False) is False:
                    _patched_init._tfshim_patched = True
                    MF.__init__ = _patched_init
            except Exception:
                pass

        try:
            if hasattr(_message_factory, "default_factory"):
                _add_getprototype_to_factory(
                    getattr(_message_factory, "default_factory")
                )
                _add_getmessageclass_to_factory(
                    getattr(_message_factory, "default_factory")
                )
        except Exception:
            pass

        try:
            _sym_db = _symbol_database.Default()
            for attr in ("_factory", "factory", "_message_factory"):
                if hasattr(_sym_db, attr):
                    fobj = getattr(_sym_db, attr)
                    _add_getprototype_to_factory(fobj)
                    _add_getmessageclass_to_factory(fobj)
            _pool = getattr(_sym_db, "pool", None)
            if _pool is not None:
                for attr in ("_factory", "factory", "_message_factory"):
                    if hasattr(_pool, attr):
                        fobj = getattr(_pool, attr)
                        _add_getprototype_to_factory(fobj)
                        _add_getmessageclass_to_factory(fobj)
        except Exception:
            pass

        try:
            _pool = _descriptor_pool.Default()
            for attr in ("_factory", "factory", "_message_factory"):
                if hasattr(_pool, attr):
                    fobj = getattr(_pool, attr)
                    _add_getprototype_to_factory(fobj)
                    _add_getmessageclass_to_factory(fobj)
        except Exception:
            pass

    except Exception:
        pass


_patch_protobuf_for_tf()



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import random

try:
    import tensorflow as tf
except Exception:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    _patch_protobuf_for_tf()
    import tensorflow as tf  # noqa: F401

from tensorflow.keras import mixed_precision
from scipy import signal

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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

MIX = False
if MIX:
    try:
        mixed_precision.set_global_policy("mixed_float16")
        print(
            "Mixed precision enabled via mixed_precision policy:",
            mixed_precision.global_policy(),
        )
    except Exception as e:
        print(
            "Mixed precision request failed; continuing with default precision:",
            repr(e),
        )
else:
    mixed_precision.set_global_policy("float32")
    print("Using full precision")



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
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## === cell 4
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")
    if READ_EEG_FILES:
        eegs = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            if len(train[train.eeg_id == name]) > 0:
                eeg_default = raw_eeg.loc[:, :].reset_index(drop=True)
                list_eeg = list()
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
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))
                list_eeg = np.concatenate(list_eeg, 0)
                eegs[name] = list_eeg
        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()



## === cell 5
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}
b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


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
    ):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.mode = mode
        self.specs = specs if specs is not None else {}
        self.eegs = eegs if eegs is not None else {}
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if CONVERTIMAGE:
            X = np.zeros((len(indexes), HIGH, LENGTH, 3), dtype="float32")
        else:
            X = np.zeros((len(indexes), HIGH, LENGTH), dtype="float32")

        if CONVERTIMAGE_EEG:
            X_eeg = np.zeros((len(indexes), HIGH, LENGTH, 3), dtype="float32")
        else:
            X_eeg = np.zeros(
                (len(indexes), 16, round(EEG_LENGTH * SFREQ)), dtype="float32"
            )

        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            img = None
            eeg = None

            if self.mode == "test":
                spec_start = 0
                eeg_start = 0
            else:
                rows = df[df.eeg_id == row.eeg_id].reset_index(drop=True)
                rows = rows.sort_values(
                    by="spectrogram_label_offset_seconds", ascending=True
                ).reset_index(drop=True)
                row_mid = rows.iloc[max(round((len(rows) + 1) / 2) - 1, 0)]
                spec_start = round(row_mid.spectrogram_label_offset_seconds / 2)
                eeg_start = (
                    round(row_mid.eeg_label_offset_seconds) + (50 - EEG_LENGTH) / 2
                )
                row = row_mid

            if row.spectrogram_id in self.specs:
                base = self.specs[row.spectrogram_id]
                img_raw = base[spec_start : (spec_start + 300), :]
                img_raw = np.nan_to_num(img_raw, nan=0.0)

                LL = img_raw[:, :100]
                RL = img_raw[:, 100:200]
                LP = img_raw[:, 200:300]
                RP = img_raw[:, 300:]

                LL = np.log(np.clip(LL, np.exp(-6), np.exp(8)))
                RL = np.log(np.clip(RL, np.exp(-6), np.exp(8)))
                LP = np.log(np.clip(LP, np.exp(-6), np.exp(8)))
                RP = np.log(np.clip(RP, np.exp(-6), np.exp(8)))

                if CONVERTIMAGE:
                    img_tmp = np.concatenate((LL, LP, RP, RL), 1)
                    img_tmp = np.nan_to_num(img_tmp, nan=0.0)
                    plt.figure()
                    img_tmp = (
                        plt.imshow(img_tmp)
                        .get_figure()
                        .gca()
                        .images[0]
                        .make_image(renderer=None)[0][:, :, :3]
                    )
                    img_tmp = np.array(
                        tf.image.resize(img_tmp, (HIGH, LENGTH)), dtype=np.uint8
                    )
                    img_tmp = img_tmp / 255.0
                    plt.close()
                    gc.collect()
                    img = img_tmp
                else:
                    img2 = np.zeros((HIGH, LENGTH), dtype="float32")
                    resize_temp = 96
                    LLr = np.array(
                        tf.image.resize(
                            np.reshape(LL, (LL.shape[0], LL.shape[1], 1)),
                            (resize_temp, LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    img2[
                        round(HIGH / 4 * 0 + (HIGH / 4 - resize_temp) / 2) : round(
                            HIGH / 4 * 0 + (HIGH / 4 + resize_temp) / 2
                        ),
                        :,
                    ] = LLr
                    RLr = np.array(
                        tf.image.resize(
                            np.reshape(RL, (RL.shape[0], RL.shape[1], 1)),
                            (resize_temp, LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    img2[
                        round(HIGH / 4 * 1 + (HIGH / 4 - resize_temp) / 2) : round(
                            HIGH / 4 * 1 + (HIGH / 4 + resize_temp) / 2
                        ),
                        :,
                    ] = RLr
                    LPr = np.array(
                        tf.image.resize(
                            np.reshape(LP, (LP.shape[0], LP.shape[1], 1)),
                            (resize_temp, LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    img2[
                        round(HIGH / 4 * 2 + (HIGH / 4 - resize_temp) / 2) : round(
                            HIGH / 4 * 2 + (HIGH / 4 + resize_temp) / 2
                        ),
                        :,
                    ] = LPr
                    RPr = np.array(
                        tf.image.resize(
                            np.reshape(RP, (RP.shape[0], RP.shape[1], 1)),
                            (resize_temp, LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    img2[
                        round(HIGH / 4 * 3 + (HIGH / 4 - resize_temp) / 2) : round(
                            HIGH / 4 * 3 + (HIGH / 4 + resize_temp) / 2
                        ),
                        :,
                    ] = RPr
                    img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-6)
                    img = img2

            if row.eeg_id in self.eegs and not CONVERTIMAGE_EEG:
                eeg_raw = self.eegs[row.eeg_id][
                    :,
                    round(eeg_start * SFREQ) : round((eeg_start + EEG_LENGTH) * SFREQ),
                ][:, :, 0]
                eeg_raw = signal.filtfilt(b, a, eeg_raw, axis=1)
                eeg = (eeg_raw - np.mean(eeg_raw, 1, keepdims=True)) / (
                    np.std(eeg_raw, 1, keepdims=True) + 1e-6
                )
            elif row.eeg_id in self.eegs and CONVERTIMAGE_EEG:
                eeg_raw = self.eegs[row.eeg_id][
                    :,
                    round(eeg_start * SFREQ) : round((eeg_start + EEG_LENGTH) * SFREQ),
                ][:, :, 0]
                eeg_raw = signal.filtfilt(b, a, eeg_raw, axis=1)
                fig, ax = plt.subplots()
                for ii in range(eeg_raw.shape[0]):
                    plt.plot(eeg_raw[ii, :] + ii * 100, color="black", linewidth=1)
                from matplotlib.backends.backend_agg import FigureCanvas

                canvas = FigureCanvas(fig)
                plt.xlim([0, eeg_raw.shape[1]])
                plt.ylim([0 - 25, ii * 100 + 25])
                ax.set_aspect("equal", adjustable="box")
                plt.axis("off")
                canvas.draw()
                eeg_img = np.array(canvas.renderer.buffer_rgba())[:, :, :3]
                plt.close()
                gc.collect()
                eeg_img = np.array(
                    tf.image.resize(eeg_img, (HIGH, LENGTH)), dtype=np.uint8
                )
                eeg_img = eeg_img / 255.0
                eeg = eeg_img

            if img is None:
                img = (
                    np.zeros((HIGH, LENGTH, 3), dtype="float32")
                    if CONVERTIMAGE
                    else np.zeros((HIGH, LENGTH), dtype="float32")
                )
            if eeg is None:
                eeg = (
                    np.zeros((HIGH, LENGTH, 3), dtype="float32")
                    if CONVERTIMAGE_EEG
                    else np.zeros((16, round(EEG_LENGTH * SFREQ)), dtype="float32")
                )

            X[j] = img
            X_eeg[j] = eeg

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)

        return X, X_eeg, y




## === cell 6
if NEEDTRAIN:
    LR_START = 1e-4
    LR_MAX = 1e-3
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 0
    LR_STEP_DECAY = 0.1
    EVERY = 2
    EPOCHS = 6

    def lrfn(epoch):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = LR_MAX * LR_STEP_DECAY ** (
                (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) // EVERY
            )
        return max(lr, 1e-4)

    rng = [i for i in range(EPOCHS)]
    y = [lrfn(x) for x in rng]
    plt.figure(figsize=(10, 4))
    plt.plot(rng, y, "o-")
    plt.xlabel("epoch", size=14)
    plt.ylabel("learning rate", size=14)
    plt.title("Step Training Schedule", size=16)
    plt.show()
    LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)




## === cell 7
def build_model(CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ):
    if CONVERTIMAGE:
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 3))
        x = inp
    else:
        inp = tf.keras.Input(shape=(HIGH, LENGTH))
        x = tf.keras.layers.Reshape((inp.shape[1], inp.shape[2], 1))(inp)
        x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False, weights=None, input_shape=(HIGH, LENGTH, 3)
    )
    base_model._name = "spectrogram_extractor"
    if PLATFORM == "local":
        base_model.load_weights(
            "./input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
        )
    if PLATFORM == "kaggle":
        base_model.load_weights(
            "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
        )

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if CONVERTIMAGE_EEG:
        inp_eeg = tf.keras.Input(shape=(HIGH, LENGTH, 3))
        x_eeg = inp_eeg
        eeg_input_shape = (HIGH, LENGTH, 3)
    else:
        inp_eeg = tf.keras.Input(shape=(16, round(EEG_LENGTH * SFREQ)))
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])
        eeg_input_shape = (16, round(EEG_LENGTH * SFREQ), 3)

    base_model_eeg = tf.keras.applications.EfficientNetB1(
        include_top=False, weights=None, input_shape=eeg_input_shape
    )
    base_model_eeg._name = "eeg_extractor"
    if PLATFORM == "local":
        base_model_eeg.load_weights(
            "./input/tf-efficientnet-imagenet-weights/efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
        )
    if PLATFORM == "kaggle":
        base_model_eeg.load_weights(
            "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
        )

    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)

    x = tf.nn.l2_normalize(x, -1)
    x_eeg = tf.nn.l2_normalize(x_eeg, -1)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 8
if NEEDTRAIN:
    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K

    all_oof = []
    all_true = []

    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        train_gen = DataGenerator(
            train.iloc[train_index],
            shuffle=True,
            batch_size=16,
            specs=spectrograms,
            eegs=eegs,
        )
        valid_gen = DataGenerator(
            train.iloc[valid_index],
            shuffle=False,
            batch_size=32,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
        )

        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        K.clear_session()

        callbacks_list = [
            LR,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"EB2_v{VER}_f{i}.h5",
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5, monitor="val_loss", mode="min"
            ),
        ]

        with strategy.scope():
            model = build_model(
                CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ
            )
        model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )

        model.load_weights(f"EB2_v{VER}_f{i}.h5")

        oof = model.predict(valid_gen, verbose=1)
        all_oof.append(oof)
        all_true.append(train.iloc[valid_index][TARGETS].values)

        del model, oof
        K.clear_session()
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)



## === cell 9
if NEEDTRAIN:
    import sys

    if PLATFORM == "local":
        sys.path.append("./input/kaggle-kl-div")
    elif PLATFORM == "kaggle":
        sys.path.append("/kaggle/input/kaggle-kl-div")
    from kaggle_kl_div import score

    oof = pd.DataFrame(all_oof.copy())
    oof["id"] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true["id"] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name="id")
    print("CV Score KL-Div for EfficientNetB2 =", cv)



## === cell 10
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
    print("Sample submission shape", sample_sub.shape)

    sub_eeg_ids = sample_sub["eeg_id"].values
    test = test.drop_duplicates(subset=["eeg_id"]).copy()

    if PLATFORM == "local":
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
    elif PLATFORM == "kaggle":
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    files_spec = os.listdir(PATH_SPEC)
    print(f"There are {len(files_spec)} test spectrogram parquets")
    spectrograms2 = {}
    for i, f in enumerate(files_spec):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    if PLATFORM == "local":
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files_eeg = os.listdir(PATH_EEG)
    print(f"There are {len(files_eeg)} eeg parquets")
    eegs2 = {}
    test_eeg_set = set(sub_eeg_ids.tolist())
    for i, f in enumerate(files_eeg):
        if i % 100 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        if name not in test_eeg_set:
            continue
        raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")
        eeg_default = raw_eeg.loc[:, :].reset_index(drop=True)
        list_eeg = []
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
            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))
        list_eeg = np.concatenate(list_eeg, 0)
        eegs2[name] = list_eeg
    print()

    eps = 1e-7
    uniform = np.full(
        (len(sample_sub), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
    )

    prior = train[TARGETS].mean(axis=0).values.astype(np.float32)
    prior = np.clip(prior, eps, 1.0)
    prior = prior / prior.sum()
    prior_mat = np.tile(prior.reshape(1, -1), (len(sample_sub), 1))

    preds = []
    can_predict = True
    if not os.path.isdir(LOAD_MODELS_FROM):
        print(
            f"WARNING: model folder not found: {LOAD_MODELS_FROM}. Will output prior-smoothed predictions."
        )
        can_predict = False

    if can_predict:
        try:
            with strategy.scope():
                model = build_model(
                    CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ
                )

            test_for_sub = pd.DataFrame({"eeg_id": sub_eeg_ids}).merge(
                test, on="eeg_id", how="left"
            )
            if "spectrogram_id" not in test_for_sub.columns:
                test_for_sub["spectrogram_id"] = -1

            test_gen = DataGenerator(
                test_for_sub,
                shuffle=False,
                batch_size=32,
                mode="test",
                specs=spectrograms2,
                eegs=eegs2,
            )

            for i in range(5):
                wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
                if not os.path.exists(wpath):
                    raise FileNotFoundError(f"Missing weights file: {wpath}")
                print(f"Fold {i + 1}")
                model.load_weights(wpath)
                pred_i = model.predict(test_gen, verbose=1)
                preds.append(pred_i)

            pred = np.mean(preds, axis=0)
            pred = np.nan_to_num(pred, nan=1.0 / len(TARGETS), posinf=1.0, neginf=0.0)
            pred = np.clip(pred, eps, 1.0)
            pred = pred / pred.sum(axis=1, keepdims=True)
        except Exception as e:
            print(
                "WARNING: inference failed; falling back to prior-smoothed uniform predictions. Error:",
                repr(e),
            )
            pred = uniform
    else:
        pred = uniform

    alpha = 0.20
    pred = (1.0 - alpha) * pred + alpha * prior_mat
    pred = np.nan_to_num(pred, nan=1.0 / len(TARGETS), posinf=1.0, neginf=0.0)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = sample_sub[["eeg_id"]].copy()
    sub[TARGETS] = pred.astype(np.float32)
    sub[TARGETS] = np.nan_to_num(
        sub[TARGETS].values, nan=1.0 / len(TARGETS), posinf=1.0, neginf=0.0
    )
    sub[TARGETS] = np.clip(sub[TARGETS].values, eps, 1.0)
    sub[TARGETS] = sub[TARGETS].values / sub[TARGETS].values.sum(axis=1, keepdims=True)

    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print("Submission shape", sub.shape)
    print(
        "Row sums (min/max):",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).max(),
    )
    sub.head()
