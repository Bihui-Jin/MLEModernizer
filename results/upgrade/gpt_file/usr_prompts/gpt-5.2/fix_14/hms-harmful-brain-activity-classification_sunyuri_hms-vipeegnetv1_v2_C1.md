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

0.5190321943945365

# 6. Current score

0.79458

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow import crash by removing the `reset_default_graph` import (it triggers a protobuf `MessageFactory` error in this environment) and keeping TF initialization otherwise unchanged. Then I make the weights-loading robust: if the expected `/kaggle/input/models20240117/*.h5` files aren’t present, the script fall back to a valid, properly-normalized prior built from the train label distribution (so you always get a valid `submission.csv`). This keeps core model/data logic intact when weights exist, but prevents runtime failure and guarantees a valid submission file with row-wise probabilities summing to 1. Finally, I add a couple of small safety guards (missing EEG/spec entries) to avoid KeyErrors during test generation.'
- What this solution (achieved 1.1728) has done: 'I fix the TensorFlow import crash causing the `MessageFactory.GetPrototype` error by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a common Kaggle Py3.12/TensorFlow/protobuf incompatibility). I keep the existing model/data pipeline unchanged, but add a safe no-GPU fallback for the distribution strategy so it also runs on CPU-only environments. To improve score toward the target (lower is better) when model weights are missing, I replace the “global prior” fallback with a more informative patient-conditional prior (computed from train) and then fall back to global prior only for unseen patients; this stays metric-consistent and should reduce KL vs a flat prior-like guess. Submission writing remains the same and still guarantees row-wise probabilities sum to 1.'
- What this solution (achieved 1.1728) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `TF_USE_CXX11_ABI=0`, and by importing TensorFlow only after those env vars are set. This is the minimal change that unblocks execution end-to-end in Kaggle’s Py3.12 environment where `MessageFactory.GetPrototype` can fail at import time. I also make the spectrogram loading robust to missing/extra files and keep memory use stable by only loading spectrograms actually referenced in `test.csv` (core logic unchanged: same arrays fed to the same generator/model). Finally, submission writing stays identical but with an explicit column order and probability normalization guards to ensure a valid `.csv` every run.'
- What this solution (achieved 1.1728) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend earlier and more completely (including disabling the upb C++ implementation), which is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle Py3.12 environment. I also make GPU selection safer by not hard-forcing `CUDA_VISIBLE_DEVICES` (which can break if only one GPU or none is available) while keeping the same distribution strategy logic. Finally, I keep the model/data pipeline unchanged, ensure predictions are always valid probabilities (clipped + renormalized), and always write a correctly-formatted `submission.csv`.'
- What this solution (achieved 1.1728) has done: 'I fix the TensorFlow/protobuf import crash by setting the required environment variables *before any TensorFlow-related imports* and by importing `google.protobuf` early to ensure the pure-Python backend is used, which directly addresses the `MessageFactory.GetPrototype` error. I also add a safe fallback if TensorFlow still fails to import: the script then generate a valid, properly-normalized submission using the already-implemented patient/global prior, so it always completes end-to-end. Additionally, I ensure the output column order matches `sample_submission.csv` exactly and keep the probability clipping/renormalization to avoid submission failures. These changes are execution/stability fixes and keep the modeling/data logic unchanged when TensorFlow and weights are available (score improvement then comes from actually running the model instead of falling back).'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend more robustly (and doing it before any TF-related import), then adding a safe fallback path if TF still cannot import so the notebook always runs end-to-end. To move the score down toward the target (lower is better) when weights aren’t available, I keep your existing patient/global-prior fallback but make it closer to the expected label distribution by using a small Dirichlet/Laplace smoothing over patient vote counts (reduces overconfident zeros and usually lowers KL). I also add a lightweight check for the model-weight directory and keep submission formatting/normalization strict so the CSV is always valid. Core model, generator, and inference logic remain unchanged when TF + weights are available.'
- What this solution (achieved 0.79458) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime *more completely and earlier* (including `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and disabling the C++/upb backend before any TF/protobuf import). If TensorFlow still can’t import, the code cleanly fall back to the patient/global prior path and still write a valid `submission.csv`. To improve the score toward your target (lower KL), I keep the same prior-based fallback core logic but make it slightly more informative by computing patient priors on the consolidated per-`eeg_id` targets (not the overlapping raw rows) and using a smaller smoothing alpha to better match the empirical distribution. I also add a small guard to ensure predictions are aligned to the submission’s `eeg_id` order and strictly normalized.'
- What this solution (achieved 0.79458) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing a compatible protobuf runtime *and* preloading TensorFlow’s bundled protobuf, then importing TF only after that; this keeps your model/training/inference logic unchanged and prevents the pipeline from falling back to priors unnecessarily. I also make the TF-availability detection more robust so the script always completes end-to-end and writes `submission.csv` even if TF still can’t load. Finally, I keep your existing probability clipping/renormalization (metric-consistent for KL and required for submission validity) and leave the patient/global prior fallback intact as the safety net.'
- What this solution (achieved 0.79458) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the model path from running by forcing the pure-Python protobuf backend *and* disabling the upb C++ implementation before any TensorFlow/protobuf modules are imported. This should allow TensorFlow to import cleanly in the Kaggle Py3.12 environment, so the script can actually load the provided fold weights (when present) instead of falling back to priors—this is the minimal change likely to move KL down toward your target. I also add a small deterministic seeding and keep the existing fallback prior path unchanged, while ensuring the submission is always aligned to `sample_submission.csv` column order and strictly normalized to sum to 1.'
- What this solution (achieved 0.79458) has done: 'I fix the TensorFlow/protobuf import crash by switching to a safe import strategy: avoid the problematic `MessageFactory.GetPrototype` path by forcing the Python protobuf backend early and, if TensorFlow still fails to import, cleanly fall back without error. I also make mixed precision enabling compatible with modern TF (use `mixed_precision.set_global_policy` when available) while keeping your model and inference logic unchanged. Finally, I ensure the fallback prior is always computed from the consolidated per-`eeg_id` targets (as you intended), and I keep strict clipping+renormalization so the submission is always valid and sums to 1 per row.'
- What this solution (achieved 0.79458) has done: 'I fix the TensorFlow/protobuf import crash that prevents the model from running (and forces the weaker prior fallback) by forcing the pure-Python protobuf implementation earlier and more reliably, and by sanitizing any preloaded protobuf modules before importing TensorFlow. This is an execution fix (not a modeling change) and should improve the score toward your target because it enables actual model inference with fold weights when present. I also add a safe fallback for the missing SciPy dependency (it’s used but not guaranteed installed) so the script always completes end-to-end and still produces a valid, normalized `submission.csv`. Finally, I keep submission ordering aligned to `sample_submission.csv` and keep strict clipping+renormalization to satisfy the competition’s probability-sum constraint.'
- What this solution (achieved 0.79458) has done: 'I fix the TensorFlow/protobuf import crash that’s preventing your model path from running (and forcing the weaker prior fallback), by enforcing a stable protobuf runtime *before* any TF import and disabling TF’s use of the C++ protobuf backend. I also add a robust “try TF import” function so the script cleanly falls back without stopping if TF still can’t load, while keeping your model/training/inference code unchanged. Finally, I add small, score-neutral safety guards: ensure `min/max` offsets are integers for indexing, ensure `r` is always within spectrogram bounds, and keep strict probability clipping+renormalization and exact sample_submission column order so the CSV is always valid.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_UPB"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_USE_CXX11_ABI", "0")

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

try:
    import google.protobuf  # noqa: F401
except Exception as _e:
    print("Warning: could not import google.protobuf early:", repr(_e))

import numpy as np
import pandas as pd

np.random.seed(0)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240117"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128
LENGTH = 256
BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}


def _try_import_tensorflow():
    """
    Fix: isolate TF import so we can safely fall back without crashing the whole run.
    Keep core model logic unchanged when TF imports successfully.
    """
    try:
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                sys.modules.pop(m, None)
        import google.protobuf  # noqa: F401

        import tensorflow as tf  # noqa: F401

        return tf, None
    except Exception as e:
        return None, e


TF_AVAILABLE = True
tf, tf_err = _try_import_tensorflow()
if tf is None:
    TF_AVAILABLE = False
    strategy = None
    VER = 1
    print(
        "TensorFlow import failed; will use prior fallback only. Reason:", repr(tf_err)
    )
else:
    try:
        try:
            tf.keras.utils.set_random_seed(0)
        except Exception:
            pass

        print("TensorFlow version =", tf.__version__)

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            print("Using CPU")
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using {len(gpus)} GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")

        VER = 1

        MIX = True
        if MIX:
            try:
                from tensorflow.keras import mixed_precision

                mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled via set_global_policy")
            except Exception as e:
                try:
                    tf.config.optimizer.set_experimental_options(
                        {"auto_mixed_precision": True}
                    )
                    print("Mixed precision enabled via experimental optimizer option")
                except Exception as e2:
                    print(
                        "Mixed precision requested but could not be enabled:",
                        repr(e),
                        repr(e2),
                    )
        else:
            print("Using full precision")

    except Exception as e:
        TF_AVAILABLE = False
        tf = None
        strategy = None
        VER = 1
        print("TensorFlow setup failed; will use prior fallback only. Reason:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

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

for c in ["min", "max"]:
    train[c] = pd.to_numeric(train[c], errors="coerce").fillna(0).astype(int)

_global_prior = train[list(TARGETS)].sum().values.astype("float64")
_global_prior = _global_prior / _global_prior.sum()
print("Global label prior (per-eeg_id):", dict(zip(TARGETS, _global_prior.round(6))))

_patient_counts = train.groupby("patient_id")[list(TARGETS)].sum().astype("float64")
alpha = 0.3
_patient_prior = (
    (_patient_counts + alpha)
    .div((_patient_counts + alpha).sum(axis=1), axis=0)
    .fillna(0.0)
)
print("Patient priors computed for patients:", _patient_prior.shape[0])



## === cell 2
try:
    import albumentations as albu
except Exception as e:
    albu = None
    print("albumentations not available; augmentation disabled. Reason:", repr(e))

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
        ):
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = augment and (albu is not None)
            self.mode = mode
            self.specs = specs if specs is not None else {}
            self.eegs = eegs if eegs is not None else {}
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.data) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            X, X_eeg, y = self.__data_generation(indexes)
            if self.augment:
                X = self.__augment_batch(X)
            return [X, X_eeg], y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
            X_eeg = np.zeros(
                (len(indexes), 4, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
            y = np.zeros((len(indexes), 6), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                elif self.mode == "valid":
                    r = int((int(row["min"]) + int(row["max"])) // 4)
                else:
                    r = np.random.randint(int(row["min"]), int(row["max"]) + 1) // 2

                spec_arr = self.specs.get(int(row.spec_id), None)
                eeg_arr = self.eegs.get(int(row.eeg_id), None)

                for k in range(4):
                    if spec_arr is not None:
                        max_r = max(0, spec_arr.shape[0] - 300)
                        rr = int(np.clip(r, 0, max_r))
                        img = spec_arr[rr : rr + 300, k * 100 : (k + 1) * 100].T
                    else:
                        img = np.ones((100, 300), dtype="float32")

                    if eeg_arr is not None:
                        img_eeg = eeg_arr[:, :, k]
                    else:
                        img_eeg = np.ones(
                            (4, round(EEG_LENGTH * SFREQ)), dtype="float32"
                        )

                    img = np.clip(img, np.exp(-4), np.exp(8))
                    img = np.log(img)
                    img = np.nan_to_num(img, nan=0.0)

                    if HIGH <= 100:
                        img = np.resize(
                            img[
                                :,
                                round((600 / 2 - LENGTH) / 2) : (
                                    round((600 / 2 - LENGTH) / 2) + LENGTH
                                ),
                            ],
                            (HIGH, LENGTH),
                        )
                    else:
                        img = img[
                            :,
                            round((600 / 2 - LENGTH) / 2) : (
                                round((600 / 2 - LENGTH) / 2) + LENGTH
                            ),
                        ]

                    ep = 1e-6
                    m = np.nanmean(img.flatten())
                    s = np.nanstd(img.flatten())
                    img = (img - m) / (s + ep)
                    img = np.nan_to_num(img, nan=0.0)

                    if HIGH <= 100:
                        X[j, :, :, k] = img
                    else:
                        X[
                            j, round((HIGH - 100) / 2) : -round((HIGH - 100) / 2), :, k
                        ] = img

                    X_eeg[j, :, :, k] = img_eeg

                if self.mode != "test":
                    y[j] = row[TARGETS].values

            return X, X_eeg, y

        def __random_transform(self, img):
            composition = albu.Compose(
                [
                    albu.HorizontalFlip(p=0.5),
                    albu.CoarseDropout(
                        max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                    ),
                ]
            )
            return composition(image=img)["image"]

        def __augment_batch(self, img_batch):
            for i in range(img_batch.shape[0]):
                img_batch[i,] = self.__random_transform(img_batch[i,])
            return img_batch




## === cell 3
if TF_AVAILABLE:
    from tensorflow.keras import layers

    def build_model():
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
        inp_eeg = tf.keras.Input(shape=(4, round(EEG_LENGTH * SFREQ), 4))

        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False, weights="imagenet"
        )
        base_model_eeg = tf.keras.applications.EfficientNetB1(
            include_top=False, weights="imagenet"
        )

        x0 = inp[:, :, :, :1]
        x1 = inp[:, :, :, 1:2]
        x2 = inp[:, :, :, 2:3]
        x3 = inp[:, :, :, 3:4]
        x = layers.Concatenate(axis=1)([x0, x1, x2, x3])
        x = layers.Concatenate(axis=3)([x, x, x])

        x0_eeg = inp_eeg[:, :, :, :1]
        x1_eeg = inp_eeg[:, :, :, 1:2]
        x2_eeg = inp_eeg[:, :, :, 2:3]
        x3_eeg = inp_eeg[:, :, :, 3:4]
        x_eeg = layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
        x_eeg = layers.Reshape((-1, 256, 1))(x_eeg)
        x_eeg = layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = layers.GlobalAveragePooling2D()(x_eeg)

        x = layers.Concatenate(axis=1)([x, x_eeg])
        x = layers.Dense(6, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.CategoricalCrossentropy()
        model.compile(loss=loss, optimizer=opt)
        return model




## === cell 4
if PLATFORM == "local":
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
elif PLATFORM == "kaggle":
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
print("Test shape", test.shape)

if PLATFORM == "local":
    PATH_SPEC = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
elif PLATFORM == "kaggle":
    PATH_SPEC = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
needed_spec_ids = set(test.spec_id.astype(int).tolist())

files_spec = os.listdir(PATH_SPEC)
print(f"There are {len(files_spec)} test spectrogram parquets")

spectrograms2 = {}
loaded = 0
for i, f in enumerate(files_spec):
    if i % 200 == 0:
        print(i, ", ", end="")
    try:
        name = int(f.split(".")[0])
    except Exception:
        continue
    if name not in needed_spec_ids:
        continue
    tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
    spectrograms2[name] = tmp.iloc[:, 1:].values
    loaded += 1

print(f"\nLoaded spectrograms: {loaded}/{len(needed_spec_ids)} needed")



## === cell 5
try:
    from scipy import signal  # type: ignore

    SCIPY_AVAILABLE = True
except Exception as e:
    SCIPY_AVAILABLE = False
    signal = None
    print(
        "scipy not available; EEG filtering will fall back to no-filter path. Reason:",
        repr(e),
    )

if PLATFORM == "local":
    PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
elif PLATFORM == "kaggle":
    PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

files_eeg = os.listdir(PATH_EEG)
print(f"There are {len(files_eeg)} test eeg parquets")

eegs2 = {}
if SCIPY_AVAILABLE:
    b, a = signal.butter(3, np.float32([1, 40]) * 2 / SFREQ, "bandpass")

test_eeg_ids = set(test.eeg_id.astype(int).tolist())

for i, f in enumerate(files_eeg):
    if i % 100 == 0:
        print(i, ", ", end="")
    try:
        name = int(f.split(".")[0])
    except Exception:
        continue
    if name not in test_eeg_ids:
        continue

    raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)

    list_eeg = []
    for region in BRAIN.keys():
        eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(BRAIN[region]):
            eeg[chan_i, :] = (
                eeg_default.loc[:, chan.split("-")[0]]
                - eeg_default.loc[:, chan.split("-")[1]]
            ).values

        eeg[np.isnan(eeg)] = 0

        if SCIPY_AVAILABLE:
            if 200 != SFREQ:
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
            eeg = signal.filtfilt(b, a, eeg, axis=1)

        eeg = (eeg - np.mean(eeg, 1, keepdims=True)) / (
            np.std(eeg, 1, keepdims=True) + 1e-6
        )
        list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

    list_eeg = np.concatenate(list_eeg, 2)
    eegs2[name] = list_eeg

print("\nLoaded EEGs:", len(eegs2), "Loaded specs:", len(spectrograms2))



## === cell 6
SUB_COLS = ["eeg_id"] + list(TARGETS)

preds = []
found_any = False

if not os.path.isdir(LOAD_MODELS_FROM):
    print(f"Model directory not found: {LOAD_MODELS_FROM}")
    weights_dir_exists = False
else:
    weights_dir_exists = True

if TF_AVAILABLE and weights_dir_exists:
    test_gen = DataGenerator(
        test, shuffle=False, batch_size=32, mode="test", specs=spectrograms2, eegs=eegs2
    )

    with strategy.scope():
        model = build_model()

    for i in range(5):
        wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
        if not os.path.exists(wpath):
            print(f"Fold {i+1}: weights not found at {wpath} (skipping)")
            continue

        print(f"Fold {i+1}: loading {wpath}")
        model.load_weights(wpath)
        pred_i = model.predict(test_gen, verbose=1)
        preds.append(pred_i)
        found_any = True

if found_any:
    pred = np.mean(preds, axis=0)
    print("\nTest preds shape", pred.shape)
    pred = np.clip(pred, 1e-8, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)
else:
    print(
        "\nNo usable model predictions (either TF unavailable or weights missing). "
        "Using patient/global prior fallback to create a valid submission."
    )
    pred = np.zeros((len(test), len(TARGETS)), dtype="float64")
    for idx, pid in enumerate(test["patient_id"].values):
        if pid in _patient_prior.index:
            pred[idx] = _patient_prior.loc[pid, list(TARGETS)].values
        else:
            pred[idx] = _global_prior
    pred = np.clip(pred, 1e-8, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub = sub.assign(**{c: pred[:, k].astype("float32") for k, c in enumerate(TARGETS)})
sub = sub[SUB_COLS]

vals = sub[list(TARGETS)].values.astype("float64")
vals = np.clip(vals, 1e-8, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub.loc[:, list(TARGETS)] = vals.astype("float32")

if PLATFORM == "local":
    sample_path = (
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
else:
    sample_path = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
sample_cols = list(pd.read_csv(sample_path, nrows=1).columns)

sub = sub[sample_cols]
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print("Row-sum stats:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max())
print(sub.head())
