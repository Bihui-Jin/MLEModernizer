# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4638136830956462

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash in the first cell by preventing TensorFlow’s protobuf-related `MessageFactory.GetPrototype` import-time error (common on Kaggle with newer `protobuf`) via a safe environment setting applied before importing TensorFlow. Next, I make model weight loading robust by searching `/kaggle/input/` for the expected `.h5` files instead of hard-failing on a non-existent dataset folder name, while keeping the exact same model and inference logic. Finally, if weights still can’t be found, the script fall back to a valid, properly-normalized uniform-probability submission so you always get a runnable end-to-end pipeline producing `submission.csv` with correct columns and row sums of 1.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation early and (crucially) disabling C++ parsing for protobuf, which prevents the `MessageFactory.GetPrototype` AttributeError on Kaggle. Next, we keep your exact inference/model logic but make sure the script always reaches submission writing by adding a safe fallback path if TensorFlow still fails to import (uniform probabilities, properly normalized). Finally, we keep the existing weight search/loading behavior but ensure the output CSV always matches `sample_submission.csv` columns and row-normalization rules.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the model from running at all (so you’re likely falling back to uniform predictions, which explains the poor score). Specifically, I force the pure-Python protobuf implementation before any TensorFlow-related import and also avoid importing TensorFlow transitively via other packages (notably `librosa`) until after that env setup. Then I keep your exact model/inference logic but make the pipeline robust: if TensorFlow still fails, or if spectrogram/EEG parquet loading fails, it still write a valid normalized `submission.csv`. These changes should restore real model inference (improving KL toward the target) while preserving your core architecture and evaluation semantics.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by moving all TensorFlow-related imports (including `librosa`, which can import TF transitively) behind a safer protobuf environment setup, and I add a robust fallback to use the spectrogram-only pipeline if EEG parquet loading fails (instead of dropping to uniform predictions). I also correct a bug in your train aggregation where `train["max"]` becomes a DataFrame, and I make sure `submission.csv` is always written with the exact sample-submission column order and row-normalized probabilities. These changes preserve your core model/inference logic while ensuring the real ensemble weights can run, which should improve KL substantially toward the target. No training behavior, architecture, or loss is changed.'
- What this solution (achieved 1.40995) has done: 'The crash is happening at TensorFlow import time due to an incompatible `protobuf` runtime (`MessageFactory.GetPrototype`), so the code never reaches real model inference and ends up producing near-uniform fallback predictions (hence the poor KL score). I fix this by enforcing a Kaggle-safe protobuf stack before importing TensorFlow: uninstalling the broken protobuf version and reinstalling a compatible one inside the notebook runtime, then importing TF normally. I also make the fallback path “soft” instead of immediate when TF import fails the first time, so we only fall back to uniform predictions if the TF re-import still fails. These changes keep your model/inference logic intact but should restore actual weight-based predictions and move the score toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far worse than the target (0.4638), and the most likely reason is that you’re still effectively submitting near-uniform predictions because the expected weight files aren’t present (you fall back when missing). The smallest score-improving change is to load whatever compatible `.h5` weights actually exist in `/kaggle/input` (instead of requiring a specific filename pattern), run inference with them, and only fall back to uniform if none can be loaded. To keep evaluation semantics identical, we won’t change the model, preprocessing, or loss—just make weight discovery/loading robust and deterministic, and ensure test columns match what `DataGenerator` expects. This should move KL substantially toward the target while remaining minimal and safe.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far worse than the target (0.4638), and the most likely cause is that you are still effectively submitting close-to-uniform predictions because real fold weights are not being found/used. I make the smallest change that increases the chance of using the intended EB2 weights: (1) fix the weight filename pattern mismatch (your code looks for `EB2_v1_f*.h5` but the files are typically `..._fold*.h5`), and (2) prefer searching only within the provided `LOAD_MODELS_FROM` dataset first (instead of arbitrary `/kaggle/input` `.h5` files), so we don’t accidentally load incompatible weights. I also force `DataGenerator(augment=augment)` to actually respect the passed `augment` flag (currently it hard-codes `self.augment=False`), which is a correctness bug but does not change inference semantics because you run `augment=False` for test. These are minimal, execution-safe changes aimed at restoring real model inference and moving KL down toward the target band.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4638), and the most likely reason is that you’re still not actually running the intended fold ensemble (so predictions are effectively close to uniform or from incompatible weights). I make weight discovery deterministic and stricter: only load files that match the expected fold naming under `LOAD_MODELS_FROM` (and common alternate patterns), and only broaden the search if still nothing is found—this avoids accidentally loading unrelated `.h5` files that degrade predictions. I also fix a subtle but important test-generator bug: in `mode="test"` your generator currently sets `r=0`, which is inconsistent with how the model was trained/validated (it uses a central window), so I switch test to use the same center-window rule as validation. These are minimal changes that preserve model architecture and inference semantics, but should materially reduce KL toward your target.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.4638), and the most likely cause is that the model inference is effectively running with missing/zeroed EEG and/or missing mel features, which can badly mismatch the weights you intended to use. I keep the exact model and generator logic, but ensure that when EEG files aren’t loaded we explicitly disable mel generation (so the mel branch is consistently zero too), and I make test-time window selection consistent with validation (use the same “center” r rule instead of sometimes defaulting to 0). I also make the fold-weight discovery a bit more robust within `LOAD_MODELS_FROM` by accepting common “best/epoch” suffix variants, without broadening to arbitrary unrelated `.h5` unless necessary. These are minimal changes aimed at restoring the intended inference semantics and moving KL down toward the target band.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4638), and the most likely reason is that the intended fold weights are never actually loaded (so you’re effectively submitting near-uniform or incompatible-weight predictions). I make weight discovery stricter and deterministic by (1) automatically detecting the correct `/kaggle/input/<dataset>` folder that actually contains `EB2_v1_*.h5` and using it as `LOAD_MODELS_FROM`, and (2) only ensembling weights that match the expected `EB2_v{VER}` prefix to avoid loading unrelated `.h5` files that degrade predictions. I also fix the fold-file selection ordering so fold0..fold4 are used in order (instead of arbitrary sorted filenames), which stabilizes predictions and typically improves KL without changing the model or preprocessing. No architecture, loss, or feature extraction is changed; this only restores the intended inference behavior and should move KL down toward your target band.'
- What this solution (achieved 1.40995) has done: 'I make the smallest changes that reduce KL by ensuring you actually load and use the intended trained fold weights instead of silently falling back to uniform (which explains a KL around ~1.41). Specifically, I (1) stop assuming an external weights dataset name and instead auto-detect the EB2 fold `.h5` files inside the already-available competition input directory, and (2) broaden the fold filename parsing slightly (still restricted to `EB2_v{VER}_*.h5`) so common naming variants are picked up reliably. I also fix the warning string formatting so you can see the real `VER` in logs, and keep all model/preprocessing logic unchanged; only weight discovery/loading is adjusted. This should move KL down toward your 0.4638 target without changing architecture, loss, or feature generation.'

# 9. Code solution

## === cell 0
import os
import sys
import warnings
import subprocess

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPLUSPLUS", "1")


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.message_factory as mf

        if hasattr(mf.MessageFactory, "GetPrototype"):
            return True
        return False
    except Exception:
        return False


def _pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


if not _ensure_compatible_protobuf():
    _pip_install(["-q", "--upgrade", "pip"])
    _pip_install(["-q", "--upgrade", "--force-reinstall", "protobuf>=3.20.3,<5"])

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False

LOAD_MODELS_FROM = "hms-harmful-brain-activity-classification"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 64
LENGTH = 256

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_TF_OK = True
_TF_IMPORT_ERR = None
try:
    import tensorflow as tf
except Exception as e:
    _TF_OK = False
    _TF_IMPORT_ERR = repr(e)

try:
    import librosa
except Exception as e:
    librosa = None
    _LIBROSA_IMPORT_ERR = repr(e)
else:
    _LIBROSA_IMPORT_ERR = None

if _TF_OK:
    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    VER = 1

    MIX = True
    if MIX:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (keras mixed_precision policy)")
        except Exception:
            try:
                tf.config.optimizer.set_experimental_options(
                    {"auto_mixed_precision": True}
                )
                print("Mixed precision enabled (experimental)")
            except Exception:
                print(
                    "Mixed precision could not be enabled; continuing with default precision"
                )
    else:
        print("Using full precision")
else:
    print("WARNING: TensorFlow failed to import; will fall back to valid submission.")
    print("TF import error:", _TF_IMPORT_ERR)

if librosa is None:
    print("WARNING: librosa failed to import; mel features will be disabled if needed.")
    print("librosa import error:", _LIBROSA_IMPORT_ERR)


def _glob_h5_under(dirpath):
    out = []
    for dp, _, fns in os.walk(dirpath):
        for fn in fns:
            if fn.lower().endswith(".h5"):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def _autodetect_models_dir(ver, platform="kaggle", current_dir=None):
    """
    Change (score-improving, minimal): find a directory under the chosen base path (or /kaggle/input)
    that actually contains EB2_v{ver}*.h5, so we load the real fold weights instead of falling back to uniform.
    """
    if current_dir is not None and os.path.isdir(current_dir):
        try:
            for fn in _glob_h5_under(current_dir):
                b = os.path.basename(fn)
                if b.startswith(f"EB2_v{ver}_") and b.lower().endswith(".h5"):
                    return os.path.dirname(fn)
        except Exception:
            pass

    root = "/kaggle/input" if platform == "kaggle" else "./input"
    if not os.path.isdir(root):
        return current_dir

    candidates = []
    try:
        for d in os.listdir(root):
            dp = os.path.join(root, d)
            if not os.path.isdir(dp):
                continue
            hits = 0
            try:
                for fn in _glob_h5_under(dp):
                    b = os.path.basename(fn)
                    if b.startswith(f"EB2_v{ver}_") and b.lower().endswith(".h5"):
                        hits += 1
            except Exception:
                continue
            if hits > 0:
                candidates.append((hits, dp))
    except Exception:
        return current_dir

    if not candidates:
        return current_dir

    candidates.sort(reverse=True)
    return candidates[0][1]


if _TF_OK and (not NEEDTRAIN):
    detected = _autodetect_models_dir(
        VER, platform=PLATFORM, current_dir=LOAD_MODELS_FROM
    )
    if detected != LOAD_MODELS_FROM:
        print("Detected model weights directory:", detected)
        LOAD_MODELS_FROM = detected
    else:
        print("Using model weights directory:", LOAD_MODELS_FROM)



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

tmp = df.groupby("eeg_id")["spectrogram_label_offset_seconds"].max()
train["max"] = tmp.values

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



## === cell 2
if _TF_OK:
    try:
        import albumentations as albu
    except Exception:

        class _NoOpCompose:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, image=None, **kwargs):
                return {"image": image}

        class _NoOp:
            def __init__(self, *args, **kwargs):
                pass

        class albu:  # noqa: N801
            Compose = _NoOpCompose
            HorizontalFlip = _NoOp
            CoarseDropout = _NoOp

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
            allow_eeg_missing=False,
            allow_mel_missing=False,
        ):
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = augment
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.allow_eeg_missing = allow_eeg_missing
            self.allow_mel_missing = allow_mel_missing
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.data) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            X, X_mel, X_eeg, y = self.__data_generation(indexes)
            if self.augment:
                X = self.__augment_batch(X)
            return [X, X_mel, X_eeg], y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
            X_mel = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
            X_eeg = np.zeros(
                (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
            y = np.zeros((len(indexes), 6), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                if self.mode in ["test", "valid"]:
                    if ("min" in row.index) and ("max" in row.index):
                        r = int((row["min"] + row["max"]) // 4)
                    else:
                        r = 75  # central-ish
                else:
                    r = np.random.randint(row["min"], row["max"] + 1) // 2

                for k in range(4):
                    img = self.specs[row.spec_id][
                        r : r + 300, k * 100 : (k + 1) * 100
                    ].T

                    img = np.clip(img, np.exp(-6), np.exp(8))
                    img = np.log(img)
                    img = np.nan_to_num(img, nan=0.0)

                    img = img[
                        :,
                        max(round((600 / 2 - LENGTH) / 2), 0) : min(
                            (round((600 / 2 - LENGTH) / 2) + LENGTH), img.shape[1]
                        ),
                    ]

                    if HIGH != 100:
                        img_rs = np.array(
                            tf.image.resize(
                                np.reshape(img, (img.shape[0], img.shape[1], 1)),
                                ((HIGH - 16), LENGTH),
                            ),
                            dtype=np.float32,
                        )[:, :, 0]
                        X[
                            j,
                            round((HIGH - img_rs.shape[0]) / 2) : round(
                                (HIGH + img_rs.shape[0]) / 2
                            ),
                            :,
                            k,
                        ] = img_rs
                    else:
                        X[j, :, :, k] = img

                    X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                        np.std(X[j, :, :, k]) + 1e-6
                    )

                    if (self.eegs is not None) and (row.eeg_id in self.eegs):
                        img_eeg = self.eegs[row.eeg_id][:, :, k]
                        X_eeg[j, 1:5, :, k] = img_eeg
                        X_eeg[j, :, :, k] = (
                            X_eeg[j, :, :, k]
                            - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                        if (librosa is not None) and (not self.allow_mel_missing):
                            for ii_x in range(img_eeg.shape[0]):
                                x = img_eeg[ii_x, :]
                                mel_spec = librosa.stft(
                                    y=x,
                                    hop_length=len(x) // LENGTH,
                                    n_fft=256,
                                    win_length=128,
                                )
                                mel_spec = np.abs(mel_spec) ** 2
                                mel_spec = mel_spec[:48, :LENGTH]
                                if ii_x == 0:
                                    mel_spec_db = librosa.amplitude_to_db(
                                        mel_spec, ref=np.median
                                    )
                                else:
                                    mel_spec_db = mel_spec_db + librosa.amplitude_to_db(
                                        mel_spec, ref=np.median
                                    )
                            mel_spec_db = mel_spec_db / img_eeg.shape[0]

                            if HIGH != 100:
                                mel_rs = np.array(
                                    tf.image.resize(
                                        np.reshape(
                                            mel_spec_db,
                                            (
                                                mel_spec_db.shape[0],
                                                mel_spec_db.shape[1],
                                                1,
                                            ),
                                        ),
                                        ((HIGH - 16), LENGTH),
                                    ),
                                    dtype=np.float32,
                                )[:, :, 0]
                                X_mel[
                                    j,
                                    round((HIGH - mel_rs.shape[0]) / 2) : round(
                                        (HIGH + mel_rs.shape[0]) / 2
                                    ),
                                    :,
                                    k,
                                ] = mel_rs
                            else:
                                X_mel[j, :, :, k] = mel_spec_db

                            X_mel[j, :, :, k] = (
                                X_mel[j, :, :, k] - np.mean(X_mel[j, :, :, k])
                            ) / (np.std(X_mel[j, :, :, k]) + 1e-6)
                    else:
                        pass

                if self.mode != "test":
                    y[j] = row[TARGETS].values

            return X, X_mel, X_eeg, y

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
if _TF_OK:
    try:
        import efficientnet.tfkeras as efn

        _EffNetB0 = efn.EfficientNetB0
    except Exception:
        _EffNetB0 = tf.keras.applications.EfficientNetB0

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="tanh",
                dilation_rate=dilation_rate,
            )(x)
            sigm_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                dilation_rate=dilation_rate,
            )(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
                x
            )
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x

    def build_model():
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
        inp_mel = tf.keras.Input(shape=(HIGH, LENGTH, 4))
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

        base_model = _EffNetB0(include_top=False, weights=None, input_shape=None)
        base_model._name = "spectrogram_extractor"
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x0 = inp[:, :, :, :1]
        x1 = inp[:, :, :, 1:2]
        x2 = inp[:, :, :, 2:3]
        x3 = inp[:, :, :, 3:4]
        x01 = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

        x_mel0 = inp_mel[:, :, :, :1]
        x_mel1 = inp_mel[:, :, :, 1:2]
        x_mel2 = inp_mel[:, :, :, 2:3]
        x_mel3 = inp_mel[:, :, :, 3:4]
        x_mel = tf.keras.layers.Concatenate(axis=1)([x_mel0, x_mel1, x_mel2, x_mel3])

        x01 = tf.keras.layers.Concatenate(axis=2)([x01, x_mel])
        x = tf.keras.layers.Concatenate(axis=3)([x01, x01, x01])

        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

        base_model_eeg = _EffNetB0(include_top=False, weights=None, input_shape=None)
        base_model_eeg._name = "eeg_extractor"
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x0_eeg = inp_eeg[:, :, :, :1]
        x1_eeg = inp_eeg[:, :, :, 1:2]
        x2_eeg = inp_eeg[:, :, :, 2:3]
        x3_eeg = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

        x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[inp, inp_mel, inp_eeg], outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.KLDivergence()
        model.compile(loss=loss, optimizer=opt)
        return model




## === cell 4
def _write_submission(test_df, probs, targets, out_path="submission.csv"):
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    sub[list(targets)] = probs
    sub = sub[["eeg_id"] + list(targets)]
    sub.to_csv(out_path, index=False)

    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sums (min/mean/max):",
        sub[list(targets)].sum(axis=1).min(),
        sub[list(targets)].sum(axis=1).mean(),
        sub[list(targets)].sum(axis=1).max(),
    )


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    if not _TF_OK:
        n = len(test)
        pred = np.ones((n, len(TARGETS)), dtype=np.float32) / float(len(TARGETS))
        _write_submission(test, pred, TARGETS, out_path="submission.csv")
    else:
        spectrograms2 = None
        eegs2 = None

        try:
            if PLATFORM == "local":
                PATH_SPEC = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            else:
                PATH_SPEC = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

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

            test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

            if "min" not in test.columns:
                test["min"] = 0
            if "max" not in test.columns:
                test["max"] = 300  # yields r = (0+300)//4 = 75

        except Exception as e:
            print(
                "WARNING: Failed while loading test spectrogram parquets; falling back to uniform submission."
            )
            print("Load error:", repr(e))
            n = len(test)
            pred = np.ones((n, len(TARGETS)), dtype=np.float32) / float(len(TARGETS))
            _write_submission(test, pred, TARGETS, out_path="submission.csv")

        if spectrograms2 is not None:
            try:
                from scipy import signal

                if PLATFORM == "local":
                    PATH_EEG = (
                        "./input/hms-harmful-brain-activity-classification/test_eegs/"
                    )
                else:
                    PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

                files_eeg = os.listdir(PATH_EEG)
                print(f"There are {len(files_eeg)} test eeg parquets")

                eegs2 = {}

                if len(filter_range) == 1:
                    if filter_range[0] > 5:
                        b, a = signal.butter(
                            3, np.float32(filter_range) * 2 / SFREQ, "lowpass"
                        )
                    else:
                        b, a = signal.butter(
                            3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                        )
                else:
                    b, a = signal.butter(
                        3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
                    )

                test_eeg_ids = set(test["eeg_id"].astype(int).values.tolist())

                for i, f in enumerate(files_eeg):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    name = int(f.split(".")[0])
                    if name not in test_eeg_ids:
                        continue

                    raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")

                    time_temp = 0
                    time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                    time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                    eeg_default = raw_eeg.loc[
                        time_start : (time_stop - 1), :
                    ].reset_index(drop=True)

                    list_eeg = []
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
                        list_eeg.append(
                            np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1))
                        )

                    list_eeg = np.concatenate(list_eeg, 2)
                    eegs2[name] = list_eeg
                print()
                print(f"Loaded EEGs for {len(eegs2)} / {len(test)} test rows")
            except Exception as e:
                print(
                    "WARNING: Failed while loading test EEG parquets; continuing with spectrogram-only (EEG zeros)."
                )
                print("EEG load error:", repr(e))
                eegs2 = None




## === cell 5
def _find_weight_paths_strict(preferred_dir, ver, n_folds=5):
    """
    Change (score-improving, minimal): detect fold weights more robustly (still only EB2_v{ver}_*.h5)
    so we actually ensemble the intended folds instead of writing uniform predictions.
    Returns {fold_index: path}.
    """
    found = {}
    if not os.path.isdir(preferred_dir):
        return found

    candidates = []
    try:
        for dp, _, fns in os.walk(preferred_dir):
            for fn in fns:
                if fn.startswith(f"EB2_v{ver}_") and fn.lower().endswith(".h5"):
                    candidates.append(os.path.join(dp, fn))
    except Exception:
        return found

    for path in candidates:
        low = os.path.basename(path).lower()
        fold_idx = None
        for i in range(n_folds):
            if (
                (f"fold{i}" in low)
                or (f"_f{i}" in low)
                or (f"-f{i}" in low)
                or low.endswith(f"f{i}.h5")
            ):
                fold_idx = i
                break
        if fold_idx is not None:
            found.setdefault(fold_idx, path)

    return found


if not NEEDTRAIN and _TF_OK:
    if ("spectrograms2" not in globals()) or (spectrograms2 is None):
        if not os.path.exists("submission.csv"):
            n = len(test)
            pred = np.ones((n, len(TARGETS)), dtype=np.float32) / float(len(TARGETS))
            _write_submission(test, pred, TARGETS, out_path="submission.csv")
    else:
        with strategy.scope():
            model = build_model()

        mel_missing = (librosa is None) or (eegs2 is None)

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            allow_eeg_missing=True,
            allow_mel_missing=mel_missing,
        )

        preds = []
        loaded_any = False

        strict_found = _find_weight_paths_strict(LOAD_MODELS_FROM, VER, n_folds=5)

        fold_paths = []
        for i in range(5):
            if i in strict_found:
                fold_paths.append(strict_found[i])

        if len(fold_paths) > 0:
            for i, wpath in enumerate(fold_paths):
                print(f"Ensemble weight fold {i} loading: {wpath}")
                try:
                    model.load_weights(wpath)
                    pred_fold = model.predict(test_gen, verbose=1)
                    preds.append(pred_fold)
                    loaded_any = True
                except Exception as e:
                    print(
                        "Failed to load/predict with weight (skipping):", wpath, repr(e)
                    )

        if not loaded_any:
            print(
                f"WARNING: No compatible EB2_v{VER} fold weights found; writing uniform fallback submission."
            )
            n = len(test)
            pred = np.ones((n, len(TARGETS)), dtype=np.float32) / float(len(TARGETS))
            _write_submission(test, pred, TARGETS, out_path="submission.csv")
        else:
            pred = np.mean(preds, axis=0).astype(np.float32)
            print("Test preds shape", pred.shape)
            _write_submission(test, pred, TARGETS, out_path="submission.csv")
