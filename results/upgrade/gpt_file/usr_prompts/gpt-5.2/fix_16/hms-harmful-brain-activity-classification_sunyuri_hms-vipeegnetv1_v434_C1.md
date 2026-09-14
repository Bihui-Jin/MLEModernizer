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

1.1320943308955929

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix two execution blockers so the notebook runs end-to-end and always writes a valid `submission.csv`. First, the protobuf `MessageFactory.GetPrototype` crash comes from an incompatible TensorFlow/protobuf combo triggered by the mixed-precision/protobuf env handling; the safest minimal fix in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow. Second, the “uniform submission” fallback is currently using `len(test)` after an index-alignment step that can accidentally expand rows; I ensure test is aligned to `sample_submission` as a 1:1 `eeg_id` list and build the fallback prior using `len(sample_sub)` so dimensions always match. These changes are score-neutral (they only ensure correct execution and correct submission formatting/summing to 1).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow-related imports* and by defensively falling back if the env vars are ignored in this runtime. I also fix a training-path bug where the script tries to read a locally-saved `train.csv` that may not exist on Kaggle (so it always build `train` directly from the official CSV when needed). For score improvement toward your (lower-is-better) target, I make inference robust and slightly better calibrated by loading both `stage2` and (if missing) falling back to `stage1` weights per fold rather than outputting a uniform prior; this preserves the model/architecture/training semantics and only affects inference weight selection. Finally, I keep strict submission alignment to `sample_submission` and enforce valid probability normalization.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is selected *before* TensorFlow is imported, and by adding a safe fallback that restarts the import path if the runtime still loads the C++ protobuf. I also make Kaggle inference reliably find and load the provided fold weights by not depending solely on directory names beginning with `models` (a common cause of silently producing a uniform submission), while keeping the model, preprocessing, and ensembling logic unchanged. Finally, I keep strict alignment to `sample_submission` and enforce probability normalization so the submission is always valid; these changes should improve score versus the current uniform/partial-model fallback behavior without altering training semantics.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by enforcing the pure-Python protobuf implementation earlier and more robustly (before any TF/Keras import paths are touched), which is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle runtime. I also make inference more reliable (and thus improve score toward your lower-is-better target) by loading weights via recursive search for `fold*_stage*.weights.h5` inside `LOAD_MODELS_FROM`, instead of assuming they sit at the directory root—this prevents silent “no models found → uniform prior” fallbacks. Finally, I keep submission alignment strictly identical to `sample_submission` and enforce safe probability normalization/clipping so the CSV is always valid for the KL metric.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation earlier and disabling the C++ implementation before any TensorFlow/Keras code is imported, which avoids the `MessageFactory.GetPrototype` error in this Kaggle runtime. I also make model-weight discovery/load more reliable (recursive search under `LOAD_MODELS_FROM`, with a safe stage2→stage1 fallback per fold) so inference doesn’t silently fall back to a uniform prior, which should improve KL score toward your (lower-is-better) target. Finally, I keep strict alignment to `sample_submission` and enforce safe clipping/renormalization so every row sums to 1 and the submission is always valid.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow import path is touched* and by hard-disabling the C++ protobuf backend, which directly addresses the `MessageFactory.GetPrototype` error. I also add a small defensive fallback that prints the detected protobuf implementation to make the runtime state obvious and fail fast with a clear error if it still loads the C++ backend. For score improvement toward your lower-is-better target, I keep the model/training logic identical and only make inference more reliable by ensuring weight discovery works robustly (recursive search already present) and by compiling models for prediction plus using float32-safe normalization/clipping to avoid accidental numerical issues that hurt KL. Finally, I keep strict alignment to `sample_submission` and always write a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by enforcing the pure-Python protobuf runtime before any TensorFlow/Keras import and by adding a safe runtime check that aborts early with a clear message if the C++ backend is still active. I also fix an inference-time KeyError/empty-selection bug in `DataGenerator` (the row matching forgot `other_vote`, so `rows` can become empty), which can silently break inference quality; this is score-relevant but does not change the model or training semantics. Finally, I keep submission alignment strictly to `sample_submission`, and I make probability normalization numerically safe so every row sums to 1 for the KL metric.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by making protobuf enforcement more robust: force the pure-Python backend early, and if TensorFlow still fails due to `MessageFactory.GetPrototype`, retry once with the C++ implementation explicitly disabled (this is execution-blocking). I also make inference reliably run under Kaggle by ensuring `DataGenerator` doesn’t reference undefined training-only variables (`df`, `TARGETS_RAW`) during `mode="test"`, which can otherwise break or silently misbehave. Finally, I keep the model/training logic unchanged but harden submission validity by clipping and renormalizing predictions in float32 and aligning test rows strictly to `sample_submission` so the produced `submission.csv` is always valid for the KL metric (and avoids score-degrading misalignment/uniform fallback).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow import occurs (including indirect imports) and by removing the unreliable “retry import” path that can’t change protobuf backend after it has loaded. I also make the script robust in Kaggle by auto-disabling mixed precision when no GPU is available (mixed_float16 on CPU can error/slow), which is score-neutral and avoids runtime failures. Finally, I keep the model/training core logic unchanged, but ensure inference always loads fold weights reliably and writes a valid `submission.csv` aligned to `sample_submission` with strictly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the execution-blocking protobuf/TensorFlow crash by enforcing a compatible protobuf version *before any TensorFlow import* (the env vars alone are not sufficient in this Kaggle runtime), and I keep the rest of the pipeline unchanged. I also make inference a bit more reliable (and thus improve KL toward your lower-is-better target) by ensuring fold weights are discovered/loaded deterministically and by avoiding any silent uniform-fallback when weights exist. Finally, I keep strict alignment to `sample_submission.csv` and enforce numerically safe probability clipping/renormalization so every row sums to 1 and the submission is always valid.'
- What this solution (achieved 1.40995) has done: 'Your current gap is 1.40995 − 1.13209 ≈ +0.278 (about 24.6% worse than target), so we should improve (lower) the KL score with minimal, inference-only changes. The most score-relevant issue I see is that test-time preprocessing is not identical to train-time preprocessing: train does bandpass filtering directly, while test does a “mirror-pad then filter then crop” trick that changes edge behavior and distribution. I make test-time EEG preprocessing match the training path (same filtering, no mirror-padding), keeping model/weights/architecture unchanged. I also remove the unintended always-true augmentation condition `np.random.rand() > 0` (harmless for Kaggle inference because NEEDTRAIN=False, but it is logically wrong) without changing inference semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995) is worse than the target (1.13209), so we should improve (lower) it with the smallest inference-only change. The biggest score-relevant issue in your current inference is that `DataGenerator(mode="test")` always uses `r_eeg=0`, so for every test EEG you feed the first 50 seconds—this is inconsistent with how training/validation pick the (center) labeled window and can systematically hurt KL. I change only the test-time offset to use the mid-window of each 50s test EEG (i.e., `r_eeg = (duration-50)/2`, which is 0 for exact-50s but also robust if lengths differ), keeping the exact same model, weights, filtering, normalization, and ensembling. I also make this choice deterministic and safe (bounds-checked) without altering train/valid behavior or any augmentation logic.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995) is worse than the target (1.13209), so we should improve (lower) it with the smallest, inference-only changes. The biggest score-relevant issue is that in `mode!="test"` you apply a fixed left/right swap augmentation even for `mode="valid"` (and also for `mode="test"` via the current `else:` branch), which creates a train/valid/test distribution mismatch and hurts calibration for KL. I make the non-train path (valid/test) deterministic and augmentation-free (no swaps/masking/sign flips/reversal), while keeping the exact same model, weights, filtering, normalization, and ensembling. I also ensure the test generator is instantiated with `stage=1` to prevent any stage-dependent augmentation logic from ever triggering in inference (even if future edits change conditions).'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX"] = "1"

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        v = version("protobuf")
        major = int(v.split(".")[0])
        if major >= 5:
            print(
                f"Detected protobuf=={v} (>=5). Installing protobuf<5 for TF compatibility..."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
    except Exception as e:
        print("Warning: protobuf version check/install skipped due to:", repr(e))


_ensure_protobuf_compatible()

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"
if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # local training
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # kaggle inference
    NEEDTRAIN = False

    import glob

    candidate_dirs = []
    for d in glob.glob("/kaggle/input/*"):
        if os.path.isdir(d):
            w = glob.glob(
                os.path.join(d, "**", "fold*_stage*.weights.h5"), recursive=True
            )
            if len(w) > 0:
                candidate_dirs.append((d, len(w)))
    if candidate_dirs:
        candidate_dirs.sort(key=lambda x: x[1], reverse=True)
        LOAD_MODELS_FROM = candidate_dirs[0][0].split(os.sep)[-1]
    else:
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

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

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg

eegs = {}  # preprocessed eegs for training
eegs_test = {}

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

import warnings

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import signal
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


def _print_protobuf_impl():
    try:
        import google.protobuf.internal.api_implementation as _api_impl

        _pb_impl = _api_impl.Type()
        print("protobuf implementation:", _pb_impl)
        return _pb_impl
    except Exception as _e:
        print("Could not determine protobuf implementation:", _e)
        return None


_pb_impl = _print_protobuf_impl()

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed. This is usually due to a protobuf incompatibility. "
        "This script forces a TF-compatible protobuf before TF import."
    ) from e

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
    print("Determinism setting not available:", e)

MIX = True
if MIX and len(gpus) > 0:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    tf.keras.mixed_precision.set_global_policy("float32")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

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
        y_data = train[TARGETS].values.astype(np.float32)
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
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

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)

            eeg = np.array(eeg, dtype=np.float32)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()




## === cell 1
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        sample_weights=False,
        mode="train",
        eegs=None,
        stage=2,
    ):

        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stage = stage
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.dataframe) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)

        if self.mode == "test":
            return x
        if self.sample_weights:
            return x, y, sample_weights
        return x, y

    def on_epoch_end(self):
        self.nan = 0
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        x_eeg = np.zeros(
            (len(indexes), EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)),
            dtype="float32",
        )
        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            if self.mode != "test":
                sample_weight = float(np.sum(row[TARGETS_RAW].values)) / 20.0

            if self.mode == "test":
                eeg_len = self.eegs[row.eeg_id].shape[1]
                total_sec = float(eeg_len) / RSFREQ
                r_eeg = max(0.0, (total_sec - 50.0) / 2.0)
                r_eeg = min(r_eeg, max(0.0, total_sec - 50.0))
            else:
                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    * (df.seizure_vote == row.seizure_vote_raw)
                    * (df.lpd_vote == row.lpd_vote_raw)
                    * (df.gpd_vote == row.gpd_vote_raw)
                    * (df.lrda_vote == row.lrda_vote_raw)
                    * (df.grda_vote == row.grda_vote_raw)
                    * (df.other_vote == row.other_vote_raw),
                    :,
                ].reset_index(drop=True)

                if len(rows) == 0:
                    rows = pd.DataFrame([row])

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
                r_eeg = float(row.eeg_label_offset_seconds)
                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0, r_eeg)
                    r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

            eeg = self.eegs[row.eeg_id][
                :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
            ]
            eeg = np.concatenate(
                (
                    eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                ),
                axis=0,
            )

            eeg = eeg[
                :,
                round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                    (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                ),
            ]

            if self.mode == "train":
                if (self.stage == 2) and (np.random.rand() > 0.5):
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                else:
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
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

                    if np.random.rand() > 0.5:
                        eeg = -eeg

                    if np.random.rand() > 0.5:
                        eeg = eeg[:, ::-1]

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)
            eeg = eeg + 1024
            eeg = eeg / 2048 * 255

            x_eeg[j] = eeg

            if self.mode != "test":
                y[j] = row[TARGETS].values / float(np.sum(row[TARGETS].values))

                if self.sample_weights:
                    sample_weights[j] = sample_weight
                else:
                    sample_weights[j] = 1.0

        return x_eeg, y, sample_weights




## === cell 2
from tensorflow.keras import optimizers


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

        self.begin = 1

    def __call__(self, step):
        if step == self.total_step:
            self.begin = 0
            self.lr_max = self.lr_max * 0.5
            self.lr_min = self.lr_min * 0.1

        step = step % self.total_step
        step = step + 1

        if (self.begin == 1) and (step < self.warm_step):
            lr = self.lr_max / self.warm_step * step
        else:
            if self.begin == 1:
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
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0 + tf.cos(step / 10 * np.pi)
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




## === cell 3
def build_model():
    inp_eeg = tf.keras.Input(
        shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)), name="eeg"
    )
    x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
        inp_eeg
    )

    x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg_raw, x_eeg_raw, x_eeg_raw])

    base_model_eeg = tf.keras.applications.EfficientNetV2B3(
        include_top=False, weights=None, include_preprocessing=True
    )

    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model_eeg.load_weights(
                f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model_eeg.load_weights(
                f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
            )

    base_model_eeg.name = "eeg_extractor"

    x_eeg = base_model_eeg(x_eeg)

    x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
        x_eeg
    )

    model = tf.keras.Model(inputs=inp_eeg, outputs=y)

    return model




## === cell 4
def train_fold(
    i,
    stage,
    train_index,
    valid_index,
    df_train_stage1,
    df_valid_stage1,
    df_train_stage2,
    df_valid_stage2,
    build_model,
    BATCHSIZE,
    EPOCHS,
    LEARN_RATE,
    TARGETS,
    TARGETS_RAW,
):

    print("#" * 25)
    print(f"### Fold {i + 1}")

    model = build_model()
    loss = tf.keras.losses.KLDivergence()

    if stage == 1:
        train_gen_stage = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            eegs=eegs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            eegs=eegs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
    elif stage == 2:
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            eegs=eegs,
            stage=stage,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2 * 2,
            mode="valid",
            eegs=eegs,
            stage=stage,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1 * 3)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1 * 3,
                    LEARN_RATE * 0.1 * 0.1 * 0.1,
                    0,
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]

    model.compile(loss=loss, optimizer=opt)

    if stage == 1:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=EPOCHS,
            callbacks=callbacks_stage,
        )
    elif stage == 2:
        history = model.fit(
            train_gen_stage,
            verbose=1,
            validation_data=valid_gen_stage,
            epochs=max(round(EPOCHS / 3), 1),
            callbacks=callbacks_stage,
        )

    model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

    loss_vals = history.history["loss"]
    val_loss_vals = history.history["val_loss"]
    epochs = range(1, len(loss_vals) + 1)
    plt.figure()
    plt.plot(epochs, loss_vals, "bo", label="loss")
    plt.plot(epochs, val_loss_vals, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss_vals), 4)}, val loss: {round(min(val_loss_vals), 4)}",
        fontsize=12,
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    if stage == 1:
        valid_stage = df_valid_stage1[TARGETS].values
    elif stage == 2:
        valid_stage = df_valid_stage2[TARGETS].values
    predict_stage = model.predict(valid_gen_stage)

    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
    gc.collect()

    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    plt.yticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    thresh = cm.max() / 2.0
    import itertools

    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[ii, jj] > -0.1:
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
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}_cm.svg"))
    plt.close()

    del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
    gc.collect()




## === cell 5
if __name__ == "__main__":
    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        mp.set_start_method("spawn")

        gkf = GroupKFold(n_splits=SPLITS)

        for i, (train_index, valid_index) in enumerate(
            gkf.split(train, train.expert_consensus, train.patient_id)
        ):
            print("#" * 25)
            print(f"### Fold {i + 1}")

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [1, 2]:
                p = mp.Process(
                    target=train_fold,
                    args=(
                        i,
                        stage,
                        train_index,
                        valid_index,
                        df_train_stage1,
                        df_valid_stage1,
                        df_train_stage2,
                        df_valid_stage2,
                        build_model,
                        BATCHSIZE,
                        EPOCHS,
                        LEARN_RATE,
                        TARGETS,
                        TARGETS_RAW,
                    ),
                )
                p.start()
                p.join()

    else:
        import glob

        sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))

        TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]
        TARGETS_RAW = [t + "_raw" for t in TARGETS]

        test = test.drop_duplicates(subset=["eeg_id"]).set_index("eeg_id")
        test = test.reindex(sample_sub["eeg_id"].values).reset_index()
        if test["spectrogram_id"].isna().any():
            raise ValueError(
                "After aligning to sample_submission, some eeg_id are missing in test.csv"
            )

        print("Test shape (aligned to sample_submission)", test.shape)
        print("Loading models from:", LOAD_MODELS_FROM)

        stage2_w = sorted(
            glob.glob(
                os.path.join(LOAD_MODELS_FROM, "**", "fold*_stage2.weights.h5"),
                recursive=True,
            )
        )
        stage1_w = sorted(
            glob.glob(
                os.path.join(LOAD_MODELS_FROM, "**", "fold*_stage1.weights.h5"),
                recursive=True,
            )
        )

        def _fold_id(p):
            bn = os.path.basename(p)
            try:
                return int(bn.split("fold")[1].split("_")[0])
            except Exception:
                return 10**9

        stage2_w = sorted(stage2_w, key=_fold_id)
        stage1_w = sorted(stage1_w, key=_fold_id)

        chosen_weights = {}
        for p in stage1_w:
            fid = _fold_id(p)
            chosen_weights[fid] = p
        for p in stage2_w:
            fid = _fold_id(p)
            chosen_weights[fid] = p  # override with stage2 if available

        models = []
        for fid in sorted(chosen_weights.keys()):
            chosen = chosen_weights[fid]
            rel = (
                os.path.relpath(chosen, LOAD_MODELS_FROM)
                if os.path.exists(LOAD_MODELS_FROM)
                else os.path.basename(chosen)
            )
            print(f"Fold {fid + 1} -> loading {rel}")
            model = build_model()
            model.load_weights(chosen)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.KLDivergence(),
            )
            models.append(model)

        if len(models) == 0:
            prior = np.ones((len(sample_sub), len(TARGETS)), dtype=np.float32) / len(
                TARGETS
            )
            sub = pd.DataFrame({"eeg_id": sample_sub.eeg_id.values})
            sub[TARGETS] = prior
            sub.to_csv("submission.csv", index=False)
            print(
                "No models found in",
                LOAD_MODELS_FROM,
                "-> wrote uniform submission.csv",
                sub.shape,
            )
        else:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

            preds_all_list = []
            start = 0
            while start < len(test):
                end = min(start + TEST_BATCHSIZE, len(test))
                batch_df = test.iloc[start:end].reset_index(drop=True)

                eegs_test = {}
                for eeg_id in batch_df.eeg_id.values:
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

                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                    if filter_range is not None:
                        eeg = signal.filtfilt(b, a, eeg, axis=1)

                    eeg = np.array(eeg, dtype=np.float32)
                    eegs_test[eeg_id] = eeg

                test_gen = DataGenerator(
                    batch_df,
                    shuffle=False,
                    sample_weights=False,
                    batch_size=TEST_BATCHSIZE,
                    mode="test",
                    eegs=eegs_test,
                    stage=1,
                )

                preds = []
                for m in models:
                    pred = m.predict(test_gen, verbose=0)
                    preds.append(pred)
                pred = np.mean(preds, axis=0)

                preds_all_list.append(pred)
                start = end

                del eegs_test, test_gen, preds, pred
                gc.collect()

            preds_all = np.concatenate(preds_all_list, axis=0).astype(np.float32)

            eps = np.float32(1e-8)
            preds_all = np.clip(preds_all, eps, np.float32(1.0))
            row_sum = np.maximum(
                preds_all.sum(axis=1, keepdims=True), eps * len(TARGETS)
            )
            preds_all = preds_all / row_sum

            sub = pd.DataFrame({"eeg_id": sample_sub.eeg_id.values})
            sub[TARGETS] = preds_all
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())

            assert len(sub) == len(
                sample_sub
            ), "Submission and sample_submission must have the same length"
            assert list(sub.columns) == list(
                sample_sub.columns
            ), "Submission columns must match sample_submission"
            row_sums = sub[TARGETS].sum(axis=1).values
            assert np.all(np.isfinite(row_sums)), "Non-finite probabilities found"
            assert (
                np.max(np.abs(row_sums - 1.0)) < 1e-3
            ), "Row probabilities must sum to 1"
