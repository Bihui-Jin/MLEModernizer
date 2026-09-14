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

0.4669932505572488

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION forcing and the deprecated mixed-precision experimental option, which is what triggers the `MessageFactory.GetPrototype` error under Python 3.12. Then I fix the missing weights path by adding an automatic fallback: if the specified model directory doesn’t exist, the script switch to creating a valid (but untrained) submission using a safe uniform-probability fallback so a `.csv` is always produced. I also make the test data generator robust by ensuring the expected `min/max` columns exist in test mode, preventing key errors. These changes are minimal, keep the model architecture intact, and guarantee an end-to-end run that outputs `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash under Python 3.12 by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error). I keep the model/data logic unchanged, but also make the code robust to missing GPUs by falling back to CPU cleanly. Finally, to move the score toward the target (lower is better) from the current uniform-fallback behavior, I ensure the provided pretrained weights can actually be found by adding a safe fallback search under `/kaggle/input/` for the expected folder, so the ensemble predictions are used instead of uniform probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the pure-Python protobuf implementation, which is the direct cause of the `MessageFactory.GetPrototype` error under Python 3.12 in Kaggle. This is a runtime-only fix and does not change the model, features, or training/inference logic. I also add a small safety fallback to load weights with `by_name=True, skip_mismatch=True` only if a strict load fails (this keeps the intended pretrained weights usage but prevents hard crashes from minor serialization/version mismatches). Finally, I keep the existing prediction normalization so the submission remains valid (rows sum to 1) and always writes `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf crash under Python 3.12 by forcing the pure-Python protobuf implementation **before** importing TensorFlow (this is the direct root-cause of the `MessageFactory.GetPrototype` error). Then we ensure the test `DataGenerator` always has the columns it expects (`spec_id`, plus `min/max` placeholders) so inference can’t randomly key-error. Finally, we keep the existing 5-fold weight-loading ensemble logic unchanged, but make the model-directory fallback search actually find nested Kaggle dataset folders (so pretrained weights are used instead of the uniform fallback), which should reduce KL toward your target. All changes are minimal and keep the model/feature logic and inference semantics intact.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf implementation, which is incompatible with TF under Python 3.12 in this environment and triggers the `MessageFactory.GetPrototype` error. Then I add a deterministic seed and a safer “same-API” weight loading sequence (build model once, re-load weights per fold) without changing the architecture or inference semantics. Finally, I make the model-directory discovery slightly more robust (still only searching under `/kaggle/input`) so pretrained weights are actually found and used (moving KL down toward your target), while keeping the uniform fallback to always produce a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf crash under Python 3.12 by forcing the pure-Python protobuf implementation **before** importing TensorFlow, which resolves the `MessageFactory.GetPrototype` AttributeError in this environment. Then we keep your model/data logic intact but make weight discovery more robust by searching for the expected fold `.h5` files anywhere under `/kaggle/input`, so you’re much less likely to fall back to uniform predictions (which is what drives the current poor KL). Finally, we ensure the test generator always has the needed `spec_id/min/max` columns and that the submission probabilities are clipped and normalized to sum to 1, writing a valid `submission.csv`. All changes are minimal and only target runtime stability + moving score downward toward the target.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment override (it triggers the `MessageFactory.GetPrototype` error under TF+Python 3.12 in Kaggle). Then I keep your model/data logic the same but make weight discovery/load more robust and deterministic so the code reliably uses the pretrained 5-fold ensemble instead of falling back to uniform probabilities (which is driving the bad KL score). Finally, I add a small safety normalization at the end (already mostly present) and ensure the submission is always written as `submission.csv` with the exact required columns and rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'We fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting the protobuf implementation to `python` *before* importing TensorFlow (this is the key runtime blocker in cell 0 under this environment). Then we keep the model/data logic unchanged but make mixed-precision opt-in safe by disabling it automatically on CPU (to avoid runtime/device issues without changing inference semantics on GPU). Finally, we keep the existing 5-fold weight loading ensemble path, but make weight discovery deterministic and robust (still only searching under `/kaggle/input`) so you actually use pretrained weights instead of falling back to uniform predictions, which should move KL down toward your target.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash by removing the protobuf implementation override that is incompatible with TensorFlow under Python 3.12 (this is what triggers the `MessageFactory.GetPrototype` error). Then I keep your model/inference logic unchanged but make weights discovery stricter and safer: if weights are missing, we still produce a valid submission, but if they exist anywhere under `/kaggle/input`, we reliably use them (which should move KL down toward your target vs. uniform fallback). Finally, I add a small guard to ensure `spec_id/min/max` exist and that predictions are always finite, clipped, and normalized to sum to 1 so the submission passes validation.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting a compatible protobuf implementation and import order before importing TensorFlow (this is the only blocker preventing the notebook from running). Then I keep your model and data logic intact but make weight discovery robust and deterministic so it reliably finds the 5 fold `.h5` files under `/kaggle/input/` and avoids the uniform-probability fallback that is currently producing the poor KL score. Finally, I keep the existing probability clipping/normalization and ensure the submission is written as `submission.csv` with the exact required columns and per-row sums of 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is what triggers the `MessageFactory.GetPrototype` error under TF + Python 3.12 in Kaggle. This is a runtime-only change and does not alter your model, features, training/inference semantics, or post-processing. I also keep your existing robust weight-search and prediction normalization so the script runs end-to-end and always writes a valid `submission.csv` with row-wise probabilities summing to 1. With the crash fixed, the code should actually reach the pretrained 5-fold ensemble path (instead of failing early), which should move KL down toward your target.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the Python protobuf implementation *before* importing TensorFlow (this is the known safe import-order workaround for TF under Python 3.12 on Kaggle). Then I keep your model/inference logic intact but add a small robustness step to ensure the weight-file discovery is deterministic (sorted) and that we don’t accidentally fall back to uniform predictions when weights exist under `/kaggle/input`. Finally, I keep the existing probability clipping/normalization and submission writing, ensuring the produced `submission.csv` is valid (correct columns, correct row count, and row sums == 1).'
- What this solution (achieved 1.40995) has done: 'We fix the immediate runtime crash by removing the incompatible protobuf implementation override that triggers the `MessageFactory.GetPrototype` error under TensorFlow + Python 3.12. Then we keep your model/inference logic identical, but make the weight discovery deterministic (stable ordering) and ensure we don’t silently miss weights due to path/key mismatches, so the pretrained 5-fold ensemble is actually used (this should move KL down toward your target from the current 1.40995). Finally, we keep the existing probability clipping/normalization and guarantee `submission.csv` is written with correct columns and per-row sums of 1. All changes are minimal and only address execution + using intended weights.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "0")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402041"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128
LENGTH = 32

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
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(0)

try:
    from tensorflow.python.framework.ops import reset_default_graph
except Exception:

    def reset_default_graph():
        return None


print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) == 0:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("Using CPU (no GPU detected)")
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print("Using 1 GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = True
if MIX and len(gpus) > 0:
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled via mixed_precision policy")
    except Exception as e:
        print("Mixed precision not enabled due to:", repr(e))
else:
    print("Using full precision (or CPU)")



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
train.head()



## === cell 2
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

if NEEDTRAIN:
    from scipy import signal

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                time_temp = train[train.eeg_id == name].eeg_median.iloc[-1]
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

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

                    eeg = signal.filtfilt(b, a, eeg, axis=1)

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)

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



## === cell 3
try:
    import albumentations as albu
except Exception:
    albu = None

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

        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.mode = mode
        self.specs = specs
        self.eegs = eegs

        self._default_spec = np.zeros((300, 400), dtype=np.float32)
        self._default_eeg = np.zeros(
            (4, round(EEG_LENGTH * SFREQ), 4), dtype=np.float32
        )

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
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")
        img_map = np.zeros((100 * 300, 3), dtype=np.float32)

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r = 0
            else:
                rmin = int(row["min"]) if "min" in row.index else 0
                rmax = int(row["max"]) if "max" in row.index else rmin
                r = np.random.randint(rmin, rmax + 1) // 2

            spec = (
                self.specs.get(int(row.spec_id), self._default_spec)
                if self.specs is not None
                else self._default_spec
            )
            eeg = (
                self.eegs.get(int(row.eeg_id), self._default_eeg)
                if self.eegs is not None
                else self._default_eeg
            )

            for k in range(4):
                img = spec[r : r + 300, k * 100 : (k + 1) * 100].T
                img_eeg = eeg[:, :, k]

                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)

                img = np.clip(img, 1, 256)
                img_map = self.cmaps[img - 1]
                img_map = np.reshape(img_map, (100, 300, 3))

                img_map = img_map[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img_map.shape[1]
                    ),
                    :,
                ]
                if HIGH != 100:
                    img_map2 = np.array(
                        tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                        dtype=np.float32,
                    )
                    X[
                        j,
                        round((HIGH - img_map2.shape[0]) / 2) : round(
                            (HIGH + img_map2.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = img_map2
                else:
                    X[j, :, :, :, k] = img_map

                X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = row[TARGETS].values

        return X, X_eeg, y

    def __random_transform(self, img):
        if albu is None:
            return img
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




## === cell 4
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
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=("imagenet" if not NEEDTRAIN else None),
        input_shape=None,
    )
    base_model._name = "spectrogram_extractor"

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=("imagenet" if not NEEDTRAIN else None),
        input_shape=None,
    )
    base_model_eeg._name = "eeg_extractor"

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

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    print("Test shape", test.shape)

    files2 = os.listdir(PATH_SPEC)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(sorted(files2)):
        if i % 200 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    if "min" not in test.columns:
        test["min"] = 0
    if "max" not in test.columns:
        test["max"] = 0

    from scipy import signal

    files2 = os.listdir(PATH_EEG)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    test_ids = set(test.eeg_id.astype(int).tolist())
    for i, f in enumerate(sorted(files2)):
        if i % 200 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        if name not in test_ids:
            continue

        raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")

        time_temp = 0
        time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
        time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

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

            eeg = signal.filtfilt(b, a, eeg, axis=1)

            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)
        eegs2[name] = list_eeg



## === cell 6
if not NEEDTRAIN:

    def find_files_recursive(root, basenames_set, max_dirs=200000):
        found = {}
        seen = 0
        for dirpath, _, filenames in os.walk(root):
            seen += 1
            if seen > max_dirs:
                break
            for fn in sorted(filenames):
                if fn in basenames_set and fn not in found:
                    found[fn] = os.path.join(dirpath, fn)
            if len(found) == len(basenames_set):
                break
        return found

    expected_basenames = [f"EB2_v{VER}_f{i}.h5" for i in range(5)]
    candidate_paths = {
        bn: os.path.join(LOAD_MODELS_FROM, bn) for bn in expected_basenames
    }
    weights_ok = os.path.isdir(LOAD_MODELS_FROM) and all(
        os.path.exists(candidate_paths[bn]) for bn in expected_basenames
    )

    if not weights_ok and PLATFORM == "kaggle":
        print(
            f"Could not find all weights in {LOAD_MODELS_FROM}. Searching under /kaggle/input ..."
        )
        found_map = find_files_recursive("/kaggle/input", set(expected_basenames))
        if len(found_map) == 5:
            candidate_paths = {bn: found_map[bn] for bn in expected_basenames}
            weights_ok = True
            print("Found all fold weights:")
            for k in expected_basenames:
                print(" -", candidate_paths[k])
        else:
            print(
                f"Only found {len(found_map)}/5 weight files. Will fall back to uniform predictions."
            )
            for k, v in sorted(found_map.items()):
                print(" -", k, "->", v)

    if weights_ok:
        preds = []
        with strategy.scope():
            model = build_model()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        for i in range(5):
            print(f"Fold {i+1}")
            wbasename = f"EB2_v{VER}_f{i}.h5"
            wpath = candidate_paths[wbasename]

            try:
                model.load_weights(wpath)
            except Exception as e:
                print(f"Strict load_weights failed for {wpath}: {repr(e)}")
                print("Retrying with by_name=True, skip_mismatch=True")
                model.load_weights(wpath, by_name=True, skip_mismatch=True)

            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = np.asarray(pred, dtype=np.float64)
        pred = np.nan_to_num(pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
    else:
        n = len(test)
        pred = np.full((n, 6), 1.0 / 6.0, dtype=np.float64)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred

    probs = sub[TARGETS].to_numpy(dtype=np.float64)
    probs = np.nan_to_num(probs, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    sub[TARGETS] = probs.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row sum check (min/max):",
        sub.iloc[:, -6:].sum(axis=1).min(),
        sub.iloc[:, -6:].sum(axis=1).max(),
    )
    sub.head()
