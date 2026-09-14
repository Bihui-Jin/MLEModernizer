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

0.4747085543035703

# 6. Current score

1.48867

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf implementation and enabling TF’s built-in Keras legacy compatibility mode, which resolves the `MessageFactory.GetPrototype` error in many Kaggle TF + protobuf setups. Then I make model weight loading robust: if the expected `models202402061` directory/weights are absent, the script automatically fall back to producing a valid (normalized) “prior” submission from train label distributions rather than crashing. This keeps the core model/inference logic intact when weights exist, but guarantees an end-to-end run and a valid `submission.csv` in all cases. Finally, I keep probability post-processing strict (clip + renormalize) to prevent submission failures.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround in Kaggle TF+protobuf environments. This change is execution-critical and does not alter your model/inference logic. I also keep the existing robust fallback behavior (prior-based submission when weights are missing) and strict post-processing (clip + renormalize) to guarantee a valid `submission.csv`. No architecture/training logic is changed; only environment setup/import order is adjusted to make the notebook run end-to-end.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by removing the forced pure-Python protobuf setting (which can be incompatible with TF builds on Kaggle) while keeping `TF_USE_LEGACY_KERAS=1` set before importing TensorFlow. I also make the GPU/strategy selection robust so it won’t crash on CPU-only environments (and won’t try to force `/gpu:0` when no GPU exists). To move the score toward your target (lower is better) without changing the model, I ensure the pretrained weights directory is correctly auto-detected inside `/kaggle/input` (including nested dataset structures) so the real model predictions are used instead of the weak prior fallback (which is what likely produced ~1.49). Finally, I keep strict clipping+renormalization and write a valid `submission.csv` with correct columns and row sums of 1.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (this is execution-critical and score-neutral by itself). Then I keep your existing weight auto-detection logic but make it more robust by selecting the *shortest* detected directory that contains all 5 fold weight files, ensuring the real pretrained model is used instead of the weak prior fallback (this should move the score down toward your target). Finally, I keep your strict clipping + renormalization and submission alignment checks so the produced `submission.csv` is always valid (correct columns and each row sums to 1).'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by removing the incompatible forced pure-Python protobuf setting and instead setting `TF_USE_LEGACY_KERAS=1` (kept) plus a safe fallback to disable XLA JIT, which is a common trigger for this protobuf error in Kaggle TF builds. I also make model-weight auto-discovery more robust by not relying on a single hardcoded `models202402061` path and by selecting a directory that contains the expected fold weight files, so the script uses the real pretrained model instead of the weak prior fallback (this should move the score down toward your target). Finally, I keep strict probability clipping + renormalization and guaranteed `submission.csv` writing with the exact required columns and alignment to `sample_submission.csv`.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting a compatible protobuf environment flag **before** importing TensorFlow (and keeping legacy Keras enabled), which is required for the notebook to run end-to-end on Kaggle. To move the score down toward your target (lower is better) without changing the model itself, I make the pretrained weight auto-discovery more robust by also checking common Kaggle working/input locations and selecting a directory that truly contains all expected fold weight files, so the script uses the real model instead of the weak prior fallback. Finally, I keep your strict probability clipping + row renormalization and ensure the submission is written as a valid `submission.csv` with the exact required columns and aligned row order.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by removing the forced pure-Python protobuf override and instead keeping `TF_USE_LEGACY_KERAS=1` plus a safe protobuf backend selection that works with Kaggle’s TF wheels. Then I keep your exact model/inference logic but make the pretrained weight auto-discovery stricter and more reliable (must contain all 5 expected `.h5` files) so the notebook uses the real ensemble rather than the weak prior fallback that likely caused the 1.48867 score. Finally, I add small guards for missing spectrogram/EEG keys during test generation to prevent runtime KeyErrors, while keeping the same feature construction and output normalization to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variable *before* importing TensorFlow (and keep legacy Keras enabled), which resolves the `MessageFactory.GetPrototype` error in many Kaggle TF builds. Then I keep your existing weight auto-discovery/inference logic intact, but make sure the script can always complete by falling back to the prior-based submission if model weights are not found. Finally, I keep strict probability clipping + renormalization and ensure the produced `submission.csv` matches `sample_submission.csv` ordering and columns so Kaggle accepts it; using the real pretrained weights (when found) should move the score down toward your target versus the weak prior baseline.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible forced pure-Python protobuf override and instead relying on TF’s default protobuf while keeping `TF_USE_LEGACY_KERAS=1` set before importing TensorFlow (this is the most common stable setup on Kaggle TF wheels). Then, to move the score down toward your target (lower is better) without changing the model itself, I strengthen model-weight auto-discovery so it can find the fold `.h5` files anywhere under `/kaggle/input` (including nested dataset folders) and actually use the pretrained ensemble instead of falling back to the weak label-prior submission. Finally, I keep the existing strict probability clipping + row renormalization and submission alignment to `sample_submission.csv` to guarantee a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation to pure-Python **before** importing TensorFlow, which is the direct cause of the `MessageFactory.GetPrototype` error in this environment. Then I keep your exact model and inference logic, but make the pretrained weight auto-discovery more robust (searching common `/kaggle/input/**` nested locations and selecting a directory that truly contains all 5 fold `.h5` files) so you don’t silently fall back to the weak prior submission that produced ~1.49. Finally, I keep your strict clipping+renormalization and submission alignment to `sample_submission.csv` to guarantee a valid `submission.csv` with row sums of 1.'
- What this solution (achieved 1.48867) has done: 'We fix the immediate TensorFlow/protobuf crash by avoiding the pure-Python protobuf override (which triggers `MessageFactory.GetPrototype` on some Kaggle TF builds) while keeping `TF_USE_LEGACY_KERAS=1` set before importing TensorFlow. Then we keep your exact model/inference pipeline but make the pretrained-weight auto-discovery more reliable by also accepting common alternative fold filename patterns (so you don’t silently fall back to the weak prior submission that caused ~1.49). Finally, we keep strict clipping + renormalization and always write a valid `submission.csv` with the correct columns and row sums of 1.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False

LOAD_MODELS_FROM = "models202402061"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    candidate_a = f"/kaggle/input/{LOAD_MODELS_FROM}"
    candidate_b = f"/kaggle/input/{LOAD_MODELS_FROM}/{LOAD_MODELS_FROM}"
    LOAD_MODELS_FROM = candidate_a if os.path.isdir(candidate_a) else candidate_b

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

VER = 1

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision option not available in this TF build:", repr(e))
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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



## === cell 3
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



## === cell 4
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
            (len(indexes), 4, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        img_map = np.zeros((100 * 300, 3))

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == "test":
                r = 0
            elif self.mode == "valid":
                r = np.random.randint(row["min"], row["max"] + 1) // 2
            else:
                r = np.random.randint(row["min"], row["max"] + 1) // 2

            spec_mat = None if self.specs is None else self.specs.get(row.spec_id, None)
            eeg_mat = None if self.eegs is None else self.eegs.get(row.eeg_id, None)

            for k in range(4):
                if spec_mat is not None:
                    img = spec_mat[r : r + 300, k * 100 : (k + 1) * 100].T
                else:
                    img = np.zeros((100, 300), dtype=np.float32)

                if eeg_mat is not None:
                    img_eeg = eeg_mat[:, :, k]
                else:
                    img_eeg = np.zeros((4, round(EEG_LENGTH * SFREQ)), dtype=np.float32)

                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)
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
                    img_map = np.array(
                        tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                        dtype=np.float32,
                    )
                    X[
                        j,
                        round((HIGH - img_map.shape[0]) / 2) : round(
                            (HIGH + img_map.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = img_map
                else:
                    X[j, :, :, :, k] = img_map

                X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                X_eeg[j, :, :, k] = img_eeg
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




## === cell 5
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
    inp_eeg = tf.keras.Input(shape=(4, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_tensor=None
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
        include_top=False, weights=None, input_tensor=None
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




## === cell 6
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    if PLATFORM == "local":
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    sub_cols = [c for c in sample_sub.columns if c != "eeg_id"]
    eps = 1e-12

    def find_weight_roots(search_roots, ver=VER):
        roots = []
        patterns = [
            [f"EB2_v{ver}_f{i}.h5" for i in range(5)],
            [f"EB2_v{ver}_fold{i}.h5" for i in range(5)],
            [f"EB2_v{ver}_F{i}.h5" for i in range(5)],
            [f"EB2_v{ver}_f{i}.weights.h5" for i in range(5)],
        ]
        for base in search_roots:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for req in patterns:
                    if all(r in files for r in req):
                        roots.append((root, req))
                        break
        roots = sorted(roots, key=lambda x: (len(x[0]), x[0]))
        return roots  # list of (root, filenames)

    expected = [os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5") for i in range(5)]
    have_all_weights = all(os.path.exists(p) for p in expected)
    chosen_filenames = [f"EB2_v{VER}_f{i}.h5" for i in range(5)]

    if not have_all_weights and PLATFORM == "kaggle":
        candidates = find_weight_roots(
            [
                "/kaggle/input",
                "/kaggle/working",
                "/kaggle/working/hms-harmful-brain-activity-classification",
            ],
            VER,
        )
        if candidates:
            LOAD_MODELS_FROM, chosen_filenames = candidates[0]
            expected = [os.path.join(LOAD_MODELS_FROM, fn) for fn in chosen_filenames]
            have_all_weights = all(os.path.exists(p) for p in expected)
            print("Auto-detected model directory:", LOAD_MODELS_FROM)
            print("Auto-detected weight filenames:", chosen_filenames)

    if have_all_weights:
        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        files2 = sorted(os.listdir(PATH2))
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()

        test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

        from scipy import signal

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = sorted(os.listdir(PATH2))
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])

            if len(test[test.eeg_id == name]) > 0:
                time_temp = 0
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
                eegs2[name] = list_eeg
        print()

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
            wpath = os.path.join(LOAD_MODELS_FROM, chosen_filenames[i])
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, eps, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        test_ids = test[["eeg_id"]].copy()
        test_ids["row_idx"] = np.arange(len(test_ids))
        merged = sample_sub[["eeg_id"]].merge(test_ids, on="eeg_id", how="left")
        if merged["row_idx"].isna().any():
            missing = int(merged["row_idx"].isna().sum())
            raise ValueError(
                f"{missing} eeg_id(s) from sample_submission not found in test.csv"
            )

        pred_aligned = pred[merged["row_idx"].astype(int).values]

        sub = pd.DataFrame({"eeg_id": sample_sub.eeg_id.values})
        sub[sub_cols] = pred_aligned.astype(np.float32)

    else:
        print(
            "WARNING: Pretrained fold weights not found; writing a prior-based submission instead."
        )
        prior = train[TARGETS].mean(axis=0).values.astype(np.float64)
        prior = np.clip(prior, eps, None)
        prior = prior / prior.sum()
        sub = sample_sub.copy()
        sub[sub_cols] = np.tile(prior, (len(sub), 1)).astype(np.float32)

    sub[sub_cols] = np.clip(sub[sub_cols].values, eps, 1.0)
    sub[sub_cols] = sub[sub_cols].div(sub[sub_cols].sum(axis=1), axis=0)

    sub.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")
    print("Submission shape", sub.shape)
    print(
        "Row prob sum stats:",
        float(sub[sub_cols].sum(axis=1).min()),
        float(sub[sub_cols].sum(axis=1).max()),
    )
    sub.head()
