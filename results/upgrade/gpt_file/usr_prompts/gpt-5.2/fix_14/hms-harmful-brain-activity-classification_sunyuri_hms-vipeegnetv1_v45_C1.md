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

0.4927873707466413

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix two runtime blockers so the notebook runs end-to-end and writes a valid `submission.csv`: (1) avoid the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by using the Kaggle CPU-only path (no CUDA device forcing) and setting safe TF env flags before importing TF, and (2) replace the invalid `tf.nn.l2_normalize` calls on KerasTensors with a Keras `Lambda` layer so Functional model construction works in Keras 3. I also add a small fallback so if the external pretrained model folder isn’t present, the script still produces a properly-normalized uniform-probability submission rather than crashing (score be worse, but it “yield” a valid submission). Core model architecture, data generation, and ensembling logic are preserved; only compatibility/robustness fixes are applied.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the correct protobuf environment flags (including `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`) before importing TensorFlow and by avoiding mixed-precision optimizer experimental options that can trigger the issue in TF/Keras 3 stacks. I also remove the hard dependency on `albumentations` (it isn’t installed by default) by providing a tiny no-op fallback so the script always runs. Finally, to move the score toward your target (lower is better) without changing the core model, I ensure the pretrained fold weights are found from Kaggle datasets (typical path is `/kaggle/input/<dataset>/...`) by searching both the provided directory and its immediate subdirectories; this should restore actual model predictions instead of the uniform fallback that caused the 1.40995 score.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by switching the protobuf implementation setting to the compiled backend (and unsetting the python implementation), which is the stable configuration for TF in Kaggle’s Python 3.12 environment. This is a runtime-blocker fix only and does not change model logic. Then, to move the score down toward your target (lower is better), I strengthen the fold-weights discovery so the script actually finds and loads the pretrained `.h5` files even when the Kaggle dataset has nested directories—avoiding the uniform-probability fallback that caused the poor 1.40995 score. Finally, I keep the submission formatting/probability normalization intact to ensure Kaggle accepts the `.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the stable workaround in Kaggle’s Py3.12 environment. Then I keep your model/data logic unchanged, but make weight discovery a bit more robust by also checking common Kaggle input roots (`/kaggle/input/<dataset>/...`) so the script actually loads your trained fold `.h5` files instead of falling back to uniform predictions (which is what’s driving the poor 1.40995 score). Finally, I keep the submission normalization guard to ensure row probabilities sum to 1 and Kaggle accepts the CSV.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf runtime crash by changing the protobuf environment configuration to the compiled backend (and unsetting the pure-Python override) before importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in Kaggle’s Py3.12 stack. I keep the model/data logic unchanged, but I also make model weight discovery a bit more robust by allowing `.weights.h5` and single-fold files, so the code is much more likely to actually load your pretrained folds instead of falling back to uniform predictions (which is driving the 1.40995 score). Finally, I keep the existing probability clipping/renormalization to guarantee the submission is valid for KL-divergence evaluation.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround in Kaggle’s Py3.12 environment. This unblocks the notebook so it can actually load the pretrained fold weights and generate non-uniform predictions, which should move the KL score down toward your target (lower is better) from the current uniform-like 1.40995. I keep your model, generator, preprocessing, and ensembling logic unchanged, only adding a small safety normalization to ensure submission rows sum to 1. The output be a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by switching to the stable compiled-protobuf setting for Kaggle Py3.12 (unset the pure-Python override) *before* importing TensorFlow, which is the current runtime blocker. Then I keep your model/data logic the same, but make weight discovery slightly more robust by also accepting common alternate fold filenames (e.g., `EFN_v{VER}_f{fold}.h5`) so the notebook is much more likely to actually load pretrained weights instead of falling back to uniform predictions (which is what’s driving the current 1.40995 KL). Finally, I keep the existing probability clipping+renormalization to guarantee submission validity (rows sum to 1) and always write `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variables *before* importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in Kaggle’s Py3.12 environment. Then I keep your model/data logic identical, but make weight discovery robust to common Kaggle dataset folder nesting so the pretrained fold `.h5` files are actually found and loaded (avoiding the uniform fallback that is driving the current 1.40995 KL). Finally, I keep the existing probability clipping and renormalization to guarantee each row sums to 1 and the submission is accepted as valid `submission.csv`. These changes are minimal and should move the score down toward your target by restoring real model predictions.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by switching to the compiled protobuf backend (and unsetting the pure-Python override) *before* importing TensorFlow, which is the known stable configuration on Kaggle’s Python 3.12 images. This is a runtime blocker and score-neutral by itself, but it enables the model to load and run. I also keep your weight discovery/loading logic intact, so you actually use pretrained fold weights instead of falling back to uniform predictions (which is what produces the poor 1.40995 KL). Finally, I keep the existing probability clipping + renormalization to guarantee the submission rows sum to 1 and Kaggle accepts the CSV.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by setting the protobuf environment to the pure-Python implementation *before* importing TensorFlow (this is a known stable workaround in Kaggle’s Py3.12 image). I keep your model/data pipeline intact, but make sure `TARGETS` exactly matches the required submission columns (and order) by reading it from `sample_submission.csv`, preventing any subtle column/order mismatch that can hurt KL scoring. I also ensure weight discovery still works as you intended and that predictions are clipped and renormalized to valid per-row probabilities (required for Kaggle acceptance). These changes are minimal and should move the score down toward the target by enabling real model inference instead of crashing/degenerating.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by adjusting the protobuf environment configuration *before* importing TensorFlow to use the compiled backend (and explicitly unsetting the pure-Python override that triggers this error on Kaggle Py3.12). This is a runtime-blocker fix that preserves your model/data logic and enables the notebook to actually run inference and load fold weights (which should improve KL from the current uniform-like 1.40995 toward your target). I keep your weight discovery/ensembling and submission normalization intact, only adding a tiny robustness guard to ensure the environment variables are set before any TF-related import happens. The script still always write a valid `submission.csv` with probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by setting the protobuf environment to the pure-Python implementation *before* importing TensorFlow, which is the most reliable workaround on Kaggle’s Python 3.12 images. I also keep your model/inference logic intact but add a tiny, score-improving robustness step to search common Kaggle input locations for your pretrained fold `.h5` weights so you don’t fall back to uniform predictions (which is what drives the current 1.40995 KL). Finally, I keep the required probability clipping+renormalization so each row sums to 1 and the generated `submission.csv` is always valid.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import gc
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep CPU-only for stability

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240208"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 100

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

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

try:
    from tensorflow.python.framework.ops import reset_default_graph
except Exception:

    def reset_default_graph():
        return None


print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(
        device="/cpu:0" if len(gpus) == 0 else "/gpu:0"
    )
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = False
print("Using full precision")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if PLATFORM == "local":
    sample_sub = pd.read_csv(
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
elif PLATFORM == "kaggle":
    sample_sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )

TARGETS = sample_sub.columns[1:].tolist()
print("Submission targets (ordered):", TARGETS)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

print("Train shape:", df.shape)


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
            eeg_default = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
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
    import albumentations as albu  # type: ignore
except Exception:

    class _NoOpCompose:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, image=None, **kwargs):
            return {"image": image}

    class _NoOpAlbu:
        Compose = _NoOpCompose

        class HorizontalFlip:
            def __init__(self, *args, **kwargs):
                pass

        class CoarseDropout:
            def __init__(self, *args, **kwargs):
                pass

    albu = _NoOpAlbu()

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
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            eeg_temp = row.eeg_id

            if self.mode == "test":
                spectrogram_temp = row.spectrogram_id
                spec_start = 0
                eeg_start = 0
            else:
                spectrogram_temp = row.spec_id
                rows = df[
                    (df.spectrogram_id == spectrogram_temp) & (df.eeg_id == eeg_temp)
                ].reset_index(drop=True)
                row = rows.iloc[np.random.permutation(len(rows))[0]]
                spec_start = round(row.spectrogram_label_offset_seconds / 2)
                eeg_start = round(row.eeg_label_offset_seconds * SFREQ)

            for k in range(4):
                img = self.specs[spectrogram_temp][
                    spec_start : spec_start + 300, k * 100 : (k + 1) * 100
                ].T
                img_eeg = self.eegs[eeg_temp][
                    :, eeg_start : (eeg_start + round(SFREQ * 50)), k
                ]
                img_eeg = img_eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = np.round(
                    (img - self.cmin) / (self.cmax - self.cmin) * 256
                ).astype(np.int16)
                img = np.clip(img, 1, 256)
                img_flat = np.reshape(img, (img.shape[0] * img.shape[1]))
                img_map = self.cmaps[img_flat - 1]
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

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y_data = row[TARGETS].values
                y[j] = y_data / sum(y_data)

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




## === cell 5
try:
    import efficientnet.tfkeras as efn

    _EFN_BACKEND = "efficientnet.tfkeras"
except Exception as e:
    efn = None
    _EFN_BACKEND = "tf.keras.applications"
    print(
        "Could not import efficientnet.tfkeras, falling back to tf.keras.applications EfficientNet. Error:",
        repr(e),
    )


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


def _make_efficientnet_b0():
    if efn is not None:
        return efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    return tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(None, None, 3)
    )


def _make_efficientnet_b2():
    if efn is not None:
        return efn.EfficientNetB2(include_top=False, weights=None, input_shape=None)
    return tf.keras.applications.EfficientNetB2(
        include_top=False, weights=None, input_shape=(None, None, 3)
    )


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = _make_efficientnet_b0()
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

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm_spec"
    )(x)

    base_model_eeg = _make_efficientnet_b2()
    base_model_eeg._name = "eeg_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model_eeg.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model_eeg.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0_eeg = inp_eeg[:, :, :, :1]
    x1_eeg = inp_eeg[:, :, :, 1:2]
    x2_eeg = inp_eeg[:, :, :, 2:3]
    x3_eeg = inp_eeg[:, :, :, 3:4]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm_eeg"
    )(x_eeg)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 6
def _resolve_weights_dir(base_dir: str) -> str | None:
    """Find a directory that contains fold weights.
    Score improvement vs uniform fallback: also search likely nested Kaggle dataset dirs,
    but keep the same fold ensembling logic once weights are found.
    """
    if not base_dir:
        return None

    expected_sets = []
    expected_sets.append([f"EB2_v{VER}_f{i}.h5" for i in range(5)])
    expected_sets.append([f"EB2_v{VER}_f{i}.weights.h5" for i in range(5)])
    expected_sets.append([f"EFN_v{VER}_f{i}.h5" for i in range(5)])
    expected_sets.append([f"EFN_v{VER}_f{i}.weights.h5" for i in range(5)])
    expected_sets.append([f"fold{i}.h5" for i in range(5)])

    def count_present(p: str, names: list[str]) -> int:
        return sum(os.path.exists(os.path.join(p, n)) for n in names)

    candidates = []
    if os.path.isdir(base_dir):
        candidates.append(base_dir)

    if PLATFORM == "kaggle":
        if not base_dir.startswith("/kaggle/"):
            candidates.append(os.path.join("/kaggle/input", base_dir))

        bn = os.path.basename(base_dir.rstrip("/"))
        candidates.append(os.path.join("/kaggle/input", bn))
        candidates.append(os.path.join("/kaggle/input", bn, bn))

        if os.path.isdir("/kaggle/input"):
            try:
                for d in os.listdir("/kaggle/input"):
                    root = os.path.join("/kaggle/input", d)
                    if os.path.isdir(root):
                        candidates.append(root)
            except Exception:
                pass

    best_dir = None
    best_score = 0

    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        for names in expected_sets:
            sc = count_present(c, names)
            if sc > best_score:
                best_score = sc
                best_dir = c

    max_depth = 5
    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        c = os.path.abspath(c)
        for root, dirs, files in os.walk(c):
            rel = os.path.relpath(root, c)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue
            for names in expected_sets:
                sc = count_present(root, names)
                if sc > best_score:
                    best_score = sc
                    best_dir = root

    if best_dir is None or best_score == 0:
        return None
    return os.path.abspath(best_dir)


def _fold_weight_path(weights_dir: str, fold: int) -> str | None:
    """Return existing weight filename for fold (accept several common patterns)."""
    cands = [
        os.path.join(weights_dir, f"EB2_v{VER}_f{fold}.h5"),
        os.path.join(weights_dir, f"EB2_v{VER}_f{fold}.weights.h5"),
        os.path.join(weights_dir, f"EFN_v{VER}_f{fold}.h5"),
        os.path.join(weights_dir, f"EFN_v{VER}_f{fold}.weights.h5"),
        os.path.join(weights_dir, f"fold{fold}.h5"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    test_eeg_ids = set(test.eeg_id.values.tolist())
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if name in test_eeg_ids:
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

    preds = []
    model = build_model()
    test_gen = DataGenerator(
        test, shuffle=False, batch_size=32, mode="test", specs=spectrograms2, eegs=eegs2
    )

    weights_dir = _resolve_weights_dir(LOAD_MODELS_FROM)
    have_weights = weights_dir is not None

    if have_weights:
        print(f"\nUsing weights from: {weights_dir}")

        loaded = 0
        for i in range(5):
            wpath = _fold_weight_path(weights_dir, i)
            if wpath is None:
                print(f"Fold {i + 1}: weight file not found, skipping.")
                continue
            print(f"\nFold {i + 1} loading: {wpath}")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
            loaded += 1

        if loaded > 0:
            pred = np.mean(preds, axis=0)
        else:
            print(
                "\nWARNING: weights_dir resolved but no expected fold files were found."
            )
            pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        print(
            f"\nWARNING: Could not find expected weight files under: {LOAD_MODELS_FROM}"
        )
        print(
            "Writing uniform-probability submission to ensure a valid .csv is produced."
        )
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)

    print("\nTest preds shape", pred.shape)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred

    p = sub[TARGETS].values.astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    denom = p.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    p = p / denom
    sub[TARGETS] = p.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row sum stats:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
    )
    print(sub.head())
