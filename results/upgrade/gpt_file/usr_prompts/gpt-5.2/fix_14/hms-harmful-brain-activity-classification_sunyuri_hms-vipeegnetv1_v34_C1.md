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

0.4687277362086465

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in Kaggle. Next, I make the model-weights loading robust by auto-detecting the actual directory/file pattern under `/kaggle/input` (or falling back to writing a valid, normalized submission from `sample_submission.csv` if no weights exist), so the notebook always produces `submission.csv`. Finally, I keep the model/inference logic intact and only add minimal guards (path checks, probability normalization, column ordering) to ensure the submission matches the required format and sums to one.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars *before any TensorFlow-related import and before Python loads `google.protobuf`*, and I also add a safe fallback to CPU if no GPU is available (so the run doesn’t die on GPU selection). To move the score toward your target (lower is better), I correct a clear preprocessing bug in the spectrogram normalization where you were dividing by `std**2` instead of `std` (this is a minimal, semantics-preserving fix that should materially improve calibration and KL). I also ensure the test dataframe is sorted by `eeg_id` before inference so predictions align deterministically with the submission rows. Finally, I keep model architecture and the overall inference loop intact, while guaranteeing the submission is always valid (correct columns, normalized probabilities, `.csv` suffix).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by enforcing a compatible protobuf runtime setup before importing TensorFlow and (when needed) pinning the Python protobuf package usage via environment variables in a way that works on Kaggle’s TF build. Then I fix a clear data bug in the spectrogram crop logic (the code was computing a crop start from a 600-wide assumption while the actual spectrogram slice is 300 wide), which was silently mis-cropping/zero-padding and hurting KL score. Finally, I keep your model and inference loop intact but add small robustness guards (safe std normalization, deterministic ordering, and always writing a valid normalized submission.csv).'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* TensorFlow (and any protobuf-using modules) are imported, and by safely falling back if the env var setup is ignored. I also fix a clear EEG preprocessing bug where channel 0/5 were left as zeros (only channels 1–4 were filled), which hurts predictions and KL; this keeps the model architecture and inference loop unchanged. Additionally, I make the test-data loading robust and deterministic by iterating over `test` rows (not all files) and by sorting by `eeg_id`, while keeping identical feature extraction semantics. Finally, I ensure a valid `submission.csv` is always written with correct columns and per-row probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf implementation env vars are set before *any* protobuf/TensorFlow-related import, and by adding a safe fallback to the pure-Python protobuf path even if something pre-imported protobuf. Then, to move the KL score down toward your target without changing the model/training core, I apply a minimal calibration fix at inference: clip + renormalize plus a small blend with a uniform prior (this is score-improving for overconfident predictions under KL). Finally, I make the script always write a valid `submission.csv` (correct columns, sorted `eeg_id`, probabilities sum to 1), even if weights are missing.'
- What this solution (achieved 1.40995) has done: 'You’re hitting the known TensorFlow/protobuf incompatibility in Kaggle’s Python 3.12 environment: even with the env vars set, importing `tensorflow` can still crash with `MessageFactory.GetPrototype`. I fix this by forcing the pure-Python protobuf implementation *and* importing `google.protobuf` early (to ensure the correct backend is selected) before importing TensorFlow. Then I keep your model/inference intact but add a small, metric-aligned calibration improvement: search a few blend-to-uniform `alpha` values on a held-out slice of training predictions (no training loop change) to pick a better alpha than the hardcoded 0.08, which should reduce KL toward your target. Finally, I keep deterministic ordering and guarantee a valid normalized `submission.csv` is written.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by ensuring protobuf is configured before any protobuf/TensorFlow import and by applying a small, compatibility-oriented monkey-patch for `MessageFactory.GetPrototype` (used by some TF builds) to map to `GetMessageClass`. This is the minimal change that unblocks execution in Kaggle Python 3.12 without altering your model, data pipeline, or training/inference logic. After TF imports successfully, the rest of your script run as-is and produce a valid `submission.csv` with correctly ordered columns and row-wise probabilities summing to 1. This should also allow your existing calibration/tuning logic to function, which is expected to reduce KL (lower is better) toward your target.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4687), so the smallest likely win is to fix a test-time mismatch in how the spectrogram time window is chosen. Right now, `mode="test"` forces `r=0`, which uses the very start of each spectrogram, while training/valid uses a center-ish window; this distribution shift can heavily hurt KL. I change only the `DataGenerator` test behavior to take a deterministic centered crop (matching the valid path more closely) without altering the model, loss, or inference loop. Everything else (weights loading, calibration, normalization, submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.4687), so we should make the smallest changes likely to reduce KL without altering your model or training approach. The biggest remaining distribution-shift bug is that test EEG crops always start at time 0, while train uses a per-eeg median offset; I change test EEG extraction to use a deterministic centered crop (same crop window length, just centered), matching the intent of labeling the central 10 seconds and reducing mismatch. I also make the test spectrogram crop deterministic-center in a way consistent with your 300-frame slice and keep your existing calibration/blend logic unchanged. These are minimal inference-time preprocessing alignment fixes and should move the score down toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far from the target (0.4687), so we should make the smallest inference-time changes that reduce distribution shift without changing the model or training logic. The biggest remaining mismatch is that test spectrogram crops use a centered 300-frame window, but the subsequent 256-frame crop is also centered, effectively “double-centering” and potentially discarding informative edges inconsistently with train/valid; we instead make the 256 crop start deterministic and aligned to the 300 crop (left-aligned within that 300) for all modes. Additionally, we ensure the `valid`/`train` r computation is clamped to the available spectrogram length to avoid silent padding differences that can hurt calibration. Everything else (architecture, weights, ensembling, alpha tuning, normalization, submission writing) remains intact.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4687), so we should make the smallest inference-time fixes that reduce train↔test preprocessing mismatch without changing the model or training loops. The biggest remaining issue is that test spectrogram crops are always centered, while train/valid use label-offset-driven crops; this distribution shift can severely hurt KL, so we compute a deterministic per-`spec_id` crop for test using a proxy derived from the training metadata (median label offset for that spectrogram). We also apply the same idea to EEG by building a per-`eeg_id` crop start from training `eeg_label_offset_seconds` and use it when extracting the 20.48s window in test (falling back to center if unseen). These changes keep architecture/loss/inference intact, but align test windows to the training labeling convention, which is expected to move KL down toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import google.protobuf  # noqa: F401

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402031"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128  # 128
LENGTH = 256  # 256

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
    print(
        "Mixed precision: not forcing experimental optimizer options (stability fix)."
    )
else:
    print("Using full precision")




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

spec_r_by_specid = (
    df.groupby("spectrogram_id")[["spectrogram_label_offset_seconds"]]
    .median()
    .rename(columns={"spectrogram_label_offset_seconds": "spec_median"})
)
spec_r_by_specid["spec_r"] = (spec_r_by_specid["spec_median"] // 2).astype(np.int32)
spec_r_by_specid = spec_r_by_specid["spec_r"].to_dict()

eeg_median_by_eegid = (
    df.groupby("eeg_id")[["eeg_label_offset_seconds"]]
    .median()["eeg_label_offset_seconds"]
    .to_dict()
)




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
        spec_r_by_specid=None,
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
        self.spec_r_by_specid = spec_r_by_specid or {}
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

            spec0 = self.specs[row.spec_id]
            max_r0 = max(spec0.shape[0] - 300, 0)

            if self.mode == "test":
                r = int(self.spec_r_by_specid.get(int(row.spec_id), max_r0 // 2))
            elif self.mode == "valid":
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = np.random.randint(row["min"], row["max"] + 1) // 2

            r = int(np.clip(r, 0, max_r0))

            for k in range(4):
                spec = self.specs[row.spec_id]
                r0 = r
                r1 = r0 + 300
                s = spec[r0 : min(r1, spec.shape[0]), k * 100 : (k + 1) * 100].T
                if s.shape[1] < 300:
                    pad = 300 - s.shape[1]
                    s = np.pad(
                        s, ((0, 0), (0, pad)), mode="constant", constant_values=0.0
                    )

                img = s
                img_eeg = self.eegs[row.eeg_id][:, :, k]

                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)
                img = np.clip(img, 1, 256)
                img_map = self.cmaps[img - 1]
                img_map = np.reshape(img_map, (100, 300, 3))

                start = 0
                end = min(start + LENGTH, img_map.shape[1])
                img_map = img_map[:, start:end, :]
                if img_map.shape[1] < LENGTH:
                    pad = LENGTH - img_map.shape[1]
                    img_map = np.pad(
                        img_map,
                        ((0, 0), (0, pad), (0, 0)),
                        mode="constant",
                        constant_values=0.0,
                    )

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

                X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / 0.229
                X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / 0.224
                X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / 0.225

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]
                X_eeg[j, 0, :, k] = 0.5 * (img_eeg[0, :] + img_eeg[1, :])
                X_eeg[j, 5, :, k] = 0.5 * (img_eeg[2, :] + img_eeg[3, :])

                mu = np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                sd = np.std(X_eeg[j, :, :, k], 1, keepdims=True)
                X_eeg[j, :, :, k] = (X_eeg[j, :, :, k] - mu) / (sd + 1e-6)

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
def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None
    )
    base_model._name = "spectrogram_extractor"

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB2(
        include_top=False, weights=None, input_shape=None
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
def _resolve_models_dir(base_dir: str) -> str | None:
    if os.path.isdir(base_dir):
        return base_dir
    parent = os.path.dirname(base_dir.rstrip("/"))
    if os.path.isdir(parent):
        for name in os.listdir(parent):
            cand = os.path.join(parent, name)
            if os.path.isdir(cand):
                for i in range(5):
                    if os.path.exists(os.path.join(cand, f"EB2_v{VER}_f{i}.h5")):
                        return cand
                if any(fn.endswith(".h5") for fn in os.listdir(cand)):
                    return cand
    return None


def _write_fallback_submission(
    test_df: pd.DataFrame, targets: list[str], out_path: str = "submission.csv"
):
    if PLATFORM == "local":
        ss_path = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        ss_path = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    ss = pd.read_csv(ss_path)
    ss = ss[["eeg_id"] + list(targets)]
    ss = ss[ss["eeg_id"].isin(test_df["eeg_id"].values)].copy()
    ss = ss.sort_values("eeg_id").reset_index(drop=True)
    ss[targets] = 1.0 / len(targets)
    ss.to_csv(out_path, index=False)
    print(f"Wrote fallback {out_path} with uniform probabilities. Shape={ss.shape}")


def _calibrate_and_normalize(
    pred: np.ndarray, eps: float = 1e-9, alpha: float = 0.08
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    if alpha > 0:
        u = np.full_like(pred, 1.0 / pred.shape[1])
        pred = (1.0 - alpha) * pred + alpha * u
        pred = pred / pred.sum(axis=1, keepdims=True)

    return pred.astype(np.float32)


def _kl_divergence(true_p: np.ndarray, pred_p: np.ndarray, eps: float = 1e-9) -> float:
    true_p = np.asarray(true_p, dtype=np.float64)
    pred_p = np.asarray(pred_p, dtype=np.float64)
    true_p = np.clip(true_p, eps, 1.0)
    pred_p = np.clip(pred_p, eps, 1.0)
    true_p = true_p / true_p.sum(axis=1, keepdims=True)
    pred_p = pred_p / pred_p.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(true_p * (np.log(true_p) - np.log(pred_p)), axis=1)))


def _tune_alpha_on_train_slice(
    model: tf.keras.Model,
    train_df: pd.DataFrame,
    specs_train: dict,
    eegs_train: dict,
    weights_path: str,
    candidate_alphas=(0.00, 0.04, 0.08, 0.12, 0.16),
    max_items: int = 256,
) -> float:
    model.load_weights(weights_path)

    valid_df = train_df.sort_values("eeg_id").head(max_items).reset_index(drop=True)
    gen = DataGenerator(
        valid_df,
        shuffle=False,
        batch_size=32,
        mode="valid",
        specs=specs_train,
        eegs=eegs_train,
    )
    pred = model.predict(gen, verbose=0)
    y_true = valid_df[list(TARGETS)].values.astype(np.float32)

    best_a, best_kl = None, None
    for a in candidate_alphas:
        pred_a = _calibrate_and_normalize(pred, alpha=float(a))
        kl = _kl_divergence(y_true, pred_a)
        if (best_kl is None) or (kl < best_kl):
            best_kl = kl
            best_a = float(a)
    print(
        f"Alpha tuning (slice n={len(valid_df)}): best_alpha={best_a} best_KL={best_kl:.6f}"
    )
    return float(best_a)




## === cell 6
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    test = test.sort_values("eeg_id").reset_index(drop=True)

    if PLATFORM == "local":
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
    else:
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    spectrograms2 = {}
    uniq_spec_ids = test["spec_id"].unique().tolist()
    print(f"Loading {len(uniq_spec_ids)} test spectrogram parquets (unique spec_id)")
    for i, sid in enumerate(uniq_spec_ids):
        if i % 100 == 0:
            print(i, ", ", end="")
        fpath = os.path.join(PATH_SPEC, f"{int(sid)}.parquet")
        tmp = pd.read_parquet(fpath)
        spectrograms2[int(sid)] = tmp.iloc[:, 1:].values

    from scipy import signal

    if PLATFORM == "local":
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    eegs2 = {}
    uniq_eeg_ids = test["eeg_id"].unique().tolist()
    print(f"\nLoading {len(uniq_eeg_ids)} test eeg parquets (unique eeg_id)")

    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, eid in enumerate(uniq_eeg_ids):
        if i % 100 == 0:
            print(i, ", ", end="")
        fpath = os.path.join(PATH_EEG, f"{int(eid)}.parquet")
        raw_eeg = pd.read_parquet(fpath)
        name = int(eid)

        n_total = raw_eeg.shape[0]  # typically 50s * 200 = 10000
        n_win = int(round(EEG_LENGTH * 200))  # keep original 200Hz indexing as before
        if name in eeg_median_by_eegid:
            time_temp = float(eeg_median_by_eegid[name])
            time_start = int(round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200))
            time_start = int(np.clip(time_start, 0, max(n_total - n_win, 0)))
        else:
            time_start = max((n_total - n_win) // 2, 0)
        time_stop = min(time_start + n_win, n_total)

        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

        list_eeg = list()
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

    resolved_models_dir = _resolve_models_dir(LOAD_MODELS_FROM)
    if resolved_models_dir is None:
        print(f"\nWARNING: Could not find model weights under {LOAD_MODELS_FROM}.")
        _write_fallback_submission(test, list(TARGETS), out_path="submission.csv")
    else:
        print(f"\nUsing weights directory: {resolved_models_dir}")

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
            spec_r_by_specid=spec_r_by_specid,  # uses training-derived proxy when available
        )

        try:
            if PLATFORM == "kaggle":
                spec_train_path = "/kaggle/input/brain-spectrograms/specs.npy"
                eeg_train_path = "/kaggle/input/brain-eegs/eegs.npy"
            else:
                spec_train_path = "./input/brain-spectrograms/specs.npy"
                eeg_train_path = "./input/brain-eegs/eegs.npy"

            can_tune = os.path.exists(spec_train_path) and os.path.exists(
                eeg_train_path
            )
            weights0 = os.path.join(resolved_models_dir, f"EB2_v{VER}_f0.h5")
            if can_tune and os.path.exists(weights0):
                specs_train = np.load(spec_train_path, allow_pickle=True).item()
                eegs_train = np.load(eeg_train_path, allow_pickle=True).item()
                tuned_alpha = _tune_alpha_on_train_slice(
                    model=model,
                    train_df=train,
                    specs_train=specs_train,
                    eegs_train=eegs_train,
                    weights_path=weights0,
                    candidate_alphas=(0.00, 0.04, 0.08, 0.12, 0.16),
                    max_items=256,
                )
            else:
                tuned_alpha = 0.08
                print("Alpha tuning skipped (missing caches/weights). Using alpha=0.08")
        except Exception as e:
            tuned_alpha = 0.08
            print(f"Alpha tuning failed ({type(e).__name__}: {e}). Using alpha=0.08")

        for i in range(5):
            weights_path = os.path.join(resolved_models_dir, f"EB2_v{VER}_f{i}.h5")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Expected weights not found: {weights_path}. "
                    f"Found dir={resolved_models_dir} files={sorted([x for x in os.listdir(resolved_models_dir) if x.endswith('.h5')])[:50]}"
                )
            print(f"\nFold {i + 1}")
            model.load_weights(weights_path)
            pred_fold = model.predict(test_gen, verbose=1)
            preds.append(pred_fold)

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = _calibrate_and_normalize(pred, eps=1e-9, alpha=float(tuned_alpha))

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[list(TARGETS)] = pred
        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)

        print("Submission shape", sub.shape)
        sums = sub[list(TARGETS)].sum(axis=1)
        print(
            "Row prob sums (min/mean/max):",
            float(sums.min()),
            float(sums.mean()),
            float(sums.max()),
        )
        print("Wrote submission.csv")
