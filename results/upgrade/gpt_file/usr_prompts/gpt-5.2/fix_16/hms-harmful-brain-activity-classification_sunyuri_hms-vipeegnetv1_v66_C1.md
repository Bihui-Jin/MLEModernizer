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

0.5383913144990755

# 6. Current score

1.12539

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'We fix two execution blockers while keeping the modeling/prediction logic intact: (1) the protobuf `MessageFactory.GetPrototype` crash is triggered by forcing the pure-Python protobuf implementation, so we remove that environment override; (2) the pretrained fold weight paths don’t exist in this Kaggle dataset, so we automatically fall back to generating a valid submission using a safe uniform probability distribution when weights are missing (rather than crashing). We also make the code robust to either `/kaggle/input/...` or `/kaggle/data/...` dataset roots, and keep the submission format/row-sum constraints enforced. These changes are score-neutral in the “weights missing” case (the only way to produce a submission) and preserve the original ensemble inference when weights are available.'
- What this solution (achieved 1.41937) has done: 'I remove the protobuf environment override that triggers the TensorFlow/protobuf `MessageFactory.GetPrototype` crash so imports work reliably in Kaggle’s runtime. Then I fix the pandas error by avoiding `keepdims` in `DataFrame.sum()` (use NumPy for row-wise normalization instead), ensuring probabilities always sum to 1. Finally, I keep the existing fallback behavior (global class-prior when weights are missing) so the notebook always produces a valid `submission.csv` end-to-end with the required columns and row-sum constraint.'
- What this solution (achieved 1.47425) has done: 'I remove the protobuf environment override that forces the C++/cpp protobuf implementation, because it directly causes TensorFlow to crash on import in this Kaggle runtime. Then, since later cells fail only because the early TensorFlow import aborted (leaving `NEEDTRAIN`, `tf`, etc. undefined), I keep the same core logic but ensure the script runs end-to-end by making the TensorFlow import reliable. Finally, I keep the existing inference + weight-loading behavior, including the safe fallback to patient/global priors when weights are missing, and ensure `submission.csv` is always written with correct columns and row-normalized probabilities.'
- What this solution (achieved 1.47425) has done: 'I fix the TensorFlow/protobuf import crash that’s blocking the whole pipeline by forcing the stable pure-Python protobuf implementation *before* importing TensorFlow in this Kaggle Python 3.12 runtime. Then I keep the existing inference logic intact, but make the missing-weights fallback more score-aligned by using the already-computed **patient prior with global backoff** (and strict row-normalization), ensuring the KL metric doesn’t blow up from invalid/unnormalized probabilities. Finally, I keep the same paths and submission schema, and ensure `submission.csv` is always produced end-to-end.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment override that forces the pure-Python protobuf runtime, which is what triggers the `MessageFactory.GetPrototype` AttributeError in this Kaggle Python 3.12 environment. Then I make the missing-weights fallback stronger (score-improving but still minimal) by predicting the **global class prior computed on the consolidated per-eeg targets** (instead of raw per-row vote counts), which better matches the training label distribution used by the model. Finally, I keep the existing model/inference logic unchanged when weights are available, and ensure the submission is always normalized, clipped, and written to `submission.csv` with the required columns.'
- What this solution (achieved 0.82395) has done: 'I fix the execution blocker causing the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the stable setup for this Kaggle Py3.12 runtime). Then I keep your existing inference logic unchanged when model weights exist, but make the “no weights found” fallback more score-aligned by using a blended patient-prior-with-global-backoff (instead of only global prior), while still strictly clipping and row-normalizing to satisfy the KL metric and submission rules. Finally, I keep paths the same but make them robust to both `/kaggle/input/...` and `/kaggle/data/...`, and ensure `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 0.9444) has done: 'I fix the TensorFlow/protobuf import crash by removing the environment override that forces the pure-Python protobuf runtime, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle Py3.12 environment. To move the score toward your target (lower is better), I keep the same “no weights found” fallback behavior but adjust the patient/global prior blending to be less patient-dominant (more global), which typically reduces overconfident mis-calibration and improves KL. I also make test-time behavior deterministic (no random patient-row mixing during inference) without changing the model, just the generator’s row selection, improving stability and usually KL. Finally, I keep strict clipping + row-normalization and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.01516) has done: 'I fix the TensorFlow/protobuf crash by forcing the stable pure-Python protobuf runtime *before* importing TensorFlow, which is the direct root cause of the `MessageFactory.GetPrototype` error in this Kaggle Py3.12 environment. Then I keep your inference/core model logic intact, but ensure the “no weights found” fallback always produces a valid, well-calibrated submission by blending patient/global priors and strictly clipping + renormalizing row-wise (score-improving vs. uniform and avoids invalid KL). I also make test-time generator behavior deterministic (no RNG dependence) while preserving the same mixing idea, improving stability without changing the model. Finally, the script always write `submission.csv` with the exact required columns and probabilities that sum to 1.'
- What this solution (achieved 1.12539) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the pure-Python protobuf implementation, which triggers the `MessageFactory.GetPrototype` error in this Kaggle Py3.12 runtime. Then I keep the exact model/inference logic, but improve the “no weights found” fallback (the only path that currently runs reliably here) by computing patient priors on the consolidated per-`eeg_id` targets (instead of per-row overlaps), with a global backoff and a mild smoothing floor to reduce KL from overconfident zeros. Finally, I keep strict clipping + row-normalization and always write a valid `submission.csv` with the required columns and row sums of 1.'
- What this solution (achieved 1.12539) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` failure in this Kaggle Py3.12 environment). Then I keep your model/inference logic unchanged, but ensure the “no weights found” fallback is still used safely and produces valid probabilities (clipped + row-normalized) so the KL metric doesn’t blow up. I also keep paths robust to both `/kaggle/input/...` and `/kaggle/data/...` as you already intended, and guarantee `submission.csv` is written with the required columns and row sums of 1. These changes should unblock execution and, since your current score is worse than target (lower is better), the improved fallback calibration should move score downward toward the target.'

# 9. Code solution

## === cell 0
import os
import glob

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from scipy import signal

print("TensorFlow version =", tf.__version__)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402182"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 128
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
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision option not applied:", repr(e))
else:
    print("Using full precision")


def _first_existing_path(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


TRAIN_CSV = _first_existing_path(
    [
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
        "/kaggle/data/hms-harmful-brain-activity-classification/train.csv",
        "./input/hms-harmful-brain-activity-classification/train.csv",
    ]
)
if TRAIN_CSV is None:
    raise FileNotFoundError("Could not locate train.csv in expected Kaggle paths.")

df = pd.read_csv(TRAIN_CSV)

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

_global_prior = train[TARGETS].sum(axis=0).astype(np.float64).values
_global_prior = _global_prior / _global_prior.sum()
_global_prior = np.clip(_global_prior, 1e-12, 1.0)
_global_prior = (_global_prior / _global_prior.sum()).astype(np.float32)
print("Global class prior:", dict(zip(TARGETS, _global_prior.round(6))))

_patient_prior = train.groupby("patient_id")[list(TARGETS)].sum().astype(np.float64)
_patient_prior = _patient_prior.div(_patient_prior.sum(axis=1), axis=0).fillna(0.0)
_patient_prior = _patient_prior.clip(1e-12, 1.0)
_patient_prior = _patient_prior.div(_patient_prior.sum(axis=1), axis=0).astype(
    np.float32
)
print("Patient prior table shape:", _patient_prior.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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



## === cell 2
try:
    import albumentations as albu  # noqa: F401
except Exception as e:
    albu = None
    print("albumentations not available (not required):", repr(e))

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
        df=None,
        seed=0,
    ):

        self.data = data.reset_index(drop=True)
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.specs = specs
        self.eegs = eegs
        self.seed = int(seed)

        if mode != "test":
            self.df = df.merge(
                self.data.iloc[:, :6], on="eeg_id", how="inner"
            ).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]

        self.b, self.a = signal.butter(
            3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            if self.mode == "test":
                row1 = self.data.iloc[i]
                rows = self.df[self.df.patient_id == row1.patient_id].reset_index(
                    drop=True
                )
                if len(rows) == 0:
                    row2 = self.df.iloc[int(row1.eeg_id) % len(self.df)]
                else:
                    pick = int(row1.eeg_id) % len(rows)
                    row2 = rows.iloc[pick]
            else:
                if j < (self.batch_size / 6 * 1):
                    target = "Seizure"
                elif j < (self.batch_size / 6 * 2):
                    target = "GPD"
                elif j < (self.batch_size / 6 * 3):
                    target = "LRDA"
                elif j < (self.batch_size / 6 * 4):
                    target = "Other"
                elif j < (self.batch_size / 6 * 5):
                    target = "GRDA"
                else:
                    target = "LPD"

                rows = self.df[self.df.expert_consensus == target].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row1 = rows.iloc[0]

                rows = self.df[self.df.patient_id_x == row1.patient_id_x].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row2 = rows.iloc[0]

                mixup1 = 0.5
                mixup2 = 1 - mixup1
                label = (
                    row1[TARGETS].values / sum(row1[TARGETS].values) * mixup1
                    + row2[TARGETS].values / sum(row2[TARGETS].values) * mixup2
                )

            for k in range(4):
                if self.mode == "test":
                    img1 = self.specs[row1.spec_id][
                        0 : 0 + 300, k * 100 : (k + 1) * 100
                    ].T
                    img1 = np.nan_to_num(img1, nan=0.0)
                    img1 = np.clip(img1, np.exp(self.cmin), np.exp(self.cmax))
                    img1 = np.log(img1)

                    img2 = self.specs[row2.spectrogram_id][
                        round(row2.spectrogram_label_offset_seconds / 2) : round(
                            row2.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img2 = np.nan_to_num(img2, nan=0.0)
                    img2 = np.clip(img2, np.exp(self.cmin), np.exp(self.cmax))
                    img2 = np.log(img2)

                    img_eeg1 = self.eegs[row1.eeg_id][:, :, k]
                    img_eeg1 = np.nan_to_num(img_eeg1, nan=0.0)

                    img_eeg2 = self.eegs[row2.eeg_id][:, :, k]
                    img_eeg2 = np.nan_to_num(img_eeg2, nan=0.0)

                    img = img1 * 0.5 + img2 * 0.5
                    img_eeg = img_eeg1 * 0.5 + img_eeg2 * 0.5
                else:
                    img1 = self.specs[row1.spectrogram_id][
                        round(row1.spectrogram_label_offset_seconds / 2) : round(
                            row1.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img2 = self.specs[row2.spectrogram_id][
                        round(row2.spectrogram_label_offset_seconds / 2) : round(
                            row2.spectrogram_label_offset_seconds / 2 + 300
                        ),
                        k * 100 : (k + 1) * 100,
                    ].T
                    img_eeg1 = self.eegs[row1.eeg_id][
                        :,
                        round(row1.eeg_label_offset_seconds * SFREQ) : round(
                            row1.eeg_label_offset_seconds * SFREQ + 50 * SFREQ
                        ),
                        k,
                    ]
                    img_eeg2 = self.eegs[row2.eeg_id][
                        :,
                        round(row2.eeg_label_offset_seconds * SFREQ) : round(
                            row2.eeg_label_offset_seconds * SFREQ + 50 * SFREQ
                        ),
                        k,
                    ]

                    img1 = np.nan_to_num(img1, nan=0.0)
                    img1 = np.clip(img1, np.exp(self.cmin), np.exp(self.cmax))
                    img1 = np.log(img1)
                    img2 = np.nan_to_num(img2, nan=0.0)
                    img2 = np.clip(img2, np.exp(self.cmin), np.exp(self.cmax))
                    img2 = np.log(img2)
                    img = img1 * mixup1 + img2 * mixup2

                    img_eeg1 = np.nan_to_num(img_eeg1, nan=0.0)
                    img_eeg2 = np.nan_to_num(img_eeg2, nan=0.0)
                    img_eeg = img_eeg1 * mixup1 + img_eeg2 * mixup2

                img = img[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img.shape[1]
                    ),
                ]
                if HIGH != 100:
                    img_r = np.array(
                        tf.image.resize(
                            np.reshape(img, (img.shape[0], img.shape[1], 1)),
                            ((HIGH - 32), LENGTH),
                        ),
                        dtype=np.float32,
                    )
                    img_r = img_r[:, :, 0]
                    X[
                        j,
                        round((HIGH - img_r.shape[0]) / 2) : round(
                            (HIGH + img_r.shape[0]) / 2
                        ),
                        :,
                        k,
                    ] = img_r
                else:
                    X[j, :, :, k] = img

                X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                    np.std(X[j, :, :, k]) + 1e-6
                )

                img_eeg = signal.filtfilt(self.b, self.a, img_eeg, axis=1)
                img_eeg = img_eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = label

                r = np.random.permutation(X.shape[0])
                X = X[r]
                X_eeg = X_eeg[r]
                y = y[r]

        return X, X_eeg, y




## === cell 3
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
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=None,
        name="spectrogram_efficientnetb0",
    )
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
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])
    x = tf.keras.layers.Concatenate(axis=3)([x, x, x])
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="eeg_efficientnetb0"
    )
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

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 4
if not NEEDTRAIN:
    TEST_CSV = _first_existing_path(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
            "/kaggle/data/hms-harmful-brain-activity-classification/test.csv",
            "./input/hms-harmful-brain-activity-classification/test.csv",
        ]
    )
    if TEST_CSV is None:
        raise FileNotFoundError("Could not locate test.csv in expected Kaggle paths.")

    test = pd.read_csv(TEST_CSV)

    train_df_for_mix = df.copy()

    print("Test shape", test.shape)

    SPEC_DIR = _first_existing_path(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/",
            "/kaggle/data/hms-harmful-brain-activity-classification/test_spectrograms/",
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/",
        ]
    )
    EEG_DIR = _first_existing_path(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/",
            "/kaggle/data/hms-harmful-brain-activity-classification/test_eegs/",
            "./input/hms-harmful-brain-activity-classification/test_eegs/",
        ]
    )
    if SPEC_DIR is None or EEG_DIR is None:
        raise FileNotFoundError(
            "Could not locate test_spectrograms/ or test_eegs/ dirs."
        )

    files2 = sorted(os.listdir(SPEC_DIR))
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(os.path.join(SPEC_DIR, f))
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    files2 = sorted(os.listdir(EEG_DIR))
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(os.path.join(EEG_DIR, f))
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
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

            list_eeg = np.concatenate(list_eeg, 2)
            eegs2[name] = list_eeg
    print()

    candidate_weight_dirs = [
        LOAD_MODELS_FROM,
        f"/kaggle/input/{os.path.basename(LOAD_MODELS_FROM)}",
        f"/kaggle/data/{os.path.basename(LOAD_MODELS_FROM)}",
    ]
    LOAD_MODELS_FROM_RESOLVED = (
        _first_existing_path(candidate_weight_dirs) or LOAD_MODELS_FROM
    )
    if LOAD_MODELS_FROM_RESOLVED != LOAD_MODELS_FROM:
        print("Resolved model dir:", LOAD_MODELS_FROM_RESOLVED)

    weight_files = [
        os.path.join(LOAD_MODELS_FROM_RESOLVED, f"EB2_v{VER}_f{i}.h5") for i in range(5)
    ]
    available = [wf for wf in weight_files if os.path.exists(wf)]
    if len(available) == 0:
        print(f"WARNING: No fold weights found under: {LOAD_MODELS_FROM_RESOLVED}")

        alpha = 0.25  # patient weight
        smooth = 0.02  # shrinkage toward global prior

        pred_list = []
        for pid in test["patient_id"].values:
            if pid in _patient_prior.index:
                pp = _patient_prior.loc[pid].values.astype(np.float32)
                vec = alpha * pp + (1.0 - alpha) * _global_prior
            else:
                vec = _global_prior.copy()

            vec = (1.0 - smooth) * vec + smooth * _global_prior
            vec = np.clip(vec, 1e-12, 1.0)
            vec = vec / vec.sum()
            pred_list.append(vec)
        pred = np.stack(pred_list, axis=0).astype(np.float32)
    else:
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
            df=train_df_for_mix,
            seed=0,
        )

        for wf in weight_files:
            if not os.path.exists(wf):
                print(f"Missing fold weights (skipping): {wf}")
                continue
            print(f"Loading weights: {wf}")
            model.load_weights(wf)
            fold_pred = model.predict(test_gen, verbose=1)
            preds.append(fold_pred)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred

    sub_vals = np.clip(sub[TARGETS].to_numpy(dtype=np.float64), 1e-12, 1.0)
    sub_vals = sub_vals / sub_vals.sum(axis=1, keepdims=True)
    sub[TARGETS] = sub_vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row-sum check (min/mean/max):",
        float(sub[TARGETS].sum(axis=1).min()),
        float(sub[TARGETS].sum(axis=1).mean()),
        float(sub[TARGETS].sum(axis=1).max()),
    )
    print("Wrote: submission.csv")
