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

0.5298586086507827

# 6. Current score

0.96161

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fix the immediate runtime failure caused by the TensorFlow/protobuf incompatibility by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I prevent the inference cell from crashing when the external pretrained weights dataset isn’t available by falling back to a valid, properly-normalized submission built from `sample_submission.csv` priors (score won’t be great, but it yield a valid submission instead of “Not yielded”). The core model/data logic is preserved; the fallback only activates when weight files are missing, otherwise the original ensemble inference runs unchanged. Finally, I ensure the output file is named exactly `submission.csv` with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.46011) has done: 'I fix the TensorFlow/protobuf crash by importing `google.protobuf` first and then forcing TensorFlow to use the pure‑Python protobuf implementation before importing TF, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. Then, to move the score down toward your target (lower is better) while keeping the overall approach intact, I improve the existing “missing weights” fallback: instead of a single global prior, it use patient-specific priors (and otherwise fall back to the global prior), which is a minimal, legitimate calibration improvement using only train metadata. Finally, I keep the normal ensemble inference path unchanged when weights exist, and I ensure `submission.csv` is always written with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.05259) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure‑Python protobuf backend *before* any TensorFlow-related import and by preventing C++ protobuf from being loaded. Then I keep your model/inference logic unchanged, but make the “missing weights” fallback slightly stronger (and still fully legitimate) by using patient-specific priors smoothed with the global prior (shrinkage), which should reduce KL versus the current fallback and move the score toward your target. Finally, I ensure `submission.csv` is always written with the exact required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.05564) has done: 'I fix the TensorFlow/protobuf crash that currently stops execution by moving the protobuf environment forcing to the very top of the notebook and importing TensorFlow only after that, plus adding a safe fallback if TF still can’t import in this Kaggle image. Since your current score (1.05259, lower is better) is far from the target (~0.53), I also improve the existing “missing weights” fallback in a minimal, legitimate way: use a patient-specific prior with shrinkage plus a small spectrogram-derived prior (from test spectrogram energy) to better match per-record characteristics without changing the model path when weights exist. The core model/inference logic is left unchanged when weight files are present; only the fallback path is strengthened. Finally, I ensure `submission.csv` is always produced with the exact required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.01913) has done: 'I fix the immediate runtime failure (`MessageFactory.GetPrototype`) by forcing the pure‑Python protobuf implementation *before* any TensorFlow import and by also forcing protobuf to use the Python backend via `google.protobuf.internal.api_implementation`. This keeps your core model/inference logic unchanged, but makes TensorFlow import reliably in the Kaggle Python 3.12 image (so you can actually use the intended ensemble path when weights are present). If weights are missing, I keep your existing priors-based fallback but make it slightly stronger and more stable by using label-count–weighted patient priors (instead of simple mean) and by guarding spectrogram-feature training to avoid extra overhead/timeouts. Finally, I ensure `submission.csv` is always written with the required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 0.96161) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf’s pure-Python backend even earlier and adding a safe TF import guard that prevents any protobuf C++ usage; this directly addresses the `MessageFactory.GetPrototype` error so the notebook runs end-to-end. I also make the fallback (used when TF or weights are unavailable) a bit stronger—but still minimal and legitimate—by (1) using a label-count–weighted global prior and (2) adding a small, smoothed “spec-id neighbor” prior computed purely from train metadata (no spectrogram file reading), which should reduce KL and move your score down toward the ~0.53 target. The main model/inference path remains unchanged when weights are present. Finally, the script always writes a valid `submission.csv` with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 0.96161) has done: 'I fix the immediate runtime failure (`MessageFactory` missing `GetPrototype`) by forcing the pure‑Python protobuf backend earlier and more robustly, including preloading `google.protobuf.message_factory` before importing TensorFlow. This change is execution-focused (score-neutral) and keeps your model/inference logic intact. Since your current score (0.96161, lower is better) is still far from the target (~0.53), I minimally strengthen only the “missing TF/weights” fallback by adding a small, legitimate patient+spec prior interaction term (a convex mixture of the already-computed priors), which should reduce KL without changing the main ensemble path when weights exist. The script still always write a valid `submission.csv` with correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 0.96161) has done: 'I fix the immediate runtime failure by applying a more robust protobuf “pure-Python” forcing that works with the protobuf version shipped in Kaggle’s Py3.12 image (the current `MessageFactory.GetPrototype` crash happens before your TF guard can help). This keeps your core model/inference logic intact, but makes TensorFlow import reliably so you can use the intended ensemble path when weights exist (which should move the KL score down toward the target versus the priors-only fallback). I also add a safe, score-neutral guard so that if TensorFlow still can’t import or weights are missing, the existing priors-based fallback still produces a valid `submission.csv`. Finally, I keep the exact required submission columns and enforce row-wise normalization.'
- What this solution (achieved 0.96161) has done: 'I fix the current hard crash (`MessageFactory` missing `GetPrototype`) by ensuring the protobuf backend is forced to pure-Python *before* any TensorFlow import, and by monkey-patching a compatible `GetPrototype` alias when only `GetMessageClass` exists (this is what breaks in the Kaggle Py3.12 image). This is execution-focused and keeps your model/data logic unchanged, but it should allow TensorFlow to import so the intended ensemble inference path can run when weights are present (which should improve KL toward your target versus the priors-only fallback). If TF still can’t load or weights are missing, the existing priors-based fallback remains and still writes a valid `submission.csv`. All outputs are kept normalized to sum to 1 with the exact required column names.'
- What this solution (achieved 0.96161) has done: 'I fix the protobuf/TensorFlow crash by making the protobuf shim robust: in this Kaggle image the failing `GetPrototype` call happens on an *instance*, so we must alias it on both the `MessageFactory` class and the returned default factory instance (and do it before importing TensorFlow). This is execution-critical and score-neutral, but it should allow TensorFlow to import so the intended ensemble inference path can run when weights exist (which should improve KL vs the priors-only fallback and move you toward the 0.53 target). I also keep the existing fallback submission path unchanged aside from ensuring it still runs if TF import fails, and keep the submission format/normalization guarantees intact. No changes are made to the model architecture, data generator logic, or prediction semantics.'
- What this solution (achieved 0.96161) has done: 'I fix the immediate crash caused by protobuf’s `MessageFactory.GetPrototype` being missing on the *instance* used by TensorFlow by installing a shim on both the class and the default factory instance, and doing it using a safer import order that works in Kaggle Py3.12. This should allow TensorFlow to import and therefore enable the intended ensemble inference path when the model weights exist (likely improving KL toward your lower-is-better target). If TensorFlow still can’t import or weights are missing, I keep your existing priors-based fallback logic intact so the notebook always produces a valid `submission.csv`. No changes are made to your model, data generator, or inference semantics beyond the protobuf compatibility fix and minor robustness guards.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"


def _install_getprototype_shim():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.message_factory as _mf

        try:
            cls = getattr(_mf, "MessageFactory", None)
            if (
                cls is not None
                and (not hasattr(cls, "GetPrototype"))
                and hasattr(cls, "GetMessageClass")
            ):

                def _GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                setattr(cls, "GetPrototype", _GetPrototype)
        except Exception:
            pass

        try:
            default_factory = _mf.GetMessageFactory()
            if (
                default_factory is not None
                and (not hasattr(default_factory, "GetPrototype"))
                and hasattr(default_factory, "GetMessageClass")
            ):

                def _GetPrototype_inst(descriptor, _df=default_factory):
                    return _df.GetMessageClass(descriptor)

                setattr(default_factory, "GetPrototype", _GetPrototype_inst)
        except Exception:
            pass

        try:
            from google.protobuf import symbol_database as _symbol_database

            _sym_db = _symbol_database.Default()
            mf = getattr(_sym_db, "pool", None)
            _ = mf
        except Exception:
            pass

    except Exception as e:
        print("Warning: protobuf preload/shim failed:", repr(e))


_install_getprototype_shim()

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402211"
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        "WARNING: TensorFlow import failed; will use fallback submission only. Error:",
        repr(e),
    )

import pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt

if TF_AVAILABLE:
    print("TensorFlow version =", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
else:
    strategy = None

VER = 1

MIX = True
if TF_AVAILABLE:
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print(
                "Mixed precision could not be enabled; continuing with default precision. Error:",
                repr(e),
            )
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



## === cell 2
try:
    import albumentations as albu
except Exception:
    albu = None

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if TF_AVAILABLE:

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
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
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

            img_map = np.zeros((100 * 300, 3))

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                if self.mode == "test":
                    r = 0
                elif self.mode == "valid":
                    r = int((row["min"] + row["max"]) // 4)
                else:
                    r = np.random.randint(row["min"], row["max"] + 1) // 2

                for k in range(4):
                    img = self.specs[row.spec_id][
                        r : r + 300, k * 100 : (k + 1) * 100
                    ].T
                    img_eeg = self.eegs[row.eeg_id][:, :, k]

                    img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                    img = np.log(img)

                    img = np.nan_to_num(img, nan=0.0)

                    img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                    img = np.reshape(img, (img.shape[0] * img.shape[1]))
                    img = np.array(img, dtype=np.int16)
                    img = np.clip(img, 0, 255)
                    img_map = self.cmaps[img]
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
                        X[j, :, :, :, k] = img

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
                    label = row[TARGETS].values
                    if self.mode == "train" and sum(label == 1):
                        label[label == 0] = 1e-2
                        label[label == 1] = 1 - 5 * 1e-2
                    y[j] = label

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




## === cell 3
if TF_AVAILABLE:

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
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None
        )
        base_model._name = "spectrogram_extractor"

        x0 = inp[:, :, :, :, 0]
        x1 = inp[:, :, :, :, 1]
        x2 = inp[:, :, :, :, 2]
        x3 = inp[:, :, :, :, 3]
        x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm_spec"
        )(x)

        base_model_eeg = tf.keras.applications.EfficientNetB2(
            include_top=False, weights=None
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
        x_eeg = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm_eeg"
        )(x_eeg)

        x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
        x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
        return model




## === cell 4
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
test.head()


def _find_any_weight_file(load_dir: str, ver: int, fold: int):
    wpath_h5 = os.path.join(load_dir, f"EB2_v{ver}_f{fold}.h5")
    wpath_weights = os.path.join(load_dir, f"EB2_v{ver}_f{fold}.weights.h5")
    if os.path.exists(wpath_h5):
        return wpath_h5
    if os.path.exists(wpath_weights):
        return wpath_weights
    return None


missing = []
for i in range(5):
    if _find_any_weight_file(LOAD_MODELS_FROM, VER, i) is None:
        missing.append(i)

if (not TF_AVAILABLE) or (len(missing) > 0):
    if not TF_AVAILABLE:
        print(
            "WARNING: TensorFlow unavailable; creating fallback submission from priors."
        )
    else:
        print(
            f"WARNING: Missing model weights for folds {missing} in {LOAD_MODELS_FROM}. "
            "Creating a fallback submission from train priors."
        )

    eeg_label_counts = df.groupby("eeg_id").size().astype(np.float64)
    train_w = train[["eeg_id", "patient_id", "spec_id"]].copy()
    train_w["w"] = (
        train_w["eeg_id"].map(eeg_label_counts).fillna(1.0).astype(np.float64)
    )

    global_prior = (
        train[list(TARGETS)].to_numpy(dtype=np.float64)
        * train_w["w"].to_numpy()[:, None]
    ).sum(axis=0)
    global_prior = np.clip(global_prior, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    patient_prior_df = (
        train.assign(w=train_w["w"].values)
        .groupby("patient_id")
        .apply(
            lambda g: (
                g[list(TARGETS)].to_numpy(dtype=np.float64) * g["w"].to_numpy()[:, None]
            ).sum(axis=0)
            / (g["w"].sum() + 1e-12)
        )
    )
    patient_prior_df = pd.DataFrame(
        patient_prior_df.tolist(), index=patient_prior_df.index, columns=list(TARGETS)
    ).astype(np.float64)

    patient_counts = train.groupby("patient_id").size().astype(np.int64)

    spec_prior_df = (
        train.assign(w=train_w["w"].values)
        .groupby("spec_id")
        .apply(
            lambda g: (
                g[list(TARGETS)].to_numpy(dtype=np.float64) * g["w"].to_numpy()[:, None]
            ).sum(axis=0)
            / (g["w"].sum() + 1e-12)
        )
    )
    spec_prior_df = pd.DataFrame(
        spec_prior_df.tolist(), index=spec_prior_df.index, columns=list(TARGETS)
    ).astype(np.float64)
    spec_counts = train.groupby("spec_id").size().astype(np.int64)

    m_patient = 20.0
    m_spec = 30.0
    alpha_interaction = 0.15  # small, conservative weight

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    probs = np.zeros((len(test), len(TARGETS)), dtype=np.float64)

    for idx, (pid, spec_id) in enumerate(
        zip(test["patient_id"].values, test["spectrogram_id"].values)
    ):
        if pid in patient_prior_df.index:
            p_pat = patient_prior_df.loc[pid].values
            p_pat = np.clip(p_pat, 1e-12, 1.0)
            p_pat = p_pat / p_pat.sum()
            n = float(patient_counts.loc[pid]) if pid in patient_counts.index else 1.0
            w_pat = n / (n + m_patient)
            p0 = w_pat * p_pat + (1.0 - w_pat) * global_prior
        else:
            p_pat = None
            p0 = global_prior

        if spec_id in spec_prior_df.index:
            p_spec = spec_prior_df.loc[spec_id].values
            p_spec = np.clip(p_spec, 1e-12, 1.0)
            p_spec = p_spec / p_spec.sum()
            ns = (
                float(spec_counts.loc[spec_id]) if spec_id in spec_counts.index else 1.0
            )
            w_spec = ns / (ns + m_spec)
            p = (1.0 - w_spec) * p0 + w_spec * p_spec
        else:
            p_spec = None
            p = p0

        if (p_pat is not None) and (p_spec is not None):
            p_int = np.clip(p_pat * p_spec, 1e-12, None)
            p_int = p_int / p_int.sum()
            p = (1.0 - alpha_interaction) * p + alpha_interaction * p_int

        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        probs[idx] = p

    sub[list(TARGETS)] = probs.astype(np.float32)
    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-wise prob sum stats:",
        float(sub[list(TARGETS)].sum(axis=1).min()),
        float(sub[list(TARGETS)].sum(axis=1).max()),
    )
    assert sub.shape[0] == test.shape[0], "Row count mismatch vs test.csv"
    assert np.allclose(
        sub[list(TARGETS)].sum(axis=1).values, 1.0, atol=1e-6
    ), "Probabilities do not sum to 1"
    assert os.path.exists("submission.csv"), "submission.csv was not created"

else:
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

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, f in enumerate(files2):
        if i % 100 == 0:
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
        print(f"Fold {i + 1}")
        wpath = _find_any_weight_file(LOAD_MODELS_FROM, VER, i)
        if wpath is None:
            raise FileNotFoundError(
                f"Missing weights file for fold {i}: expected "
                f"{os.path.join(LOAD_MODELS_FROM, f'EB2_v{VER}_f{i}.h5')} "
                f"or {os.path.join(LOAD_MODELS_FROM, f'EB2_v{VER}_f{i}.weights.h5')}"
            )
        model.load_weights(wpath)
        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)

    pred = np.mean(preds, axis=0)
    print()
    print("Test preds shape", pred.shape)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, 1e-12, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)
    sub[list(TARGETS)] = pred.astype(np.float32)

    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-wise prob sum stats:",
        float(sub[list(TARGETS)].sum(axis=1).min()),
        float(sub[list(TARGETS)].sum(axis=1).max()),
    )
    assert sub.shape[0] == test.shape[0], "Row count mismatch vs test.csv"
    assert np.allclose(
        sub[list(TARGETS)].sum(axis=1).values, 1.0, atol=1e-5
    ), "Probabilities do not sum to 1"
    assert os.path.exists("submission.csv"), "submission.csv was not created"
