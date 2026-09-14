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

0.4702548889012179

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix two execution blockers: the protobuf/TensorFlow import crash by forcing TensorFlow to use the pure-Python protobuf implementation, and the EfficientNet ImageNet weight download (blocked on Kaggle) by switching to `weights=None` so it never tries to fetch from the internet. I also correct the cell numbering to start at 1, keep your model architecture and data pipeline intact, and ensure the code always writes a valid `submission.csv` with probabilities clipped and renormalized to sum to 1. These changes are required for end-to-end execution; they may reduce score versus ImageNet initialization, but they at least yield a valid submission and allow later weight-loading to work if the expected `.h5` files exist. Finally, I make the model-weight loading more robust by using `tf.keras.backend.clear_session()` per fold to avoid graph bloat in long runs.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by switching the protobuf implementation to `upb` (the default in Kaggle’s TF builds) and explicitly disabling TF’s C++ protobuf usage, which avoids the `MessageFactory.GetPrototype` AttributeError on Python 3.12. Then I re-enable ImageNet initialization for EfficientNet **only when weights are not available**, because your current `weights=None` fallback produces very weak uniform-ish predictions (score 1.40995) and moving back toward pretrained features should substantially reduce KL toward your ~0.47 target without changing the architecture or training semantics. I also keep the “no-internet” constraint by catching any weight-download failure and falling back to `weights=None` safely. Finally, I keep the existing robust submission writing (clip + renormalize) to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by switching back to the pure-Python protobuf implementation (the only robust option on Kaggle Py3.12 when this specific `MessageFactory.GetPrototype` error appears), ensuring the script can import TensorFlow and run end-to-end. Then I keep your existing “try ImageNet weights, else fallback to None” logic but make it safer in Kaggle’s no-internet environment by forcing `weights=None` unless a local EfficientNet weights file is already present in `/kaggle/input` (this avoids repeated download attempts/timeouts and keeps execution stable). Finally, I keep your ensemble/load-weights behavior intact and continue to clip+renormalize predictions so the submission always sums to 1 and is accepted; if your fold `.h5` weights exist, score should move down toward the target versus the current uniform-fallback behavior.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash that stops execution by switching from the fragile pure-Python protobuf override to the stable `upb`-based default used in Kaggle TF builds on Python 3.12, while keeping logging quiet. I also make EfficientNet weight selection deterministic and Kaggle-safe: only use ImageNet weights if they are already cached locally (no internet attempts), otherwise fall back to `weights=None` without repeated failures/timeouts. This preserves your model architecture and inference logic, but enables the intended stronger feature extractor when cached weights exist, which should move KL down toward the target versus the current uniform-like fallback behavior. Finally, I keep the existing clipping+renormalization so the submission is always valid and sums to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by switching to the most robust Kaggle/Py3.12 setup: force the pure-Python protobuf backend before importing TensorFlow. I also keep your existing “use ImageNet weights only if already cached locally, otherwise use weights=None” behavior, but make the cache-check actually look for EfficientNetB0 filenames in common Keras cache locations so it can improve score when weights are present without attempting any internet download. Finally, I ensure the script always reaches submission writing and produces a valid `submission.csv` with strictly positive, row-normalized probabilities (required for the KL metric and Kaggle submission validator). These are minimal, execution-unblocking changes that should move your score down toward the target when cached weights and/or provided fold weights exist.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash that stops execution by making the protobuf backend selection robust on Kaggle Py3.12 (try the fast C++/upb backend first, and fall back to pure-Python only if needed). I also ensure TensorFlow is imported only after the environment variables are set, so the setting actually takes effect. These changes are execution-critical and score-neutral by themselves, but they enable your intended fold-weight inference (which should move KL down toward your ~0.47 target versus the current broken run/uniform fallback). I keep your model/data logic intact and keep the submission post-processing (clip + renormalize) to guarantee a valid CSV with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running on Kaggle Python 3.12 by setting the protobuf environment variables *before* any TensorFlow import and using a robust fallback (default/upb → pure-python). Then I keep your model and data pipeline intact, but improve score toward the target by also checking for EfficientNet cached weights in Kaggle’s common cache directories (including `/kaggle/input` and `/kaggle/working`) so pretrained weights are used when available without attempting any internet download. Finally, I make submission writing always succeed with strictly positive, row-normalized probabilities that sum to 1 and match the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def _import_tf_with_robust_protobuf():
    """
    Try importing TensorFlow with default protobuf (fast/upb in Kaggle builds).
    If that hits the known 'MessageFactory.GetPrototype' issue, fall back to
    pure-python protobuf which is slower but robust.
    """
    try:
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        try:
            import importlib
            import sys

            for m in list(sys.modules.keys()):
                if m == "tensorflow" or m.startswith("tensorflow."):
                    del sys.modules[m]
            tf = importlib.import_module("tensorflow")
            return tf
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed with both default protobuf and pure-python protobuf.\n"
                f"Default error: {repr(e1)}\n"
                f"Pure-python error: {repr(e2)}"
            )


import gc
import numpy as np
import pandas as pd

tf = _import_tf_with_robust_protobuf()

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202401312"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

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
print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    device = "/gpu:0" if len(gpus) == 1 else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)
    print(f"Using {len(gpus)} GPU(s)")
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
        print("Mixed precision not enabled due to:", repr(e))
else:
    print("Using full precision")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from scipy import signal


def _butter_band(filter_range, sfreq):
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / sfreq, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / sfreq, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / sfreq, "bandpass")
    return b, a


B_BUTTER, A_BUTTER = _butter_band(filter_range, SFREQ)


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
        specs_path=None,
        eegs_path=None,
    ):
        self.data = data.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.mode = mode

        self.specs = specs
        self.eegs = eegs
        self.specs_path = specs_path
        self.eegs_path = eegs_path

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

    def _load_spec(self, spec_id: int):
        if self.specs is not None and spec_id in self.specs:
            return self.specs[spec_id]
        if self.specs_path is None:
            raise KeyError(f"spec_id={spec_id} not found and specs_path is None")
        fpath = os.path.join(self.specs_path, f"{int(spec_id)}.parquet")
        tmp = pd.read_parquet(fpath)
        return tmp.iloc[:, 1:].values

    def _load_eeg(self, eeg_id: int):
        if self.eegs is not None and eeg_id in self.eegs:
            return self.eegs[eeg_id]
        if self.eegs_path is None:
            raise KeyError(f"eeg_id={eeg_id} not found and eegs_path is None")
        fpath = os.path.join(self.eegs_path, f"{int(eeg_id)}.parquet")
        raw_eeg = pd.read_parquet(fpath)

        time_temp = 0.0 if self.mode == "test" else 0.0
        time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
        time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)
        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

        list_eeg = []
        for region in BRAIN.keys():
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(BRAIN[region]):
                a, b = chan.split("-")
                eeg[chan_i, :] = (eeg_default.loc[:, a] - eeg_default.loc[:, b]).values
            eeg[np.isnan(eeg)] = 0

            if 200 != SFREQ:
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

            eeg = signal.filtfilt(B_BUTTER, A_BUTTER, eeg, axis=1)
            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)  # (4, T, 4)
        return list_eeg

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == "test":
                r = 0
            elif self.mode == "valid":
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = np.random.randint(row["min"], row["max"] + 1) // 2

            spec = self._load_spec(int(row.spec_id))
            eeg_arr = self._load_eeg(int(row.eeg_id))

            for k in range(4):
                img = spec[r : r + 300, k * 100 : (k + 1) * 100].T  # (100,300)
                img_eeg = eeg_arr[:, :, k]  # (4, T)

                img = np.clip(img, np.exp(-6), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                start = max(round((600 / 2 - LENGTH) / 2), 0)
                stop = min(round((600 / 2 - LENGTH) / 2) + LENGTH, img.shape[1])
                img = img[:, start:stop]

                if HIGH != 100:
                    img_rs = (
                        tf.image.resize(
                            np.reshape(img, (img.shape[0], img.shape[1], 1)),
                            ((HIGH - 32), LENGTH),
                        )
                        .numpy()[:, :, 0]
                        .astype(np.float32)
                    )
                    top = round((HIGH - img_rs.shape[0]) / 2)
                    X[j, top : top + img_rs.shape[0], :, k] = img_rs
                else:
                    X[j, :, :, k] = img

                X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                    np.std(X[j, :, :, k]) + 1e-6
                )

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = row[TARGETS].values.astype(np.float32)

        return X, X_eeg, y




## === cell 2
def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    def _imagenet_if_cached_else_none():
        candidates = [
            os.path.expanduser("~/.keras/models"),
            "/root/.keras/models",
            "/kaggle/working/.keras/models",
            "/kaggle/input",  # some datasets ship keras caches
        ]
        patterns = (
            "efficientnetb0_notop.h5",
            "efficientnetb0.h5",
            "efficientnetb0_notop_tf_dim_ordering_tf_kernels_autoaugment.h5",
            "efficientnetb0_tf_dim_ordering_tf_kernels_autoaugment.h5",
        )
        for d in candidates:
            if not os.path.isdir(d):
                continue
            for p in patterns:
                if os.path.exists(os.path.join(d, p)):
                    return "imagenet"
                if d == "/kaggle/input":
                    try:
                        for sub in os.listdir(d):
                            subd = os.path.join(d, sub)
                            if os.path.isdir(subd) and os.path.exists(
                                os.path.join(subd, p)
                            ):
                                return "imagenet"
                    except Exception:
                        pass
        return None

    w_spec = _imagenet_if_cached_else_none()
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=w_spec, name="efficientnetb0_spec"
        )
    except Exception as e:
        print(
            "WARNING: EfficientNetB0 weights unavailable for spec; using None. Err:",
            repr(e),
        )
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, name="efficientnetb0_spec"
        )
    base_model._name = "spectrogram_extractor"

    x0 = inp[:, :, :, :1]
    x1 = inp[:, :, :, 1:2]
    x2 = inp[:, :, :, 2:3]
    x3 = inp[:, :, :, 3:4]
    x01 = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])
    x = tf.keras.layers.Concatenate(axis=3)([x01, x01, x01])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    w_eeg = _imagenet_if_cached_else_none()
    try:
        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=w_eeg, name="efficientnetb0_eeg"
        )
    except Exception as e:
        print(
            "WARNING: EfficientNetB0 weights unavailable for eeg; using None. Err:",
            repr(e),
        )
        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, name="efficientnetb0_eeg"
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




## === cell 3
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        SPECS_PATH = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        EEGS_PATH = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        SPECS_PATH = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        EEGS_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    print("Test shape", test.shape)
    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=None,
        eegs=None,
        specs_path=SPECS_PATH,
        eegs_path=EEGS_PATH,
    )

    preds = []
    found_any_weights = False
    for i in range(5):
        print(f"Fold {i+1}")
        wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
        if os.path.exists(wpath):
            found_any_weights = True

            tf.keras.backend.clear_session()
            gc.collect()

            with strategy.scope():
                model = build_model()
                model.load_weights(wpath)

            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
        else:
            print(f"WARNING: weights not found at {wpath}")

    if found_any_weights:
        pred = np.mean(preds, axis=0)
    else:
        print(
            "WARNING: No fold weights found. Writing uniform probabilities fallback submission."
        )
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float64)

    print("Test preds shape", pred.shape)

    pred = np.asarray(pred, dtype=np.float64)
    pred = np.clip(pred, 1e-9, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred.astype(np.float32)
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row prob sum min/max:",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).max(),
    )
    print(sub.head())
