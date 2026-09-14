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

0.5750454971627468

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf/tensorflow interaction by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it triggers the `MessageFactory.GetPrototype` error in this environment). Then I make model weight loading robust: instead of hard-failing when the external `/kaggle/input/models202402111` dataset isn’t attached, the script automatically fall back to producing a valid (properly normalized) submission using the competition’s sample-submission prior (uniform probabilities), ensuring a `.csv` is always generated. This keeps the core model/inference logic intact when weights are present, and only changes behavior to avoid crashing when they are not. Finally, I ensure column order exactly matches `sample_submission.csv` and every row sums to 1 to prevent submission rejection.'
- What this solution (achieved 1.40995) has done: 'We need to fix the immediate crash happening before any submission is written: the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` occurs on importing TensorFlow due to an incompatible protobuf runtime in this Kaggle Python 3.12 environment. The minimal robust fix is to force protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids the missing `GetPrototype` path. I keep the rest of the pipeline unchanged (model, generator, inference) so behavior/score only changes insofar as the code can now run and actually use the provided weights when present. I also ensure we always emit a valid `submission.csv` with correct columns and row-wise probability normalization.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf override and instead (a) deferring the TensorFlow import until after we decide whether we actually need it, and (b) adding a robust fallback that still produces a valid, normalized `submission.csv` when TensorFlow can’t be imported or model weights are missing. This keeps the core model/inference code intact when weights + TensorFlow are available, but prevents the notebook from dying early (which is currently blocking any chance to score better). To move the score toward the target (lower is better) from the current 1.40995, the main necessary change is to actually run the real model inference instead of always falling back; the changes below ensure that happens whenever possible in the Kaggle environment. Submission formatting is kept strict: correct columns, correct order, and per-row probabilities sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf to use the pure-Python implementation *before* any TensorFlow import, which avoids the `MessageFactory.GetPrototype` error in this Python 3.12 Kaggle environment. I also make sure the notebook never silently proceeds with `TF_AVAILABLE=True` when import actually failed, so the model inference path is only used when TensorFlow is truly usable. These changes are execution-blocking bug fixes and are necessary to actually run the real model instead of the uniform-probability fallback (which should move the KL score down toward your target). Submission formatting and core model/generator logic are preserved, and the code always write a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by not forcing the pure-Python protobuf implementation, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle Python 3.12 environment. Then I ensure TensorFlow is only imported when needed and that the script always produces a valid `submission.csv` with the exact required columns and per-row probabilities summing to 1. This should allow the real model inference path to run (when weights are available), which is the minimal legitimate change likely to reduce KL from the current uniform-fallback-like behavior toward your target. All model architecture, data generator logic, and inference averaging remain unchanged.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation **before** any TensorFlow import, which is the minimal change that unblocks the real model inference path in this Python 3.12 Kaggle environment. I also make TensorFlow availability detection stricter (so we don’t proceed into model code after a partial/failed import), and keep the existing robust fallback to `sample_submission.csv` only when TF or weights are genuinely unavailable. Finally, I keep submission formatting identical but add a small safety normalization helper to guarantee each row sums to 1 and is finite, preventing “invalid submission” failures. These changes should reduce KL from the current uniform-like fallback score (1.40995) toward your target (~0.575) by enabling the intended pretrained fold ensemble inference.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402111"
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

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy import signal

VER = 1
MIX = True
print("Mixed precision experimental flag disabled for stability in this environment")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))


def _resolve_models_dir(requested_path: str, ver: int = 1, folds: int = 5) -> str:
    """
    Fix FileNotFoundError for model weights by auto-detecting the actual dataset folder under /kaggle/input.
    Keeps core inference logic unchanged; only makes the path robust.
    """
    expected = [f"EB2_v{ver}_f{i}.h5" for i in range(folds)]
    if os.path.isdir(requested_path):
        if all(os.path.exists(os.path.join(requested_path, f)) for f in expected):
            return requested_path

    base = "/kaggle/input"
    if os.path.isdir(base):
        candidates = []
        for root, _, files in os.walk(base):
            if any(f in files for f in expected):
                candidates.append(root)
        for root in candidates:
            if all(os.path.exists(os.path.join(root, f)) for f in expected):
                return root
        if candidates:
            return candidates[0]

    return requested_path


if PLATFORM == "kaggle":
    resolved = _resolve_models_dir(LOAD_MODELS_FROM, ver=VER, folds=5)
    if resolved != LOAD_MODELS_FROM:
        print(f"Resolved LOAD_MODELS_FROM: {LOAD_MODELS_FROM} -> {resolved}")
    LOAD_MODELS_FROM = resolved


def _normalize_probs(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Safety: clip, replace non-finite, renormalize rows to sum=1 (prevents submission rejection)."""
    arr = np.asarray(arr, dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return (arr / s).astype(np.float32)




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



## === cell 4
TF_AVAILABLE = False
tf = None
strategy = None

try:
    import tensorflow as tf  # noqa: F401

    _ = tf.constant(0)

    TF_AVAILABLE = True
    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    strategy = None
    print(
        "WARNING: TensorFlow import failed; will fall back to sample_submission probabilities."
    )
    print("TF import error:", repr(e))

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
            df=None,
        ):

            if mode != "test":
                self.df = df.merge(
                    data.iloc[:, :6], on="eeg_id", how="inner"
                ).reset_index(drop=True)

            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
            self.data = data.reset_index(drop=True)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.b, self.a = signal.butter(
                3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
            )
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.data) / self.batch_size))

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

            img_map = np.zeros((100 * 300, 3), dtype=np.float32)

            for j, i in enumerate(indexes):
                if self.mode == "test":
                    row = self.data.iloc[i]
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
                    row2 = rows.iloc[1]
                    mixup = 0.3
                    label = (row1[TARGETS].values / sum(row1[TARGETS].values)) * (
                        1 - mixup
                    ) + (row2[TARGETS].values / sum(row2[TARGETS].values)) * mixup

                for k in range(4):
                    if self.mode == "test":
                        spec_id = (
                            row.spec_id
                            if "spec_id" in row.index
                            else row.spectrogram_id
                        )
                        img = self.specs[int(spec_id)][0:300, k * 100 : (k + 1) * 100].T
                        img_eeg = self.eegs[int(row.eeg_id)][:, :, k]
                        img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                        img = np.log(img)
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
                        img1 = np.clip(img1, np.exp(self.cmin), np.exp(self.cmax))
                        img1 = np.log(img1)
                        img1 = np.nan_to_num(img1, nan=0.0)
                        img2 = np.clip(img2, np.exp(self.cmin), np.exp(self.cmax))
                        img2 = np.log(img2)
                        img2 = np.nan_to_num(img2, nan=0.0)
                        img = img1 * (1 - mixup) + img2 * mixup
                        img_eeg1 = np.nan_to_num(img_eeg1, nan=0.0)
                        img_eeg2 = np.nan_to_num(img_eeg2, nan=0.0)
                        img_eeg = img_eeg1 * (1 - mixup) + img_eeg2 * mixup

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

            return X, X_eeg, y




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
if TF_AVAILABLE:
    try:
        import efficientnet.tfkeras as efn  # original dependency

        _USING_EFN = True
    except Exception:
        efn = None
        _USING_EFN = False
        from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2

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

        b0_input_shape = (HIGH * 4, LENGTH, 3)

        if _USING_EFN:
            base_model = efn.EfficientNetB0(
                include_top=False, weights=None, input_shape=b0_input_shape
            )
        else:
            base_model = EfficientNetB0(
                include_top=False, weights=None, input_shape=b0_input_shape
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

        x0 = inp[:, :, :, :, 0]
        x1 = inp[:, :, :, :, 1]
        x2 = inp[:, :, :, :, 2]
        x3 = inp[:, :, :, :, 3]
        x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

        x0_eeg = inp_eeg[:, :, :, :1]
        x1_eeg = inp_eeg[:, :, :, 1:2]
        x2_eeg = inp_eeg[:, :, :, 2:3]
        x3_eeg = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        x_eeg = tf.keras.layers.ZeroPadding2D(padding=((4, 4), (0, 0)))(x_eeg)

        if _USING_EFN:
            base_model_eeg = efn.EfficientNetB2(
                include_top=False,
                weights=None,
                input_shape=(32, round(EEG_LENGTH * SFREQ), 3),
            )
        else:
            base_model_eeg = EfficientNetB2(
                include_top=False,
                weights=None,
                input_shape=(32, round(EEG_LENGTH * SFREQ), 3),
            )
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
    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    expected_files = [
        os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5") for i in range(5)
    ]
    missing = [p for p in expected_files if not os.path.exists(p)]

    if (not TF_AVAILABLE) or missing:
        if not TF_AVAILABLE:
            print(
                "WARNING: TensorFlow unavailable; falling back to sample_submission probabilities."
            )
        if missing:
            print(
                "WARNING: Model weights not found; falling back to sample_submission probabilities."
            )
            print("Missing files:\n" + "\n".join(missing))

        sub = sample_sub.copy()
        sub[TARGETS] = _normalize_probs(sub[TARGETS].to_numpy(dtype=np.float64))

        sub = sub[["eeg_id"] + list(sample_sub.columns[1:])]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row sum min/max:",
            sub[TARGETS].sum(axis=1).min(),
            sub[TARGETS].sum(axis=1).max(),
        )
        print("Saved to submission.csv")
    else:
        test_spec_ids = set(test["spec_id"].astype(int).tolist())
        test_eeg_ids = set(test["eeg_id"].astype(int).tolist())

        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_spec_ids:
                continue
            tmp = pd.read_parquet(f"{PATH2}{f}")
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()
        print(f"Loaded {len(spectrograms2)} spectrograms for test")

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids:
                continue

            eeg_default = pd.read_parquet(f"{PATH2}{f}")

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
        print(f"Loaded {len(eegs2)} EEGs for test")

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
            model.load_weights(os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5"))
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = _normalize_probs(pred)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred

        sub = sub[["eeg_id"] + list(sample_sub.columns[1:])]

        sub[TARGETS] = _normalize_probs(sub[TARGETS].to_numpy(dtype=np.float64))

        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row sum min/max:",
            sub[TARGETS].sum(axis=1).min(),
            sub[TARGETS].sum(axis=1).max(),
        )
        print("Saved to submission.csv")
