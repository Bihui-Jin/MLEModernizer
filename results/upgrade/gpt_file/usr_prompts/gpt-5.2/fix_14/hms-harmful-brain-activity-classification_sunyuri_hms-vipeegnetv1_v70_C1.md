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

0.4518571697565199

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime/import failure by disabling mixed precision on this Kaggle image (it’s triggering a protobuf/GetPrototype crash) while keeping the model and inference logic unchanged. Then I fix the missing weights issue by auto-detecting the correct `/kaggle/input/...` directory and weight filenames; if no weights are present, the script fall back to a safe uniform-probability submission so you always get a valid `submission.csv`. I also remove the non-installed `albumentations` dependency (it isn’t used in your current pipeline) to prevent import errors. These changes are execution/stability-focused and do not alter the model architecture, loss, or prediction post-processing when weights are available.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash causing `MessageFactory.GetPrototype` errors by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a common Kaggle TF stability issue) while keeping your model/training/inference logic unchanged. I also make the model-weight directory resolution more robust by auto-searching under `/kaggle/input` for the expected `.h5` files so folds actually load, since your current score suggests you may be falling back to uniform predictions. Finally, I keep the submission formatting and probability normalization intact and ensure `submission.csv` is always produced.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate crash (`MessageFactory.GetPrototype`) by switching protobuf to the C++ implementation (and unsetting the problematic env vars) before importing TensorFlow; this is the root cause preventing any model inference from running. I also add a small compatibility fallback so the script still runs even if TensorFlow import fails for any reason, producing a valid uniform-probability `submission.csv` rather than crashing. Finally, I keep your model/inference logic unchanged, but make model directory/weight discovery slightly more robust (including searching nested subdirectories under `/kaggle/input`) so you’re less likely to fall back to uniform predictions—this should improve KL toward your target if weights exist.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the common stable configuration on Kaggle for TF+protobuf version mismatches. I also ensure SciPy is imported safely (or provide a deterministic fallback) since your pipeline uses `scipy.signal` and missing SciPy would otherwise crash mid-inference. Finally, I keep your model/inference logic unchanged but make weight discovery a bit more robust (case-insensitive matching) so folds actually load when weights exist, which should move KL down from the current uniform-like score toward your target.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by enforcing a stable protobuf runtime configuration *before* importing TensorFlow; this is an execution blocker preventing any real inference and likely causing your high KL from uniform fallback predictions. I also make the GPU strategy selection robust so it doesn’t request `/gpu:0` when no GPU is available (another common runtime failure). Finally, I keep your model and prediction logic unchanged, but ensure weights are actually discovered by allowing recursive search under the resolved models directory so fold weights load when they exist—this should move the score down toward your target without changing the architecture or loss.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the C++ protobuf backend (and unsetting conflicting env vars) before importing TensorFlow, with a safe fallback to pure-Python protobuf only if needed. This should allow the model to actually run inference instead of failing early and/or falling back to uniform predictions, which is likely why your KL is far from the target. I also make `CUDA_VISIBLE_DEVICES` safe (don’t request GPU IDs that may not exist) and ensure the submission always matches `sample_submission.csv` column order and is normalized to sum to 1. Core model, generator, and inference logic remain unchanged; the changes are strictly stability/IO/calibration-safe.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf runtime to pure-Python *before* any TensorFlow import, which avoids the `MessageFactory.GetPrototype` failure that currently prevents real inference. Then I make the TF import attempt deterministic (single stable path) and keep your strategy/model/inference logic unchanged. Finally, I keep the existing robust weight-directory search so fold weights actually load; this should move the KL score down substantially from the current uniform-like 1.40995 toward your ~0.45 target, while still always producing a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash that stops the pipeline at cell 0 by setting a stable protobuf env configuration *before* importing TensorFlow and by doing the TF import in a clean subprocess-safe way (no change to your model logic). Then I fix a logic bug in the generator where the “HIGH==100” branch accidentally writes the wrong variable (it assigns `img` instead of the computed RGB `img_map`), which can severely damage predictions and inflate KL. Finally, I keep your weight discovery/loading logic intact but make the fallback deterministic and ensure the submission is always written with correct column order and row-normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by setting a stable protobuf environment *and* forcing TensorFlow to use the pure-Python protobuf backend before any TF import, with a clean fallback to uniform predictions only if TF truly can’t import. Then I correct a real logic bug in the generator’s `HIGH == 100` branch that currently writes the wrong variable into `X` (this can badly degrade predictions and inflate KL). Finally, I keep your model/weights/inference logic the same but make sure model weight discovery still works and that the submission is always properly normalized and written as `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` import crash by forcing a safe protobuf runtime configuration *and* retrying TF import with the alternate backend if the first attempt fails; this should unblock real model inference instead of falling back to uniform predictions (which is consistent with the current high KL). I also correct a real logic bug in the `DataGenerator` where the `HIGH == 100` branch writes the wrong variable (`img` instead of the computed RGB `img_map`), which can severely corrupt spectrogram inputs and hurt score. Finally, I keep your model/weights/inference flow unchanged, but make the Kaggle dataset path resolution a bit more robust (directly try the known competition directory) so fold weights are more likely to be found and loaded.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by setting a single, stable protobuf backend configuration before any TensorFlow import (and avoid the unreliable “import-then-switch-backend” flow that still triggers `MessageFactory.GetPrototype`). Then I keep your model/inference logic intact but apply the already-identified generator bugfix in the `HIGH == 100` branch (write `img_map` into `X`, not the pre-colormap `img`) because it can strongly degrade predictions when that branch is used. Finally, I keep your existing robust weight discovery and ensure the script always writes a valid `submission.csv` with correct column order and row-normalized probabilities; this should move KL down toward the target by enabling real inference instead of fallback behavior.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # do not assume GPU ids exist
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import numpy as np
import pandas as pd

_TF_OK = True
_TF_IMPORT_ERR = None
tf = None

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    tf = None
    _TF_OK = False
    _TF_IMPORT_ERR = repr(e)

import matplotlib
import matplotlib.pyplot as plt  # noqa: F401

print("Python OK")
if _TF_OK:
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
else:
    print("WARNING: TensorFlow failed to import:", _TF_IMPORT_ERR)
    strategy = None

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402202"
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

VER = 1

MIX = False
if _TF_OK and MIX:
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision requested but could not be enabled:", repr(e))
else:
    print("Using full precision (mixed precision disabled for stability)")



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



## === cell 2
try:
    import albumentations as albu  # noqa: F401

    _HAS_ALBU = True
except Exception:
    _HAS_ALBU = False

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if _TF_OK:

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

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                if self.mode == "test":
                    r = 0
                elif self.mode == "valid":
                    r = int((row["min"] + row["max"]) // 4)
                else:
                    r = np.random.randint(row["min"], row["max"] + 1) // 2

                for k in range(4):
                    spec = self.specs[row.spec_id]

                    max_r = max(spec.shape[0] - 300, 0)
                    rr = int(np.clip(r, 0, max_r))

                    img = spec[rr : rr + 300, k * 100 : (k + 1) * 100].T
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
                    label = row[TARGETS].values
                    if self.mode == "train" and sum(label == 1):
                        label[label == 0] = 1e-2
                        label[label == 1] = 1 - 5 * 1e-2
                    y[j] = label

            return X, X_eeg, y

        def __random_transform(self, img):
            if not _HAS_ALBU:
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
if _TF_OK:

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




## === cell 4
def _resolve_models_dir(preferred_dir: str) -> str | None:
    if preferred_dir and os.path.isdir(preferred_dir):
        return preferred_dir

    direct_candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    ]
    for cand in direct_candidates:
        if os.path.isdir(cand):
            try:
                if any(fn.lower().endswith(".h5") for fn in os.listdir(cand)):
                    return cand
            except Exception:
                pass

    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None

    candidates = []
    for d in sorted(os.listdir(base)):
        cand = os.path.join(base, d)
        if os.path.isdir(cand):
            candidates.append(cand)
            try:
                for sd in sorted(os.listdir(cand)):
                    cand2 = os.path.join(cand, sd)
                    if os.path.isdir(cand2):
                        candidates.append(cand2)
            except Exception:
                pass

    best = None
    best_hits = -1
    for cand in candidates:
        try:
            files = os.listdir(cand)
        except Exception:
            continue
        h5s = [fn for fn in files if fn.lower().endswith(".h5")]
        if not h5s:
            continue

        hits = 0
        for fn in h5s:
            lo = fn.lower()
            if "eb2" in lo:
                hits += 2
            if f"v{VER}".lower() in lo:
                hits += 1
            if "_f" in lo:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best = cand

    return best


def _iter_h5_files(root_dir: str):
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".h5"):
                yield os.path.join(r, fn)


def _find_weight_file(models_dir: str, fold: int, ver: int) -> str | None:
    fn = f"EB2_v{ver}_f{fold}.h5"
    p = os.path.join(models_dir, fn)
    if os.path.isfile(p):
        return p

    h5_paths = list(_iter_h5_files(models_dir))
    if not h5_paths:
        return None

    want_fold = f"_f{fold}"
    want_ver = f"v{ver}"
    for path in h5_paths:
        xl = os.path.basename(path).lower()
        if want_fold in xl and want_ver in xl:
            return path
    for path in h5_paths:
        xl = os.path.basename(path).lower()
        if want_fold in xl:
            return path
    return None


if PLATFORM == "local":
    _sample_path = (
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
else:
    _sample_path = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
_sample = pd.read_csv(_sample_path)
SUB_COLS = [c for c in _sample.columns if c != "eeg_id"]


if not _TF_OK:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )

    pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float64)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[SUB_COLS] = pred.astype(np.float32)
    sub.to_csv("submission.csv", index=False)
    print("Wrote fallback uniform submission.csv because TensorFlow import failed.")
    print("Submission shape", sub.shape)

else:
    if not NEEDTRAIN:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
        elif PLATFORM == "kaggle":
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )
        print("Test shape", test.shape)

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
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()

        test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

        try:
            from scipy import signal  # type: ignore

            _HAS_SCIPY = True
        except Exception as e:
            _HAS_SCIPY = False
            _SCIPY_ERR = repr(e)
            signal = None

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        test_eeg_ids = set(test.eeg_id.values.tolist())

        if _HAS_SCIPY:
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
        else:
            print(
                "WARNING: scipy.signal not available, EEG filtering/resampling will be skipped:",
                _SCIPY_ERR,
            )

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])

            if name in test_eeg_ids:
                time_temp = 0
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

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

                    if _HAS_SCIPY:
                        if 200 != SFREQ:
                            eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)
                        eeg = signal.filtfilt(b, a, eeg, axis=1)

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs2[name] = list_eeg
        print()

        models_dir = _resolve_models_dir(LOAD_MODELS_FROM)
        if models_dir is None:
            print(
                f"WARNING: Could not find models directory. Tried: {LOAD_MODELS_FROM}"
            )
        else:
            print(f"Using models directory: {models_dir}")
            try:
                print(
                    "Model dir .h5 files found (first 20, recursive):",
                    sorted(
                        [
                            os.path.relpath(p, models_dir)
                            for p in list(_iter_h5_files(models_dir))
                        ]
                    )[:20],
                )
            except Exception:
                pass

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

        if models_dir is not None:
            for i in range(5):
                wpath = _find_weight_file(models_dir, fold=i, ver=VER)
                if wpath is None:
                    print(
                        f"WARNING: No weights found for fold {i} (ver {VER}). Skipping."
                    )
                    continue
                print(f"Fold {i+1} loading: {wpath}")
                model.load_weights(wpath)
                pred = model.predict(test_gen, verbose=1)
                preds.append(pred)

        if len(preds) == 0:
            print("WARNING: No fold predictions produced (no weights loaded).")
            pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float64)
        else:
            pred = np.mean(preds, axis=0)

        print("Test preds shape", pred.shape)

        pred = np.asarray(pred, dtype=np.float64)
        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[SUB_COLS] = pred.astype(np.float32)

        sub = sub[["eeg_id"] + SUB_COLS]
        sub.to_csv("submission.csv", index=False)

        print("Submission shape", sub.shape)
        print(
            "Row sums (min/mean/max):",
            sub[SUB_COLS].sum(axis=1).min(),
            sub[SUB_COLS].sum(axis=1).mean(),
            sub[SUB_COLS].sum(axis=1).max(),
        )
