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

0.3300212734455402

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the runtime crash caused by an incompatible EfficientNet wheel/TensorFlow+protobuf combo by removing the external `efficientnet` dependency and switching to `tf.keras.applications.EfficientNetB0` (same backbone idea, minimal architectural change) so the notebook runs on Kaggle’s Python 3.12 environment. I also remove unused `albumentations` import (not installed) and add a robust, lightweight fallback that uses `sample_submission.csv` if model weight files are missing, ensuring a valid `submission.csv` is always written. Finally, I enforce numerical safety for the submission by clipping to small epsilon and renormalizing so each row sums to 1 (required for this competition). These changes are execution/stability focused and should yield a valid submission; if pretrained weights load successfully, the score should move toward the target compared to a uniform fallback.'
- What this solution (achieved 1.39779) has done: 'I fix the immediate runtime crash by removing the problematic `librosa` import (it’s not needed for your current `DATATYPE` and it triggers the protobuf/TF incompatibility in this environment), and by avoiding any internal TensorFlow graph reset calls. I also make GPU strategy selection safe when no GPU exists (fall back to CPU) so the notebook runs on all Kaggle runtimes. To nudge the score toward your target (lower is better) with minimal semantic change, I enable reading spectrogram parquets (`DATATYPE` already includes `"spe"`) and keep the rest of the pipeline/model unchanged; missing model weights still fall back to a valid prior-based submission. Finally, I keep the submission numerically safe (clip + renormalize) so rows sum to 1 and Kaggle accepts the file.'
- What this solution (achieved 1.39779) has done: 'I fix the immediate crash happening at `import tensorflow as tf` (the protobuf `MessageFactory.GetPrototype` AttributeError) by forcing TensorFlow to use the Python protobuf implementation before TF is imported, which is the standard workaround on Kaggle Python 3.12 when protobuf wheels mismatch. I also make model-weight loading more robust by accepting both `.h5` and `.weights.h5` filenames so the intended ensemble is actually used when weights exist (this should improve KL toward your target, since your current score suggests the fallback/untrained path is being used). Finally, I keep your existing prediction safety (clip + renormalize) and submission format unchanged to ensure Kaggle acceptance.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and pinning it early, before anything that might indirectly import protobuf/TensorFlow. Then I keep your model/data pipeline unchanged, but make weight discovery more robust by also checking common Kaggle-input nesting patterns so your actual trained ensemble loads instead of falling back to a prior (which is likely why the score is stuck around 1.39). Finally, I keep the existing numeric safety (clip + renormalize) and ensure a valid `submission.csv` is always written with correct columns and row sums of 1.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime *before* any protobuf/TensorFlow-related imports, and by adding a safe fallback that skips TF entirely (and still writes a valid submission) if TF cannot be imported in this Kaggle Python 3.12 environment. To move the score down toward your target (lower is better) with minimal semantic change, I ensure the intended trained weights are actually discovered/loaded when present (more robust search patterns), because your current score strongly suggests the prior/untrained fallback path is being used. I also keep the existing probability clipping + renormalization to guarantee each row sums to 1 (required by Kaggle) and keep the model architecture/training logic unchanged.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is selected *before any TensorFlow/protobuf import happens*, and by clearing any already-imported protobuf modules to avoid the stale C++ runtime causing `MessageFactory.GetPrototype` errors. To move the KL score down toward your target, I also make weight discovery less brittle (without changing model logic) so the intended trained fold weights are actually found/loaded when present—your current 1.39779 strongly suggests it’s falling back to the prior/untrained path. Finally, I keep your existing probability safety (clip + renormalize) and guarantee a valid `submission.csv` is written even if TF still cannot import.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime earlier and more robustly (including disabling the C++ implementation and clearing any preloaded protobuf modules before importing TensorFlow). This unblocks the full inference path so your pretrained fold weights can actually load; your current KL=1.39779 is consistent with the fallback prior being used, so enabling real model predictions should move the score down toward the 0.33 target. I also make weight-file discovery tolerant of common filename patterns (including `*.keras`) while keeping your model and prediction logic unchanged. Finally, I keep the existing numeric safety (clip + renormalize) to guarantee each row sums to 1 and the submission is accepted.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow import, and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at the very top (plus clearing any already-loaded protobuf modules). This should unblock TensorFlow so the intended EfficientNet-based model inference can run instead of falling back to the train-prior baseline (which is consistent with the current poor KL score). I also make the runtime robust by only importing `scipy` when needed and by ensuring the submission probabilities are always finite, clipped, and row-normalized (required for Kaggle validity). Core model/data logic is preserved; changes are limited to import-order/robustness so you can actually load weights and predict.'

# 9. Code solution

## === cell 0
import os
import io
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX", "1")

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        del sys.modules[_m]

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024040502"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024
BATCHSIZE = 16

READ_SPEC_FILES = True
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

spectrograms = {}
eegs = {}
imgs = {}
stfts = {}

spectrograms2 = {}
eegs2 = {}
imgs2 = {}
stfts2 = {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["PYTHONHASHSEED"] = str(SEED)

TF_AVAILABLE = False
tf = None
strategy = None
try:
    import tensorflow as tf  # noqa: E402

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) >= 1:
        if len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print("Using 1 GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("Using CPU")

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision requested but not available; continuing.")
    else:
        print("Using full precision")

    TF_AVAILABLE = True
except Exception as e:
    print("WARNING: TensorFlow could not be imported; will use fallback submission.")
    print("TF import error:", repr(e))

VER = 1



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if PLATFORM == "local":
    TRAIN_CSV = "./input/hms-harmful-brain-activity-classification/train.csv"
    TEST_CSV = "./input/hms-harmful-brain-activity-classification/test.csv"
    SAMPLE_SUB = (
        "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
    TRAIN_SPEC_DIR = (
        "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    TRAIN_EEG_DIR = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    TEST_SPEC_DIR = (
        "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TEST_EEG_DIR = "./input/hms-harmful-brain-activity-classification/test_eegs/"
else:
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    SAMPLE_SUB = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
    TRAIN_SPEC_DIR = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    TRAIN_EEG_DIR = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    )
    TEST_SPEC_DIR = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TEST_EEG_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

df = pd.read_csv(TRAIN_CSV)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## === cell 2
if NEEDTRAIN:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]

    if READ_SPEC_FILES * READ_EEG_FILES * READ_IMG_FILES * READ_STFT_FILES:
        train_max = df.groupby("eeg_id")[["eeg_label_offset_seconds"]].agg(
            {"eeg_label_offset_seconds": "max"}
        )
        train_min = df.groupby("eeg_id")[["eeg_label_offset_seconds"]].agg(
            {"eeg_label_offset_seconds": "min"}
        )

        train_max.columns = ["eeg_label_offset_seconds_max"]
        train_min.columns = ["eeg_label_offset_seconds_min"]

        df2 = df.merge(train_max, on="eeg_id")
        df2 = df2.merge(train_min, on="eeg_id")

        df2["s_max"] = abs(
            df2.eeg_label_offset_seconds - df2.eeg_label_offset_seconds_max
        )
        df2["s_min"] = abs(
            df2.eeg_label_offset_seconds - df2.eeg_label_offset_seconds_min
        )

        xx = df2.loc[:, ["s_max", "s_min"]].min(1)
        df2 = df2.iloc[:, :15]
        df2["selected"] = xx

        df2 = df2.sort_values("selected", ascending=False).reset_index(drop=True)
        df3 = df2.drop_duplicates("eeg_id").reset_index(drop=True)

        train = pd.DataFrame()
        train = pd.concat([train, df3]).reset_index(drop=True)
        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data
        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 3
if TF_AVAILABLE:
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
            imgs=None,
            stfts=None,
            targets=None,
        ):
            self.targets = targets
            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["cividis"](np.linspace(0, 1, 256))[:, :3]

            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.imgs = imgs
            self.stfts = stfts
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.data) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y = self.__data_generation(indexes)
            return x, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (len(indexes), 6, round(20 * SFREQ), 3, 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros(
                    (len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32"
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")

            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    r_spe = round(
                        getattr(row, "spectrogram_label_offset_seconds", 0) / 2
                    )
                    r_eeg = round(getattr(row, "eeg_label_offset_seconds", 0) * SFREQ)

                if self.mode == "train":
                    x1 = np.random.rand() * (LENGTH / 2 - 20)
                    x2 = np.random.rand() * (LENGTH / 2 - 20)
                    if np.random.rand() < 0.5:
                        x1 = x1 + LENGTH / 2
                        x2 = x2 + LENGTH / 2
                    x_spe_min = round(min(x1, x2))
                    x_spe_max = round(max(x1, x2))

                    x1 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                    x2 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                    if np.random.rand() < 0.5:
                        x1 = x1 + EEG_LENGTH * SFREQ / 2
                        x2 = x2 + EEG_LENGTH * SFREQ / 2
                    else:
                        x1 = x1 + 10 * SFREQ / 2
                        x2 = x2 + 10 * SFREQ / 2
                    x_eeg_min = round(min(x1, x2))
                    x_eeg_max = round(max(x1, x2))

                    x1 = np.random.rand() * (LENGTH / 2 - 42)
                    x2 = np.random.rand() * (LENGTH / 2 - 42)
                    if np.random.rand() < 0.5:
                        x1 = x1 + LENGTH / 2
                        x2 = x2 + LENGTH / 2
                    else:
                        x1 = x1 + 42
                        x2 = x2 + 42
                    x_img_min = round(min(x1, x2))
                    x_img_max = round(max(x1, x2))

                for k in range(4):
                    if "spe" in DATATYPE:
                        spe = self.specs[row.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T
                        spe = np.nan_to_num(spe, nan=0.0)
                        spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                        spe = np.log(spe)
                        spe = np.round(
                            (spe - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                        spe = np.array(spe, dtype=np.int16)
                        spe = self.cmaps[spe]
                        spe = np.reshape(spe, (100, 300, 3))
                        spe = spe[
                            :,
                            max(round((600 / 2 - 256) / 2), 0) : min(
                                (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
                            ),
                            :,
                        ]
                        if self.mode == "train":
                            spe[:, x_spe_min:x_spe_max, :] = 0
                        x_spe[
                            j,
                            round((HIGH - spe.shape[0]) / 2) : round(
                                (HIGH + spe.shape[0]) / 2
                            ),
                            :,
                            :,
                            k,
                        ] = spe
                        x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                    if "eeg" in DATATYPE:
                        eeg = self.eegs[row.eeg_id][
                            :, r_eeg : r_eeg + round(50 * SFREQ), k
                        ]
                        eeg1 = eeg[:, round(10 * SFREQ) : round(30 * SFREQ)]
                        eeg2 = eeg[:, round(15 * SFREQ) : round(35 * SFREQ)]
                        eeg3 = eeg[:, round(20 * SFREQ) : round(40 * SFREQ)]
                        x_eeg[j, 1:5, :, 0, k] = eeg1
                        x_eeg[j, 1:5, :, 1, k] = eeg2
                        x_eeg[j, 1:5, :, 2, k] = eeg3
                        x_eeg[j, :, :, :, k] = (
                            x_eeg[j, :, :, :, k]
                            - np.mean(x_eeg[j, :, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, :, k], 1, keepdims=True) + 1e-6)

                    if "img" in DATATYPE:
                        if self.mode == "test":
                            img = self.imgs[row.eeg_id][:, :, k, :]
                        else:
                            img = self.imgs[row.sign_id][:, :, k, :]
                        x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        if self.mode == "test":
                            stft = self.stfts[row.eeg_id][:, :, k]
                        else:
                            stft = self.stfts[row.sign_id][:, :, k]
                        stft = np.nan_to_num(stft, nan=0.0)
                        cmin = 0
                        cmax = 60
                        stft = np.clip(stft, cmin, cmax)
                        stft = np.round((stft - cmin) / (cmax - cmin) * 255)
                        shape0, shape1 = stft.shape[0], stft.shape[1]
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)
                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (shape0, shape1, 3))
                        x_stft[
                            j,
                            round((HIGH - stft.shape[0]) / 2) : round(
                                (HIGH + stft.shape[0]) / 2
                            ),
                            :,
                            :,
                            k,
                        ] = stft
                        x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                if self.mode != "test":
                    label = row[self.targets].values.astype("float32")
                    if self.mode == "train" and np.sum(label == 1):
                        xx_eps = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx_eps
                        label[label == 1] = 1 - 5 * xx_eps
                    y[j] = label

            x = []
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "img" in DATATYPE:
                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img
                x.append(x_img)
            if "stft" in DATATYPE:
                x.append(x_stft)

            return x, y




## === cell 4
if TF_AVAILABLE:

    def build_model(TARGETS_PRETRAIN):
        inp = []

        def _make_backbone(name: str):
            base = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=None,
            )
            base._name = name
            return base

        y = None

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [inp_spe[:, :, :, :, i] for i in range(4)]
            )
            base_model_spe = _make_backbone("spe_extractor")
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = tf.nn.l2_normalize(x_spe, -1)
            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg = tf.keras.layers.Concatenate(axis=1)(
                [inp_eeg[:, :, :, :, i] for i in range(4)]
            )
            base_model_eeg = _make_backbone("eeg_extractor")
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.nn.l2_normalize(x_eeg, -1)
            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [inp_img[:, :, :, :, i] for i in range(4)]
            )
            base_model_img = _make_backbone("img_extractor")
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = tf.nn.l2_normalize(x_img, -1)
            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [inp_stft[:, :, :, :, i] for i in range(4)]
            )
            base_model_stft = _make_backbone("stft_extractor")
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = tf.nn.l2_normalize(x_stft, -1)
            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                if y is not None
                else x_stft
            )

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

    def my_loss(y_ture, y_pred):
        y_pred1 = y_pred[:, 5:6]
        y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 5
test = pd.read_csv(TEST_CSV)
print("Test shape", test.shape)
test.head()

if not TF_AVAILABLE:
    y = df[TARGETS].values.astype("float64")
    y = y / np.clip(y.sum(axis=1, keepdims=True), 1e-12, None)
    prior = y.mean(axis=0)
    prior = prior / prior.sum()
    pred = np.tile(prior[None, :], (len(test), 1)).astype("float32")

    eps = 1e-8
    pred = np.nan_to_num(pred, nan=1.0 / len(TARGETS), posinf=1.0, neginf=0.0)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Wrote fallback prior-based submission.csv")
    print("Submission shape", sub.shape)
    print(sub.head())
else:
    if "spe" in DATATYPE and READ_SPEC_FILES:
        files2 = os.listdir(TEST_SPEC_DIR)
        print(f"There are {len(files2)} test spectrogram parquets")
        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{TEST_SPEC_DIR}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()
    elif "spe" in DATATYPE:
        raise RuntimeError(
            "DATATYPE includes 'spe' but READ_SPEC_FILES is False; cannot build 'spe' inputs."
        )

    from scipy import signal

    files2 = os.listdir(TEST_EEG_DIR)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}

    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    test_eeg_ids = set(test.eeg_id.values.tolist())

    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        if name not in test_eeg_ids:
            continue

        eeg_default = pd.read_parquet(f"{TEST_EEG_DIR}{f}")

        list_eeg = []
        list_img = []
        list_stft = []
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

            time_temp = 0
            time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
            time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

            list_img.append(eeg[:, time_start:time_stop])

            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)

        if "eeg" in DATATYPE:
            eegs2[name] = list_eeg

        if "img" in DATATYPE:
            eeg_all_region = np.concatenate(list_img, 0)

            fig = plt.figure(clear=True)
            fig.patch.set_facecolor("black")
            amp = 200
            for ii in range(eeg_all_region.shape[0]):
                jj = ii * amp + (ii // 4) * amp
                plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
            plt.xlim(-10, eeg_all_region.shape[1] + 10)
            plt.ylim(-amp / 2, eeg_all_region.shape[0] * amp + amp / 2 * 5)
            plt.axis("off")

            byte_stream = io.BytesIO()
            plt.savefig(byte_stream, format="png", bbox_inches="tight")
            byte_stream.seek(0)
            img = Image.open(byte_stream)
            img = np.array(img)[:, :, :1]
            byte_stream.truncate()
            plt.close("all")

            img = np.concatenate((img, img, img), 2)
            img = np.array(
                tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
            )
            img = img[:, :, 0:1]

            img = np.concatenate(
                [
                    img[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                    img[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                    img[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                    img[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                ],
                -1,
            )

            img[:, :, 0] = -img[:, :, 0]
            img[:, :, 2] = -img[:, :, 2]
            img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))

            eeg_all_region2 = eeg_all_region[
                :,
                round(eeg_all_region.shape[1] * 1 / 4) : round(
                    eeg_all_region.shape[1] * 3 / 4
                ),
            ]
            fig = plt.figure(clear=True)
            fig.patch.set_facecolor("black")
            amp = 150
            for ii in range(eeg_all_region2.shape[0]):
                jj = ii * amp + (ii // 4) * amp
                plt.plot(eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5)
            plt.xlim(-5, eeg_all_region2.shape[1] + 5)
            plt.ylim(-amp / 2, eeg_all_region2.shape[0] * amp + amp / 2 * 5)
            plt.axis("off")

            byte_stream = io.BytesIO()
            plt.savefig(byte_stream, format="png", bbox_inches="tight")
            byte_stream.seek(0)
            img2 = Image.open(byte_stream)
            img2 = np.array(img2)[:, :, :1]
            byte_stream.truncate()
            plt.close("all")

            img2 = np.concatenate((img2, img2, img2), 2)
            img2 = np.array(
                tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
            )
            img2 = img2[:, :, 0:1]

            img2 = np.concatenate(
                [
                    img2[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                    img2[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                    img2[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                    img2[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                ],
                -1,
            )

            img2[:, :, 0] = -img2[:, :, 0]
            img2[:, :, 2] = -img2[:, :, 2]
            img2 = np.reshape(img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1))

            eeg_all_region3 = eeg_all_region[
                :,
                round(eeg_all_region.shape[1] * 2 / 5) : round(
                    eeg_all_region.shape[1] * 3 / 5
                ),
            ]
            fig = plt.figure(clear=True)
            fig.patch.set_facecolor("black")
            amp = 100
            for ii in range(eeg_all_region3.shape[0]):
                jj = ii * amp + (ii // 4) * amp
                plt.plot(eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5)
            plt.xlim(-2, eeg_all_region3.shape[1] + 2)
            plt.ylim(-amp / 2, eeg_all_region3.shape[0] * amp + amp / 2 * 5)
            plt.axis("off")

            byte_stream = io.BytesIO()
            plt.savefig(byte_stream, format="png", bbox_inches="tight")
            byte_stream.seek(0)
            img3 = Image.open(byte_stream)
            img3 = np.array(img3)[:, :, :1]
            byte_stream.truncate()
            plt.close("all")

            img3 = np.concatenate((img3, img3, img3), 2)
            img3 = np.array(
                tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
            )
            img3 = img3[:, :, 0:1]

            img3 = np.concatenate(
                [
                    img3[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                    img3[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                    img3[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                    img3[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                ],
                -1,
            )

            img3[:, :, 0] = -img3[:, :, 0]
            img3[:, :, 2] = -img3[:, :, 2]
            img3 = np.reshape(img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1))

            img = np.concatenate([img, img2, img3], -1)
            imgs2[name] = img

    print()

    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2 if "spe" in DATATYPE else None,
        eegs=eegs2 if "eeg" in DATATYPE else None,
        imgs=imgs2 if "img" in DATATYPE else None,
        stfts=stfts2 if "stft" in DATATYPE else None,
        targets=TARGETS,
    )

    model_dir = Path(LOAD_MODELS_FROM)
    candidate_model_dirs = [
        model_dir,
        model_dir / model_dir.name,
        model_dir / "models",
        model_dir / "weights",
        model_dir / "model",
        model_dir / "checkpoints",
        model_dir / "checkpoint",
    ]

    if not model_dir.exists() and PLATFORM == "kaggle":
        base = Path("/kaggle/input")
        if base.exists():
            hits = list(base.rglob(model_dir.name))
            for h in hits[:50]:
                if h.is_dir():
                    candidate_model_dirs.extend(
                        [h, h / "models", h / "weights", h / "checkpoints"]
                    )

    if model_dir.exists() and model_dir.is_dir():
        for p in model_dir.glob("*"):
            if p.is_dir():
                candidate_model_dirs.extend(
                    [p, p / "models", p / "weights", p / "checkpoints"]
                )

    seen = set()
    candidate_model_dirs2 = []
    for p in candidate_model_dirs:
        if p.exists() and p.is_dir():
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                candidate_model_dirs2.append(p)
    candidate_model_dirs = candidate_model_dirs2
    if not candidate_model_dirs:
        candidate_model_dirs = [model_dir]  # keep original for messaging

    def _find_weight_path(fold: int, stage: int):
        candidates = []
        for md in candidate_model_dirs:
            candidates.extend(
                [
                    md / f"f{fold}_stage{stage}.h5",
                    md / f"f{fold}_stage{stage}.weights.h5",
                    md / f"f{fold}_stage{stage}.keras",
                    md / f"fold{fold}_stage{stage}.h5",
                    md / f"fold{fold}_stage{stage}.weights.h5",
                    md / f"fold{fold}_stage{stage}.keras",
                    md / f"f{fold+1}_stage{stage}.h5",
                    md / f"f{fold+1}_stage{stage}.weights.h5",
                    md / f"f{fold+1}_stage{stage}.keras",
                    md / f"fold{fold+1}_stage{stage}.h5",
                    md / f"fold{fold+1}_stage{stage}.weights.h5",
                    md / f"fold{fold+1}_stage{stage}.keras",
                    md / f"fold{fold}.h5",
                    md / f"fold{fold}.weights.h5",
                    md / f"fold{fold}.keras",
                    md / f"fold{fold+1}.h5",
                    md / f"fold{fold+1}.weights.h5",
                    md / f"fold{fold+1}.keras",
                ]
            )
        for p in candidates:
            if p.exists():
                return p
        return None

    weight_paths = [_find_weight_path(i, STAGETEST) for i in range(5)]
    have_weights = all(p is not None for p in weight_paths)

    if have_weights:
        preds = []
        with strategy.scope():
            model = build_model(TARGETS)
        for i in range(5):
            wp = weight_paths[i]
            print(f"Fold {i+1} weights: {wp}")
            model.load_weights(str(wp))
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
        pred = np.mean(preds, axis=0)
    else:
        print(
            f"WARNING: Missing model weights in {LOAD_MODELS_FROM} (searched: {candidate_model_dirs}). "
            "Writing prior-based submission instead."
        )
        y = df[TARGETS].values.astype("float64")
        y = y / np.clip(y.sum(axis=1, keepdims=True), 1e-12, None)
        prior = y.mean(axis=0)
        prior = prior / prior.sum()
        pred = np.tile(prior[None, :], (len(test), 1)).astype("float32")

    eps = 1e-8
    pred = np.nan_to_num(pred, nan=1.0 / len(TARGETS), posinf=1.0, neginf=0.0)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sums (min/mean/max):",
        sub[TARGETS].sum(1).min(),
        sub[TARGETS].sum(1).mean(),
        sub[TARGETS].sum(1).max(),
    )
