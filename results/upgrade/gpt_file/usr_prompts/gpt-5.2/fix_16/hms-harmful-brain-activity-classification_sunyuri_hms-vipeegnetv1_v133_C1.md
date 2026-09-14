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

0.3389501665016398

# 6. Current score

1.03204

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The protobuf/TensorFlow import crash is coming from an incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` setup for this Kaggle image, so I force the pure-Python protobuf implementation before importing TensorFlow to make the notebook start reliably. Next, your inference currently hard-fails because the expected weight directory `/kaggle/input/models2024032601` isn’t available; to keep the pipeline end-to-end, I add a safe fallback that generates a valid, properly-normalized submission using `sample_submission.csv`’s class-prior (global mean) computed from `train.csv` when weights are missing. This keeps the core model code intact (no architecture/training changes) while ensuring a submission is always produced and should score meaningfully better than uniform guesses. I also fix the cell numbering to start at 1 per your required format.'
- What this solution (achieved 1.29417) has done: 'I fix the runtime crash that prevents TensorFlow from importing by setting the protobuf implementation to the pure-Python backend *before any TensorFlow/protobuf-related imports* and by avoiding any accidental early TF import paths. Then, to move the score toward your target (lower is better) without changing the model/training logic, I improve the existing “missing weights” fallback from a single global prior to a patient-conditional prior (computed from `train.csv`), with a safe fallback to the global prior for unseen patients; this is still a legitimate calibration step aligned with the KL metric. Finally, I ensure the produced `submission.csv` always has the exact required columns/order and each row sums to 1.0.'
- What this solution (achieved 1.29417) has done: 'I fix the immediate runtime crash by forcing protobuf to use the pure-Python implementation *before* anything that might import `google.protobuf`/TensorFlow, and (critically) by downgrading the protobuf package at runtime to a TF-compatible version if the environment has a newer incompatible protobuf that triggers the `MessageFactory.GetPrototype` error. This is a correctness/stability fix and does not change your modeling logic. After TensorFlow imports cleanly, the rest of your pipeline stays the same, including the existing patient-conditional prior fallback when model weights are missing. Finally, I keep the submission writing logic unchanged but ensure the file is always produced as `submission.csv` with correct columns/order and row-wise probability normalization.'
- What this solution (achieved 1.29417) has done: 'Your current score (1.29417, lower is better) is far above the target (0.33895), and the main reason is that you’re still falling back to a weak “patient-conditional prior” because the expected model weights directory is not present. The smallest high-impact change is to actually use a real pretrained weights source that exists in the Kaggle dataset: load Keras weights directly from the provided `/kaggle/input/hms-harmful-brain-activity-classification/` folder by auto-discovering `f*_stage3.h5` files there (and only falling back to priors if truly none exist). This keeps your model architecture/inference loop intact while enabling the intended ensemble inference, which should move KL much closer to the target band. I also ensure strict submission validity (column order + row-wise sum-to-1 + clipping) remains unchanged.'
- What this solution (achieved 1.01346) has done: 'Your current score (1.29417, lower is better) is far from the target (0.33895), and the main limiter is that the code usually falls back to patient/global priors because it requires *all* 5 fold weight files to exist. I make the smallest change that enables real model inference when only some folds are available: load and ensemble over whatever fold weight files are actually found, and only fall back to priors if none are found. To keep evaluation semantics valid for KL, I also apply a tiny probability floor + renormalization after prediction (and in the fallback) to avoid extreme zeros that can blow up KL. Paths and core model/data logic stay the same; this just unblocks using pretrained weights when present.'
- What this solution (achieved 1.01346) has done: 'Your current score (1.01346, lower is better) is still far above the target (0.33895), so we should carefully improve predictions without changing the model or feature pipeline. The biggest low-risk gain is to align inference-time preprocessing with what the training likely used: enable spectrogram loading (`READ_SPEC_FILES=True`) so the model gets its expected multi-modal inputs rather than partially-empty tensors. To keep KL stable, I keep the existing probability floor+renormalization and also ensure we average predictions only across successfully-loaded folds. All paths stay the same and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.01346) has done: 'I make two minimal, score-relevant adjustments aimed at reducing KL (lower is better) without changing your model architecture or training/inference loop. First, I fix the spectrogram preprocessing normalization bug: you currently divide by `std**2` instead of `std`, which can badly mis-scale inputs and hurt predictions. Second, I ensure model inference runs with `training=False` explicitly (disables dropout/other training-time behavior) and keep the existing probability floor+renormalization for KL stability. These changes are tiny, keep core semantics intact, and should move your 1.01346 closer to the 0.33895 target.'
- What this solution (achieved 1.01346) has done: 'We make two tiny, inference-only fixes that should reduce KL (lower is better) without changing your model architecture or data pipeline: (1) ensure inference runs with `training=False` so BatchNorm/Dropout behave deterministically as in eval mode, and (2) apply the same probability floor + renormalization after each fold prediction (not just after averaging) to prevent any near-zeros from dominating KL. These are minimal, score-relevant calibration/stability changes and keep all paths and submission format identical. The rest of the logic (data loading, model definition, ensembling) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.03204) has done: 'I make two minimal, inference-only fixes that are directly score-relevant for KL divergence while keeping your model, data loading, and ensembling logic intact. First, I fix the `stft` channel normalization bug (it currently divides by `std**2` instead of `std`), which can severely mis-scale inputs if `stft` is ever enabled and is a clear correctness issue. Second, I apply a small “logit/temperature sharpening” calibration on the final averaged probabilities (not per-fold, and only at inference) to reduce over-smooth predictions; this commonly lowers KL for this competition and is a tiny post-processing step that preserves evaluation semantics (still valid probabilities summing to 1). The rest of the pipeline (weight discovery, generator, model definition, prediction, submission formatting) stays unchanged and still always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import io
import sys
import warnings
import subprocess

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    try:
        if pb_ver is not None:
            major = int(str(pb_ver).split(".")[0])
            if major >= 5:
                raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception:
        print(
            "Adjusting protobuf to a TensorFlow-compatible version (protobuf==4.25.3)..."
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

from scipy import signal

print("TensorFlow version =", tf.__version__)

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print("DATATYPE =", DATATYPE)

LOAD_MODELS_FROM = "models2024032601"
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
NSPLIT = 5
BATCHSIZE = 16

READ_SPEC_FILES = True
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

spectrograms, eegs, imgs, stfts = {}, {}, {}, {}
spectrograms2, eegs2, imgs2, stfts2 = {}, {}, {}, {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enable warning:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision option warning:", repr(e))
else:
    print("Using full precision")

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

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

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



## === cell 2
TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}


class DataGenerator(tf.keras.utils.Sequence):
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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
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
            x_eeg = np.zeros((len(indexes), 4, round(50 * SFREQ), 4), dtype="float32")
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")

        y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            elif self.mode == "valid":
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = round(row.eeg_label_offset_seconds * SFREQ)
            else:
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = round(row.eeg_label_offset_seconds * SFREQ)

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

                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
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
                    spe = np.array(
                        tf.image.resize(spe, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

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

                    x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / 0.229
                    x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / 0.224
                    x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / 0.225

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]
                    eeg = eeg[:, round(0 * SFREQ) : round(50 * SFREQ)]
                    x_eeg[j, :, :, k] = eeg

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
                    stft = np.round(
                        (stft - np.min(stft))
                        / (np.max(stft) - np.min(stft) + 1e-6)
                        * 255
                    )

                    shape0, shape1 = stft.shape[0], stft.shape[1]
                    stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                    stft = np.array(stft, dtype=np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (shape0, shape1, 3))
                    stft = np.array(
                        tf.image.resize(stft, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

                    x_stft[
                        j,
                        round((HIGH - stft.shape[0]) / 2) : round(
                            (HIGH + stft.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = stft
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / 0.229
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / 0.224
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / 0.225

            if self.mode != "test":
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        if "eeg" in DATATYPE:
            for i_eeg in range(x_eeg.shape[0]):
                xx = np.std(x_eeg[i_eeg, :, :, :], 1, keepdims=True)
                xx = np.mean(xx)
                x_eeg[i_eeg, :, :, :] = (
                    x_eeg[i_eeg, :, :, :]
                    - np.mean(x_eeg[i_eeg, :, :, :], 1, keepdims=True)
                ) / (xx + 1e-6)

        x = []
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "img" in DATATYPE:
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)
        if "stft" in DATATYPE:
            x.append(x_stft)

        return x, y




## === cell 3
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


def _effnet_b0_backbone(name_prefix: str):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, name=f"{name_prefix}_efficientnetb0"
    )

    def _clone_fn(layer):
        cfg = layer.get_config()
        cfg["name"] = f"{name_prefix}_{cfg['name']}"
        return layer.__class__.from_config(cfg)

    cloned = tf.keras.models.clone_model(base, clone_function=_clone_fn)
    cloned._name = f"{name_prefix}_efficientnetb0"
    return cloned


def build_model(TARGETS_PRETRAIN):
    inp = []
    y = None

    l2norm = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1, name="concat_spe_4")(
            [x_spe1, x_spe2, x_spe3, x_spe4]
        )

        base_model_spe = _effnet_b0_backbone("spe_extractor")
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
        x_spe = l2norm(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(4, round(50 * SFREQ), 4), name="inp_eeg")

        kernal_num = 64
        x_eeg1 = inp_eeg[:, :, :, 0:1]
        x_eeg2 = inp_eeg[:, :, :, 1:2]
        x_eeg3 = inp_eeg[:, :, :, 2:3]
        x_eeg4 = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_4")(
            [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
        )

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 1, (1, 8), padding="valid", name="eeg_conv1"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu1")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn1")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool1")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 1, (1, 6), padding="valid", name="eeg_conv2"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu2")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn2")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool2")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 1, (1, 4), padding="valid", name="eeg_conv3"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu3")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn3")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool3")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 2, (4, 4), padding="valid", strides=(4, 1), name="eeg_conv4"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu4")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn4")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool4")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 4, (1, 8), padding="valid", name="eeg_conv5"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu5")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn5")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool5")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 4, (1, 4), padding="valid", name="eeg_conv6"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu6")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn6")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool6")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 8, (4, 4), padding="valid", strides=(4, 1), name="eeg_conv7"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu7")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn7")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool7")(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 16, (1, 4), padding="valid", name="eeg_conv8"
        )(x_eeg)
        x_eeg = tf.keras.layers.Activation("relu", name="eeg_relu8")(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization(name="eeg_bn8")(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2), name="eeg_pool8")(x_eeg)

        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = l2norm(x_eeg)

        inp.append(inp_eeg)
        if y is not None:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_spe_eeg")([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
        x_img1 = inp_img[:, :, :, :, 0]
        x_img2 = inp_img[:, :, :, :, 1]
        x_img3 = inp_img[:, :, :, :, 2]
        x_img4 = inp_img[:, :, :, :, 3]
        x_img = tf.keras.layers.Concatenate(axis=1, name="concat_img_4")(
            [x_img1, x_img2, x_img3, x_img4]
        )

        base_model_img = _effnet_b0_backbone("img_extractor")
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = l2norm(x_img)

        inp.append(inp_img)
        if y is not None:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_all")([y, x_img])
        else:
            y = x_img

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
        x_stft1 = inp_stft[:, :, :, :, 0]
        x_stft2 = inp_stft[:, :, :, :, 1]
        x_stft3 = inp_stft[:, :, :, :, 2]
        x_stft4 = inp_stft[:, :, :, :, 3]
        x_stft = tf.keras.layers.Concatenate(axis=1, name="concat_stft_4")(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )

        base_model_stft = _effnet_b0_backbone("stft_extractor")
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
        x_stft = l2norm(x_stft)

        inp.append(inp_stft)
        if y is not None:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_all2")([y, x_stft])
        else:
            y = x_stft

    y = tf.keras.layers.Dense(
        len(TARGETS_PRETRAIN),
        activation="softmax",
        dtype="float32",
        name="head_softmax",
    )(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    return model


def my_loss(y_ture, y_pred):
    y_pred1 = y_pred[:, 5:6]
    y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
    y_pred2 = y_pred[:, 0:5]
    y_pred = tf.concat((y_pred2, y_pred1), axis=1)
    return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 4
if not NEEDTRAIN:
    sample_path = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        if PLATFORM == "kaggle"
        else "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
    sample = pd.read_csv(sample_path)

    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    def _discover_weight_paths(stage: int, nsplit: int):
        preferred = [
            os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{stage}.h5")
            for i in range(nsplit)
        ]
        if all(os.path.exists(p) for p in preferred):
            return preferred

        search_roots = []
        if PLATFORM == "kaggle":
            search_roots = [
                "/kaggle/input/hms-harmful-brain-activity-classification",
                "/kaggle/input",
            ]
        else:
            search_roots = [
                "./input/hms-harmful-brain-activity-classification",
                "./input",
            ]

        found = {i: None for i in range(nsplit)}
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for i in range(nsplit):
                    fname = f"f{i}_stage{stage}.h5"
                    if fname in filenames and found[i] is None:
                        found[i] = os.path.join(dirpath, fname)
                if all(found[i] is not None for i in range(nsplit)):
                    break
            if all(found[i] is not None for i in range(nsplit)):
                break

        return [found[i] for i in range(nsplit)]

    weight_paths = _discover_weight_paths(STAGETEST, NSPLIT)
    existing_weight_paths = [
        p for p in weight_paths if (p is not None and os.path.exists(p))
    ]
    have_any_weights = len(existing_weight_paths) > 0
    print("Weight paths (all):", weight_paths)
    print("Existing weights:", existing_weight_paths)
    print("Have any weights:", have_any_weights)

    PROB_FLOOR = 1e-6

    TEMPERATURE = 0.90  # <1 => slightly sharper; small change to avoid destabilizing
    EPS_LOGIT = 1e-12

    if not have_any_weights:
        print(
            "Warning: No model weights found, will create prior-based baseline submission."
        )

        y = df[TARGETS].values.astype(np.float64)
        y = np.clip(y, 0.0, None)
        y = y / (y.sum(axis=1, keepdims=True) + 1e-12)
        df_norm = df[["patient_id"]].copy()
        for i, t in enumerate(TARGETS):
            df_norm[t] = y[:, i]

        global_prior = df_norm[TARGETS].mean(axis=0).values.astype(np.float64)
        global_prior = np.clip(global_prior, PROB_FLOOR, 1.0)
        global_prior = global_prior / global_prior.sum()

        pat_prior = df_norm.groupby("patient_id")[list(TARGETS)].mean()
        pat_prior = pat_prior.div(pat_prior.sum(axis=1), axis=0).astype(np.float64)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        test_pats = test["patient_id"].values
        out = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
        for idx, pid in enumerate(test_pats):
            if pid in pat_prior.index:
                vec = pat_prior.loc[pid].values
                vec = np.clip(vec, PROB_FLOOR, 1.0)
                vec = vec / vec.sum()
                out[idx] = vec
            else:
                out[idx] = global_prior

        out = np.clip(out, PROB_FLOOR, 1.0)
        out = out / out.sum(axis=1, keepdims=True)

        p = np.clip(out, PROB_FLOOR, 1.0)
        logits = np.log(p + EPS_LOGIT)
        logits = logits / TEMPERATURE
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = np.clip(p, PROB_FLOOR, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        out = p

        for j, t in enumerate(TARGETS):
            sub[t] = out[:, j].astype(np.float32)

        sub = sub[sample.columns]
        probs = sub[TARGETS].values.astype(np.float64)
        probs = np.clip(probs, PROB_FLOOR, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        sub[TARGETS] = probs.astype(np.float32)

        sub.to_csv("submission.csv", index=False)
        print("Submission saved: submission.csv")
        print("Submission shape", sub.shape)
        print(
            "Row prob sum stats:",
            float(sub[TARGETS].sum(axis=1).min()),
            float(sub[TARGETS].sum(axis=1).max()),
        )
        print(sub.head())
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                PATH_SPE = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            else:
                PATH_SPE = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

            files_spe = os.listdir(PATH_SPE)
            print(f"There are {len(files_spe)} test spectrogram parquets")
            spectrograms2 = {}
            for i, f in enumerate(files_spe):
                if i % 200 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_SPE}{f}")
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values
            print()

        if PLATFORM == "local":
            PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        else:
            PATH_EEG = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
            )

        files_eeg = os.listdir(PATH_EEG)
        print(f"There are {len(files_eeg)} test eeg parquets")

        eegs2, imgs2, stfts2 = {}, {}, {}
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        for i, f in enumerate(files_eeg):
            if i % 200 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")
            name = int(f.split(".")[0])

            if len(test[test.eeg_id == name]) > 0:
                list_eeg, list_img, list_stft = [], [], []
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

                    time_temp = 0
                    time_start = round(
                        time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ
                    )
                    time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)
                    list_img.append(eeg[:, time_start:time_stop])

                    if "stft" in DATATYPE:
                        raise RuntimeError(
                            "STFT requested but librosa is intentionally not imported due to environment incompatibility."
                        )

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)

                if "stft" in DATATYPE:
                    list_stft = np.concatenate(list_stft, 2)
                    stfts2[name] = list_stft

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
                        tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
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
                        plt.plot(
                            eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5
                        )
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
                        tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
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
                    img2 = np.reshape(
                        img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1)
                    )

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
                        plt.plot(
                            eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5
                        )
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
                        tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
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
                    img3 = np.reshape(
                        img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1)
                    )

                    img = np.concatenate([img, img2, img3], -1)
                    imgs2[name] = img

        print()

        preds = []
        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            imgs=imgs2,
            stfts=stfts2,
            targets=TARGETS,
        )

        with strategy.scope():
            model = build_model(TARGETS)

        for fold_i, wpath in enumerate(existing_weight_paths):
            print(f"Fold weights {fold_i+1}/{len(existing_weight_paths)}: {wpath}")
            try:
                model.load_weights(wpath)
            except Exception as e:
                print(
                    f"Warning: failed to load weights at {wpath} -> skipping. Error: {repr(e)}"
                )
                continue

            pred_i = model.predict(test_gen, verbose=1)
            pred_i = pred_i.astype(np.float64)
            pred_i = np.clip(pred_i, PROB_FLOOR, 1.0)
            pred_i = pred_i / pred_i.sum(axis=1, keepdims=True)
            preds.append(pred_i)

        if len(preds) == 0:
            raise RuntimeError(
                "No fold predictions produced (all weight loads failed). Cannot create model-based submission."
            )

        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)

        pred = pred.astype(np.float64)
        pred = np.clip(pred, PROB_FLOOR, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        p = np.clip(pred, PROB_FLOOR, 1.0)
        logits = np.log(p + EPS_LOGIT)
        logits = logits / TEMPERATURE
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = np.clip(p, PROB_FLOOR, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        pred = p

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred.astype(np.float32)

        sub = sub[sample.columns]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row prob sum stats:",
            float(sub[TARGETS].sum(axis=1).min()),
            float(sub[TARGETS].sum(axis=1).max()),
        )
        print(sub.head())
