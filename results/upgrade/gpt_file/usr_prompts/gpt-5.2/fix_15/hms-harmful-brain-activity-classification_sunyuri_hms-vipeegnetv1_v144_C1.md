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

0.354723620109739

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError seen in this environment. Then I fix the missing-weights runtime error by auto-discovering available `.h5` weights under `/kaggle/input/` (including nested competition folders) and using them if found; if not found, the script fall back to a valid, well-formed uniform-probability submission so a `.csv` is always produced. I also make the weight-loading stage robust to different fold/stage filename patterns while keeping the model and prediction logic unchanged. Finally, I add strict submission column/order enforcement and probability normalization to ensure Kaggle accepts the file.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow-related import happens*, and also by importing `google.protobuf` early to ensure the env var takes effect. Then I keep the existing inference/core model logic unchanged, but make weight discovery slightly more robust (also searching `/kaggle/input` for any `.h5` if the preferred folder is missing) so the script actually loads trained weights instead of frequently falling back to uniform predictions (which is why the score is currently far from the target). Finally, I keep strict submission column ordering and per-row probability normalization so the submission is always valid for Kaggle’s KL metric.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* TensorFlow import and additionally forcing the legacy protobuf API implementation (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). Then I keep the model/inference logic unchanged but make weight discovery more reliable by searching all `.h5` files under `/kaggle/input` and preferring files that match the requested stage/fold patterns, which should reduce the chance of falling back to uniform predictions (and thus improve KL toward the target). Finally, I ensure the submission is always valid by strict column ordering, clipping, and per-row normalization to sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation and additionally pinning protobuf to the Python backend before *any* TensorFlow/Keras import occurs, which addresses the `MessageFactory.GetPrototype` error. Next, I keep your model/inference logic unchanged but make the weight discovery actually find weights in Kaggle datasets by searching both `/kaggle/input/**` and `/kaggle/input/**/models*.h5` with better scoring, so you don’t silently fall back to uniform predictions (which is the main reason the score is far from the target). Finally, I ensure the submission is always valid (correct columns/order, probabilities clipped and normalized) and always written as `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* disabling C++ protos before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` error in this Kaggle Python 3.12 environment. Then I keep your model and inference flow unchanged, but make weight discovery prefer the competition dataset folder and `.weights.h5` files so it’s much more likely to actually load trained folds instead of silently falling back to uniform predictions (which caused the poor 1.40995 score). Finally, I ensure the submission is created without aborting the notebook (remove `SystemExit`) and strictly match the sample submission column order with per-row normalization so Kaggle accepts it.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate protobuf/TensorFlow crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend earlier and additionally disabling the upb/C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `google.protobuf.internal.api_implementation._SetType("python")` before importing TensorFlow. Then, to move the KL score down toward the target, I prevent the pipeline from silently producing uniform predictions when no weights are found by also searching the official competition dataset folder for any `.h5` weights and preferring likely fold/stage matches. Finally, I keep your model/inference logic unchanged while ensuring the submission is always written as `submission.csv` with correct columns/order and strictly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the protobuf Python implementation is forced *before* any protobuf/TensorFlow import and by guarding the TensorFlow import with a safe fallback that still produces a valid submission if TF cannot load. I also fix a logic bug in `DataGenerator` where `x_eeg2` is allocated but never filled, yet later normalized, which can introduce NaNs and degrade predictions; this is score-impacting but does not change the model architecture or training/inference semantics. Finally, I keep your weight auto-discovery and submission normalization, but make the script always reach `submission.csv` creation even if TensorFlow is unavailable, avoiding runtime termination.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python implementation earlier and more safely, and by importing TensorFlow only after that configuration is guaranteed to be applied. Then I remove the `SystemExit` hard-stop so the notebook always completes and writes `submission.csv` even in fallback paths. Finally, I fix a real logic bug in `DataGenerator` where `x_eeg2` is computed/normalized but never returned to the model (making the normalization pointless and potentially introducing instability); this is score-neutral for architecture (since the model never consumed it) but improves correctness and avoids unintended NaN/scale issues during batch building.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash that currently stops execution by forcing protobuf’s Python implementation in a way that reliably takes effect before TensorFlow loads (and falling back gracefully if TF still cannot import). Then I keep your model/inference logic intact, but correct a score-impacting DataGenerator label bug (`sum(label == 1)` → `np.sum(label == 1)`) that currently prevents intended smoothing and can worsen KL. Finally, I keep your robust weight auto-discovery, but also ensure submission writing always happens and the CSV is strictly aligned to the sample submission columns with per-row normalization.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime earlier and (critically) setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` to avoid the legacy API path that triggers this error under Python 3.12. Then I keep your model/inference flow the same but add a robust fallback around TensorFlow import so the notebook always completes and writes `submission.csv`. Finally, I keep your weight auto-discovery logic intact while ensuring `strategy` is always defined correctly (CPU fallback if no GPU), and I preserve strict probability clipping/normalization so Kaggle accepts the submission.'

# 9. Code solution

## === cell 0
import os, io, glob, re, sys
import numpy as np
import pandas as pd
from PIL import Image

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_LEGACY_API", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    tf = None

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024040101"
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

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

READ_EXTRA_SPEC_FILES = True
READ_EXTRA_EEG_FILES = True
READ_EXTRA_IMG_FILES = True
READ_EXTRA_STFT_FILES = True

spectrograms, eegs, imgs, stfts = {}, {}, {}, {}
spectrograms2, eegs2, imgs2, stfts2 = {}, {}, {}, {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if TF_AVAILABLE:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        tf.random.set_seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print("TensorFlow version =", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("Using 1 GPU")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("Using CPU")

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision could not be enabled:", repr(e))
    else:
        print("Using full precision")
else:
    print("WARNING: TensorFlow import failed; will fall back to uniform submission.")
    print("TensorFlow import error:", repr(TF_IMPORT_ERROR))
    strategy = None

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
            self.specs = specs if specs is not None else {}
            self.eegs = eegs if eegs is not None else {}
            self.imgs = imgs if imgs is not None else {}
            self.stfts = stfts if stfts is not None else {}
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
                x_eeg2 = np.zeros(
                    (len(indexes), 4, round(50 * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros(
                    (len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32"
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")

            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    r_spe = (
                        round(row.spectrogram_label_offset_seconds / 2)
                        if "spectrogram_label_offset_seconds" in row
                        else 0
                    )
                    r_eeg = (
                        round(row.eeg_label_offset_seconds * SFREQ)
                        if "eeg_label_offset_seconds" in row
                        else 0
                    )

                if self.mode == "train":
                    x1 = np.random.rand() * LENGTH / 2
                    x2 = np.random.rand() * LENGTH / 2
                    if np.random.rand() < 0.5:
                        x1 = x1 + LENGTH / 2
                        x2 = x2 + LENGTH / 2
                    x_spe_min = round(min(x1, x2))
                    x_spe_max = round(max(x1, x2))

                for k in range(4):
                    if "spe" in DATATYPE:
                        if row.spectrogram_id in self.specs:
                            spe = self.specs[row.spectrogram_id][
                                r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                            ].T
                        else:
                            spe = np.zeros((100, 300), dtype=np.float32)

                        if (spe.shape[0] != 100) or (spe.shape[1] != 300):
                            spe2 = np.zeros((100, 300))
                            spe2[: spe.shape[0], : spe.shape[1]] = spe
                            spe = spe2

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
                            round((spe.shape[1] - LENGTH) / 2) : -round(
                                (spe.shape[1] - LENGTH) / 2
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
                        if row.eeg_id in self.eegs:
                            eeg = self.eegs[row.eeg_id][
                                :, r_eeg : r_eeg + round(50 * SFREQ), k
                            ]
                        else:
                            eeg = np.zeros((4, int(50 * SFREQ)), dtype=np.float32)

                        if eeg.shape[1] < 50 * SFREQ:
                            eeg = np.concatenate((eeg, eeg), 1)
                            eeg = eeg[:, : int(50 * SFREQ)]

                        x_eeg2[j, :, :, k] = eeg

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
                        if row.eeg_id in self.imgs:
                            img = self.imgs[row.eeg_id][:, :, k, :]
                            x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        if row.eeg_id in self.stfts:
                            stft = self.stfts[row.eeg_id][:, :, :, k]
                        else:
                            stft = np.zeros((4, 64, 128), dtype=np.float32)

                        stft = np.concatenate(
                            [
                                stft[0, :, :],
                                stft[1, :, :],
                                stft[2, :, :],
                                stft[3, :, :],
                            ],
                            1,
                        )
                        stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                        stft = np.log(stft)
                        stft = np.nan_to_num(stft, nan=0.0)

                        stft = np.round(
                            (stft - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)
                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (64, 128 * 4, 3))

                        x_stft[j, :, :, :, k] = stft
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
                    label = row[self.targets].values
                    if self.mode == "train" and (np.sum(label == 1) > 0):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label == 1] = 1 - 5 * xx
                    y[j] = label

            if "eeg" in DATATYPE:
                x_eeg2 = np.nan_to_num(x_eeg2, nan=0.0, posinf=0.0, neginf=0.0)

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




## === cell 2
if TF_AVAILABLE:

    def build_model(TARGETS_PRETRAIN):
        inp = list()
        y = None

        l2n = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_tensor=None,
                name="efficientnetb0_spe",
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = l2n(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg1 = inp_eeg[:, :, :, :, 0]
            x_eeg2 = inp_eeg[:, :, :, :, 1]
            x_eeg3 = inp_eeg[:, :, :, :, 2]
            x_eeg4 = inp_eeg[:, :, :, :, 3]
            x_eeg = tf.keras.layers.Concatenate(axis=1)(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_tensor=None,
                name="efficientnetb0_eeg",
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2n(x_eeg)

            inp.append(inp_eeg)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            else:
                y = x_eeg

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img1 = inp_img[:, :, :, :, 0]
            x_img2 = inp_img[:, :, :, :, 1]
            x_img3 = inp_img[:, :, :, :, 2]
            x_img4 = inp_img[:, :, :, :, 3]
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [x_img1, x_img2, x_img3, x_img4]
            )

            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_tensor=None,
                name="efficientnetb0_img",
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = l2n(x_img)

            inp.append(inp_img)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
            else:
                y = x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4))
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_tensor=None,
                name="efficientnetb0_stft",
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = l2n(x_stft)

            inp.append(inp_stft)
            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            else:
                y = x_stft

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




## === cell 3
def _discover_weights_dir(preferred_dir: str) -> str | None:
    if preferred_dir and os.path.isdir(preferred_dir):
        return preferred_dir

    base = os.path.basename(preferred_dir) if preferred_dir else ""
    candidates = []

    if base:
        for p in glob.glob(f"/kaggle/input/**/{base}", recursive=True):
            if os.path.isdir(p):
                candidates.append(p.rstrip("/"))

    for p in glob.glob(
        "/kaggle/input/hms-harmful-brain-activity-classification/**", recursive=True
    ):
        if os.path.isdir(p) and (
            glob.glob(os.path.join(p, "*.h5"))
            or glob.glob(os.path.join(p, "**", "*.h5"), recursive=True)
        ):
            candidates.append(p.rstrip("/"))

    for p in glob.glob("/kaggle/input/**", recursive=True):
        if os.path.isdir(p) and (
            glob.glob(os.path.join(p, "*.h5"))
            or glob.glob(os.path.join(p, "**", "*.h5"), recursive=True)
        ):
            candidates.append(p.rstrip("/"))

    for p in glob.glob("/kaggle/input/**/*.h5", recursive=True):
        candidates.append(os.path.dirname(p).rstrip("/"))

    candidates = sorted(set(candidates))
    return candidates[0] if candidates else None


def _parse_fold_stage_from_name(path: str) -> tuple[int | None, int | None]:
    name = os.path.basename(path).lower()
    fold = None
    stage = None

    m = re.search(r"(?:^|[^a-z0-9])f(?:old)?\s*([0-9]+)", name)
    if m:
        try:
            fold = int(m.group(1))
        except Exception:
            pass

    m = re.search(r"stage\s*([0-9]+)", name)
    if m:
        try:
            stage = int(m.group(1))
        except Exception:
            pass

    return fold, stage


def _all_h5_files(weights_dir: str) -> list[str]:
    hits = []
    hits.extend(glob.glob(os.path.join(weights_dir, "*.h5")))
    hits.extend(glob.glob(os.path.join(weights_dir, "**", "*.h5"), recursive=True))
    return sorted(set([h for h in hits if os.path.isfile(h)]))


def _find_weight_file(weights_dir: str, fold: int, stage: int) -> str | None:
    patterns = [
        os.path.join(weights_dir, f"f{fold}_stage{stage}.weights.h5"),
        os.path.join(weights_dir, f"fold{fold}_stage{stage}.weights.h5"),
        os.path.join(weights_dir, f"stage{stage}_f{fold}.weights.h5"),
        os.path.join(weights_dir, f"stage{stage}_fold{fold}.weights.h5"),
        os.path.join(weights_dir, f"f{fold}_stage{stage}.h5"),
        os.path.join(weights_dir, f"fold{fold}_stage{stage}.h5"),
        os.path.join(weights_dir, f"stage{stage}_f{fold}.h5"),
        os.path.join(weights_dir, f"stage{stage}_fold{fold}.h5"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p

    globs_ = [
        os.path.join(weights_dir, f"**/*f{fold}*stage{stage}*.weights.h5"),
        os.path.join(weights_dir, f"**/*fold{fold}*stage{stage}*.weights.h5"),
        os.path.join(weights_dir, f"**/*stage{stage}*f{fold}*.weights.h5"),
        os.path.join(weights_dir, f"**/*stage{stage}*fold{fold}*.weights.h5"),
        os.path.join(weights_dir, f"**/*f{fold}*stage{stage}*.h5"),
        os.path.join(weights_dir, f"**/*fold{fold}*stage{stage}*.h5"),
        os.path.join(weights_dir, f"**/*stage{stage}*f{fold}*.h5"),
        os.path.join(weights_dir, f"**/*stage{stage}*fold{fold}*.h5"),
    ]
    for g in globs_:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]

    all_hits = _all_h5_files(weights_dir)
    if not all_hits:
        return None

    scored = []
    for p in all_hits:
        f, s = _parse_fold_stage_from_name(p)
        score = 0
        if s is not None and s == stage:
            score += 50
        if f is not None and f == fold:
            score += 25
        bn = os.path.basename(p).lower()
        if bn.endswith(".weights.h5"):
            score += 10
        if "best" in bn:
            score += 5
        if "final" in bn:
            score += 2
        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, best_path = scored[0]
    if best_score <= 0:
        return None
    return best_path




## === cell 4
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

    sub_cols = list(sample_sub.columns)
    target_cols = [c for c in sub_cols if c != "eeg_id"]

    if not TF_AVAILABLE:
        sub = sample_sub.copy()
        sub[target_cols] = 1.0 / len(target_cols)
        sub.to_csv("submission.csv", index=False)
        print(
            "Submission written (TF unavailable): submission.csv", "shape:", sub.shape
        )
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
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

        from scipy import signal

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        imgs2 = {}
        stfts2 = {}
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        test_eeg_ids = set(test.eeg_id.astype(int).tolist())

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])

            if name in test_eeg_ids:
                list_eeg = []
                list_img = []
                list_stft = []

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
                            "STFT path requires cupy and is disabled in this run."
                        )

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
                    byte_stream.truncate(0)
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
                    byte_stream.truncate(0)
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
                    byte_stream.truncate(0)
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

        weights_dir = _discover_weights_dir(LOAD_MODELS_FROM)
        if weights_dir is None:
            print(
                "WARNING: No weights directory found under /kaggle/input. Will write uniform submission."
            )
            sub = sample_sub.copy()
            sub[target_cols] = 1.0 / len(target_cols)
            sub.to_csv("submission.csv", index=False)
            print("Submission written:", "submission.csv", "shape:", sub.shape)
        else:
            print("Using weights directory:", weights_dir)

            preds = []
            with strategy.scope():
                model = build_model(TARGETS)

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

            loaded_any = False
            for i in range(NSPLIT):
                wfile = _find_weight_file(weights_dir, i, STAGETEST)
                if wfile is None:
                    print(f"Fold {i+1}: WARNING no weight file found; skipping fold.")
                    continue
                print(f"Fold {i+1}: loading {wfile}")
                model.load_weights(wfile)
                pred = model.predict(test_gen, verbose=1)
                preds.append(pred)
                loaded_any = True

            if not loaded_any:
                print("WARNING: No fold weights loaded. Will write uniform submission.")
                sub = sample_sub.copy()
                sub[target_cols] = 1.0 / len(target_cols)
                sub.to_csv("submission.csv", index=False)
                print("Submission written:", "submission.csv", "shape:", sub.shape)
            else:
                pred = np.mean(preds, axis=0)
                print("\nTest preds shape", pred.shape)

                pred = np.clip(pred, 1e-7, 1.0)
                pred = pred / pred.sum(axis=1, keepdims=True)

                sub = sample_sub.copy()
                sub[target_cols] = pred[: len(sub), :]

                sub[target_cols] = np.clip(sub[target_cols].values, 1e-7, 1.0)
                sub[target_cols] = sub[target_cols].values / sub[
                    target_cols
                ].values.sum(axis=1, keepdims=True)

                sub.to_csv("submission.csv", index=False)
                print("Submission written: submission.csv")
                print("Submission shape", sub.shape)
                print(sub.head())
                print(
                    "Row sums min/max:",
                    sub[target_cols].sum(axis=1).min(),
                    sub[target_cols].sum(axis=1).max(),
                )
