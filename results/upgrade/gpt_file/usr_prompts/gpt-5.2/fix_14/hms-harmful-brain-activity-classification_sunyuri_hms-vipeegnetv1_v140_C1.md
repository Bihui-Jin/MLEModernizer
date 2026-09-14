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

0.4466148402231433

# 6. Current score

1.39684

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the submission merge explosion by ensuring we generate exactly one prediction per `eeg_id` (the test set has unique `eeg_id`s, so we can safely deduplicate and align to `sample_submission`). Finally, I keep the model/inference logic unchanged but make the final submission strictly match `sample_submission` ordering and shape, with normalized probabilities that sum to 1, so a valid `submission.csv` is always written.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf implementation even earlier (before any TensorFlow-related import side effects) and by clearing any preloaded `google.protobuf` modules if present. Then I remove an unnecessary dependency on `efficientnet.tfkeras` (which can indirectly trigger protobuf/TensorFlow import issues) by always using `tf.keras.applications.EfficientNetB0` while keeping the exact same model wiring (include_top=False, weights=None). Finally, I keep the rest of the inference/submission logic unchanged so it still produces one prediction per `eeg_id`, aligned to `sample_submission`, with strictly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment configuration and module cleanup to the very top, and by also setting `TF_USE_LEGACY_KERAS=1` (a common Kaggle+TF workaround) before importing TensorFlow. Then I fix a major inference bug that is likely causing the poor score: `DataGenerator` currently uses `row.eeg_id` in test mode but the test dataframe doesn’t contain `eeg_id` after merge as an attribute-safe field in all contexts; I ensure we always access via `row["eeg_id"]` / `row["spectrogram_id"]` to avoid silent misreads and wrong lookups. Finally, I keep the model and averaging logic identical, but make sure that if no weights are found we still emit a valid, normalized submission aligned exactly to `sample_submission`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf version *before* importing TensorFlow (and cleaning any preloaded protobuf modules), which is the root cause of the `MessageFactory.GetPrototype` error in this Kaggle Python 3.12 image. Then I remove the dependency on `librosa` (often not available by default) by switching the optional STFT branch to use `tf.signal.stft` only when `DATATYPE` includes `"stft"`; with your current `DATATYPE=["eeg"]` this is score-neutral but prevents import/runtime failures. Finally, I keep your model/inference logic intact, but ensure robust test EEG loading (skip missing ids safely) and always write a submission aligned exactly to `sample_submission`, with strictly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4466), and the biggest likely cause is a label/prediction semantic mismatch: the model outputs 6 classes, but your custom `my_loss` collapses class index 5 into “other”, while your submission currently uses the raw 6-class softmax directly. I keep the architecture and inference loop intact, but apply the exact same “collapse to other” post-processing to predictions before writing the CSV, ensuring evaluation semantics match training. I also load weights into a freshly built model per weight file to avoid any subtle state carryover across heads (minimal risk, often more stable), and keep the existing normalization/alignment guarantees unchanged. These are small, directly score-relevant fixes that should reduce KL divergence substantially without changing the core approach.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is much worse than the target (0.4466), and the biggest score-relevant issue still present is a train/infer semantic mismatch: `my_loss` collapses class-5 into “other”, but the model is currently built with 6 outputs and weights were trained under that loss, making inference more stable if we mirror the same collapse *inside the graph* before the softmax output is exposed. I keep your architecture and loops the same, but add a tiny “collapse+renormalize” post-softmax layer to the model used for inference so predictions match the loss semantics exactly (not just in numpy after predict). I also compile with the custom loss before loading weights (no training, but ensures the graph/loss is consistent and avoids subtle Keras legacy quirks), and keep the final probability clipping/normalization and sample_submission alignment unchanged.'
- What this solution (achieved 1.40717) has done: 'We make two score-relevant, minimal adjustments without changing your architecture or inference loop: (1) remove the extra “collapse_other” layer during inference because your model already outputs exactly the 6 competition classes and collapsing class-5 into itself can distort probabilities and worsen KL, and (2) apply a small, fixed blending with the empirical class prior from `train.csv` (a standard KL stabilizer) to reduce overconfident/peaky predictions that typically inflate KL on this competition. Both changes preserve evaluation semantics (still outputs 6 probabilities summing to 1 for the required columns) and keep everything else (data loading, generator, weight ensembling, submission alignment) the same. This should move your KL down from 1.41 toward the 0.45 target without risky refactors.'
- What this solution (achieved 1.40244) has done: 'Your current KL (1.407, lower-is-better) is far from the target (0.4466), so we should improve score with minimal, low-risk changes that don’t touch the model/feature core. The biggest score-relevant issue still present is that inference outputs can be overconfident; for KL this is punished heavily, and your fixed prior-mix (0.08) is likely too weak for these weights/features, so we increase smoothing strength in a controlled way. We also make the smoothing more robust by mixing toward a per-row prior (based on the training global prior but with a tiny uniform component) and by applying a very small temperature (>1) to soften predictions before mixing—both are purely post-processing and preserve evaluation semantics. Finally, we keep the same CSV alignment and normalization guarantees so it always produces a valid submission.'
- What this solution (achieved 1.39684) has done: 'Your current KL (1.402, lower-is-better) is still far from the target (0.4466), so we should move it down with minimal, low-risk changes that don’t touch the model or feature pipeline. The biggest score lever you’re using is post-processing; right now the prior-mix (0.25) and temperature (1.6) may still leave predictions too peaky for KL, so we increase smoothing moderately and make it adaptive to ensemble size (stronger smoothing when fewer/no weights load). We also apply a small probability floor before normalization (slightly higher than 1e-12) to reduce KL blow-ups on near-zero classes. These changes preserve evaluation semantics (valid probabilities summing to 1) and keep everything else intact.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

try:
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            sys.modules.pop(m, None)
except Exception:
    pass

import io
import warnings
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
)  # kept to preserve original imports/semantics
from tensorflow.keras.applications import EfficientNetB0

warnings.filterwarnings("ignore")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024032701"
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
print("TensorFlow version =", tf.__version__)
print("EfficientNet source = tf.keras.applications.EfficientNetB0")

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

MIX = False
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
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
df.head()



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

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

        num_all = 0
        for i in range(len(TARGETS)):
            num_all = max(num_all, sum(np.argmax(df3[TARGETS].values, 1) == i))

        train = pd.DataFrame()
        train = pd.concat([train, df3]).reset_index(drop=True)
        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.head()
        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 2
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
        self.targets = targets if targets is not None else []
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
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

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

        y = np.zeros((len(indexes), max(len(self.targets), 1)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            eeg_id = int(row["eeg_id"]) if "eeg_id" in row.index else None
            spectrogram_id = (
                int(row["spectrogram_id"]) if "spectrogram_id" in row.index else None
            )

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            elif self.mode == "valid":
                r_spe = round(float(row["spectrogram_label_offset_seconds"]) / 2)
                r_eeg = round(float(row["eeg_label_offset_seconds"]) * SFREQ)
            else:
                r_spe = round(float(row["spectrogram_label_offset_seconds"]) / 2)
                r_eeg = round(float(row["eeg_label_offset_seconds"]) * SFREQ)

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
                    spe = self.specs[spectrogram_id][
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
                    x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (0.225**2)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]
                    eeg = eeg[:, round(0 * SFREQ) : round(50 * SFREQ)]
                    x_eeg[j, :, :, k] = eeg

                if "img" in DATATYPE:
                    if self.mode == "test":
                        img = self.imgs[eeg_id][:, :, k, :]
                    else:
                        img = self.imgs[int(row["sign_id"])][:, :, k, :]
                    x_img[j, :, :, :, k] = img

                if "stft" in DATATYPE:
                    if self.mode == "test":
                        stft = self.stfts[eeg_id][:, :, k]
                    else:
                        stft = self.stfts[int(row["sign_id"])][:, :, k]

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
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test" and len(self.targets) > 0:
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j, : len(self.targets)] = label

        alpha = 0

        if "eeg" in DATATYPE:
            for i_eeg in range(x_eeg.shape[0]):
                xx = np.std(x_eeg[i_eeg, :, :, :], 1, keepdims=True)
                xx = np.mean(xx)
                x_eeg[i_eeg, :, :, :] = (
                    x_eeg[i_eeg, :, :, :]
                    - np.mean(x_eeg[i_eeg, :, :, :], 1, keepdims=True)
                ) / (xx + 1e-6)

        x = list()
        if "spe" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xx) + x_spe[::-1, :, :, :, :] * xx
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xx) + x_eeg[::-1, :, :, :] * xx
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)

        if "stft" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_stft.shape[0], 1, 1, 1, 1))
                x_stft = x_stft * (1 - xx) + x_stft[::-1, :, :, :, :] * xx
            x.append(x_stft)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 3
def _make_efficientnet_b0():
    return EfficientNetB0(include_top=False, weights=None, input_shape=None)


def build_model(TARGETS_PRETRAIN):
    inp = list()

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])

        base_model_spe = _make_efficientnet_b0()
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(4, round(50 * SFREQ), 4))

        kernal_num = 64
        x_eeg1 = inp_eeg[:, :, :, 0:1]
        x_eeg2 = inp_eeg[:, :, :, 1:2]
        x_eeg3 = inp_eeg[:, :, :, 2:3]
        x_eeg4 = inp_eeg[:, :, :, 3:4]
        x_eeg = tf.keras.layers.Concatenate(axis=1)([x_eeg1, x_eeg2, x_eeg3, x_eeg4])

        x_eeg = tf.keras.layers.Conv2D(kernal_num * 1, (1, 8), padding="valid")(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(kernal_num * 1, (1, 6), padding="valid")(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(kernal_num * 1, (1, 4), padding="valid")(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 2, (4, 4), padding="valid", strides=(4, 1)
        )(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(kernal_num * 4, (1, 8), padding="valid")(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(kernal_num * 4, (1, 4), padding="valid")(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(
            kernal_num * 8, (4, 4), padding="valid", strides=(4, 1)
        )(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.Conv2D(kernal_num * 16, (1, 4), padding="valid")(x_eeg)
        x_eeg = tf.keras.layers.ReLU()(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.AveragePooling2D((1, 2))(x_eeg)

        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)

        inp.append(inp_eeg)
        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
        x_img1 = inp_img[:, :, :, :, 0]
        x_img2 = inp_img[:, :, :, :, 1]
        x_img3 = inp_img[:, :, :, :, 2]
        x_img4 = inp_img[:, :, :, :, 3]
        x_img = tf.keras.layers.Concatenate(axis=1)([x_img1, x_img2, x_img3, x_img4])

        base_model_img = _make_efficientnet_b0()
        base_model_img._name = "img_extractor"

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_img)

        inp.append(inp_img)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
        else:
            y = x_img

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_stft1 = inp_stft[:, :, :, :, 0]
        x_stft2 = inp_stft[:, :, :, :, 1]
        x_stft3 = inp_stft[:, :, :, :, 2]
        x_stft4 = inp_stft[:, :, :, :, 3]
        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )

        base_model_stft = _make_efficientnet_b0()
        base_model_stft._name = "stft_extractor"

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(
            x_stft
        )

        inp.append(inp_stft)
        if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
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




## === cell 4
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    test = test.copy()
    sample_sub = sample_sub.copy()
    test["eeg_id"] = test["eeg_id"].astype(int)
    sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(int)

    test = test.drop_duplicates(subset=["eeg_id"]).reset_index(drop=True)
    test = sample_sub[["eeg_id"]].merge(test, on="eeg_id", how="left")
    if test["spectrogram_id"].isna().any():
        raise RuntimeError("sample_submission eeg_id not fully present in test.csv")

    print("Test shape (aligned to sample_submission)", test.shape)
    print("Sample submission shape", sample_sub.shape)

    if "spe" in DATATYPE:
        if PLATFORM == "local":
            PATH2 = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        else:
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
    else:
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    test_eeg_ids = set(test["eeg_id"].astype(int).tolist())

    def _compute_stft_mel_like(eeg_2d: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(eeg_2d, dtype=tf.float32)  # (ch, t)
        x = tf.transpose(x, [1, 0])  # (t, ch)
        stft = tf.signal.stft(
            x,
            frame_length=128,
            frame_step=max(1, round(50 * SFREQ / 256)),
            fft_length=512,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )  # (frames, freq, ch)
        mag = tf.abs(stft)
        mag = tf.reduce_mean(mag, axis=-1)  # (frames, freq)
        mag = tf.math.log(mag + 1e-6)
        mag = mag - tf.reduce_min(mag)
        mag = mag / (tf.reduce_max(mag) + 1e-6)
        mag = tf.cast(mag, tf.float32)
        mag = mag[..., tf.newaxis]  # (frames, freq, 1)
        return mag.numpy()

    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        if name not in test_eeg_ids:
            continue

        eeg_default = pd.read_parquet(f"{PATH2}{f}")

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

            if "stft" in DATATYPE:
                st = _compute_stft_mel_like(
                    eeg[:, round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ)]
                )
                list_stft.append(st)

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
            imgs2[name] = img

    print()

    if "eeg" in DATATYPE:
        missing = [
            eid for eid in test["eeg_id"].astype(int).tolist() if eid not in eegs2
        ]
        if len(missing) > 0:
            print(
                f"WARNING: {len(missing)} eeg_ids missing in eegs2. Backfilling zeros."
            )
            for eid in missing:
                eegs2[eid] = np.zeros((4, 50 * SFREQ, 4), dtype=np.float32)

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

    def _resolve_weight_path(base_dir: str, fname_no_ext: str) -> str:
        p1 = os.path.join(base_dir, f"{fname_no_ext}.h5")
        p2 = os.path.join(base_dir, f"{fname_no_ext}.weights.h5")
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return p1

    preds = []
    loaded_paths = []

    for i in range(NSPLIT):
        for j in range(5):
            wpath = _resolve_weight_path(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}_{j}")
            print(f"Fold {i+1}, stage {STAGETEST}, head {j} -> {wpath}")
            if not os.path.exists(wpath):
                print("  WARNING: weight file not found, skipping.")
                continue

            model = build_model(TARGETS)
            model.compile(optimizer="adam", loss=my_loss)
            model.load_weights(wpath)

            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)
            loaded_paths.append(wpath)

    if len(preds) > 0:
        pred = np.mean(preds, axis=0)
        print("\nTest preds shape", pred.shape)
        print(f"Ensembled {len(preds)} weight files.")
    else:
        print("\nWARNING: No weights loaded. Writing uniform predictions.")
        pred = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float64)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[list(TARGETS)] = np.asarray(pred, dtype=np.float64)

    prior = df[list(TARGETS)].values.astype(np.float64)
    prior = prior / (prior.sum(axis=1, keepdims=True) + 1e-12)
    prior = prior.mean(axis=0)  # shape (6,)
    prior = prior / (prior.sum() + 1e-12)

    prior = 0.98 * prior + 0.02 * (np.ones_like(prior) / len(prior))
    prior = prior / (prior.sum() + 1e-12)

    p = sub[list(TARGETS)].values.astype(np.float64)
    p = np.clip(p, 1e-6, 1.0)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)

    temp = 2.2
    p = np.power(np.clip(p, 1e-12, 1.0), 1.0 / temp)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)

    if len(preds) == 0:
        mix_eps = 0.60
    elif len(preds) <= 3:
        mix_eps = 0.45
    else:
        mix_eps = 0.35

    p = (1.0 - mix_eps) * p + mix_eps * prior.reshape(1, -1)

    p = np.clip(p, 1e-6, 1.0)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    sub[list(TARGETS)] = p

    sub = sub[["eeg_id"] + list(TARGETS)]

    if len(sub) != len(sample_sub):
        raise RuntimeError(
            f"Submission length mismatch: {len(sub)} vs {len(sample_sub)}"
        )
    if not np.allclose(sub[list(TARGETS)].sum(axis=1).values, 1.0, atol=1e-5):
        raise RuntimeError("Submission probabilities do not sum to 1.")

    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row sum stats:",
        float(sub[list(TARGETS)].sum(1).min()),
        float(sub[list(TARGETS)].sum(1).max()),
    )
    sub.head()
