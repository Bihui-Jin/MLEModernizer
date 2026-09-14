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

0.352073344576038

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime failure by automatically falling back to an available model directory (or to a safe no-weights path) instead of hard-crashing when `/kaggle/input/models2024040601` is missing. To keep the core logic intact, the inference loop, model, and generator stay the same; only the weight-loading is made robust and use whichever fold/stage files actually exist. If no weights are found at all, the script still produce a valid `submission.csv` by outputting a uniform probability distribution (so you at least get a “yielded” score and a valid file). I also align the cell numbering to start at 1 to match your required format and add a small sanity check on prediction shape/columns before writing the CSV.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.35207), and the biggest reason is that you’re likely not actually using the intended pretrained weights (the script often falls back to uniform predictions or mismatched/random weights). I make the smallest changes that (1) reliably find the correct weight folder/files by preferring directories that match your `LOAD_MODELS_FROM` name and contain the expected `f*_stage*.h5` set, and (2) ensure inference uses the exact same EfficientNet implementation that the weights were trained with (by preferring `efficientnet.tfkeras` when available and failing loudly if weights exist but cannot be loaded). These changes preserve your model architecture, data pipeline, and prediction semantics; they only improve robustness of weight discovery/loading to move the score down toward the target. The submission writing remains the same, with strict probability normalization.'
- What this solution (achieved 1.40995) has done: 'Your current gap to the target is large (1.40995 vs 0.35207; lower is better), and the most likely cause is that inference isn’t actually using the intended pretrained weights (or is using only a small subset) due to the very narrow weight-file pattern. I make the smallest change that broadens weight discovery within the model directory (still preserving fold+stage semantics), so the script can load the correct `.h5` files even if they’re named slightly differently (e.g., `stage2_f0.h5`, `fold0_stage2.h5`, etc.). I also rebuild the model inside the fold loop to avoid any cross-fold weight/state carryover (same architecture, just safer loading), and I keep the same prediction normalization and submission schema. No training logic, model layers, feature extraction, or loss is changed—only weight discovery/loading robustness to move the score down toward your target.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (1.40995 vs 0.35207; lower is better), and the most likely reason is that inference is not actually loading the intended pretrained weights (or is loading incompatible weights) while still producing “valid-looking” probabilities. I make minimal changes that (1) robustly locate the correct weight files under the *competition dataset* input tree (where Kaggle actually places shared model folders), (2) ensure we only average predictions across folds that successfully load, and (3) make the EfficientNet weight-loading path safer by using `by_name=True, skip_mismatch=True` when needed (without changing architecture), so you get real pretrained inference instead of near-random outputs. I also keep your probability normalization but make it numerically safer (avoid division by 0) to prevent any submission invalidation. No model layers, feature pipeline, loss, or inference loop structure are changed—only weight discovery/loading robustness to move the KL down toward your target.'
- What this solution (achieved 1.40995) has done: 'Your score is far above the target (1.40995 vs 0.35207; lower is better), and the biggest likely cause is that the script is either (a) not finding/loading the intended weights at all, or (b) loading incompatible weights with `skip_mismatch=True` which can silently produce near-random behavior. I make minimal changes to (1) reliably locate weights inside the competition dataset tree and common nested “models” folders, (2) enforce “only use fully-compatible weights” (no skip-mismatch fallback) so we don’t average garbage folds, and (3) only average folds that successfully load and produce finite predictions. This preserves your model architecture, generator, and inference flow; it just makes weight discovery/loading stricter and more reliable, which should move KL down toward the target. The submission formatting and probability normalization remain unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far from the target (0.35207), and the most likely cause is that inference is effectively untrained because no compatible `.h5` weights are being loaded. I make the smallest changes that (1) correctly discover weights that may be stored inside nested subfolders under the model dataset directory, and (2) stop silently mixing in bad folds by verifying that loaded weights actually change the model outputs (quick checksum on one batch). This preserves your model, data generation, and inference averaging, but makes weight usage real and consistent, which should move the KL down toward the target. The submission formatting/normalization stays the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import subprocess, sys


def _pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version
    import google.protobuf as _pb

    if Version(_pb.__version__) >= Version("5.0.0"):
        try:
            _pip_install("protobuf<5")
        except Exception as e:
            print("protobuf downgrade attempt failed; continuing. Error:", repr(e))
except Exception as e:
    print("protobuf check failed; continuing. Error:", repr(e))

EFFNET_WHL = (
    "/kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl"
)
EFFNET_WHL_DIR = "/kaggle/input/tf-efficientnet-whl-files"
if os.path.exists(EFFNET_WHL):
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={EFFNET_WHL_DIR}",
                EFFNET_WHL,
            ]
        )
        print("Installed efficientnet wheel from:", EFFNET_WHL)
    except Exception as e:
        print(
            "Efficientnet wheel install failed; will fallback to tf.keras.applications. Error:",
            repr(e),
        )
else:
    print("Efficientnet wheel not found; will fallback to tf.keras.applications")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 2
print(DATATYPE)
LOAD_MODELS_FROM = "models2024040601"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 256  # 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024

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


## === cell 1
import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # noqa: F401
import librosa

print("TensorFlow version =", tf.__version__)

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

MIX = True
if MIX:
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision policy set to mixed_float16")
    except Exception as e:
        print("Mixed precision not available; continuing. Error:", repr(e))
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


## === cell 2
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


## === cell 3
try:
    import albumentations as albu  # noqa: F401
except Exception as e:
    albu = None
    print("albumentations not available; continuing. Error:", repr(e))

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
            x_eeg = np.zeros(
                (len(indexes), 6, round(20 * SFREQ), 3, 4), dtype="float32"
            )
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

                    if (spe.shape[0] != 100) or (spe.shape[1] != 300):
                        spe2 = np.zeros((100, 300))
                        spe2[: spe.shape[0], : spe.shape[1]] = spe
                        spe = spe2

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
                    eeg = self.eegs[row.eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]

                    if eeg.shape[1] < 50 * SFREQ:
                        eeg = np.concatenate((eeg, eeg), 1)
                        eeg = eeg[:, : 50 * SFREQ]

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

                    if (stft.shape[1] != 64) or (stft.shape[2] != 256):
                        stft2 = np.zeros((4, 64, 128))
                        stft2[:, : stft.shape[1], : stft.shape[2]] = stft
                        stft = stft2

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
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test":
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0

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




## === cell 4
try:
    import efficientnet.tfkeras as efn  # type: ignore

    _USING_EXTERNAL_EFN = True
    print("Using efficientnet.tfkeras EfficientNetB0")
except Exception as e:
    efn = None
    _USING_EXTERNAL_EFN = False
    print(
        "efficientnet.tfkeras not available; using tf.keras.applications.EfficientNetB0. Error:",
        repr(e),
    )


def _make_efficientnet_b0(name: str):
    if _USING_EXTERNAL_EFN:
        return efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name=name
        )
    return tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name=name
    )


def _l2norm_layer(name: str):
    return tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1), name=name)


def build_model(TARGETS_PRETRAIN):
    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])

        base_model_spe = _make_efficientnet_b0(name="efficientnetb0_spe")
        base_model_spe._name = "spe_extractor"
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_spe.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_spe.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = _l2norm_layer("spe_l2norm")(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
        x_eeg1 = inp_eeg[:, :, :, :, 0]
        x_eeg2 = inp_eeg[:, :, :, :, 1]
        x_eeg3 = inp_eeg[:, :, :, :, 2]
        x_eeg4 = inp_eeg[:, :, :, :, 3]

        x_eeg = tf.keras.layers.Concatenate(axis=1)([x_eeg1, x_eeg2, x_eeg3, x_eeg4])

        base_model_eeg = _make_efficientnet_b0(name="efficientnetb0_eeg")
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

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = _l2norm_layer("eeg_l2norm")(x_eeg)

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

        base_model_img = _make_efficientnet_b0(name="efficientnetb0_img")
        base_model_img._name = "img_extractor"
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_img.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_img.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = _l2norm_layer("img_l2norm")(x_img)

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

        base_model_stft = _make_efficientnet_b0(name="efficientnetb0_stft")
        base_model_stft._name = "stft_extractor"
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_stft.load_weights(
                    "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_stft.load_weights(
                    "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                )

        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        x_stft = _l2norm_layer("stft_l2norm")(x_stft)

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




## === cell 5
if not NEEDTRAIN:
    if "spe" in DATATYPE:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
        elif PLATFORM == "kaggle":
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )
        print("Test shape", test.shape)
        test.head()

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
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values

    from scipy import signal
    import re

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"\nThere are {len(files2)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            list_eeg = list()
            list_img = list()
            list_stft = list()
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
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE:
                    mel_spec = librosa.feature.melspectrogram(
                        y=eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        sr=SFREQ,
                        hop_length=round(50 * SFREQ / 256),
                        n_fft=512,
                        n_mels=128,
                        fmin=0,
                        fmax=40,
                        win_length=128,
                    )
                    mel_spec = np.mean(mel_spec, 0)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.min).astype(
                        np.float32
                    )
                    list_stft.append(
                        np.reshape(
                            mel_spec_db, (mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                        )
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

    def _find_weight_files(preferred_dir: str, stage: int):
        def _collect_from_dir_recursive(root: str, max_depth: int = 3):
            if not root or not os.path.isdir(root):
                return []
            stage_pat = re.compile(
                rf"(?:^|[^0-9])stage{stage}(?:[^0-9]|$)", re.IGNORECASE
            )
            fold_pat = re.compile(
                r"(?:^|[^0-9])f([0-4])(?:[^0-9]|$)|(?:^|[^0-9])fold([0-4])(?:[^0-9]|$)",
                re.IGNORECASE,
            )

            hits = []
            root = os.path.abspath(root)
            base_depth = root.rstrip(os.sep).count(os.sep)

            for cur, dirs, files in os.walk(root):
                cur_depth = cur.rstrip(os.sep).count(os.sep) - base_depth
                if cur_depth > max_depth:
                    dirs[:] = []
                    continue
                for fn in files:
                    if not fn.lower().endswith(".h5"):
                        continue
                    if not stage_pat.search(fn):
                        continue
                    m = fold_pat.search(fn)
                    if not m:
                        continue
                    fold_str = m.group(1) if m.group(1) is not None else m.group(2)
                    if fold_str is None:
                        continue
                    fold = int(fold_str)
                    hits.append((fold, os.path.join(cur, fn)))

            best = {}
            for fold, path in sorted(hits, key=lambda x: x[1]):
                best[fold] = path
            return sorted(best.items(), key=lambda x: x[0])

        search_roots = []
        if preferred_dir:
            search_roots.append(preferred_dir)

        comp_root = "/kaggle/input/hms-harmful-brain-activity-classification"
        token = os.path.basename(preferred_dir.rstrip("/")) if preferred_dir else ""
        if token:
            search_roots.extend(
                [
                    os.path.join(comp_root, token),
                    os.path.join(comp_root, token, "models"),
                    os.path.join(comp_root, token, "weights"),
                    os.path.join(comp_root, token, "ckpt"),
                    os.path.join(comp_root, token, "checkpoints"),
                ]
            )

        base = "/kaggle/input"
        if token and os.path.isdir(base):
            for dname in sorted(os.listdir(base)):
                if token not in dname:
                    continue
                cand_dir = os.path.join(base, dname)
                if not os.path.isdir(cand_dir):
                    continue
                search_roots.extend(
                    [
                        cand_dir,
                        os.path.join(cand_dir, "models"),
                        os.path.join(cand_dir, "weights"),
                        os.path.join(cand_dir, "ckpt"),
                        os.path.join(cand_dir, "checkpoints"),
                    ]
                )

        for root in search_roots:
            found = _collect_from_dir_recursive(root, max_depth=4)
            if found:
                return found

        found_all = []
        if os.path.isdir(base):
            for dname in sorted(os.listdir(base)):
                cand_dir = os.path.join(base, dname)
                if not os.path.isdir(cand_dir):
                    continue
                found_all.extend(_collect_from_dir_recursive(cand_dir, max_depth=2))
        best = {}
        for fold, path in found_all:
            if fold not in best:
                best[fold] = path
        return sorted(best.items(), key=lambda x: x[0])

    def _prediction_checksum(pred_arr: np.ndarray) -> float:
        pred_arr = np.asarray(pred_arr, dtype=np.float32)
        pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.mean(np.abs(pred_arr)))

    preds = []
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

    weight_files = _find_weight_files(LOAD_MODELS_FROM, STAGETEST)
    if len(weight_files) == 0:
        print(
            f"\nWARNING: No weights found for stage {STAGETEST}. "
            "Will generate a valid (but likely low-scoring) uniform submission."
        )
        pred = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
    else:
        print("\nUsing weights:")
        for fold, path in weight_files:
            print(f"  fold {fold}: {path}")

        x0, _ = test_gen[0]

        for fold, path in weight_files:
            print(f"Fold {fold+1}")
            model = build_model(TARGETS)

            try:
                p_before = model.predict(x0, verbose=0)
                cs_before = _prediction_checksum(p_before)
            except Exception:
                cs_before = None

            try:
                model.load_weights(path)
            except Exception as e:
                print(
                    f"ERROR: load_weights failed for fold {fold} at {path}. "
                    "Skipping this fold to avoid averaging incompatible weights. Error:",
                    repr(e),
                )
                continue

            try:
                p_after = model.predict(x0, verbose=0)
                cs_after = _prediction_checksum(p_after)
                if cs_before is not None and abs(cs_after - cs_before) < 1e-6:
                    print(
                        f"WARNING: fold {fold} weights appear to not change outputs (checksum {cs_before:.8f}->{cs_after:.8f}). "
                        "Skipping to avoid averaging a likely-bad/no-op load."
                    )
                    continue
            except Exception as e:
                print(
                    f"WARNING: checksum test failed for fold {fold}; continuing with full prediction. Error:",
                    repr(e),
                )

            pred_fold = model.predict(test_gen, verbose=1)
            pred_fold = np.asarray(pred_fold, dtype=np.float32)

            if not np.isfinite(pred_fold).all():
                print(
                    f"ERROR: non-finite predictions for fold {fold}. Skipping this fold."
                )
                continue

            preds.append(pred_fold)

        if len(preds) == 0:
            print(
                "\nWARNING: No valid folds produced predictions. "
                "Will generate a valid uniform submission."
            )
            pred = np.full(
                (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
            )
        else:
            pred = np.mean(preds, axis=0)
            print()
            print("Test preds shape", pred.shape)

            pred = np.asarray(pred, dtype=np.float32)
            pred = np.clip(pred, 1e-8, 1.0)
            pred = pred / np.sum(pred, axis=1, keepdims=True)

    if pred.shape[0] != len(test) or pred.shape[1] != len(TARGETS):
        raise ValueError(
            f"Prediction shape mismatch: got {pred.shape}, expected ({len(test)}, {len(TARGETS)})"
        )

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[list(TARGETS)] = pred

    sub = sub[["eeg_id"] + list(TARGETS)]
    row_sums = sub[list(TARGETS)].sum(axis=1).values.astype(np.float32)
    row_sums = np.maximum(row_sums, 1e-12)
    sub.loc[:, list(TARGETS)] = sub[list(TARGETS)].values / row_sums[:, None]

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-sum check (min/max):",
        sub[list(TARGETS)].sum(axis=1).min(),
        sub[list(TARGETS)].sum(axis=1).max(),
    )
    print("Wrote submission.csv")
