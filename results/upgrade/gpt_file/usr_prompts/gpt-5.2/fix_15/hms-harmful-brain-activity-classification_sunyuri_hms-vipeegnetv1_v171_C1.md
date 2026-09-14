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

0.3261303143352472

# 6. Current score

1.48454

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the early import/runtime crash by disabling the protobuf C-implementation (this avoids the `MessageFactory.GetPrototype` error in some Kaggle TF environments). Next, I make the model-weight loading robust: if the expected `/kaggle/input/models2024040601/*.h5` files are not present, the code fall back to producing a valid, properly-normalized submission using the competition’s `sample_submission.csv` prior (so a `.csv` is always generated). Finally, I keep the original architecture and inference loop intact when weights exist, but add path discovery and clearer diagnostics so the notebook runs end-to-end reliably.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variable setup to the very top (before any TensorFlow/protobuf-related imports) and forcing pure-Python protobuf consistently. Then I ensure the weight-loading discovery works reliably in Kaggle’s `/kaggle/input` layout and that the script always produces a properly-normalized `submission.csv`. Finally, to improve the score toward your target (lower is better), I keep the same model/inference flow but add a minimal, metric-aligned calibration fallback: when weights are missing (or partially missing), use a smoothed prior computed from train vote distributions rather than the flat sample submission prior.'
- What this solution (achieved 1.39779) has done: 'I fix the early crash caused by an incompatible protobuf runtime by setting the protobuf implementation env vars *before any other imports* and also forcing the pure-Python protobuf module to be imported early (this avoids the `MessageFactory.GetPrototype` issue). I keep your model/inference logic unchanged, but make GPU strategy selection robust when no GPU is present to prevent another runtime failure. Finally, I keep the existing “smoothed train prior” fallback (which is score-improving vs uniform when weights are missing) and ensure the produced `submission.csv` is always valid with rows summing to 1.'
- What this solution (achieved 1.39779) has done: 'I fix the protobuf/TensorFlow crash by forcing pure-Python protobuf before *any* protobuf/TensorFlow-related import and by pinning a safe python implementation version (this directly addresses the `MessageFactory.GetPrototype` AttributeError). Then I keep your model/inference logic intact but make the TF import more robust by disabling TF’s use of the C++ protobuf backend via environment variables and ensuring they’re set early enough. Finally, I keep the existing “smoothed train prior” fallback (since your current score is far from the lower-is-better target) and ensure the submission is always normalized and written as `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I fix the early TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *and* removing any already-imported `google.protobuf` modules before importing TensorFlow, which is the reliable way to prevent this incompatibility in Kaggle’s Python 3.12 environment. I keep your model and inference logic unchanged, but add a safe fallback if SciPy is unavailable (so the pipeline still runs end-to-end and writes `submission.csv`). Finally, I ensure the produced submission is always valid (probabilities clipped and row-normalized), and keep the existing train-prior fallback/blending since your current score is far from the lower-is-better target.'
- What this solution (achieved 1.39766) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf backend before any TensorFlow import, and additionally disabling the C++ protobuf implementation in a way that reliably works on Kaggle’s Python 3.12 images. Then I keep your model/data/inference logic unchanged, but add a safe, metric-aligned “prior + tiny uniform” smoothing at the very end of prediction post-processing to avoid overconfident near-zero probabilities that can blow up KL divergence. Finally, I make sure the script always produces a valid `submission.csv` with the required columns and row-normalized probabilities even when no weight files are found.'
- What this solution (achieved 1.39766) has done: 'I fix the early TensorFlow/protobuf crash by setting the protobuf environment variables even earlier and ensuring any already-imported protobuf modules are purged before TensorFlow is imported. I keep your model, generator, and inference logic unchanged, but add a safe fallback to avoid importing TensorFlow at all if it still fails (so the notebook always finishes and writes a valid `submission.csv`). Because your current score is far from the lower-is-better target, I also make the fallback prediction slightly more metric-aligned by using the smoothed train prior (already computed) and ensuring final predictions are clipped and normalized to prevent KL blow-ups. All paths and submission column names remain exactly as required.'
- What this solution (achieved 1.48454) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf backend *before any other imports* and also sanitizing any preloaded protobuf modules, which is the root cause of the current runtime failure. Then I keep your model/inference logic intact, but add a safe “TF import failed → prior-only submission” path that still produces a valid, normalized `submission.csv`. Finally, to move the KL score down toward your target (lower is better) when the model can’t run or weights are missing, I upgrade the fallback from a simple mean prior to an `eeg_id`-aggregated train prior (grouped by `eeg_id`, then averaged), which is metric-aligned yet preserves the intended semantics.'
- What this solution (achieved 1.48454) has done: 'I make the pipeline always run end-to-end by (1) preventing the protobuf/TensorFlow `MessageFactory.GetPrototype` crash from aborting execution, and (2) fixing the EfficientNet input shape issue that currently throws a negative-dimension error when applying the network to the EEG branch. The core model structure (EfficientNetB0 backbones + pooling + l2norm + softmax head, same generator/inference flow) is preserved; the fix is a minimal shape correction (proper NHWC layout + correct concat axis) so EfficientNet receives a valid image-like tensor. If TensorFlow (or weights) still can’t be used in the environment, the script fall back to a properly normalized, train-prior-based submission so a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_PROTOBUF_CPP_IMPLEMENTATION", "python")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf") or k.startswith("protobuf"):
        del sys.modules[k]

import io
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TF_AVAILABLE = True
_TF_IMPORT_ERR = None
try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0

    print("TensorFlow version =", tf.__version__)
except Exception as e:
    _TF_AVAILABLE = False
    _TF_IMPORT_ERR = e
    print("WARNING: TensorFlow import failed; will use prior-only submission fallback.")
    print("TF import error:", repr(e))

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024040601"
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

strategy = None
if _TF_AVAILABLE:
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("Using CPU")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("Using 1 GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision not enabled:", repr(e))
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

_vote = df[TARGETS].astype(np.float32).values
_vote = np.clip(_vote, 0.0, None)
_vote_sum = _vote.sum(axis=1, keepdims=True)
_vote_sum[_vote_sum == 0] = 1.0
_prob_row = _vote / _vote_sum

df_prob = pd.DataFrame(_prob_row, columns=TARGETS)
df_prob["eeg_id"] = df["eeg_id"].values
_prob_by_eeg = df_prob.groupby("eeg_id", sort=False)[list(TARGETS)].mean()
train_prior = _prob_by_eeg.mean(axis=0).values.astype(np.float32)

train_prior = np.clip(train_prior, 1e-8, 1.0)
train_prior = train_prior / train_prior.sum()
print("Train prior (eeg_id-aggregated):", dict(zip(TARGETS, train_prior.round(6))))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import matplotlib  # already imported; kept for colormap use

if not _TF_AVAILABLE:

    class _DummySequence:  # minimal placeholder
        pass

    class _DummyKerasUtils:
        Sequence = _DummySequence

    class _DummyKeras:
        utils = _DummyKerasUtils()

    class _DummyTF:
        keras = _DummyKeras()

    tf = _DummyTF()

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
                        eeg = eeg[:, : round(50 * SFREQ)]

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
                    img = self.imgs[row.eeg_id][:, :, k, :]
                    x_img[j, :, :, :, k] = img

                if "stft" in DATATYPE:
                    raise RuntimeError(
                        "stft mode not supported in this minimal inference run"
                    )

            if self.mode != "test":
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0
        x = []
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
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)

        if "stft" in DATATYPE:
            x.append(x_stft)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 2
def build_model(TARGETS_PRETRAIN):
    if not _TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow not available: {_TF_IMPORT_ERR!r}")

    inp = []
    y = None

    l2n = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe = tf.keras.layers.Concatenate(axis=-1, name="cat_spe")(
            [inp_spe[:, :, :, :, i] for i in range(4)]
        )
        base_model_spe = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_spe._name = "spe_extractor"
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
        x_spe = l2n(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4), name="inp_eeg")
        x_eeg = tf.keras.layers.Concatenate(axis=-1, name="cat_eeg")(
            [inp_eeg[:, :, :, :, i] for i in range(4)]
        )
        base_model_eeg = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_eeg._name = "eeg_extractor"
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = l2n(x_eeg)
        inp.append(inp_eeg)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_feat_spe_eeg")([y, x_eeg])
            if y is not None
            else x_eeg
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
        x_img = tf.keras.layers.Concatenate(axis=-1, name="cat_img")(
            [inp_img[:, :, :, :, i] for i in range(4)]
        )
        base_model_img = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_img._name = "img_extractor"
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = l2n(x_img)
        inp.append(inp_img)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_feat_all")([y, x_img])
            if y is not None
            else x_img
        )

    if "stft" in DATATYPE:
        raise RuntimeError("stft not enabled in this run")

    y = tf.keras.layers.Dense(
        len(TARGETS_PRETRAIN),
        activation="softmax",
        dtype="float32",
        name="head_softmax",
    )(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_multimodal")
    return model




## === cell 3
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPE = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        SAMPLE_SUB_PATH = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPE = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        SAMPLE_SUB_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"

    print("Test shape", test.shape)

    if not _TF_AVAILABLE:
        pred = np.tile(train_prior[None, :], (len(test), 1)).astype(np.float32)
        eps_mix = 0.01
        uniform = np.full((1, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
        pred = (1.0 - eps_mix) * pred + eps_mix * uniform
        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred
        sub.to_csv("submission.csv", index=False)
        print("Wrote submission.csv (TF unavailable fallback). Shape:", sub.shape)
        print("Row sum min/max:", sub[TARGETS].sum(1).min(), sub[TARGETS].sum(1).max())
    else:
        spectrograms2 = {}
        if "spe" in DATATYPE:
            files2 = os.listdir(PATH_SPE)
            print(f"There are {len(files2)} test spectrogram parquets")
            for i, f in enumerate(files2):
                if i % 200 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(os.path.join(PATH_SPE, f))
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values
            print()

        try:
            from scipy import signal  # type: ignore

            _HAVE_SCIPY = True
        except Exception as e:
            print(
                "WARNING: scipy not available, using no-op signal processing:", repr(e)
            )
            _HAVE_SCIPY = False

            class _SignalFallback:
                @staticmethod
                def butter(*args, **kwargs):
                    return None, None

                @staticmethod
                def resample_poly(x, up, down, axis=1):
                    return x

                @staticmethod
                def filtfilt(b, a, x, axis=1):
                    return x

            signal = _SignalFallback()

        files2 = os.listdir(PATH_EEG)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        imgs2 = {}

        if _HAVE_SCIPY:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        else:
            b, a = None, None

        test_eeg_ids = set(test.eeg_id.astype(int).values.tolist())

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids:
                continue

            eeg_default = pd.read_parquet(os.path.join(PATH_EEG, f))

            list_eeg = []
            list_img = []

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
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            if "eeg" in DATATYPE:
                eegs2[name] = np.concatenate(list_eeg, 2)

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

                imgs2[name] = np.concatenate([img, img2, img3], -1)

        print()

        def _find_weight_file(base_dir: str, fname: str):
            candidates = [
                os.path.join(base_dir, fname),
                os.path.join(base_dir, os.path.basename(base_dir), fname),
            ]
            if os.path.isdir(base_dir):
                try:
                    for subdir in os.listdir(base_dir):
                        p = os.path.join(base_dir, subdir, fname)
                        candidates.append(p)
                except Exception:
                    pass
            for p in candidates:
                if p and os.path.exists(p):
                    return p
            return None

        preds = []
        try:
            with strategy.scope():
                model = build_model(TARGETS)
        except Exception as e:
            print(
                "WARNING: model build failed, using prior-only fallback. Error:",
                repr(e),
            )
            pred = np.tile(train_prior[None, :], (len(test), 1)).astype(np.float32)
            eps_mix = 0.01
            uniform = np.full((1, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
            pred = (1.0 - eps_mix) * pred + eps_mix * uniform
            pred = np.clip(pred, 1e-8, 1.0)
            pred = pred / pred.sum(axis=1, keepdims=True)
            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = pred
            sub.to_csv("submission.csv", index=False)
            print(
                "Wrote submission.csv (model build failed fallback). Shape:", sub.shape
            )
            raise SystemExit(0)

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2 if "spe" in DATATYPE else None,
            eegs=eegs2 if "eeg" in DATATYPE else None,
            imgs=imgs2 if "img" in DATATYPE else None,
            stfts=None,
            targets=TARGETS,
        )

        missing = []
        for i in range(5):
            fname = f"f{i}_stage{STAGETEST}.h5"
            wpath = _find_weight_file(LOAD_MODELS_FROM, fname)
            if wpath is None:
                missing.append(os.path.join(LOAD_MODELS_FROM, fname))
                continue
            print(f"Fold {i+1} loading weights: {wpath}")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        if len(preds) == 0:
            print(
                "WARNING: No model weights found. Creating a valid submission using an eeg_id-aggregated smoothed train prior."
            )
            pred = np.tile(train_prior[None, :], (len(test), 1)).astype(np.float32)
        else:
            pred = np.mean(preds, axis=0).astype(np.float32)
            if len(preds) < 5:
                blend = 0.15 * (5 - len(preds)) / 5.0
                pred = (1.0 - blend) * pred + blend * train_prior[None, :]

        print("Test preds shape", pred.shape)

        eps_mix = 0.01
        uniform = np.full((1, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
        pred = (1.0 - eps_mix) * pred + eps_mix * uniform

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print("Row sum min/max:", sub[TARGETS].sum(1).min(), sub[TARGETS].sum(1).max())
        print(sub.head())

        if missing:
            print("Missing weight files (first few):")
            print("\n".join(missing[:5]))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/751328560.py in <cell line: 0>()
    282             with strategy.scope():
--> 283                 model = build_model(TARGETS)
    284         except Exception as e:

/tmp/ipykernel_55/2615835935.py in build_model(TARGETS_PRETRAIN)
     23         base_model_spe._name = "spe_extractor"
---> 24         x_spe = base_model_spe(x_spe)
     25         x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 

ValueError: Exception encountered when calling layer 'normalization' (type Normalization).

Dimensions must be equal, but are 12 and 3 for '{{node spe_extractor/normalization/sub}} = Sub[T=DT_FLOAT](spe_extractor/rescaling/add, spe_extractor/normalization/sub/y)' with input shapes: [?,128,256,12], [1,1,1,3].

Call arguments received by layer 'normalization' (type Normalization):
  • inputs=tf.Tensor(shape=(None, 128, 256, 12), dtype=float32)

During handling of the above exception, another exception occurred:

SystemExit                                Traceback (most recent call last)
    [... skipping hidden 1 frame]

/tmp/ipykernel_55/751328560.py in <cell line: 0>()
    301             )
--> 302             raise SystemExit(0)
    303 

SystemExit: 0

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
    [... skipping hidden 1 frame]

/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py in showtraceback(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)
   2090                     stb = ['An exception has occurred, use %tb to see '
   2091                            'the full traceback.\n']
-> 2092                     stb.extend(self.InteractiveTB.get_exception_only(etype,
   2093                                                                      value))
   2094                 else:

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in get_exception_only(self, etype, value)
    752         value : exception value
    753         """
--> 754         return ListTB.structured_traceback(self, etype, value)
    755 
    756     def show_exception_only(self, etype, evalue):

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, context)
    627             chained_exceptions_tb_offset = 0
    628             out_list = (
--> 629                 self.structured_traceback(
    630                     etype, evalue, (etb, chained_exc_ids),
    631                     chained_exceptions_tb_offset, context)

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1365         else:
   1366             self.tb = tb
-> 1367         return FormattedTB.structured_traceback(
   1368             self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1369 

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1265         if mode in self.verbose_modes:
   1266             # Verbose modes need a full traceback
-> 1267             return VerboseTB.structured_traceback(
   1268                 self, etype, value, tb, tb_offset, number_of_lines_of_context
   1269             )

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)
   1122         """Return a nice text document describing the traceback."""
   1123 
-> 1124         formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,
   1125                                                                tb_offset)
   1126 

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in format_exception_as_a_whole(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)
   1080 
   1081 
-> 1082         last_unique, recursion_repeat = find_recursion(orig_etype, evalue, records)
   1083 
   1084         frames = self.format_records(records, last_unique, recursion_repeat)

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in find_recursion(etype, value, records)
    380     # first frame (from in to out) that looks different.
    381     if not is_recursion_error(etype, value, records):
--> 382         return len(records), 0
    383 
    384     # Select filename, lineno, func_name to track frames with

TypeError: object of type 'NoneType' has no len()
