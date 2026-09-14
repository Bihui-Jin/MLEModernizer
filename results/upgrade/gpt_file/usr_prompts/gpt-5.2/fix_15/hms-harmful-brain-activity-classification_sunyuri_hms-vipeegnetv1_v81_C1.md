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

0.3535640618406237

# 6. Current score

1.40598

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40594) has done: 'The timeout is dominated by (1) Python-heavy per-sample preprocessing inside `DataGenerator.__data_generation` (especially `tf.image.resize` calls and repeated normalization per-k) and (2) repeated work across 5 folds (the generator redoes the same expensive numpy/TF ops every fold). I keep the exact model and inference semantics, but cache the fully-prepared model inputs per `eeg_id` for test mode (since test has one row per `eeg_id`) so preprocessing happens once, not 5×. I also replace per-sample `tf.image.resize` with a deterministic, equivalent NumPy “area resize” that matches the downsampling pattern here (300->96 in height) and precompute constants (colormap LUT, slice indices, normalization factors) to remove repeated overhead. All I/O paths and predictions (mean over 5 folds, same weights) remain unchanged.'
- What this solution (achieved 1.40594) has done: 'I fix the runtime crash happening before any training/inference by forcing TensorFlow to use the Python protobuf implementation (this avoids the known `MessageFactory.GetPrototype` incompatibility in some Kaggle TF/protobuf builds). I keep the model, preprocessing, and fold-averaging logic identical, only adding this environment safeguard early enough to take effect. I also make the TF GPU strategy selection robust to CPU-only environments to prevent device errors, and ensure the submission is always written with correct column order and probabilities summing to 1. These changes are correctness/stability fixes and should allow your existing (already-scored) approach to run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 1.40598) has done: 'I fix the crash caused by the TensorFlow/protobuf `MessageFactory.GetPrototype` incompatibility by ensuring the protobuf env vars are set before any TF-related import and by importing `google.protobuf` early (this is a stability fix and score-neutral). I also fix a couple of logic bugs that can silently break preprocessing: the incorrect `if "col" in row` checks (Series membership checks values, not index) be replaced with safe `if "col" in self.data.columns`, and the spectrogram normalization division be corrected to use std (not variance) to better match standard ImageNet preprocessing (this should improve KL toward your target). Finally, I keep the model and fold-averaging unchanged while making submission assembly simpler/safer (guaranteed column order and row alignment, probabilities sum to 1).'
- What this solution (achieved 1.40598) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars early and (most importantly) forcing a compatible protobuf version before importing TensorFlow, with a safe fallback that keeps the rest of your pipeline unchanged. I also make the environment setup robust to Kaggle’s CPU-only/GPU setups and ensure the submission is always written with the required column order and row-wise probability normalization. These changes are stability/correctness oriented and should let your existing 5-fold weight-averaged inference run end-to-end (which is necessary before we can meaningfully move the score toward the target). No model architecture, preprocessing semantics, or fold-averaging logic is altered beyond what’s required to avoid the runtime error and produce a valid `submission.csv`.'
- What this solution (achieved 1.40598) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf implementation/version environment variables are set before any protobuf/TensorFlow import and by forcing the pure-Python protobuf backend consistently (this is the root cause of the `MessageFactory.GetPrototype` error in some Kaggle TF/protobuf builds). I keep your model, preprocessing, fold-averaging, and submission logic unchanged, only making the environment setup deterministic and compatible so inference can actually run end-to-end. I also make GPU selection safer by not hard-setting `CUDA_VISIBLE_DEVICES` (which can accidentally hide devices) and leaving TensorFlow to enumerate what Kaggle provides. These changes are stability/correctness fixes and should allow your existing approach to execute and generate a valid `submission.csv` (score should remain in the same regime, enabling further score-directed iterations afterward).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_PROTOBUF_IMPLEMENTATION", "python")

import io
import gc
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import google.protobuf  # noqa: F401
except Exception as e:
    print("Warning: could not import google.protobuf early:", repr(e))

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed (likely protobuf incompatibility). "
        "Tried forcing pure-Python protobuf via environment variables."
    ) from e

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img'
STAGE = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024022901"
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

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

if (
    "CUDA_VISIBLE_DEVICES" in os.environ
    and os.environ["CUDA_VISIBLE_DEVICES"].strip() == ""
):
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

print("TensorFlow version =", tf.__version__)

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

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled (optimizer experimental option)")
    except Exception:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (global policy)")
        except Exception as e:
            print("Could not enable mixed precision:", repr(e))
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
print("Targets:", list(TARGETS))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _area_resize_gray(img2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Deterministic area resize for 2D float32 array (no external deps)."""
    in_h, in_w = img2d.shape
    h_crop = (in_h // out_h) * out_h
    w_crop = (in_w // out_w) * out_w
    if h_crop <= 0 or w_crop <= 0:
        img2d = img2d[: max(out_h, 1), : max(out_w, 1)]
        return img2d.astype(np.float32, copy=False)
    img2d = img2d[:h_crop, :w_crop]
    sh = h_crop // out_h
    sw = w_crop // out_w
    return (
        img2d.reshape(out_h, sh, out_w, sw)
        .mean(axis=(1, 3))
        .astype(np.float32, copy=False)
    )


def _rasterize_eeg_traces_to_img4(
    eeg_all_region: np.ndarray, img_high: int, img_wide: int, amp: float = 200.0
) -> np.ndarray:
    """
    Create (img_high, img_wide, 4) float32 image similar to the Matplotlib pipeline.
    """
    n_lines, t = eeg_all_region.shape

    H = img_high * 4
    W = img_wide

    x_idx = (np.linspace(0, W - 1, t)).astype(np.int32)

    ii = np.arange(n_lines, dtype=np.float32)
    jj = ii * amp + (ii // 4.0) * amp

    y_min = -amp / 2.0
    y_max = n_lines * amp + (amp / 2.0) * 5.0
    y_span = y_max - y_min

    y = eeg_all_region.astype(np.float32) + jj[:, None]
    y_norm = (y - y_min) / y_span
    y_pix = (H - 1 - np.round(y_norm * (H - 1))).astype(np.int32)
    y_pix = np.clip(y_pix, 0, H - 1)

    canvas = np.zeros((H, W), dtype=np.float32)
    for line in range(n_lines):
        canvas[y_pix[line], x_idx] = 1.0

    img = canvas[:, :, None]  # (H, W, 1)

    img4 = np.concatenate(
        [
            img[0 * img_high : 1 * img_high, :, :],
            img[1 * img_high : 2 * img_high, :, :],
            img[2 * img_high : 3 * img_high, :, :],
            img[3 * img_high : 4 * img_high, :, :],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    img4[:, :, 0] = -img4[:, :, 0]
    img4[:, :, 2] = -img4[:, :, 2]
    return img4




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
    ):

        self.cmin = -4
        self.cmax = 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3].astype(
            np.float32, copy=False
        )

        self.data = data.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = False
        self.mode = mode
        self.specs = specs if specs is not None else {}
        self.eegs = eegs if eegs is not None else {}
        self.imgs = imgs if imgs is not None else {}

        self._spe_pad_top = 16  # (HIGH - (HIGH-32))/2 = 16
        self._spe_out_h = HIGH - 32  # 96
        self._spe_out_w = LENGTH
        self._spe_crop_l = max(round((600 / 2 - LENGTH) / 2), 0)
        self._spe_crop_r = self._spe_crop_l + LENGTH

        self._eeg_t = round(EEG_LENGTH * SFREQ)
        self._eeg_center_l = round((50 - EEG_LENGTH) / 2 * SFREQ)
        self._eeg_center_r = round((50 + EEG_LENGTH) / 2 * SFREQ)

        self._spe_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._spe_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._test_cache = {} if self.mode == "test" else None

        self._has_spe_off = "spectrogram_label_offset_seconds" in self.data.columns
        self._has_eeg_off = "eeg_label_offset_seconds" in self.data.columns

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
            x_eeg = np.zeros((len(indexes), 6, self._eeg_t, 4), dtype="float32")
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")
        y = np.zeros((len(indexes), 6), dtype="float32")

        if self.mode == "test" and self._test_cache is not None:
            for j, i in enumerate(indexes):
                row = self.data.iloc[i]
                eid = int(row.eeg_id)
                cached = self._test_cache.get(eid, None)
                if cached is None:
                    cached = self._build_single_test_item(row)
                    self._test_cache[eid] = cached

                off = 0
                if "spe" in DATATYPE:
                    x_spe[j] = cached[off]
                    off += 1
                if "eeg" in DATATYPE:
                    x_eeg[j] = cached[off]
                    off += 1
                if "img" in DATATYPE:
                    x_img[j] = cached[off]
                    off += 1

            x = []
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "img" in DATATYPE:
                x.append(x_img)
            return tuple(x), y

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            else:
                r_spe = (
                    round(row.spectrogram_label_offset_seconds / 2)
                    if self._has_spe_off
                    else 0
                )
                r_eeg = (
                    round(row.eeg_label_offset_seconds * SFREQ)
                    if self._has_eeg_off
                    else 0
                )

            if self.mode == "train":
                x1 = np.random.rand() * (256 / 2 - 20)
                x2 = np.random.rand() * (256 / 2 - 20)
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_spe_min += 128
                    x_spe_max += 128

                x1 = np.random.rand() * (2048 / 2 - 500)
                x2 = np.random.rand() * (2048 / 2 - 500)
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_eeg_min += 1024
                    x_eeg_max += 1024

                x1 = np.random.rand() * (256 / 2 - 64)
                x2 = np.random.rand() * (256 / 2 - 64)
                x_img_min = round(min(x1, x2))
                x_img_max = round(max(x1, x2))
                if np.random.rand() < 0.5:
                    x_img_min += 128
                    x_img_max += 128

            for k in range(4):
                if "spe" in DATATYPE:
                    if row.spectrogram_id in self.specs:
                        spe_src = self.specs[row.spectrogram_id]
                    else:
                        spe_src = np.zeros((300, 400), dtype=np.float32)

                    spe = spe_src[r_spe : r_spe + 300, k * 100 : (k + 1) * 100].T
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
                    spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                    spe = np.array(spe, dtype=np.int16)
                    spe = self.cmaps[spe]
                    spe = np.reshape(spe, (100, 300, 3))
                    spe = spe[
                        :, self._spe_crop_l : min(self._spe_crop_r, spe.shape[1]), :
                    ]

                    if (
                        spe.shape[0] != self._spe_out_h
                        or spe.shape[1] != self._spe_out_w
                    ):
                        spe_f = spe.astype(np.float32, copy=False)
                        spe_rs = np.empty(
                            (self._spe_out_h, self._spe_out_w, 3), dtype=np.float32
                        )
                        for c in range(3):
                            spe_rs[:, :, c] = _area_resize_gray(
                                spe_f[:, :, c], self._spe_out_h, self._spe_out_w
                            )
                        spe = spe_rs
                    else:
                        spe = spe.astype(np.float32, copy=False)

                    if self.mode == "train":
                        spe[:, x_spe_min:x_spe_max, :] = 0

                    x_spe[
                        j, self._spe_pad_top : self._spe_pad_top + spe.shape[0], :, :, k
                    ] = spe

                    x_spe[j, :, :, :, k] = (
                        x_spe[j, :, :, :, k] - self._spe_means
                    ) / self._spe_stds

                if "eeg" in DATATYPE:
                    if row.eeg_id in self.eegs:
                        eeg_src = self.eegs[row.eeg_id]
                    else:
                        eeg_src = np.zeros((4, 5000, 4), dtype=np.float32)

                    eeg = eeg_src[
                        :,
                        r_eeg + self._eeg_center_l : r_eeg + self._eeg_center_r,
                        k,
                    ]
                    if self.mode == "train":
                        eeg[:, x_eeg_min:x_eeg_max] = 0
                    x_eeg[j, 1:5, :, k] = eeg
                    m = np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                    s = np.std(x_eeg[j, :, :, k], 1, keepdims=True)
                    x_eeg[j, :, :, k] = (x_eeg[j, :, :, k] - m) / (s + 1e-6)

                if "img" in DATATYPE:
                    if row.eeg_id in self.imgs:
                        img = self.imgs[row.eeg_id][:, :, k]
                    else:
                        img = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
                        if np.random.randn() > 0:
                            img = -img
                        img[:, x_img_min:x_img_max] = 0
                    x_img[j, :, :, k] = img

            if self.mode != "test":
                label = row[TARGETS].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0
        if (self.mode == "train") and (np.random.random() > 0.5):
            alpha = 1.0
            xx = [np.random.beta(alpha, alpha) for _ in range(y.shape[0])]

        x = []
        if "spe" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xxr = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xxr) + x_spe[::-1, :, :, :, :] * xxr
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xxr = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xxr) + x_eeg[::-1, :, :, :] * xxr
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xxr = np.reshape(xx, (x_img.shape[0], 1, 1, 1))
                x_img = x_img * (1 - xxr) + x_img[::-1, :, :, :] * xxr
            x.append(x_img)

        if (self.mode == "train") and (alpha > 0):
            xxr = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xxr) + y[::-1, :] * xxr

        x = tuple(x)
        return x, y

    def _build_single_test_item(self, row):
        """Build one cached test item for a single row; exact same transforms as __data_generation."""
        out = []
        r_spe = 0
        r_eeg = 0

        if "spe" in DATATYPE:
            x_spe = np.zeros((HIGH, LENGTH, 3, 4), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros((6, self._eeg_t, 4), dtype="float32")
        if "img" in DATATYPE:
            x_img = np.zeros((IMG_HIGH, IMG_WIDE, 4), dtype="float32")

        for k in range(4):
            if "spe" in DATATYPE:
                if row.spectrogram_id in self.specs:
                    spe_src = self.specs[row.spectrogram_id]
                else:
                    spe_src = np.zeros((300, 400), dtype=np.float32)

                spe = spe_src[r_spe : r_spe + 300, k * 100 : (k + 1) * 100].T
                spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                spe = np.log(spe)
                spe = np.nan_to_num(spe, nan=0.0)
                spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
                spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                spe = np.array(spe, dtype=np.int16)
                spe = self.cmaps[spe]
                spe = np.reshape(spe, (100, 300, 3))
                spe = spe[:, self._spe_crop_l : min(self._spe_crop_r, spe.shape[1]), :]

                if spe.shape[0] != self._spe_out_h or spe.shape[1] != self._spe_out_w:
                    spe_f = spe.astype(np.float32, copy=False)
                    spe_rs = np.empty(
                        (self._spe_out_h, self._spe_out_w, 3), dtype=np.float32
                    )
                    for c in range(3):
                        spe_rs[:, :, c] = _area_resize_gray(
                            spe_f[:, :, c], self._spe_out_h, self._spe_out_w
                        )
                    spe = spe_rs
                else:
                    spe = spe.astype(np.float32, copy=False)

                x_spe[self._spe_pad_top : self._spe_pad_top + spe.shape[0], :, :, k] = (
                    spe
                )
                x_spe[:, :, :, k] = (
                    x_spe[:, :, :, k] - self._spe_means
                ) / self._spe_stds

            if "eeg" in DATATYPE:
                if row.eeg_id in self.eegs:
                    eeg_src = self.eegs[row.eeg_id]
                else:
                    eeg_src = np.zeros((4, 5000, 4), dtype=np.float32)

                eeg = eeg_src[
                    :, r_eeg + self._eeg_center_l : r_eeg + self._eeg_center_r, k
                ]
                x_eeg[1:5, :, k] = eeg
                m = np.mean(x_eeg[:, :, k], 1, keepdims=True)
                s = np.std(x_eeg[:, :, k], 1, keepdims=True)
                x_eeg[:, :, k] = (x_eeg[:, :, k] - m) / (s + 1e-6)

            if "img" in DATATYPE:
                if row.eeg_id in self.imgs:
                    img = self.imgs[row.eeg_id][:, :, k]
                else:
                    img = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
                x_img[:, :, k] = img

        if "spe" in DATATYPE:
            out.append(x_spe)
        if "eeg" in DATATYPE:
            out.append(x_eeg)
        if "img" in DATATYPE:
            out.append(x_img)
        return tuple(out)




## === cell 3
def build_model():
    inp = []
    y = None

    l2_layer = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])

        base_model_spe = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            name="efficientnetb0_spe",
        )
        base_model_spe._name = "spe_extractor"

        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = l2_layer(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))  # (6, T, 4)

        x_eeg = tf.keras.layers.Permute((2, 1, 3))(inp_eeg)  # (T, 6, 4)
        x_eeg = tf.keras.layers.Reshape((round(EEG_LENGTH * SFREQ), 6 * 4, 1))(
            x_eeg
        )  # (T, 24, 1)

        x_eeg = tf.keras.layers.Reshape(
            (round(EEG_LENGTH * SFREQ) // 4, (6 * 4) * 4, 1)
        )(
            x_eeg
        )  # (750,96,1)
        x_eeg = tf.keras.layers.Concatenate(axis=3)(
            [x_eeg, x_eeg, x_eeg]
        )  # (750, 96, 3)

        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            name="efficientnetb0_eeg",
        )
        base_model_eeg._name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = l2_layer(x_eeg)

        inp.append(inp_eeg)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg]) if y is not None else x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1)([x_img1, x_img2, x_img3, x_img4])
        x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_tensor=None,
            name="efficientnetb0_img",
        )
        base_model_img._name = "img_extractor"

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = l2_layer(x_img)

        inp.append(inp_img)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_img]) if y is not None else x_img

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 4
def _find_existing_weights_dir(preferred_dir: str, stage: int) -> str:
    """
    BUGFIX: Kaggle datasets may not mount under the expected name.
    We search /kaggle/input for a directory containing the expected fold weight files.
    """
    expected = [f"f{i}_stage{stage}.h5" for i in range(5)]

    if os.path.isdir(preferred_dir):
        ok = all(os.path.isfile(os.path.join(preferred_dir, e)) for e in expected)
        if ok:
            return preferred_dir

    base = "/kaggle/input"
    candidates = []
    if os.path.isdir(base):
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if not os.path.isdir(p):
                continue
            candidates.append(p)
            for dd in os.listdir(p):
                pp = os.path.join(p, dd)
                if os.path.isdir(pp):
                    candidates.append(pp)

    for c in candidates:
        ok = all(os.path.isfile(os.path.join(c, e)) for e in expected)
        if ok:
            return c

    return preferred_dir




## === cell 5
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPE = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPE = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    print("Test shape", test.shape)

    need_spe_ids = set(test.spectrogram_id.values.tolist())
    files_spe = os.listdir(PATH_SPE)
    spe_map = {int(f.split(".")[0]): f for f in files_spe if f.endswith(".parquet")}
    spectrograms2 = {}
    need_list = [sid for sid in need_spe_ids if sid in spe_map]
    print(f"Need {len(need_list)} / {len(files_spe)} test spectrogram parquets")
    for i, sid in enumerate(need_list):
        if i % 200 == 0:
            print(i, ", ", end="")
        f = spe_map[sid]
        tmp = pd.read_parquet(f"{PATH_SPE}{f}")
        spectrograms2[sid] = tmp.iloc[:, 1:].values
    print()

    try:
        from scipy import signal  # type: ignore

        HAVE_SCIPY = True
    except Exception as e:
        HAVE_SCIPY = False
        signal = None
        print("SciPy not available; proceeding without filtering/resampling:", repr(e))

    files_eeg = os.listdir(PATH_EEG)
    eeg_map = {int(f.split(".")[0]): f for f in files_eeg if f.endswith(".parquet")}
    test_eeg_ids = set(test.eeg_id.values.tolist())
    need_eeg_list = [eid for eid in test_eeg_ids if eid in eeg_map]
    print(f"Need {len(need_eeg_list)} / {len(files_eeg)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}

    if HAVE_SCIPY:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
    else:
        b, a = None, None

    for i, name in enumerate(need_eeg_list):
        if i % 200 == 0:
            print(i, ", ", end="")

        f = eeg_map[name]
        eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")

        list_eeg = []
        list_img = []
        for region in BRAIN.keys():
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(BRAIN[region]):
                a0, a1 = chan.split("-")
                eeg[chan_i, :] = (
                    eeg_default.loc[:, a0] - eeg_default.loc[:, a1]
                ).values

            eeg[np.isnan(eeg)] = 0

            if HAVE_SCIPY and (200 != SFREQ):
                eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

            if HAVE_SCIPY:
                eeg = signal.filtfilt(b, a, eeg, axis=1)

            time_temp = 0
            time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
            time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

            list_img.append(eeg[:, time_start:time_stop])
            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)
        eegs2[name] = list_eeg

        eeg_all_region = np.concatenate(list_img, 0)
        imgs2[name] = _rasterize_eeg_traces_to_img4(
            eeg_all_region, img_high=IMG_HIGH, img_wide=IMG_WIDE, amp=200.0
        )

    print()

    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
    )

    preds = []
    with strategy.scope():
        model = build_model()

    resolved_weights_dir = _find_existing_weights_dir(LOAD_MODELS_FROM, STAGE)
    print("Resolved weights dir:", resolved_weights_dir)

    missing = []
    for i in range(5):
        wpath = os.path.join(resolved_weights_dir, f"f{i}_stage{STAGE}.h5")
        if not os.path.isfile(wpath):
            missing.append(wpath)
    if len(missing) > 0:
        print(
            "WARNING: Missing weight files. Will still run inference with random init."
        )
        for p in missing[:10]:
            print(" missing:", p)

    for i in range(5):
        print(f"Fold {i+1}")
        wpath = os.path.join(resolved_weights_dir, f"f{i}_stage{STAGE}.h5")
        if os.path.isfile(wpath):
            model.load_weights(wpath)
        pred_i = model.predict(test_gen, verbose=1)
        preds.append(pred_i)

    pred = np.mean(preds, axis=0)
    print("Test preds shape", pred.shape)

    pred = np.clip(pred, 1e-9, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame(pred, columns=list(TARGETS))
    sub.insert(0, "eeg_id", test.eeg_id.values)
    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row-sum stats:",
        float(sub[list(TARGETS)].sum(axis=1).min()),
        float(sub[list(TARGETS)].sum(axis=1).max()),
    )
    print(sub.head())
