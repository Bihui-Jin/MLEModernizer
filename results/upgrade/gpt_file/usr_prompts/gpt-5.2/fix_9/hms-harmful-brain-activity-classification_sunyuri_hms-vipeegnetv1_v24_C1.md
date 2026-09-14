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

0.4764342231961925

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash in the first cell by avoiding the protobuf/TensorFlow conflict that triggers the `MessageFactory.GetPrototype` AttributeError, without changing the modeling logic. Then I fix the model-building error by ensuring the two EfficientNetB0 backbones have unique names (Keras now requires globally unique operation/model names), which unblocks inference and submission creation. Finally, I make submission writing more robust by aligning columns to `sample_submission.csv` order and ensuring probabilities are valid (clipped + row-normalized), so Kaggle accepts the file.'
- What this solution (achieved 1.40995) has done: 'You’re currently blocked by a TensorFlow/protobuf runtime crash in cell 0; I fix that by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is a common Kaggle workaround and is score-neutral. Then I ensure the script always produces a valid `submission.csv` by making the submission column order match `sample_submission.csv` exactly and by clipping + row-normalizing probabilities (also score-neutral but prevents “row sums != 1” failures). Finally, because your current score (1.40995, lower-is-better) is far from the target (0.4764), I also fix a key inference alignment issue: the generator currently predicts per-row of `test.csv` (9850) which is fine, but we must ensure predictions align to the exact `eeg_id` order in `sample_submission.csv` (some competitions require that); this can materially improve score without changing model logic.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash in TensorFlow caused by the protobuf `MessageFactory.GetPrototype` incompatibility by forcing a compatible protobuf implementation/version *before* importing TensorFlow (this is score-neutral but required for the notebook to run). Then I correct a key inference issue in your data generator for `mode="test"`: right now it always uses `r=0`, which ignores most of each spectrogram and hurts score; I switch to using the centered window similarly to validation (this preserves the same modeling logic and feature pipeline, but uses the intended representative segment). Finally, I keep your submission alignment/normalization safeguards and ensure the output CSV matches `sample_submission.csv` column order and sums to 1 per row.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars even earlier and making the setting idempotent, then importing TensorFlow only after that (this is required for the notebook to run). Next, I fix a core inference bug that is likely driving the very poor KL score: the test `DataGenerator` currently uses a hard-coded spectrogram offset (`r=150`) which can be out-of-bounds for many spectrogram arrays; I compute a safe centered crop based on the actual spectrogram length and also clamp/round `r` for train/valid. Finally, I keep the existing submission alignment/normalization but ensure `r` is always valid so predictions are based on real data instead of accidental empty/incorrect crops, which should move the score down toward the target without changing the model/training logic.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by making the protobuf env-var workaround happen before any TensorFlow/protobuf-related import and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early and consistently. Then I address a likely major score issue without changing your model/training logic: the spectrogram crop start `r` is currently computed from seconds but used as spectrogram-row indices; I convert offset seconds to spectrogram row index (0.5s per row) and clamp safely, so the model sees the intended time window instead of an almost-random/shifted crop. Finally, I keep your submission alignment and probability normalization, ensuring a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys, gc, math
import numpy as np
import pandas as pd

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240201"
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

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

try:
    from tensorflow.python.framework.ops import reset_default_graph  # noqa: F401
except Exception:
    reset_default_graph = None

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version =", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = True
if MIX:
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (global policy)")
    except Exception as e:
        print(
            "Mixed precision policy set failed, continuing in default precision:",
            repr(e),
        )
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
try:
    import albumentations as albu
except Exception:
    albu = None

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
    ):

        self.cmin = -6
        self.cmax = 8
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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _safe_r(self, row, spec_arr):
        """
        FIX (score-critical): spectrogram_label_offset_seconds are in seconds, but spec_arr is indexed by rows.
        Each spectrogram row corresponds to 0.5 seconds (20 Hz time bins). Convert seconds -> rows and clamp.
        """
        n_rows = int(spec_arr.shape[0])
        crop_h = 300
        if n_rows <= crop_h:
            return 0
        max_r = n_rows - crop_h

        def sec_to_row(sec):
            try:
                return int(np.round(float(sec) * 2.0))
            except Exception:
                return 0

        if self.mode == "test":
            r = (n_rows - crop_h) // 2
        elif self.mode == "valid":
            r = int((sec_to_row(row["min"]) + sec_to_row(row["max"])) // 2)
        else:
            lo = sec_to_row(row["min"])
            hi = sec_to_row(row["max"])
            if hi < lo:
                lo, hi = hi, lo
            r = int(np.random.randint(lo, hi + 1))

        if r < 0:
            r = 0
        if r > max_r:
            r = max_r
        return int(r)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            spec_arr = self.specs[row.spec_id]
            r = self._safe_r(row, spec_arr)

            for k in range(4):
                img = spec_arr[r : r + 300, k * 100 : (k + 1) * 100].T
                img_eeg = self.eegs[row.eeg_id][:, :, k]

                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

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

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y[j] = row[TARGETS].values

        return X, X_eeg, y

    def __random_transform(self, img):
        if albu is None:
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




## === cell 2
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
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="spectrogram_effnetb0"
    )

    if PLATFORM == "local":
        wpath = "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
    else:
        wpath = "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
    if os.path.exists(wpath):
        base_model.load_weights(wpath)
    else:
        print(
            f"WARNING: EfficientNetB0 weights not found at {wpath}. Using random initialization for base_model."
        )

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=1)([x0, x1, x2, x3])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)

    base_model_eeg = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name="eeg_effnetb0"
    )
    if os.path.exists(wpath):
        base_model_eeg.load_weights(wpath)
    else:
        print(
            f"WARNING: EfficientNetB0 weights not found at {wpath}. Using random initialization for base_model_eeg."
        )

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
        PATH_SPEC = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        sample_sub_path = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPEC = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        sample_sub_path = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"

    print("Test shape", test.shape)

    files2 = os.listdir(PATH_SPEC)
    print(f"There are {len(files2)} test spectrogram parquets")

    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    from scipy import signal

    test_eeg_ids = set(test.eeg_id.astype(int).tolist())

    files2 = os.listdir(PATH_EEG)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    for i, f in enumerate(files2):
        name = int(f.split(".")[0])
        if name not in test_eeg_ids:
            continue

        if i % 100 == 0:
            print(i, ", ", end="")
        raw_eeg = pd.read_parquet(f"{PATH_EEG}{f}")

        time_temp = 0
        time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
        time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

        list_eeg = []
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
            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)
        eegs2[name] = list_eeg
    print()

    preds = []
    with strategy.scope():
        model = build_model()

    test_gen = DataGenerator(
        test, shuffle=False, batch_size=32, mode="test", specs=spectrograms2, eegs=eegs2
    )

    loaded_any = False
    if os.path.isdir(LOAD_MODELS_FROM):
        for i in range(5):
            w = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
            if not os.path.exists(w):
                print(f"WARNING: missing fold weights: {w} (skipping)")
                continue
            print(f"Fold {i+1}")
            model.load_weights(w)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)
            loaded_any = True
    else:
        print(
            f"WARNING: model directory not found: {LOAD_MODELS_FROM}. Will predict with untrained model."
        )

    if loaded_any:
        pred = np.mean(preds, axis=0)
    else:
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)

    print("Test preds shape", pred.shape)

    pred = np.clip(pred, 1e-8, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sample_sub = pd.read_csv(sample_sub_path)
    target_cols = [c for c in sample_sub.columns if c != "eeg_id"]

    pred_df = pd.DataFrame(pred, columns=target_cols)
    pred_df["eeg_id"] = test["eeg_id"].values
    pred_df = pred_df.groupby("eeg_id", as_index=False)[target_cols].mean()

    sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

    miss = sub[target_cols].isna().any(axis=1)
    if miss.any():
        sub.loc[miss, target_cols] = 1.0 / len(target_cols)

    vals = sub[target_cols].to_numpy(dtype=np.float32)
    vals = np.clip(vals, 1e-8, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub[target_cols] = vals

    sub = sub[["eeg_id"] + target_cols]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row sum stats:",
        sub[target_cols].sum(axis=1).min(),
        sub[target_cols].sum(axis=1).max(),
    )
    print(sub.head())
