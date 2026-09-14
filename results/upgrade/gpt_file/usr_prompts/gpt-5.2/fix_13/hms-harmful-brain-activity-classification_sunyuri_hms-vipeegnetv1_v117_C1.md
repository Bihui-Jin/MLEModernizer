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

0.347466739170762

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf implementation and making GPU/mixed-precision setup safe for Kaggle’s TensorFlow build. Then I fix the submission length mismatch by generating predictions aggregated to one row per `eeg_id` (the required submission key), and merging them onto `sample_submission.csv` in the same order/length. Finally, I add guards so that if any modalities aren’t loaded (e.g., `READ_SPEC_FILES=False`), the code won’t crash and still produce a valid probability submission that sums to 1 for every row.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf-related TensorFlow crash by forcing a compatible protobuf runtime implementation *before* importing TensorFlow, and by making the optional determinism call safe. Then I keep your inference/model logic intact but ensure the script always finds the model directory on Kaggle (either `/kaggle/input/...` or `/kaggle/working/...`) so weights actually load; right now missing weights can silently produce uniform predictions and a very poor score. Finally, I add a tiny, score-aligned post-processing step (probability “sharpening” via power + renormalization) which typically reduces KL divergence when predictions are under-confident, without changing the model architecture or training loop.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash by removing the incompatible forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"` setting (this is what triggers the missing `_message` symbol in this environment), and instead force the safe pure-Python protobuf backend before importing TensorFlow. Because cell 0 currently fails, downstream cells never see `tf`, `NEEDTRAIN`, etc., so I keep the same core logic but make sure all globals are defined by making the first cell succeed. I also keep the existing inference/submission logic intact and only add small guards to ensure the weights path resolves and a valid `submission.csv` is always written with probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the runtime crash in the first cell caused by an incompatible protobuf implementation selection (your environment is hitting `MessageFactory.GetPrototype` issues). The minimal safe fix is to stop forcing the pure-Python protobuf backend and instead let TensorFlow use its default compatible protobuf runtime in Kaggle’s TF build. Then I keep the rest of your pipeline intact (same model, same inference, same aggregation), only adding small guards so the script can run end-to-end even if SciPy isn’t available by gracefully falling back to uniform predictions (still valid). This should both unblock execution and, when weights exist, restore non-uniform model predictions (improving KL toward your target).'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError seen in this environment. This is an execution-unblocking change and does not alter your model/training logic. I also ensure the weights directory resolution falls back safely to known Kaggle locations so your `.h5` files actually load (otherwise you get near-uniform predictions and a very poor KL score). Finally, I keep your existing inference/aggregation and submission formatting, only adding a tiny guard to keep probability normalization valid.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by removing the forced protobuf runtime override that’s incompatible with this Kaggle TF build, letting TensorFlow use its bundled compatible protobuf implementation. I also make the TF import/setup more robust (safe determinism/mixed-precision toggles) without changing your model, generator, or inference logic. Then I keep the existing prediction/merge pipeline intact so it still outputs exactly 9850 rows with probabilities summing to 1, and still loads weights if present (avoiding uniform predictions). These changes are execution-unblocking and should move the KL score down toward your target by ensuring real model weights can be used.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the safe pure-Python protobuf implementation *before* importing TensorFlow (the current error is a known incompatibility in some Kaggle/Python 3.12 environments). Then I ensure TensorFlow loads cleanly even if optional determinism/mixed-precision toggles are unsupported, without changing your model architecture or inference loop. Finally, to move the KL score down toward your target (lower is better) with minimal semantics change, I make the probability “sharpening” conditional: only apply it when real weights are loaded, and reduce the sharpening strength to avoid overconfident miscalibration (a common cause of worse KL).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import io
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix  # kept (original import)

PLATFORM = "kaggle"  # local / kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024031301"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    cand1 = f"/kaggle/input/{LOAD_MODELS_FROM}"
    cand2 = f"/kaggle/working/{LOAD_MODELS_FROM}"
    cand3 = (
        f"/kaggle/input/hms-harmful-brain-activity-classification/{LOAD_MODELS_FROM}"
    )
    cand4 = (
        f"/kaggle/working/hms-harmful-brain-activity-classification/{LOAD_MODELS_FROM}"
    )
    for c in (cand1, cand2, cand3, cand4):
        if os.path.exists(c):
            LOAD_MODELS_FROM = c
            break
print("LOAD_MODELS_FROM =", LOAD_MODELS_FROM)

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024
BATCHSIZE = 16
AMP = 200

READ_SPEC_FILES = "spe" in DATATYPE
READ_EEG_FILES = "eeg" in DATATYPE
READ_IMG_FILES = "img" in DATATYPE
READ_STFT_FILES = "stft" in DATATYPE

spectrograms, eegs, imgs, stfts = {}, {}, {}, {}
spectrograms2, eegs2, imgs2, stfts2 = {}, {}, {}, {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("WARNING: enable_op_determinism failed:", repr(e))

print("TensorFlow version =", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(
        device="/gpu:0" if len(gpus) == 1 else "/cpu:0"
    )
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled (optimizer experimental option)")
    except Exception:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled (mixed_precision policy)")
        except Exception as e:
            print("WARNING: Mixed precision could not be enabled:", repr(e))
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
import matplotlib as _mpl

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
    ):

        self.cmin = -4
        self.cmax = 6
        self.cmaps = _mpl.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
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
                (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            elif self.mode == "valid":
                r_spe = round(getattr(row, "spectrogram_label_offset_seconds", 0) / 2)
                r_eeg = round(getattr(row, "eeg_label_offset_seconds", 0) * SFREQ)
            else:
                r_spe = round(getattr(row, "spectrogram_label_offset_seconds", 0) / 2)
                r_eeg = round(getattr(row, "eeg_label_offset_seconds", 0) * SFREQ)

            if self.mode == "train":
                x1 = np.random.rand() * LENGTH / 2
                x2 = np.random.rand() * LENGTH / 2
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
                    if row.spectrogram_id in self.specs:
                        spe = self.specs[row.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T
                        spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                        spe = np.log(spe)
                        spe = np.nan_to_num(spe, nan=0.0)
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
                        spe = np.array(
                            tf.image.resize(spe, ((HIGH - 32), LENGTH)),
                            dtype=np.float32,
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
                            :,
                            r_eeg
                            + round((50 - EEG_LENGTH) / 2 * SFREQ) : r_eeg
                            + round((50 + EEG_LENGTH) / 2 * SFREQ),
                            k,
                        ]
                        x_eeg[j, 1:5, :, k] = eeg
                        x_eeg[j, :, :, k] = (
                            x_eeg[j, :, :, k]
                            - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    if row.eeg_id in self.imgs:
                        img = self.imgs[row.eeg_id][:, :, k]
                        x_img[j, :, :, k] = img

                if "stft" in DATATYPE:
                    if row.eeg_id in self.stfts:
                        stft = self.stfts[row.eeg_id][:, :, k]
                        stft = np.clip(stft, np.exp(-4), np.exp(4))
                        stft = np.log(stft)
                        stft = np.nan_to_num(stft, nan=0.0)
                        stft = np.round((stft - (-4)) / (4 - (-4)) * 255)
                        shape0, shape1 = stft.shape[0], stft.shape[1]
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)
                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (shape0, shape1, 3))
                        stft = np.array(
                            tf.image.resize(stft, ((HIGH - 32), LENGTH)),
                            dtype=np.float32,
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
                label = row[TARGETS].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        x = []
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "img" in DATATYPE:
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)
        if "stft" in DATATYPE:
            x.append(x_stft)

        if len(x) == 1:
            x = x[0]
        else:
            x = tuple(x)

        return x, y




## === cell 2
def _make_effnet_b0(name: str):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=None, name=name
    )
    return base


def build_model():
    l2n = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    inp = []
    y = None

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe = tf.keras.layers.Concatenate(axis=1, name="cat_spe_4")(
            [inp_spe[:, :, :, :, i] for i in range(4)]
        )

        base_model_spe = _make_effnet_b0("spe_efficientnetb0")
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
        x_spe = l2n(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(6, round(EEG_LENGTH * SFREQ), 4), name="inp_eeg"
        )
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="cat_eeg_4")(
            [inp_eeg[:, :, :, i : i + 1] for i in range(4)]
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3, name="rgb_eeg")(
            [x_eeg, x_eeg, x_eeg]
        )

        base_model_eeg = _make_effnet_b0("eeg_efficientnetb0")
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = l2n(x_eeg)

        inp.append(inp_eeg)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_spe_eeg")([y, x_eeg])
            if y is not None
            else x_eeg
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="inp_img")
        x_img = tf.keras.layers.Concatenate(axis=1, name="cat_img_4")(
            [inp_img[:, :, :, i : i + 1] for i in range(4)]
        )
        x_img = tf.keras.layers.Concatenate(axis=3, name="rgb_img")(
            [x_img, x_img, x_img]
        )

        base_model_img = _make_effnet_b0("img_efficientnetb0")
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = l2n(x_img)

        inp.append(inp_img)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_prev_img")([y, x_img])
            if y is not None
            else x_img
        )

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
        x_stft = tf.keras.layers.Concatenate(axis=1, name="cat_stft_4")(
            [inp_stft[:, :, :, :, i] for i in range(4)]
        )

        base_model_stft = _make_effnet_b0("stft_efficientnetb0")
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
        x_stft = l2n(x_stft)

        inp.append(inp_stft)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_prev_stft")([y, x_stft])
            if y is not None
            else x_stft
        )

    y = tf.keras.layers.Dense(
        6, activation="softmax", dtype="float32", name="head_softmax"
    )(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    return model




## === cell 3
try:
    from scipy import signal  # type: ignore

    HAVE_SCIPY = True
except Exception as e:
    print(
        "WARNING: scipy.signal import failed; will fall back to uniform predictions.",
        repr(e),
    )
    HAVE_SCIPY = False

if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        PATH_SPE = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        PATH_SPE = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )

    print("Test shape", test.shape)

    test_unique = test.drop_duplicates(subset=["eeg_id"]).reset_index(drop=True)

    spectrograms2, eegs2, imgs2, stfts2 = {}, {}, {}, {}

    if HAVE_SCIPY:
        if READ_SPEC_FILES and ("spe" in DATATYPE):
            files_spe = os.listdir(PATH_SPE)
            print(f"There are {len(files_spe)} test spectrogram parquets")
            for i, f in enumerate(files_spe):
                if i % 200 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(os.path.join(PATH_SPE, f))
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values
            print()

        files_eeg = os.listdir(PATH_EEG)
        print(f"There are {len(files_eeg)} test eeg parquets")

        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        test_eeg_ids = set(test.eeg_id.values.tolist())

        need_eeg_processing = (
            (READ_EEG_FILES and ("eeg" in DATATYPE))
            or (READ_IMG_FILES and ("img" in DATATYPE))
            or (READ_STFT_FILES and ("stft" in DATATYPE))
        )

        for i, f in enumerate(files_eeg):
            if i % 200 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids:
                continue
            if not need_eeg_processing:
                continue

            eeg_default = pd.read_parquet(os.path.join(PATH_EEG, f))

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
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)
                list_img.append(eeg[:, time_start:time_stop])

                if "stft" in DATATYPE and READ_STFT_FILES:
                    ff, tt, pp = signal.spectrogram(
                        eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        fs=SFREQ,
                        nperseg=232,
                        noverlap=194,
                        window="hann",
                        nfft=512,
                        scaling="density",
                        mode="magnitude",
                    )
                    pp = pp[:, ff <= 20, :]
                    pp = np.mean(pp, 0)
                    list_stft.append(np.reshape(pp, (pp.shape[0], pp.shape[1], 1)))

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if "stft" in DATATYPE and READ_STFT_FILES:
                list_stft = np.concatenate(list_stft, 2)
                stfts2[name] = list_stft

            if "eeg" in DATATYPE and READ_EEG_FILES:
                eegs2[name] = list_eeg

            if "img" in DATATYPE and READ_IMG_FILES:
                eeg_all_region = np.concatenate(list_img, 0)
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                for ii in range(eeg_all_region.shape[0]):
                    jj = ii * AMP + (ii // 4) * AMP
                    plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-10, eeg_all_region.shape[1] + 10)
                plt.ylim(-AMP / 2, eeg_all_region.shape[0] * AMP + AMP / 2 * 5)
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
                imgs2[name] = img

        print()
    else:
        print("Proceeding without loading/parsing EEG/spec/img due to missing SciPy.")

    preds = []
    with strategy.scope():
        model = build_model()

    test_gen = DataGenerator(
        test_unique,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        imgs=imgs2,
        stfts=stfts2,
    )

    loaded_any = False
    for fold in range(5):
        print(f"Fold {fold+1}")
        w_path = os.path.join(LOAD_MODELS_FROM, f"f{fold}_stage{STAGETEST}.h5")

        if not os.path.exists(w_path):
            print(f"WARNING: weights not found: {w_path} (skipping load for this fold)")
            continue

        model.load_weights(w_path)
        loaded_any = True

        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)

    if not loaded_any:
        print("WARNING: No model weights were loaded; using uniform predictions.")
        pred = np.full((len(test_unique), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        pred = np.mean(preds, axis=0)

    print("Test preds shape", pred.shape)

    pred = np.clip(pred, 1e-7, 1.0)
    pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)

    if loaded_any:
        SHARPEN_P = 1.12
        pred = np.power(pred, SHARPEN_P)
        pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-12)

    pred_df = pd.DataFrame(pred, columns=list(TARGETS))
    pred_df.insert(0, "eeg_id", test_unique["eeg_id"].astype(np.int64).values)

    sub = sample_sub[["eeg_id"]].copy()
    sub["eeg_id"] = sub["eeg_id"].astype(np.int64)
    sub = sub.merge(pred_df, on="eeg_id", how="left")

    for c in TARGETS:
        if sub[c].isna().any():
            sub[c] = sub[c].fillna(1.0 / 6.0)

    sub[list(TARGETS)] = np.clip(sub[list(TARGETS)].values, 1e-7, 1.0)
    sub[list(TARGETS)] = sub[list(TARGETS)].values / np.maximum(
        sub[list(TARGETS)].values.sum(axis=1, keepdims=True), 1e-12
    )

    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row prob sum min/max:",
        sub[list(TARGETS)].sum(axis=1).min(),
        sub[list(TARGETS)].sum(axis=1).max(),
    )
    print(sub.head())
