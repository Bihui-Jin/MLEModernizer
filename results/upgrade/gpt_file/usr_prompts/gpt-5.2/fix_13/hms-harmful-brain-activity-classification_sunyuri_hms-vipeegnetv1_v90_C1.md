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

0.3464995089196601

# 6. Current score

0.96631

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure‑python protobuf environment override, which is causing the `MessageFactory.GetPrototype` error in this Kaggle/Python 3.12 setup. Then I make inference robust to missing external fold weight files: if the expected `.h5` weights are not present under `/kaggle/input`, the script fall back to a valid, properly-normalized probability submission based on the empirical class prior from `train.csv` (score-safe baseline) instead of crashing. This keeps the core model/data pipeline intact and only changes behavior when weights are unavailable, ensuring an end-to-end run that always writes `submission.csv` in the required format. The submission always have the correct columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from importing TensorFlow under Python 3.12 by safely forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep your existing “use fold weights if present, otherwise fall back to a class-prior submission” logic intact, so it always runs end-to-end and writes a valid `submission.csv`. Finally, I make the fallback slightly closer to the target KL metric by applying light Dirichlet/Laplace smoothing to the class prior (still a legitimate label-distribution baseline) and ensuring strict row-wise normalization.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf override, which is the direct cause of the `MessageFactory.GetPrototype` AttributeError in this Kaggle Python 3.12 environment. I also make SciPy an optional dependency: if it is not available, the code safely fall back to the already-present class-prior submission path instead of crashing mid-inference. To move the KL score toward your target (lower is better) while keeping core logic intact, I improve the fallback distribution from a global prior to a patient-conditioned prior (computed from `train.csv` only), with light smoothing and strict normalization, which is a legitimate calibration improvement without changing the model. The script always write a valid `submission.csv` with the required columns summing to 1 per row.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround in Kaggle’s Python 3.12 environment. I also add a safe fallback if TensorFlow still can’t import for any reason: the code still produce a valid `submission.csv` using the existing patient-conditioned prior logic. Finally, I keep the model/data logic unchanged and only touch import/guard code paths and submission robustness (normalization and column alignment are preserved).'
- What this solution (achieved 0.96631) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure‑Python protobuf override (it’s the direct trigger for `MessageFactory.GetPrototype` in this environment) and instead make TensorFlow import optional with a safe fallback. To move the score toward your lower-is-better target while keeping the “no-weights fallback” core approach, I strengthen the fallback from a patient-conditioned prior to a patient-conditioned *and* consensus-conditioned prior (still trained only from `train.csv`), with light smoothing and strict normalization. I also make the prior mapping robust to unseen `expert_consensus` values and ensure the submission columns match `sample_submission.csv` exactly and sum to 1 per row. No changes are made to your model architecture/training logic; only import stability and fallback calibration are adjusted.'
- What this solution (achieved 0.96631) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround for this Kaggle Python 3.12 environment. I also make mixed precision opt-in/off by default to avoid any instability while keeping the model architecture and inference logic the same (this is score-neutral when using the same weights). Finally, I keep your existing “use fold weights if present, otherwise prior fallback” behavior intact, ensuring the script always writes a valid `submission.csv` with correct columns and row-wise normalization.'
- What this solution (achieved 0.96631) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf override that triggers `MessageFactory.GetPrototype` in this Kaggle/Python 3.12 environment, and instead make TensorFlow import optional (keeping your existing fallback behavior). Then I correct a small but impactful bug in the training-time masking augmentation (`x_eeg_max`/`x_img_max` incorrectly using `min`), which is a logic error and can hurt learned calibration when you do have weights. Finally, I keep submission generation unchanged except for ensuring robust probability normalization and CSV output as you already do, so the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.96631) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround for the `MessageFactory.GetPrototype` error under Kaggle’s Python 3.12 image. This is an execution fix (not a modeling change) and allow the existing inference path (loading fold `.h5` weights and predicting) to run when weights are present, which should move your score down toward the target versus the current fallback-heavy behavior. I also keep the existing “if TF/SciPy/weights missing, write a calibrated prior-based submission” safety net intact, ensuring a valid `submission.csv` is always produced with correct columns and row-wise normalization. No architecture/training logic is changed.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import io
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TF_AVAILABLE = True
try:
    import tensorflow as tf  # noqa: F401

    print("TensorFlow version =", tf.__version__)
except Exception as e:
    _TF_AVAILABLE = False
    tf = None
    print("Warning: TensorFlow failed to import:", repr(e))

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img'
STAGE = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024030101"
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

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

MIX = False
if _TF_AVAILABLE:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism enable warning:", repr(e))

    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision enable warning:", repr(e))
    else:
        print("Using full precision")

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
else:
    strategy = None

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
if _TF_AVAILABLE:
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
            self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.imgs = imgs
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
                    (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")

            y = np.zeros((len(indexes), 6), dtype="float32")

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
                    x1 = np.random.rand() * (256 / 2 - 20)
                    x2 = np.random.rand() * (256 / 2 - 20)
                    x_spe_min = round(min(x1, x2))
                    x_spe_max = round(max(x1, x2))
                    if np.random.rand() < 0.5:
                        x_spe_min = x_spe_min + 128
                        x_spe_max = x_spe_max + 128

                    x1 = np.random.rand() * (2048 / 2 - 500)
                    x2 = np.random.rand() * (2048 / 2 - 500)
                    x_eeg_min = round(min(x1, x2))
                    x_eeg_max = round(max(x1, x2))
                    if np.random.rand() < 0.5:
                        x_eeg_min = x_eeg_min + 1024
                        x_eeg_max = x_eeg_max + 1024

                    x1 = np.random.rand() * (256 / 2 - 64)
                    x2 = np.random.rand() * (256 / 2 - 64)
                    x_img_min = round(min(x1, x2))
                    x_img_max = round(max(x1, x2))
                    if np.random.rand() < 0.5:
                        x_img_min = x_img_min + 128
                        x_img_max = x_img_max + 128

                for k in range(4):
                    if "spe" in DATATYPE:
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
                            max(round((600 / 2 - LENGTH) / 2), 0) : min(
                                round((600 / 2 - LENGTH) / 2) + LENGTH, spe.shape[1]
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
                        eeg = self.eegs[row.eeg_id][
                            :,
                            r_eeg
                            + round((50 - EEG_LENGTH) / 2 * SFREQ) : r_eeg
                            + round((50 + EEG_LENGTH) / 2 * SFREQ),
                            k,
                        ]
                        if self.mode == "train":
                            eeg[:, x_eeg_min:x_eeg_max] = 0

                        x_eeg[j, 1:5, :, k] = eeg
                        x_eeg[j, :, :, k] = (
                            x_eeg[j, :, :, k]
                            - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                    if "img" in DATATYPE:
                        img = self.imgs[row.eeg_id][:, :, k]

                        if self.mode == "train":
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
                    xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1))
                    x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx

                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img

                x.append(x_img)

            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (y.shape[0], 1))
                y = y * (1 - xx) + y[::-1, :] * xx

            return x, y




## === cell 2
if _TF_AVAILABLE:

    def _maybe_load_imagenet_notop_weights(model):
        if not NEEDTRAIN:
            return
        candidates = []
        if PLATFORM == "local":
            candidates.append(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        else:
            candidates.append(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

        for p in candidates:
            if os.path.exists(p):
                model.load_weights(p)
                return
        print(
            "Warning: EfficientNet ImageNet notop weights not found; continuing without them."
        )

    def build_model():
        l2_layer = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )

        inp = []
        y = None

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
                include_top=False, weights=None, name="efficientnetb0_spe"
            )
            base_model_spe._name = "spe_extractor"
            _maybe_load_imagenet_notop_weights(base_model_spe)

            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = l2_layer(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))
            x_eeg1 = inp_eeg[:, :, :, 0:1]
            x_eeg2 = inp_eeg[:, :, :, 1:2]
            x_eeg3 = inp_eeg[:, :, :, 2:3]
            x_eeg4 = inp_eeg[:, :, :, 3:4]
            x_eeg = tf.keras.layers.Concatenate(axis=1)(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )
            x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_eeg"
            )
            base_model_eeg._name = "eeg_extractor"
            _maybe_load_imagenet_notop_weights(base_model_eeg)

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2_layer(x_eeg)

            inp.append(inp_eeg)

            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            else:
                y = x_eeg

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
            x_img1 = inp_img[:, :, :, 0:1]
            x_img2 = inp_img[:, :, :, 1:2]
            x_img3 = inp_img[:, :, :, 2:3]
            x_img4 = inp_img[:, :, :, 3:4]
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [x_img1, x_img2, x_img3, x_img4]
            )
            x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_img"
            )
            base_model_img._name = "img_extractor"
            _maybe_load_imagenet_notop_weights(base_model_img)

            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = l2_layer(x_img)

            inp.append(inp_img)

            if y is not None:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
            else:
                y = x_img

        y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
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

    def _class_prior_from_train(
        train_df: pd.DataFrame, target_cols, alpha_per_class: float = 1.0
    ):
        votes = train_df[list(target_cols)].to_numpy(dtype=np.float64)
        counts = votes.sum(axis=0)
        counts = counts + float(alpha_per_class)
        prior = counts / counts.sum()
        prior = np.clip(prior, 1e-12, 1.0)
        prior = prior / prior.sum()
        return prior.astype(np.float32)

    def _patient_consensus_prior_maps_from_train(
        train_df: pd.DataFrame, target_cols, alpha_per_class: float = 1.0
    ):
        cols = ["patient_id", "expert_consensus"] + list(target_cols)
        tmp = train_df[cols].copy()
        tmp["expert_consensus"] = (
            tmp["expert_consensus"].astype(str).fillna("nan").str.strip()
        )

        grp_pc = tmp.groupby(["patient_id", "expert_consensus"])[
            list(target_cols)
        ].sum()
        counts_pc = grp_pc.to_numpy(dtype=np.float64) + float(alpha_per_class)
        probs_pc = counts_pc / counts_pc.sum(axis=1, keepdims=True)
        probs_pc = np.clip(probs_pc, 1e-12, 1.0)
        probs_pc = probs_pc / probs_pc.sum(axis=1, keepdims=True)
        pc_map = {
            (pid, cons): probs_pc[i].astype(np.float32)
            for i, (pid, cons) in enumerate(grp_pc.index)
        }

        grp_p = tmp.groupby("patient_id")[list(target_cols)].sum()
        counts_p = grp_p.to_numpy(dtype=np.float64) + float(alpha_per_class)
        probs_p = counts_p / counts_p.sum(axis=1, keepdims=True)
        probs_p = np.clip(probs_p, 1e-12, 1.0)
        probs_p = probs_p / probs_p.sum(axis=1, keepdims=True)
        p_map = {
            pid: probs_p[i].astype(np.float32) for i, pid in enumerate(grp_p.index)
        }

        grp_c = tmp.groupby("expert_consensus")[list(target_cols)].sum()
        counts_c = grp_c.to_numpy(dtype=np.float64) + float(alpha_per_class)
        probs_c = counts_c / counts_c.sum(axis=1, keepdims=True)
        probs_c = np.clip(probs_c, 1e-12, 1.0)
        probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)
        c_map = {
            cons: probs_c[i].astype(np.float32) for i, cons in enumerate(grp_c.index)
        }

        return pc_map, p_map, c_map

    def _find_weights_dir_and_pattern(expected_stage: int):
        preferred = LOAD_MODELS_FROM
        if os.path.isdir(preferred):
            expected_files = [f"f{i}_stage{expected_stage}.h5" for i in range(5)]
            if all(
                os.path.exists(os.path.join(preferred, fn)) for fn in expected_files
            ):
                return preferred

        search_root = "/kaggle/input" if PLATFORM == "kaggle" else "./input"
        expected_files = [f"f{i}_stage{expected_stage}.h5" for i in range(5)]
        for root, dirs, files in os.walk(search_root):
            file_set = set(files)
            if all(fn in file_set for fn in expected_files):
                return root
        return None

    def _write_prior_submission(reason: str):
        print("Falling back to prior submission. Reason:", reason)
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        vote_cols = [c for c in sample_sub.columns if c != "eeg_id"]

        global_prior = _class_prior_from_train(df, TARGETS, alpha_per_class=1.0)
        pc_map, p_map, c_map = _patient_consensus_prior_maps_from_train(
            df, TARGETS, alpha_per_class=1.0
        )

        tmp = df[["patient_id", "expert_consensus"]].copy()
        tmp["expert_consensus"] = (
            tmp["expert_consensus"].astype(str).fillna("nan").str.strip()
        )
        patient_top_consensus = (
            tmp.groupby(["patient_id", "expert_consensus"])
            .size()
            .reset_index(name="n")
            .sort_values(["patient_id", "n"], ascending=[True, False])
            .drop_duplicates("patient_id")
            .set_index("patient_id")["expert_consensus"]
            .to_dict()
        )

        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        test_patients = test["patient_id"].values
        out = np.zeros((len(test), len(vote_cols)), dtype=np.float32)
        for i, pid in enumerate(test_patients):
            cons = patient_top_consensus.get(pid, None)
            if cons is not None and (pid, cons) in pc_map:
                out[i] = pc_map[(pid, cons)]
            elif pid in p_map:
                out[i] = p_map[pid]
            else:
                out[i] = global_prior

        p = out.astype(np.float64)
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        sub[vote_cols] = p.astype(np.float32)

        sub.to_csv("submission.csv", index=False)
        print("Submission saved to submission.csv (prior fallback)")
        print("Submission shape", sub.shape)
        print(
            "Row sums (min/mean/max):",
            sub[vote_cols].sum(axis=1).min(),
            sub[vote_cols].sum(axis=1).mean(),
            sub[vote_cols].sum(axis=1).max(),
        )

    if not _TF_AVAILABLE:
        _write_prior_submission("TensorFlow import failed")
    else:
        try:
            from scipy import signal  # noqa: F401

            _HAS_SCIPY = True
        except Exception as e:
            print("Warning: SciPy not available:", repr(e))
            _HAS_SCIPY = False

        weights_dir = _find_weights_dir_and_pattern(STAGE)
        if weights_dir is None or (not _HAS_SCIPY):
            if weights_dir is None:
                _write_prior_submission(
                    f"could not locate fold weights for stage{STAGE}"
                )
            else:
                _write_prior_submission(
                    "SciPy missing; cannot run EEG preprocessing for inference"
                )
        else:
            print("Using weights_dir:", weights_dir)

            files_spe = os.listdir(PATH_SPE)
            print(f"There are {len(files_spe)} test spectrogram parquets")
            spectrograms2 = {}
            for i, f in enumerate(files_spe):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_SPE}{f}")
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values
            print()

            from scipy import signal

            files_eeg = os.listdir(PATH_EEG)
            print(f"There are {len(files_eeg)} test eeg parquets")

            eegs2 = {}
            imgs2 = {}
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

            test_eeg_ids = set(test.eeg_id.astype(int).tolist())

            for i, f in enumerate(files_eeg):
                if i % 100 == 0:
                    print(i, ", ", end="")
                name = int(f.split(".")[0])
                if name not in test_eeg_ids:
                    continue

                eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")

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
                    time_start = round(
                        time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ
                    )
                    time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                    list_img.append(eeg[:, time_start:time_stop])
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)
                eegs2[name] = list_eeg

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
                imgs2[name] = img

            print()
            print(
                "Prepared eegs2:",
                len(eegs2),
                "imgs2:",
                len(imgs2),
                "spectrograms2:",
                len(spectrograms2),
            )

            missing = sorted(list(test_eeg_ids - set(eegs2.keys())))
            if len(missing) > 0:
                raise RuntimeError(
                    f"Missing preprocessed EEG/img for {len(missing)} test eeg_id(s). Example: {missing[:5]}"
                )

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
                imgs=imgs2,
            )

            for i in range(5):
                print(f"Fold {i+1}")
                wpath = os.path.join(weights_dir, f"f{i}_stage{STAGE}.h5")
                if not os.path.exists(wpath):
                    raise FileNotFoundError(f"Missing model weights: {wpath}")
                model.load_weights(wpath)
                pred = model.predict(test_gen, verbose=1)
                preds.append(pred)

            pred = np.mean(preds, axis=0)
            print("Test preds shape", pred.shape)

            sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
            vote_cols = [c for c in sample_sub.columns if c != "eeg_id"]

            sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
            sub[vote_cols] = pred

            p = sub[vote_cols].values.astype(np.float64)
            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum(axis=1, keepdims=True)
            sub[vote_cols] = p.astype(np.float32)

            sub.to_csv("submission.csv", index=False)
            print("Submission saved to submission.csv")
            print("Submission shape", sub.shape)
            print(
                "Row sums (min/mean/max):",
                sub[vote_cols].sum(axis=1).min(),
                sub[vote_cols].sum(axis=1).mean(),
                sub[vote_cols].sum(axis=1).max(),
            )
            print(sub.head())
