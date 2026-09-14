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

0.3438441016922353

# 6. Current score

1.19354

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a common Kaggle/Python 3.12 incompatibility) while keeping the model and data logic unchanged. Next, I make inference robust to missing external weight files by automatically falling back to an untrained “prior” prediction built from the training vote distribution; this guarantees a valid `submission.csv` is always written. This fallback is score-oriented (usually much better than random) but still minimal and does not change the core model definition or evaluation semantics. Finally, I keep the submission format identical to `sample_submission.csv` and enforce per-row probability normalization.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars before *any* TensorFlow-related import (your current cell ordering means TF can be imported before those vars are applied in some notebook/run contexts). Then I keep your existing inference logic, but improve the fallback path (used when weights are missing) to use a per-patient prior (falls back to global prior if patient unseen), which is a minimal, metric-aligned calibration change and should reduce KL from 1.419 toward your target. Finally, I make sure the written `submission.csv` is always valid (correct columns, probabilities clipped and row-normalized) without changing the model architecture or training semantics.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the Python protobuf implementation *and* disabling the C++ protobuf runtime before any TensorFlow import (the current env vars are not sufficient in some Kaggle Py3.12 images). Next, I keep your model/inference logic unchanged, but make the fallback prior slightly more robust by mapping test `patient_id` to a safe per-patient prior and enforcing correct target-column ordering (score-neutral to slightly positive, no architecture/training changes). Finally, I ensure the code always writes a valid `submission.csv` with the exact sample submission columns and strictly row-normalized probabilities.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring the required protobuf environment variables are set before any TensorFlow import (your current “cell 0” isn’t executed in this script format, so TF imports too early). I keep the model/data logic unchanged, and only add a safe TensorFlow import fallback (force pure-Python protobuf + retry) to make execution robust in Kaggle Py3.12. Then I ensure the script always completes and writes a valid `submission.csv` with correct columns and per-row normalization (score-neutral). This should unblock the run and allow your existing weights (if present) to be used; if missing, it still produces the calibrated prior fallback you already had.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime *before any import that could transitively load protobuf/TensorFlow*, and by setting the additional env var `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus disabling the C++ implementation in a way that is effective in Kaggle’s Py3.12 runtime. Then I keep your model/inference logic unchanged and ensure the pipeline always reaches submission writing, even if TF still cannot import, by falling back to the already-present patient/global prior path (this is metric-aligned and should improve KL vs random, moving score toward the target). Finally, I ensure the produced `submission.csv` has the exact required columns/order and strictly row-normalized probabilities.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables in the very first executable cell (cell 1) before any other imports can transitively load protobuf/TensorFlow, and by adding a safe re-try TensorFlow import after forcing the pure-Python protobuf implementation. This should restore the intended path where your pretrained fold weights (if available) are used; using the real model instead of the prior-only fallback is the minimal change most likely to move KL down toward the target. I also keep the existing patient/global prior fallback intact (so a valid `submission.csv` is always produced) and make the TensorFlow import failure non-fatal (so the script continues to submission writing). No model architecture/training/inference logic is changed beyond making TF import robust.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable setup to the first executed cell (cell 1) so it is applied before any import that can transitively load protobuf/TensorFlow; in your current script, cell 0 isn’t guaranteed to run, which is why you still see `MessageFactory.GetPrototype`. I also make the TensorFlow import failure non-fatal so the pipeline always reaches the fallback prior and writes a valid `submission.csv`. These changes preserve your model/inference logic and only affect import robustness and completion. This should also improve score versus the prior-only path when TensorFlow successfully imports and weights exist, moving KL down toward the target.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by moving the protobuf environment-variable setup into the very first executed cell (cell 1) and ensuring it happens before any other import that might load protobuf (including PIL/matplotlib). Then I keep your model/inference logic unchanged, but make TF import failures non-fatal so the script always reaches the existing patient/global prior fallback and writes `submission.csv`. Finally, I keep the submission column order identical to `sample_submission.csv` and enforce strict per-row probability normalization (score-neutral), so Kaggle accepts the file and you can benefit from real weights when TF imports correctly.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable setup into the first executed cell (cell 1) and ensuring no imports that can transitively load protobuf/TensorFlow happen before it. Then I make the TensorFlow import retry more robust by explicitly switching protobuf to the Python implementation and clearing any already-imported protobuf/tensorflow modules before retrying. These changes are execution-critical but score-neutral by themselves; they should allow your intended path (loading pretrained fold weights) to run, which is the minimal legitimate way to improve KL substantially from the current prior-heavy behavior. Finally, I keep your existing prior fallback and strict submission normalization so a valid `submission.csv` is always produced.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable setup into the first executed cell and ensuring no protobuf-dependent imports happen before it; this is the root cause of the `MessageFactory.GetPrototype` error in your current cell ordering. Then I make the TensorFlow import fallback robust by forcing the pure-Python protobuf implementation and clearing already-imported protobuf/tensorflow modules before retrying, but keep your model/inference logic unchanged. Finally, I keep the existing prior-based fallback and strict per-row probability normalization so the script always finishes and writes a valid `submission.csv`; when TensorFlow successfully imports and weights exist, your intended model path run, which is the minimal legitimate way to move KL down toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024032201"
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
import sys

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_AVAILABLE = True
tf = None

try:
    import tensorflow as tf  # noqa: F401
except Exception as e1:
    print("TensorFlow import failed on first attempt. Error:", repr(e1))
    TF_AVAILABLE = False
    try:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP"] = "0"

        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf") or m.startswith("tensorflow"):
                sys.modules.pop(m, None)

        import tensorflow as tf  # noqa: F401

        TF_AVAILABLE = True
        print("TensorFlow import succeeded on retry. TF version =", tf.__version__)
    except Exception as e2:
        TF_AVAILABLE = False
        print(
            "TensorFlow import failed on retry; will use prior-only fallback. Error:",
            repr(e2),
        )

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if TF_AVAILABLE:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("enable_op_determinism not available:", repr(e))

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision option not available:", repr(e))
    else:
        print("Using full precision")

    print("TensorFlow version =", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    import albumentations as albu  # noqa: F401
except Exception as e:
    albu = None
    print("albumentations not available (ok, unused):", repr(e))

TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
TARS2 = {x: y for y, x in TARS.items()}

if TF_AVAILABLE:

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
                        img = self.imgs[row.eeg_id][:, :, k, :]
                        x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        pass

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
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
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
if TF_AVAILABLE:
    from tensorflow.keras import layers as KL

    def _effnet_b0_backbone(name: str):
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            name=name,
        )
        return base

    def build_model(TARGETS_PRETRAIN):
        inp = list()
        y = None

        l2norm = KL.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm")

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe = KL.Concatenate(axis=1)(
                [
                    inp_spe[:, :, :, :, 0],
                    inp_spe[:, :, :, :, 1],
                    inp_spe[:, :, :, :, 2],
                    inp_spe[:, :, :, :, 3],
                ]
            )
            base_model_spe = _effnet_b0_backbone("spe_extractor")
            x_spe = base_model_spe(x_spe)
            x_spe = KL.GlobalAveragePooling2D()(x_spe)
            x_spe = l2norm(x_spe)
            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg = KL.Concatenate(axis=1)(
                [
                    inp_eeg[:, :, :, :, 0],
                    inp_eeg[:, :, :, :, 1],
                    inp_eeg[:, :, :, :, 2],
                    inp_eeg[:, :, :, :, 3],
                ]
            )
            base_model_eeg = _effnet_b0_backbone("eeg_extractor")
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = KL.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2norm(x_eeg)
            inp.append(inp_eeg)
            y = KL.Concatenate(axis=1)([y, x_eeg]) if y is not None else x_eeg

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img = KL.Concatenate(axis=1)(
                [
                    inp_img[:, :, :, :, 0],
                    inp_img[:, :, :, :, 1],
                    inp_img[:, :, :, :, 2],
                    inp_img[:, :, :, :, 3],
                ]
            )
            base_model_img = _effnet_b0_backbone("img_extractor")
            x_img = base_model_img(x_img)
            x_img = KL.GlobalAveragePooling2D()(x_img)
            x_img = l2norm(x_img)
            inp.append(inp_img)
            y = KL.Concatenate(axis=1)([y, x_img]) if y is not None else x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_stft = KL.Concatenate(axis=1)(
                [
                    inp_stft[:, :, :, :, 0],
                    inp_stft[:, :, :, :, 1],
                    inp_stft[:, :, :, :, 2],
                    inp_stft[:, :, :, :, 3],
                ]
            )
            base_model_stft = _effnet_b0_backbone("stft_extractor")
            x_stft = base_model_stft(x_stft)
            x_stft = KL.GlobalAveragePooling2D()(x_stft)
            x_stft = l2norm(x_stft)
            inp.append(inp_stft)
            y = KL.Concatenate(axis=1)([y, x_stft]) if y is not None else x_stft

        y = KL.Dense(len(TARGETS_PRETRAIN), activation="softmax", dtype="float32")(y)
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
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    print("Test shape", test.shape)
    test.head()

    TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]

    vote_sums_global = df[TARGETS].sum(axis=0).values.astype(np.float64)
    prior_global = vote_sums_global / max(vote_sums_global.sum(), 1.0)
    prior_global = np.clip(prior_global, 1e-8, 1.0)
    prior_global = prior_global / prior_global.sum()
    print("Fallback global prior probs:", dict(zip(TARGETS, prior_global.round(6))))

    vote_sums_by_patient = (
        df.groupby("patient_id")[list(TARGETS)].sum().astype(np.float64)
    )
    denom = vote_sums_by_patient.sum(axis=1).replace(0.0, 1.0)
    prior_by_patient = (vote_sums_by_patient.T / denom).T
    prior_by_patient = prior_by_patient.clip(lower=1e-8)
    prior_by_patient = prior_by_patient.div(prior_by_patient.sum(axis=1), axis=0)

    if TF_AVAILABLE:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
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

        test_eeg_ids = set(test.eeg_id.values.tolist())

        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids:
                continue

            eeg_default = pd.read_parquet(f"{PATH2}{f}")

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
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if "stft" in DATATYPE:
                stfts2[name] = np.concatenate(list_stft, 2)

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

        print()

    possible_model_roots = [LOAD_MODELS_FROM]
    if PLATFORM == "kaggle":
        possible_model_roots.append(
            os.path.join("/kaggle/working", os.path.basename(LOAD_MODELS_FROM))
        )

    def _find_weights_path(fname: str) -> str:
        for root in possible_model_roots:
            cand = os.path.join(root, fname)
            if os.path.exists(cand):
                return cand
        return os.path.join(possible_model_roots[0], fname)

    pred = None

    if TF_AVAILABLE:
        weight_paths = []
        for i in range(5):
            wpath = _find_weights_path(f"f{i}_stage{STAGETEST}.h5")
            if os.path.exists(wpath):
                weight_paths.append(wpath)

        if len(weight_paths) == 0:
            print(
                f"No weight files found for stage {STAGETEST} under roots {possible_model_roots}. Writing fallback prior submission."
            )
            pred_list = []
            for pid in test["patient_id"].values:
                if pid in prior_by_patient.index:
                    pred_list.append(prior_by_patient.loc[pid, list(TARGETS)].values)
                else:
                    pred_list.append(prior_global.copy())
            pred = np.vstack(pred_list).astype(np.float64)
        else:
            preds = []
            with strategy.scope():
                model = build_model(TARGETS)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=tf.keras.losses.KLDivergence(),
                )

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

            for i in range(5):
                wpath = _find_weights_path(f"f{i}_stage{STAGETEST}.h5")
                if not os.path.exists(wpath):
                    print(f"Fold {i + 1}: missing weights, skipping: {wpath}")
                    continue
                print(f"Fold {i + 1}: loading {wpath}")
                model.load_weights(wpath)
                pred_i = model.predict(test_gen, verbose=1)
                preds.append(pred_i)

            if len(preds) == 0:
                print("All folds missing weights; writing fallback prior submission.")
                pred_list = []
                for pid in test["patient_id"].values:
                    if pid in prior_by_patient.index:
                        pred_list.append(
                            prior_by_patient.loc[pid, list(TARGETS)].values
                        )
                    else:
                        pred_list.append(prior_global.copy())
                pred = np.vstack(pred_list).astype(np.float64)
            else:
                pred = np.mean(preds, axis=0).astype(np.float64)
    else:
        pred_list = []
        for pid in test["patient_id"].values:
            if pid in prior_by_patient.index:
                pred_list.append(prior_by_patient.loc[pid, list(TARGETS)].values)
            else:
                pred_list.append(prior_global.copy())
        pred = np.vstack(pred_list).astype(np.float64)

    print("Test preds shape", pred.shape)

    pred = np.clip(pred, 1e-8, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub = sub[sample_sub.columns]

    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-12, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    sub.loc[:, TARGETS] = vals

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row sums (min/mean/max):",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).mean(),
        sub[TARGETS].sum(axis=1).max(),
    )
    print("Wrote submission.csv")
