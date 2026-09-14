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

0.3181784697594211

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the missing weight files issue by auto-detecting available `.h5` weight files in `/kaggle/input` and using them if present; if none are available, the script still run end-to-end and produce a valid, normalized submission using a safe fallback (uniform probabilities). Finally, I make the cell numbering valid (start at 1) and add a strict submission column/order alignment check to ensure the CSV matches `sample_submission.csv` and rows sum to 1.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf implementation *and* pinning the protobuf runtime to the compatible API before TensorFlow is imported, plus adding a safe fallback to avoid TensorFlow entirely if the environment still can’t import it. To move the score toward your target (lower-is-better), I replace the uniform-probability fallback (which yields very poor KL) with a score-safe “prior” fallback: predicting the normalized mean label distribution from `train.csv` for every test row (this is a minimal, legitimate calibration change and usually much better than uniform for KL). I also fix the cell numbering (must start at 1) and make sure the submission columns exactly match `sample_submission.csv` and each row sums to 1. Core model logic/architecture is unchanged; only robustness and prediction fallback/calibration are adjusted.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* monkey-patching the missing `MessageFactory.GetPrototype` method before importing TensorFlow, which addresses the exact `AttributeError` you hit. I keep the model, generator, and prediction logic unchanged, but make the TensorFlow availability check reliable so the script always runs end-to-end. I also correct the cell numbering (must start at 1) and keep the existing “train label prior” fallback, which is a minimal and legitimate calibration that should improve KL versus uniform when weights are missing or TF can’t load. Finally, I enforce submission column order and per-row probability normalization so the output CSV is always valid.'
- What this solution (achieved 1.41937) has done: 'I fix the immediate protobuf crash by making the monkey-patch apply to the actual `MessageFactory` class used by `google.protobuf` in this environment (patching the module attribute was not sufficient). Then I keep the model/data pipeline unchanged, but ensure the script always reaches submission writing even if TensorFlow still cannot import or if no weights are found (using the existing train-label-prior fallback). Finally, I correct the cell numbering to start at 1 and keep strict submission column order + per-row normalization so the CSV is always valid for Kaggle.'
- What this solution (achieved 1.41937) has done: 'I fix the immediate runtime crash by making the protobuf monkey-patch apply reliably before TensorFlow import, including patching the actual C++/python MessageFactory class instance that triggers `GetPrototype` calls. Then I ensure the notebook always runs end-to-end even if TensorFlow still can’t import, keeping your existing “train label prior” fallback (score-improving vs uniform for KL) unchanged in semantics. I also correct the cell numbering to start at 1 (Kaggle/your validator requirement) and keep the strict submission column order + per-row normalization so the CSV is always valid. No model architecture, generator logic, or loss logic is changed.'
- What this solution (achieved 1.41937) has done: 'I fix the remaining protobuf/TensorFlow crash by strengthening the protobuf `MessageFactory.GetPrototype` monkey-patch so it reliably applies to the exact class TensorFlow ends up using in this environment (including the C++-backed implementation), and ensure the patch runs strictly before any TensorFlow import. This is a runtime/stability fix only; it does not change your model architecture or inference logic. Once TensorFlow can import, your existing weight-loading + prediction path run (which should improve KL vs the current fallback-prior-only behavior and move the score toward the target). I also correct the cell numbering to start at 1 and keep the submission column order + per-row normalization checks intact so a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os
import io

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _patch_protobuf_message_factory():
    """
    Fix for environments where google.protobuf MessageFactory lacks GetPrototype,
    which TensorFlow tries to call during import/graph construction.

    We patch multiple likely MessageFactory classes/instances (public + internal)
    and also attempt to patch the C++-backed implementation if present.
    """
    try:
        from google.protobuf import message_factory as mf

        def _add_getprototype(obj_or_cls):
            if hasattr(obj_or_cls, "GetPrototype"):
                return
            if hasattr(obj_or_cls, "GetMessageClass"):

                def GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                try:
                    setattr(obj_or_cls, "GetPrototype", GetPrototype)
                except Exception:
                    pass

        if hasattr(mf, "MessageFactory"):
            _add_getprototype(mf.MessageFactory)
            try:
                _add_getprototype(mf.MessageFactory())
            except Exception:
                pass

        try:
            from google.protobuf.internal import message_factory as imf

            if hasattr(imf, "MessageFactory"):
                _add_getprototype(imf.MessageFactory)
                try:
                    _add_getprototype(imf.MessageFactory())
                except Exception:
                    pass
        except Exception:
            pass

        try:
            from google.protobuf.pyext import _message as cpp_message  # type: ignore

            if hasattr(cpp_message, "MessageFactory"):
                _add_getprototype(cpp_message.MessageFactory)
                try:
                    _add_getprototype(cpp_message.MessageFactory())
                except Exception:
                    pass
        except Exception:
            pass

    except Exception:
        pass


_patch_protobuf_message_factory()

from PIL import Image

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024040901"
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # noqa: F401

TF_AVAILABLE = True
try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        f"WARNING: TensorFlow import failed; will use fallback predictions. Error: {repr(e)}"
    )

if "stft" in DATATYPE:
    import librosa  # noqa: F401

if TF_AVAILABLE:
    print("TensorFlow version =", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")
else:
    strategy = None

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
if TF_AVAILABLE:
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

MIX = True
if TF_AVAILABLE:
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision not available; continuing.")
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    import albumentations as albu  # noqa: F401
except Exception as e:
    print(f"albumentations import skipped due to runtime incompatibility: {repr(e)}")
    albu = None

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
            self.specs = specs if specs is not None else {}
            self.eegs = eegs if eegs is not None else {}
            self.imgs = imgs if imgs is not None else {}
            self.stfts = stfts if stfts is not None else {}
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
                x_eeg2 = np.zeros(
                    (len(indexes), 4, round(50 * SFREQ), 4), dtype="float32"
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
                    rows = df.loc[df.eeg_id == row.eeg_id, :].reset_index(drop=True)
                    for lk in TARGETS:
                        rows = rows.loc[
                            rows[lk] == row.get(lk + "_raw", row.get(lk)), :
                        ].reset_index(drop=True)
                    rows_eeg = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )

                    r_spe = round(
                        getattr(row, "spectrogram_label_offset_seconds", 0) / 2
                    )
                    r_eeg = round(rows_eeg.eeg_label_offset_seconds[0] * SFREQ)

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
                        if row.spectrogram_id in self.specs:
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

                            eeg = (eeg - np.mean(eeg, 1, keepdims=True)) / (
                                np.std(eeg, 1, keepdims=True) + 1e-6
                            )
                            x_eeg2[j, :, :, k] = eeg

                    if "img" in DATATYPE:
                        if row.eeg_id in self.imgs:
                            img = self.imgs[row.eeg_id][:, :, k, :]
                            x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        if row.eeg_id in self.stfts:
                            stft = self.stfts[row.eeg_id][:, :, k]

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
                x.append(x_eeg2)

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




## === cell 2
if TF_AVAILABLE:

    def build_model(TARGETS_PRETRAIN):
        def l2norm_layer():
            return tf.keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))

        inp = []
        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1, name="concat_spe_quads")(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_spe",
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
            x_spe = l2norm_layer()(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4), name="inp_eeg")
            x_eeg1 = inp_eeg[:, :, :, :, 0]
            x_eeg2 = inp_eeg[:, :, :, :, 1]
            x_eeg3 = inp_eeg[:, :, :, :, 2]
            x_eeg4 = inp_eeg[:, :, :, :, 3]

            x_eeg1 = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_12")(
                [x_eeg1, x_eeg2]
            )
            x_eeg2 = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_34")(
                [x_eeg3, x_eeg4]
            )
            x_eeg = tf.keras.layers.Concatenate(axis=2, name="concat_eeg_time")(
                [x_eeg1, x_eeg2]
            )

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_eeg",
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
            x_eeg = l2norm_layer()(x_eeg)

            inp.append(inp_eeg)

            if "spe" in DATATYPE:
                y = tf.keras.layers.Concatenate(axis=1, name="concat_spe_eeg")(
                    [y, x_eeg]
                )
            else:
                y = x_eeg

        if "eeg" in DATATYPE:
            inp_eeg2 = tf.keras.Input(shape=(4, round(50 * SFREQ), 4), name="inp_eeg2")
            inp.append(inp_eeg2)

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
            x_img1 = inp_img[:, :, :, :, 0]
            x_img2 = inp_img[:, :, :, :, 1]
            x_img3 = inp_img[:, :, :, :, 2]
            x_img4 = inp_img[:, :, :, :, 3]
            x_img = tf.keras.layers.Concatenate(axis=1, name="concat_img_quads")(
                [x_img1, x_img2, x_img3, x_img4]
            )

            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_img",
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
            x_img = l2norm_layer()(x_img)

            inp.append(inp_img)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="concat_prev_img")(
                    [y, x_img]
                )
            else:
                y = x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1, name="concat_stft_quads")(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_stft",
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
            x_stft = l2norm_layer()(x_stft)

            inp.append(inp_stft)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="concat_prev_stft")(
                    [y, x_stft]
                )
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




## === cell 3
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

    TARGETS_SUB = [c for c in sample_sub.columns if c != "eeg_id"]

    print("Test shape", test.shape)
    test.head()

    prior = df[TARGETS].astype(np.float64).sum(axis=0).values
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()
    print("Train prior:", dict(zip(TARGETS, prior.round(6))))

    if "spe" in DATATYPE and TF_AVAILABLE:
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
            if i % 100 == 0:
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

    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
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

                if "stft" in DATATYPE:
                    import librosa  # local import

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
                if TF_AVAILABLE:
                    img = np.array(
                        tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                else:
                    img = img.astype(np.float32) / 255.0
                    img = img[: IMG_HIGH * 4, :IMG_WIDE, :]
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
                if TF_AVAILABLE:
                    img2 = np.array(
                        tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                else:
                    img2 = img2.astype(np.float32) / 255.0
                    img2 = img2[: IMG_HIGH * 4, :IMG_WIDE, :]
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
                if TF_AVAILABLE:
                    img3 = np.array(
                        tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                else:
                    img3 = img3.astype(np.float32) / 255.0
                    img3 = img3[: IMG_HIGH * 4, :IMG_WIDE, :]
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

    def _discover_weight_files(load_dir: str, stage: int):
        paths = []
        if os.path.isdir(load_dir):
            for i in range(5):
                p = os.path.join(load_dir, f"f{i}_stage{stage}.h5")
                if os.path.isfile(p):
                    paths.append(p)
        return paths

    weight_files = _discover_weight_files(LOAD_MODELS_FROM, STAGETEST)

    if len(weight_files) == 0 and PLATFORM == "kaggle":
        for root, _, files in os.walk("/kaggle/input"):
            for i in range(5):
                fn = f"f{i}_stage{STAGETEST}.h5"
                if fn in files:
                    weight_files.append(os.path.join(root, fn))
        weight_files = sorted(set(weight_files))

    print(f"Discovered {len(weight_files)} weight file(s) for stage {STAGETEST}.")
    if len(weight_files) == 0:
        print(
            "WARNING: No model weights found in /kaggle/input. "
            "Will generate a valid submission using the train label prior (better than uniform for KL)."
        )

    if TF_AVAILABLE and len(weight_files) > 0:
        preds = []
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

        for wf in weight_files:
            print(f"Predicting with weights: {wf}")
            model.load_weights(wf)
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0).astype(np.float32)
        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, 1e-7, 1.0)
        pred = pred / np.sum(pred, axis=1, keepdims=True)
    else:
        pred = np.tile(prior.astype(np.float32), (len(test), 1))

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[list(TARGETS)] = pred

    sub = sub[["eeg_id"] + TARGETS_SUB].copy()

    p = sub[TARGETS_SUB].to_numpy(dtype=np.float64)
    p = np.clip(p, 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sub[TARGETS_SUB] = p.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print("Submission shape", sub.shape)
    print(
        "Row-sum stats:",
        float(sub[TARGETS_SUB].sum(axis=1).min()),
        float(sub[TARGETS_SUB].sum(axis=1).max()),
    )
    sub.head()
