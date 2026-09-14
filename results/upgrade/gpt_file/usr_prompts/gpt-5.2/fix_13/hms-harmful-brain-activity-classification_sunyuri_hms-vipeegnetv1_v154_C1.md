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

0.3430339504409042

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error on Kaggle/Py3.12). Then I ensure the produced submission has exactly the same `eeg_id` set and row count/order as `sample_submission.csv` (your current code writes 9850 rows from `test.csv`, which can mismatch the official expected IDs/order and triggers the “same length” error). Finally, I make prediction-to-submission alignment robust by merging predictions onto `sample_submission` and filling any missing IDs with a uniform distribution while renormalizing, ensuring valid probabilities that sum to 1.'
- What this solution (achieved 1.40995) has done: 'I first fix the TensorFlow/protobuf crash on Kaggle Py3.12 by forcing the pure-Python protobuf implementation early and also setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` for better compatibility before importing TensorFlow. Next, to improve the score toward your target (lower is better), I avoid loading EfficientNet with random weights by switching `weights=None` to `weights="imagenet"` (same architecture; just proper pretrained initialization), which typically yields a large KL improvement for this competition when using pretrained fold weights and/or any finetuning. Finally, I keep the existing robust submission alignment/renormalization logic so the script always writes a valid `submission.csv` with exactly the sample submission IDs/order and row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash coming from the TensorFlow↔protobuf incompatibility on Kaggle Py3.12 by ensuring the protobuf pure-Python backend is forced *before* TensorFlow is imported, and by defensively importing TensorFlow in a way that avoids the `MessageFactory.GetPrototype` path. Then I fix a hidden runtime bug in the generator where `x_eeg2` is allocated but never filled (it’s currently normalized anyway), by removing the unused normalization block to prevent future shape/NaN issues without changing model semantics. Finally, I keep your submission alignment/renormalization logic but make it stricter about column order/dtypes so the produced `submission.csv` is always valid and row-probabilities sum to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash by forcing a protobuf version that is compatible with TensorFlow on Kaggle/Python 3.12 (the current env var setting still allows the incompatible protobuf to be used, causing `MessageFactory.GetPrototype` to fail). I also add a defensive import fallback so the notebook continues even if TF still fails, and in that case it reliably write a valid `submission.csv` in the exact `sample_submission.csv` order with probabilities summing to 1. These changes are execution-critical and score-neutral when TensorFlow loads successfully (the model/prediction path remains unchanged). No model/training logic is altered.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow import crash on Kaggle/Python 3.12 by forcing the pure-Python protobuf implementation *and* pre-importing `google.protobuf` before importing TensorFlow, then retrying the TF import once (this addresses the `MessageFactory.GetPrototype` AttributeError). I also add a safe guard so that feature generation for images does not call `tf.image.resize` when TensorFlow is unavailable (preventing a secondary crash in fallback mode). Finally, I keep your model/weights/inference logic unchanged, but ensure that the script always writes `submission.csv` in exact `sample_submission.csv` order with strictly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'I fix the root-cause TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation earlier and more robustly (including disabling the C++ implementation and avoiding duplicate/partial imports), so TF can import on Kaggle Py3.12 and your real model inference runs instead of the uniform fallback (which is why the score is currently very poor). I also make the TF-failure fallback deterministic and always produce a valid `submission.csv` in exact `sample_submission.csv` order with probabilities that sum to 1 (score-neutral when TF works, but prevents invalid outputs). Finally, I keep your model/inference logic intact and only add minimal guards around environment/device config so it doesn’t crash on single-GPU/CPU setups.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash that prevents the real model inference from running (currently you hit `MessageFactory.GetPrototype` and fall back to uniform predictions, which explains the very poor score). The minimal reliable fix in Kaggle/Py3.12 is to force protobuf’s pure-Python implementation and also remove any already-imported `google.protobuf*` modules before importing TensorFlow, so TF cannot latch onto the incompatible C++ backend. I also make the fallback path use the competition’s expected ID order (`sample_submission.csv`) and keep the strict row-wise probability normalization (score-neutral when TF works). No model architecture, generator logic, or inference semantics are changed.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import io
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_AVAILABLE = True
tf = None
strategy = None

try:
    import google.protobuf  # noqa: F401
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print("ERROR: TensorFlow failed to import; will fall back to uniform submission.")
    print("TF import exception:", repr(e))

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [1, 2, 3]
STAGETEST = 3
print("DATATYPE =", DATATYPE)

LOAD_MODELS_FROM = "models2024040301"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

threshold = 0.2

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

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0, 1")

if TF_AVAILABLE:
    print("TensorFlow version =", tf.__version__)

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        gpus = []

    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(
            device="/gpu:0" if len(gpus) == 1 else "/cpu:0"
        )
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
    except Exception as e:
        print("Warning: could not enable TF determinism:", repr(e))

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Warning: could not enable mixed precision:", repr(e))
    else:
        print("Using full precision")
else:

    class _DummyStrategy:
        def scope(self):
            from contextlib import nullcontext

            return nullcontext()

    strategy = _DummyStrategy()
    VER = 1
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _safe_load_dict_npy(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


if not NEEDTRAIN:
    if "spe" in DATATYPE:
        spectrograms = (
            _safe_load_dict_npy("/kaggle/input/brain-spectrograms/specs.npy") or {}
        )
        if len(spectrograms) > 0:
            print("Loaded train spectrogram dict:", len(spectrograms))
    if "eeg" in DATATYPE:
        eegs = _safe_load_dict_npy("/kaggle/input/brain-eegs/eegs.npy") or {}
        if len(eegs) > 0:
            print("Loaded train eeg dict:", len(eegs))
    if "img" in DATATYPE:
        imgs = _safe_load_dict_npy("/kaggle/input/brain-imgs/imgs.npy") or {}
        if len(imgs) > 0:
            print("Loaded train img dict:", len(imgs))
    if "stft" in DATATYPE:
        stfts = _safe_load_dict_npy("/kaggle/input/brain-stfts/stfts.npy") or {}
        if len(stfts) > 0:
            print("Loaded train stft dict:", len(stfts))



## === cell 2
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

            self.data = data.reset_index(drop=True)
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
                x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")

            n_targets = len(self.targets) if self.targets is not None else 6
            y = np.zeros((len(indexes), n_targets), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                row_spe = row
                row_eeg = row
                row_img = row
                row_stft = row

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    r_spe = round(row_spe.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row_eeg.eeg_label_offset_seconds * SFREQ)

                for k in range(4):
                    if "spe" in DATATYPE:
                        spe = self.specs[row_spe.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T

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
                        eeg = self.eegs[row_eeg.eeg_id][
                            :, r_eeg : r_eeg + round(50 * SFREQ), k
                        ]

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
                        img = self.imgs[row_img.eeg_id][:, :, k, :]
                        x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        stft = self.stfts[row_stft.eeg_id][:, :, :, k]

                        if (stft.shape[1] != 64) or (stft.shape[2] != 256):
                            stft2 = np.zeros((4, 64, 128))
                            stft2[:, : stft.shape[1], : stft.shape[2]] = stft
                            stft = stft2

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
                    label = 0
                    if "spe" in DATATYPE:
                        label = label + row_spe[self.targets].values
                    if "eeg" in DATATYPE:
                        label = label + row_spe[self.targets].values
                    if "img" in DATATYPE:
                        label = label + row_img[self.targets].values
                    if "stft" in DATATYPE:
                        label = label + row_stft[self.targets].values
                    label = label / len(DATATYPE)

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
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img
                x.append(x_img)
            if "stft" in DATATYPE:
                x.append(x_stft)

            return x, y

    def build_model(TARGETS_PRETRAIN):
        l2_layer = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )

        inp = []
        y = None

        effnet_weights = "imagenet"

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_spe[:, :, :, :, 0],
                    inp_spe[:, :, :, :, 1],
                    inp_spe[:, :, :, :, 2],
                    inp_spe[:, :, :, :, 3],
                ]
            )
            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=effnet_weights,
                input_shape=None,
                name="efficientnetb0_spe",
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = l2_layer(x_spe)
            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_eeg[:, :, :, :, 0],
                    inp_eeg[:, :, :, :, 1],
                    inp_eeg[:, :, :, :, 2],
                    inp_eeg[:, :, :, :, 3],
                ]
            )
            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=effnet_weights,
                input_shape=None,
                name="efficientnetb0_eeg",
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2_layer(x_eeg)
            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_img[:, :, :, :, 0],
                    inp_img[:, :, :, :, 1],
                    inp_img[:, :, :, :, 2],
                    inp_img[:, :, :, :, 3],
                ]
            )
            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=effnet_weights,
                input_shape=None,
                name="efficientnetb0_img",
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = l2_layer(x_img)
            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4))
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_stft[:, :, :, :, 0],
                    inp_stft[:, :, :, :, 1],
                    inp_stft[:, :, :, :, 2],
                    inp_stft[:, :, :, :, 3],
                ]
            )
            base_model_stft = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=effnet_weights,
                input_shape=None,
                name="efficientnetb0_stft",
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = l2_layer(x_stft)
            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                if y is not None
                else x_stft
            )

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN), activation="softmax", dtype="float32"
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

else:
    DataGenerator = None
    build_model = None



## === cell 3
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

print("Test shape", test.shape, "Sample submission shape", sample_sub.shape)

sample_eeg_ids = sample_sub["eeg_id"].values
test_eeg_ids = test["eeg_id"].values
print("Unique eeg_id in test:", len(pd.unique(test_eeg_ids)))
print("Unique eeg_id in sample_submission:", len(pd.unique(sample_eeg_ids)))


def _try_load_test_dicts():
    out = {}
    candidates = [
        "/kaggle/input/brain-test-eegs/eegs.npy",
        "/kaggle/input/brain-test-specs/specs.npy",
        "/kaggle/input/brain-test-imgs/imgs.npy",
        "/kaggle/input/brain-test-stfts/stfts.npy",
    ]
    for p in candidates:
        if os.path.exists(p):
            out[p] = np.load(p, allow_pickle=True).item()
    return out


loaded_any_test_cache = False
test_cache = _try_load_test_dicts()

if "spe" in DATATYPE:
    if any("brain-test-specs" in k for k in test_cache.keys()):
        spectrograms2 = [v for k, v in test_cache.items() if "brain-test-specs" in k][0]
        loaded_any_test_cache = True
        print("Loaded cached test spectrogram dict:", len(spectrograms2))
    else:
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

if ("eeg" in DATATYPE) or ("img" in DATATYPE) or ("stft" in DATATYPE):
    if any("brain-test-eegs" in k for k in test_cache.keys()) and ("eeg" in DATATYPE):
        eegs2 = [v for k, v in test_cache.items() if "brain-test-eegs" in k][0]
        loaded_any_test_cache = True
        print("Loaded cached test eeg dict:", len(eegs2))
    if any("brain-test-imgs" in k for k in test_cache.keys()) and ("img" in DATATYPE):
        imgs2 = [v for k, v in test_cache.items() if "brain-test-imgs" in k][0]
        loaded_any_test_cache = True
        print("Loaded cached test img dict:", len(imgs2))
    if any("brain-test-stfts" in k for k in test_cache.keys()) and ("stft" in DATATYPE):
        stfts2 = [v for k, v in test_cache.items() if "brain-test-stfts" in k][0]
        loaded_any_test_cache = True
        print("Loaded cached test stft dict:", len(stfts2))

    if not loaded_any_test_cache:
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

        test_eeg_ids_set = set(test.eeg_id.values.tolist())
        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids_set:
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

                if "stft" in DATATYPE:
                    frequencies, times, Sxx = signal.spectrogram(
                        eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        SFREQ,
                        nperseg=256,
                        noverlap=219,
                        nfft=320,
                    )
                    valid_freq = (frequencies > 0.0) & (frequencies <= 20)
                    Sxx_filtered = Sxx[:, valid_freq, :-1]
                    Sxx_filtered = np.reshape(
                        Sxx_filtered,
                        (
                            Sxx_filtered.shape[0],
                            Sxx_filtered.shape[1],
                            Sxx_filtered.shape[2],
                            1,
                        ),
                    )
                    list_stft.append(Sxx_filtered)

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)

            if "stft" in DATATYPE:
                list_stft = np.concatenate(list_stft, -1)
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

                img = (
                    np.array(
                        tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                    if TF_AVAILABLE
                    else (
                        np.array(
                            Image.fromarray(img.astype(np.uint8)).resize(
                                (IMG_WIDE, IMG_HIGH * 4), resample=Image.BILINEAR
                            ),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
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
                img2 = (
                    np.array(
                        tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                    if TF_AVAILABLE
                    else (
                        np.array(
                            Image.fromarray(img2.astype(np.uint8)).resize(
                                (IMG_WIDE, IMG_HIGH * 4), resample=Image.BILINEAR
                            ),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
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
                img3 = (
                    np.array(
                        tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                    if TF_AVAILABLE
                    else (
                        np.array(
                            Image.fromarray(img3.astype(np.uint8)).resize(
                                (IMG_WIDE, IMG_HIGH * 4), resample=Image.BILINEAR
                            ),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
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




## === cell 4
def _write_fallback_submission(sample_sub_df, targets, out_path="submission.csv"):
    sub = sample_sub_df.copy()
    for c in targets:
        sub[c] = 1.0 / len(targets)
    sub = sub[["eeg_id"] + list(targets)]
    sub.to_csv(out_path, index=False)
    print("Wrote fallback submission:", out_path, "shape", sub.shape)
    print(
        "Row-sum check (min/max):",
        sub[targets].sum(axis=1).min(),
        sub[targets].sum(axis=1).max(),
    )
    print(sub.head())
    return sub


required_ok = True
if "spe" in DATATYPE and (
    not isinstance(spectrograms2, dict) or len(spectrograms2) == 0
):
    required_ok = False
if "eeg" in DATATYPE and (not isinstance(eegs2, dict) or len(eegs2) == 0):
    required_ok = False
if "img" in DATATYPE and (not isinstance(imgs2, dict) or len(imgs2) == 0):
    required_ok = False
if "stft" in DATATYPE and (not isinstance(stfts2, dict) or len(stfts2) == 0):
    required_ok = False

weights_missing = False
for i in range(NSPLIT):
    wpath = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
    if not os.path.exists(wpath):
        print("Missing model weights:", wpath)
        weights_missing = True

if (not TF_AVAILABLE) or (not required_ok) or weights_missing:
    print(
        "Warning: TF unavailable and/or missing required test feature dict(s) and/or model weights. Writing safe fallback submission."
    )
    _ = _write_fallback_submission(sample_sub, list(TARGETS), out_path="submission.csv")
else:
    preds = []

    test_for_pred = pd.DataFrame({"eeg_id": sample_sub["eeg_id"].values})
    test_for_pred = test_for_pred.merge(test, on="eeg_id", how="left")

    with strategy.scope():
        model = build_model(TARGETS)

    test_gen = DataGenerator(
        test_for_pred,
        shuffle=False,
        batch_size=BATCHSIZE * 2,
        mode="test",
        specs=spectrograms2 if "spe" in DATATYPE else None,
        eegs=eegs2 if "eeg" in DATATYPE else None,
        imgs=imgs2 if "img" in DATATYPE else None,
        stfts=stfts2 if "stft" in DATATYPE else None,
        targets=TARGETS,
    )

    for i in range(NSPLIT):
        print(f"Fold {i+1}")
        wpath = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
        model.load_weights(wpath)
        pred_i = model.predict(test_gen, verbose=1)
        preds.append(pred_i)

    pred = np.mean(preds, axis=0)
    print("Test preds shape", pred.shape)

    pred = np.nan_to_num(
        pred,
        nan=1.0 / len(TARGETS),
        posinf=1.0 / len(TARGETS),
        neginf=1.0 / len(TARGETS),
    )
    pred = np.clip(pred, 1e-7, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = sample_sub[["eeg_id"]].copy()
    pred_df = pd.DataFrame(pred, columns=list(TARGETS))
    pred_df["eeg_id"] = test_for_pred["eeg_id"].values
    sub = sub.merge(pred_df, on="eeg_id", how="left")

    for c in TARGETS:
        sub[c] = (
            pd.to_numeric(sub[c], errors="coerce")
            .fillna(1.0 / len(TARGETS))
            .astype(np.float64)
        )

    rs = sub[list(TARGETS)].sum(axis=1).values.reshape(-1, 1)
    rs[rs == 0] = 1.0
    sub[list(TARGETS)] = (sub[list(TARGETS)].values / rs).astype(np.float32)

    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)

    print("Submission shape", sub.shape)
    print(
        "Row-sum check (min/max):",
        sub[list(TARGETS)].sum(axis=1).min(),
        sub[list(TARGETS)].sum(axis=1).max(),
    )
    print(sub.head())
