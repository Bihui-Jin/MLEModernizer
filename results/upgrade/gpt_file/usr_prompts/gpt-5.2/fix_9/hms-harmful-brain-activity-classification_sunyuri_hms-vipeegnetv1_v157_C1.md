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

0.3380172047363531

# 6. Current score

0.75645

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime. I also fix the missing model-weights issue by falling back to a stable, valid baseline submission (the class prior from `train.csv`) when the expected `/kaggle/input/models2024040302` directory is not available, ensuring the notebook always produces `submission.csv`. Additionally, I make sure the submission columns exactly match `sample_submission.csv` and that each row sums to 1 with safe clipping/renormalization. These changes are minimal, keep your existing model/inference logic intact when weights exist, and guarantee an end-to-end run with a valid CSV.'
- What this solution (achieved 1.15381) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf runtime *and* ensuring TensorFlow is imported only after that setting (plus a safe fallback to the Python implementation if needed). Then I fix the hidden logic bug in `DataGenerator` where `x_eeg2` is allocated but never filled (leading to invalid normalization behavior) by populating it from the raw 50s EEG window as intended. Finally, I keep your existing inference logic intact but make the model-weights fallback smarter than a pure class prior by using per-`patient_id` priors (falls back to global prior when unseen), which should legitimately reduce KL and move score toward your ~0.338 target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.75645) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation earlier and also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before *any* TensorFlow-related import, plus adding a safe fallback to run without TF by producing a calibrated prior-based submission if TF still fails. I keep your model/inference logic unchanged when TensorFlow imports successfully and when model weights exist. To move the score toward the target (lower is better) with minimal change, I improve the fallback (used when TF/models aren’t available) from a simple patient prior to a smoothed blend of patient prior + global prior, which is generally more robust and should reduce KL. Finally, I ensure the output CSV strictly matches `sample_submission.csv` columns and each row sums to 1 with clipping and renormalization.'
- What this solution (achieved 0.75645) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before any protobuf/TensorFlow import* and also forcing the Python implementation via the `google.protobuf` API when available (this directly addresses the `MessageFactory.GetPrototype` AttributeError). I also ensure the script doesn’t fail hard if TensorFlow imports but model weights are missing: it fall back to the existing smoothed patient/global prior submission so a valid `submission.csv` is always produced. Finally, I keep your model/data logic unchanged, only adding a small safety renormalization when writing the model-based submission to guarantee every row sums to 1 and is KL-safe.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf.internal import api_implementation as _pb_api_impl

    try:
        _pb_api_impl._SetType("python")  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    pass

import io
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_AVAILABLE = True
tf_import_error = None
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)

PLATFORM = "kaggle"  # local / kaggle
NEEDTRAIN = False

DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3

LOAD_MODELS_FROM = "models2024040302"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

threshold = 0.2

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 128

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if TF_AVAILABLE:
    print("TensorFlow version =", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism setting not available:", repr(e))

    MIX = True
    if MIX:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled via keras mixed_precision policy")
        except Exception as e:
            print("Mixed precision not enabled (fallback to full precision):", repr(e))
    else:
        print("Using full precision")
else:
    print("WARNING: TensorFlow import failed; will use fallback submission only.")
    print("TF import error:", tf_import_error)
    strategy = None  # not used without TF

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
if TF_AVAILABLE:
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
            self.specs = specs if specs is not None else {}
            self.eegs = eegs if eegs is not None else {}
            self.imgs = imgs if imgs is not None else {}
            self.stfts = stfts if stfts is not None else {}
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
                x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")
            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

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

                        x_eeg2[j, :, :, k] = eeg

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

            if "eeg" in DATATYPE:
                for i_eeg in range(x_eeg2.shape[0]):
                    xx = np.std(x_eeg2[i_eeg, :, :, :], 1, keepdims=True)
                    xx = np.mean(xx)
                    x_eeg2[i_eeg, :, :, :] = (
                        x_eeg2[i_eeg, :, :, :]
                        - np.mean(x_eeg2[i_eeg, :, :, :], 1, keepdims=True)
                    ) / (xx + 1e-6)

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




## === cell 2
if TF_AVAILABLE:

    def build_model(TARGETS_PRETRAIN):
        l2norm = tf.keras.layers.UnitNormalization(axis=-1, name="l2_norm")

        inp = []
        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe = tf.keras.layers.Concatenate(axis=1, name="spe_concat")(
                [inp_spe[:, :, :, :, i] for i in range(4)]
            )
            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_spe",
            )
            base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D(name="spe_gap")(x_spe)
            x_spe = l2norm(x_spe)
            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4))
            x_eeg = tf.keras.layers.Concatenate(axis=1, name="eeg_concat")(
                [inp_eeg[:, :, :, :, i] for i in range(4)]
            )
            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_eeg",
            )
            base_model_eeg._name = "eeg_extractor"
            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="eeg_gap")(x_eeg)
            x_eeg = l2norm(x_eeg)
            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1, name="spe_eeg_fuse")([y, x_eeg])
                if ("spe" in DATATYPE)
                else x_eeg
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4))
            x_img = tf.keras.layers.Concatenate(axis=1, name="img_concat")(
                [inp_img[:, :, :, :, i] for i in range(4)]
            )
            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_img",
            )
            base_model_img._name = "img_extractor"
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D(name="img_gap")(x_img)
            x_img = l2norm(x_img)
            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1, name="fuse_with_img")([y, x_img])
                if (("spe" in DATATYPE) or ("eeg" in DATATYPE))
                else x_img
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4))
            x_stft = tf.keras.layers.Concatenate(axis=1, name="stft_concat")(
                [inp_stft[:, :, :, :, i] for i in range(4)]
            )
            base_model_stft = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="efficientnetb0_stft",
            )
            base_model_stft._name = "stft_extractor"
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D(name="stft_gap")(x_stft)
            x_stft = l2norm(x_stft)
            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1, name="fuse_with_stft")([y, x_stft])
                if (("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE))
                else x_stft
            )

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN),
            activation="softmax",
            dtype="float32",
            name="head_softmax",
        )(y)
        return tf.keras.Model(inputs=inp, outputs=y)




## === cell 3
def _make_fallback_submission(
    train_df: pd.DataFrame, test_df: pd.DataFrame, sample_sub: pd.DataFrame, targets
):
    vote = train_df[list(targets)].values.astype("float64")
    vote = np.clip(vote, 0.0, None)
    vote_sum = vote.sum(axis=1, keepdims=True)
    vote_sum[vote_sum == 0] = 1.0
    prob = vote / vote_sum

    df_prob = pd.DataFrame(prob, columns=targets)
    df_prob["patient_id"] = train_df["patient_id"].values

    patient_prior = df_prob.groupby("patient_id")[list(targets)].mean()
    global_prior = df_prob[list(targets)].mean().values.astype("float64")
    global_prior = np.clip(global_prior, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    patient_counts = train_df.groupby("patient_id").size()

    sub = sample_sub.copy()
    prob_cols = [c for c in sample_sub.columns if c != "eeg_id"]

    preds = np.zeros((len(test_df), len(targets)), dtype="float64")
    for i, pid in enumerate(test_df["patient_id"].values):
        if pid in patient_prior.index:
            p_pat = patient_prior.loc[pid, list(targets)].values.astype("float64")
            p_pat = np.clip(p_pat, 1e-12, None)
            p_pat = p_pat / p_pat.sum()
            n = float(patient_counts.loc[pid]) if pid in patient_counts.index else 1.0
            alpha = n / (n + 20.0)
            p = alpha * p_pat + (1.0 - alpha) * global_prior
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
        else:
            p = global_prior
        preds[i] = p

    for c in prob_cols:
        if c in targets:
            sub[c] = preds[:, list(targets).index(c)].astype("float32")
        else:
            sub[c] = np.float32(1.0 / 6.0)

    arr = sub[prob_cols].values.astype("float64")
    arr = np.clip(arr, 1e-8, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    sub[prob_cols] = arr.astype("float32")
    return sub




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

    if not TF_AVAILABLE:
        sub = _make_fallback_submission(df, test, sample_sub, TARGETS)
        sub.to_csv("submission.csv", index=False)
        print("Saved: submission.csv (fallback; TF unavailable)")
        print("Submission shape", sub.shape)
    else:
        candidate_model_dirs = [
            LOAD_MODELS_FROM,
            os.path.join("/kaggle/input", os.path.basename(LOAD_MODELS_FROM)),
            os.path.join("/kaggle/working", os.path.basename(LOAD_MODELS_FROM)),
        ]
        model_dir = None
        for d in candidate_model_dirs:
            if d and os.path.isdir(d):
                model_dir = d
                break

        if model_dir is None:
            print(
                "WARNING: Could not find model directory. Falling back to smoothed patient-prior submission. "
                f"Tried: {candidate_model_dirs}"
            )
            sub = _make_fallback_submission(df, test, sample_sub, TARGETS)
            sub.to_csv("submission.csv", index=False)
            print("Saved: submission.csv (fallback; no model dir)")
            print("Submission shape", sub.shape)
        else:
            print("Using model_dir:", model_dir)

            if "spe" in DATATYPE:
                if PLATFORM == "local":
                    PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
                elif PLATFORM == "kaggle":
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
            elif PLATFORM == "kaggle":
                PATH2 = (
                    "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
                )

            files2 = os.listdir(PATH2)
            print(f"There are {len(files2)} test eeg parquets")

            eegs2 = {}
            imgs2 = {}
            stfts2 = {}
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
            test_eeg_ids = set(test.eeg_id.values.tolist())

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

                    if "stft" in DATATYPE:
                        frequencies, times, Sxx = signal.spectrogram(
                            eeg[
                                :,
                                round(time_temp * SFREQ) : round(
                                    (time_temp + 50) * SFREQ
                                ),
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
                    byte_stream.truncate(0)
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
                        plt.plot(
                            eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5
                        )
                    plt.xlim(-5, eeg_all_region2.shape[1] + 5)
                    plt.ylim(-amp / 2, eeg_all_region2.shape[0] * amp + amp / 2 * 5)
                    plt.axis("off")

                    byte_stream = io.BytesIO()
                    plt.savefig(byte_stream, format="png", bbox_inches="tight")
                    byte_stream.seek(0)
                    img2 = Image.open(byte_stream)
                    img2 = np.array(img2)[:, :, :1]
                    byte_stream.truncate(0)
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
                        plt.plot(
                            eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5
                        )
                    plt.xlim(-2, eeg_all_region3.shape[1] + 2)
                    plt.ylim(-amp / 2, eeg_all_region3.shape[0] * amp + amp / 2 * 5)
                    plt.axis("off")

                    byte_stream = io.BytesIO()
                    plt.savefig(byte_stream, format="png", bbox_inches="tight")
                    byte_stream.seek(0)
                    img3 = Image.open(byte_stream)
                    img3 = np.array(img3)[:, :, :1]
                    byte_stream.truncate(0)
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

            with strategy.scope():
                model = build_model(TARGETS)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-3),
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

            preds = []
            missing_weights = False
            for i in range(NSPLIT):
                print(f"Fold {i+1}")
                wpath = os.path.join(model_dir, f"f{i}_stage{STAGETEST}.h5")
                if not os.path.exists(wpath):
                    print(f"WARNING: Missing weights: {wpath}")
                    missing_weights = True
                    break
                model.load_weights(wpath)
                pred_i = model.predict(test_gen, verbose=1)
                preds.append(pred_i)

            if missing_weights or len(preds) == 0:
                print(
                    "Falling back to smoothed patient-prior submission (missing weights)."
                )
                sub = _make_fallback_submission(df, test, sample_sub, TARGETS)
                sub.to_csv("submission.csv", index=False)
                print("Saved: submission.csv (fallback; missing weights)")
                print("Submission shape", sub.shape)
            else:
                pred = np.mean(preds, axis=0)
                print("Test preds shape", pred.shape)

                pred = np.clip(pred, 1e-8, 1.0)
                pred = pred / pred.sum(axis=1, keepdims=True)

                sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
                for c in TARGETS:
                    sub[c] = pred[:, list(TARGETS).index(c)]

                sub = sub[["eeg_id"] + [c for c in sample_sub.columns if c != "eeg_id"]]

                arr = sub[sub.columns[1:]].values.astype("float64")
                arr = np.clip(arr, 1e-8, 1.0)
                arr = arr / arr.sum(axis=1, keepdims=True)
                sub[sub.columns[1:]] = arr.astype("float32")

                sub.to_csv("submission.csv", index=False)
                print("Submission shape", sub.shape)
                print(
                    "Row-sum stats:",
                    float(sub[sub.columns[1:]].sum(axis=1).min()),
                    float(sub[sub.columns[1:]].sum(axis=1).max()),
                )
                print("Saved: submission.csv")
