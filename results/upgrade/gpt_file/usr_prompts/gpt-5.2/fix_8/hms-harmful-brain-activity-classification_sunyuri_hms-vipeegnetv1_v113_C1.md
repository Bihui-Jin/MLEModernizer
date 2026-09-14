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

0.4305725754210282

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the model build crash caused by calling `tf.nn.l2_normalize` directly on a KerasTensor (Keras 3 disallows raw TF ops in the Functional graph) by replacing it with an equivalent Keras Layer (`Lambda`) so the architecture/semantics stay the same. I also remove the unnecessary pip installs that can fail/offline in Kaggle and make the run brittle, while keeping the EfficientNet fallback logic intact. Finally, I keep the existing inference/submission logic but ensure the code always reaches `submission.csv` creation with valid probabilities that sum to 1. This should run end-to-end and yield a valid submission (and if weights exist, it use them; otherwise it writes uniform probabilities).'
- What this solution (achieved 1.40995) has done: 'I fix the runtime crash happening at import time (the `MessageFactory.GetPrototype` error), which is caused by an incompatible `protobuf` runtime being pulled in when importing `librosa` (and also `albumentations`, which isn’t used). To keep core modeling logic intact, I remove/guard those unused imports so TensorFlow and the rest of the pipeline can run end-to-end. Then I ensure `LOAD_MODELS_FROM` points to an existing directory (either the intended Kaggle dataset path or a local fallback) so your trained weights actually load; this should materially improve the KL score versus the uniform fallback. Finally, I keep the same prediction and submission formatting but make it robust to missing weights and guarantee valid row-wise probability sums.'
- What this solution (achieved 1.40995) has done: 'I remove the import-time crash (`MessageFactory.GetPrototype`) by eliminating the unused `matplotlib` and `scipy` dependencies from the inference pipeline, since they are what typically trigger the protobuf/lib mismatch in Kaggle’s Py3.12 images. To keep your core logic intact (same model, same inputs/outputs, same weight-loading/inference flow), I switch the test “img” feature creation to a lightweight NumPy-based renderer (no plotting) that produces the same `(64,256,4)` shape expected by your model. I also ensure the code reads `train.csv` safely without triggering the problematic imports, and always writes a valid `submission.csv` with the correct columns and row-wise probability sums. This should restore weight loading (when present) and materially reduce KL from the uniform fallback, moving the score toward your target.'
- What this solution (achieved 1.40995) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by forcing Python to use the pure-Python protobuf implementation before TensorFlow loads, which avoids the common Kaggle Py3.12 protobuf binary incompatibility. I also add a small safe fallback so the script can still run even if TensorFlow cannot import (it then write a valid uniform-probability submission), ensuring you always get a `.csv` submission file. To improve score toward your target, I additionally make weight-loading more robust by searching common Kaggle input/working locations for the fold `.h5` files (without changing the model or inference semantics), so if the weights exist they actually be used instead of the uniform fallback. All other logic (model, generator, feature creation, averaging folds, normalization) is preserved.'
- What this solution (achieved 1.40995) has done: 'We fix the import-time `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by explicitly forcing the pure-Python protobuf implementation *and* importing `google.protobuf` before TensorFlow, which avoids the binary protobuf incompatibility seen on Kaggle Py3.12. We also make the script robust if that still fails by cleanly falling back to a valid uniform-probability submission (score-neutral vs your current fallback behavior), ensuring a `.csv` is always produced. To improve score toward your target (lower is better), we keep your exact model/inference logic but make weight discovery slightly more robust by additionally scanning `/kaggle/input/**` for matching `f*_stage*.h5` files when `LOAD_MODELS_FROM` doesn’t exist, so your trained weights are more likely to be loaded instead of using uniform predictions. All model architecture, preprocessing, generator behavior, and prediction post-processing semantics remain unchanged.'

# 9. Code solution

## === cell 0
import os, sys, io

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024031101"
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

AMP = 200

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

import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401
except Exception as e:
    print("Warning: google.protobuf pre-import failed:", repr(e))

TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        "WARNING: TensorFlow failed to import; will fallback to uniform submission. Reason:",
        repr(e),
    )

from PIL import Image  # kept as in original; harmless

if TF_AVAILABLE:
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
    except Exception as e:
        print("Warning: enable_op_determinism not available:", repr(e))

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Warning: could not enable mixed precision:", repr(e))
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
print(df.head(2))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if TF_AVAILABLE and NEEDTRAIN:
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
        df3 = df2.drop_duplicates("eeg_id")

        df2 = df2.drop(df3.index).reset_index(drop=True)
        df3 = df3.reset_index(drop=True)
        df2 = df2.iloc[np.random.permutation(len(df2))].reset_index(drop=True)

        num_all = list()
        for i in range(len(TARGETS)):
            num_all.append(sum(np.argmax(df3[TARGETS].values, 1) == i))

        train = pd.DataFrame()
        for i in range(len(TARGETS)):
            train_temp = df2.iloc[np.argmax(df2[TARGETS].values, 1) == i].reset_index(
                drop=True
            )

            train_temp_df3 = df3.iloc[
                np.argmax(df3[TARGETS].values, 1) == i
            ].reset_index(drop=True)
            train_temp_df3 = train_temp_df3[["eeg_id", "eeg_label_offset_seconds"]]
            train_temp_df3 = train_temp_df3.rename(
                {"eeg_label_offset_seconds": "seconds"}, axis=1
            )

            train_temp = train_temp.merge(train_temp_df3, how="left", on="eeg_id")
            train_temp = train_temp[
                ~(
                    abs(train_temp.eeg_label_offset_seconds - train_temp.seconds)
                    < (EEG_LENGTH / 3 * 2)
                )
            ]
            train_temp = train_temp.drop("seconds", axis=1)

            train_temp = train_temp.sort_values(
                "eeg_label_offset_seconds", ascending=True
            ).reset_index(drop=True)
            for ii in range(100):
                train_temp_temp = train_temp.drop_duplicates("eeg_id")
                train_temp = train_temp.drop(train_temp_temp.index).reset_index(
                    drop=True
                )
                train_temp_temp = train_temp_temp.reset_index(drop=True)

                train = pd.concat([train, train_temp_temp])

                train_temp_temp = train_temp_temp[
                    ["eeg_id", "eeg_label_offset_seconds"]
                ]
                train_temp_temp = train_temp_temp.rename(
                    {"eeg_label_offset_seconds": "seconds"}, axis=1
                )

                train_temp = train_temp.merge(
                    train_temp_temp, how="left", on="eeg_id"
                ).reset_index(drop=True)
                train_temp = train_temp[
                    ~(
                        abs(train_temp.eeg_label_offset_seconds - train_temp.seconds)
                        < (EEG_LENGTH / 3 * 2)
                    )
                ].reset_index(drop=True)
                train_temp = train_temp.drop("seconds", axis=1)

                if len(train_temp) == 0:
                    break
        train = pd.concat([train, df3]).reset_index(drop=True)
        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.head()
        train.to_csv("train.csv", index=False)

        for i in range(len(TARGETS)):
            print(
                f"{TARGETS[i]}, {sum(np.argmax(df3[TARGETS].values, 1) == i)} -> {sum(np.argmax(train[TARGETS].values, 1) == i)}"
            )

    else:
        train = pd.read_csv("train.csv")

        for i in range(len(TARGETS)):
            print(f"{TARGETS[i]}, {sum(np.argmax(train[TARGETS].values, 1) == i)}")



## === cell 2
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
        ):

            self.cmin = -4
            self.cmax = 6
            self.cmaps = None  # only needed for spe/stft; DATATYPE defaults to ["img"]
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
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row.eeg_label_offset_seconds * SFREQ)
                else:
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row.eeg_label_offset_seconds * SFREQ)

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
                    if "img" in DATATYPE:
                        if self.mode == "test":
                            img = self.imgs[row.eeg_id][:, :, k]
                        else:
                            img = self.imgs[row.sign_id][:, :, k]
                        x_img[j, :, :, k] = img

                if self.mode != "test":
                    label = row[TARGETS].values
                    if self.mode == "train" and sum(label == 1):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label == 1] = 1 - 5 * xx
                    y[j] = label

            alpha = 0

            x = list()
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

    try:
        import efficientnet.tfkeras as efn  # type: ignore

        _USING_EFN = True
        print("Using efficientnet.tfkeras")
    except Exception as e:
        efn = None
        _USING_EFN = False
        print(
            "efficientnet.tfkeras not available; using tf.keras.applications EfficientNet. Reason:",
            repr(e),
        )

    def _get_backbone(name: str, include_top=False, weights=None, input_shape=None):
        if _USING_EFN:
            return getattr(efn, name)(
                include_top=include_top, weights=weights, input_shape=input_shape
            )
        return getattr(tf.keras.applications, name)(
            include_top=include_top, weights=weights, input_shape=input_shape
        )

    def _l2norm_layer(axis=-1, name=None):
        return tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=axis), name=name
        )

    def build_model():
        inp = list()

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

            base_model_img = _get_backbone(
                "EfficientNetB4", include_top=False, weights=None, input_shape=None
            )
            base_model_img._name = "img_extractor"
            if NEEDTRAIN:
                if PLATFORM == "local":
                    base_model_img.load_weights(
                        "./input/tf-efficientnet-imagenet-weights/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_img.load_weights(
                        "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                    )

            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = _l2norm_layer(axis=-1, name="img_l2norm")(x_img)

            inp.append(inp_img)
            y = x_img

        y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 3
def _make_img_from_eeg(eeg_default: pd.DataFrame) -> np.ndarray:
    time_start = int(round((50 - EEG_LENGTH) / 2 * SFREQ))
    time_stop = int(round((50 + EEG_LENGTH) / 2 * SFREQ))
    out = np.zeros((IMG_HIGH, IMG_WIDE, 4), dtype=np.float32)

    for k, region in enumerate(BRAIN.keys()):
        eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(BRAIN[region]):
            a, b = chan.split("-")
            eeg[chan_i, :] = (eeg_default[a].values - eeg_default[b].values).astype(
                np.float32
            )
        eeg = np.nan_to_num(eeg, nan=0.0)

        if SFREQ != 200:
            old_n = eeg.shape[1]
            new_n = int(round(old_n * SFREQ / 200))
            x_old = np.linspace(0.0, 1.0, old_n, endpoint=False, dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, new_n, endpoint=False, dtype=np.float32)
            eeg_rs = np.empty((eeg.shape[0], new_n), dtype=np.float32)
            for ci in range(eeg.shape[0]):
                eeg_rs[ci] = np.interp(x_new, x_old, eeg[ci]).astype(np.float32)
            eeg = eeg_rs

        seg = eeg[:, time_start:time_stop]
        seg_mean = seg.mean(axis=0)
        seg_mean = np.nan_to_num(seg_mean, nan=0.0)

        old_n = seg_mean.shape[0]
        x_old = np.linspace(0.0, 1.0, old_n, endpoint=False, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, IMG_WIDE, endpoint=False, dtype=np.float32)
        line = np.interp(x_new, x_old, seg_mean).astype(np.float32)

        scale = np.percentile(np.abs(line), 95) + 1e-6
        line = np.clip(line / scale, -1.0, 1.0)

        rows = ((line + 1.0) * 0.5 * (IMG_HIGH - 1)).astype(np.int32)
        img = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
        cols = np.arange(IMG_WIDE)
        img[rows, cols] = 1.0
        img = np.maximum(img, np.roll(img, 1, axis=0))
        img = np.maximum(img, np.roll(img, -1, axis=0))

        out[:, :, k] = img

    out[:, :, 0] = -out[:, :, 0]
    out[:, :, 2] = -out[:, :, 2]
    return out


def _discover_weight_paths(load_dir: str, stage: int, nfolds: int = 5):
    candidate_weight_paths = []
    for i in range(nfolds):
        candidate_weight_paths.append(os.path.join(load_dir, f"f{i}_stage{stage}.h5"))

    if os.path.isdir(load_dir):
        try:
            for subdir in os.listdir(load_dir):
                subpath = os.path.join(load_dir, subdir)
                if os.path.isdir(subpath):
                    for i in range(nfolds):
                        candidate_weight_paths.append(
                            os.path.join(subpath, f"f{i}_stage{stage}.h5")
                        )
        except Exception as e:
            print("Warning: could not scan subdirectories for weights:", repr(e))

    if (not os.path.isdir(load_dir)) and PLATFORM == "kaggle":
        roots = ["/kaggle/input", "/kaggle/working"]
        for root in roots:
            try:
                if not os.path.isdir(root):
                    continue
                for d in os.listdir(root):
                    dpath = os.path.join(root, d)
                    if not os.path.isdir(dpath):
                        continue
                    for i in range(nfolds):
                        candidate_weight_paths.append(
                            os.path.join(dpath, f"f{i}_stage{stage}.h5")
                        )
                    try:
                        for subdir in os.listdir(dpath):
                            subpath = os.path.join(dpath, subdir)
                            if os.path.isdir(subpath):
                                for i in range(nfolds):
                                    candidate_weight_paths.append(
                                        os.path.join(subpath, f"f{i}_stage{stage}.h5")
                                    )
                    except Exception:
                        pass
            except Exception:
                pass

    seen = set()
    candidate_weight_paths = [
        p for p in candidate_weight_paths if not (p in seen or seen.add(p))
    ]
    return candidate_weight_paths


if PLATFORM == "kaggle":
    candidates = [
        LOAD_MODELS_FROM,
        f"/kaggle/input/{os.path.basename(LOAD_MODELS_FROM)}",
        "/kaggle/input/models2024031101",
        "/kaggle/working/models2024031101",
        "/kaggle/input/hms-harmful-brain-activity-classification/models2024031101",
    ]
    for cand in candidates:
        if isinstance(cand, str) and os.path.isdir(cand):
            if cand != LOAD_MODELS_FROM:
                print(f"LOAD_MODELS_FROM not found; using fallback dir: {cand}")
            LOAD_MODELS_FROM = cand
            break
print("LOAD_MODELS_FROM =", LOAD_MODELS_FROM)

if not TF_AVAILABLE:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )

    pred = np.ones((len(test), 6), dtype=np.float32) / 6.0
    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub = sub[["eeg_id"] + list(TARGETS)]
    sub.to_csv("submission.csv", index=False)
    print("TensorFlow unavailable; wrote uniform submission.csv", sub.shape)
else:
    if not NEEDTRAIN:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
        elif PLATFORM == "kaggle":
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = os.listdir(PATH2)
        print(f"\nThere are {len(files2)} test eeg parquets")

        eegs2 = {}
        imgs2 = {}
        stfts2 = {}

        test_eeg_ids = set(test.eeg_id.values.tolist())

        for i, f in enumerate(files2):
            if i % 200 == 0:
                print(i, ", ", end="")
            name = int(f.split(".")[0])
            if name not in test_eeg_ids:
                continue

            eeg_default = pd.read_parquet(f"{PATH2}{f}")

            if "img" in DATATYPE:
                imgs2[name] = _make_img_from_eeg(eeg_default)

        model = build_model()
        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            imgs=imgs2,
            stfts=stfts2,
        )

        preds = []
        weights_found = False

        candidate_weight_paths = _discover_weight_paths(LOAD_MODELS_FROM, STAGETEST, 5)

        for i in range(5):
            preferred = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            wpath = None
            if os.path.exists(preferred):
                wpath = preferred
            else:
                for p in candidate_weight_paths:
                    if p.endswith(f"f{i}_stage{STAGETEST}.h5") and os.path.exists(p):
                        wpath = p
                        break

            if wpath is not None:
                weights_found = True
                print(f"\nFold {i+1} loading weights:", wpath)
                model.load_weights(wpath)
                pred = model.predict(test_gen, verbose=1)
                preds.append(pred)
            else:
                print(
                    f"\nFold {i+1} weights missing for pattern f{i}_stage{STAGETEST}.h5 under:",
                    LOAD_MODELS_FROM,
                )

        if weights_found:
            pred = np.mean(preds, axis=0)
        else:
            print(
                "No weights found; writing uniform probabilities submission to avoid invalid/no-submission."
            )
            pred = np.ones((len(test), 6), dtype=np.float32) / 6.0

        print("\nTest preds shape", pred.shape)

        pred = np.clip(pred, 1e-9, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred

        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)

        print("Submission shape", sub.shape)
        rs = sub.iloc[:, -6:].sum(axis=1)
        print("Row-sum stats:", float(rs.min()), float(rs.max()))
        print(sub.head())
