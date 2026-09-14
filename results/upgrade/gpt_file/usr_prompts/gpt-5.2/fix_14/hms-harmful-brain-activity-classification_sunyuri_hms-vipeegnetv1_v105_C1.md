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

0.35366687509036

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` AttributeError seen on Python 3.12. Then I fix the missing weights issue by adding a safe fallback: if the external `models2024030401` Kaggle dataset is not mounted, the script still run end-to-end and write a valid `submission.csv` using a calibrated uniform distribution (score won’t be good, but it yield a valid submission). If weights are present, the original ensemble prediction path is unchanged. I also make the GPU strategy selection robust when no GPU is available.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash on Python 3.12 by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` via `os.environ[...]` (not `setdefault`) to ensure it actually takes effect in Kaggle. Then, to move your score down toward the target (lower-is-better) without changing the model architecture/training, I replace the uniform fallback with a label-prior fallback computed from `train.csv` (mean normalized vote distribution), which is a legitimate calibration baseline for KL divergence and should substantially improve over 1/6 each class. Finally, I keep the weights/ensemble path unchanged when weights exist, and ensure the submission is valid (non-NaN, clipped, row-normalized, correct column order, `.csv` output).'
- What this solution (achieved 1.39779) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* importing protobuf once before TensorFlow is imported, which is the known safe ordering on Kaggle Python 3.12. Then I keep your existing ensemble/weights logic unchanged, but add a minimal, score-improving (lower-is-better) safety calibration: after averaging fold predictions, blend a small amount of the train prior to reduce overconfident zeros that hurt KL divergence. Finally, I keep the submission-writing logic intact while ensuring probabilities are finite, clipped, and row-normalized so the `.csv` is always valid.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation even earlier and by avoiding importing `google.protobuf` before TensorFlow (that import ordering can still trigger the `MessageFactory.GetPrototype` issue on Py3.12). Then I keep your ensemble logic unchanged, but make the post-ensemble calibration slightly stronger and more stable for KL divergence by blending a bit more of the train-set class prior (and enforcing strict clipping/renormalization), which should move the score down toward the target without changing architecture/training. Finally, I keep all paths and the submission format intact, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash on Python 3.12 by ensuring the pure-Python protobuf implementation is enforced before TensorFlow import, and by importing protobuf once early in a safe order. Then I keep your model/ensemble logic intact but strengthen the post-ensemble probability calibration slightly (still just a prior blend + clipping/renorm) to reduce KL penalties from overconfident predictions, which should move the score down toward your target. Finally, I make the weights discovery more robust (directory existence + glob) and keep the guaranteed-valid submission writing (finite, clipped, row-normalized, correct column order, `submission.csv`).'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf implementation *and* preventing any `google.protobuf` import before TensorFlow loads (that import ordering is what triggers `MessageFactory.GetPrototype` on Py3.12). Then I keep your ensemble/model path unchanged, but adjust only the post-ensemble calibration for KL divergence: increase the prior-blend slightly (still just a convex blend + clip/renorm) to reduce overconfident predictions and move the score down toward your target. Finally, I make submission generation robust (finite probs, clipped, row-normalized, correct column order) so it always writes a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash on Python 3.12 by enforcing the pure-Python protobuf implementation and importing protobuf before any TensorFlow import (the current ordering still triggers `MessageFactory.GetPrototype` in this environment). Then I keep your ensemble/model logic unchanged but slightly adjust the post-ensemble prior-blending strength (a calibration-only change) to move the KL score down toward your target, since your current score is much worse (higher) than desired. Finally, I keep the same I/O paths and ensure the submission probabilities are finite, clipped, row-normalized, and written to `submission.csv` with the exact required columns.'
- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash on Python 3.12 by enforcing the pure-Python protobuf implementation and (crucially) avoiding any `google.protobuf` import before TensorFlow is imported, since that import ordering is what triggers the `MessageFactory.GetPrototype` failure. I keep your model/ensemble logic intact, but slightly strengthen the post-ensemble prior blending (calibration only) to reduce KL divergence from overconfident predictions—this should move the score down toward your target because lower is better and the current score is far worse than desired. I also make the weights discovery/prediction path more robust (skip missing folds instead of hard-failing) while still producing a valid `submission.csv` with finite, clipped, row-normalized probabilities in the correct column order.'
- What this solution (achieved 1.39779) has done: 'I fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow import* and by ensuring we don’t import `google.protobuf` ahead of TensorFlow (this import order is what triggers the `MessageFactory.GetPrototype` error on Py3.12). Then, to move your KL score downward toward the target while keeping the same model/ensemble logic, I reduce the overly-strong prior blending (currently `alpha=0.90`, which tends to collapse predictions toward the prior and can score poorly) to a mild calibration blend. Finally, I keep the submission writing path unchanged but add a small safety floor/renorm to guarantee valid probabilities and avoid KL blow-ups from zeros/NaNs.'

# 9. Code solution

## === cell 0
import os
import io

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from PIL import Image

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024030401"
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
AMP = 150

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
    print("Determinism setting not available:", repr(e))

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision option not available:", repr(e))
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
if NEEDTRAIN:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]

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
        df3 = df2.drop_duplicates("eeg_id").reset_index(drop=True)

        num_all = 0
        for i in range(len(TARGETS)):
            num_all = max(num_all, sum(np.argmax(df3[TARGETS].values, 1) == i))

        train = pd.DataFrame()
        for i in range(len(TARGETS)):
            train_temp = df2.iloc[np.argmax(df2[TARGETS].values, 1) == i].reset_index(
                drop=True
            )
            train_temp = train_temp.iloc[
                np.random.permutation(len(train_temp))
            ].reset_index(drop=True)
            ii = 1
            while len(train_temp.groupby("eeg_id").head(ii)) < num_all:
                ii = ii + 1
            print(ii)
            if ii > 1:
                if len(train_temp.groupby("eeg_id").head(ii)) > num_all:
                    train_temp1 = train_temp.groupby("eeg_id").head(ii - 1)
                    train_temp2 = train_temp.groupby("eeg_id").head(ii)
                    train_temp2 = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.drop_duplicates(keep=False).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.iloc[
                        np.random.permutation(len(train_temp2))
                    ].reset_index(drop=True)
                    train_temp2 = train_temp2[
                        : (num_all - len(train_temp.groupby("eeg_id").head(ii - 1)))
                    ]
                    train_temp = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
            else:
                train_temp = train_temp.groupby("eeg_id").head(ii)
            print(len(train_temp))
            train = pd.concat([train, train_temp]).reset_index(drop=True)

        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")




## === cell 2
def _make_backbone(name: str):
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # was "imagenet"
        input_shape=None,
        name=name,
    )
    return backbone


def build_model():
    inp = []
    y = None

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe = tf.keras.layers.Concatenate(axis=1, name="cat_spe_regions")(
            [
                inp_spe[:, :, :, :, 0],
                inp_spe[:, :, :, :, 1],
                inp_spe[:, :, :, :, 2],
                inp_spe[:, :, :, :, 3],
            ]
        )
        base_model_spe = _make_backbone("spe_extractor")
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
        x_spe = tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm_spe"
        )(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(6, round(EEG_LENGTH * SFREQ), 4), name="inp_eeg"
        )
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="cat_eeg_regions")(
            [
                inp_eeg[:, :, :, 0:1],
                inp_eeg[:, :, :, 1:2],
                inp_eeg[:, :, :, 2:3],
                inp_eeg[:, :, :, 3:4],
            ]
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3, name="eeg_to_rgb")(
            [x_eeg, x_eeg, x_eeg]
        )
        base_model_eeg = _make_backbone("eeg_extractor")
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm_eeg"
        )(x_eeg)
        inp.append(inp_eeg)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_modal_eeg")([y, x_eeg])
            if y is not None
            else x_eeg
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="inp_img")
        x_img = tf.keras.layers.Concatenate(axis=1, name="cat_img_regions")(
            [
                inp_img[:, :, :, 0:1],
                inp_img[:, :, :, 1:2],
                inp_img[:, :, :, 2:3],
                inp_img[:, :, :, 3:4],
            ]
        )
        x_img = tf.keras.layers.Concatenate(axis=3, name="img_to_rgb")(
            [x_img, x_img, x_img]
        )
        base_model_img = _make_backbone("img_extractor")
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm_img"
        )(x_img)
        inp.append(inp_img)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_modal_img")([y, x_img])
            if y is not None
            else x_img
        )

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
        x_stft = tf.keras.layers.Concatenate(axis=1, name="cat_stft_regions")(
            [
                inp_stft[:, :, :, :, 0],
                inp_stft[:, :, :, :, 1],
                inp_stft[:, :, :, :, 2],
                inp_stft[:, :, :, :, 3],
            ]
        )
        base_model_stft = _make_backbone("stft_extractor")
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
        x_stft = tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=-1), name="l2norm_stft"
        )(x_stft)
        inp.append(inp_stft)
        y = (
            tf.keras.layers.Concatenate(axis=1, name="cat_modal_stft")([y, x_stft])
            if y is not None
            else x_stft
        )

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32", name="head")(y)
    model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
    return model




## === cell 3
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
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
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
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)
                    spe = np.nan_to_num(spe, nan=0.0)
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
                    spe = np.array(
                        tf.image.resize(spe, ((HIGH - 32), LENGTH)), dtype=np.float32
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
                    x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (0.225**2)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :,
                        r_eeg
                        + round((50 - EEG_LENGTH) / 2 * SFREQ) : r_eeg
                        + round((50 + EEG_LENGTH) / 2 * SFREQ),
                        k,
                    ]
                    x_eeg[j, 1:5, :, k] = eeg
                    x_eeg[j, :, :, k] = (
                        x_eeg[j, :, :, k] - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                    ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    if self.mode == "test":
                        img = self.imgs[row.eeg_id][:, :, k]
                    else:
                        img = self.imgs[row.sign_id][:, :, k]
                    x_img[j, :, :, k] = img

                if "stft" in DATATYPE:
                    if self.mode == "test":
                        stft = self.stfts[row.eeg_id][:, :, k]
                    else:
                        stft = self.stfts[row.sign_id][:, :, k]

                    stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                    stft = np.log(stft)
                    stft = np.nan_to_num(stft, nan=0.0)
                    stft = np.round((stft - self.cmin) / (self.cmax - self.cmin) * 255)
                    shape0, shape1 = stft.shape[0], stft.shape[1]
                    stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                    stft = np.array(stft, dtype=np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (shape0, shape1, 3))
                    stft = np.array(
                        tf.image.resize(stft, ((HIGH - 32), LENGTH)), dtype=np.float32
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
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

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

        return x, y




## === cell 4
if not NEEDTRAIN:
    if "spe" in DATATYPE:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
            PATH_SPE = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        else:
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )
            PATH_SPE = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

        print("Test shape", test.shape)

        files2 = os.listdir(PATH_SPE)
        print(f"There are {len(files2)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH_SPE}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()
    else:
        if PLATFORM == "local":
            test = pd.read_csv(
                "./input/hms-harmful-brain-activity-classification/test.csv"
            )
        else:
            test = pd.read_csv(
                "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
            )

    from scipy import signal

    if PLATFORM == "local":
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH_EEG)
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

        eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")

        list_eeg = []
        list_img = []
        list_stft = []

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

            time_temp = 0
            time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
            time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

            list_img.append(eeg[:, time_start:time_stop])

            if "stft" in DATATYPE:
                ff, tt, pp = signal.spectrogram(
                    eeg[:, round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ)],
                    fs=SFREQ,
                    nperseg=232,
                    noverlap=194,
                )
                pp = pp[:, ff <= 40, :]
                pp = np.mean(pp, 0)
                list_stft.append(np.reshape(pp, (pp.shape[0], pp.shape[1], 1)))

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
                tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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

    votes = df[TARGETS].astype("float32").values
    votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = votes / row_sum
    prior = probs.mean(axis=0).astype("float32")
    prior = np.clip(prior, 1e-7, 1.0)
    prior = prior / prior.sum()
    print("Train prior probs:", prior.tolist())

    pattern = os.path.join(LOAD_MODELS_FROM, f"f*_stage{STAGETEST}.h5")
    weight_files = []
    if tf.io.gfile.exists(LOAD_MODELS_FROM):
        weight_files = sorted(tf.io.gfile.glob(pattern))
    weights_available = len(weight_files) > 0

    if not weights_available:
        print(f"WARNING: Weights directory not found or empty: {LOAD_MODELS_FROM}")
        pred = np.tile(prior[None, :], (len(test), 1)).astype("float32", copy=False)
    else:
        preds = []
        with strategy.scope():
            model = build_model()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2 if "spe" in DATATYPE else None,
            eegs=eegs2 if "eeg" in DATATYPE else None,
            imgs=imgs2 if "img" in DATATYPE else None,
            stfts=stfts2 if "stft" in DATATYPE else None,
        )

        for i in range(5):
            wpath = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            if not tf.io.gfile.exists(wpath):
                print(f"WARNING: Missing weights file, skipping fold: {wpath}")
                continue
            print(f"Fold {i + 1}")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        if len(preds) == 0:
            print("WARNING: No fold weights were loadable; falling back to prior only.")
            pred = np.tile(prior[None, :], (len(test), 1)).astype("float32", copy=False)
        else:
            pred = np.mean(preds, axis=0)
            print("\nTest preds shape", pred.shape)

            alpha = 0.15

            pred = pred.astype("float32", copy=False)
            pred = np.nan_to_num(
                pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0
            )
            pred = np.clip(pred, 1e-7, 1.0)
            pred = pred / pred.sum(axis=1, keepdims=True)
            pred = (1.0 - alpha) * pred + alpha * prior[None, :]

            pred = np.clip(pred, 1e-7, 1.0)
            pred = pred / pred.sum(axis=1, keepdims=True)

    if PLATFORM == "local":
        sample_path = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        sample_path = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    sample = pd.read_csv(sample_path)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub = sub[sample.columns]  # enforce correct column order

    p = sub.iloc[:, 1:].to_numpy(dtype="float32")
    p = np.nan_to_num(p, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    p = np.clip(p, 1e-7, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    sub.iloc[:, 1:] = p

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-sum stats:",
        float(sub.iloc[:, -6:].sum(axis=1).min()),
        float(sub.iloc[:, -6:].sum(axis=1).max()),
    )
    assert sub.shape[0] == sample.shape[0]
    assert list(sub.columns) == list(sample.columns)
    assert np.allclose(sub.iloc[:, 1:].sum(axis=1).values, 1.0, atol=1e-4)
