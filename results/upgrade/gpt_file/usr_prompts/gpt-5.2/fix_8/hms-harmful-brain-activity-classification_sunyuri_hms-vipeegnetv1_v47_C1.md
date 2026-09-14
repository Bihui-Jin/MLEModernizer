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

0.4855181355199378

# 6. Current score

1.48867

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I first fix the TensorFlow import crash caused by a protobuf incompatibility by forcing the pure-Python protobuf backend before importing TensorFlow. Next, I fix the Kaggle path resolution bug where you pass a duplicated competition folder name into `_kaggle_path`, which currently points to a non-existent nested path. Finally, because the external pretrained fold weights dataset (`models202402081`) is not available in your environment, I add a safe fallback that generates a valid, well-formed submission by using the label prior from `train.csv` (this is score-stable and avoids runtime failure while still producing reasonable KL vs. uniform). The core model/data logic is preserved; the only behavioral change is the inference-time fallback when weights are missing so the notebook can complete and write `submission.csv`.'
- What this solution (achieved 1.48867) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf backend (the current `"cpp"` setting is what triggers the `_message` import error in this environment). Because cell 0 currently fails, downstream cells don’t define `PLATFORM`, `NEEDTRAIN`, `tf`, etc.; fixing the TF import resolve the cascading `NameError`s. I also make path resolution robust by pointing `LOAD_MODELS_FROM` through `_kaggle_path()` and keeping the existing safe fallback that writes a valid `submission.csv` when external fold weights aren’t present. These changes are execution/robustness fixes and preserve the original modeling/inference logic (including the prior-based fallback).'

# 9. Code solution

## === cell 0
import os, sys, gc, math

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402081"

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 128
LENGTH = 32

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
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Mixed precision not enabled (unsupported in this TF build):", repr(e))
else:
    print("Using full precision")

HAVE_EFN = False
HAVE_ALBU = False
HAVE_LIBROSA = False
try:
    import efficientnet.tfkeras as efn  # type: ignore

    HAVE_EFN = True
except Exception as e:
    print("Optional import efficientnet.tfkeras disabled:", repr(e))

try:
    import albumentations as albu  # type: ignore

    HAVE_ALBU = True
except Exception as e:
    print("Optional import albumentations disabled:", repr(e))

try:
    import librosa  # type: ignore

    HAVE_LIBROSA = True
except Exception as e:
    print("Optional import librosa disabled:", repr(e))


def _kaggle_path(*parts):
    """
    Robust path resolution for Kaggle variants where data may be under
    /kaggle/input or /kaggle/data (and sometimes nested).
    """
    candidates = [
        os.path.join("/kaggle/input", *parts),
        os.path.join("/kaggle/data", *parts),
        os.path.join(
            "/kaggle/input", "hms-harmful-brain-activity-classification", *parts
        ),
        os.path.join(
            "/kaggle/data", "hms-harmful-brain-activity-classification", *parts
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
else:
    resolved = _kaggle_path(LOAD_MODELS_FROM)
    if os.path.exists(resolved):
        LOAD_MODELS_FROM = resolved
    else:
        LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

print("LOAD_MODELS_FROM =", LOAD_MODELS_FROM)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(_kaggle_path("train.csv"))

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

label_prior = train[TARGETS].mean(axis=0).values.astype(np.float64)
label_prior = np.clip(label_prior, 1e-12, 1.0)
label_prior = label_prior / label_prior.sum()
print("Label prior:", dict(zip(TARGETS, np.round(label_prior, 6))))



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
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_mel, X_eeg, y = self.__data_generation(indexes)
        return [X, X_mel, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_mel = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            eeg_temp = row.eeg_id

            if self.mode == "test":
                spectrogram_temp = row.spectrogram_id
                spec_start = 0
                eeg_start = 0
            elif self.mode == "valid":
                spectrogram_temp = row.spec_id
                rows = df[
                    (df.spectrogram_id == spectrogram_temp) & (df.eeg_id == eeg_temp)
                ].reset_index(drop=True)
                row = rows.iloc[np.random.permutation(len(rows))[0]]
                spec_start = round(row.spectrogram_label_offset_seconds / 2)
                eeg_start = round(row.eeg_label_offset_seconds * SFREQ)
            else:
                spectrogram_temp = row.spec_id
                rows = df[
                    (df.spectrogram_id == spectrogram_temp) & (df.eeg_id == eeg_temp)
                ].reset_index(drop=True)
                row = rows.iloc[np.random.permutation(len(rows))[0]]
                spec_start = round(row.spectrogram_label_offset_seconds / 2)
                eeg_start = round(row.eeg_label_offset_seconds * SFREQ)

            for k in range(4):
                img = self.specs[spectrogram_temp][
                    spec_start : spec_start + 300, k * 100 : (k + 1) * 100
                ].T
                img_eeg = self.eegs[eeg_temp][
                    :, eeg_start : (eeg_start + round(SFREQ * 50)), k
                ]
                img_eeg = img_eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]

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

                X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / (0.229**2)
                X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / (0.224**2)
                X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / (0.225**2)

                mel_spec_db = None
                for ii_x in range(img_eeg.shape[0]):
                    x = img_eeg[ii_x, :].astype(np.float32)

                    if HAVE_LIBROSA:
                        mel_spec = librosa.stft(
                            y=x, hop_length=len(x) // LENGTH, n_fft=256, win_length=128
                        )
                        mel_spec = np.abs(mel_spec) ** 2
                        mel_spec = mel_spec[:48, :LENGTH]
                        mel_spec = np.log(mel_spec + 1e-10)
                    else:
                        x_tf = tf.convert_to_tensor(x[None, :], dtype=tf.float32)
                        stft = tf.signal.stft(
                            x_tf,
                            frame_length=128,
                            frame_step=max(1, len(x) // LENGTH),
                            fft_length=256,
                            window_fn=tf.signal.hann_window,
                            pad_end=False,
                        )
                        mel_spec = tf.math.square(tf.abs(stft))[0]
                        mel_spec = tf.transpose(mel_spec)
                        mel_spec = mel_spec[:48, :LENGTH]
                        mel_spec = tf.math.log(mel_spec + 1e-10)
                        mel_spec = mel_spec.numpy()

                    mel_spec = np.clip(mel_spec, -6, 16)
                    if mel_spec_db is None:
                        mel_spec_db = mel_spec
                    else:
                        mel_spec_db = mel_spec_db + mel_spec

                mel_spec_db = mel_spec_db / img_eeg.shape[0]
                if HIGH != 100:
                    mel_spec_db = np.array(
                        tf.image.resize(
                            np.reshape(
                                mel_spec_db,
                                (mel_spec_db.shape[0], mel_spec_db.shape[1], 1),
                            ),
                            ((HIGH - 16), LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]

                mel_spec_db = np.round((mel_spec_db - (-6)) / (16 - (-6)) * 256)
                mel_spec_db = np.reshape(
                    mel_spec_db, (mel_spec_db.shape[0] * mel_spec_db.shape[1])
                )
                mel_spec_db = np.array(mel_spec_db, dtype=np.int16)
                mel_spec_db = self.cmaps[mel_spec_db - 1]
                mel_spec_db = np.reshape(mel_spec_db, (HIGH - 16, LENGTH, 3))

                X_mel[
                    j,
                    round((HIGH - mel_spec_db.shape[0]) / 2) : round(
                        (HIGH + mel_spec_db.shape[0]) / 2
                    ),
                    :,
                    :,
                    k,
                ] = mel_spec_db

                X_mel[j, :, :, 0, k] = (X_mel[j, :, :, 0, k] - 0.485) / (0.229**2)
                X_mel[j, :, :, 1, k] = (X_mel[j, :, :, 1, k] - 0.456) / (0.224**2)
                X_mel[j, :, :, 2, k] = (X_mel[j, :, :, 2, k] - 0.406) / (0.225**2)

                X_eeg[j, 1, :, k] = img_eeg[0, :]
                X_eeg[j, 2, :, k] = img_eeg[1, :]
                X_eeg[j, 3, :, k] = img_eeg[2, :]
                X_eeg[j, 4, :, k] = img_eeg[3, :]

                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k] - np.mean(X_eeg[j, :, :, k], 1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

            if self.mode != "test":
                y_data = row[TARGETS].values
                y[j] = y_data / np.sum(y_data)

        return X, X_mel, X_eeg, y

    def __random_transform(self, img):
        if not HAVE_ALBU:
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




## === cell 3
def _make_efficientnet_b2_backbone():
    if HAVE_EFN:
        return efn.EfficientNetB2(include_top=False, weights=None, input_shape=None)
    return tf.keras.applications.EfficientNetB2(
        include_top=False, weights=None, input_shape=(HIGH, LENGTH * 8, 3)
    )


def _make_efficientnet_b0_backbone():
    if HAVE_EFN:
        return efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    return tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(32, round(EEG_LENGTH * SFREQ), 3),
    )


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_mel = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

    base_model = _make_efficientnet_b2_backbone()
    base_model._name = "spectrogram_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        else:
            base_model.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

    x0 = inp_mel[:, :, :, :, 0]
    x1 = inp_mel[:, :, :, :, 1]
    x2 = inp_mel[:, :, :, :, 2]
    x3 = inp_mel[:, :, :, :, 3]
    x_mel = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

    x = tf.keras.layers.Concatenate(axis=2)([x, x_mel])
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm_spec"
    )(x)

    base_model_eeg = _make_efficientnet_b0_backbone()
    base_model_eeg._name = "eeg_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model_eeg.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        else:
            base_model_eeg.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0_eeg = inp_eeg[:, :, :, :1]
    x1_eeg = inp_eeg[:, :, :, 1:2]
    x2_eeg = inp_eeg[:, :, :, 2:3]
    x3_eeg = inp_eeg[:, :, :, 3:4]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])
    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])  # (B,24,T,3)

    pad_h = 32 - 24
    x_eeg = tf.keras.layers.ZeroPadding2D(padding=((0, pad_h), (0, 0)))(x_eeg)

    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
    x_eeg = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm_eeg"
    )(x_eeg)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_mel, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 4
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(_kaggle_path("test.csv"))
    print("Test shape", test.shape)

    expected_first_weight = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f0.h5")
    if not os.path.exists(expected_first_weight):
        print("WARNING: Model weights not found at:", expected_first_weight)
        print(
            "Falling back to label-prior predictions to produce a valid submission.csv"
        )

        pred = np.tile(label_prior[None, :], (len(test), 1)).astype(np.float64)
        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred.astype(np.float32)
        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row prob sum stats:",
            float(sub[TARGETS].sum(axis=1).min()),
            float(sub[TARGETS].sum(axis=1).max()),
        )
        print(sub.head())
    else:
        spec_npy_candidates = [
            "/kaggle/input/brain-spectrograms/specs.npy",
            "/kaggle/input/brain-spectrograms/specs_test.npy",
            "/kaggle/input/brain-spectrograms/specs2.npy",
        ]
        eeg_npy_candidates = [
            "/kaggle/input/brain-eegs/eegs.npy",
            "/kaggle/input/brain-eegs/eegs_test.npy",
            "/kaggle/input/brain-eegs/eegs2.npy",
        ]

        spectrograms2 = None
        for p in spec_npy_candidates:
            if os.path.exists(p):
                spectrograms2 = np.load(p, allow_pickle=True).item()
                print("Loaded spectrogram dict from", p, "keys=", len(spectrograms2))
                break

        eegs2 = None
        for p in eeg_npy_candidates:
            if os.path.exists(p):
                eegs2 = np.load(p, allow_pickle=True).item()
                print("Loaded eeg dict from", p, "keys=", len(eegs2))
                break

        if spectrograms2 is None:
            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            else:
                PATH2 = _kaggle_path("test_spectrograms") + "/"

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

        if eegs2 is None:
            from scipy import signal

            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
            else:
                PATH2 = _kaggle_path("test_eegs") + "/"

            files2 = os.listdir(PATH2)
            print(f"There are {len(files2)} test eeg parquets")

            eegs2 = {}
            if len(filter_range) == 1:
                if filter_range[0] > 5:
                    b, a = signal.butter(
                        3, np.float32(filter_range) * 2 / SFREQ, "lowpass"
                    )
                else:
                    b, a = signal.butter(
                        3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                    )
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
                )

            for i, f in enumerate(files2):
                if i % 100 == 0:
                    print(i, ", ", end="")
                eeg_default = pd.read_parquet(f"{PATH2}{f}")
                name = int(f.split(".")[0])

                if len(test[test.eeg_id == name]) > 0:
                    list_eeg = []
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
                        list_eeg.append(
                            np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1))
                        )

                    list_eeg = np.concatenate(list_eeg, 2)
                    eegs2[name] = list_eeg
            print()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
        )

        with strategy.scope():
            model = build_model()

        preds = []
        for i in range(5):
            print(f"Fold {i+1}")
            wpath = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
            if not os.path.exists(wpath):
                raise FileNotFoundError(f"Missing model weights: {wpath}")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = np.asarray(pred, dtype=np.float64)
        pred = np.clip(pred, 1e-9, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred.astype(np.float32)

        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(
            "Row prob sum stats:",
            float(sub[TARGETS].sum(axis=1).min()),
            float(sub[TARGETS].sum(axis=1).max()),
        )
        print(sub.head())
