# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.13

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

0.2869336254823406

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blockers by defining all missing global configuration variables (like `NEEDTRAIN`, paths, targets, and feature flags) and by adding a small protobuf compatibility shim to prevent the `MessageFactory.GetPrototype` crash that can happen in some Kaggle TF environments. I also ensure the inference path (NEEDTRAIN=False) loads test data from the provided `/kaggle/input/hms-harmful-brain-activity-classification` structure, creates the required dicts (eegs/specs/stfts/imgs), and always writes a valid `submission.csv` with the exact columns and row-wise probabilities summing to 1. To keep core logic unchanged and meet the 600s limit, the script default to inference-only and fall back to a uniform-probability submission if no model weights are found at `LOAD_MODELS_FROM` (so you still get a valid CSV instead of a crash). These changes are correctness/stability focused; they don’t alter the model architecture, loss, generator logic, or prediction normalization.'
- What this solution (achieved 1.39779) has done: 'I fix the protobuf shim that currently still triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by making the patch robust to different protobuf versions and by applying it before TensorFlow import. Then I ensure inference produces a non-uniform submission when weights are absent by adding a tiny, score-improving (but core-logic-preserving) fallback: use the normalized mean class distribution from `train.csv` as priors instead of uniform probabilities (still sums to 1 and respects the metric). I also make sure the code always locates the dataset correctly under `/kaggle/input/hms-harmful-brain-activity-classification` even if the competition data is nested one level deeper. These changes are minimal, do not alter the model/training loop/architecture, and should reduce KL divergence versus uniform output when models are missing.'

# 9. Code solution

## === cell 0
"""
Performance-focused refactor notes (correctness-preserving):
- Avoid repeated pandas .iloc/.loc inside __getitem__ by caching needed columns as NumPy arrays.
- Precompute per-sign_id candidate df indices and the deterministic "valid" pick (median eeg_sub_id) once.
  This removes O(batch_size) pandas filtering and sorting every batch while preserving the same selection rule.
- Keep training augmentations and label/weight computations identical (same RNG source: np.random with fixed seeds).
- Use tf.keras Sequence exactly as before (no change to model/fit semantics).

Minimal score-improvement changes (inference only):
- Fix generator data-source bug: during inference we must read from the *_test dicts passed into DataGenerator,
  not from empty global dicts (otherwise predictions can fail or degrade to priors/uniform).
- Fix batching slice off-by-one that can drop one row per batch boundary and misalign predictions to eeg_id order.
- Make DataGenerator return (x, y) for test mode to be maximally compatible with Keras predict() APIs.
"""

import os, gc, time, io, warnings

warnings.filterwarnings("ignore")

try:
    from google.protobuf import message_factory as _message_factory  # type: ignore

    if hasattr(_message_factory, "MessageFactory"):
        MF = _message_factory.MessageFactory  # class
        if not hasattr(MF, "GetPrototype"):
            if hasattr(MF, "GetMessageClass"):

                def _GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                setattr(MF, "GetPrototype", _GetPrototype)
            else:

                def _GetPrototype(self, descriptor):
                    return None

                setattr(MF, "GetPrototype", _GetPrototype)
except Exception:
    pass

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import zoom

import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

PLATFORM = "kaggle"

NEEDTRAIN = False

LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
if not os.path.exists(os.path.join(LOAD_DATA_FROM, "train.csv")):
    nested = os.path.join(LOAD_DATA_FROM, "hms-harmful-brain-activity-classification")
    if os.path.exists(os.path.join(nested, "train.csv")):
        LOAD_DATA_FROM = nested

LOAD_MODELS_FROM = "./models"

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
TARGETS_RAW = [f"{c}_raw" for c in TARGETS]

DATATYPE = ["eeg"]  # allowed: "spe", "eeg", "stft", "img"

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

STFT_TIME = 0.5
STFT_HIGH = 80
STFT_WIDE = 200

SPE_HIGH = 128
SPE_WIDE = 256

IMG_HIGH = 512
IMG_WIDE = 512
IMG_LENGTH = 10

filter_range = None
filter_range2 = (0.5, 40.0)
b, a = signal.butter(
    3, np.array([0.5, 40.0], dtype=np.float32) * 2 / RSFREQ, "bandpass"
)

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]

SPLITS = 5
BATCHSIZE = 16
EPOCHS = 6
LEARN_RATE = 1e-3

TEST_BATCHSIZE = 64

eegs, stfts, spectrograms, imgs = {}, {}, {}, {}
eegs_test, stfts_test, spectrograms_test, imgs_test = {}, {}, {}, {}

df = None
train = None
SIGNID_TO_DF_IDXS = {}
SIGNID_TO_VALID_DF_PICK = None
if NEEDTRAIN:
    train = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
    df = train.copy()

    for c in TARGETS:
        train[f"{c}_raw"] = train[c].astype(np.float32)
    votes = train[TARGETS].values.astype(np.float32)
    votes_sum = np.sum(votes, axis=1, keepdims=True)
    votes_sum[votes_sum == 0] = 1.0
    train[TARGETS] = votes / votes_sum

    train["sign_id"] = np.arange(len(train), dtype=np.int64)

    SIGNID_TO_DF_IDXS = {
        int(sid): np.array([int(sid)], dtype=np.int64)
        for sid in train["sign_id"].values
    }

    SIGNID_TO_VALID_DF_PICK = {}
    for sid, idxs in SIGNID_TO_DF_IDXS.items():
        if len(idxs) == 0:
            continue
        eeg_sub_ids = df.loc[idxs, "eeg_sub_id"].to_numpy()
        pick = idxs[np.argsort(eeg_sub_ids, kind="mergesort")[len(idxs) // 2]]
        SIGNID_TO_VALID_DF_PICK[int(sid)] = int(pick)




## === cell 1
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        sample_weights=False,
        mode="train",
        eegs=None,
        stfts=None,
        specs=None,
        imgs=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.sample_weights = bool(sample_weights)
        self.mode = mode

        self.eegs = eegs if eegs is not None else {}
        self.stfts = stfts if stfts is not None else {}
        self.specs = specs if specs is not None else {}
        self.imgs = imgs if imgs is not None else {}

        cols_needed = ["eeg_id", "spectrogram_id"]
        if "sign_id" in self.dataframe.columns:
            cols_needed.append("sign_id")
        if self.mode != "test":
            cols_needed += ["expert_consensus"]
            for c in TARGETS_RAW:
                cols_needed.append(c)
            for c in TARGETS:
                cols_needed.append(c)
            cols_needed += [
                "seizure_vote_raw",
                "lpd_vote_raw",
                "gpd_vote_raw",
                "lrda_vote_raw",
                "grda_vote_raw",
            ]
        cols_needed = [c for c in cols_needed if c in self.dataframe.columns]

        self._col = {}
        for c in cols_needed:
            self._col[c] = self.dataframe[c].to_numpy()

        if self.mode != "test":
            self._targets = self.dataframe[TARGETS].to_numpy(
                dtype=np.float32, copy=False
            )
            self._targets_raw = self.dataframe[TARGETS_RAW].to_numpy(
                dtype=np.float32, copy=False
            )
        else:
            self._targets = None
            self._targets_raw = None

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)

        if self.mode == "test":
            return x, y
        return x, y, sample_weights

    def on_epoch_end(self):
        self.nan = 0
        self.indexes = np.arange(len(self.dataframe), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        bs = len(indexes)

        if "spe" in DATATYPE:
            x_spe = np.zeros((bs, 4, SPE_HIGH, SPE_WIDE), dtype=np.float32)
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    bs,
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype=np.float32,
            )
        if "img" in DATATYPE:
            x_img = np.zeros((bs, IMG_HIGH, IMG_WIDE, 3), dtype=np.float32)

        y = np.zeros((bs, len(TARGETS)), dtype=np.float32)
        sample_weights = np.zeros((bs, 1), dtype=np.float32)

        for j, i in enumerate(indexes):
            eeg_id = (
                int(self._col["eeg_id"][i])
                if "eeg_id" in self._col
                else int(self.dataframe.iloc[i].eeg_id)
            )
            spectrogram_id = (
                int(self._col["spectrogram_id"][i])
                if "spectrogram_id" in self._col
                else int(self.dataframe.iloc[i].spectrogram_id)
            )

            if "sign_id" in self._col:
                sign_id = int(self._col["sign_id"][i])
            else:
                sign_id = int(i)

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0.0
            else:
                if self.sample_weights:
                    sample_weight = float(np.sum(self._targets_raw[i])) / 20.0
                else:
                    sample_weight = 1.0

                df_idxs = SIGNID_TO_DF_IDXS.get(sign_id, None)
                if df_idxs is None or len(df_idxs) == 0:
                    row0 = self.dataframe.iloc[int(i)]
                    rows = df.loc[
                        (df.eeg_id == row0.eeg_id)
                        * (df.seizure_vote == row0.seizure_vote_raw)
                        * (df.lpd_vote == row0.lpd_vote_raw)
                        * (df.gpd_vote == row0.gpd_vote_raw)
                        * (df.lrda_vote == row0.lrda_vote_raw)
                        * (df.grda_vote == row0.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    if self.mode == "train":
                        rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                            drop=True
                        )
                        row = rows.loc[0, :]
                    else:
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = float(row.eeg_label_offset_seconds)
                else:
                    if self.mode == "train":
                        pick = int(df_idxs[np.random.randint(0, len(df_idxs))])
                    else:
                        pick = int(SIGNID_TO_VALID_DF_PICK[sign_id])
                    row = df.iloc[pick]
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = float(row.eeg_label_offset_seconds)

                if self.mode == "train":
                    r_eeg = r_eeg + np.random.random() * 10 - 5
                    r_eeg = max(0.0, r_eeg)
                    r_eeg = min(r_eeg, self.eegs[eeg_id].shape[1] / RSFREQ - 50)

            if "spe" in DATATYPE:
                spe = []
                spec = self.specs[spectrogram_id]
                for k in range(4):
                    spe.append(
                        np.reshape(
                            spec[r_spe : (r_spe + 300), k * 100 : (k + 1) * 100].T,
                            (1, 100, 300),
                        )
                    )
                spe = np.concatenate(spe, axis=0)

            if "eeg" in DATATYPE:
                eeg = self.eegs[eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]
                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

            if "stft" in DATATYPE:
                stft_t = self.stfts[-eeg_id]
                r_stft = (np.where(stft_t >= (r_eeg - float(np.min(stft_t)))))[0][0]
                r_stft2 = (np.where(stft_t <= (50 + r_eeg - float(np.min(stft_t)))))[0][
                    -1
                ]
                stft = self.stfts[eeg_id][:, :, r_stft:r_stft2]
                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = self.imgs[sign_id]

            if "spe" in DATATYPE:
                spe[np.isnan(spe)] = 0
                exp_min, exp_max = -4, 6
                spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                spe = np.log(spe)

                if (spe.shape[1] != SPE_HIGH) or (spe.shape[2] != SPE_WIDE):
                    spe2 = np.zeros(
                        (spe.shape[0], SPE_HIGH, SPE_WIDE), dtype=np.float32
                    )
                    for k in range(4):
                        scaled_arr = zoom(
                            spe[k],
                            (SPE_HIGH / spe.shape[1], SPE_WIDE / spe.shape[2]),
                            order=1,
                        )
                        spe2[k, :, :] = scaled_arr
                    spe = spe2.copy()

                if self.mode == "train":
                    spe2 = spe.copy()
                    if np.random.rand() > 0.5:
                        spe[0] = spe2[2]
                        spe[2] = spe2[0]
                    if np.random.rand() > 0.5:
                        spe[1] = spe2[3]
                        spe[3] = spe2[1]
                    if np.random.rand() > 0.5:
                        spe = spe[::-1, :, :]
                    if np.random.rand() > 0.5:
                        for ii in range(spe.shape[0]):
                            m1 = round(np.random.rand() * spe.shape[2] / 2)
                            m2 = round(np.random.rand() * spe.shape[2] / 2)
                            if np.random.rand() > 0.5:
                                m1 = spe.shape[2] - m1
                                m2 = spe.shape[2] - m2
                            m_min = min(m1, m2)
                            m_max = min(max(m1, m2), m_min + round(spe.shape[2] * 0.05))
                            spe[ii, :, m_min:m_max] = 0

                spe = (spe - exp_min) / (exp_max - exp_min) * 255
                spe = np.clip(spe, a_min=0, a_max=255)
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]

                if self.mode == "train":
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0
                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][np.random.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][np.random.permutation(8), :]

                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]
                    if np.random.rand() > 0.5:
                        eeg = -eeg
                    if np.random.rand() > 0.5:
                        eeg = eeg[:, ::-1]
                else:
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eeg = eeg + 1024
                eeg = eeg / 2048 * 255
                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                exp_min, exp_max = 0, 8
                stft = np.clip(stft, a_min=0, a_max=np.exp(exp_max))
                stft = np.log1p(stft)

                if self.mode == "train":
                    stft[0 : round(stft.shape[0] / 2), :, :] = stft[
                        0 : round(stft.shape[0] / 2), :, :
                    ][np.random.permutation(stft.shape[0] // 2), :, :]
                    stft[-round(stft.shape[0] / 2) :, :, :] = stft[
                        -round(stft.shape[0] / 2) :, :, :
                    ][np.random.permutation(stft.shape[0] // 2), :, :]
                    stft2 = stft.copy()
                    stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                        0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                    ]
                    stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                        3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                    ]
                    stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                        1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                    ]
                    stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                        2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                    ]
                    if np.random.rand() > 0.5:
                        stft = stft[::-1, :, :]
                else:
                    stft2 = stft.copy()
                    stft[0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)] = stft2[
                        0 * stft.shape[0] // 4 : (1 * stft.shape[0] // 4)
                    ]
                    stft[1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)] = stft2[
                        3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)
                    ]
                    stft[2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)] = stft2[
                        1 * stft.shape[0] // 4 : (2 * stft.shape[0] // 4)
                    ]
                    stft[3 * stft.shape[0] // 4 : (4 * stft.shape[0] // 4)] = stft2[
                        2 * stft.shape[0] // 4 : (3 * stft.shape[0] // 4)
                    ]

                stft = (stft - exp_min) / (exp_max - exp_min) * 255
                stft = np.clip(stft, a_min=0, a_max=255)

                if j == 0:
                    x_stft = np.zeros(
                        (bs, stft.shape[0], stft.shape[1], stft.shape[2]),
                        dtype=np.float32,
                    )
                x_stft[j] = stft

            if "img" in DATATYPE:
                img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
                if self.mode == "train":
                    img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                    img[10:18, :, :] = img[10:18, :, :][np.random.permutation(8), :, :]
                    if np.random.rand() > 0.5:
                        img = img[::-1, :, :]
                for ii in range(img.shape[0]):
                    axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                    start_temp = round(
                        max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                    )
                    end_temp = round(
                        min(
                            img_save.shape[0],
                            axis_temp + img_save.shape[1] / img.shape[0],
                        )
                    )
                    temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                    img_save[start_temp:end_temp, :] = (
                        img_save[start_temp:end_temp, :]
                        + img[
                            ii, temp_temp : round(temp_temp + end_temp - start_temp), :
                        ]
                    )
                img_save = np.clip(img_save, a_min=0, a_max=1)
                img = np.reshape(img_save, (img_save.shape[0], img_save.shape[1], 1))
                img = np.concatenate((img, img, img), -1)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                yy = self._targets[i]
                y[j] = yy / float(np.sum(yy))
                sample_weights[j] = sample_weight

        x = {}
        if "spe" in DATATYPE:
            x["spe"] = x_spe
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        if "stft" in DATATYPE:
            x["stft"] = x_stft
        if "img" in DATATYPE:
            x["img"] = x_img
        return x, y, sample_weights




## === cell 2
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = total_step

        if warmth_rate == 0:
            self.warm_step = 1
        else:
            self.warm_step = int(warmth_rate)

        self.lr_max = lr_max
        self.lr_min = lr_min

        self.begin = 1

    def __call__(self, step):
        if step == self.total_step:
            self.begin = 0
            self.lr_max = self.lr_max * 0.5
            self.lr_min = self.lr_min * 0.1

        step = step % self.total_step
        step = step + 1

        if (self.begin == 1) and (step < self.warm_step):
            lr = self.lr_max / self.warm_step * step
        else:
            if self.begin == 1:
                if self.total_step == 1:
                    lr = self.lr_max
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0
                        + tf.cos(
                            (step - self.warm_step)
                            / (self.total_step - self.warm_step)
                            * np.pi
                        )
                    )
            else:
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0 + tf.cos(step / 10 * np.pi)
                )

        return np.float32(lr)


class IniToOne(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOne, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        for i in range(filter_count):
            kernel[i % filter_length, 0, i] = 1.0
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOne(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOne, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class IniToOneAtten(tf.keras.initializers.Initializer):
    def __init__(self):
        super(IniToOneAtten, self).__init__()

    def __call__(self, shape, dtype=None):
        assert len(shape) == 3
        filter_length, input_channel, filter_count = shape

        kernel = np.zeros(shape, dtype=np.float32)
        kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
            (filter_length) // 2 + 1 - (filter_length - 1) // 2
        )
        kernel = tf.convert_to_tensor(kernel, dtype=dtype)
        return kernel

    def get_config(self):
        return {}


class SumToOneAtten(tf.keras.constraints.Constraint):
    def __init__(self):
        super(SumToOneAtten, self).__init__()

    def __call__(self, w):
        w = tf.abs(w)
        w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
        return w_normed

    def get_config(self):
        return {}


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="gelu"),
                tf.keras.layers.Dense(feat_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, weights = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), weights


class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)


def build_model():
    inp = list()
    y = 0
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
        x_spe = tf.keras.layers.Reshape(
            (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
        )(inp_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [
                x_spe[:, 0, :, :, :],
                x_spe[:, 1, :, :, :],
                x_spe[:, 2, :, :, :],
                x_spe[:, 3, :, :, :],
            ]
        )

        base_model_spe = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_spe.load_weights(
                    f"./input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_spe.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                )
        base_model_spe.name = "spe_extractor"

        x_spe = base_model_spe(x_spe)

        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.5)(x_spe)

        inp.append(inp_spe)

        y = x_spe * 1

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
        )
        x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        strides = 10
        if PLATFORM == "local":
            eeg_embed = tf.keras.layers.Conv1D(
                filters=strides * 3,
                kernel_size=strides,
                strides=strides,
                padding="same",
                use_bias=False,
                activation=None,
                kernel_initializer=IniToOne(),
                kernel_constraint=SumToOne(),
                input_shape=(None, 1),
            )
        else:
            eeg_embed = tf.keras.layers.Conv1D(
                filters=strides * 3,
                kernel_size=strides,
                strides=strides,
                padding="same",
                use_bias=False,
                activation=None,
            )

        x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)

        x_eeg = tf.keras.layers.Concatenate(axis=-1)(
            [
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 0 * strides : 1 * strides]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 1 * strides : 2 * strides]
                ),
                tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                    x_eeg[:, :, :, 2 * strides : 3 * strides]
                ),
            ]
        )
        x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
        x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
        x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

        base_model_eeg = tf.keras.applications.EfficientNetV2M(
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
        base_model_eeg.name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)

        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]

        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)

        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(16, STFT_HIGH, STFT_WIDE), name="stft")
        x_stft = tf.keras.layers.Reshape(
            (inp_stft.shape[1], inp_stft.shape[2], inp_stft.shape[3], 1)
        )(inp_stft)
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        x_stft = tf.keras.layers.Concatenate(axis=1)(
            [
                x_stft[:, 0, :, :, :],
                x_stft[:, 1, :, :, :],
                x_stft[:, 2, :, :, :],
                x_stft[:, 3, :, :, :],
                x_stft[:, 4, :, :, :],
                x_stft[:, 5, :, :, :],
                x_stft[:, 6, :, :, :],
                x_stft[:, 7, :, :, :],
                x_stft[:, 8, :, :, :],
                x_stft[:, 9, :, :, :],
                x_stft[:, 10, :, :, :],
                x_stft[:, 11, :, :, :],
                x_stft[:, 12, :, :, :],
                x_stft[:, 13, :, :, :],
                x_stft[:, 14, :, :, :],
                x_stft[:, 15, :, :, :],
            ]
        )

        base_model_stft = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights=None, include_preprocessing=True
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_stft.load_weights(
                    f"./input/pre-trained-weights/{base_model_stft.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_stft.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_stft.name}_notop.h5"
                )
        base_model_stft.name = "stft_extractor"

        x_stft = base_model_stft(x_stft)

        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

        x_stft = tf.keras.layers.Dropout(0.5)(x_stft)

        inp.append(inp_stft)

        if y == 0:
            y = x_stft * 1
        else:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_tensor=inp_img
        )
        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_img.load_weights(
                    f"./input/pre-trained-weights/{base_model_img.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_img.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_img.name}_notop.h5"
                )
        base_model_img.name = "img_extractor"
        x_img = base_model_img.output

        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

        inp.append(inp_img)

        if y == 0:
            y = x_img * 1
        else:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])

    y = y_eeg * 1
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 3
def train_fold(
    i,
    stage,
    train_index,
    valid_index,
    df_train_stage1,
    df_valid_stage1,
    df_train_stage2,
    df_valid_stage2,
    build_model,
    BATCHSIZE,
    EPOCHS,
    LEARN_RATE,
    TARGETS,
    TARGETS_RAW,
):

    print("#" * 25)
    print(f"### Fold {i + 1}")

    model = build_model()
    loss = tf.keras.losses.KLDivergence()

    if stage == 1:
        train_gen_stage = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
        epochs_to_run = EPOCHS
    else:
        train_gen_stage = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen_stage = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
        model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
        callbacks_stage = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(
                    max(round(EPOCHS / 3), 1),
                    LEARN_RATE * 0.1,
                    LEARN_RATE * 0.1 * 0.1 * 0.1,
                    0,
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
        ]
        epochs_to_run = max(round(EPOCHS / 3), 1)

    model.compile(loss=loss, optimizer=opt, jit_compile=True)

    history = model.fit(
        train_gen_stage,
        epochs=epochs_to_run,
        verbose=1,
        validation_data=valid_gen_stage,
        callbacks=callbacks_stage,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=16,
    )

    model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

    loss_hist = history.history["loss"]
    val_loss_hist = history.history["val_loss"]
    epochs = range(1, len(loss_hist) + 1)
    plt.figure()
    plt.plot(epochs, loss_hist, "bo", label="loss")
    plt.plot(epochs, val_loss_hist, "b", label="val_loss")
    plt.title(
        f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
        fontsize=12,
    )
    plt.legend()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
    plt.close()

    if stage == 1:
        valid_stage = df_valid_stage1[TARGETS].values
    else:
        valid_stage = df_valid_stage2[TARGETS].values

    predict_stage = model.predict(
        valid_gen_stage,
        verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=16,
    )

    del train_gen_stage, valid_gen_stage, history, model
    tf.keras.backend.clear_session()
    gc.collect()

    cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
    cm = cm / np.sum(cm, 1, keepdims=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    plt.yticks(
        tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
    )
    thresh = cm.max() / 2.0
    import itertools

    for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[ii, jj] > -0.1:
            plt.text(
                jj,
                ii,
                str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                horizontalalignment="center",
                color="white" if cm[ii, jj] > thresh else "black",
                fontsize=10,
            )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join("models", f"fold{i}_stage{stage}_cm.svg"))
    plt.close()

    del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
    gc.collect()




## === cell 4
def _find_weight_files(load_models_from: str, max_folds: int = 100):
    """
    Inference-only helper:
    Try multiple likely directories for weights to increase chance we use real models.
    This does not change model logic; only where we look for already-trained weights.
    """
    candidates = []

    candidates.append(load_models_from)

    candidates += [
        "/kaggle/input/models",
        "/kaggle/input/model",
        "/kaggle/input/weights",
    ]

    try:
        for d in os.listdir("/kaggle/input"):
            candidates.append(os.path.join("/kaggle/input", d))
            candidates.append(os.path.join("/kaggle/input", d, "models"))
            candidates.append(os.path.join("/kaggle/input", d, "weights"))
    except Exception:
        pass

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    weight_files = []
    for base in candidates:
        if not base or not os.path.exists(base):
            continue
        for model_i in range(max_folds):
            wf = os.path.join(base, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(wf):
                weight_files.append((model_i, wf))
    best_by_fold = {}
    for fold_i, path in weight_files:
        if fold_i not in best_by_fold:
            best_by_fold[fold_i] = path
    return sorted([(k, v) for k, v in best_by_fold.items()], key=lambda x: x[0])


def _compute_priors(train_csv_path: str, targets: list[str]):
    """
    Inference-only helper:
    Compute global prior and patient-conditioned prior from train.csv vote distributions.
    These priors are legitimate, label-only statistics and often reduce KL vs uniform/global.
    """
    tr = pd.read_csv(train_csv_path, usecols=["patient_id"] + targets)
    votes = tr[targets].values.astype(np.float32)
    vs = votes.sum(axis=1, keepdims=True)
    vs[vs == 0] = 1.0
    probs = votes / vs

    global_prior = probs.mean(axis=0).astype(np.float32)
    global_prior = np.clip(global_prior, 1e-7, 1.0)
    global_prior = global_prior / global_prior.sum()

    df_probs = pd.DataFrame(probs, columns=targets)
    df_probs["patient_id"] = tr["patient_id"].values
    patient_prior_df = df_probs.groupby("patient_id")[targets].mean()
    patient_prior_df = patient_prior_df.clip(lower=1e-7)
    patient_prior_df = patient_prior_df.div(patient_prior_df.sum(axis=1), axis=0)

    del tr, votes, vs, probs, df_probs
    gc.collect()
    return global_prior, patient_prior_df




## === cell 5
if __name__ == "__main__":
    if NEEDTRAIN:
        if not os.path.exists("models"):
            os.makedirs("models")

        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=SPLITS)

        for i, (train_index, valid_index) in enumerate(
            gkf.split(train, train.expert_consensus, train.patient_id)
        ):
            print("#" * 25)
            print(f"### Fold {i + 1}")

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [1, 2]:
                train_fold(
                    i,
                    stage,
                    train_index,
                    valid_index,
                    df_train_stage1,
                    df_valid_stage1,
                    df_train_stage2,
                    df_valid_stage2,
                    build_model,
                    BATCHSIZE,
                    EPOCHS,
                    LEARN_RATE,
                    TARGETS,
                    TARGETS_RAW,
                )

    else:
        preds_all_batches = []
        models = []

        weight_files = _find_weight_files(LOAD_MODELS_FROM, max_folds=100)

        if len(weight_files) > 0:
            print(f"Found {len(weight_files)} fold weight files.")
            for model_i, wf in weight_files:
                print(f"Fold {model_i + 1}: {wf}")
                model = build_model()
                model.load_weights(wf)
                model.compile(jit_compile=True)
                models.append(model)
        else:
            print(
                f"No weights found (searched from {LOAD_MODELS_FROM} + common Kaggle locations). "
                f"Will create patient-conditioned prior submission (from train.csv) instead."
            )

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        prior_global = None
        prior_by_patient = None
        if len(models) == 0:
            try:
                prior_global, prior_by_patient = _compute_priors(
                    os.path.join(LOAD_DATA_FROM, "train.csv"), TARGETS
                )
                print(
                    "Using global prior:",
                    dict(zip(TARGETS, prior_global.round(6).tolist())),
                )
                print(f"Computed patient priors for {len(prior_by_patient)} patients")
            except Exception as e:
                print(
                    "Failed to compute priors from train.csv, using uniform. Reason:",
                    repr(e),
                )
                prior_global = np.full(
                    (len(TARGETS),), 1.0 / len(TARGETS), dtype=np.float32
                )
                prior_by_patient = None

        if "spe" in DATATYPE:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            files_test = os.listdir(PATH_test)
            print(f"There are {len(files_test)} test spectrogram parquets")
            for i, f in enumerate(files_test):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_test}{f}")
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values
            print()

        PATH_test_eeg = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
        if ("eeg" in DATATYPE) or ("stft" in DATATYPE) or ("img" in DATATYPE):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            start_idx = 0
            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")
                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test_eeg, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for channel in BRAIN:
                    a0, a1 = channel.split("-")
                    eeg_temp = (eeg_default.loc[:, a0] - eeg_default.loc[:, a1]).values
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                if SFREQ != RSFREQ:
                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                if "stft" in DATATYPE:
                    eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                    nperseg = round(RSFREQ * 1)
                    ff, tt, ss = signal.spectrogram(
                        eeg2,
                        axis=1,
                        fs=RSFREQ,
                        nperseg=nperseg,
                        noverlap=round(nperseg - RSFREQ * STFT_TIME),
                        nfft=RSFREQ * 5,
                    )
                    ss[np.isnan(ss)] = 0
                    ss = ss[:, (ff > 0) * (ff <= 20), :]
                    ss = np.concatenate(
                        (
                            ss[0 : round(EEG_CHANNEL_USED / 2), :, :],
                            ss[-round(EEG_CHANNEL_USED / 2) :, :, :],
                        ),
                        axis=0,
                    )
                    for k in range(4):
                        ss[k, :, :] = np.mean(ss[k * 4 : (k + 1) * 4, :, :], 0)
                    ss = ss[:4, :, :]
                    ss = np.array(ss, dtype=np.float32)
                    tt = np.array(tt, dtype=np.float32)

                if "img" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)
                    train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                    for j in range(len(train_plot)):
                        eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
                        eeg_plot = eeg_plot[
                            :,
                            round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                                (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                            ),
                        ]
                        img_save = np.zeros(
                            (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                        )
                        for ii in range(eeg_plot.shape[0]):
                            fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                            fig.patch.set_facecolor("black")
                            plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)
                            plt.xlim(-5, eeg_plot.shape[1] + 5)
                            plt.ylim(0, 200)
                            plt.axis("off")
                            byte_stream = io.BytesIO()
                            plt.savefig(
                                byte_stream, format="png", bbox_inches="tight", dpi=100
                            )
                            byte_stream.seek(0)
                            img = Image.open(byte_stream)
                            img = np.array(img)[:, :, :1]
                            img = img / 255
                            img = np.array(img, dtype=np.float32)
                            byte_stream.truncate()
                            plt.close("all")
                            if img.shape != (36, IMG_WIDE, 1):
                                img = np.concatenate((img, img, img), 2)
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]
                            img_save[ii, :, :] = img
                        imgs_test[int(train_plot.sign_id[j])] = img_save

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eegshape = eeg.shape[1]
                eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
                if filter_range is not None:
                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = eeg[:, eegshape : eegshape * 2]
                eeg = np.array(eeg, dtype=np.float32)

                if "eeg" in DATATYPE:
                    eegs_test[int(eeg_id)] = eeg
                if "stft" in DATATYPE:
                    stfts_test[int(eeg_id)] = ss
                    stfts_test[-int(eeg_id)] = tt

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    batch_df = test.iloc[start_idx : i + 1, :].reset_index(drop=True)

                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )

                    if len(models) == 0:
                        if prior_by_patient is not None:
                            pid = batch_df["patient_id"].values
                            pred_list = []
                            for p in pid:
                                if p in prior_by_patient.index:
                                    pred_list.append(
                                        prior_by_patient.loc[p, TARGETS].values.astype(
                                            np.float32
                                        )
                                    )
                                else:
                                    pred_list.append(prior_global.astype(np.float32))
                            pred = np.stack(pred_list, axis=0)
                        else:
                            pred = np.tile(
                                prior_global.reshape(1, -1), (len(batch_df), 1)
                            ).astype(np.float32)
                    else:
                        preds = []
                        for m in models:
                            out = []
                            for bidx in range(len(test_gen)):
                                xb, _ = test_gen[bidx]
                                pb = m.predict_on_batch(xb)
                                pb = pb.numpy() if hasattr(pb, "numpy") else pb
                                out.append(pb)
                            preds.append(np.concatenate(out, axis=0))
                        pred = np.mean(preds, axis=0)

                    start_idx = i + 1

                    del eegs_test
                    gc.collect()
                    eegs_test = {}

                    preds_all_batches.append(pred)

            print()

        if len(preds_all_batches) == 0:
            if prior_global is None:
                preds_all = np.full(
                    (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
                )
            else:
                if prior_by_patient is not None:
                    pred_list = []
                    for p in test["patient_id"].values:
                        if p in prior_by_patient.index:
                            pred_list.append(
                                prior_by_patient.loc[p, TARGETS].values.astype(
                                    np.float32
                                )
                            )
                        else:
                            pred_list.append(prior_global.astype(np.float32))
                    preds_all = np.stack(pred_list, axis=0).astype(np.float32)
                else:
                    preds_all = np.tile(
                        prior_global.reshape(1, -1), (len(test), 1)
                    ).astype(np.float32)
        else:
            preds_all = np.concatenate(preds_all_batches, axis=0).astype(np.float32)

        preds_all = np.clip(preds_all, 1e-7, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub = sample_sub[["eeg_id"]].merge(sub, on="eeg_id", how="left")
        if sub[TARGETS].isna().any().any():
            fill = (
                prior_global
                if prior_global is not None
                else np.full((len(TARGETS),), 1.0 / len(TARGETS), dtype=np.float32)
            )
            sub[TARGETS] = sub[TARGETS].fillna(pd.Series(fill, index=TARGETS))
        arr = sub[TARGETS].values.astype(np.float32)
        arr = np.clip(arr, 1e-7, 1.0)
        arr = arr / np.sum(arr, axis=1, keepdims=True)
        sub[TARGETS] = arr

        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
        print("Saved to submission.csv")
