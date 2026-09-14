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

0.3323337130116107

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.39779) has done: 'The script now safely handles missing TensorFlow/weight files, skips unavailable pre‑processed arrays, and falls back to a simple prior‑based prediction (the average class proportions from the training set) so a valid `submission.csv` is always written. This fixes the import error, the missing `.npy` files, and the missing model weights while keeping the original logic untouched when everything is available.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024
@author: yuri
"""
import os, warnings, gc, time
import numpy as np, pandas as pd
from scipy import signal

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
except Exception as e:
    class _DummySeq:
        pass

    class _DummyUtils:
        Sequence = _DummySeq

    class _DummyKeras:
        utils = _DummyUtils()

    class _DummyTF:
        keras = _DummyKeras()

    tf = _DummyTF()
    optimizers = type("opt", (), {"schedules": type("sch", (), {})})

    def clone_model(x):
        return x


if hasattr(tf, "config"):
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy()
    np.random.seed(2024)
    os.environ["PYTHONHASHSEED"] = str(2024)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(2024)
    tf.keras.utils.set_random_seed(2024)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

PLATFORM = "kaggle"
NEEDTRAIN = False
DATATYPE = ["eeg"]  # keep original setting
LOAD_MODELS_FROM = "models20241114a"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 200
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 20
EEG_LENGTH_USED = 50
SPE_HIGH, SPE_WIDE = 100, 256
STFT_HIGH, STFT_WIDE = 32, round(50 / 0.4)
IMG_HIGH, IMG_WIDE = 324, 324
filter_range = [0.5, 45]
filter_range2 = [0.1, 35]
BATCHSIZE = 8
TEST_BATCHSIZE = 32
SPLITS = 5

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # seizure, lpd, gpd, lrda, grda, other
row_sums = df[TARGETS].sum(axis=1).replace(0, np.nan)
norm_votes = df[TARGETS].div(row_sums, axis=0).fillna(0)
PRIOR = norm_votes.mean(axis=0).values  # shape (6,)

print("Train shape:", df.shape)
print("Targets:", list(TARGETS))
print("Prior probabilities:", PRIOR)



## === cell 2
spectrograms = {}
eegs = {}
stfts = {}
imgs = {}

if not NEEDTRAIN:
    if PLATFORM == "local":
        datapath = "./input/preprocess"
    else:
        datapath = "/kaggle/input/preprocess"

    def _load_npy(name):
        try:
            return np.load(
                os.path.join(datapath, f"{name}.npy"), allow_pickle=True
            ).item()
        except Exception:
            return {}

    if "eeg" in DATATYPE:
        eegs = _load_npy("eegs")
    if "stft" in DATATYPE:
        stfts = _load_npy("stfts")
    if "img" in DATATYPE:
        imgs = _load_npy("imgs")
    if "spe" in DATATYPE:
        spectrograms = _load_npy("spectrograms")




## === cell 3
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
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.mode = mode
        self.eegs = eegs
        self.stfts = stfts
        self.specs = specs
        self.imgs = imgs
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, y, sample_weights = self.__data_generation(indexes)
        return x, y, sample_weights

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if "spe" in DATATYPE:
            x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    len(indexes),
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype="float32",
            )
        if "stft" in DATATYPE:
            x_stft = np.zeros(
                (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]
            sign_id = row.sign_id
            if self.mode != "test":
                sample_weight = sum(row[TARGETS_RAW].values) / 20

            if self.mode == "test":
                r_spe = r_eeg = r_stft = 0
            else:
                rows = df.loc[
                    (df.eeg_id == row.eeg_id)
                    * (df.seizure_vote == row.seizure_vote_raw)
                    * (df.lpd_vote == row.lpd_vote_raw)
                    * (df.gpd_vote == row.gpd_vote_raw)
                    * (df.lrda_vote == row.lrda_vote_raw)
                    * (df.grda_vote == row.grda_vote_raw),
                    :,
                ].reset_index(drop=True)
                if self.mode == "train":
                    rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                        drop=True
                    )
                    row = rows.loc[0, :]
                elif self.mode == "valid":
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )
                r_spe = round(row.spectrogram_label_offset_seconds / 2)
                r_eeg = row.eeg_label_offset_seconds

            if "spe" in DATATYPE:
                spe = []
                for k in range(4):
                    spe.append(
                        np.reshape(
                            self.specs[row.spectrogram_id][
                                r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                            ].T,
                            (1, 100, 300),
                        )
                    )
                spe = np.concatenate(spe, axis=0)

            if "eeg" in DATATYPE:
                eeg = self.eegs[row.eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]

            if "stft" in DATATYPE:
                stft_t = self.stfts[-row.eeg_id]
                r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                if stft.shape[2] < STFT_WIDE:
                    stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                    stft = stft[:, :, :STFT_WIDE]

            if "img" in DATATYPE:
                img = self.imgs[sign_id]

            if "spe" in DATATYPE:
                spe = np.nan_to_num(spe, nan=0.0)
                spe = np.clip(spe, 1e-6, 1e6)
                spe = np.log2(spe)
                spe = spe[
                    :,
                    :,
                    round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                        (spe.shape[2] - SPE_WIDE) / 2
                    ),
                ]
                spe = spe[[0, 2, 3, 1], :, :]
                spe = (spe - np.mean(spe, keepdims=True)) / (
                    np.std(spe, keepdims=True) + 1e-6
                )
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]
                eeg_save = np.zeros((x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32)
                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )
                for ii in range(eeg_save.shape[0]):
                    eeg_save[ii, :] = eeg[
                        ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                    ]
                eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                    np.std(eeg_save, keepdims=True) + 1e-6
                )
                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                stft = np.nan_to_num(stft, nan=0.0)
                stft = np.clip(stft, 1e-6, 1e6)
                stft = np.log2(stft)
                stft_save = np.zeros(
                    (round(stft.shape[0] / 2 * stft.shape[1]), stft.shape[2] * 2),
                    dtype=np.float32,
                )
                for ii in range(stft.shape[0]):
                    stft_save[
                        ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                        (ii % 2) * stft.shape[2] : (ii % 2 + 1) * stft.shape[2],
                    ] = stft[ii, :, :]
                stft = (stft_save - np.mean(stft_save, keepdims=True)) / (
                    np.std(stft_save, keepdims=True) + 1e-6
                )
                x_stft[j] = stft

            if "img" in DATATYPE:
                img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)
                img = np.reshape(img_save, (IMG_HIGH, IMG_WIDE, 1))
                img = np.concatenate((img, img, img), -1)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                sample_weights[j] = sample_weight if self.sample_weights else 1

        x = []
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "stft" in DATATYPE:
            x.append(x_stft)
        if "img" in DATATYPE:
            x.append(x_img)
        return x, y, sample_weights




## === cell 4
def build_model():
    inp = []
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
        x_spe = tf.keras.layers.Concatenate(axis=1)(
            [
                inp_spe[:, 0, :, :],
                inp_spe[:, 1, :, :],
                inp_spe[:, 2, :, :],
                inp_spe[:, 3, :, :],
            ]
        )
        x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
        x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])
        base_model_spe = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_spe._name = "spe_extractor"
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            )
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])
        base_model_eeg = efn.EfficientNetB5(
            include_top=False, weights=None, input_shape=None
        )
        base_model_eeg._name = "eeg_extractor"
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)
        inp.append(inp_eeg)
        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
        else:
            y = x_eeg

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
        x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
            inp_stft
        )
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])
        base_model_stft = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_stft._name = "stft_extractor"
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
        inp.append(inp_stft)
        if any(t in DATATYPE for t in ("spe", "eeg")):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
        else:
            y = x_stft

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
        base_model_img = efn.EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_img._name = "img_extractor"
        x_img = base_model_img(inp_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        inp.append(inp_img)
        if any(t in DATATYPE for t in ("spe", "eeg", "stft")):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
        else:
            y = x_img

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    return tf.keras.Model(inputs=inp, outputs=y)




## === cell 5
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape:", test.shape)

use_prior = False
models = []
if os.path.isdir(LOAD_MODELS_FROM):
    try:
        model_template = build_model()
        for model_i in range(SPLITS):
            weight_path = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
            if not os.path.isfile(weight_path):
                raise FileNotFoundError
            model = clone_model(model_template)
            model.load_weights(weight_path)
            models.append(model)
    except Exception as e:
        print("Model loading failed:", e)
        use_prior = True
else:
    use_prior = True

if use_prior:
    preds_all = np.tile(PRIOR, (len(test), 1))
    print("Using prior-based predictions.")
else:
    preds_all = None
    if "spe" in DATATYPE:
        PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
        files_test = os.listdir(PATH_test)
        print(f"There are {len(files_test)} test spectrogram parquets")
        for f in files_test:
            tmp = pd.read_parquet(os.path.join(PATH_test, f))
            name = int(f.split(".")[0])
            spectrograms_test[name] = tmp.iloc[:, 1:].values

    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
    if any(t in DATATYPE for t in ("eeg", "stft", "img")):
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")
        for i, eeg_id in enumerate(test.eeg_id):
            if i % 100 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(os.path.join(PATH_test, f"{eeg_id}.parquet"))
            eeg = []
            for channel in BRAIN:
                ch0, ch1 = channel.split("-")
                eeg_temp = (eeg_default.loc[:, ch0] - eeg_default.loc[:, ch1]).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(eeg_temp.reshape(1, -1))
            eeg = np.concatenate(eeg, axis=0)
            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                ff, tt, ss = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) & (ff <= 20), :]

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, -1024, 1024)
                img = np.zeros((EEG_CHANNEL_USED, IMG_HIGH, IMG_WIDE), dtype=np.float32)
                imgs_test[test.loc[i, "sign_id"]] = img

            eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.clip(eeg, -1024, 1024)

            if "eeg" in DATATYPE:
                eegs_test[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts_test[eeg_id] = ss
                stfts_test[-eeg_id] = tt

            if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test)):
                batch_start = max(i - TEST_BATCHSIZE + 1, 0)
                batch_df = test.iloc[batch_start : i + 1]
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
                batch_preds = [m.predict(test_gen, verbose=0) for m in models]
                pred = np.mean(batch_preds, axis=0)
                eegs_test.clear()
                gc.collect()
                if preds_all is None:
                    preds_all = pred.copy()
                else:
                    preds_all = np.concatenate((preds_all, pred), axis=0)

preds_all = np.clip(preds_all, 0, 1)
preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = preds_all
sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
print(sub.head())
