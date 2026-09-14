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

0.272543691039171

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
"""
Environment setup, imports, constants, and loading of training metadata.
- Added missing imports (numpy, pandas, tensorflow, scipy.signal).
- Fixed optimizers import.
- Defined paths, constants, and the list of EEG channel pairs (BRAIN).
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
from tensorflow.keras import optimizers

warnings.filterwarnings("ignore")

SEED = 2024
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    base_input = "./input"
else:
    PLATFORM = "kaggle"
    base_input = "/kaggle/input"

LOAD_MODELS_FROM = "modelsxxxxxxx"
for d in os.listdir(base_input):
    if d.startswith("models"):
        LOAD_MODELS_FROM = d
        break

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

DATATYPE = ["eeg"]

print("DATATYPE:", DATATYPE)

SFREQ = 200
RSFREQ = 200
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
TEST_BATCHSIZE = 128
filter_range = [0.5, 45]

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # vote columns
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

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
"""
DataGenerator for inference.
- Returns only the input dictionary `x` when mode == "test".
- Keeps the original preprocessing logic for EEG data.
"""


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        batch_size=32,
        shuffle=False,
        mode="test",
        eegs=None,
        specs=None,
        stfts=None,
        imgs=None,
    ):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.eegs = eegs
        self.specs = specs
        self.stfts = stfts
        self.imgs = imgs
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        x, _, _ = self.__data_generation(indexes)
        return x

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if "eeg" in DATATYPE:
            x_eeg = np.zeros(
                (
                    len(indexes),
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                dtype="float32",
            )
        y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
        sample_weights = np.zeros((len(indexes), 1), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.dataframe.iloc[i]

            if "eeg" in DATATYPE:
                eeg = self.eegs[row.eeg_id][
                    :,
                    round(row.eeg_label_offset_seconds * RSFREQ) : round(
                        (row.eeg_label_offset_seconds + 50) * RSFREQ
                    ),
                ]
                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )
                eeg = np.clip(eeg, -1024, 1024)
                eeg = (eeg + 1024) / 2048 * 255
                x_eeg[j] = eeg

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                sample_weights[j] = 1

        x = {}
        if "eeg" in DATATYPE:
            x["eeg"] = x_eeg
        return x, y, sample_weights




## === cell 2
"""
Cosine annealing learning‑rate scheduler (kept unchanged).
"""


class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super().__init__()
        self.total_step = total_step
        self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.begin = 1

    def __call__(self, step):
        if step == self.total_step:
            self.begin = 0
            self.lr_max *= 0.5
            self.lr_min *= 0.1
        step = step % self.total_step + 1
        if self.begin == 1 and step < self.warm_step:
            lr = self.lr_max / self.warm_step * step
        else:
            if self.begin == 1:
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




## === cell 3
"""
Model construction.
- Fixed tensor dimensions so the EEG tensor is 4‑D before EfficientNetV2B3.
"""


def build_model():
    inp = []
    y = None

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
        eeg_embed = tf.keras.layers.Conv1D(
            filters=strides * 3,
            kernel_size=strides,
            strides=strides,
            padding="same",
            use_bias=False,
            activation=None,
        )
        x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)  # (B, ch, t', f)

        x_eeg = tf.keras.layers.Permute((2, 1, 3))(x_eeg)  # (B, t', ch, f)

        base_model_eeg = tf.keras.applications.EfficientNetV2B3(
            include_top=False,
            weights=None,
            input_shape=(
                int(round(EEG_LENGTH_USED * RSFREQ / strides)),  # time' dimension
                EEG_CHANNEL_USED,
                strides * 3,
            ),
        )
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        inp.append(inp_eeg)
        y = x_eeg if y is None else tf.keras.layers.Concatenate(axis=1)([y, x_eeg])

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    return tf.keras.Model(inputs=inp, outputs=y)




## === cell 4
"""
Main inference script:
- Loads any pretrained fold weights if present; otherwise falls back to uniform predictions.
- Reads each test EEG file, builds batches, runs the model(s), aggregates predictions,
  normalises them, and writes `submission.csv`.
"""

if __name__ == "__main__":
    preds_all = []
    models = []

    model_template = build_model()

    for model_i in range(100):
        weight_path = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
        if os.path.exists(weight_path):
            print(f"Loading weights for fold {model_i + 1}")
            m = tf.keras.models.clone_model(model_template)
            m.load_weights(weight_path)
            models.append(m)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape:", test.shape)

    if not models:
        print("No pretrained models found – using uniform probabilities.")
        uniform_prob = np.full(
            (len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
        )
        preds_all = uniform_prob
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
        eegs_test = {}

        for i, eeg_id in enumerate(test.eeg_id):
            if i % 100 == 0:
                print(i, ", ", end="")

            eeg_raw = pd.read_parquet(os.path.join(PATH_test, f"{eeg_id}.parquet"))

            eeg = []
            for pair in BRAIN:
                ch_a, ch_b = pair.split("-")
                diff = eeg_raw.loc[:, ch_a] - eeg_raw.loc[:, ch_b]
                diff = diff.fillna(0).values
                eeg.append(diff.reshape(1, -1))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.clip(eeg, -1024, 1024)
            eeg = (eeg + 1024) / 2048 * 255

            eegs_test[eeg_id] = eeg

            if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                start_idx = max(i - TEST_BATCHSIZE + 1, 0)
                test_batch = test.loc[start_idx:i, :]

                test_gen = DataGenerator(
                    test_batch,
                    batch_size=TEST_BATCHSIZE,
                    shuffle=False,
                    mode="test",
                    eegs=eegs_test,
                )

                batch_preds = [m.predict(test_gen, verbose=0) for m in models]
                batch_mean = np.mean(batch_preds, axis=0)

                if isinstance(preds_all, list) and len(preds_all) == 0:
                    preds_all = batch_mean.copy()
                else:
                    preds_all = np.concatenate((preds_all, batch_mean), axis=0)

                eegs_test = {}
                gc.collect()

    preds_all = np.clip(preds_all, 0, 1)
    row_sums = preds_all.sum(axis=1, keepdims=True)
    preds_all = preds_all / np.where(row_sums == 0, 1, row_sums)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = preds_all
    sub.to_csv("submission.csv", index=False)
    print("Submission shape:", sub.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2908521717.py in <cell line: 0>()
     10     models = []
     11 
---> 12     model_template = build_model()
     13 
     14     for model_i in range(100):

/tmp/ipykernel_55/2100978572.py in build_model()
     38 
     39         # EfficientNetV2B3 expects (height, width, channels)
---> 40         base_model_eeg = tf.keras.applications.EfficientNetV2B3(
     41             include_top=False,
     42             weights=None,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet_v2.py in EfficientNetV2B3(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)
   1203     name="efficientnetv2-b3",
   1204 ):
-> 1205     return EfficientNetV2(
   1206         width_coefficient=1.2,
   1207         depth_coefficient=1.4,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet_v2.py in EfficientNetV2(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, min_depth, bn_momentum, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, weights_name)
    909 
    910     # Determine proper input shape
--> 911     input_shape = imagenet_utils.obtain_input_shape(
    912         input_shape,
    913         default_size=default_size,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/imagenet_utils.py in obtain_input_shape(input_shape, default_size, min_size, data_format, require_flatten, weights)
    388                     input_shape[0] is not None and input_shape[0] < min_size
    389                 ) or (input_shape[1] is not None and input_shape[1] < min_size):
--> 390                     raise ValueError(
    391                         "Input size must be at least "
    392                         f"{min_size}x{min_size}; Received: "

ValueError: Input size must be at least 32x32; Received: input_shape=(1000, 16, 30)
