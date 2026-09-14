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

0.5243078618326658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PLATFORM = "kaggle"  # local or kaggle platform
NEEDTRAIN = False  # training not required for inference
READ_SPEC_FILES = False
READ_EEG_FILES = False
LOAD_MODELS_FROM = "models20240130"

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # seconds
SFREQ = 100  # resampled frequency
HIGH = 512  # spectrogram freq dimension
LENGTH = 256  # spectrogram time dimension

CONVERTIMAGE = False
CONVERTIMAGE_EEG = False

filter_range = [0.5, 40]  # EEG band‑pass range
BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}
VER = 1

import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy import signal

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

MIX = True
if MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
else:
    print("Using full precision")

try:
    import efficientnet.tfkeras as efn

    EfficientNetB2 = efn.EfficientNetB2
    EfficientNetB1 = efn.EfficientNetB1
    print("Using external efficientnet package")
except Exception as e:
    from tensorflow.keras.applications import EfficientNetB2, EfficientNetB1

    efn = None
    print("External efficientnet import failed, using tf.keras applications:", e)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()

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
print("Train non‑overlap eeg_id shape:", train.shape)
train.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


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
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        if CONVERTIMAGE:
            X = np.zeros((len(indexes), HIGH, LENGTH, 3), dtype="float32")
        else:
            X = np.zeros((len(indexes), HIGH, LENGTH), dtype="float32")

        if CONVERTIMAGE_EEG:
            X_eeg = np.zeros((len(indexes), HIGH, LENGTH, 3), dtype="float32")
        else:
            X_eeg = np.zeros(
                (len(indexes), 16, round(EEG_LENGTH * SFREQ)), dtype="float32"
            )

        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == "test":
                spec_start = 0
                eeg_start = 0
            else:
                rows = df[df.eeg_id == row.eeg_id].reset_index(drop=True)
                row = rows.iloc[np.random.randint(len(rows))]
                spec_start = round(row.spectrogram_label_offset_seconds / 2)
                eeg_start = round(row.eeg_label_offset_seconds) + (50 - EEG_LENGTH) / 2

            img = self.specs[row.spectrogram_id][spec_start : (spec_start + 300), :]
            img = np.nan_to_num(img, nan=0.0)

            LL = img[:, :100]
            RL = img[:, 100:200]
            LP = img[:, 200:300]
            RP = img[:, 300:]

            for part in (LL, RL, LP, RP):
                np.clip(part, np.exp(-6), np.exp(8), out=part)
                np.log(part, out=part)

            if CONVERTIMAGE:
                img = np.concatenate((LL, LP, RP, RL), 1)
                img = np.nan_to_num(img, nan=0.0)
                plt.figure()
                img = (
                    plt.imshow(img)
                    .get_figure()
                    .gca()
                    .images[0]
                    .make_image(renderer=None)[0][:, :, :3]
                )
                img = np.array(tf.image.resize(img, (HIGH, LENGTH)), dtype=np.uint8)
                img = img / 255
                plt.close()
                gc.collect()
            else:
                img = np.zeros((HIGH, LENGTH), dtype="float32")
                resize_temp = 96

                def place(part, idx):
                    resized = np.array(
                        tf.image.resize(
                            np.reshape(part, (part.shape[0], part.shape[1], 1)),
                            (resize_temp, LENGTH),
                        ),
                        dtype=np.float32,
                    )[:, :, 0]
                    start = round(HIGH / 4 * idx + (HIGH / 4 - resize_temp) / 2)
                    end = round(HIGH / 4 * idx + (HIGH / 4 + resize_temp) / 2)
                    img[start:end, :] = resized

                place(LL, 0)
                place(RL, 1)
                place(LP, 2)
                place(RP, 3)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)

            eeg = self.eegs[row.eeg_id][
                :, round(eeg_start * SFREQ) : round((eeg_start + EEG_LENGTH) * SFREQ)
            ][:, :, 0]
            eeg = signal.filtfilt(b, a, eeg, axis=1)

            if CONVERTIMAGE_EEG:
                fig, ax = plt.subplots()
                for i in range(eeg.shape[0]):
                    plt.plot(eeg[i, :] + i * 100, color="black", linewidth=1)
                canvas = FigureCanvas(fig)
                plt.xlim([0, eeg.shape[1]])
                plt.ylim([0 - 25, i * 100 + 25])
                ax.set_aspect("equal", adjustable="box")
                plt.axis("off")
                canvas.draw()
                eeg = np.array(canvas.renderer.buffer_rgba())[:, :, :3]
                plt.close()
                gc.collect()
                eeg = np.array(tf.image.resize(eeg, (HIGH, LENGTH)), dtype=np.uint8)
                eeg = eeg / 255
            else:
                eeg = (eeg - np.mean(eeg, 1, keepdims=True)) / (
                    np.std(eeg, 1, keepdims=True) + 1e-6
                )

            X[j] = img
            X_eeg[j] = eeg

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)

        return X, X_eeg, y




## === cell 2
def build_model(CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ):
    if CONVERTIMAGE:
        inp = tf.keras.Input(shape=(HIGH, LENGTH, 3))
        x = inp
    else:
        inp = tf.keras.Input(shape=(HIGH, LENGTH))
        x = tf.keras.layers.Reshape((inp.shape[1], inp.shape[2], 1))(inp)
        x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

    base_model = EfficientNetB2(include_top=False, weights="imagenet")
    base_model._name = "spectrogram_extractor"
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if CONVERTIMAGE_EEG:
        inp_eeg = tf.keras.Input(shape=(HIGH, LENGTH, 3))
        x_eeg = inp_eeg
    else:
        inp_eeg = tf.keras.Input(shape=(16, round(EEG_LENGTH * SFREQ)))
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

    base_model_eeg = EfficientNetB1(include_top=False, weights="imagenet")
    base_model_eeg._name = "eeg_extractor"
    x_eeg = base_model_eeg(x_eeg)
    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)

    x = tf.nn.l2_normalize(x, -1)
    x_eeg = tf.nn.l2_normalize(x_eeg, -1)

    x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
    x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model




## === cell 3
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    else:
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test spectrogram parquets")
    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test EEG parquets")
    eegs2 = {}
    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        raw_eeg = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])
        if len(test[test.eeg_id == name]) > 0:
            eeg_default = raw_eeg.loc[:, :].reset_index(drop=True)
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
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))
            list_eeg = np.concatenate(list_eeg, 0)
            eegs2[name] = list_eeg

    preds = []
    model = build_model(CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ)
    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
    )
    for i in range(5):
        print(f"Fold {i+1}")
        weight_path = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
        model.load_weights(weight_path)
        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    print("\nTest preds shape", pred.shape)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission saved as submission.csv")
    print("Submission shape", sub.shape)
    print("Probability sums (should be 1.0):", sub[TARGETS].sum(axis=1).head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/62605107.py in <cell line: 0>()
     60     # Ensemble inference over 5 folds
     61     preds = []
---> 62     model = build_model(CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ)
     63     test_gen = DataGenerator(
     64         test,

/tmp/ipykernel_55/86099575.py in build_model(CONVERTIMAGE, HIGH, LENGTH, CONVERTIMAGE_EEG, EEG_LENGTH, SFREQ)
     29     x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
     30 
---> 31     x = tf.nn.l2_normalize(x, -1)
     32     x_eeg = tf.nn.l2_normalize(x_eeg, -1)
     33 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```
