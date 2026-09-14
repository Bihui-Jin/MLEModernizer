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

0.5383913144990755

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, matplotlib
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from scipy import signal

PLATFORM = "kaggle"  # 'local' or 'kaggle'
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402182"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # seconds
SFREQ = 100
HIGH = 128
LENGTH = 256
filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

try:
    import albumentations as albu
except Exception:
    albu = None  # augmentation won't be used in this script

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]  # ['seizure_vote', ..., 'other_vote']


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data,
        batch_size=32,
        shuffle=False,
        augment=False,
        mode="train",
        specs=None,
        eegs=None,
        df=None,
    ):
        self.mode = mode
        if mode != "test":
            self.df = df.merge(data.iloc[:, :6], on="eeg_id", how="inner").reset_index(
                drop=True
            )
        else:
            self.df = data.copy()
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.specs = specs
        self.eegs = eegs
        self.cmin, self.cmax = -4, 6
        self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
        self.b, self.a = signal.butter(
            3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        if self.mode == "test":
            return [X, X_eeg]  # Keras predict expects only inputs
        else:
            return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")
        for j, i in enumerate(indexes):
            if self.mode == "test":
                row1 = self.data.iloc[i]
                rows = self.df[self.df.patient_id == row1.patient_id].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row2 = rows.iloc[0]
            else:
                class_order = ["Seizure", "GPD", "LRDA", "Other", "GRDA", "LPD"]
                target = class_order[min(j // (self.batch_size // 6), 5)]
                rows = self.df[self.df.expert_consensus == target].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row1 = rows.iloc[0]
                rows = self.df[self.df.patient_id_x == row1.patient_id_x].reset_index(
                    drop=True
                )
                rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                    drop=True
                )
                row2 = rows.iloc[0]
                mixup1, mixup2 = 0.5, 0.5
                label = (
                    row1[TARGETS].values / row1[TARGETS].sum() * mixup1
                    + row2[TARGETS].values / row2[TARGETS].sum() * mixup2
                )
                y[j] = label

            for k in range(4):
                if self.mode == "test":
                    spec1 = self.specs[row1.spec_id][0:300, k * 100 : (k + 1) * 100].T
                    spec2 = self.specs[row2.spec_id][0:300, k * 100 : (k + 1) * 100].T
                else:
                    start1 = round(row1.spectrogram_label_offset_seconds / 2)
                    start2 = round(row2.spectrogram_label_offset_seconds / 2)
                    spec1 = self.specs[row1.spectrogram_id][
                        start1 : start1 + 300, k * 100 : (k + 1) * 100
                    ].T
                    spec2 = self.specs[row2.spectrogram_id][
                        start2 : start2 + 300, k * 100 : (k + 1) * 100
                    ].T
                spec1 = np.nan_to_num(spec1, 0.0)
                spec2 = np.nan_to_num(spec2, 0.0)
                spec1 = np.clip(spec1, np.exp(self.cmin), np.exp(self.cmax))
                spec2 = np.clip(spec2, np.exp(self.cmin), np.exp(self.cmax))
                spec1 = np.log(spec1)
                spec2 = np.log(spec2)
                img = (
                    (spec1 + spec2) * 0.5
                    if self.mode == "test"
                    else spec1 * mixup1 + spec2 * mixup2
                )
                img = img[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        round((600 / 2 - LENGTH) / 2) + LENGTH, img.shape[1]
                    ),
                ]
                if HIGH != 100:
                    img = (
                        tf.image.resize(tf.expand_dims(img, -1), ((HIGH - 32), LENGTH))
                        .numpy()
                        .squeeze()
                    )
                X[j, :, :, k] = img
                X[j, :, :, k] = (X[j, :, :, k] - np.mean(X[j, :, :, k])) / (
                    np.std(X[j, :, :, k]) + 1e-6
                )

                if self.mode == "test":
                    eeg1 = self.eegs[row1.eeg_id][:, :, k]
                    eeg2 = self.eegs[row2.eeg_id][:, :, k]
                else:
                    start1 = round(row1.eeg_label_offset_seconds * SFREQ)
                    start2 = round(row2.eeg_label_offset_seconds * SFREQ)
                    eeg1 = self.eegs[row1.eeg_id][:, start1 : start1 + 50 * SFREQ, k]
                    eeg2 = self.eegs[row2.eeg_id][:, start2 : start2 + 50 * SFREQ, k]
                eeg1 = np.nan_to_num(eeg1, 0.0)
                eeg2 = np.nan_to_num(eeg2, 0.0)
                eeg = (
                    (eeg1 + eeg2) * 0.5
                    if self.mode == "test"
                    else eeg1 * mixup1 + eeg2 * mixup2
                )
                eeg = signal.filtfilt(self.b, self.a, eeg, axis=1)
                eeg = eeg[
                    :,
                    round((50 - EEG_LENGTH) / 2 * SFREQ) : round(
                        (50 + EEG_LENGTH) / 2 * SFREQ
                    ),
                ]
                for ch in range(4):
                    X_eeg[j, ch + 1, :, k] = eeg[ch, :]
                X_eeg[j, :, :, k] = (
                    X_eeg[j, :, :, k]
                    - np.mean(X_eeg[j, :, :, k], axis=1, keepdims=True)
                ) / (np.std(X_eeg[j, :, :, k], axis=1, keepdims=True) + 1e-6)
        return X, X_eeg, y

    def __random_transform(self, img):
        if albu is None:
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
            img_batch[i] = self.__random_transform(img_batch[i])
        return img_batch


def build_model():
    inp = layers.Input(shape=(HIGH, LENGTH, 4), name="spec_input")
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    inp_eeg = layers.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4), name="eeg_input")
    y = layers.Reshape((6, round(EEG_LENGTH * SFREQ), 4))(inp_eeg)
    y = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(y)
    y = layers.MaxPooling2D((2, 2))(y)
    y = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(y)
    y = layers.GlobalAveragePooling2D()(y)

    combined = layers.Concatenate()([x, y])
    output = layers.Dense(6, activation="softmax", dtype="float32")(combined)

    model = models.Model(inputs=[inp, inp_eeg], outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.KLDivergence()
    )
    return model


if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    spect_path = (
        "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        if PLATFORM == "local"
        else "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    spec_files = os.listdir(spect_path)
    print(f"There are {len(spec_files)} test spectrogram parquets")
    spectrograms2 = {}
    for i, f in enumerate(spec_files):
        if i % 100 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(os.path.join(spect_path, f))
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values

    test = test.rename(columns={"spectrogram_id": "spec_id"})

    eeg_path = (
        "./input/hms-harmful-brain-activity-classification/test_eegs/"
        if PLATFORM == "local"
        else "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    )
    eeg_files = os.listdir(eeg_path)
    print(f"\nThere are {len(eeg_files)} test eeg parquets")
    eegs2 = {}
    for i, f in enumerate(eeg_files):
        if i % 100 == 0:
            print(i, ", ", end="")
        eeg_default = pd.read_parquet(os.path.join(eeg_path, f))
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
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))
            eegs2[name] = np.concatenate(list_eeg, 2)

    model = build_model()
    test_gen = DataGenerator(
        test,
        shuffle=False,
        batch_size=32,
        mode="test",
        specs=spectrograms2,
        eegs=eegs2,
        df=test,
    )

    preds = []
    for i in range(5):
        ckpt_path = os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
        if os.path.isfile(ckpt_path):
            print(f"Loading checkpoint {ckpt_path}")
            model.load_weights(ckpt_path)
        else:
            print(f"Checkpoint {ckpt_path} not found – using random weights")
        pred = model.predict(test_gen, verbose=1)
        preds.append(pred)

    pred = np.mean(preds, axis=0)
    print("Test preds shape", pred.shape)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub_path = "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Submission written to {sub_path}, shape {sub.shape}")
    row_sums = sub[TARGETS].sum(axis=1)
    print("Row sum stats:", row_sums.min(), row_sums.max(), row_sums.mean())

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
