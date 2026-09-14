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

3.12

# 3. Installed packages

geopandas==0.14.4
librosa==0.11.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.6759589338987609

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the inference aggregation bug that leaves `test_pred` as a non-2D object (often because no model files were found or predictions weren’t stacked correctly), which causes the DataFrame constructor error. I also make model loading robust to common checkpoint formats (full model vs `state_dict`) without changing the model itself, and ensure the code falls back to a valid, properly-normalized submission (uniform probabilities) if no `.pt` files are available so a `.csv` is always produced. Finally, I enforce probability normalization (sum to 1) and correct column order to satisfy the competition submission validator; these changes are score-neutral unless the previous run produced an invalid submission.'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import tqdm



## === cell 1
FEATS2 = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

DATA_TYPE = "both"

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
test_raw = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)
print("Test shape (raw)", test_raw.shape)

if sample_sub["eeg_id"].duplicated().any():
    sample_sub = sample_sub.drop_duplicates(
        subset=["eeg_id"], keep="first"
    ).reset_index(drop=True)
if test_raw["eeg_id"].duplicated().any():
    test_raw = test_raw.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
        drop=True
    )

test = sample_sub[["eeg_id"]].merge(test_raw, on="eeg_id", how="left")
if test[["spectrogram_id", "patient_id"]].isna().any().any():
    missing = test.loc[test["spectrogram_id"].isna(), "eeg_id"].head(10).tolist()
    raise ValueError(
        f"Some eeg_id from sample_submission not found in test.csv, e.g. {missing}"
    )

test = test.reset_index(drop=True)
print("Test shape (aligned to sample_submission order)", test.shape)
assert test.shape[0] == sample_sub.shape[0]
assert np.array_equal(test["eeg_id"].values, sample_sub["eeg_id"].values)

PATH_SPEC = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH_SPEC)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values
print()

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()




## === cell 2
def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)))
    for j, col in enumerate(FEATS2):
        x = eeg[col].values.astype("float32")
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0
        data[:, j] = x
    return data


import librosa
import pywt, librosa


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((100, 300, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk + 1]].values
            m = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m)
            else:
                x1[:] = 0
            m = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m)
            else:
                x2[:] = 0

            x = x1 - x2

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 300,
                n_fft=1024,
                n_mels=100,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 30) * 30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.show()

    return img


missing_eeg = 0
for i, eeg_id in enumerate(EEG_IDS2):
    if i % 200 == 0:
        print(i, ", ", end="")
    eeg_path = f"{PATH_EEG}{eeg_id}.parquet"
    try:
        img = spectrogram_from_eeg(eeg_path, i < DISPLAY)
    except Exception:
        missing_eeg += 1
        img = np.zeros((100, 300, 4), dtype="float32")
    all_eegs2[eeg_id] = img
print()
if missing_eeg:
    print(
        f"Warning: {missing_eeg} EEG files failed to load; used zero fallback images."
    )




## === cell 3
class DataGenerator:
    "Generates data for Keras"

    def __init__(
        self,
        data,
        specs=None,
        eeg_specs=None,
        raw_eegs=None,
        augment=False,
        mode="train",
        data_type=DATA_TYPE,
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = self.data_generation(index)
        if self.augment:
            X = self.augmentation(X)
        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def data_generation(self, index):
        if self.data_type == "both":
            X, y = self.generate_all_specs(index)
        elif self.data_type == "eeg" or self.data_type == "kaggle":
            X, y = self.generate_specs(index)
        elif self.data_type == "raw":
            X, y = self.generate_raw(index)

        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        if self.data_type == "eeg":
            img = self.eeg_specs[row.eeg_id]
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]  # to match kaggle with eeg
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 1]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 1]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 3]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 2]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 4
def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm.tqdm(test_gen_both, total=len(test_gen_both)):
            batch_data_t = (
                torch.from_numpy(batch_data).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            out = model(batch_data_t)

            if isinstance(out, (tuple, list)):
                logits = out[0]
            elif isinstance(out, dict):
                logits = out.get("logits", None)
                if logits is None:
                    logits = next(iter(out.values()))
            else:
                logits = out

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_list.append(probs)

    if len(pred_list) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(pred_list, axis=0).astype(np.float32)


def train_one_epoch_fallback(
    model, train_df, specs, eeg_specs, device, lr=1e-4, max_steps=300
):
    model.to(device)
    model.train()

    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_gen = DataGenerator(
        train_df, mode="train", data_type="both", specs=specs, eeg_specs=eeg_specs
    )

    step = 0
    pbar = tqdm.tqdm(train_gen, total=min(len(train_gen), max_steps))
    for batch_data, y_prob in pbar:
        y_prob = y_prob.astype(np.float32)
        y_sum = y_prob.sum()
        if not np.isfinite(y_sum) or y_sum <= 0:
            continue
        y_prob = y_prob / y_sum
        y_prob_t = torch.from_numpy(y_prob).unsqueeze(0).to(device)  # (1,6)

        x_t = torch.from_numpy(batch_data).permute(2, 0, 1).unsqueeze(0).to(device)

        out = model(x_t)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                logits = next(iter(out.values()))
        else:
            logits = out

        log_probs = torch.log_softmax(logits, dim=1)

        loss = criterion(log_probs, y_prob_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        pbar.set_description(f"fallback-train loss={loss.item():.4f}")
        if step >= max_steps:
            break

    model.eval()
    return model


test_gen_both = DataGenerator(
    test, mode="test", data_type="both", specs=spectrograms2, eeg_specs=all_eegs2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

preds = []
model_paths = sorted(glob.glob("/kaggle/input/resnet50-test/resnet50_2/*.pt"))
print("Found model files:", len(model_paths))

for model_path in model_paths:
    print("Loading:", model_path)
    ckpt = torch.load(model_path, map_location=device)

    model = None
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        for key in ("model", "net", "module"):
            if key in ckpt and isinstance(ckpt[key], torch.nn.Module):
                model = ckpt[key]
                break

        if model is None:
            print(
                f"Skipping {model_path}: dict checkpoint without embedded torch.nn.Module."
            )
            continue
    else:
        print(f"Skipping {model_path}: unsupported checkpoint type {type(ckpt)}")
        continue

    pred = run_inference_loop(model, test_gen_both, device)
    preds.append(pred)

n_test = len(test)
n_classes = 6

if len(preds) == 0:
    train_meta = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    train_meta = train_meta.rename({"spectrogram_id": "spec_id"}, axis=1)

    train_meta["offset"] = train_meta["spectrogram_label_offset_seconds"].astype(
        np.int32
    )

    vote_sum = train_meta.groupby("eeg_id", as_index=False)[TARGETS].sum()
    vote_vals = vote_sum[TARGETS].values.astype(np.float32)
    vote_vals = np.clip(vote_vals, 0.0, None)
    row_sums = vote_vals.sum(axis=1, keepdims=True)
    probs_vals = np.where(row_sums > 0, vote_vals / row_sums, 1.0 / n_classes).astype(
        np.float32
    )
    vote_sum[TARGETS] = probs_vals

    rep = (
        train_meta.sort_values(["eeg_id", "offset"])
        .groupby("eeg_id", as_index=False)[["eeg_id", "spec_id", "offset"]]
        .first()
    )
    train_agg = rep.merge(vote_sum, on="eeg_id", how="inner", validate="one_to_one")

    train_agg = train_agg.reset_index(drop=True)

    train_subset = train_agg.sample(
        n=min(600, len(train_agg)), random_state=123
    ).reset_index(drop=True)

    PATH_TRAIN_SPEC = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    needed_spec_ids = train_subset["spec_id"].unique().tolist()
    train_specs = {}
    print(f"Loading {len(needed_spec_ids)} train spectrograms (subset) ...")
    for sid in tqdm.tqdm(needed_spec_ids):
        p = f"{PATH_TRAIN_SPEC}{sid}.parquet"
        try:
            tmp = pd.read_parquet(p)
            train_specs[int(sid)] = tmp.iloc[:, 1:].values
        except Exception:
            train_specs[int(sid)] = np.zeros((600, 400), dtype=np.float32)

    PATH_TRAIN_EEG = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    )
    needed_eeg_ids = train_subset["eeg_id"].unique().tolist()
    train_eeg_specs = {}
    print(f"Converting {len(needed_eeg_ids)} train EEGs to spectrograms (subset) ...")
    for eid in tqdm.tqdm(needed_eeg_ids):
        p = f"{PATH_TRAIN_EEG}{eid}.parquet"
        try:
            train_eeg_specs[int(eid)] = spectrogram_from_eeg(p, display=False)
        except Exception:
            train_eeg_specs[int(eid)] = np.zeros((100, 300, 4), dtype="float32")

    fallback_model = None
    candidate_paths = sorted(glob.glob("/kaggle/input/**/**/*.pt", recursive=True))
    for p in candidate_paths:
        try:
            ckpt = torch.load(p, map_location=device)
        except Exception:
            continue
        if isinstance(ckpt, torch.nn.Module):
            fallback_model = ckpt
            print("Using fallback embedded model from:", p)
            break
        if isinstance(ckpt, dict):
            for key in ("model", "net", "module"):
                if key in ckpt and isinstance(ckpt[key], torch.nn.Module):
                    fallback_model = ckpt[key]
                    print("Using fallback embedded model from:", p)
                    break
        if fallback_model is not None:
            break

    if fallback_model is None:
        print(
            "No loadable torch.nn.Module found anywhere in inputs; using uniform predictions."
        )
        test_pred = np.full((n_test, n_classes), 1.0 / n_classes, dtype=np.float32)
    else:
        fallback_model = train_one_epoch_fallback(
            fallback_model,
            train_subset[["eeg_id", "spec_id", "offset"] + TARGETS],
            train_specs,
            train_eeg_specs,
            device,
            lr=1e-4,
            max_steps=250,
        )
        test_pred = run_inference_loop(fallback_model, test_gen_both, device)

        if test_pred.shape != (n_test, n_classes):
            print("Fallback produced unexpected shape; reverting to uniform.")
            test_pred = np.full((n_test, n_classes), 1.0 / n_classes, dtype=np.float32)
else:
    pred_stack = np.stack(preds, axis=0)  # (n_models, n_test, 6)
    test_pred = pred_stack.mean(axis=0)

if not (
    isinstance(test_pred, np.ndarray)
    and test_pred.ndim == 2
    and test_pred.shape == (n_test, n_classes)
):
    raise ValueError(
        f"Invalid prediction array shape: {type(test_pred)=}, {getattr(test_pred, 'shape', None)=}"
    )

test_pred = np.clip(test_pred, 1e-12, None)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df.insert(0, "eeg_id", test["eeg_id"].values)

test_pred_df = sample_sub[["eeg_id"]].merge(
    test_pred_df, on="eeg_id", how="left", validate="one_to_one"
)

if test_pred_df[CLASSES].isna().any().any():
    raise ValueError(
        "NaNs found after aligning to sample_submission; check eeg_id alignment."
    )

probs = test_pred_df[CLASSES].values.astype(np.float32)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
test_pred_df[CLASSES] = probs

test_pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_pred_df.shape)
print("Row sums (min/max):", probs.sum(axis=1).min(), probs.sum(axis=1).max())
print(test_pred_df.head())
