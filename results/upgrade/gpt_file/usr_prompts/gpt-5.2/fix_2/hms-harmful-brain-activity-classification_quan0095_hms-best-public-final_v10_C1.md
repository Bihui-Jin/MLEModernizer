# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]



## === cell 3
from scipy import signal



## === cell 4
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
TEST_CSV = f"{BASE_PATH}/test.csv"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TEST_SPEC_PATH = f"{BASE_PATH}/test_spectrograms/"
TEST_EEG_PATH = f"{BASE_PATH}/test_eegs/"
TRAIN_SPEC_PATH = f"{BASE_PATH}/train_spectrograms/"
TRAIN_EEG_PATH = f"{BASE_PATH}/train_eegs/"

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG:
    test = pd.read_csv(TRAIN_CSV).head(40)
    SPEC_PATH = TRAIN_SPEC_PATH
    EEG_PATH = TRAIN_EEG_PATH
else:
    test = pd.read_csv(TEST_CSV)
    SPEC_PATH = TEST_SPEC_PATH
    EEG_PATH = TEST_EEG_PATH

print("test:", test.shape)



## === cell 5
spec_directory_path = "spec_spectrograms/"
eeg_directory_path = "eeg_spectrograms/"
raw_10s_directory_path = "eeg_10s_raws/"
raw_50s_directory_path = "eeg_50s_raws/"

for p in [
    spec_directory_path,
    eeg_directory_path,
    raw_10s_directory_path,
    raw_50s_directory_path,
]:
    os.makedirs(p, exist_ok=True)



## === cell 6
SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")




## === cell 7
def stft_spec_from_eeg(parquet_path):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = []
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]].copy()
            mean_value = float(eeg_1.mean())
            eeg_1 = eeg_1.fillna(mean_value).values

            eeg_2 = eeg[COLS[kk + 1]].copy()
            mean_value = float(eeg_2.mean())
            eeg_2 = eeg_2.fillna(mean_value).values

            new_eeg = eeg_1 - eeg_2
            fs = 200
            nperseg = 70
            noverlap = 0
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )

            spec = np.abs(spec)
            spec = np.log1p(spec).astype("float32")

            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)

    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)

    def _slice_make(start_s, stop_s):
        time_start = round(start_s * 200)
        time_stop = round(stop_s * 200)
        eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
            drop=True
        )

        list_eeg = []
        for region in RAW_FEATS.keys():
            eeg_arr = np.zeros(
                (len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32
            )
            for chan_i, chan in enumerate(RAW_FEATS[region]):
                c1, c2 = chan.split("-")
                eeg_1 = eeg_default.loc[:, c1].copy()
                eeg_1 = eeg_1.fillna(float(eeg_1.mean())).values

                eeg_2 = eeg_default.loc[:, c2].copy()
                eeg_2 = eeg_2.fillna(float(eeg_2.mean())).values

                new_eeg = eeg_1 - eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
                new_eeg = np.clip(new_eeg, -1024, 1024)
                eeg_arr[chan_i, :] = new_eeg

            eeg_arr = np.reshape(eeg_arr, (4, 200, EEG_LENGTH))
            eeg_arr = np.concatenate(
                [
                    eeg_arr[0, :, :],
                    eeg_arr[1, :, :],
                    eeg_arr[2, :, :],
                    eeg_arr[3, :, :],
                ],
                1,
            )
            list_eeg.append(eeg_arr)

        eeg_c = np.concatenate(list_eeg, 1)
        eeg_c /= 104.0
        return eeg_c

    eeg_c = _slice_make((50 - EEG_LENGTH) / 2, (50 + EEG_LENGTH) / 2)
    eeg_l = _slice_make(18, 28)
    eeg_r = _slice_make(22, 32)
    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)

    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg_arr = np.zeros(
            (len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32
        )
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            c1, c2 = chan.split("-")
            eeg_1 = eeg_default.loc[:, c1].copy()
            eeg_1 = eeg_1.fillna(float(eeg_1.mean())).values

            eeg_2 = eeg_default.loc[:, c2].copy()
            eeg_2 = eeg_2.fillna(float(eeg_2.mean())).values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg_arr[chan_i, :] = new_eeg

        eeg_arr = np.reshape(eeg_arr, (4, 200, EEG_LENGTH))
        eeg_arr = np.concatenate(
            (eeg_arr[0, :, :], eeg_arr[1, :, :], eeg_arr[2, :, :], eeg_arr[3, :, :]), 1
        )
        list_eeg.append(eeg_arr)

    eeg_arr = np.concatenate(list_eeg, 1)
    eeg_arr /= 104.0
    return eeg_arr




## === cell 8
from joblib import Parallel, delayed


def save_features(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img50 = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img50)

    imgstft = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", imgstft)


need_build = False
if len(test) > 0:
    sample_eeg_id = str(test.iloc[0]["eeg_id"])
    need_build = not (
        os.path.exists(os.path.join(spec_directory_path, sample_eeg_id + ".npy"))
        and os.path.exists(os.path.join(eeg_directory_path, sample_eeg_id + ".npy"))
        and os.path.exists(os.path.join(raw_50s_directory_path, sample_eeg_id + ".npy"))
        and os.path.exists(
            os.path.join(raw_10s_directory_path, sample_eeg_id + "_c.npy")
        )
    )
if need_build:
    _ = Parallel(n_jobs=2, prefer="processes")(
        delayed(save_features)(row) for _, row in test.iterrows()
    )




## === cell 9
class Config:
    seed = 2024
    num_folds = 5


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 10
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader
from skimage.transform import resize




## === cell 11
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = int(row.eeg_id)

        spec_img = np.load(os.path.join(self.spec_data_path, f"{eeg_id}.npy")).astype(
            "float32"
        )
        raw_50s_img = np.load(
            os.path.join(self.raw_50s_data_path, f"{eeg_id}.npy")
        ).astype("float32")
        raw_10s_l_img = np.load(
            os.path.join(self.raw_10s_data_path, f"{eeg_id}_l.npy")
        ).astype("float32")
        raw_10s_c_img = np.load(
            os.path.join(self.raw_10s_data_path, f"{eeg_id}_c.npy")
        ).astype("float32")
        raw_10s_r_img = np.load(
            os.path.join(self.raw_10s_data_path, f"{eeg_id}_r.npy")
        ).astype("float32")
        eeg_img = np.load(os.path.join(self.eeg_data_path, f"{eeg_id}.npy")).astype(
            "float32"
        )

        eeg_img = resize(eeg_img, self.test_imgsize, preserve_range=True).astype(
            "float32"
        )
        spec_img = resize(spec_img, self.test_imgsize, preserve_range=True).astype(
            "float32"
        )
        raw_10s_l_img = resize(
            raw_10s_l_img, self.test_imgsize, preserve_range=True
        ).astype("float32")
        raw_10s_c_img = resize(
            raw_10s_c_img, self.test_imgsize, preserve_range=True
        ).astype("float32")
        raw_10s_r_img = resize(
            raw_10s_r_img, self.test_imgsize, preserve_range=True
        ).astype("float32")
        raw_50s_img = resize(
            raw_50s_img, self.test_imgsize, preserve_range=True
        ).astype("float32")

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

        eps = 1e-6

        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 12
class Net(nn.Module):
    def __init__(self, back_bone, device_id=None):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )

        for m in [
            self.spec_model,
            self.eeg_model,
            self.raw_50s_model,
            self.raw_10s_model,
        ]:
            m.fc_norm = nn.Identity()
            m.head_drop = nn.Identity()
            m.head = nn.Identity()

        self.head = nn.Linear(384 * 4, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(384, 6)
        self.head4 = nn.Linear(384, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

        spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

        feature = torch.cat(
            (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
        )
        logits = self.head(feature)
        logits_1 = self.head1(spec_feature)
        logits_2 = self.head2(eeg_feature)
        logits_3 = self.head3(raw_50s_feature)
        logits_4 = self.head4(raw_10s_feature)
        return logits, logits_1, logits_2, logits_3, logits_4




## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model_weights = [
    "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
]
backbone = "vit_small_patch14_reg4_dinov2.lvd142m"

vit_models = []
for w in model_weights:
    if os.path.exists(w):
        m = Net(backbone).to(device)
        state = torch.load(w, map_location=device)
        m.load_state_dict(state, strict=True)
        m.eval()
        vit_models.append(m)

print("loaded models:", len(vit_models))



## === cell 14
test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data,
    batch_size=8 if device.type == "cpu" else 32,
    pin_memory=(device.type == "cuda"),
    num_workers=2,
    drop_last=False,
)

result = {}

with torch.no_grad():
    for (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in test_loader:
        spec_imgs = (
            torch.from_numpy(np.asarray(spec_imgs)).to(device).float()
            if isinstance(spec_imgs, np.ndarray)
            else spec_imgs.to(device).float()
        )
        eeg_imgs = (
            torch.from_numpy(np.asarray(eeg_imgs)).to(device).float()
            if isinstance(eeg_imgs, np.ndarray)
            else eeg_imgs.to(device).float()
        )
        raw_50s_imgs = (
            torch.from_numpy(np.asarray(raw_50s_imgs)).to(device).float()
            if isinstance(raw_50s_imgs, np.ndarray)
            else raw_50s_imgs.to(device).float()
        )
        raw_10s_l_imgs = (
            torch.from_numpy(np.asarray(raw_10s_l_imgs)).to(device).float()
            if isinstance(raw_10s_l_imgs, np.ndarray)
            else raw_10s_l_imgs.to(device).float()
        )
        raw_10s_c_imgs = (
            torch.from_numpy(np.asarray(raw_10s_c_imgs)).to(device).float()
            if isinstance(raw_10s_c_imgs, np.ndarray)
            else raw_10s_c_imgs.to(device).float()
        )
        raw_10s_r_imgs = (
            torch.from_numpy(np.asarray(raw_10s_r_imgs)).to(device).float()
            if isinstance(raw_10s_r_imgs, np.ndarray)
            else raw_10s_r_imgs.to(device).float()
        )

        if len(vit_models) == 0:
            probs = torch.full(
                (spec_imgs.size(0), N_CLASSES), 1.0 / N_CLASSES, device=device
            )
        else:
            ensemble_probs = torch.zeros((spec_imgs.size(0), N_CLASSES), device=device)
            for model in vit_models:
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs_l = logits_l.softmax(dim=1)
                probs_c = logits_c.softmax(dim=1)
                probs_r = logits_r.softmax(dim=1)
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0
            probs = ensemble_probs / len(vit_models)

        probs = probs.detach().cpu().numpy()

        for j, eeg_id in enumerate(eeg_ids):
            eeg_id_int = int(eeg_id)
            result[eeg_id_int] = probs[j].astype(np.float64)

del vit_models
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("predicted eeg_ids:", len(result))



## === cell 15
sub = pd.read_csv(SAMPLE_SUB)

pred_mat = np.zeros((len(sub), N_CLASSES), dtype=np.float64)
uniform = np.full((N_CLASSES,), 1.0 / N_CLASSES, dtype=np.float64)

for i, eeg_id in enumerate(sub["eeg_id"].values):
    eeg_id_int = int(eeg_id)
    p = result.get(eeg_id_int, uniform).copy()

    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    pred_mat[i] = p

for k, c in enumerate(CLASSES):
    sub[c] = pred_mat[:, k]

row_sums = sub[CLASSES].sum(axis=1).values
if not np.allclose(row_sums, 1.0, atol=1e-6):
    sub[CLASSES] = sub[CLASSES].div(sub[CLASSES].sum(axis=1), axis=0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("wrote submission.csv with shape:", sub.shape)



## === cell 16
if not DEBUG:
    for p in [
        spec_directory_path,
        eeg_directory_path,
        raw_50s_directory_path,
        raw_10s_directory_path,
    ]:
        try:
            pass
        except Exception:
            pass
