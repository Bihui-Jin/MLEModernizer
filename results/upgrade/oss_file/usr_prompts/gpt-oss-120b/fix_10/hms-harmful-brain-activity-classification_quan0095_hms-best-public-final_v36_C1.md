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
import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy import signal
from scipy.signal import butter
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import gc

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
def stft_spec_from_eeg(parquet_path=None, df=None):
    EEG_LENGTH = 50
    if df is None:
        df = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg = df.iloc[time_start:time_stop]

    list_eeg = []
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 256, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1 = eeg_1.fillna(mean_value).values

            eeg_2 = eeg[COLS[kk + 1]]
            mean_value = eeg_2.mean()
            eeg_2 = eeg_2.fillna(mean_value).values

            new_eeg = eeg_1 - eeg_2
            f, t, spec = signal.spectrogram(
                new_eeg, fs=200, nperseg=39, noverlap=0, nfft=256
            )
            spec = np.log1p(np.abs(spec)).astype("float32")
            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img




## === cell 4
SFREQ = 200
filter_range = [0.5, 40]
b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}


def raw10seeg_from_eeg(parquet_path=None, df=None, eeg_id=None):
    EEG_LENGTH = 10
    if df is None:
        df = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = df.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1 = eeg_1.fillna(mean_value).values
            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2 = eeg_2.fillna(mean_value).values
            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate([eeg[0], eeg[1], eeg[2], eeg[3]], 1)
        list_eeg.append(eeg)
    img_c = np.concatenate(list_eeg, 1) / 104

    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)
    eeg_default = df.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1 = eeg_1.fillna(mean_value).values
            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2 = eeg_2.fillna(mean_value).values
            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate([eeg[0], eeg[1], eeg[2], eeg[3]], 1)
        list_eeg.append(eeg)
    img_l = np.concatenate(list_eeg, 1) / 104

    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)
    eeg_default = df.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1 = eeg_1.fillna(mean_value).values
            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2 = eeg_2.fillna(mean_value).values
            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate([eeg[0], eeg[1], eeg[2], eeg[3]], 1)
        list_eeg.append(eeg)
    img_r = np.concatenate(list_eeg, 1) / 104
    return img_l, img_c, img_r


def raw50seeg_from_eeg(parquet_path=None, df=None):
    EEG_LENGTH = 50
    if df is None:
        df = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_default = df.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1 = eeg_1.fillna(mean_value).values
            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2 = eeg_2.fillna(mean_value).values
            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate((eeg[0], eeg[1], eeg[2], eeg[3]), 1)
        list_eeg.append(eeg)
    eeg = np.concatenate(list_eeg, 1) / 104
    return eeg




## === cell 5
TARGET_IMG_SIZE = (518, 518)

if DEBUG:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print("Test shape:", test.shape)




## === cell 6
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




## === cell 7
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize
        self.spec_path = SPEC_PATH
        self.eeg_path = EEG_PATH

    def __len__(self):
        return len(self.df)

    def _resize_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy array (H, W) to a torch tensor and resize to TARGET_IMG_SIZE.
        The resulting tensor has shape (H_resized, W_resized, 1) to match the
        original code's (height, width, channel) layout.
        """
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=TARGET_IMG_SIZE,
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # (1, H_resized, W_resized)
        tensor = tensor.permute(1, 2, 0)  # (H_resized, W_resized, 1)
        return tensor

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        eeg_id = str(row.eeg_id)
        spec_id = str(row.spectrogram_id)

        spec = pd.read_parquet(f"{self.spec_path}{spec_id}.parquet")
        spec_arr = spec.values[:, 1:].T.astype("float32")
        spec_arr = spec_arr[:, :300]

        spec_arr = self._resize_tensor(spec_arr)

        eeg_df = pd.read_parquet(f"{self.eeg_path}{eeg_id}.parquet")

        img_l, img_c, img_r = raw10seeg_from_eeg(df=eeg_df)
        img_l = self._resize_tensor(img_l)
        img_c = self._resize_tensor(img_c)
        img_r = self._resize_tensor(img_r)

        img_50 = raw50seeg_from_eeg(df=eeg_df)
        img_50 = self._resize_tensor(img_50)

        eeg_img = stft_spec_from_eeg(df=eeg_df)
        eeg_img = self._resize_tensor(eeg_img)

        eps = 1e-6
        spec_arr = torch.clamp(spec_arr, np.exp(-4), np.exp(8))
        spec_arr = torch.log(spec_arr)
        spec_arr = torch.nan_to_num(spec_arr, nan=0.0)
        img_mean = spec_arr.mean()
        img_std = spec_arr.std()
        spec_arr = (spec_arr - img_mean) / (img_std + eps)

        return (
            spec_arr.numpy(),
            eeg_img.numpy(),
            img_50.numpy(),
            img_l.numpy(),
            img_c.numpy(),
            img_r.numpy(),
            eeg_id,
        )




## === cell 8
class Net(nn.Module):
    def __init__(self, back_bone):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone,
            num_classes=6,
            pretrained=False,
            in_chans=1,
            dynamic_img_pad=True,
            dynamic_img_size=True,
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
        spec_imgs = spec_imgs.permute(0, 3, 1, 2).contiguous()
        eeg_imgs = eeg_imgs.permute(0, 3, 1, 2).contiguous()
        raw_50s_imgs = raw_50s_imgs.permute(0, 3, 1, 2).contiguous()
        raw_10s_imgs = raw_10s_imgs.permute(0, 3, 1, 2).contiguous()

        spec_f = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_f = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw_50s_f = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw_10s_f = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

        concat_f = torch.cat((spec_f, eeg_f, raw_50s_f, raw_10s_f), dim=1)
        logits = self.head(concat_f)
        logits1 = self.head1(spec_f)
        logits2 = self.head2(eeg_f)
        logits3 = self.head3(raw_50s_f)
        logits4 = self.head4(raw_10s_f)
        return logits, logits1, logits2, logits3, logits4




## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_exp_5_bestlb.pth",
]
for w in model_weights:
    model = Net("vit_small_patch14_reg4_dinov2.lvd142m").to(device)
    try:
        state = torch.load(w, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {w}")
    except FileNotFoundError:
        print(f"Warning: weight file {w} not found – using untrained model.")
    model.eval()
    vit_models.append(model)



## === cell 10
num_workers = max(1, os.cpu_count() - 1)

test_dataset = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=num_workers,
    drop_last=False,
)

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

result_4 = {}
with torch.no_grad():
    for (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l,
        raw_10s_c,
        raw_10s_r,
        eeg_ids,
    ) in test_loader:
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_l = raw_10s_l.to(device).float()
        raw_10s_c = raw_10s_c.to(device).float()
        raw_10s_r = raw_10s_r.to(device).float()

        spec_perm = spec_imgs.permute(0, 3, 1, 2).contiguous()
        eeg_perm = eeg_imgs.permute(0, 3, 1, 2).contiguous()
        raw_50_perm = raw_50s_imgs.permute(0, 3, 1, 2).contiguous()
        raw_10_stack = (
            torch.cat([raw_10s_l, raw_10s_c, raw_10s_r], dim=0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
        for model in vit_models:
            spec_f = model.spec_model.forward_features(spec_perm)[:, 0]
            eeg_f = model.eeg_model.forward_features(eeg_perm)[:, 0]
            raw_50_f = model.raw_50s_model.forward_features(raw_50_perm)[:, 0]

            raw_10_f_all = model.raw_10s_model.forward_features(raw_10_stack)[:, 0]
            B = spec_imgs.shape[0]
            raw_10_f_l, raw_10_f_c, raw_10_f_r = torch.split(raw_10_f_all, B, dim=0)

            for raw_10_f in (raw_10_f_l, raw_10_f_c, raw_10_f_r):
                concat_f = torch.cat((spec_f, eeg_f, raw_50_f, raw_10_f), dim=1)
                logits = model.head(concat_f)
                probs = logits.softmax(dim=1)
                ensemble_probs += probs / 3.0  # average over three views

        ensemble_probs /= len(vit_models)  # average over models
        probs_np = ensemble_probs.cpu().numpy()
        for idx, eid in enumerate(eeg_ids):
            eid = int(eid)
            if eid not in result_4:
                result_4[eid] = np.zeros(6, dtype=np.float32)
            result_4[eid] += probs_np[idx]



## === cell 11
sub_rows = []
for eeg_id, probs in result_4.items():
    sub_rows.append([eeg_id] + probs.tolist())

df_sub = pd.DataFrame(sub_rows, columns=["eeg_id"] + CLASSES)

df_sub = df_sub.set_index("eeg_id")
df_sub = df_sub.reindex(test["eeg_id"], fill_value=np.nan).reset_index()

uniform = np.full(len(CLASSES), 1.0 / len(CLASSES), dtype=np.float32)
df_sub[CLASSES] = df_sub[CLASSES].apply(lambda col: col.fillna(uniform[col.name]))

prob_sum = df_sub[CLASSES].sum(axis=1)
df_sub[CLASSES] = df_sub[CLASSES].div(prob_sum, axis=0)

df_sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(df_sub.head())



## === cell 12
for path in []:
    if os.path.isdir(path):
        for f in os.listdir(path):
            try:
                os.remove(os.path.join(path, f))
            except Exception:
                pass
