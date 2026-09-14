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

0.2876283318496322

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
DEBUG = False
PROCESS_DATA = False  # Skip heavy data preprocessing by default.



## === cell 1
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import librosa
from scipy.signal import butter, lfilter, spectrogram




## === cell 2
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
        img = np.zeros((128, 256, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            eeg_2 = eeg[COLS[kk + 1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            new_eeg = eeg_1 - eeg_2
            f, t, spec = spectrogram(new_eeg, fs=200, nperseg=39, noverlap=0, nfft=256)
            spec = np.log1p(np.abs(spec)).astype("float32")
            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img




## === cell 3
SFREQ = 200
filter_range = [0.5, 40]
RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}
b, a = butter(3, np.float32(filter_range) * 2 / SFREQ, btype="band")


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            new_eeg = eeg_1 - eeg_2
            new_eeg = np.clip(lfilter(b, a, new_eeg), -1024, 1024)
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate([eeg[0], eeg[1], eeg[2], eeg[3]], 1)
        list_eeg.append(eeg)
    eeg_c = np.concatenate(list_eeg, 1) / 104
    return eeg_c, eeg_c, eeg_c


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = []
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            new_eeg = eeg_1 - eeg_2
            new_eeg = np.clip(lfilter(b, a, new_eeg), -1024, 1024)
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate([eeg[0], eeg[1], eeg[2], eeg[3]], 1)
        list_eeg.append(eeg)
    eeg = np.concatenate(list_eeg, 1) / 104
    return eeg




## === cell 4
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

default_path = "./data/hms-harmful-brain-activity-classification/"
kaggle_path = "/kaggle/input/hms-harmful-brain-activity-classification/"

if os.path.isdir(kaggle_path):
    BASE_PATH = kaggle_path
elif os.path.isdir(default_path):
    BASE_PATH = default_path
else:
    raise FileNotFoundError("Dataset base directory not found.")

if DEBUG:
    test = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))[:40]
    SPEC_PATH = os.path.join(BASE_PATH, "train_spectrograms/")
    EEG_PATH = os.path.join(BASE_PATH, "train_eegs/")
else:
    test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
    SPEC_PATH = os.path.join(BASE_PATH, "test_spectrograms/")
    EEG_PATH = os.path.join(BASE_PATH, "test_eegs/")

print("Test shape:", test.shape)

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



## === cell 5
from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def save(row):
    try:
        eeg_id = row["eeg_id"]
        spec_id = row["spectrogram_id"]
        spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
        spec_arr = spec.values[:, 1:].T.astype("float32")[:, :300]
        np.save(f"{spec_directory_path}{eeg_id}.npy", spec_arr)
        img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
        np.save(f"{raw_10s_directory_path}{eeg_id}_l.npy", img_l)
        np.save(f"{raw_10s_directory_path}{eeg_id}_c.npy", img_c)
        np.save(f"{raw_10s_directory_path}{eeg_id}_r.npy", img_r)
        img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
        np.save(f"{raw_50s_directory_path}{eeg_id}.npy", img)
        img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
        np.save(f"{eeg_directory_path}{eeg_id}.npy", img)
    except Exception as e:
        if DEBUG:
            print(f"Failed to process {eeg_id}: {e}")


if PROCESS_DATA:
    Parallel(n_jobs=4)(delayed(save)(row) for _, row in test.iterrows())
else:
    if DEBUG:
        print("Skipping heavy preprocessing (PROCESS_DATA=False)")




## === cell 6
class Config:
    seed = 2024
    num_folds = 5




## === cell 7
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 8
try:
    import timm
except ImportError:
    timm = None



## === cell 9
import torch.utils.data as data
from torch.utils.data import DataLoader
from skimage.transform import resize




## === cell 10
class ImageFolder(data.Dataset):
    def __init__(self, df, imgsize):
        self.df = df.reset_index(drop=True)
        self.imgsize = imgsize
        self.spec_path = spec_directory_path
        self.eeg_path = eeg_directory_path
        self.raw50_path = raw_50s_directory_path
        self.raw10_path = raw_10s_directory_path

    def __len__(self):
        return len(self.df)

    def _load_npy(self, path, shape):
        if os.path.exists(path):
            return np.load(path).astype("float32")
        return np.zeros(shape, dtype="float32")

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        eid = row.eeg_id  # keep as integer for consistency
        spec = self._load_npy(os.path.join(self.spec_path, f"{eid}.npy"), (300, 128))
        eeg = self._load_npy(os.path.join(self.eeg_path, f"{eid}.npy"), (128, 256, 4))
        raw50 = self._load_npy(
            os.path.join(self.raw50_path, f"{eid}.npy"), (512, 512, 4)
        )
        raw10_l = self._load_npy(
            os.path.join(self.raw10_path, f"{eid}_l.npy"), (128, 256, 4)
        )
        raw10_c = self._load_npy(
            os.path.join(self.raw10_path, f"{eid}_c.npy"), (128, 256, 4)
        )
        raw10_r = self._load_npy(
            os.path.join(self.raw10_path, f"{eid}_r.npy"), (128, 256, 4)
        )
        spec = resize(spec, self.imgsize)
        raw10_l = resize(raw10_l, self.imgsize)
        raw10_c = resize(raw10_c, self.imgsize)
        raw10_r = resize(raw10_r, self.imgsize)
        raw50 = resize(raw50, self.imgsize)
        spec = np.expand_dims(spec, -1)
        eeg = np.expand_dims(eeg, -1)
        raw50 = np.expand_dims(raw50, -1)
        raw10_l = np.expand_dims(raw10_l, -1)
        raw10_c = np.expand_dims(raw10_c, -1)
        raw10_r = np.expand_dims(raw10_r, -1)
        eps = 1e-6
        spec = np.clip(spec, np.exp(-4), np.exp(8))
        spec = np.log(spec)
        spec = np.nan_to_num(spec, nan=0.0)
        spec = (spec - spec.mean()) / (spec.std() + eps)
        return spec, eeg, raw50, raw10_l, raw10_c, raw10_r, eid




## === cell 11
class Net(nn.Module):
    def __init__(self, backbone_name, device):
        super().__init__()
        self.device = device
        if timm and backbone_name:
            self.spec_model = timm.create_model(
                backbone_name, num_classes=6, pretrained=False, in_chans=1
            )
            self.eeg_model = timm.create_model(
                backbone_name,
                num_classes=6,
                pretrained=False,
                in_chans=1,
                dynamic_img_pad=True,
                dynamic_img_size=True,
            )
            self.raw50_model = timm.create_model(
                backbone_name, num_classes=6, pretrained=False, in_chans=1
            )
            self.raw10_model = timm.create_model(
                backbone_name, num_classes=6, pretrained=False, in_chans=1
            )
            for m in [
                self.spec_model,
                self.eeg_model,
                self.raw50_model,
                self.raw10_model,
            ]:
                m.fc_norm = nn.Identity()
                m.head_drop = nn.Identity()
                m.head = nn.Identity()
            self.head = nn.Linear(384 * 4, 6)
        else:
            self.head = nn.Identity()

    def forward(self, spec_imgs, eeg_imgs, raw50_imgs, raw10_imgs):
        if timm and hasattr(self, "spec_model"):
            spec_imgs = spec_imgs.permute(0, 3, 1, 2).contiguous()
            eeg_imgs = eeg_imgs.permute(0, 3, 1, 2).contiguous()
            raw50_imgs = raw50_imgs.permute(0, 3, 1, 2).contiguous()
            raw10_imgs = raw10_imgs.permute(0, 3, 1, 2).contiguous()
            spec_f = self.spec_model.forward_features(spec_imgs)[:, 0]
            eeg_f = self.eeg_model.forward_features(eeg_imgs)[:, 0]
            raw50_f = self.raw50_model.forward_features(raw50_imgs)[:, 0]
            raw10_f = self.raw10_model.forward_features(raw10_imgs)[:, 0]
            feats = torch.cat([spec_f, eeg_f, raw50_f, raw10_f], dim=1)
            logits = self.head(feats)
            return logits, logits, logits, logits, logits
        else:
            batch = spec_imgs.shape[0]
            zeros = torch.zeros(batch, 6, device=self.device)
            return zeros, zeros, zeros, zeros, zeros




## === cell 12
class DummyModel(nn.Module):
    """Returns zero logits for any input; used when pretrained weights are unavailable."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, spec_imgs, eeg_imgs, raw50_imgs, raw10_imgs):
        batch = spec_imgs.shape[0]
        zeros = torch.zeros(batch, 6, device=self.device)
        return zeros, zeros, zeros, zeros, zeros


model_weights = [
    "/kaggle/input/hms-stage2/fold_0_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_exp_3_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_exp_3_bestlb.pth",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_models = []
for w_path in model_weights:
    if os.path.exists(w_path):
        try:
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(w_path, map_location=device)
            model.load_state_dict(state)
        except Exception:
            model = DummyModel(device).to(device)
    else:
        model = DummyModel(device).to(device)
    model.eval()
    vit_models.append(model)



## === cell 13
result_4 = {}
try:
    test_data = ImageFolder(test, (518, 518))
    test_loader = DataLoader(
        test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
    )
    with torch.no_grad():
        for batch in test_loader:
            spec_imgs, eeg_imgs, raw50_imgs, raw10_l, raw10_c, raw10_r, eeg_ids = batch
            spec_imgs = spec_imgs.to(device).float()
            eeg_imgs = eeg_imgs.to(device).float()
            raw50_imgs = raw50_imgs.to(device).float()
            raw10_l = raw10_l.to(device).float()
            raw10_c = raw10_c.to(device).float()
            raw10_r = raw10_r.to(device).float()
            ensemble = torch.zeros(spec_imgs.shape[0], 6, device=device)
            for mdl in vit_models:
                logits_l, _, _, _, _ = mdl(spec_imgs, eeg_imgs, raw50_imgs, raw10_l)
                logits_c, _, _, _, _ = mdl(spec_imgs, eeg_imgs, raw50_imgs, raw10_c)
                logits_r, _, _, _, _ = mdl(spec_imgs, eeg_imgs, raw50_imgs, raw10_r)
                probs = (
                    logits_l.softmax(1) + logits_c.softmax(1) + logits_r.softmax(1)
                ) / 3.0
                ensemble += probs
            ensemble /= len(vit_models)
            ensemble_np = ensemble.cpu().numpy()
            for i, eid in enumerate(eeg_ids):
                result_4[int(eid)] = (
                    result_4.get(int(eid), np.zeros(6)) + ensemble_np[i]
                )
except Exception as e:
    if DEBUG:
        print("Inference failed or data missing:", e)



## === cell 14
if not result_4:
    train_path = os.path.join(BASE_PATH, "train.csv")
    train_df = pd.read_csv(train_path)
    class_totals = train_df[CLASSES].sum().values.astype(float)
    class_probs = class_totals / class_totals.sum()
    for eid in test["eeg_id"]:
        result_4[int(eid)] = class_probs.copy()



## === cell 15
train_path = os.path.join(BASE_PATH, "train.csv")
train_df = pd.read_csv(train_path)
class_totals = train_df[CLASSES].sum().values.astype(float)
class_probs = class_totals / class_totals.sum()

sub_rows = []
for eid, probs in result_4.items():
    sub_rows.append([eid, *probs])
df = pd.DataFrame(sub_rows, columns=["eeg_id"] + CLASSES)

prob_array = df[CLASSES].values.astype(float)
row_sums = prob_array.sum(axis=1, keepdims=True)
zero_mask = row_sums.squeeze() == 0
prob_array[zero_mask] = class_probs
row_sums[zero_mask] = 1.0
df[CLASSES] = prob_array / row_sums

df.to_csv("submission.csv", index=False)
print(df.head())



## === cell 16
if not DEBUG:
    for p in [
        spec_directory_path,
        eeg_directory_path,
        raw_50s_directory_path,
        raw_10s_directory_path,
    ]:
        if os.path.isdir(p):
            for f in os.listdir(p):
                os.remove(os.path.join(p, f))
