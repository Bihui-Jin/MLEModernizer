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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
timm==1.0.19
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

0.5329145195513019

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import albumentations as A

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import timm
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter

from tqdm import tqdm



## === cell 1
path = "/kaggle/input/hms-harmful-brain-activity-classification"
batch_size = 24
device = "cuda" if tc.cuda.is_available() else "cpu"
domain = "test"  # keep as provided

read_all_eegs = False
read_all_specs = False
read_all_l2i = False
is_augument = False

tc.backends.cudnn.benchmark = True



## === cell 2
df = pd.read_csv(f"{path}/{domain}.csv")
df.head()



## === cell 3
if domain == "train":
    train_df = df.copy()
    label_cols = train_df.columns[-6:]
    eeg_ids = train_df.eeg_id.unique()
    train_df = df.groupby("eeg_id")[["patient_id"]].agg("first")
    aux = df.groupby("eeg_id")[label_cols].agg("sum")
    si = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg("first")

    for k in si:
        train_df[k] = si[k].values

    for label in label_cols:
        train_df[label] = aux[label].values

    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train_df[label_cols] = y_data

    train_df = train_df.reset_index()
    train_df = train_df.loc[train_df.eeg_id.isin(eeg_ids)]
    print(f"Train dataframe with unique eeg_id has shape: {train_df.shape}")

    df = train_df




## === cell 4
def butter_lowpass_filter(data, cutoff_freq=40, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def butter_highpass_filter(data, cutoff_freq=1, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def _eeg_folder_for_domain(domain_):
    return "train_eegs" if domain_ == "train" else "test_eegs"


def _spec_folder_for_domain(domain_):
    return "train_spectrograms" if domain_ == "train" else "test_spectrograms"


def get_eeg_from_parquet(idx, metadata, eeg_ids, domain_="train"):
    eeg_id = eeg_ids[idx]
    eeg_path = f"{path}/{_eeg_folder_for_domain(domain_)}/{eeg_id}.parquet"
    eeg = pd.read_parquet(eeg_path).iloc[0:10_000, :]
    eeg = eeg.values.T
    return eeg


def get_spec_from_parquet(idx, metadata, eeg_ids, domain_="train"):
    eeg_id = eeg_ids[idx]
    occurances_df = metadata.loc[metadata["eeg_id"] == eeg_id, :]

    spec_id = occurances_df.loc[:, "spectrogram_id"].values[0]
    if domain_ == "train":
        spec_offset = int(
            occurances_df.loc[:, "spectrogram_label_offset_seconds"].values[0] // 2
        )
    else:
        spec_offset = 0

    spec_path = f"{path}/{_spec_folder_for_domain(domain_)}/{spec_id}.parquet"
    x = pd.read_parquet(spec_path)
    x = (
        x.values[spec_offset : 300 + spec_offset, 1:]
        .reshape(300, 4, 100)
        .transpose((1, 2, 0))
    )
    return x


def process_spec(x):
    x = np.clip(x, np.exp(-4), np.exp(8))
    x = np.log(x)

    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    x = (x - mu) / (sigma + 1e-5)
    x = np.nan_to_num(x, nan=0.0)
    return x


def line2img(x):
    C, L = x.shape

    mx = x.max(axis=1, keepdims=True)
    mn = x.min(axis=1, keepdims=True)
    nx = (x - mn) / (mx - mn + 1e-5)

    sw = 3
    H, W = 128 - sw, 1024 - sw

    h_inds = (nx * (H - 1)).astype(np.int32)
    w_inds = np.linspace(0, W - 1, L)[None, :].repeat(C, axis=0).astype(np.int32)
    chans = np.array(range(C))[:, None]

    img = np.zeros((C, H + sw, W + sw), dtype=np.uint8)
    for i in range(sw):
        for j in range(sw):
            img[chans, h_inds + i, w_inds + j] = 1

    img = cv2.resize(
        img.transpose(1, 2, 0), (512, 64), interpolation=cv2.INTER_AREA
    ).transpose(2, 0, 1)
    return img




## === cell 5
class TrainingDatasetEEG(Dataset):
    def __init__(self, metadata, train=True):
        self.train = train
        self.metadata = metadata

        neeg_ids = np.array(
            sorted(metadata["eeg_id"].unique()), dtype=np.int64
        ).squeeze()
        if read_all_eegs and domain == "train":
            eeg_ids = np.array(
                [
                    eeg_id
                    for eeg_id, is_corrupted in zip(neeg_ids, all_is_corrupted_list)
                ]
            ).astype(np.int64)
        else:
            eeg_ids = neeg_ids

        num_eeg_ids = eeg_ids.shape[0]
        train_size_percentage = 85
        train_set_size = int(train_size_percentage / 100 * num_eeg_ids)
        self.train_set_size = train_set_size
        eeg_ids_train = eeg_ids[:train_set_size]
        eeg_ids_valid = eeg_ids[train_set_size:]
        self.eeg_ids = eeg_ids_train if train else eeg_ids_valid

        names = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]
        pairs = list(zip(range(20), names))
        pairs = {k.lower(): v for v, k in pairs}
        cen_locs = [
            "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
            "f7,f3,fz,fz,f4,f8".split(","),
            "t3,c3,cz,cz,c4,t4".split(","),
            "t5,p3,pz,pz,p4,t6".split(","),
            "o1,o1,o1,o2,o2,o2".split(","),
        ]
        cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
        cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)
        self.cen_ilocs = cen_ilocs

        self.transform = A.Compose(
            [
                A.Blur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.2
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.OneOf(
                    [
                        A.CoarseDropout(
                            max_holes=12, max_height=60, max_width=64, p=0.05
                        ),
                        A.GridDropout(p=0.05),
                    ],
                    p=0.1,
                ),
                A.ElasticTransform(p=0.05),
                A.GridDistortion(p=0.05),
                A.OpticalDistortion(p=0.05),
            ],
            p=0.2,
        )

    def __len__(self):
        return self.eeg_ids.shape[0]

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        idx_global = idx if self.train else idx + self.train_set_size
        occurances_df = self.metadata.loc[self.metadata["eeg_id"] == eeg_id, :]

        if read_all_eegs and domain == "train":
            eeg = all_eegs[idx_global].copy()
        else:
            eeg = get_eeg_from_parquet(
                idx_global - self.train_set_size,
                self.metadata,
                self.eeg_ids,
                domain_="train",
            )
            for i in range(eeg.shape[0]):
                nans = np.isnan(eeg[i]).sum()
                if nans / 1e4 > 0.7:
                    nidx = idx + 1
                    if nidx >= self.__len__():
                        nidx = 0
                    return self.__getitem__(nidx)
            eeg = np.nan_to_num(eeg, nan=0)

        cutoff_freq = 40
        gn = 0.0
        direction = 1
        polarity = 1.0

        if is_augument and self.train:
            if np.random.rand() > 0.8:
                cutoff_freq = int(np.random.choice(range(16, 30, 4), size=1)[0])

            if np.random.rand() > 0.8:
                gnmx = eeg.mean(1)[:, None] * float(
                    np.random.choice((0.025, 0.05, 0.1), size=1)[0]
                )
                polarity_gn = np.random.choice((-1.0, 1.0), size=eeg[0].shape)[None, :]
                gn = polarity_gn * np.random.random(eeg[0].shape)[None, :] * gnmx

            if np.random.rand() > 0.8:
                direction = -1

            if np.random.rand() > 0.8:
                polarity = float(np.random.choice((-1.0, 1.0), size=1)[0])

        eeg = eeg[:, ::direction] + gn
        eeg *= polarity

        x1 = butter_lowpass_filter(eeg.T, cutoff_freq=cutoff_freq).T
        x1 = np.clip(x1, -1024, 1024)

        x2_ = eeg[self.cen_ilocs, :]  # 5,6,L
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)
        x2_ = x2_.reshape(4 * 6, -1)

        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=cutoff_freq).T
        x2 = np.clip(x2, -1024, 1024)

        x3_ = eeg[self.cen_ilocs[[0, 2, 4], :][:, [0, 1, 4, 5]], :]
        x3_ = np.split(x3_, 3, axis=0)
        x3_ = np.concatenate([x3_[i] - x3_[i + 1] for i in range(2)], 0)
        x3_ = x3_.reshape(2 * 4, -1)

        x3 = butter_lowpass_filter(x3_.T, cutoff_freq=cutoff_freq).T
        x3 = np.clip(x3, -1024, 1024)

        if read_all_specs and domain == "train":
            x4 = all_specs[idx_global]
        else:
            x4 = get_spec_from_parquet(
                idx_global - self.train_set_size,
                self.metadata,
                self.eeg_ids,
                domain_="train",
            )
            x4 = process_spec(x4)

        if read_all_l2i and domain == "train":
            x5 = all_imgs[idx_global]
        else:
            x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
            x5 = line2img(x5).astype(np.float32)

        if is_augument and self.train:
            for i in range(16):
                augmented = self.transform(image=x5[i])
                x5[i] = augmented["image"]

        x5 = (x5 - 0.5) / 0.5

        targets = occurances_df.iloc[:, -6:].values.squeeze()

        x1 = tc.from_numpy(np.ascontiguousarray(x1)).type(tc.float32) / 128.0
        x2 = tc.from_numpy(np.ascontiguousarray(x2)).type(tc.float32) / 32.0
        x3 = tc.from_numpy(np.ascontiguousarray(x3)).type(tc.float32)
        x4 = tc.from_numpy(np.ascontiguousarray(x4)).type(tc.float32)
        x5 = tc.from_numpy(np.ascontiguousarray(x5)).type(tc.float32)
        targets = tc.from_numpy(targets).type(tc.float32)
        return (x1, x2, x3, x4, x5), targets.squeeze()




## === cell 6
class DatasetEEG(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata.reset_index(drop=True)

        names = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]
        pairs = list(zip(range(20), names))
        pairs = {k.lower(): v for v, k in pairs}
        cen_locs = [
            "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
            "f7,f3,fz,fz,f4,f8".split(","),
            "t3,c3,cz,cz,c4,t4".split(","),
            "t5,p3,pz,pz,p4,t6".split(","),
            "o1,o1,o1,o2,o2,o2".split(","),
        ]
        cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
        cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)
        self.cen_ilocs = cen_ilocs

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        eeg_id = int(self.metadata.loc[idx, "eeg_id"])

        eeg_path = f"{path}/test_eegs/{eeg_id}.parquet"
        eeg = pd.read_parquet(eeg_path).iloc[:10_000, :]
        eeg = eeg.values.T
        eeg = np.nan_to_num(eeg, nan=0)

        cutoff_freq = 40

        x1 = butter_lowpass_filter(eeg.T, cutoff_freq=cutoff_freq).T
        x1 = np.clip(x1, -1024, 1024)

        x2_ = eeg[self.cen_ilocs, :]
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)
        x2_ = x2_.reshape(4 * 6, -1)

        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=cutoff_freq).T
        x2 = np.clip(x2, -1024, 1024)

        x3_ = eeg[self.cen_ilocs[[0, 2, 4], :][:, [0, 1, 4, 5]], :]
        x3_ = np.split(x3_, 3, axis=0)
        x3_ = np.concatenate([x3_[i] - x3_[i + 1] for i in range(2)], 0)
        x3_ = x3_.reshape(2 * 4, -1)

        x3 = butter_lowpass_filter(x3_.T, cutoff_freq=cutoff_freq).T
        x3 = np.clip(x3, -1024, 1024)

        spec_id = int(self.metadata.loc[idx, "spectrogram_id"])
        spec_path = f"{path}/test_spectrograms/{spec_id}.parquet"
        x4 = pd.read_parquet(spec_path)
        x4 = x4.values[:300, 1:].reshape(300, 4, 100).transpose((1, 2, 0))
        x4 = process_spec(x4)

        x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
        x5 = line2img(x5).astype(np.float32)
        x5 = (x5 - 0.5) / 0.5

        x1 = tc.from_numpy(np.ascontiguousarray(x1)).type(tc.float32) / 128.0
        x2 = tc.from_numpy(np.ascontiguousarray(x2)).type(tc.float32) / 32.0
        x3 = tc.from_numpy(np.ascontiguousarray(x3)).type(tc.float32)
        x4 = tc.from_numpy(np.ascontiguousarray(x4)).type(tc.float32)
        x5 = tc.from_numpy(np.ascontiguousarray(x5)).type(tc.float32)
        return x1, x2, x3, x4, x5, eeg_id




## === cell 7
import torch

p_dropout = 0.3 if domain == "train" else 0.0


class BBlock(nn.Module):
    def __init__(self, o, g, s=2, h=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.c = nn.Sequential(
            nn.Conv1d(o, o * s, 3, padding=1, groups=g),
            nn.BatchNorm1d(o * s),
            nn.SiLU(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
            nn.BatchNorm1d(o * s),
            nn.SiLU(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
        )
        self.c_res = nn.Conv1d(o, o * s, 1)
        self.c_add_norm = nn.Sequential(nn.BatchNorm1d(o * s), nn.SiLU())
        o_ = o if h is None else h
        self.c_final = nn.Sequential(
            nn.Conv1d(o * s, o_, 3, padding=1, groups=g), nn.BatchNorm1d(o_), nn.SiLU()
        )

    def forward(self, x):
        c = self.c(x)
        c_res = self.c_res(x)
        c_norm = self.c_add_norm(c + c_res)
        c_final = self.c_final(c_norm)
        return c_final


class BBlock2(nn.Module):
    def __init__(
        self, o, g, s=2, h=None, is_strided=False, stride=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.c = nn.Sequential(
            nn.Conv2d(o, o * s, 3, padding=1, groups=g),
            nn.BatchNorm2d(o * s),
            nn.SiLU(),
            nn.Conv2d(o * s, o * s, 3, padding=1, groups=g),
            nn.BatchNorm2d(o * s),
            nn.SiLU(),
            nn.Conv2d(o * s, o * s, 3, padding=1, groups=g),
        )
        self.c_res = nn.Conv2d(o, o * s, 1, groups=g)
        self.c_add_norm = nn.Sequential(nn.BatchNorm2d(o * s), nn.SiLU())
        o_ = o if h is None else h
        self.c_final = nn.Sequential(
            nn.Conv2d(o * s, o_, 3, stride if is_strided else 1, padding=1, groups=g),
            nn.BatchNorm2d(o_),
            nn.SiLU(),
        )

    def forward(self, x):
        c = self.c(x)
        c_res = self.c_res(x)
        c_norm = self.c_add_norm(c + c_res)
        c_final = self.c_final(c_norm)
        return c_final


class TBlock(nn.Module):
    def __init__(self, o, g, s=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.c = nn.Sequential(
            nn.Conv1d(o, o * s, 3, padding=1, groups=g),
            nn.Tanh(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
            nn.Tanh(),
            nn.Conv1d(o * s, o * s, 3, padding=1, groups=g),
        )
        self.c_res = nn.Conv1d(o, o * s, 1, groups=g)
        self.c_add_norm = nn.Tanh()
        self.c_final = nn.Sequential(
            nn.Conv1d(o * s, o, 3, padding=1, groups=g), nn.Tanh()
        )

    def forward(self, x):
        c = self.c(x)
        c_res = self.c_res(x)
        c_norm = self.c_add_norm(c + c_res)
        c_final = self.c_final(c_norm)
        return c_final


class EEGFeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.c1 = TBlock(20, 20, s=2)

        self.c2 = nn.Sequential(
            nn.Conv1d(190, 190, 1, groups=190),
            nn.Tanh(),
            nn.AvgPool1d(5),
            nn.Conv1d(190, 32, 3, padding=1, groups=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )

        self.c3 = nn.Sequential(
            BBlock(32, 1, s=2, h=16),
            BBlock(16, 1, s=2, h=24),
            nn.AvgPool1d(5),
            BBlock(24, 1, s=2, h=32),
            BBlock(32, 1, s=2, h=48),
            nn.AvgPool1d(5),
            BBlock(48, 1, s=2, h=64),
            BBlock(64, 1, s=2, h=72),
            nn.AvgPool1d(5),
            BBlock(72, 1, s=2, h=96),
        )

        self.d1 = nn.Sequential(
            BBlock(1, 1, s=4, h=4),
            nn.AvgPool1d(5),
            BBlock(4, 1, s=2, h=8),
            nn.AvgPool1d(5),
            BBlock(8, 1, s=2, h=12),
            nn.AvgPool1d(5),
            BBlock(12, 1, s=2, h=16),
            nn.AvgPool1d(5),
            BBlock(16, 1, s=2, h=24),
        )
        self.d2 = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(480, 128), nn.SiLU())

        self.e1 = nn.Sequential(
            BBlock(1, 1, s=4, h=4),
            nn.AvgPool1d(5),
            BBlock(4, 1, s=2, h=8),
            nn.AvgPool1d(5),
            BBlock(8, 1, s=2, h=12),
            nn.AvgPool1d(5),
            BBlock(12, 1, s=2, h=16),
            nn.AvgPool1d(5),
            BBlock(16, 1, s=2, h=24),
        )
        self.e2 = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(576, 128), nn.SiLU())

        self.f1 = nn.Sequential(
            BBlock(4, 1, s=4, h=8),
            nn.AvgPool1d(5),
            BBlock(8, 1, s=2, h=12),
            nn.AvgPool1d(5),
            BBlock(12, 1, s=2, h=16),
            nn.AvgPool1d(5),
            BBlock(16, 1, s=2, h=24),
            nn.AvgPool1d(5),
            BBlock(24, 1, s=2, h=32),
        )
        self.f2 = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(192, 96), nn.SiLU())

        self.g1 = nn.Sequential(
            BBlock(1, 1, s=4, h=8),
            nn.AvgPool1d(2),
            BBlock(8, 1, s=2, h=16),
            nn.AvgPool1d(2),
            BBlock(16, 1, s=2, h=32),
            nn.AvgPool1d(2),
            BBlock(32, 1, s=2, h=64),
            nn.AvgPool1d(2),
            BBlock(64, 1, s=2, h=128),
            nn.AvgPool1d(2),
            BBlock(128, 1, s=1, h=256),
            BBlock(256, 1, s=1, h=324),
            BBlock(324, 1, s=1, h=512),
        )
        self.g2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(512 * 8, 512), nn.SiLU()
        )

        self.h1 = nn.Sequential(
            BBlock2(1, 1, s=2, h=8),
            nn.AvgPool2d(2),
            BBlock2(8, 1, s=2, h=12),
            nn.AvgPool2d(2),
            BBlock2(12, 1, s=2, h=24),
            nn.AvgPool2d(2),
            BBlock2(24, 1, s=2, h=32),
            nn.AvgPool2d(2),
            BBlock2(32, 1, s=2, h=48),
            nn.AvgPool2d(2),
            BBlock2(48, 1, s=2, h=64),
        )
        self.h2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(64 * 4, 256), nn.SiLU()
        )

        self.i1 = nn.Sequential(
            BBlock2(1, 1, s=2, h=8, is_strided=True, stride=(1, 2)),
            BBlock2(8, 1, s=2, h=12, is_strided=True, stride=(1, 2)),
            BBlock2(12, 1, s=2, h=24),
            nn.AvgPool2d(2),
            BBlock2(24, 1, s=2, h=32),
            nn.AvgPool2d(2),
            BBlock2(32, 1, s=2, h=48),
            nn.AvgPool2d(2),
            BBlock2(48, 1, s=2, h=64),
        )
        self.i2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(64 * 16, 512), nn.SiLU()
        )

        self.j1 = timm.create_model("tf_efficientnet_b0", pretrained=False, in_chans=4)
        self.j1.classifier = nn.Linear(self.j1.classifier.in_features, 512)
        self.j2 = nn.Sequential(
            nn.Dropout(p_dropout), nn.Linear(512 * 4, 512), nn.SiLU()
        )

    def ind_feature_extractor(self, x):
        d1 = tc.cat([self.d1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        d2 = self.d2(d1.mean(-1))
        return d2

    def total_feature_extractor(self, x):
        c1 = self.c1(x)
        c2 = []
        for i in range(20):
            X = c1[:, i, :]
            for j in range(i + 1, 20):
                xij = X - c1[:, j, :]
                c2.append(xij)

        c2 = self.c2(F.tanh(tc.stack(c2, 1)))
        c3 = self.c3(c2)
        return c3.mean(-1)

    def montage_features_extractor(self, x):
        e1 = tc.cat([self.e1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        e2 = self.e2(e1.mean(-1))
        return e2

    def montage_features_extractor2(self, x):
        with tc.no_grad():
            x = self.norm(x, 2)
        g1 = tc.cat([self.g1(x[:, i, :].unsqueeze(1)) for i in range(x.shape[1])], 1)
        g2 = self.g2(g1.mean(-1))
        return g2

    def norm(self, x, axis=1):
        mx = x.max(axis, keepdims=True)[0]
        mn = x.min(axis, keepdims=True)[0]
        n = (x - mn) / (mx - mn + 1e-5)
        return n

    def gr_montage_features_extractor(self, x):
        N, S, L = x.shape
        x = x.view(N, 4, 6, L)
        f1 = tc.cat([self.f1(x[:, :, i, :]).mean(-1) for i in range(6)], 1)
        f2 = self.f2(f1)
        return f2

    def kagg_spec_feature_extractor(self, x):
        if is_augument and self.training:
            with tc.no_grad():
                if tc.rand([]) > 0.85:
                    x = tc.flip(x, dims=[-2])
                if tc.rand([]) > 0.85:
                    x = tc.flip(x, dims=[-1])
        h1 = tc.cat(
            [self.h1(x[:, i, ...].unsqueeze(1)).mean((-2, -1)) for i in range(4)], 1
        )
        h2 = self.h2(h1)
        return h2

    def eeg_img_feature_extractor(self, x):
        if is_augument and self.training:
            with tc.no_grad():
                if tc.rand([]) > 0.85:
                    x = tc.flip(x, dims=[-2])
                if tc.rand([]) > 0.85:
                    x = tc.flip(x, dims=[-1])
        i1 = tc.cat(
            [self.i1(x[:, i, ...].unsqueeze(1)).mean((-2, -1)) for i in range(16)], 1
        )
        i2 = self.i2(i1)
        return i2

    def eeg_img_feature_extractor2(self, x):
        with tc.no_grad():
            N, C, H, W = x.shape
            if is_augument and self.training:
                if tc.rand([]) > 0.85:
                    x = tc.flip(x, dims=[-2])
                if tc.rand([]) > 0.85:
                    x = tc.flip(x, dims=[-1])
            x = x.view(N, 4, 4, H, W)
        j1 = tc.cat([self.j1(x[:, :, i, ...]) for i in range(4)], 1)
        j2 = self.j2(j1)
        return j2

    def forward(self, x1, x2, x3, x4, x5):
        h1 = self.total_feature_extractor(x1)
        h2 = self.ind_feature_extractor(x1)
        h3 = self.montage_features_extractor(x2)
        h4 = self.gr_montage_features_extractor(x2)
        h5 = self.montage_features_extractor2(x3)
        h6 = self.kagg_spec_feature_extractor(x4)
        h7 = self.eeg_img_feature_extractor(x5)
        h8 = self.eeg_img_feature_extractor2(x5)
        h = tc.cat([h1, h2, h3, h4, h5, h6, h7, h8], 1)
        return h


class EEGBasedClassifier(nn.Module):
    def __init__(
        self, load_weights=True, load_out_weights=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feature_extractor = EEGFeatureExtractor()
        self.out = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(96 + 128 + 128 + 96 + 512 + 256 + 512 + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p_dropout),
            nn.Linear(512, 6),
        )

        self.load_out_weights = load_out_weights
        if load_weights:
            ckpt_path = "/kaggle/input/hms-dataset/HMS_EEG_MODEL.pth"
            if os.path.exists(ckpt_path):
                checkpoint = tc.load(ckpt_path, map_location=tc.device(device))["model"]
                new_weights = self.feature_extractor.state_dict()
                for key in checkpoint.keys():
                    if "out" in key:
                        continue
                    nkey = ".".join(key.split(".")[1:])
                    if (
                        nkey in new_weights
                        and new_weights[nkey].shape == checkpoint[key].shape
                    ):
                        new_weights[nkey] = checkpoint[key]
                self.feature_extractor.load_state_dict(new_weights, strict=False)

                if load_out_weights:
                    new_out = self.out.state_dict()
                    for key in checkpoint.keys():
                        if "out" in key:
                            nkey = ".".join(key.split(".")[1:])
                            if (
                                nkey in new_out
                                and new_out[nkey].shape == checkpoint[key].shape
                            ):
                                new_out[nkey] = checkpoint[key]
                    self.out.load_state_dict(new_out, strict=False)
            else:
                print(
                    f"[WARN] Checkpoint not found at {ckpt_path}. Running with randomly initialized weights."
                )

    def forward(self, x1, x2, x3, x4, x5):
        if self.load_out_weights:
            h = self.feature_extractor(x1, x2, x3, x4, x5)
        else:
            with tc.no_grad():
                h = self.feature_extractor(x1, x2, x3, x4, x5)
        out = self.out(h)
        return out




## === cell 8
model = EEGBasedClassifier(load_weights=True, load_out_weights=True).to(device)
n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("trainable params:", n_params)



## === cell 9
from tqdm import tqdm


def test(model, criterion, ret=False):
    test_dataset = TrainingDatasetEEG(df, train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 32),
        shuffle=False,
        num_workers=min(batch_size, 4),
    )
    print("Starting Testing")
    model.eval()
    running_loss = 0.0
    ya = []
    yp = []
    for i, data in enumerate(test_dataloader, 0):
        (x1, x2, x3, x4, x5), targets = data
        ya.append(targets.squeeze())

        with tc.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            with tc.no_grad():
                outputs = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                outputs = F.softmax(outputs, -1)
                outputs = tc.log(outputs + 1e-5)
                loss = criterion(outputs, targets.to(device))
                yp.append(outputs.detach().cpu().numpy().squeeze())

        running_loss += loss.item()
    print(f"Test Loss: {running_loss / max(i,1):.6f}")
    del loss, outputs

    print("Finished Testing")
    print("")
    if ret:
        return np.concatenate(ya), np.concatenate(yp)


def train(model, optimizer, criterion, scaler, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loss_count = 0

        for i, data in enumerate(train_dataloader, 0):
            (x1, x2, x3, x4, x5), targets = data

            with tc.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                outputs = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                outputs = F.log_softmax(outputs, -1)
                loss = criterion(outputs, targets.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            loss_count += 1
            if i % 100 == 99:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_count:.6f}"
                )
                running_loss = 0.0
                loss_count = 0
            del loss, outputs

        if scheduler is not None:
            scheduler.step()

        test(model, criterion)

    print("Finished Training")


def submit(model):
    dataset = DatasetEEG(df)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=min(batch_size, 4)
    )
    print("Starting submission inference")
    model.eval()
    rows = []

    for x1, x2, x3, x4, x5, eeg_ids in dataloader:
        with tc.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            with tc.no_grad():
                outputs = F.softmax(
                    model(
                        x1.to(device),
                        x2.to(device),
                        x3.to(device),
                        x4.to(device),
                        x5.to(device),
                    ),
                    -1,
                )

        outputs = outputs.detach().cpu().numpy()
        eeg_ids = np.asarray(eeg_ids)

        outputs = np.clip(outputs, 1e-12, 1.0)
        outputs = outputs / outputs.sum(axis=1, keepdims=True)

        for output, eeg_id in zip(outputs, eeg_ids):
            rows.append(
                {
                    "eeg_id": int(eeg_id),
                    "seizure_vote": float(output[0]),
                    "lpd_vote": float(output[1]),
                    "gpd_vote": float(output[2]),
                    "lrda_vote": float(output[3]),
                    "grda_vote": float(output[4]),
                    "other_vote": float(output[5]),
                }
            )

    sub = pd.DataFrame(rows)

    ss_path = f"{path}/sample_submission.csv"
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path)
        sub = ss[["eeg_id"]].merge(sub, on="eeg_id", how="left")
        vote_cols = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
        sub[vote_cols] = sub[vote_cols].fillna(1.0 / 6.0)
        sub[vote_cols] = sub[vote_cols].clip(1e-12, 1.0)
        sub[vote_cols] = sub[vote_cols].div(sub[vote_cols].sum(axis=1), axis=0)

    print("completed submission dataframe:", sub.shape)
    return sub




## === cell 10
if domain == "train":
    lr = 1e-6
    epochs = 100

    train_dataset = TrainingDatasetEEG(df, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(batch_size, 4),
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = tc.optim.AdamW(model.parameters(), lr=lr)
    scaler = tc.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train(model, optimizer, criterion, scaler, scheduler, epochs)
else:
    submission = submit(model)
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")
