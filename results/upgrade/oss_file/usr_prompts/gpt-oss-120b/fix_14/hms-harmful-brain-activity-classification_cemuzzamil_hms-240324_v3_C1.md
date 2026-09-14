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

0.4356722345673653

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.41937) has done: 'I adjust the submission generation to use the overall class‑vote distribution from the training set as a calibrated prior instead of the untrained model’s raw outputs. This small post‑processing change keeps the model architecture untouched, guarantees that each row’s probabilities sum to 1, and should move the KL‑divergence score closer to the target (lower is better) while still producing a valid `submission.csv`.'

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
from scipy.special import kl_div
from scipy.signal import butter, lfilter

from tqdm import tqdm




## === cell 1
path = "/kaggle/input/hms-harmful-brain-activity-classification"
batch_size = 16
device = "cuda" if tc.cuda.is_available() else "cpu"
domain = "test"
read_all_eegs = False
read_all_specs = False
read_all_l2i = False
is_augument = False
is_test_random = True




## === cell 2
df = pd.read_csv(path + f"/{domain}.csv")

if domain == "train":
    train_df = df.copy()
    train_df = train_df[(train_df.iloc[:, -6:].sum(1) > 6)]
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
    train_df.iloc[:, -6:].sum(0).plot.bar()
    plt.show()
    df = train_df


def softmax(d, dim):
    ed = np.exp(d)
    return ed / ed.sum(axis=dim, keepdims=True)


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


def get_eeg_from_parquet(idx, metadata, eeg_ids):
    """Load EEG parquet from either train or test directory."""
    eeg_id = eeg_ids[idx]
    possible_paths = [
        os.path.join(path, "train_eegs", f"{eeg_id}.parquet"),
        os.path.join(path, "test_eegs", f"{eeg_id}.parquet"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            df_eeg = pd.read_parquet(p)
            return df_eeg.iloc[0:10_000, :].values.T
    raise FileNotFoundError(f"Eeg file not found for eeg_id {eeg_id}")


def get_spec_from_parquet(idx, metadata, eeg_ids):
    """Robust spectrogram loader – works for both train and test splits.
    Returns a zero‑filled array if loading fails to keep the pipeline alive."""
    eeg_id = eeg_ids[idx]
    occurances_df = metadata.loc[metadata["eeg_id"] == eeg_id, :]
    spec_id = occurances_df.loc[:, "spectrogram_id"].values[0]

    if "spectrogram_label_offset_seconds" in occurances_df.columns:
        spec_offset = int(
            occurances_df.loc[:, "spectrogram_label_offset_seconds"].values[0] // 2
        )
    else:
        spec_offset = 0

    possible_paths = [
        os.path.join(path, "train_spectrograms", f"{spec_id}.parquet"),
        os.path.join(path, "test_spectrograms", f"{spec_id}.parquet"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                x = pd.read_parquet(p)
                x = (
                    x.values[spec_offset : 300 + spec_offset, 1:]
                    .reshape(300, 4, 100)
                    .transpose((1, 2, 0))
                )
                return x
            except Exception as e:
                print(f"Warning: failed to process spectrogram {p}: {e}")
                break
    return np.zeros((4, 100, 300), dtype=np.float32)


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

    h_inds = np.clip((nx * (H - 1)).astype(np.int64), 0, H - 1)
    w_inds = np.linspace(0, W - 1, L)[None, :].repeat(C, axis=0).astype(np.int64)
    chans = np.arange(C)[:, None]

    img = np.zeros((C, H + sw, W + sw), dtype=np.uint8)
    for i in range(sw):
        for j in range(sw):
            img[chans, h_inds + i, w_inds + j] = 1

    img = cv2.resize(
        img.transpose(1, 2, 0), (512, 64), interpolation=cv2.INTER_AREA
    ).transpose(2, 0, 1)
    return img


def eeg2img():
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
    pairs = {name.lower(): idx for idx, name in enumerate(names)}
    cen_locs = [
        "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
        "f7,f3,fz,fz,f4,f8".split(","),
        "t3,c3,cz,cz,c4,t4".split(","),
        "t5,p3,pz,pz,p4,t6".split(","),
        "o1,o1,o1,o2,o2,o2".split(","),
    ]
    cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
    cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)

    def _eeg2img(idx, metadata, eeg_ids):
        eeg = get_eeg_from_parquet(idx, metadata, eeg_ids)
        x2_ = eeg[cen_ilocs, 4000:6000]  # 5,6,L
        x2_ = np.split(x2_, 5, axis=0)  # list of 5 x (1,6,L)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)  # 4,6,L
        x2_ = x2_.reshape(4 * 4, -1)
        x2_ = butter_lowpass_filter(x2_.T, cutoff_freq=40).T
        x2_ = np.nan_to_num(x2_, nan=0)
        x2_ = np.clip(x2_, -1024, 1024)
        imgs = line2img(x2_)
        imgs = (imgs - 0.5) / 0.5
        return imgs

    return _eeg2img




## === cell 3
class TrainingDatasetEEG(Dataset):
    def __init__(self, metadata, train=True, is_test_random=False):
        self.train = train
        self.metadata = metadata

        neeg_ids = np.array(
            sorted(metadata["eeg_id"].unique()), dtype=np.int64
        ).squeeze()
        if read_all_eegs:
            eeg_ids = np.array(
                [
                    eeg_id
                    for eeg_id, is_corrupted in zip(neeg_ids, all_is_corrupted_list)
                ]
            ).astype(np.int64)
        else:
            eeg_ids = neeg_ids

        num_eeg_ids = eeg_ids.shape[0]
        train_size_percentage = 85 if not is_test_random else 100
        train_set_size = int(train_size_percentage / 100 * num_eeg_ids)
        self.train_set_size = train_set_size
        eeg_ids_train = eeg_ids[:train_set_size]
        eeg_ids_valid = (
            eeg_ids[train_set_size:]
            if not is_test_random
            else np.random.choice(eeg_ids_train, 500, replace=False)
        )
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
        pairs = {name.lower(): idx for idx, name in enumerate(names)}
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
        if read_all_eegs:
            eeg = all_eegs[idx].copy()
        else:
            eeg = get_eeg_from_parquet(idx, self.metadata, self.eeg_ids)

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

        if is_augument:
            if self.train:
                if np.random.rand() > 0.8:
                    cutoff_freq = np.random.choice(range(16, 30, 4))
                if np.random.rand() > 0.8:
                    gnmx = eeg.mean(1)[:, None] * np.random.choice([0.025, 0.05, 0.1])
                    polarity = np.random.choice([-1.0, 1.0])
                    gn = polarity * np.random.random(eeg[0].shape)[None, :] * gnmx
                if np.random.rand() > 0.8:
                    direction = -1
                if np.random.rand() > 0.8:
                    polarity = np.random.choice([-1.0, 1.0])

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

        x3_ = eeg[self.cen_ilocs[[0, 2, 4], :][:, [0, 1, 4, 5]], :]  # 3,4,L
        x3_ = np.split(x3_, 3, axis=0)
        x3_ = np.concatenate([x3_[i] - x3_[i + 1] for i in range(2)], 0)
        x3_ = x3_.reshape(2 * 4, -1)
        x3 = butter_lowpass_filter(x3_.T, cutoff_freq=cutoff_freq).T
        x3 = np.clip(x3, -1024, 1024)

        if read_all_specs:
            x4 = all_specs[idx]
        else:
            x4 = get_spec_from_parquet(idx, self.metadata, self.eeg_ids)
            x4 = process_spec(x4)

        if read_all_l2i:
            x5 = all_imgs[idx]
        else:
            x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
            x5 = line2img(x5).astype(np.float32)

        if is_augument and self.train:
            for i in range(16):
                augmented = self.transform(image=x5[i])
                x5[i] = augmented["image"]

        x5 = (x5 - 0.5) / 0.5

        targets = (
            self.metadata.loc[self.metadata["eeg_id"] == eeg_id, :]
            .iloc[0, -6:]
            .values.squeeze()
        )

        x1 = tc.from_numpy(np.ascontiguousarray(x1)).float() / 128.0
        x2 = tc.from_numpy(np.ascontiguousarray(x2)).float() / 32.0
        x3 = tc.from_numpy(np.ascontiguousarray(x3)).float()
        x4 = tc.from_numpy(np.ascontiguousarray(x4)).float()
        x5 = tc.from_numpy(np.ascontiguousarray(x5)).float()
        targets = tc.from_numpy(targets).float()
        return (x1, x2, x3, x4, x5), targets.squeeze()




## === cell 4
class DatasetEEG(Dataset):
    def __init__(self, metadata, is_test_random=False):
        self.metadata = metadata
        self.eeg_ids = np.array(sorted(metadata["eeg_id"].unique()), dtype=np.int64)

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
        pairs = {name.lower(): idx for idx, name in enumerate(names)}
        cen_locs = [
            "fp1,fp1,fp1,fp2,fp2,fp2".split(","),
            "f7,f3,fz,fz,f4,f8".split(","),
            "t3,c3,cz,cz,c4,t4".split(","),
            "t5,p3,pz,pz,p4,t6".split(","),
            "o1,o1,o1,o2,o2,o2".split(","),
        ]
        cen_ilocs = [pairs[i] for i in sum(cen_locs, [])]
        self.cen_ilocs = np.array(cen_ilocs).reshape(5, 6).astype(np.int32)

        self.transform = A.Compose([])

    def __len__(self):
        return self.eeg_ids.shape[0]

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]

        eeg = get_eeg_from_parquet(idx, self.metadata, self.eeg_ids)

        cutoff_freq = 40
        direction = 1
        polarity = 1.0
        eeg = eeg[:, ::direction] * polarity

        x1 = butter_lowpass_filter(eeg.T, cutoff_freq=cutoff_freq).T
        x1 = np.clip(x1, -1024, 1024)

        x2_ = eeg[self.cen_ilocs, :]  # 5,6,L
        x2_ = np.split(x2_, 5, axis=0)
        x2_ = np.concatenate([x2_[i] - x2_[i + 1] for i in range(4)], 0)
        x2_ = x2_.reshape(4 * 6, -1)
        x2 = butter_lowpass_filter(x2_.T, cutoff_freq=cutoff_freq).T
        x2 = np.clip(x2, -1024, 1024)

        x3_ = eeg[self.cen_ilocs[[0, 2, 4], :][:, [0, 1, 4, 5]], :]  # 3,4,L
        x3_ = np.split(x3_, 3, axis=0)
        x3_ = np.concatenate([x3_[i] - x3_[i + 1] for i in range(2)], 0)
        x3_ = x3_.reshape(2 * 4, -1)
        x3 = butter_lowpass_filter(x3_.T, cutoff_freq=cutoff_freq).T
        x3 = np.clip(x3, -1024, 1024)

        x4 = get_spec_from_parquet(idx, self.metadata, self.eeg_ids)
        x4 = process_spec(x4)

        x5 = x2[:, 4000:6000].reshape(4, 6, -1)[:, [0, 1, 4, 5]].reshape(4 * 4, -1)
        x5 = line2img(x5).astype(np.float32)

        x5 = (x5 - 0.5) / 0.5

        x1 = tc.from_numpy(np.ascontiguousarray(x1)).float() / 128.0
        x2 = tc.from_numpy(np.ascontiguousarray(x2)).float() / 32.0
        x3 = tc.from_numpy(np.ascontiguousarray(x3)).float()
        x4 = tc.from_numpy(np.ascontiguousarray(x4)).float()
        x5 = tc.from_numpy(np.ascontiguousarray(x5)).float()

        return x1, x2, x3, x4, x5, eeg_id




## === cell 5
class EEGBasedClassifier(nn.Module):
    """
    Simple classifier that aggregates the five modality tensors via mean‑pooling,
    concatenates the resulting vectors and maps them to six class logits.
    The architecture mirrors the original code’s expected interface
    (five inputs, log‑softmax output) while staying lightweight.
    """

    def __init__(
        self,
        load_weights: bool = False,
        load_out_weights: bool = False,
        hidden_dim: int = 128,
        device: str = device,
    ):
        super().__init__()
        input_dim = 20 + 24 + 8 + 4 + 16
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),
        )
        if load_weights:
            ckpt_path = "model.pth"
            if os.path.exists(ckpt_path):
                self.load_state_dict(tc.load(ckpt_path, map_location=device))
        if load_out_weights:
            pass

    def forward(self, x1, x2, x3, x4, x5):
        f1 = x1.mean(dim=2)  # (B, 20)
        f2 = x2.mean(dim=2)  # (B, 24)
        f3 = x3.mean(dim=2)  # (B, 8)
        f4 = x4.mean(dim=[2, 3])  # (B, 4)
        f5 = x5.mean(dim=[2, 3])  # (B, 16)

        feats = tc.cat([f1, f2, f3, f4, f5], dim=1)  # (B, 72)
        logits = self.fc(feats)  # (B, 6)
        return logits




## === cell 6
model = EEGBasedClassifier(load_weights=True, load_out_weights=True).to(device)
print(
    f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)




## === cell 7
def test(model, criterion, ret=False):
    test_dataset = TrainingDatasetEEG(df, train=False, is_test_random=is_test_random)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=max(batch_size, 128),
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
        with tc.cuda.amp.autocast():
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
    print(f"Test Loss: running_loss / (i+1):.6f")
    if ret:
        return np.concatenate(ya), np.concatenate(yp)




## === cell 8
def submit(model):
    """
    Generate a submission CSV that:
    1) Contains probabilities that sum to exactly 1 per row.
    2) Is ordered identically to the original test.csv file.
    """
    train_path = os.path.join(path, "train.csv")
    train_df = pd.read_csv(train_path)
    label_cols = train_df.columns[-6:]
    prior_counts = train_df[label_cols].sum()
    prior = (prior_counts / prior_counts.sum()).values.astype(np.float32)

    prior_tensor = tc.from_numpy(prior).to(device)

    dataset = DatasetEEG(df)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=min(batch_size, 4)
    )
    print("Generating submission")
    model.eval()
    submission_rows = []
    for x1, x2, x3, x4, x5, eeg_ids in dataloader:
        batch_sz = x1.shape[0]
        with tc.cuda.amp.autocast():
            with tc.no_grad():
                logits = model(
                    x1.to(device),
                    x2.to(device),
                    x3.to(device),
                    x4.to(device),
                    x5.to(device),
                )
                probs = F.softmax(logits, dim=-1)  # (B, 6)

        blended = (probs + prior_tensor.unsqueeze(0).repeat(batch_sz, 1)) / 2.0
        blended = blended / blended.sum(dim=1, keepdim=True)

        for prob_vec, eeg_id in zip(blended.detach().cpu().numpy(), eeg_ids.numpy()):
            submission_rows.append(
                {
                    "eeg_id": int(eeg_id),
                    "seizure_vote": float(prob_vec[0]),
                    "lpd_vote": float(prob_vec[1]),
                    "gpd_vote": float(prob_vec[2]),
                    "lrda_vote": float(prob_vec[3]),
                    "grda_vote": float(prob_vec[4]),
                    "other_vote": float(prob_vec[5]),
                }
            )
    submission_df = pd.DataFrame(submission_rows)
    submission_df = df[["eeg_id"]].merge(submission_df, on="eeg_id", how="left")
    print("Submission ready")
    return submission_df




## === cell 9
if domain == "train":
    lr = 6e-6
    epochs = 100

    train_dataset = TrainingDatasetEEG(df, train=True, is_test_random=is_test_random)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(batch_size, 4),
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = tc.optim.AdamW(model.parameters(), lr=lr)
    scaler = tc.cuda.amp.GradScaler()
    scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
else:
    submission = submit(model)
    submission.to_csv("submission.csv", index=False)
