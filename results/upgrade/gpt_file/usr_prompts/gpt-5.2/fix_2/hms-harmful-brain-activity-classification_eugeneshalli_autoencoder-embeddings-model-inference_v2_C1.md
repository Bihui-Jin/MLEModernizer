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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

1.0647442880318188

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the immediate import crash by removing the unused `umap` import that triggers a protobuf incompatibility in this environment. Then I make the model-weight loading robust by searching `/kaggle/input` for the expected `.pth` files (since the hardcoded dataset paths don’t exist), and if weights truly aren’t available, fall back to a valid uniform-probability submission so you always get a `.csv` to submit. Finally, I fix the dataset output type/shape for Albumentations (CHW→HWC) and ensure the output probabilities are finite and sum to 1 per row, preventing submission rejection.'

# 9. Code solution

## === cell 0
import os
import gc
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import albumentations as albu



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
TARGETS = train_df.columns[-6:].tolist()
TARGETS



## === cell 2
test = pd.read_csv(f"{DATA_DIR}/test.csv")
print("Test shape", test.shape)
test.head()



## === cell 3
import pywt, librosa

USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


def spectrogram_from_eeg(parquet_path, display=False, eeg_id=None):
    eeg = pd.read_parquet(parquet_path)

    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))

    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"EEG {eeg_id} - Spectrogram {NAMES[k]}")

    if display:
        plt.show()

    return img




## === cell 4
PATH_EEG_TEST = f"{DATA_DIR}/test_eegs/"

DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...\n")
for i, eeg_id in enumerate(tqdm(EEG_IDS2)):
    img = spectrogram_from_eeg(
        f"{PATH_EEG_TEST}{eeg_id}.parquet", display=(i < DISPLAY), eeg_id=eeg_id
    )
    all_eegs2[eeg_id] = img

print("Done. Built", len(all_eegs2), "test spectrogram tensors.")




## === cell 5
class EEGDataset(Dataset):
    def __init__(self, data, augment=False, mode="train", eeg_specs=None):
        self.data = data.reset_index(drop=True)
        self.augment = augment
        self.mode = mode
        self.eeg_specs = eeg_specs if eeg_specs is not None else {}

        self._aug = albu.Compose([albu.HorizontalFlip(p=0.5)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img = self.eeg_specs[row.eeg_id]  # HWC (128,256,4)
        if self.augment:
            img = self._aug(image=img)["image"]

        x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        if self.mode == "test":
            return x

        y = row[TARGETS].values.astype("float32")
        y = torch.from_numpy(y)
        return x, y




## === cell 6
class ResNetBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size, modify=False, bn=True, scale_factor=(1, 1)
    ):
        super().__init__()
        self.modify = modify
        if modify == "downsample":
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                stride=2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=in_channels * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            if bn:
                self.bn1 = nn.BatchNorm2d(in_channels * 2)
                self.bn2 = nn.BatchNorm2d(in_channels * 2)
            else:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()

        elif modify == "upsample":
            self.conv1 = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                stride=2,
                kernel_size=kernel_size,
                output_padding=scale_factor,
                padding=kernel_size // 2,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(in_channels // 2)
            self.bn2 = nn.BatchNorm2d(in_channels // 2)
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.conv2 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)

        self.act = nn.ReLU()

        if modify == "downsample":
            self.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                stride=2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        if modify == "upsample":
            self.proj = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                stride=2,
                kernel_size=kernel_size,
                output_padding=scale_factor,
                padding=kernel_size // 2,
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.modify:
            x = self.proj(x)
        out = x + out
        out = self.act(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, 7, 1, 7 // 2)
        self.rnb1 = ResNetBlock(16, 3, modify="downsample")
        self.rnb2 = ResNetBlock(32, 3, modify="downsample")
        self.rnb3 = ResNetBlock(64, 3, modify="downsample")
        self.rnb4 = ResNetBlock(128, 3, modify="downsample")
        self.rnb5 = ResNetBlock(256, 3, modify="downsample")
        self.rnb6 = ResNetBlock(512, 3, modify="downsample")
        self.rnb7 = ResNetBlock(1024, 3, modify="downsample")
        self.rnb8 = ResNetBlock(2048, 3, modify="downsample")

    def forward(self, x):
        x = self.conv(x)
        x = self.rnb1(x)
        x = self.rnb2(x)
        x = self.rnb3(x)
        x = self.rnb4(x)
        x = self.rnb5(x)
        x = self.rnb6(x)
        x = self.rnb7(x)
        x = self.rnb8(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnb1 = ResNetBlock(4096, 3, modify="upsample", scale_factor=(0, 1))
        self.rnb2 = ResNetBlock(2048, 3, modify="upsample")
        self.rnb3 = ResNetBlock(1024, 3, modify="upsample")
        self.rnb4 = ResNetBlock(512, 3, modify="upsample")
        self.rnb5 = ResNetBlock(256, 3, modify="upsample")
        self.rnb6 = ResNetBlock(128, 3, modify="upsample")
        self.rnb7 = ResNetBlock(64, 3, modify="upsample")
        self.rnb8 = ResNetBlock(32, 3, modify="upsample")
        self.conv = nn.Conv2d(16, 4, 3, 1, 3 // 2)

    def forward(self, x):
        x = self.rnb1(x)
        x = self.rnb2(x)
        x = self.rnb3(x)
        x = self.rnb4(x)
        x = self.rnb5(x)
        x = self.rnb6(x)
        x = self.rnb7(x)
        x = self.rnb8(x)
        x = self.conv(x)
        return x




## === cell 7
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(Encoder(), Decoder())

    def forward(self, x):
        return self.net(x)


class SimpleNet1(nn.Module):
    def __init__(self):
        super(SimpleNet1, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 128)
        self.output = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x




## === cell 8
def find_first_checkpoint(patterns, search_root="/kaggle/input"):
    root = Path(search_root)
    for pat in patterns:
        hits = list(root.rglob(pat))
        if hits:
            hits = sorted(hits, key=lambda p: str(p))
            return str(hits[0])
    return None


ae_ckpt = find_first_checkpoint(
    patterns=[
        "best_model.pth",
        "*autoencoder*.pth",
        "*ae*.pth",
    ],
    search_root="/kaggle/input",
)
clf_ckpt = find_first_checkpoint(
    patterns=[
        "model_simplenet_20_e4.pth",
        "*simplenet*.pth",
        "*SimpleNet*.pth",
        "*net*.pth",
    ],
    search_root="/kaggle/input",
)

print("Found AE checkpoint:", ae_ckpt)
print("Found CLF checkpoint:", clf_ckpt)



## === cell 9
device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = EEGDataset(test, mode="test", eeg_specs=all_eegs2)
test_loader = DataLoader(
    test_ds,
    shuffle=False,
    batch_size=64,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

pred = None

if ae_ckpt is None or clf_ckpt is None:
    print(
        "WARNING: Required model checkpoints were not found in /kaggle/input. Writing uniform-probability submission."
    )
    pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
else:
    preds = []
    for i in range(5):
        print("#" * 25)
        print(f"### Testing Fold {i+1}")

        model_ae = SimpleAE()
        model_ae = nn.DataParallel(model_ae)

        state_ae = torch.load(ae_ckpt, map_location="cpu")
        model_ae.load_state_dict(state_ae)

        model_ae.to(device).eval()
        encoder = nn.DataParallel(model_ae.module.net[0]).to(device).eval()

        model = SimpleNet1()
        state_clf = torch.load(clf_ckpt, map_location="cpu")
        model.load_state_dict(state_clf)
        model.to(device).eval()

        fold_preds = []
        with torch.inference_mode():
            for xb in test_loader:
                xb = xb.to(device, non_blocking=True)
                emb = encoder(xb)
                embs = emb.squeeze(2, 3)
                logits = model(embs)
                probs = F.softmax(logits, dim=1)
                fold_preds.append(probs.detach().cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        preds.append(fold_preds)

    pred = np.mean(preds, axis=0).astype(np.float32)

pred = np.nan_to_num(pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
row_sum = pred.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
pred = pred / row_sum

print(
    "\nTest preds shape",
    pred.shape,
    "row_sum range",
    pred.sum(1).min(),
    pred.sum(1).max(),
)



## === cell 10
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())



## === cell 11
sums = sub.iloc[:, -6:].sum(axis=1)
print("Row sum min/max:", sums.min(), sums.max())
