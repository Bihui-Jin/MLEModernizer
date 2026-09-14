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

0.6075068408753687

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with the correct columns and row-wise probabilities summing to 1. The main runtime blockers are missing external model-weight directories (causing `FileNotFoundError`), an undefined `targets` variable in the dataset, incorrect `predictions3` assignment, and excessive `print()` inside `__getitem__` that would make inference unusably slow. Since no valid score has been produced yet, I keep the core model/inference logic intact but add a safe fallback that outputs the class prior from `train.csv` when no pretrained weights are available, ensuring a reasonable KL baseline instead of crashing. I also ensure deterministic ordering and numeric stability in the final normalization.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far above the target (0.6075), so we should improve accuracy with the smallest changes that don’t alter the core model/training setup. The biggest, low-risk gain here is fixing an ensemble bug: you’re accidentally applying the resnet50 predictions under the `r34` weight and resnet34 under `r50`, which likely hurts calibration and increases KL. I also fix a dataset bug where EEG features get overwritten inside the region loop (it doesn’t change architecture, just corrects the intended feature construction) and make the spectrogram window for test deterministic and centered instead of always starting at 0, which better matches the train-time intent and should reduce distribution shift. Everything else (models, inference, softmax, submission format, and prior fallback) stays the same.'
- What this solution (achieved 1.41937) has done: 'I make two minimal inference-time fixes that should reduce your KL (lower-is-better) without changing the model architectures or training approach: (1) fix the ResNet input tensor ordering so it matches what timm ResNets expect (N,3,H,W), because your current dataset produces (N,256,16,128) which badly scrambles channels/spatial axes and strongly harms predictions. (2) clamp/clean the spectrogram window index `r` for train-mode (even though you’re using test-mode here) to avoid occasional out-of-bounds / negative indexing if you reuse the dataset later, keeping evaluation semantics intact. Everything else (feature construction, weights loading, ensembling weights, softmax, and submission normalization) stays the same so runtime and behavior remain stable.'
- What this solution (achieved 1.41937) has done: 'I make two minimal, score-relevant fixes without changing your model architectures or inference approach: (1) remove a redundant/incorrect second `permute` in the dataset that scrambles the input tensor layout before it reaches timm models (this is very likely inflating KL), and (2) ensure the tensor is returned as `float32` in `NCHW` format as timm expects. Everything else (spectrogram/EEG feature construction, weight loading, softmax, ensembling, and submission writing) stays the same to keep behavior stable and runtime within limits. These changes should improve predictive correctness and calibration, moving the KL (lower is better) toward your target.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.41937, lower-is-better) is far from the target (0.6075), so we should improve prediction correctness with the smallest possible, score-relevant fixes. The biggest issue is that your dataset returns a 12-channel tensor (because you triple the 4 channels), while timm ResNets/ViT are trained/defined for 3-channel input—this channel mismatch typically destroys feature extraction and inflates KL even if code runs. I keep your exact feature construction, models, and ensembling logic, but change only the final tensor formatting so the model receives a proper 3-channel image derived from your existing 4-channel representation. I also enforce a stable, centered spectrogram crop for test (already intended in your code) and keep the prior fallback/normalization to guarantee a valid submission.'
- What this solution (achieved 1.41937) has done: 'I make two minimal, score-relevant fixes that preserve your modeling/inference core logic and keep the same submission semantics. First, I adapt the dataset output to exactly match what timm ResNets/ViT expect: a 3-channel image with a square-ish aspect ratio, by resizing the constructed 3-channel tensor to (224,224) for *all* models (currently ResNets get a (3,128,512) tensor, which is a big distribution mismatch and tends to hurt KL). Second, I avoid recomputing the dataset/loader inside every checkpoint loop (same predictions, just faster/less variance from repeated construction), keeping runtime safely under limits while not altering the ensemble logic. Everything else (feature construction, models, softmax, ensembling weights, prior fallback, and row-wise normalization) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import pywt
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    model1 = "resnet50d"
    model2 = "vit_base_patch16_224"
    model3 = "resnet34d"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
    FOLDS = 5
    AMP = True


class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    )
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    )
    out = "/kaggle/working/"




## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


def spectrogram_from_eeg(parquet_path, display=False):
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
            plt.title(f"Spectrogram {NAMES[k]}")

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
        plt.title("EEG Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img




## === cell 3
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")



## === cell 4
all_eegs = {}
test_eeg_files = sorted(
    [f for f in os.listdir(paths.test_eeg) if f.endswith(".parquet")]
)

for i in tqdm(test_eeg_files, desc="Building EEG mel-specs"):
    sp = spectrogram_from_eeg(os.path.join(paths.test_eeg, i))
    name = int(i.split(".")[0])
    all_eegs[name] = np.array(sp)



## === cell 5
all_eegs



## === cell 6
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 7
import os

all_spectrograms = {}
test_spec_files = sorted(
    [f for f in os.listdir(paths.test_spec) if f.endswith(".parquet")]
)

for i in tqdm(test_spec_files, desc="Loading test spectrograms"):
    sp = pd.read_parquet(os.path.join(paths.test_spec, i))
    name = int(i.split(".")[0])
    all_spectrograms[name] = np.array(sp)



## === cell 8
all_spectrograms



## === cell 9
targets = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = all_spectrograms,
        eegs: dict[int, np.ndarray] = all_eegs,
        out_size: int = 224,
    ):
        self.traindf = traindf
        self.specs = specs
        self.eeg = eegs
        self.mode = mode
        self.out_size = out_size

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.traindf.iloc[idx]

        spec = self.specs[row.spectrogram_id]
        if self.mode == "test":
            r = max(0, (spec.shape[0] - 300) // 2)
        else:
            r = int((row["min"] + row["max"]) // 4)
            r = max(0, min(r, max(0, spec.shape[0] - 300)))

        img_eeg = self.eeg[row.eeg_id]
        X[:, :, 4:] = img_eeg

        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        X = torch.from_numpy(X).to(torch.float32)  # (H, W, 8)

        spectograms = X[:, :, 0:4]  # (H,W,4)
        eegs = X[:, :, 4:8]  # (H,W,4)

        x4 = torch.cat([spectograms, eegs], dim=1)  # (H, 2W, 4) => (128,512,4)

        x3 = x4[:, :, 0:3].clone()
        x3[:, :, 2] = x3[:, :, 2] + 0.5 * x4[:, :, 3]

        x = x3.permute(2, 0, 1).contiguous()  # (C,H,W) = (3,128,512)

        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.mode != "test":
            y = row[targets].values.astype(np.float32)

        return {"data": x, "target": y}




## === cell 10
customdataset = CustomDataset(test_df, config, mode="test")



## === cell 11
customdataset[0]



## === cell 12
test_df.iloc[0].spectrogram_id



## === cell 13
from torch.utils.data import DataLoader

test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
X = customdataset[0]["data"]
y = customdataset[0]["target"]
y




## === cell 14
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model1,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 15
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in tqdm(test_loader, desc="Inference", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            ypred = model(x)
            ypred = softmax(ypred)
        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 16
def try_load_folder_predictions(folder, model_ctor, test_loader):
    if (folder is None) or (not os.path.isdir(folder)):
        return None
    ckpts = sorted([f for f in os.listdir(folder) if not f.startswith(".")])
    if len(ckpts) == 0:
        return None

    fold_preds = []
    for f in ckpts:
        ckpt_path = os.path.join(folder, f)
        dd = torch.load(ckpt_path, map_location="cpu")
        model = model_ctor(config)
        state = dd.get("model", dd)
        model.load_state_dict(state, strict=True)
        model.to(device)
        pred = inference_function(test_loader, model, device)["predictions"]
        fold_preds.append(pred)

    fold_preds = np.array(fold_preds, dtype=np.float32)
    return fold_preds.mean(axis=0)


predictions = try_load_folder_predictions(
    "/kaggle/input/resnet5010ep2", Custommodel, test_loader
)
predictions




## === cell 17
class Custommodelkk(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodelkk, self).__init__()
        self.model = timm.create_model(
            config.model3,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 18
predictions3 = try_load_folder_predictions(
    "/kaggle/input/resnet34d2", Custommodelkk, test_loader
)
predictions3



## === cell 19
predictions



## === cell 20
from torchvision.transforms import transforms

tras = transforms.Compose([transforms.Resize((224, 224))])


class Custommodel2(nn.Module):
    def __init__(self, config, transform, numclass: int = 6):
        super(Custommodel2, self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained=False,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x




## === cell 21
predictions2 = try_load_folder_predictions(
    "/kaggle/input/visiontransformer",
    lambda cfg: Custommodel2(cfg, tras),
    test_loader,
)
predictions2



## === cell 22
train_df = pd.read_csv(paths.train_csv)
prior = train_df[targets].sum(axis=0).values.astype(np.float32)
prior = prior / prior.sum()
prior_preds = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)

if predictions is None:
    predictions = prior_preds
if predictions2 is None:
    predictions2 = prior_preds
if predictions3 is None:
    predictions3 = prior_preds

predictions = np.asarray(predictions, dtype=np.float32)
predictions2 = np.asarray(predictions2, dtype=np.float32)
predictions3 = np.asarray(predictions3, dtype=np.float32)

r34 = 0.40
r50 = 0.40
vit = 0.2

finalpred = predictions3 * r34 + predictions2 * vit + predictions * r50

finalpred = np.clip(finalpred, 1e-8, 1.0)
finalpred = finalpred / np.sum(finalpred, axis=1, keepdims=True)

finalpred[:2], finalpred.shape, finalpred.sum(axis=1).min(), finalpred.sum(axis=1).max()



## === cell 23
finalpred



## === cell 24
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = finalpred
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(
    "Row-sum min/max:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
sub.head()



## === cell 25
print(np.sum(finalpred))
