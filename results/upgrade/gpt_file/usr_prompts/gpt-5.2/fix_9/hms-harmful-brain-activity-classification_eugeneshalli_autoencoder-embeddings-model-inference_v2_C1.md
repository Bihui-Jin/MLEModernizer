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

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate import crash by removing the unused `umap` import that triggers a protobuf incompatibility in this environment. Then I make the model-weight loading robust by searching `/kaggle/input` for the expected `.pth` files (since the hardcoded dataset paths don’t exist), and if weights truly aren’t available, fall back to a valid uniform-probability submission so you always get a `.csv` to submit. Finally, I fix the dataset output type/shape for Albumentations (CHW→HWC) and ensure the output probabilities are finite and sum to 1 per row, preventing submission rejection.'
- What this solution (achieved 1.40995) has done: 'Your current score is worse than the target (lower is better), and the biggest reason is that you’re effectively using a single checkpoint (or falling back to uniform) while looping “5 folds” without actually loading different fold weights. I keep your model and feature logic identical, but make checkpoint discovery fold-aware so each fold loads its own weights when available; otherwise it gracefully fall back to the single checkpoint (preserving current behavior). I also ensure the autoencoder weights are loaded safely whether the checkpoint was saved from a `DataParallel` model or not (common Kaggle pitfall that can silently break or degrade embeddings). Finally, I keep the submission post-processing (finite + row-normalized) unchanged to preserve valid CSV semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (1.0647), so we should improve predictive quality while keeping your model/feature pipeline unchanged. The biggest low-risk gain here is fixing the test-time augmentation pipeline: Albumentations `HorizontalFlip` is inappropriate for this spectrogram representation and can significantly degrade predictions, so we disable it for test inference while preserving the dataset/model logic. Additionally, we make checkpoint selection stricter so that each fold uses a distinct checkpoint when available (avoiding accidentally reusing the same weight file across folds). Finally, we keep the same probability normalization to ensure valid submissions and stable KL behavior.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is worse than the target (1.0647), so we should improve predictive quality with very small, low-risk inference fixes. The biggest likely issue is that some checkpoints were trained under `DataParallel` and your classifier is *not* wrapped, so keys mismatch and `strict=False` can silently leave parts of the classifier uninitialized, hurting score; we load the classifier weights with an explicit “strip/add `module.` prefix” fallback before allowing any non-strict load. Next, we ensure the encoder output is flattened robustly (avoiding `squeeze` shape surprises) and we add a tiny probability floor + renormalization (still valid probabilities) which typically improves KL stability without changing core logic. No model architecture, feature extraction, or training loop is changed; only safer checkpoint loading and numerically-stable post-processing.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower is better) is still far from the target (1.0647), so we should improve *inference correctness* with minimal risk while keeping your model/feature pipeline unchanged. The biggest likely issue is that you always wrap the AE in `DataParallel`, which can cause key-mismatch and partial/non-loading depending on how the checkpoint was saved; we instead instantiate AE **without** `DataParallel` and load weights flexibly, only wrapping the encoder for multi-GPU if needed. Next, we ensure the classifier checkpoint is searched/selected more reliably by preferring filenames that contain the fold key *and* “simplenet” to reduce the chance of accidentally loading unrelated `.pth` files. Finally, we keep the exact same softmax + probability floor + row-normalization so submissions remain valid and KL-stable.'
- What this solution (achieved 1.40995) has done: 'I make two minimal inference-only fixes that usually improve KL without changing your model or feature extraction. First, I remove the accidental dependency on `test` row order by building predictions per unique `eeg_id` (your dataset returns one spectrogram per `eeg_id`, but you were averaging/assigning per-row), then map back to `test.eeg_id` so every row is correctly aligned. Second, I disable the inappropriate Albumentations `HorizontalFlip` for test (already off) and also ensure the dataset never even constructs that augmentation in test mode to avoid any chance of accidental usage; everything else (AE/encoder, classifier, softmax, normalization, checkpoint loading) stays the same. These are small correctness fixes that should move the score down (better) toward the 1.0647 target while keeping behavior stable and producing a valid submission CSV.'
- What this solution (achieved 1.40995) has done: 'We make two minimal inference-only fixes that tend to improve KL without changing your model, features, or training approach. First, we ensure the classifier’s `Dropout` is definitely disabled by setting `model.eval()` (already done) and also forcing deterministic inference settings (no randomness) so the ensemble mean is stable and not accidentally noisy across runs. Second, we adjust the KL-stability post-processing to use a slightly larger probability floor (still tiny) and renormalize; with KL-div, overly confident near-zero probabilities can hurt more than they help, and this is a low-risk way to move the score down toward your 1.0647 target. Everything else (spectrogram creation, AE encoder, SimpleNet1, checkpoint loading/selection, and submission schema) stays the same and still produces `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I make two minimal inference-only changes aimed at lowering KL (improving score) without changing your model/feature pipeline: (1) apply temperature scaling to soften probabilities (KL heavily penalizes overconfident wrong predictions), and (2) ensemble each model’s probabilities with a small uniform prior (Dirichlet-like smoothing) to reduce extreme near-zeros. Both are simple post-processing steps that preserve semantics (still a valid probability distribution per row) and are often a low-risk way to move from ~1.41 toward ~1.06 on this metric. I keep checkpoint discovery/loading and spectrogram creation unchanged, and still write a valid `submission.csv`. Parameters are set conservatively to improve calibration rather than chase maximum performance.'

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

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



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

        if self.mode != "test":
            self._aug = albu.Compose([albu.HorizontalFlip(p=0.5)])
        else:
            self._aug = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img = self.eeg_specs[row.eeg_id]  # HWC (128,256,4)
        if self.augment and self._aug is not None:
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
def find_all_checkpoints(patterns, search_root="/kaggle/input"):
    root = Path(search_root)
    hits_all = []
    for pat in patterns:
        hits = list(root.rglob(pat))
        hits_all.extend(hits)
    uniq = sorted({str(p) for p in hits_all})
    return uniq


def pick_fold_checkpoint(candidates, fold_idx, prefer_substrings=()):
    """
    Change rationale (score): reduce accidental selection of unrelated .pth files by preferring
    filenames containing both fold key and model-type substrings (e.g., 'simplenet').
    """
    if not candidates:
        return None
    keys = [
        f"fold{fold_idx+1}",
        f"fold_{fold_idx+1}",
        f"fold-{fold_idx+1}",
        f"fold{fold_idx}",
        f"fold_{fold_idx}",
        f"fold-{fold_idx}",
        f"f{fold_idx+1}",
        f"f{fold_idx}",
    ]
    cand_sorted = list(candidates)
    if prefer_substrings:
        pref = []
        other = []
        for p in cand_sorted:
            lp = str(p).lower()
            if all(s.lower() in lp for s in prefer_substrings):
                pref.append(p)
            else:
                other.append(p)
        cand_sorted = pref + other

    for k in keys:
        for p in cand_sorted:
            if k.lower() in str(p).lower():
                return str(p)
    return None


def _extract_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        return state["state_dict"]
    return state


def load_state_dict_flexible(model, state, allow_non_strict_fallback=True):
    """
    Change rationale (score): avoid partial/non-loading due to 'module.' prefix mismatch.
    Try strict loads after stripping/adding module.; only then (optionally) strict=False.
    """
    state = _extract_state_dict(state)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a state_dict-like dict.")

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    state_keys = set(state.keys())

    if state_keys == model_keys:
        model.load_state_dict(state, strict=True)
        return True

    if any(k.startswith("module.") for k in state.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        if set(stripped.keys()) == model_keys:
            model.load_state_dict(stripped, strict=True)
            return True

    added = {("module." + k): v for k, v in state.items()}
    if set(added.keys()) == model_keys:
        model.load_state_dict(added, strict=True)
        return True

    if allow_non_strict_fallback:
        model.load_state_dict(state, strict=False)
        return False

    raise RuntimeError("Could not align checkpoint keys to model keys.")


ae_candidates = find_all_checkpoints(
    patterns=[
        "best_model.pth",
        "*autoencoder*.pth",
        "*auto*encoder*.pth",
        "*ae*.pth",
        "*AE*.pth",
    ],
    search_root="/kaggle/input",
)
clf_candidates = find_all_checkpoints(
    patterns=[
        "model_simplenet_20_e4.pth",
        "*simplenet*.pth",
        "*SimpleNet*.pth",
        "*net*.pth",
    ],
    search_root="/kaggle/input",
)

print("AE candidates found:", len(ae_candidates))
print("CLF candidates found:", len(clf_candidates))
print("Example AE candidate:", ae_candidates[0] if ae_candidates else None)
print("Example CLF candidate:", clf_candidates[0] if clf_candidates else None)



## === cell 9
device = "cuda" if torch.cuda.is_available() else "cpu"

test_unique = test[["eeg_id"]].drop_duplicates().reset_index(drop=True)

test_ds = EEGDataset(test_unique, mode="test", augment=False, eeg_specs=all_eegs2)
test_loader = DataLoader(
    test_ds,
    shuffle=False,
    batch_size=64,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

pred_unique = None

TEMPERATURE = 1.25  # soften logits -> less peaky probs
UNIFORM_MIX = 0.06  # mix a small uniform prior to avoid near-zero probabilities

if len(ae_candidates) == 0 or len(clf_candidates) == 0:
    print(
        "WARNING: Required model checkpoints were not found in /kaggle/input. Writing uniform-probability submission."
    )
    pred_unique = np.full((len(test_unique), 6), 1.0 / 6.0, dtype=np.float32)
else:
    preds = []
    used_pairs = set()
    for i in range(5):
        print("#" * 25)
        print(f"### Testing Fold {i+1}")

        ae_ckpt = pick_fold_checkpoint(ae_candidates, i, prefer_substrings=())
        clf_ckpt = pick_fold_checkpoint(
            clf_candidates, i, prefer_substrings=("simplenet",)
        )

        if ae_ckpt is None:
            ae_ckpt = ae_candidates[0]
        if clf_ckpt is None:
            clf_ckpt = clf_candidates[0]

        pair = (ae_ckpt, clf_ckpt)
        if pair in used_pairs:
            print("Fold ckpt pair already used; skipping to avoid overweighting:", pair)
            continue
        used_pairs.add(pair)

        print("Using AE ckpt:", ae_ckpt)
        print("Using CLF ckpt:", clf_ckpt)

        model_ae = SimpleAE()
        state_ae = torch.load(ae_ckpt, map_location="cpu")
        load_state_dict_flexible(model_ae, state_ae, allow_non_strict_fallback=True)
        model_ae.to(device).eval()

        encoder = model_ae.net[0].to(device).eval()
        if torch.cuda.device_count() > 1:
            encoder = nn.DataParallel(encoder)

        model = SimpleNet1()
        state_clf = torch.load(clf_ckpt, map_location="cpu")
        load_state_dict_flexible(model, state_clf, allow_non_strict_fallback=False)
        model.to(device).eval()

        fold_preds = []
        with torch.inference_mode():
            for xb in test_loader:
                xb = xb.to(device, non_blocking=True)
                emb = encoder(xb)
                embs = torch.flatten(emb, start_dim=1)
                logits = model(embs)

                logits = logits / TEMPERATURE

                probs = F.softmax(logits, dim=1)

                probs = (1.0 - UNIFORM_MIX) * probs + (UNIFORM_MIX / probs.shape[1])

                fold_preds.append(probs.detach().cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        preds.append(fold_preds)

        del model_ae, encoder, model, state_ae, state_clf, fold_preds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(preds) == 0:
        print(
            "WARNING: No fold predictions were produced; writing uniform-probability submission."
        )
        pred_unique = np.full((len(test_unique), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        pred_unique = np.mean(preds, axis=0).astype(np.float32)

pred_unique = np.nan_to_num(
    pred_unique, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0
).astype(np.float32)

pred_unique = np.clip(pred_unique, 1e-4, 1.0)
row_sum = pred_unique.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
pred_unique = pred_unique / row_sum

print(
    "\nUnique EEG preds shape",
    pred_unique.shape,
    "row_sum range",
    pred_unique.sum(1).min(),
    pred_unique.sum(1).max(),
)

pred_map = {eid: pred_unique[i] for i, eid in enumerate(test_unique.eeg_id.values)}
pred = np.vstack([pred_map[eid] for eid in test.eeg_id.values]).astype(np.float32)



## === cell 10
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())



## === cell 11
sums = sub.iloc[:, -6:].sum(axis=1)
print("Row sum min/max:", sums.min(), sums.max())
