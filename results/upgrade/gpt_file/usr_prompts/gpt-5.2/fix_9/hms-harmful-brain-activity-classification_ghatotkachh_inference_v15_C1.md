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

0.5421195897242879

# 6. Current score

1.40893

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime errors by correcting missing variables (`targets`), removing debug prints that would stall inference, and ensuring the dataset indexing logic works for test data without relying on non-existent `min/max` columns. Since the referenced pretrained weight directories do not exist in your environment, I replace that section with a score-safe fallback that outputs a properly normalized probability distribution (uniform across the 6 classes) so a valid `submission.csv` is always produced. I also fix the cell numbering to be sequential and ensure the final probabilities sum to 1 per row to avoid submission rejection. These changes are minimal and focused on making the notebook run end-to-end and generate a valid CSV.'
- What this solution (achieved 1.39779) has done: 'The crash comes from merging `sub` with `sample_sub` in a way that explodes the number of rows (cartesian product due to duplicate `eeg_id` values), so `finalpred` (9850 rows) can’t be assigned into a 778,942-row dataframe. I fix this by simply using `sample_submission.csv` as the submission template (it already has the exact required ordering/length) and then assigning predictions by aligned row order after asserting lengths match. I also add small safety checks to guarantee `finalpred` has the right shape and that each row sums to 1 (required for submission validity). Core modeling/inference logic is unchanged.'
- What this solution (achieved 1.40395) has done: 'Your current score (1.39779, lower is better) is far from the target (0.5421), so we should improve legitimately without changing your core model/dataset logic. The biggest practical issue is that you almost certainly fall back to the “prior mean” because the referenced weight directories don’t exist; we instead use `timm` pretrained backbones (same architectures) to produce non-uniform, data-dependent predictions, which typically reduces KL noticeably versus a constant prior. To better match the competition’s KL on soft labels (vote distributions), we also add a minimal temperature scaling + blending with the empirical prior (both are post-processing/calibration; no training loop changes). Finally, we keep your submission alignment/normalization safeguards unchanged to ensure a valid CSV.'
- What this solution (achieved 1.39138) has done: 'We keep your exact feature construction and both `timm` pretrained backbones, but fix a key evaluation-metric mismatch: KL is computed against *soft vote distributions*, so returning raw ImageNet-pretrained outputs is poorly calibrated. We add a minimal, deterministic post-processing step that (a) renormalizes model probabilities after a small “power/Dirichlet” smoothing toward the uniform distribution, and (b) slightly increases the prior blend to reduce overconfident mistakes—both typically reduce KL without changing core modeling. We also make the prior computation more faithful by using the *aggregated per-eeg_id* vote distribution (test is per eeg_id), which is a small but meaningful alignment improvement. All changes are limited to post-processing and prior estimation; the dataset, models, and inference loops remain the same, and a valid `submission.csv` is always produced.'
- What this solution (achieved 1.41201) has done: 'We keep your exact feature extraction and both `timm` pretrained models, but improve how their probabilities are calibrated for KL on soft targets by tuning only the post-processing. Specifically, we (1) compute a more faithful prior by weighting each `eeg_id` by its total votes (so high-consensus labels influence the prior more), and (2) slightly adjust temperature, smoothing power, and prior blending to reduce overconfident errors that inflate KL. We also make inference deterministic and use autocast during inference (no change to logic) to stabilize outputs and keep runtime within limits. The submission writing/alignment logic remains the same and still guarantees per-row probability sums to 1.'
- What this solution (achieved 1.41124) has done: 'Your current KL (1.412) is far above the target (0.542), so we should legitimately improve calibration toward softer, less overconfident probabilities (KL heavily punishes confident mistakes). I keep your exact feature creation and both `timm` pretrained models, but make two minimal metric-aligned changes: (1) blend with the empirical prior using a per-row uncertainty gate (more prior when the ensemble is peaky), and (2) use a slightly higher temperature plus slightly stronger smoothing to reduce overconfidence. These are pure post-processing changes (no architecture/training/feature changes) and preserve submission formatting and normalization guarantees.'
- What this solution (achieved 1.40893) has done: 'We keep your exact feature construction, models, and inference loops, but fix a metric-alignment issue that likely keeps KL high: the competition evaluates per `eeg_id` soft-label distribution, so averaging predictions across duplicate `eeg_id` (if any) and then re-expanding to the required row order reduces noise and typically lowers KL. We also make the prior blend slightly more conservative and stable by blending toward the vote-weighted prior with a small constant floor (prevents overconfident spikes that KL punishes), without changing the core modeling. Finally, we keep your strict normalization checks and ensure the produced `submission.csv` matches `sample_submission.csv` exactly.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
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

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 2
class config:
    model1 = "resnet34d"
    model2 = "vit_base_patch16_224"
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




## === cell 3
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
        plt.title("Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img




## === cell 4
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 5
all_eegs = {}
for fn in os.listdir(paths.test_eeg):
    if not fn.endswith(".parquet"):
        continue
    eeg_id = int(fn.split(".")[0])
    sp = spectrogram_from_eeg(os.path.join(paths.test_eeg, fn))
    all_eegs[eeg_id] = np.asarray(sp, dtype=np.float32)

print("Loaded test EEG-derived specs:", len(all_eegs))



## === cell 6
all_spectrograms = {}
for fn in os.listdir(paths.test_spec):
    if not fn.endswith(".parquet"):
        continue
    spec_id = int(fn.split(".")[0])
    sp = pd.read_parquet(os.path.join(paths.test_spec, fn))
    all_spectrograms[spec_id] = np.asarray(sp, dtype=np.float32)

print("Loaded test spectrograms:", len(all_spectrograms))



## === cell 7
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
targets = TARGETS  # keep original variable name used later




## === cell 8
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = None,
        eegs: dict[int, np.ndarray] = None,
    ):
        self.traindf = traindf
        self.specs = specs if specs is not None else {}
        self.eeg = eegs if eegs is not None else {}
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.traindf.iloc[idx]

        if self.mode == "test":
            spec_arr = self.specs[int(row.spectrogram_id)]
            r = max(0, (spec_arr.shape[0] - 300) // 2)
        else:
            if ("min" in row.index) and ("max" in row.index):
                r = int((row["min"] + row["max"]) // 4)
            else:
                spec_arr = self.specs[int(row.spectrogram_id)]
                r = max(0, (spec_arr.shape[0] - 300) // 2)

        for region in range(4):
            spec_arr = self.specs[int(row.spectrogram_id)]
            img = spec_arr[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        X = torch.tensor(X)
        spectograms = [X[:, :, i : i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=0)

        eegs = [X[:, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=0)

        x = torch.cat([spectograms, eegs], dim=1)
        x = torch.cat([x, x, x], dim=2)
        x = x.permute(2, 0, 1)

        if self.mode != "test":
            y = row[targets].values.astype(np.float32)

        return {"data": x, "target": y}




## === cell 9
customdataset = CustomDataset(
    test_df, config, mode="test", specs=all_spectrograms, eegs=all_eegs
)
_ = customdataset[0]["data"].shape
print("One sample tensor shape:", _)



## === cell 10
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)




## === cell 11
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model1,
            pretrained=True,
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




## === cell 12
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    use_amp = device.type == "cuda"
    for batch in tqdm(test_loader, total=len(test_loader), desc="Infer", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ylogits = model(x)
            else:
                ylogits = model(x)
        preds.append(ylogits.detach().cpu().numpy())
    return {"logits": np.concatenate(preds, axis=0)}




## === cell 13
from torchvision.transforms import transforms

tras = transforms.Compose([transforms.Resize((224, 224))])


class Custommodel2(nn.Module):
    def __init__(self, config, transform, numclass: int = 6):
        super(Custommodel2, self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained=True,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x




## === cell 14
def try_predict_from_weight_dir(weight_dir: str, model_builder, loader):
    if not os.path.isdir(weight_dir):
        return None

    preds_ens = []
    for fn in sorted(os.listdir(weight_dir)):
        fp = os.path.join(weight_dir, fn)
        if not (
            os.path.isfile(fp)
            and (fn.endswith(".pt") or fn.endswith(".pth") or fn.endswith(".bin"))
        ):
            continue
        dd = torch.load(fp, map_location="cpu")
        model = model_builder()
        if isinstance(dd, dict) and "model" in dd:
            model.load_state_dict(dd["model"], strict=True)
        else:
            model.load_state_dict(dd, strict=True)
        model.to(device)
        pred = inference_function(loader, model, device)["logits"]
        preds_ens.append(pred)

    if len(preds_ens) == 0:
        return None
    return np.mean(np.stack(preds_ens, axis=0), axis=0)


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def smooth_probs(p: np.ndarray, power: float = 0.85, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    p = p**power
    p = p / p.sum(axis=1, keepdims=True)
    return p


def entropy01(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalized entropy in [0,1]. Lower => peakier => more overconfident.
    """
    p = np.clip(p, eps, 1.0)
    H = -(p * np.log(p)).sum(axis=1)
    Hmax = np.log(p.shape[1])
    return (H / Hmax).astype(np.float32)


def average_per_eeg_id(pred: np.ndarray, eeg_ids: np.ndarray) -> np.ndarray:
    dfp = pd.DataFrame(pred, columns=TARGETS)
    dfp["eeg_id"] = eeg_ids
    mean_by = dfp.groupby("eeg_id", as_index=False)[TARGETS].mean()
    m = mean_by.set_index("eeg_id")[TARGETS]
    out = m.loc[eeg_ids].to_numpy(np.float32)
    return out


logits1 = try_predict_from_weight_dir(
    "/kaggle/input/resnet34d2", lambda: Custommodel(config), test_loader
)
logits2 = try_predict_from_weight_dir(
    "/kaggle/input/visiontransformer", lambda: Custommodel2(config, tras), test_loader
)

if logits1 is None:
    m1 = Custommodel(config).to(device)
    logits1 = inference_function(test_loader, m1, device)["logits"]
    del m1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if logits2 is None:
    m2 = Custommodel2(config, tras).to(device)
    logits2 = inference_function(test_loader, m2, device)["logits"]
    del m2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

n_test = len(test_df)

train_df = pd.read_csv(paths.train_csv, usecols=["eeg_id"] + TARGETS)
train_grp = train_df.groupby("eeg_id", as_index=False)[TARGETS].sum()
vote_totals = train_grp[TARGETS].sum(axis=1).astype(np.float64).values  # weights
row_sum = train_grp[TARGETS].sum(axis=1).replace(0, np.nan)
probs = (train_grp[TARGETS].div(row_sum, axis=0)).fillna(0.0).to_numpy(np.float64)

w = np.where(vote_totals > 0, vote_totals, 0.0)
w_sum = w.sum()
if w_sum <= 0:
    prior = probs.mean(axis=0)
else:
    prior = (probs * w[:, None]).sum(axis=0) / w_sum

prior = prior.astype(np.float32)
prior = np.clip(prior, 1e-8, 1.0)
prior = prior / prior.sum()

T = 3.2
p1 = softmax_np(logits1 / T, axis=1).astype(np.float32)
p2 = softmax_np(logits2 / T, axis=1).astype(np.float32)

p1 = smooth_probs(p1, power=0.70)
p2 = smooth_probs(p2, power=0.70)

ens = 0.5 * (p1 + p2)

eeg_ids = test_df["eeg_id"].values
if pd.Series(eeg_ids).duplicated().any():
    ens = average_per_eeg_id(ens, eeg_ids)

ent = entropy01(ens)  # 0 (peaky) .. 1 (flat)
alpha_min, alpha_max = 0.28, 0.62
alpha_row = alpha_min + (1.0 - ent) * (alpha_max - alpha_min)  # peaky => higher alpha

alpha_floor = 0.04
alpha_eff = np.clip(alpha_floor + (1.0 - alpha_floor) * alpha_row, 0.0, 0.95)

finalpred = (1.0 - alpha_eff[:, None]) * ens + alpha_eff[:, None] * prior[None, :]

finalpred = np.asarray(finalpred, dtype=np.float32)
if finalpred.shape != (n_test, 6):
    raise ValueError(
        f"finalpred has wrong shape {finalpred.shape}, expected {(n_test, 6)}"
    )

row_sums = finalpred.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)
finalpred = finalpred / row_sums
finalpred = np.clip(finalpred, 1e-8, 1.0)
finalpred = finalpred / finalpred.sum(axis=1, keepdims=True)

print(
    "finalpred shape:",
    finalpred.shape,
    "row sum min/max:",
    float(finalpred.sum(1).min()),
    float(finalpred.sum(1).max()),
    "alpha_eff min/max:",
    float(alpha_eff.min()),
    float(alpha_eff.max()),
)



## === cell 15
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

if len(sample_sub) != len(test_df):
    test_map = pd.DataFrame(
        {"eeg_id": test_df["eeg_id"].values, "_row": np.arange(len(test_df))}
    )
    ss = sample_sub[["eeg_id"]].merge(
        test_map, on="eeg_id", how="left", validate="one_to_one"
    )
    if ss["_row"].isna().any():
        raise ValueError("Could not align test_df to sample_submission by eeg_id.")
    finalpred_aligned = finalpred[ss["_row"].astype(int).values]
    sub = sample_sub.copy()
    sub[TARGETS] = finalpred_aligned
else:
    sub = sample_sub.copy()
    sub[TARGETS] = finalpred

p = sub[TARGETS].to_numpy(np.float64)
p = np.clip(p, 1e-15, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sub[TARGETS] = p

out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print(
    "Row-sum min/max:",
    float(sub[TARGETS].sum(axis=1).min()),
    float(sub[TARGETS].sum(axis=1).max()),
)
sub.head()
