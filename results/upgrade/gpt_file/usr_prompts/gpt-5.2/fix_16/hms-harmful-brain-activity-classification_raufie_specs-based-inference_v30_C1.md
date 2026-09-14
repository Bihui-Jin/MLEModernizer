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

0.5201514181786011

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the failure that causes `predictions` to be empty (and thus 0‑D), by adding a safe fallback when the external weight directory is missing/empty and by guarding the normalization step. I also make the checkpoint loading more robust to different saved formats (common in Kaggle) while keeping the same model and inference logic. Finally, I ensure we always write a valid `submission.csv` with the required columns and probabilities that sum to 1, even if no weights are found (uniform probabilities fallback, score-worse but valid). These changes are minimal and directly address the runtime error and submission validity.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.52015), so we should make a small, low-risk improvement that preserves your exact model/inference pipeline. The biggest issue is that you are computing the Kaggle spectrogram “r” offset from `row["min"]`/`row["max"]`, which don’t exist in `test.csv` and (as written) don’t exist in your loaded train metadata either—this typically leads to either a crash or inconsistent behavior if you ever run on train/valid. I fix this by using the correct spectrogram time-offset column (`spectrogram_label_offset_seconds`) when available and a safe centered fallback otherwise, keeping the same slicing logic. I also add a tiny numerical guard for empty/short spectrogram arrays so every test sample gets a well-formed input, which should reduce pathological predictions and move KL down toward your target without changing the model.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.52015), so we need a small but meaningful correctness fix that improves KL without changing the model/training logic. The biggest issue is that your `CustomModel.__reshape_input` concatenates along the wrong dimension (it stacks channels into the height axis), producing a malformed input distribution for EfficientNet; fixing this to properly form an 8-channel image and then adapt it to 3 channels via a minimal, deterministic 1x1 conv should materially improve predictions. I also make test-time spectrogram offset handling closer to the competition’s center-10s intent by using the (existing) centered crop consistently and add a safe fallback when an EEG spectrogram is missing to avoid occasional crashes/bad rows. These changes preserve your overall pipeline (same feature sources, same backbone, same softmax inference, same submission semantics) but should move KL down toward the target.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (lower is better), so the smallest meaningful improvement is to fix a likely major input mismatch at inference time: you currently create a timm backbone without setting `num_classes`, then manually slice children and add your own head, which can easily diverge from the architecture used when the provided weights were trained. I keep your feature extraction, dataset construction, and inference logic the same, but change the backbone creation to `num_classes=6` and use the backbone’s own classifier head (reset to Identity) so the loaded checkpoint matches and predictions become non-degenerate. I also load checkpoints on CPU first (safer across formats) and keep strict loading when possible, falling back to strict=False only when needed (same as you already do). This should materially reduce KL toward the target without changing the overall approach or adding any new training.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.52015), so we should make a small but high-impact correctness fix without changing your model or inference loop. The most likely issue is that the Kaggle spectrogram crop is currently taken from the *middle of the full spectrogram* (300 rows), but these spectrograms are 10-minute windows where the labeled event is centered; using the centered 10-second window (as intended by the dataset design) is critical and typically yields a large KL improvement. I adjust the crop start `r` to center on 300 seconds (i.e., row ~300) when possible, with safe clipping for short arrays, while keeping the exact same normalization and model forward path. I also add a tiny guard to ensure the crop always has exactly 300 rows (pad if needed), preventing rare shape/pathology issues that can spike KL.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.52015), so we need a correctness fix that improves predictions without changing the overall approach. The biggest low-risk issue is that the model is almost certainly being fed spectrogram crops from the wrong time window: test spectrograms are already centered 10-minute windows, and you should take the centered 10-second crop (middle of the time axis), not a hard-coded “seconds index=300” that may not match the actual row scale. I change the crop start `r` to be computed from `spec_arr.shape[0]//2` (center crop), keeping the exact same crop size, normalization, and model/inference loop. I also add a tiny guard to ensure we always slice exactly 300 rows (pad if needed) and keep submission probabilities properly normalized.'
- What this solution (achieved 1.40995) has done: 'I fix the root-cause mismatch between `test_df` rows and the number of predictions by ensuring the test metadata is deduplicated to one row per `eeg_id` (as required by the submission), while preserving the same alignment to `sample_submission.csv`. I also change the expensive “preload all spectrogram/eeg parquets into dicts” logic to lazy-on-demand loading inside the Dataset, which prevents accidental key/shape issues and avoids memory blowups that can lead to unstable behavior. Finally, I keep the model/inference logic intact, but make the dataloader output count deterministically match the submission length so `submission.csv` is always produced validly.'

# 9. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
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
from typing import Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b3"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b3_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


model_weights = sorted(glob("/kaggle/input/hba-efficientnet-weights/*.pth"))
if len(model_weights) == 0 and os.path.exists(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]

if len(model_weights) == 0:
    discovered = sorted(
        glob("/kaggle/input/**/hba*efficientnet*/*.pth", recursive=True)
        + glob("/kaggle/input/**/*efficientnet*b3*/*.pth", recursive=True)
    )
    discovered = [p for p in discovered if os.path.getsize(p) > 100_000]
    model_weights = discovered

print(f"Found {len(model_weights)} model weights")
if len(model_weights) > 0:
    print("Example weight path:", model_weights[0])

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



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
        plt.title(f"EEG Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img


def plot_spectrogram(spectrogram_path: str):
    sample_spect = pd.read_parquet(spectrogram_path)

    split_spect = {
        "LL": sample_spect.filter(regex="^LL", axis=1),
        "RL": sample_spect.filter(regex="^RL", axis=1),
        "RP": sample_spect.filter(regex="^RP", axis=1),
        "LP": sample_spect.filter(regex="^LP", axis=1),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(
            np.log(split_spect[split_name]).T,
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Log(Value)")
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels(
            [column_name[3:] for column_name in split_spect[split_name].columns]
        )
        frequencies = [
            column_name[3:] for column_name in split_spect[split_name].columns
        ]
        ax.set_yticks(
            np.arange(0, len(split_spect[split_name].columns), label_interval)
        )
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)



## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Raw test dataframe shape is: {test_df.shape}")

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
required_eeg_order = sample_sub["eeg_id"].values

test_df = test_df.drop_duplicates(subset=["eeg_id"], keep="first")

test_df = test_df.set_index("eeg_id").reindex(required_eeg_order).reset_index()

if test_df["spectrogram_id"].isna().any():
    test_df["spectrogram_id"] = test_df["spectrogram_id"].ffill().bfill()
if test_df["patient_id"].isna().any():
    test_df["patient_id"] = test_df["patient_id"].ffill().bfill()

print("Aligned & deduped test_df shape:", test_df.shape)
print("Aligned test_df head eeg_id:", test_df["eeg_id"].head().tolist())
test_df.head()



## === cell 4
print("Using lazy loading for spectrograms/EEGs (no full preload).")




## === cell 5
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True

        self.model = timm.create_model(
            config.MODEL, pretrained=False, num_classes=num_classes
        )

        self.input_conv = nn.Conv2d(8, 3, kernel_size=1, bias=False)

        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(
                0, global_pool=""
            )  # remove classifier & pooling safely

        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 8, 128, 256)

        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = x  # keep all 8 channels
        elif self.USE_EEG_SPECTROGRAMS:
            x = x[:, 4:8, :, :]  # EEG-only (4ch)
            z = torch.zeros(
                (x.size(0), 4, x.size(2), x.size(3)), device=x.device, dtype=x.dtype
            )
            x = torch.cat([z, x], dim=1)
        else:
            x = x[:, 0:4, :, :]  # spectrogram-only (4ch)
            z = torch.zeros(
                (x.size(0), 4, x.size(2), x.size(3)), device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, z], dim=1)

        x = self.input_conv(x)  # (B, 3, 128, 256)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        if hasattr(self.model, "forward_features"):
            x = self.model.forward_features(x)
        else:
            x = nn.Sequential(*list(self.model.children())[:-2])(x)
        x = self.custom_layers(x)
        return x




## === cell 6
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        spectrogram_dir: str = None,
        eeg_dir: str = None,
        cache_size: int = 256,
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode

        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.spectrogram_dir = (
            spectrogram_dir if spectrogram_dir is not None else paths.TEST_SPECTROGRAMS
        )
        self.eeg_dir = eeg_dir if eeg_dir is not None else paths.TEST_EEGS

        self.crop_h = 300
        self.dst_w = 256

        self._spec_cache: Dict[int, np.ndarray] = {}
        self._eeg_cache: Dict[int, np.ndarray] = {}
        self._cache_size = int(cache_size)

    def __len__(self):
        return len(self.df)

    def _cache_put(self, cache: dict, key: int, value: np.ndarray):
        if key in cache:
            return
        if len(cache) >= self._cache_size:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def _load_spec_arr(self, spectrogram_id: int) -> Optional[np.ndarray]:
        if self.spectrograms is not None:
            return self.spectrograms.get(spectrogram_id, None)

        if spectrogram_id in self._spec_cache:
            return self._spec_cache[spectrogram_id]

        fpath = os.path.join(self.spectrogram_dir, f"{spectrogram_id}.parquet")
        if not os.path.exists(fpath):
            return None

        aux = pd.read_parquet(fpath)
        arr = aux.iloc[:, 1:].values
        del aux
        self._cache_put(self._spec_cache, spectrogram_id, arr)
        return arr

    def _load_eeg_img(self, eeg_id: int) -> Optional[np.ndarray]:
        if self.eeg_spectrograms is not None:
            return self.eeg_spectrograms.get(eeg_id, None)

        if eeg_id in self._eeg_cache:
            return self._eeg_cache[eeg_id]

        fpath = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
        if not os.path.exists(fpath):
            return None

        eeg_img = spectrogram_from_eeg(fpath, display=False)
        self._cache_put(self._eeg_cache, eeg_id, eeg_img)
        return eeg_img

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        if self.mode == "test":
            y = np.float32(0.0)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        spec_arr = self._load_spec_arr(int(row.spectrogram_id))
        if (
            spec_arr is None
            or not hasattr(spec_arr, "shape")
            or spec_arr.ndim != 2
            or spec_arr.shape[0] < 1
        ):
            spec_arr = np.zeros((self.crop_h, 400), dtype=np.float32)

        crop_h = self.crop_h

        r = int(spec_arr.shape[0] // 2 - crop_h // 2)
        r = int(np.clip(r, 0, max(0, spec_arr.shape[0] - crop_h)))

        if spec_arr.shape[0] < crop_h:
            pad = crop_h - spec_arr.shape[0]
            spec_arr = np.pad(
                spec_arr, ((0, pad), (0, 0)), mode="constant", constant_values=0.0
            )
            r = 0

        eeg_img = self._load_eeg_img(int(row.eeg_id))
        if (
            eeg_img is None
            or not hasattr(eeg_img, "shape")
            or eeg_img.shape != (128, 256, 4)
        ):
            eeg_img = np.zeros((128, 256, 4), dtype=np.float32)

        X[:, :, 4:] = eeg_img

        for region in range(4):
            c0 = region * 100
            c1 = min((region + 1) * 100, spec_arr.shape[1])
            region_block = spec_arr[r : r + crop_h, c0:c1]

            if region_block.shape[1] < 100:
                region_block = np.pad(
                    region_block,
                    ((0, 0), (0, 100 - region_block.shape[1])),
                    mode="constant",
                    constant_values=0.0,
                )

            img = region_block.T  # (100, crop_h)

            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            flat = img.flatten()
            mu = np.nanmean(flat)
            std = np.nanstd(flat)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            if img.shape[1] != self.dst_w:
                img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                img_t = torch.nn.functional.interpolate(
                    img_t,
                    size=(img_t.shape[-2], self.dst_w),
                    mode="bilinear",
                    align_corners=False,
                )
                img = img_t.squeeze(0).squeeze(0).numpy()

            X[14:-14, :, region] = img / 2.0

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 7
test_dataset = CustomDataset(
    test_df,
    config,
    mode="test",
    augment=False,
    specs=None,
    eeg_specs=None,
    spectrogram_dir=paths.TEST_SPECTROGRAMS,
    eeg_dir=paths.TEST_EEGS,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X, y = test_dataset[0]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Test dataset length:", len(test_dataset))




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            if X.ndim != 4 or X.shape[-1] != 8:
                raise ValueError(f"Bad batch X shape {tuple(X.shape)} at step={step}")
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)

                if isinstance(y_preds, (tuple, list)):
                    y_preds = y_preds[0]
                if y_preds.ndim > 2:
                    y_preds = y_preds.reshape(y_preds.shape[0], -1)

                if y_preds.ndim != 2:
                    raise ValueError(
                        f"Bad model output shape {tuple(y_preds.shape)} at step={step}"
                    )
                if y_preds.shape[1] != 6:
                    raise ValueError(
                        f"Expected 6 logits, got {y_preds.shape[1]} at step={step}"
                    )

                y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())
    if len(preds) == 0:
        return {"predictions": np.zeros((0, 6), dtype=np.float32)}
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 9
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                if any(key.startswith("model.") for key in sd.keys()):
                    sd = {key.replace("model.", "", 1): val for key, val in sd.items()}
                if any(key.startswith("module.") for key in sd.keys()):
                    sd = {key.replace("module.", "", 1): val for key, val in sd.items()}
                return sd
        if all(isinstance(k, str) for k in ckpt.keys()):
            sd = ckpt
            if any(key.startswith("module.") for key in sd.keys()):
                sd = {key.replace("module.", "", 1): val for key, val in sd.items()}
            return sd
    return ckpt


def _remap_state_dict_for_this_model(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd

    out = {}
    for k, v in sd.items():
        nk = k

        for pref in ("backbone.", "encoder.", "net.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
                break

        if nk.startswith("classifier."):
            nk = "custom_layers.2." + nk[len("classifier.") :]
        if nk.startswith("head.fc."):
            nk = "custom_layers.2." + nk[len("head.fc.") :]
        if nk.startswith("fc."):
            nk = "custom_layers.2." + nk[len("fc.") :]

        out[nk] = v
    return out


per_model_predictions = []

if len(model_weights) == 0:
    print(
        "WARNING: No model weights found. Will create uniform predictions as fallback."
    )
else:
    for model_weight in model_weights:
        model = CustomModel(config)

        checkpoint = torch.load(model_weight, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        state_dict = _remap_state_dict_for_this_model(state_dict)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            with torch.no_grad():
                model.input_conv.weight.zero_()
                for out_c in range(3):
                    model.input_conv.weight[out_c, :, 0, 0] = 1.0 / 8.0
            res = model.load_state_dict(state_dict, strict=False)
            missing = list(res.missing_keys) if hasattr(res, "missing_keys") else []
            unexpected = (
                list(res.unexpected_keys) if hasattr(res, "unexpected_keys") else []
            )
            print("Loaded with strict=False due to:", str(e).split("\n")[0])
            if len(missing) > 0:
                print("Missing keys:", missing[:10])
            if len(unexpected) > 0:
                print("Unexpected keys:", unexpected[:10])

        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        pm = prediction_dict["predictions"]

        if not (isinstance(pm, np.ndarray) and pm.ndim == 2 and pm.shape[1] == 6):
            raise ValueError(
                f"Per-model predictions bad shape: {getattr(pm, 'shape', None)} for weight={model_weight}"
            )
        if pm.shape[0] != len(test_df):
            raise ValueError(
                f"Per-model predictions rows {pm.shape[0]} != test_df rows {len(test_df)} for weight={model_weight}"
            )

        per_model_predictions.append(pm.astype(np.float32, copy=False))

        del model
        torch.cuda.empty_cache()
        gc.collect()

if len(per_model_predictions) > 0:
    stacked = np.stack(per_model_predictions, axis=0)  # (n_models, n_samples, 6)
    predictions = stacked.mean(axis=0)  # (n_samples, 6)
else:
    n = len(test_df)
    predictions = np.full((n, 6), 1.0 / 6.0, dtype=np.float32)

if not (
    isinstance(predictions, np.ndarray)
    and predictions.ndim == 2
    and predictions.shape == (len(test_df), 6)
):
    print(
        "WARNING: predictions has unexpected shape, fallback to uniform. Shape:",
        getattr(predictions, "shape", None),
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)

print("Predictions shape:", predictions.shape)



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

pred = np.asarray(predictions, dtype=np.float64)
if pred.ndim != 2 or pred.shape[1] != 6:
    raise ValueError(f"Unexpected predictions shape {pred.shape}, expected (n, 6)")

pred = np.clip(pred, 1e-12, 1.0)
row_sums = pred.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)
pred = pred / row_sums

sub = sample_sub.copy()
if len(sub) != pred.shape[0]:
    raise ValueError(
        f"Submission length {len(sub)} does not match predictions {pred.shape[0]}"
    )
sub[TARGETS] = pred.astype(np.float32)

vals = sub[TARGETS].values.astype(np.float64)
vals = np.clip(vals, 1e-12, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print(f"Saved submission to: {out_path}")
print(f"Submission shape: {sub.shape}")
print(
    "Row-sum min/max:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
print("Unique eeg_id count:", sub["eeg_id"].nunique())
sub.head()
