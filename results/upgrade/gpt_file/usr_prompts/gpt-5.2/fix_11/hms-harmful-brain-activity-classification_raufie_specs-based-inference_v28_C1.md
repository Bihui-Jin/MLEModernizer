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

0.5678983843462729

# 6. Current score

1.4076

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weights path by making the script robust: it try to load the provided checkpoint if present, but if it’s not available in this environment it fall back to a deterministic, valid probability baseline so a submission is always produced. I also fix the dataset bug where spectrogram and EEG parts are written inside the wrong loop (overwriting and mis-shaping `X`), and define the missing `label_cols`/`min`/`max` dependencies for test mode so the test pipeline runs. Finally, I ensure predictions have shape `(len(test_df), 6)` and are properly normalized to sum to 1 per row to satisfy the submission constraints and avoid the pandas assignment error.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is much worse than the target (0.5679), so we should improve while keeping the exact model and inference logic intact. The biggest likely issue is that the test spectrogram slicing uses a hardcoded `r=0`, which can mismatch how the model was trained (typically centered window); we switch test-mode `r` to a centered crop derived from each spectrogram’s time length to better align with training semantics. We also ensure deterministic behavior and fix a subtle boolean bug (`&` vs `and`) in the model’s input routing that can change behavior unexpectedly. These are minimal, low-risk changes that should move the KL score down toward the target without changing architecture, loss, or the overall pipeline.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far above the target (0.5679), so we should make a small, metric-aligned improvement without changing the model or training logic. The biggest fix is to make inference match how these EfficientNet checkpoints are typically trained: add the exact same input normalization (ImageNet mean/std) right before the backbone, which materially affects logits and KL. I also switch to GPU-friendly inference (AMP autocast) and compile-free deterministic settings while keeping outputs equivalent (no early stopping/sampling), plus a safer spectrogram crop bound to avoid edge/corner mismatches. These changes are minimal and focused on reducing KL by improving calibration and input consistency.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (lower-is-better), so the smallest safe improvements are to (1) make test-time spectrogram cropping align more closely with train-time by using the fixed center 10s window (rows 100:200 out of 300) instead of a full 300-row crop, and (2) ensure the EfficientNet input normalization matches the actual number of channels by applying mean/std per-channel (your current 3-channel ImageNet stats are being broadcast across 12 channels, which distorts inputs). These changes keep the same model architecture and inference flow, but correct two inference-time mismatches that commonly inflate KL. I also keep the existing probability normalization/clipping to guarantee a valid submission. The rest of the pipeline (feature extraction, model, weights loading, ensembling) stays intact.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.5679), so we make two minimal inference-time fixes that commonly reduce KL without changing the model architecture or training logic. First, we correct a key shape bug in `CustomModel.__reshape_input`: it currently concatenates along width and then repeats width, which scrambles channel semantics; we instead concatenate spectrogram+EEG along the channel axis and only replicate channels to 3 for EfficientNet compatibility. Second, we make test-time spectrogram crop selection deterministic and consistent with the already-intended “center window” logic by using a fixed centered 300-row crop (then the existing 100:200 slice) rather than sometimes anchoring at `r=0`. These changes keep the same feature extraction, same model, same weights, same softmax, and still guarantee valid probability rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'We keep your model and inference core unchanged, but make two small, metric-aligned fixes that typically reduce KL for this competition. First, we make test-time spectrogram cropping match the intended “centered 10s within a centered 300-row window” consistently by always taking a centered 300-row crop (when available) instead of sometimes anchoring at `r=0` for shorter spectrograms. Second, we change the final probability clipping from `1e-8` to `1e-4` (then renormalize) to reduce extreme probabilities that are heavily penalized by KL when the model is overconfident/miscalibrated. These are minimal changes, preserve architecture/loss/feature extraction, and still guarantee a valid submission with row sums = 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.5679), so we should make the smallest inference-only corrections likely to reduce KL without changing the model/training design. The biggest remaining mismatch is that your spectrogram pipeline extracts a centered 10s window (columns 100:200) but you currently place it into a 256-wide canvas using hardcoded `78:178`, which is shifted and can materially hurt predictions; we instead center it exactly (`78:178` computed from width) to preserve the intended “center window” semantics robustly. We also ensure deterministic, stable inference by using `inference_mode()` (no semantic change vs `no_grad()`) and by clipping/renormalizing as before. Everything else (model, weights loading, feature extraction, softmax, ensembling, submission format) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.5679), so we should make a small inference-time fix that improves calibration without changing the model, training loop, features, or loss. The biggest low-risk gain here is correcting spectrogram normalization: you currently standardize each region using the full 100×100 patch (including large zero-padded borders), which distorts mean/std; we instead compute mean/std only on the actually-filled area before writing into the canvas. This preserves the same spectrogram extraction, same crop, same model, and same softmax, but makes inputs closer to the intended distribution, which typically reduces KL. Everything else (weights loading, ensembling, probability clipping/renorm, and submission format) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.5679), so we make two very small inference-time fixes that typically reduce KL without changing the model, training, loss, or feature extraction intent. First, we normalize the EEG-derived spectrogram channels to the same rough distribution as the Kaggle spectrogram channels (log/standardize), because currently the model sees mismatched scales across the 8 channels which often breaks calibrated probabilities. Second, we apply a tiny temperature scaling during softmax (T>1) to slightly reduce overconfidence, which KL heavily penalizes; this preserves the same logits/model and only changes the probability calibration. Everything else (cropping logic, EfficientNet, weights loading, submission formatting, and row-sum normalization) stays the same.'
- What this solution (achieved 1.4076) has done: 'Your current KL (1.40995, lower-is-better) is much worse than the target (0.5679), so we make the smallest inference-only calibration changes that usually reduce KL without changing the model, features, or training semantics. First, we add a very light “prior-mix” (shrinkage) toward the empirical class prior from `train.csv` to reduce overconfident miscalibration that KL heavily penalizes, while keeping the model’s relative preferences intact. Second, we keep your existing temperature scaling but tune it slightly downward (less smoothing than 1.15) because over-smoothing can also hurt KL if the model is underconfident; this is a minimal single-parameter change. Everything else (feature extraction, cropping, EfficientNet, checkpoint loading, and submission formatting/normalization) stays the same and still produces a valid `submission.csv`.'

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

from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b1"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False

    TTA_TEMPERATURE = 1.07

    PRIOR_MIX_ALPHA = 0.08


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b1_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


discovered_weights = sorted(glob("/kaggle/input/hba-efficientnet-weights/*.pth"))
model_weights = (
    discovered_weights if len(discovered_weights) > 0 else [paths.MODEL_WEIGHTS]
)
print("Discovered weight files:", len(discovered_weights))



## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
label_cols = TARGETS


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
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquets")
all_spectrograms = {}

for file_path in tqdm(paths_spectrograms):
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split(".")[0])
    all_spectrograms[name] = aux.iloc[:, 1:].values
    del aux

if config.VISUALIZE and len(paths_spectrograms) > 0:
    idx = np.random.randint(0, len(paths_spectrograms))
    plot_spectrogram(paths_spectrograms[idx])



## === cell 5
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets")
all_eegs = {}
counter = 0

for file_path in tqdm(paths_eegs):
    eeg_id = file_path.split("/")[-1].split(".")[0]
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs[int(eeg_id)] = eeg_spectrogram
    counter += 1




## === cell 6
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(config.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

        self.register_buffer(
            "_imgnet_mean_3",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imgnet_std_3",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def __reshape_input(self, x):
        """
        Keep: same inputs and same EfficientNet expectation (3-channel NCHW).
        """
        spectrograms = x[:, :, :, 0:4]  # (B, H, W, 4)
        eegs = x[:, :, :, 4:8]  # (B, H, W, 4)

        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=3)  # (B, H, W, 8)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs  # (B, H, W, 4)
        else:
            x = spectrograms  # (B, H, W, 4)

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        c = x.shape[1]
        rep = int(math.ceil(3 / c))
        x = x.repeat(1, rep, 1, 1)[:, :3, :, :]  # (B, 3, H, W)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = (x - self._imgnet_mean_3) / self._imgnet_std_3
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 7
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = all_spectrograms if specs is None else specs
        self.eeg_spectrograms = all_eegs if eeg_specs is None else eeg_specs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        spec = self.spectrograms[int(row.spectrogram_id)]

        if self.mode == "test":
            t = int(spec.shape[0])
            if t >= 300:
                r = (t - 300) // 2
            else:
                r = 0
            r = int(np.clip(r, 0, max(0, t - 300)))
        else:
            r = int((row["min"] + row["max"]) // 4)

        t0, t1 = 100, 200  # 100 rows ~= 10 seconds

        patch_w = t1 - t0
        x0 = (X.shape[1] - patch_w) // 2
        x1 = x0 + patch_w

        for region in range(4):
            img_src = spec[:, region * 100 : (region + 1) * 100]
            if img_src.shape[0] < 300:
                pad = 300 - img_src.shape[0]
                top = pad // 2
                bot = pad - top
                img_src = np.pad(img_src, ((top, bot), (0, 0)), mode="edge")
                rr = 0
            else:
                rr = r

            img_full = img_src[rr : rr + 300].T  # (100, 300)
            img = img_full[:, t0:t1]  # (100, 100)

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, x0:x1, region] = img / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)].astype(
            np.float32
        )  # (128,256,4)

        ep = 1e-6
        eeg_img = np.clip(eeg_img, ep, None)
        eeg_img = np.log(eeg_img)

        mu = np.nanmean(eeg_img, axis=(0, 1), keepdims=True)
        std = np.nanstd(eeg_img, axis=(0, 1), keepdims=True)
        eeg_img = (eeg_img - mu) / (std + ep)
        eeg_img = np.nan_to_num(eeg_img, nan=0.0, posinf=0.0, neginf=0.0)

        X[:, :, 4:] = eeg_img

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )
        return transforms(image=img)["image"]




## === cell 8
test_dataset = CustomDataset(test_df, config, mode="test")
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




## === cell 9
def inference_function(test_loader, model, device, temperature: float = 1.0):
    model.eval()
    preds = []
    use_amp = device.type == "cuda"
    temperature = float(max(1e-6, temperature))
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    logits = model(X)
            probs = torch.softmax((logits.float() / temperature), dim=1)
            preds.append(probs.detach().cpu().numpy())

    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
predictions_list = []

existing_weights = [w for w in model_weights if os.path.exists(w)]
if len(existing_weights) == 0:
    print(
        "WARNING: No model weights found. Falling back to uniform probabilities baseline."
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    for model_weight in existing_weights:
        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        model.to(device)

        prediction_dict = inference_function(
            test_loader, model, device, temperature=config.TTA_TEMPERATURE
        )
        predictions_list.append(prediction_dict["predictions"])

        del model, checkpoint, state
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0).astype(np.float32)

print("Predictions shape:", predictions.shape)



## === cell 11
try:
    train_df = pd.read_csv(paths.TRAIN_CSV, usecols=TARGETS)
    prior = train_df[TARGETS].sum(axis=0).values.astype(np.float64)
    prior = prior / np.maximum(prior.sum(), 1e-12)
    prior = prior.astype(np.float32)
    print("Computed train prior:", dict(zip(TARGETS, prior.round(6))))
except Exception as e:
    print("WARNING: failed to compute train prior, using uniform. Error:", repr(e))
    prior = np.full((6,), 1.0 / 6.0, dtype=np.float32)

alpha = float(np.clip(getattr(config, "PRIOR_MIX_ALPHA", 0.0), 0.0, 0.3))
if alpha > 0:
    predictions = (1.0 - alpha) * predictions + alpha * prior[None, :]

predictions = np.clip(predictions, 1e-4, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission saved to: {out_path}")
print(f"Submission shape: {sub.shape}")
print(
    "Row-sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
