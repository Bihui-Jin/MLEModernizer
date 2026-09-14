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

0.7486796901779933

# 6. Current score

1.43616

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.44269) has done: 'I fix the missing-weight crash by automatically falling back to a local timm pretrained EfficientNet when the Kaggle dataset `/kaggle/input/hba-efficientnet-weights/` is not available, keeping the same model head and inference flow so it runs end-to-end. I also fix the dataset bug where `label_cols`/`min`/`max` are referenced but never defined for test mode, and correct an indentation/loop logic issue that repeatedly overwrote the EEG channels. Finally, I ensure predictions always have shape `(len(test_df), 6)`, are proper probabilities (sum to 1), and the submission CSV matches the required columns exactly.'
- What this solution (achieved 1.44269) has done: 'Your current score is much worse than the target (lower-is-better), so we should improve predictive calibration/quality without changing the model architecture or training loop (you do pure inference). The biggest score-killer here is a silent bug: when slicing Kaggle spectrograms you always start at `r=0`, ignoring the row’s `spectrogram_label_offset_seconds`/window alignment; that makes the input mismatched and harms KL a lot. I minimally fix this by computing the correct start row `r` from the offset (0.5 sec per row in the parquet) with safe clipping, while keeping all feature engineering and the model exactly the same. I also add a tiny epsilon floor before log to avoid `-inf` in rare cases (stability, not a logic change), and keep the submission formatting/probability normalization identical.'
- What this solution (achieved 1.44168) has done: 'Your current score (1.44269, lower-is-better) is far from the target (0.74868), so we should improve input/label alignment and inference correctness while keeping the same model and inference-only approach. The biggest remaining issue is that `test.csv` does not have `spectrogram_label_offset_seconds`, but your dataset code still tries to use it; this silently forces `r=0` for all test samples and likely mis-centers the 10s window, hurting KL. I change the test-time spectrogram crop to use the *center* of the available test spectrogram (since test is exactly 10 minutes and the labeled 10s is centered), while keeping the existing offset-based behavior for train/if-present. I also add a tiny guard so the computed `r` is always valid even if the spectrogram length is slightly different, without changing normalization, model, or output semantics.'
- What this solution (achieved 1.44189) has done: 'Your current score is much worse than the target (lower-is-better), so we should make the smallest fixes that improve input correctness and probability calibration without changing the model/inference approach. The biggest quality issue left is that test-time spectrogram cropping is only “centered” in an arbitrary way, while train-time labels are for the central 10s of the 10-minute spectrogram; we should explicitly crop the *true center 10s window* (300 frames) for test, and keep the offset-based behavior when an offset exists. I also fix a region-order mismatch (you defined `NAMES` as `LL, LP, RP, RR` but the Kaggle spectrogram regions are `LL, RL, RP, LP`), which silently feeds the model swapped channels and hurts KL. Finally, I make the softmax output slightly safer for KL by enforcing a tiny floor and renormalizing (no change to semantics, just prevents extreme probabilities).'
- What this solution (achieved 1.44191) has done: 'Your current score (1.44189, lower-is-better) is far from the target (0.74868), so we should fix the most likely remaining *data alignment* issues while keeping the same model and inference-only flow. The biggest remaining bug is that the Kaggle spectrogram crop uses `spec[r:r+300, region*100:(region+1)*100]` assuming regions are stored in contiguous 100-column blocks, but in this competition the parquet columns are frequency bins prefixed by region (LL/RL/RP/LP), so slicing by raw column indices can mix regions/frequencies and badly degrade KL. I minimally change spectrogram loading to build a correct `(time, 400)` matrix in the exact `NAMES = ["LL","RL","RP","LP"]` order by selecting columns by prefix, then keep the existing crop/normalization and model unchanged. This should materially improve signal correctness without changing architecture, loops, or loss/metric semantics.'
- What this solution (achieved 1.44154) has done: 'Your current score (1.44191, lower-is-better) is far from the target (0.74868), so we should make the smallest fixes that improve correctness/calibration without changing the model or inference loop. The biggest remaining issue is that the test spectrogram parquet’s first column is typically the time index/`time` and should be used to center-crop the *true middle 10s window*; right now we drop it and then center by raw row count, which can be slightly misaligned. I (1) keep your exact feature construction, but compute the crop start `r` using the time column when available (fall back to your existing center logic otherwise), and (2) apply the same safe log/standardization but also clamp extreme standardized values to reduce KL blow-ups from rare artifacts (still produces valid probabilities and doesn’t alter the model). These are minimal, inference-only changes that should move KL down toward the target while preserving your core approach.'
- What this solution (achieved 1.43526) has done: 'Your current KL (1.44154, lower-is-better) is far from the target (0.74868), so we should make the smallest inference-only fixes that improve signal correctness without changing the model or inference flow. The biggest remaining quality issue is that you normalize each spectrogram crop using its own mean/std; that causes a train/test feature mismatch because the EfficientNet weights were trained with a fixed, global normalization. I keep your exact feature construction and cropping, but change to the same fixed normalization constants used in common HMS baselines (per-region global mean/std) and remove the per-sample standardization (still standardization, just consistent). I also ensure the “time” column is only treated as time if it’s actually a time-like column (otherwise it was accidentally dropping a real feature column for some files), which can materially hurt inputs.'
- What this solution (achieved 1.43616) has done: 'Your current KL (1.43526, lower-is-better) is still far above the target (0.74868), so we should make the smallest inference-only fixes that improve input correctness without changing the model or the overall inference pipeline. The biggest remaining likely issue is that the Kaggle spectrogram crop is always taken around the middle of the *available rows*, but the parquet often contains a `time`/seconds column and may not be uniformly sampled; using the true time-centered 10s window (±5s) is more faithful to how labels were defined. I adjust `_compute_r` to prefer a time-based center crop when a time column exists (using a robust detection you already built), and fall back to the existing behavior otherwise. I also ensure the selected crop always yields exactly 300 frames by padding when a spectrogram is shorter than expected (rare but catastrophic for consistency), without changing the model, features, or softmax/probability normalization.'

# 9. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pywt
import random
import timm
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b3"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b3_epoch_20.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


model_weights = [paths.MODEL_WEIGHTS]



## === cell 2
USE_WAVELET = None

NAMES = ["LL", "RL", "RP", "LP"]

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

SPEC_MEAN = np.array([-0.525, -0.525, -0.525, -0.525], dtype=np.float32)
SPEC_STD = np.array([0.29, 0.29, 0.29, 0.29], dtype=np.float32)


def maddest(d, axis: int = None):
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
    os.environ["PYTHONHASHSEED"] = str(seed)


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
all_spectrograms: Dict[int, np.ndarray] = {}
all_spectrogram_times: Dict[int, np.ndarray] = {}

for file_path in tqdm(paths_spectrograms):
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split(".")[0])

    time_col = None
    aux_feat = aux
    if aux.shape[1] > 1:
        c0 = aux.columns[0]
        if (isinstance(c0, str) and (c0.lower() in ["time", "seconds", "t"])) or (
            not isinstance(c0, str) and str(c0).lower() in ["time", "seconds", "t"]
        ):
            time_col = aux.iloc[:, 0].to_numpy()
            aux_feat = aux.iloc[:, 1:]
        else:
            v0 = aux.iloc[:, 0].to_numpy()
            if np.issubdtype(v0.dtype, np.number):
                prefixed = sum(
                    (
                        isinstance(c, str)
                        and (
                            c.startswith("LL")
                            or c.startswith("RL")
                            or c.startswith("RP")
                            or c.startswith("LP")
                        )
                    )
                    for c in aux.columns[1:]
                )
                if prefixed >= 350:
                    time_col = v0
                    aux_feat = aux.iloc[:, 1:]

    region_blocks = []
    for reg in NAMES:
        cols = [c for c in aux_feat.columns if isinstance(c, str) and c.startswith(reg)]
        cols_sorted = sorted(
            cols, key=lambda x: float(x.split("_")[1]) if "_" in x else x
        )
        block = aux_feat[cols_sorted].to_numpy(dtype=np.float32)

        if block.shape[1] != 100:
            if block.shape[1] > 100:
                block = block[:, :100]
            else:
                pad = np.zeros((block.shape[0], 100 - block.shape[1]), dtype=np.float32)
                block = np.concatenate([block, pad], axis=1)

        region_blocks.append(block)

    spec_400 = np.concatenate(region_blocks, axis=1)  # (time, 400)
    all_spectrograms[name] = spec_400
    if time_col is not None:
        all_spectrogram_times[name] = time_col

    del aux, aux_feat, region_blocks, spec_400, time_col

if config.VISUALIZE and len(paths_spectrograms) > 0:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)



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
    def __init__(self, config, num_classes: int = 6, pretrained_backbone: bool = False):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(config.MODEL, pretrained=pretrained_backbone)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
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
        spec_times: Dict[int, np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs if specs is not None else all_spectrograms
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else all_eegs
        self.spectrogram_times = (
            spec_times if spec_times is not None else all_spectrogram_times
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def _compute_r(self, row, spec: np.ndarray, spectrogram_id: int) -> int:
        if "spectrogram_label_offset_seconds" in self.df.columns:
            offset_seconds = float(
                row.get("spectrogram_label_offset_seconds", 0.0) or 0.0
            )
            r = int(round(offset_seconds / 0.5))
        else:
            t = self.spectrogram_times.get(spectrogram_id, None)
            if t is not None and len(t) == spec.shape[0]:
                t = t.astype(np.float64)
                t_center = 0.5 * (float(t[0]) + float(t[-1]))
                t_start = t_center - 5.0  # start of the central 10 seconds
                r = int(np.argmin(np.abs(t - t_start)))
            else:
                r = int(round((spec.shape[0] - 300) / 2))

        r = int(r)
        r = max(0, min(r, max(0, spec.shape[0] - 300)))
        return r

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")

        row = self.df.iloc[index]
        spectrogram_id = int(row.spectrogram_id)
        spec = self.spectrograms[spectrogram_id]  # (time, 400)

        if spec.shape[0] < 300:
            pad_len = 300 - spec.shape[0]
            spec = np.pad(spec, ((0, pad_len), (0, 0)), mode="edge")

        r = self._compute_r(row, spec, spectrogram_id)

        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img + 1e-12)

            img = (img - SPEC_MEAN[region]) / (SPEC_STD[region] + 1e-6)
            img = np.nan_to_num(img, nan=0.0)

            img = np.clip(img, -6.0, 6.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 8
test_dataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=all_spectrograms,
    eeg_specs=all_eegs,
    spec_times=all_spectrogram_times,
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




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)
                y_preds = softmax(y_preds)
            preds.append(y_preds.detach().to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
predictions = []

for model_weight in model_weights:
    test_dataset = CustomDataset(
        test_df,
        config,
        mode="test",
        augment=False,
        specs=all_spectrograms,
        eeg_specs=all_eegs,
        spec_times=all_spectrogram_times,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    use_ckpt = isinstance(model_weight, str) and os.path.exists(model_weight)
    model = CustomModel(config, pretrained_backbone=(not use_ckpt))

    if use_ckpt:
        checkpoint = torch.load(model_weight, map_location="cpu")
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        print(
            f"WARNING: weights not found at {model_weight}. Using timm pretrained backbone instead."
        )

    model.to(device)

    prediction_dict = inference_function(test_loader, model, device)
    predictions.append(prediction_dict["predictions"])

    torch.cuda.empty_cache()
    gc.collect()

predictions = np.array(predictions)
predictions = np.mean(predictions, axis=0)

print("Predictions shape:", predictions.shape)



## === cell 11
assert predictions.shape[0] == len(
    test_df
), f"Expected {len(test_df)} rows, got {predictions.shape[0]}"
assert predictions.shape[1] == 6, f"Expected 6 columns, got {predictions.shape[1]}"

predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
