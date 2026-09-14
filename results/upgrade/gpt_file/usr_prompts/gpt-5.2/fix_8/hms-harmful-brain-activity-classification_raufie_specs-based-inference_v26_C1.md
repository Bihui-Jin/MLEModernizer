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

0.551701239162851

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the missing weight file error by making the script robust to absent `/kaggle/input/hba-efficientnet-weights/*` and instead falling back to a valid, submission-producing baseline when no weights are available. I also fix the dataset bugs that would break inference even with weights (undefined `label_cols`, missing `min/max` in test, and the incorrect indentation that overwrote EEG channels inside the loop). Finally, I ensure predictions always have shape `(len(test), 6)` and each row sums to 1, so the submission CSV is valid and won’t fail format checks.'
- What this solution (achieved 1.39779) has done: 'I keep your model/inference pipeline unchanged and instead fix the main score killer: in `mode="test"` you currently hardcode `r=0`, which mis-centers the spectrogram crop for every sample and typically hurts KL a lot. I compute a stable, per-row `r` for test using the actual spectrogram length (center-crop), while preserving the exact same preprocessing and tensor shapes. I also make the crop bounds safe for varying spectrogram lengths to avoid silent misalignment and NaNs. These are minimal, inference-only fixes that should reduce KL (move your 1.39779 closer to the 0.5517 target) without changing architecture or training.'
- What this solution (achieved 1.39779) has done: 'I keep your architecture and inference loop unchanged and focus on two minimal, score-relevant fixes in the dataset preprocessing: (1) correct the spectrogram region mapping so it matches your defined region names (LL/LP/RP/RR) instead of accidentally pulling RL, and (2) make the test-time crop `r` consistent with your train logic by centering on the middle 50s-equivalent window using `spec_h//2 - 150` (still bounded safely). These changes should reduce systematic input misalignment, which is a common source of large KL in this competition, moving your 1.39779 closer to the 0.5517 target. Everything else (model, softmax, normalization, submission formatting) stays the same to preserve core logic and stability. The script still runs end-to-end and writes a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.39779) has done: 'I make two minimal, score-relevant fixes that preserve your architecture and inference loop. First, the Kaggle spectrogram region mapping is currently wrong (it accidentally uses RL and swaps LP/RP); I align it to the intended four regions `LL, LP, RP, RR` by slicing the correct 100-column blocks in order. Second, your test-time crop uses a hardcoded `300`-row window that can mis-center compared to the train logic; I compute `r` based on the actual crop height implied by the assignment (`X[14:-14,...]` → 100 rows), keeping it safely bounded. These changes reduce systematic input misalignment, which is a major driver of high KL, and should move your 1.39779 closer to the 0.5517 target.'
- What this solution (achieved 1.39779) has done: 'I fix the shape/broadcasting bug in `CustomDataset.__data_generation` where the spectrogram crop is being transposed and then sliced in a way that produces `(100, 56)` instead of the expected `(100, 256)`. This is causing inference to crash before any submission can be written. The minimal correction is to keep the crop in `(time, freq)` order, normalize it, and then resize the frequency axis to exactly 256 with simple interpolation so the downstream model input shape stays identical. No model architecture, loss, or inference semantics are changed; this simply makes the preprocessing consistent and runnable end-to-end and should also reduce KL by avoiding malformed inputs.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far from the target (0.5517), so we should make a small but meaningful fix that improves calibration toward the KL metric without changing the model or training. The biggest remaining score killer is that you normalize each spectrogram patch independently (per-sample, per-region z-score), while these EfficientNet weights were almost certainly trained with the original global normalization constants from the training set. I compute train-derived `spec_mu/spec_std` once (fast, from a small random subset of spectrogram parquet files under the same preprocessing) and then use those fixed constants at test time instead of per-patch normalization; this preserves architecture/inference but makes preprocessing consistent with training expectations and typically reduces KL substantially. I also keep all shapes identical and keep the existing robust cropping and 256-frequency interpolation so it still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'We keep your model and preprocessing structure unchanged, but fix a remaining systematic input error that can strongly hurt KL: the test spectrogram columns are not ordered into contiguous 4×100 region blocks, yet the dataset assumes they are. I build the spectrogram tensors by explicitly concatenating `LL_*`, `LP_*`, `RP_*`, `RR_*` columns (and sort within each group by frequency), so each region slice `region*100:(region+1)*100` really corresponds to the intended brain region. This is an inference-only alignment fix (no training/architecture changes) and should materially reduce KL from 1.39779 toward your 0.5517 target. Everything else (global normalization, crop sizing, softmax, row-normalization, and submission writing) stays the same.'

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
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_9.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


_discovered = glob("/kaggle/input/hba-efficientnet-weights/*.pth")
if os.path.exists(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]
else:
    model_weights = sorted(_discovered)

print(
    f"Discovered {len(_discovered)} weight files; using {len(model_weights)} weight files."
)



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
label_cols = TARGETS  # referenced by dataset


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


def _sorted_region_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]

    def _key(cn: str):
        try:
            return float(cn.split("_", 1)[1])
        except Exception:
            return cn

    cols = sorted(cols, key=_key)
    return cols


for file_path in tqdm(paths_spectrograms):
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split(".")[0])

    spec_df = aux.iloc[:, 1:]
    ll = spec_df[_sorted_region_cols(spec_df, "LL_")]
    lp = spec_df[_sorted_region_cols(spec_df, "LP_")]
    rp = spec_df[_sorted_region_cols(spec_df, "RP_")]
    rr = spec_df[_sorted_region_cols(spec_df, "RR_")]

    if min(len(ll.columns), len(lp.columns), len(rp.columns), len(rr.columns)) == 0:
        all_spectrograms[name] = spec_df.values
    else:
        all_spectrograms[name] = pd.concat([ll, lp, rp, rr], axis=1).values

    del aux

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
TRAIN_SPEC_DIR = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
_train_spec_paths = glob(os.path.join(TRAIN_SPEC_DIR, "*.parquet"))


def estimate_global_spec_norm(train_spec_paths, max_files: int = 64, seed: int = 20):
    rng = np.random.default_rng(seed)
    if len(train_spec_paths) == 0:
        return 0.0, 1.0
    take = min(max_files, len(train_spec_paths))
    chosen = rng.choice(train_spec_paths, size=take, replace=False)

    n = 0
    mean = 0.0
    M2 = 0.0

    for p in chosen:
        df = pd.read_parquet(p)
        arr = df.iloc[:, 1:].values  # (time, 400)
        arr = np.clip(arr, np.exp(-4), np.exp(8))
        arr = np.log(arr).astype(np.float64, copy=False)
        flat = arr.reshape(-1)
        flat = flat[np.isfinite(flat)]
        for x in flat:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

    if n < 2:
        return float(mean), 1.0
    var = M2 / (n - 1)
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std


SPEC_MU, SPEC_STD = estimate_global_spec_norm(
    _train_spec_paths, max_files=64, seed=config.SEED
)
print(
    f"Estimated global spec norm: mu={SPEC_MU:.6f}, std={SPEC_STD:.6f} (from up to 64 train spectrograms)"
)




## === cell 7
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

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
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




## === cell 8
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        spec_mu: float = 0.0,
        spec_std: float = 1.0,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs if specs is not None else all_spectrograms
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else all_eegs
        self.spec_mu = float(spec_mu)
        self.spec_std = float(spec_std) if float(spec_std) > 0 else 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    @staticmethod
    def _resize_freq_to_256(img_tf: np.ndarray) -> np.ndarray:
        """
        Bugfix: ensure the (time, freq) spectrogram patch always matches (100, 256)
        expected by X[14:-14, :, region], without changing downstream model logic.
        Uses linear interpolation along the frequency axis only.
        """
        if img_tf.ndim != 2:
            raise ValueError(f"Expected 2D (time,freq) array, got shape {img_tf.shape}")
        t, f = img_tf.shape
        if f == 256:
            return img_tf.astype(np.float32, copy=False)
        x_old = np.linspace(0.0, 1.0, f, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        out = np.empty((t, 256), dtype=np.float32)
        for i in range(t):
            out[i] = np.interp(x_new, x_old, img_tf[i].astype(np.float32, copy=False))
        return out

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        spec = self.spectrograms[int(row.spectrogram_id)]
        spec_h = spec.shape[0]  # time axis length in rows

        crop_h = X[14:-14, :, 0].shape[0]  # 100 time rows to place

        if self.mode == "test":
            r = int(spec_h // 2 - crop_h // 2)
            r = max(0, min(r, max(0, spec_h - crop_h)))
        else:
            r = int((row["min"] + row["max"]) // 4)
            r = max(0, min(r, max(0, spec_h - crop_h)))

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        region_to_slice = {0: 0, 1: 1, 2: 2, 3: 3}  # LL, LP, RP, RR
        for region in range(4):
            src_region = region_to_slice[region]
            patch_tf = spec[r : r + crop_h, src_region * 100 : (src_region + 1) * 100]

            patch_tf = np.clip(patch_tf, np.exp(-4), np.exp(8))
            patch_tf = np.log(patch_tf)

            ep = 1e-6
            patch_tf = (patch_tf - self.spec_mu) / (self.spec_std + ep)
            patch_tf = np.nan_to_num(patch_tf, nan=0.0)

            patch_tf = self._resize_freq_to_256(patch_tf) / 2.0
            X[14:-14, :, region] = patch_tf

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




## === cell 9
test_dataset = CustomDataset(
    test_df, config, mode="test", augment=False, spec_mu=SPEC_MU, spec_std=SPEC_STD
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




## === cell 10
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.detach().to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 11
if len(model_weights) == 0:
    print("No model weights found. Falling back to train prior probabilities baseline.")
    train_df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    votes = train_df[TARGETS].values.astype(np.float64)
    probs = votes / np.clip(votes.sum(axis=1, keepdims=True), 1.0, None)
    prior = probs.mean(axis=0)
    prior = prior / prior.sum()
    predictions = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)
else:
    predictions = []
    for model_weight in model_weights:
        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])

        del model
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

print("Predictions shape:", predictions.shape)



## === cell 12
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(f"Predictions must be (n,6). Got {predictions.shape}")

predictions = np.clip(predictions, 1e-7, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)
out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
