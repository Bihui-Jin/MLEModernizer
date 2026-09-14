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

0.5693814071469879

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41035) has done: 'I fix the missing model-weights runtime error by falling back to an untrained EfficientNet (uniform-probability output) when the external `/kaggle/input/hba-efficientnet-weights/` files are not available, so the notebook always runs end-to-end and writes `submission.csv`. I also fix the submission shape error by guaranteeing that `predictions` is always a `(len(test_df), 6)` array, and by using the correct loader variable during inference. Finally, I enforce proper probability normalization (row sums to 1, positive, finite) to avoid Kaggle submission failures under the KL metric. These changes are correctness/stability-oriented (score-neutral to slightly worse vs trained weights), but they unblock producing a valid submission.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41035, lower-is-better) is far worse than the target (0.56938), and the main reason is that inference often falls back to a randomly initialized model because the external weights path may not exist. The smallest safe improvement is to (1) auto-discover weights from *existing* Kaggle dataset directories in your environment (both `/kaggle/input/...` and `/kaggle/data/...`) without changing the model, and (2) if no weights are found, use a simple label-prior fallback computed from `train.csv` (still a valid probabilistic baseline, typically much better than uniform for KL). These changes preserve your architecture, feature extraction, and inference loop, but replace the “random/uniform” behavior with a more informative and stable prediction source. The submission formatting and normalization safeguards are kept intact.'
- What this solution (achieved 1.41937) has done: 'Your current score is much worse than the target (lower-is-better), so the safest way to move toward 0.569 is to actually load the intended pretrained weights (instead of frequently falling back to the train-prior baseline). I keep your exact model and feature pipeline, but make weight discovery precise (prefer `tf_efficientnet_b0_epoch_6.pth`, then other `b0` weights) and robust to different checkpoint key formats (common `state_dict` / `model_state_dict` variants) so a valid weight file is much more likely to load successfully. I also ensure the checkpoint keys are normalized (strip `module.` / `model.` prefixes) to avoid strict-load failures that currently push you into the weaker fallback. Submission formatting and probability normalization stay the same.'
- What this solution (achieved 1.41937) has done: 'You’re far worse than the target (lower-is-better), and the most likely cause is that you still often fail to load the intended pretrained checkpoint because the head layer names/shapes don’t match under `strict=True`, which silently pushes you into the much weaker train-prior fallback. I keep your exact model and feature pipeline, but make checkpoint loading robust by (1) allowing `strict=False` and (2) explicitly dropping incompatible classifier/head keys so the backbone weights still load. This is a minimal, score-relevant change: getting the pretrained backbone active typically moves KL a lot closer to ~0.57 than a fixed prior. I also stop scanning every `/kaggle/input/*/*.pth` to avoid picking unrelated weights and wasting time; we only consider the explicit file and the specific `hba-efficientnet-weights` dataset locations.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.56938), and the most likely remaining issue is a subtle mismatch between what the pretrained weights expect and what your model currently feeds into EfficientNet. I keep your model backbone/head and inference loop intact, but fix `__reshape_input` to produce a standard 3x512x512 tensor (currently it concatenates along the wrong dimension, yielding a nonstandard shape that makes loaded weights ineffective). I also make the test spectrogram/eeg caches align exactly to `test.csv` ids (same data, just indexed lookup) to avoid any accidental KeyError/fallback behavior. These are minimal semantic fixes that should let the intended pretrained weights actually work and move KL much closer toward the target.'
- What this solution (achieved 1.41937) has done: 'I keep your model/dataset/inference logic intact and focus on two minimal score-relevant fixes: (1) ensure the pretrained checkpoint actually loads into the same module names you defined by mapping common timm EfficientNet keys (e.g., `conv_stem/bn1/blocks/conv_head/classifier`) into your `features.*` layout, instead of silently dropping most backbone weights; and (2) avoid averaging across multiple potentially unrelated `.pth` files by selecting the single best-matching checkpoint (prefer exact `tf_efficientnet_b0_epoch_6.pth`) to prevent dilution. If weight loading still fails, your existing train-prior fallback remains unchanged (so the notebook always produces a valid `submission.csv`). These changes should move KL substantially closer to your target by making inference use meaningful pretrained features rather than an effectively-random backbone.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.419, lower-is-better) is far from the target (0.569), and the most likely reason is that the pretrained weights are still not being applied correctly at inference time (so the model behaves close to an untrained/random backbone). I make a minimal, score-relevant change to load the checkpoint into the *actual timm backbone module* (`self.model`) instead of trying to remap keys into `self.features`, while still keeping your exact forward path and head. If the checkpoint includes a classifier head with mismatched shape, we drop only those incompatible keys and load the rest strictly enough to activate the pretrained feature extractor. This should materially improve KL without changing the core architecture, feature extraction, or inference semantics, and still guarantees a valid `submission.csv`.'

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
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_6.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


def _discover_weight_paths() -> List[str]:
    patterns = [
        "/kaggle/input/hba-efficientnet-weights/*.pth",
        "/kaggle/data/hba-efficientnet-weights/*.pth",
    ]
    found = []
    for pat in patterns:
        found.extend(glob(pat))

    found.append(paths.MODEL_WEIGHTS)

    found = sorted(set(found))

    def _priority(p: str):
        base = os.path.basename(p)
        if base == "tf_efficientnet_b0_epoch_6.pth":
            return (0, p)
        if "tf_efficientnet_b0" in base or "efficientnet_b0" in base or "b0" in base:
            return (1, p)
        return (2, p)

    found = sorted(found, key=_priority)
    return found


model_weights = _discover_weight_paths()
print("Discovered weights:", model_weights[:10], f"(total={len(model_weights)})")

model_weights = model_weights[:1]
print("Using weight candidates:", model_weights)




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
    coeff = pywt.wavedec(x, wavelet, mode="per")  # multilevel 1D DWT
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


def plot_spectrogram(spectrogram_path: str):
    """
    Visualize spectrogram recordings from a parquet file.
    """
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
test_spec_ids = set(test_df["spectrogram_id"].astype(int).unique().tolist())
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquets on disk")

all_spectrograms = {}
for file_path in tqdm(paths_spectrograms, desc="Load test spectrograms"):
    name = int(os.path.basename(file_path).split(".")[0])
    if name not in test_spec_ids:
        continue
    aux = pd.read_parquet(file_path)
    all_spectrograms[name] = aux.iloc[:, 1:].values
    del aux

print(f"Loaded {len(all_spectrograms)} spectrograms referenced by test.csv")

if config.VISUALIZE:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)




## === cell 5
test_eeg_ids = set(test_df["eeg_id"].astype(int).unique().tolist())
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets on disk")

all_eegs = {}
counter = 0
for file_path in tqdm(paths_eegs, desc="Build EEG spectrograms"):
    eeg_id = int(os.path.basename(file_path).split(".")[0])
    if eeg_id not in test_eeg_ids:
        continue
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs[eeg_id] = eeg_spectrogram
    counter += 1

print(f"Built {len(all_eegs)} EEG-derived spectrograms referenced by test.csv")




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

    def __reshape_input(self, x):
        """
        Reshapes input (N, 128, 256, 8) -> (N, 3, 512, 512) image (NCHW).

        Change (score-critical): the original code concatenated along dim=2 then dim=3,
        which produced a nonstandard shape and makes pretrained EfficientNet weights largely ineffective.
        We keep the same content (kaggle spectrograms + EEG spectrograms), but place them as:
          - width: concatenate regions (4*128=512)
          - height: concatenate kaggle+eeg (256+256=512)
          - channels: replicate to 3
        """
        spect = x[:, :, :, 0:4]  # (N, 128, 256, 4)
        eeg = x[:, :, :, 4:8]  # (N, 128, 256, 4)

        spect = spect.permute(0, 3, 1, 2)
        eeg = eeg.permute(0, 3, 1, 2)

        spect = spect[:, :, :, 64:192]  # (N, 4, 128, 128)
        eeg = eeg[:, :, :, 64:192]  # (N, 4, 128, 128)

        spect = torch.cat(
            [spect[:, i : i + 1] for i in range(4)], dim=3
        )  # (N,1,128,512)
        eeg = torch.cat([eeg[:, i : i + 1] for i in range(4)], dim=3)  # (N,1,128,512)

        if self.USE_KAGGLE_SPECTROGRAMS and self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spect, eeg], dim=2)  # (N,1,256,512)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eeg  # (N,1,128,512)
        else:
            x = spect  # (N,1,128,512)

        x = torch.cat([x, x], dim=2)  # height*2
        if x.shape[2] < 512:
            x = torch.cat([x, x], dim=2)  # ensure reach 512 for the single-source case
        x = x[:, :, :512, :]  # (N,1,512,512)

        x = x.repeat(1, 3, 1, 1)  # (N,3,512,512)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 7
label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = all_spectrograms,
        eeg_specs: Dict[int, np.ndarray] = all_eegs,
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = all_spectrograms
        self.eeg_spectrograms = all_eegs

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
        r = 0  # test mode always uses centered window

        for region in range(4):
            img = self.spectrograms[int(row.spectrogram_id)][
                r : r + 300, region * 100 : (region + 1) * 100
            ].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        img_eeg = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = img_eeg

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
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
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
def _compute_train_prior(train_csv_path: str, alpha: float = 1.0) -> np.ndarray:
    train_df = pd.read_csv(train_csv_path, usecols=label_cols)
    counts = train_df[label_cols].sum(axis=0).values.astype(np.float64)
    prior = (counts + alpha) / (counts.sum() + alpha * len(label_cols))
    prior = prior.astype(np.float32)
    prior = np.clip(prior, 1e-8, 1.0)
    prior = prior / prior.sum()
    return prior


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if len(ckpt) > 0 and all(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    return ckpt


def _normalize_state_dict_keys(state: dict) -> dict:
    if not isinstance(state, dict):
        return state
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


def _strip_prefix_if_present(state: dict, prefix: str) -> dict:
    if not isinstance(state, dict):
        return state
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    print(f"Stripped prefix '{prefix}' from checkpoint keys where applicable.")
    return out


def _drop_classifier_keys_for_backbone_load(state: dict) -> dict:
    """
    Change (score-relevant, minimal): ensure we can load the pretrained EfficientNet backbone
    even if the checkpoint contains a classifier head with incompatible output dim.
    We only drop classifier/global_pool/fc keys; backbone conv/bn/blocks remain.
    """
    if not isinstance(state, dict):
        return state
    drop_prefixes = ("classifier.", "fc.", "head.", "global_pool.")
    cleaned = {}
    dropped = 0
    for k, v in state.items():
        if k.startswith(drop_prefixes):
            dropped += 1
            continue
        cleaned[k] = v
    if dropped > 0:
        print(
            f"Dropped {dropped} classifier/global_pool keys to load backbone weights."
        )
    return cleaned


def _load_checkpoint_into_timm_backbone(custom_model: nn.Module, state: dict) -> bool:
    """
    Change (score-critical): load weights directly into custom_model.model (the timm EfficientNet),
    instead of trying to remap into custom_model.features indices. This avoids silent partial loads
    that keep the backbone effectively random, which causes very poor KL.
    """
    if not isinstance(state, dict):
        return False

    for pref in ["net.", "backbone.", "encoder."]:
        state = _strip_prefix_if_present(state, pref)

    state = _strip_prefix_if_present(state, "model.")

    state_bb = _drop_classifier_keys_for_backbone_load(state)

    try:
        missing, unexpected = custom_model.model.load_state_dict(state_bb, strict=False)
        print("Loaded checkpoint into timm backbone (partial ok).")
        if len(missing) > 0:
            print(f"Backbone missing keys (ok): {len(missing)}")
        if len(unexpected) > 0:
            print(f"Backbone unexpected keys (ok): {len(unexpected)}")
        return True
    except Exception as e:
        print(f"WARNING: backbone load_state_dict failed due to: {repr(e)}")
        return False


predictions = []
loaded_any_weights = False

for model_weight in model_weights:
    if not os.path.exists(model_weight):
        continue

    ds = CustomDataset(test_df, config, mode="test", augment=False)
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    model = CustomModel(config).to(device)

    try:
        checkpoint = torch.load(model_weight, map_location="cpu")
        state = _extract_state_dict(checkpoint)
        state = _normalize_state_dict_keys(state)

        ok = _load_checkpoint_into_timm_backbone(model, state)
        if not ok:
            raise RuntimeError("Could not load checkpoint into timm backbone.")

        print(f"Using weights for inference: {model_weight}")
        loaded_any_weights = True
        pred_dict = inference_function(loader, model, device)
        predictions.append(pred_dict["predictions"])
    except Exception as e:
        print(f"WARNING: failed to use weights ({model_weight}) due to: {repr(e)}")
        print("Skipping this weight file.")

    torch.cuda.empty_cache()
    gc.collect()

if loaded_any_weights and len(predictions) > 0:
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
else:
    print("No valid weights loaded; using train label prior fallback for predictions.")
    prior = _compute_train_prior(paths.TRAIN_CSV, alpha=1.0)
    predictions = np.tile(prior[None, :], (len(test_df), 1))

print("Predictions shape:", predictions.shape)




## === cell 11
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

if (
    predictions.ndim != 2
    or predictions.shape[1] != 6
    or predictions.shape[0] != len(test_df)
):
    print(
        "WARNING: unexpected predictions shape; falling back to uniform distribution."
    )
    predictions = np.full((len(test_df), 6), 1 / 6, dtype=np.float32)

predictions = np.nan_to_num(predictions, nan=1 / 6, posinf=1 / 6, neginf=1 / 6).astype(
    np.float32
)
predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Wrote:", out_path)
sub.head()
