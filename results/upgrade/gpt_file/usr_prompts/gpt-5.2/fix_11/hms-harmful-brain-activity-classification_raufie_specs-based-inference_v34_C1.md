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

0.5623183646753916

# 6. Current score

1.48867

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Your code doesn’t currently yield a Kaggle score mainly because it depends on external model weights at `/kaggle/input/hba-efficientnet-weights/`, which aren’t present in the provided data paths; this prevents a valid end-to-end submission in this environment. I add a minimal, deterministic fallback that still produces a valid `submission.csv` with correct columns and probabilities summing to 1, by using the competition’s empirical class prior from `train.csv` (a strong baseline for KL divergence). If the weights are available, your original inference path run unchanged; otherwise it automatically switch to the prior-based submission (no architecture/training changes). This ensures you always get a valid submission and a reasonable baseline score that moves you toward the target versus “not yielded”.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.39779, lower-is-better) is far worse than the target (0.5623), so we should improve it with the smallest changes that keep your overall pipeline intact. The biggest low-risk gain here is to post-process model probabilities (when weights exist) with a tiny, validated calibration toward the empirical class prior, which often reduces KL divergence by avoiding overconfident errors. When weights do not exist, your prior-only fallback is already the right direction, but we compute the prior more robustly by aggregating votes at the `eeg_id` level (matching the test granularity) and applying a tiny epsilon to prevent any exact zeros. Finally, we ensure the submission is always valid (strict row sums to 1, no NaNs, no zeros) without changing your model, features, or inference loop.'
- What this solution (achieved 1.39779) has done: 'We’re currently far worse than the target (1.48867 vs 0.5623, lower-is-better), so we should make the smallest reliable changes that reduce KL without touching your model/feature/inference core. The highest-impact minimal fix is to correctly convert train vote *counts* into training-distribution probabilities by normalizing per-row before averaging; your current prior sums votes per `eeg_id` and then averages, which over-weights eegs with many overlapping labeled segments and can distort the true label distribution. Then we reuse that corrected prior both for the fallback submission and for the existing post-hoc blending step, and we add a tiny “safe probability” renormalization to guarantee strictly positive, sum-to-1 rows (important for KL). Finally, we keep your architecture/inference identical and only adjust the prior computation + reuse it consistently.'
- What this solution (achieved 1.48697) has done: 'Your current score is much worse than the target (1.39779 vs 0.5623, lower-is-better), so we should improve the submission with minimal, low-risk changes that don’t touch your model/inference core. The biggest reliable gain in your current pipeline is to compute a *test-granularity-matched* class prior by aggregating vote probabilities at the `eeg_id` level (train has many overlapping segments per eeg; test has one row per eeg). Then we use that corrected prior both for the fallback (no weights) and the existing post-hoc blending (weights present), which typically reduces KL by preventing overconfident miscalibration. Finally, we keep your “always-valid submission” safeguards but ensure the prior computation is consistent and robust (strictly positive, normalized).'
- What this solution (achieved 1.48697) has done: 'Your current score (1.48697, lower-is-better) is far from the target (0.5623), so we should make a small, low-risk improvement that reduces KL without changing your model/feature/inference core. The most reliable knob here is probability calibration: KL heavily penalizes overconfident wrong predictions, so we slightly strengthen your existing “blend with prior” step (still the same logic) and make it robust by selecting the blend weight via a simple patient-grouped CV on `train.csv` using only the targets (no EEG/spectrogram features). This preserves your architecture and inference loop, but chooses a better `alpha` than the current fixed 0.04, which should move the score downward toward the target. We also keep the no-weights fallback unchanged (prior submission), and we ensure strict positivity + row-sum-to-1 in all cases.'
- What this solution (achieved 1.48697) has done: 'Your current score (1.48697, lower-is-better) is still far from the target (0.5623), so the most likely “minimal change, reliable KL gain” is to improve probability calibration without touching your model, features, or inference loop. Right now `choose_alpha_via_cv()` mistakenly evaluates blending using `y_true` as if it were the model predictions, which makes the chosen `alpha` meaningless and can worsen calibration on test. I fix `choose_alpha_via_cv()` to select `alpha` by simulating the *kind* of miscalibration we actually have (overconfident predictions) via a simple temperature sharpening/softening of `y_true` (still label-only, no feature/model changes), then pick `alpha` that minimizes KL on patient-grouped folds. Finally, I slightly expand the alpha grid to include stronger blending values (often helps KL a lot) and keep the existing safe-normalization guarantees so submission validity is unchanged.'
- What this solution (achieved 1.48697) has done: 'Your score is far worse than the target (1.48697 vs 0.5623; lower is better), so we should reduce KL with the smallest, safest changes that don’t touch your model/feature extraction/inference loop. The most direct low-risk fix is to correct a bug in `CustomDataset.__data_generation`: right now the EEG spectrogram (`X[:,:,4:]`) is overwritten inside the `for region` loop, so the final `X` depends only on the last region’s kaggle-spectrogram slice; moving the EEG assignment outside the loop preserves intended semantics and typically improves predictions. Next, we make the prior-blend calibration a bit more robust by (1) including a slightly wider alpha grid and (2) using a more realistic `gamma_sim` (closer to typical overconfidence) so the chosen alpha better reduces KL. All outputs remain strictly positive and row-normalized to guarantee a valid submission.'
- What this solution (achieved 1.48697) has done: 'We’re far worse than the target (1.48697 vs 0.5623; lower is better), so we want a small, low-risk KL improvement without changing your model/feature/inference core. The biggest remaining calibration win is to make the prior-blend weight selection actually reflect your model’s behavior: instead of simulating “model-like” predictions by power-transforming `y_true`, we estimate an “overconfidence temperature” from the model’s own test predictions (no labels) and use that to choose `alpha` via patient-grouped CV on train labels. This keeps the same blending logic, but makes the chosen `alpha` meaningful and typically reduces KL by better damping overconfident outputs. I also make sure the chosen alpha search remains deterministic and keep the submission validity safeguards unchanged.'
- What this solution (achieved 1.48867) has done: 'Your current score (1.48697; lower-is-better) is far from the target (0.5623), so we should make a small change that reliably reduces KL without touching your model, features, or inference loop. The most impactful low-risk fix is to compute the prior used for blending in a way that matches the competition’s label construction: first aggregate raw vote counts to `eeg_id`, then normalize to probabilities (instead of averaging per-row probabilities), which avoids bias from overlapping segments and different vote totals. This corrected prior is then used identically in both the no-weights fallback and the post-hoc blending path, keeping all other logic unchanged. Finally, we keep your strict positivity and row-sum-to-1 safeguards so the submission remains valid.'

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
    MODEL = "tf_efficientnet_b7"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b7_epoch_0.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


model_weights = sorted(glob("/kaggle/input/hba-efficientnet-weights/*.pth"))
if len(model_weights) == 0 and os.path.exists(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]
print(f"Found {len(model_weights)} weight file(s).")



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

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

seed_everything(config.SEED)



## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 4
def _safe_normalize_rows(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=False
    )
    mat = np.clip(mat, 0.0, None)
    mat = mat + eps
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def compute_train_prior(
    train_csv_path: str,
    target_cols: List[str],
    eps: float = 1e-6,
) -> np.ndarray:
    usecols = ["eeg_id"] + list(target_cols)
    train_df = pd.read_csv(train_csv_path, usecols=usecols)

    eeg_votes = train_df.groupby("eeg_id", sort=False)[target_cols].sum()
    votes = eeg_votes.values.astype(np.float64, copy=False)
    row_sums = np.clip(votes.sum(axis=1, keepdims=True), 1.0, None)
    eeg_probs = votes / row_sums
    eeg_probs = _safe_normalize_rows(eeg_probs, eps=eps)

    prior = eeg_probs.mean(axis=0)
    prior = np.maximum(prior, eps)
    prior = prior / prior.sum()
    return prior.astype(np.float64)


def make_prior_submission(
    train_csv_path: str,
    test_df: pd.DataFrame,
    target_cols: List[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    prior = compute_train_prior(train_csv_path, target_cols, eps=eps)

    sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    for i, c in enumerate(target_cols):
        sub[c] = prior[i]

    sub[target_cols] = sub[target_cols].astype(np.float64)
    sub[target_cols] = sub[target_cols].div(sub[target_cols].sum(axis=1).values, axis=0)
    return sub


def blend_with_prior(
    probs: np.ndarray, prior: np.ndarray, alpha: float = 0.04, eps: float = 1e-6
) -> np.ndarray:
    probs = probs.astype(np.float64, copy=False)
    prior = prior.astype(np.float64, copy=False)

    probs = np.nan_to_num(
        probs,
        nan=1.0 / probs.shape[1],
        posinf=1.0 / probs.shape[1],
        neginf=1.0 / probs.shape[1],
    )
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    blended = (1.0 - alpha) * probs + alpha * prior[None, :]
    blended = _safe_normalize_rows(blended, eps=eps)
    return blended


def _kl_divergence_rowwise(
    p_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-6
) -> float:
    p_true = _safe_normalize_rows(p_true, eps=eps)
    p_pred = _safe_normalize_rows(p_pred, eps=eps)
    return float(np.mean(np.sum(p_true * (np.log(p_true) - np.log(p_pred)), axis=1)))


def _power_calibration(
    probs: np.ndarray, gamma: float = 1.0, eps: float = 1e-6
) -> np.ndarray:
    probs = _safe_normalize_rows(probs, eps=eps)
    probs = np.clip(probs, eps, None) ** gamma
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def estimate_gamma_from_pred_confidence(
    probs: np.ndarray,
    target_mean_max: float = 0.55,
    gamma_min: float = 0.6,
    gamma_max: float = 1.6,
) -> float:
    probs = _safe_normalize_rows(np.asarray(probs, dtype=np.float64), eps=1e-6)
    mean_max = float(np.mean(np.max(probs, axis=1)))
    if not np.isfinite(mean_max):
        return 1.0
    ratio = mean_max / max(target_mean_max, 1e-6)
    gamma = 1.0 / ratio
    gamma = float(np.clip(gamma, gamma_min, gamma_max))
    return gamma


def choose_alpha_via_cv(
    train_csv_path: str,
    target_cols: List[str],
    candidate_alphas: List[float],
    eps: float = 1e-6,
    n_folds: int = 5,
    seed: int = 20,
    gamma_sim: float = 0.85,
) -> float:
    usecols = ["eeg_id", "patient_id"] + list(target_cols)
    df = pd.read_csv(train_csv_path, usecols=usecols)

    votes = df[target_cols].values.astype(np.float64, copy=False)
    row_sums = np.clip(votes.sum(axis=1, keepdims=True), 1.0, None)
    row_probs = votes / row_sums
    row_probs = _safe_normalize_rows(row_probs, eps=eps)

    tmp = df[["eeg_id", "patient_id"]].copy()
    for i, c in enumerate(target_cols):
        tmp[c] = row_probs[:, i]
    eeg_df = (
        tmp.groupby("eeg_id", sort=False)
        .agg({**{c: "mean" for c in target_cols}, "patient_id": "first"})
        .reset_index()
    )

    patients = eeg_df["patient_id"].values
    unique_patients = np.unique(patients)

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)
    folds = np.array_split(unique_patients, n_folds)

    prior = compute_train_prior(train_csv_path, target_cols, eps=eps)

    y_true = eeg_df[target_cols].values.astype(np.float64, copy=False)
    y_true = _safe_normalize_rows(y_true, eps=eps)

    y_model_like = _power_calibration(y_true, gamma=gamma_sim, eps=eps)

    best_alpha = candidate_alphas[0]
    best_score = float("inf")

    for alpha in candidate_alphas:
        fold_scores = []
        for k in range(n_folds):
            val_patients = folds[k]
            is_val = np.isin(patients, val_patients)

            y_pred = blend_with_prior(
                y_model_like[is_val], prior=prior, alpha=alpha, eps=eps
            )
            fold_scores.append(_kl_divergence_rowwise(y_true[is_val], y_pred, eps=eps))

        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score - 1e-12:
            best_score = mean_score
            best_alpha = alpha

    scored = []
    for alpha in candidate_alphas:
        fold_scores = []
        for k in range(n_folds):
            val_patients = folds[k]
            is_val = np.isin(patients, val_patients)
            y_pred = blend_with_prior(
                y_model_like[is_val], prior=prior, alpha=alpha, eps=eps
            )
            fold_scores.append(_kl_divergence_rowwise(y_true[is_val], y_pred, eps=eps))
        scored.append((alpha, float(np.mean(fold_scores))))
    scored.sort(key=lambda x: x[1])
    tol = 1e-5
    best_val = scored[0][1]
    best_small_alpha = min(a for a, s in scored if s <= best_val + tol)

    return float(best_small_alpha)


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

if len(model_weights) == 0:
    print(
        "No external weight files found. Writing a prior-based baseline submission to submission.csv"
    )
    sub = make_prior_submission(paths.TRAIN_CSV, test_df, TARGETS, eps=1e-6)
    sub.to_csv("submission.csv", index=False)
    print(f"Submission shape: {sub.shape}")
    print(sub.head())
    print(
        "Row-sum stats:",
        float(sub[TARGETS].sum(axis=1).min()),
        float(sub[TARGETS].sum(axis=1).max()),
    )



## === cell 5
if len(model_weights) > 0:
    paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
    print(f"There are {len(paths_spectrograms)} spectrogram parquets")
    all_spectrograms = {}

    for file_path in tqdm(paths_spectrograms):
        aux = pd.read_parquet(file_path)
        name = int(file_path.split("/")[-1].split(".")[0])
        all_spectrograms[name] = aux.iloc[:, 1:].values
        del aux

    if config.VISUALIZE:
        idx = np.random.randint(0, len(paths_spectrograms))
        spectrogram_path = paths_spectrograms[idx]
        plot_spectrogram(spectrogram_path)



## === cell 6
if len(model_weights) > 0:
    paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
    print(f"There are {len(paths_eegs)} EEG spectrograms")
    all_eegs = {}
    counter = 0

    for file_path in tqdm(paths_eegs):
        eeg_id = file_path.split("/")[-1].split(".")[0]
        eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
        all_eegs[int(eeg_id)] = eeg_spectrogram
        counter += 1



## === cell 7
if len(model_weights) > 0:

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
            spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
            spectrograms = torch.cat(spectrograms, dim=1)

            eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
            eegs = torch.cat(eegs, dim=1)

            if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
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
if len(model_weights) > 0:

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
            if self.mode == "test":
                r = 0
            else:
                r = int((row["min"] + row["max"]) // 4)

            eeg_img = self.eeg_spectrograms[row.eeg_id]
            X[:, :, 4:] = eeg_img

            for region in range(4):
                img = self.spectrograms[row.spectrogram_id][
                    r : r + 300, region * 100 : (region + 1) * 100
                ].T

                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                ep = 1e-6
                mu = np.nanmean(img.flatten())
                std = np.nanstd(img.flatten())
                img = (img - mu) / (std + ep)
                img = np.nan_to_num(img, nan=0.0)
                X[14:-14, :, region] = img[:, 22:-22] / 2.0

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
if len(model_weights) > 0:
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



## === cell 10
if len(model_weights) > 0:

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
                preds.append(y_preds.to("cpu").numpy())
        return {"predictions": np.concatenate(preds, axis=0)}




## === cell 11
if len(model_weights) > 0:
    predictions = []

    for model_weight in model_weights:
        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])

        del model, checkpoint, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    print("Predictions shape:", predictions.shape)



## === cell 12
if len(model_weights) > 0:
    prior = compute_train_prior(paths.TRAIN_CSV, TARGETS, eps=1e-6)

    candidate_alphas = [
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.16,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    gamma_sim_est = estimate_gamma_from_pred_confidence(
        predictions, target_mean_max=0.55
    )
    print("Estimated gamma_sim from test prediction confidence:", gamma_sim_est)

    chosen_alpha = choose_alpha_via_cv(
        paths.TRAIN_CSV,
        TARGETS,
        candidate_alphas=candidate_alphas,
        eps=1e-6,
        n_folds=5,
        seed=config.SEED,
        gamma_sim=gamma_sim_est,
    )
    print("Chosen prior-blend alpha:", chosen_alpha)

    predictions = blend_with_prior(
        predictions, prior=prior, alpha=chosen_alpha, eps=1e-6
    )

    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    sub[TARGETS] = predictions

    sub[TARGETS] = _safe_normalize_rows(sub[TARGETS].values, eps=1e-6)

    sub.to_csv("submission.csv", index=False)
    print(f"Submission shape: {sub.shape}")
    print(sub.head())
    print(
        "Row-sum stats:",
        float(sub[TARGETS].sum(axis=1).min()),
        float(sub[TARGETS].sum(axis=1).max()),
    )
    print(
        "Min prob per class:",
        {c: float(sub[c].min()) for c in TARGETS},
    )
