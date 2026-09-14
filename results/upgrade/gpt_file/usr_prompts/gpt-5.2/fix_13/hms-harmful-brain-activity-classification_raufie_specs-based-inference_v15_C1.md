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

0.5850530216171643

# 6. Current score

0.8329

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the runtime failure caused by missing external model weights by falling back to a deterministic “prior” prediction computed from the training vote distributions; this keeps the pipeline end-to-end and produces a valid `submission.csv`. I also fix dataset bugs that would break inference even if weights existed (undefined `label_cols`, missing `min/max` columns in `test_df`, and an indentation bug that overwrote EEG features inside the loop). Finally, I guarantee the submission has the correct column order, correct row count, and per-row probabilities that sum to 1 (required to avoid submission rejection). These changes are correctness/stability oriented; without the provided weights, they also provide a reasonable baseline likely closer to the target than uniform guessing.'
- What this solution (achieved 1.06434) has done: 'Your current score is much worse than the target (lower is better), and the main reason is that you are using a single global prior for every test row when model weights are absent. I keep your exact feature extraction, dataset, model, and inference code intact, but replace the fallback prior with a slightly more informative and still leak-free prior: compute a patient-level vote prior from the training set and use it for test rows with known `patient_id`, otherwise back off to the global prior. This is a minimal change that should move KL divergence closer to the target by accounting for strong patient-specific label tendencies without altering the core approach. I also add a tiny Dirichlet-style smoothing to avoid overconfident zeros (stability for KL).'
- What this solution (achieved 1.06434) has done: 'I keep your pipeline and model code unchanged, but make the “no weights available” fallback prior more informative to reduce KL toward the target. Specifically, instead of only a patient-level prior, I use a smoothed backoff chain `patient_id → (patient_id, spectrogram_id cluster) → global`, where the middle level is a lightweight spectrogram-conditioned prior computed from train metadata (no leakage; uses only train labels). I also ensure all priors are computed from normalized vote probabilities with the same Dirichlet-style smoothing you already use, preserving valid non-zero probabilities and per-row normalization. This is a minimal change localized to the fallback prediction block and should improve score when weights are missing or not found.'
- What this solution (achieved 1.06434) has done: 'Your current score (1.06434, lower-is-better) is still far from the target (0.58505), and the biggest limitation is that when weights are missing you predict using only metadata-based priors. To move KL closer to the target with minimal code change and without touching the model/feature pipeline, I make that fallback prior more informative by adding an `eeg_id`-conditioned prior (train has many overlapping windows per `eeg_id`, while test has one row per `eeg_id`). I use a conservative backoff chain `eeg_id → (patient_id, spectrogram_id) → patient_id → spectrogram_id → global`, keeping the same Dirichlet-style smoothing and strict per-row normalization to avoid invalid submissions. This change is localized to the fallback block and should improve score whenever weights are absent.'
- What this solution (achieved 0.81137) has done: 'I keep your model/dataset/inference pipeline unchanged and only improve the “no weights available” (or weak-weights) probability fallback to reduce KL toward the target. The current fallback uses hard backoff to a few priors; I instead use a conservative convex mixture of available priors (eeg_id/patient_id/spectrogram_id/global), which is typically better under KL than picking a single prior and is still fully leak-free (train-metadata only). I also switch the smoothing from “add alpha to probabilities” to proper Dirichlet posterior mean using vote counts (add alpha to counts before normalizing), which is more consistent with the competition’s vote targets and should move the score down. Finally, I keep the same output formatting/normalization guarantees so the submission remains valid.'
- What this solution (achieved 0.80442) has done: 'Your current score (0.81137, lower-is-better) is still above the target (0.58505), so we should cautiously improve (lower) KL without changing the model/feature pipeline. Since you are almost certainly running in “no weights available” mode, the only lever is the fallback prior; the smallest, most on-metric improvement is to replace fixed mixture weights with *data-driven* weights based on how reliable each conditional prior is in training (larger vote mass ⇒ lower-variance Dirichlet posterior ⇒ better expected KL). Concretely, we compute per-group total vote counts for `eeg_id`, `patient_id`, `spectrogram_id`, and `(patient_id, spectrogram_id)` from `train.csv`, and use those as mixture weights (plus a small global base weight) at inference time. This preserves your exact priors and smoothing, but makes the ensemble adaptively trust `eeg_id` when it has lots of training support and back off smoothly otherwise, which should move the score down toward the target while keeping the submission valid.'
- What this solution (achieved 1.22804) has done: 'Your current score (0.80442, lower-is-better) is still above the target (0.58505), so we should make a small, low-risk improvement without changing your model/feature pipeline. The biggest lever (since weights are likely missing) is the metadata fallback prior: I keep your same priors and supports, but make the mixture weights less “spiky” by applying a concave transform to supports (square-root) and slightly increasing the always-on global base weight to reduce over-trusting noisy, highly specific groups under KL. This is a minimal, localized change that typically improves KL by preventing overconfident conditional priors when group support is moderate. The submission formatting, column order, and per-row normalization remain unchanged.'
- What this solution (achieved 1.24935) has done: 'Your current score (1.22804, lower-is-better) is far above the target (0.58505), so we should cautiously improve (lower) KL without changing your model/feature pipeline. The biggest low-risk lever is still the “no weights available” fallback: right now the global base weight is large and supports are square-root transformed, which can over/under-trust priors in a way that’s suboptimal under KL. I make a minimal, localized change to use a *log1p* support transform (less extreme than raw or sqrt at high supports) and reduce the always-on global base weight so informative `eeg_id`/patient/spec priors influence more when supported. I also add a tiny uniform mixing floor to prevent any prior from becoming too sharp (helps KL stability) while keeping per-row normalization and submission formatting identical.'
- What this solution (achieved 0.93369) has done: 'Your current score is worse than the target (lower is better), and since weights are likely missing the only lever is the metadata-based fallback; the last change (log1p supports + smaller global weight + 2% uniform floor) made KL much worse, so we should revert that oversmoothing. I keep your exact prior computation and mixture-by-support core logic, but switch back to the previously better-behaving concave support transform (sqrt) and reduce the uniform mixing floor to a tiny epsilon (to avoid KL issues without washing out signal). I also set a moderate always-on global base weight (not too large) so informative eeg/patient/spec priors can dominate when well-supported, while still backing off safely when they’re not. Everything else (feature extraction, dataset, model, inference, submission formatting) stays identical and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.83647) has done: 'Your current score (0.93369, lower-is-better) is still far above the target (0.58505), and since model weights are absent the only meaningful lever is the metadata-based fallback prior. I keep your exact prior computation and mixture-by-support core logic, but adjust the fallback mixture weights to be slightly more “specificity-favoring” under KL: reduce the always-on global base weight, and slightly down-weight the noisier `spectrogram_id` prior while keeping `eeg_id`/`patient_id`/joint strong when supported. I also reduce the uniform mixing epsilon a bit (it can wash out signal under KL), while still keeping a nonzero floor for stability/valid submissions. Everything else (feature extraction, dataset/model, inference, submission formatting) remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.80144) has done: 'Your current score (0.83647, lower-is-better) is still well above the target (0.58505), so we should make a small, low-risk improvement that only affects the “no weights found” fallback prior (since that is what’s driving score). I keep your exact priors, smoothing, and mixture-by-support structure, but (1) lower the always-on global base weight so supported conditional priors influence more, and (2) slightly reduce the uniform mixing epsilon (it can wash out signal under KL) while keeping a nonzero floor for submission safety. This is a minimal/localized tweak that should move KL downward without changing your model/dataset/feature pipeline or submission semantics. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.8329) has done: 'Your current score (0.80144, lower-is-better) is still well above the target (0.58505), so we should make a small, low-risk improvement confined to the metadata fallback (since weights are likely missing). The main tweak is to make the support-based mixture less overconfident by adding a small per-row “prior sharpening control”: we increase the Dirichlet smoothing for low-support groups by blending each conditional prior slightly back toward the global prior as support decreases (a classic shrinkage step that typically improves KL). This preserves your existing priors, the same mixture-by-support core logic, and the same output semantics, but reduces variance/overfitting of noisy group priors (especially joint/spec groups). Submission formatting, normalization, and file path remain unchanged.'

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
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b2"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b2_epoch_9.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )


model_weights = [x for x in glob("/kaggle/input/hba-efficientnet-weights/*.pth")]
if len(model_weights) == 0 and os.path.exists(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]

print("Found model weights:", model_weights)



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
    os.environ["PYTHONHASHSEED"] = str(seed)


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



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
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)



## === cell 5
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets")
all_eegs = {}
counter = 0

for file_path in tqdm(paths_eegs):
    eeg_id = file_path.split("/")[-1].split(".")[0]
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1 and config.VISUALIZE)
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




## === cell 7
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
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs

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

        r = 0

        for region in range(4):
            spec = self.spectrograms[int(row.spectrogram_id)]
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        if self.mode != "test" and all(c in row.index for c in label_cols):
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
train_df = pd.read_csv(paths.TRAIN_CSV)

alpha = 0.5

vote_counts = train_df[label_cols].astype(np.float32).values
vote_counts = np.nan_to_num(vote_counts, nan=0.0, posinf=0.0, neginf=0.0)
row_count_sums = vote_counts.sum(axis=1, keepdims=True)
row_count_sums = np.where(row_count_sums == 0, 1.0, row_count_sums)
vote_probs = vote_counts / row_count_sums  # kept for compatibility/inspection if needed


def dirichlet_mean_from_counts(counts_sum: np.ndarray, alpha: float) -> np.ndarray:
    counts_sum = counts_sum.astype(np.float32)
    p = counts_sum + alpha
    p = np.clip(p, 1e-8, None)
    p = p / p.sum()
    return p.astype(np.float32)


global_prior = dirichlet_mean_from_counts(vote_counts.sum(axis=0), alpha=alpha)
print("Global fallback prior:", dict(zip(label_cols, global_prior.round(6))))


def build_group_prior_map(
    df: pd.DataFrame, group_cols: List[str], label_cols: List[str], alpha: float
) -> Dict:
    tmp = df[group_cols + label_cols].copy()
    for c in label_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).astype(np.float32)
    grouped = tmp.groupby(group_cols, sort=False)[label_cols].sum()

    priors = (grouped + alpha).div(
        grouped.sum(axis=1) + alpha * len(label_cols), axis=0
    )
    priors = priors.clip(lower=1e-8)
    priors = priors.div(priors.sum(axis=1), axis=0)

    if len(group_cols) == 1:
        return {int(k): priors.loc[k].values.astype(np.float32) for k in priors.index}
    else:
        return {
            tuple(int(x) for x in k): priors.loc[k].values.astype(np.float32)
            for k in priors.index
        }


patient_prior_map = build_group_prior_map(
    train_df, ["patient_id"], label_cols, alpha=alpha
)
spec_prior_map = build_group_prior_map(
    train_df, ["spectrogram_id"], label_cols, alpha=alpha
)
joint_prior_map = build_group_prior_map(
    train_df, ["patient_id", "spectrogram_id"], label_cols, alpha=alpha
)
eeg_prior_map = build_group_prior_map(train_df, ["eeg_id"], label_cols, alpha=alpha)


def build_group_support_map(
    df: pd.DataFrame, group_cols: List[str], label_cols: List[str]
) -> Dict:
    tmp = df[group_cols + label_cols].copy()
    for c in label_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).astype(np.float32)
    support = tmp.groupby(group_cols, sort=False)[label_cols].sum().sum(axis=1)

    if len(group_cols) == 1:
        return {int(k): float(support.loc[k]) for k in support.index}
    else:
        return {tuple(int(x) for x in k): float(support.loc[k]) for k in support.index}


patient_support_map = build_group_support_map(train_df, ["patient_id"], label_cols)
spec_support_map = build_group_support_map(train_df, ["spectrogram_id"], label_cols)
joint_support_map = build_group_support_map(
    train_df, ["patient_id", "spectrogram_id"], label_cols
)
eeg_support_map = build_group_support_map(train_df, ["eeg_id"], label_cols)

predictions = None

if len(model_weights) > 0:
    fold_preds = []
    for model_weight in model_weights:
        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        fold_preds.append(prediction_dict["predictions"])
        del model
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.mean(np.stack(fold_preds, axis=0), axis=0).astype(np.float32)
else:
    base_global_support = 5.0  # was 6.0; slightly less global dominance to let strong eeg/patient priors help

    uniform_floor = (np.ones(6, dtype=np.float32) / 6.0).astype(np.float32)
    uniform_eps = 0.0005

    def _w(s: float) -> float:
        s = float(s)
        if not np.isfinite(s) or s <= 0:
            return 0.0
        return float(math.sqrt(s))

    def _shrink_to_global(
        p: np.ndarray, support: float, tau: float = 25.0
    ) -> np.ndarray:
        """
        Support-adaptive shrinkage: p' = (support/(support+tau))*p + (tau/(support+tau))*global.
        tau controls how quickly we trust group priors; keeps semantics (still a prior) and improves KL stability.
        """
        if p is None:
            return None
        s = float(support) if np.isfinite(support) else 0.0
        if s <= 0:
            return global_prior
        lam = s / (s + tau)
        out = lam * p.astype(np.float32) + (1.0 - lam) * global_prior
        out = np.clip(out, 1e-8, None)
        out = out / out.sum()
        return out.astype(np.float32)

    preds_list = []
    for eid, pid, sid in zip(
        test_df["eeg_id"].values,
        test_df["patient_id"].values,
        test_df["spectrogram_id"].values,
    ):
        eid_int = int(eid)
        pid_int = int(pid)
        sid_int = int(sid)
        key_joint = (pid_int, sid_int)

        p_global = global_prior

        spec_sup = spec_support_map.get(sid_int, 0.0)
        pat_sup = patient_support_map.get(pid_int, 0.0)
        joint_sup = joint_support_map.get(key_joint, 0.0)
        eeg_sup = eeg_support_map.get(eid_int, 0.0)

        p_spec = spec_prior_map.get(sid_int, None)
        p_patient = patient_prior_map.get(pid_int, None)
        p_joint = joint_prior_map.get(key_joint, None)
        p_eeg = eeg_prior_map.get(eid_int, None)

        if p_spec is not None:
            p_spec = _shrink_to_global(p_spec, spec_sup, tau=35.0)
        if p_patient is not None:
            p_patient = _shrink_to_global(p_patient, pat_sup, tau=25.0)
        if p_joint is not None:
            p_joint = _shrink_to_global(p_joint, joint_sup, tau=15.0)
        if p_eeg is not None:
            p_eeg = _shrink_to_global(p_eeg, eeg_sup, tau=20.0)

        w_global = base_global_support
        w_spec = (0.65 * _w(spec_sup)) if p_spec is not None else 0.0
        w_patient = _w(pat_sup) if p_patient is not None else 0.0
        w_joint = _w(joint_sup) if p_joint is not None else 0.0
        w_eeg = _w(eeg_sup) if p_eeg is not None else 0.0

        parts: List[Tuple[np.ndarray, float]] = [(p_global, w_global)]
        if p_spec is not None and w_spec > 0:
            parts.append((p_spec, w_spec))
        if p_patient is not None and w_patient > 0:
            parts.append((p_patient, w_patient))
        if p_joint is not None and w_joint > 0:
            parts.append((p_joint, w_joint))
        if p_eeg is not None and w_eeg > 0:
            parts.append((p_eeg, w_eeg))

        wsum = float(sum(w for _, w in parts))
        if wsum <= 0:
            p = p_global.copy()
        else:
            p = np.zeros(6, dtype=np.float32)
            for vec, w in parts:
                p += (w / wsum) * vec.astype(np.float32)

        p = (1.0 - uniform_eps) * p + uniform_eps * uniform_floor
        p = np.clip(p, 1e-8, None)
        p = p / p.sum()
        preds_list.append(p.astype(np.float32))

    predictions = np.stack(preds_list, axis=0).astype(np.float32)

predictions = np.asarray(predictions, dtype=np.float32)
assert predictions.shape == (
    len(test_df),
    6,
), f"Bad predictions shape: {predictions.shape}"
predictions = np.clip(predictions, 1e-8, None)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    "Predictions shape:",
    predictions.shape,
    "row-sum (first 3):",
    predictions[:3].sum(axis=1),
)



## === cell 11
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

sub[TARGETS] = sub[TARGETS].astype(np.float32)
sub[TARGETS] = sub[TARGETS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
row_sum = sub[TARGETS].sum(axis=1).values
row_sum = np.where(row_sum == 0, 1.0, row_sum)
sub[TARGETS] = sub[TARGETS].div(row_sum, axis=0)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
