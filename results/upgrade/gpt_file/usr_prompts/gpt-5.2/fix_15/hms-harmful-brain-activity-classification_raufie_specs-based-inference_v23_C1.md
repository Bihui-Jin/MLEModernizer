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

0.5547362836844233

# 6. Current score

1.44089

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the missing model weights issue by falling back to a deterministic, valid-probability baseline when the external `/kaggle/input/hba-efficientnet-weights/` directory is not available, so the notebook always runs end-to-end and produces `submission.csv`. I also fix a couple of logic/runtime issues that would break inference: undefined `label_cols`, incorrect spectrogram slice index usage (`min/max` not present in test), and the accidental overwriting/indentation bug that repeats EEG assignment inside the region loop. Finally, I guarantee the submission has exactly 6 target columns, is aligned to `test_df.eeg_id`, and each row sums to 1 (required by the metric/submission checker). These changes are score-neutral-to-safe given no training is happening here and are strictly to ensure a valid submission is produced.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.40995, lower-is-better) suggests the predictions are too close to uniform/poorly calibrated, likely because the external weights aren’t available and/or because the test spectrogram slice is always taken from the top (r=0), which is a mismatch vs the 10-minute spectrograms’ informative center. I keep the model/inference logic identical, but (1) fix the spectrogram windowing to consistently take the center 300-frame slice at inference, and (2) add a tiny, competition-safe prior smoothing step that blends predictions slightly toward the empirical class prior from `train.csv`, which generally improves KL when predictions are weak. If weights are available, these changes should still be modest and mostly calibration-like; if weights are missing, they should materially improve over pure uniform. The submission format/row-sum constraints remain enforced exactly.'
- What this solution (achieved 1.48697) has done: 'The timeout is dominated by two expensive data pipelines: (1) precomputing EEG-derived mel spectrograms for every test EEG with multiprocessing, and (2) repeatedly reading spectrogram parquet files inside `__getitem__` across multiple DataLoader workers (duplicated IO and cache per worker). To preserve identical model/inference logic while cutting wall time, the script below (a) builds a fast `spectrogram_id -> path` map without scanning the whole directory, (b) replaces the per-recording `librosa.stft` loop with a provably-equivalent `librosa.feature.melspectrogram` call using the same mel basis/FFT/hop/win settings, (c) removes the up-front EEG precompute by computing EEG spectrograms lazily inside the Dataset with an LRU cache (so each needed EEG is computed once per worker process and only for actually iterated items), and (d) reduces DataLoader workers to avoid duplicated parquet IO and cache fragmentation while keeping pinned-memory/nonblocking GPU transfers. These changes keep the same tensors fed to the model (up to negligible floating point differences) and do not alter architecture, loss, training, or evaluation semantics.'
- What this solution (achieved 1.40042) has done: 'You’re currently far worse than the target (KL: 1.48697 vs 0.5547; lower is better), and with missing external weights the pipeline collapses to a weak prior baseline. The smallest score-relevant improvement without changing core model/feature logic is to add a simple internal training step on `train.csv` (same model, same forward, same softmax/KL-style objective) and then run inference on test. To keep this stable and within the time budget, I train for a single short epoch on a small, deterministic subset of training EEGs and freeze the backbone so only the final linear layer learns calibration, which materially improves KL vs pure prior/untrained logits while preserving your architecture and data generation. I keep your existing prior-blend and temperature scaling (calibration), but reduce the prior blend slightly because once we have a minimally trained head, over-blending can hurt KL. The script still writes a valid `submission.csv` with correct columns and rows summing to 1.'
- What this solution (achieved 1.43002) has done: 'Your current KL (1.40042, lower-is-better) is far worse than the target (0.5547), so we should improve score cautiously without changing your core model/feature pipeline. The smallest high-impact fix is to make the internal “head training” use the same per-`eeg_id` aggregated soft labels that the competition expects (since test is one row per `eeg_id`, but train has many overlapping rows per `eeg_id`), reducing label noise and KL mismatch. I keep your exact model, inputs, loss, inference, and prior-blend, but (1) aggregate train targets by `eeg_id`, (2) train on one representative row per `eeg_id` (deterministically), and (3) adjust the internal-training subset size/steps slightly upward to move KL materially toward the target while staying within the 600s budget. Submission writing and probability normalization remain unchanged.'
- What this solution (achieved 1.4298) has done: 'Your current KL (1.43002, lower-is-better) is far worse than the target (0.5547), so we should improve score with minimal, calibration-like changes while preserving your model/feature pipeline and training loop structure. The largest safe gain here is to better match the competition’s “one row per eeg_id” evaluation by aggregating test predictions by `eeg_id` (your current code predicts per `test_df` row, which can duplicate eeg_ids in some setups and hurts alignment/consistency). Next, I make the internal head-training targets align more closely with the evaluation by using an `eeg_id`-level *sum-normalized* vote distribution (not mean-of-row-probabilities), which is a small semantic fix that reduces label noise without changing architecture or loss. Finally, I tune the post-hoc prior blend slightly downward when internal training is used, since it can over-smooth and worsen KL once the head has learned anything.'
- What this solution (achieved 1.41418) has done: 'You’re far above the target KL (1.4298 vs 0.5547; lower is better), and the biggest limiter here is that the “internal head training” is both too short and too weakly regularized to learn meaningful class separation from noisy inputs, so predictions stay close to the prior and score poorly. I keep your exact architecture, feature building, inference, and KL-style objective, but (1) increase internal head-training steps modestly and (2) unfreeze only the last EfficientNet block (still minimal) so the model can adapt slightly to this domain without changing the training approach. Because training produce stronger posteriors, I reduce the prior-blend alpha a bit to avoid over-smoothing, which should move KL downward toward the target. All submission formatting, row alignment, and probability normalization remain exactly enforced.'
- What this solution (achieved 1.41588) has done: 'Your current KL (1.41418, lower-is-better) is far above the target (0.5547), so we should improve performance with the smallest changes that keep your architecture and feature pipeline intact. The biggest score issue is a train/test mismatch: internal training uses *train* spectrogram offsets implicitly (always center), but those offsets exist in `train.csv` and should be applied so the model sees the same “labeled 10s window” it be judged on; we add offset-based windowing for training only. Next, we reduce label noise further by training on an `eeg_id`-level *vote-sum-normalized* target (already done) but using a deterministic representative row per `eeg_id` whose `spectrogram_label_offset_seconds` is closest to the median (more “central”/stable) instead of just `.first()`. Finally, we make the prediction aggregation match the evaluation more closely by aggregating test predictions by `eeg_id` using a simple mean (kept) but slightly lowering prior-blend for the internal-training case since better-aligned training should produce less prior-like outputs.'
- What this solution (achieved 1.41528) has done: 'Your current KL (1.41588, lower-is-better) is far above the target (0.5547), so the smallest safe way to move toward the target without changing the core model/feature logic is to fix two calibration/alignment issues that directly affect the metric. First, during internal training your targets are already `eeg_id`-aggregated, but the model is trained with per-sample KL; adding a tiny label-smoothing floor to the target probabilities reduces overconfident mistakes and typically improves KL without changing the architecture or loss type. Second, your prior blend is applied **before** `eeg_id` aggregation; moving the prior blend to **after** aggregation (and using a slightly stronger alpha only in the internal-training case) better matches the evaluation granularity and tends to reduce KL when predictions are weak/noisy. These are minimal, metric-aligned changes and keep the same data generation, model forward, and inference semantics.'
- What this solution (achieved 1.41173) has done: 'Your current KL (1.41528; lower is better) is far above the target (0.5547), so we should improve score with minimal, metric-aligned calibration changes rather than altering your model/features. The most impactful low-risk fix is to correct the spectrogram offset-to-frame conversion during internal training: the Kaggle spectrograms are ~0.5s per row, so `spectrogram_label_offset_seconds` should map to frames via `*2` (not `/2`), otherwise training crops the wrong time window and learns noise. Next, we make the internal-training label floor a bit smaller and reduce internal-training prior blending (now that the training crop is correct, too much prior hurts KL), while keeping probability normalization and submission alignment exactly the same. No architecture, feature extraction, inference loop, or loss type is changed.'
- What this solution (achieved 1.43863) has done: 'You’re currently much worse than the target KL (1.41173 vs 0.5547; lower is better), so we should make the smallest metric-aligned calibration changes that preserve your model/data pipeline. I keep architecture, feature extraction, and the training/inference loops intact, but adjust only post-processing: (1) tune the prior-blend strength to be stronger when predictions are weak (internal-training case), and (2) choose the softmax temperature automatically from a small fixed grid using an internal validation split at the `eeg_id` level (no extra training, just calibration) to directly reduce KL. These changes are strictly about better probability calibration for KL divergence and keep the submission format/row-sum constraints exactly enforced. The script still runs end-to-end and writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.44089) has done: 'Your current KL (1.43863; lower is better) is far above the target (0.5547), so we need a modest, metric-aligned improvement without changing the architecture or feature pipeline. The biggest low-risk gain is to calibrate the **prior-blend strength `alpha`** on an internal `eeg_id`-level validation split (similar to how you already grid-search temperature), because your predictions are still weak and KL is very sensitive to over/under-smoothing. I keep your model, training loop, and inference unchanged, but add a small grid search over `alpha` (applied **after** `eeg_id` aggregation exactly as in submission) and then use the best `alpha` for the final submission. This should move the score downward toward the target while keeping runtime within budget (uses the already-loaded `train_sub` and reuses precomputed val predictions once).'

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
from typing import Dict, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 2 if multiprocessing.cpu_count() >= 4 else 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    SAMPLE_SUB = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )


model_weights = glob("/kaggle/input/hba-efficientnet-weights/*.pth")
if len(model_weights) == 0 and os.path.exists(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]

print("Discovered model weights:", model_weights)



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


_SR = 200
_N_FFT = 1024
_N_MELS = 128
_FMIN = 0
_FMAX = 20
_WIN_LENGTH = 128
_MEL_BASIS = librosa.filters.mel(
    sr=_SR,
    n_fft=_N_FFT,
    n_mels=_N_MELS,
    fmin=_FMIN,
    fmax=_FMAX,
    htk=False,
    norm="slaney",
).astype(np.float32)


def _mel_from_signal(x: np.ndarray, hop_length: int) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=x,
        sr=_SR,
        n_fft=_N_FFT,
        hop_length=hop_length,
        win_length=_WIN_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=_N_MELS,
        fmin=_FMIN,
        fmax=_FMAX,
        htk=False,
        norm="slaney",
    ).astype(np.float32, copy=False)
    return mel_spec


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

        xs = []
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            xs.append(x.astype(np.float32, copy=False))
            signals.append(x)

        hop_length = len(xs[0]) // 256  # identical to original
        acc = np.zeros((128, 256), dtype=np.float32)

        for x in xs:
            mel_spec = _mel_from_signal(x, hop_length=hop_length)

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec = mel_spec[:, :width]

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
            mel_spec_db = (mel_spec_db + 40) / 40
            acc += mel_spec_db

        img[:, :, k] = acc / 4.0

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
spec_path_by_id: Dict[int, str] = {
    int(sid): os.path.join(paths.TEST_SPECTROGRAMS, f"{int(sid)}.parquet")
    for sid in test_df["spectrogram_id"].unique()
}
missing = [sid for sid, p in spec_path_by_id.items() if not os.path.exists(p)]
if len(missing):
    raise FileNotFoundError(
        f"Missing spectrogram parquet(s) for ids (sample): {missing[:5]}"
    )

if config.VISUALIZE:
    spectrogram_path = next(iter(spec_path_by_id.values()))
    plot_spectrogram(spectrogram_path)



## === cell 5
eeg_path_by_id: Dict[int, str] = {
    int(eid): os.path.join(paths.TEST_EEGS, f"{int(eid)}.parquet")
    for eid in test_df["eeg_id"].unique()
}
missing_eeg = [eid for eid, p in eeg_path_by_id.items() if not os.path.exists(p)]
if len(missing_eeg):
    raise FileNotFoundError(
        f"Missing EEG parquet(s) for ids (sample): {missing_eeg[:5]}"
    )

all_eegs: Dict[int, np.ndarray] = {}  # kept for API compatibility; not prefilled.

if config.VISUALIZE:
    _ = spectrogram_from_eeg(next(iter(eeg_path_by_id.values())), display=True)




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




## === cell 7
class _LRUCache:
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._d: Dict[int, np.ndarray] = {}
        self._order: List[int] = []

    def get(self, k: int) -> Optional[np.ndarray]:
        v = self._d.get(k, None)
        if v is not None:
            try:
                self._order.remove(k)
            except ValueError:
                pass
            self._order.append(k)
        return v

    def put(self, k: int, v: np.ndarray):
        if k in self._d:
            self._d[k] = v
            try:
                self._order.remove(k)
            except ValueError:
                pass
            self._order.append(k)
            return
        self._d[k] = v
        self._order.append(k)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._d.pop(old, None)


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        spec_paths: Dict[int, str] = None,
        eeg_paths: Dict[int, str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode

        self.spectrograms = specs
        self.spec_paths = spec_paths

        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else all_eegs
        self.eeg_paths = eeg_paths

        self._spec_cache = _LRUCache(max_items=256)
        self._eeg_cache = _LRUCache(max_items=256)

    def __len__(self):
        return len(self.df)

    def _load_spec(self, spectrogram_id: int) -> np.ndarray:
        if self.spectrograms is not None:
            return self.spectrograms[spectrogram_id]
        v = self._spec_cache.get(spectrogram_id)
        if v is not None:
            return v
        p = self.spec_paths[spectrogram_id]
        aux = pd.read_parquet(p)
        arr = aux.iloc[:, 1:].values  # identical to original
        self._spec_cache.put(spectrogram_id, arr)
        return arr

    def _load_eeg_spec(self, eeg_id: int) -> np.ndarray:
        v = self.eeg_spectrograms.get(eeg_id, None)
        if v is not None:
            return v
        v = self._eeg_cache.get(eeg_id)
        if v is not None:
            return v
        p = self.eeg_paths[eeg_id]
        eeg_spec = spectrogram_from_eeg(p, display=False)
        self._eeg_cache.put(eeg_id, eeg_spec)
        return eeg_spec

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
        spec = self._load_spec(int(row.spectrogram_id))

        total_t = spec.shape[0]
        window = 300

        if self.mode != "test" and "spectrogram_label_offset_seconds" in row.index:
            center_frame = int(round(float(row.spectrogram_label_offset_seconds) * 2.0))
            r = center_frame - window // 2
            r = max(0, min(r, max(0, total_t - window)))
        else:
            if total_t >= window:
                r = (total_t - window) // 2
            else:
                r = 0  # safety

        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T  # (100,300)

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self._load_eeg_spec(int(row.eeg_id))  # (128,256,4)
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
test_dataset = CustomDataset(
    test_df,
    config,
    mode="test",
    augment=False,
    specs=None,  # lazy load
    eeg_specs=None,  # lazy compute
    spec_paths=spec_path_by_id,
    eeg_paths=eeg_path_by_id,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(config.NUM_WORKERS > 0),
    prefetch_factor=2 if config.NUM_WORKERS > 0 else None,
)
X, y = test_dataset[0]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")




## === cell 9
def inference_function(test_loader, model, device, temperature: float = 1.35):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.inference_mode():
                logits = model(X)
                logits = logits / float(temperature)
                y_preds = torch.softmax(logits, dim=1)
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_df = pd.read_csv(paths.TRAIN_CSV)

votes = train_df[TARGETS].to_numpy(dtype=np.float64)
votes_sum = votes.sum(axis=1, keepdims=True)
votes_sum = np.where(votes_sum == 0, 1.0, votes_sum)
train_probs = (votes / votes_sum).astype(np.float32)

votes_by_eeg = train_df.groupby("eeg_id", sort=False)[TARGETS].sum().astype(np.float64)
votes_by_eeg_sum = (
    votes_by_eeg.sum(axis=1).to_numpy(dtype=np.float64, copy=False).reshape(-1, 1)
)
votes_by_eeg_sum = np.where(votes_by_eeg_sum == 0, 1.0, votes_by_eeg_sum)
probs_by_eeg = (votes_by_eeg.to_numpy(dtype=np.float64) / votes_by_eeg_sum).astype(
    np.float32
)

prior = probs_by_eeg.mean(axis=0).astype(np.float32)
prior = np.clip(prior, 1e-8, 1.0)
prior = prior / prior.sum()
print(
    "Empirical prior (eeg_id-level, sum->normalize):",
    dict(zip(TARGETS, prior.tolist())),
    "sum=",
    float(prior.sum()),
)

existing_weights = [w for w in model_weights if os.path.exists(w)]
use_internal_training = len(existing_weights) == 0
if use_internal_training:
    print(
        "No model weights found. Training a lightweight head on a deterministic eeg_id-level target."
    )

    train_eegs_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    )
    train_specs_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )

    eeg_targets_df = votes_by_eeg.copy()
    eeg_targets_df["eeg_id"] = eeg_targets_df.index.values
    eeg_targets_df = eeg_targets_df.reset_index(drop=True)
    eeg_targets_arr = eeg_targets_df[TARGETS].to_numpy(dtype=np.float64)
    eeg_targets_sum = eeg_targets_arr.sum(axis=1, keepdims=True)
    eeg_targets_sum = np.where(eeg_targets_sum == 0, 1.0, eeg_targets_sum)
    eeg_targets_probs = (eeg_targets_arr / eeg_targets_sum).astype(np.float32)

    _label_floor = 0.0015
    eeg_targets_probs = np.clip(eeg_targets_probs, _label_floor, 1.0)
    eeg_targets_probs = eeg_targets_probs / eeg_targets_probs.sum(axis=1, keepdims=True)

    eeg_targets_df[TARGETS] = eeg_targets_probs

    sort_cols = ["eeg_id", "spectrogram_id", "eeg_sub_id"]
    needed_cols = [
        "eeg_id",
        "spectrogram_id",
        "spectrogram_label_offset_seconds",
        "eeg_sub_id",
    ]
    tmp = train_df[needed_cols].sort_values(sort_cols, kind="mergesort").copy()
    med = tmp.groupby("eeg_id", sort=False)[
        "spectrogram_label_offset_seconds"
    ].transform("median")
    tmp["_abs_dev"] = (tmp["spectrogram_label_offset_seconds"] - med).abs()
    eeg_rep_row = (
        tmp.sort_values(
            ["eeg_id", "_abs_dev", "spectrogram_id", "eeg_sub_id"], kind="mergesort"
        )
        .groupby("eeg_id", sort=False)
        .first()
        .reset_index()[["eeg_id", "spectrogram_id", "spectrogram_label_offset_seconds"]]
    )
    train_eeg_level = eeg_rep_row.merge(eeg_targets_df, on="eeg_id", how="inner")
    train_eeg_level = train_eeg_level.reset_index(drop=True)
    print("eeg_id-level training rows:", len(train_eeg_level))

    rng = np.random.RandomState(config.SEED)
    uniq_eeg = train_eeg_level["eeg_id"].to_numpy()
    rng.shuffle(uniq_eeg)

    max_train_eegs = 1408
    keep_eeg = set(uniq_eeg[:max_train_eegs])
    train_sub = train_eeg_level[train_eeg_level["eeg_id"].isin(keep_eeg)].copy()
    train_sub = train_sub.reset_index(drop=True)

    train_spec_path_by_id: Dict[int, str] = {
        int(sid): os.path.join(train_specs_dir, f"{int(sid)}.parquet")
        for sid in train_sub["spectrogram_id"].unique()
    }
    train_eeg_path_by_id: Dict[int, str] = {
        int(eid): os.path.join(train_eegs_dir, f"{int(eid)}.parquet")
        for eid in train_sub["eeg_id"].unique()
    }

    spec_ok = train_sub["spectrogram_id"].map(
        lambda x: os.path.exists(train_spec_path_by_id[int(x)])
    )
    eeg_ok = train_sub["eeg_id"].map(
        lambda x: os.path.exists(train_eeg_path_by_id[int(x)])
    )
    train_sub = train_sub[spec_ok & eeg_ok].reset_index(drop=True)
    if len(train_sub) == 0:
        raise RuntimeError("Training subset is empty after file checks unexpectedly.")
    print("Training eeg_id rows after file checks:", len(train_sub))

    train_dataset = CustomDataset(
        train_sub,
        config,
        mode="train",
        augment=False,
        specs=None,
        eeg_specs=None,
        spec_paths=train_spec_path_by_id,
        eeg_paths=train_eeg_path_by_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=max(0, config.NUM_WORKERS),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(max(0, config.NUM_WORKERS) > 0),
        prefetch_factor=2 if max(0, config.NUM_WORKERS) > 0 else None,
    )

    model = CustomModel(config)
    model.to(device)

    for p in model.features.parameters():
        p.requires_grad = False
    try:
        for p in model.features[-1].parameters():
            p.requires_grad = True
        print("Unfroze last feature stage: model.features[-1]")
    except Exception as e:
        print(
            "Could not unfreeze model.features[-1]; keeping full backbone frozen. Reason:",
            repr(e),
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1.2e-3, weight_decay=1e-3)

    def kl_loss_from_logits(
        logits: torch.Tensor, target_probs: torch.Tensor
    ) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=1)
        t = torch.clamp(target_probs, 1e-8, 1.0)
        t = t / t.sum(dim=1, keepdim=True)
        return torch.mean(torch.sum(t * (torch.log(t) - log_p), dim=1))

    model.train()
    n_steps = 0
    max_steps = 520
    pbar = tqdm(train_loader, desc="Head training", unit="batch")
    for Xb, yb in pbar:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(Xb)
        loss = kl_loss_from_logits(logits, yb)
        loss.backward()
        optimizer.step()
        n_steps += 1
        if n_steps % 10 == 0:
            pbar.set_postfix({"loss": float(loss.detach().cpu().item())})
        if n_steps >= max_steps:
            break

    predictions = inference_function(test_loader, model, device, temperature=1.35)[
        "predictions"
    ]

    del train_dataset, train_loader
    torch.cuda.empty_cache()
    gc.collect()
else:
    predictions_list = []
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

        pred_dict = inference_function(test_loader, model, device, temperature=1.35)
        predictions_list.append(pred_dict["predictions"])

        del model, checkpoint, state
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)

predictions = np.asarray(predictions, dtype=np.float32)
if predictions.shape != (len(test_df), len(TARGETS)):
    raise ValueError(
        f"Predictions shape {predictions.shape} does not match expected {(len(test_df), len(TARGETS))}"
    )

print(
    "Predictions ready (pre-aggregation, no prior blend yet):",
    predictions.shape,
    "row-sum min/max:",
    float(predictions.sum(1).min()),
    float(predictions.sum(1).max()),
)




## === cell 11
def _kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    yt = np.clip(y_true, eps, 1.0)
    yt = yt / yt.sum(axis=1, keepdims=True)
    yp = np.clip(y_pred, eps, 1.0)
    yp = yp / yp.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(yt * (np.log(yt) - np.log(yp)), axis=1)))


def _calibrate_temperature_on_val(
    model: nn.Module,
    base_df: pd.DataFrame,
    spec_paths: Dict[int, str],
    eeg_paths: Dict[int, str],
    device,
    prior: np.ndarray,
    seed: int,
) -> float:
    uniq = base_df["eeg_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(128, int(round(0.15 * len(uniq))))
    val_ids = set(uniq[:n_val])
    val_df = base_df[base_df["eeg_id"].isin(val_ids)].reset_index(drop=True)
    if len(val_df) < 64:
        return 1.35

    val_loader = DataLoader(
        CustomDataset(
            val_df,
            config,
            mode="train",  # to apply training offset windowing for a consistent label window
            augment=False,
            specs=None,
            eeg_specs=None,
            spec_paths=spec_paths,
            eeg_paths=eeg_paths,
        ),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # avoid extra IO duplication; small val anyway
        pin_memory=True,
        drop_last=False,
    )

    temps = [1.0, 1.15, 1.35, 1.6, 1.9]
    best_t = 1.35
    best_kl = 1e9

    model.eval()
    with torch.inference_mode():
        for t in temps:
            all_p = []
            all_y = []
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                logits = model(Xb) / float(t)
                pb = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_p.append(pb)
                all_y.append(yb.numpy())
            p = np.concatenate(all_p, axis=0).astype(np.float32)
            y = np.concatenate(all_y, axis=0).astype(np.float32)

            dfp = pd.DataFrame(p, columns=TARGETS)
            dfp["eeg_id"] = val_df["eeg_id"].values
            agg_p = (
                dfp.groupby("eeg_id", sort=False)[TARGETS]
                .mean()
                .to_numpy(dtype=np.float32)
            )

            dfy = pd.DataFrame(y, columns=TARGETS)
            dfy["eeg_id"] = val_df["eeg_id"].values
            agg_y = (
                dfy.groupby("eeg_id", sort=False)[TARGETS]
                .mean()
                .to_numpy(dtype=np.float32)
            )

            k = _kl_divergence(agg_y, agg_p)
            if k < best_kl:
                best_kl = k
                best_t = t

    print(f"Chose temperature={best_t} from grid {temps} using val KL={best_kl:.6f}")
    return float(best_t)


def _choose_alpha_on_val_from_cached_preds(
    val_eeg_ids: np.ndarray,
    val_pred_probs: np.ndarray,
    val_true_probs: np.ndarray,
    prior: np.ndarray,
    use_internal_training: bool,
) -> float:
    """
    Change is score-relevant for KL: tune post-aggregation prior blending alpha on a
    held-out eeg_id validation split to reduce KL when predictions are weak/noisy.
    This preserves core model/inference logic and only adjusts final probability calibration.
    """
    dfp = pd.DataFrame(val_pred_probs, columns=TARGETS)
    dfp["eeg_id"] = val_eeg_ids
    agg_p = dfp.groupby("eeg_id", sort=False)[TARGETS].mean().to_numpy(dtype=np.float32)

    dfy = pd.DataFrame(val_true_probs, columns=TARGETS)
    dfy["eeg_id"] = val_eeg_ids
    agg_y = dfy.groupby("eeg_id", sort=False)[TARGETS].mean().to_numpy(dtype=np.float32)

    if use_internal_training:
        alphas = [0.00, 0.04, 0.08, 0.10, 0.12, 0.16, 0.20]
        default_alpha = 0.10
    else:
        alphas = [0.06, 0.12, 0.18, 0.22, 0.26, 0.30]
        default_alpha = 0.22

    best_a = default_alpha
    best_kl = 1e9
    for a in alphas:
        p = np.clip(agg_p, 1e-8, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        p = (1.0 - a) * p + a * prior[None, :]
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        k = _kl_divergence(agg_y, p)
        if k < best_kl:
            best_kl = k
            best_a = a

    print(
        f"Chose prior_blend_alpha={best_a} from grid {alphas} using val KL={best_kl:.6f}"
    )
    return float(best_a)


chosen_temperature = 1.35
chosen_alpha = 0.10 if use_internal_training else 0.22

if use_internal_training:
    try:
        chosen_temperature = _calibrate_temperature_on_val(
            model=model,
            base_df=train_sub,
            spec_paths=train_spec_path_by_id,
            eeg_paths=train_eeg_path_by_id,
            device=device,
            prior=prior,
            seed=config.SEED,
        )
    except Exception as e:
        print("Temperature calibration failed; falling back to 1.35. Reason:", repr(e))
        chosen_temperature = 1.35

    predictions = inference_function(
        test_loader, model, device, temperature=chosen_temperature
    )["predictions"]
    predictions = np.asarray(predictions, dtype=np.float32)

    try:
        uniq = train_sub["eeg_id"].unique()
        rng = np.random.RandomState(config.SEED)
        rng.shuffle(uniq)
        n_val = max(128, int(round(0.15 * len(uniq))))
        val_ids = set(uniq[:n_val])
        val_df = train_sub[train_sub["eeg_id"].isin(val_ids)].reset_index(drop=True)

        if len(val_df) >= 64:
            val_loader = DataLoader(
                CustomDataset(
                    val_df,
                    config,
                    mode="train",
                    augment=False,
                    specs=None,
                    eeg_specs=None,
                    spec_paths=train_spec_path_by_id,
                    eeg_paths=train_eeg_path_by_id,
                ),
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
            model.eval()
            all_p = []
            all_y = []
            with torch.inference_mode():
                for Xb, yb in tqdm(
                    val_loader, desc="Val preds (for alpha)", unit="batch"
                ):
                    Xb = Xb.to(device, non_blocking=True)
                    logits = model(Xb) / float(chosen_temperature)
                    pb = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    all_p.append(pb)
                    all_y.append(yb.numpy())
            val_p = np.concatenate(all_p, axis=0).astype(np.float32)
            val_y = np.concatenate(all_y, axis=0).astype(np.float32)
            val_eids = val_df["eeg_id"].to_numpy()

            chosen_alpha = _choose_alpha_on_val_from_cached_preds(
                val_eeg_ids=val_eids,
                val_pred_probs=val_p,
                val_true_probs=val_y,
                prior=prior,
                use_internal_training=True,
            )
        else:
            chosen_alpha = 0.10
    except Exception as e:
        print("Alpha calibration failed; using default. Reason:", repr(e))
        chosen_alpha = 0.10



## === cell 12
pred_df = pd.DataFrame(predictions, columns=TARGETS)
pred_df["eeg_id"] = test_df["eeg_id"].values
pred_agg = pred_df.groupby("eeg_id", sort=False)[TARGETS].mean().reset_index()

sample_sub = pd.read_csv(paths.SAMPLE_SUB)
sub = sample_sub[["eeg_id"]].merge(pred_agg, on="eeg_id", how="left")

miss_mask = sub[TARGETS].isna().any(axis=1)
if miss_mask.any():
    sub.loc[miss_mask, TARGETS] = prior[None, :]

alpha = float(chosen_alpha) if use_internal_training else 0.22

vals = sub[TARGETS].to_numpy(dtype=np.float32)
vals = np.clip(vals, 1e-8, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
vals = (1.0 - alpha) * vals + alpha * prior[None, :]

vals = np.clip(vals, 1e-8, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print("Used temperature:", chosen_temperature, "prior_blend_alpha:", alpha)
sub.head()
