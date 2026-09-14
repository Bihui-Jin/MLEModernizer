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

0.5506465570004035

# 6. Current score

1.40452

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the smallest fixes needed to (1) ensure the notebook runs end-to-end in your environment and (2) produce a valid `submission.csv`. The key issues preventing a score are: missing cell numbers (Kaggle executes top-to-bottom), an external weights path that likely doesn’t exist, a `label_cols` NameError, and a couple of correctness/performance bugs in dataset generation (EEG assignment inside the wrong loop and spectrogram time indexing using nonexistent `min/max` columns for test). I keep your model and inference logic the same, but add a safe fallback to uniform probabilities if weights are unavailable so you always get a valid submission, and I fix dataset indexing/assembly so predictions are well-formed and more likely to improve KL versus a broken pipeline.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.55065), so we should improve the KL by making predictions less overconfident and better calibrated without changing your model or data pipeline. The smallest high-impact change here is to apply a mild temperature scaling to the model logits before softmax (this preserves the exact model and inference flow, only adjusts calibration). I also keep your ensembling intact, but compute the ensemble in logit-space (average logits then softmax) which is typically better for KL than averaging already-softmaxed probabilities, while staying within the same core logic. Finally, I keep the strict probability normalization/clipping to guarantee valid submissions.'
- What this solution (achieved 1.40995) has done: 'We should move the KL down (lower-is-better) toward 0.5506 from 1.40995, so the minimal high-impact fix is to correct a small but important preprocessing mismatch: your EEG-derived spectrogram channels are not normalized like the Kaggle spectrogram channels, which typically hurts calibration and KL. I keep your model, weights, and inference unchanged, but add the same log-clip/log/standardize pipeline to the EEG-derived spectrograms before they are inserted into `X[:, :, 4:]`. I also make the softmax numerically stable in float32 and keep the probability clipping/renormalization to guarantee valid submissions.'
- What this solution (achieved 1.40995) has done: 'We should move KL down (lower-is-better) from 1.40995 toward 0.55065, so the most likely minimal win is to fix a preprocessing mismatch: your EEG-derived spectrograms are currently in a different scale than the Kaggle spectrogram channels (you log-normalize them later, but from an already dB-normalized input), which can make the model effectively see out-of-distribution inputs and output poorly-calibrated probabilities. I adjust `spectrogram_from_eeg()` to output “power-like” (positive) mel spectrogram values (not dB-normalized), and keep the exact same log/standardize pipeline already used in `__data_generation` for both Kaggle spectrograms and EEG spectrograms. I keep your model, weights loading, logit-temperature, and logit-space ensembling unchanged. This is a small, targeted change that should improve calibration and reduce KL without altering the core training/inference approach.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down (lower-is-better) from 1.40995 toward 0.55065, so the smallest likely gain without changing your architecture/training is to improve probability calibration and reduce overconfident mistakes. I (1) apply a small, safe label-smoothing-style blending of the final probabilities with a uniform prior, which tends to reduce KL when predictions are miscalibrated, and (2) slightly increase the logit temperature to further soften probabilities. These are post-processing/inference-only changes, preserving your core model/data pipeline and still producing a valid submission with rows summing to 1. Everything else is kept the same to minimize risk and runtime.'
- What this solution (achieved 1.40995) has done: 'Your KL (lower-is-better) is still far above target, so the smallest, safest way to improve toward 0.5506 without changing your model or data pipeline is to (1) reduce overconfidence further via slightly stronger calibration at inference time. I do this by modestly increasing the logit temperature and slightly increasing the uniform-probability blend, both of which typically reduce KL when predictions are miscalibrated. I also add a tiny epsilon to the per-image standardization denominator for the Kaggle spectrogram slices (mirroring the EEG branch) to avoid rare NaN/Inf spikes that can badly hurt KL. Everything else (architecture, feature construction, ensemble method, submission format) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is still far above the target (0.55065), so we should make predictions less overconfident and more robustly calibrated with minimal, inference-only changes. I keep your model, dataset construction, and ensembling exactly the same, but (1) soften probabilities a bit more by slightly increasing the logit temperature and (2) blend a bit more with a uniform prior, which typically reduces KL when predictions are miscalibrated. I also ensure the Kaggle-spectrogram standardization cannot ever produce extreme values (by matching the small epsilon/NaN guards already used elsewhere), avoiding rare spikes that can disproportionately hurt KL. These are small, safe adjustments that preserve semantics and should move the score downward toward the target.'
- What this solution (achieved 1.40806) has done: 'Your KL (lower-is-better) is far above the target, so we should improve calibration and reduce overconfident errors with minimal, inference-only changes that don’t alter your model or feature pipeline. I (1) add a tiny amount of logit centering/variance normalization per-sample before softmax (a standard calibration trick that preserves ranking but reduces extreme confidence), (2) slightly increase the softening strength by a small step (temperature + uniform blend) to move KL downward, and (3) add a very small “prior” based on the train label distribution (blended at low weight) to further reduce KL when the model is uncertain. These are small post-processing steps and keep the architecture, data generation, and ensembling logic intact while making the output probabilities more robust.'
- What this solution (achieved 1.40781) has done: 'Your current KL (lower-is-better) is still far above the target, so we should make the output probabilities less overconfident in a controlled, inference-only way without touching your model, feature construction, or training loops. The smallest reliable lever for KL in this competition is calibration/post-processing, so I slightly strengthen the existing softening: increase the logit temperature a bit and increase the uniform+prior blending a bit to reduce extreme probabilities. I also add a tiny extra probability floor before renormalization to avoid any near-zero entries that can spike KL when the true class has non-trivial mass. Everything else (data reading, spectrogram construction, architecture, ensembling, submission schema) stays identical, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.4069) has done: 'Your KL is still far above the target (lower is better), so the most “minimal-but-high-impact” lever left without touching architecture/training is inference calibration. I keep your entire pipeline intact, but adjust three inference-only knobs that directly affect KL: (1) slightly increase the logit temperature to soften overconfident outputs, (2) slightly increase blending with the empirical class prior (more grounded than uniform), and (3) slightly reduce the uniform blend to avoid over-flattening when the prior already stabilizes probabilities. These are tiny parameter changes (no new logic), should reduce extreme probabilities, and are the safest way to move KL downward toward ~0.55.'
- What this solution (achieved 1.40639) has done: 'Your current KL (1.4069, lower-is-better) is still far above the target (0.5506), so we should make a small inference-only calibration change that tends to reduce KL without touching your architecture or feature pipeline. The most direct knob is to slightly strengthen probability “softening” by increasing the uniform/prior blending a bit while keeping your logit-temperature and logit-space ensembling unchanged. To avoid rare KL spikes from tiny probabilities, we also raise the probability floor epsilon slightly (then renormalize), which is metric-aligned and preserves submission validity. Everything else (data loading, spectrogram extraction, model, inference loop, submission format) stays the same.'
- What this solution (achieved 1.40618) has done: 'Your current KL (1.40639, lower-is-better) is still far above the target (0.55065), so we should make a minimal, inference-only calibration adjustment that reliably reduces KL by avoiding overconfident mistakes. I keep your model, data loading, spectrogram construction, and ensembling unchanged, but slightly increase the softening via (1) a small bump in logit temperature and (2) a small increase in prior/uniform blending. I also make the logits-to-probability epsilon match your final probability floor to prevent any “tiny-probability” spikes before blending/renorm that can hurt KL. These are tiny parameter-level changes and should move the score downward toward the target without changing core logic.'
- What this solution (achieved 1.40558) has done: 'To move your KL score downward toward the target (lower is better), the most reliable minimal lever left (without touching the model/data pipeline) is inference-time calibration. I slightly strengthen the existing probability “softening” by a small increase in logit temperature and a small increase in prior blending (and a tiny decrease in uniform blending to avoid over-flattening), which usually reduces KL when the model is miscalibrated. I keep your logit-space ensembling, preprocessing, and submission formatting identical, and still guarantee probabilities are clipped and renormalized to sum to 1. These are parameter-only changes, so runtime and core logic remain the same.'
- What this solution (achieved 1.405) has done: 'Your KL is still much worse than the target (lower is better), so we should make the smallest inference-only calibration change that reliably reduces overconfidence without touching your model, data construction, or training. The most conservative lever here is to slightly increase probability softening by (1) raising the logit temperature a bit and (2) increasing the class-prior blend a bit (while slightly decreasing the uniform blend to avoid over-flattening). This keeps your logit-space ensembling, preprocessing, and probability normalization identical, but nudges outputs toward better-calibrated distributions for KL. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.40452) has done: 'Your current KL (1.405, lower-is-better) is still far above the target (0.5506), so we should move it downward by making predictions less overconfident in the smallest inference-only way. I keep your model, data loading, feature construction, and logit-space ensembling unchanged, and only adjust the post-logit calibration knobs that directly affect KL. Specifically, I slightly increase the softmax temperature and slightly increase the prior/uniform blending while keeping your normalization/clipping logic intact, which typically reduces KL when the model is miscalibrated. This is a minimal, low-risk change that still produces a valid `submission.csv` with rows summing to 1.'

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

from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # keep 0 for Kaggle stability
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False

    TTA_LOGIT_TEMPERATURE = 4.15  # was 3.90

    PROBA_UNIFORM_BLEND = 0.20  # was 0.22 (slightly down to avoid over-flattening when prior is increased)
    PROBA_PRIOR_BLEND = 0.33  # was 0.28

    LOGIT_NORM_EPS = 1e-6

    PROBA_FLOOR_EPS = 8e-6


class paths:
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


weight_globs = [
    "/kaggle/input/hba-efficientnet-weights/*.pth",
    "/kaggle/input/*/*.pth",
]
model_weights = []
for g in weight_globs:
    model_weights.extend(glob(g))
model_weights = sorted(list(set(model_weights)))
print(f"Discovered {len(model_weights)} model weight file(s).")
model_weights[:5]



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
label_cols = TARGETS  # fix NameError in dataset when mode != 'test'


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
                power=2.0,
            ).astype(np.float32)

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec = mel_spec[:, :width]

            mel_spec = np.clip(mel_spec, 1e-10, None)

            if mel_spec.shape[1] != 256:
                if mel_spec.shape[1] > 256:
                    mel_spec = mel_spec[:, :256]
                else:
                    pad = 256 - mel_spec.shape[1]
                    mel_spec = np.pad(mel_spec, ((0, 0), (0, pad)), mode="edge")

            img[:, :, k] += mel_spec

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(
                np.log(np.clip(img[:, :, k], 1e-10, None)),
                aspect="auto",
                origin="lower",
            )
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(config.SEED)



## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
train_df_for_prior = pd.read_csv(paths.TRAIN_CSV, usecols=TARGETS)
train_votes = train_df_for_prior[TARGETS].sum(axis=0).values.astype(np.float64)
prior = train_votes / np.clip(train_votes.sum(), 1.0, None)
prior = prior.astype(np.float32)
print("Class prior (from train votes):", prior, "sum:", prior.sum())

del train_df_for_prior
gc.collect()



## === cell 5
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquets")
all_spectrograms: Dict[int, np.ndarray] = {}

for file_path in tqdm(paths_spectrograms):
    aux = pd.read_parquet(file_path)
    name = int(os.path.basename(file_path).split(".")[0])
    all_spectrograms[name] = aux.iloc[:, 1:].values
    del aux

print("Loaded spectrograms:", len(all_spectrograms))



## === cell 6
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquet files")
all_eegs: Dict[int, np.ndarray] = {}
counter = 0

for file_path in tqdm(paths_eegs):
    eeg_id = int(os.path.basename(file_path).split(".")[0])
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1 and config.VISUALIZE)
    all_eegs[eeg_id] = eeg_spectrogram
    counter += 1

print("Loaded EEG-derived spectrograms:", len(all_eegs))




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
        r = max(0, (spec.shape[0] - 300) // 2)

        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)].astype(np.float32)

        eeg_img = np.clip(eeg_img, np.exp(-4), np.exp(8))
        eeg_img = np.log(eeg_img)

        ep = 1e-6
        mu = np.nanmean(eeg_img, axis=(0, 1), keepdims=True)
        std = np.nanstd(eeg_img, axis=(0, 1), keepdims=True)
        eeg_img = (eeg_img - mu) / (std + ep)
        eeg_img = np.nan_to_num(eeg_img, nan=0.0, posinf=0.0, neginf=0.0)

        X[:, :, 4:] = eeg_img / 2.0

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 9
test_dataset = CustomDataset(test_df, config, mode="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X0, y0 = test_dataset[0]
print(f"X shape: {X0.shape}")
print(f"y shape: {y0.shape}")




## === cell 10
def inference_function(
    test_loader, model, device, temperature: float = 1.0, return_logits: bool = True
):
    model.eval()
    outs = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(X)
                if temperature is not None and float(temperature) != 1.0:
                    logits = logits / float(temperature)
            if return_logits:
                outs.append(logits.detach().cpu().numpy())
            else:
                outs.append(nn.Softmax(dim=1)(logits).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)




## === cell 11
def logits_to_proba(logits: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    l = logits.astype(np.float32)
    l = l - np.mean(l, axis=1, keepdims=True)
    s = np.std(l, axis=1, keepdims=True)
    l = l / (s + float(config.LOGIT_NORM_EPS))

    l = l - np.max(l, axis=1, keepdims=True)
    exp = np.exp(l)
    p = exp / np.sum(exp, axis=1, keepdims=True)

    p = np.clip(p, float(config.PROBA_FLOOR_EPS), 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p




## === cell 12
all_model_logits = []

if len(model_weights) == 0:
    print("No model weights found; using uniform probabilities fallback.")
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
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

        logits = inference_function(
            test_loader,
            model,
            device,
            temperature=config.TTA_LOGIT_TEMPERATURE,
            return_logits=True,
        )
        all_model_logits.append(logits)

        del model, checkpoint, state_dict, logits
        torch.cuda.empty_cache()
        gc.collect()

    avg_logits = np.mean(np.stack(all_model_logits, axis=0), axis=0).astype(np.float32)
    predictions = logits_to_proba(avg_logits, eps=float(config.PROBA_FLOOR_EPS))

beta = float(config.PROBA_PRIOR_BLEND)
if beta > 0:
    prior_row = np.broadcast_to(prior.reshape(1, -1), predictions.shape).astype(
        np.float32
    )
    predictions = (1.0 - beta) * predictions + beta * prior_row

alpha = float(config.PROBA_UNIFORM_BLEND)
if alpha > 0:
    uniform = np.full_like(predictions, 1.0 / predictions.shape[1], dtype=np.float32)
    predictions = (1.0 - alpha) * predictions + alpha * uniform

predictions = np.clip(predictions, float(config.PROBA_FLOOR_EPS), 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    "Predictions shape:",
    predictions.shape,
    "Row sums (min/max):",
    predictions.sum(1).min(),
    predictions.sum(1).max(),
)



## === cell 13
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")
print(f"Submission shape: {sub.shape}")
sub.head()
