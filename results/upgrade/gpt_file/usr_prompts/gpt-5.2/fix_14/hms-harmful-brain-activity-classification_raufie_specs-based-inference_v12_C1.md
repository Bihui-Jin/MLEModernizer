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

0.5681837655146406

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the inference crash by making the weight discovery robust to both `.pth` and `.pt` files and by using the already-declared `MODEL_WEIGHTS` path as a fallback. If no weights are available in the environment, I still produce a valid `submission.csv` by outputting a safe uniform probability distribution (so you can at least submit and get a score rather than failing). I also prevent the downstream `AxisError` by ensuring `predictions` is always a 2D `(n_samples, 6)` array before normalization. Finally, I keep the model/data logic intact and only touch the parts needed to run end-to-end and write a correct submission.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.56818), so we should improve predictions while keeping the same model and pipeline. The biggest likely issue is that your test DataFrame is unsorted while you load spectrogram/eeg dicts by ID, so `__getitem__` depends on matching IDs and can silently misalign if types differ; we enforce integer IDs and stable ordering. Next, we make the softmax numerically safer and ensure the final probabilities are properly normalized with a slightly larger clip epsilon (still negligible to semantics) to avoid KL blow-ups from tiny values. Finally, we reduce accidental device/CPU overhead and memory spikes by using non_blocking transfers and `inference_mode()` (no semantic change), which helps keep inference stable within Kaggle runtime.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.56818), so we make small, low-risk fixes that preserve your model/pipeline but remove two big KL-divergence pitfalls: (1) mis-normalized targets at training time (you were effectively predicting “vote-count proportions” but never ensuring they sum to 1 in the data) and (2) overly-confident probabilities (KL punishes near-zero predictions heavily). Concretely, we (a) run a quick out-of-fold temperature calibration on the already-produced test logits/probabilities using the provided train labels (no architecture/training changes) and (b) apply a tiny Dirichlet-style smoothing before final normalization to avoid extreme probabilities. These are post-processing steps aligned to the KL metric and should materially reduce the score without changing the core model inference. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'We keep your model/inference core intact and focus on two KL-sensitive fixes that are very likely driving the high (bad) score: (1) your “temperature calibration” is currently invalid because it tries to fit train labels against *test* predictions (mismatched distributions/IDs), and (2) your dataset is accidentally overwriting the EEG channels inside the per-region loop, which is harmless but inefficient and can create subtle inconsistencies. Concretely, we remove the train-based temperature fitting step (and instead apply a safe, small temperature > 1 to reduce overconfidence) and move EEG assignment outside the region loop (same data, same semantics, less risk). We also add a tiny per-row normalization safety check at the end (already mostly done) to ensure KL never blows up from numerical drift. These are minimal changes aimed at reducing your KL from 1.40995 toward the 0.568 target without changing architecture, training, or feature extraction.'
- What this solution (achieved 1.40995) has done: 'We keep your model, feature extraction, and inference loop intact, but fix a likely major score driver: the spectrogram crop index `r` is always `0` for test, which can misalign the 10‑minute spectrogram context and inflate KL. We compute `r` for test using the same train metadata aggregation per `spectrogram_id` (mean center from `spectrogram_label_offset_seconds`), without changing any training. We also ensure `spectrogram_id`/`eeg_id` dtypes match across train/test and add a safe fallback to `r=0` only when a spectrogram id is missing in the mapping. This should materially lower (improve) the KL score from 1.40995 toward the 0.568 target while keeping the pipeline behavior consistent and producing a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is much worse than the target (0.56818), so we should improve without changing the model/feature pipeline. The biggest low-risk improvement for this metric is to reduce overconfidence and avoid tiny probabilities: we increase the fixed temperature slightly and increase the Dirichlet-style smoothing alpha a bit, which typically lowers KL when predictions are too sharp. We also ensure test `eeg_id` order exactly matches `sample_submission.csv` order (some competitions score strictly by row alignment), while still using the same predictions per eeg_id. Finally, we keep all inference logic identical otherwise and still write a valid `submission.csv` with row sums exactly 1.'
- What this solution (achieved 1.40995) has done: 'I fix the crash in the test dataframe alignment step by removing the invalid one-to-one merge validation and instead reindexing `test.csv` to exactly match `sample_submission.csv` order while handling any duplicate `eeg_id` safely. This keeps your inference logic unchanged but ensures predictions line up with the required submission row order (a common hidden score-killer). I also add a small guard to ensure `test_df` is unique per `eeg_id` before dataset creation, which prevents accidental duplication/misalignment without changing the model. These changes are score-improving toward your target because they correct row mapping errors that can inflate KL dramatically.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far worse than the target (0.56818), so we should reduce KL with minimal risk while keeping the same model/inference pipeline. The biggest low-risk KL improvement here is to reduce overconfidence more aggressively and add slightly stronger probability smoothing, since KL heavily penalizes near-zero probabilities on true classes. I keep your ensemble/inference identical, but adjust only post-processing: increase the fixed temperature and smoothing alpha, and use a slightly larger epsilon in clipping to prevent extreme probabilities. I also ensure the submission is aligned exactly to `sample_submission.csv` order (already mostly true) and keep row-sum normalization strict.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far worse than the target (0.56818), so the most likely issue is not the model but a data/feature misalignment that makes predictions effectively wrong per row. I make two minimal, high-impact fixes while preserving your architecture and inference loop: (1) stop shifting spectrogram windows using a train-derived `r_map` (test spectrograms are already centered), and always use the centered crop for test; (2) fix the spectrogram crop height to 256 (not 300) so it exactly matches your `(128,256,8)` tensor layout (this mismatch can distort inputs). I keep your temperature+smoothing post-processing, but tune it slightly less aggressive (closer to typical KL-friendly calibration) to improve toward the target without changing the model.'
- What this solution (achieved 1.40995) has done: 'I fix the dataset broadcast error by making the spectrogram crop width consistent with the model’s expected `(128,256,8)` layout (your current `22:-22` slice produces width 56, which cannot fill 256). I do this with a minimal, score-neutral change: compute the correct center crop dynamically from the available spectrogram width and place it into the 256-wide canvas with padding if needed. I also make `__getitem__` robust to rare missing `eeg_id`/`spectrogram_id` keys (so inference doesn’t crash) while keeping your model, inference loop, and post-processing intact. The rest of the pipeline remains unchanged and write a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.56818), so we should make small, low-risk fixes that reduce misalignment and KL blow-ups without changing the model or feature extraction. The biggest likely score killer is that your test-time spectrogram crop always uses `r=0`, while your model was trained on a centered 256-row crop; we compute a per-`spectrogram_id` center-row using the test spectrogram parquet itself (no train leakage, same semantics: center crop). Next, we ensure the spectrogram array we store preserves the true time axis by not blindly dropping the first column (some parquets include `time`, others don’t), which can distort inputs. Finally, we keep your existing temperature+smoothing but make it slightly less aggressive (still conservative) to avoid washing out useful signal while still preventing overconfidence that hurts KL.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far above the target (0.56818), so we should improve without changing the model or feature pipeline. The most likely high-impact, low-risk fix is correcting the spectrogram axis handling: the parquet is typically shaped as (time, freq), but the current code crops rows as if they were frequency, causing a severe train/test feature mismatch and bad KL. I minimally change the dataset to crop along the time axis (columns) with a centered 256-window and keep frequency on the vertical axis, while preserving the same normalization and the same `(128,256,8)` tensor layout. I also compute the test crop start `r` from the spectrogram time length (width) instead of height, and keep your existing temperature + smoothing post-processing unchanged.'

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
    MODEL = "tf_efficientnet_b4"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = "/kaggle/input/hba-efficientnet-weights/efficient_net_weights.pt"
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


model_weights = sorted(glob("/kaggle/input/hba-efficientnet-weights/*.pth")) + sorted(
    glob("/kaggle/input/hba-efficientnet-weights/*.pt")
)
if os.path.exists(paths.MODEL_WEIGHTS) and paths.MODEL_WEIGHTS not in model_weights:
    model_weights.append(paths.MODEL_WEIGHTS)

print("Found weight files:", len(model_weights))
for p in model_weights[:5]:
    print("  ", p)



## === cell 2
model_weights



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
            plt.title(f"EEG Spectrogram {NAMES[k]}")

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
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
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

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

seed_everything(config.SEED)




## === cell 4
def build_test_r_map(test_df: pd.DataFrame, test_spec_dir: str) -> Dict[int, int]:
    r_map_local: Dict[int, int] = {}
    uniq = test_df["spectrogram_id"].dropna().astype(np.int64).unique()
    for sid in tqdm(uniq, desc="Building test r_map"):
        fp = os.path.join(test_spec_dir, f"{int(sid)}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            aux = pd.read_parquet(fp)
            if aux.shape[1] > 0 and str(aux.columns[0]).lower() in (
                "time",
                "t",
                "seconds",
            ):
                mat = aux.iloc[:, 1:].values
            else:
                mat = aux.values

            time_len = mat.shape[0]
            r0 = max(0, (time_len - 256) // 2)
            r_map_local[int(sid)] = int(r0)
        except Exception:
            continue
    return r_map_local


print("Will compute r_map from test spectrogram shapes (center crop on time axis).")



## === cell 5
test_df_raw = pd.read_csv(paths.TEST_CSV)
test_df_raw["eeg_id"] = test_df_raw["eeg_id"].astype(np.int64)
test_df_raw["spectrogram_id"] = test_df_raw["spectrogram_id"].astype(np.int64)

test_df_raw = test_df_raw.sort_values(["eeg_id", "spectrogram_id"]).drop_duplicates(
    subset=["eeg_id"], keep="first"
)

sample_sub = pd.read_csv(paths.SAMPLE_SUB, usecols=["eeg_id"])
sample_sub["eeg_id"] = sample_sub["eeg_id"].astype(np.int64)

test_df = sample_sub.merge(test_df_raw, on="eeg_id", how="left")

missing = test_df["spectrogram_id"].isna().sum()
if missing:
    print(f"WARNING: {missing} eeg_id rows missing spectrogram_id after merge.")
    test_df["spectrogram_id"] = test_df["spectrogram_id"].fillna(-1).astype(np.int64)
else:
    test_df["spectrogram_id"] = test_df["spectrogram_id"].astype(np.int64)

print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 6
r_map = build_test_r_map(test_df, paths.TEST_SPECTROGRAMS)
print(
    f"Built r_map for {len(r_map)} / {test_df['spectrogram_id'].nunique()} spectrogram_id."
)



## === cell 7
paths_spectrograms = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(paths_spectrograms)} spectrogram parquets")
all_spectrograms = {}

for file_path in tqdm(paths_spectrograms, desc="Loading test spectrograms"):
    aux = pd.read_parquet(file_path)
    name = int(file_path.split("/")[-1].split(".")[0])
    if aux.shape[1] > 0 and str(aux.columns[0]).lower() in ("time", "t", "seconds"):
        mat = aux.iloc[:, 1:].values
    else:
        mat = aux.values
    all_spectrograms[name] = mat
    del aux

if config.VISUALIZE:
    idx = np.random.randint(0, len(paths_spectrograms))
    spectrogram_path = paths_spectrograms[idx]
    plot_spectrogram(spectrogram_path)



## === cell 8
paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets")
all_eegs = {}
counter = 0

for file_path in tqdm(paths_eegs, desc="Building EEG spectrograms"):
    eeg_id = int(file_path.split("/")[-1].split(".")[0])
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs[eeg_id] = eeg_spectrogram
    counter += 1




## === cell 9
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




## === cell 10
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        r_map: Dict[int, int] = None,
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = all_spectrograms if specs is None else specs
        self.eeg_spectrograms = all_eegs if eeg_specs is None else eeg_specs
        self.r_map = {} if r_map is None else r_map

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

        eeg_id = int(row.eeg_id)
        spec_id = int(row.spectrogram_id)

        if self.mode == "test":
            r = int(self.r_map.get(spec_id, 0))
        else:
            r = int((row["min"] + row["max"]) // 4)

        eeg_img = self.eeg_spectrograms.get(eeg_id, None)
        if eeg_img is not None:
            X[:, :, 4:] = eeg_img

        spec = self.spectrograms.get(spec_id, None)
        if spec is not None:
            for region in range(4):
                time_crop = spec[
                    r : r + 256, region * 100 : (region + 1) * 100
                ]  # (256, 100)
                img = (
                    time_crop.T
                )  # (100, 256) with freq(vertical)=100, time(horizontal)=256

                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                ep = 1e-6
                mu = np.nanmean(img.flatten())
                std = np.nanstd(img.flatten())
                img = (img - mu) / (std + ep)
                img = np.nan_to_num(img, nan=0.0)

                target_h = 100  # 14:-14 => 100 rows
                target_w = 256

                img_h, img_w = img.shape
                if img_h >= target_h:
                    h0 = (img_h - target_h) // 2
                    img_hc = img[h0 : h0 + target_h, :]
                else:
                    pad_top = (target_h - img_h) // 2
                    pad_bot = target_h - img_h - pad_top
                    img_hc = np.pad(img, ((pad_top, pad_bot), (0, 0)), mode="constant")

                if img_w >= target_w:
                    w0 = (img_w - target_w) // 2
                    img_wc = img_hc[:, w0 : w0 + target_w]
                else:
                    pad_left = (target_w - img_w) // 2
                    pad_right = target_w - img_w - pad_left
                    img_wc = np.pad(
                        img_hc, ((0, 0), (pad_left, pad_right)), mode="constant"
                    )

                X[14:-14, :, region] = img_wc / 2.0

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 11
test_dataset = CustomDataset(test_df, config, mode="test", r_map=r_map)
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




## === cell 12
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device, non_blocking=True)
            with torch.inference_mode():
                logits = model(X)
                y_preds = torch.softmax(logits, dim=1)
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}


def load_model_weights(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)} for {os.path.basename(ckpt_path)}"
        )
    return model


predictions_list = []

if len(model_weights) == 0:
    print(
        "WARNING: No weights found under /kaggle/input/hba-efficientnet-weights/. "
        "Falling back to uniform predictions to generate a valid submission.csv."
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    for model_weight in model_weights:
        print("Loading weights:", model_weight)
        model = CustomModel(config)
        model = load_model_weights(model, model_weight)
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions_list.append(prediction_dict["predictions"])

        del model
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)  # (n_samples, 6)

predictions = np.asarray(predictions)
if predictions.ndim == 1:
    predictions = predictions.reshape(-1, 6)
print("Predictions shape:", predictions.shape)




## === cell 13
def smooth_and_normalize_probs(
    p: np.ndarray, alpha: float = 0.06, eps: float = 1e-4
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    k = p.shape[1]
    p = (p + alpha) / (1.0 + alpha * k)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def inv_softmax_from_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


if len(model_weights) == 0:
    calibrated = predictions
else:
    T_FIXED = 1.5
    logits = inv_softmax_from_probs(predictions)
    calibrated = softmax_np(logits / T_FIXED).astype(np.float32)

predictions = smooth_and_normalize_probs(calibrated, alpha=0.06, eps=1e-4)
print("Post-processed predictions shape:", predictions.shape)



## === cell 14
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)

p = sub[TARGETS].values.astype(np.float64)
p = np.clip(p, 1e-4, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sub[TARGETS] = p.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print(
    "Row-sum stats (min/mean/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).mean(),
    sub[TARGETS].sum(axis=1).max(),
)
sub.head()
