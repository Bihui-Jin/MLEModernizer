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

0.8225230189177692

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the runtime blocker by making the code robust to missing external model-weight files: if the expected `/kaggle/input/hba-efficientnet-weights/*.pth` files are not available, the script still run end-to-end by using a safe fallback prediction (uniform probabilities) and writing a valid `submission.csv`. I also fix the shape/logic issues that cause the “Columns must be same length as key” error by ensuring predictions are always `(len(test_df), 6)` and by removing loader-variable mixups in inference. Finally, I add a small, score-safe post-processing step to enforce non-negative probabilities and exact row-wise sum-to-one normalization (required by the competition), without changing the model’s core logic.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 score is driven largely by the “no weights found → uniform predictions” fallback, which is far from the target 0.8225 on this KL metric. I keep your exact model/dataset/inference logic, but make the weight discovery robust to the actual environment by searching common Kaggle input locations (including the competition dataset folders) and accepting both `.pth` and `.pt`. To avoid inference failures when checkpoints are stored under different keys, I load the state dict with a small, safe key-stripping fallback (still the same architecture/weights, just more tolerant loading). With weights successfully loaded, your existing pipeline should move materially closer to the target score while still writing a valid `submission.csv` with properly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'Your current score is far from the target mainly because you’re still often falling back to uniform predictions (or loading mismatched checkpoints), which is weak on KL divergence. I keep your exact dataset/model/inference pipeline, but make checkpoint discovery and loading more precise: prefer files that match your configured model name (efficientnet_b3) and ignore irrelevant `.pth/.pt` files that can silently “load” with `strict=False` and degrade predictions. I also add a tiny, score-safe safeguard to skip checkpoints whose state_dict clearly doesn’t fit (too many missing/unexpected keys), so the ensemble only averages compatible models. These changes should materially improve the KL score toward the 0.8225 target while still producing a valid normalized `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your score gap is large (1.40995 vs target 0.8225, lower is better), and the most likely cause is that your ensemble is still frequently using the uniform fallback or producing poorly calibrated probabilities for KL. I keep your exact model/dataset/inference pipeline, but tighten checkpoint discovery to avoid accidentally loading unrelated `.pth/.pt` files (which can “load” with `strict=False` yet hurt predictions), and I add an explicit compatibility check based on tensor shapes so only truly matching EfficientNet-B3 checkpoints are averaged. Finally, because KL is sensitive to overconfident wrong predictions, I add a very small temperature smoothing + blend with uniform (a calibration-only post-process) to move the KL toward the target without changing the model architecture or training. The script still always write a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current gap to the target is large (1.40995 vs 0.82252, lower is better), and the biggest likely driver is not the model itself but (1) possible test-row misalignment (predictions not guaranteed to be in the exact same order as `sample_submission.csv`) and (2) overly-restrictive checkpoint compatibility filtering that can accidentally drop all usable weights and trigger the uniform fallback. I make two minimal, score-relevant fixes: align the final predictions to `sample_submission.csv` order by mapping `eeg_id -> prediction`, and slightly relax the checkpoint compatibility check to avoid false rejections while still skipping truly mismatched shapes. I also ensure deterministic evaluation behavior in inference (`torch.inference_mode()` + `cudnn` determinism) without changing the model or preprocessing logic. These changes should move the KL score materially down toward the target while still producing a valid `submission.csv` with per-row probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995) is far worse than the target (0.82252, lower is better), so we should cautiously improve score without changing the core model/data pipeline. The largest likely issue is that you’re accidentally averaging in unrelated `.pth/.pt` files because checkpoint discovery is too broad, and then forcing `strict=False` loads that can silently degrade predictions; I constrain discovery to EfficientNet-B3-relevant filenames and add a stricter compatibility gate based on the classifier head weight shape. I also make state-dict loading robust to common key prefixes (`model.`, `net.`, `backbone.`) while still requiring a near-complete match, so we load the *right* weights (or cleanly fall back). Finally, I keep your existing probability normalization and alignment, but reduce the post-hoc smoothing slightly (less uniform blending) because with correct weights it can overly flatten predictions and hurt KL.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995) is far from the target (0.82252; lower is better), and the most likely reason in this code is that it still often falls back to uniform predictions because it can’t find/load any compatible checkpoints in your environment. I make checkpoint discovery deterministic and restricted to actual Kaggle “input” locations, and broaden the loader to accept common head-key names (classifier/fc/head) so real EfficientNet-B3 checkpoints don’t get falsely rejected by the `custom_layers.2.weight` check. I also replace the too-strict “70% of all model keys must overlap” gate with a safer compatibility check based on matching tensor shapes over the *overlapping* keys (so we don’t reject checkpoints that simply store fewer keys but are otherwise correct). These are minimal changes that preserve your exact model/data/inference logic, but should load valid weights more often and reduce KL toward the target; submission writing/normalization stays intact.'
- What this solution (achieved 1.40995) has done: 'Your current score is much worse than the target (KL: 1.40995 vs 0.8225; lower is better), and the most likely reason is that you are still usually falling back to uniform predictions because the checkpoint loader rejects real checkpoints too often. I keep your exact model/dataset/inference pipeline, but make checkpoint loading more tolerant in a score-safe way: accept common alternative classifier-head key names by remapping them into your `custom_layers.2.*`, and relax the compatibility gate to require shape-matching on overlapping tensors rather than requiring a specific head key to exist. I also keep your probability normalization/alignment, but reduce the post-hoc smoothing slightly (temperature closer to 1 and smaller uniform blend) so that once real weights load, you don’t unnecessarily flatten predictions and hurt KL. These are minimal changes aimed specifically at getting real weights to load and produce non-uniform predictions, which should move the KL substantially toward 0.8225.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995) is far worse than the target (0.82252, lower is better), and the most likely reason (given this code) is that you’re still ending up with the uniform-probability fallback because no compatible EfficientNet-B3 checkpoints are actually being found/loaded. I make the checkpoint discovery also look in `/kaggle/data/**` (your environment clearly has weights/data mirrored there), and I make the head-weight remapping handle the most common EfficientNet head key `model.classifier.*` / `classifier.*` stored under different wrapper dicts so valid weights aren’t falsely rejected. Finally, I slightly relax the overlap threshold in the shape-compatibility check (still requiring exact shape matches on overlapping keys) so we load real checkpoints more often; this should move KL down toward the target while keeping the model, preprocessing, and inference semantics unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995) is far above the target (0.8225, lower is better), and the most likely driver in this notebook is that you’re still producing near-uniform predictions because no checkpoints are being found/loaded in this environment. I make the smallest score-relevant fix by explicitly searching `/kaggle/input/**` and `/kaggle/data/**` for EfficientNet-B3 weights while also excluding the competition dataset folders (which contain no weights but many `.parquet` files) to avoid wasted globbing and timeouts. I also make checkpoint loading more tolerant to common wrapper formats by (1) unwrapping `{'model': ...}` / `{'state_dict': ...}` and (2) stripping common prefixes and remapping the final classifier to your `custom_layers.2.*` more broadly (including `*_fc.*`/`*_classifier.*` variants), while still enforcing strict shape-compatibility on overlapping tensors to avoid loading wrong models. These changes keep your model/dataset/inference logic intact, but should greatly reduce fallback-to-uniform and move KL down toward the target while still writing a valid normalized `submission.csv`.'

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
    MODEL = "tf_efficientnet_b3"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b3_epoch_29.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    SAMPLE_SUB = (
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )


def discover_weight_files() -> List[str]:
    """
    Score-relevant fix: search common Kaggle locations for weights, but explicitly
    EXCLUDE the competition dataset folders (parquets only) to avoid slow scans and
    reduce the chance of "no weights found -> uniform fallback".
    """
    exclude_markers = [
        "/hms-harmful-brain-activity-classification/",
        "/train_eegs/",
        "/test_eegs/",
        "/train_spectrograms/",
        "/test_spectrograms/",
        "/example_figures/",
    ]

    patterns = [
        "/kaggle/input/**/tf_efficientnet_b3*.pth",
        "/kaggle/input/**/tf_efficientnet_b3*.pt",
        "/kaggle/input/**/*efficientnet*b3*.pth",
        "/kaggle/input/**/*efficientnet*b3*.pt",
        "/kaggle/input/**/hba*weight*/*.pth",
        "/kaggle/input/**/hba*weight*/*.pt",
        "/kaggle/data/**/tf_efficientnet_b3*.pth",
        "/kaggle/data/**/tf_efficientnet_b3*.pt",
        "/kaggle/data/**/*efficientnet*b3*.pth",
        "/kaggle/data/**/*efficientnet*b3*.pt",
        "/kaggle/data/**/hba*weight*/*.pth",
        "/kaggle/data/**/hba*weight*/*.pt",
    ]
    found = []
    for p in patterns:
        found.extend(glob(p, recursive=True))

    def is_excluded(fp: str) -> bool:
        fpl = fp.lower()
        return any(m in fpl for m in exclude_markers)

    found = [
        f
        for f in found
        if os.path.isfile(f)
        and (f.endswith(".pth") or f.endswith(".pt"))
        and (not is_excluded(f))
    ]

    model_l = config.MODEL.lower()
    tokens = []
    if "efficientnet" in model_l:
        tokens.append("efficientnet")
    if "b3" in model_l:
        tokens.append("b3")
    if "tf_" in model_l:
        tokens.append("tf")

    def score_path(fp: str) -> int:
        base = os.path.basename(fp).lower()
        s = 0
        for t in tokens:
            if t in base:
                s += 10
        if "epoch" in base:
            s += 2
        if "fold" in base:
            s += 1
        if "best" in base:
            s += 2
        if "last" in base:
            s += 1
        return s

    found_sorted = sorted(found, key=score_path, reverse=True)

    scored = [f for f in found_sorted if score_path(f) >= 10]
    found_sorted = scored if len(scored) > 0 else found_sorted

    seen = set()
    out = []
    for f in found_sorted:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


discovered = discover_weight_files()
model_weights = (
    discovered
    if len(discovered) > 0
    else ([paths.MODEL_WEIGHTS] if os.path.exists(paths.MODEL_WEIGHTS) else [])
)

print(f"Discovered weight files: {len(model_weights)}")
if len(model_weights) > 0:
    for wf in model_weights[:10]:
        print(" -", wf)
if len(model_weights) == 0:
    print(
        "WARNING: No model weight files found. Will generate a valid submission using a safe fallback (uniform probs)."
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        self.spectrograms = specs if specs is not None else all_spectrograms
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else all_eegs

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

        r = 0 if self.mode == "test" else int((row["min"] + row["max"]) // 4)

        for region in range(4):
            spec = self.spectrograms[int(row.spectrogram_id)]
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

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
            with torch.inference_mode():
                y_preds = model(X)
                y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for k in ["model", "state_dict", "model_state_dict", "net", "weights"]:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                return checkpoint[k]
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(
        isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()
    ):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _maybe_strip_known_prefixes(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    prefixes = ["model.", "net.", "backbone.", "encoder.", "module.model."]
    keys = list(state_dict.keys())
    for p in prefixes:
        if all(
            (isinstance(k, str) and k.startswith(p)) for k in keys[: min(50, len(keys))]
        ):
            return {k[len(p) :]: v for k, v in state_dict.items()}
    return state_dict


def _maybe_remap_classifier_to_custom_head(model: nn.Module, state_dict: dict) -> dict:
    """
    Score-relevant fix: broaden head remapping for common EfficientNet wrappers so valid
    checkpoints load (avoids uniform fallback). Architecture stays identical.
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    msd = model.state_dict()
    target_w = "custom_layers.2.weight"
    target_b = "custom_layers.2.bias"
    if target_w in state_dict and target_b in state_dict:
        return state_dict

    cand_w = [
        "classifier.weight",
        "fc.weight",
        "head.weight",
        "model.classifier.weight",
        "model.fc.weight",
        "model.head.weight",
        "module.classifier.weight",
        "module.fc.weight",
        "module.head.weight",
        "model._fc.weight",
        "_fc.weight",
        "model.classifier._linear.weight",
        "classifier._linear.weight",
    ]
    cand_b = [
        "classifier.bias",
        "fc.bias",
        "head.bias",
        "model.classifier.bias",
        "model.fc.bias",
        "model.head.bias",
        "module.classifier.bias",
        "module.fc.bias",
        "module.head.bias",
        "model._fc.bias",
        "_fc.bias",
        "model.classifier._linear.bias",
        "classifier._linear.bias",
    ]

    src_w = next((k for k in cand_w if k in state_dict), None)
    src_b = next((k for k in cand_b if k in state_dict), None)

    if src_w is None or src_b is None:
        return state_dict

    try:
        sw, sb = state_dict[src_w], state_dict[src_b]
        if (
            hasattr(sw, "shape")
            and hasattr(sb, "shape")
            and tuple(sw.shape) == tuple(msd[target_w].shape)
            and tuple(sb.shape) == tuple(msd[target_b].shape)
        ):
            new_sd = dict(state_dict)
            new_sd[target_w] = sw
            new_sd[target_b] = sb
            return new_sd
    except Exception:
        return state_dict

    return state_dict


def _shape_compatible(model: nn.Module, state_dict: dict) -> bool:
    """
    Keep safety: only accept checkpoints whose overlapping tensors exactly match shapes,
    but avoid rejecting "thin" state_dicts too aggressively.
    """
    if not isinstance(state_dict, dict):
        return False
    msd = model.state_dict()

    overlap = 0
    for k, v in state_dict.items():
        if k in msd and hasattr(v, "shape") and hasattr(msd[k], "shape"):
            overlap += 1
            if tuple(v.shape) != tuple(msd[k].shape):
                return False

    return overlap >= 5


predictions_list = []

if len(model_weights) == 0:
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    loaded_ok = 0

    for model_weight in model_weights:
        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")

        state = _extract_state_dict(checkpoint)
        state = _strip_module_prefix(state)
        state = _maybe_strip_known_prefixes(state)
        state = _maybe_remap_classifier_to_custom_head(model, state)

        if not _shape_compatible(model, state):
            print(
                f"Skipping incompatible checkpoint (shape mismatch on overlapping keys): {model_weight}"
            )
            del model, checkpoint, state
            torch.cuda.empty_cache()
            gc.collect()
            continue

        try:
            model.load_state_dict(state, strict=True)
            compatible = True
            missing = []
            unexpected = []
        except RuntimeError:
            incompat = model.load_state_dict(state, strict=False)
            missing = list(getattr(incompat, "missing_keys", []))
            unexpected = list(getattr(incompat, "unexpected_keys", []))

            total_params = len(list(model.state_dict().keys()))
            badness = (len(missing) + len(unexpected)) / max(1, total_params)
            compatible = badness <= 0.02

        if not compatible:
            print(
                f"Skipping incompatible checkpoint: {model_weight}\n"
                f"  missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )
            del model, checkpoint, state
            torch.cuda.empty_cache()
            gc.collect()
            continue

        print(f"Using checkpoint: {model_weight}")
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions_list.append(prediction_dict["predictions"])
        loaded_ok += 1

        del model, checkpoint, state
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    if loaded_ok == 0:
        print(
            "WARNING: No compatible checkpoints loaded. Falling back to uniform probabilities."
        )
        predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)

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

preds = np.asarray(predictions, dtype=np.float64)
if preds.ndim != 2 or preds.shape[1] != 6:
    raise ValueError(f"Predictions must be (N,6). Got {preds.shape}")

eps = 1e-12
preds = np.clip(preds, eps, None)
preds = preds / preds.sum(axis=1, keepdims=True)

T = 1.01
preds = np.power(preds, 1.0 / T)
preds = np.clip(preds, eps, None)
preds = preds / preds.sum(axis=1, keepdims=True)

alpha = 0.005
preds = (1.0 - alpha) * preds + alpha * (1.0 / 6.0)
preds = np.clip(preds, eps, None)
preds = preds / preds.sum(axis=1, keepdims=True)

sample_sub = pd.read_csv(paths.SAMPLE_SUB)
sub_eeg_ids = sample_sub["eeg_id"].values

pred_map = {int(eid): preds[i] for i, eid in enumerate(test_df["eeg_id"].values)}
aligned = np.vstack([pred_map[int(eid)] for eid in sub_eeg_ids])

aligned = np.clip(aligned, eps, None)
aligned = aligned / aligned.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": sub_eeg_ids})
sub[TARGETS] = aligned.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
