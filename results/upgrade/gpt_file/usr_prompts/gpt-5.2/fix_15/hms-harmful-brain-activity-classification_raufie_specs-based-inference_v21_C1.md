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

0.8042135403855587

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the missing model-weights issue by automatically falling back to a “no-weights” mode that produces a valid probability submission (uniform) when the expected Kaggle dataset `/kaggle/input/hba-efficientnet-weights/` is not available. Next, I fix a hard runtime bug in the dataset (`label_cols` is undefined for non-test modes) and a logic/indent issue that incorrectly assigns EEG spectrogram channels inside the loop. Finally, I guarantee that predictions always have shape `(len(test_df), 6)` and each row sums to 1.0, so `submission.csv` is always written with the correct columns and valid probabilities.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) indicates the code is very likely running in the “no weights found → uniform predictions” fallback, which produces a poor KL score. The smallest change that should move you toward the target is to correctly locate and load the model weights if they exist anywhere under `/kaggle/input/` (including common dataset directory nesting), while keeping the exact same model and inference logic. I add a robust weight-file discovery step (search by basename) and keep the uniform fallback only if nothing is found. This should improve the score substantially without changing architecture, training, features, or post-processing semantics.'
- What this solution (achieved 1.40995) has done: 'I fix the submission-building bug by avoiding a one-to-one merge validation that fails when `eeg_id` is duplicated in either `sample_submission` or `test_df`, and instead construct the submission by aligning to the sample submission’s `eeg_id` order via a mapping. This keeps the core model/inference logic unchanged and ensures every `eeg_id` in the submission gets exactly one probability row. I also enforce float dtype consistency for `eeg_id` keys during mapping to prevent silent mismatches, and keep the probability normalization/clipping so the submission is always valid for the KL metric. These changes are score-neutral except that they ensure your (potentially good) model predictions actually land on the correct rows rather than failing before writing.'
- What this solution (achieved 1.40995) has done: 'Your current score strongly suggests the code is still effectively producing near-uniform predictions (or otherwise miscalibrated outputs), which usually happens when weights aren’t actually loaded correctly. I make the smallest score-relevant changes by (1) tightening weight discovery to prefer an exact basename match and erroring if a found checkpoint can’t be loaded strictly (so we don’t silently fall back to bad outputs), and (2) ensuring the spectrogram slice index `r` is always valid for each test spectrogram (so inputs aren’t accidentally empty/shifted). These changes keep the same model, features, and inference; they only prevent silent failure modes and invalid indexing that degrade KL. The submission writing and probability normalization remain unchanged and submission-valid.'
- What this solution (achieved 1.40995) has done: 'The current gap is large (1.40995 vs target 0.8042, lower-is-better), and the most likely cause is still that the intended weights are not being loaded, so you’re effectively submitting near-uniform predictions. I make the smallest score-relevant changes to (1) robustly discover the *correct* EfficientNet-B3 checkpoint under `/kaggle/input/` by prioritizing filename + model-name matches, and (2) load it in a way that accepts common wrapper keys while still preventing silent partial/incorrect loads. I also fix a subtle evaluation-metric issue by renormalizing predictions **after** the `groupby(eeg_id)` averaging (averaging softmax outputs no longer sums exactly to 1), which can otherwise increase KL. Core model, feature extraction, and inference remain the same.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is still consistent with “effectively uniform” predictions, so the smallest score-relevant improvement is to (a) ensure we actually find/load the intended weights and (b) avoid silently producing uniform outputs when weights exist but load/inference is wrong. I keep the exact same model, preprocessing, and inference, but make weight discovery more robust (also looking for `.pt`/`.bin`) and make checkpoint loading tolerant to common key-mangling (`module.`, `model.`, Lightning `_orig_mod.`) while still verifying that *most* tensors match so we don’t accidentally run random weights. Finally, I add a deterministic test-time augmentation that preserves semantics (average of original + horizontally flipped input) to nudge KL down without changing architecture or training.'
- What this solution (achieved 1.40995) has done: 'The score gap is large (1.40995 vs target 0.8042, lower-is-better), and your pipeline is still very likely producing near-uniform predictions because the intended weights aren’t actually being found/loaded. I make the smallest score-relevant fixes by (1) making weight discovery search for *any* `tf_efficientnet_b3_epoch_24.pth` under `/kaggle/input/` and also accept `.pth/.pt/.bin` generally, (2) relaxing the checkpoint “match ratio” check so it doesn’t erroneously reject a valid checkpoint (which forces the uniform fallback), while still guarding against totally incompatible weights, and (3) fixing the TTA flip dimension (your current flip is on the height axis of the original tensor, not the final image width), so the augmentation is actually meaningful and consistent. Core model, feature extraction, and inference remain the same; these changes mainly prevent silent fallback/incorrect TTA that inflate KL.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.8042), and the most likely reason is still that you’re effectively not using the trained weights (or only partially loading them), leading to near-uniform/weak predictions. I keep the exact same model, dataset, preprocessing, and inference, but make weight loading more reliable by (1) prioritizing a checkpoint that matches both the basename and the model name token, and (2) expanding the “verified” load to accept a common case where the head layer names differ while still refusing clearly incompatible checkpoints. Finally, I fix the TTA flip to be unquestionably correct by flipping the final image-width axis in model space (after reshape) rather than guessing on the pre-reshape tensor, without changing outputs when TTA is disabled. These are minimal changes aimed specifically at preventing silent random/unloaded weights and making TTA consistent, which should reduce KL toward your target.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant fixes without changing your model/dataset/core inference flow: (1) correct the TTA flip axis so it flips the true image-width after the model’s reshape (your current flip is on the wrong dimension, making TTA ineffective or harmful), and (2) make the checkpoint loader accept the common case where checkpoints store `features.*` weights under the original timm backbone name (e.g., `model.*`), by remapping those keys to your `features.*` module before matching—this prevents an accidental “mostly random backbone” situation that would keep KL high. These changes should improve predictions away from near-uniform/weak outputs and move KL down toward your 0.804 target. Everything else (architecture, preprocessing, loss/metric semantics, submission format) stays the same, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'The main issue behind your 1.40995 KL is still most consistent with “weights not actually being used”, i.e., the code ends up producing uniform (or near-uniform) predictions. I make the smallest score-relevant change by implementing the missing backbone-key remapping in `_remap_backbone_keys_to_features()` so checkpoints that store timm’s backbone under its native names load correctly into your wrapped `features` module (instead of leaving most of the network random). I also slightly relax the verification threshold in `_load_state_dict_forgiving_but_verified()` to avoid incorrectly rejecting a valid checkpoint (which would otherwise trigger the uniform fallback), while still refusing clearly incompatible weights. Everything else (model architecture, spectrogram/EEG feature extraction, inference flow, and submission formatting/normalization) stays the same.'
- What this solution (achieved 1.40995) has done: 'I make the smallest score-relevant fixes that reduce the chance you’re silently running with poorly-loaded weights, since your current KL (1.40995, lower-is-better) is still consistent with “effectively random/untrained” inference. First, I fix the backbone key remapping so it correctly maps timm EfficientNet-B3 backbone parameters into your `features.*` module (your current remap likely doesn’t match `features.0.*`, `features.1.*`, etc., so most of the backbone can remain random). Second, I make the loader verification check backbone coverage more directly (still strict enough to avoid totally incompatible checkpoints) to prevent accidental fallbacks to weak partial loads. These changes preserve the exact same model architecture, preprocessing, inference loop, and submission formatting, and should move KL down toward your 0.804 target.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant fixes aimed at reducing KL from 1.40995 toward 0.8042 without changing your model architecture or feature extraction. First, I fix a likely backbone-weight remapping failure by mapping checkpoint keys to your `features.*` module using a reliable “suffix match with longest-key preference” approach, which prevents the backbone from staying mostly random. Second, I relax the backbone-coverage verification threshold slightly (while still refusing clearly incompatible checkpoints) because EfficientNet checkpoints often miss a few buffers/keys and your current 70% cutoff can incorrectly reject valid weights and trigger the uniform fallback. Everything else (data loading, spectrogram construction, model forward, inference loop, TTA semantics, and submission formatting) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your KL gap to the target (1.40995 → 0.8042, lower-is-better) is large, and the most likely score-killer left is still “weights not truly loading into the wrapped `features` backbone”, causing near-random/uniform-like outputs. I keep your architecture and inference intact, but make weight-loading deterministic and correct by mapping checkpoint keys into `features.*` via a direct module-to-module remap (timm backbone → wrapper `features`), instead of the current ambiguous suffix-matching. I also make the backbone-coverage verification focus on *parameter counts* (not tensor counts) to avoid mistakenly accepting a weak partial load or rejecting a valid load—both cases can keep KL high. Everything else (data, spectrogram construction, TTA behavior, submission formatting, probability normalization) stays the same.'

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
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b3_epoch_24.pth"
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


def _discover_weight_file(expected_path: str) -> List[str]:
    """
    Change (score-relevant): make checkpoint discovery robust so we don't fall back to uniform
    predictions when weights exist but the dataset path differs.

    Priority:
      1) exact expected path
      2) exact basename match anywhere under /kaggle/input
      3) any ckpt containing model token
    """
    if os.path.exists(expected_path) and os.path.isfile(expected_path):
        return [expected_path]

    base = os.path.basename(expected_path)
    base_noext, ext0 = os.path.splitext(base)

    exts = [ext0.lower(), ".pth", ".pt", ".bin"]
    exts = list(dict.fromkeys(exts))  # unique, keep order

    hits = []
    for ext in exts:
        hits.extend(glob(f"/kaggle/input/**/{base_noext}{ext}", recursive=True))
    hits = [p for p in hits if os.path.isfile(p)]
    if len(hits) > 0:
        preferred = [p for p in hits if "hba-efficientnet-weights" in p.lower()]
        hits = preferred if len(preferred) > 0 else hits
        return sorted(hits)

    model_token = config.MODEL.lower()
    any_ckpt = []
    for ext in [".pth", ".pt", ".bin"]:
        any_ckpt.extend(glob(f"/kaggle/input/**/*{ext}", recursive=True))
    any_ckpt = [p for p in any_ckpt if os.path.isfile(p)]
    model_named = [p for p in any_ckpt if model_token in os.path.basename(p).lower()]
    if len(model_named) > 0:
        return sorted(model_named)

    return []


model_weights = [paths.MODEL_WEIGHTS]
available_model_weights = []
for w in model_weights:
    available_model_weights.extend(_discover_weight_file(w))

if len(available_model_weights) > 1:
    model_token = config.MODEL.lower()
    token_hits = [
        w for w in available_model_weights if model_token in os.path.basename(w).lower()
    ]
    if len(token_hits) > 0:
        available_model_weights = token_hits

if len(available_model_weights) > 1:
    preferred = [w for w in available_model_weights if "hba-efficientnet-weights" in w]
    if len(preferred) > 0:
        available_model_weights = preferred
if len(available_model_weights) > 1:
    available_model_weights = [sorted(available_model_weights)[0]]

if len(available_model_weights) == 0:
    print(f"WARNING: No model weights found for expected path: {paths.MODEL_WEIGHTS}")
    print(
        "         Will generate a valid submission using uniform probabilities (score will be poor but submission-valid)."
    )
else:
    print(f"Found model weight file: {available_model_weights[0]}")




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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
label_cols = TARGETS




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
        Reshapes input (128, 256, 8) -> (3, 512, 512) image.
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
        self.df = df
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

        if self.mode == "test":
            r = 0
        else:
            r = int((row["min"] + row["max"]) // 4)

        eeg_img = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        spect = self.spectrograms[int(row.spectrogram_id)]
        max_r = max(0, spect.shape[0] - 300)
        if r < 0:
            r = 0
        elif r > max_r:
            r = max_r

        for region in range(4):
            img = spect[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
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
def _extract_state_dict(ckpt):
    """
    Change (score-relevant): support common wrapper formats so weights actually load.
    """
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "net", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_known_prefixes(state_dict: dict) -> dict:
    """
    Change (score-relevant): handle common prefix wrappers without changing core model.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ["module.", "model.", "net.", "_orig_mod."]
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _remap_backbone_keys_to_features(state_dict: dict, model: nn.Module) -> dict:
    """
    Change (score-relevant): deterministically map a timm backbone checkpoint into this wrapper.

    Previous suffix-matching could map incorrectly or not at all, leaving the backbone random.
    Here we do a direct map:
      - backbone keys (as in model.model.state_dict()) -> "features.{idx}." + key
      - head keys are left as-is (custom head is different anyway)
    """
    if not isinstance(state_dict, dict):
        return state_dict

    ckpt_keys = list(state_dict.keys())

    if any(k.startswith("features.") for k in ckpt_keys) or any(
        k.startswith("custom_layers.") for k in ckpt_keys
    ):
        return state_dict

    orig_sd = model.model.state_dict()
    wrapper_features_prefix = "features.0."

    remapped = {}
    for k, v in state_dict.items():
        if k in orig_sd:
            remapped[wrapper_features_prefix + k] = v
        else:
            remapped[k] = v
    return remapped


def _load_state_dict_forgiving_but_verified(model: nn.Module, state: dict) -> None:
    """
    Change (score-relevant): verify loading by matched PARAMETER COUNT in the backbone.

    Counting tensors can be misleading; parameter-count coverage better detects "mostly random"
    backbone situations that would yield near-uniform predictions and high KL.
    """
    model_sd = model.state_dict()

    backbone_total_params = 0
    backbone_matched_params = 0

    for k, v in model_sd.items():
        if k.startswith("features.") and hasattr(v, "numel"):
            backbone_total_params += int(v.numel())

    filtered = {}
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if v.shape == model_sd[k].shape:
                filtered[k] = v
                if k.startswith("features.") and hasattr(v, "numel"):
                    backbone_matched_params += int(v.numel())

    backbone_ratio = backbone_matched_params / max(1, backbone_total_params)

    if backbone_ratio < 0.55:
        raise RuntimeError(
            f"Checkpoint seems incompatible/partial: backbone matched {backbone_matched_params}/{backbone_total_params} "
            f"params ({backbone_ratio:.1%}). Refusing to run mostly-random weights."
        )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if len(unexpected) > 0:
        print(f"Note: unexpected keys after filtering: {len(unexpected)}")
    if len(missing) > 0:
        print(f"Note: missing keys after filtering: {len(missing)}")


def inference_function(test_loader, model, device, use_tta: bool = True):
    """
    Horizontal flip TTA on width axis in model-input space (B,C,H,W).
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                logits1 = model(X)
                p1 = softmax(logits1)

                if use_tta:
                    Xr = model._CustomModel__reshape_input(X)
                    Xrf = torch.flip(Xr, dims=[3])
                    logits2 = model.custom_layers(model.features(Xrf))
                    p2 = softmax(logits2)
                    y_preds = 0.5 * (p1 + p2)
                else:
                    y_preds = p1

            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
if len(available_model_weights) > 0:
    predictions_list = []
    for model_weight in available_model_weights:
        model = CustomModel(config)
        checkpoint = torch.load(model_weight, map_location="cpu")
        state = _extract_state_dict(checkpoint)
        state = _strip_known_prefixes(state)

        state = _remap_backbone_keys_to_features(state, model)

        _load_state_dict_forgiving_but_verified(model, state)

        model.to(device)

        pred_dict = inference_function(test_loader, model, device, use_tta=True)
        predictions_list.append(pred_dict["predictions"])

        del model, checkpoint, state
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
else:
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)

print("Predictions shape:", predictions.shape)




## === cell 11
if (
    predictions.ndim != 2
    or predictions.shape[1] != 6
    or predictions.shape[0] != len(test_df)
):
    raise ValueError(
        f"Predictions must be (n_test, 6). Got {predictions.shape}, expected ({len(test_df)}, 6)."
    )

predictions = np.clip(predictions, 1e-12, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sample_sub = pd.read_csv(paths.SAMPLE_SUB)

test_eeg_ids = test_df["eeg_id"].astype(np.int64).values
pred_df = pd.DataFrame(predictions, columns=TARGETS)
pred_df["eeg_id"] = test_eeg_ids

pred_df = pred_df.groupby("eeg_id", as_index=False)[TARGETS].mean()

p2 = pred_df[TARGETS].values.astype(np.float64)
p2 = np.clip(p2, 1e-12, 1.0)
p2 = p2 / p2.sum(axis=1, keepdims=True)
pred_df[TARGETS] = p2.astype(np.float32)

sub = sample_sub[["eeg_id"]].copy()
sub["eeg_id"] = sub["eeg_id"].astype(np.int64)
sub = sub.merge(pred_df, on="eeg_id", how="left")

missing = sub[TARGETS].isna().any(axis=1)
if missing.any():
    sub.loc[missing, TARGETS] = 1.0 / 6.0

p = sub[TARGETS].values.astype(np.float64)
p = np.clip(p, 1e-12, 1.0)
p = p / p.sum(axis=1, keepdims=True)
sub[TARGETS] = p.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print(sub.head())
