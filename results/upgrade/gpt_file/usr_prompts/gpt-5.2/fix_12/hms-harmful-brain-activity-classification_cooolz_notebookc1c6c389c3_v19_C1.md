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
mne==1.10.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.402780406294065

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the script robust to missing external weight folders by auto-disabling those model ensembles and falling back to a valid, metric-safe submission (uniform probabilities) so you always get a submit-able `submission.csv`. I also fix GPU-only code paths by using `cuda` only when available and keeping all spectrogram generation on CPU otherwise, preventing the “no NVIDIA driver” crash. I prevent `torch.load` from trying to load directories by filtering weight paths to actual files with known model extensions. Finally, I add a normalization/safety step to ensure every prediction row sums to 1 and has the exact required columns, avoiding submission-format failures.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.40278), and the most likely reason is that your run is falling back to uniform predictions because no weights are actually being found at the configured paths. I make the smallest change that improves score toward target: auto-detect and use the *real* HMS dataset root in this environment and search common weight locations (without changing any model code) so inference actually runs when weights exist. I also fix one small but impactful bug in `brain_lead()` where the last region is duplicated (`RP` twice), which can hurt EEG-model predictions without altering the architecture/training semantics. Finally, I keep your probability normalization/safety logic intact to guarantee a valid submission.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.40278), and the most likely cause is that you are still effectively doing weak inference (badly loaded weights / wrong state_dict key) or degrading predictions via a data bug. I make two minimal, score-relevant fixes without changing the model architectures or inference semantics: (1) load checkpoints robustly by extracting the correct `state_dict` when weights are saved as Lightning/EMA wrappers, and (2) fix a clear bug in `get_spec()` where `r` never increments, so all 4 regions currently use the same 300-row slice. I also ensure the mel-spectrogram precompute runs in `eval()`/no-grad and on CPU unless CUDA is available, improving numerical stability and avoiding accidental training-mode behavior. These are small, safe changes that typically move KL down substantially compared to the current broken/degenerate inputs.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far above the target (0.40278), and the most likely reason is that you’re still effectively submitting weak predictions because no real weights are being loaded from the configured folders. I make the smallest change that increases the chance of actually finding and using model checkpoints in this environment by expanding the weight-directory auto-detection to search under the HMS dataset root and common Kaggle working locations, without touching model architectures or inference logic. I also make state-dict loading slightly more robust to Lightning-style key prefixes like `net.`/`module.` so valid checkpoints don’t silently load poorly under `strict=False`. These changes should materially lower KL (toward target) while preserving your core pipeline and still guaranteeing a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) suggests the inference is still effectively near-uniform/weak, so the smallest likely-to-help fixes are to (1) ensure EEG preprocessing uses the intended floating-point precision (float32) to match typical training/inference behavior and avoid filter/numerics drift, and (2) make checkpoint loading stricter in a safe way by explicitly validating that a meaningful fraction of weights actually matched (so you don’t silently run with mostly-random init under `strict=False`). If matching fails, we fall back to uniform for that checkpoint only (still producing a valid submission), rather than averaging in junk predictions that inflate KL. These changes keep your model architectures, feature extraction, and inference loops intact while making the ensemble less likely to be poisoned by bad/partial checkpoint loads. The submission writing and probability normalization remain unchanged and metric-safe.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), and given the code already has the key bugfixes, the remaining likely cause is that you’re still not loading any real checkpoints (so you’re effectively uniform or near-random). I make two minimal, directly score-relevant changes: (1) broaden weight-file discovery to recursively search under `/kaggle/input`, `/kaggle/working`, and the HMS root for common checkpoint filenames so your existing inference actually runs with weights when they exist, and (2) add a tiny safeguard to stop “poisoning” the ensemble with near-uniform predictions by skipping checkpoints whose output entropy is essentially uniform across the test set (often happens with wrong/empty loads). These changes preserve your model architectures and inference loops and still guarantee a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your score is far above the target (lower-is-better), and the most likely reason is that you are still averaging in weak/incorrectly-loaded checkpoints or falling back to near-uniform predictions. I make two minimal, directly score-relevant fixes without changing model architectures or inference semantics: (1) tighten checkpoint validation to ensure the model actually loaded meaningful weights (skip if too few keys/shapes match), and (2) fix a known HRNet/NetMix head dimension mismatch by inferring the correct feature size at init (this avoids silently-bad loads that poison the ensemble). I also ensure spectrogram-from-EEG preprocessing uses the same central crop as the dataset (consistent 50s middle 10k samples) and keep the probability normalization to guarantee valid submissions. These changes should move KL down toward your target by preventing “junk” ensemble members and enabling mix checkpoints to load properly when present.'
- What this solution (achieved 1.40995) has done: 'Your score is far above the target (lower-is-better), and the most likely reason is that you are still effectively submitting near-uniform predictions because no usable checkpoints are being found/loaded in this environment. I make two minimal, score-relevant changes: (1) broaden checkpoint discovery to also accept weights provided as a single file path and to search a bit more reliably under `/kaggle/input` for any plausible `.ckpt/.pth/.pt` files, and (2) make state-dict loading more robust by also stripping common Lightning prefixes like `model.model.` and `ema_model.` so real weights actually load and aren’t skipped. These changes preserve your model architectures and inference loops, keep the same preprocessing, and still guarantee a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the most likely cause is still that you are submitting a near-uniform distribution (either because no real checkpoints are found/loaded, or because the “near-uniform” filter is incorrectly skipping usable models). I make two minimal, directly score-relevant changes: (1) make weight-file discovery actually pick up checkpoints placed in nested folders by recursively searching *even when* a weight directory exists but contains no files at its top level, and (2) relax/disable the “near-uniform” skip so we don’t accidentally discard valid but cautious models (which would force the uniform fallback). These preserve your model architectures, preprocessing, and inference loops, and keep the same submission normalization so the CSV remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.40278), and the most likely reason is that your ensemble is still being “poisoned” by checkpoints that technically pass the loose `strict=False` load but are not actually compatible (or are random-ish), plus some models are likely being skipped leaving you near-uniform. I make two minimal, score-relevant fixes without changing any model architecture or inference semantics: (1) strengthen checkpoint validation by requiring a higher match ratio and also verifying that the matched tensors cover key layers (classifier + early stem), and (2) stop re-creating the dataset/loader inside every checkpoint loop so that inference is consistent and less error-prone (same data order, less overhead). These changes should reduce KL substantially by ensuring only truly compatible checkpoints contribute to the mean, while still guaranteeing a valid submission even if no good checkpoints exist. The uniform fallback and probability normalization remain intact to keep submissions metric-safe.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target, and the most plausible minimal fix is to stop “poisoning” the ensemble with checkpoints that technically load but are actually incompatible (leading to near-uniform or junk predictions). I keep all models/inference intact and only add a stricter, shape-aware load that filters each checkpoint to the intersection of matching tensors, then validates that key layers (stem + head) are present before using it. This typically turns “bad-load-but-runs” models into “skipped”, which should move KL materially down toward the target without changing architectures or preprocessing. I also enforce a deterministic DataLoader order and ensure the mel transform always runs on the same device as intended (already mostly true), keeping submission formatting unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import copy
import json
import pickle
import random

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchaudio

import albumentations as A
import librosa
import timm
import mne
from tqdm import tqdm
from torch.utils.data import DataLoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

EEG_TEST_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
SPEC_TEST_DIR = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
)




## === cell 2
def list_weight_files(path: str):
    exts = (".pt", ".pth", ".bin", ".ckpt")
    if not isinstance(path, str):
        return []
    if os.path.isfile(path) and path.lower().endswith(exts):
        return [path]
    if not os.path.isdir(path):
        return []
    out = []
    for x in sorted(os.listdir(path)):
        fp = os.path.join(path, x)
        if os.path.isfile(fp) and fp.lower().endswith(exts):
            out.append(fp)
    return out


def resolve_hms_root_and_paths():
    candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification",
        "/kaggle/working/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/working/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    ]
    root = None
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "test.csv")):
            root = c
            break
    if root is None:
        root = "/kaggle/input/hms-harmful-brain-activity-classification"

    data_csv = os.path.join(root, "test.csv")
    eeg_dir = os.path.join(root, "test_eegs")
    spec_dir = os.path.join(root, "test_spectrograms")
    return root, data_csv, eeg_dir, spec_dir


def find_first_existing_dir(paths):
    for p in paths:
        if isinstance(p, str) and os.path.isdir(p):
            return p
    return None


def recursive_find_weight_files(
    roots,
    max_files=80,
    exts=(".pt", ".pth", ".bin", ".ckpt"),
    name_hints=(),
):
    found = []
    seen = set()
    for r in roots:
        if not (isinstance(r, str) and os.path.isdir(r)):
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            base = os.path.basename(dirpath).lower()
            if base in {
                "train_eegs",
                "test_eegs",
                "train_spectrograms",
                "test_spectrograms",
            }:
                dirnames[:] = []
                continue

            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                if name_hints:
                    if not any(h in lfn for h in name_hints):
                        continue
                fp = os.path.join(dirpath, fn)
                if fp in seen:
                    continue
                if os.path.isfile(fp):
                    found.append(fp)
                    seen.add(fp)
                    if len(found) >= max_files:
                        return sorted(found)
    return sorted(found)


HMS_ROOT, CFG["data"], EEG_TEST_DIR, SPEC_TEST_DIR = resolve_hms_root_and_paths()
print("HMS_ROOT:", HMS_ROOT)
print("CFG['data']:", CFG["data"])
print("EEG_TEST_DIR exists:", os.path.isdir(EEG_TEST_DIR))
print("SPEC_TEST_DIR exists:", os.path.isdir(SPEC_TEST_DIR))

spec_w_dir = find_first_existing_dir(
    [
        CFG["weights_spec"],
        "/kaggle/input/hms-baseline",
        "/kaggle/data/hms-baseline",
        "/kaggle/working/hms-baseline",
        os.path.join(HMS_ROOT, "hms-baseline"),
        os.path.join(HMS_ROOT, "weights_spec"),
        os.path.join("/kaggle/working", "hms-baseline"),
        os.path.join("/kaggle/working", "weights_spec"),
    ]
)
eeg_w_dir = find_first_existing_dir(
    [
        CFG["weights_eeg"],
        "/kaggle/input/hms-eeg",
        "/kaggle/data/hms-eeg",
        "/kaggle/working/hms-eeg",
        os.path.join(HMS_ROOT, "hms-eeg"),
        os.path.join(HMS_ROOT, "weights_eeg"),
        os.path.join("/kaggle/working", "hms-eeg"),
        os.path.join("/kaggle/working", "weights_eeg"),
    ]
)
mix_w_dir = find_first_existing_dir(
    [
        CFG["weights_mix"],
        "/kaggle/input/hms-mix",
        "/kaggle/data/hms-mix",
        "/kaggle/working/hms-mix",
        os.path.join(HMS_ROOT, "hms-mix"),
        os.path.join(HMS_ROOT, "weights_mix"),
        os.path.join("/kaggle/working", "hms-mix"),
        os.path.join("/kaggle/working", "weights_mix"),
    ]
)

CFG["weights_spec"] = list_weight_files(CFG["weights_spec"]) or (
    list_weight_files(spec_w_dir) if spec_w_dir else []
)
if spec_w_dir and len(CFG["weights_spec"]) == 0:
    CFG["weights_spec"] = recursive_find_weight_files(
        [spec_w_dir], max_files=30, name_hints=()
    )

CFG["weights_eeg"] = list_weight_files(CFG["weights_eeg"]) or (
    list_weight_files(eeg_w_dir) if eeg_w_dir else []
)
if eeg_w_dir and len(CFG["weights_eeg"]) == 0:
    CFG["weights_eeg"] = recursive_find_weight_files(
        [eeg_w_dir], max_files=30, name_hints=()
    )

CFG["weights_mix"] = list_weight_files(CFG["weights_mix"]) or (
    list_weight_files(mix_w_dir) if mix_w_dir else []
)
if mix_w_dir and len(CFG["weights_mix"]) == 0:
    CFG["weights_mix"] = recursive_find_weight_files(
        [mix_w_dir], max_files=30, name_hints=()
    )

if (
    len(CFG["weights_spec"]) == 0
    and len(CFG["weights_eeg"]) == 0
    and len(CFG["weights_mix"]) == 0
):
    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/data",
        HMS_ROOT,
    ]
    CFG["weights_spec"] = recursive_find_weight_files(
        search_roots,
        max_files=30,
        name_hints=("spec", "baseline", "efficientnet", "b5", "effnet"),
    )
    CFG["weights_eeg"] = recursive_find_weight_files(
        search_roots,
        max_files=30,
        name_hints=("eeg", "efficientnet", "b5", "effnet"),
    )
    CFG["weights_mix"] = recursive_find_weight_files(
        search_roots,
        max_files=30,
        name_hints=("mix", "hrnet", "w18"),
    )

print("weights_spec dir:", spec_w_dir, "n:", len(CFG["weights_spec"]))
print("weights_eeg  dir:", eeg_w_dir, "n:", len(CFG["weights_eeg"]))
print("weights_mix  dir:", mix_w_dir, "n:", len(CFG["weights_mix"]))

HAVE_ANY_WEIGHTS = (
    len(CFG["weights_spec"]) + len(CFG["weights_eeg"]) + len(CFG["weights_mix"])
) > 0
print("HAVE_ANY_WEIGHTS:", HAVE_ANY_WEIGHTS)



## === cell 3
NAMES = ["LL", "LP", "RP", "RR"]
FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


class MelTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=200,
            hop_length=10000 // 256,
            n_fft=1024,
            n_mels=128,
            f_min=0,
            f_max=20,
            win_length=128,
        )

    def forward(self, x: torch.Tensor):
        return self.wave_transform(x)


def spectrogram_from_eeg(
    parquet_path: str, transform_func: nn.Module, transform_device: torch.device
):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    transform_func.eval()
    with torch.no_grad():
        for k in range(4):
            COLS = FEATS[k]
            for kk in range(4):
                x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
                m = np.nanmean(x)
                if np.isnan(x).mean() < 1:
                    x = np.nan_to_num(x, nan=m)
                else:
                    x[:] = 0

                x_tensor = torch.from_numpy(x.astype(np.float32)).to(transform_device)
                mel_spec = transform_func(x_tensor)  # [n_mels, time]
                mel_spec = mel_spec.detach().to("cpu").numpy()

                width = (mel_spec.shape[1] // 32) * 32
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(
                    np.float32
                )[:, :width]
                img[:, :, k] += mel_spec_db

            img[:, :, k] /= 4.0

    return img


EEG_SPECS_PKL = "eeg_specs_dict.pkl"
NEED_EEG_SPECS = len(CFG["weights_mix"]) > 0

if NEED_EEG_SPECS and (not os.path.exists(EEG_SPECS_PKL)):
    print("Precomputing eeg_specs_dict.pkl (needed for mix model)...")
    transform_device = device  # use GPU if available else CPU
    transform_func = MelTransform().to(transform_device)
    all_fs = [x for x in os.listdir(EEG_TEST_DIR) if x.endswith(".parquet")]
    all_specs = {}
    for item in tqdm(all_fs):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = os.path.join(EEG_TEST_DIR, f"{eeg_id}.parquet")
        eeg_spec = spectrogram_from_eeg(eeg_path, transform_func, transform_device)
        all_specs[eeg_id] = eeg_spec
    with open(EEG_SPECS_PKL, "wb") as file:
        pickle.dump(all_specs, file)
    del all_specs
    gc.collect()
else:
    if NEED_EEG_SPECS:
        print("Found existing eeg_specs_dict.pkl; will reuse.")
    else:
        print("Mix weights not found; skipping eeg_specs_dict.pkl generation.")




## === cell 4
class AlaskaDataIter:
    def __init__(
        self,
        df,
        training_flag=False,
        shuffle=False,
        use_spec=False,
        use_eeg=False,
        use_mix=False,
        ll=0,
        rr=20,
    ):
        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.df = df

        self.train_trans = A.Compose([A.HorizontalFlip(p=0.5)])

        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]

        self.LL = ["Fp1", "F7", "T3", "T5", "O1"]
        self.RR = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix

        if self.use_mix:
            if not os.path.exists(EEG_SPECS_PKL):
                raise FileNotFoundError(
                    f"{EEG_SPECS_PKL} not found, but use_mix=True. "
                    f"Either generate it or disable mix inference."
                )
            with open(EEG_SPECS_PKL, mode="rb") as f:
                self.eeg_specs = pickle.load(f)

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RR]

        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp_lead = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)
        return data

    def get_eeg(self, dp, is_training):
        eeg_path = os.path.join(EEG_TEST_DIR, f"{dp['eeg_id']}.parquet")
        eeg = pd.read_parquet(eeg_path)

        offset = 0
        eeg = eeg.iloc[int(offset * 200) : int(offset * 200) + 10000]

        waves = eeg.values
        waves = np.transpose(waves, axes=[1, 0])

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0

        waves = np.asarray(waves, dtype=np.float32)
        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        waves = self.brain_lead(waves)
        return waves

    def get_spec(self, dp, is_training):
        spec_path = os.path.join(SPEC_TEST_DIR, f"{dp['spectrogram_id']}.parquet")
        spec = pd.read_parquet(spec_path)

        spec = spec.values[:, 1:]

        images = []
        r = 0
        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
            images.append(img)
            r += 300

        images = np.stack(images, -1)
        data = np.transpose(images, [2, 0, 1])
        return data

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)
        eeg_spec = self.eeg_specs[str(dp["eeg_id"])]

        kg_spec = np.transpose(kg_spec, axes=[1, 2, 0])

        X[14:-14, :, :4] = kg_spec[:, 22:-22]
        X[:, :, 4:] = eeg_spec

        X = np.transpose(X, [2, 0, 1])
        return X

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            data = self.get_eeg(dp, is_training)
        elif self.use_spec:
            data = self.get_spec(dp, is_training)
        elif self.use_mix:
            data = self.get_mix(dp, is_training)
        else:
            raise ValueError("One of use_eeg/use_spec/use_mix must be True.")
        return data.astype(np.float32)




## === cell 5
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 6
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 7
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)

        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            self.model.eval()
            dummy = torch.zeros(1, 3, 128, 512, dtype=torch.float32)
            feat = self.model.forward_features(dummy)
            feat = self.avg(feat)
            feat_dim = int(feat.view(1, -1).shape[1])

        self.fc = nn.Linear(feat_dim, 6, bias=True)

    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)

        x2 = [x[:, i + 4 : i + 5, :, :] for i in range(4)]
        x2 = torch.cat(x2, dim=2)

        x = torch.cat([x1, x2], dim=3)
        x = torch.cat([x, x, x], dim=1)

        x = self.model.forward_features(x)
        x = self.avg(x)

        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        return x




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}


def normalize_probs(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = p / s
    return p.astype(np.float32)


def load_state_dict_flexible(ckpt_path: str, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "weights",
            "params",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict):
        prefixes = (
            "model.",
            "net.",
            "module.",
            "ema_model.",
            "model_ema.",
            "student.",
            "teacher.",
            "model.model.",
            "net.model.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if any(k.startswith(prefix) for k in sd.keys()):
                    sd = {k.replace(prefix, "", 1): v for k, v in sd.items()}
                    changed = True
    return sd


def filter_state_dict_by_shape(model: nn.Module, state_dict: dict):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            filtered[k] = v
    return filtered


def load_with_match_check(
    model: nn.Module,
    state_dict: dict,
    min_match_ratio: float = 0.93,
    min_num_matches: int = 120,
    required_any_keys=(),
):
    model_sd = model.state_dict()
    match = 0
    total = len(model_sd)
    for k, v in model_sd.items():
        if (
            k in state_dict
            and hasattr(state_dict[k], "shape")
            and state_dict[k].shape == v.shape
        ):
            match += 1
    ratio = match / max(total, 1)

    if required_any_keys:
        ok_required = False
        for rk in required_any_keys:
            if (
                rk in model_sd
                and rk in state_dict
                and hasattr(state_dict[rk], "shape")
                and state_dict[rk].shape == model_sd[rk].shape
            ):
                ok_required = True
                break
        if not ok_required:
            return False, ratio, match, total

    if match < min_num_matches or ratio < min_match_ratio:
        return False, ratio, match, total

    filtered = filter_state_dict_by_shape(model, state_dict)
    model.load_state_dict(filtered, strict=False)
    return True, ratio, match, total


def looks_near_uniform(pred: np.ndarray, tol: float = 0.03) -> bool:
    if pred.ndim != 2 or pred.shape[1] != 6:
        return True
    m = pred.mean(axis=0)
    return float(np.max(np.abs(m - (1.0 / 6.0)))) < tol




## === cell 9
test_df = pd.read_csv(CFG["data"])
print("test_df shape:", test_df.shape)
test_df.head()



## === cell 10
if not HAVE_ANY_WEIGHTS:
    print("No weights found -> using uniform probabilities fallback.")
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    predictions_all = []

    spec_dataset = None
    spec_loader = None
    eeg_dataset = None
    eeg_loader = None
    mix_dataset = None
    mix_loader = None

    g = torch.Generator()
    g.manual_seed(42)

    if len(CFG["weights_spec"]) > 0:
        spec_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_spec=True
        )
        spec_loader = DataLoader(
            spec_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            generator=g,
            persistent_workers=(CFG["num_worker"] > 0),
        )

    if len(CFG["weights_eeg"]) > 0:
        eeg_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        eeg_loader = DataLoader(
            eeg_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            generator=g,
            persistent_workers=(CFG["num_worker"] > 0),
        )

    if len(CFG["weights_mix"]) > 0:
        mix_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_mix=True
        )
        mix_loader = DataLoader(
            mix_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            generator=g,
            persistent_workers=(CFG["num_worker"] > 0),
        )

    print("infer with weights_spec")
    for model_weight in CFG["weights_spec"]:
        model = NetSpec()
        state_dict = load_state_dict_flexible(model_weight, map_location=device)

        ok, ratio, match, total = load_with_match_check(
            model,
            state_dict,
            min_match_ratio=0.93,
            min_num_matches=120,
            required_any_keys=("fc.weight", "model.conv_stem.weight"),
        )
        if not ok:
            print(
                f"[skip] spec ckpt low match match={match}/{total} ratio={ratio:.3f}: {model_weight}"
            )
            del model, state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model.to(device)
        pred = inference_function(spec_loader, model, device)["predictions"]
        pred = normalize_probs(pred)
        predictions_all.append(pred)

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("infer with weights_eeg")
    for model_weight in CFG["weights_eeg"]:
        model = NetEeg()
        state_dict = load_state_dict_flexible(model_weight, map_location=device)

        ok, ratio, match, total = load_with_match_check(
            model,
            state_dict,
            min_match_ratio=0.93,
            min_num_matches=120,
            required_any_keys=("fc.weight", "model.conv_stem.weight"),
        )
        if not ok:
            print(
                f"[skip] eeg ckpt low match match={match}/{total} ratio={ratio:.3f}: {model_weight}"
            )
            del model, state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model.to(device)
        pred = inference_function(eeg_loader, model, device)["predictions"]
        pred = normalize_probs(pred)
        predictions_all.append(pred)

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("infer with weights_mix")
    for model_weight in CFG["weights_mix"]:
        model = NetMix()
        state_dict = load_state_dict_flexible(model_weight, map_location=device)

        ok, ratio, match, total = load_with_match_check(
            model,
            state_dict,
            min_match_ratio=0.90,
            min_num_matches=120,
            required_any_keys=("fc.weight",),
        )
        if not ok:
            print(
                f"[skip] mix ckpt low match match={match}/{total} ratio={ratio:.3f}: {model_weight}"
            )
            del model, state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model.to(device)
        pred = inference_function(mix_loader, model, device)["predictions"]
        pred = normalize_probs(pred)
        predictions_all.append(pred)

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(predictions_all) == 0:
        print("All checkpoints were skipped (bad matches) -> uniform fallback.")
        predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        predictions_all = np.array(predictions_all)  # [n_models, n_samples, 6]
        predictions = np.mean(predictions_all, axis=0)

predictions = normalize_probs(predictions)
print(
    "predictions shape:",
    predictions.shape,
    "row-sum range:",
    predictions.sum(1).min(),
    predictions.sum(1).max(),
)



## === cell 11
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

sub = sub[["eeg_id"] + TARGETS]
probs = sub[TARGETS].values
probs = normalize_probs(probs)
sub[TARGETS] = probs

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Saved {out_path} shape={sub.shape}")
sub.head()
