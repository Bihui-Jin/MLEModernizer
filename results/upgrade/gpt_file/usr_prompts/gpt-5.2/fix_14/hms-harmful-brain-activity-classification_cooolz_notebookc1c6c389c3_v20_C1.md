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

0.4373425895911577

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the script robust to missing external weight directories by falling back to a valid, score-safe baseline (uniform probabilities) when no model files are found, so a submission is always produced. I also fix the CUDA hard-failure by selecting CPU when CUDA isn’t available, and avoid the heavy precompute cell that tried to create EEG spectrograms on GPU. Finally, I ensure only actual weight files are loaded (not directories) and guarantee the submission has the correct columns, shape, and row-wise probability sums to 1.0.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4373), and the main reason is that the notebook is almost certainly falling back to uniform predictions because the referenced weight directories (`/kaggle/input/hms-baseline`, `/kaggle/input/hms-eeg`, `/kaggle/input/hms-mix`) don’t exist in your environment. I make a minimal change to auto-discover any `.pt/.pth/.bin` weights inside `/kaggle/input/**` (common when Kaggle datasets are mounted under different names), while keeping your exact model code and inference flow. I also ensure the most relevant branch runs first (mix -> eeg -> spec) only if corresponding weights are found, and keep the uniform fallback as a safe backstop. These changes should materially improve the score toward the target by actually using available weights, without altering the model architecture, data pipeline, or prediction semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), and the most likely cause is that you’re still not actually loading usable checkpoints (or you’re loading incompatible checkpoint dicts), so the pipeline effectively behaves like a weak baseline. I make minimal, inference-only fixes to (1) load common Kaggle checkpoint formats robustly (`state_dict`, `model`, `module.*`), (2) prefer the most relevant single checkpoint per group to reduce the chance of averaging mismatched folds/architectures, and (3) fix a small but important bug in `brain_lead` (RP duplicated instead of RR), which can materially hurt EEG-based inference while preserving the exact modeling approach. These changes keep the same models/feature extraction/loss semantics and only improve correctness and checkpoint usage to move KL toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4373), and the most likely remaining cause is still “not actually using a good checkpoint”: either no weights are being found, or the one picked is incompatible/mismatched, leading to near-uniform/garbage probabilities. I make the smallest inference-only change to (1) expand checkpoint discovery to include common extensions like `.ckpt` and (2) automatically choose the checkpoint that best matches each model’s expected parameter keys/shapes (instead of filename heuristics), while keeping your exact dataset/model/inference logic intact. Additionally, I fix a small bug in `get_spec` where the region slice index `r` never increments (all 4 “regions” are identical), which can materially degrade spec/mix inference without changing the overall approach. These changes should move KL substantially toward the target by ensuring the intended inputs and best-compatible weights are actually used.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4373), and the most likely cause is that inference is still not actually using the intended checkpoints correctly (or is using an incompatible model head), producing near-uniform/poor probabilities. I make the smallest inference-only fixes that improve checkpoint compatibility: (1) correct the `NetMix` classifier input dimension for `hrnet_w18` (it should be 1024, not 2048), and (2) ensure the `use_mix` branch is reachable by writing `eeg_specs_dict.pkl` even when precompute wasn’t run (CPU-safe lazy precompute). These changes preserve the core model architectures/training semantics and only fix mismatches/availability so real weights can be applied, which should materially reduce KL toward the target. The rest of the pipeline (data reading, transforms, inference loop, submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL is far above the target, and the most likely cause is that inference is still effectively weak due to loading *test EEG windows from the wrong temporal location* (always from offset 0), while your own `spectrogram_from_eeg()` uses the centered 10,000 samples. I make the smallest correctness-only fix: extract the centered 50s window in `get_eeg()` exactly like your mix-precompute does, without changing any model code, architecture, or inference loop. This should materially improve EEG and MIX branches (both depend on correct EEG segment content), moving KL toward the target. I also keep submission normalization/clipping as-is to ensure the file remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4373), and the most likely reason is that the script is still selecting no/poor checkpoints and falling back to a weak baseline-like behavior. I make two minimal, inference-only fixes that preserve your model architectures and loops: (1) normalize/robustify checkpoint discovery so it prefers actual HMS competition checkpoints under `/kaggle/input/hms-harmful-brain-activity-classification/` and avoids random unrelated weights, and (2) when multiple model types are available, compute predictions for each available branch (mix/eeg/spec) and average them (simple ensemble), which typically reduces KL without changing any single model’s semantics. I also set `model.eval()` and wrap inference with `torch.inference_mode()` (no numerical change, but safer/faster) and keep the required probability normalization to guarantee a valid submission. These changes are directly aimed at moving KL down toward the target while keeping the core logic intact.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower-is-better) is far worse than the target (0.4373), so we should improve predictive correctness rather than tune for marginal gains. The biggest score-risking issue left is that `NetSpec` is created with `in_chans=3` but your spec input is 4-channel (and you later concatenate into 3 channels manually), which can cause incompatible/partial checkpoint loading and effectively weak inference; I change it to `in_chans=3` *and* load the backbone weights more strictly by filtering only matching keys/shapes (instead of `strict=False` with large missing/unexpected), so the model actually uses the checkpoint as intended. Additionally, I prevent the expensive/slow mix precompute from running by default (it can blow the 600s budget), and instead only enable mix if `eeg_specs_dict.pkl` already exists; this keeps the run reliable and focuses on the likely-working EEG/SPEC branches to reduce KL toward the target. Finally, I fix a subtle data issue: `get_spec()` currently slices frequency/time windows assuming a 1200x400 layout, but the official spectrogram parquet has a time axis of 300 and 4 regions stacked vertically; I adjust the slicing to the correct layout used in common baselines (4 vertical regions, full time), which materially improves spec inputs without changing the overall approach.'
- What this solution (achieved 1.40995) has done: 'Your KL is far above the target, so the most likely issue is still that you are not actually using any meaningful checkpoints and are effectively submitting uniform/near-random probabilities. I make the smallest changes that increase the chance of loading real, compatible HMS checkpoints by (1) also searching for `.safetensors` weights (common on Kaggle), and (2) making checkpoint discovery prefer directories/files whose path names strongly indicate HMS/HBAC rather than any random “spec/eeg/mix” match under `/kaggle/input`. I also ensure the correct test-eeg/spectrogram directories are used via a single base path (no semantic change), and keep the exact models/inference logic intact so any score change comes purely from actually loading better weights. If no usable weights are found, the uniform fallback remains to guarantee a valid submission.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant fixes that keep your core model/inference logic intact: (1) correct the test spectrogram slicing so it matches the actual parquet layout (300 frequency bins × 4 regions, time on rows), because the current code effectively swaps time/frequency and harms SPEC/MIX predictions; and (2) ensure the DataLoader uses a safe worker configuration for Kaggle (often avoids silent slowdowns/instability) without changing semantics. These changes should materially reduce KL from the current weak baseline behavior toward your target by feeding the models correctly structured inputs and making inference reliable. Everything else (architectures, softmax outputs, ensembling/normalization, submission format) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.4373), and the most likely reason is still that you are either (a) not loading any real HMS checkpoints, or (b) failing to load them correctly because `.safetensors` isn’t supported by `torch.load`, causing silent skips and ending up with weak/uniform predictions. I make a minimal, inference-only fix to correctly load `.safetensors` checkpoints (if present) using `safetensors.torch.load_file` when available, while keeping the exact same model architectures, preprocessing, and inference loop. I also slightly adjust the weight discovery preference to look for common HMS checkpoint keywords (fold/effnet/hrnet) so we’re more likely to pick a real competition model rather than unrelated weights, without changing any prediction semantics. These changes should increase the chance that your existing branches (eeg/spec/mix) actually run with meaningful weights, moving KL down toward the target.'
- What this solution (achieved 1.40995) has done: 'I make minimal inference-only fixes that reduce your KL toward the target by ensuring you actually load and use compatible checkpoints (right now you’re likely still hitting the uniform/near-random path, or partially loading mismatched weights). Concretely: (1) add a strict path filter so weight discovery only considers HMS/HBAC-related directories under `/kaggle/input`, (2) make state-dict extraction more robust for Lightning-style checkpoints and common key prefixes (e.g., `model.`, `net.`), and (3) avoid silently accepting low-coverage loads by requiring a higher matched-shape ratio when selecting the “most compatible” checkpoint. These changes keep your exact models, preprocessing, and inference loop semantics intact, but significantly increase the odds that the intended weights are used correctly, which should move KL down toward the target.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant correctness fixes in the data pipeline that can materially reduce KL without changing your model architectures, losses, or inference loop. First, your EEG bandpass currently keeps only 0–20 Hz (`rr=20`), which is an overly aggressive low-pass that typically degrades HMS performance; I restore the common baseline band (0–40 Hz) by changing the default `rr` to 40 while leaving everything else unchanged. Second, your `get_spec()` currently assumes the spectrogram parquet is laid out as `(time, 1200)` (4×300), but in this competition the saved spectrograms are commonly `(freq, time)` with 4 regions stacked vertically; I implement a robust reader that detects the layout and always returns `(4, time, freq)` consistently. These are targeted input-shape/content fixes (not model changes) and should move KL down from 1.40995 toward your 0.437 target while keeping runtime within limits and preserving submission validity.'

# 9. Code solution

## === cell 0
import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
import gc

import albumentations as A
import os
import librosa
import pickle
import timm
from tqdm import tqdm
import mne

import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 0,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
}

DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TEST_EEG_DIR = f"{DATA_ROOT}/test_eegs"
TEST_SPEC_DIR = f"{DATA_ROOT}/test_spectrograms"




## === cell 2
def _is_weight_file(fn: str) -> bool:
    fn = fn.lower()
    return (
        fn.endswith(".pt")
        or fn.endswith(".pth")
        or fn.endswith(".bin")
        or fn.endswith(".ckpt")
        or fn.endswith(".safetensors")
    )


def _list_weight_files(path: str):
    if not isinstance(path, str) or (not path):
        return []
    if not os.path.exists(path):
        return []
    if os.path.isfile(path):
        return [path] if _is_weight_file(path) else []
    files = []
    for x in sorted(os.listdir(path)):
        fp = os.path.join(path, x)
        if os.path.isfile(fp) and _is_weight_file(x):
            files.append(fp)
    return files


def _discover_weight_files_under_kaggle_input(max_files_per_group=50):
    """
    Change rationale (score -> target): restrict discovery to HMS/HBAC-related areas to avoid
    selecting unrelated checkpoints that load "successfully" but yield near-random probabilities.
    """
    roots = [DATA_ROOT, "/kaggle/input"]
    spec, eeg, mix = [], [], []

    def _looks_like_hms(fp: str) -> bool:
        l = fp.lower()
        return (
            ("hms-harmful-brain-activity-classification" in l)
            or ("harmful" in l)
            or ("hbac" in l)
            or ("/hms" in l)
            or ("hms_" in l)
            or ("harmful-brain" in l)
        )

    def _path_priority(fp: str) -> int:
        l = fp.lower()
        p = 0
        if (
            "hms-harmful-brain-activity-classification" in l
            or "hbac" in l
            or "harmful" in l
        ):
            p += 80
        if "checkpoint" in l or "ckpt" in l:
            p += 8
        if "fold" in l:
            p += 6
        if "efficientnet" in l or "effnet" in l:
            p += 4
        if "hrnet" in l:
            p += 4
        if "baseline" in l:
            p += 2
        return p

    candidates = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not _is_weight_file(fn):
                    continue
                fp = os.path.join(dirpath, fn)
                if not _looks_like_hms(fp):
                    continue
                candidates.append(fp)

    candidates = sorted(
        set(candidates), key=lambda fp: (-_path_priority(fp), len(fp), fp)
    )

    for fp in candidates:
        lfp = fp.lower()
        if "mix" in lfp:
            mix.append(fp)
        elif "eeg" in lfp:
            eeg.append(fp)
        elif ("spec" in lfp) or ("spect" in lfp) or ("baseline" in lfp):
            spec.append(fp)

        if (
            len(spec) >= max_files_per_group
            and len(eeg) >= max_files_per_group
            and len(mix) >= max_files_per_group
        ):
            break

    return {
        "spec": spec[:max_files_per_group],
        "eeg": eeg[:max_files_per_group],
        "mix": mix[:max_files_per_group],
    }


CFG["weights_spec"] = _list_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = _list_weight_files(CFG["weights_eeg"])
CFG["weights_mix"] = _list_weight_files(CFG["weights_mix"])

if (len(CFG["weights_spec"]) + len(CFG["weights_eeg"]) + len(CFG["weights_mix"])) == 0:
    discovered = _discover_weight_files_under_kaggle_input(max_files_per_group=200)
    CFG["weights_spec"] = discovered["spec"]
    CFG["weights_eeg"] = discovered["eeg"]
    CFG["weights_mix"] = discovered["mix"]

print(
    "Weights discovered -> spec:",
    len(CFG["weights_spec"]),
    "| eeg:",
    len(CFG["weights_eeg"]),
    "| mix:",
    len(CFG["weights_mix"]),
)
CFG



## === cell 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 4
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from tqdm import tqdm

data_dir = TEST_EEG_DIR
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


class TransformMel(nn.Module):
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

    def forward(self, x):
        return self.wave_transform(x)


def spectrogram_from_eeg(parquet_path, transform_func, device_infer):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            x_tensor = torch.from_numpy(x.astype(np.float32)).to(device_infer)
            mel_spec = transform_func(x_tensor).detach().cpu().numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0
    return img


def _ensure_eeg_specs_dict_for_mix():
    """
    Change rationale (score -> target + timeout safety): do NOT precompute mix inputs unless the
    file already exists. Precomputing for 9850 EEGs can exceed the 600s time limit.
    """
    return


_ensure_eeg_specs_dict_for_mix()




## === cell 5
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
        rr=40,  # Change rationale (score -> target): restore common EEG band upper cutoff (0-40Hz) vs 0-20Hz which often removes discriminative content and worsens KL.
    ):

        self.ll = 0
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None
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
            if not os.path.exists("eeg_specs_dict.pkl"):
                raise FileNotFoundError(
                    "eeg_specs_dict.pkl not found but use_mix=True. Provide it or disable mix."
                )
            with open("eeg_specs_dict.pkl", mode="rb") as f:
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
        eeg_path = f"{TEST_EEG_DIR}/{dp['eeg_id']}.parquet"
        eeg = pd.read_parquet(eeg_path)

        middle = (len(eeg) - 10_000) // 2
        eeg = eeg.iloc[middle : middle + 10_000]

        waves = eeg.values
        waves = np.transpose(waves, axes=[1, 0])

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0

        waves = np.array(waves, dtype=np.float64)
        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        waves = self.brain_lead(waves)
        return waves

    def get_spec(self, dp, is_training):
        """
        Change rationale (score -> target): robustly parse test spectrogram parquet layout.
        Kaggle HMS spectrogram parquets are commonly either:
          (A) (time, 1201) with first col = time, remaining 1200 = 4 regions * 300 freqs
          (B) (freq*4, time+1) or (freq*4, time) style with regions stacked vertically.
        This function detects layout and always returns (4, time, freq) for the model.
        """
        spec_path = f"{TEST_SPEC_DIR}/{dp['spectrogram_id']}.parquet"
        arr = pd.read_parquet(spec_path).values

        if arr.shape[1] in (1201, 301, 401, 501) and np.all(np.isfinite(arr[:, 0])):
            maybe = arr[:, 1:]
            if maybe.shape[1] % 300 == 0:
                arr = maybe

        if arr.ndim == 2 and arr.shape[1] == 1200:
            t = arr.shape[0]
            spec = arr.reshape(t, 4, 300)  # (time, region, freq)
            spec = np.transpose(spec, (1, 0, 2))  # (4, time, freq)
        elif arr.ndim == 2 and arr.shape[0] == 1200:
            t = arr.shape[1]
            spec = arr.reshape(4, 300, t)  # (region, freq, time)
            spec = np.transpose(spec, (0, 2, 1))  # (4, time, freq)
        else:
            if arr.ndim == 2 and arr.shape[0] % 4 == 0:
                per = arr.shape[0] // 4
                spec = arr.reshape(4, per, arr.shape[1])  # (region, freq?, time?)
                if per == 300:
                    spec = np.transpose(spec, (0, 2, 1))  # (4, time, 300)
                else:
                    spec = np.transpose(spec, (0, 2, 1))  # (4, time, freq_like)
            else:
                raise ValueError(
                    f"Unexpected spectrogram shape: {arr.shape} for {spec_path}"
                )

        spec = np.clip(spec, np.exp(-4), np.exp(8))
        spec = np.log(spec)
        spec = np.nan_to_num(spec, nan=0.0)
        return spec.astype(np.float32)

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




## === cell 6
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




## === cell 7
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




## === cell 8
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)
        self.fc = nn.Linear(1024, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)

        x2 = [x[:, i + 4 : i + 5, :, :] for i in range(4)]
        x2 = torch.cat(x2, dim=2)

        x = torch.cat([x1, x2], dim=3)
        x = torch.cat([x, x, x], dim=1)

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




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.inference_mode():
                y_preds = model(X)
            preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 11
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

try:
    from safetensors.torch import load_file as _safetensors_load_file
except Exception:
    _safetensors_load_file = None


def _load_checkpoint_any(fp: str):
    fp_l = fp.lower()
    if fp_l.endswith(".safetensors"):
        if _safetensors_load_file is None:
            raise RuntimeError(
                "safetensors is not available but a .safetensors checkpoint was selected."
            )
        return _safetensors_load_file(fp)
    return torch.load(fp, map_location="cpu")


def _extract_state_dict(ckpt):
    """
    Change rationale (score -> target): handle Lightning and nested formats more robustly so we
    don't end up with near-empty loads (which behave like a weak baseline).
    """
    if isinstance(ckpt, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _strip_any_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = ["module.", "model.", "net.", "encoder.", "backbone."]
    sd = state_dict
    for pref in prefixes:
        if any(k.startswith(pref) for k in sd.keys()):
            sd = {k[len(pref) :]: v for k, v in sd.items()}
    return sd


def _compat_score_for_model(model, state_dict):
    if not isinstance(state_dict, dict):
        return -1.0
    msd = model.state_dict()
    matched = 0
    matched_shape = 0
    total = len(msd)
    for k, v in state_dict.items():
        if k in msd:
            matched += 1
            try:
                if tuple(v.shape) == tuple(msd[k].shape):
                    matched_shape += 1
            except Exception:
                pass
    if total == 0:
        return -1.0
    return (matched_shape / total) + 0.05 * (matched / total)


def _select_most_compatible_weights(files, model_ctor, topk=1):
    """
    Change rationale (score -> target): pick checkpoints with high matched-shape ratio so we avoid
    partially-incompatible weights that load but predict poorly.
    """
    if not files:
        return []
    scored = []
    for fp in files[:200]:
        try:
            model = model_ctor()
            ckpt = _load_checkpoint_any(fp)
            sd = _strip_any_known_prefixes(_extract_state_dict(ckpt))
            score = _compat_score_for_model(model, sd)
            scored.append((score, fp))
        except Exception:
            continue
        finally:
            try:
                del model, ckpt, sd
            except Exception:
                pass
            gc.collect()

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    picked = [fp for s, fp in scored[:topk] if s >= 0.35]
    return picked


CFG["weights_mix"] = _select_most_compatible_weights(CFG["weights_mix"], NetMix, topk=1)
CFG["weights_eeg"] = _select_most_compatible_weights(CFG["weights_eeg"], NetEeg, topk=1)
CFG["weights_spec"] = _select_most_compatible_weights(
    CFG["weights_spec"], NetSpec, topk=1
)

print(
    "Weights selected -> spec:",
    len(CFG["weights_spec"]),
    "| eeg:",
    len(CFG["weights_eeg"]),
    "| mix:",
    len(CFG["weights_mix"]),
)
print(
    "Selected paths:",
    {"spec": CFG["weights_spec"], "eeg": CFG["weights_eeg"], "mix": CFG["weights_mix"]},
)


def _filter_state_dict_to_model(model, state_dict):
    msd = model.state_dict()
    out = {}
    for k, v in state_dict.items():
        if k in msd:
            try:
                if tuple(v.shape) == tuple(msd[k].shape):
                    out[k] = v
            except Exception:
                continue
    return out


def _run_ensemble(weights, dataset_kwargs, model_ctor):
    preds_list = []
    for model_weight in weights:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, **dataset_kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        model = model_ctor()

        ckpt = _load_checkpoint_any(model_weight)
        state_dict = _strip_any_known_prefixes(_extract_state_dict(ckpt))
        state_dict = _filter_state_dict_to_model(model, state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(state_dict) < 0.7 * len(model.state_dict()):
            print(
                f"Skip low-coverage checkpoint: {model_weight} | loaded={len(state_dict)} missing={len(missing)} unexpected={len(unexpected)}"
            )
            del model, test_loader, test_dataset, ckpt, state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model.to(device)
        pred_dict = inference_function(test_loader, model, device)
        preds_list.append(pred_dict["predictions"])

        del model, test_loader, test_dataset, ckpt, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(preds_list) == 0:
        return None
    return np.mean(np.stack(preds_list, axis=0), axis=0)


branch_preds = []
branch_names = []

if len(CFG["weights_mix"]) > 0 and os.path.exists("eeg_specs_dict.pkl"):
    print("infer with weights_mix:", len(CFG["weights_mix"]))
    p = _run_ensemble(
        CFG["weights_mix"], dataset_kwargs={"use_mix": True}, model_ctor=NetMix
    )
    if p is not None:
        branch_preds.append(p)
        branch_names.append("mix")
elif len(CFG["weights_mix"]) > 0:
    print(
        "mix weights found but eeg_specs_dict.pkl missing -> skip mix to avoid timeout."
    )

if len(CFG["weights_eeg"]) > 0:
    print("infer with weights_eeg:", len(CFG["weights_eeg"]))
    p = _run_ensemble(
        CFG["weights_eeg"], dataset_kwargs={"use_eeg": True}, model_ctor=NetEeg
    )
    if p is not None:
        branch_preds.append(p)
        branch_names.append("eeg")

if len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec:", len(CFG["weights_spec"]))
    p = _run_ensemble(
        CFG["weights_spec"], dataset_kwargs={"use_spec": True}, model_ctor=NetSpec
    )
    if p is not None:
        branch_preds.append(p)
        branch_names.append("spec")

if len(branch_preds) == 0:
    print("No usable model weights found; writing uniform-probability submission.")
    pred = np.full((len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
else:
    print("Averaging branches:", branch_names)
    pred = np.mean(np.stack(branch_preds, axis=0), axis=0)

pred = np.asarray(pred, dtype=np.float64)
if pred.shape != (len(test_df), len(TARGETS)):
    raise ValueError(
        f"Predictions shape {pred.shape} does not match expected {(len(test_df), len(TARGETS))}."
    )

pred = np.clip(pred, 1e-9, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
