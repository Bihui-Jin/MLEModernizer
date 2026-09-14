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

1.1936010238944363

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blocker by making the weight loading robust: if the external weight file is missing in this environment, the code fall back to a valid, deterministic baseline prediction (uniform probabilities) so a submission CSV is still produced. I also fix the dataset bug where `label_cols` is undefined (and where EEG spectrogram assignment is incorrectly inside the region loop), and ensure the test inference uses the correct DataLoader variable. Finally, I add a strict post-processing step to guarantee probabilities are finite, non-negative, and sum to 1 per row (required by the competition), which is score-neutral/stability-focused.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 is worse than the target 1.1936 (lower is better), and the main issue is that you’re falling back to uniform predictions because the external weights path doesn’t exist in this environment. The smallest score-improving change that preserves your core model/inference logic is to make weight discovery robust by also searching common Kaggle input/working locations for `*.pt` weight files and loading the first match. If no weights are found, we keep your exact uniform fallback to still guarantee a valid submission. This should move the score toward the target without altering architecture, preprocessing, loss, or inference semantics (only restoring intended weights usage).'
- What this solution (achieved 1.40995) has done: 'Your current score is worse than the target (1.40995 vs 1.1936; lower is better), and the biggest likely cause is still that you’re not successfully loading the intended EfficientNet weights (or you’re loading the wrong file among many matches). I make the weight discovery stricter and deterministic by prioritizing the exact expected filename first and then only accepting checkpoints whose tensors actually match your model’s `state_dict` shapes (so we don’t silently load incompatible weights or fall back to uniform). This preserves your architecture and inference logic, but increases the chance you use the real trained weights, which should move the score toward the target. If no compatible weights are found, the code still produce a valid submission via the same uniform fallback.'
- What this solution (achieved 1.40995) has done: 'Your score is worse than the target (1.40995 vs 1.1936; lower is better), and given your prior notes the dominant cause is likely still that you aren’t actually using the intended trained weights (or you’re averaging in uniform fallbacks). I make weight loading stricter: only average predictions from checkpoints that both (1) match the model’s tensor shapes and (2) load successfully with `strict=True`; if none load, we keep your uniform fallback to guarantee a valid submission. This is a minimal change that preserves your model/inference logic and metric semantics, but prevents “poisoning” the ensemble with uniform predictions that push KL worse. I also add a deterministic preference order that prioritizes the exact expected filename and then the shortest path, so you consistently pick the most likely correct checkpoint first.'
- What this solution (achieved 1.40995) has done: 'Your score is worse than the target (1.40995 vs 1.1936; lower is better), and the most likely reason is still that you’re not actually loading the intended trained weights, so predictions are effectively weak/uniform. I keep your exact model/dataset/inference logic, but make checkpoint loading robust to common training-save formats by (1) stripping `module.` prefixes (DDP) and (2) allowing a deterministic “filtered strict” load that drops only clearly-non-matching keys (e.g., EMA/meta) while still requiring full shape matches for the backbone/head. This should increase the probability that a real compatible checkpoint successfully loads, moving the score toward the target without changing architecture or evaluation semantics. If nothing loads, the existing uniform fallback remains unchanged to guarantee a valid submission.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995) is worse than the target (1.1936; lower is better), and the most likely reason is that you’re still not loading the “right” trained weights (or you’re ensembling multiple mismatched checkpoints). I keep your exact model/dataset/inference core logic, but make checkpoint selection deterministic and stricter: prefer checkpoints whose keys indicate they belong to your `tf_efficientnet_b4` backbone and include a compatible classifier head, then stop after loading the single best candidate instead of averaging across many unknown `.pt/.pth` files. This should move predictions away from near-uniform/poisoned ensembles toward a more meaningful set of probabilities, improving KL toward the target while still falling back to uniform if nothing compatible exists. Submission writing and probability normalization stay the same to guarantee a valid CSV.'
- What this solution (achieved 1.40995) has done: 'Your current score is worse than the target (1.40995 vs 1.1936; lower is better), and the most likely cause is still that no compatible checkpoint is being found/loaded so you fall back to uniform predictions. I keep your exact model/dataset/inference logic, but make checkpoint loading more robust to common Kaggle formats by (1) also handling nested keys like `model`/`net`, (2) trying a deterministic “best candidate” list, and (3) loading with a strict-first then filtered load that also strips a leading `model.` prefix (in addition to `module.`). To avoid accidentally selecting an unrelated `.pt` from other datasets, I tighten discovery to prefer checkpoints located under the current competition folder name. If no checkpoint still loads, the same uniform fallback and probability normalization remain, ensuring a valid submission CSV.'
- What this solution (achieved 1.40995) has done: 'Your score is worse than the target (1.40995 vs 1.1936; lower is better), and the most likely cause is that you still fall back to uniform predictions because no compatible checkpoint is actually found/loaded. I keep your exact model/dataset/inference logic, but broaden checkpoint discovery to include common Kaggle weight extensions (especially `.ckpt`) and improve compatibility probing by also handling checkpoints where weights live under nested keys like `model_state_dict/state_dict` inside a `.ckpt`. I also make the “filtered load” actually safe by switching it to `strict=False` (your current code can never succeed because you filter keys then require strict=True), which increases the chance a real checkpoint loads without changing architecture or predictions when a strict load already works. If no checkpoint still loads, the same uniform fallback and probability normalization remain to guarantee a valid submission.'
- What this solution (achieved 1.40995) has done: 'Your current score is worse than the target (1.40995 vs 1.19360; lower is better), and the most likely reason is that you’re still ending up with near-uniform predictions due to either (a) no weights being found/loaded or (b) a mismatched checkpoint being partially loaded. I keep your exact model, feature construction, and inference loop, but make weight loading more reliable by (1) explicitly extracting the correct `state_dict` from common Lightning/torch formats, (2) allowing safe dtype/device normalization of tensors in the checkpoint, and (3) improving compatibility checks so a valid checkpoint is actually accepted and loaded. If no compatible checkpoint exists, the uniform fallback remains unchanged to guarantee a valid submission. Finally, I keep your strict probability normalization to satisfy Kaggle’s row-sum requirement and avoid invalid submissions.'
- What this solution (achieved 1.40995) has done: 'Your current score is worse than the target (1.40995 vs 1.1936; lower is better), and the biggest likely reason is still that you’re effectively producing weak predictions (often near-uniform) because no truly compatible checkpoint is getting loaded. I keep your exact model/dataset/inference logic, but make weight loading reliably succeed for more real checkpoints by (1) handling another common prefix (`"model.model."`) and (2) adding a minimal key-remap fallback for EfficientNet heads (`classifier.*` ↔ `custom_layers.2.*`) when shapes match, which doesn’t change architecture but helps load otherwise-compatible weights. I also broaden compatibility slightly (still strict on shapes) to accept checkpoints that match most of the backbone even if the head naming differs, then rely on the remap/filtered load to complete loading. This should move predictions away from uniform and improve KL toward your target while preserving the same core semantics and still falling back to uniform if nothing can be loaded.'

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


def _discover_weight_paths(primary_path: str) -> List[str]:
    """
    Change (score-toward-target): broaden discovery to include common checkpoint extensions
    (notably .ckpt used by Lightning) to increase chance we load real trained weights
    instead of falling back to uniform predictions (improves KL).
    """
    candidates: List[str] = []
    if primary_path and os.path.exists(primary_path):
        candidates.append(primary_path)

    preferred_substrings = ["hms-harmful-brain-activity-classification", "hms"]

    search_roots = ["/kaggle/input", "/kaggle/working"]
    patterns = [
        "efficient_net_weights.pt",
        "efficient*_weights*.pt",
        "*efficientnet*.pt",
        "*effnet*.pt",
        "*.pt",
        "*.pth",
        "*.bin",
        "*.ckpt",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pat in patterns:
            matches = glob(os.path.join(root, "**", pat), recursive=True)
            for m in matches:
                if os.path.isfile(m):
                    candidates.append(m)

    candidates = list(dict.fromkeys(candidates))

    def pref_rank(p: str) -> Tuple[int, int, int, str]:
        bn = os.path.basename(p).lower()
        in_comp_folder = 0 if any(s in p.lower() for s in preferred_substrings) else 1
        exact = 0 if bn == "efficient_net_weights.pt" else 1
        short = len(p)
        return (in_comp_folder, exact, short, p)

    candidates = sorted(candidates, key=pref_rank)
    return candidates


def _extract_state_dict(checkpoint):
    """
    Change (score-toward-target): robustly extract the real parameter dict from common formats
    (plain torch, Lightning, and nested wrappers). This increases chance we load real weights.
    """
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]

        for k in ["model_state_dict", "model", "net", "weights", "params"]:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                return checkpoint[k]

        for k in ["model", "net"]:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                inner = checkpoint[k]
                if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                    return inner["state_dict"]
                if "model_state_dict" in inner and isinstance(
                    inner["model_state_dict"], dict
                ):
                    return inner["model_state_dict"]

    return checkpoint


def _strip_prefixes(state_dict: dict) -> dict:
    """
    Change (score-toward-target): strip common training wrappers prefixes to increase load success
    without changing model architecture/semantics.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    out = state_dict

    prefixes = ["module.", "model.", "model.model."]

    for pref in prefixes:
        if any(isinstance(k, str) and k.startswith(pref) for k in out.keys()):
            out = {k.replace(pref, "", 1): v for k, v in out.items()}

    return out


def _coerce_state_dict_tensors(state_dict: dict) -> dict:
    """
    Change (score-toward-target): ensure checkpoint tensors are CPU float tensors.
    This avoids rare load failures due to device/dtype (e.g., fp16/bf16) mismatches,
    improving the probability that a real checkpoint loads and improves KL.
    """
    if not isinstance(state_dict, dict):
        return {}
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            vv = v.detach().cpu()
            if vv.dtype in (torch.float16, torch.bfloat16):
                vv = vv.float()
            out[k] = vv
        else:
            out[k] = v
    return out


def _remap_head_keys_for_custom_model(model: nn.Module, state_dict: dict) -> dict:
    """
    Change (score-toward-target): minimal key remap to load otherwise-compatible EfficientNet
    checkpoints whose classifier naming differs. This keeps architecture identical but increases
    chance weights load, moving predictions away from uniform (better KL).

    Supported remaps (only when shapes match model.state_dict()):
      - classifier.{weight,bias}  -> custom_layers.2.{weight,bias}
      - fc.{weight,bias}          -> custom_layers.2.{weight,bias}
    """
    if not isinstance(state_dict, dict):
        return {}
    msd = model.state_dict()
    out = dict(state_dict)

    def try_map(src_w, src_b, dst_w, dst_b):
        if src_w in out and dst_w in msd:
            if hasattr(out[src_w], "shape") and tuple(out[src_w].shape) == tuple(
                msd[dst_w].shape
            ):
                out[dst_w] = out[src_w]
        if src_b in out and dst_b in msd:
            if hasattr(out[src_b], "shape") and tuple(out[src_b].shape) == tuple(
                msd[dst_b].shape
            ):
                out[dst_b] = out[src_b]

    try_map(
        "classifier.weight",
        "classifier.bias",
        "custom_layers.2.weight",
        "custom_layers.2.bias",
    )
    try_map("fc.weight", "fc.bias", "custom_layers.2.weight", "custom_layers.2.bias")

    return out


def _filter_state_dict_to_model(model: nn.Module, state_dict: dict) -> dict:
    """
    Change (score-toward-target): keep only exact name+shape matches.
    This preserves core logic but increases chance a real checkpoint loads.
    """
    if not isinstance(state_dict, dict):
        return {}
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) == tuple(model_sd[k].shape):
                filtered[k] = v
    return filtered


def _state_dict_is_compatible(model: nn.Module, state_dict: dict) -> bool:
    """
    Change (score-toward-target): accept checkpoints that match a large fraction of params
    exactly by name+shape. This avoids mistakenly rejecting valid checkpoints due to a few
    extra non-model keys, improving chances to load real weights.

    Note: we still require strict shape matches on all intersecting keys; threshold slightly
    lowered to allow head key renames that will be handled by remap/filter later.
    """
    if not isinstance(state_dict, dict):
        return False
    model_sd = model.state_dict()
    intersect = [k for k in state_dict.keys() if k in model_sd]
    if len(intersect) < max(10, int(0.35 * len(model_sd))):
        return False
    for k in intersect:
        v = state_dict[k]
        mv = model_sd[k]
        if not hasattr(v, "shape") or not hasattr(mv, "shape"):
            return False
        if tuple(v.shape) != tuple(mv.shape):
            return False
    return True


def _weight_preference_score(
    weight_path: str, state_dict: dict
) -> Tuple[int, int, int, int, str]:
    """
    Change (score-toward-target): deterministic selection of most likely correct checkpoint.
    """
    bn = os.path.basename(weight_path).lower()
    p = weight_path.lower()

    in_comp = (
        0 if ("hms-harmful-brain-activity-classification" in p or "/hms" in p) else 1
    )
    exact = 0 if bn == "efficient_net_weights.pt" else 1
    model_hint = (
        0 if ("efficientnet" in p or "effnet" in p or "tf_efficientnet_b4" in p) else 1
    )

    has_head = 1
    if isinstance(state_dict, dict):
        head_keys = [
            "custom_layers.2.weight",
            "custom_layers.2.bias",
            "classifier.weight",
            "classifier.bias",
            "fc.weight",
            "fc.bias",
        ]
        has_head = 0 if any(k in state_dict for k in head_keys) else 1

    return (in_comp, exact, model_hint, has_head, weight_path)


model_weights = _discover_weight_paths(paths.MODEL_WEIGHTS)
print(
    "Discovered weight candidates:", model_weights[:10], f"(total={len(model_weights)})"
)

if len(model_weights) == 0:
    model_weights = [paths.MODEL_WEIGHTS]




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
label_cols = TARGETS


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

if config.VISUALIZE:
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
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
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

        if self.mode == "test":
            r = 0
        else:
            r = int((row["min"] + row["max"]) // 4)

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
def _safe_row_normalize(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum(axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    probs = probs / s
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)




## === cell 11
predictions = []

successful_preds = []
successful_weight_paths: List[str] = []

compatible_ranked: List[Tuple[Tuple[int, int, int, int, str], str]] = []
_probe_model = CustomModel(config)
for wp in model_weights:
    if not os.path.exists(wp):
        continue
    try:
        ckpt = torch.load(wp, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        sd = _strip_prefixes(sd)
        sd = _coerce_state_dict_tensors(sd)

        sd = _remap_head_keys_for_custom_model(_probe_model, sd)

        if _state_dict_is_compatible(_probe_model, sd):
            score = _weight_preference_score(wp, sd)
            compatible_ranked.append((score, wp))
    except Exception:
        continue
del _probe_model
gc.collect()

compatible_ranked.sort(key=lambda x: x[0])
compatible_weights = [wp for _, wp in compatible_ranked]

print(f"Compatible weight files found: {len(compatible_weights)}")
if len(compatible_weights) > 0:
    print("Top compatible candidates:", compatible_weights[:5])

model_weights_to_try = (
    compatible_weights[:3] if len(compatible_weights) > 0 else model_weights[:1]
)
print(f"Trying {len(model_weights_to_try)} checkpoint candidate(s).")

for model_weight in model_weights_to_try:
    if not os.path.exists(model_weight):
        continue

    print(f"Loading weights from: {model_weight}")
    model = CustomModel(config)
    try:
        checkpoint = torch.load(model_weight, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        state_dict = _strip_prefixes(state_dict)
        state_dict = _coerce_state_dict_tensors(state_dict)

        state_dict = _remap_head_keys_for_custom_model(model, state_dict)

        if not _state_dict_is_compatible(model, state_dict):
            raise ValueError("Incompatible state_dict for this CustomModel")

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            filtered = _filter_state_dict_to_model(model, state_dict)
            model.load_state_dict(filtered, strict=False)

        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        successful_preds.append(prediction_dict["predictions"])
        successful_weight_paths.append(model_weight)

        break

    except Exception as e:
        print(f"WARNING: failed to use {model_weight}: {type(e).__name__}: {e}")
    finally:
        try:
            del model, checkpoint, state_dict
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

if len(successful_preds) == 0:
    print(
        "WARNING: no valid weights could be loaded for inference. "
        "Falling back to uniform probabilities to produce a valid submission."
    )
    predictions = np.full(
        (len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32
    )
else:
    print(f"Using predictions from {len(successful_preds)} checkpoint(s).")
    print("Used weight:", successful_weight_paths[0])
    predictions = successful_preds[0].astype(np.float32)

predictions = _safe_row_normalize(predictions)
print("Predictions shape:", predictions.shape)




## === cell 12
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

row_sums = sub[TARGETS].sum(axis=1).values
if not np.allclose(row_sums, 1.0, atol=1e-4):
    sub[TARGETS] = _safe_row_normalize(sub[TARGETS].values)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")
print(f"Submission shape: {sub.shape}")
sub.head()
