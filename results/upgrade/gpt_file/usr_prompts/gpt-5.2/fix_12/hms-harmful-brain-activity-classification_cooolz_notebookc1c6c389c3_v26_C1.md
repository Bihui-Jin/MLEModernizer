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

0.4568878693320788

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing-weights path issue by falling back to a safe, score-neutral baseline (uniform probabilities) when the expected model directories aren’t present, so the notebook always produces a valid `submission.csv`. I also make the code robust to CPU-only environments by removing hard-coded `.to('cuda')` in preprocessing and gating heavy EEG-spec precomputation/inference behind “weights available” checks. Finally, I fix a couple of logic bugs that would break shapes/keys (wrong brain lead list and inconsistent dict keys) and enforce row-wise probability normalization to satisfy the submission constraint. These changes preserve the intended ensemble inference flow when weights exist, but avoid runtime errors and always yield a valid CSV.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4569), and the biggest reason is that your code likely fell back to uniform predictions because the specified weight directories don’t exist in this runtime. I make a minimal, score-relevant change to actually load weights from any available local directories (including `/kaggle/input`), without changing the model architectures or inference logic. Additionally, I make the EEG-spec precompute conditional on having mix weights (as you intended) and ensure `DataLoader` doesn’t crash on Kaggle by clamping workers to what’s available. These changes should move the score significantly toward the target by enabling real model inference instead of a uniform baseline.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), and the most likely cause is that you’re not actually loading the intended competition weights (so you effectively submit near-uniform predictions). I make minimal, score-relevant fixes to (1) correctly locate weight files inside `/kaggle/input` (including common `.pt/.pth` layouts), and (2) robustly load checkpoints that wrap the state dict under keys like `state_dict`, `model`, or `model_state_dict` (without changing any model code). I also make sure the mix-spec precompute runs when mix weights exist and the lookup doesn’t KeyError by falling back to an on-the-fly EEG mel-spec for missing ids. These changes preserve your ensemble inference logic and should move the score substantially toward your target by ensuring real model inference happens.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the most likely reason is that the script is still not actually loading the intended trained weights (so you’re effectively predicting close to uniform or random-init). I make minimal, score-relevant fixes to (1) prefer checkpoints whose tensor shapes match the current model (avoids averaging in broken loads), and (2) make state-dict key cleaning safer (only strip prefixes when appropriate), which increases the chance of a correct load without changing any model/inference logic. I also add a tiny uniform-mixture “prior smoothing” (very small) to reduce extreme probabilities and improve KL stability; this keeps semantics (probabilities) but is often beneficial for KL. The submission writing and row-normalization remain unchanged and guaranteed.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), which strongly suggests the script is still not loading any meaningful trained weights and/or is skipping most checkpoints due to the strict shape-match filter, so predictions collapse to near-uniform. I make the smallest score-relevant changes to (1) improve weight autodiscovery by prioritizing common “best/fold” checkpoint names and the exact competition dataset subdir, and (2) relax the checkpoint acceptance gate from a strict ratio-based skip to a “try-load then accept if it actually changed many parameters,” which preserves your model/inference logic but greatly increases the chance real weights get used. Finally, I slightly reduce the uniform prior smoothing (still keeping it for KL stability) so it doesn’t wash out the model signal once weights load. These changes keep architecture/training/inference semantics the same and should move the score materially toward your target.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the most likely remaining cause is still “no meaningful weights loaded,” either because `hrnet_w18` head shapes don’t match (your `NetMix` uses a fixed `Linear(2048,6)`), or because the overlap gate skips otherwise-loadable checkpoints. I make two minimal, score-relevant fixes: (1) make `NetMix` infer its feature dimension with a tiny dummy forward and build `fc` accordingly (preserves the same backbone and head concept, but prevents silent mis-load/skip due to wrong `in_features`), and (2) relax the checkpoint skip rule to allow `strict=False` loads even with low overlap, while still skipping obvious non-model artifacts (no tensor keys). Finally, I keep your probability smoothing/normalization but slightly reduce smoothing so it doesn’t wash out signal once real weights load.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the most likely reason is that you’re still averaging in many “wrong” checkpoints (or partially-loaded weights) because everything is loaded with `strict=False`, which can silently yield near-random heads and harm KL. I make a minimal, score-relevant change to only accept checkpoints that have a reasonable key+shape overlap with the current model (so we keep genuine trained weights and skip mismatched artifacts), while keeping the same models, transforms, and ensemble inference flow. I also ensure the `NetMix` dummy forward used to infer `in_features` runs on CPU and in `eval()` to avoid accidental device/BN issues during initialization. Finally, I keep your probability smoothing/normalization unchanged so submission validity and KL stability are preserved.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower is better), so we should push performance up with minimal, score-relevant changes. The biggest likely issue is that your overlap filter is skipping most real checkpoints (especially for EfficientNet where feature dims vary), causing predictions to fall back toward uniform; I relax the acceptance criterion to require sufficient *shape-matched* keys rather than a high ratio over whatever keys exist in the file. I also compute the `fc` input feature size for `NetSpec` and `NetEeg` via a tiny dummy forward (as you already do for `NetMix`) so their heads match common saved checkpoints and stop being skipped. Finally, I reduce the uniform prior smoothing slightly so it doesn’t wash out model signal once weights are correctly loaded, while keeping normalization/clipping to satisfy the KL metric and submission constraints.'
- What this solution (achieved 1.40995) has done: 'The current gap to the target is large (1.40995 vs 0.4569, lower-is-better), and the most likely cause is still “no (or wrong) weights loaded”, so the ensemble effectively behaves close to uniform. I make a minimal, score-relevant change to the checkpoint acceptance logic so we don’t skip valid checkpoints just because the model is large (your current absolute `ok < 80` gate is too strict for some backbones and can reject correct weights). I also restrict each weight list to the most relevant few checkpoints (first N after your existing sorting) so we avoid averaging in many partially-mismatched or auxiliary checkpoints that can wash out signal and hurt KL. Finally, I keep your probability smoothing/normalization semantics but reduce the smoothing slightly so it doesn’t overpower model outputs once real weights are used.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4569), and the most likely reason is still that you are not loading any real trained checkpoints (so you end up near-uniform). I make two minimal, score-relevant changes: (1) improve weight auto-discovery to also include Kaggle “working” and any `.ckpt` files (common in Lightning) so real weights are found, and (2) make checkpoint loading accept Lightning-style `state_dict` keys by stripping the common `model.`/`net.`/`backbone.` prefixes when present. I also set inference-time dropout to deterministic by disabling the `self.dropout` modules (keeping the architecture the same but preventing accidental train-mode randomness) to stabilize and usually reduce KL. The rest of your ensemble, transforms, and probability normalization remain unchanged so evaluation semantics stay the same and a valid `submission.csv` is always produced.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), which strongly suggests most or all checkpoints are still being skipped and you’re effectively near-uniform. I make the smallest score-relevant changes to (1) improve weight autodiscovery so it actually finds likely model checkpoints under `/kaggle/input` for this competition, and (2) adjust the checkpoint acceptance gate to prefer shape-matched keys (instead of a too-strict ratio vs total model keys) so valid checkpoints aren’t rejected. I also ensure inference truly runs in eval mode without forcing dropout probabilities to zero (which can unintentionally mismatch training-time behavior), and keep your probability smoothing/normalization and submission writing unchanged so the CSV stays valid.'

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




## === cell 2
def list_weight_files(root: str):
    if (root is None) or (not isinstance(root, str)) or (not os.path.exists(root)):
        return []
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".pt", ".pth", ".bin", ".ckpt")):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _sort_ckpts(files):
    """
    Score-relevant: prefer 'best/final/fold' checkpoints first, increasing the chance
    we actually load trained weights instead of random auxiliary artifacts.
    """
    if not files:
        return files

    def key(fp):
        name = os.path.basename(fp).lower()
        pri = 0
        if any(s in name for s in ["best", "final", "last"]):
            pri -= 20
        if any(s in name for s in ["fold"]):
            pri -= 10
        if any(s in name for s in ["epoch"]):
            pri -= 5
        if any(s in name for s in ["ema"]):
            pri -= 2
        if any(s in name for s in ["optimizer", "sched", "scheduler"]):
            pri += 10
        return (pri, len(name), name)

    return sorted(files, key=key)


def _autodiscover_weights():
    """
    Score-relevant: expand discovery to common Kaggle dataset layouts, but prioritize:
    - official competition dataset subtree
    - directories containing 'hms' + ('baseline'/'eeg'/'mix') keywords
    This increases probability we find real trained checkpoints (vs random unrelated weights).
    """
    roots = []
    for base in ["/kaggle/input", "/kaggle/working"]:
        if os.path.exists(base):
            comp = os.path.join(base, "hms-harmful-brain-activity-classification")
            if os.path.exists(comp):
                roots.append(comp)
            roots.append(base)

    if not roots:
        return {"spec": [], "eeg": [], "mix": []}

    spec_kw = ("spec", "baseline", "spectro", "spectrogram")
    eeg_kw = ("eeg", "raw", "wave")
    mix_kw = ("mix", "fusion", "combo")

    spec, eeg, mix = [], [], []
    all_w = []

    def looks_hms_related(path: str) -> bool:
        p = path.lower()
        return ("hms" in p) or ("harmful" in p) or ("brain" in p) or ("eeg" in p)

    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if not looks_hms_related(dirpath):
                continue
            ldir = dirpath.lower()
            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith((".pt", ".pth", ".bin", ".ckpt")):
                    continue
                fp = os.path.join(dirpath, fn)
                all_w.append(fp)
                hay = lfn + " " + ldir
                if any(k in hay for k in mix_kw):
                    mix.append(fp)
                elif any(k in hay for k in eeg_kw):
                    eeg.append(fp)
                elif any(k in hay for k in spec_kw):
                    spec.append(fp)

    all_w = _sort_ckpts(sorted(set(all_w)))
    spec = _sort_ckpts(sorted(set(spec)))
    eeg = _sort_ckpts(sorted(set(eeg)))
    mix = _sort_ckpts(sorted(set(mix)))

    if (len(spec) + len(eeg) + len(mix)) == 0 and len(all_w) > 0:
        spec = all_w

    return {"spec": spec, "eeg": eeg, "mix": mix}


CFG["weights_spec"] = _sort_ckpts(list_weight_files(CFG["weights_spec"]))
CFG["weights_eeg"] = _sort_ckpts(list_weight_files(CFG["weights_eeg"]))
CFG["weights_mix"] = _sort_ckpts(list_weight_files(CFG["weights_mix"]))

if (len(CFG["weights_spec"]) + len(CFG["weights_eeg"]) + len(CFG["weights_mix"])) == 0:
    found = _autodiscover_weights()
    CFG["weights_spec"] = found["spec"]
    CFG["weights_eeg"] = found["eeg"]
    CFG["weights_mix"] = found["mix"]

CFG["weights_spec"] = CFG["weights_spec"][:5]
CFG["weights_eeg"] = CFG["weights_eeg"][:5]
CFG["weights_mix"] = CFG["weights_mix"][:5]

CFG["num_worker"] = int(max(0, min(CFG["num_worker"], os.cpu_count() or 2)))

print(
    "Weights found (capped to top-5 each):",
    {
        "spec": len(CFG["weights_spec"]),
        "eeg": len(CFG["weights_eeg"]),
        "mix": len(CFG["weights_mix"]),
    },
)
if len(CFG["weights_spec"]) > 0:
    print("Example spec weight:", CFG["weights_spec"][0])
if len(CFG["weights_eeg"]) > 0:
    print("Example eeg weight:", CFG["weights_eeg"][0])
if len(CFG["weights_mix"]) > 0:
    print("Example mix weight:", CFG["weights_mix"][0])
CFG



## === cell 3
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from tqdm import tqdm

data_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, x):
        return self.wave_transform(x)


transform_func = MelTransform().to(_DEVICE)


def spectrogram_from_eeg(parquet_path, display=False):
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

            x_tensor = torch.from_numpy(x).to(_DEVICE)
            mel_spec = transform_func(x_tensor)
            mel_spec = mel_spec.detach().to("cpu").numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]

            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

    return img


if len(CFG["weights_mix"]) > 0:
    all_fs = os.listdir(data_dir)
    all_specs = {}

    for item in tqdm(all_fs, desc="Precompute EEG mel specs"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        eeg_spec = spectrogram_from_eeg(eeg_path)
        all_specs[str(eeg_id)] = eeg_spec  # ensure str keys for consistent lookup

    with open("eeg_specs_dict.pkl", "wb") as file:
        pickle.dump(all_specs, file)
else:
    if not os.path.exists("eeg_specs_dict.pkl"):
        with open("eeg_specs_dict.pkl", "wb") as file:
            pickle.dump({}, file)




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
        print(self.ll, self.rr)
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None
        self.df = df

        self.train_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )

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
        eeg_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp["eeg_id"]}.parquet'
        eeg = pd.read_parquet(eeg_path)

        offset = 0
        eeg = eeg.iloc[int(offset * 200) : int(offset * 200) + 10000]

        waves = eeg.values
        waves = np.transpose(waves, axes=[1, 0])

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i, :], nan=m)
            else:
                waves[i] = 0

        waves = np.array(waves, dtype=np.float64)
        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        waves = self.brain_lead(waves)
        return waves

    def get_spec(self, dp, is_training):
        spec_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp["spectrogram_id"]}.parquet'
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

        images = np.stack(images, -1)
        data = np.transpose(images, [2, 0, 1])
        return data

    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype="float32")
        kg_spec = self.get_spec(dp, is_training=False)

        eeg_id_str = str(dp["eeg_id"])
        if eeg_id_str in self.eeg_specs:
            eeg_spec = self.eeg_specs[eeg_id_str]
        else:
            eeg_path = f'/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp["eeg_id"]}.parquet'
            eeg_spec = spectrogram_from_eeg(eeg_path)

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
            raise ValueError("One of use_eeg/use_spec/use_mix must be True")

        return data.astype(np.float32)




## === cell 5
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 400, device="cpu")
            feat = self.model.forward_features(dummy)
            feat = self.avg(feat).view(1, -1)
            in_features = int(feat.shape[1])
        if was_training:
            self.model.train()

        self.fc = nn.Linear(in_features, 6, bias=True)

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
        image = image[:, :, : int(40 / 100 * h + 5), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 160, 320, device="cpu")
            feat = self.model.forward_features(dummy)
            feat = self.avg(feat).view(1, -1)
            in_features = int(feat.shape[1])
        if was_training:
            self.model.train()

        self.fc = nn.Linear(in_features, 6, bias=True)

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

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 512, device="cpu")
            feat = self.model.forward_features(dummy)
            feat = self.avg(feat).view(1, -1)
            in_features = int(feat.shape[1])
        if was_training:
            self.model.train()

        self.fc = nn.Linear(in_features, 6, bias=True)

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




## === cell 8
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for _, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().to("cpu").numpy())

    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 9
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
n_test = len(test_df)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

predictions_accum = []


def _extract_state_dict(ckpt):
    """
    Score-relevant robustness: handle common checkpoint wrappers so we can load trained weights.
    """
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "ema", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return ckpt


def _clean_state_dict_keys(sd, model_keys=None):
    """
    Score-relevant: strip common Lightning/wrapper prefixes only if it increases matches.
    """
    if not isinstance(sd, dict):
        return sd
    if model_keys is None:
        model_keys = set()

    prefixes = ["module.", "model.", "net.", "backbone.", "encoder."]

    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                cand = nk[len(p) :]
                if (cand in model_keys) and (nk not in model_keys):
                    nk = cand
                    break
        out[nk] = v
    return out


def _state_dict_overlap(model, sd):
    """
    Score-relevant: estimate whether the checkpoint is intended for this model by counting
    matching keys with matching shapes.
    """
    if not isinstance(sd, dict):
        return 0, 0
    msd = model.state_dict()
    ok = 0
    tot = 0
    for k, v in sd.items():
        if k in msd and torch.is_tensor(v) and torch.is_tensor(msd[k]):
            tot += 1
            if tuple(v.shape) == tuple(msd[k].shape):
                ok += 1
    return ok, tot


def _has_any_tensor(sd):
    if isinstance(sd, dict):
        return any(torch.is_tensor(v) for v in sd.values())
    return False


def _run_ensemble(weights, build_dataset_fn, build_model_fn):
    if len(weights) == 0:
        return None
    local_preds = []
    for model_weight in weights:
        ds = build_dataset_fn()
        dl = DataLoader(
            ds,
            batch_size=CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_model_fn()

        ckpt = torch.load(model_weight, map_location="cpu")
        raw_sd = _extract_state_dict(ckpt)

        if not _has_any_tensor(raw_sd):
            print(f"Skip {os.path.basename(model_weight)} (no tensors found).")
            del model, ckpt, raw_sd, dl, ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model_keys = set(model.state_dict().keys())
        state_dict = _clean_state_dict_keys(raw_sd, model_keys=model_keys)

        ok, tot = _state_dict_overlap(model, state_dict)

        ok_over_ckpt = ok / max(1, tot)
        if ok < 50 or ok_over_ckpt < 0.20:
            print(
                f"Skip {os.path.basename(model_weight)} due to low shape-match "
                f"(ok={ok}, ok/ckpt={ok_over_ckpt:.3f}, model_keys={len(model_keys)}, tot_in_ckpt={tot})."
            )
            del model, ckpt, state_dict, raw_sd, dl, ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded {os.path.basename(model_weight)}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, ok_shape={ok}, ok/ckpt={ok_over_ckpt:.3f}, model_keys={len(model_keys)}"
        )

        model.to(device)

        pred = inference_function(dl, model, device)["predictions"]
        local_preds.append(pred)

        del model, ckpt, state_dict, raw_sd, dl, ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if len(local_preds) == 0:
        return None

    local_preds = np.mean(np.array(local_preds), axis=0)
    return local_preds


print("infer with weights_eeg:", len(CFG["weights_eeg"]))
pred_eeg = _run_ensemble(
    CFG["weights_eeg"],
    build_dataset_fn=lambda: AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=True, ll=0, rr=40
    ),
    build_model_fn=lambda: NetEeg(),
)
if pred_eeg is not None:
    predictions_accum.append(pred_eeg)

print("infer with weights_spec:", len(CFG["weights_spec"]))
pred_spec = _run_ensemble(
    CFG["weights_spec"],
    build_dataset_fn=lambda: AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_spec=True
    ),
    build_model_fn=lambda: NetSpec(),
)
if pred_spec is not None:
    predictions_accum.append(pred_spec)

print("infer with weights_mix:", len(CFG["weights_mix"]))
pred_mix = _run_ensemble(
    CFG["weights_mix"],
    build_dataset_fn=lambda: AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_mix=True
    ),
    build_model_fn=lambda: NetMix(),
)
if pred_mix is not None:
    predictions_accum.append(pred_mix)

if len(predictions_accum) == 0:
    predictions = np.full((n_test, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
else:
    predictions = np.mean(np.array(predictions_accum), axis=0).astype(np.float32)

prior_eps = 0.0002
uniform = np.full_like(predictions, 1.0 / predictions.shape[1], dtype=np.float32)
predictions = (1.0 - prior_eps) * predictions + prior_eps * uniform

predictions = np.clip(predictions, 1e-7, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

predictions.shape



## === cell 11
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sum (min/max):", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
