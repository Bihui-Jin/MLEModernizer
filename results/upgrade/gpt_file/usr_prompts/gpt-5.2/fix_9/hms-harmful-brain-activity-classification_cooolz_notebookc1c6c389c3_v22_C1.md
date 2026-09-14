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

0.3991544337423225

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix three execution blockers while keeping your modeling logic intact: (1) make weight discovery robust when the referenced `/kaggle/input/hms-*` folders don’t exist by falling back to a safe uniform-probability submission (so you still get a valid `.csv`), (2) remove hard CUDA assumptions so the notebook runs on CPU-only environments, and (3) fix directory/listing issues (filter to real weight files) plus ensure prediction array shape is always `(N, 6)` and rows sum to 1. If no weights are found, the code skip inference and directly write a valid calibrated (uniform) submission. If weights are found, it run your exact inference pipeline and then write the submission.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 is far worse than the 0.399 target (lower is better), and the main reason is that your script is likely falling back to uniform predictions because the weight paths point to non-existent `/kaggle/input/hms-baseline`, `/kaggle/input/hms-eeg`, `/kaggle/input/hms-mix`. I minimally change only the weight-path discovery so it also searches the provided dataset directory (`/kaggle/input/hms-harmful-brain-activity-classification/`) and common subfolders for `.pt/.pth/.bin` files, enabling your existing inference logic to actually run. I also include spec-weights inference (your `NetSpec` exists but wasn’t used), which is a small, direct improvement consistent with your ensemble approach and KL metric. Everything else (models, preprocessing, inference, submission format/normalization) remains intact.'
- What this solution (achieved 1.40995) has done: 'Your current score strongly suggests the code is still effectively producing near-uniform predictions (or otherwise uninformative ones), so the smallest reliable path toward the 0.399 target is to ensure your inference actually loads compatible weights and doesn’t silently mis-load checkpoints. I minimally add a robust checkpoint loader that correctly extracts `state_dict` from common Kaggle checkpoint formats (e.g., `{"state_dict":...}`, `{"model":...}`) and strips `module.` prefixes, which often prevents “random-weight inference” even when files exist. I also make weight discovery prioritize likely files (by name) and cap the number loaded to keep runtime under 600s, without changing any model architecture or preprocessing. Finally, I keep your normalization/sum-to-1 guarantees intact so the submission is always valid.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) strongly suggests the pipeline is still effectively producing weak/uniform-like predictions, likely because (a) no compatible weights are being found/loaded or (b) checkpoints are being averaged even when they don’t match the model (leading to random-init outputs). I keep your architecture and inference logic unchanged, but make weight discovery search only plausible checkpoint locations and add a strict compatibility gate so we only ensemble checkpoints that actually load with near-zero missing keys (otherwise skip them instead of polluting the mean). If no compatible weights remain, we still fall back to a valid uniform submission; otherwise we should move substantially toward the 0.399 target by ensuring we’re using real trained weights. I also make DataLoader workers safe for Kaggle CPU environments to avoid silent hangs/timeouts that can prevent proper inference.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the 0.399 target, and the most likely cause is that you’re still not loading any real compatible checkpoints, so predictions remain uniform/weak. I make the smallest change that directly increases the chance of finding and correctly loading actual weight files by expanding the fallback search roots to include the competition dataset directory (not just `/kaggle/input`) and tightening discovery to likely subfolders/files. I also ensure we don’t accidentally include non-weight artifacts by filtering with a slightly stricter filename heuristic while keeping your ensemble/inference logic unchanged. This should move the score toward the target by making your existing trained models actually run instead of falling back to uniform outputs.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the 0.399 target, and the most likely reason is that inference is still not using the intended trained weights (so outputs are near-uniform/weak). I make weight discovery actually look in the competition dataset for common checkpoint patterns, and I also accept weights stored under `model_ema` (a frequent format) so more real checkpoints load successfully. To avoid “poisoning” the ensemble with wrong-architecture checkpoints, I keep your compatibility gate but make it stricter in a safe way and add a sanity check that predicted rows sum to 1 and have finite values. Core model classes, preprocessing, inference loop, and submission schema remain unchanged.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the 0.399 target, and the most likely reason is that your “weights discovery” still finds zero (or wrong) checkpoints so you fall back to uniform predictions. I make a minimal, targeted change: tighten weight discovery to search only realistic checkpoint subfolders within the provided dataset directory and, crucially, automatically relax the compatibility gate when it would otherwise reject all weights (so we don’t end up uniform). I also add a safer checkpoint loader path that tries both full-path and `*.pt/*.pth` files under the discovered directories and logs how many checkpoints were actually used. Everything else—model classes, preprocessing, inference loop, and submission normalization—remains unchanged.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the target (0.399), and the most likely cause is that you are still either (a) not loading any real trained checkpoints, or (b) loading incompatible checkpoints that produce essentially random/uniform-like outputs. I keep your model classes and inference loop intact, but (1) broaden weight discovery to also search recursively under the competition dataset directory for plausible checkpoint filenames, and (2) make checkpoint compatibility checking safer by allowing “head-only mismatch” (common when fc layer names differ) while still rejecting truly wrong architectures. This should move predictions away from uniform/random toward meaningful probabilities, improving KL substantially. I also ensure we never average predictions with wrong row counts and always produce a valid `submission.csv` with row-sum-to-1 probabilities.'

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
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
    "max_missing_keys_ratio": 0.005,
    "max_unexpected_keys_ratio": 0.02,
    "force_num_workers": None,  # set to 0 to force single-process loading; None = auto
    "max_weight_files_per_family": 8,
    "compat_fallback_max_missing_keys_ratio": 0.25,
    "compat_fallback_max_unexpected_keys_ratio": 0.25,
    "head_only_missing_key_frac_thr": 0.90,
    "head_only_unexpected_key_frac_thr": 0.90,
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def list_weight_files(path):
    if not isinstance(path, str) or (not os.path.exists(path)):
        return []
    if os.path.isfile(path):
        return [path]
    files = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp) and fn.lower().endswith((".pt", ".pth", ".bin", ".ckpt")):
            files.append(fp)
    return files


def _weight_priority_key(fp: str) -> tuple:
    fn = os.path.basename(fp).lower()
    score = 0
    for kw, w in [
        ("best", 80),
        ("final", 50),
        ("fold", 25),
        ("epoch", 10),
        ("checkpoint", 10),
        ("ckpt", 10),
        ("ema", 12),
    ]:
        if kw in fn:
            score += w
    if "loss" in fn or "kl" in fn:
        score += 10
    return (-score, fn)


def _looks_like_weight_filename(fp: str) -> bool:
    """
    NOTE (score improvement): filter out common non-model artifacts that can exist in dataset trees.
    """
    fn = os.path.basename(fp).lower()
    bad_kw = [
        "token",
        "vocab",
        "config",
        "args",
        "opt",
        "optimizer",
        "sched",
        "scheduler",
        "augment",
        "albument",
        "labels",
        "submission",
        "log",
        "events",
    ]
    if any(k in fn for k in bad_kw):
        return False
    good_kw = [
        "model",
        "fold",
        "epoch",
        "best",
        "final",
        "ema",
        "checkpoint",
        "ckpt",
        "weights",
    ]
    if not any(k in fn for k in good_kw):
        return False
    return fn.endswith((".pt", ".pth", ".ckpt", ".bin"))


def discover_weight_files(primary_path, fallback_roots, max_files=8):
    """
    NOTE (score improvement): If the original folders don't exist, search inside the provided
    competition dataset directory recursively for plausible checkpoint names. This directly increases
    the chance we actually run trained inference instead of uniform fallback (major KL gain).
    """
    files = [
        fp for fp in list_weight_files(primary_path) if _looks_like_weight_filename(fp)
    ]
    if len(files) > 0:
        files = sorted(files, key=_weight_priority_key)
        return files[:max_files]

    found = []
    for root in fallback_roots:
        if not isinstance(root, str) or (not os.path.exists(root)):
            continue

        preferred_subs = [
            root,
            os.path.join(root, "models"),
            os.path.join(root, "model"),
            os.path.join(root, "weights"),
            os.path.join(root, "checkpoints"),
            os.path.join(root, "checkpoint"),
            os.path.join(root, "ckpt"),
            os.path.join(root, "outputs"),
            os.path.join(root, "output"),
            os.path.join(root, "train"),
            os.path.join(root, "exp"),
            os.path.join(root, "experiment"),
        ]
        for base in preferred_subs:
            if not os.path.exists(base) or not os.path.isdir(base):
                continue
            for fn in os.listdir(base):
                fp = os.path.join(base, fn)
                if os.path.isfile(fp) and fp.lower().endswith(
                    (".pt", ".pth", ".bin", ".ckpt")
                ):
                    if _looks_like_weight_filename(fp):
                        found.append(fp)

        walked = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".pt", ".pth", ".bin", ".ckpt")):
                    fp = os.path.join(dirpath, fn)
                    if _looks_like_weight_filename(fp):
                        found.append(fp)
            walked += 1
            if walked >= 350:  # bound traversal for <600s runtime
                break

        if len(found) >= max_files * 40:
            break

    found = sorted(set(found), key=_weight_priority_key)
    return found[:max_files]


FALLBACK_WEIGHT_ROOTS = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification",
    "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    "/kaggle/input",
    "/kaggle/data",
]

CFG["weights_spec"] = discover_weight_files(
    CFG["weights_spec"],
    FALLBACK_WEIGHT_ROOTS,
    max_files=CFG["max_weight_files_per_family"],
)
CFG["weights_eeg"] = discover_weight_files(
    CFG["weights_eeg"],
    FALLBACK_WEIGHT_ROOTS,
    max_files=CFG["max_weight_files_per_family"],
)
CFG["weights_mix"] = discover_weight_files(
    CFG["weights_mix"],
    FALLBACK_WEIGHT_ROOTS,
    max_files=CFG["max_weight_files_per_family"],
)

print("Discovered spec weights:", [os.path.basename(x) for x in CFG["weights_spec"]])
print("Discovered eeg  weights:", [os.path.basename(x) for x in CFG["weights_eeg"]])
print("Discovered mix  weights:", [os.path.basename(x) for x in CFG["weights_mix"]])



## === cell 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
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


transform_func_mel = TransformMel().to(device)


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

            x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
            mel_spec = transform_func_mel(x_tensor)
            mel_spec = mel_spec.detach().cpu().numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

    return img




## === cell 3
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

        self.ll = 0
        self.rr = rr
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

        self.eeg_specs = None
        if self.use_mix:
            if os.path.exists("eeg_specs_dict.pkl"):
                with open("eeg_specs_dict.pkl", mode="rb") as f:
                    self.eeg_specs = pickle.load(f)
            else:
                self.eeg_specs = None

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RP]  # preserving original logic

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
        eeg_path = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
            % (dp["eeg_id"])
        )
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

        waves = np.array(waves, dtype=np.float64)
        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        waves = self.brain_lead(waves)
        return waves

    def get_spec(self, dp, is_training):
        spec_path = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet"
            % (dp["spectrogram_id"])
        )
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

        if self.eeg_specs is not None and str(dp["eeg_id"]) in self.eeg_specs:
            eeg_spec = self.eeg_specs[str(dp["eeg_id"])]
        else:
            eeg_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
                % (dp["eeg_id"])
            )
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
            raise ValueError("One of use_eeg/use_spec/use_mix must be True.")
        return data.astype(np.float32)




## === cell 4
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

        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 5
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
        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 6
class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("hrnet_w18", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
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
        x = x.view((2 * bs if CFG["flip"] else bs), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.softmax(x, -1)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 7
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().cpu().numpy())
    prediction_dict = {
        "predictions": (
            np.concatenate(preds, axis=0)
            if len(preds)
            else np.zeros((0, 6), dtype=np.float32)
        )
    }
    return prediction_dict


def normalize_probs(p, eps=1e-8):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "model_ema",
            "net",
            "weights",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        looks_like_sd = any(isinstance(v, torch.Tensor) for v in ckpt.values())
        if looks_like_sd:
            return ckpt
    return ckpt


def load_weights_robust(model, weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)
    sd = _extract_state_dict(ckpt)
    if isinstance(sd, dict):
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def _is_head_key(k: str) -> bool:
    k = str(k)
    head_markers = ("fc.", "classifier", "head", "last_linear", "logits", "final")
    return any(m in k for m in head_markers)


def _is_checkpoint_compatible(
    model, missing, unexpected, miss_ratio_thr, unexp_ratio_thr
):
    """
    NOTE (score improvement): Many checkpoints differ only in classifier head naming. Accepting those
    avoids rejecting good weights and falling back to uniform/random predictions (large KL harm).
    We still reject when mismatches are widespread (wrong backbone/architecture).
    """
    total_keys = max(1, len(model.state_dict()))
    miss_ratio = len(missing) / total_keys
    unexp_ratio = len(unexpected) / total_keys
    if (miss_ratio <= miss_ratio_thr) and (unexp_ratio <= unexp_ratio_thr):
        return True

    if len(missing) > 0:
        miss_head_frac = sum(_is_head_key(k) for k in missing) / len(missing)
    else:
        miss_head_frac = 1.0
    if len(unexpected) > 0:
        unexp_head_frac = sum(_is_head_key(k) for k in unexpected) / len(unexpected)
    else:
        unexp_head_frac = 1.0

    if (miss_head_frac >= CFG["head_only_missing_key_frac_thr"]) and (
        unexp_head_frac >= CFG["head_only_unexpected_key_frac_thr"]
    ):
        return (miss_ratio <= CFG["compat_fallback_max_missing_keys_ratio"]) and (
            unexp_ratio <= CFG["compat_fallback_max_unexpected_keys_ratio"]
        )

    return False


def _effective_num_workers():
    if CFG["force_num_workers"] is not None:
        return int(CFG["force_num_workers"])
    return CFG["num_worker"] if torch.cuda.is_available() else 0




## === cell 8
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 9
have_any_weights = (
    len(CFG["weights_eeg"]) + len(CFG["weights_mix"]) + len(CFG["weights_spec"])
) > 0

predictions_list = []
num_workers = _effective_num_workers()

strict_used = 0
relaxed_used = 0


def _run_family(weights, family_name, model_ctor, dataset_kwargs):
    global strict_used, relaxed_used
    fam_preds = []

    for model_weight in weights:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, **dataset_kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG["batch_size"],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = model_ctor()
        model.to(device)
        missing, unexpected = load_weights_robust(model, model_weight, device)

        compatible = _is_checkpoint_compatible(
            model,
            missing,
            unexpected,
            CFG["max_missing_keys_ratio"],
            CFG["max_unexpected_keys_ratio"],
        )
        print(
            f"[{family_name}] loaded: {os.path.basename(model_weight)} | missing={len(missing)} unexpected={len(unexpected)} | compatible={compatible}"
        )
        if not compatible:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        pred = inference_function(test_loader, model, device)["predictions"]
        if pred.shape != (len(test_df), 6):
            print(f"[{family_name}] skipped due to bad pred shape: {pred.shape}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        fam_preds.append(pred)
        strict_used += 1

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return fam_preds


if have_any_weights:
    print("infer with weights_spec")
    predictions_list.extend(
        _run_family(CFG["weights_spec"], "spec", NetSpec, {"use_spec": True})
    )

    print("infer with weights_eeg")
    predictions_list.extend(
        _run_family(CFG["weights_eeg"], "eeg", NetEeg, {"use_eeg": True})
    )

    print("infer with weights_mix")
    predictions_list.extend(
        _run_family(CFG["weights_mix"], "mix", NetMix, {"use_mix": True})
    )

    print(f"Checkpoints used: {len(predictions_list)}")
    if len(predictions_list) == 0:
        have_any_weights = False

if not have_any_weights:
    print(
        "No usable model weights found in provided paths; writing uniform-probability submission."
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    predictions = np.mean(np.stack(predictions_list, axis=0), axis=0)
    predictions = normalize_probs(predictions)

predictions = normalize_probs(predictions)
predictions.shape



## === cell 10
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

sub[TARGETS] = normalize_probs(sub[TARGETS].values)

sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
