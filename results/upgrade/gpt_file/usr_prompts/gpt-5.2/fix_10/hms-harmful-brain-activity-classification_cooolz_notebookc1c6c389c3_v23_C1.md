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

0.467728428302768

# 6. Current score

0.78876

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first make the script robust to missing external weight directories by falling back to a safe, valid baseline prediction (uniform probabilities) so it always produces a submission CSV end-to-end. I also remove the hard CUDA dependency in the EEG→mel code path by selecting CPU when CUDA isn’t available, preventing the “no NVIDIA driver” crash. Next, I filter the weight file lists to actual model checkpoint files only (not directories), fixing the `IsADirectoryError` when `torch.load` is called. Finally, I ensure the prediction array is always shaped `(len(test_df), 6)` and normalized to sum to 1 per row, preventing the pandas assignment error and submission rejection.'
- What this solution (achieved 1.40995) has done: 'Your current 1.40995 is far worse than the target 0.4677 (lower is better), and the main reason is that you’re almost certainly falling back to the uniform-probability baseline because the weight paths (`/kaggle/input/hms-baseline`, `/kaggle/input/hms-eeg`, `/kaggle/input/hms-mix`) don’t exist in your environment. I keep the model logic identical, but make the code auto-discover any available `.pt/.pth/.bin` checkpoints under `/kaggle/input` and use them; if none are found, it still produces a valid submission. I also fix a small but impactful bug in `brain_lead` where the RR group is mistakenly duplicated as RP, which hurts EEG feature construction and worsen predictions when EEG/mix models are used. Finally, I load checkpoints robustly whether they are raw `state_dict` or wrapped dicts (common in Kaggle), which prevents silent partial loads that degrade score.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far above the target (0.4677), and the biggest likely cause is that you’re still effectively using weak/unintended checkpoints (or none that match) because checkpoint discovery is too broad and `strict=False` can silently load mismatched weights. I make checkpoint selection *more precise* by only using checkpoints that live in the competition dataset tree and by filtering to filenames that look like real model weights, reducing the chance of averaging garbage weights that hurt KL. I also harden checkpoint loading by stripping common prefixes (`module.`, `model.`) and only falling back to `strict=False` if a strict load fails—this preserves your architecture/inference logic but makes it much more likely you’re actually using the intended trained weights. Finally, I fix a real bug in `brain_lead` construction (RR/RP mix-up in prior plan context) by ensuring the lead groups are exactly `[LL, LP, RP, RR]` and remain consistent; this is a minimal correctness fix that can materially improve EEG-based predictions when weights exist.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995; lower is better) is far above the target (0.4677), and the most likely cause is that you are loading no (or wrong) checkpoints from nonexistent `/kaggle/input/hms-*/` paths, so you effectively submit near-uniform probabilities. I keep your model/dataset/inference logic identical, but change checkpoint discovery to search all of `/kaggle/input` (not just the competition dataset folder) and to prefer weights from datasets that match your configured names (`hms-baseline`, `hms-eeg`, `hms-mix`) so it actually uses your intended trained weights when available. I also make the test dataloader deterministic (`shuffle=False` is already set; we additionally disable any accidental randomness via seeds) and ensure checkpoint selection is stable by sorting by filename, which helps avoid score variance. These are minimal execution- and correctness-focused changes that should move KL down toward your target without changing the core approach.'
- What this solution (achieved 1.40995) has done: 'Your current score is far above the target (lower is better), and the most likely reason is that you’re averaging many unrelated checkpoints (auto-discovery grabs arbitrary `.pt/.pth/.bin` under `/kaggle/input`), which typically produces near-uniform/garbage probabilities and a high KL. I keep your exact model/dataset/inference logic, but make checkpoint discovery *strictly limited* to the HMS competition dataset tree and only accept filenames that look like real model checkpoints, so we either (a) load the intended weights or (b) fall back cleanly to uniform rather than averaging junk. I also cap the number of checkpoints used per model type to a small number (sorted, deterministic) to avoid diluting predictions with mismatched weights, which should move KL down toward your target. Finally, I keep the existing normalization/clipping so the submission remains valid.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far above the target (0.4677), and the most likely reason is you’re still not loading the intended checkpoints (so you’re close to uniform) or you’re loading incompatible ones (so predictions are noisy/poor). I make checkpoint discovery slightly broader (search all of `/kaggle/input`, but still filtered and capped) and route discovered checkpoints into the correct model type (spec/eeg/mix) based on filename tokens, so you don’t accidentally run a spec checkpoint through an EEG model (or vice versa). I also make checkpoint loading more robust by handling common “model.”/“module.” prefixes and “state_dict” nesting, and I keep everything else (models, feature extraction, inference, normalization, submission format) unchanged. These minimal changes should legitimately reduce KL and move you closer to the target without altering the solution’s core logic.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4677), and the most likely cause is that you’re still not loading any real trained checkpoints (so you submit near-uniform) because the hardcoded weight dirs don’t exist in this environment. I make checkpoint discovery explicitly search within the HMS competition dataset folder (where weights could plausibly exist) and, if none are found, fall back to a simple label-prior baseline computed from `train.csv` vote distributions (usually much better than uniform for KL) while keeping your model/inference logic unchanged. I also fix a real bug in `get_spec()` where `r` is never advanced, causing all 4 regions to use the same slice; this hurts any spec/mix model predictions when weights are available. Finally, I ensure deterministic, correct file selection and keep the submission normalization exactly as required.'
- What this solution (achieved 0.76744) has done: 'Your current KL (1.419) is far above the target (0.468; lower is better), and this script is very likely still not using any real trained checkpoints—so the best minimal legitimate improvement is to make the fallback baseline stronger (and stable) when checkpoints are missing/invalid. I keep all model/inference logic unchanged, but (1) broaden checkpoint discovery to search all of `/kaggle/input` (still filtered/capped) so it can actually find your weight datasets if present, and (2) improve the no-checkpoint fallback by using a per-patient label-prior (computed only from `train.csv` votes) with a global-prior fallback for unseen patients. This typically reduces KL substantially versus a single global prior/uniform because label distribution is patient-dependent, and it doesn’t change evaluation semantics or introduce leakage (patient_id exists in both train/test). Finally, I keep the existing probability clipping/renormalization so the submission always validates.'
- What this solution (achieved 0.78876) has done: 'Your current KL (0.76744; lower is better) is still far above the target (0.4677), so we should improve the fallback predictions without changing your model logic. The smallest legitimate lever here is the patient-prior baseline used when checkpoints are missing/poor: we compute a more accurate patient prior by first consolidating overlapping train rows to unique `(eeg_id, eeg_sub_id)` (to avoid overweighting recordings with many overlapping windows), then forming patient vote priors from that de-duplicated table. We also apply a tiny blend between patient prior and global prior to stabilize patients with few samples, which usually lowers KL versus a raw patient prior while keeping semantics identical (still pure train-derived priors). Everything else (models, transforms, inference, normalization, submission format/paths) stays the same.'

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
    "train_csv": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "weights_mix": "/kaggle/input/hms-mix",
    "flip": True,
}




## === cell 2
def _list_weight_files(path: str):
    if not isinstance(path, str) or (not path) or (not os.path.exists(path)):
        return []
    if os.path.isfile(path):
        return [path]
    files = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp) and fn.lower().endswith((".pt", ".pth", ".bin", ".ckpt")):
            files.append(fp)
    return files


def _looks_like_weight_name(fn_lower: str) -> bool:
    return any(
        t in fn_lower
        for t in (
            "fold",
            "epoch",
            "best",
            "ckpt",
            "checkpoint",
            "model",
            "weight",
            "weights",
            "state",
        )
    )


def _discover_checkpoints_under(
    base_dirs=("/kaggle/input/hms-harmful-brain-activity-classification",),
    max_files_total=60,
):
    ckpts = []
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for fn in files:
                lfn = fn.lower()
                if not lfn.endswith((".pt", ".pth", ".bin", ".ckpt")):
                    continue
                if not _looks_like_weight_name(lfn):
                    continue
                ckpts.append(os.path.join(root, fn))

    ckpts = sorted(list(dict.fromkeys(ckpts)))

    def _rank(p):
        lp = p.lower()
        pref = 0
        if "hms" in lp:
            pref += 10
        if "harmful" in lp or "brain" in lp or "eeg" in lp or "spect" in lp:
            pref += 5
        name_bump = 1 if _looks_like_weight_name(os.path.basename(lp)) else 0
        return (pref, name_bump, lp)

    ckpts = sorted(ckpts, key=_rank, reverse=True)
    if max_files_total is not None:
        ckpts = ckpts[:max_files_total]
    return ckpts


def _cap_and_sort(files, cap=5):
    files = [f for f in (files or []) if isinstance(f, str) and os.path.isfile(f)]
    files = sorted(files)
    if cap is not None:
        files = files[:cap]
    return files


def _split_discovered_by_type(paths):
    spec, eeg, mix = [], [], []
    for p in paths or []:
        lp = p.lower()
        base = os.path.basename(lp)
        if any(t in base for t in ("mix", "hrnet")):
            mix.append(p)
        elif any(t in base for t in ("eeg", "wave", "raw")):
            eeg.append(p)
        elif any(
            t in base for t in ("spec", "spect", "efficientnet", "b5", "baseline")
        ):
            spec.append(p)
        else:
            spec.append(p)
    return spec, eeg, mix


CFG["weights_spec"] = _cap_and_sort(_list_weight_files(CFG["weights_spec"]), cap=5)
CFG["weights_eeg"] = _cap_and_sort(_list_weight_files(CFG["weights_eeg"]), cap=5)
CFG["weights_mix"] = _cap_and_sort(_list_weight_files(CFG["weights_mix"]), cap=5)

if (len(CFG["weights_spec"]) + len(CFG["weights_eeg"]) + len(CFG["weights_mix"])) == 0:
    discovered = _discover_checkpoints_under(
        base_dirs=(
            "/kaggle/input/hms-harmful-brain-activity-classification",
            "/kaggle/input",
        ),
        max_files_total=80,
    )
    d_spec, d_eeg, d_mix = _split_discovered_by_type(discovered)
    CFG["weights_spec"] = _cap_and_sort(d_spec, cap=5)
    CFG["weights_eeg"] = _cap_and_sort(d_eeg, cap=5)
    CFG["weights_mix"] = _cap_and_sort(d_mix, cap=5)

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


transform_func = TransformMel().to(DEVICE)


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

            x_tensor = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
            mel_spec = transform_func(x_tensor).detach().cpu().numpy()

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
    for item in tqdm(all_fs, desc="Building EEG mel specs (for mix model)"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        eeg_spec = spectrogram_from_eeg(eeg_path)
        all_specs[eeg_id] = eeg_spec

    with open("eeg_specs_dict.pkl", "wb") as file:
        pickle.dump(all_specs, file)




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

        self.ll = 0
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  ##decided by self.parse_file

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
            data = np.zeros((4, 300, 100), dtype=np.float32)

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
    def __init__(
        self,
    ):
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
        for X in tqdm_test_loader:
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 9
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 10
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _strip_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in ("module.", "model.", "net."):
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _train_global_prior_probs_from_votes(train_csv_path, targets, alpha=1.0):
    df = pd.read_csv(train_csv_path, usecols=targets)
    sums = df[targets].sum(axis=0).values.astype(np.float64)
    sums = sums + float(alpha)
    p = sums / sums.sum()
    return p.astype(np.float32)


def _train_patient_prior_probs_from_votes(
    train_csv_path, targets, alpha=1.0, blend=0.20
):
    usecols = ["eeg_id", "eeg_sub_id", "patient_id"] + list(targets)
    df = pd.read_csv(train_csv_path, usecols=usecols)

    df = df.drop_duplicates(subset=["eeg_id", "eeg_sub_id"], keep="first").reset_index(
        drop=True
    )

    grp = df.groupby("patient_id", sort=False)[targets].sum()
    sums = grp.values.astype(np.float64) + float(alpha)
    probs = sums / sums.sum(axis=1, keepdims=True)

    global_sums = df[targets].sum(axis=0).values.astype(np.float64) + float(alpha)
    global_p = (global_sums / global_sums.sum()).astype(np.float32)

    blend = float(blend)
    probs = (1.0 - blend) * probs + blend * global_p[None, :]

    patient2p = {
        pid: probs[i].astype(np.float32) for i, pid in enumerate(grp.index.values)
    }
    return patient2p, global_p


def _predict_with_weights(weight_files, dataset_kwargs, model_ctor):
    weight_files = [wf for wf in (weight_files or []) if os.path.isfile(wf)]
    if len(weight_files) == 0:
        return None

    filtered = []
    for wf in weight_files:
        try:
            if os.path.getsize(wf) >= 500_000:  # ~0.5MB minimum to avoid junk
                filtered.append(wf)
        except Exception:
            continue
    weight_files = filtered
    if len(weight_files) == 0:
        return None

    local_preds = []
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, **dataset_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    for model_weight in weight_files:
        model = model_ctor()
        ckpt = torch.load(model_weight, map_location=device)
        state_dict = _strip_prefixes(_extract_state_dict(ckpt))

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        local_preds.append(prediction_dict["predictions"])
        del model, ckpt, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    local_preds = np.array(local_preds)
    return np.mean(local_preds, axis=0)


print("infer with weights_eeg")
pred_eeg = _predict_with_weights(CFG["weights_eeg"], {"use_eeg": True}, NetEeg)
if pred_eeg is not None:
    predictions_accum.append(pred_eeg)

print("infer with weights_spec")
pred_spec = _predict_with_weights(CFG["weights_spec"], {"use_spec": True}, NetSpec)
if pred_spec is not None:
    predictions_accum.append(pred_spec)

print("infer with weights_mix")
pred_mix = _predict_with_weights(CFG["weights_mix"], {"use_mix": True}, NetMix)
if pred_mix is not None:
    predictions_accum.append(pred_mix)

if len(predictions_accum) == 0:
    patient2p, global_p = _train_patient_prior_probs_from_votes(
        CFG["train_csv"], TARGETS, alpha=1.0, blend=0.20
    )
    predictions = np.zeros((n_test, 6), dtype=np.float32)
    test_patients = test_df["patient_id"].values
    for i, pid in enumerate(test_patients):
        predictions[i] = patient2p.get(pid, global_p)
else:
    predictions = np.mean(np.stack(predictions_accum, axis=0), axis=0).astype(
        np.float32
    )

if predictions.shape != (n_test, 6):
    raise ValueError(
        f"Predictions have wrong shape {predictions.shape}, expected {(n_test, 6)}"
    )

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
    "Row sums (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
