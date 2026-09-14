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

0.4277328111095559

# 6. Current score

0.79439

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing weight-paths issue by making the script fall back to a uniform-probability submission when the expected model weight folders are not present in this Kaggle environment (so it runs end-to-end and writes a valid `.csv`). I also prevent `os.listdir()` from crashing and filter out directories/non-weight files so `torch.load()` never receives a directory path. Finally, I add a safety normalization step before writing the submission to guarantee each row sums to 1 (required by the competition) and keep the rest of your model/inference logic unchanged when weights are available.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4277), and the log shows you’re falling back to uniform predictions because no weights are found at the configured paths. The smallest change that legitimately improves score is to point `weights_spec` and `weights_eeg` to directories that actually exist in this environment (your dataset already contains spectrogram/EEG parquet files but not those external weight datasets). I keep your model/inference core logic identical, but add an automatic fallback search for weight files inside `/kaggle/input/` (and keep uniform predictions only if none are found). I also make `num_workers` safe in Kaggle and ensure we load either raw `state_dict` or common checkpoint wrappers without changing the network.'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target because the notebook is still effectively producing weak/uninformed predictions (most likely it’s not finding any real trained weights, or it’s loading incompatible checkpoints with `strict=False` and silently leaving random initialization). The smallest legitimate improvement toward the target is to (1) only load weight files that actually match each model type, and (2) load checkpoints robustly while refusing clearly-mismatched state_dicts (so we don’t average in random models), while keeping your architecture/inference unchanged. If no compatible weights are found, we still fall back to a safe baseline, but we use a class-prior distribution estimated from `train.csv` (still valid, no leakage) instead of uniform, which should reduce KL substantially versus uniform. Finally, we keep your probability normalization to guarantee valid submissions.'
- What this solution (achieved 1.43814) has done: 'Your score is still far above the target because you’re effectively producing an almost-constant prior prediction (no compatible weights are found/loaded). The smallest legitimate way to move toward the target without changing your model or training is to use a stronger constant baseline: compute a *patient-aggregated* train prior (averaging per patient first, then across patients) and apply mild Dirichlet/Laplace smoothing to avoid overconfident zeros—this typically reduces KL vs a raw global-row mean prior on this dataset. I keep your inference/weight-loading logic intact, only improving the “no weights” fallback and adding a couple of safety checks so the submission always aligns to `sample_submission.csv` row order and columns. This should move the score down toward the target while remaining stable and minimal.'
- What this solution (achieved 1.41973) has done: 'Your current score is far worse than the target because no compatible weights are being found/loaded, so the script falls back to a constant prior; the smallest legitimate improvement is to make that fallback prior better calibrated to the test distribution without changing your model/inference. I keep all model code identical and only change the “no weights” fallback to a per-patient mixture prior (patient-mean + global-mean) with light smoothing, which is usually safer than a single global prior and tends to reduce KL. I also add a deterministic “public baseline” mixture with the competition’s sample_submission (uniform) so probabilities are never overly peaked, which typically improves KL when you have no model signal. Submission writing/alignment stays the same and still guarantees row sums of 1.'
- What this solution (achieved 0.74875) has done: 'Your score is far above the target because the code is still not finding/using any compatible weights and falls back to a constant prior. To move the KL down toward the target with minimal changes (and without touching the model/inference core logic), I only improve the fallback prior to better match the test distribution by using per-patient priors derived from `train.csv` when available (same patient_id appears in test), blended with global and uniform priors with smoothing. I also make the spectrogram construction bug-free (currently `r` is never incremented, so all 4 regions use the same slice), which is a small correctness fix that can materially improve predictions if any weights do load. Everything else (architecture, softmax inference, ensembling, submission alignment/normalization) stays the same.'
- What this solution (achieved 0.75498) has done: 'Your current score (0.74875, lower-is-better) is still far above the target (0.4277), and since no compatible weights are typically found in this environment, the only legitimate lever that keeps core logic intact is improving the constant fallback calibration. I keep all model/inference code unchanged and only adjust the “no weights” fallback to better match the competition’s label-generation process by aggregating votes at the `eeg_id` level (the evaluation unit) before computing priors, then using a slightly stronger patient/global mixture with gentle smoothing. This should reduce KL versus the current row-level prior because it removes bias from overlapping subsamples in train. I also fix a small correctness bug in `brain_lead` (RR region was never used), which is minimal and only affects outputs if EEG weights ever do load.'
- What this solution (achieved 0.83114) has done: 'Your current score is far worse than the target (lower is better), and since no compatible weights are being found/loaded, the only lever that preserves your core model/inference logic is improving the constant fallback calibration. I keep all model code unchanged and only adjust the “no weights” fallback prior: compute priors at the evaluation unit (`eeg_id`), then blend per-patient, global, and uniform priors with a slightly stronger smoothing that typically reduces KL by avoiding overconfident distributions. I also make the fallback mixture weights configurable and slightly more conservative (more global/uniform), which often improves KL when patient overlap between train/test is partial. Submission alignment and row-sum normalization remain exactly as before to guarantee a valid `.csv`.'
- What this solution (achieved 0.88345) has done: 'Your current score (0.83114, lower-is-better) is still far above the target (0.42773), and in this environment you’re effectively in the “no compatible weights found” regime, so the only minimal lever that preserves your model/inference core logic is improving the fallback probabilities. I keep all model code untouched and only adjust the fallback prior to match the evaluation unit and distribution better: compute per-`eeg_id` vote distributions, then (crucially) build **per-patient priors weighted by each patient’s number of unique eegs** (instead of giving every patient equal weight), which better reflects the marginal label distribution and typically lowers KL. I also make the patient/global/uniform mixture slightly more conservative (more global, slightly less patient) while keeping smoothing and normalization safeguards identical. This should move KL down toward the target without changing the architecture or inference semantics.'
- What this solution (achieved 0.8836) has done: 'Your current score (0.88345, lower-is-better) is still far above the target (0.42773), and the logs/plans indicate you’re effectively in the “no compatible weights found” regime, so improving the fallback probabilities is the smallest legitimate lever that can reduce KL without touching your model/inference core logic. I keep all model/dataloader/inference code unchanged, and only refine the prior fallback to better match the competition’s label-generation process by (1) estimating **test-time class prior from `train.csv` using only patients present in `test.csv`**, aggregated at `eeg_id` level, and (2) blending that with the existing per-patient prior and global prior using the same smoothing/normalization safeguards. This typically reduces KL versus a generic global prior because the test set’s patient mix differs from the full training population, while remaining leakage-free and constant-time. The submission writing/alignment/row-sum validation remains identical.'
- What this solution (achieved 0.76262) has done: 'Your current score (0.8836, lower-is-better) is still far worse than the target (0.4277), and your logs/plans indicate you’re effectively in the “no compatible weights found” regime, so the smallest lever that can legitimately reduce KL (without changing the model/inference core logic) is improving the constant fallback probabilities. I keep all model/dataloader/inference code unchanged and only adjust the fallback prior to better match the evaluation unit and label-generation process by computing priors using a **Dirichlet-multinomial posterior mean** from `train.csv` (at `eeg_id` level), then blending per-patient and “test-patient-only” priors in a more robust way. Concretely, I (1) compute per-eeg vote totals, (2) form posteriors with a tunable `prior_alpha` interpreted as Dirichlet strength (not just additive smoothing on probabilities), and (3) compute per-patient priors from those posteriors and use them for patients seen in test; unseen patients use a blended default prior. This should move KL down toward the target while remaining stable, leakage-free, and minimal.'
- What this solution (achieved 0.764) has done: 'Your current KL (0.76262, lower-is-better) is still well above the target (0.42773), and in this environment you’re almost certainly in the “no compatible weights found” regime, so the only minimal lever is improving the fallback probabilities while keeping your model/inference untouched. I keep all architecture/inference exactly the same and only adjust the fallback prior to better match the evaluation: compute a Dirichlet posterior **per eeg_id** (vote-counts) then form a **test-patient-only** global prior and **per-patient** priors, and finally blend them with slightly more weight on the test-patient prior (less uniform), which typically reduces KL. I also make the fallback safer for patients not in train (avoid `KeyError`/bad casting) and keep the same strict row-sum normalization and sample_submission alignment. No training is introduced, and runtime stays small.'
- What this solution (achieved 0.75941) has done: 'Your current KL (0.764, lower-is-better) is still far above the target (0.4277), and in this environment you’re effectively using the “no weights found” fallback, so the only minimal lever is to make that fallback prior closer to the test distribution. I keep your model/inference path unchanged and only refine `_compute_train_priors` to use a more appropriate evaluation-unit estimate: compute a test-patient-weighted prior from `train.csv` aggregated at `eeg_id`, then blend it with per-patient priors and a small uniform component for safety. This is still leakage-free (uses only train labels) and constant-time, and it should reduce KL versus your current mixture because it better matches the test patient composition while avoiding overconfident probabilities. Submission alignment and row-sum normalization remain identical to guarantee a valid CSV.'
- What this solution (achieved 0.79086) has done: 'Your current KL (0.75941, lower-is-better) is still far above the target (0.4277), and based on the code path it’s very likely you’re still in the “no compatible weights found” regime—so the only minimal lever that preserves core model/inference logic is improving the fallback probabilities. I keep the model code untouched and only adjust `_compute_train_priors` to (1) build per-patient priors using **vote-count aggregation** (Dirichlet posterior on summed votes) instead of averaging per-eeg probabilities, and (2) blend that patient posterior with a **test-patient-frequency-weighted** global posterior; this typically reduces KL by better matching how labels are generated. I also keep your existing smoothing/normalization and submission alignment, so output remains valid and stable. No training, no architecture changes, and runtime stays small.'
- What this solution (achieved 0.79439) has done: 'Your current KL (0.79086, lower-is-better) is still far above the target (0.42773), and the code path indicates you are likely still in the “no compatible weights found” regime, so the only minimal lever (without touching model/inference core logic) is improving the fallback prior calibration. I keep your Dirichlet-on-vote-counts approach, but change the global/test-patient priors to be computed from **vote-count aggregation with the same Dirichlet posterior** (rather than averaging per-eeg posteriors), which is a small but often material correction for this metric. I also make the per-patient priors use the same vote-count Dirichlet posterior (already true) and adjust nothing else about architecture, dataloading, inference, or submission alignment/normalization. This should legitimately move KL downward toward the target while remaining stable and fast.'

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
    "flip": True,
    "train_csv_for_prior": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    "sample_submission": "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
    "prior_alpha": 0.20,
    "mix_patient": 0.35,
    "mix_global": 0.15,
    "mix_uniform": 0.05,
    "mix_test_patients_global": 0.45,
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 2
def _list_weight_files(path):
    if path is None:
        return []
    if not os.path.isdir(path):
        return []
    files = []
    for fn in sorted(os.listdir(path)):
        full = os.path.join(path, fn)
        if not os.path.isfile(full):
            continue
        lfn = fn.lower()
        if lfn.endswith((".pt", ".pth", ".bin", ".ckpt")):
            files.append(full)
    return files


def _find_weight_dirs_under_input(prefer_keywords):
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return []
    dirs = []
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if os.path.isdir(full):
            name = d.lower()
            if any(k in name for k in prefer_keywords):
                dirs.append(full)
    return dirs


def _auto_find_weights(prefer_keywords):
    base = "/kaggle/input"
    candidates = _find_weight_dirs_under_input(prefer_keywords)
    weight_files = []
    for d in candidates:
        weight_files.extend(_list_weight_files(d))

    if len(weight_files) == 0 and os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            full = os.path.join(base, d)
            if not os.path.isdir(full):
                continue
            for subd in sorted(os.listdir(full)):
                subfull = os.path.join(full, subd)
                if os.path.isdir(subfull):
                    weight_files.extend(_list_weight_files(subfull))

    seen = set()
    out = []
    for p in weight_files:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _filter_weight_files_by_name(weight_files, include_any=(), exclude_any=()):
    out = []
    for p in weight_files:
        name = os.path.basename(p).lower()
        if include_any and not any(k in name for k in include_any):
            continue
        if exclude_any and any(k in name for k in exclude_any):
            continue
        out.append(p)
    return out


CFG["weights_spec"] = _list_weight_files(CFG["weights_spec"])
CFG["weights_eeg"] = _list_weight_files(CFG["weights_eeg"])

if len(CFG["weights_spec"]) == 0:
    CFG["weights_spec"] = _auto_find_weights(
        prefer_keywords=["spec", "spect", "baseline", "efficientnet", "hms"]
    )
if len(CFG["weights_eeg"]) == 0:
    CFG["weights_eeg"] = _auto_find_weights(
        prefer_keywords=["eeg", "wave", "efficientnet", "hms"]
    )

spec_filtered = _filter_weight_files_by_name(
    CFG["weights_spec"],
    include_any=("spec", "spect", "baseline", "eff", "b5"),
    exclude_any=("eeg", "wave"),
)
eeg_filtered = _filter_weight_files_by_name(
    CFG["weights_eeg"],
    include_any=("eeg", "wave", "eff", "b5"),
    exclude_any=("spec", "spect", "baseline"),
)
if len(spec_filtered) > 0:
    CFG["weights_spec"] = spec_filtered
if len(eeg_filtered) > 0:
    CFG["weights_eeg"] = eeg_filtered

print("weights_spec:", len(CFG["weights_spec"]), CFG["weights_spec"][:3])
print("weights_eeg :", len(CFG["weights_eeg"]), CFG["weights_eeg"][:3])




## === cell 3
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
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

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        if self.use_eeg:
            eeg_path = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet"
                % (dp["eeg_id"])
            )
            eeg = pd.read_parquet(eeg_path)

            offset = 0  # test has full 50s already
            eeg = eeg.iloc[offset * 200 : (offset + 50) * 200]

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
            data = waves
        else:
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
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 5
class Transform(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=50, power=1
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
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 6
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to("cpu").numpy())
    prediction_dict = {"predictions": np.concatenate(preds)}
    return prediction_dict




## === cell 7
test_df = pd.read_csv(CFG["data"])
test_df.head(5)




## === cell 8
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def _sanitize_state_dict_keys(sd):
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


def _load_state_dict_safely(model, ckpt_obj):
    sd = _extract_state_dict(ckpt_obj)
    sd = _sanitize_state_dict_keys(sd)
    if not isinstance(sd, dict):
        return False, "checkpoint_state_not_dict"

    model_sd = model.state_dict()
    common = [
        k
        for k in sd.keys()
        if k in model_sd
        and hasattr(sd[k], "shape")
        and hasattr(model_sd[k], "shape")
        and sd[k].shape == model_sd[k].shape
    ]

    if len(common) < 10:
        return False, f"too_few_matching_keys={len(common)}"

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0.9 * len(model_sd):
        return False, f"mostly_missing_keys={len(missing)}/{len(model_sd)}"
    return (
        True,
        f"loaded_common_keys={len(common)} missing={len(missing)} unexpected={len(unexpected)}",
    )


def _compute_train_priors(
    train_csv_path,
    test_df_in,
    alpha=0.20,
    mix_patient=0.35,
    mix_global=0.15,
    mix_uniform=0.05,
    mix_test_patients_global=0.45,
):
    """
    Score-relevant minimal change (fallback only):
    - Keep eeg_id-level aggregation.
    - Keep Dirichlet posterior mean smoothing parameter alpha.
    - CHANGE (small but often material for KL): compute global prior and test-patient-global prior
      from *aggregated vote counts* then Dirichlet posterior mean, instead of averaging per-eeg
      posteriors. This matches the generative process of vote counts and tends to calibrate the
      constant fallback better.
    """
    uniform = np.full((6,), 1.0 / 6.0, dtype=np.float64)
    if not (train_csv_path and os.path.isfile(train_csv_path)):
        return uniform, uniform, {}, uniform

    usecols = ["eeg_id", "patient_id"] + TARGETS
    df = pd.read_csv(train_csv_path, usecols=usecols)

    agg = df.groupby("eeg_id", sort=False).agg(
        {"patient_id": "first", **{t: "sum" for t in TARGETS}}
    )

    votes = agg[TARGETS].to_numpy(dtype=np.float64)
    votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)
    votes[votes < 0] = 0.0

    K = votes.shape[1]

    denom = votes.sum(axis=1, keepdims=True) + alpha * K
    denom[denom == 0] = 1.0
    probs_eeg = (votes + alpha) / denom  # (n_eeg, 6)

    pid_series = pd.to_numeric(agg["patient_id"], errors="coerce").astype("Int64")

    votes_df = pd.DataFrame(votes, columns=TARGETS, index=agg.index)
    votes_df["patient_id"] = pid_series.values

    votes_by_patient = (
        votes_df.dropna(subset=["patient_id"]).groupby("patient_id")[TARGETS].sum()
    )
    if len(votes_by_patient) > 0:
        vpat = votes_by_patient.to_numpy(dtype=np.float64)
        vpat = np.nan_to_num(vpat, nan=0.0, posinf=0.0, neginf=0.0)
        vpat[vpat < 0] = 0.0
        denom_pat = vpat.sum(axis=1, keepdims=True) + alpha * K
        denom_pat[denom_pat == 0] = 1.0
        probs_patient = (vpat + alpha) / denom_pat  # (n_patients, 6)
        patient_index = votes_by_patient.index.astype(np.int64).to_numpy()

        patient_counts_train_eegs = (
            votes_df.dropna(subset=["patient_id"])
            .groupby("patient_id")
            .size()
            .to_numpy(dtype=np.float64)
        )
        w = patient_counts_train_eegs / (
            patient_counts_train_eegs.sum()
            if patient_counts_train_eegs.sum() > 0
            else 1.0
        )
        patient_prior_raw = (probs_patient * w[:, None]).sum(axis=0)
    else:
        probs_patient = np.zeros((0, K), dtype=np.float64)
        patient_index = np.zeros((0,), dtype=np.int64)
        patient_prior_raw = probs_eeg.mean(axis=0)

    vglob = votes.sum(axis=0).astype(np.float64)
    vglob = np.nan_to_num(vglob, nan=0.0, posinf=0.0, neginf=0.0)
    vglob[vglob < 0] = 0.0
    global_prior_raw = (vglob + alpha) / (vglob.sum() + alpha * K + 1e-12)

    test_pids = (
        pd.to_numeric(pd.Series(test_df_in["patient_id"].values), errors="coerce")
        .dropna()
        .astype(np.int64)
    )
    test_pid_set = set(test_pids.unique().tolist())
    if len(test_pid_set) > 0:
        mask = votes_df["patient_id"].isin(list(test_pid_set)).values
        if mask.any():
            vtestglob = votes[mask].sum(axis=0).astype(np.float64)
            vtestglob = np.nan_to_num(vtestglob, nan=0.0, posinf=0.0, neginf=0.0)
            vtestglob[vtestglob < 0] = 0.0
            test_patients_global_raw = (vtestglob + alpha) / (
                vtestglob.sum() + alpha * K + 1e-12
            )
        else:
            test_patients_global_raw = global_prior_raw.copy()
    else:
        test_patients_global_raw = global_prior_raw.copy()

    def _clip_norm(p):
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        return p

    patient_prior = _clip_norm(patient_prior_raw)
    global_prior = _clip_norm(global_prior_raw)
    test_patients_global = _clip_norm(test_patients_global_raw)

    per_patient = {}
    if len(patient_index) > 0:
        for i, pid in enumerate(patient_index.tolist()):
            per_patient[int(pid)] = _clip_norm(probs_patient[i].astype(np.float64))

    s = mix_patient + mix_global + mix_uniform + mix_test_patients_global
    if s <= 0:
        mix_patient, mix_global, mix_uniform, mix_test_patients_global = (
            0.0,
            0.0,
            1.0,
            0.0,
        )
        s = 1.0
    mix_patient, mix_global, mix_uniform, mix_test_patients_global = (
        mix_patient / s,
        mix_global / s,
        mix_uniform / s,
        mix_test_patients_global / s,
    )

    default_prior = (
        mix_patient * patient_prior
        + mix_global * global_prior
        + mix_uniform * uniform
        + mix_test_patients_global * test_patients_global
    )
    default_prior = _clip_norm(default_prior)

    return patient_prior, global_prior, per_patient, default_prior


predictions = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_workers = int(CFG["num_worker"])
num_workers = max(0, min(num_workers, os.cpu_count() or 0))

loaded_any = False

if len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec")
    for model_weight in CFG["weights_spec"]:
        test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = NetSpec()
        ckpt = torch.load(model_weight, map_location="cpu")

        ok, msg = _load_state_dict_safely(model, ckpt)
        if not ok:
            print(f"Skip spec weight (incompatible): {model_weight} ({msg})")
            del model, ckpt, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
            continue

        print(f"Loaded spec weight: {os.path.basename(model_weight)} ({msg})")
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])
        loaded_any = True

        del model, ckpt, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

if len(CFG["weights_eeg"]) > 0:
    print("infer with weights_eeg")
    for model_weight in CFG["weights_eeg"]:
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = NetEeg()
        ckpt = torch.load(model_weight, map_location="cpu")

        ok, msg = _load_state_dict_safely(model, ckpt)
        if not ok:
            print(f"Skip eeg weight (incompatible): {model_weight} ({msg})")
            del model, ckpt, test_loader, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
            continue

        print(f"Loaded eeg weight: {os.path.basename(model_weight)} ({msg})")
        model.to(device)

        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])
        loaded_any = True

        del model, ckpt, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

if not loaded_any:
    patient_prior, global_prior, per_patient, default_prior = _compute_train_priors(
        CFG["train_csv_for_prior"],
        test_df,
        alpha=float(CFG["prior_alpha"]),
        mix_patient=float(CFG["mix_patient"]),
        mix_global=float(CFG["mix_global"]),
        mix_uniform=float(CFG["mix_uniform"]),
        mix_test_patients_global=float(CFG["mix_test_patients_global"]),
    )

    preds = np.zeros((len(test_df), 6), dtype=np.float64)
    pids = (
        pd.to_numeric(pd.Series(test_df["patient_id"].values), errors="coerce")
        .astype("Int64")
        .values
    )
    for i in range(len(test_df)):
        pid = pids[i]
        if pd.isna(pid):
            preds[i] = default_prior
        else:
            preds[i] = per_patient.get(int(pid), default_prior)

    preds = np.clip(preds, 1e-12, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)
    predictions = preds.astype(np.float32)

    print(
        "No compatible model weights found. Using eeg_id-aggregated Dirichlet-on-votes priors with vote-count-based global/test-patient priors."
    )
    print("Default prior:", default_prior)
else:
    predictions = np.array(predictions, dtype=np.float32)  # (n_models, n_samples, 6)
    predictions = np.mean(predictions, axis=0)  # (n_samples, 6)



## === cell 9
predictions = np.asarray(predictions, dtype=np.float64)
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(
        f"predictions must have shape (n_samples, 6), got {predictions.shape}"
    )

predictions = np.clip(predictions, 1e-12, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

if CFG.get("sample_submission") and os.path.isfile(CFG["sample_submission"]):
    sub = pd.read_csv(CFG["sample_submission"])
    sub = sub[["eeg_id"] + TARGETS]
    id_to_row = {int(eid): i for i, eid in enumerate(test_df.eeg_id.values)}
    idx = [id_to_row.get(int(eid), None) for eid in sub.eeg_id.values]
    if any(i is None for i in idx):
        raise ValueError(
            "Some eeg_id in sample_submission not found in test_df; cannot align safely."
        )
    sub[TARGETS] = predictions[np.array(idx, dtype=np.int64)]
else:
    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    sub[TARGETS] = predictions

sub[TARGETS] = sub[TARGETS].to_numpy(dtype=np.float64)
row_sums = sub[TARGETS].sum(axis=1).to_numpy()
if not np.allclose(row_sums, 1.0, atol=1e-6):
    arr = sub[TARGETS].to_numpy()
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    sub[TARGETS] = arr

sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sum stats:",
    float(sub[TARGETS].sum(axis=1).min()),
    float(sub[TARGETS].sum(axis=1).max()),
)
