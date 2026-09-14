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

0.4077395548344591

# 6. Current score

0.79884

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing-weights path bug by detecting whether the external weight directories exist; if they don’t (as in your environment), the script fall back to a valid baseline submission (uniform probabilities that sum to 1) so a `.csv` is always produced. I also fix the hard CUDA usage in the EEG→mel-spectrogram preprocessing so it runs on CPU-only machines without crashing. Finally, I make the weights discovery robust (only load actual model files, not directories) and add a safety normalization step so each prediction row sums to 1 (required to avoid submission failure). These changes are strictly to unblock execution and produce a valid submission end-to-end; when pretrained weights are available, the original inference ensemble logic remains intact.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) indicates the notebook is almost certainly falling back to the uniform-probability baseline because no pretrained weights are being found at the configured `/kaggle/input/hms-*` paths. The smallest change that should move the score substantially toward your target is to point `weights_spec/weights_eeg/weights_mix` to directories that actually exist in this environment (under the HMS dataset folder) and robustly load common checkpoint formats (plain `state_dict`, nested `state_dict`, or full checkpoint dicts). This preserves your model architectures, preprocessing, inference loop, and averaging logic; it only fixes weight discovery/loading so the intended ensemble inference actually runs. I also keep the existing row-normalization (required for valid KL submissions) and leave the uniform fallback intact if no weights are present.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) strongly suggests you are still submitting the uniform fallback because no pretrained weight files are actually being found/loaded. I make the smallest change that materially improves score: broaden weight discovery to also search common subdirectories inside the dataset folder and load checkpoints robustly (including nested keys and removing common prefixes), so your existing ensemble inference runs instead of the uniform baseline. I also fix a small but critical bug in `brain_lead` (RR mistakenly uses RP twice) that affects EEG feature construction and should improve predictions without changing the model or training logic. Finally, I keep the strict row-normalization/clipping required for a valid KL-divergence submission.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is consistent with still using the uniform fallback because no real checkpoints are being found under the HMS dataset folder. The smallest change that should move the score strongly toward your target is to (1) point weight discovery at the *actual* Kaggle Dataset input(s) by scanning `/kaggle/input` for common “hms” model folders/checkpoints, and (2) harden checkpoint loading so different training frameworks’ key names/prefixes load correctly into your unchanged model classes. I’m also fixing a subtle but impactful bug in `get_spec` (the spectrogram slice index `r` never increments, so all 4 regions were identical), which improves feature correctness without changing the overall pipeline. Everything else (architectures, softmax, flip-TTA, ensembling, normalization, and submission schema) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.4077), and the most likely cause is that you’re still effectively producing near-uniform predictions because no compatible pretrained checkpoints are being loaded into the models (or they’re loading with many missing keys). I keep your exact architectures/inference logic but (1) tighten weight discovery to prefer checkpoints whose filenames indicate they belong to the correct model type (spec/eeg/mix) instead of mixing arbitrary `.pth` files, and (2) make checkpoint loading more compatible by also handling “model.model.*” prefixes and by filtering out non-matching tensor shapes before `load_state_dict` so more layers actually load. Finally, I add a small “sanity gate”: if a checkpoint loads with *too many* missing keys (likely wrong model), we skip it instead of averaging garbage into the ensemble, which should move score down toward the target without changing the pipeline semantics. All changes are strictly in weight discovery/loading/selection (not model definitions, preprocessing, or inference loops), and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.4077), and the most likely reason is that you are still submitting the uniform fallback because there are no real pretrained checkpoints being found under the configured paths. I make the smallest change that can materially improve score without touching architectures or inference semantics: explicitly support single-file weight paths (not just directories) and widen discovery to include `.safetensors` (common on Kaggle) while still filtering by model kind. I also cap the number of loaded checkpoints per kind to avoid accidentally averaging hundreds of mismatched models (which can push predictions toward uniform), keeping the ensemble logic identical but more stable. Everything else (feature extraction, models, softmax, TTA flip, averaging, and per-row normalization) stays the same, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.4077), and the most likely reason is still that no real pretrained checkpoints are being loaded, so predictions stay near-uniform. The minimal change to move the score toward target is to stop scanning arbitrary `/kaggle/input` datasets (which can mix in incompatible checkpoints or none at all) and instead only load weights if they are explicitly present under your configured HMS dataset paths; otherwise we keep the uniform fallback. To prevent “garbage” averaging from partially-loaded/incorrect checkpoints, we also make the skip-gate stricter (require most keys to load) so only genuinely compatible weights contribute. These changes keep your architectures, preprocessing, inference loops, softmax, TTA flip, ensembling-by-mean, and submission formatting identical—only weight discovery/selection is tightened to improve prediction quality when valid weights exist.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.4077) and is consistent with still falling back to near-uniform predictions because no compatible checkpoints are actually present/loaded from the dataset paths. The smallest score-relevant improvement without changing model/inference semantics is to explicitly support Kaggle “HMS Baseline” style models where weights are shipped inside the **competition dataset itself** as `.pth`/`.pt` files in nested folders, and to fail fast into the uniform fallback only when truly nothing exists. I also tighten the weight-kind preference to avoid pulling unrelated checkpoints (which can average into noise), and I fix one small but impactful bug-risk: ensure `eeg_id` keys used for `eeg_specs` are consistently `str` to avoid silent KeyError/empty mix features. Everything else (architectures, softmax, TTA flip, mean ensembling, normalization, and submission schema) stays the same.'
- What this solution (achieved 1.41937) has done: 'Your current score is much worse than the target (lower-is-better), and the most likely cause is that you are still producing near-uniform predictions because no compatible checkpoints exist in the HMS dataset folder (so the ensemble never truly runs). Since this environment appears to contain only the competition data (no pretrained weights), the smallest legitimate way to move toward the target without changing your model/inference core is to add a “prior smoothing” fallback: compute the global class distribution from `train.csv` votes and use that as the fallback probability vector instead of uniform. This keeps evaluation semantics valid (probabilities sum to 1) and should reduce KL versus uniform because the true label distribution is highly imbalanced. I also keep your existing weight-loading logic unchanged; if real compatible weights exist, the script still use them and ignore the fallback.'
- What this solution (achieved 1.12336) has done: 'I keep your inference and submission pipeline unchanged, but improve the fallback behavior used when no compatible checkpoints load (which is the likely cause of your high KL). Specifically, instead of using a single global prior for every test row, I compute a per-`patient_id` prior from `train.csv` (with a global-prior backoff for unseen patients) and use that only in the fallback path; this is still label-leak-free and typically reduces KL versus a single prior. I also compute the priors using `eeg_id`-unique consolidated vote totals to better match the test granularity (one row per `eeg_id`) without altering any model logic. Everything else (weight discovery/loading, preprocessing, models, averaging, normalization, and CSV format) stays the same.'
- What this solution (achieved 0.79884) has done: 'Your current score (1.12336, lower-is-better) is still far above the target (0.4077), and given the provided environment it’s very likely you’re still in the “no compatible checkpoints loaded” fallback path. The smallest score-relevant improvement without changing any model/inference core is to make the fallback probabilities better match the test distribution by using a per-patient prior that is *smoothed toward the global prior* (to avoid overconfident patient priors that can increase KL when a patient’s test label differs). Concretely, I compute patient priors from train at the eeg_id level as you already do, but apply a simple convex blend with the global prior controlled by an effective sample-size parameter derived from each patient’s total votes. Everything else (weight discovery/loading, preprocessing, architectures, softmax, TTA, mean ensembling, and submission formatting) stays unchanged, and the script still always produces a valid `submission.csv`.'

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
    "weights_spec": "/kaggle/input/hms-harmful-brain-activity-classification/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-harmful-brain-activity-classification/hms-eeg",
    "weights_mix": "/kaggle/input/hms-harmful-brain-activity-classification/hms-mix",
    "flip": True,
    "train_csv": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
}




## === cell 2
def _is_weight_file(fp: str) -> bool:
    exts = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")
    return isinstance(fp, str) and os.path.isfile(fp) and fp.lower().endswith(exts)


def _discover_weight_files(path):
    if path is None or (not isinstance(path, str)):
        return []
    if _is_weight_file(path):
        return [path]
    if not os.path.isdir(path):
        return []
    files = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if _is_weight_file(fp):
            files.append(fp)
    return files


def _shallow_walk_for_ckpts(root, max_depth=6):
    exts = (".pt", ".pth", ".bin", ".ckpt", ".safetensors")
    out = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _prefer_ckpts_for_kind(files, kind):
    """
    Directly score-relevant: reduce chance of averaging incompatible checkpoints.
    Keep core logic (mean ensembling) unchanged; only rank/filter candidates.
    """
    if not files:
        return []
    kind = kind.lower()
    pos_terms = {
        "spec": ["spec", "spect", "spectrogram", "baseline", "efficientnet"],
        "eeg": ["eeg", "wave", "raw", "efficientnet"],
        "mix": ["mix", "fusion", "both", "concat", "hrnet"],
    }.get(kind, [])
    neg_terms = {
        "spec": ["eeg", "mix", "fusion", "hrnet"],
        "eeg": ["spec", "spect", "baseline", "mix", "fusion", "hrnet"],
        "mix": ["spec", "spect", "baseline", "efficientnet_b5"],
    }.get(kind, [])

    def score_fp(fp):
        name = os.path.basename(fp).lower()
        s = 0
        for t in pos_terms:
            if t in name:
                s += 2
        for t in neg_terms:
            if t in name:
                s -= 3
        for t in ["best", "final", "fold", "epoch", "ema", "checkpoint"]:
            if t in name:
                s += 1
        return s

    scored = [(score_fp(fp), fp) for fp in files]
    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = [fp for s, fp in scored if s >= -1]
    return filtered if filtered else files


def _find_weight_files_local_only(preferred_path, fallback_roots, kind=None):
    """
    Directly score-relevant: only search within intended competition dataset roots,
    but do it robustly enough to actually find nested ckpts when present.
    """
    candidates = []
    if isinstance(preferred_path, str):
        candidates = _discover_weight_files(preferred_path)

    if len(candidates) == 0:
        for root in fallback_roots:
            if isinstance(root, str) and os.path.isdir(root):
                candidates = _discover_weight_files(root)
                if len(candidates) > 0:
                    break

    if len(candidates) == 0:
        for root in fallback_roots:
            if isinstance(root, str) and os.path.isdir(root):
                candidates = _shallow_walk_for_ckpts(root, max_depth=6)
                if len(candidates) > 0:
                    break

    if kind is not None:
        candidates = _prefer_ckpts_for_kind(candidates, kind)
    return candidates


DATASET_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
NESTED_ROOT = os.path.join(DATASET_ROOT, "hms-harmful-brain-activity-classification")

CFG["weights_spec"] = _find_weight_files_local_only(
    CFG["weights_spec"],
    fallback_roots=[
        DATASET_ROOT,
        NESTED_ROOT,
        os.path.join(DATASET_ROOT, "hms-baseline"),
        os.path.join(NESTED_ROOT, "hms-baseline"),
        os.path.join(DATASET_ROOT, "baseline"),
        os.path.join(NESTED_ROOT, "baseline"),
        os.path.join(DATASET_ROOT, "weights"),
        os.path.join(NESTED_ROOT, "weights"),
        os.path.join(DATASET_ROOT, "models"),
        os.path.join(NESTED_ROOT, "models"),
    ],
    kind="spec",
)
CFG["weights_eeg"] = _find_weight_files_local_only(
    CFG["weights_eeg"],
    fallback_roots=[
        DATASET_ROOT,
        NESTED_ROOT,
        os.path.join(DATASET_ROOT, "hms-eeg"),
        os.path.join(NESTED_ROOT, "hms-eeg"),
        os.path.join(DATASET_ROOT, "eeg"),
        os.path.join(NESTED_ROOT, "eeg"),
        os.path.join(DATASET_ROOT, "weights"),
        os.path.join(NESTED_ROOT, "weights"),
        os.path.join(DATASET_ROOT, "models"),
        os.path.join(NESTED_ROOT, "models"),
    ],
    kind="eeg",
)
CFG["weights_mix"] = _find_weight_files_local_only(
    CFG["weights_mix"],
    fallback_roots=[
        DATASET_ROOT,
        NESTED_ROOT,
        os.path.join(DATASET_ROOT, "hms-mix"),
        os.path.join(NESTED_ROOT, "hms-mix"),
        os.path.join(DATASET_ROOT, "mix"),
        os.path.join(NESTED_ROOT, "mix"),
        os.path.join(DATASET_ROOT, "weights"),
        os.path.join(NESTED_ROOT, "weights"),
        os.path.join(DATASET_ROOT, "models"),
        os.path.join(NESTED_ROOT, "models"),
    ],
    kind="mix",
)

MAX_CKPTS_PER_KIND = 6
CFG["weights_spec"] = CFG["weights_spec"][:MAX_CKPTS_PER_KIND]
CFG["weights_eeg"] = CFG["weights_eeg"][:MAX_CKPTS_PER_KIND]
CFG["weights_mix"] = CFG["weights_mix"][:MAX_CKPTS_PER_KIND]

print(
    "Discovered weights (local-only, after kind preference + cap):",
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


_MEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_func = TransformMel().to(_MEL_DEVICE)


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

            x_tensor = torch.from_numpy(x).to(_MEL_DEVICE)
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

    for item in tqdm(all_fs, desc="Precompute EEG mel-spectrograms (for mix model)"):
        eeg_id = item.rsplit(".", 1)[0]
        eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
        eeg_spec = spectrogram_from_eeg(eeg_path)
        all_specs[str(eeg_id)] = eeg_spec

    with open("eeg_specs_dict.pkl", "wb") as file:
        pickle.dump(all_specs, file)
else:
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
            r += 300

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
            raise ValueError("One of use_eeg/use_spec/use_mix must be True")
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
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)

        x2 = [x[:, i + 4 : i + 5, :, :] for i in range(4)]
        x2 = torch.cat(x2, dim=2)

        x = torch.cat([x1, x2], dim=3)
        x = torch.cat([x, x, x], dim=1)

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
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for _, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.to("cpu").numpy())
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

have_any_weights = (
    len(CFG["weights_spec"]) + len(CFG["weights_eeg"]) + len(CFG["weights_mix"])
) > 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Have any weights:", have_any_weights)


def _extract_state_dict(obj):
    sd = obj
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "weights",
            "ema_state_dict",
            "checkpoint",
        ):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break

    if isinstance(sd, dict):
        filtered = {}
        for k, v in sd.items():
            if torch.is_tensor(v):
                filtered[k] = v
        if len(filtered) > 0:
            sd = filtered

        for pref in ("module.",):
            if any(isinstance(k, str) and k.startswith(pref) for k in sd.keys()):
                sd = {k.replace(pref, "", 1): v for k, v in sd.items()}

        def strip_prefix_if_all(d, pref):
            keys = list(d.keys())
            if len(keys) > 0 and all(
                isinstance(k, str) and k.startswith(pref) for k in keys
            ):
                return {k[len(pref) :]: v for k, v in d.items()}
            return d

        for pref in (
            "model.model.",
            "model.",
            "net.",
            "encoder.",
            "backbone.",
            "module.model.",
        ):
            sd = strip_prefix_if_all(sd, pref)

    return sd


def _filter_state_dict_by_shape(model, state_dict):
    msd = model.state_dict()
    out = {}
    kept, dropped = 0, 0
    for k, v in state_dict.items():
        if (
            k in msd
            and hasattr(v, "shape")
            and hasattr(msd[k], "shape")
            and v.shape == msd[k].shape
        ):
            out[k] = v
            kept += 1
        else:
            dropped += 1
    return out, kept, dropped


def _should_skip_loaded(missing_keys, total_keys, max_missing_ratio=0.20):
    if total_keys <= 0:
        return True
    return (len(missing_keys) / total_keys) > max_missing_ratio


def _normalize_prob_vec(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.full_like(p, 1.0 / len(p), dtype=np.float64)
        return p.astype(np.float32)
    p = p / s
    return p.astype(np.float32)


def _train_patient_priors_fallback(
    train_csv_path: str, targets: list[str], smooth_strength: float = 50.0
) -> tuple[np.ndarray, dict]:
    """
    Directly score-relevant fallback improvement (when no checkpoints exist):
    - Still per-patient priors (leakage-free).
    - NEW: smooth each patient prior toward the global prior based on how many
      total votes that patient contributes (effective sample size).
      This reduces overconfident priors that can increase KL on test rows where
      the patient's label distribution shifts.
    Core inference/model logic is unchanged; only the fallback distribution is improved.
    """
    try:
        usecols = ["eeg_id", "patient_id"] + list(targets)
        tr = pd.read_csv(train_csv_path, usecols=usecols)

        agg = {t: "sum" for t in targets}
        tr_eeg = tr.groupby("eeg_id", as_index=False).agg(
            {**{"patient_id": "first"}, **agg}
        )

        global_votes = tr_eeg[targets].sum(axis=0).astype(np.float64).values
        global_prior = _normalize_prob_vec(global_votes)

        patient_priors = {}
        for pid, g in tr_eeg.groupby("patient_id"):
            votes = g[targets].sum(axis=0).astype(np.float64).values
            total_votes = float(np.maximum(votes.sum(), 0.0))

            if total_votes <= 0:
                patient_priors[int(pid)] = global_prior
                continue

            patient_prior_raw = _normalize_prob_vec(votes).astype(np.float64)
            global_prior64 = global_prior.astype(np.float64)

            alpha = total_votes / (total_votes + float(smooth_strength))
            blended = alpha * patient_prior_raw + (1.0 - alpha) * global_prior64
            patient_priors[int(pid)] = _normalize_prob_vec(blended)

        return global_prior, patient_priors
    except Exception as e:
        print(
            "Could not compute patient priors fallback; using uniform. Error:", repr(e)
        )
        global_prior = np.full((len(targets),), 1.0 / len(targets), dtype=np.float32)
        return global_prior, {}


if not have_any_weights:
    global_prior, patient_priors = _train_patient_priors_fallback(
        CFG["train_csv"], TARGETS, smooth_strength=50.0
    )
    preds = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)
    for i, pid in enumerate(test_df["patient_id"].values):
        preds[i] = patient_priors.get(int(pid), global_prior)
    predictions = preds
else:
    predictions_list = []

    if len(CFG["weights_spec"]) > 0:
        print("infer with weights_spec")
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_spec=True
        )
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        for model_weight in CFG["weights_spec"]:
            model = NetSpec()
            ckpt = torch.load(model_weight, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            state_dict, kept, dropped = _filter_state_dict_by_shape(model, state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            total_keys = len(model.state_dict())
            if _should_skip_loaded(missing, total_keys):
                print(
                    f"Skip spec (likely wrong ckpt): {os.path.basename(model_weight)} | kept={kept} dropped={dropped} missing={len(missing)}/{total_keys} unexpected={len(unexpected)}"
                )
                del model, ckpt, state_dict
                torch.cuda.empty_cache()
                gc.collect()
                continue
            print(
                f"Loaded spec: {os.path.basename(model_weight)} | kept={kept} dropped={dropped} missing={len(missing)}/{total_keys} unexpected={len(unexpected)}"
            )
            model.to(device)
            prediction_dict = inference_function(test_loader, model, device)
            predictions_list.append(prediction_dict["predictions"])
            del model, ckpt, state_dict
            torch.cuda.empty_cache()
            gc.collect()

    if len(CFG["weights_eeg"]) > 0:
        print("infer with weights_eeg")
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_eeg=True
        )
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        for model_weight in CFG["weights_eeg"]:
            model = NetEeg()
            ckpt = torch.load(model_weight, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            state_dict, kept, dropped = _filter_state_dict_by_shape(model, state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            total_keys = len(model.state_dict())
            if _should_skip_loaded(missing, total_keys):
                print(
                    f"Skip eeg (likely wrong ckpt): {os.path.basename(model_weight)} | kept={kept} dropped={dropped} missing={len(missing)}/{total_keys} unexpected={len(unexpected)}"
                )
                del model, ckpt, state_dict
                torch.cuda.empty_cache()
                gc.collect()
                continue
            print(
                f"Loaded eeg: {os.path.basename(model_weight)} | kept={kept} dropped={dropped} missing={len(missing)}/{total_keys} unexpected={len(unexpected)}"
            )
            model.to(device)
            prediction_dict = inference_function(test_loader, model, device)
            predictions_list.append(prediction_dict["predictions"])
            del model, ckpt, state_dict
            torch.cuda.empty_cache()
            gc.collect()

    if len(CFG["weights_mix"]) > 0:
        print("infer with weights_mix")
        test_dataset = AlaskaDataIter(
            test_df, training_flag=False, shuffle=False, use_mix=True
        )
        test_loader = DataLoader(
            test_dataset,
            CFG["batch_size"],
            num_workers=CFG["num_worker"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        for model_weight in CFG["weights_mix"]:
            model = NetMix()
            ckpt = torch.load(model_weight, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            state_dict, kept, dropped = _filter_state_dict_by_shape(model, state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            total_keys = len(model.state_dict())
            if _should_skip_loaded(missing, total_keys):
                print(
                    f"Skip mix (likely wrong ckpt): {os.path.basename(model_weight)} | kept={kept} dropped={dropped} missing={len(missing)}/{total_keys} unexpected={len(unexpected)}"
                )
                del model, ckpt, state_dict
                torch.cuda.empty_cache()
                gc.collect()
                continue
            print(
                f"Loaded mix: {os.path.basename(model_weight)} | kept={kept} dropped={dropped} missing={len(missing)}/{total_keys} unexpected={len(unexpected)}"
            )
            model.to(device)
            prediction_dict = inference_function(test_loader, model, device)
            predictions_list.append(prediction_dict["predictions"])
            del model, ckpt, state_dict
            torch.cuda.empty_cache()
            gc.collect()

    if len(predictions_list) == 0:
        print(
            "Warning: weights were discovered but none were compatible enough; falling back to patient priors."
        )
        global_prior, patient_priors = _train_patient_priors_fallback(
            CFG["train_csv"], TARGETS, smooth_strength=50.0
        )
        preds = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)
        for i, pid in enumerate(test_df["patient_id"].values):
            preds[i] = patient_priors.get(int(pid), global_prior)
        predictions = preds
    else:
        predictions = np.array(predictions_list, dtype=np.float32)
        predictions = np.mean(predictions, axis=0)

predictions = np.asarray(predictions, dtype=np.float32)
if (
    predictions.ndim != 2
    or predictions.shape[1] != len(TARGETS)
    or predictions.shape[0] != len(test_df)
):
    raise ValueError(
        f"Bad predictions shape {predictions.shape}; expected ({len(test_df)}, {len(TARGETS)})"
    )

row_sums = predictions.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums <= 0, 1.0, row_sums)
predictions = predictions / row_sums
predictions = np.clip(predictions, 1e-7, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    "Predictions shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)



## === cell 11
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row-sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
