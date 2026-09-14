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

0.6177724945174508

# 6. Current score

0.829

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate path/weight-loading failures by (1) pointing the script at the actual competition input directory and (2) making weight discovery robust (only real model files; skip missing dirs) and adding a safe fallback to uniform predictions when no weights are available. I also remove the hard CUDA dependency in the EEG→mel-spec preprocessing (use CPU when no GPU), and avoid generating the heavy `eeg_specs_dict.pkl` entirely unless mix-inference is requested. Finally, I guarantee the submission has the correct column order, float probabilities that sum to 1 per row, and is written to `submission.csv` end-to-end.'
- What this solution (achieved 1.40995) has done: 'Your current score is much worse than the target (lower-is-better), and the biggest likely cause is that you’re effectively submitting near-uniform predictions because no model weights are being found/loaded from the specified directories. I make a minimal, score-relevant fix by automatically resolving the correct Kaggle input paths for weights (both `/kaggle/input/...` and the provided `/kaggle/data/...` mirrors), then re-run the same ensemble logic so real checkpoints are used if present. To avoid silent mis-loading (which can also yield near-uniform softmax outputs), I add a strict key-match fallback that tries common checkpoint key prefixes and reports how many parameters actually loaded. These changes keep your model architectures, preprocessing, inference loop, and submission semantics identical, but should move the score substantially down toward the target by using the intended trained weights.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.6178), and the most likely reason is that the ensemble is still effectively running with randomly initialized weights (or no weights), producing near-uniform probabilities. I make the smallest score-relevant changes to (1) point the weight folders at the actual available dataset paths in your environment, (2) ensure we recursively discover checkpoint files under those folders, and (3) enforce a more robust state_dict extraction/loading that avoids “almost nothing loaded” cases. This preserves your model architectures, preprocessing, inference loop, and averaging logic, but makes it much more likely you are using real trained checkpoints, which should move KL down toward the target. The submission writing and probability normalization remain the same.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995; lower-is-better) is far above the target (0.6178), and the most likely cause is still that no real checkpoints are being loaded so the submission is close to uniform. I make a minimal, score-relevant change to weight discovery: in addition to searching a few fixed folders, we also search inside the mounted competition dataset for any `.pth/.pt/.bin` files whose names look like model checkpoints, then use those for the existing EEG/SPEC/MIX inference (no architecture or inference changes). I also ensure we don’t accidentally load obviously-wrong files (e.g., very tiny files) and keep the exact same softmax+averaging+normalization semantics for the submission. This should substantially reduce KL toward the target by actually using trained weights when they exist in your environment.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the most likely reason is still that no real trained checkpoints are being loaded, so the ensemble effectively behaves like near-random/uniform predictions. I make the smallest changes that increase the chance of loading correct weights without changing any model architectures or inference semantics: (1) restrict checkpoint discovery to realistic model files (and avoid accidentally loading unrelated `.pth`/`.pt` like optimizer snapshots), (2) filter discovered files by filename patterns and size, and (3) skip checkpoints that load with extremely low parameter match (which otherwise silently degrades to random). This should move KL down toward the target by ensuring inference uses actual trained parameters when present, while keeping your preprocessing, models, flip-TTA, softmax, and averaging exactly the same. The submission writing stays identical, still guaranteeing valid probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far from the target (0.6178), and the most likely reason is that you’re still not actually using meaningful trained checkpoints (or you’re mixing incompatible checkpoints), so predictions collapse toward near-uniform. I make the smallest score-relevant change: if no compatible weights are found for a given model type, we not “fallback-discover” arbitrary checkpoints and reuse them across EEG/SPEC/MIX (that often loads poorly and harms KL), and we also remove the aggressive “skip if <60% params matched” filter so valid but slightly different checkpoints still contribute instead of leaving you with few/no models. Finally, I keep your exact inference/softmax/averaging logic, but add a very light safeguard to ensure we only attempt to load checkpoints that match the intended architecture by checking key shapes before rejecting them. These changes should move KL downward toward the target by increasing the chance that at least one real, compatible checkpoint is used rather than uniform predictions.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the target (0.6178), and the most likely cause is still that you’re not actually loading any real checkpoints (so you end up with near-uniform/random-ish probabilities). I make a minimal, score-relevant change to weight discovery: if the configured weight folders yield zero checkpoints, we additionally search only under `/kaggle/input` and `/kaggle/data` for likely checkpoint files (excluding the competition data folders) and then route candidates to the correct model type by a quick dry-run “load compatibility” check against each architecture. This keeps your models, preprocessing, inference, softmax, averaging, and submission semantics the same, but makes it much more likely at least one compatible trained checkpoint is used, pulling KL down toward the target. I also fix a small bug in `brain_lead` (RR was never used) which is a correctness fix that can improve EEG-feature quality without changing the approach.'
- What this solution (achieved 1.40995) has done: 'Your current KL is much worse than the target (lower-is-better), and the most likely remaining issue is still that you are effectively not loading any meaningful trained checkpoints, so predictions are near-uniform. I make a minimal, score-relevant change: search specifically for model checkpoints inside the provided competition dataset tree (including the mirrored `/kaggle/data/...`), but only under small/likely “model/ckpt/weights” folders and only for realistic checkpoint extensions—this avoids scanning huge EEG/spectrogram folders and fits the time limit. Then, if weights are found, we keep your exact same architectures and inference averaging, but use a slightly less strict compatibility gate (still shape-based) to avoid throwing away valid checkpoints due to naming differences. This should move the score down toward your target by increasing the chance that real trained weights are actually used.'
- What this solution (achieved 1.41937) has done: 'Your KL is far above the target (lower is better), and the most likely reason is still that no meaningful trained checkpoints are being loaded, so predictions stay close to uniform. I make a minimal, score-relevant fix by (1) explicitly disabling the slow/low-yield “scan anywhere for checkpoints” fallback (which typically finds none or incompatible files in this environment) and instead (2) use a guaranteed-valid, metric-aligned baseline: class-prior probabilities computed from `train.csv` vote distributions. This preserves your inference/model code paths (no architecture or loop changes), still produces a valid submission, and should move the score substantially down toward the target versus uniform. The submission remain correctly normalized and written to `submission.csv`.'
- What this solution (achieved 1.64506) has done: 'Your current KL (1.41937, lower-is-better) is far above the target (0.6178), and the most likely remaining issue is that inference is still effectively producing weak/near-uniform predictions—either because no checkpoints are found/loaded, or because test-time inference is mis-scaled. I make a minimal, metric-aligned improvement that does not change any model architecture or inference loops: use a stronger fallback than global class priors by computing per-patient class priors from `train.csv` (patient-level label distribution) and mapping them to `test.csv` patients, with a safe fallback to global priors for unseen patients. This is directly relevant to KL because it yields better-calibrated probabilities than uniform/global priors when models are absent/ineffective, while still producing valid row-normalized probabilities. If real checkpoints do get loaded, the existing ensemble path is unchanged.'
- What this solution (achieved 0.83217) has done: 'Your current KL (1.645, lower-is-better) is far from the target (0.618), and the most direct way to move toward the target without changing any model/inference logic is to make the “no-weights / ineffective-weights” fallback much more metric-aligned. I keep your model code untouched, but replace the patient-mean prior with an aggregated patient prior using summed votes (more faithful to label distribution), and add a small Dirichlet/Laplace smoothing so no class probability collapses to zero (helps KL stability). I also add a very conservative mixture with the global prior (only in fallback) to reduce overconfident patient priors for patients with few samples, which typically reduces KL. Submission formatting, normalization, paths, and the whole ensemble path remain unchanged.'
- What this solution (achieved 0.83217) has done: 'Your current KL (0.83217; lower is better) is still above the target (0.61777), and given this script typically ends up in the “no model preds” branch, the safest way to move toward target is to make that fallback prior more informative without changing any model/inference logic. I keep the exact same patient-vote-summed prior with Laplace smoothing, but also condition it on test `spectrogram_id` (which correlates strongly with EEG segment labels) by building a smoothed spectrogram-level prior from train and shrinking it toward the patient/global prior based on sample counts. This is a minimal, metric-aligned change (better calibrated probabilities for KL) and only affects the fallback path when no checkpoints are used; if weights do load, predictions are unchanged. Submission writing and normalization remain identical and still guarantee valid probabilities summing to 1.'
- What this solution (achieved 0.83217) has done: 'Your current KL (0.832) is worse than the target (0.618), so we should cautiously improve calibration without changing any model/inference code paths. Since this notebook often ends up in the “no model preds” fallback, the smallest score-relevant change is to strengthen that fallback prior: add an `eeg_id`-level prior (train has many rows per eeg_id) and combine it with the existing spectrogram+patient+global shrinkage by sample-count weights. This remains a pure prior-based fallback (no architecture/loop/feature changes) and typically reduces KL by making probabilities more specific yet still smoothed. All submission formatting/normalization stays identical and we still write a valid `submission.csv`.'
- What this solution (achieved 0.829) has done: 'Your current KL (0.832, lower-is-better) is still worse than the target (0.618), and given this code often falls back to priors (no usable checkpoints), the most score-relevant minimal change is to improve the fallback prior calibration without touching any model architectures or inference loops. I keep the same vote-summed + Laplace smoothing + patient/eeg/spec shrinkage structure, but make the shrinkage data-adaptive by using each group’s total vote mass (a proxy for annotator/evidence count) instead of row counts, which better matches the KL objective. I also add one very small, conservative blend of the final prior with the global prior (only in fallback) to reduce overconfident tails that can hurt KL. All paths, submission formatting, and probability normalization remain unchanged, and if real model predictions exist the behavior is identical.'
- What this solution (achieved 0.829) has done: 'Your current KL (0.829; lower-is-better) is still above the target (0.6178), and this script most likely reaches that score via the fallback prior path (few/no compatible checkpoints). To move cautiously toward the target without touching model architectures or inference semantics, I only strengthen the fallback by adding a simple, metric-aligned “expert_consensus → vote distribution” prior learned from `train.csv`, then shrink/blend it with the existing spectrogram/eeg/patient/global priors using the same smoothing/normalization approach. This is minimal (just adds one more prior source) and should reduce KL by better calibrating rows where metadata correlates with labels. If checkpoints do load, predictions remain unchanged.'

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
}




## === cell 2
def _resolve_kaggle_path(p: str) -> str:
    """
    Why: Assets may exist under /kaggle/data/... instead of /kaggle/input/...
    Keeping identical logic but making paths valid prevents falling back to random/uniform predictions.
    """
    if p is None or not isinstance(p, str):
        return p
    if os.path.exists(p):
        return p
    if p.startswith("/kaggle/input/"):
        alt = p.replace("/kaggle/input/", "/kaggle/data/", 1)
        if os.path.exists(alt):
            return alt
    if p.startswith("/kaggle/data/"):
        alt = p.replace("/kaggle/data/", "/kaggle/input/", 1)
        if os.path.exists(alt):
            return alt
    return p


def _looks_like_model_ckpt(filename: str) -> bool:
    """
    Why: During discovery, we can accidentally pick up non-model torch files.
    Filtering by common naming patterns is a minimal, score-relevant change to increase odds
    we load real trained weights (reducing KL) without changing model/inference logic.
    """
    fn = filename.lower()
    good_tokens = (
        "best",
        "checkpoint",
        "ckpt",
        "fold",
        "epoch",
        "model",
        "state",
        "weights",
        "ema",
    )
    bad_tokens = ("optimizer", "sched", "scheduler", "scaler", "rng", "train_state")
    if any(t in fn for t in bad_tokens):
        return False
    return any(t in fn for t in good_tokens)


def list_weight_files(path):
    """
    Why: Many Kaggle notebooks store weights in nested subdirectories (e.g., fold*/best.pth).
    Recursive discovery is a minimal change that helps ensure we actually find and load real checkpoints.
    """
    if path is None:
        return []
    if isinstance(path, (list, tuple)):
        files = []
        for p in path:
            files.extend(list_weight_files(p))
        seen = set()
        out = []
        for fp in files:
            if fp not in seen:
                out.append(fp)
                seen.add(fp)
        return out

    path = _resolve_kaggle_path(path)
    if not (isinstance(path, str) and os.path.exists(path)):
        return []

    exts = {".pt", ".pth", ".bin"}
    out = []
    if os.path.isfile(path) and (os.path.splitext(path)[1].lower() in exts):
        return [path]

    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(root, fn))
    return sorted(out)


def _pick_existing_weight_root(preferred: str, candidates: list[str]) -> str:
    """
    Why: When the configured weight dataset doesn't exist, try a small set of likely alternatives
    within the already-mounted competition dataset. This is still 'weight discovery', not changing modeling.
    """
    preferred = _resolve_kaggle_path(preferred)
    if isinstance(preferred, str) and os.path.exists(preferred):
        return preferred
    for c in candidates:
        c2 = _resolve_kaggle_path(c)
        if os.path.exists(c2):
            return c2
    return preferred


def _discover_ckpts_in_competition_tree(
    comp_roots: list[str], max_files: int = 200
) -> list[str]:
    """
    Why (score-relevant): In this environment, trained weights are often stored INSIDE the competition dataset
    folder (or its mirror). Prior logic avoided scanning that tree entirely, which can leave us with zero weights
    and near-uniform predictions (high KL).
    Minimal fix: scan only likely small subfolders (models/weights/ckpt/checkpoint) under the competition root,
    while still excluding massive data folders, to find real checkpoints without timing out.
    """
    exts = {".pt", ".pth", ".bin"}
    likely_dirs = (
        "models",
        "model",
        "weights",
        "weight",
        "checkpoints",
        "checkpoint",
        "ckpt",
        "ckpts",
    )
    bad_dirs = {
        "train_eegs",
        "test_eegs",
        "train_spectrograms",
        "test_spectrograms",
        "example_figures",
    }

    hits = []
    for cr in comp_roots:
        cr = _resolve_kaggle_path(cr)
        if not (isinstance(cr, str) and os.path.exists(cr)):
            continue

        for ld in likely_dirs:
            p = os.path.join(cr, ld)
            if not os.path.exists(p):
                continue
            for root, dirs, files in os.walk(p):
                base = os.path.basename(root)
                if base in bad_dirs:
                    dirs[:] = []
                    continue
                for fn in files:
                    if os.path.splitext(fn)[1].lower() not in exts:
                        continue
                    if not _looks_like_model_ckpt(fn):
                        continue
                    fp = os.path.join(root, fn)
                    try:
                        if os.path.getsize(fp) < 1_000_000:
                            continue
                    except OSError:
                        continue
                    hits.append(fp)
                    if len(hits) >= max_files:
                        return sorted(set(hits))

        try:
            for name in os.listdir(cr):
                p = os.path.join(cr, name)
                if not os.path.isdir(p):
                    continue
                lname = name.lower()
                if lname in bad_dirs:
                    continue
                if not any(tok in lname for tok in likely_dirs):
                    continue
                for root, dirs, files in os.walk(p):
                    base = os.path.basename(root)
                    if base in bad_dirs:
                        dirs[:] = []
                        continue
                    for fn in files:
                        if os.path.splitext(fn)[1].lower() not in exts:
                            continue
                        if not _looks_like_model_ckpt(fn):
                            continue
                        fp = os.path.join(root, fn)
                        try:
                            if os.path.getsize(fp) < 1_000_000:
                                continue
                        except OSError:
                            continue
                        hits.append(fp)
                        if len(hits) >= max_files:
                            return sorted(set(hits))
        except Exception:
            pass

    return sorted(set(hits))


CFG["data"] = _resolve_kaggle_path(CFG["data"])

_comp_root = _resolve_kaggle_path(
    "/kaggle/input/hms-harmful-brain-activity-classification"
)
_alt_comp_root = _resolve_kaggle_path(
    "/kaggle/data/hms-harmful-brain-activity-classification"
)
CFG["_comp_roots"] = [p for p in [_comp_root, _alt_comp_root] if os.path.exists(p)]

default_weight_candidates = []
for r in CFG["_comp_roots"]:
    for sub in ("weights", "checkpoints", "models", "model", "ckpt", "ckpts"):
        default_weight_candidates.append(os.path.join(r, sub))

CFG["weights_spec"] = _pick_existing_weight_root(
    CFG.get("weights_spec"), candidates=default_weight_candidates
)
CFG["weights_eeg"] = _pick_existing_weight_root(
    CFG.get("weights_eeg"), candidates=default_weight_candidates
)
CFG["weights_mix"] = _pick_existing_weight_root(
    CFG.get("weights_mix"), candidates=default_weight_candidates
)

CFG["weights_spec"] = list_weight_files(CFG.get("weights_spec"))
CFG["weights_eeg"] = list_weight_files(CFG.get("weights_eeg"))
CFG["weights_mix"] = list_weight_files(CFG.get("weights_mix"))

CFG["weights_spec"] = [
    p for p in CFG["weights_spec"] if _looks_like_model_ckpt(os.path.basename(p))
]
CFG["weights_eeg"] = [
    p for p in CFG["weights_eeg"] if _looks_like_model_ckpt(os.path.basename(p))
]
CFG["weights_mix"] = [
    p for p in CFG["weights_mix"] if _looks_like_model_ckpt(os.path.basename(p))
]

if len(CFG["weights_eeg"]) + len(CFG["weights_spec"]) + len(CFG["weights_mix"]) == 0:
    comp_hits = _discover_ckpts_in_competition_tree(CFG["_comp_roots"], max_files=240)
    if comp_hits:
        print(
            f"Found {len(comp_hits)} checkpoint candidates inside competition tree (targeted scan)."
        )
    else:
        print("No checkpoint candidates found inside competition tree (targeted scan).")

print("Resolved paths:")
print("  data:", CFG["data"])
print("  comp_roots:", CFG["_comp_roots"])
print("Found weights (initial):")
print("  spec:", len(CFG["weights_spec"]))
print("  eeg :", len(CFG["weights_eeg"]))
print("  mix :", len(CFG["weights_mix"]))
if len(CFG["weights_spec"]) > 0:
    print("  first spec weight:", CFG["weights_spec"][0])
if len(CFG["weights_eeg"]) > 0:
    print("  first eeg weight:", CFG["weights_eeg"][0])
if len(CFG["weights_mix"]) > 0:
    print("  first mix weight:", CFG["weights_mix"][0])



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

data_dir = _resolve_kaggle_path(
    "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
)
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

_SPEC_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        image = self.wave_transform(x)
        return image


transform_func = TransformMel().to(_SPEC_DEVICE)


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

            x_tensor = torch.from_numpy(x.astype(np.float32)).to(_SPEC_DEVICE)
            mel_spec = transform_func(x_tensor)
            mel_spec = mel_spec.detach().cpu().numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

    return img


if len(CFG.get("weights_mix", [])) > 0:
    if not os.path.exists("eeg_specs_dict.pkl"):
        all_fs = os.listdir(data_dir)
        all_specs = {}

        for item in tqdm(all_fs, desc="Building eeg_specs_dict.pkl"):
            eeg_id = item.rsplit(".", 1)[0]
            eeg_path = _resolve_kaggle_path(
                f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{eeg_id}.parquet"
            )
            all_specs[eeg_id] = spectrogram_from_eeg(eeg_path)

        with open("eeg_specs_dict.pkl", "wb") as file:
            pickle.dump(all_specs, file)
        del all_specs
        gc.collect()




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
        eeg_path = _resolve_kaggle_path(
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
        spec_path = _resolve_kaggle_path(
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
            raise ValueError("Must set one of use_eeg/use_spec/use_mix=True")
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
        self.model = timm.create_model("efficientnet_b2", pretrained=False, in_chans=4)
        self.fc = nn.Linear(1408, 6, bias=True)
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
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.to("cpu").numpy())
    return {"predictions": np.concatenate(preds)}




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

all_model_preds = []


def _extract_state_dict(state):
    """
    Why: Many checkpoints wrap weights under different keys; extracting correctly ensures we load trained parameters
    instead of leaving the model near-random (which inflates KL).
    """
    if isinstance(state, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state


def _normalize_state_dict_keys(sd):
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


def _load_with_fallbacks(model, sd):
    """
    Why: If we load with the wrong prefix, strict=False will 'succeed' but match almost nothing.
    We try a few common prefix-stripping patterns and keep the best match.
    """
    if not isinstance(sd, dict):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return missing, unexpected, None

    sd = _normalize_state_dict_keys(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    matched = len(model.state_dict()) - len(missing)

    if matched < max(10, int(0.2 * len(model.state_dict()))):
        candidates = []
        for prefix in ("model.", "net.", "module.model.", "module.net.", "backbone."):
            candidates.append(
                {
                    (k[len(prefix) :] if k.startswith(prefix) else k): v
                    for k, v in sd.items()
                }
            )
        best = (matched, missing, unexpected)
        for alt_sd in candidates:
            m2, u2 = model.load_state_dict(alt_sd, strict=False)
            matched2 = len(model.state_dict()) - len(m2)
            if matched2 > best[0]:
                best = (matched2, m2, u2)
        matched, missing, unexpected = best[0], best[1], best[2]

    return missing, unexpected, matched


def _shape_match_ratio(model, sd):
    """
    Why (score-relevant): A checkpoint can have many key-name matches but wrong tensor shapes (different arch),
    producing poor loads or silent partial loads. We compute a cheap compatibility ratio to skip clearly-wrong ones.
    """
    if not isinstance(sd, dict):
        return 0.0
    msd = model.state_dict()
    same = 0
    total = 0
    for k, v in sd.items():
        if k in msd:
            total += 1
            try:
                if tuple(v.shape) == tuple(msd[k].shape):
                    same += 1
            except Exception:
                pass
    return same / max(1, total)


def run_ensemble(weight_list, dataset_kwargs, model_ctor):
    preds_list = []
    if len(weight_list) == 0:
        return preds_list

    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, **dataset_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(CFG["num_worker"] > 0),
    )

    for model_weight in weight_list:
        model = model_ctor()
        try:
            state = torch.load(model_weight, map_location="cpu")
        except Exception as e:
            print(f"Skip {model_weight} (torch.load failed): {e}")
            del model
            gc.collect()
            continue

        state_dict = _extract_state_dict(state)
        if isinstance(state_dict, dict):
            state_dict = _normalize_state_dict_keys(state_dict)

        compat = (
            _shape_match_ratio(model, state_dict)
            if isinstance(state_dict, dict)
            else 0.0
        )
        if isinstance(state_dict, dict) and compat < 0.05:
            print(
                f"Skip {os.path.basename(model_weight)} due to extremely low shape-compatibility ({compat:.2%})."
            )
            del model, state, state_dict
            gc.collect()
            continue

        missing, unexpected, matched = _load_with_fallbacks(model, state_dict)

        if matched is not None:
            total = len(model.state_dict())
            ratio = matched / max(1, total)
            print(
                f"Loaded {os.path.basename(model_weight)}: matched_params={matched}/{total} ({ratio:.2%}), "
                f"shape_compat={compat:.2%}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
        else:
            print(
                f"Loaded {os.path.basename(model_weight)}: shape_compat={compat:.2%}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        preds_list.append(prediction_dict["predictions"])

        del model, state, state_dict, prediction_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return preds_list


def _route_discovered_checkpoints_to_archs(
    discovered: list[str],
) -> dict[str, list[str]]:
    """
    Why (score-relevant): If the configured weight dirs are empty, we must still locate weights.
    But mixing random checkpoints across architectures often produces near-uniform output.
    Minimal fix: try each discovered ckpt against each model type and keep it only where it is compatible.
    """
    routed = {"eeg": [], "spec": [], "mix": []}
    if not discovered:
        return routed

    discovered = sorted(discovered)[:160]

    probes = {"eeg": NetEeg(), "spec": NetSpec(), "mix": NetMix()}

    for fp in discovered:
        try:
            state = torch.load(fp, map_location="cpu")
        except Exception:
            continue
        sd = _extract_state_dict(state)
        if isinstance(sd, dict):
            sd = _normalize_state_dict_keys(sd)

        best_type = None
        best_score = 0.0
        for t, model in probes.items():
            score = _shape_match_ratio(model, sd) if isinstance(sd, dict) else 0.0
            if score > best_score:
                best_score = score
                best_type = t

        if best_type is not None and best_score >= 0.20:
            routed[best_type].append(fp)

    for k in routed:
        routed[k] = sorted(routed[k])[:10]
    return routed


if len(CFG["weights_eeg"]) + len(CFG["weights_spec"]) + len(CFG["weights_mix"]) == 0:
    comp_hits = _discover_ckpts_in_competition_tree(CFG["_comp_roots"], max_files=240)
    routed = _route_discovered_checkpoints_to_archs(comp_hits)
    CFG["weights_eeg"] = routed["eeg"]
    CFG["weights_spec"] = routed["spec"]
    CFG["weights_mix"] = routed["mix"]

print("infer with weights_eeg")
all_model_preds.extend(run_ensemble(CFG["weights_eeg"], dict(use_eeg=True), NetEeg))

print("infer with weights_spec")
all_model_preds.extend(run_ensemble(CFG["weights_spec"], dict(use_spec=True), NetSpec))

print("infer with weights_mix")
all_model_preds.extend(run_ensemble(CFG["weights_mix"], dict(use_mix=True), NetMix))

if len(all_model_preds) == 0:
    train_csv = _resolve_kaggle_path(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    if not os.path.exists(train_csv):
        train_csv = _resolve_kaggle_path("/kaggle/data/train.csv")

    usecols = ["patient_id", "spectrogram_id", "eeg_id", "expert_consensus"] + TARGETS
    train_df = pd.read_csv(train_csv, usecols=usecols)

    votes = train_df[TARGETS].to_numpy(np.float64)
    votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)
    votes = np.maximum(votes, 0.0)

    global_counts = votes.sum(axis=0)

    alpha = 0.5
    global_prior = (global_counts + alpha) / (global_counts.sum() + alpha * 6.0)

    train_df_counts = train_df[
        ["patient_id", "spectrogram_id", "eeg_id", "expert_consensus"]
    ].copy()
    for j, c in enumerate(TARGETS):
        train_df_counts[c] = votes[:, j]

    patient_counts = train_df_counts.groupby("patient_id")[TARGETS].sum()
    spec_counts = train_df_counts.groupby("spectrogram_id")[TARGETS].sum()
    eeg_counts = train_df_counts.groupby("eeg_id")[TARGETS].sum()

    cons_counts = train_df_counts.groupby("expert_consensus")[TARGETS].sum()

    patient_mass = patient_counts.sum(axis=1).astype(np.float64)
    spec_mass = spec_counts.sum(axis=1).astype(np.float64)
    eeg_mass = eeg_counts.sum(axis=1).astype(np.float64)
    cons_mass = cons_counts.sum(axis=1).astype(np.float64)

    preds = np.zeros((n_test, 6), dtype=np.float32)
    test_cons = None
    if "expert_consensus" in test_df.columns:
        test_cons = test_df["expert_consensus"].values

    for i, (pid, sid, eid) in enumerate(
        zip(
            test_df["patient_id"].values,
            test_df["spectrogram_id"].values,
            test_df["eeg_id"].values,
        )
    ):
        if pid in patient_counts.index:
            cnt_p = patient_counts.loc[pid].to_numpy(np.float64)
            cnt_p = np.maximum(cnt_p, 0.0)
            p_patient = (cnt_p + alpha) / (cnt_p.sum() + alpha * 6.0)

            m_p = float(patient_mass.loc[pid])
            w_p = m_p / (m_p + 80.0)
            p_pg = w_p * p_patient + (1.0 - w_p) * global_prior
        else:
            p_pg = global_prior

        if eid in eeg_counts.index:
            cnt_e = eeg_counts.loc[eid].to_numpy(np.float64)
            cnt_e = np.maximum(cnt_e, 0.0)
            p_eeg = (cnt_e + alpha) / (cnt_e.sum() + alpha * 6.0)

            m_e = float(eeg_mass.loc[eid])
            w_e = m_e / (m_e + 60.0)
            p_pge = w_e * p_eeg + (1.0 - w_e) * p_pg
        else:
            p_pge = p_pg

        if sid in spec_counts.index:
            cnt_s = spec_counts.loc[sid].to_numpy(np.float64)
            cnt_s = np.maximum(cnt_s, 0.0)
            p_spec = (cnt_s + alpha) / (cnt_s.sum() + alpha * 6.0)

            m_s = float(spec_mass.loc[sid])
            w_s = m_s / (m_s + 140.0)
            p = w_s * p_spec + (1.0 - w_s) * p_pge
        else:
            p = p_pge

        if test_cons is not None:
            ec = test_cons[i]
            if ec in cons_counts.index:
                cnt_c = cons_counts.loc[ec].to_numpy(np.float64)
                cnt_c = np.maximum(cnt_c, 0.0)
                p_cons = (cnt_c + alpha) / (cnt_c.sum() + alpha * 6.0)

                m_c = float(cons_mass.loc[ec])
                w_c = m_c / (m_c + 120.0)
                p = w_c * p_cons + (1.0 - w_c) * p

        p = 0.97 * p + 0.03 * global_prior

        s = float(np.sum(p))
        if not np.isfinite(s) or s <= 0:
            p = np.ones(6, dtype=np.float64) / 6.0
        else:
            p = p / s
        preds[i] = p.astype(np.float32)

    predictions = preds
    print(
        "No model preds. Using smoothed vote-summed priors with vote-mass shrinkage at spectrogram_id + eeg_id "
        "shrunk to patient/global priors, plus (if available) expert_consensus shrinkage for KL calibration. "
        f"Global prior: {global_prior.astype(np.float32)}"
    )
else:
    predictions = np.mean(np.stack(all_model_preds, axis=0), axis=0).astype(np.float32)

print("Predictions shape:", predictions.shape)



## === cell 11
predictions = np.asarray(predictions, dtype=np.float32)
if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != 6
):
    raise ValueError(
        f"Bad predictions shape {predictions.shape}, expected ({len(test_df)}, 6)"
    )

predictions = np.nan_to_num(
    predictions, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0
)

row_sums = predictions.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums <= 0, 1.0, row_sums)
predictions = predictions / row_sums

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sum stats:",
    float(sub[TARGETS].sum(axis=1).min()),
    float(sub[TARGETS].sum(axis=1).max()),
)
