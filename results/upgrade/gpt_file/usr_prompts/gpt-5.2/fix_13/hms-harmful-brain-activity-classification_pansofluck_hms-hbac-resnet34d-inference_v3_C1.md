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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.5744040164959602

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the pipeline robust to missing pretrained fold weights by falling back to a valid baseline submission (uniform probabilities) so the notebook always produces `submission.csv` end-to-end. I also fix the `MergeError` by avoiding a strict one-to-one merge/validate and instead aligning predictions to `sample_submission` order via a mapping keyed by `eeg_id`, which is safe even if duplicates exist. These changes are score-neutral when weights are available (the original inference path is preserved), and they ensure a valid CSV is always written with rows summing to 1. Finally, I keep the same core model/dataset/inference logic untouched when weights exist.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) suggests the notebook is falling back to the uniform-probability baseline at inference time, which usually happens when the fold weight files aren’t found/loaded. The smallest change that should move you toward the target is to correctly resolve the real weights directory inside the provided dataset tree and to make the weight-loading robust to the common checkpoint formats (raw `state_dict` vs nested dict). I keep the same model, preprocessing, inference loop, and ensembling; the only changes are (1) broader weight-path discovery (including nested input/data locations) and (2) safer checkpoint parsing so weights actually load, avoiding the uniform fallback. This should improve the KL score substantially without changing the core approach.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the target (0.5744), and the most likely reason is still that inference is not actually using the intended trained fold weights (so predictions are effectively untrained/uniform-ish). I make a minimal, score-relevant patch to (1) broaden checkpoint discovery to also accept common fold filenames beyond `best_model_fold{k}.pth`, and (2) make `extract_state_dict` robust to the very common `module.` prefix / Lightning-style nesting so weights load successfully. This keeps the exact same model, preprocessing, inference loop, and fold-mean ensembling; it only changes how we locate and load the same weights. The submission writing/alignment logic stays identical and still guarantees row sums of 1.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the target (0.5744), and the most likely cause is that the notebook is still not loading the intended trained weights and is effectively producing near-uniform predictions. I make a minimal, score-relevant patch to (1) ensure `TRAINED_MODEL` resolves to a real directory that contains *all* fold checkpoints (not just fold0), and (2) make `extract_state_dict` robust to the common “timm backbone nested under `model.`” / Lightning-style prefixes so `load_state_dict(strict=True)` succeeds instead of silently failing earlier and triggering the uniform fallback. This preserves the exact same model, transforms, inference loop, and fold-mean ensembling; only checkpoint discovery/loading is adjusted. The submission alignment and row-sum normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current gap to the target is large (1.40995 vs 0.5744, lower-is-better), and the most likely cause is still that inference is not actually using the intended trained fold checkpoints (or is loading them but with key mismatches that silently fail earlier in your workflow). I make the smallest score-relevant changes to (1) discover checkpoints more reliably (including searching recursively under the dataset tree and accepting `.ckpt`), (2) robustly extract the true `state_dict` (including handling `ema_state_dict` and partial prefix stripping), and (3) load with an explicit, safe fallback to `strict=False` only when keys mismatch (so we still use as much trained information as possible instead of reverting to uniform). The model/dataset/transforms/inference loop/ensemble remain the same, and the submission alignment and row-sum normalization stay intact. This should move the KL score substantially downward toward your target without changing core evaluation semantics.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (1.40995 vs 0.5744, lower-is-better), and the most likely cause is still that the notebook is effectively not using the intended trained weights and/or is loading them only partially (so predictions stay near-uniform). I make a minimal, score-relevant patch to (1) resolve the correct checkpoint directory more reliably by explicitly searching for a directory that contains fold checkpoint files (instead of relying on the “all folds exist” requirement), and (2) load weights in a way that maximizes actually-applied trained parameters by auto-mapping common head key differences (`fc.*`/`classifier.*`/`head.*`) and only then falling back to `strict=False`. This keeps the exact same model, preprocessing, inference loop, and fold-mean ensembling; it only improves checkpoint discovery/loading so inference uses the trained models rather than default-like outputs. The submission alignment and row-sum normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.5744), which strongly suggests the notebook is still not actually loading the intended trained fold checkpoints and is effectively outputting near-uniform predictions. I make the smallest score-relevant changes to ensure we (1) resolve the correct checkpoint directory more reliably by also searching under `DATA` (your dataset root) and (2) correctly load weights from common Kaggle artifacts (including a top-level `model` object, and `module.` prefixes) while keeping the same model, transforms, inference loop, and fold-mean ensembling. I also force `.eval()` after loading and use AMP autocast during inference (no change to core semantics, but reduces numeric instability and speeds execution) to stay within time. The submission alignment and per-row normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.5744), and the most likely reason is that you’re still not actually applying the pretrained fold checkpoints (either because the directory search misses them or because `load_state_dict` silently ends up mostly-unloaded). I make minimal, score-relevant changes to (1) resolve the correct checkpoint directory by directly searching for fold checkpoint files (instead of globbing many directories), and (2) improve checkpoint parsing/loading so we reliably load the real model weights (handling `state_dict` nesting, `module.` prefixes, and `head.`/`fc.` mismatches) without changing the model, transforms, inference loop, or ensembling logic. I also ensure we never accidentally pick a random/incorrect checkpoint by ranking candidates by “fold coverage” and preferred keywords. These changes should move predictions away from near-uniform and substantially reduce KL toward your target while keeping the core approach identical.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (1.40995 vs 0.5744, lower-is-better), and the most likely cause is that your inference is still effectively running without the intended trained checkpoints (or loading the wrong artifact), yielding near-uniform predictions. I make the smallest score-relevant changes to (1) resolve the correct checkpoint directory by explicitly preferring directories that contain *multiple fold checkpoints* and deprioritizing unrelated `.pth` files (e.g., backbone weights), and (2) make checkpoint loading stricter-but-safe by selecting the best-matching state_dict (including handling Lightning `state_dict` keys like `model.model.*`) and refusing “bad” candidates rather than silently loading mostly-mismatched weights. This preserves your model, transforms, inference loop, and ensembling; it only improves which weights are found/loaded so predictions actually reflect the trained model. The submission writing/alignment and row-sum normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your score gap to the target is large (1.40995 vs 0.5744, lower-is-better), which is most consistent with inference still not applying the intended trained fold weights (falling back to near-uniform / poorly loaded checkpoints). I make the smallest score-relevant changes to ensure we actually find and load the correct fold checkpoints: (1) expand `TRAINED_MODEL` resolution to also consider the competition dataset tree and common “weights/output” subfolders, and (2) make checkpoint loading accept the common “full model saved” case by extracting `model.state_dict()` when present. I not change the model architecture, transforms, or inference loop; only weight discovery/loading is adjusted so predictions reflect trained parameters. The submission alignment and per-row probability normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the target (0.5744), and the most likely reason in this notebook is that inference is still not actually using the intended trained fold weights (so predictions stay near-uniform). I make minimal, score-relevant changes to (1) ensure checkpoint discovery prefers the intended “fold” artifacts and doesn’t get derailed by unrelated `.pth` files, and (2) make checkpoint parsing/loading robust to the most common Kaggle/Lightning/timm checkpoint structures so weights truly get applied. I not change the model architecture, dataset, transforms, inference loop, or ensembling—only the weight finding/loading reliability—so evaluation semantics remain the same. The submission alignment and per-row normalization stay intact to guarantee a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



## === cell 2
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

DATA = INPUT / "hms-harmful-brain-activity-classification"
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TRAINED_MODEL = INPUT / "hms-hbac-resnet34-baseline"

TMP = ROOT / "tmp"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(exist_ok=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

RANDAM_SEED = 1086
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)




## === cell 3
def _find_fold_weight_file(trained_model_dir: Path, fold_id: int) -> tp.Optional[Path]:
    """
    Score-relevant, minimal:
    Prefer fold-specific checkpoint filenames and avoid accidentally picking generic backbone weights.

    Patch: prioritize "best"/"checkpoint" containing names over generic "*fold{fold}*.pth" to reduce
    the chance we grab an unrelated artifact and end up effectively untrained.
    """
    if trained_model_dir is None or (not Path(trained_model_dir).exists()):
        return None
    trained_model_dir = Path(trained_model_dir)

    patterns_primary = [
        f"best_model_fold{fold_id}.pth",
        f"best_model_fold{fold_id}.pt",
        f"best_model_fold{fold_id}.ckpt",
        f"best_fold{fold_id}.pth",
        f"best_fold{fold_id}.pt",
        f"best_fold{fold_id}.ckpt",
        f"checkpoint_fold{fold_id}.pth",
        f"checkpoint_fold{fold_id}.pt",
        f"checkpoint_fold{fold_id}.ckpt",
        f"*fold{fold_id}*best*.pth",
        f"*fold{fold_id}*best*.pt",
        f"*fold{fold_id}*best*.ckpt",
        f"*fold{fold_id}*checkpoint*.pth",
        f"*fold{fold_id}*checkpoint*.pt",
        f"*fold{fold_id}*checkpoint*.ckpt",
        f"*fold{fold_id}*ckpt*.pth",
        f"*fold{fold_id}*ckpt*.pt",
        f"*fold{fold_id}*ckpt*.ckpt",
    ]
    patterns_fallback = [
        f"fold{fold_id}.pth",
        f"fold{fold_id}.pt",
        f"fold{fold_id}.ckpt",
        f"fold_{fold_id}.pth",
        f"fold_{fold_id}.pt",
        f"fold_{fold_id}.ckpt",
        f"*fold{fold_id}*.pth",
        f"*fold{fold_id}*.pt",
        f"*fold{fold_id}*.ckpt",
        f"*fold_{fold_id}*.pth",
        f"*fold_{fold_id}*.pt",
        f"*fold_{fold_id}*.ckpt",
    ]

    def _bad_name(p: Path) -> bool:
        n = p.name.lower()
        return any(
            tok in n for tok in ["imagenet", "backbone", "encoder", "pretrain"]
        ) and ("fold" not in n)

    def _pick_best(matches: tp.List[Path]) -> Path:
        def _score(p: Path) -> tp.Tuple[int, int, int, str]:
            n = p.name.lower()
            key = 0
            if "best" in n:
                key -= 3
            if "checkpoint" in n or "ckpt" in n:
                key -= 2
            if n.endswith(".ckpt"):
                key -= 1
            return (key, len(p.name), len(p.parts), str(p))

        matches.sort(key=_score)
        return matches[0]

    for patterns, use_rglob in [
        (patterns_primary, False),
        (patterns_primary, True),
        (patterns_fallback, False),
        (patterns_fallback, True),
    ]:
        for pat in patterns:
            it = (
                trained_model_dir.rglob(pat)
                if use_rglob
                else trained_model_dir.glob(pat)
            )
            matches = [p for p in it if (p.is_file() and not _bad_name(p))]
            if matches:
                return _pick_best(matches)

    return None


def trained_weights_available(
    trained_model_dir: tp.Optional[Path], n_folds: int
) -> bool:
    if trained_model_dir is None:
        return False
    trained_model_dir = Path(trained_model_dir)
    if not trained_model_dir.exists():
        return False
    for fold_id in range(n_folds):
        if _find_fold_weight_file(trained_model_dir, fold_id) is None:
            return False
    return True


def _count_folds_found(d: Path, n_folds: int) -> int:
    if d is None or (not d.exists()) or (not d.is_dir()):
        return 0
    return sum(
        _find_fold_weight_file(d, fold_id) is not None for fold_id in range(n_folds)
    )


def _keyword_rank(p: Path) -> int:
    name = str(p).lower()
    score = 0
    for token, w in [
        ("hms", 3),
        ("hbac", 3),
        ("harmful", 2),
        ("baseline", 1),
        ("resnet", 1),
        ("resnest", 1),
        ("fold", 2),
        ("ckpt", 1),
        ("checkpoint", 1),
        ("best", 1),
        ("output", 1),
        ("weights", 1),
    ]:
        if token in name:
            score -= w
    return score


def resolve_trained_model_dir(preferred: Path, n_folds: int) -> tp.Optional[Path]:
    """
    Score-relevant, minimal:
    Pick a directory that maximizes fold coverage.

    Patch: explicitly prefer directories containing >=2 fold checkpoints (multi-fold artifacts),
    and only then accept single-fold dirs, to reduce the chance we resolve to the wrong place.
    """
    preferred = Path(preferred) if preferred is not None else None

    if preferred is not None and preferred.exists():
        cand_dirs = [preferred]
        for sub in [
            "weights",
            "weight",
            "checkpoints",
            "checkpoint",
            "ckpt",
            "outputs",
            "output",
            "models",
            "model",
        ]:
            cand_dirs.append(preferred / sub)
        for d in list(cand_dirs):
            if d.exists() and d.is_dir():
                for child in d.iterdir():
                    if child.is_dir() and child.name.lower() in {
                        "weights",
                        "checkpoints",
                        "ckpt",
                        "output",
                        "outputs",
                        "models",
                        "model",
                    }:
                        cand_dirs.append(child)

        for d in cand_dirs:
            if d.exists() and trained_weights_available(d, n_folds):
                return d

    search_roots: tp.List[Path] = []
    for base in [
        preferred,
        INPUT,
        DATA,
        ROOT / "kaggle" / "input",
        ROOT / "kaggle" / "data",
        ROOT / "data",
    ]:
        if base is None:
            continue
        base = Path(base)
        if base.exists():
            search_roots.append(base)

    fold_like_files: tp.List[Path] = []
    for root in search_roots:
        for ext in ("*.pth", "*.pt", "*.ckpt"):
            for p in root.rglob(ext):
                n = p.name.lower()
                if "fold" in n:
                    fold_like_files.append(p)

    if not fold_like_files:
        return preferred if (preferred is not None and preferred.exists()) else None

    candidate_dirs = set()
    for p in fold_like_files:
        candidate_dirs.add(p.parent)
        if p.parent.parent is not None:
            candidate_dirs.add(p.parent.parent)

    scored: tp.List[tp.Tuple[int, int, int, int, str, Path]] = []
    for d in candidate_dirs:
        folds_found = _count_folds_found(d, n_folds)
        if folds_found == 0:
            continue
        multi_fold_penalty = 0 if folds_found >= 2 else 50
        scored.append(
            (
                multi_fold_penalty,
                -folds_found,
                _keyword_rank(d),
                len(d.parts),
                str(d),
                d,
            )
        )

    if not scored:
        return preferred if (preferred is not None and preferred.exists()) else None

    scored.sort()
    best_dir = scored[0][-1]
    return best_dir


TRAINED_MODEL = resolve_trained_model_dir(TRAINED_MODEL, N_FOLDS)
print("Resolved TRAINED_MODEL directory:", TRAINED_MODEL)



## === cell 4
test = pd.read_csv(DATA / "test.csv")
sample_sub = pd.read_csv(DATA / "sample_submission.csv")

assert "eeg_id" in test.columns
assert "eeg_id" in sample_sub.columns
assert list(sample_sub.columns) == ["eeg_id"] + CLASSES



## === cell 5
existing = set(p.stem for p in TEST_SPEC_SPLIT.glob("*.npy"))
need = [int(s) for s in test["spectrogram_id"].values if str(s) not in existing]

for spec_id in tqdm(need, desc="Caching test spectrograms to .npy"):
    spec = pd.read_parquet(TEST_SPEC / f"{spec_id}.parquet")
    spec_arr = spec.fillna(0).values[:, 1:].T.astype("float32")  # (Hz, Time)
    spec_arr = np.nan_to_num(spec_arr, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(TEST_SPEC_SPLIT / f"{spec_id}.npy", spec_arr)

print(
    f"Cached/available test spectrograms: {len(list(TEST_SPEC_SPLIT.glob('*.npy')))} / {len(test)}"
)




## === cell 6
class HMSHBACSpecModel(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool, in_channels: int, num_classes: int
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        h = self.model(x)
        return h




## === cell 7
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)  # (Hz, Time)

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # (Hz, Time, 1)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img




## === cell 8
class CFG:
    model_name = "resnest50d_4s2x40d"
    img_size = 512
    max_epoch = 7
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"




## === cell 9
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device,
    *args,
    **kwargs,
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    img_paths = []
    labels = np.full((len(test), N_CLASSES), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

    test_data = {"image_paths": img_paths, "labels": [l for l in labels]}
    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose(
        [
            A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
            ToTensorV2(p=1.0),
        ]
    )
    return test_transform




## === cell 10
def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    use_amp = (device.type == "cuda") and bool(getattr(CFG, "enable_amp", True))
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", leave=False):
            x = to_device(batch["data"], device)
            if use_amp:
                with amp.autocast(device_type="cuda", dtype=torch.float16):
                    y = model(x)
            else:
                y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr




## === cell 11
test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    num_workers=2,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
print("Inference device:", device)




## === cell 12
def extract_state_dict(ckpt: tp.Any) -> tp.Dict[str, torch.Tensor]:
    """
    Score-relevant, minimal:
    Expand common nestings (including Lightning) and normalize prefixes so we actually load trained weights.

    Patch: handle additional common nesting keys ('model_ema', 'ema', 'module') and cases where ckpt['model']
    itself is a state_dict-like mapping with tensor values. This increases the chance weights truly load,
    improving KL toward the target.
    """
    state = None

    if isinstance(ckpt, dict):
        for k in (
            "ema_state_dict",
            "model_ema",
            "ema",
            "state_dict",
            "model_state_dict",
            "model",
            "module",
            "net",
            "weights",
        ):
            if k in ckpt:
                v = ckpt[k]
                if isinstance(v, dict):
                    if len(v) > 0 and all(
                        isinstance(vv, torch.Tensor) for vv in v.values()
                    ):
                        state = v
                        break
                    for kk in ("state_dict", "model_state_dict", "ema_state_dict"):
                        if kk in v and isinstance(v[kk], dict):
                            state = v[kk]
                            break
                    if state is not None:
                        break
                if isinstance(v, nn.Module):
                    state = v.state_dict()
                    break

        if (
            state is None
            and len(ckpt) > 0
            and all(isinstance(v, torch.Tensor) for v in ckpt.values())
        ):
            state = ckpt

    if state is None and isinstance(ckpt, nn.Module):
        state = ckpt.state_dict()

    if state is None:
        raise ValueError(
            f"Unrecognized checkpoint format with type={type(ckpt)} and keys={list(ckpt.keys())[:10] if isinstance(ckpt, dict) else 'NA'}"
        )

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    pref_candidates = [
        "model.model.",
        "model.",
        "net.",
        "ema_model.",
        "student.",
        "backbone.",
    ]
    for pref in pref_candidates:
        if len(state) > 0 and all(k.startswith(pref) for k in state.keys()):
            state = {k[len(pref) :]: v for k, v in state.items()}
            break

    keys = list(state.keys())
    if keys:
        n_pref = sum(k.startswith(("model.", "net.", "backbone.")) for k in keys)
        if n_pref / len(keys) > 0.7:

            def _strip(k: str) -> str:
                for pref in ("model.", "net.", "backbone."):
                    if k.startswith(pref):
                        return k[len(pref) :]
                return k

            state = {_strip(k): v for k, v in state.items()}

    return state


def _remap_head_keys_to_timm(
    state: tp.Dict[str, torch.Tensor]
) -> tp.Dict[str, torch.Tensor]:
    """
    Score-relevant, minimal:
    Remap final classifier naming differences so load_state_dict succeeds.
    """
    new_state = dict(state)

    if any(k.startswith("fc.") for k in new_state.keys()):
        if not any(k.startswith("head.fc.") for k in new_state.keys()):
            for k in list(new_state.keys()):
                if k.startswith("fc."):
                    new_state["head." + k] = new_state[k]

    if any(k.startswith("classifier.") for k in new_state.keys()):
        for suffix in ("weight", "bias"):
            src = f"classifier.{suffix}"
            dst = f"head.fc.{suffix}"
            if src in new_state and dst not in new_state:
                new_state[dst] = new_state[src]

    for suffix in ("weight", "bias"):
        src = f"head.{suffix}"
        dst = f"head.fc.{suffix}"
        if src in new_state and dst not in new_state:
            new_state[dst] = new_state[src]

    return new_state


def _try_load_best_state_dict(
    model: nn.Module, state: tp.Dict[str, torch.Tensor]
) -> bool:
    """
    Score-relevant, minimal:
    Try a small set of key-normalizations and keep the one that matches best.

    Patch: accept a slightly lower threshold (0.75 vs 0.85) because some checkpoints legitimately omit
    optimizer-only buffers or have minor naming differences; skipping them can force uniform fallback,
    which is much worse for KL.
    """
    candidates = []
    candidates.append(("as_is", state))
    candidates.append(("remap_head", _remap_head_keys_to_timm(state)))
    candidates.append(
        (
            "strip_model_prefix",
            {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state.items()
            },
        )
    )
    candidates.append(
        (
            "strip_net_prefix",
            {
                k.replace("net.", "", 1) if k.startswith("net.") else k: v
                for k, v in state.items()
            },
        )
    )
    candidates.append(
        (
            "strip_backbone_prefix",
            {
                k.replace("backbone.", "", 1) if k.startswith("backbone.") else k: v
                for k, v in state.items()
            },
        )
    )

    best = None
    best_loaded_ratio = -1.0

    total_model_keys = len(model.state_dict())

    for name, sd in candidates:
        try:
            incompatible = model.load_state_dict(sd, strict=False)
            missing = len(incompatible.missing_keys)
            loaded_ratio = (total_model_keys - missing) / max(1, total_model_keys)
            if loaded_ratio > best_loaded_ratio:
                best_loaded_ratio = loaded_ratio
                best = (name, sd, incompatible, loaded_ratio)
        except Exception:
            continue

    if best is None:
        return False

    name, sd, incompatible, loaded_ratio = best
    missing = len(incompatible.missing_keys)
    unexpected = len(incompatible.unexpected_keys)
    print(
        f"Best checkpoint variant: {name} | loaded_ratio={loaded_ratio:.3f} | missing={missing} unexpected={unexpected}"
    )

    if loaded_ratio < 0.75:
        print(
            "Checkpoint match too low; treating as unavailable for this fold to avoid near-random predictions."
        )
        return False

    model.load_state_dict(sd, strict=False)
    return True


available_fold_paths: tp.List[tp.Tuple[int, Path]] = []
if TRAINED_MODEL is not None and Path(TRAINED_MODEL).exists():
    for fold_id in range(N_FOLDS):
        p = _find_fold_weight_file(TRAINED_MODEL, fold_id)
        if p is not None:
            available_fold_paths.append((fold_id, p))

print("Available fold checkpoints:", [(f, str(p)) for f, p in available_fold_paths])

fold_preds = []
if len(available_fold_paths) > 0:
    for fold_id, model_path in available_fold_paths:
        print(f"\n[fold {fold_id}] Using weight file: {model_path}")

        model = HMSHBACSpecModel(
            model_name=CFG.model_name,
            pretrained=False,
            num_classes=N_CLASSES,
            in_channels=1,
        )

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = extract_state_dict(ckpt)

        ok = _try_load_best_state_dict(model, state_dict)
        if not ok:
            print(f"Skipping fold {fold_id} due to poor checkpoint compatibility.")
            del model, ckpt, state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        model.eval()
        test_pred_fold = run_inference_loop(model, test_loader, device)
        fold_preds.append(test_pred_fold.astype(np.float32))

        del model, ckpt, state_dict, test_pred_fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if len(fold_preds) > 0:
    test_pred = np.mean(np.stack(fold_preds, axis=0), axis=0)
else:
    test_pred = np.full((len(test), N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)



## === cell 13
test_pred = np.nan_to_num(
    test_pred, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
)
row_sum = test_pred.sum(axis=1, keepdims=True)
row_sum = np.where(row_sum <= 0, 1.0, row_sum)
test_pred = test_pred / row_sum

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df.insert(0, "eeg_id", test["eeg_id"].values)



## === cell 14
pred_map = test_pred_df.drop_duplicates(subset=["eeg_id"], keep="first").set_index(
    "eeg_id"
)[CLASSES]
aligned = pred_map.reindex(sample_sub["eeg_id"].values)

sub = pd.concat([sample_sub[["eeg_id"]].copy(), aligned.reset_index(drop=True)], axis=1)

missing = sub[CLASSES].isna().any(axis=1)
if missing.any():
    sub.loc[missing, CLASSES] = 1.0 / N_CLASSES

probs = sub[CLASSES].to_numpy(dtype=np.float32)
probs = np.nan_to_num(
    probs, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
)
probs = probs / probs.sum(axis=1, keepdims=True)
sub[CLASSES] = probs

sub.to_csv("submission.csv", index=False)

print("submission.csv written. shape=", sub.shape)
print(sub.head())
print("Row-sum stats:", sub[CLASSES].sum(axis=1).min(), sub[CLASSES].sum(axis=1).max())
assert len(sub) == len(
    sample_sub
), "Submission and sample_submission must have the same length."
assert np.allclose(
    sub[CLASSES].sum(axis=1).values, 1.0, atol=1e-4
), "Each row must sum to 1."
