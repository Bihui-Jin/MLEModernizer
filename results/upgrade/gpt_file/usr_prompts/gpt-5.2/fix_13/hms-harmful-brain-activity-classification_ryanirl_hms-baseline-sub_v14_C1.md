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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
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

0.3662436226673654

# 6. Current score

0.82226

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I first make the script robust to the missing `/kaggle/input/hms-models/` dataset by adding a safe fallback that skips model loading when those files aren’t present, instead producing a valid probability submission from a sensible prior (the normalized mean vote distribution from train). Then I fix the Polars row indexing bug (`df_test[i]` returns a Series) by switching to `df_test.row(i, named=True)` and making `gen_ensemble_pred` accept either a dict-like row or a 1-row DataFrame. Finally, I guard against the “no models loaded” case so inference never tries to stack an empty list, and ensure the final submission CSV has correct columns, row order, and rows sum to 1 for the KL metric.'
- What this solution (achieved 1.40465) has done: 'Your current score is high (bad) because the code usually can’t load any trained models from `/kaggle/input/hms-models/`, so it falls back to a global mean prior that’s poorly calibrated per-recording for KL. The smallest legitimate improvement toward your target is to keep the same fallback approach but make it *per-eeg_id* by averaging the normalized vote distributions from all train rows sharing that `eeg_id` (and using a smoothed global prior only when an `eeg_id` is unseen). This preserves your overall pipeline and avoids changing any model/training logic, while typically yielding a much lower KL than a single global prior. I’m also keeping the row-sum-to-1 safety normalization exactly as required for the metric.'
- What this solution (achieved 1.06579) has done: 'Your current score is much worse than the target (lower is better), so we should make a small, low-risk improvement that preserves your fallback-based core logic. The biggest remaining issue is that the “per-eeg_id prior” can’t help much because almost all test `eeg_id`s are unseen in train, so the code falls back to a global prior; we can legitimately improve this by switching the fallback to a **per-patient_id prior** computed from train (still only using metadata/votes, no leakage). We keep your existing per-eeg_id prior as first choice, then back off to per-patient_id, then finally to the global prior; this typically reduces KL noticeably while keeping identical inference/model logic when models are available. Finally, we keep the strict row-normalization to ensure the submission is always valid for the KL metric.'
- What this solution (achieved 0.75113) has done: 'Your current score (1.06579, lower is better) is far from the target (0.36624), and since models usually aren’t loading, the only lever is improving the fallback probabilities without changing the model/inference core logic. The smallest, legitimate improvement is to compute a stronger hierarchical prior from train votes: normalize votes to probabilities per row, then average them per patient (and globally), and finally blend patient/global with a small weight to reduce overconfidence for KL. We keep the existing backoff order (eeg_id → patient_id → global), but make the patient/global priors **Dirichlet-smoothed in probability space** and ensure every prediction is safely clipped and renormalized. This should reduce KL meaningfully while preserving your pipeline behavior when models are present and still producing a valid submission.csv.'
- What this solution (achieved 0.81813) has done: 'Your current score (0.75113, lower is better) is still far above the target (0.36624), and since models often don’t load the only safe lever is improving the fallback probability estimates without changing your model/inference core. I keep the same hierarchical backoff (eeg_id → patient_id → global), but compute the priors from **raw vote counts with Dirichlet smoothing** (rather than averaging already-normalized row probabilities), which is a better empirical Bayes estimate and typically reduces KL. I also replace the fixed patient/global blend with an **evidence-weighted blend** based on the number of training rows and total votes for that patient, so patients with little data back off more to global (stabilizing KL). Finally, I keep the same strict clipping + renormalization to guarantee valid submissions.'
- What this solution (achieved 0.81813) has done: 'Your current score (0.81813, lower-is-better) is still far from the target (0.36624), and since models usually don’t load, the only safe lever is improving the fallback probabilities while keeping your inference/model core unchanged. The smallest high-impact fix is to make the patient/global fallback **eeg_id-aware via spectrogram_id**: many train/test share `spectrogram_id`, so we can back off using a `spectrogram_id` prior before patient/global, which is legitimate metadata-only information and typically much more specific than patient-only. I also compute the `spectrogram_id` prior from **raw vote counts with the same Dirichlet smoothing** you already use (consistent with your latest plan), and keep the same clipping+renormalization to guarantee valid KL submissions. No model architecture/training logic is changed; this only improves the fallback path.'
- What this solution (achieved 0.81813) has done: 'Your current score is still much worse than the target (lower-is-better), and since models usually aren’t loading the only safe lever is to improve the metadata-only fallback priors. I keep your existing hierarchical backoff (eeg_id → spectrogram_id → patient_id → global) and the same Dirichlet-smoothed vote-count logic, but I add a very small and legitimate enhancement: use an additional prior keyed by `(patient_id, spectrogram_id)` when available in train, which is often more specific than either alone. I also fix a submission-alignment risk by writing predictions in exactly the `sample_submission.csv` row order (no sorting), which prevents accidental row permutation from hurting KL. All changes preserve your model/inference core; they only affect the fallback path and the final CSV row alignment.'
- What this solution (achieved 0.81813) has done: 'Your current KL (0.81813, lower-is-better) is still far above the target (0.36624), so we should cautiously improve the metadata-only fallback path without touching the model/inference core. The biggest low-risk gain is to make the spectrogram-based prior stronger by using a hierarchical key `(patient_id, spectrogram_id)` **and** letting the spectrogram prior also blend with global based on its evidence (total votes), which reduces overconfident/overfit priors that can hurt KL. I’m also fixing a silent bug in your `compute_chain` spectrogram helper (`[:, 0]` drops 3 of 4 channels), keeping the logic the same but making it correct; this only affects inference if models ever use the spectrogram path later. Finally, I keep strict clipping + renormalization and ensure predictions are written exactly in `sample_submission.csv` row order.'
- What this solution (achieved 0.8202) has done: 'Your current score is still far above the target (lower-is-better), and since the external model directory often isn’t available the only safe lever is improving the metadata-only fallback probabilities. I keep your exact hierarchical backoff and Dirichlet-smoothed vote-count logic, but fix two small issues that can materially hurt KL: (1) make the blend weight for the `(patient_id, spectrogram_id)` prior use the **pat+spec evidence scale** (not the patient-scale function), and (2) correct `compute_chain()` to return all 4 chains (it currently drops 3/4 due to `[:, 0]`), which preserves the intended core feature extraction if/when spectrogram-based inference is used. I also apply a tiny, KL-friendly uniform mix-in (very small epsilon) only on the fallback path to reduce overconfident zeros while keeping row sums exactly 1. All changes are minimal, keep your architecture/training logic untouched, and still write a valid `submission.csv`.'
- What this solution (achieved 0.82226) has done: 'Your current KL (0.8202, lower is better) is still far above the target (0.3662), and because the external model directory often isn’t available the only safe lever is improving the metadata-only fallback probabilities without changing any model/inference architecture. The smallest high-impact improvement is to replace the fixed “uniform epsilon” mix with a KL-friendlier Dirichlet-style smoothing that adds a tiny pseudocount to every class (prevents near-zeros) while preserving relative structure of the prior; this typically reduces KL more consistently than blending toward uniform. I also tighten the final probability sanitation (float64, renormalize once at the very end) to avoid any accidental numeric drift and keep rows summing to exactly 1. No training loops, model definitions, or feature extraction logic are changed; only the fallback prior calibration and final output normalization are adjusted.'
- What this solution (achieved 0.82226) has done: 'Your current KL (0.82226, lower-is-better) is still far above the target (0.36624), and because models typically don’t load, the only safe lever is improving the metadata-only fallback probabilities without touching model/training/feature logic. The smallest high-impact change is to use the train labels’ **consensus pattern per `spectrogram_id`** (via `expert_consensus`) to build a class-conditional prior; many train/test share `spectrogram_id`, and this gives a sharper, more informative prior than vote-marginals alone while remaining leakage-free. Concretely, when `spectrogram_id` is known, we blend the current spectrogram vote-count prior with an `expert_consensus`-conditioned prior (weighted by evidence) and keep your existing hierarchical backoff unchanged otherwise. Finally, we keep the same strict clipping + renormalization so every row sums to 1 for a valid KL submission.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List, Dict, Any, Union



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state, strict=True)
    return model




## === cell 4
import sys
import glob
import importlib.util
from types import ModuleType

HMS_MODELS_DIR = "/kaggle/input/hms-models/"


def _load_module_from_path(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _find_first(patterns: List[str]) -> str:
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"Could not find any file matching: {patterns}")


EegModel0 = None
EegModel1 = None
eeg_cnn_rnn_w1_path = None
eeg_cnn_rnn_path = None

if os.path.exists(HMS_MODELS_DIR):
    try:
        eeg_cnn_rnn_w1_path = _find_first(
            [
                os.path.join(HMS_MODELS_DIR, "**", "eeg_cnn_rnn_w1.py"),
                os.path.join(HMS_MODELS_DIR, "eeg_cnn_rnn_w1.py"),
            ]
        )
        eeg_cnn_rnn_path = _find_first(
            [
                os.path.join(HMS_MODELS_DIR, "**", "eeg_cnn_rnn.py"),
                os.path.join(HMS_MODELS_DIR, "eeg_cnn_rnn.py"),
            ]
        )

        mod0 = _load_module_from_path("eeg_cnn_rnn_w1_mod", eeg_cnn_rnn_w1_path)
        mod1 = _load_module_from_path("eeg_cnn_rnn_mod", eeg_cnn_rnn_path)

        EegModel0 = getattr(mod0, "EegModel")
        EegModel1 = getattr(mod1, "EegModel")

        print("Loaded model code from:")
        print(" -", eeg_cnn_rnn_w1_path)
        print(" -", eeg_cnn_rnn_path)
    except Exception as e:
        print(
            "Warning: could not load external model code; will use fallback submission."
        )
        print("Reason:", repr(e))
else:
    print("Warning: /kaggle/input/hms-models not found; will use fallback submission.")



## === cell 5
import glob

models_0: List[nn.Module] = []
models_1: List[nn.Module] = []

if EegModel0 is not None and EegModel1 is not None and os.path.exists(HMS_MODELS_DIR):
    model_0_files = glob.glob(
        "/kaggle/input/hms-models/new_distil/send_kaggle/*"
    ) + glob.glob("/kaggle/input/hms-models/new_fold/send_kaggle/*")

    for fold_path in model_0_files:
        if not os.path.isfile(fold_path):
            continue
        try:
            model = EegModel0()
            model = load_model(fold_path, model)
            model = model.to(device)
            models_0.append(model)
        except Exception:
            continue

    for fold_path in glob.glob("/kaggle/input/hms-models/baseline_eeg_diff/*"):
        model_path = os.path.join(fold_path, "model_best_val_g10.pt")
        if not os.path.exists(model_path):
            alt = glob.glob(os.path.join(fold_path, "*.pt"))
            if not alt:
                continue
            model_path = sorted(alt)[0]
        try:
            model = EegModel1()
            model = load_model(model_path, model)
            model = model.to(device)
            models_1.append(model)
        except Exception:
            continue

print(
    "Loaded models:",
    len(models_0),
    "+",
    len(models_1),
    "=",
    len(models_0) + len(models_1),
)



## === cell 6
import polars as pl
import librosa
import numpy as np

from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    spectrogram = librosa.stft(
        eeg,
        n_fft=1024,
        hop_length=39,
        win_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
        out=None,
        dtype=None,
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    spec0 = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    spec1 = compute_spec(eeg2)
    return (spec0 + spec1) / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack([(spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1))])
    lp = np.stack([(spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1))])
    rp = np.stack([(spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2))])
    rl = np.stack([(spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2))])

    chain = np.stack([ll, lp, rp, rl])  # (4, 1, 4, F, T)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    return chain




## === cell 7
import numpy as np
import math

from typing import Union
from typing import Tuple
from typing import List


def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode=mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding

    return array




## === cell 8
from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)

    TARGET_LEN = 2500
    if eeg.shape[-1] < TARGET_LEN:
        eeg = np.pad(eeg, (0, TARGET_LEN - eeg.shape[-1]), mode="edge")
    elif eeg.shape[-1] > TARGET_LEN:
        eeg = eeg[:TARGET_LEN]
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack(
        [
            (
                compute_eeg(Fp1 - F7),
                compute_eeg(F7 - T3),
                compute_eeg(T3 - T5),
                compute_eeg(T5 - O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_eeg(Fp1 - F3),
                compute_eeg(F3 - C3),
                compute_eeg(C3 - P3),
                compute_eeg(P3 - O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_eeg(Fp2 - F4),
                compute_eeg(F4 - C4),
                compute_eeg(C4 - P4),
                compute_eeg(P4 - O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_eeg(Fp2 - F8),
                compute_eeg(F8 - T4),
                compute_eeg(T4 - T6),
                compute_eeg(T6 - O2),
            )
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 9
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 10
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, 2_500)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, 2_500)
    return x


def flip_h(eeg):
    temp0 = eeg[0].copy()
    temp1 = eeg[1].copy()

    eeg[0] = eeg[3]
    eeg[1] = eeg[2]
    eeg[3] = temp0
    eeg[2] = temp1
    return eeg


def flip_v(eeg):
    temp0 = eeg[:, 0].copy()
    temp1 = eeg[:, 1].copy()

    eeg[:, 0] = eeg[:, 3]
    eeg[:, 1] = eeg[:, 2]
    eeg[:, 3] = temp0
    eeg[:, 2] = temp1
    return eeg


def tta(x):
    x0 = x.copy().reshape(4, 4, -1)
    x1 = x.copy().reshape(4, 4, -1)
    x2 = x.copy().reshape(4, 4, -1)

    x0 = flip_v(x0).reshape(16, 1, -1)
    x1 = flip_h(x1).reshape(16, 1, -1)
    x2 = flip_v(flip_h(x2)).reshape(16, 1, -1)
    return x, x0, x1, x2


_train_meta_votes = df_train.select(
    ["eeg_id", "spectrogram_id", "patient_id", "expert_consensus"] + LABELS
)

_votes_np = _train_meta_votes.select(LABELS).to_numpy().astype(np.float64)
_votes_np = np.where(np.isfinite(_votes_np), _votes_np, 0.0)
_votes_np = np.clip(_votes_np, 0.0, None)

_eeg_ids = _train_meta_votes["eeg_id"].to_numpy()
_spec_ids = _train_meta_votes["spectrogram_id"].to_numpy()
_patient_ids = _train_meta_votes["patient_id"].to_numpy()
_cons = _train_meta_votes["expert_consensus"].to_list()

_global_counts = _votes_np.sum(axis=0)
_global_counts = np.clip(_global_counts, 0.0, None)

DIRICHLET_ALPHA = 0.5  # per-class pseudo-count

_global_prior = (_global_counts + DIRICHLET_ALPHA).astype(np.float64)
_global_prior = _global_prior / _global_prior.sum()
_global_prior = _global_prior.astype(np.float32)

df_counts = pl.DataFrame(
    {
        "eeg_id": _eeg_ids,
        "spectrogram_id": _spec_ids,
        "patient_id": _patient_ids,
        "expert_consensus": _cons,
        **{c: _votes_np[:, j].astype(np.float32) for j, c in enumerate(LABELS)},
        "row_weight": np.ones(len(_votes_np), dtype=np.float32),
    }
)

_df_eeg_counts = df_counts.group_by("eeg_id").sum()
_df_spec_counts = df_counts.group_by("spectrogram_id").sum()
_df_patient_counts = df_counts.group_by("patient_id").sum()
_df_pat_spec_counts = df_counts.group_by(["patient_id", "spectrogram_id"]).sum()

_df_cons_counts = df_counts.group_by("expert_consensus").sum()
_df_spec_cons_counts = df_counts.group_by(["spectrogram_id", "expert_consensus"]).sum()

_eeg_prior_map: Dict[int, np.ndarray] = {}
for r in _df_eeg_counts.iter_rows(named=True):
    cnt = np.array([r[c] for c in LABELS], dtype=np.float64)
    cnt = np.clip(cnt, 0.0, None)
    p = cnt + DIRICHLET_ALPHA
    p = p / p.sum()
    _eeg_prior_map[int(r["eeg_id"])] = p.astype(np.float32)

_spec_prior_map: Dict[int, np.ndarray] = {}
_spec_strength_map: Dict[int, float] = {}
for r in _df_spec_counts.iter_rows(named=True):
    cnt = np.array([r[c] for c in LABELS], dtype=np.float64)
    cnt = np.clip(cnt, 0.0, None)
    strength = float(cnt.sum())
    p = cnt + DIRICHLET_ALPHA
    p = p / p.sum()
    sid = int(r["spectrogram_id"])
    _spec_prior_map[sid] = p.astype(np.float32)
    _spec_strength_map[sid] = strength

_patient_prior_map: Dict[int, np.ndarray] = {}
_patient_strength_map: Dict[int, float] = {}
for r in _df_patient_counts.iter_rows(named=True):
    cnt = np.array([r[c] for c in LABELS], dtype=np.float64)
    cnt = np.clip(cnt, 0.0, None)
    strength = float(cnt.sum())
    p = cnt + DIRICHLET_ALPHA
    p = p / p.sum()
    pid = int(r["patient_id"])
    _patient_prior_map[pid] = p.astype(np.float32)
    _patient_strength_map[pid] = strength

_pat_spec_prior_map: Dict[Tuple[int, int], np.ndarray] = {}
_pat_spec_strength_map: Dict[Tuple[int, int], float] = {}
for r in _df_pat_spec_counts.iter_rows(named=True):
    cnt = np.array([r[c] for c in LABELS], dtype=np.float64)
    cnt = np.clip(cnt, 0.0, None)
    strength = float(cnt.sum())
    p = cnt + DIRICHLET_ALPHA
    p = p / p.sum()
    key = (int(r["patient_id"]), int(r["spectrogram_id"]))
    _pat_spec_prior_map[key] = p.astype(np.float32)
    _pat_spec_strength_map[key] = strength

_cons_prior_map: Dict[str, np.ndarray] = {}
_cons_strength_map: Dict[str, float] = {}
for r in _df_cons_counts.iter_rows(named=True):
    cons = str(r["expert_consensus"])
    cnt = np.array([r[c] for c in LABELS], dtype=np.float64)
    cnt = np.clip(cnt, 0.0, None)
    strength = float(cnt.sum())
    p = cnt + DIRICHLET_ALPHA
    p = p / p.sum()
    _cons_prior_map[cons] = p.astype(np.float32)
    _cons_strength_map[cons] = strength

_spec_cons_prior_map: Dict[Tuple[int, str], np.ndarray] = {}
_spec_cons_strength_map: Dict[Tuple[int, str], float] = {}
for r in _df_spec_cons_counts.iter_rows(named=True):
    sid = int(r["spectrogram_id"])
    cons = str(r["expert_consensus"])
    cnt = np.array([r[c] for c in LABELS], dtype=np.float64)
    cnt = np.clip(cnt, 0.0, None)
    strength = float(cnt.sum())
    p = cnt + DIRICHLET_ALPHA
    p = p / p.sum()
    key = (sid, cons)
    _spec_cons_prior_map[key] = p.astype(np.float32)
    _spec_cons_strength_map[key] = strength

PATIENT_BLEND_MAX = 0.90
PATIENT_BLEND_MIN = 0.10
PATIENT_BLEND_TAU = 60.0  # votes needed to reach ~63% of max-min range

SPEC_BLEND_MAX = 0.85
SPEC_BLEND_MIN = 0.15
SPEC_BLEND_TAU = 80.0

CONS_BLEND_MAX = 0.65
CONS_BLEND_MIN = 0.10
CONS_BLEND_TAU = 80.0


def _patient_blend_w(total_votes: float) -> float:
    w = PATIENT_BLEND_MIN + (PATIENT_BLEND_MAX - PATIENT_BLEND_MIN) * (
        total_votes / (total_votes + PATIENT_BLEND_TAU)
    )
    return float(np.clip(w, PATIENT_BLEND_MIN, PATIENT_BLEND_MAX))


def _spec_blend_w(total_votes: float) -> float:
    w = SPEC_BLEND_MIN + (SPEC_BLEND_MAX - SPEC_BLEND_MIN) * (
        total_votes / (total_votes + SPEC_BLEND_TAU)
    )
    return float(np.clip(w, SPEC_BLEND_MIN, SPEC_BLEND_MAX))


def _cons_blend_w(total_votes: float) -> float:
    w = CONS_BLEND_MIN + (CONS_BLEND_MAX - CONS_BLEND_MIN) * (
        total_votes / (total_votes + CONS_BLEND_TAU)
    )
    return float(np.clip(w, CONS_BLEND_MIN, CONS_BLEND_MAX))


def _get_ids_from_row(
    df_row: Union[pl.DataFrame, Dict[str, Any]]
) -> Tuple[int, Optional[int], Optional[int]]:
    if isinstance(df_row, dict):
        eeg_id = int(df_row["eeg_id"])
        patient_id = (
            int(df_row["patient_id"])
            if "patient_id" in df_row and df_row["patient_id"] is not None
            else None
        )
        spec_id = (
            int(df_row["spectrogram_id"])
            if "spectrogram_id" in df_row and df_row["spectrogram_id"] is not None
            else None
        )
        return eeg_id, patient_id, spec_id

    eeg_id = int(df_row["eeg_id"].item())
    patient_id = (
        int(df_row["patient_id"].item()) if "patient_id" in df_row.columns else None
    )
    spec_id = (
        int(df_row["spectrogram_id"].item())
        if "spectrogram_id" in df_row.columns
        else None
    )
    return eeg_id, patient_id, spec_id


FALLBACK_SMOOTH_ALPHA = 0.02  # tiny pseudocount to avoid near-zeros for KL stability


def _smooth_probs_dirichlet(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / (p.sum() + 1e-18)
    k = p.shape[-1]
    p2 = p + (FALLBACK_SMOOTH_ALPHA / k)
    p2 = np.clip(p2, 1e-12, None)
    p2 = p2 / p2.sum()
    return p2.astype(np.float32)


def _infer_spec_consensus_probs(spec_id: int) -> Optional[np.ndarray]:
    total = 0.0
    probs = np.zeros(len(LABELS), dtype=np.float64)

    for cons, p_cons in _cons_prior_map.items():
        key = (spec_id, cons)
        strength = _spec_cons_strength_map.get(key, 0.0)
        if strength <= 0:
            continue
        probs += float(strength) * np.asarray(p_cons, dtype=np.float64)
        total += float(strength)

    if total <= 0:
        return None

    probs = np.clip(probs, 0.0, None)
    probs = probs / (probs.sum() + 1e-18)
    return probs.astype(np.float32)


def _fallback_prior(
    eeg_id: int, patient_id: Optional[int], spec_id: Optional[int]
) -> np.ndarray:
    if eeg_id in _eeg_prior_map:
        return _smooth_probs_dirichlet(_eeg_prior_map[eeg_id].copy())

    if patient_id is not None and spec_id is not None:
        key = (patient_id, spec_id)
        if key in _pat_spec_prior_map:
            p_ps = _pat_spec_prior_map[key]
            strength = _pat_spec_strength_map.get(key, 0.0)
            w = _spec_blend_w(strength)
            p = w * p_ps + (1.0 - w) * _global_prior
            p = np.clip(p, 1e-12, None)
            p = p / p.sum()
            return _smooth_probs_dirichlet(p.astype(np.float32))

    if spec_id is not None and spec_id in _spec_prior_map:
        p_spec = _spec_prior_map[spec_id]
        strength = _spec_strength_map.get(spec_id, 0.0)
        w = _spec_blend_w(strength)
        p = w * p_spec + (1.0 - w) * _global_prior

        p_cons = _infer_spec_consensus_probs(spec_id)
        if p_cons is not None:
            w2 = _cons_blend_w(strength)
            p = (1.0 - w2) * p + w2 * p_cons

        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        return _smooth_probs_dirichlet(p.astype(np.float32))

    if patient_id is not None and patient_id in _patient_prior_map:
        p_pat = _patient_prior_map[patient_id]
        strength = _patient_strength_map.get(patient_id, 0.0)
        w = _patient_blend_w(strength)
        p = w * p_pat + (1.0 - w) * _global_prior
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        return _smooth_probs_dirichlet(p.astype(np.float32))

    return _smooth_probs_dirichlet(_global_prior.copy())


@torch.no_grad()
def gen_ensemble_pred(
    models_0, models_1, df_row: Union[pl.DataFrame, Dict[str, Any]]
) -> np.ndarray:
    eeg_id, patient_id, spec_id = _get_ids_from_row(df_row)

    if (models_0 is None or len(models_0) == 0) and (
        models_1 is None or len(models_1) == 0
    ):
        return _fallback_prior(eeg_id, patient_id, spec_id)

    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    x = compute_eeg_from_file(filepath)
    x[np.isnan(x) | np.isinf(x)] = 0

    x0 = proc_0(x)
    x1 = proc_1(x)

    x11, x12, x13, x14 = tta(x1)
    x01, x02, x03, x04 = tta(x0)

    x11 = torch.tensor(x11, device=device).unsqueeze(0)
    x12 = torch.tensor(x12, device=device).unsqueeze(0)
    x13 = torch.tensor(x13, device=device).unsqueeze(0)
    x14 = torch.tensor(x14, device=device).unsqueeze(0)

    x01 = torch.tensor(x01, device=device).unsqueeze(0)
    x02 = torch.tensor(x02, device=device).unsqueeze(0)
    x03 = torch.tensor(x03, device=device).unsqueeze(0)
    x04 = torch.tensor(x04, device=device).unsqueeze(0)

    preds = []

    for model in models_0:
        model.eval()
        preds.append(model(x01).exp().detach().cpu().numpy().reshape(-1))
        preds.append(model(x02).exp().detach().cpu().numpy().reshape(-1))
        preds.append(model(x03).exp().detach().cpu().numpy().reshape(-1))
        preds.append(model(x04).exp().detach().cpu().numpy().reshape(-1))

    for model in models_1:
        model.eval()
        preds.append(model(x11).exp().detach().cpu().numpy().reshape(-1))
        preds.append(model(x12).exp().detach().cpu().numpy().reshape(-1))
        preds.append(model(x13).exp().detach().cpu().numpy().reshape(-1))
        preds.append(model(x14).exp().detach().cpu().numpy().reshape(-1))

    if len(preds) == 0:
        return _fallback_prior(eeg_id, patient_id, spec_id)

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum()
    return preds.astype(np.float32)




## === cell 11
from tqdm.auto import tqdm

preds_final = []
for i in tqdm(range(len(df_test))):
    row = df_test.row(i, named=True)
    pred = gen_ensemble_pred(models_0, models_1, row)
    preds_final.append(pred)

preds_final = np.stack(preds_final, axis=0)
print(
    "preds_final shape:",
    preds_final.shape,
    "sum range:",
    preds_final.sum(axis=1).min(),
    preds_final.sum(axis=1).max(),
)



## === cell 12
df_sub = sample_submission.to_pandas()[["eeg_id"]].copy()

pred_map = {
    int(eid): preds_final[i] for i, eid in enumerate(df_test["eeg_id"].to_list())
}

preds_aligned = np.zeros((len(df_sub), len(LABELS)), dtype=np.float64)
missing = 0
for i, eid in enumerate(df_sub["eeg_id"].tolist()):
    p = pred_map.get(int(eid))
    if p is None:
        missing += 1
        p = _fallback_prior(int(eid), None, None)
    preds_aligned[i] = np.asarray(p, dtype=np.float64)

for j, c in enumerate(LABELS):
    df_sub[c] = preds_aligned[:, j]

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Missing eeg_id filled with fallback prior:", missing)
print("Wrote submission.csv with shape:", df_sub.shape)
