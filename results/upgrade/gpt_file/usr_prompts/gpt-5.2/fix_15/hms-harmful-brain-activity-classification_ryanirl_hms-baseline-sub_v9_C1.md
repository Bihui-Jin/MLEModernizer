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

0.4540712510783468

# 6. Current score

1.4199

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.4209) has done: 'I fix the shape mismatch causing the runtime error by reshaping the EEG tensor dynamically based on the actual preprocessed length (instead of hardcoding 2500), while preserving the same preprocessing and model forward semantics. I also make the pipeline robust to missing/failed reads by falling back to uniform probabilities for that sample, so the loop completes and a submission is always produced. Finally, I harden the final normalization so every row sums to exactly one (within floating tolerance), preventing the submission-format assertion failure. These changes are score-neutral relative to the intended model (they just make it run correctly end-to-end).'
- What this solution (achieved 1.42048) has done: 'Your current score (1.4209, lower-is-better) is far worse than the target (0.4541), so we need a small but meaningful improvement without changing the model or preprocessing logic. The biggest likely issue is a label-order mismatch: many HMS checkpoints are trained with a different class order (often GPD/GRDA swapped relative to your `LABELS`), which can destroy KL score even if the model is good. I add an order-alignment step that detects a plausible mismatch using the known train label priors and permutes predicted columns accordingly, then keep your existing normalization so the submission remains valid. This is minimal, preserves your architecture/inference, and is very likely to move the score substantially toward the target if the checkpoint uses a different order.'
- What this solution (achieved 1.4209) has done: 'Your current KL (1.42048, lower-is-better) is far from the target (0.45407), so we need a small but high-impact fix that doesn’t change your model/inference core. The most likely remaining issue is that you permute `preds_final` but you keep the original `LABELS` header, which silently writes probabilities under the wrong class names and can severely hurt KL. I minimally fix this by permuting the column names alongside the prediction columns so the submission header matches the permuted probabilities, then keep your existing row-wise normalization/validity checks intact. This change is directly tied to score correctness (label alignment) and should substantially move the score toward the target without altering the model.'
- What this solution (achieved 1.4209) has done: 'Your KL (lower-is-better) is far above target, so we need a correctness fix that can materially improve score without changing your model or preprocessing. The biggest issue is in the submission assembly: you permute the prediction columns (LABELS_SUB) but then you reorder the DataFrame using the original LABELS list, which can swap probabilities under the wrong class names and severely hurt KL. I minimally fix this by writing predictions into a fixed 6-column array keyed by the original LABELS names (so permutation is applied to values, not headers), then normalize once at the end to guarantee valid probabilities. This keeps your inference/model intact and directly addresses evaluation semantics.'
- What this solution (achieved 1.42071) has done: 'Your KL score is much worse than the target, so the most likely “minimal-but-high-impact” fix is correcting evaluation semantics rather than changing the model. In your pipeline you are reading EEG parquet files but never selecting the central 10 seconds that the labels refer to; using the whole 50 seconds can easily degrade KL. I minimally crop the EEG to the central 10 seconds (same preprocessing/model, just correct windowing), keep your existing robust normalization and fallback behavior, and keep the submission schema identical. This change is directly aligned with the competition’s label definition and should move the score materially toward the target without altering architecture or training logic.'
- What this solution (achieved 1.42056) has done: 'Your score is far worse than the target (KL 1.4207 vs 0.4541; lower is better), so the most likely remaining issue is evaluation-semantics correctness rather than model capacity. I keep your model and preprocessing core intact, but fix the EEG windowing: your current code crops to 2000 raw samples (10s) *before* filtering/downsamping, which effectively becomes ~2.5s after binning, misaligning with the labeled 10s center segment. I instead crop the central 10 seconds *after* the bandpass filter, ensuring the model sees the intended 10s segment at the correct sampling rate, then downsample as before. This is a minimal change that should materially reduce KL toward the target without altering architecture, loss, or inference semantics beyond correcting the time window.'
- What this solution (achieved 1.4199) has done: 'We fix the Polars aggregation bug that stops the validation-permutation step by replacing the invalid `pl.first(pl.col(c))` usage with the correct `pl.col(c).first()`. Because cell 10 currently crashes, the test predictions are never created, which then causes the `NameError` in cell 11; once cell 10 runs, cell 11 run unchanged and write `submission.csv`. These changes are execution/correctness fixes and keep the model/inference logic and metric semantics intact. The resulting script run end-to-end and always produce a valid, row-normalized submission CSV.'
- What this solution (achieved 1.4199) has done: 'Your current KL (1.4199, lower-is-better) is far worse than the target (0.4541), so we focus on a minimal but high-impact correctness fix rather than changing the model. The main remaining issue is that you permute the prediction *values* in cell 10 but still write them under the original `LABELS` column names in cell 11, which can silently swap class probabilities and severely hurt KL. I keep your model/inference exactly the same, but apply the permutation consistently: either permute both values and headers, or (safer) un-permute back into the canonical `LABELS` order before writing the submission. I also keep the final row-wise normalization/NaN handling exactly as you have to ensure a valid submission.'
- What this solution (achieved 1.4199) has done: 'Your current KL (1.4199, lower-is-better) is far from the target (0.4541), so the most likely remaining minimal fix is correctness of the EEG windowing relative to what the model was trained on. Right now `compute_eeg()` bandpasses then center-crops to 10s and *then* downsamples, which changes the effective receptive time context vs pipelines that downsample first and then crop the resulting 10s-equivalent window (common in HMS baselines). I make a minimal, localized change to crop **after** downsampling to preserve the intended 10-second window at the model’s working sampling rate, while keeping the same model, same preprocessing ops, and the same submission/normalization logic. Everything else (architecture, inference, permutation check, output columns) stays intact and it still write a valid `submission.csv`.'
- What this solution (achieved 1.4199) has done: 'I make two minimal, score-relevant corrections without changing your model architecture or inference loop: (1) fix an obvious channel-construction bug in `compute_eeg_chain` where the RP chain mistakenly duplicates `C4-P4` instead of using `P4-O2`, which corrupts a quarter of the input features and can heavily worsen KL; and (2) make the permutation-selection step more reliable by evaluating it on a slightly larger (still fast) validation subset so it’s less likely to choose the wrong mapping. These are localized, semantics-preserving fixes (same preprocessing ops, same model forward, same normalization) and should move KL down toward your target. The script still run end-to-end and write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.4199) has done: 'Your KL (1.4199, lower-is-better) is far from the target (0.4541), so we need a small but meaningful improvement without changing the model or overall pipeline. The biggest likely remaining mismatch is the test EEG windowing: the model was trained on the central 10 seconds of each 50s EEG, but your current `compute_eeg()` always center-crops; for test (which is exactly 50s) that’s fine, but for train validation (used to choose the permutation) many rows are not centered on the labeled offset, so the permutation choice can be wrong and then applied to test, hurting KL. I minimally fix the permutation-selection validation to use the correct labeled 10s segment from train by using `eeg_label_offset_seconds` to crop the proper 10s window before preprocessing (same filter/bin/standardize/model forward). This keeps architecture and inference intact, only corrects the validation semantics so the selected permutation is more likely correct, moving KL down toward your target.'

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

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




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
import sys, glob, importlib.util, inspect

MODELS_DIR = "/kaggle/input/hms-models"

LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _import_eeg_model(models_dir: str):
    if not os.path.isdir(models_dir):
        return None, None

    py_files = sorted(glob.glob(os.path.join(models_dir, "**", "*.py"), recursive=True))
    if not py_files:
        return None, None

    last_err = None
    for py_path in py_files:
        try:
            with open(py_path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if "class EegModel" not in txt and "EegModel" not in txt:
                continue

            mod_name = f"imported_eeg_model_{abs(hash(py_path))}"
            spec = importlib.util.spec_from_file_location(mod_name, py_path)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)

            if hasattr(module, "EegModel") and inspect.isclass(module.EegModel):
                return module.EegModel, py_path
        except Exception as e:
            last_err = e
            continue

    print(
        f"Warning: Could not import EegModel from {models_dir}. Last error: {last_err}"
    )
    return None, None


class FallbackEegModel(nn.Module):
    """
    Minimal log-probability model:
    - input x shape: (B, 16, 1, T) where T is time steps after preprocessing
    - output: (B, 6) log-probabilities (so downstream .exp() yields probabilities)
    """

    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(2)
        x = self.net(x).squeeze(-1)  # (B, 64)
        logits = self.fc(x)  # (B, 6)
        return torch.log_softmax(logits, dim=-1)


EegModel, EEGMODEL_SRC = _import_eeg_model(MODELS_DIR)
if EegModel is None:
    print("Using fallback model (no external model class found).")
    CurrModel = FallbackEegModel
else:
    print("Imported EegModel from:", EEGMODEL_SRC)
    CurrModel = EegModel



## === cell 5
torch.manual_seed(0)
np.random.seed(0)

model = CurrModel()
weights_path = "/kaggle/input/hms-models/baseline_eeg_diff/baseline_eeg_diff_cnn_rnn_stage_5_fold_0/model_best_val_g10.pt"
if os.path.isfile(weights_path):
    model = load_model(weights_path, model)
else:
    print(
        f"Warning: weights not found at {weights_path}. Proceeding without pretrained weights."
    )

model = model.to(device)
model.eval()
models = [model]
len(models)



## === cell 6
import librosa
from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1], dtype=np.float32)


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    chain = np.asarray(chain, dtype=np.float32)
    chain[~np.isfinite(chain)] = 0.0
    return chain


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
    s1 = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    s2 = compute_spec(eeg2)
    return (s1 + s2) / 2


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

    ll = np.stack([spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1)])
    lp = np.stack([spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1)])
    rp = np.stack([spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2)])
    rl = np.stack([spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)
    return chain




## === cell 7
import math
from typing import Union


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
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(
        N=order, Wn=np.asarray(cutoff_freq) / (0.5 * fs), btype=btype, analog=False
    )
    return filtfilt(b, a, eeg_data)


def _center_crop_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x)
    n = x.shape[-1]
    if n <= out_len:
        return x
    start = (n - out_len) // 2
    end = start + out_len
    return x[..., start:end]


def _crop_by_offset_seconds_200hz(
    x: np.ndarray, offset_seconds: float, win_seconds: float = 10.0, fs: int = 200
) -> np.ndarray:
    """
    Score-relevant fix: for TRAIN validation used to select output permutation,
    crop the exact labeled 10s window using eeg_label_offset_seconds, instead of center-crop.
    This does NOT change the model or preprocessing ops; it only selects the correct time window.
    """
    x = np.asarray(x)
    if not np.isfinite(offset_seconds):
        return _center_crop_1d(x, int(win_seconds * fs))
    start = int(round(float(offset_seconds) * fs))
    out_len = int(round(float(win_seconds) * fs))
    end = start + out_len
    if start < 0:
        start = 0
        end = out_len
    if end > x.shape[-1]:
        end = x.shape[-1]
        start = max(0, end - out_len)
    return x[..., start:end]


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)  # 200Hz -> 50Hz
    eeg = _center_crop_1d(eeg, out_len=500)  # 10 seconds at 50Hz
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
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_chain_with_offset(
    df_eeg: pl.DataFrame, offset_seconds: float
) -> np.ndarray:
    """
    Score-relevant fix for validation: apply offset-based crop at 200Hz first,
    then run the exact same compute_eeg() preprocessing pipeline.
    """
    Fp1 = _crop_by_offset_seconds_200hz(df_eeg["Fp1"].to_numpy(), offset_seconds)
    Fp2 = _crop_by_offset_seconds_200hz(df_eeg["Fp2"].to_numpy(), offset_seconds)
    F3 = _crop_by_offset_seconds_200hz(df_eeg["F3"].to_numpy(), offset_seconds)
    F4 = _crop_by_offset_seconds_200hz(df_eeg["F4"].to_numpy(), offset_seconds)
    F7 = _crop_by_offset_seconds_200hz(df_eeg["F7"].to_numpy(), offset_seconds)
    F8 = _crop_by_offset_seconds_200hz(df_eeg["F8"].to_numpy(), offset_seconds)
    C3 = _crop_by_offset_seconds_200hz(df_eeg["C3"].to_numpy(), offset_seconds)
    C4 = _crop_by_offset_seconds_200hz(df_eeg["C4"].to_numpy(), offset_seconds)
    P3 = _crop_by_offset_seconds_200hz(df_eeg["P3"].to_numpy(), offset_seconds)
    P4 = _crop_by_offset_seconds_200hz(df_eeg["P4"].to_numpy(), offset_seconds)
    T3 = _crop_by_offset_seconds_200hz(df_eeg["T3"].to_numpy(), offset_seconds)
    T4 = _crop_by_offset_seconds_200hz(df_eeg["T4"].to_numpy(), offset_seconds)
    T5 = _crop_by_offset_seconds_200hz(df_eeg["T5"].to_numpy(), offset_seconds)
    T6 = _crop_by_offset_seconds_200hz(df_eeg["T6"].to_numpy(), offset_seconds)
    O1 = _crop_by_offset_seconds_200hz(df_eeg["O1"].to_numpy(), offset_seconds)
    O2 = _crop_by_offset_seconds_200hz(df_eeg["O2"].to_numpy(), offset_seconds)

    ll = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain


def compute_eeg_from_file_with_offset(
    filepath: str, offset_seconds: float
) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain_with_offset(df_eeg, offset_seconds)
    return chain




## === cell 9
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def _uniform_pred() -> np.ndarray:
    return (np.ones((len(LABELS),), dtype=np.float32) / len(LABELS)).astype(np.float32)


@torch.no_grad()
def gen_ensemble_pred(models, eeg_id: int) -> np.ndarray:
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    try:
        x = compute_eeg_from_file(filepath)
        x = np.asarray(x, dtype=np.float32)
        x[~np.isfinite(x)] = 0.0
    except Exception:
        return _uniform_pred()

    if x.ndim != 2 or x.shape[0] != 4:
        return _uniform_pred()

    T = int(x.shape[1])
    if T <= 0:
        return _uniform_pred()

    x = np.repeat(x, repeats=4, axis=0)  # (16, T)

    xt = torch.tensor(x, device=device)
    xt = xt - xt.mean(dim=-1, keepdim=True)
    xt = xt / (xt.std(dim=-1, keepdim=True) + 1e-5)
    xt = xt.reshape(16, 1, T).unsqueeze(0)  # (B=1,16,1,T)

    preds = []
    for m in models:
        m.eval()
        p = m(xt).exp()
        p = p.detach().float().cpu().numpy().reshape(-1)
        if p.shape[0] != len(LABELS) or not np.all(np.isfinite(p)):
            p = _uniform_pred()
        preds.append(p)

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-12, None)
    s = float(preds.sum())
    if not np.isfinite(s) or s <= 0:
        preds = _uniform_pred().astype(np.float64)
    else:
        preds = (preds / s).astype(np.float64)

    return preds.astype(np.float32)




## === cell 10
def _row_votes_to_prob(df_pl: pl.DataFrame, labels: List[str]) -> np.ndarray:
    v = df_pl.select(labels).to_numpy()
    v = np.asarray(v, dtype=np.float64)
    v[~np.isfinite(v)] = 0.0
    s = v.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    p = v / s
    p = np.clip(p, 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _kl_divergence_rowwise(true_p: np.ndarray, pred_p: np.ndarray) -> float:
    true_p = np.asarray(true_p, dtype=np.float64)
    pred_p = np.asarray(pred_p, dtype=np.float64)
    true_p = np.clip(true_p, 1e-12, None)
    pred_p = np.clip(pred_p, 1e-12, None)
    true_p = true_p / true_p.sum(axis=1, keepdims=True)
    pred_p = pred_p / pred_p.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(true_p * np.log(true_p / pred_p), axis=1)))


val_eeg_ids = (
    df_train.select("eeg_id")
    .unique()
    .sort("eeg_id")
    .head(512)  # keep under 600s, but more robust than 128
    .to_series()
    .to_list()
)

df_train_first = (
    df_train.sort(["eeg_id", "label_id"])
    .group_by("eeg_id")
    .agg(
        [pl.col("eeg_label_offset_seconds").first().alias("eeg_label_offset_seconds")]
        + [pl.col(c).first().alias(c) for c in LABELS]
    )
)

df_val = df_train_first.filter(pl.col("eeg_id").is_in(val_eeg_ids)).sort("eeg_id")
y_true = _row_votes_to_prob(df_val, LABELS)

TRAIN_EEG_DIR = os.path.join(DATA_DIR, "train_eegs/")


@torch.no_grad()
def _predict_for_ids_with_offsets(
    eeg_ids: List[int], offsets: List[float], eeg_dir: str
) -> np.ndarray:
    """
    Score-relevant fix: predict on the correct labeled window for TRAIN validation only.
    This affects only the permutation-selection step and keeps test inference unchanged.
    """
    out = np.zeros((len(eeg_ids), len(LABELS)), dtype=np.float64)
    for i, (eid, off) in enumerate(zip(eeg_ids, offsets)):
        filepath = os.path.join(eeg_dir, f"{int(eid)}.parquet")
        try:
            x = compute_eeg_from_file_with_offset(filepath, float(off))
            x = np.asarray(x, dtype=np.float32)
            x[~np.isfinite(x)] = 0.0
            if x.ndim != 2 or x.shape[0] != 4 or x.shape[1] <= 0:
                out[i] = _uniform_pred().astype(np.float64)
                continue

            T = int(x.shape[1])
            x = np.repeat(x, repeats=4, axis=0)  # (16, T)

            xt = torch.tensor(x, device=device)
            xt = xt - xt.mean(dim=-1, keepdim=True)
            xt = xt / (xt.std(dim=-1, keepdim=True) + 1e-5)
            xt = xt.reshape(16, 1, T).unsqueeze(0)

            preds = []
            for m in models:
                p = m(xt).exp().detach().float().cpu().numpy().reshape(-1)
                if p.shape[0] != len(LABELS) or not np.all(np.isfinite(p)):
                    p = _uniform_pred()
                preds.append(p.astype(np.float64))
            p_mean = np.mean(np.stack(preds, axis=0), axis=0)
            p_mean = np.clip(p_mean, 1e-12, None)
            p_mean = p_mean / p_mean.sum()
            out[i] = p_mean
        except Exception:
            out[i] = _uniform_pred().astype(np.float64)

    out = np.clip(out, 1e-12, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


y_pred = _predict_for_ids_with_offsets(
    df_val["eeg_id"].to_list(),
    df_val["eeg_label_offset_seconds"].to_list(),
    TRAIN_EEG_DIR,
)

idx = {lab: i for i, lab in enumerate(LABELS)}
perm_identity = list(range(6))
perm_swap_gpd_grda = [
    idx["seizure_vote"],
    idx["lpd_vote"],
    idx["grda_vote"],
    idx["lrda_vote"],
    idx["gpd_vote"],
    idx["other_vote"],
]

kl_id = _kl_divergence_rowwise(y_true, y_pred[:, perm_identity])
kl_swap = _kl_divergence_rowwise(y_true, y_pred[:, perm_swap_gpd_grda])

best_perm = perm_identity if kl_id <= kl_swap else perm_swap_gpd_grda
print("Val KL identity:", kl_id, "Val KL swap_gpd_grda:", kl_swap)
print("Selected permutation:", best_perm)

test_eeg_ids = df_test["eeg_id"].to_list()
preds_test = np.zeros((len(test_eeg_ids), len(LABELS)), dtype=np.float64)
for i, eid in enumerate(tqdm(test_eeg_ids)):
    preds_test[i] = gen_ensemble_pred(models, int(eid)).astype(np.float64)

preds_test = np.clip(preds_test, 1e-12, None)
preds_test = preds_test / preds_test.sum(axis=1, keepdims=True)

inv_best_perm = np.empty(len(best_perm), dtype=int)
for label_pos, model_out_pos in enumerate(best_perm):
    inv_best_perm[model_out_pos] = label_pos
preds_test_canonical = preds_test[:, inv_best_perm]

preds_test_canonical = np.clip(preds_test_canonical, 1e-12, None)
preds_test_canonical = preds_test_canonical / preds_test_canonical.sum(
    axis=1, keepdims=True
)

preds_test_canonical.shape, preds_test_canonical[:1], preds_test_canonical.sum(
    axis=1
).min(), preds_test_canonical.sum(axis=1).max()



## === cell 11
df_pred = pd.DataFrame({"eeg_id": test_eeg_ids})

for j, col in enumerate(LABELS):
    df_pred[col] = preds_test_canonical[:, j]

df_pred = df_pred.groupby("eeg_id", as_index=False)[LABELS].mean()

sub_order = sample_submission["eeg_id"].to_list()
df_sub = pd.DataFrame({"eeg_id": sub_order}).merge(df_pred, on="eeg_id", how="left")

missing = df_sub[LABELS].isna().any(axis=1).values
if np.any(missing):
    df_sub.loc[missing, LABELS] = 1.0 / len(LABELS)

pred_mat = df_sub[LABELS].to_numpy(dtype=np.float64)
pred_mat[~np.isfinite(pred_mat)] = 0.0
pred_mat = np.clip(pred_mat, 1e-12, None)
pred_mat = pred_mat / pred_mat.sum(axis=1, keepdims=True)
df_sub[LABELS] = pred_mat

row_sums = df_sub[LABELS].sum(axis=1).values
assert np.all(np.isfinite(row_sums))
assert (
    np.max(np.abs(row_sums - 1.0)) < 1e-6
), f"Row sums not 1. Max dev: {np.max(np.abs(row_sums - 1.0))}"

df_sub.to_csv("submission.csv", index=False)
df_sub.head()
