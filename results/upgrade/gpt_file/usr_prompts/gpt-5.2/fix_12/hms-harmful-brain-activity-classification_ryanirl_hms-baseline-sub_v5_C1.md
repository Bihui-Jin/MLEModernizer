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

0.4733206221861924

# 6. Current score

1.16511

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the dependency on the missing external dataset/module (`/kaggle/input/hms-models/` and `eeg_cnn_rnn`) by replacing it with a small, local PyTorch model that preserves the same inference semantics (input → 6-class probabilities that sum to 1). I also fix the dataframe row-iteration bug (Polars row selection) and the submission construction bug by ensuring `preds_final` is a proper `(n_test, 6)` numpy array before assigning to the 6 label columns. Finally, I add a safe fallback so that if an EEG parquet is missing or malformed, the code outputs a valid uniform probability row (still sums to 1), ensuring a valid `submission.csv` is always written end-to-end.'
- What this solution (achieved 1.40724) has done: 'The runtime error comes from `compute_eeg_chain()` returning a 3D array `[4, 4, T]` while the model expects a 2D signal `[4, T]`; I fix the stacking so each of the 4 chains is flattened/averaged into a single channel, producing `[4, T]`. I also harden the preprocessing so it always returns a consistent time length (pad/trim) and avoids NaNs/Infs after normalization, preventing intermittent failures. Finally, I keep the model/inference semantics identical (still outputs 6-class probabilities summing to 1) and ensure `submission.csv` is written with the required columns.'
- What this solution (achieved 1.40724) has done: 'Your current score is far worse than the target (lower is better), and the biggest issue is that the model is completely untrained, so its outputs are essentially random and score poorly on KL-divergence. I keep your exact preprocessing and model architecture, but add a minimal training step on `train.csv` + `train_eegs/` using the same forward pass (probabilities) and optimize KL-divergence against the normalized vote distribution. To keep changes minimal and runtime <600s, I train on a small, deterministic subset of training rows and then run the same test-time inference code to produce `submission.csv`. I also add a small probability floor in the model output to avoid `log(0)` instability in KL.'
- What this solution (achieved 1.32354) has done: 'The timeout is dominated by per-file parquet reads + CPU `filtfilt` during inference for 9,850 EEGs, and by slow batch assembly during training due to repeated CPU tensor allocations/copies. I keep the exact same preprocessing, model, and training semantics, but (1) batch inference to reduce Python overhead and GPU kernel launch overhead, (2) switch parquet reads to a faster path (PyArrow direct-to-NumPy, multithreaded), (3) avoid repeated per-step tensor allocations in training by preallocating and slicing, and (4) use pinned memory + non-blocking transfers for faster CPU→GPU copies. These are provably equivalent transformations (same computations, just reorganized and cached) and should bring runtime under 600s.'
- What this solution (achieved 1.28656) has done: 'The timeout is dominated by expensive per-file parquet reads plus SciPy `filtfilt` on every EEG (both while preloading training cache and while running inference over 9,850 test files). To keep identical model/training logic and outputs, the main speedups are: (1) eliminate the slow `pyarrow -> pandas -> numpy` conversion by converting Arrow columns directly to NumPy, (2) reuse preallocated NumPy/Torch buffers to avoid repeated allocations/copies, and (3) avoid per-sample Python overhead in preprocessing by doing the same operations in-place on preallocated arrays. These changes are provably equivalent (same filter, same math, same model) and only reduce constant factors and memory churn. Training steps and inference batch sizes remain unchanged; no approximations, early stopping, or reduced precision are introduced.'
- What this solution (achieved 1.28656) has done: 'The timeout is dominated by repeated Parquet reads and `filtfilt` calls for thousands of EEGs during both training cache preload and test inference, plus unnecessary per-sample Python-loop overhead. I keep the exact same feature pipeline (diff pairs → lowpass `filtfilt` → downsample → region means → `diff` → per-channel standardization) and the same model/training loop semantics, but make I/O and preprocessing faster by (1) using Polars’ PyArrow-backed parquet reads with thread support instead of manual per-column Arrow conversions, (2) caching computed EEG chains for both train and test within the run so each file is filtered at most once, and (3) vectorizing/collapsing small Python loops in batch assembly while preserving determinism. These changes are equivalent in outputs aside from negligible floating-point ordering differences, and they reduce constant factors enough to fit the 600s budget.'
- What this solution (achieved 1.16511) has done: 'I keep your exact feature pipeline (diff pairs → lowpass `filtfilt` → downsample → region means → `diff` → per-channel standardization) and the same model/training loop, but make two small changes aimed at reducing KL: (1) train on a larger, more representative cached set by increasing `max_steps` moderately and slightly increasing batch size (no early stopping or approximations), and (2) blend your model probabilities with the global class prior from `train.csv` (a standard KL-stabilizing calibration that preserves valid probabilities and avoids overconfident mistakes). I also ensure the prior is computed at the same granularity as your training targets (per `eeg_id` aggregated votes) and that blending is applied identically for every test row with a fixed weight. These are minimal, low-risk changes that typically move scores down from ~1.2 toward the 0.47 target without changing the core architecture or preprocessing. The script still runs end-to-end and writes a valid `submission.csv` with rows summing to 1.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os
import math
import gc

import torch
import torch.nn as nn

from typing import List, Tuple



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")
TRAIN_EEG_DIR = os.path.join(DATA_DIR, "train_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(df_test.shape)
df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)



## === cell 3
from scipy.signal import butter, filtfilt
from functools import lru_cache


@lru_cache(maxsize=None)
def _butter_coeffs(fs=200, cutoff_freq=22, order=4):
    b, a = butter(
        N=order,
        Wn=cutoff_freq / (0.5 * fs),
        btype="low",
        analog=False,
    )
    return b, a


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4):
    b, a = _butter_coeffs(fs=fs, cutoff_freq=cutoff_freq, order=order)
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg)
    eeg = eeg[::2]  # 200Hz -> 100Hz
    return eeg


_EEG_COLS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "F7",
    "F8",
    "C3",
    "C4",
    "P3",
    "P4",
    "T3",
    "T4",
    "T5",
    "T6",
    "O1",
    "O2",
]
_EEG_COL_IDX = {c: i for i, c in enumerate(_EEG_COLS)}

_DIFF_PAIRS = [
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),  # LL (4)
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),  # LP (4)
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),  # RP (4)
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),  # RL (4)
]
_DIFF_A = np.fromiter(
    (_EEG_COL_IDX[a] for a, b in _DIFF_PAIRS), count=len(_DIFF_PAIRS), dtype=np.int64
)
_DIFF_B = np.fromiter(
    (_EEG_COL_IDX[b] for a, b in _DIFF_PAIRS), count=len(_DIFF_PAIRS), dtype=np.int64
)


def compute_eeg_chain_from_array(x: np.ndarray) -> np.ndarray:
    diffs = (x[:, _DIFF_A] - x[:, _DIFF_B]).T  # [16, T]
    b, a = _butter_coeffs(fs=200, cutoff_freq=22, order=4)
    diffs_f = filtfilt(b, a, diffs, axis=-1)
    diffs_f = diffs_f[:, ::2]  # [16, T100]
    ll = diffs_f[0:4].mean(axis=0)
    lp = diffs_f[4:8].mean(axis=0)
    rp = diffs_f[8:12].mean(axis=0)
    rl = diffs_f[12:16].mean(axis=0)
    return np.stack([ll, lp, rp, rl], axis=0)  # [4, T100]


def _read_eeg_cols_polars(filepath: str) -> np.ndarray:
    df = pl.read_parquet(
        filepath,
        columns=_EEG_COLS,
        use_pyarrow=True,
        pyarrow_options={"use_threads": True},
    )
    df = df.fill_null(0)
    x = df.to_numpy()  # [T,16]
    if x.dtype != np.float64:
        x = x.astype(np.float64, copy=False)
    if np.isnan(x).any():
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@lru_cache(maxsize=32768)
def compute_eeg_from_file(filepath: str) -> np.ndarray:
    x = _read_eeg_cols_polars(filepath)
    chain = compute_eeg_chain_from_array(x)
    return chain.astype(np.float64, copy=False)




## === cell 4
class EegModel(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)  # [B, 64]
        logits = self.head(x)  # [B, 6]
        probs = torch.softmax(logits, dim=1)
        probs = probs / probs.sum(dim=1, keepdim=True)
        probs = torch.clamp(probs, 1e-6, 1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs




## === cell 5
models: List[nn.Module] = []
model = EegModel().to(device)
models.append(model)

print("Num models:", len(models))



## === cell 6
TARGET_LEN = 5000  # 50s at 100Hz


def _fix_length_np(x: np.ndarray, target_len: int) -> np.ndarray:
    t = x.shape[-1]
    if t == target_len:
        return x
    if t > target_len:
        return x[:, :target_len]
    pad = target_len - t
    return np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)


def _preprocess_chain_to_tensor(x_np: np.ndarray, device: torch.device) -> torch.Tensor:
    x_np = _fix_length_np(x_np, TARGET_LEN)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    x = torch.diff(x, dim=-1)  # [4, T-1]
    denom = torch.std(x, dim=-1, keepdim=True) + 1e-5
    x = x / denom
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.unsqueeze(0)  # [1, 4, T-1]
    return x


@torch.no_grad()
def gen_ensemble_pred_batch(
    models: List[nn.Module], eeg_ids: List[int], batch_size: int = 64
) -> np.ndarray:
    out = np.zeros((len(eeg_ids), 6), dtype=np.float32)
    for m in models:
        m.eval()

    use_pin = device.type == "cuda"
    xb_cpu = torch.empty(
        (batch_size, 4, TARGET_LEN - 1), dtype=torch.float32, pin_memory=use_pin
    )

    i = 0
    pbar = tqdm(total=len(eeg_ids), desc="infer", leave=False)
    while i < len(eeg_ids):
        cur_bs = min(batch_size, len(eeg_ids) - i)

        bad_mask = np.zeros(cur_bs, dtype=bool)

        fps = [
            os.path.join(EEG_DIR, f"{int(eeg_ids[i + j])}.parquet")
            for j in range(cur_bs)
        ]

        for j, filepath in enumerate(fps):
            try:
                x_np = compute_eeg_from_file(filepath)  # [4, T]
                if not (
                    isinstance(x_np, np.ndarray)
                    and x_np.ndim == 2
                    and x_np.shape[0] == 4
                ):
                    raise ValueError(f"Bad EEG shape {getattr(x_np, 'shape', None)}")
                x_np = _fix_length_np(x_np, TARGET_LEN)
                x_t = torch.from_numpy(x_np.astype(np.float32, copy=False))
                x_t = torch.diff(x_t, dim=-1)
                denom = torch.std(x_t, dim=-1, keepdim=True) + 1e-5
                x_t = x_t / denom
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
                xb_cpu[j].copy_(x_t)
            except Exception:
                bad_mask[j] = True
                xb_cpu[j].zero_()

        x = xb_cpu[:cur_bs].to(device, non_blocking=True)

        preds = []
        for m in models:
            preds.append(m(x).detach())
        p_mean = torch.mean(torch.stack(preds, dim=0), dim=0)  # [B,6]
        p_np = p_mean.detach().cpu().numpy().astype(np.float32, copy=False)

        if bad_mask.any():
            p_np[bad_mask] = np.ones(6, dtype=np.float32) / 6.0

        s = p_np.sum(axis=1, keepdims=True)
        ok = np.isfinite(s) & (s > 0)
        p_np = np.where(ok, p_np / s, (np.ones_like(p_np) / 6.0)).astype(
            np.float32, copy=False
        )

        out[i : i + cur_bs] = p_np
        i += cur_bs
        pbar.update(cur_bs)

    pbar.close()
    return out


@torch.no_grad()
def gen_ensemble_pred(models: List[nn.Module], eeg_id: int) -> np.ndarray:
    filepath = os.path.join(EEG_DIR, f"{int(eeg_id)}.parquet")
    try:
        x = compute_eeg_from_file(filepath)  # [4, T]
        if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 4):
            raise ValueError(
                f"Bad EEG shape {getattr(x, 'shape', None)} for eeg_id={eeg_id}"
            )
        x = _preprocess_chain_to_tensor(x, device=device)
    except Exception:
        return np.ones(6, dtype=np.float32) / 6.0

    preds = []
    for m in models:
        m.eval()
        pred = m(x)  # [1, 6]
        preds.append(pred.detach().cpu().numpy().reshape(-1))

    preds = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
    s = float(preds.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones(6, dtype=np.float32) / 6.0
    preds = preds / s
    return preds




## === cell 7
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _row_to_target_probs(row: dict) -> np.ndarray:
    y = np.array([row[c] for c in LABELS], dtype=np.float32)
    s = float(y.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones(6, dtype=np.float32) / 6.0
    y = y / s
    y = np.clip(y, 1e-6, 1.0)
    y = y / y.sum()
    return y


def _build_training_cache(
    df_train: pl.DataFrame,
    max_steps: int,
    batch_size: int,
) -> Tuple[torch.Tensor, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df_u = (
        df_train.group_by("eeg_id")
        .agg([pl.col(c).sum().alias(c) for c in LABELS])
        .sort("eeg_id")
    )

    eeg_ids_all = df_u["eeg_id"].to_numpy()

    y_votes = df_u.select(LABELS).to_numpy().astype(np.float32, copy=False)
    y_sums = y_votes.sum(axis=1, keepdims=True)
    y_probs = np.where(
        (np.isfinite(y_sums)) & (y_sums > 0),
        y_votes / y_sums,
        (np.ones_like(y_votes) / 6.0),
    )
    y_probs = np.clip(y_probs, 1e-6, 1.0)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    n = df_u.height
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)

    need = max_steps * batch_size
    usable = []
    for ii in idx:
        usable.append(int(ii))
        if len(usable) >= need:
            break

    use_pin = device.type == "cuda"
    xs = torch.empty(
        (len(usable), 4, TARGET_LEN - 1), dtype=torch.float32, pin_memory=use_pin
    )
    usable_out = np.empty((len(usable),), dtype=np.int64)
    good = np.zeros(len(usable), dtype=bool)

    for j, ri in enumerate(tqdm(usable, desc="preload_train_eegs", leave=False)):
        eeg_id = int(eeg_ids_all[ri])
        fp = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
        try:
            x_np = compute_eeg_from_file(fp)
            if not (
                isinstance(x_np, np.ndarray) and x_np.ndim == 2 and x_np.shape[0] == 4
            ):
                continue
            x_np = _fix_length_np(x_np, TARGET_LEN)
            x_t = torch.from_numpy(x_np.astype(np.float32, copy=False))
            x_t = torch.diff(x_t, dim=-1)
            denom = torch.std(x_t, dim=-1, keepdim=True) + 1e-5
            x_t = x_t / denom
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
            xs[j].copy_(x_t)
            usable_out[j] = int(ri)
            good[j] = True
        except Exception:
            continue

    if not good.any():
        return torch.empty((0, 4, TARGET_LEN - 1), dtype=torch.float32), (
            y_probs,
            np.array([], dtype=np.int64),
            eeg_ids_all,
        )

    if not good.all():
        xs = xs[good]
        usable_out = usable_out[good]

    return xs, (y_probs, usable_out, eeg_ids_all)


def train_one_model(
    model: nn.Module,
    df_train: pl.DataFrame,
    max_steps: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    kld = nn.KLDivLoss(reduction="batchmean")

    xs_cpu, pack = _build_training_cache(
        df_train, max_steps=max_steps, batch_size=batch_size
    )
    if xs_cpu.numel() == 0:
        print("No usable training EEG files found; skipping training.")
        model.eval()
        return

    y_probs, usable_out, eeg_ids_all = pack

    use_pin = device.type == "cuda"
    xb_cpu = torch.empty(
        (batch_size, 4, TARGET_LEN - 1), dtype=torch.float32, pin_memory=use_pin
    )
    yb = np.empty((batch_size, 6), dtype=np.float32)

    rng = np.random.default_rng(0)
    cache_n = xs_cpu.shape[0]

    pbar = tqdm(range(max_steps), desc="train_steps", leave=False)
    for step in pbar:
        pick = rng.integers(0, cache_n, size=batch_size, endpoint=False)

        xs_cpu_local = xs_cpu
        y_probs_local = y_probs
        usable_out_local = usable_out
        for b in range(batch_size):
            ci = int(pick[b])
            xb_cpu[b].copy_(xs_cpu_local[ci])
            yb[b] = y_probs_local[int(usable_out_local[ci])]

        x = xb_cpu.to(device, non_blocking=True)
        y = torch.as_tensor(yb, dtype=torch.float32, device=device)

        opt.zero_grad(set_to_none=True)
        p = model(x)  # [B,6] probs
        loss = kld(torch.log(p), y)
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

    model.eval()
    del xs_cpu
    gc.collect()


train_one_model(models[0], df_train, max_steps=900, batch_size=48, lr=1e-3)



## === cell 8
eeg_ids = df_test["eeg_id"].to_list()

preds_final = gen_ensemble_pred_batch(models, eeg_ids, batch_size=64)

print(
    "preds_final shape:",
    preds_final.shape,
    "row sums:",
    preds_final.sum(axis=1).min(),
    preds_final.sum(axis=1).max(),
)



## === cell 9
df_u = df_train.group_by("eeg_id").agg([pl.col(c).sum().alias(c) for c in LABELS])
prior_votes = df_u.select(LABELS).to_numpy().astype(np.float64, copy=False)
prior_sum = prior_votes.sum(axis=0, keepdims=False)
if (not np.isfinite(prior_sum).all()) or float(prior_sum.sum()) <= 0:
    prior = np.ones(6, dtype=np.float64) / 6.0
else:
    prior = prior_sum / float(prior_sum.sum())
prior = np.clip(prior, 1e-12, 1.0)
prior = prior / prior.sum()

alpha = 0.15  # fixed, deterministic shrinkage toward prior to improve KL stability
preds_final = (1.0 - alpha) * preds_final.astype(
    np.float64, copy=False
) + alpha * prior[None, :]
preds_final = np.clip(preds_final, 1e-12, None)
preds_final = (preds_final / preds_final.sum(axis=1, keepdims=True)).astype(
    np.float32, copy=False
)



## === cell 10
df_sub = pd.DataFrame({"eeg_id": eeg_ids})
df_sub[LABELS] = preds_final

vals = df_sub[LABELS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)
df_sub[LABELS] = vals

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
df_sub.head()
