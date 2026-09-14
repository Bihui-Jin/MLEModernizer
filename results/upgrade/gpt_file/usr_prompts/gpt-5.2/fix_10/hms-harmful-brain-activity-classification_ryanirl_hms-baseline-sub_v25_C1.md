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
cudf-polars-cu12==25.6.0
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.3433825606701745

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing external model modules by loading the packaged Kaggle dataset copy (if present) and, if it’s not available, fall back to a safe uniform-probability submission so a valid CSV is always produced. I fix the `compute_spec()` pipeline so the EEG-derived spectrogram has the exact shape expected by `proc_eeg_spec` (4×96×224), which removes the reshape runtime error. I also correct the label list to include all 6 required vote columns (it was missing `grda_vote`), which fixes the submission DataFrame column-length mismatch. Finally, I add strict probability normalization/clipping to guarantee each row sums to 1, preventing Kaggle submission failures and keeping metric semantics intact.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower is better), so the safest way to move toward 0.343 is to make your predictions less “confident-wrong” and more calibrated without changing the model or features. I (1) add per-row probability smoothing (a small mix with uniform) and (2) add temperature scaling on the model log-prob outputs before ensembling, both of which usually reduce KL on this competition when a model is miscalibrated. These are minimal post-processing changes that keep the same architecture, inference loop, and semantics (still valid probabilities summing to 1). I also make the final normalization robust and deterministic but keep your I/O paths and submission schema unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far above the target (0.34338), so we should make the smallest safe changes that reduce KL without changing the model/feature core. The biggest low-risk gain here is fixing an input bug: in `compute_eeg_chain` the `ekg` signal is incorrectly taken from `O2` instead of the actual `EKG` column, which can severely harm predictions. I also switch inference to `torch.inference_mode()` (same semantics, slightly safer) and add a final “safety” probability smoothing at submission time (very small mix with uniform) to reduce catastrophic overconfidence if any row becomes too peaky. Everything else (models, preprocessing, inference loop, output columns/paths) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.34338), so we should improve score with the smallest safe changes that don’t alter the model/feature core. The most likely remaining high-impact issue is a mismatch in the time/frequency axes of the EEG-derived spectrogram: `compute_spec()` currently selects `2:98` on the *frequency* axis instead of the *time/frames* axis, which can severely degrade model inputs while still “working.” I fix `compute_spec()` so it always outputs `(4, 96, 228)` with `96` being the time frames (as your `proc_eeg_spec` expects), keeping all downstream shapes identical. Everything else (models, preprocessing steps, ensembling, temperature scaling, uniform mixing, submission schema) is left unchanged.'
- What this solution (achieved 1.40995) has done: 'Your KL is much worse than the target (lower is better), so we should make the smallest safe changes that typically reduce KL without touching the model, features, or inference loop. The biggest likely remaining issue is that `compute_kaggle_spec_from_file()` currently includes the parquet index/first column in the spectrogram array, which can corrupt the model’s input while still producing the expected shape. I fix it by explicitly dropping the first column when present (keeping the exact downstream shape/semantics), and I also make `proc_kspec()` robust to minor width mismatches so it always feeds the model a consistent `(4,96,224)` tensor. Everything else (models, preprocessing logic, temperature scaling, uniform mixing, and submission formatting) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.34338), so we need a small, safe improvement that doesn’t change your model or feature pipeline. The most likely remaining “silent” quality issue is in `process_spec()`: it currently drops the first column unconditionally (`spec = spec[:, 1:]`), which can misalign the 400-bin layout when the parquet already has exactly 400 feature columns; this can severely hurt predictions while still producing valid shapes. I make `process_spec()` width-aware: only drop the first column when the incoming spectrogram has 401 columns (index-like extra), otherwise keep all 400 bins. Everything else (models, preprocessing, temperature scaling/uniform mixing, inference loop, and submission formatting) stays the same to preserve core logic and semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.34338), so we need a small change that improves correctness without changing the model or training. The most likely remaining silent bug is in `proc_kspec`: it slices `x[:, 2:98]`, which is slicing the width (frequency bins) instead of the intended time axis, corrupting the Kaggle spectrogram input while still producing a valid tensor. I fix `proc_kspec` to slice the time axis (`x[:, 2:98, :]`) and add a tiny width-guard so the function always returns a consistent `(4,96,224)` without shifting real bins. Everything else (models, feature generation, temperature scaling/uniform mixing, inference loop, submission schema/path) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is still far above the target (0.34338), so the smallest likely high-impact fix is to reduce a remaining silent preprocessing mismatch that can corrupt the model input while still producing “valid” shapes. I (1) ensure the torchaudio spectrogram uses the same scaling as training by switching `power=None` (complex) to `power=2.0` (magnitude-squared), so your `abs/log` pipeline matches typical spectrogram expectations and avoids under/over-scaling. I also (2) make `compute_spec()` robust to NaNs/Infs in the raw EEG chain before the transform (these can explode KL) and (3) keep your existing temperature + uniform mixing intact (calibration help) while leaving model architecture/inference loop unchanged. This should improve correctness/calibration without changing core semantics and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.34338), so we should make the smallest changes that improve correctness/calibration without changing the model or feature/training core. The biggest likely remaining silent issue is a preprocessing mismatch in `proc_eeg_spec`: it currently adds `+1` and does not apply the same log/standardization used in `proc_kspec`, which can shift distributions and hurt KL. I align `proc_eeg_spec` to use the same stable `log` + per-channel standardization (while keeping the same input shape `(4,96,224)` and using the same computed spectrogram), and I slightly increase the final uniform-mix smoothing (postprocessing only) to reduce catastrophic overconfidence. Everything else (model loading, feature extraction, inference loop, file paths, submission schema) stays unchanged.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os
import glob
import sys
import math

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
print()

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




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
MODEL_ROOTS = [
    "/kaggle/input/hms-models/",
    os.path.join(DATA_DIR, "hms-models/"),
]

import_ok = False
last_import_err = None
for root in MODEL_ROOTS:
    try:
        if os.path.isdir(root):
            sys.path.append(root)
        from eeg_cnn_rnn_att import EegModel as EegModel  # noqa: F401
        from spc_cnn_att import SpectrogramCnnModel as SpcModel  # noqa: F401
        from comb_model import MultimodalModel  # noqa: F401

        import_ok = True
        break
    except Exception as e:
        last_import_err = e
        continue

print("External model imports available:", import_ok)
if not import_ok:
    print("Import error (will use uniform fallback):", repr(last_import_err))



## === cell 5
models_multimodal = []
if import_ok:
    for model_path in sorted(glob.glob("/kaggle/input/hms-models/stage_6/stage_6/*")):
        try:
            model = MultimodalModel(None, None, None)
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception as e:
            print("Corrupted/unloadable:", model_path, "err:", repr(e))

print("Loaded multimodal models:", len(models_multimodal))




## === cell 6
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 7
import scipy
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2.0
)


@torch.no_grad()
def compute_spec(chain: np.ndarray) -> np.ndarray:
    """
    Produce (4, 96, 228) with correct axes:
      torchaudio Spectrogram returns (C, Freq, Frames).
    """
    chain = np.asarray(chain, dtype=np.float32)
    chain = np.nan_to_num(chain, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor(chain, dtype=torch.float32)  # (4, T)
    x = spectrogram(x)  # (4, Freq, Frames), now real-valued power spectrogram

    target_T = 96
    T = x.shape[-1]
    if T >= target_T:
        start_t = (T - target_T) // 2
        x = x[:, :, start_t : start_t + target_T]  # (4, Freq, 96)
    else:
        pad_total = target_T - T
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        x = torch.nn.functional.pad(x, (pad_l, pad_r))  # pad on Frames axis

    x = x / 15.0
    x = torch.log(x.clamp(min=math.exp(-4), max=math.exp(7)))

    F = x.shape[1]
    target_F = 228
    if F >= target_F:
        start_f = (F - target_F) // 2
        x = x[:, start_f : start_f + target_F, :]  # (4, 228, 96)
    else:
        pad_total = target_F - F
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r))

    x = x.permute(0, 2, 1).contiguous()  # (4, 96, 228)
    return x.cpu().numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
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
                compute_spec_eeg(Fp1, F7),
                compute_spec_eeg(F7, T3),
                compute_spec_eeg(T3, T5),
                compute_spec_eeg(T5, O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_spec_eeg(Fp1, F3),
                compute_spec_eeg(F3, C3),
                compute_spec_eeg(C3, P3),
                compute_spec_eeg(P3, O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F4),
                compute_spec_eeg(F4, C4),
                compute_spec_eeg(C4, P4),
                compute_spec_eeg(P4, O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F8),
                compute_spec_eeg(F8, T4),
                compute_spec_eeg(T4, T6),
                compute_spec_eeg(T6, O2),
            )
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]  # (4, T)

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)  # (4, 96, 228)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 8
def process_spec(spec: np.ndarray) -> np.ndarray:
    if spec.shape[1] == 401:
        spec = spec[:, 1:]
    spec = np.stack(
        [
            spec[:, 0:100].T,
            spec[:, 100:200].T,
            spec[:, 200:300].T,
            spec[:, 300:400].T,
        ]
    )
    return spec


def compute_kaggle_spec_from_file(filepath: str) -> np.ndarray:
    df = pl.read_parquet(filepath)

    if df.width == 401:
        df = df.drop(df.columns[0])
    else:
        c0 = df.columns[0]
        if c0 in ("", "__index_level_0__", "index", "row_id"):
            df = df.drop(c0)

    spec = df.to_numpy().astype(np.float32)
    spec = process_spec(spec)
    return spec




## === cell 9
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




## === cell 10
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
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

    ekg = (
        df_eeg["EKG"].to_numpy() if "EKG" in df_eeg.columns else df_eeg["O2"].to_numpy()
    )
    ekg = butter_filter(ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass")
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1)
    ekg = ekg.reshape(1, -1)

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
    mid = np.stack([compute_eeg(Fz - Cz), compute_eeg(Cz - Pz)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain, mid, ekg


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 11
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 12
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})

import albumentations as A
import cv2

spec_transforms = A.Compose(
    [
        A.Resize(height=96, width=224, interpolation=cv2.INTER_CUBIC, p=1.0),
    ]
)


def proc_kspec(x):
    x = x.copy()

    if x.ndim != 3:
        raise ValueError(f"proc_kspec expected 3D array (4,T,F), got shape={x.shape}")

    if x.shape[1] >= 98:
        x = x[:, 2:98, :]
    else:
        target_T = 96
        pad_total = max(0, target_T - x.shape[1])
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        if pad_total > 0:
            x = np.pad(x, ((0, 0), (pad_l, pad_r), (0, 0)), mode="edge")
        x = x[:, :target_T, :]

    x[np.isnan(x) | np.isinf(x)] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)

    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)

    x = x.transpose(1, 2, 0)  # (T, F, 4)
    x = np.ascontiguousarray(x, dtype=np.float32)

    x = spec_transforms(image=x)["image"]  # -> (96, 224, 4)
    x = x.transpose(2, 0, 1)  # (4, 96, 224)
    x = x.reshape(4, 96, 224)
    return x


def proc_eeg_spec(x):
    x = x.copy()
    x = x[:, :, 2:-2]  # (4, 96, 224)

    x[np.isnan(x) | np.isinf(x)] = 0.0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)

    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)

    x = x.reshape(4, 96, 224).astype(np.float32, copy=False)
    return x


def proc_eeg(eeg, mid, ekg):
    eeg = eeg.copy()
    mid = mid.copy()
    ekg = ekg.copy()

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    mid = mid / mad_std
    mid = mid.clip(-10, 10)

    ekg = ekg / (MAD(ekg, axis=-1).reshape(-1) + 1e-5)

    eeg = eeg.reshape(16, -1)
    eeg = np.concatenate([eeg, mid, ekg], axis=0)
    eeg = eeg.reshape(19, 2_500)
    return eeg


TEMP = 1.35  # >1 softens probabilities
MIX_UNIFORM = 0.12  # mix with uniform to improve calibration


@torch.inference_mode()
def gen_ensemble_pred(df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = int(df_row["eeg_id"].item())
    spc_id = int(df_row["spectrogram_id"].item())
    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")

    if (not import_ok) or (len(models_multimodal) == 0):
        return np.ones(6, dtype=np.float32) / 6.0

    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    kspec = torch.tensor(kspec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg_spec = compute_spec_from_file(eeg_filepath)
    eeg_spec = proc_eeg_spec(eeg_spec)
    eeg_spec = torch.tensor(eeg_spec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg, mid, ekg = compute_eeg_from_file(eeg_filepath)
    eeg = proc_eeg(eeg, mid, ekg)
    eeg = torch.tensor(eeg, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    for model in models_multimodal:
        model.eval()
        out = model(eeg, eeg_spec, kspec)  # expected log-probs
        logp = out.detach() / TEMP
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
        p = logp.exp().cpu().numpy().reshape(-1).astype(np.float64)
        preds.append(p)

    preds = np.mean(preds, axis=0)

    uniform = np.ones_like(preds) / preds.size
    preds = (1.0 - MIX_UNIFORM) * preds + MIX_UNIFORM * uniform

    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum()
    return preds.astype(np.float32)




## === cell 13
preds_final = []
for i in tqdm(range(len(df_test))):
    pred = gen_ensemble_pred(df_test[i])
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float32)
print(
    "preds_final shape:",
    preds_final.shape,
    "min/max:",
    float(preds_final.min()),
    float(preds_final.max()),
)



## === cell 14
eeg_ids = df_test["eeg_id"].to_list()

probs = preds_final.astype(np.float64)

SUB_MIX_UNIFORM = 0.06
uniform = np.full(
    (probs.shape[0], probs.shape[1]), 1.0 / probs.shape[1], dtype=np.float64
)
probs = (1.0 - SUB_MIX_UNIFORM) * probs + SUB_MIX_UNIFORM * uniform

probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)

df_sub = pd.DataFrame({"eeg_id": eeg_ids})
for j, col in enumerate(LABELS):
    df_sub[col] = probs[:, j]

row_sums = df_sub[LABELS].sum(axis=1).to_numpy()
assert np.allclose(
    row_sums, 1.0, atol=1e-5
), f"Row sums not 1 (min={row_sums.min()}, max={row_sums.max()})"

df_sub.to_csv("submission.csv", index=False)
df_sub.head()
