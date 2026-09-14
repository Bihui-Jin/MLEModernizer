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

0.3414374905736105

# 6. Current score

1.18278

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I first remove the hard dependency on a missing `/kaggle/input/hms-models/` package by adding a safe fallback that produces valid probability predictions when those external model files are unavailable, so the notebook runs end-to-end and writes `submission.csv`. Next, I fix the EEG-derived spectrogram reshape bug by making `proc_eeg_spec` robust to the actual time dimension returned by `compute_spec` (pad/crop to the expected width), which unblocks inference even if models are later present. Finally, I ensure `preds_final` is always a proper `(n_test, 6)` float array aligned to `eeg_id`, and that each row sums to 1 (with clipping), so the submission passes Kaggle format checks.'
- What this solution (achieved 1.4582) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.3414), and the biggest reason is that when external models are missing you fall back to a global class-prior for every test row, which performs poorly on KL. Keeping the same overall inference loop and submission semantics, I replace that fallback with a lightweight, fully-local heuristic that uses the provided test spectrogram parquet to produce per-row probabilities, then mix it slightly with the train prior for stability and clip/renormalize for valid KL. This changes only the “no external models” path (core multimodal model path stays untouched) and should move the score substantially toward the target while still finishing within the time limit. I also cache the computed train prior once (instead of recomputing it 9850 times) to both speed up and reduce minor numeric drift.'
- What this solution (achieved 1.58212) has done: 'You’re far above (worse than) the target KL (1.4582 vs 0.3414, lower-is-better), and the biggest lever without changing your core multimodal path is improving the “no external models” fallback so it better matches the label distribution. I keep the same overall inference flow, but (1) compute a more realistic prior by normalizing per-sample vote distributions before averaging, and (2) upgrade the heuristic to use multiple robust spectrogram features (band energies, temporal variability, spectral entropy/flatness) and calibrate it by blending with the improved prior. I also add a tiny cache for reading test spectrogram parquets so we don’t repeatedly parse the same file when spectrogram_ids repeat (speed/stability, no semantic change). All changes are confined to the fallback branch and prior computation; the external-model inference remains untouched.'
- What this solution (achieved 1.45036) has done: 'Your current KL (1.58212, lower-is-better) is far worse than the target (0.3414), so we should improve performance rather than “match by degrading.” With external models missing, the only lever is the fallback predictor: I keep your same inference flow and submission semantics, but make the fallback more label-aligned by (1) calibrating the heuristic using a simple, fixed “difficulty/uncertainty” gate derived from spectrogram entropy/flatness/temporal variability, and (2) using a per-class temperature + prior-mixing schedule so uncertain cases revert more to the (well-estimated) train prior while confident cases rely more on the heuristic. I also fix a silent bug in `compute_eeg_chain` where `ekg` incorrectly uses `O2` (should be `EKG` when present); this only affects the external-model path but is correctness-preserving and low-risk. These are minimal, score-relevant changes and still write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.1456) has done: 'Your current KL (1.45036, lower-is-better) is still far from the target (0.3414), so we should improve the fallback branch (since external models are typically unavailable). I keep your exact pipeline structure, but make the fallback predictor more label-aligned by calibrating the heuristic mapping using statistics learned from train spectrograms (simple linear/logit mapping fit with ridge on aggregated per-spectrogram features), then blend with the existing train prior for stability. This preserves the “no-training neural model” core logic while making predictions much closer to the empirical label distribution, which should reduce KL substantially. I also add a small cache for computed fallback features to keep runtime within the 600s limit.'
- What this solution (achieved 1.15478) has done: 'Your current KL (1.1456, lower-is-better) is still far above the target (0.3414), so we should improve predictions rather than “match by degrading.” Keeping your exact inference structure and fallback “train-fitted linear calibrator on spectrogram features,” I fix a bug where the uncertainty-gating uses the wrong feature indices (it was accidentally reading `asym_lr` as entropy, etc.), which can badly miscalibrate the prior mixing and hurt KL. I also make the calibrator fit use a deterministic, spread-out subset of `spectrogram_id`s (instead of “first N after sort”), which reduces sampling bias with minimal runtime impact. Finally, I keep the same output semantics but add a small epsilon-smoothing (Dirichlet-like) after blending to reduce overconfident zeros, which is directly beneficial for KL and won’t change your core logic.'
- What this solution (achieved 1.15001) has done: 'Your current KL (1.15478, lower-is-better) is still far above the target (0.3414), so we should improve the fallback predictor (the only path that consistently runs when external models are absent). Keeping your exact “train-fitted linear calibrator on spectrogram features → uncertainty-gated prior blend” core logic, I make two minimal, score-relevant fixes: (1) fit the calibrator on logit-transformed targets (instead of raw log-probabilities) so the mapping matches the softmax inference better, and (2) add a tiny, deterministic per-class bias-correction step (computed on the same fit subset) to align average predicted probabilities to average training targets, which typically reduces KL without changing model structure. I also ensure the cache keys remain filepaths and keep runtime bounded by reusing the same subset and not adding any extra loops over all rows.'
- What this solution (achieved 1.16479) has done: 'Your current KL (1.15001, lower-is-better) is still far above the target (0.3414), so we should improve the fallback path (the external models are typically unavailable). Keeping the same fallback core logic (train-fitted linear calibrator on spectrogram features → uncertainty-gated prior blend), I make two minimal, KL-relevant calibration tweaks: (1) add a per-class temperature vector (estimated on the same train subset) to better match class-wise over/under-confidence, and (2) slightly increase the Dirichlet-like smoothing epsilon (and make it uncertainty-dependent) to reduce overconfident near-zeros that are heavily penalized by KL. Everything else (feature extraction, ridge fit, softmax inference, submission schema) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.18278) has done: 'Your current KL (1.16479, lower-is-better) is still far above the target (0.3414), so we should improve the fallback (no-external-models) predictions with minimal, metric-aligned calibration rather than changing the modeling pipeline. I keep your existing “train-fitted linear calibrator on spectrogram features → uncertainty-gated prior blend” intact, but replace the heuristic per-class temperature estimation with a direct, stable optimization of temperature that minimizes KL on the same train subset. I also make the final prior-blend weight `alpha` slightly self-calibrating using how peaked the prediction is (max-probability), which helps KL without changing feature extraction or the inference loop. Everything still runs end-to-end, stays within Kaggle constraints, and writes a valid `submission.csv` with rows summing to 1.'

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




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    sd = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(sd, dict) and "model_state_dict" in sd:
        model.load_state_dict(sd["model_state_dict"])
    else:
        model.load_state_dict(sd)
    return model




## === cell 4
HAVE_EXTERNAL_MODELS = False
EegModel = None
SpcModel = None
MultimodalModel = None

try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_att import EegModel as EegModel  # noqa: F401
    from spc_cnn_att import SpectrogramCnnModel as SpcModel  # noqa: F401
    from comb_model import MultimodalModel  # noqa: F401

    HAVE_EXTERNAL_MODELS = True
except Exception as e:
    print(
        "External model package not found; using fallback predictor. Import error:",
        repr(e),
    )
    HAVE_EXTERNAL_MODELS = False



## === cell 5
models_multimodal = []

if HAVE_EXTERNAL_MODELS:
    for model_path in sorted(
        glob.glob(
            "/kaggle/input/hms-models/final_multimodal_sanity/final_multimodal_sanity/*"
        )
    ):
        try:
            model = MultimodalModel(None, None, None)
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception as e:
            print("Corrupted:", model_path, "err:", repr(e))

    for model_path in sorted(glob.glob("/kaggle/input/hms-models/stage_6/stage_6/*")):
        try:
            model = MultimodalModel(None, None, None)
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception as e:
            print("Corrupted:", model_path, "err:", repr(e))

print("Total multimodal models:", len(models_multimodal))




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
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
)


@torch.no_grad()
def compute_spec(chain):
    chain = torch.as_tensor(chain, dtype=torch.float32)
    chain = spectrogram(chain)
    chain = chain[:, :, 2:98]
    chain = torch.abs(chain) / 15.0
    chain = torch.log(chain.clamp(min=math.exp(-4), max=math.exp(7)))
    chain = chain.mean(axis=1)
    return chain.cpu().numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
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

    chain = np.stack([ll, lp, rp, rl])[:, 0]

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 8
def process_spec(spec: np.ndarray) -> np.ndarray:
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


_SPEC_CACHE: Dict[str, np.ndarray] = {}


def compute_kaggle_spec_from_file(filepath: str) -> np.ndarray:
    if filepath in _SPEC_CACHE:
        return _SPEC_CACHE[filepath]
    spec = pl.read_parquet(filepath).to_numpy().astype(np.float32)
    spec = process_spec(spec)
    _SPEC_CACHE[filepath] = spec
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

    if "EKG" in df_eeg.columns:
        ekg = df_eeg["EKG"].to_numpy()
    else:
        ekg = df_eeg["O2"].to_numpy()
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
        A.Resize(
            height=96,
            width=224,
            interpolation=cv2.INTER_CUBIC,
            always_apply=True,
            p=1.0,
        ),
    ]
)


def proc_kspec(x):
    x = x.copy()
    x = x[:, 2:98]
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)

    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)

    x = x.transpose(1, 2, 0)
    x = spec_transforms(image=x)["image"]
    x = x.transpose(2, 0, 1)

    x = x.reshape(4, 96, 224)
    return x


def _pad_or_crop_last_dim(x: np.ndarray, target: int, mode: str = "edge") -> np.ndarray:
    """Pad/crop along last dim to `target`."""
    cur = x.shape[-1]
    if cur == target:
        return x
    if cur > target:
        return x[..., :target]
    pad = target - cur
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode=mode)


def proc_eeg_spec(x):
    x = x.copy()
    x = x[:, :, 2:-2]
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x + 1

    x = _pad_or_crop_last_dim(x, 96 * 224, mode="edge")
    x = x.reshape(4, 96, 224)
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


def flip_h(eeg):
    eeg = eeg.copy()
    eeg = eeg[::-1]
    return eeg.copy()


def flip_v(eeg):
    eeg = eeg.copy()
    eeg = eeg[:, ::-1]
    return eeg.copy()




## === cell 13
def _safe_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _kl_divergence_rowwise(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """y_true,y_pred in (N,C), already normalized; returns (N,) KL(y_true||y_pred)."""
    eps = 1e-12
    yt = np.clip(y_true, eps, 1.0)
    yp = np.clip(y_pred, eps, 1.0)
    yt = yt / yt.sum(axis=1, keepdims=True)
    yp = yp / yp.sum(axis=1, keepdims=True)
    return np.sum(yt * (np.log(yt) - np.log(yp)), axis=1)


y_train_votes = df_train.select(LABELS).to_numpy().astype(np.float64)
row_sums = y_train_votes.sum(axis=1, keepdims=True)
row_sums = np.clip(row_sums, 1.0, None)
y_train_prob = y_train_votes / row_sums
TRAIN_PRIOR = y_train_prob.mean(axis=0).astype(np.float64)
TRAIN_PRIOR = np.clip(TRAIN_PRIOR, 1e-8, 1.0)
TRAIN_PRIOR = TRAIN_PRIOR / TRAIN_PRIOR.sum()


_FALLBACK_FEAT_CACHE: Dict[str, np.ndarray] = {}


def _fallback_features_from_specfile(spc_filepath: str) -> np.ndarray:
    if spc_filepath in _FALLBACK_FEAT_CACHE:
        return _FALLBACK_FEAT_CACHE[spc_filepath]
    spec = compute_kaggle_spec_from_file(spc_filepath)  # (4, 100, T)
    x = np.asarray(spec, dtype=np.float64)
    x[np.isnan(x) | np.isinf(x)] = 0.0
    x = np.clip(x, 0.0, None)
    x = np.log1p(x)

    low = x[:, :20, :].mean()
    mid = x[:, 20:60, :].mean()
    high = x[:, 60:, :].mean()
    total = x.mean()

    band_ts = x.mean(axis=1)  # (4, T)
    t_std = np.clip(np.std(band_ts, axis=-1).mean(), 0.0, 5.0)
    t_diff = np.clip(np.mean(np.abs(np.diff(band_ts, axis=-1))), 0.0, 5.0)

    xf = x.mean(axis=2)  # (4, 100)
    p = xf / (xf.sum(axis=1, keepdims=True) + 1e-12)
    ent = (-np.sum(p * np.log(p + 1e-12), axis=1) / np.log(p.shape[1])).mean()
    ent = float(np.clip(ent, 0.0, 1.0))

    gm = np.exp(np.mean(np.log(xf + 1e-12), axis=1))
    am = np.mean(xf + 1e-12, axis=1)
    flat = float(np.clip(np.mean(gm / am), 0.0, 2.0))

    montage_energy = x.mean(axis=(1, 2))  # (4,)
    asym_lr = float(
        np.abs(
            (montage_energy[0] + montage_energy[1])
            - (montage_energy[2] + montage_energy[3])
        )
    )

    feat = np.array(
        [
            1.0,
            float(low),
            float(mid),
            float(high),
            float(total),
            float(t_std),
            float(t_diff),
            float(ent),
            float(flat),
            float(asym_lr),
        ],
        dtype=np.float64,
    )
    _FALLBACK_FEAT_CACHE[spc_filepath] = feat
    return feat


def _fit_fallback_calibrator(
    df_train: pl.DataFrame, data_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (W, feat_mean, b, global_temp) where:
      raw_logits = (feat - feat_mean) @ W + b, W shape (F, 6), b shape (6,)
      global_temp is optimized on the same fit subset to directly reduce KL (helps metric).
    """
    spec_dir_train = os.path.join(data_dir, "train_spectrograms/")

    y_votes = df_train.select(["spectrogram_id"] + LABELS).to_pandas()
    yv = y_votes[LABELS].values.astype(np.float64)
    rs = np.clip(yv.sum(axis=1, keepdims=True), 1.0, None)
    yp = yv / rs
    y_votes[LABELS] = yp

    y_agg = y_votes.groupby("spectrogram_id")[LABELS].mean()

    spc_ids_all = y_agg.index.to_numpy()
    spc_ids_all = np.sort(spc_ids_all)

    max_fit = 3000
    if spc_ids_all.shape[0] > max_fit:
        idx = np.linspace(0, spc_ids_all.shape[0] - 1, num=max_fit, dtype=int)
        spc_ids = spc_ids_all[idx]
    else:
        spc_ids = spc_ids_all

    X_list = []
    Y_list = []
    for spc_id in tqdm(
        spc_ids, desc="Fitting fallback calibrator (train spectrogram features)"
    ):
        spc_fp = os.path.join(spec_dir_train, f"{int(spc_id)}.parquet")
        if not os.path.exists(spc_fp):
            continue
        feat = _fallback_features_from_specfile(spc_fp)
        X_list.append(feat)
        Y_list.append(y_agg.loc[spc_id].values.astype(np.float64))

    X = np.stack(X_list, axis=0)  # (N, F)
    Y = np.stack(Y_list, axis=0)  # (N, 6)

    feat_mean = X.mean(axis=0, keepdims=True)
    Xc = X - feat_mean

    T = _safe_logit(Y)

    lam = 2.0
    XtX = Xc.T @ Xc
    F = XtX.shape[0]
    W = np.linalg.solve(XtX + lam * np.eye(F), Xc.T @ T)  # (F, 6)

    raw0 = Xc @ W  # (N, 6)
    p0 = _safe_softmax(raw0, axis=1)
    eps_t = 1e-6
    ybar = Y.mean(axis=0) + eps_t
    pbar = p0.mean(axis=0) + eps_t
    b0 = (_safe_logit(ybar) - _safe_logit(pbar)).astype(np.float64)

    def eval_mean_kl(temp: float) -> float:
        logits = (raw0 + b0.reshape(1, -1)) / float(temp)
        p = _safe_softmax(logits, axis=1)
        return float(np.mean(_kl_divergence_rowwise(Y, p)))

    temps = np.linspace(0.7, 1.6, 19)
    kls = np.array([eval_mean_kl(t) for t in temps], dtype=np.float64)
    best_idx = int(np.argmin(kls))
    best_temp = float(temps[best_idx])

    logits_bt = raw0 / best_temp
    p_bt = _safe_softmax(logits_bt, axis=1).mean(axis=0) + eps_t
    b = (_safe_logit(ybar) - _safe_logit(p_bt)).astype(np.float64)  # (6,)

    return (
        W.astype(np.float64),
        feat_mean.reshape(-1).astype(np.float64),
        b.astype(np.float64),
        float(best_temp),
    )


FALLBACK_W, FALLBACK_FEAT_MEAN, FALLBACK_BIAS, FALLBACK_GLOBAL_TEMP = (
    _fit_fallback_calibrator(df_train, DATA_DIR)
)


def _fallback_pred_from_kaggle_spec(spc_filepath: str) -> np.ndarray:
    """
    Use calibrated linear logits from train-fitted mapping, then blend with TRAIN_PRIOR
    using an uncertainty gate.
    """
    feat = _fallback_features_from_specfile(spc_filepath).astype(np.float64)
    feat_c = feat - FALLBACK_FEAT_MEAN
    logits = feat_c @ FALLBACK_W + FALLBACK_BIAS  # (6,)

    ent = float(np.clip(feat[7], 0.0, 1.0))
    flat = float(np.clip(feat[8], 0.0, 2.0))
    t_std = float(np.clip(feat[5], 0.0, 5.0))

    unc = (
        0.55 * ent
        + 0.25 * np.clip(flat / 1.2, 0.0, 1.0)
        + 0.20 * np.clip(1.0 - (t_std / 1.2), 0.0, 1.0)
    )
    unc = float(np.clip(unc, 0.0, 1.0))

    temp_unc = 0.90 + 0.70 * unc
    logits_adj = logits / (FALLBACK_GLOBAL_TEMP * temp_unc)
    p_h = _safe_softmax(logits_adj, axis=-1)

    pmax = float(np.max(p_h))
    peaked = float(np.clip((pmax - 1.0 / 6.0) / (1.0 - 1.0 / 6.0), 0.0, 1.0))
    alpha = (0.88 - 0.36 * unc) * (0.80 + 0.20 * peaked)
    alpha = float(np.clip(alpha, 0.35, 0.95))

    p = alpha * p_h + (1.0 - alpha) * TRAIN_PRIOR

    eps = 4e-4 + 6e-4 * unc
    p = (1.0 - 6 * eps) * p + eps

    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    return p.astype(np.float32)


@torch.no_grad()
def gen_ensemble_pred(df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = int(df_row["eeg_id"].item())
    spc_id = int(df_row["spectrogram_id"].item())
    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")

    if (not HAVE_EXTERNAL_MODELS) or (len(models_multimodal) == 0):
        return _fallback_pred_from_kaggle_spec(spc_filepath)

    preds = []

    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    kspec = torch.tensor(kspec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg_spec = compute_spec_from_file(eeg_filepath)
    eeg_spec = proc_eeg_spec(eeg_spec)
    eeg_spec = torch.tensor(eeg_spec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg, mid, ekg = compute_eeg_from_file(eeg_filepath)
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = proc_eeg(eeg, mid, ekg)
    eeg = torch.tensor(eeg, dtype=torch.float32, device=device).unsqueeze(0)

    for model in models_multimodal:
        model.eval()
        out = model(eeg, eeg_spec, kspec)
        out = out.exp().detach().float().cpu().numpy().reshape(-1)
        preds.append(out)

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-8, 1e9)
    preds = preds / preds.sum()
    return preds.astype(np.float32)




## === cell 14
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



## === cell 15
if preds_final.ndim != 2 or preds_final.shape[1] != len(LABELS):
    raise ValueError(
        f"preds_final must be (n_test, {len(LABELS)}), got {preds_final.shape}"
    )

preds_final = np.clip(preds_final, 1e-6, 1.0)
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)

df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index=False)

print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
