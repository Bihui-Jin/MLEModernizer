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

0.4034617965434584

# 6. Current score

1.47385

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-models/` package by falling back to a safe baseline predictor when the pretrained model files aren’t available, so the notebook runs end-to-end and always writes a valid `submission.csv`. I also fix the EEG tensor shaping bug by making the preprocessing produce exactly 16 channels × 2500 time steps (matching your intended reshape) via a deterministic crop/pad, preventing the runtime error. Finally, I ensure predictions are valid probabilities (non-negative, sum to 1) and that `preds_final` is a proper 2D array so assignment into the submission dataframe works.'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest issue is that you are applying `.exp()` to the model outputs assuming they are log-probabilities; if the loaded model already outputs logits (common with KLDivLoss usage), this produces badly miscalibrated probabilities and a much worse KL. I make a minimal, metric-aligned fix by converting model outputs to probabilities via `softmax` (and only falling back to `exp()` if the output already looks like log-probs), which preserves the model and inference flow but corrects probability normalization. I also add a small, safe probability floor and re-normalization (already present) to avoid invalid rows for Kaggle. No architecture, feature extraction, or loop structure is changed; only the output-to-probability mapping is fixed.'
- What this solution (achieved 1.62868) has done: 'Your current score is much worse than the target (lower-is-better), so we should make a small, metric-aligned improvement without changing the model or feature pipeline. The biggest safe gain here is to match training label semantics: the competition’s targets are *vote distributions per eeg_id*, but your inference produces one prediction per test eeg_id while your PRIOR is computed per-row (with many overlapping subsamples), which biases the distribution; we compute a patient-balanced, eeg_id-aggregated prior and use it for (a) fallback and (b) light post-hoc blending to stabilize probabilities. This keeps core logic identical (same model, same preprocessing, same inference loop) and only adjusts the final probability calibration in a way that typically reduces KL divergence. We also ensure the ensemble mean is done in probability-space (already) and add a tiny blend with the improved prior to reduce overconfident errors.'
- What this solution (achieved 1.62868) has done: 'We keep your modeling and feature pipeline intact, but fix two issues that are likely inflating KL substantially: (1) the test set contains repeated `eeg_id`s, so you are currently writing duplicate rows instead of one prediction per `eeg_id`; we predict once per unique `eeg_id` and then re-align to `sample_submission.csv`. (2) Your `_outputs_to_probs` heuristic can mis-detect logits vs log-probs; we make it deterministic by always using `softmax` (safe for logits and log-probs up to a constant) and keep your probability floor + renormalization. Finally, we reduce the prior blend from 0.10 to a smaller 0.02 to avoid washing out the model signal (a minimal calibration change aimed at lowering KL toward your target).'
- What this solution (achieved 1.48697) has done: 'Your current score (1.62868, lower-is-better) is far from the target (0.40346), so we need a small but meaningful KL-aligned improvement without changing the model, preprocessing, or inference loop. The biggest safe issue is that your vote “prior” is computed as an unweighted mean over patients, which can be miscalibrated versus the competition’s label distribution (vote distribution per EEG sample); we switch to an eeg_id-aggregated global prior (still derived only from train labels) to better match the test-time marginal distribution. We keep your softmax conversion and unique-eeg_id prediction, but reduce the prior blend weight slightly and add a final “temperature” calibration on probabilities (power + renorm) which often reduces KL by softening overconfident outputs while preserving semantics. All changes are localized to prior computation and the final post-processing; the model and feature pipeline remain identical.'
- What this solution (achieved 1.39779) has done: 'We make two minimal, metric-aligned changes that typically reduce KL without touching your model architecture or feature pipeline: (1) compute the prior in a way that matches the evaluation unit (vote distribution per `eeg_id`, weighted by how many train rows each `eeg_id` contributes, rather than unweighted mean over `eeg_id`s), and (2) slightly increase the prior blending strength in post-calibration to curb overconfident probabilities (a common KL failure mode) while keeping your existing power/temperature softening. Everything else (EEG preprocessing, tensor shape, inference loop, softmax, and submission writing) stays the same, and the script still produces a valid `submission.csv` with rows summing to 1. These are small changes aimed at moving your score down (lower is better) toward the target band rather than chasing the best possible score.'
- What this solution (achieved 1.47385) has done: 'We keep your model, preprocessing, and inference loop intact, and only make two minimal changes aimed at reducing KL (lower-is-better) toward your target: compute a more label-unit-matched prior by aggregating train vote-distributions at the `eeg_id` level (instead of averaging over overlapping subsamples), and apply the same prior-blend + power calibration to the fallback path as well so the output distribution is consistent. This should reduce miscalibration/overconfidence that KL heavily penalizes, without changing architecture or training. Everything else (softmax on outputs, 16×2500 shaping, unique `eeg_id` prediction, and submission formatting) is preserved. The script still runs end-to-end and writes a valid `submission.csv` with row sums equal to 1.'

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
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    state = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model




## === cell 4
import glob
import sys

HMS_MODELS_DIR = "/kaggle/input/hms-models/"
has_hms_models = os.path.isdir(HMS_MODELS_DIR)

CurrModel = None
if has_hms_models:
    sys.path.append(HMS_MODELS_DIR)
    try:
        from eeg_cnn_rnn import EegModel  # type: ignore

        CurrModel = EegModel
    except Exception:
        CurrModel = None

if CurrModel is None:

    class FallbackEegModel(nn.Module):
        def __init__(self, n_classes: int = 6):
            super().__init__()
            self.n_classes = n_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            return torch.full(
                (b, self.n_classes),
                -np.log(self.n_classes),
                device=x.device,
                dtype=x.dtype,
            )

    CurrModel = FallbackEegModel

print(
    "Using pretrained model code:",
    (
        "yes"
        if has_hms_models and CurrModel.__name__ != "FallbackEegModel"
        else "no (fallback)"
    ),
)



## === cell 5
models = []

fold_dirs = ["/kaggle/input/hms-models/baseline_eeg_diff/*"]

for fold_dir in fold_dirs:
    for fold_path in glob.glob(fold_dir):
        model_path = os.path.join(fold_path, "model_best_val_g10.pt")
        if os.path.exists(model_path):
            model = CurrModel()
            model = load_model(model_path, model)
            model = model.to(device)
            models.append(model)

print("Loaded models:", len(models))



## === cell 6
import librosa
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
    chain = preprocess_chain(
        chain
    )  # defined in original codebase; unused in this notebook
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
        N=order,
        Wn=(
            np.array(cutoff_freq) / (0.5 * fs)
            if isinstance(cutoff_freq, (list, tuple, np.ndarray))
            else cutoff_freq / (0.5 * fs)
        ),
        btype=btype,
        analog=False,
    )
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
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


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 9
TARGET_T = 2500


def _to_16xT(chain_4x4xT: np.ndarray) -> np.ndarray:
    x = chain_4x4xT.reshape(-1, chain_4x4xT.shape[-1])  # (16, T)
    T = x.shape[-1]
    if T < TARGET_T:
        pad = TARGET_T - T
        x = np.pad(x, ((0, 0), (0, pad)), mode="edge")
    elif T > TARGET_T:
        x = x[:, :TARGET_T]
    return x.astype(np.float32)




## === cell 10
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

vote_mat = df_train.select(LABELS).to_numpy()
vote_sum = vote_mat.sum(axis=1, keepdims=True)
vote_sum[vote_sum == 0] = 1.0
row_probs = vote_mat / vote_sum

train_tmp = df_train.select(["eeg_id"]).with_columns(
    [pl.Series(name=lab, values=row_probs[:, j]) for j, lab in enumerate(LABELS)]
)

eeg_level = train_tmp.group_by("eeg_id").agg(
    [pl.col(lab).mean().alias(lab) for lab in LABELS]
)

PRIOR = eeg_level.select(LABELS).to_numpy().mean(axis=0).astype(np.float64)
PRIOR = np.clip(PRIOR, 1e-12, None)
PRIOR = PRIOR / PRIOR.sum()

PRIOR




## === cell 11
@torch.no_grad()
def _outputs_to_probs(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    return torch.softmax(y, dim=-1)


def _post_calibrate_probs(
    p: np.ndarray, prior: np.ndarray, alpha: float, power: float
) -> np.ndarray:
    p = (1.0 - alpha) * p + alpha * prior
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()

    if power != 1.0:
        p = np.power(p, power)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
    return p.astype(np.float64)


@torch.no_grad()
def gen_ensemble_pred(models: List[nn.Module], eeg_id: int) -> np.ndarray:
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    alpha = 0.03
    power = 0.95

    if len(models) == 0:
        return _post_calibrate_probs(PRIOR.copy(), PRIOR, alpha=alpha, power=power)

    x = compute_eeg_from_file(filepath)
    x = _to_16xT(x)
    x[np.isnan(x) | np.isinf(x)] = 0.0

    x = torch.tensor(x, device=device)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
    x = x.reshape(16, 1, TARGET_T).unsqueeze(0)  # (1,16,1,2500)

    preds = []
    for model in models:
        model.eval()
        out = model(x)
        prob = _outputs_to_probs(out)
        prob = prob.detach().cpu().numpy().reshape(-1)
        preds.append(prob)

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = _post_calibrate_probs(preds, PRIOR, alpha=alpha, power=power)
    return preds




## === cell 12
test_eeg_ids = df_test.select("eeg_id").to_series().to_numpy()
unique_eeg_ids = np.unique(test_eeg_ids)

pred_map: Dict[int, np.ndarray] = {}
for eeg_id in tqdm(unique_eeg_ids, desc="Predict unique eeg_id"):
    pred_map[int(eeg_id)] = gen_ensemble_pred(models, int(eeg_id))

sub_eeg_ids = sample_submission["eeg_id"].to_list()
preds_final = np.vstack([pred_map[int(eid)] for eid in sub_eeg_ids]).astype(np.float64)

print(
    preds_final.shape,
    preds_final.min(),
    preds_final.max(),
    np.allclose(preds_final.sum(axis=1), 1.0, atol=1e-6),
)



## === cell 13
df_sub = pd.DataFrame({"eeg_id": sample_submission["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index=False)

df_sub.head()
print("Wrote submission.csv with shape:", df_sub.shape)
