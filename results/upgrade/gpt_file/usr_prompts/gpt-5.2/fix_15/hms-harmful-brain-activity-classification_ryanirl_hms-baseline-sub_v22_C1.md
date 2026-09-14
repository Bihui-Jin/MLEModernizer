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

0.3527710471868035

# 6. Current score

1.39829

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the shape bug in `compute_spec()`/`proc_eeg_spec()` that causes the transpose error by making the spectrogram pipeline produce a consistent `(4, T, 96)` array (channels × time × freq), and then resize along the time axis safely. I also remove the strict assertion that fails due to floating-point summation and instead enforce normalization robustly (still guaranteeing valid probability rows that sum to 1 within numerical tolerance). These are execution/stability fixes and keep the same core “prior model fallback + feature extraction” logic, while ensuring a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the runtime error in `proc_eeg()` caused by mismatched time lengths between the EEG chain, `mid`, and `ekg` after filtering/binning, by center-cropping or padding all three to a consistent target length before concatenation (keeping the same signal processing and model behavior). I also correct an obvious bug where `ekg` was accidentally taken from the `O2` channel instead of the dedicated `EKG` column when present, which should improve feature correctness and move the score down toward the target. Finally, I ensure the pipeline completes end-to-end and always writes a valid `submission.csv` with properly normalized probabilities.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by CPU-side feature extraction inside `_cpu_features_for_eeg_id`, especially repeated `scipy.signal.filtfilt` calls and heavy per-row object conversions (`pyarrow -> pandas -> numpy`). To preserve identical core logic and outputs, I keep the exact same feature formulas and model inference, but remove redundant work by (1) reading Parquet directly to NumPy via PyArrow without creating Pandas objects, (2) caching all Butterworth filter coefficients within each worker (still deterministic), and (3) applying filters across stacked signals in a vectorized way (same `filtfilt` math, just fewer Python calls). I also reduce scheduler overhead in the main loop by using `executor.map` with a bounded `chunksize` while keeping the same parallelism and inference behavior. No sampling, early stopping, or precision reduction is introduced.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by per-file parquet reads plus expensive CPU `filtfilt` calls repeated for both EEG and spectrogram branches, and by per-sample GPU spectrogram calls in a Python loop. The optimizations below keep the exact same feature definitions and model semantics, but reduce overhead by (1) reading parquet with `pyarrow` (faster than polars here) and directly to NumPy, (2) computing all required bandpass-filtered signals in just two `filtfilt` passes per EEG (one for EEG features, one for spec features) instead of many small calls, (3) batching the GPU spectrogram computation and the albumentations resize so the GPU/CPU work happens in vectorized batches, and (4) using a thread pool (I/O + SciPy releases the GIL) to avoid process spawn/IPC overhead while preserving determinism. All changes are runtime-only and do not alter architecture, losses, or the mathematical operations used to generate features/predictions.'
- What this solution (achieved 1.39744) has done: 'We fix the batching crash by ensuring every `spec_chain` fed to `np.stack` has an identical time length `T` (some EEG files differ slightly in length), using a deterministic center-crop/pad at the CPU feature stage before queueing. This is an execution/stability fix and does not change the model logic; it only guarantees the existing spectrogram pipeline can batch correctly. Additionally, we read the `EKG` column when present (instead of implicitly using `O2`) in the fast PyArrow path to keep feature extraction consistent with the intended logic, which should also move the KL score down toward the target. The rest of the pipeline (prior-model fallback, feature formulas, normalization, submission writing) is preserved.'
- What this solution (achieved 1.40079) has done: 'Your current pipeline is effectively a constant “train prior” predictor, so the KL score is dominated by mismatch between the train-set mean distribution and the test-set distribution; the smallest legitimate way to move the score down toward your target (lower is better) is to calibrate that constant distribution using a simple temperature smoothing toward uniform. I keep the exact same core logic (PriorModel + feature extraction + batching) and only change the post-processing to use a tuned convex mix `pred = (1-α)*pred + α*uniform`, which is guaranteed to keep valid probabilities summing to 1. This is stable, fast, and directly targets KL reduction without changing architecture/training loops. I also ensure all normalizations remain numerically safe.'
- What this solution (achieved 1.40523) has done: 'I make one score-focused change: adjust the constant-prediction calibration by changing the uniform-smoothing strength, because your pipeline is effectively predicting the train prior and KL can improve a lot from better calibration even without changing the model/features. Specifically, I increase the convex mix toward uniform (still producing valid probabilities summing to 1), which is the smallest legitimate lever to reduce KL when predictions are overconfident/mismatched to test. I also keep all existing normalization/clipping exactly as-is for submission validity and stability. No architecture, feature extraction, loss, or training changes are introduced.'
- What this solution (achieved 1.39829) has done: 'Your current model is effectively constant (train PRIOR) and the only score-sensitive lever you’re using is the convex “smooth toward uniform” calibration; we therefore keep everything else identical and only tune that single parameter to move KL closer to the target. Since lower is better and your current score (1.40523) is far worse than the target (0.35277), we should reduce over-smoothing toward uniform and bias predictions closer to the PRIOR. I make a minimal change: decrease `SMOOTH_ALPHA` (still keeping strict probability validity via clipping+renorm) and leave all feature extraction/inference unchanged. This is the smallest legitimate adjustment expected to reduce KL without altering architecture/training/feature logic.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os
import math

import torch
import torch.nn as nn

from typing import Tuple, Dict



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

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    state = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model




## === cell 4
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_votes = df_train.select(LABELS).to_numpy().astype(np.float64)
train_votes_sum = train_votes.sum(axis=1, keepdims=True)
train_probs = train_votes / np.clip(train_votes_sum, 1.0, None)
PRIOR = train_probs.mean(axis=0)
PRIOR = PRIOR / PRIOR.sum()

print("Using PRIOR (fallback, sums to 1):", PRIOR, "sum=", PRIOR.sum())


class PriorModel(nn.Module):
    def __init__(self, prior: np.ndarray):
        super().__init__()
        self.register_buffer(
            "log_prior", torch.log(torch.tensor(prior, dtype=torch.float32))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        return self.log_prior.unsqueeze(0).expand(bs, -1)


models_eeg = [PriorModel(PRIOR).to(device).eval()]
models_eeg_spc = [PriorModel(PRIOR).to(device).eval()]

len(models_eeg) + len(models_eeg_spc)




## === cell 5
def MAD(signal, axis=-1):
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import butter, filtfilt

_BUTTER_CACHE: Dict[
    Tuple[int, Tuple[float, ...], int, str], Tuple[np.ndarray, np.ndarray]
] = {}


def _butter_cached(fs: int, cutoff_freq, order: int, btype: str):
    if isinstance(cutoff_freq, np.ndarray):
        cutoff_key = tuple(map(float, cutoff_freq.tolist()))
    elif isinstance(cutoff_freq, (list, tuple)):
        cutoff_key = tuple(map(float, cutoff_freq))
    else:
        cutoff_key = (float(cutoff_freq),)
    key = (int(fs), cutoff_key, int(order), str(btype))
    ba = _BUTTER_CACHE.get(key)
    if ba is None:
        wn = np.asarray(cutoff_key, dtype=np.float64) / (0.5 * fs)
        b, a = butter(N=order, Wn=wn, btype=btype, analog=False)
        _BUTTER_CACHE[key] = (b, a)
        return b, a
    return ba


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = _butter_cached(fs=fs, cutoff_freq=cutoff_freq, order=order, btype=btype)
    return filtfilt(b, a, eeg_data, axis=-1)




## === cell 6
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
).to(device)


@torch.no_grad()
def compute_spec_batch(chain_b: np.ndarray) -> np.ndarray:
    """
    Input: chain_b shape (B, 4, T) float
    Output: (B, 4, T_frames, 96) where 96 corresponds to freq bins [2:98].
    """
    x = torch.as_tensor(chain_b, dtype=torch.float32, device=device)  # (B,4,T)
    B = x.shape[0]
    x = x.reshape(B * 4, x.shape[-1])  # (B*4,T)
    x = spectrogram(x)  # (B*4, F, Frames) complex
    x = x[:, 2:98, :]  # (B*4, 96, Frames)
    x = torch.abs(x) / 15.0
    x = torch.log(x.clamp(min=math.exp(-4), max=math.exp(7)))
    x = x.transpose(1, 2).contiguous()  # (B*4, Frames, 96)
    x = x.reshape(B, 4, x.shape[1], 96)  # (B,4,Frames,96)
    return x.cpu().numpy()


@torch.no_grad()
def compute_spec(chain: np.ndarray) -> np.ndarray:
    """
    Input: chain shape (4, T) float
    Output: (4, T_frames, 96) where 96 corresponds to freq bins [2:98].
    """
    return compute_spec_batch(chain[None, ...])[0]


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


_EEG_COLS = [
    "Fp1",
    "Fp2",
    "Fz",
    "Cz",
    "Pz",
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
    "EKG",
]
_EEG_COLS_NO_EKG = _EEG_COLS[:-1]
_COL2IDX = {c: i for i, c in enumerate(_EEG_COLS)}


def compute_spec_chain_from_matrix(mat: np.ndarray) -> np.ndarray:
    Fp1 = mat[:, _COL2IDX["Fp1"]]
    Fp2 = mat[:, _COL2IDX["Fp2"]]
    F7 = mat[:, _COL2IDX["F7"]]
    F8 = mat[:, _COL2IDX["F8"]]
    T3 = mat[:, _COL2IDX["T3"]]
    T4 = mat[:, _COL2IDX["T4"]]
    T5 = mat[:, _COL2IDX["T5"]]
    T6 = mat[:, _COL2IDX["T6"]]
    O1 = mat[:, _COL2IDX["O1"]]
    O2 = mat[:, _COL2IDX["O2"]]
    F3 = mat[:, _COL2IDX["F3"]]
    F4 = mat[:, _COL2IDX["F4"]]
    C3 = mat[:, _COL2IDX["C3"]]
    C4 = mat[:, _COL2IDX["C4"]]
    P3 = mat[:, _COL2IDX["P3"]]
    P4 = mat[:, _COL2IDX["P4"]]

    ll = np.stack(
        [
            compute_spec_eeg(Fp1, F7),
            compute_spec_eeg(F7, T3),
            compute_spec_eeg(T3, T5),
            compute_spec_eeg(T5, O1),
        ]
    )
    lp = np.stack(
        [
            compute_spec_eeg(Fp1, F3),
            compute_spec_eeg(F3, C3),
            compute_spec_eeg(C3, P3),
            compute_spec_eeg(P3, O1),
        ]
    )
    rp = np.stack(
        [
            compute_spec_eeg(Fp2, F4),
            compute_spec_eeg(F4, C4),
            compute_spec_eeg(C4, P4),
            compute_spec_eeg(P4, O2),
        ]
    )
    rl = np.stack(
        [
            compute_spec_eeg(Fp2, F8),
            compute_spec_eeg(F8, T4),
            compute_spec_eeg(T4, T6),
            compute_spec_eeg(T6, O2),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]  # (4, T)

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)  # (4, Frames, 96)
    return chain


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    mat = df_eeg.select(_EEG_COLS_NO_EKG).to_numpy()
    full = np.zeros((mat.shape[0], len(_EEG_COLS)), dtype=mat.dtype)
    for i, c in enumerate(_EEG_COLS_NO_EKG):
        full[:, _COL2IDX[c]] = mat[:, i]
    return compute_spec_chain_from_matrix(full)


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 7
import albumentations as A

spec_resize = A.Resize(height=96, width=224)


def proc_eeg_spec(x: np.ndarray) -> np.ndarray:
    """
    x is (4, T_frames, 96). Convert to (96, T_frames, 4) for albumentations.
    Output: (4, 96, 224) float32.
    """
    x = x.copy()
    x[np.isnan(x) | np.isinf(x)] = 0.0

    if x.ndim != 3 or x.shape[0] != 4 or x.shape[2] != 96:
        raise ValueError(f"Unexpected eeg_spec shape {x.shape}, expected (4, T, 96)")

    x = x.transpose(2, 1, 0)  # (96, T, 4)
    x = spec_resize(image=x)["image"]  # (96, 224, 4)
    x = x.transpose(2, 0, 1)  # (4, 96, 224)

    x = x + 1.0
    return x.astype(np.float32)


def proc_eeg_spec_batch(x_b: np.ndarray) -> np.ndarray:
    """
    Input: (B,4,T_frames,96)
    Output: (B,4,96,224) float32
    """
    B = x_b.shape[0]
    out = np.empty((B, 4, 96, 224), dtype=np.float32)
    for i in range(B):
        out[i] = proc_eeg_spec(x_b[i])
    return out




## === cell 8
def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
):
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




## === cell 9
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain_from_matrix(
    mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Fp1 = mat[:, _COL2IDX["Fp1"]]
    Fp2 = mat[:, _COL2IDX["Fp2"]]
    Fz = mat[:, _COL2IDX["Fz"]]
    Cz = mat[:, _COL2IDX["Cz"]]
    Pz = mat[:, _COL2IDX["Pz"]]
    F3 = mat[:, _COL2IDX["F3"]]
    F4 = mat[:, _COL2IDX["F4"]]
    F7 = mat[:, _COL2IDX["F7"]]
    F8 = mat[:, _COL2IDX["F8"]]
    C3 = mat[:, _COL2IDX["C3"]]
    C4 = mat[:, _COL2IDX["C4"]]
    P3 = mat[:, _COL2IDX["P3"]]
    P4 = mat[:, _COL2IDX["P4"]]
    T3 = mat[:, _COL2IDX["T3"]]
    T4 = mat[:, _COL2IDX["T4"]]
    T5 = mat[:, _COL2IDX["T5"]]
    T6 = mat[:, _COL2IDX["T6"]]
    O1 = mat[:, _COL2IDX["O1"]]
    O2 = mat[:, _COL2IDX["O2"]]

    if mat.shape[1] > _COL2IDX["EKG"] and not np.all(mat[:, _COL2IDX["EKG"]] == 0):
        ekg = mat[:, _COL2IDX["EKG"]]
    else:
        ekg = O2

    ekg = butter_filter(ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass")
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1)
    ekg = ekg.reshape(1, -1)

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
    mid = np.stack([compute_eeg(Fz - Cz), compute_eeg(Cz - Pz)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain, mid, ekg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    cols = _EEG_COLS if "EKG" in df_eeg.columns else _EEG_COLS_NO_EKG
    mat = df_eeg.select(cols).to_numpy()
    full = np.zeros((mat.shape[0], len(_EEG_COLS)), dtype=mat.dtype)
    for i, c in enumerate(cols):
        full[:, _COL2IDX[c]] = mat[:, i]
    return compute_eeg_chain_from_matrix(full)


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 10
def _center_crop_or_pad_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Execution fix: ensure all modalities have consistent length before concat."""
    x = np.asarray(x)
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        start = (x.shape[-1] - target_len) // 2
        end = start + target_len
        return x[..., start:end]
    pad_total = target_len - x.shape[-1]
    pad_l = pad_total // 2
    pad_r = pad_total - pad_l
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (pad_l, pad_r)
    return np.pad(x, pad_width, mode="edge")


def proc_eeg(eeg, mid, ekg):
    """
    Keep identical features (16 eeg + 2 mid + 1 ekg) but enforce consistent *binned*
    time length before concatenation.
    """
    eeg = np.asarray(eeg, dtype=np.float32).copy()  # expected (4, 2500) or (16,2500)
    mid = np.asarray(mid, dtype=np.float32).copy()  # expected (2, 2500)
    ekg = np.asarray(ekg, dtype=np.float32).copy()  # expected (1, 2500)

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = (eeg / mad_std).clip(-10, 10)
    mid = (mid / mad_std).clip(-10, 10)
    ekg = ekg / (MAD(ekg, axis=-1).reshape(-1) + 1e-5)

    target_len = 2500
    eeg = _center_crop_or_pad_1d(eeg, target_len)
    mid = _center_crop_or_pad_1d(mid, target_len)
    ekg = _center_crop_or_pad_1d(ekg, target_len)

    if eeg.shape[0] == 4:
        eeg = np.repeat(eeg, repeats=4, axis=0)  # (16, 2500)
    elif eeg.shape[0] != 16:
        raise ValueError(f"Unexpected eeg channels {eeg.shape[0]} (expected 4 or 16)")

    x = np.concatenate([eeg, mid, ekg], axis=0)  # (19, 2500)
    if x.shape != (19, 2500):
        raise ValueError(
            f"Unexpected proc_eeg output shape {x.shape}, expected (19,2500)"
        )
    return x.astype(np.float32)




## === cell 11
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq


def _center_crop_or_pad_chain_2d(chain: np.ndarray, target_T: int) -> np.ndarray:
    """
    Bugfix: spec_chain time length varies slightly across EEG parquet files, which
    breaks np.stack batching. We deterministically center-crop/pad along time axis.
    Input: (4, T). Output: (4, target_T).
    """
    chain = np.asarray(chain)
    if chain.ndim != 2 or chain.shape[0] != 4:
        raise ValueError(f"Unexpected chain shape {chain.shape}, expected (4,T)")
    return _center_crop_or_pad_1d(chain, target_T)


SPEC_CHAIN_T = 10000


def _cpu_features_for_eeg_id(eeg_id: int):
    eeg_filepath = os.path.join(EEG_DIR, f"{int(eeg_id)}.parquet")

    table = pq.read_table(eeg_filepath, columns=_EEG_COLS)
    mat = table.to_pandas(self_destruct=True).to_numpy(dtype=np.float64, copy=False)
    if np.isnan(mat).any():
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    colpos = {c: i for i, c in enumerate(_EEG_COLS)}
    Fp1, Fp2 = mat[:, colpos["Fp1"]], mat[:, colpos["Fp2"]]
    Fz, Cz, Pz = mat[:, colpos["Fz"]], mat[:, colpos["Cz"]], mat[:, colpos["Pz"]]
    F3, F4 = mat[:, colpos["F3"]], mat[:, colpos["F4"]]
    F7, F8 = mat[:, colpos["F7"]], mat[:, colpos["F8"]]
    C3, C4 = mat[:, colpos["C3"]], mat[:, colpos["C4"]]
    P3, P4 = mat[:, colpos["P3"]], mat[:, colpos["P4"]]
    T3, T4 = mat[:, colpos["T3"]], mat[:, colpos["T4"]]
    T5, T6 = mat[:, colpos["T5"]], mat[:, colpos["T6"]]
    O1, O2 = mat[:, colpos["O1"]], mat[:, colpos["O2"]]
    EKG = mat[:, colpos["EKG"]]

    diffs = np.stack(
        [
            Fp1 - F7,
            F7 - T3,
            T3 - T5,
            T5 - O1,
            Fp1 - F3,
            F3 - C3,
            C3 - P3,
            P3 - O1,
            Fp2 - F4,
            F4 - C4,
            C4 - P4,
            P4 - O2,
            Fp2 - F8,
            F8 - T4,
            T4 - T6,
            T6 - O2,
        ],
        axis=0,
    )  # (16, T)

    b_eeg, a_eeg = _butter_cached(
        fs=200, cutoff_freq=np.array([0.25, 50.0]), order=4, btype="bandpass"
    )
    diffs_eeg = filtfilt(b_eeg, a_eeg, diffs, axis=-1)
    diffs_eeg = bin_array(diffs_eeg, bin_size=4, mode="reflect").mean(axis=-1)

    ll = diffs_eeg[0:4]
    lp = diffs_eeg[4:8]
    rp = diffs_eeg[8:12]
    rl = diffs_eeg[12:16]
    eeg = np.stack([ll, lp, rp, rl])[:, 0]  # (4, 2500-ish)

    mid_raw = np.stack([Fz - Cz, Cz - Pz], axis=0)
    mid_raw = filtfilt(b_eeg, a_eeg, mid_raw, axis=-1)
    mid = bin_array(mid_raw, bin_size=4, mode="reflect").mean(axis=-1)

    ekg_src = EKG if (EKG.size > 0 and not np.all(EKG == 0)) else O2
    b_ekg, a_ekg = _butter_cached(
        fs=200, cutoff_freq=np.array([0.50, 20.0]), order=4, btype="bandpass"
    )
    ekg = filtfilt(b_ekg, a_ekg, ekg_src, axis=-1)
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1).reshape(1, -1)

    b_sp, a_sp = _butter_cached(
        fs=200, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )
    diffs_spec = filtfilt(b_sp, a_sp, diffs, axis=-1)
    ll_s = diffs_spec[0:4]
    lp_s = diffs_spec[4:8]
    rp_s = diffs_spec[8:12]
    rl_s = diffs_spec[12:16]
    chain = np.stack([ll_s, lp_s, rp_s, rl_s])[:, 0]  # (4, T_raw)

    chain = _center_crop_or_pad_chain_2d(chain, SPEC_CHAIN_T)

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    return (
        int(eeg_id),
        chain.astype(np.float32, copy=False),
        eeg.astype(np.float32, copy=False),
        mid.astype(np.float32, copy=False),
        ekg.astype(np.float32, copy=False),
    )


eeg_ids = df_test["eeg_id"].to_numpy()
preds_final = np.zeros((eeg_ids.shape[0], len(LABELS)), dtype=np.float64)
id_to_row = {int(eid): i for i, eid in enumerate(eeg_ids.tolist())}

max_workers = min(8, max(1, (os.cpu_count() or 2)))
chunksize = 64
BATCH_GPU = 64

pending_ids = []
pending_spec_chains = []
pending_eeg = []
pending_mid = []
pending_ekg = []


def _flush_batch():
    if not pending_ids:
        return

    spec_b = np.stack(pending_spec_chains, axis=0)  # (B,4,T) now fixed T
    eeg_spec_b = compute_spec_batch(spec_b)  # (B,4,Frames,96)
    eeg_spec_b = proc_eeg_spec_batch(eeg_spec_b)  # (B,4,96,224)
    eeg_spec_t = torch.as_tensor(eeg_spec_b, dtype=torch.float32, device=device)

    eeg_feats = [
        proc_eeg(e, m, k) for e, m, k in zip(pending_eeg, pending_mid, pending_ekg)
    ]
    eeg_t = torch.as_tensor(
        np.stack(eeg_feats, axis=0), dtype=torch.float32, device=device
    )

    with torch.no_grad():
        preds_accum = []
        for model in models_eeg:
            preds_accum.append(model(eeg_t).exp().detach().cpu().numpy())
        for model in models_eeg_spc:
            preds_accum.append(model(eeg_spec_t).exp().detach().cpu().numpy())

    pred = np.mean(np.stack(preds_accum, axis=0), axis=0)
    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=1, keepdims=True)

    UNIFORM = np.full((1, len(LABELS)), 1.0 / len(LABELS), dtype=np.float64)
    SMOOTH_ALPHA = 0.10  # reduced from 0.70 to better match constant prior predictions
    pred = (1.0 - SMOOTH_ALPHA) * pred + SMOOTH_ALPHA * UNIFORM
    pred = np.clip(pred, 1e-12, None)
    pred = pred / pred.sum(axis=1, keepdims=True)

    for k, eeg_id_i in enumerate(pending_ids):
        preds_final[id_to_row[eeg_id_i]] = pred[k].astype(np.float64, copy=False)

    pending_ids.clear()
    pending_spec_chains.clear()
    pending_eeg.clear()
    pending_mid.clear()
    pending_ekg.clear()


with ThreadPoolExecutor(max_workers=max_workers) as ex:
    it = ex.map(_cpu_features_for_eeg_id, eeg_ids.tolist(), chunksize=chunksize)
    for eeg_id_i, spec_chain, eeg_chain, mid, ekg in tqdm(it, total=len(eeg_ids)):
        pending_ids.append(eeg_id_i)
        pending_spec_chains.append(spec_chain)
        pending_eeg.append(eeg_chain)
        pending_mid.append(mid)
        pending_ekg.append(ekg)

        if len(pending_ids) >= BATCH_GPU:
            _flush_batch()

_flush_batch()

preds_final = np.clip(preds_final, 1e-12, None)
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)

preds_final.shape, preds_final.sum(axis=1).min(), preds_final.sum(axis=1).max()



## === cell 12
df_sub = pd.DataFrame({"eeg_id": eeg_ids.tolist()})
for j, col in enumerate(LABELS):
    df_sub[col] = preds_final[:, j]

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

assert list(df_sub.columns) == ["eeg_id"] + LABELS

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote", out_path, "shape=", df_sub.shape)
print(
    "Row sum check:", df_sub[LABELS].sum(axis=1).min(), df_sub[LABELS].sum(axis=1).max()
)
df_sub.head()
