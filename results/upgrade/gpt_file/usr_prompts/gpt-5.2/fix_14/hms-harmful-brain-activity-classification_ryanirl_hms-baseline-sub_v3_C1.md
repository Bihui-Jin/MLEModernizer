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

0.4794500545624416

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os

import torch
import torch.nn as nn

from typing import Optional
from collections import OrderedDict



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")
TRAIN_EEG_DIR = os.path.join(DATA_DIR, "train_eegs/")

DATA_DIR = os.path.normpath(DATA_DIR)
SPEC_DIR = os.path.normpath(SPEC_DIR)
EEG_DIR = os.path.normpath(EEG_DIR)
TRAIN_EEG_DIR = os.path.normpath(TRAIN_EEG_DIR)

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(df_test.head())



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 3
import timm
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


class NewModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.m0 = timm.create_model(
            "fastvit_t8.apple_in1k",
            pretrained=pretrained,
            num_classes=6,
            in_chans=4,
            features_only=True,
        )
        self.pool_0 = SelectAdaptivePool2d(
            pool_type="avg", flatten=True, input_fmt="NCHW"
        )
        self.fc = nn.Sequential(nn.Linear(384 * 4, 6), nn.Sigmoid())

    def forward(self, x):
        x0 = self.pool_0(self.m0(x[:, 0])[-1])
        x1 = self.pool_0(self.m0(x[:, 1])[-1])
        x2 = self.pool_0(self.m0(x[:, 2])[-1])
        x3 = self.pool_0(self.m0(x[:, 3])[-1])

        embed = torch.concat([x0, x1, x2, x3], dim=1)
        out = self.fc(embed)
        out = out + 0.001
        out = out / out.sum(axis=1).unsqueeze(1)

        return out, embed

    @torch.no_grad()
    def predict(self, x):
        x = torch.Tensor(x).unsqueeze(0).to(device)
        pred, _ = self.forward(x)
        pred = pred.detach().cpu().numpy()
        return pred.reshape(-1)




## === cell 4
def load_model(path: str, model: nn.Module) -> nn.Module:
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    )
    return model




## === cell 5
class EegModel(nn.Module):
    def __init__(self, n_classes: int = 6, in_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)  # [B, 64]
        return self.head(z)  # [B, 6]


models = [EegModel().to(device)]
for m in models:
    m.eval()

print("n_models:", len(models))



## === cell 6
import librosa
from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    chain = chain.astype(np.float32, copy=False)
    chain = np.nan_to_num(chain, nan=0.0, posinf=0.0, neginf=0.0)
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
    spec_ = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    spec_ = spec_ + compute_spec(eeg2)
    return spec_ / 2


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

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)
    return chain




## === cell 7
from scipy.signal import butter, filtfilt

_FS = 200
_CUTOFF_FREQ = 22
_ORDER = 4
_B_BUTTER, _A_BUTTER = butter(
    N=_ORDER, Wn=_CUTOFF_FREQ / (0.5 * _FS), btype="low", analog=False
)


def _fast_lowpass_approx_1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1:
        return x
    x = x.astype(np.float32, copy=False)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4):
    return filtfilt(_B_BUTTER, _A_BUTTER, eeg_data)


def compute_eeg(eeg: np.ndarray, *, fast: bool = False) -> np.ndarray:
    if fast:
        eeg = _fast_lowpass_approx_1d(eeg, k=9)
    else:
        eeg = butter_filter(eeg)
    eeg = eeg[::2]
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame, *, fast: bool = False) -> np.ndarray:
    """
    Keep shape [4, T] for Conv1d(in_ch=4).
    """
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
            compute_eeg(Fp1 - F7, fast=fast),
            compute_eeg(F7 - T3, fast=fast),
            compute_eeg(T3 - T5, fast=fast),
            compute_eeg(T5 - O1, fast=fast),
        ],
        axis=0,
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3, fast=fast),
            compute_eeg(F3 - C3, fast=fast),
            compute_eeg(C3 - P3, fast=fast),
            compute_eeg(P3 - O1, fast=fast),
        ],
        axis=0,
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4, fast=fast),
            compute_eeg(F4 - C4, fast=fast),
            compute_eeg(C4 - P4, fast=fast),
            compute_eeg(P4 - O2, fast=fast),
        ],
        axis=0,
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8, fast=fast),
            compute_eeg(F8 - T4, fast=fast),
            compute_eeg(T4 - T6, fast=fast),
            compute_eeg(T6 - O2, fast=fast),
        ],
        axis=0,
    )

    chain = np.stack([ll, lp, rp, rl], axis=0)[:, 0]  # [4, T]
    chain = chain.astype(np.float32, copy=False)
    chain = np.nan_to_num(chain, nan=0.0, posinf=0.0, neginf=0.0)
    return chain


def compute_eeg_from_file(filepath: str, *, fast: bool = False) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg, fast=fast)
    return chain




## === cell 8
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

_TARGET_EPS = 1e-3
_TARGET_T = 8192  # deterministic fixed length; keeps Conv1d semantics and avoids variable-length batching.

_FAST_EEG_PREPROC = True


def _fix_length_1d_torch(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    x: [C, T] or [1, C, T]
    Returns same rank tensor with T==target_t via center-crop or symmetric zero-pad.
    """
    if x.dim() == 2:
        C, T = x.shape
        if T == target_t:
            return x
        if T > target_t:
            start = (T - target_t) // 2
            return x[:, start : start + target_t]
        pad_total = target_t - T
        left = pad_total // 2
        right = pad_total - left
        return torch.nn.functional.pad(x, (left, right), mode="constant", value=0.0)

    if x.dim() == 3:
        B, C, T = x.shape
        if T == target_t:
            return x
        if T > target_t:
            start = (T - target_t) // 2
            return x[:, :, start : start + target_t]
        pad_total = target_t - T
        left = pad_total // 2
        right = pad_total - left
        return torch.nn.functional.pad(x, (left, right), mode="constant", value=0.0)

    raise ValueError(f"Unexpected tensor rank: {x.dim()}")


def _votes_to_prob(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(6, 1.0 / 6.0, dtype=np.float64)
    p = v / s
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    p = p + _TARGET_EPS
    p = p / p.sum()
    return p.astype(np.float64)


def _smooth_pred_prob(p: np.ndarray, eps: float = _TARGET_EPS) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum()
    p = p + eps
    p = p / p.sum()
    return p


def build_train_index(df_train: pl.DataFrame, max_rows: int = 1500) -> pd.DataFrame:
    dfp = df_train.select(["eeg_id"] + LABELS).to_pandas()
    dfp = dfp.dropna()
    dfp = dfp.sample(n=min(max_rows, len(dfp)), random_state=42).reset_index(drop=True)
    return dfp


_EEG_CACHE: "OrderedDict[tuple, Optional[torch.Tensor]]" = OrderedDict()
_EEG_CACHE_MAX = 256

_TEST_EEG_CACHE: "OrderedDict[tuple, Optional[torch.Tensor]]" = OrderedDict()
_TEST_EEG_CACHE_MAX = 2048

_MISS = object()


def _cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return _MISS


def _cache_set(cache: OrderedDict, cache_max: int, key, value):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > cache_max:
        cache.popitem(last=False)


@torch.no_grad()
def _prepare_x_from_eeg_id(eeg_id: int, base_dir: str) -> Optional[torch.Tensor]:
    base_dir = os.path.normpath(base_dir)
    is_test = os.path.basename(base_dir) == "test_eegs"
    cache = _TEST_EEG_CACHE if is_test else _EEG_CACHE
    cache_max = _TEST_EEG_CACHE_MAX if is_test else _EEG_CACHE_MAX

    key = (int(eeg_id), base_dir, bool(_FAST_EEG_PREPROC))
    cached = _cache_get(cache, key)
    if cached is not _MISS:
        return cached

    filepath = os.path.join(base_dir, f"{int(eeg_id)}.parquet")
    if not os.path.exists(filepath):
        _cache_set(cache, cache_max, key, None)
        return None
    try:
        x = compute_eeg_from_file(filepath, fast=_FAST_EEG_PREPROC)  # [4,T]
        x = torch.tensor(x, device=device, dtype=torch.float32)
        x = torch.diff(x, dim=-1)  # [4,T-1]
        x = x / (torch.std(x, dim=-1, keepdims=True) + 1e-5)
        x = _fix_length_1d_torch(x, _TARGET_T)  # [4,target_T]
        x = x.unsqueeze(0)  # [1,4,target_T]
        _cache_set(cache, cache_max, key, x)
        return x
    except Exception:
        _cache_set(cache, cache_max, key, None)
        return None


def _precompute_train_cache(df_idx: pd.DataFrame):
    uniq_ids = pd.unique(df_idx["eeg_id"].astype(np.int64))
    for eeg_id in tqdm(uniq_ids, desc="cache_train_eeg", leave=False):
        _ = _prepare_x_from_eeg_id(int(eeg_id), TRAIN_EEG_DIR)


def train_one_model(
    model: nn.Module, df_idx: pd.DataFrame, steps: int = 350, lr: float = 2e-3
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    n = len(df_idx)
    if n == 0:
        model.eval()
        return

    for t in tqdm(range(steps), desc="train", leave=False):
        r = df_idx.iloc[t % n]
        eeg_id = int(r["eeg_id"])
        x = _prepare_x_from_eeg_id(eeg_id, TRAIN_EEG_DIR)
        if x is None:
            continue

        y = _votes_to_prob(r[LABELS].to_numpy())
        y = torch.tensor(y, device=device, dtype=torch.float32).unsqueeze(0)  # [1,6]

        opt.zero_grad(set_to_none=True)
        logits = model(x)  # [1,6]
        log_probs = torch.log_softmax(logits, dim=1)
        loss = criterion(log_probs, y)
        loss.backward()
        opt.step()

    model.eval()


train_idx = build_train_index(df_train, max_rows=1500)
_precompute_train_cache(train_idx)
for m in models:
    train_one_model(m, train_idx, steps=350, lr=2e-3)




## === cell 9
@torch.no_grad()
def gen_ensemble_pred(models, eeg_id: int) -> np.ndarray:
    """
    Ensure output is valid probabilities for KL-div evaluation.
    """
    x = _prepare_x_from_eeg_id(eeg_id, EEG_DIR)
    if x is None:
        return np.full(6, 1.0 / 6.0, dtype=np.float64)

    preds = []
    for model in models:
        logits = model(x)  # [B,6]
        probs = torch.softmax(logits.detach().float(), dim=1)
        probs = probs.cpu().numpy().reshape(-1)
        if probs.size != 6 or (not np.all(np.isfinite(probs))):
            probs = np.full(6, 1.0 / 6.0, dtype=np.float64)
        preds.append(probs)

    preds = np.mean(preds, axis=0).astype(np.float64)
    preds = _smooth_pred_prob(preds, eps=_TARGET_EPS)
    return preds




## === cell 10
@torch.no_grad()
def gen_ensemble_pred_batch(models, xs: torch.Tensor) -> np.ndarray:
    xs = _fix_length_1d_torch(xs, _TARGET_T)

    preds = []
    for model in models:
        logits = model(xs)  # [B,6]
        probs = torch.softmax(logits.detach().float(), dim=1)  # [B,6]
        preds.append(probs)
    probs = torch.stack(preds, dim=0).mean(dim=0)  # [B,6]
    probs = probs.clamp(1e-6, 1.0)
    probs = probs / probs.sum(dim=1, keepdim=True)

    probs = probs + _TARGET_EPS
    probs = probs / probs.sum(dim=1, keepdim=True)

    return probs.detach().cpu().numpy().astype(np.float64)


from concurrent.futures import ThreadPoolExecutor

preds_final = []
test_eeg_ids = df_test["eeg_id"].to_list()

BATCH_SIZE = 64
PREFETCH = 256
N_WORKERS = min(8, (os.cpu_count() or 4))


def _load_one(eid: int):
    return eid, _prepare_x_from_eeg_id(int(eid), EEG_DIR)


batch_x = []
n_total = len(test_eeg_ids)

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    for i in tqdm(range(0, n_total, PREFETCH), desc="infer_prefetch"):
        chunk = test_eeg_ids[i : i + PREFETCH]
        for eid, x in ex.map(_load_one, chunk):
            if x is None:
                preds_final.append(np.full(6, 1.0 / 6.0, dtype=np.float64))
                continue

            batch_x.append(x)  # [1,4,T_fixed]
            if len(batch_x) >= BATCH_SIZE:
                xs = torch.cat(batch_x, dim=0)  # [B,4,T_fixed]
                probs = gen_ensemble_pred_batch(models, xs)  # [B,6]
                preds_final.extend([p for p in probs])
                batch_x = []

if len(batch_x) > 0:
    xs = torch.cat(batch_x, dim=0)
    probs = gen_ensemble_pred_batch(models, xs)
    preds_final.extend([p for p in probs])

preds_final = np.asarray(preds_final, dtype=np.float64)

if preds_final.shape[0] != len(test_eeg_ids):
    preds_final = np.asarray(
        [
            gen_ensemble_pred(models, int(eid))
            for eid in tqdm(test_eeg_ids, desc="infer_fallback")
        ],
        dtype=np.float64,
    )

preds_final = np.nan_to_num(
    preds_final, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0
)
preds_final = np.clip(preds_final, 1e-6, 1.0)
row_sums = preds_final.sum(axis=1, keepdims=True)
bad = ~np.isfinite(row_sums) | (row_sums <= 0)
if np.any(bad):
    preds_final[bad.reshape(-1)] = 1.0 / 6.0
    row_sums = preds_final.sum(axis=1, keepdims=True)
preds_final = preds_final / row_sums

preds_final = preds_final + _TARGET_EPS
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)



## === cell 11
df_pred = pd.DataFrame({"eeg_id": test_eeg_ids})
df_pred[LABELS] = preds_final

df_sub = sample_submission.to_pandas()[["eeg_id"]].merge(
    df_pred, on="eeg_id", how="left"
)

for c in LABELS:
    if df_sub[c].isna().any():
        df_sub[c] = df_sub[c].fillna(1.0 / 6.0)

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.nan_to_num(probs, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
probs = np.clip(probs, 1e-6, 1.0)
row_sums = probs.sum(axis=1, keepdims=True)
bad = ~np.isfinite(row_sums) | (row_sums <= 0)
if np.any(bad):
    probs[bad.reshape(-1)] = 1.0 / 6.0
    row_sums = probs.sum(axis=1, keepdims=True)
probs = probs / row_sums

probs = probs + _TARGET_EPS
probs = probs / probs.sum(axis=1, keepdims=True)

df_sub[LABELS] = probs

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape=", df_sub.shape)
print(df_sub.head())
print("Row-sum check (min/max):", probs.sum(axis=1).min(), probs.sum(axis=1).max())
