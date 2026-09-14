# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os
import glob

import torch
import torch.nn as nn

from typing import List



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/data/hms-harmful-brain-activity-classification/"
    if os.path.exists(alt):
        DATA_DIR = alt
    else:
        alt2 = "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/"
        if os.path.exists(alt2):
            DATA_DIR = alt2

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
import timm
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


class NewModel(nn.Module):
    def __init__(self, pretrained: bool = True):
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
        self.fc = nn.Sequential(
            nn.Linear(384 * 4, 6),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x0 = x[:, 0:1].repeat(1, 4, 1, 1)
        x1 = x[:, 1:2].repeat(1, 4, 1, 1)
        x2 = x[:, 2:3].repeat(1, 4, 1, 1)
        x3 = x[:, 3:4].repeat(1, 4, 1, 1)

        f0 = self.pool_0(self.m0(x0)[-1])
        f1 = self.pool_0(self.m0(x1)[-1])
        f2 = self.pool_0(self.m0(x2)[-1])
        f3 = self.pool_0(self.m0(x3)[-1])

        embed = torch.concat([f0, f1, f2, f3], dim=1)
        out = self.fc(embed)
        out = out + 0.001
        out = out / out.sum(dim=1, keepdim=True)
        return out, embed

    @torch.no_grad()
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        pred, _ = self.forward(x)
        return pred.detach().cpu().numpy().reshape(-1)




## === cell 4
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model




## === cell 5
class EegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        z = self.conv(x)
        z = self.pool(z)
        logits = self.head(z)
        return logits




## === cell 6
ensembles = []

fold_dirs = [
    "/kaggle/input/hms-models/eeg_cnn_rnn_stage_3/*",
]

models: List[nn.Module] = []
found_any_ckpt = False
for fold_dir in fold_dirs:
    for fold_path in glob.glob(fold_dir):
        model_path = os.path.join(fold_path, "model_best_test.pt")
        if os.path.exists(model_path):
            found_any_ckpt = True
            m = EegModel()
            m = load_model(model_path, m)
            m = m.to(device)
            models.append(m)

if not models:
    models = [EegModel().to(device)]

for m in models:
    m.eval()

spec_model = NewModel(pretrained=True).to(device).eval()

print(
    "Loaded EEG models:",
    len(models),
    "| found_any_ckpt:",
    found_any_ckpt,
    "| spec_model: yes",
)



## === cell 7
import librosa
from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    chain = chain.astype(np.float32)
    m = np.nanmean(chain)
    s = np.nanstd(chain) + 1e-6
    chain = (chain - m) / s
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
    spectrogram = librosa.power_to_db((np.abs(spectrogram) ** 2), ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    sp = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    sp = sp + compute_spec(eeg2)
    return sp / 2


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

    chain = np.stack([ll, lp, rp, rl], axis=0).mean(axis=1)  # (4, F, T)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)
    return chain




## === cell 8
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs: int = 200, cutoff_freq: float = 22, order: int = 4):
    b, a = butter(
        N=order,
        Wn=cutoff_freq / (0.5 * fs),
        btype="low",
        analog=False,
    )
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg)
    eeg = eeg[::2]
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

    chain = np.stack([ll, lp, rp, rl], axis=0).mean(axis=1)  # (4, T)
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    chain = np.nan_to_num(chain, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
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

train_votes = df_train.select(LABELS).to_numpy().astype(np.float64)
train_votes_sum = train_votes.sum(axis=1, keepdims=True)
train_votes_sum[train_votes_sum == 0] = 1.0
train_probs = train_votes / train_votes_sum
prior = train_probs.mean(axis=0)
prior = np.clip(prior, 1e-7, 1.0)
prior = (prior / prior.sum()).astype(np.float32)
PRIOR_TORCH = torch.tensor(prior, device=device, dtype=torch.float32).view(1, -1)

TEMP = 2.10
ALPHA = 0.48
UNIFORM_MIX = 2.0e-2




## === cell 10
def resolve_eeg_path(eeg_id: int) -> str:
    candidate_paths = [
        os.path.join(EEG_DIR, f"{eeg_id}.parquet"),
        os.path.join(
            "/kaggle/data/hms-harmful-brain-activity-classification/test_eegs/",
            f"{eeg_id}.parquet",
        ),
        os.path.join(
            "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/test_eegs/",
            f"{eeg_id}.parquet",
        ),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"EEG parquet not found for eeg_id={eeg_id}. Tried: {candidate_paths}"
    )




## === cell 11
@torch.no_grad()
def gen_ensemble_pred(models: List[nn.Module], df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = df_row["eeg_id"].item()
    filepath = resolve_eeg_path(eeg_id)

    x0 = compute_eeg_from_file(filepath)  # (4, T)
    x1 = x0[:, ::-1].copy()

    def _prep_eeg(x_np: np.ndarray) -> torch.Tensor:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / (torch.std(x, dim=-1, keepdim=True) + 1e-5)
        return x.unsqueeze(0)  # (1, 4, T)

    xb = [_prep_eeg(x0), _prep_eeg(x1)]

    eeg_preds = []
    for model in models:
        probs_tta = []
        for x in xb:
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]

            logits = out.float()
            logits = logits - logits.mean(dim=1, keepdim=True)
            logits = logits / TEMP
            prob = torch.softmax(logits, dim=1)

            prob = (1.0 - ALPHA) * prob + ALPHA * PRIOR_TORCH
            probs_tta.append(prob)

        prob = torch.mean(torch.stack(probs_tta, dim=0), dim=0)  # average TTA
        pred = prob.detach().cpu().numpy().reshape(-1)
        eeg_preds.append(pred)

    eeg_preds = np.mean(np.stack(eeg_preds, axis=0), axis=0)
    eeg_preds = np.nan_to_num(eeg_preds, nan=0.0, posinf=0.0, neginf=0.0)

    sp = compute_spec_from_file(filepath)  # (4, F, T)
    sp_t = torch.tensor(sp, dtype=torch.float32, device=device).unsqueeze(
        0
    )  # (1,4,F,T)
    sp_rev = torch.flip(sp_t, dims=[-1])  # time-reversal TTA in spec-time

    probs_spec = []
    for sp_in in (sp_t, sp_rev):
        out_spec = spec_model(sp_in)
        if isinstance(out_spec, (tuple, list)):
            out_spec = out_spec[0]
        probs_spec.append(out_spec.float())
    spec_prob = (
        torch.mean(torch.stack(probs_spec, dim=0), dim=0)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    spec_prob = np.nan_to_num(spec_prob, nan=0.0, posinf=0.0, neginf=0.0)

    preds = 0.5 * eeg_preds + 0.5 * spec_prob

    preds = (1.0 - UNIFORM_MIX) * preds + (UNIFORM_MIX / 6.0)

    preds = np.clip(preds, 1e-7, 1.0)
    s = preds.sum()
    if not np.isfinite(s) or s <= 0 or preds.shape != (6,):
        preds = prior.astype(np.float32)
        s = preds.sum()
    preds = preds / s
    return preds.astype(np.float32)




## === cell 12
preds_final: List[np.ndarray] = []
for i in tqdm(range(len(df_test))):
    pred = gen_ensemble_pred(models, df_test[i])
    preds_final.append(pred)

preds_final = np.stack(preds_final, axis=0)  # (n_test, 6)
print(
    "preds_final shape:",
    preds_final.shape,
    "row_sums min/max:",
    preds_final.sum(1).min(),
    preds_final.sum(1).max(),
)



## === cell 13
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final

p = df_sub[LABELS].to_numpy(dtype=np.float64)
p = np.clip(p, 1e-12, 1.0)
p = p / p.sum(axis=1, keepdims=True)
df_sub[LABELS] = p.astype(np.float32)

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
