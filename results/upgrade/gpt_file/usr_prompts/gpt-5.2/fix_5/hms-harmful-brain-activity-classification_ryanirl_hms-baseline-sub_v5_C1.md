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

import torch
import torch.nn as nn

from typing import List



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



## === cell 3
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4):
    b, a = butter(
        N=order,
        Wn=cutoff_freq / (0.5 * fs),
        btype="low",
        analog=False,
    )
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg)
    eeg = eeg[::2]  # 200Hz -> 100Hz
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

    ll_segs = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ],
        axis=0,
    )  # [4, T]
    lp_segs = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ],
        axis=0,
    )
    rp_segs = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ],
        axis=0,
    )
    rl_segs = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ],
        axis=0,
    )

    ll = ll_segs.mean(axis=0)
    lp = lp_segs.mean(axis=0)
    rp = rp_segs.mean(axis=0)
    rl = rl_segs.mean(axis=0)

    chain = np.stack([ll, lp, rp, rl], axis=0)  # [4, T]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




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


def train_one_model(
    model: nn.Module,
    df_train: pl.DataFrame,
    max_steps: int = 300,
    batch_size: int = 16,
    lr: float = 2e-3,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    kld = nn.KLDivLoss(reduction="batchmean")

    df_u = df_train.unique(subset=["eeg_id"], keep="first")

    n = df_u.height
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)

    usable = []
    for i in idx:
        row = df_u.row(int(i), named=True)
        eeg_id = int(row["eeg_id"])
        fp = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
        if os.path.exists(fp):
            usable.append(int(i))
        if len(usable) >= max_steps * batch_size:
            break

    if len(usable) == 0:
        print("No usable training EEG files found; skipping training.")
        model.eval()
        return

    ptr = 0
    pbar = tqdm(range(max_steps), desc="train_steps", leave=False)
    for step in pbar:
        xb = []
        yb = []
        for _ in range(batch_size):
            if ptr >= len(usable):
                ptr = 0
            ri = usable[ptr]
            ptr += 1

            row = df_u.row(int(ri), named=True)
            eeg_id = int(row["eeg_id"])
            fp = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
            try:
                x_np = compute_eeg_from_file(fp)
                if not (
                    isinstance(x_np, np.ndarray)
                    and x_np.ndim == 2
                    and x_np.shape[0] == 4
                ):
                    continue
                x = _preprocess_chain_to_tensor(x_np, device=device)  # [1,4,T]
                xb.append(x)
                yb.append(_row_to_target_probs(row))
            except Exception:
                continue

        if len(xb) == 0:
            continue

        x = torch.cat(xb, dim=0)  # [B,4,T]
        y = torch.as_tensor(
            np.stack(yb, axis=0), dtype=torch.float32, device=device
        )  # [B,6]

        opt.zero_grad(set_to_none=True)
        p = model(x)  # [B,6] probs
        loss = kld(torch.log(p), y)
        loss.backward()
        opt.step()

        pbar.set_postfix({"loss": float(loss.detach().cpu().item())})

    model.eval()


train_one_model(models[0], df_train, max_steps=300, batch_size=16, lr=2e-3)



## === cell 8
eeg_ids = df_test["eeg_id"].to_list()

preds_final = np.zeros((len(eeg_ids), 6), dtype=np.float32)
for i, eeg_id in enumerate(tqdm(eeg_ids, total=len(eeg_ids))):
    preds_final[i] = gen_ensemble_pred(models, eeg_id)

print(
    "preds_final shape:",
    preds_final.shape,
    "row sums:",
    preds_final.sum(axis=1).min(),
    preds_final.sum(axis=1).max(),
)



## === cell 9
df_sub = pd.DataFrame({"eeg_id": eeg_ids})
df_sub[LABELS] = preds_final

vals = df_sub[LABELS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)
df_sub[LABELS] = vals

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
df_sub.head()
