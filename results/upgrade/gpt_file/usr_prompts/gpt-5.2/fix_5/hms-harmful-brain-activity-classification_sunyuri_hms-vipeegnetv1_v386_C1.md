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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import gc
import random
import warnings

warnings.filterwarnings("ignore")

NEEDTRAIN = True  # train the model (we enable on Kaggle to avoid "no weights found" uniform fallback)
LOAD_MODELS_FROM = "models"  # kept for compatibility; we write/read local models/

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = True
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    PLATFORM = "unknown"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img ***
print("DATATYPE:", DATATYPE, "PLATFORM:", PLATFORM)

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 3
SPLITS = 3

TEST_BATCHSIZE = 64

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "device:", device)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
_ss_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
_ss = pd.read_csv(_ss_path, nrows=5)
TARGETS = list(_ss.columns[1:])  # exclude eeg_id
print("Train shape:", df.shape)
print("Targets:", TARGETS)

TARGETS_RAW = [t + "_raw" for t in TARGETS]

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")




## === cell 1
def make_train_table(df_in: pd.DataFrame) -> pd.DataFrame:
    train = df_in.drop_duplicates(
        [
            "eeg_id",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
    ).reset_index(drop=True)
    train["sign_id"] = train.index.values
    df_in = df_in.copy()
    df_in["sign_id"] = df_in.index.values

    y_data = train[TARGETS].values.astype(np.float32)
    train[TARGETS_RAW] = y_data
    y_prob = y_data / np.clip(y_data.sum(axis=1, keepdims=True), 1e-6, None)
    train[TARGETS] = y_prob
    return train


train = make_train_table(df)
print("Consolidated train:", train.shape)


def _fix_length_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Bugfix: some parquet EEGs are not exactly 50s long, causing variable tensor widths after pooling
    and DataLoader stack() failures. Enforce a deterministic fixed window by center-cropping or
    padding with edge values (stable, minimal distortion).
    """
    cur = x.shape[-1]
    if cur == target_len:
        return x
    if cur > target_len:
        start = (cur - target_len) // 2
        return x[..., start : start + target_len]
    pad_total = target_len - cur
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    if cur == 0:
        return np.zeros((x.shape[0], target_len), dtype=x.dtype)
    return np.pad(x, ((0, 0), (pad_left, pad_right)), mode="edge")


def read_eeg_parquet(eeg_id: int, base_path: str) -> np.ndarray:
    """Return (16, 10000) float32 EEG after montage + filter + clipping."""
    p = os.path.join(base_path, f"{int(eeg_id)}.parquet")
    eeg_default = pd.read_parquet(p)

    eeg = []
    for channel in BRAIN:
        a0, a1 = channel.split("-")
        eeg_temp = (eeg_default.loc[:, a0] - eeg_default.loc[:, a1]).values
        eeg_temp = np.nan_to_num(eeg_temp, nan=0.0).astype(np.float32)
        eeg.append(eeg_temp[None, :])
    eeg = np.concatenate(eeg, axis=0)  # (16, T)

    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1).astype(np.float32)

    fixed_len = int(RSFREQ * EEG_LENGTH_USED)
    eeg = _fix_length_1d(eeg.astype(np.float32), fixed_len).astype(np.float32)

    eeg = np.clip(eeg, a_min=-1024, a_max=1024).astype(np.float32)
    if filter_range is not None:
        eeg = signal.filtfilt(b, a, eeg, axis=1).astype(np.float32)
    eeg = np.clip(eeg, a_min=-255, a_max=255).astype(np.float32)

    eeg = (eeg + 255.0) / 2.0
    return eeg.astype(np.float32)




## === cell 2
class EEGDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, eeg_dir: str, mode: str = "train"):
        self.df = meta_df.reset_index(drop=True)
        self.eeg_dir = eeg_dir
        self.mode = mode
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _get_eeg(self, eeg_id: int) -> np.ndarray:
        if eeg_id not in self._cache:
            self._cache[eeg_id] = read_eeg_parquet(eeg_id, self.eeg_dir)
        return self._cache[eeg_id]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg_id = int(row.eeg_id)
        x = self._get_eeg(eeg_id)  # (16, 10000)

        x = np.concatenate([x[0:8], x[-8:]], axis=0)  # (16, 10000)

        x_t = torch.from_numpy(x).float().unsqueeze(0)  # (1,16,10000)
        x_t = F.avg_pool1d(x_t, kernel_size=10, stride=10)  # (1,16,1000)

        if self.mode != "test":
            y = torch.tensor(row[TARGETS].values.astype(np.float32))
            sw = float(np.sum(row[TARGETS_RAW].values) / 20.0)
            return x_t, y, torch.tensor([sw], dtype=torch.float32)
        else:
            return (
                x_t,
                torch.zeros(len(TARGETS), dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
            )


class SimpleEEGCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))  # W/2
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))  # W/4
        x = F.silu(self.bn3(self.conv3(x)))
        x = x.mean(dim=(2, 3))  # GAP over H,W -> (B,128)
        x = self.drop(x)
        x = self.head(x)  # logits
        return x


def kl_div_loss_per_sample(p_logits, target_probs, eps=1e-7):
    """
    Score-improvement fix: return per-sample KL so we can apply per-sample weights correctly.
    Core loss remains KLDiv between target distribution and predicted softmax.
    """
    log_p = F.log_softmax(p_logits, dim=1)
    t = torch.clamp(target_probs, eps, 1.0)
    t = t / t.sum(dim=1, keepdim=True)
    return F.kl_div(log_p, t, reduction="none").sum(dim=1)




## === cell 3
from sklearn.model_selection import GroupKFold


def train_one_fold(fold, tr_idx, va_idx, eeg_dir):
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)

    tr_ds = EEGDataset(tr_df, eeg_dir=eeg_dir, mode="train")
    va_ds = EEGDataset(va_df, eeg_dir=eeg_dir, mode="valid")

    tr_loader = DataLoader(
        tr_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    va_loader = DataLoader(
        va_ds, batch_size=BATCHSIZE * 2, shuffle=False, num_workers=2, pin_memory=True
    )

    model = SimpleEEGCNN(n_classes=len(TARGETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)

    best = 1e9
    best_path = os.path.join("models", f"fold{fold}.pt")
    for epoch in range(EPOCHS):
        model.train()
        tr_losses = []
        for xb, yb, sw in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True).view(-1)  # (B,)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            per_sample = kl_div_loss_per_sample(logits, yb)  # (B,)
            loss = (per_sample * sw).sum() / torch.clamp(sw.sum(), min=1e-6)

            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb, sw in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = kl_div_loss_per_sample(logits, yb).mean()
                va_losses.append(loss.item())

        tr_l = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_l = float(np.mean(va_losses)) if va_losses else float("nan")
        print(f"Fold {fold} epoch {epoch+1}/{EPOCHS} loss {tr_l:.4f} val {va_l:.4f}")

        if va_l < best:
            best = va_l
            torch.save({"model": model.state_dict()}, best_path)

    del tr_ds, va_ds, tr_loader, va_loader, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_path


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    eeg_dir_train = os.path.join(LOAD_DATA_FROM, "train_eegs")
    gkf = GroupKFold(n_splits=SPLITS)

    fold_paths = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train.patient_id)):
        p = train_one_fold(fold, tr_idx, va_idx, eeg_dir_train)
        fold_paths.append(p)
    print("Saved fold models:", fold_paths)




## === cell 4
def predict_test(test_df: pd.DataFrame, eeg_dir: str, weight_paths):
    test_ds = EEGDataset(test_df, eeg_dir=eeg_dir, mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=TEST_BATCHSIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    models = []
    for wp in weight_paths:
        m = SimpleEEGCNN(n_classes=len(TARGETS)).to(device)
        ckpt = torch.load(wp, map_location=device)
        m.load_state_dict(ckpt["model"], strict=True)
        m.eval()
        models.append(m)

    all_preds = []
    with torch.no_grad():
        for xb, _, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits_ens = None
            for m in models:
                lg = m(xb)
                logits_ens = lg if logits_ens is None else (logits_ens + lg)
            logits_ens = logits_ens / max(len(models), 1)
            prob = F.softmax(logits_ens, dim=1).detach().cpu().numpy()
            all_preds.append(prob)

    preds = np.concatenate(all_preds, axis=0).astype(np.float32)
    preds = np.clip(preds, 1e-7, 1.0)
    preds = preds / np.sum(preds, axis=1, keepdims=True)
    return preds


if __name__ == "__main__":
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    eeg_dir_test = os.path.join(LOAD_DATA_FROM, "test_eegs")

    weight_paths = []
    for fold in range(SPLITS):
        p = os.path.join("models", f"fold{fold}.pt")
        if os.path.exists(p):
            weight_paths.append(p)

    if not weight_paths:
        preds = np.full((len(test), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)
    else:
        preds = predict_test(test, eeg_dir_test, weight_paths)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    for i, c in enumerate(TARGETS):
        sub[c] = preds[:, i]

    vals = sub[TARGETS].values.astype(np.float64)
    vals = np.clip(vals, 1e-7, 1.0)
    vals = vals / np.sum(vals, axis=1, keepdims=True)
    sub[TARGETS] = vals.astype(np.float32)

    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", sub.shape)
    print(sub.head())
