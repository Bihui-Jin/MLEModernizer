# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.2846556071065502

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash caused by an incompatible `torchaudio/torch` import in this Kaggle/Python 3.13 environment by making those imports optional and only required when `DATATYPE` actually includes `stft`. I also fix a couple of Kaggle-inference path/availability issues that prevent producing a submission: (1) the code currently expects preprocessed `eegs.npy` (not available on Kaggle) even though it can read raw parquet test EEGs, and (2) it may fail to find model weights and still proceed. Finally, I ensure the script always writes a valid `submission.csv` with correct columns and row-wise probabilities summing to 1 (with a safe normalization/clip), so you always get a valid submission file end-to-end.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before TensorFlow/Keras imports, which is a common incompatibility in Python 3.13 Kaggle images. I also ensure inference always has the correct `TARGETS` list even when `NEEDTRAIN=False` by reading it from `sample_submission.csv`, preventing accidental column mismatch and helping score versus uniform fallback. Finally, I make model-weight discovery robust (avoid the dummy `modelsxxxxxxx` path causing “no models found”) by defaulting to a local `models/` directory if present, and I keep the submission probabilities strictly normalized and clipped to valid ranges.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the TensorFlow/protobuf dependency entirely and switching the script to PyTorch (which is available on Kaggle) while keeping the same core “EEG-only EfficientNet-like image backbone” semantics via a simple CNN over the same reshaped EEG “image” representation. I also fix inference-time issues that currently cause a fallback to uniform predictions (missing weights, missing preprocess `.npy`) by training on-the-fly on Kaggle using a patient-group split and then ensembling folds for test predictions. Finally, I ensure the submission matches `sample_submission.csv` column order exactly and that every row is strictly normalized and clipped so it passes Kaggle validation and improves KL divergence toward your target.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import gc
import time
import math
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
    NEEDTRAIN = True  # ensure we actually train and improve score vs uniform submission
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
EPOCHS = 3  # keep runtime under 600s on Kaggle while still improving score materially vs uniform
SPLITS = 3  # reduced folds to fit runtime; still group-based for correctness

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


def read_eeg_parquet(eeg_id: int, base_path: str) -> np.ndarray:
    """Return (16, 10000) float32 EEG after montage + filter + clipping, matching original semantics."""
    p = os.path.join(base_path, f"{int(eeg_id)}.parquet")
    eeg_default = pd.read_parquet(p)

    eeg = []
    for channel in BRAIN:
        a0, a1 = channel.split("-")
        eeg_temp = (eeg_default.loc[:, a0] - eeg_default.loc[:, a1]).values
        eeg_temp = np.nan_to_num(eeg_temp, nan=0.0).astype(np.float32)
        eeg.append(eeg_temp[None, :])
    eeg = np.concatenate(eeg, axis=0)

    if SFREQ != RSFREQ:
        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1).astype(np.float32)

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
                torch.zeros(6, dtype=torch.float32),
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


def kl_div_loss(p_logits, target_probs, eps=1e-7):
    log_p = F.log_softmax(p_logits, dim=1)
    t = torch.clamp(target_probs, eps, 1.0)
    t = t / t.sum(dim=1, keepdim=True)
    return F.kl_div(log_p, t, reduction="batchmean")




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
            sw = sw.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = kl_div_loss(logits, yb)
            loss = loss * sw.mean()
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
                loss = kl_div_loss(logits, yb)
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1275640043.py in <cell line: 0>()
     74     fold_paths = []
     75     for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train.patient_id)):
---> 76         p = train_one_fold(fold, tr_idx, va_idx, eeg_dir_train)
     77         fold_paths.append(p)
     78     print("Saved fold models:", fold_paths)

/tmp/ipykernel_55/1275640043.py in train_one_fold(fold, tr_idx, va_idx, eeg_dir)
     25         model.train()
     26         tr_losses = []
---> 27         for xb, yb, sw in tr_loader:
     28             xb = xb.to(device, non_blocking=True)
     29             yb = yb.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 211, in collate
    return [
           ^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 212, in <listcomp>
    collate(samples, collate_fn_map=collate_fn_map)
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 155, in collate
    return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 272, in collate_tensor_fn
    return torch.stack(batch, 0, out=out)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: stack expects each tensor to be equal size, but got [1, 16, 1000] at entry 0 and [1, 16, 1440] at entry 1


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
