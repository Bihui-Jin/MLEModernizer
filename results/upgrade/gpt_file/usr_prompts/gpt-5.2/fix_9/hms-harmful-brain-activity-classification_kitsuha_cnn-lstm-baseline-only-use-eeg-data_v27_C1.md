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

1.166161429369046

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, glob, warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SAFE_SINGLE_WORKER = True




## === cell 1
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
df = pd.read_csv(PATH + "train.csv")
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()




## === cell 2
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds"]
].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
train.columns = ["spectrogram_id", "min"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head(5)




## === cell 3
ycol = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
cd = {
    "Seizure": "seizure_vote",
    "GPD": "gpd_vote",
    "LRDA": "lrda_vote",
    "Other": "other_vote",
    "GRDA": "grda_vote",
    "LPD": "lpd_vote",
}

eeg_id_col = train.iloc[:, 0]  # eeg_id
prob_cols = train.iloc[:, -7:-1]  # 6 probability columns
label_col = train.iloc[:, -1]  # expert_consensus

prob_cols = prob_cols.astype("float32")
prob_cols_normalized = prob_cols.div(prob_cols.sum(axis=1), axis=0)

normalized_data = pd.concat([eeg_id_col, prob_cols_normalized, label_col], axis=1)
normalized_data.head(5)




## === cell 4
normalized_csv_path = "normalized_data.csv"
normalized_data.to_csv(normalized_csv_path, index=False)
print("Wrote:", normalized_csv_path, "shape:", normalized_data.shape)




## === cell 5
EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
train_path = normalized_csv_path  # local CSV we just created


def _pick_parquet_engine():
    try:
        import fastparquet  # noqa: F401

        return "fastparquet"
    except Exception:
        try:
            import pyarrow  # noqa: F401

            return "pyarrow"
        except Exception:
            return "auto"


_PARQUET_ENGINE = _pick_parquet_engine()
print("Parquet engine:", _PARQUET_ENGINE)


def _nanmean_impute_inplace(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean).astype(x.dtype, copy=False)
    inds = np.where(np.isnan(x))
    x[inds] = col_mean[inds[1]]
    return x


class _NoOpBandpass:
    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data


class _LRUCache:
    def __init__(self, max_items: int = 64):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


class EEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path, cache_items: int = 32):
        self.csv = pd.read_csv(csv_file)
        self.eeg_path = eeg_path
        self.bandpass = _NoOpBandpass()

        self.FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

        self._eeg_ids = self.csv["eeg_id"].astype(np.int64).to_numpy()
        self._labels = self.csv[ycol].astype(np.float32).to_numpy()

        self._cache = _LRUCache(max_items=cache_items)

    def __len__(self):
        return self._eeg_ids.shape[0]

    def _load_eeg(self, eeg_id: int) -> np.ndarray:
        cached = self._cache.get(eeg_id)
        if cached is not None:
            return cached

        eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"
        eeg_df = pd.read_parquet(
            eeg_file_path, columns=self.FEATS, engine=_PARQUET_ENGINE
        )
        eeg_data = eeg_df.to_numpy(dtype=np.float32, copy=True)

        eeg_data = _nanmean_impute_inplace(eeg_data)
        eeg_data = self.bandpass(eeg_data).astype(np.float32, copy=False)

        mid_index = eeg_data.shape[0] // 2
        start_index = max(0, mid_index - 5000)
        end_index = min(eeg_data.shape[0], mid_index + 5000)
        eeg_data = eeg_data[start_index:end_index]

        desired_len = 10000
        if eeg_data.shape[0] < desired_len:
            pad = desired_len - eeg_data.shape[0]
            eeg_data = np.pad(eeg_data, ((0, pad), (0, 0)), mode="constant")
        elif eeg_data.shape[0] > desired_len:
            eeg_data = eeg_data[:desired_len]

        self._cache.put(eeg_id, eeg_data)
        return eeg_data

    def __getitem__(self, idx):
        eeg_id = int(self._eeg_ids[idx])
        eeg_data = self._load_eeg(eeg_id)

        eeg_tensor = torch.from_numpy(eeg_data).transpose(0, 1).contiguous()  # (C,T)
        labels = torch.from_numpy(self._labels[idx])
        return eeg_tensor, labels


dataset_full = EEGDataset(train_path, EEG_PATH, cache_items=128)

MAX_TRAIN_EEGS = 4096
subset_n = min(MAX_TRAIN_EEGS, len(dataset_full))
subset_indices = np.arange(subset_n, dtype=np.int64)
dataset = torch.utils.data.Subset(dataset_full, subset_indices)

if SAFE_SINGLE_WORKER:
    num_workers = 0
else:
    num_workers = min(4, os.cpu_count() or 2)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
print(
    "Train dataset size(full/subset):",
    len(dataset_full),
    "/",
    len(dataset),
    "num_workers:",
    num_workers,
)




## === cell 6
X, y = dataset_full[0]
print(f"Sample 1: X shape {X.shape}, y shape {y.shape}, y sum {y.sum().item():.6f}")




## === cell 7
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=6):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1)  # (B,T,C)
        x, _ = self.lstm(x)

        att_weights = F.softmax(self.attention(x), dim=1)  # (B,T,1)
        x = torch.sum(att_weights * x, dim=1)  # (B,H*2)

        x = self.fc(x)
        return x




## === cell 8
input_channels = 8
num_classes = 6

model = EEGNet(in_channels=input_channels, num_classes=num_classes).to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-12)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=1)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.6f}")




## === cell 9
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
TEST_EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


test_df = pd.read_csv(test_path)
unique_test_eeg_ids = (
    pd.Series(test_df["eeg_id"].astype(np.int64).unique(), name="eeg_id")
    .sort_values(kind="mergesort")
    .reset_index(drop=True)
)
unique_test_csv_path = "test_unique_eeg_ids.csv"
unique_test_eeg_ids.to_frame().to_csv(unique_test_csv_path, index=False)
print("Unique test eeg_ids:", len(unique_test_eeg_ids), "of", len(test_df))


class TestEEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path, cache_items: int = 64):
        self.csv = pd.read_csv(csv_file)
        self.eeg_path = eeg_path
        self.bandpass = _NoOpBandpass()
        self.FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]
        self._eeg_ids = self.csv["eeg_id"].astype(np.int64).to_numpy()
        self._cache = _LRUCache(max_items=cache_items)

    def __len__(self):
        return self._eeg_ids.shape[0]

    def __getitem__(self, idx):
        eeg_id = int(self._eeg_ids[idx])

        cached = self._cache.get(eeg_id)
        if cached is None:
            eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"
            eeg_df = pd.read_parquet(
                eeg_file_path, columns=self.FEATS, engine=_PARQUET_ENGINE
            )
            eeg_data = eeg_df.to_numpy(dtype=np.float32, copy=True)

            eeg_data = _nanmean_impute_inplace(eeg_data)
            eeg_data = self.bandpass(eeg_data).astype(np.float32, copy=False)

            mid_index = eeg_data.shape[0] // 2
            start_index = max(0, mid_index - 5000)
            end_index = min(eeg_data.shape[0], mid_index + 5000)
            eeg_data = eeg_data[start_index:end_index]

            desired_len = 10000
            if eeg_data.shape[0] < desired_len:
                pad = desired_len - eeg_data.shape[0]
                eeg_data = np.pad(eeg_data, ((0, pad), (0, 0)), mode="constant")
            elif eeg_data.shape[0] > desired_len:
                eeg_data = eeg_data[:desired_len]

            self._cache.put(eeg_id, eeg_data)
        else:
            eeg_data = cached

        eeg_tensor = torch.from_numpy(eeg_data).transpose(0, 1).contiguous()
        return eeg_tensor


testdataset = TestEEGDataset(unique_test_csv_path, TEST_EEG_PATH, cache_items=256)

if SAFE_SINGLE_WORKER:
    test_num_workers = 0
else:
    test_num_workers = min(4, os.cpu_count() or 2)

test_dataloader = DataLoader(
    testdataset,
    batch_size=64,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(test_num_workers > 0),
    prefetch_factor=2 if test_num_workers > 0 else None,
)

model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions.append(probabilities.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
print("Predictions shape (unique eeg_id):", predictions.shape)




## === cell 10
pred = np.clip(predictions.astype(np.float64, copy=False), 1e-8, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
results_df = pd.DataFrame(pred, columns=columns)
results_df.head()




## === cell 11
sample_sub_path = PATH + "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

if len(results_df) != len(unique_test_eeg_ids):
    raise RuntimeError(
        f"Prediction length mismatch vs unique eeg_id list: preds={len(results_df)} unique_eeg_ids={len(unique_test_eeg_ids)}"
    )

pred_map = unique_test_eeg_ids.to_frame().copy()
for c in columns:
    pred_map[c] = results_df[c].values

sub = sample_sub[["eeg_id"]].merge(
    pred_map, on="eeg_id", how="left", validate="many_to_one"
)

if sub[columns].isna().any().any():
    sub[columns] = sub[columns].fillna(1.0 / len(columns))

sub = sub[sample_sub.columns]
sub[columns] = sub[columns].clip(1e-8, 1.0)
sub[columns] = sub[columns].div(sub[columns].sum(axis=1), axis=0)

row_sums = sub[columns].sum(axis=1).values
print("Submission rows:", len(sub), "Sample rows:", len(sample_sub))
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))
assert len(sub) == len(sample_sub)
assert np.allclose(row_sums, 1.0, atol=1e-6)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv. Submission shape", sub.shape)
sub.head()
