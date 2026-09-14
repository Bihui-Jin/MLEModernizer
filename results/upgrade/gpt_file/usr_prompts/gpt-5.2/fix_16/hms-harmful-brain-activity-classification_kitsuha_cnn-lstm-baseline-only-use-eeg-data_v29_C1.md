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
import os, gc, glob, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.signal import butter, sosfilt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
PATH_CANDIDATES = [
    "/kaggle/input/hms-harmful-brain-activity-classification/",
    "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification/",
]
PATH = None
for p in PATH_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")):
        PATH = p
        break
if PATH is None:
    raise FileNotFoundError(f"Could not find train.csv in any of: {PATH_CANDIDATES}")

TRAIN_CSV = os.path.join(PATH, "train.csv")
TEST_CSV = os.path.join(PATH, "test.csv")
SAMPLE_SUB = os.path.join(PATH, "sample_submission.csv")

EEG_TRAIN_PATH = os.path.join(PATH, "train_eegs/")
EEG_TEST_PATH = os.path.join(PATH, "test_eegs/")

WORK_NORM_CSV = "/kaggle/working/normalized_data.csv"
SUB_PATH = "/kaggle/working/submission.csv"

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

print("Using PATH:", PATH)



## === cell 2
df = pd.read_csv(TRAIN_CSV)
print("Train shape:", df.shape)
print("Targets:", TARGETS)
df.head()



## === cell 3
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds"]
].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
train.columns = ["spectrogram_id", "min"]

tmp = df.groupby("eeg_id")[["spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

votes = df[TARGETS].to_numpy(dtype=np.float32, copy=True)
row_sum = votes.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
row_probs = votes / row_sum

row_probs_df = pd.DataFrame(row_probs, columns=TARGETS)
row_probs_df["eeg_id"] = df["eeg_id"].to_numpy()

avg_probs = row_probs_df.groupby("eeg_id")[TARGETS].mean()

y_data = avg_probs.to_numpy(dtype=np.float32, copy=True)
eps = 1e-6
y_data = np.clip(y_data, eps, 1.0)
y_data = y_data / y_data.sum(axis=1, keepdims=True)

for i, t in enumerate(TARGETS):
    train[t] = y_data[:, i]

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## === cell 4
eeg_id_col = train["eeg_id"]
prob_cols = train[TARGETS].astype("float32")
label_col = train["target"]

normalized_data = pd.concat(
    [
        eeg_id_col.reset_index(drop=True),
        prob_cols.reset_index(drop=True),
        label_col.reset_index(drop=True),
    ],
    axis=1,
)

normalized_data = normalized_data.drop_duplicates(
    subset=["eeg_id"], keep="first"
).reset_index(drop=True)

normalized_data.to_csv(WORK_NORM_CSV, index=False)
print("Wrote:", WORK_NORM_CSV, "shape:", normalized_data.shape)
normalized_data.head()



## === cell 5
ycol = TARGETS

try:
    import pyarrow.parquet as pq  # preferred on Kaggle when available
except Exception:
    pq = None

try:
    import fastparquet  # noqa: F401

    _HAS_FASTPARQUET = True
except Exception:
    _HAS_FASTPARQUET = False


def read_parquet_columns(path, columns):
    if pq is not None:
        table = pq.read_table(path, columns=columns)
        return table.to_pandas()
    if _HAS_FASTPARQUET:
        return pd.read_parquet(path, columns=columns, engine="fastparquet")
    return pd.read_parquet(path, columns=columns)


class EEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path, cache_size=-1):
        self.csv = pd.read_csv(csv_file)
        self.eeg_path = eeg_path
        self.sos = self.butter_bandpass_filter_init()
        self.FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

        self._cache = {}
        self._cache_order = []
        self._cache_size = int(cache_size)

        self.eeg_ids = self.csv["eeg_id"].to_numpy()
        self.labels = self.csv[ycol].to_numpy(dtype=np.float32, copy=True)

    def butter_bandpass_filter_init(self):
        lowcut, highcut, fs, order = 0.5, 40.0, 200.0, 5
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype="band", output="sos")

    def butter_bandpass_filter(self, data):
        return sosfilt(self.sos, data, axis=0)

    def __len__(self):
        return len(self.eeg_ids)

    @staticmethod
    def _mean_impute_inplace(x: np.ndarray) -> np.ndarray:
        if not np.isnan(x).any():
            return x
        col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(np.isnan(x))
        x[inds] = col_means[inds[1]]
        return x

    @staticmethod
    def _zscore_inplace(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)
        x -= mu
        x /= sd
        return x

    def _cache_get(self, key):
        return self._cache.get(key, None)

    def _cache_put(self, key, value):
        if self._cache_size == 0:
            return
        if key in self._cache:
            return
        self._cache[key] = value
        if self._cache_size > 0:
            self._cache_order.append(key)
            if len(self._cache_order) > self._cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])

        cached = self._cache_get(eeg_id)
        if cached is not None:
            eeg_data = cached
        else:
            eeg_file_path = os.path.join(self.eeg_path, f"{eeg_id}.parquet")
            try:
                eeg_df = read_parquet_columns(eeg_file_path, columns=self.FEATS)
                eeg_data_np = eeg_df.to_numpy(dtype=np.float32, copy=True)

                eeg_data_np = self._mean_impute_inplace(eeg_data_np)
                eeg_data_np = self.butter_bandpass_filter(eeg_data_np).astype(
                    np.float32, copy=False
                )

                mid_index = eeg_data_np.shape[0] // 2
                start_index = max(0, mid_index - 5000)
                end_index = min(eeg_data_np.shape[0], mid_index + 5000)
                eeg_data_np = eeg_data_np[start_index:end_index]

                target_len = 10000
                cur_len = eeg_data_np.shape[0]
                if cur_len < target_len:
                    pad = np.zeros(
                        (target_len - cur_len, eeg_data_np.shape[1]), dtype=np.float32
                    )
                    eeg_data_np = np.concatenate([eeg_data_np, pad], axis=0)
                elif cur_len > target_len:
                    eeg_data_np = eeg_data_np[:target_len]

                eeg_data_np = self._zscore_inplace(eeg_data_np)
            except Exception:
                eeg_data_np = np.zeros((10000, len(self.FEATS)), dtype=np.float32)

            eeg_data = torch.from_numpy(eeg_data_np.T.copy())
            self._cache_put(eeg_id, eeg_data)

        labels = torch.from_numpy(self.labels[idx].copy())
        timestamps = torch.arange(0, 10000, dtype=torch.float32)
        return eeg_data, labels, timestamps


dataset = EEGDataset(WORK_NORM_CSV, EEG_TRAIN_PATH, cache_size=-1)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

X0, y0, t0 = dataset[0]
print("Sample shapes:", X0.shape, y0.shape, t0.shape)




## === cell 6
class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(TLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x, timestamps):
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


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

        self.tlstm = TLSTM(input_size=64, hidden_size=128)

        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x, timestamps):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

        x = x.permute(0, 2, 1)

        x, _ = self.tlstm(x, timestamps)

        att_weights = F.softmax(self.attention(x), dim=1)  # [B, T', 1]
        x = torch.sum(att_weights * x, dim=1)  # [B, 256]
        x = self.fc(x)
        return x




## === cell 7
input_channels = 8
num_classes = 6

model = EEGNet(in_channels=input_channels, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, timestamps in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, timestamps=timestamps)
        log_probs = F.log_softmax(logits, dim=1)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.6f}")




## === cell 8
class TestEEGDataset(Dataset):
    def __init__(self, eeg_ids, eeg_path, cache_size=-1):
        self.eeg_ids = np.asarray(eeg_ids)
        self.eeg_path = eeg_path
        self.sos = self.butter_bandpass_filter_init()
        self.FEATS = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

        self._cache = {}
        self._cache_order = []
        self._cache_size = int(cache_size)

    def __len__(self):
        return len(self.eeg_ids)

    def butter_bandpass_filter_init(self):
        lowcut, highcut, fs, order = 0.5, 40.0, 200.0, 5
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], analog=False, btype="band", output="sos")

    def butter_bandpass_filter(self, data):
        return sosfilt(self.sos, data, axis=0)

    @staticmethod
    def _mean_impute_inplace(x: np.ndarray) -> np.ndarray:
        if not np.isnan(x).any():
            return x
        col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(np.isnan(x))
        x[inds] = col_means[inds[1]]
        return x

    @staticmethod
    def _zscore_inplace(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)
        x -= mu
        x /= sd
        return x

    def _cache_get(self, key):
        return self._cache.get(key, None)

    def _cache_put(self, key, value):
        if self._cache_size == 0:
            return
        if key in self._cache:
            return
        self._cache[key] = value
        if self._cache_size > 0:
            self._cache_order.append(key)
            if len(self._cache_order) > self._cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])

        cached = self._cache_get(eeg_id)
        if cached is not None:
            return cached

        eeg_file_path = os.path.join(self.eeg_path, f"{eeg_id}.parquet")
        try:
            eeg_df = read_parquet_columns(eeg_file_path, columns=self.FEATS)
            eeg_data_np = eeg_df.to_numpy(dtype=np.float32, copy=True)

            eeg_data_np = self._mean_impute_inplace(eeg_data_np)
            eeg_data_np = self.butter_bandpass_filter(eeg_data_np).astype(
                np.float32, copy=False
            )

            mid_index = eeg_data_np.shape[0] // 2
            start_index = max(0, mid_index - 5000)
            end_index = min(eeg_data_np.shape[0], mid_index + 5000)
            eeg_data_np = eeg_data_np[start_index:end_index]

            target_len = 10000
            cur_len = eeg_data_np.shape[0]
            if cur_len < target_len:
                pad = np.zeros(
                    (target_len - cur_len, eeg_data_np.shape[1]), dtype=np.float32
                )
                eeg_data_np = np.concatenate([eeg_data_np, pad], axis=0)
            elif cur_len > target_len:
                eeg_data_np = eeg_data_np[:target_len]

            eeg_data_np = self._zscore_inplace(eeg_data_np)
        except Exception:
            eeg_data_np = np.zeros((10000, len(self.FEATS)), dtype=np.float32)

        eeg_data = torch.from_numpy(eeg_data_np.T.copy())
        self._cache_put(eeg_id, eeg_data)
        return eeg_data


sample_sub = pd.read_csv(SAMPLE_SUB)
sub_eeg_ids = sample_sub["eeg_id"].to_numpy()

testdataset = TestEEGDataset(sub_eeg_ids, EEG_TEST_PATH, cache_size=-1)
test_dataloader = DataLoader(
    testdataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        bsz = inputs.size(0)
        timestamps = (
            torch.arange(0, 10000, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        logits = model(inputs, timestamps=timestamps)
        probs = torch.softmax(logits, dim=1)
        predictions.append(probs.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
print("Predictions shape:", predictions.shape, "Expected:", len(sample_sub))



## === cell 9
if "normalized_data" not in globals():
    normalized_data = pd.read_csv(WORK_NORM_CSV)

pred = predictions.astype("float64")
eps = 1e-6
pred = np.clip(pred, eps, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

prior = normalized_data[TARGETS].mean(axis=0).to_numpy(dtype=np.float64)
prior = np.clip(prior, eps, 1.0)
prior = prior / prior.sum()

alpha = 0.42  # keep as-is (calibration already chosen to be closer to target than raw)
pred = (1.0 - alpha) * pred + alpha * prior[None, :]
pred = np.clip(pred, eps, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

beta = 0.09  # keep as-is
uniform = np.full((1, len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float64)
pred = (1.0 - beta) * pred + beta * uniform
pred = np.clip(pred, eps, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub = sample_sub[["eeg_id"]].copy()
for i, c in enumerate(TARGETS):
    sub[c] = pred[:, i]

row_sums = sub[TARGETS].sum(axis=1).to_numpy()
row_sums[row_sums == 0] = 1.0
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

sub.to_csv(SUB_PATH, index=False)
print("Wrote submission:", SUB_PATH)
print("Submission shape:", sub.shape)
print(
    "Min/Max row sum:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
sub.head()



## === cell 10
assert list(sub.columns) == ["eeg_id"] + TARGETS
assert np.isfinite(sub[TARGETS].values).all()
assert len(sub) == len(sample_sub)
assert np.allclose(sub[TARGETS].sum(axis=1).values, 1.0, atol=1e-6)
assert np.array_equal(sub["eeg_id"].to_numpy(), sample_sub["eeg_id"].to_numpy())
print("All checks passed.")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
