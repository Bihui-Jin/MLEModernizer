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
import os, gc, glob, warnings, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
TRAIN_CSV = os.path.join(PATH, "train.csv")
TEST_CSV = os.path.join(PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(PATH, "sample_submission.csv")

SPEC_PATH = os.path.join(PATH, "train_spectrograms") + "/"
TEST_SPEC_PATH = os.path.join(PATH, "test_spectrograms") + "/"

df = pd.read_csv(TRAIN_CSV)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



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



## === cell 3
ycol = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

normalized_data = train[["eeg_id", "spectrogram_id"] + ycol + ["target"]].copy()
normalized_data[ycol] = normalized_data[ycol].astype("float32")
normalized_data[ycol] = normalized_data[ycol].div(
    normalized_data[ycol].sum(axis=1), axis=0
)

normalized_data.to_csv("normalized_data.csv", index=False)
train_path = "normalized_data.csv"
print("Wrote:", train_path, "shape:", normalized_data.shape)



## === cell 4
norm_df = pd.read_csv(train_path)
idx = np.arange(len(norm_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.05
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

train_csv_path = "train_split.csv"
val_csv_path = "val_split.csv"
norm_df.iloc[trn_idx].to_csv(train_csv_path, index=False)
norm_df.iloc[val_idx].to_csv(val_csv_path, index=False)
print("Split sizes:", len(trn_idx), len(val_idx))




## === cell 5
class SpectrogramDataset(Dataset):
    """
    Uses spectrogram parquet (freq_bins x time) as a 1D sequence:
    - channels = selected freq bins (fixed count)
    - time = spectrogram time steps
    This preserves the same model interface: (channels, time).
    """

    def __init__(self, csv_file, spec_path, ycol=None, n_freq_bins=8):
        self.csv = pd.read_csv(csv_file)
        self.spec_path = spec_path
        self.ycol = ycol
        self.n_freq_bins = n_freq_bins

    def __len__(self):
        return len(self.csv)

    @staticmethod
    def _to_float32(x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return x

    @staticmethod
    def _fill_nan(x: np.ndarray) -> np.ndarray:
        if not np.isnan(x).any():
            return x
        x = x.copy()
        x[np.isnan(x)] = 0.0
        return x

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        spec_id = int(row["spectrogram_id"])
        spec_file_path = f"{self.spec_path}{spec_id}.parquet"

        spec_df = pd.read_parquet(spec_file_path)
        spec = spec_df.values
        spec = self._to_float32(spec)
        spec = self._fill_nan(spec)

        m = spec.mean()
        s = spec.std() + 1e-6
        spec = (spec - m) / s

        if spec.shape[0] >= self.n_freq_bins:
            spec = spec[: self.n_freq_bins]
        else:
            pad = np.zeros(
                (self.n_freq_bins - spec.shape[0], spec.shape[1]), dtype=np.float32
            )
            spec = np.concatenate([spec, pad], axis=0)

        x = torch.from_numpy(spec)  # (channels, time)

        if self.ycol is None:
            return x

        labels = torch.tensor(
            row[self.ycol].values.astype(np.float32), dtype=torch.float32
        )
        return x, labels


train_dataset = SpectrogramDataset(train_csv_path, SPEC_PATH, ycol=ycol, n_freq_bins=8)
val_dataset = SpectrogramDataset(val_csv_path, SPEC_PATH, ycol=ycol, n_freq_bins=8)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

X0, y0 = train_dataset[0]
print("Train sample shapes:", X0.shape, y0.shape)




## === cell 6
class CNNLSTM(nn.Module):
    def __init__(self, in_channels=8, num_classes=6):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.75)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
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
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        x = x.permute(0, 2, 1)  # (batch, seq, channels)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        att_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq, 1)
        x = torch.sum(att_weights * x, dim=1)  # (batch, features)

        x = self.fc(x)
        return x




## === cell 7
input_channels = 8
num_classes = 6

model = CNNLSTM(in_channels=input_channels, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)


def eval_kl(model, loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            target_probs = labels.clamp(min=1e-7)
            target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
            loss = criterion(log_probs, target_probs)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)

        log_probs = F.log_softmax(outputs, dim=1)
        target_probs = labels.clamp(min=1e-7)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)

        loss = criterion(log_probs, target_probs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss = eval_kl(model, val_loader)
    print(f"Epoch {epoch+1}, Train KL: {train_loss:.6f}, Val KL: {val_loss:.6f}")




## === cell 8
class TestSpectrogramDataset(Dataset):
    def __init__(self, csv_file, spec_path, n_freq_bins=8):
        self.csv = pd.read_csv(csv_file)
        self.spec_path = spec_path
        self.n_freq_bins = n_freq_bins

    def __len__(self):
        return len(self.csv)

    @staticmethod
    def _to_float32(x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return x

    @staticmethod
    def _fill_nan(x: np.ndarray) -> np.ndarray:
        if not np.isnan(x).any():
            return x
        x = x.copy()
        x[np.isnan(x)] = 0.0
        return x

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        spec_id = int(row["spectrogram_id"])
        spec_file_path = f"{self.spec_path}{spec_id}.parquet"

        spec_df = pd.read_parquet(spec_file_path)
        spec = spec_df.values
        spec = self._to_float32(spec)
        spec = self._fill_nan(spec)

        m = spec.mean()
        s = spec.std() + 1e-6
        spec = (spec - m) / s

        if spec.shape[0] >= self.n_freq_bins:
            spec = spec[: self.n_freq_bins]
        else:
            pad = np.zeros(
                (self.n_freq_bins - spec.shape[0], spec.shape[1]), dtype=np.float32
            )
            spec = np.concatenate([spec, pad], axis=0)

        x = torch.from_numpy(spec)  # (channels, time)
        return x


testdataset = TestSpectrogramDataset(TEST_CSV, TEST_SPEC_PATH, n_freq_bins=8)
test_dataloader = DataLoader(
    testdataset,
    batch_size=64,
    shuffle=False,  # must align with test.csv order
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predictions.append(probabilities.detach().cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
print("Predictions shape:", predictions.shape)



## === cell 9
test_data = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

columns = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
results_df = pd.DataFrame(predictions, columns=columns)

assert len(results_df) == len(test_data) == len(sample_sub), (
    len(results_df),
    len(test_data),
    len(sample_sub),
)

sub = pd.DataFrame({"eeg_id": test_data["eeg_id"].values})
sub[columns] = results_df[columns].values

sub[columns] = sub[columns].clip(lower=1e-7)
sub[columns] = sub[columns].div(sub[columns].sum(axis=1), axis=0)

sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
print(
    "Row sum stats:",
    float(sub[columns].sum(axis=1).min()),
    float(sub[columns].sum(axis=1).max()),
)
sub.head()



## === cell 10
print(sub.columns.tolist())
print(sub.isna().sum())
print("Saved to submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
