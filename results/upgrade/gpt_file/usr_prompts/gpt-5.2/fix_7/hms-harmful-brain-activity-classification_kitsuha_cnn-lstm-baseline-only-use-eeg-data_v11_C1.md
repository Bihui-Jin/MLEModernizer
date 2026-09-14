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

1.0735758289935646

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.4764) has done: 'The failure is caused by creating many duplicate rows during `sample.merge(pred_df, ...)` because `test.csv` can contain repeated `eeg_id` values, so `pred_df` has duplicates and the merge explodes the row count. I fix this by ensuring we generate exactly one prediction per `eeg_id` (matching the sample submission) via a deterministic groupby aggregation, then align the predictions to `sample_submission.csv` order without any many-to-many merge. I also keep the probabilities safely normalized and clipped to avoid invalid rows (sum≠1 or zeros) which can hurt KL-div and can break submission checks. Core model/training logic is unchanged; only submission construction and id alignment are fixed so it runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
df = pd.read_csv(PATH + "train.csv")

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
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

normalized_data = train[["eeg_id"] + ycol + ["target"]].copy()
normalized_data[ycol] = normalized_data[ycol].astype("float32")
normalized_data[ycol] = normalized_data[ycol].div(
    normalized_data[ycol].sum(axis=1), axis=0
)

_eps = 1e-6
vals = normalized_data[ycol].to_numpy(dtype=np.float64)
vals = np.clip(vals, _eps, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
normalized_data[ycol] = vals.astype(np.float32)

normalized_data.head(5)



## === cell 4
output_file = "normalized_data.csv"
normalized_data.to_csv(output_file, index=False)
print("Wrote:", os.path.abspath(output_file), "rows:", len(normalized_data))



## === cell 5
EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
train_path = "/kaggle/working/normalized_data.csv"  # keep path unchanged

EEG_CHANNELS = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "O1",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F7",
    "T3",
    "T5",
    "F8",
    "T4",
    "T6",
    "Fz",
    "Cz",
    "Pz",
    "EKG",
]


def center_crop_or_pad(eeg_array_2d: np.ndarray, target_len: int = 10000) -> np.ndarray:
    """Ensure time dimension length equals target_len by center-cropping or padding."""
    n = eeg_array_2d.shape[0]
    if n == target_len:
        return eeg_array_2d
    if n > target_len:
        mid = n // 2
        half = target_len // 2
        start = max(0, mid - half)
        end = start + target_len
        if end > n:
            end = n
            start = end - target_len
        return eeg_array_2d[start:end]
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(
        eeg_array_2d,
        ((pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )


def nanmean_impute_inplace(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    col_means = np.nanmean(x, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(
        np.float32, copy=False
    )
    inds = np.where(np.isnan(x))
    x[inds] = col_means[inds[1]]
    return x


def load_eeg_as_array(eeg_file_path: str, channels: list[str]) -> np.ndarray:
    """Load parquet and return (T, C) float32 with a fixed channel order; missing -> zeros."""
    eeg_df = pd.read_parquet(eeg_file_path)
    T = len(eeg_df)
    out = np.zeros((T, len(channels)), dtype=np.float32)
    cols = eeg_df.columns
    for j, ch in enumerate(channels):
        if ch in cols:
            out[:, j] = eeg_df[ch].to_numpy(dtype=np.float32, copy=False)
    out = nanmean_impute_inplace(out)
    return out


class EEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path, max_items: int | None = 6000):
        csv = pd.read_csv(csv_file)
        if max_items is not None and len(csv) > max_items:
            csv = csv.iloc[:max_items].reset_index(drop=True)
        self.eeg_ids = csv["eeg_id"].to_numpy(dtype=np.int64)
        self.labels = csv[ycol].to_numpy(dtype=np.float32)
        self.eeg_path = eeg_path

    def __len__(self):
        return self.eeg_ids.shape[0]

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])
        eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"

        eeg_data = load_eeg_as_array(eeg_file_path, EEG_CHANNELS)
        eeg_data = center_crop_or_pad(eeg_data, target_len=10000)  # (T, C)
        eeg_data = torch.from_numpy(eeg_data).transpose(0, 1)  # (C, T)

        labels = torch.from_numpy(self.labels[idx])
        return eeg_data, labels


dataset = EEGDataset(train_path, EEG_PATH, max_items=6000)

num_workers = min(4, (os.cpu_count() or 2))
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

X0, y0 = dataset[0]
print("Sample shapes:", X0.shape, y0.shape)




## === cell 6
class EEGNet(nn.Module):
    def __init__(self, in_channels=20, num_classes=6, weight_decay=0.01):
        super(EEGNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=64, stride=2, padding=16)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(0.25)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, num_classes)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.pool(x).squeeze(-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

    def l2_regularization(self):
        l2_reg = torch.tensor(0.0, device=self.conv1.weight.device)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)) and m.weight is not None:
                l2_reg = l2_reg + (m.weight**2).sum()
        return self.weight_decay * l2_reg




## === cell 7
input_channels = X0.shape[0]
num_classes = 6

model = EEGNet(input_channels, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        log_probs = F.log_softmax(logits, dim=1)

        loss = criterion(log_probs, labels)
        loss = loss + model.l2_regularization()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.6f}")



## === cell 8
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
TEST_EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"


class TestEEGDataset(Dataset):
    def __init__(self, csv_file, eeg_path):
        csv = pd.read_csv(csv_file)
        self.eeg_ids = csv["eeg_id"].to_numpy(dtype=np.int64)
        self.eeg_path = eeg_path

    def __len__(self):
        return self.eeg_ids.shape[0]

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])
        eeg_file_path = f"{self.eeg_path}{eeg_id}.parquet"

        eeg_data = load_eeg_as_array(eeg_file_path, EEG_CHANNELS)
        eeg_data = center_crop_or_pad(eeg_data, target_len=10000)
        eeg_data = torch.from_numpy(eeg_data).transpose(0, 1)
        return eeg_data


testdataset = TestEEGDataset(test_path, TEST_EEG_PATH)

test_num_workers = num_workers
test_dataloader = DataLoader(
    testdataset,
    batch_size=32,
    shuffle=False,
    num_workers=test_num_workers,
    pin_memory=True,
    persistent_workers=(test_num_workers > 0),
    prefetch_factor=4 if test_num_workers > 0 else None,
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
print("Predictions shape:", predictions.shape)



## === cell 9
columns = TARGETS

sample_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample = pd.read_csv(sample_path)
test_data = pd.read_csv(test_path)

pred_row_df = pd.DataFrame(predictions, columns=columns)
pred_row_df.insert(0, "eeg_id", test_data["eeg_id"].to_numpy(dtype=np.int64))

pred_unique = pred_row_df.groupby("eeg_id", sort=False)[columns].mean().reset_index()

sub = sample[["eeg_id"]].merge(pred_unique, on="eeg_id", how="left")

missing = sub[columns].isna().any(axis=1)
if missing.any():
    sub.loc[missing, columns] = 1.0 / len(columns)

probs = sub[columns].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-4, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
sub[columns] = probs.astype(np.float32)

assert len(sub) == len(
    sample
), f"Invalid submission length: {len(sub)} vs sample {len(sample)}"
row_sums = sub[columns].sum(axis=1)
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))
assert np.all(np.isfinite(probs)), "Non-finite probabilities found"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", "shape:", sub.shape)
print(sub.head())
