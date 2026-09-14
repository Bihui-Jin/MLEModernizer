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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



## === cell 1
import os
from pathlib import Path

PATH_IN = Path("/kaggle/input/")
PATH_TMP = Path("/kaggle/temp/")
PATH_OUT = Path("/kaggle/working/")

COMP_PATH = PATH_IN / "hms-harmful-brain-activity-classification"

TRAIN_CSV = COMP_PATH / "train.csv"
TEST_CSV = COMP_PATH / "test.csv"
SAMPLE_SUB = COMP_PATH / "sample_submission.csv"

TRAIN_EEG_DIR = COMP_PATH / "train_eegs"
TEST_EEG_DIR = COMP_PATH / "test_eegs"



## === cell 2
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
device



## === cell 3
VOTE_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def votes_to_prob(votes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    votes = np.asarray(votes, dtype=np.float32)
    s = votes.sum(axis=-1, keepdims=True)
    s = np.maximum(s, eps)
    p = votes / s
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return p




## === cell 4
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.lstm = nn.LSTM(input_size=19, hidden_size=19, batch_first=True)

        self.dropout1 = nn.Dropout(p=0.2)

        self.conv1 = nn.Conv1d(in_channels=19, out_channels=6, kernel_size=3, stride=1)
        self.batch_norm1 = nn.BatchNorm1d(num_features=6)
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv5 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(in_features=6, out_features=6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.contiguous()
        if x.is_cuda:
            try:
                x, _ = self.lstm(x)
            except RuntimeError as e:
                msg = str(e)
                if "CUDNN_STATUS_NOT_SUPPORTED" in msg or "cudnn" in msg.lower():
                    with torch.backends.cudnn.flags(enabled=False):
                        x, _ = self.lstm(x)
                else:
                    raise
        else:
            x, _ = self.lstm(x)

        x = self.dropout1(x)

        x = x.permute(0, 2, 1).contiguous()

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.max_pool1(x)

        x = self.conv5(x)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x




## === cell 5
from collections import OrderedDict


def _read_eeg_parquet_cached(fp: Path, cache: OrderedDict, cache_max_items: int = 256):
    key = str(fp)
    arr = cache.get(key, None)
    if arr is not None:
        cache.move_to_end(key)
        return arr

    df = pd.read_parquet(fp, engine="pyarrow")
    if "EKG" in df.columns:
        df = df.drop("EKG", axis=1)
    arr = np.ascontiguousarray(df.to_numpy(), dtype=np.float32)

    cache[key] = arr
    if len(cache) > cache_max_items:
        cache.popitem(last=False)
    return arr


class TrainDataset(Dataset):
    def __init__(
        self, meta_df: pd.DataFrame, eeg_dir: Path, max_samples: int | None = None
    ):
        self.meta_df = meta_df.reset_index(drop=True)
        if max_samples is not None:
            self.meta_df = self.meta_df.iloc[:max_samples].reset_index(drop=True)
        self.eeg_dir = eeg_dir

        self.targets = votes_to_prob(self.meta_df[VOTE_COLS].values)

        self._cache = OrderedDict()

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        eeg_id = str(int(self.meta_df.loc[idx, "eeg_id"]))
        fp = self.eeg_dir / f"{eeg_id}.parquet"

        arr = _read_eeg_parquet_cached(fp, self._cache, cache_max_items=256)

        x = torch.from_numpy(arr)  # (T, 19)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)  # (6,)
        return x, y




## === cell 6
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True)  # pads with zeros by default
    y = torch.stack(ys, dim=0).contiguous()
    return x_pad.contiguous(), y




## === cell 7
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)

n_train = int(0.95 * len(idx))
train_idx = idx[:n_train]
val_idx = idx[n_train:]

train_meta = train_df.iloc[train_idx].reset_index(drop=True)
val_meta = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = TrainDataset(train_meta, TRAIN_EEG_DIR)
val_ds = TrainDataset(val_meta, TRAIN_EEG_DIR)

num_workers = min(8, os.cpu_count() or 2)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=pad_collate,
    persistent_workers=True,
    prefetch_factor=4,
)
val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=pad_collate,
    persistent_workers=True,
    prefetch_factor=4,
)

len(train_ds), len(val_ds)



## === cell 8
model = CustomModel().to(device)

criterion = nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def run_eval(loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous()
            y = y.to(device, non_blocking=True).contiguous()

            p = model(x)  # already softmax probs
            p = torch.clamp(p, 1e-6, 1.0)
            p = p / p.sum(dim=1, keepdim=True)

            loss = criterion(torch.log(p), y)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


EPOCHS = 2
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for x, y in train_loader:
        x = x.to(device, non_blocking=True).contiguous()
        y = y.to(device, non_blocking=True).contiguous()

        optimizer.zero_grad(set_to_none=True)

        p = model(x)  # probs
        p = torch.clamp(p, 1e-6, 1.0)
        p = p / p.sum(dim=1, keepdim=True)

        loss = criterion(torch.log(p), y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    val_loss = run_eval(val_loader)
    print(
        f"epoch {epoch+1}/{EPOCHS} | train_loss={np.mean(train_losses):.6f} | val_loss={val_loss:.6f}"
    )

model.eval()




## === cell 9
class TestDataset(Dataset):
    def __init__(self, eeg_ids, data_dir: Path):
        self.eeg_ids = [str(int(e)) for e in eeg_ids]
        self.data_dir = data_dir
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        fp = self.data_dir / f"{eeg_id}.parquet"

        arr = _read_eeg_parquet_cached(fp, self._cache, cache_max_items=256)

        x = torch.from_numpy(arr)
        return x, eeg_id


def pad_collate_test(batch):
    xs, eeg_ids = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True)
    return x_pad.contiguous(), list(eeg_ids)


test_dataset = TestDataset(test_df["eeg_id"].values, TEST_EEG_DIR)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=pad_collate_test,
    persistent_workers=True,
    prefetch_factor=4,
)

len(test_dataset), test_df.shape



## === cell 10
predictions_list = []
eeg_ids = []

model.eval()
with torch.no_grad():
    for inputs, eeg_id_batch in test_loader:
        inputs = inputs.to(device, non_blocking=True).contiguous()
        outputs = model(inputs)  # already probabilities

        p = outputs.detach().cpu().numpy()
        p = np.clip(p, 1e-6, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        predictions_list.append(p)
        eeg_ids.extend(eeg_id_batch)

predictions = (
    np.concatenate(predictions_list, axis=0)
    if predictions_list
    else np.zeros((0, len(VOTE_COLS)), dtype=np.float32)
)
len(eeg_ids), predictions.shape



## === cell 11
pred_df = pd.DataFrame(predictions, columns=VOTE_COLS)
pred_df.insert(0, "eeg_id", [int(e) for e in eeg_ids])

sub = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

missing_mask = sub[VOTE_COLS].isna().any(axis=1)
if missing_mask.any():
    sub.loc[missing_mask, VOTE_COLS] = 1.0 / len(VOTE_COLS)

vals = sub[VOTE_COLS].values.astype(np.float64)
vals = np.clip(vals, 1e-6, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[VOTE_COLS] = vals

assert len(sub) == len(sample_sub), (len(sub), len(sample_sub))
row_sums = sub[VOTE_COLS].sum(axis=1).values
assert np.allclose(row_sums, 1.0, atol=1e-6), (row_sums.min(), row_sums.max())

sub.head()



## === cell 12
out_path = PATH_OUT / "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub), "cols:", sub.shape[1])
out_path
