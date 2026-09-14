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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedGroupKFold

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)



## === cell 2
base_path = "/kaggle/input/hms-harmful-brain-activity-classification"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "test.csv")
sample_submission_csv_path = os.path.join(base_path, "sample_submission.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_raw = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_submission_csv_path)

grp_cols = ["eeg_id", "spectrogram_id", "patient_id", "expert_consensus"]
vote_agg = train_raw.groupby(grp_cols, as_index=False)[TARGET_COLS].sum()
df = vote_agg.copy()

df["eeg_path"] = base_path + "/train_eegs/" + df["eeg_id"].astype(str) + ".parquet"
df["spec_path"] = (
    base_path + "/train_spectrograms/" + df["spectrogram_id"].astype(str) + ".parquet"
)
df["class_name"] = df["expert_consensus"].copy()

class_name_to_label = {
    "Seizure": 0,
    "LPD": 1,
    "GPD": 2,
    "LRDA": 3,
    "GRDA": 4,
    "Other": 5,
}
df["class_label"] = df["class_name"].map(class_name_to_label)

test["eeg_path"] = base_path + "/test_eegs/" + test["eeg_id"].astype(str) + ".parquet"
test["spec_path"] = (
    base_path + "/test_spectrograms/" + test["spectrogram_id"].astype(str) + ".parquet"
)
if "spectrogram_label_offset_seconds" not in test.columns:
    test["spectrogram_label_offset_seconds"] = 0.0
if "class_label" not in test.columns:
    test["class_label"] = 0

print("Train/val pool rows:", len(df), "Kaggle test rows:", len(test))



## === cell 3
df.head()



## === cell 4
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_classes = 6
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
model = model.to(device)



## === cell 5

_PARQUET_ENGINE = None
_HAVE_PYARROW = False
_HAVE_FASTPARQUET = False
try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401

    _HAVE_PYARROW = True
    _PARQUET_ENGINE = "pyarrow"
except Exception:
    _HAVE_PYARROW = False

if not _HAVE_PYARROW:
    try:
        import fastparquet  # noqa: F401

        _HAVE_FASTPARQUET = True
        _PARQUET_ENGINE = "fastparquet"
    except Exception:
        _HAVE_FASTPARQUET = False
        _PARQUET_ENGINE = None

_TABLE_CACHE = {}
_TABLE_CACHE_ORDER = []
_TABLE_CACHE_MAX_ITEMS = 96  # tuned: spectrogram reuse is high; keep bounded memory


def _table_cache_get(path):
    return _TABLE_CACHE.get(path, None)


def _table_cache_put(path, table):
    if path in _TABLE_CACHE:
        return
    _TABLE_CACHE[path] = table
    _TABLE_CACHE_ORDER.append(path)
    if len(_TABLE_CACHE_ORDER) > _TABLE_CACHE_MAX_ITEMS:
        old = _TABLE_CACHE_ORDER.pop(0)
        _TABLE_CACHE.pop(old, None)


def read_parquet_subset(parquet_file_path, offset_seconds, length, is_eeg=True):
    offset_seconds = float(offset_seconds) if offset_seconds is not None else 0.0
    start_row = int(offset_seconds * 200) if is_eeg else int(offset_seconds / 2)
    end_row = start_row + (10000 if is_eeg else 300)

    if _HAVE_PYARROW:
        table = _table_cache_get(parquet_file_path)
        if table is None:
            if is_eeg:
                table = pq.read_table(parquet_file_path)
            else:
                pf = pq.ParquetFile(parquet_file_path)
                cols = [c for c in pf.schema.names if c != "time"]
                table = pf.read(columns=cols)
            _table_cache_put(parquet_file_path, table)

        sliced = table.slice(start_row, end_row - start_row)
        return sliced.to_pandas(types_mapper=None)

    cache_key = (parquet_file_path, start_row, end_row, is_eeg)
    cached = _TABLE_CACHE.get(cache_key, None)
    if cached is not None:
        return cached

    if _PARQUET_ENGINE is None:
        dfp = pd.read_parquet(parquet_file_path)
    else:
        dfp = pd.read_parquet(parquet_file_path, engine=_PARQUET_ENGINE)

    if (not is_eeg) and ("time" in dfp.columns):
        dfp = dfp.drop(columns=["time"])
    out = dfp.iloc[start_row:end_row]
    _TABLE_CACHE[cache_key] = out
    if len(_TABLE_CACHE) > (_TABLE_CACHE_MAX_ITEMS * 2):
        _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
    return out


def convert_2d_to_3d(data_2d: pd.DataFrame):
    x = data_2d.to_numpy(copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
    x = np.repeat(x[:, :, None], 3, axis=2)
    return x


class SPECTROGRAM_Dataset(Dataset):
    """
    Return soft target distribution from vote columns when available.
    This aligns KLDivLoss with the competition's KL metric while keeping your model and loop structure.
    """

    def __init__(
        self, dataframe: pd.DataFrame, transform, return_soft_targets: bool = True
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_soft_targets = return_soft_targets
        self.has_votes = all(c in self.dataframe.columns for c in TARGET_COLS)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        spec_data = read_parquet_subset(
            row["spec_path"],
            row.get("spectrogram_label_offset_seconds", 0.0),
            300,
            is_eeg=False,
        )
        spec_data_3d = convert_2d_to_3d(spec_data)
        data_img = Image.fromarray(spec_data_3d, mode="RGB")
        features = self.transform(data_img)

        if self.return_soft_targets and self.has_votes:
            votes = row[TARGET_COLS].to_numpy(dtype=np.float32)
            s = float(votes.sum())
            if s <= 0:
                target = np.ones(6, dtype=np.float32) / 6.0
            else:
                target = votes / s
            target = torch.tensor(target, dtype=torch.float32)
            label = int(row["class_label"]) if pd.notnull(row["class_label"]) else 0
            return features, target, label
        else:
            label = int(row["class_label"]) if pd.notnull(row["class_label"]) else 0
            return features, label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 6
lr = 0.001
num_epochs = 100
batch_size = 32
factor = 0.8
num_workers = 2



## === cell 7
labels = df["class_label"].values
groups = df["eeg_id"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

train_idx, val_idx = next(sgkf.split(X=df, y=labels, groups=groups))
train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[
    val_idx
].reset_index(drop=True)

train_dataset = SPECTROGRAM_Dataset(train_df, transform, return_soft_targets=True)
val_dataset = SPECTROGRAM_Dataset(val_df, transform, return_soft_targets=True)
test2_dataset = SPECTROGRAM_Dataset(test, transform, return_soft_targets=False)

loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
test2_loader = DataLoader(
    test2_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

print(f"Size of train: {len(train_dataset)}")
print(f"Size of val: {len(val_dataset)}")
print(f"Size of test2 (Kaggle test): {len(test2_dataset)}")



## === cell 8
criterion = torch.nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=factor, patience=1)



## === cell 9
best_val_loss = float("inf")
training_losses = []
validation_losses = []
accumulation_steps = 4

use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)
amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    batch_count = 0

    for i, batch in enumerate(train_loader):
        features, target_dist, _ = batch
        features = features.to(device, non_blocking=True)
        target_dist = target_dist.to(device, non_blocking=True)

        with autocast(device_type=amp_device_type, enabled=use_amp):
            outputs = model(features)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, target_dist) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item() * features.size(0)
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")

    train_loss /= len(train_loader.dataset)
    training_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            features, target_dist, _ = batch
            features = features.to(device, non_blocking=True)
            target_dist = target_dist.to(device, non_blocking=True)
            with autocast(device_type=amp_device_type, enabled=use_amp):
                outputs = model(features)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, target_dist)
            val_loss += loss.item() * features.size(0)

    val_loss /= len(val_loader.dataset)
    validation_losses.append(val_loss)
    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved with Validation Loss: {val_loss:.4f}")

if not os.path.exists("best_model.pth"):
    torch.save(model.state_dict(), "best_model.pth")
    print("WARNING: best_model.pth was missing; saved current model state.")



## === cell 10
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_probs = []
with torch.no_grad():
    for features, _ in test2_loader:
        features = features.to(device, non_blocking=True)
        outputs = model(features)
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs.detach().cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)

eps = 1e-7
all_probs = np.clip(all_probs, eps, 1.0)
all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)

print(all_probs.shape)



## === cell 11
pred_df = pd.DataFrame(all_probs, columns=TARGET_COLS)
pred_df["eeg_id"] = test["eeg_id"].values

submission = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

missing_mask = submission[TARGET_COLS].isna().any(axis=1)
if missing_mask.any():
    submission.loc[missing_mask, TARGET_COLS] = 1.0 / 6.0

vals = submission[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, eps, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
submission[TARGET_COLS] = vals

submission = submission[["eeg_id"] + TARGET_COLS]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    "Row sums (min/max):",
    submission[TARGET_COLS].sum(axis=1).min(),
    submission[TARGET_COLS].sum(axis=1).max(),
)
print("Saved submission.csv")
