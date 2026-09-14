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

# 5. Target score

1.0598384737455746

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
from sklearn.metrics import classification_report

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

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

temp = train_raw[["eeg_id", "expert_consensus"]].drop_duplicates()
df = train_raw.loc[temp.index].reset_index(drop=True)

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

eeg_ids = []
for class_label in df["class_label"].dropna().unique():
    eeg_id = df[df["class_label"] == class_label].iloc[0]["eeg_id"]
    eeg_ids.append(eeg_id)

test_df = df[df["eeg_id"].isin(eeg_ids)]  # quick internal test
df = df[~df["eeg_id"].isin(eeg_ids)]  # train & val
df = df.iloc[:1000,].reset_index(drop=True)



## === cell 3
df.head()



## === cell 4
model = models.efficientnet_v2_l(weights=None)
num_classes = 6
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)

weights_path = "/kaggle/input/efficientnet-v2-l/efficientnet_v2_l-59c71312.pth"
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location="cpu")
    state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("classifier.1")
    }
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded external weights from: {weights_path}")
else:
    print(f"WARNING: weights not found at {weights_path}. Training from scratch.")

model = model.to(device)




## === cell 5
def read_parquet_subset(parquet_file_path, offset_seconds, length, is_eeg=True):
    offset_seconds = int(offset_seconds)
    start_row = int(offset_seconds * 200) if is_eeg else int(offset_seconds / 2)
    end_row = start_row + (10000 if is_eeg else 300)
    dfp = pd.read_parquet(parquet_file_path)
    return dfp.iloc[start_row:end_row]


def convert_2d_to_3d(data_2d: pd.DataFrame):
    x = data_2d.to_numpy()
    x = np.clip(x, 0, 255)
    x = np.nan_to_num(x)
    x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
    return x.astype(np.uint8)


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
            row["spectrogram_label_offset_seconds"],
            300,
            is_eeg=False,
        ).copy()

        if "time" in spec_data.columns:
            spec_data.drop("time", axis=1, inplace=True)

        spec_data_3d = convert_2d_to_3d(spec_data)
        data_img = Image.fromarray(spec_data_3d)
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

for train_idx, val_idx in sgkf.split(X=df, y=labels, groups=groups):
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

train_dataset = SPECTROGRAM_Dataset(train_df, transform, return_soft_targets=True)
val_dataset = SPECTROGRAM_Dataset(val_df, transform, return_soft_targets=True)

test_dataset = SPECTROGRAM_Dataset(test_df, transform, return_soft_targets=False)
test2_dataset = SPECTROGRAM_Dataset(test, transform, return_soft_targets=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
test2_loader = DataLoader(
    test2_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

print(f"Size of train: {len(train_dataset)}")
print(f"Size of val: {len(val_dataset)}")
print(f"Size of test: {len(test_dataset)}")
print(f"Size of test2: {len(test2_dataset)}")



## === cell 8
criterion = torch.nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=factor, patience=1)



## === cell 9
best_val_loss = float("inf")
training_losses = []
validation_losses = []
epochs_no_improve = 0
n_patience = 1
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
        if batch_count % 10 == 0:
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
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved with Validation Loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == n_patience:
        print("Early stopping triggered.")
        break

if not os.path.exists("best_model.pth"):
    torch.save(model.state_dict(), "best_model.pth")
    print("WARNING: best_model.pth was missing; saved current model state.")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4211790870.py in <cell line: 0>()
     24         target_dist = target_dist.to(device, non_blocking=True)
     25 
---> 26         with autocast(device_type=amp_device_type, enabled=use_amp):
     27             outputs = model(features)
     28             log_probs = F.log_softmax(outputs, dim=1)

/usr/local/lib/python3.11/dist-packages/typing_extensions.py in wrapper(*args, **kwargs)
   3002                 def wrapper(*args, **kwargs):
   3003                     warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
-> 3004                     return arg(*args, **kwargs)
   3005 
   3006                 if asyncio.coroutines.iscoroutinefunction(arg):

TypeError: autocast.__init__() got an unexpected keyword argument 'device_type'

## === cell 10
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(features)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())

print(
    classification_report(
        all_targets,
        all_preds,
        target_names=["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
        zero_division=0,
    )
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2162648005.py in <cell line: 0>()
      1 # Sanity-check classification report on the small internal "test_df"
----> 2 model.load_state_dict(torch.load("best_model.pth", map_location=device))
      3 model = model.to(device)
      4 model.eval()
      5 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'

## === cell 11
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/768588176.py in <cell line: 0>()
      1 # Predict on Kaggle test set
----> 2 model.load_state_dict(torch.load("best_model.pth", map_location=device))
      3 model = model.to(device)
      4 model.eval()
      5 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pth'

## === cell 12
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

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3064321164.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame(all_probs, columns=TARGET_COLS)
      2 pred_df["eeg_id"] = test["eeg_id"].values
      3 
      4 submission = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")
      5 

NameError: name 'all_probs' is not defined
