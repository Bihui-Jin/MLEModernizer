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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np
import pandas as pd

import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm



## === cell 1
INPUT_IMAGE_SIZE = (400, 400)
TRAIN_SIZE = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 3  # short training to stay within time limits
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")



## === cell 2
train_eeg_files = glob.glob(
    "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/*"
)
test_eeg_files = glob.glob(
    "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/*"
)
train_spectrogram_files = glob.glob(
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/*"
)
test_spectrogram_files = glob.glob(
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/*"
)
print(f"Total number of files in train_eegs: {len(train_eeg_files)}")
print(f"Total number of files in test_eegs: {len(test_eeg_files)}")
print(f"Total number of files in train_spectrograms: {len(train_spectrogram_files)}")
print(f"Total number of files in test_spectrograms: {len(test_spectrogram_files)}")



## === cell 3
train_csv = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
test_csv = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
)
sample_csv = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
print(f"number of entries in train_csv {len(train_csv)}")
print(f"number of entries in test_csv {len(test_csv)}")
print(f"number of entries in sample_csv {len(sample_csv)}")



## === cell 4
columns_to_drop = [
    "eeg_id",
    "eeg_sub_id",
    "eeg_label_offset_seconds",
    "spectrogram_sub_id",
    "spectrogram_label_offset_seconds",
    "label_id",
    "patient_id",
    "expert_consensus",
]



## === cell 5
spectro_csv = (
    train_csv[train_csv["spectrogram_sub_id"] == 0]
    .drop(columns_to_drop, axis=1)
    .reset_index(drop=True)
)




## === cell 6
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, "constant")




## === cell 7
class HMS_dataset(Dataset):
    def __init__(
        self, csv_file, root_dir, image_size=(400, 400), targets_available=True
    ):
        self.hms_dataset = csv_file
        self.root_dir = root_dir
        self.image_size = image_size
        self.targets_available = targets_available
        self.transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self):
        return len(self.hms_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(
            self.root_dir, str(self.hms_dataset.iloc[idx, 0]) + ".parquet"
        )
        spectro = np.log(
            pd.read_parquet(file_name)
            .drop("time", axis=1)
            .fillna(0)
            .replace(0, 1e-6)
            .to_numpy()
            .T
        )
        spectro_pil = Image.fromarray(spectro)
        spectro_torch = self.transform(spectro_pil)
        if self.targets_available:
            votes_label = self.hms_dataset.iloc[idx, 1:].to_numpy()
            torch_labels = torch.tensor(votes_label, dtype=torch.float)
            sample = (spectro_torch, torch_labels)
        else:
            sample = spectro_torch

        return sample




## === cell 8
root_train = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
hms_data = HMS_dataset(spectro_csv, root_train, INPUT_IMAGE_SIZE)
print(len(hms_data))



## === cell 9
train_size = int(TRAIN_SIZE * len(hms_data))
test_size = len(hms_data) - train_size
train_set, val_set = torch.utils.data.random_split(hms_data, [train_size, test_size])
print(len(train_set), len(val_set))



## === cell 10
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)




## === cell 11
class Conv_net(nn.Module):
    def __init__(self, n_classes):
        super(Conv_net, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_net = nn.Sequential(
            nn.Linear(128 * 10 * 10, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        return x




## === cell 12
model = Conv_net(6).to(DEVICE)



## === cell 13
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.KLDivLoss(
    reduction="batchmean"
)  # expects log‑probs input, prob target




## === cell 14
def train_one_epoch(model, optimizer, loss_fn, data_loader):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(data_loader, 1):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        preds = F.log_softmax(model(x), dim=1)
        target = F.softmax(y, dim=1)
        loss = loss_fn(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(
                f"train batch {batch_idx}/{len(data_loader)} loss {(total_loss/10):.4f}"
            )
            total_loss = 0.0


def validate(model, loss_fn, data_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            preds = F.log_softmax(model(x), dim=1)
            target = F.softmax(y, dim=1)
            loss = loss_fn(preds, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)




## === cell 15
for epoch in range(1, EPOCHS + 1):
    print(f"=== Epoch {epoch}/{EPOCHS} ===")
    train_one_epoch(model, optimizer, loss_fn, train_loader)
    val_loss = validate(model, loss_fn, val_loader)
    print(f"Validation KLDivLoss: {val_loss:.4f}")



## === cell 17
model.eval()



## === cell 18
spectro_test_csv = test_csv.drop(["eeg_id", "patient_id"], axis=1)
root_test = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
test_hms_data = HMS_dataset(spectro_test_csv, root_test, targets_available=False)
test_loader = DataLoader(dataset=test_hms_data, batch_size=BATCH_SIZE, shuffle=False)



## === cell 19
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference", ncols=100):
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        all_preds.append(probs.cpu().numpy())
test_preds = np.vstack(all_preds)  # shape (num_test, 6)



## === cell 20
submission = pd.DataFrame(test_preds, columns=sample_csv.columns[1:])
submission.insert(0, "eeg_id", test_csv["eeg_id"].values)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission.shape)
