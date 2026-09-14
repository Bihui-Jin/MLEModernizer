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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

0.586116777154513

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import random
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skimage.transform import resize
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def resolve_path(relative_path: str) -> str:
    """
    Return an existing absolute path for a given relative path.
    Works for Kaggle (`/kaggle/input/...`), the repository layout
    (`kaggle/data/...`), and a local notebook where data lives under `./input/...`.
    """
    candidates = [
        os.path.join("/kaggle/input", relative_path),  # Kaggle default
        os.path.join("input", relative_path),  # Local input folder
        os.path.join("kaggle", "data", relative_path),  # Provided Kaggle/data tree
        os.path.join("./kaggle", "data", relative_path),  # Explicit relative path
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not locate {relative_path} in any known location.")


class SimpleMRIDataset(Dataset):
    """Simplified dataset for MRI data with basic preprocessing."""

    def __init__(self, df, data_dir, target_size=(32, 32, 32)):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["BraTS21ID"]).zfill(5)
        patient_path = os.path.join(self.data_dir, patient_id)

        modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        volume_data = []

        for modality in modalities:
            modality_path = os.path.join(patient_path, modality)
            dicom_files = (
                sorted([f for f in os.listdir(modality_path) if f.endswith(".dcm")])
                if os.path.isdir(modality_path)
                else []
            )

            if len(dicom_files) == 0:
                volume = np.zeros(self.target_size, dtype=np.float32)
            else:
                try:
                    middle_idx = len(dicom_files) // 2
                    dicom_file = os.path.join(modality_path, dicom_files[middle_idx])
                    ds = pydicom.dcmread(dicom_file)
                    slice_data = ds.pixel_array.astype(np.float32)

                    slice_resized = resize(
                        slice_data, self.target_size[:2], preserve_range=True
                    )
                    volume = np.repeat(
                        slice_resized[:, :, np.newaxis], self.target_size[2], axis=2
                    )
                    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                except Exception:
                    volume = np.zeros(self.target_size, dtype=np.float32)

            volume_data.append(volume)

        multi_modal_volume = np.stack(volume_data, axis=0)  # (4, D, H, W)
        return torch.FloatTensor(multi_modal_volume), torch.FloatTensor(
            [row["MGMT_value"]]
        )


class Simple3DCNN(nn.Module):
    """Simplified 3D CNN for better performance."""

    def __init__(self, in_channels=4, num_classes=1):
        super(Simple3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_and_train_model(train_df, val_df, train_data_dir):
    """Train the 3D CNN with a modest number of epochs to stay within runtime limits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SimpleMRIDataset(train_df, train_data_dir)
    val_dataset = SimpleMRIDataset(val_df, train_data_dir)

    batch_size = 8
    num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = Simple3DCNN(in_channels=4, num_classes=1).to(device)

    pos = train_df["MGMT_value"].sum()
    neg = len(train_df) - pos
    if pos > 0:
        pos_weight = torch.tensor(neg / pos, device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 5
    best_val_auc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy().flatten())

        val_auc = roc_auc_score(val_labels, val_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_auc




## === cell 1
train_labels_path = resolve_path(
    "rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
train_data_dir = resolve_path(
    "rsna-miccai-brain-tumor-radiogenomic-classification/train"
)
test_data_dir = resolve_path("rsna-miccai-brain-tumor-radiogenomic-classification/test")

train_labels = pd.read_csv(train_labels_path)

exclude_ids = ["00109", "00123", "00709"]
train_labels = train_labels[~train_labels["BraTS21ID"].astype(str).isin(exclude_ids)]

train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels["MGMT_value"],
)

model, best_val_auc = build_and_train_model(train_df, val_df, train_data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

val_dataset = SimpleMRIDataset(val_df, train_data_dir)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

val_predictions = []
patient_ids = []

with torch.no_grad():
    for i, (images, _) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        start_idx = i * val_loader.batch_size
        end_idx = start_idx + len(probs)
        batch_ids = val_df.iloc[start_idx:end_idx]["BraTS21ID"].values

        for pid, prob in zip(batch_ids, probs):
            patient_ids.append(str(pid).zfill(5))
            val_predictions.append(float(prob))

val_df_out = pd.DataFrame({"BraTS21ID": patient_ids, "MGMT_value": val_predictions})
val_df_out.to_csv("validation_predictions.csv", index=False)

test_patient_ids = sorted(
    [
        f
        for f in os.listdir(test_data_dir)
        if os.path.isdir(os.path.join(test_data_dir, f))
    ]
)

test_predictions = []
for patient_id in test_patient_ids:
    patient_path = os.path.join(test_data_dir, patient_id)
    try:
        modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        volume_data = []
        for modality in modalities:
            modality_path = os.path.join(patient_path, modality)
            dicom_files = (
                sorted([f for f in os.listdir(modality_path) if f.endswith(".dcm")])
                if os.path.isdir(modality_path)
                else []
            )

            if len(dicom_files) == 0:
                volume = np.zeros((32, 32, 32), dtype=np.float32)
            else:
                try:
                    middle_idx = len(dicom_files) // 2
                    dicom_file = os.path.join(modality_path, dicom_files[middle_idx])
                    ds = pydicom.dcmread(dicom_file)
                    slice_data = ds.pixel_array.astype(np.float32)
                    slice_resized = resize(slice_data, (32, 32), preserve_range=True)
                    volume = np.repeat(slice_resized[:, :, np.newaxis], 32, axis=2)
                    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                except Exception:
                    volume = np.zeros((32, 32, 32), dtype=np.float32)

            volume_data.append(volume)

        multi_modal_volume = np.stack(volume_data, axis=0)  # (4,32,32,32)
        image_tensor = torch.FloatTensor(multi_modal_volume).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(image_tensor)
            prob = torch.sigmoid(out).cpu().numpy()[0, 0]

        test_predictions.append({"BraTS21ID": patient_id, "MGMT_value": float(prob)})
    except Exception:
        test_predictions.append({"BraTS21ID": patient_id, "MGMT_value": 0.5})

test_df_out = pd.DataFrame(test_predictions)
test_df_out.to_csv("submission.csv", index=False)

print(f"Best validation ROC-AUC: {best_val_auc:.4f}")
print(f"Validation predictions saved: {len(val_df_out)} patients")
print(f"Test predictions saved: {len(test_df_out)} patients")
