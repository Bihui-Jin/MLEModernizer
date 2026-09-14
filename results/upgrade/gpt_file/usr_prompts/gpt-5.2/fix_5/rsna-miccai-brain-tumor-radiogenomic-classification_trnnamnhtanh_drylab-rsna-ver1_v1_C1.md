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

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.47529) has done: 'I first fix why you aren’t getting a Kaggle score by ensuring the script always produces a valid `submission.csv` with the exact row order/IDs from `sample_submission.csv` (this avoids missing/extra IDs or formatting mismatches). Then I make a minimal correctness tweak in the training loop: `BCEWithLogitsLoss` expects `labels` to have the same shape as `outputs`, so I reshape labels to `(B,1)` to prevent silent broadcasting and unstable learning that can hurt AUC. Finally, I reuse the exact same preprocessing code path for test inference by using the existing `SimpleMRIDataset` (same core logic, just less duplication), which reduces train/test preprocessing drift and should improve AUC toward the target without changing the model architecture or training approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import warnings

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class SimpleMRIDataset(Dataset):
    """Simplified dataset for MRI data with basic preprocessing + caching."""

    _middle_dicom_path_cache = (
        {}
    )  # (data_dir, patient_id, modality) -> dicom_path or None
    _tensor_cache = (
        {}
    )  # (data_dir, patient_id, target_size) -> torch.FloatTensor volume (4, D, H, W)

    def __init__(
        self, df, data_dir, target_size=(32, 32, 32), cache_tensors: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.target_size = tuple(target_size)
        self.cache_tensors = cache_tensors

    @property
    def has_target(self) -> bool:
        return "MGMT_value" in self.df.columns

    def __len__(self):
        return len(self.df)

    def _sorted_dicom_files(self, modality_path: str):
        try:
            files = [f for f in os.listdir(modality_path) if f.endswith(".dcm")]
        except Exception:
            return []

        if not files:
            return []

        def dicom_sort_key(fname: str):
            fpath = os.path.join(modality_path, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                inst = getattr(ds, "InstanceNumber", None)
                if inst is not None:
                    return (0, int(inst))
            except Exception:
                pass
            return (1, fname)

        return [f for f in sorted(files, key=dicom_sort_key)]

    def _get_middle_dicom_path(self, patient_id: str, modality: str):
        key = (self.data_dir, patient_id, modality)
        if key in SimpleMRIDataset._middle_dicom_path_cache:
            return SimpleMRIDataset._middle_dicom_path_cache[key]

        patient_path = os.path.join(self.data_dir, patient_id)
        modality_path = os.path.join(patient_path, modality)
        dicom_files = self._sorted_dicom_files(modality_path)
        if len(dicom_files) == 0:
            dicom_path = None
        else:
            middle_idx = len(dicom_files) // 2
            dicom_path = os.path.join(modality_path, dicom_files[middle_idx])

        SimpleMRIDataset._middle_dicom_path_cache[key] = dicom_path
        return dicom_path

    def _build_patient_tensor(self, patient_id: str):
        modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        volume_data = []
        ts = self.target_size

        for modality in modalities:
            dicom_path = self._get_middle_dicom_path(patient_id, modality)
            if dicom_path is None:
                volume = np.zeros(ts, dtype=np.float32)
            else:
                try:
                    ds = pydicom.dcmread(dicom_path, force=True)
                    slice_data = ds.pixel_array.astype(np.float32)
                    slice_resized = resize(
                        slice_data,
                        ts[:2],
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.float32, copy=False)

                    volume = np.repeat(slice_resized[:, :, np.newaxis], ts[2], axis=2)
                    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
                except Exception:
                    volume = np.zeros(ts, dtype=np.float32)

            volume_data.append(volume)

        multi_modal_volume = np.stack(volume_data, axis=0)  # (4, H, W, D)
        return torch.from_numpy(multi_modal_volume).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["BraTS21ID"]).zfill(5)

        tensor_key = (self.data_dir, patient_id, self.target_size)
        if self.cache_tensors and tensor_key in SimpleMRIDataset._tensor_cache:
            x = SimpleMRIDataset._tensor_cache[tensor_key]
        else:
            x = self._build_patient_tensor(patient_id)
            if self.cache_tensors:
                SimpleMRIDataset._tensor_cache[tensor_key] = x

        y = float(row["MGMT_value"]) if self.has_target else 0.0
        return x, torch.FloatTensor([y])


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
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


def _seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_and_train_model(train_df, val_df, train_data_dir):
    """Build and train the 3D CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SimpleMRIDataset(train_df, train_data_dir, cache_tensors=True)
    val_dataset = SimpleMRIDataset(val_df, train_data_dir, cache_tensors=True)

    batch_size = 8
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    model = Simple3DCNN(in_channels=4, num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)

            if labels.ndim == 1:
                labels = labels.view(-1, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)

                if labels.ndim == 1:
                    labels = labels.view(-1, 1)

                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                val_preds.extend(preds.ravel().tolist())
                val_labels.extend(labels.detach().cpu().numpy().ravel().tolist())

        val_auc = roc_auc_score(np.array(val_labels), np.array(val_preds))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_auc




## === cell 1
if __name__ == "__main__":
    seed_everything(42)

    train_labels_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    train_data_dir = (
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
    )
    test_data_dir = (
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    )
    sample_sub_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"

    train_labels = pd.read_csv(train_labels_path)
    train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)

    exclude_ids = ["00109", "00123", "00709"]
    train_labels = train_labels[
        ~train_labels["BraTS21ID"].isin(exclude_ids)
    ].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels["MGMT_value"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    model, best_val_auc = build_and_train_model(train_df, val_df, train_data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    val_dataset = SimpleMRIDataset(val_df, train_data_dir, cache_tensors=True)
    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    val_predictions = np.empty(len(val_df), dtype=np.float32)
    patient_ids = val_df["BraTS21ID"].astype(str).str.zfill(5).values

    with torch.no_grad():
        offset = 0
        for images, _labels in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = (
                torch.sigmoid(outputs)
                .detach()
                .cpu()
                .numpy()
                .ravel()
                .astype(np.float32, copy=False)
            )
            val_predictions[offset : offset + len(probs)] = probs
            offset += len(probs)

    val_predictions_df = pd.DataFrame(
        {"BraTS21ID": patient_ids, "MGMT_value": val_predictions}
    )
    val_predictions_df.to_csv("validation_predictions.csv", index=False)

    sample_sub = pd.read_csv(sample_sub_path)
    sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

    test_df = sample_sub.copy()
    test_df["MGMT_value"] = 0.0  # placeholder required by Dataset
    test_dataset = SimpleMRIDataset(test_df, test_data_dir, cache_tensors=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    test_probs = []
    try:
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel().tolist()
                test_probs.extend(probs)
    except Exception as e:
        print(f"Inference error encountered: {e}")
        test_probs = [0.5] * len(sample_sub)

    if len(test_probs) != len(sample_sub):
        if len(test_probs) < len(sample_sub):
            fill_value = float(np.mean(test_probs)) if len(test_probs) else 0.5
            test_probs = test_probs + [fill_value] * (len(sample_sub) - len(test_probs))
        else:
            test_probs = test_probs[: len(sample_sub)]

    submission = pd.DataFrame(
        {
            "BraTS21ID": sample_sub["BraTS21ID"].values,
            "MGMT_value": np.array(test_probs, dtype=np.float32),
        }
    )
    submission.to_csv("submission.csv", index=False)

    print(f"Best validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Validation predictions saved: {len(val_predictions_df)} patients")
    print(f"Test predictions saved: {len(submission)} patients")
    print("Wrote: submission.csv")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1386164010.py in <cell line: 0>()
     28     val_df = val_df.reset_index(drop=True)
     29 
---> 30     model, best_val_auc = build_and_train_model(train_df, val_df, train_data_dir)
     31 
     32     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

/tmp/ipykernel_55/1625042689.py in build_and_train_model(train_df, val_df, train_data_dir)
    262 
    263             loss = criterion(outputs, labels)
--> 264             loss.backward()
    265             optimizer.step()
    266 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: adaptive_avg_pool3d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
