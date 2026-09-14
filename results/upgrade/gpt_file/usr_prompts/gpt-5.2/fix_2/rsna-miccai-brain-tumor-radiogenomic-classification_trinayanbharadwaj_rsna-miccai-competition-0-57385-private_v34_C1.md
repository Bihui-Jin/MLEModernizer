# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pydicom as dicom
from skimage.transform import resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

IMG_PX_SIZE = (
    224  # keep modest for speed/memory; core logic still "resize slice to fixed square"
)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BAD_TRAIN_CASES = {"00109", "00123", "00709"}


def list_case_ids(path_dir):
    return sorted([d.name for d in os.scandir(path_dir) if d.is_dir()])


train_ids = [cid for cid in list_case_ids(TRAIN_DIR) if cid not in BAD_TRAIN_CASES]
test_ids = list_case_ids(TEST_DIR)

print("Train cases:", len(train_ids), "Test cases:", len(test_ids))

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_map = dict(
    zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(np.float32))
)

missing = [cid for cid in train_ids if cid not in labels_map]
print("Missing labels:", len(missing))




## === cell 2
def _sorted_modality_dirs(case_dir):
    return sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])


def _choose_slice_from_modality(modality_dir, min_sum=100000, min_norm_sum=5000):
    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(modality_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    for fp in dcm_files:
        ds = dicom.dcmread(fp)
        arr = ds.pixel_array.astype(np.float32)
        if arr.sum() <= min_sum:
            continue
        arr_rs = resize(
            arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        mx = float(arr_rs.max())
        if mx <= 0:
            continue
        arr_rs = arr_rs / mx
        if arr_rs.sum() <= min_norm_sum:
            continue
        return arr_rs
    if len(dcm_files) == 0:
        return np.zeros((IMG_PX_SIZE, IMG_PX_SIZE), dtype=np.float32)
    mid = dcm_files[len(dcm_files) // 2]
    ds = dicom.dcmread(mid)
    arr = ds.pixel_array.astype(np.float32)
    arr_rs = resize(
        arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    mx = float(arr_rs.max())
    return arr_rs / mx if mx > 0 else arr_rs


def load_case_image(case_root, modality_index):
    """
    Returns HxWx3 float32 image for a single case and modality index:
    0: FLAIR, 1: T1w, 2: T1wCE, 3: T2w
    """
    mods = _sorted_modality_dirs(case_root)
    if len(mods) <= modality_index:
        arr = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE), dtype=np.float32)
    else:
        arr = _choose_slice_from_modality(mods[modality_index])
    stacked = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
    return stacked




## === cell 3
class RSNADataset(Dataset):
    def __init__(self, case_ids, root_dir, labels_map=None, modality_index=0):
        self.case_ids = case_ids
        self.root_dir = root_dir
        self.labels_map = labels_map
        self.modality_index = modality_index

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        case_root = os.path.join(self.root_dir, cid)
        img = load_case_image(case_root, self.modality_index)  # H W 3 float32 in [0,1]
        x = torch.from_numpy(img.transpose(2, 0, 1)).float()
        if self.labels_map is None:
            return cid, x
        y = torch.tensor(self.labels_map[cid]).float()
        return cid, x, y




## === cell 4
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x).flatten(1)
        x = self.head(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 5
def train_one_modality(modality_index, epochs=3, batch_size=8, lr=1e-3):
    rng = np.random.RandomState(SEED + modality_index)
    shuffled = train_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(0.1 * len(shuffled)))
    val_ids = shuffled[:n_val]
    tr_ids = shuffled[n_val:]

    train_ds = RSNADataset(
        tr_ids, TRAIN_DIR, labels_map=labels_map, modality_index=modality_index
    )
    val_ds = RSNADataset(
        val_ids, TRAIN_DIR, labels_map=labels_map, modality_index=modality_index
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for _, x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * x.size(0)
        tr_loss /= max(1, len(train_ds))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for _, x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).unsqueeze(1)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += float(loss.item()) * x.size(0)
        va_loss /= max(1, len(val_ds))

        print(
            f"Modality {modality_index} Epoch {ep+1}/{epochs} - train_loss={tr_loss:.4f} val_loss={va_loss:.4f}"
        )

    return model


model_flair = train_one_modality(modality_index=0, epochs=3, batch_size=8, lr=1e-3)
model_t2w = train_one_modality(modality_index=3, epochs=3, batch_size=8, lr=1e-3)




## === cell 6
def predict_modality(model, modality_index, batch_size=8):
    ds = RSNADataset(test_ids, TEST_DIR, labels_map=None, modality_index=modality_index)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_ids = []
    all_probs = []
    with torch.no_grad():
        for cids, x in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            all_ids.extend(list(cids))
            all_probs.extend(list(probs))
    return np.array(all_ids), np.array(all_probs, dtype=np.float32)


ids1, p1 = predict_modality(model_flair, modality_index=0, batch_size=8)
ids4, p4 = predict_modality(model_t2w, modality_index=3, batch_size=8)

assert (ids1 == ids4).all(), "Test ID ordering mismatch between modalities"



## === cell 7
pred = (p1.astype(np.float32) + p4.astype(np.float32)) / 2.0
pred = np.clip(pred, 1e-6, 1 - 1e-6)

sub_df = pd.DataFrame({"BraTS21ID": ids1, "MGMT_value": pred})
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str)

sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(np.float32)

print(sub_df.head())
print("Submission rows:", len(sub_df), "NaNs:", sub_df["MGMT_value"].isna().sum())



## === cell 8
try:
    ds_vis = RSNADataset(test_ids[:10], TEST_DIR, labels_map=None, modality_index=3)
    plt.figure(figsize=(12, 6))
    for i in range(min(6, len(ds_vis))):
        cid, x = ds_vis[i]
        img = x.numpy().transpose(1, 2, 0)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(cid)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Visualization skipped due to:", repr(e))



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print("Columns:", list(sub_df.columns))
print(
    "Min/Max pred:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)
