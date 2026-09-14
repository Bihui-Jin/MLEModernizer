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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.91331

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, pathlib

print("Skipping external pip installs (offline Kaggle environment).")



## === cell 1
from pathlib import Path

distill_code = r"""
import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Paths (Kaggle-standard)
ROOT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, "sample_submission.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def set_seed(seed: int = 42):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


class Plant2020Dataset(Dataset):
    def __init__(self, csv_path: str, train: bool, image_size: int = 224):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.train = train
        self.image_size = image_size

        if self.train:
            self.labels = self.df[TARGET_COLS].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df), 4), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # Minimal replacement for Resize->ToTensor->Normalize without torchvision dependency.
        img = img.resize((self.image_size, self.image_size))
        x = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
        x = np.transpose(x, (2, 0, 1))  # CHW

        # Use the same normalization constants as the original config.
        mean = np.array([0.49139968, 0.48215841, 0.44653091], dtype=np.float32)[:, None, None]
        std = np.array([0.24703223, 0.24348513, 0.26158784], dtype=np.float32)[:, None, None]
        x = (x - mean) / std
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        path = get_image_path(image_id)
        img = Image.open(path).convert("RGB")
        x = self._transform(img)
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y


class SmallCNN(nn.Module):
    # Minimal CNN to replace timm model (no external deps); outputs 4 logits (BCEWithLogitsLoss).
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


def mean_columnwise_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Local validation metric helper (not required for submission, but keeps semantics aligned).
    # Compute ROC AUC per column, then mean, skipping degenerate columns.
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        # If only one class present, ROC AUC is undefined; skip (shouldn't happen often here).
        if np.all(yt == 0) or np.all(yt == 1):
            continue
        # Rank-based AUC computation without sklearn:
        order = np.argsort(yp)
        yt_sorted = yt[order]
        n_pos = yt_sorted.sum()
        n_neg = len(yt_sorted) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        # Mann–Whitney U statistic
        ranks = np.arange(1, len(yt_sorted) + 1)
        sum_ranks_pos = ranks[yt_sorted == 1].sum()
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        auc = u / (n_pos * n_neg)
        aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else 0.0


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * x.size(0)
    return running / max(1, len(loader.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("-student_only", action="store_true")
    parser.add_argument("-test_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # Data
    full_train = pd.read_csv(TRAIN_CSV)
    # Deterministic split similar to original random_split lengths [0.8, 0.2] with seed 42
    rng = np.random.default_rng(42)
    idx = np.arange(len(full_train))
    rng.shuffle(idx)
    n_tr = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    tr_df = full_train.iloc[tr_idx].reset_index(drop=True)
    va_df = full_train.iloc[va_idx].reset_index(drop=True)

    # Write temp split CSVs to match the dataset interface
    tr_csv = "train_split.csv"
    va_csv = "val_split.csv"
    tr_df.to_csv(tr_csv, index=False)
    va_df.to_csv(va_csv, index=False)

    train_ds = Plant2020Dataset(tr_csv, train=True, image_size=224)
    val_ds = Plant2020Dataset(va_csv, train=True, image_size=224)
    test_ds = Plant2020Dataset(TEST_CSV, train=False, image_size=224)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = SmallCNN(num_classes=4).to(device)

    # Training setup (preserve BCEWithLogitsLoss and cosine schedule intent in a minimal way)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train fixed number of epochs (3) as in original config
    num_epochs = 3
    for epoch in range(args.start_epoch, num_epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_probs = predict(model, val_loader, device)
        val_true = val_ds.labels
        val_auc = mean_columnwise_roc_auc(val_true, val_probs)
        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={tr_loss:.4f} | val_mean_auc={val_auc:.4f}")

    # Inference + submission (ensure correct order/columns from sample_submission.csv)
    test_probs = predict(model, test_loader, device)

    sub = pd.read_csv(SAMPLE_SUB_CSV)
    # Align by image_id order of sample_submission (same as test.csv typically)
    sub[TARGET_COLS] = test_probs
    save_path = "submission.csv"
    sub.to_csv(save_path, index=False)
    print(f"Saved submission to {save_path} with shape {sub.shape} and columns {list(sub.columns)}")


if __name__ == "__main__":
    main()
"""
Path("distill.py").write_text(distill_code)
print("Wrote distill.py")



## === cell 2
from pathlib import Path

config_text = r"""
# Kept for compatibility; simplified distill.py does not require torchdistill/timm configs.
"""
Path("config.yaml").write_text(config_text)
print("Wrote config.yaml")



## === cell 3
import subprocess, sys

subprocess.run(
    [
        sys.executable,
        "distill.py",
        "--config",
        "config.yaml",
        "--device",
        "cuda",
        "--seed",
        "10",
        "-student_only",
        "--log",
        "log.log",
    ],
    check=True,
)



## === cell 4
from pathlib import Path
import pandas as pd

print("Submission exists:", Path("submission.csv").exists())
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print("Columns:", list(sub.columns))
assert sub.shape[1] == 5
assert sub.columns.tolist() == [
    "image_id",
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
]
assert sub["image_id"].isna().sum() == 0
print("submission.csv looks valid.")
