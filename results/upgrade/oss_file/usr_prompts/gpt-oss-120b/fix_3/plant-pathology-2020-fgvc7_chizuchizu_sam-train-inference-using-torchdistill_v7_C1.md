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

0.91219

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import timm

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = Path("/kaggle/input/plant-pathology-2020-fgvc7")
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
SAMPLE_SUBMISSION = DATA_ROOT / "sample_submission.csv"
IMG_DIR = DATA_ROOT / "images"




## === cell 1
def get_image_path(image_id: str) -> str:
    """Return absolute path to an image given its id."""
    return str(IMG_DIR / f"{image_id}.jpg")




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, inf: bool = False, transform=None):
        """
        df must contain at least the column 'image_id'.
        If inf is False, the dataframe must also contain the label columns:
        ['healthy', 'multiple_diseases', 'rust', 'scab'].
        """
        self.df = df.reset_index(drop=True)
        self.inf = inf
        self.transform = transform

        if not inf:
            self.labels = self.df[
                ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df), 4), dtype=np.float32)

        self.paths = self.df["image_id"].apply(get_image_path).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
        return img, label




## === cell 3
def mean_column_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean ROC‑AUC over each column (class)."""
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return float(np.mean(aucs))




## === cell 4
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    return epoch_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    all_targets = []
    all_logits = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_targets.append(targets)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    return mean_column_auc(targets, probs), probs




## === cell 5
transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784],
        ),
    ]
)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_len = int(0.8 * len(train_df))
val_len = len(train_df) - train_len
train_subset, val_subset = random_split(
    train_df, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
)

train_dataset = PlantDataset(pd.DataFrame(train_subset), inf=False, transform=transform)
val_dataset = PlantDataset(pd.DataFrame(val_subset), inf=False, transform=transform)
test_dataset = PlantDataset(test_df, inf=True, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)

model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=4)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0)

best_auc = 0.0
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_auc, _ = evaluate(model, val_loader)
    scheduler.step()
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}"
    )
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_model.ckpt")

model.load_state_dict(torch.load("best_model.ckpt", map_location=DEVICE))
_, test_probs = evaluate(model, test_loader)

submission = pd.read_csv(SAMPLE_SUBMISSION)
submission = submission.set_index("image_id").loc[test_df["image_id"]].reset_index()
submission[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1683788289.py in <cell line: 0>()
     22 )
     23 
---> 24 train_dataset = PlantDataset(pd.DataFrame(train_subset), inf=False, transform=transform)
     25 val_dataset = PlantDataset(pd.DataFrame(val_subset), inf=False, transform=transform)
     26 test_dataset = PlantDataset(test_df, inf=True, transform=transform)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    884         else:
    885             if index is None or columns is None:
--> 886                 raise ValueError("DataFrame constructor not properly called!")
    887 
    888             index = ensure_index(index)

ValueError: DataFrame constructor not properly called!
