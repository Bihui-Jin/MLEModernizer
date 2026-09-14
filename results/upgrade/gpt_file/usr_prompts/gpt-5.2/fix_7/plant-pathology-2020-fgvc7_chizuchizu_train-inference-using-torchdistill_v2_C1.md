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

0.85396

# 6. Current score

0.95404

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.95404) has done: 'The crashes come from building image paths without the “.jpg” extension; `image_id` values like `Train_737` and `Test_0` must be mapped to `Train_737.jpg` etc. I fix the dataset to robustly resolve filenames (try exact name, then add `.jpg`, and fall back to a quick directory lookup cache), which unblocks both training and test inference. I also make the submission creation align predictions to `test_df` order (not relying on `sample_submission` order), ensuring a valid `.csv` with correct columns is always produced. Core model/training logic is unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(10)

ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(ROOT, "images")
TRAIN_CSV = os.path.join(ROOT, "train.csv")
TEST_CSV = os.path.join(ROOT, "test.csv")
SAMPLE_SUB = os.path.join(ROOT, "sample_submission.csv")
SAVE_PATH = "submission.csv"

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("IMG_DIR exists:", os.path.isdir(IMG_DIR))




## === cell 1
class Plant2020Dataset(Dataset):
    """
    Bugfix: competition image_id values don't include '.jpg' in CSV.
    Resolve to an existing file by trying:
      1) img_dir/image_id
      2) img_dir/image_id + '.jpg'
      3) cached lookup in directory (built once)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        target_cols=None,
        transform=None,
        infer=False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_cols = target_cols
        self.transform = transform
        self.infer = infer

        self._fname_map = None
        try:
            files = os.listdir(self.img_dir)
            self._fname_map = {
                os.path.splitext(f)[0]: f for f in files if f.lower().endswith(".jpg")
            }
        except Exception:
            self._fname_map = None

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, image_id: str) -> str:
        p0 = os.path.join(self.img_dir, image_id)
        if os.path.exists(p0):
            return p0
        p1 = p0 + ".jpg"
        if os.path.exists(p1):
            return p1
        if self._fname_map is not None and image_id in self._fname_map:
            p2 = os.path.join(self.img_dir, self._fname_map[image_id])
            if os.path.exists(p2):
                return p2
        raise FileNotFoundError(
            f"Image file not found for image_id='{image_id}' in dir='{self.img_dir}'"
        )

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        path = self._resolve_path(image_id)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.infer:
            y = torch.zeros(4, dtype=torch.float32)
        else:
            y = torch.tensor(
                self.df.loc[idx, self.target_cols].values.astype(np.float32),
                dtype=torch.float32,
            )
        return img, y


train_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

assert "image_id" in train_df.columns and "image_id" in test_df.columns
for c in TARGET_COLS:
    assert c in train_df.columns, f"Missing target col {c} in train.csv"
    assert c in sample_sub.columns, f"Missing target col {c} in sample_submission.csv"

n = len(train_df)
perm = np.random.RandomState(42).permutation(n)
split = int(0.8 * n)
tr_idx, va_idx = perm[:split], perm[split:]
tr_df, va_df = train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[
    va_idx
].reset_index(drop=True)

train_ds = Plant2020Dataset(
    tr_df, IMG_DIR, TARGET_COLS, transform=train_tfms, infer=False
)
val_ds = Plant2020Dataset(va_df, IMG_DIR, TARGET_COLS, transform=val_tfms, infer=False)
test_ds = Plant2020Dataset(
    test_df, IMG_DIR, TARGET_COLS, transform=val_tfms, infer=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

print("train/val/test sizes:", len(train_ds), len(val_ds), len(test_ds))



## === cell 3
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 4)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0)


@torch.no_grad()
def mean_columnwise_auc_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> float:
    from sklearn.metrics import roc_auc_score

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    aucs = []
    for k in range(y.shape[1]):
        yk = y[:, k]
        if np.unique(yk).size < 2:
            continue
        aucs.append(roc_auc_score(yk, probs[:, k]))
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def validate():
    model.eval()
    outs, tars = [], []
    total_loss = 0.0
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        outs.append(logits)
        tars.append(yb)
    outs = torch.cat(outs, dim=0)
    tars = torch.cat(tars, dim=0)
    auc = mean_columnwise_auc_from_logits(outs, tars)
    return total_loss / len(val_loader.dataset), auc




## === cell 4
best_auc = -1.0
best_state = None

num_epochs = 1
for epoch in range(num_epochs):
    tr_loss = train_one_epoch()
    va_loss, va_auc = validate()
    scheduler.step()
    print(
        f"epoch {epoch+1}/{num_epochs} train_loss={tr_loss:.5f} val_loss={va_loss:.5f} val_mean_auc={va_auc:.5f}"
    )
    if va_auc == va_auc and va_auc > best_auc:
        best_auc = va_auc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state)




## === cell 5
@torch.no_grad()
def predict_test_logits():
    model.eval()
    all_logits = []
    for xb, _ in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.detach().cpu())
    return torch.cat(all_logits, dim=0)


test_logits = predict_test_logits()
test_probs = torch.sigmoid(test_logits).numpy().astype(np.float32)

sub = test_df[["image_id"]].copy()
for i, c in enumerate(TARGET_COLS):
    sub[c] = test_probs[:, i]

sub = sub[["image_id"] + TARGET_COLS]
sub.to_csv(SAVE_PATH, index=False)

print("Wrote:", os.path.abspath(SAVE_PATH))
print(sub.head())
print("shape:", sub.shape)
print("cols:", sub.columns.tolist())
assert os.path.exists(SAVE_PATH) and SAVE_PATH.endswith(".csv")
assert sub.shape[0] == len(test_df) and sub.shape[1] == 1 + len(TARGET_COLS)
