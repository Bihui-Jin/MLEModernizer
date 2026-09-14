# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(rel_path: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {rel_path} under candidates: {DATA_ROOT_CANDIDATES}"
    )


train_csv_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")


def find_dir(dir_name: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, dir_name)
        if os.path.isdir(p):
            return p
        p2 = os.path.join(root, "paddy-disease-classification", dir_name)
        if os.path.isdir(p2):
            return p2
    raise FileNotFoundError(
        f"Could not find dir {dir_name} under candidates: {DATA_ROOT_CANDIDATES}"
    )


train_images_dir = find_dir("train_images")
test_images_dir = find_dir("test_images")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print("train.csv:", train_df.shape, "sample_submission.csv:", sample_sub.shape)
print("train_images_dir:", train_images_dir)
print("test_images_dir:", test_images_dir)



## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import torchvision
    import torchvision.transforms as T
except Exception as e:
    raise RuntimeError(
        "torchvision is required for this solution in Kaggle environment."
    ) from e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

labels = sorted(train_df["label"].unique().tolist())
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}
num_classes = len(labels)
print("num_classes:", num_classes, labels)


def train_image_path(row):
    return os.path.join(train_images_dir, row["label"], row["image_id"])


def test_image_path(image_id):
    return os.path.join(test_images_dir, image_id)


IMG_SIZE = 224
train_tfms = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_tfms = train_tfms


class PaddyDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        if self.is_train:
            img_path = train_image_path(r)
            y = label2idx[r["label"]]
        else:
            img_path = test_image_path(r["image_id"])
            y = -1

        img = Image.open(img_path).convert("RGB")
        x = (train_tfms if self.is_train else test_tfms)(img)
        if self.is_train:
            return x, y
        return x, r["image_id"]




## === cell 2
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(splitter.split(train_df, train_df["label"]))
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = PaddyDataset(tr_df, is_train=True)
val_ds = PaddyDataset(va_df, is_train=True)

BATCH_SIZE = 32
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

len(train_loader), len(val_loader)



## === cell 3
from torchvision.models import resnet18, ResNet18_Weights

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def accuracy_from_logits(logits, y):
    preds = logits.argmax(1)
    return (preds == y).float().mean().item()




## === cell 4
EPOCHS = (
    2  # keep within runtime; produces a valid submission; adjust only if needed later.
)

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    n_tr = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        tr_loss += loss.item() * bs
        tr_acc += (logits.argmax(1) == yb).float().sum().item()
        n_tr += bs

    model.eval()
    va_loss = 0.0
    va_acc = 0.0
    n_va = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            va_loss += loss.item() * bs
            va_acc += (logits.argmax(1) == yb).float().sum().item()
            n_va += bs

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss {tr_loss/n_tr:.4f} acc {tr_acc/n_tr:.4f} | "
        f"val loss {va_loss/n_va:.4f} acc {va_acc/n_va:.4f}"
    )



## === cell 5
test_df = sample_sub[["image_id"]].copy()
test_ds = PaddyDataset(test_df, is_train=False)
test_loader = DataLoader(
    test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_ids = []
all_preds = []

with torch.no_grad():
    for xb, img_ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred_idx = logits.argmax(1).cpu().numpy().tolist()
        preds = [idx2label[i] for i in pred_idx]
        all_ids.extend(list(img_ids))
        all_preds.extend(preds)

sub = pd.DataFrame({"image_id": all_ids, "label": all_preds})

sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
assert sub["label"].isna().sum() == 0, "Some test images were not predicted."
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
