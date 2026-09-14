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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9998

# 6. Current score

0.71423

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.71423) has done: 'I fix the test file listing bug that’s causing the `FileNotFoundError` by filtering `os.listdir(TEST_DIR)` (and train preview) to include only real image files, excluding the stray `test/` directory entry. I also make the image reader more robust by raising a clearer error if a non-file slips through, without changing the model/training logic. Finally, I ensure the script always writes a valid `sub.csv` with the exact required columns and ordering from `sample_submission.csv`, so Kaggle accepts it.'

# 9. Code solution

## === cell 0
import os
import random
import time

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"


def list_image_files(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    out = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and name.lower().endswith(exts):
            out.append(name)
    return sorted(out)




## === cell 1
import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



## === cell 2
labels = pd.read_csv(TRAIN_CSV)

fig = plt.figure(figsize=(25, 8))
train_imgs = list_image_files(TRAIN_DIR)
for idx, img in enumerate(np.random.choice(train_imgs, 20, replace=False)):
    ax = fig.add_subplot(4, 20 // 4, idx + 1, xticks=[], yticks=[])
    im = Image.open(os.path.join(TRAIN_DIR, img))
    plt.imshow(im)
    lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
    ax.set_title(f"Label: {lab}")
plt.show()



## === cell 3
test_img = list_image_files(TEST_DIR)
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

labels.head()



## === cell 4
labels.loc[labels["data_type"] == "train", "has_cactus"].value_counts()



## === cell 5
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

print("train:", train.shape, "valid:", valid.shape)




## === cell 6
def reader_fn(row, root_train=TRAIN_DIR, root_test=TEST_DIR):
    if row["data_type"] == "train":
        path = os.path.join(root_train, row["id"])
    else:
        path = os.path.join(root_test, row["id"])

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Expected an image file but got missing/non-file path: {path}"
        )

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image (cv2.imread returned None): {path}"
        )
    image = image[:, :, ::-1]  # BGR -> RGB
    label = float(row["has_cactus"])
    return image, label




## === cell 7
def augs(p=0.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.75
            ),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        p=p,
    )




## === cell 8
def get_transforms(size=32, p=0.5, train=True):
    if train:
        aug = augs(p=p)
    else:
        aug = None

    def _tfm(image):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        if aug is not None:
            image = aug(image=image)["image"]
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
            IMAGENET_STD, dtype=np.float32
        )
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(image)

    return _tfm




## === cell 9
train_tfm = get_transforms(size=32, p=0.5, train=True)
val_tfm = get_transforms(size=32, p=0.0, train=False)




## === cell 10
class CactusDataset(Dataset):
    def __init__(self, df, transform, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, label = reader_fn(row)
        x = self.transform(image)
        if self.is_test:
            return {"image": x, "id": row["id"]}
        y = torch.tensor([label], dtype=torch.float32)
        return {"image": x, "label": y}


batch_size = 64
workers = 2  # safe on Kaggle; improves throughput without changing logic

train_ds = CactusDataset(train, transform=train_tfm, is_test=False)
val_ds = CactusDataset(valid, transform=val_tfm, is_test=False)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

print("batches:", len(train_dl), len(val_dl))



## === cell 11
test_ds = CactusDataset(test_df, transform=val_tfm, is_test=True)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)




## === cell 12
class Net(nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        arch: str = "densenet169",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if arch != "densenet169":
            raise ValueError("This script preserves the original choice: densenet169")

        weights = (
            torchvision.models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = torchvision.models.densenet169(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits




## === cell 13
model = Net(num_classes=1, p=0.2, arch="densenet169", pretrained=True).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)




## === cell 14
def step_fn(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    inp = batch["image"].to(device, non_blocking=True)
    return model(inp)




## === cell 15
def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target = target.detach().cpu().numpy().reshape(-1)
    preds = (torch.sigmoid(preds).detach().cpu().numpy().reshape(-1) > thresh).astype(
        int
    )
    return accuracy_score(target, preds)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target = target.detach().cpu().numpy().reshape(-1)
    preds = torch.sigmoid(preds).detach().cpu().numpy().reshape(-1)
    return roc_auc_score(target, preds)




## === cell 16
def run_one_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_targets = []
    all_logits = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_targets.append(targets.detach().cpu())
        all_logits.append(logits.detach().cpu())

    all_targets = torch.cat(all_targets, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    avg_loss = total_loss / len(loader.dataset)
    acc = bce_accuracy(all_targets, all_logits)
    auc = roc_auc(all_targets, all_logits)
    return avg_loss, acc, auc




## === cell 17
total_epochs = 9
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

best_auc = -1.0
best_state = None

for epoch in range(1, total_epochs + 1):
    t0 = time.time()
    tr_loss, tr_acc, tr_auc = run_one_epoch(model, train_dl, optimizer=optimizer)
    va_loss, va_acc, va_auc = run_one_epoch(model, val_dl, optimizer=None)
    scheduler.step()

    if va_auc > best_auc:
        best_auc = va_auc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    dt = time.time() - t0
    lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:02d}/{total_epochs} | lr={lr:.6f} | "
        f"train loss={tr_loss:.4f} acc={tr_acc:.4f} auc={tr_auc:.5f} | "
        f"val loss={va_loss:.4f} acc={va_acc:.4f} auc={va_auc:.5f} | {dt:.1f}s"
    )

print("best val auc:", best_auc)



## === cell 18
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()



## === cell 19
all_ids = []
all_probs = []

with torch.no_grad():
    for batch in test_dl:
        imgs = batch["image"].to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        all_probs.append(probs)
        all_ids.extend(batch["id"])

all_probs = np.concatenate(all_probs, axis=0)
preds_df = pd.DataFrame({"id": all_ids, "has_cactus": all_probs})

sub = pd.read_csv(SAMPLE_SUB)
preds_df = sub[["id"]].merge(preds_df, on="id", how="left")
preds_df["has_cactus"] = preds_df["has_cactus"].fillna(0.5).astype(float)

preds_df.to_csv("sub.csv", index=False)
preds_df.head()



## === cell 20
print("Wrote submission:", os.path.abspath("sub.csv"))
print("submission shape:", preds_df.shape)
print("na counts:\n", preds_df.isna().sum())
print(preds_df.head())
