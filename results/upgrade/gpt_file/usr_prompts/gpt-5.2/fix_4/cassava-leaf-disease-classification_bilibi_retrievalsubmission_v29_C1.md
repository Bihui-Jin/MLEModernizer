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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8779087337564219

# 6. Current score

0.70254

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16069) has done: 'I remove the broken dependencies on missing external Kaggle datasets/modules (`utils`, `model.*`, custom config, and `build_transforms`) and replace them with an equivalent, self-contained PyTorch inference pipeline that loads images from `test_images` and writes a correctly sized `submission.csv`. To keep the core idea (ResNet50 inference) while making it runnable, I use `torchvision`’s ResNet-50 with a 5-class head and standard ImageNet normalization. Because your referenced weights file path doesn’t exist in this environment, the script safely fall back to the untrained head; this is score-suboptimal but guarantees a valid submission (since current score is “Not yielded”). Finally, I ensure the submission exactly matches `sample_submission.csv` ordering and length to prevent the “same length as the answers” error.'
- What this solution (achieved 0.73804) has done: 'Your score is far below the target because the current script is doing pure inference with a random 5‑class head (no Cassava-trained weights), so predictions are essentially noise. The smallest legitimate change that preserves your core model (ResNet50 + linear 5-class head) and inference semantics is to add a brief supervised fine-tuning step on `train.csv` using the existing training images, then run the same test inference and write `submission.csv`. To keep runtime under 600s and changes minimal, this trains only the final `fc` layer for 1 short epoch and uses a simple stratified split for a quick sanity-check accuracy (not used for early stopping). This should move accuracy substantially upward toward the target band without changing the model architecture or loss.'
- What this solution (achieved 0.70254) has done: 'Your current gap to the target is about 0.14 accuracy, so we need a modest but real boost without changing the core ResNet50+linear-head setup. The smallest high-impact change is to fine-tune not only `fc` but also the last ResNet stage (`layer4`) for the same single epoch, which typically yields a noticeable accuracy lift while preserving architecture and training semantics. To make that single epoch more effective without adding extra epochs or changing the objective, I also switch the train transform from plain resize to a standard “resize then random resized crop” (common for ImageNet fine-tuning) and use class-weighted cross entropy to better handle Cassava’s label imbalance. All changes keep the same inference pipeline and still write a valid `submission.csv` matching `sample_submission.csv` order/length.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision import models, transforms


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_ROOT = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.exists(TEST_IMG_DIR), f"Missing test image dir: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train image dir: {TRAIN_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission columns"
test_image_ids = sample_sub["image_id"].tolist()

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"

weights_path = "../input/wwwwww/weight_epoch_13.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 1
class CassavaTestDataset(Dataset):
    def __init__(self, img_dir: str, image_ids, tfm):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.tfm = tfm

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        fp = os.path.join(self.img_dir, image_id)
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(img)
        return img, image_id


class CassavaTrainDataset(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, tfm):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])
        fp = os.path.join(self.img_dir, image_id)
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(img)
        return img, label


train_tfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_tfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_ds = CassavaTestDataset(TEST_IMG_DIR, test_image_ids, test_tfm)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

x0, id0 = next(iter(test_loader))
x0.shape, id0[:3]



## === cell 2
num_classes = 5
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)

loaded_external_weights = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    try:
        model.load_state_dict(state, strict=True)
        loaded_external_weights = True
        print(f"Loaded weights (strict) from: {weights_path}")
    except Exception:
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_external_weights = True
        print(f"Loaded weights (non-strict) from: {weights_path}")
        print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
else:
    print(
        f"WARNING: weights not found at {weights_path}. Will fine-tune on train.csv as a fallback."
    )

model = model.to(device)




## === cell 3
def stratified_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    val_idx = []
    for lbl, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.extend(idx[:n_val].tolist())
    val_idx = set(val_idx)
    trn_df = df.loc[~df.index.isin(val_idx)].reset_index(drop=True)
    val_df = df.loc[df.index.isin(val_idx)].reset_index(drop=True)
    return trn_df, val_df


if not loaded_external_weights:
    trn_df, val_df = stratified_split(train_df, val_frac=0.1, seed=42)

    train_ds = CassavaTrainDataset(TRAIN_IMG_DIR, trn_df, train_tfm)
    val_ds = CassavaTrainDataset(TRAIN_IMG_DIR, val_df, test_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    counts = trn_df["label"].value_counts().sort_index()
    weight = (counts.sum() / (counts + 1e-6)).to_numpy(dtype=np.float32)
    weight = weight / weight.mean()  # normalize to keep loss scale stable
    class_weight = torch.tensor(weight, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)

    optimizer = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 3e-4},
            {"params": model.fc.parameters(), "lr": 3e-3},
        ]
    )

    model.train()
    epochs = 1  # keep identical epoch count; no early stopping or relaxed convergence
    for epoch in range(epochs):
        running_loss = 0.0
        n = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            n += bs

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        model.train()
        print(
            f"Epoch {epoch+1}/{epochs} - train_loss: {running_loss/max(1,n):.4f} - val_acc: {correct/max(1,total):.4f}"
        )

model.eval()



## === cell 4
pred_image_ids = []
pred_labels = []

with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        pred_image_ids.extend(list(image_ids))
        pred_labels.extend(preds.tolist())

assert len(pred_image_ids) == len(
    sample_sub
), "Prediction count must match sample_submission length"
assert (
    pred_image_ids == test_image_ids
), "Prediction order must match sample_submission order"

sub = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})
sub_path = "./submission.csv"
sub.to_csv(sub_path, index=False)

print(sub.head())
print("Wrote:", sub_path, "rows:", len(sub))



## === cell 5
check = pd.read_csv("./submission.csv")
assert (
    check.shape[0] == sample_sub.shape[0]
), "Submission must have same number of rows as sample_submission"
assert list(check.columns) == [
    "image_id",
    "label",
], "Submission must have columns: image_id,label"
assert (
    check["image_id"].tolist() == sample_sub["image_id"].tolist()
), "Submission image_id ordering mismatch"
check.describe(include="all")
