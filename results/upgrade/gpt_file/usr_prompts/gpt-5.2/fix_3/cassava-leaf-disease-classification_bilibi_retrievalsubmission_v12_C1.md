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

0.8774554245995769

# 6. Current score

0.74701

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09417) has done: 'The errors come from relying on external notebook packages (`baseline-infer`, `utils`, `configs`, `model`) that aren’t available in this Kaggle environment, which prevents model/config/transform imports and then cascades into missing symbols like `torch`/`Dataset`. To make it run end-to-end with minimal conceptual change (still “load a pretrained CNN and run inference on test_images”), I replace those unavailable modules with a torchvision ResNet50 backbone, load weights if present (otherwise run with ImageNet weights), and use standard ImageNet preprocessing. I also fix the submission-length issue by iterating over `sample_submission.csv` order (guaranteeing exactly 2676 rows) instead of `os.listdir()`, and write `submission.csv` with the required columns. This should yield a valid submission file consistently and typically a reasonable accuracy baseline (though exact score depends on whether the referenced competition weights exist).'
- What this solution (achieved 0.74701) has done: 'Your current score is extremely low because the model is effectively an ImageNet-pretrained ResNet50 with a randomly initialized 5-class head (since the referenced cassava weights file doesn’t exist), so predictions are near-random for this dataset. To move toward the 0.877 target with minimal conceptual change (still ResNet50 inference), I (1) automatically train only the final classification head on the provided `train.csv` images for a single short epoch and (2) then run the same test-time inference pipeline to produce `submission.csv`. This preserves the core architecture and loss (standard cross-entropy) and keeps the pipeline end-to-end within the time limit, while giving a large accuracy jump compared to the untrained head. I also keep ordering aligned to `sample_submission.csv` as you already do, to guarantee a valid submission.'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
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

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

weights_path = "../input/baseline-weights/epoch_35.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "num_files:",
    len(os.listdir(TRAIN_DIR)) if os.path.isdir(TRAIN_DIR) else None,
)
print(
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
    "num_files:",
    len(os.listdir(TEST_DIR)) if os.path.isdir(TEST_DIR) else None,
)
print("Train csv exists:", os.path.isfile(TRAIN_CSV_PATH))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))



## === cell 1
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 5)

external_weights_loaded = False
if os.path.isfile(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    external_weights_loaded = True
    print("Loaded weights:", weights_path)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print(
        "Weights not found at:",
        weights_path,
        " -> will fit only the classifier head from train.csv for a small boost vs random.",
    )

model = model.to(device)



## === cell 2
img_size = 224
infer_tfms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_tfms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TrainDataset(Dataset):
    def __init__(self, train_dir: str, train_csv_path: str, transform=None):
        self.train_dir = train_dir
        self.df = pd.read_csv(train_csv_path)
        assert {"image_id", "label"}.issubset(self.df.columns)
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        fp = os.path.join(self.train_dir, image_id)
        img = cv2.imread(fp)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class InferDataset(Dataset):
    """
    Uses sample_submission.csv order to guarantee exact length and correct image_id alignment.
    """

    def __init__(self, test_dir: str, sample_sub_path: str, transform=None):
        self.test_dir = test_dir
        self.df = pd.read_csv(sample_sub_path)
        assert "image_id" in self.df.columns, "sample_submission must contain image_id"
        self.image_ids = self.df["image_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        fp = os.path.join(self.test_dir, image_id)
        img = cv2.imread(fp)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id




## === cell 3
if not external_weights_loaded:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    train_ds = TrainDataset(TRAIN_DIR, TRAIN_CSV_PATH, transform=train_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-4)

    model.train()
    start = datetime.datetime.now()
    running_loss = 0.0
    correct = 0
    seen = 0

    for step, (xb, yb) in enumerate(train_loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == yb).sum().detach().cpu())
        seen += int(yb.shape[0])

        if step % 100 == 0:
            elapsed = (datetime.datetime.now() - start).total_seconds()
            print(
                f"step {step}/{len(train_loader)} "
                f"loss {running_loss/step:.4f} "
                f"acc {correct/seen:.4f} "
                f"elapsed {elapsed:.1f}s"
            )

    print(
        "Finished head training.",
        "avg_loss:",
        running_loss / max(1, len(train_loader)),
        "train_acc:",
        correct / max(1, seen),
    )
else:
    print("External weights loaded; skipping head training.")

model.eval()



## === cell 4
infer_ds = InferDataset(TEST_DIR, SAMPLE_SUB_PATH, transform=infer_tfms)
infer_loader = DataLoader(
    infer_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

xb, ids = next(iter(infer_loader))
print("Batch:", xb.shape, "Example id:", ids[0], "Total test rows:", len(infer_ds))



## === cell 5
pred_image_ids = []
pred_labels = []

with torch.no_grad():
    for xb, ids in infer_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
        pred_labels.extend(preds.tolist())
        pred_image_ids.extend(list(ids))

sub = pd.DataFrame({"image_id": pred_image_ids, "label": pred_labels})

sample = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
assert len(sub) == len(sample), "Submission length mismatch after merge"
assert sub["label"].isna().sum() == 0, "Some test images missing predictions"

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))
print(sub.head())



## === cell 6
check = pd.read_csv("./submission.csv")
print("submission.csv shape:", check.shape)
print("columns:", list(check.columns))
print("label dtype:", check["label"].dtype)
print(
    "unique labels:",
    sorted(check["label"].unique())[:10],
    "count:",
    check["label"].nunique(),
)
