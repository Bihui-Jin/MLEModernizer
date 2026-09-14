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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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

0.6128739800543971

# 6. Current score

0.80531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12145) has done: 'I fix the immediate runtime/import errors by replacing the removed `pytorch_lightning.metrics.functional.accuracy` import with the supported `torchmetrics` equivalent (or removing it since it isn’t used). I also fix the model definition to output the correct number of classes (5 for Cassava) and remove the dependency on a missing external checkpoint by running inference with ImageNet-pretrained ResNet50 weights (a minimal, legitimate change that also yields a reasonable baseline score instead of crashing). Finally, I ensure predictions are generated for every row in `sample_submission.csv`, moved to the right device, and written to `submission.csv` with the exact required columns.'
- What this solution (achieved 0.80531) has done: 'Your current score is far below the target, so we should legitimately improve accuracy with minimal logic changes. The main issue is that you’re using an ImageNet-pretrained ResNet50 without any Cassava fine-tuning, which yields near-random performance; we add a short, straightforward supervised fine-tuning step on `train.csv` (same model/loss, standard PyTorch loop) and then run inference. To keep changes minimal and stable, we use a simple train/val split, basic torchvision augmentations, and class-weighted cross-entropy to handle label imbalance—no early stopping, no architectural changes. We also switch inference to batched DataLoader inference (same predictions, just faster and less error-prone) and still write `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os
import json
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

import pytorch_lightning as pl  # noqa: F401
from sklearn import metrics, model_selection, preprocessing  # noqa: F401

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomCrop,
    MultiplicativeNoise,
)  # noqa: F401

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
try:
    weights = torchvision.models.ResNet50_Weights.DEFAULT
except Exception:
    weights = None

base = torchvision.models.resnet50(weights=weights)
in_features = base.fc.in_features
base.fc = nn.Sequential(
    nn.Dropout(p=0.8),
    nn.Linear(in_features, 512, bias=False),
    nn.BatchNorm1d(512),
)




## === cell 2
class Modified_Model(nn.Module):
    def __init__(self, mymodel, num_classes, classify=False):
        super(Modified_Model, self).__init__()
        self.model = mymodel
        self.logits = nn.Linear(512, num_classes)
        self.classify = classify

    def forward(self, x):
        x = self.model(x)
        if self.classify:
            x = self.logits(x)  # return class scores
            return x
        else:
            return x


class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x




## === cell 3
my_model = Modified_Model(base, num_classes=5, classify=True)
model = Model(my_model)



## === cell 4
loss_fn = nn.CrossEntropyLoss()




## === cell 5
class LitModel(pl.LightningModule):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def forward(self, x):
        logits = self.model(x)
        return logits




## === cell 6
model = my_model



## === cell 7
ckpt_path = "/kaggle/input/resnet-new-lighting/resnet_50_new.ckpt"
dicts = None
if os.path.exists(ckpt_path):
    dicts = torch.load(ckpt_path, map_location="cpu")



## === cell 8
if dicts is not None:
    state = dicts.get("state_dict", dicts)
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("[INFO] Loaded checkpoint.")
    print("[INFO] Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print("[INFO] Checkpoint not found; using ImageNet-pretrained ResNet50 backbone.")



## === cell 9
from torch.utils.data import Dataset, DataLoader

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

train_df = pd.read_csv(train_csv_path)
sample_submission_df = pd.read_csv(sample_sub_path)

trn_df, val_df = model_selection.train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)

train_tfms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
eval_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform, labeled=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.labeled:
            y = int(self.df.loc[idx, "label"])
            return image, y
        return image, img_id


label_counts = train_df["label"].value_counts().sort_index()
counts = label_counts.values.astype(np.float32)
weights = counts.sum() / (counts + 1e-6)
weights = weights / weights.mean()
class_weights = torch.tensor(weights, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

batch_size = 32 if torch.cuda.is_available() else 16
num_workers = 2

train_loader = DataLoader(
    CassavaDataset(trn_df, train_img_dir, train_tfms, labeled=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    CassavaDataset(val_df, train_img_dir, eval_tfms, labeled=True),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


epochs = 3
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    seen = 0
    t0 = time.time()
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * yb.size(0)
        seen += int(yb.size(0))

    tr_loss = running_loss / max(seen, 1)
    va_loss, va_acc = evaluate(model, val_loader)
    print(
        f"[INFO] Epoch {epoch}/{epochs} "
        f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
        f"time={(time.time()-t0):.1f}s"
    )



## === cell 10
from torch.utils.data import DataLoader

test_loader = DataLoader(
    CassavaDataset(sample_submission_df, test_img_dir, eval_tfms, labeled=False),
    batch_size=64 if torch.cuda.is_available() else 32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
pred_rows = []
with torch.no_grad():
    for xb, img_ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
        for iid, p in zip(img_ids, preds):
            pred_rows.append((iid, int(p)))

pred_df = pd.DataFrame(pred_rows, columns=["image_id", "label"])

pred_df = sample_submission_df[["image_id"]].merge(pred_df, on="image_id", how="left")
assert pred_df["label"].isna().sum() == 0, "Missing predictions for some test images."
assert len(pred_df) == len(sample_submission_df)

pred_df.to_csv("submission.csv", index=False)
print("[INFO] Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())
