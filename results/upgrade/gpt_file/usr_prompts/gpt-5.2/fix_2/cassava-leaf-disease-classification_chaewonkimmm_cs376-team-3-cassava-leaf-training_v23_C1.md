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
imbalanced-learn==0.13.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
tqdm==4.67.1

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

0.7855

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.7855) has done: 'I remove the `imblearn/SMOTE` import (it’s incompatible with the installed scikit-learn version and isn’t used elsewhere), which unblocks the notebook immediately. I also fix the missing pretrained `.pth` by training the same ResNet34 head on the provided `train_images` within the notebook (same architecture/criterion family; just no external weights). For inference, I filter out the stray `test_images/` subdirectory so only real `.jpg` files are read, and I align predictions to `sample_submission.csv` ordering to ensure a valid `submission.csv`. Finally, I make inference deterministic and efficient (`eval()`, `no_grad()`) so it completes reliably and writes a correct CSV.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
image_path = "../input/cassava-leaf-disease-classification/train_images/"

train_image_paths = [os.path.join(image_path, x) for x in dfx.image_id.values]
train_targets = dfx.label.values.astype(int)

len(train_image_paths), len(train_targets), dfx.head()



## === cell 2
len(train_image_paths), len(train_targets)



## === cell 3
"""torch module dataset"""


class CassavaDataset(Dataset):
    def __init__(self, files, targets=None, transform=None):
        self.files = files
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.targets is None:
            return image

        label = int(self.targets[idx])
        return image, label




## === cell 4
"""Dataset Initialization (kept, but not used for training below; original code created a dataset here)."""
cassava_data = CassavaDataset(train_image_paths, train_targets)



## === cell 5
batch_size = 16

cassava_loader = DataLoader(
    cassava_data, batch_size=batch_size, shuffle=True, num_workers=2
)
classes = ("0", "1", "2", "3", "4")
len(cassava_loader)



## === cell 6
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



## === cell 7

train_idx, val_idx = train_test_split(
    np.arange(len(train_image_paths)),
    test_size=0.1,
    random_state=42,
    stratify=train_targets,
)

train_files = [train_image_paths[i] for i in train_idx]
train_labels = train_targets[train_idx]
val_files = [train_image_paths[i] for i in val_idx]
val_labels = train_targets[val_idx]

input_size_train = 384
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((input_size_train, input_size_train)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((input_size_train, input_size_train)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

train_ds = CassavaDataset(train_files, train_labels, transform=train_transform)
val_ds = CassavaDataset(val_files, val_labels, transform=val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(train_ds), len(val_ds)



## === cell 8
resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)
resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)

resnet




## === cell 9
def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


epochs = 2  # minimal to keep runtime bounded and still better than random

for epoch in range(epochs):
    resnet.train()
    train_loss = 0.0
    train_acc = 0.0
    n_train = 0

    for xb, yb in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{epochs} - train", leave=False
    ):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = resnet(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        train_loss += loss.item() * bs
        train_acc += accuracy_from_logits(out.detach(), yb) * bs
        n_train += bs

    resnet.eval()
    val_loss = 0.0
    val_acc = 0.0
    n_val = 0
    with torch.no_grad():
        for xb, yb in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{epochs} - val", leave=False
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = resnet(xb)
            loss = criterion(out, yb)

            bs = xb.size(0)
            val_loss += loss.item() * bs
            val_acc += accuracy_from_logits(out, yb) * bs
            n_val += bs

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"train_loss={train_loss/n_train:.4f}, train_acc={train_acc/n_train:.4f} | "
        f"val_loss={val_loss/n_val:.4f}, val_acc={val_acc/n_val:.4f}"
    )



## === cell 10
"""For Submission"""
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 11
test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

test_image_ids = submission_df["image_id"].tolist()
test_files = [os.path.join(test_path, img_id) for img_id in test_image_ids]

test_transform = transforms.Compose(
    [
        transforms.Resize((input_size_train, input_size_train)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

test_ds = CassavaDataset(test_files, targets=None, transform=test_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_ds), test_ds[0].shape



## === cell 12
resnet.eval()
y_preds = []

with torch.no_grad():
    for xb in tqdm(test_loader, desc="Inference", leave=False):
        xb = xb.to(device, non_blocking=True)
        out = resnet(xb)
        preds = torch.argmax(out, dim=1).detach().cpu().numpy().tolist()
        y_preds.extend(preds)

len(y_preds), y_preds[:10]



## === cell 13
"""Submission CSV"""
df_sub = pd.DataFrame({"image_id": test_image_ids, "label": y_preds})
assert df_sub.shape[0] == submission_df.shape[0], (df_sub.shape, submission_df.shape)

df_sub.to_csv("submission.csv", index=False)
df_sub.head()



## === cell 14
print("Wrote:", os.path.abspath("submission.csv"))
print(pd.read_csv("submission.csv").head())
print("Rows:", pd.read_csv("submission.csv").shape[0])
