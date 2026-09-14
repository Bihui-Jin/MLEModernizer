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

No external packages required in the script and installed.

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

0.6145361136294953

# 6. Current score

0.85987

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.85987) has done: 'I remove the failing `imblearn` import (it’s incompatible with the installed sklearn), and fix the execution order/undefined variables by defining the ResNeXt model before creating the optimizer. Since the referenced pretrained weights file doesn’t exist, I fall back to torchvision’s built-in pretrained weights (same architecture) so the pipeline can run end-to-end and reach a reasonable accuracy target. I also fix the dataset bug (`sample` undefined), make transforms deterministic for validation/test, and fix test image enumeration to ignore nested directories so predictions align with `sample_submission.csv`. Finally, I write a valid `submission.csv` with the exact required columns and row order.'

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
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn import model_selection



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
torch.cuda.is_available()



## === cell 2
dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

df_train, df_valid = model_selection.train_test_split(
    dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
)



## === cell 3
print(df_train.label.value_counts())



## === cell 4
try:
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    resnet = models.resnext50_32x4d(weights=weights)
except Exception:
    resnet = models.resnext50_32x4d(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)
resnet = resnet.to(device)



## === cell 5
input_size = 384
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)



## === cell 6
_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

image_path = "../input/cassava-leaf-disease-classification/train_images/"
train_image_paths = [os.path.join(image_path, x) for x in _train.image_id.values]
valid_image_paths = [os.path.join(image_path, x) for x in df_valid.image_id.values]

train_targets = _train.label.values
valid_targets = df_valid.label.values



## === cell 7
dfx_full = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
image_path = "../input/cassava-leaf-disease-classification/train_images/"
full_train_image_paths = [os.path.join(image_path, x) for x in dfx_full.image_id.values]
full_train_targets = dfx_full.label.values



## === cell 8
len(train_image_paths), len(train_targets)



## === cell 9
len(valid_image_paths), len(valid_targets)



## === cell 10
"""torch module dataset"""


class CassavaDataset(Dataset):
    def __init__(self, data, targets=None, transform=None):
        self.files = list(data)
        self.targets = None if targets is None else np.asarray(targets)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = Image.open(name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.targets is None:
            return image
        label = int(self.targets[idx])
        return image, label




## === cell 11
"""Dataset Initialization"""
cassava_data = CassavaDataset(
    train_image_paths, train_targets, transform=train_transform
)
cassava_valid = CassavaDataset(
    valid_image_paths, valid_targets, transform=valid_transform
)



## === cell 12
batch_size = 16

cassava_loader = DataLoader(
    cassava_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
valid_loader = DataLoader(
    cassava_valid,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
classes = ("0", "1", "2", "3", "4")




## === cell 13
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_image(img_tensor, label):
    img_tensor = (
        denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0)).cpu().numpy()
    )
    plt.imshow(np.clip(img_tensor, 0, 1))
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()




## === cell 14
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)




## === cell 15
def run_one_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0
    correct = 0
    total_loss = 0.0

    for batch in loader:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)




## === cell 16
epochs = 2
for e in range(epochs):
    tr_loss, tr_acc = run_one_epoch(resnet, cassava_loader, optimizer=optimizer)
    va_loss, va_acc = run_one_epoch(resnet, valid_loader, optimizer=None)
    print(
        f"Epoch {e+1}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )



## === cell 17
resnet.eval()
va_loss, va_acc = run_one_epoch(resnet, valid_loader, optimizer=None)
print("Validation accuracy:", va_acc)



## === cell 18
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 19
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_files = []
for f in os.listdir(test_dir):
    fp = os.path.join(test_dir, f)
    if os.path.isfile(fp) and f.lower().endswith((".jpg", ".jpeg", ".png")):
        test_files.append(f)

required_order = submission_df["image_id"].tolist()
test_files_set = set(test_files)
missing = [x for x in required_order if x not in test_files_set]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images referenced by sample_submission, e.g. {missing[:5]}"
    )

test_paths = [os.path.join(test_dir, x) for x in required_order]
test_ds = CassavaDataset(test_paths, targets=None, transform=valid_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 20
resnet.eval()
all_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device, non_blocking=True)
        logits = resnet(images)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)

len(all_preds), len(required_order)



## === cell 21
sub = pd.DataFrame({"image_id": required_order, "label": all_preds})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
