# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

root = "/kaggle"


class CSVDataset(Dataset):
    def __init__(
        self,
        annotations_df,
        img_dir,
        transform=None,
        target_transform=None,
        aug=True,
        is_train=False,
    ):
        self.df = annotations_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample = {"image": image}
        if self.is_train:
            label = int(self.df.iloc[idx, 1])
            sample["label"] = label
        return sample




## === cell 1
"""
Utility functions: transforms, model creation, checkpoint I/O.
"""


def get_transforms(aug=True):
    """Return (train_transform, val_transform)."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if aug:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, val_tf


def get_model(name, width_mult=1.0, dropout=0.0):
    """Instantiate a pretrained model with a 5‑class head."""
    if name.lower() == "resnet":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 5),
        )
        return model
    elif name.lower() == "mobilenet":
        model = models.mobilenet_v2(pretrained=True, width_mult=width_mult)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 5),
        )
        return model
    else:
        raise ValueError(f"Unsupported model name: {name}")


def save_model(model, name, tag, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_{tag}.pth")
    torch.save(model.state_dict(), path)


def load_model(model, name, tag, folder, device):
    path = os.path.join(folder, f"{name}_{tag}.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model




## === cell 2
"""
Configuration, data loading, training (light fine‑tuning), and model preparation.
"""


def resolve_dir(base_path, target):
    """Return the first existing directory among possible layouts."""
    candidates = [
        os.path.join(base_path, target),
        os.path.join(base_path, "cassava-leaf-disease-classification", target),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"{target} directory not found under {base_path}")


args = {}
args["name"] = "mobilenet_384_finetune"
args["batch_size"] = 32
args["width_mult"] = 1.8
args["dropout"] = 0.0
args["aug"] = True  # enable augmentations for training
args["model"] = "resnet"  # custom ResNet with 5‑class head
args["gpu_id"] = 0
args["epochs"] = 5  # a few epochs for reasonable accuracy
args["lr"] = 1e-4

data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")
save_dir = "/kaggle/working/pretrained"  # writable location

img_dir_test = resolve_dir(data_dir, "test_images")
test_files = [f for f in os.listdir(img_dir_test) if f.lower().endswith(".jpg")]
test_pd = pd.DataFrame({"image_id": test_files})

_, test_transforms = get_transforms(aug=False)

test_dataset = CSVDataset(
    test_pd,
    img_dir_test,
    transform=test_transforms,
    aug=False,
    is_train=False,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0
)

train_csv_path = os.path.join(data_dir, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_split = int(0.9 * len(train_df))
train_df_split = train_df.iloc[:val_split]
val_df_split = train_df.iloc[val_split:]

img_dir_train = resolve_dir(data_dir, "train_images")

train_transforms, val_transforms = get_transforms(aug=args["aug"])

train_dataset = CSVDataset(
    train_df_split,
    img_dir_train,
    transform=train_transforms,
    aug=args["aug"],
    is_train=True,
)

val_dataset = CSVDataset(
    val_df_split,
    img_dir_train,
    transform=val_transforms,
    aug=False,
    is_train=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args["batch_size"],
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=0,
)

device = f"cuda:{args['gpu_id']}" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"])

best_val_acc = 0.0
for epoch in range(1, args["epochs"] + 1):
    net.train()
    running_loss = 0.0
    for batch in train_loader:
        imgs = batch["image"].float().to(device)
        labels = batch["label"].long().to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].float().to(device)
            labels = batch["label"].long().to(device)
            outputs = net(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(
        f"Epoch {epoch}/{args['epochs']} - loss: {epoch_loss:.4f} - val_acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(net, args["name"], "best", save_dir)

net = load_model(net, args["name"], "best", save_dir, device)




## === cell 3
"""
Inference and submission generation.
"""

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


print(f"Number of total parameters: {human_format(num_params)}")

pred_list = []
net.eval()
with torch.no_grad():
    for data in test_dataloader:
        imgs = data["image"].float().to(device)
        outputs = net(imgs)
        pred_list += outputs.argmax(dim=1).tolist()

sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

pred_dict = dict(zip(test_pd["image_id"], pred_list))

fallback_label = int(pred_list[0]) if pred_list else 0
final_labels = sample_sub["image_id"].map(pred_dict).fillna(fallback_label).astype(int)

final_submission = sample_sub.copy()
final_submission["label"] = final_labels

submission_path = os.path.join("/kaggle/working", "submission.csv")
final_submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
