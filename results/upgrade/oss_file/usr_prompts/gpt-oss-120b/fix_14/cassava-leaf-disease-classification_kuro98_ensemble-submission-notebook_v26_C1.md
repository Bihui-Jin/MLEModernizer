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

3.12

# 3. Installed packages

geopandas==0.14.4
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
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision import models
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")




## === cell 1
seed = 3407
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 2
class CassavaDataset(Dataset):
    """Dataset for Cassava images.

    For training it returns (image_tensor, label).
    For test it returns (image_tensor, image_id).
    """

    def __init__(self, img_dir, df=None, transform=None, is_test=False, preload=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        if df is not None:
            self.filenames = df["image_id"].tolist()
            self.labels = df["label"].tolist()
        else:
            all_entries = os.listdir(img_dir)
            self.filenames = [
                entry
                for entry in all_entries
                if os.path.isfile(os.path.join(img_dir, entry))
                and entry.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            self.labels = None

        self.preload = preload
        if self.preload:
            self._cached_images = []
            for fname in self.filenames:
                path = os.path.join(self.img_dir, fname)
                img = Image.open(path).convert("RGB")
                self._cached_images.append(img)
        else:
            self._cached_images = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self._cached_images is not None:
            img = self._cached_images[idx]
        else:
            fname = self.filenames[idx]
            path = os.path.join(self.img_dir, fname)
            img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        extra = self.filenames[idx] if self.is_test else self.labels[idx]
        return img, extra




## === cell 3
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

train_df = pd.read_csv(train_csv_path)

train_len = int(0.9 * len(train_df))
val_len = len(train_df) - train_len
train_split, val_split = random_split(
    train_df, [train_len, val_len], generator=torch.Generator().manual_seed(seed)
)

train_set_df = train_df.iloc[train_split.indices].reset_index(drop=True)
val_set_df = train_df.iloc[val_split.indices].reset_index(drop=True)




## === cell 4
train_transform = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

batch_size = 64
num_workers = 8

train_dataset = CassavaDataset(
    train_img_dir,
    df=train_set_df,
    transform=train_transform,
    is_test=False,
    preload=True,
)
val_dataset = CassavaDataset(
    train_img_dir, df=val_set_df, transform=test_transform, is_test=False, preload=True
)

pin_mem = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
    prefetch_factor=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
    prefetch_factor=2,
)




## === cell 5
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

if isinstance(model.heads, nn.Linear):
    in_features = model.heads.in_features
elif isinstance(model.heads, nn.Sequential):
    linear_layer = None
    for layer in model.heads.modules():
        if isinstance(layer, nn.Linear):
            linear_layer = layer
            break
    if linear_layer is None:
        raise RuntimeError("No linear layer found in model.heads")
    in_features = linear_layer.in_features
else:
    raise RuntimeError("Unexpected type for model.heads")

model.heads = nn.Linear(in_features, 5)
model = model.to(device)

if device.type == "cuda":
    model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

epochs = 12
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(train_loader)
    val_acc = evaluate(val_loader)
    scheduler.step()
    print(f"Epoch {epoch}: Train loss {train_loss:.4f} | Val Acc {val_acc:.4f}")




## === cell 6
test_dataset = CassavaDataset(
    test_img_dir, transform=test_transform, is_test=True, preload=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
    prefetch_factor=2,
)

model.eval()
all_names = []
all_preds = []

with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_names.extend(fnames)
        all_preds.extend(preds)

submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()
