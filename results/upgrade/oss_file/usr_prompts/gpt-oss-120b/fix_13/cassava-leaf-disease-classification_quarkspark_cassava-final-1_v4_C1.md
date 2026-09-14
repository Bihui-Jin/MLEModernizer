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

geopandas==0.14.4
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
import os, sys, time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sz = 224
proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"




## === cell 1
def softmax_torch(X, theta=1.0, axis=None):
    if axis is None:
        axis = 1 if X.dim() > 1 else 0
    X = X * float(theta)
    return F.softmax(X, dim=axis)




## === cell 2
def light_model(num_classes, pretrained=True):
    squeezenet_custom = torchvision.models.squeezenet1_0(pretrained=pretrained)
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_4 = light_model(4, pretrained=True)  # minority classifier
squeezenet_custom_2 = light_model(2, pretrained=True)  # binary classifier

squeezenet_custom_4 = torch.compile(squeezenet_custom_4)
squeezenet_custom_2 = torch.compile(squeezenet_custom_2)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(400),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 4
minority_idx = {0: 0, 1: 1, 2: 2, 3: 4}
inverse_minority_idx = {v: k for k, v in minority_idx.items()}

binary_idx = {0: 0, 1: 3}




## === cell 5
class CassavaDataset(Dataset):
    def __init__(self, df, img_root, transform, binary=False):
        self.paths = [os.path.join(img_root, img_id) for img_id in df["image_id"]]
        if binary:
            self.labels = (df["label"] == 3).astype(int).values
        else:
            self.labels = df["label"].map(inverse_minority_idx).astype(int).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        img = self.transform(img)
        label = self.labels[idx]
        return img, label




## === cell 6
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

train_df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))

val_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
train_df = train_df.drop(val_df.index).reset_index(drop=True)

train_min_df = train_df[train_df["label"] != 3].reset_index(drop=True)
val_min_df = val_df[val_df["label"] != 3].reset_index(drop=True)

batch_size = 64
num_workers = 8  # increased for faster data loading

common_loader_kwargs = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

train_min_loader = DataLoader(
    CassavaDataset(train_min_df, train_dir, leaf_transform, binary=False),
    **common_loader_kwargs,
)

val_min_loader = DataLoader(
    CassavaDataset(val_min_df, train_dir, leaf_transform, binary=False),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

train_bin_loader = DataLoader(
    CassavaDataset(train_df, train_dir, leaf_transform, binary=True),
    **common_loader_kwargs,
)

val_bin_loader = DataLoader(
    CassavaDataset(val_df, train_dir, leaf_transform, binary=True),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 7
def train_model(model, loader, epochs=30, lr=1e-4):
    """Train a model for a few more epochs with a smaller learning‑rate.
    This helps the network converge better and raise accuracy toward the target."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # mixed‑precision scaler
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)  # [B, C, 1, 1]
                loss = criterion(outputs.squeeze(-1).squeeze(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(loader.dataset):.4f}")


print("Training minority (4‑class) model...")
train_model(squeezenet_custom_4, train_min_loader, epochs=30, lr=1e-4)

print("\nTraining binary (2‑class) model...")
train_model(squeezenet_custom_2, train_bin_loader, epochs=30, lr=1e-4)

squeezenet_custom_4.to(device).eval()
squeezenet_custom_2.to(device).eval()




## === cell 8
def prediction_logic_batch(minority_logits, binary_logits, thresh_3=0.5):
    preds_minority = F.softmax(minority_logits.squeeze(-1).squeeze(-1), dim=1)  # [B,4]
    preds_binary = F.softmax(binary_logits.squeeze(-1).squeeze(-1), dim=1)  # [B,2]

    minority_cls = torch.argmax(preds_minority, dim=1).cpu().numpy()
    binary_cls = torch.argmax(preds_binary, dim=1).cpu().numpy()
    cls_3_prob = preds_binary[:, 1].cpu().numpy()  # prob of “healthy”

    use_binary = cls_3_prob >= thresh_3

    minority_map = np.array([minority_idx[i] for i in range(4)])  # [0,1,2,4]
    binary_map = np.array([binary_idx[i] for i in range(2)])  # [0,3]

    pred = np.empty_like(minority_cls)
    pred[~use_binary] = minority_map[minority_cls[~use_binary]]
    pred[use_binary] = binary_map[binary_cls[use_binary]]
    return pred




## === cell 9
val_paths = [os.path.join(train_dir, img_id) for img_id in val_df["image_id"]]
val_labels = val_df["label"].to_numpy()

all_minority_logits = []
all_binary_logits = []

batch_size = 64
for start in range(0, len(val_paths), batch_size):
    batch_paths = val_paths[start : start + batch_size]
    imgs = torch.stack(
        [leaf_transform(Image.open(p).convert("RGB")) for p in batch_paths]
    ).to(device, non_blocking=True)

    imgs_flipped = torch.flip(imgs, dims=[3])

    with torch.inference_mode():
        minority_out_orig = squeezenet_custom_4(imgs)
        binary_out_orig = squeezenet_custom_2(imgs)
        minority_out_flip = squeezenet_custom_4(imgs_flipped)
        binary_out_flip = squeezenet_custom_2(imgs_flipped)

    minority_out = (minority_out_orig + minority_out_flip) / 2.0
    binary_out = (binary_out_orig + binary_out_flip) / 2.0

    all_minority_logits.append(minority_out.cpu())
    all_binary_logits.append(binary_out.cpu())

minority_logits_all = torch.cat(all_minority_logits, dim=0)
binary_logits_all = torch.cat(all_binary_logits, dim=0)

thresholds = np.arange(0.30, 0.71, 0.02)  # finer search
best_thresh = 0.5
best_acc = 0.0
for t in thresholds:
    pred = prediction_logic_batch(minority_logits_all, binary_logits_all, thresh_3=t)
    acc = (pred == val_labels).mean()
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

optimal_thresh = best_thresh
print(f"Optimal binary threshold = {optimal_thresh:.3f} (val accuracy {best_acc:.4f})")



## === cell 10
test_image_ids = sample_df["image_id"].tolist()
test_paths = [os.path.join(test_dir, img_id) for img_id in test_image_ids]

test_preds = []
batch_size = 64
for start in range(0, len(test_paths), batch_size):
    batch_ids = test_image_ids[start : start + batch_size]
    batch_paths = test_paths[start : start + batch_size]
    imgs = torch.stack(
        [leaf_transform(Image.open(p).convert("RGB")) for p in batch_paths]
    ).to(device, non_blocking=True)

    imgs_flipped = torch.flip(imgs, dims=[3])

    with torch.inference_mode():
        minority_out_orig = squeezenet_custom_4(imgs)
        binary_out_orig = squeezenet_custom_2(imgs)
        minority_out_flip = squeezenet_custom_4(imgs_flipped)
        binary_out_flip = squeezenet_custom_2(imgs_flipped)

    minority_out = (minority_out_orig + minority_out_flip) / 2.0
    binary_out = (binary_out_orig + binary_out_flip) / 2.0

    batch_pred = prediction_logic_batch(
        minority_out, binary_out, thresh_3=optimal_thresh
    )
    test_preds.extend(zip(batch_ids, batch_pred.tolist()))



## === cell 11
sub = pd.DataFrame.from_records(test_preds, columns=["image_id", "label"])
assert len(sub) == len(sample_df), "Submission length mismatch"
os.chdir("/kaggle/working")
sub.to_csv("submission.csv", index=False)
