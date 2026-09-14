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
scipy==1.15.3
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
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True



## === cell 1
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 2
resnet_model = models.resnext50_32x4d(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
resnet_model = resnet_model.to(device)
resnet_model.eval()

effnet_model = models.efficientnet_b4(pretrained=True)
effnet_model.classifier[1] = nn.Linear(effnet_model.classifier[1].in_features, 5)
effnet_model = effnet_model.to(device)
effnet_model.eval()

sub_aug = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

to_tensor = lambda img: torch.from_numpy(img).permute(2, 0, 1).float()

train_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_path)
class_counts = train_df["label"].value_counts().sort_index()
prior_probs = class_counts.values.astype(np.float32)
prior_probs = prior_probs / prior_probs.sum()  # shape (5,)



## === cell 3
train_images_path = "../input/cassava-leaf-disease-classification/train_images"


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, aug):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        with Image.open(img_path) as im:
            img = np.array(im.convert("RGB"))
        img = self.aug(image=img)["image"]
        img_tensor = to_tensor(img)
        label = int(row["label"])
        return img_tensor, label


val_frac = 0.10
val_size = int(len(train_df) * val_frac)
train_subset = train_df.iloc[:-val_size]
val_subset = train_df.iloc[-val_size:]

train_dataset = CassavaDataset(train_subset, train_images_path, sub_aug)
val_dataset = CassavaDataset(val_subset, train_images_path, sub_aug)

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

for param in resnet_model.parameters():
    param.requires_grad = False
for param in resnet_model.fc.parameters():
    param.requires_grad = True

for param in effnet_model.parameters():
    param.requires_grad = False
for param in effnet_model.classifier[1].parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [
        {"params": resnet_model.fc.parameters()},
        {"params": effnet_model.classifier[1].parameters()},
    ],
    lr=1e-3,
)

epochs = 2
for epoch in range(epochs):
    resnet_model.train()
    effnet_model.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        res_logits = resnet_model(imgs)
        eff_logits = effnet_model(imgs)
        loss = criterion(res_logits, labels) + criterion(eff_logits, labels)
        loss.backward()
        optimizer.step()
    resnet_model.eval()
    effnet_model.eval()
    correct = total = 0
    with torch.inference_mode():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            res_logits = resnet_model(imgs)
            eff_logits = effnet_model(imgs)
            probs = F.softmax((res_logits * 0.10) + (eff_logits * 0.90), dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} - Val acc: {acc:.4f}")



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

image_paths = [
    os.path.join(test_images_path, img_id) for img_id in sample_sub["image_id"]
]
raw_images = []
for p in image_paths:
    with Image.open(p) as im:
        raw_images.append(np.array(im.convert("RGB")))

N = len(raw_images)
batch_size = 64  # fits GPU memory

resnet_sum = torch.zeros(N, 5, device=device, dtype=torch.float32)
effnet_sum = torch.zeros(N, 5, device=device, dtype=torch.float32)

tta_count = 20

with torch.inference_mode():
    for _ in range(tta_count):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_imgs = raw_images[start:end]

            aug_tensors = [to_tensor(sub_aug(image=img)["image"]) for img in batch_imgs]
            batch_tensor = torch.stack(aug_tensors).to(device)  # (B, C, H, W)

            res_out = resnet_model(batch_tensor)  # (B, 5)
            eff_out = effnet_model(batch_tensor)  # (B, 5)

            resnet_sum[start:end] += res_out
            effnet_sum[start:end] += eff_out

resnet_logits = resnet_sum / tta_count
effnet_logits = effnet_sum / tta_count



## === cell 5
logits_ensemble = (resnet_logits * 0.10) + (effnet_logits * 0.90)  # (N,5)
ensemble_probs = F.softmax(logits_ensemble, dim=1).cpu().numpy()

prior_weight = 0.02
ensemble_probs = ensemble_probs * (1.0 - prior_weight) + prior_probs * prior_weight
ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

pred_labels = ensemble_probs.argmax(axis=1)

submission_df = pd.DataFrame({"image_id": sample_sub["image_id"], "label": pred_labels})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(submission_df.head())
