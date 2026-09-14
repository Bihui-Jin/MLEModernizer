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

0.4817165306739196

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10538) has done: 'I fix the Albumentations augmentation error by replacing the outdated `RandomResizedCrop` (which now requires a `size` argument) with a simple `Resize` transform that works with the installed version. This allows the augmentation pipeline to be created, and consequently the inference loop can run without the `NameError`. No other logic is changed, preserving the original model and prediction workflow, so the script now runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 0.84081) has done: 'I add a lightweight fine‑tuning stage that trains the existing EfficientNet‑B4 on the provided training images for one epoch using the same Albumentations pipeline. This keeps the model architecture unchanged, introduces only a minimal training loop, and should raise the validation‑style accuracy from the current ~0.10 toward the target 0.48 while still producing a valid `submission.csv`. No other logic is altered.'
- What this solution (achieved 0.25374) has done: 'I reduce the model’s predictive strength to move the accuracy down toward the target. First, I skip loading the fine‑tuned checkpoint and remove the one‑epoch training loop so the model stays at its ImageNet‑pretrained state. Second, I lower test‑time augmentation by using a single TTA pass (tta_count = 1). These minimal adjustments keep the original architecture and pipeline intact while expectedly lowering the validation‑style score from 0.84 toward the target range.'
- What this solution (achieved 0.83445) has done: 'I add a very small fine‑tuning phase (one epoch) to give the pretrained EfficientNet‑B4 a modest accuracy boost, and increase test‑time augmentation passes from 1 to 5 to gain a little extra performance without altering the core model architecture or training logic. These minimal changes should raise the validation‑style score from ~0.25 toward the target 0.48 while keeping the pipeline simple and deterministic.'
- What this solution (achieved 0.83632) has done: 'The changes keep the exact model, loss, and training loop but speed up I/O and augmentation work: increase DataLoader workers and enable persistent workers for training, and rewrite the test‑time loop to use a batched DataLoader (removing the per‑image Python overhead) while still applying the same five TTA augmentations and keeping the original order of predictions.'
- What this solution (achieved 0.26046) has done: 'I lower the model’s predictive strength to move the validation‑style accuracy closer to the target (≈0.48).  
The changes are minimal:  
1) Skip the one‑epoch fine‑tuning so the model remains at its ImageNet‑pretrained state.  
2) Reduce test‑time augmentation passes from 5 to 1, removing the extra TTA boost.  
These adjustments keep the same architecture and data pipeline while expectedly decreasing the score toward the desired range.'
- What this solution (achieved 0.23505) has done: 'I re‑enable a very light fine‑tuning pass (one epoch) but reduce its impact by using a larger batch size (256) and a smaller learning rate so the model improves only modestly. Then I raise test‑time augmentation from 1 to 3 passes to give a gentle boost. These minimal changes are expected to move the validation‑style accuracy from ~0.26 toward the target ≈0.48 without overshooting.'
- What this solution (achieved 0.78214) has done: 'I reduced the training batch size from 256 to 32 to avoid CUDA OOM and kept the modest one‑epoch fine‑tuning, which should raise the validation‑style accuracy toward the target while preserving the original architecture and inference pipeline.'
- What this solution (achieved 0.41704) has done: 'I lower the validation‑style accuracy to move it toward the target by disabling the fine‑tuning pass (`do_train = False`) and reducing test‑time augmentation to a single pass (`tta_count = 1`). These minimal changes keep the core model and data pipeline unchanged while safely decreasing the score into the desired range and still producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I enable a lightweight fine‑tuning stage that uses only a small random subset (20 %) of the training data and a lower learning rate, then increase test‑time augmentation to three passes. These minimal tweaks should raise the validation‑style accuracy from 0.417 toward the target 0.482 without drastically changing the original pipeline.'
- What this solution (achieved 0.61099) has done: 'I lower the test‑time augmentation from three passes to a single pass, which should reduce the validation‑style accuracy from the current 0.61 toward the target of ~0.48 while keeping the existing architecture and fine‑tuning unchanged. This minimal change keeps the core logic intact and only adjusts the `tta_count` parameter.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights



## === cell 1
model_path = "../input/en-b4-tta-calr-clahe/model(14).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 2
sub_aug = A.Compose(
    [
        A.Resize(height=512, width=512, p=1.0),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(
            clip_limit=(1.0, 20.0), tile_grid_size=(32, 32), always_apply=True, p=1.0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


class CassavaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = np.array(Image.open(img_path).convert("RGB"))
        aug = self.transform(image=image)
        img = aug["image"]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        label = int(row["label"])
        return img_tensor, label


do_train = True  # keep fine‑tuning enabled
train_frac = 0.20  # use 20 % of training data
np.random.seed(42)  # deterministic subset selection
train_dataset = CassavaDataset(train_csv_path, train_images_path, sub_aug)

subset_size = int(train_frac * len(train_dataset))
subset_indices = np.random.choice(len(train_dataset), size=subset_size, replace=False)

train_loader = DataLoader(
    torch.utils.data.Subset(train_dataset, subset_indices),
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # gentle learning rate

if do_train:
    model.train()
    for epoch in range(1):  # single epoch
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/1] - Loss: {epoch_loss:.4f}")

model.eval()  # ensure eval mode for inference



## === cell 3
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 1  # reduce TTA passes to lower accuracy toward target


class CassavaTestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.image_ids = df["image_id"].tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        image = np.array(Image.open(img_path).convert("RGB"))
        aug = self.transform(image=image)
        img = aug["image"]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor, idx


test_dataset = CassavaTestDataset(sample_sub, test_images_path, sub_aug)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

num_samples = len(sample_sub)
num_classes = 5
pred_sum = torch.zeros(num_samples, num_classes, device=device)

with torch.no_grad():
    for _ in range(tta_count):
        for batch_imgs, batch_idxs in test_loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            outputs = model(batch_imgs)  # shape (B, 5)
            pred_sum[batch_idxs] += outputs

pred_avg = pred_sum / tta_count
_, pred_labels = torch.max(pred_avg, dim=1)

submission = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"],
        "label": pred_labels.cpu().numpy().astype(int),
    }
)
submission.to_csv("submission.csv", index=False)
print("Submission file saved. Preview:")
print(submission.head())
