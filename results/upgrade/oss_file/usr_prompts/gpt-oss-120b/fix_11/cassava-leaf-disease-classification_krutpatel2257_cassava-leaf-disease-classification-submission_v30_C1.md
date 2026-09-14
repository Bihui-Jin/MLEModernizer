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
try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm




## === cell 1
BASE_DATA_DIR = os.path.join("..", "input", "cassava-leaf-disease-classification")

config = {
    "DATA": {
        "IMAGES": os.path.join(
            BASE_DATA_DIR, "train_images"
        ),  # corrected training images folder
        "LABELS": os.path.join(
            BASE_DATA_DIR, "train.csv"
        ),  # corrected training csv path
        "SUB_IMAGES": os.path.join(BASE_DATA_DIR, "test_images"),
        "SUB_LABELS": os.path.join(BASE_DATA_DIR, "sample_submission.csv"),
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 16,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 15,
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "RESNET_50",
}




## === cell 2
sample_sub_path = config["DATA"]["SUB_LABELS"]
test_images_path = config["DATA"]["SUB_IMAGES"]




## === cell 3
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 on supported GPUs

if config["MODEL_TYPE"] == "EFFICIENT_NET_B4" and EfficientNet is not None:
    model = EfficientNet.from_pretrained(
        "efficientnet-b4", num_classes=config["CLASSES"]
    )
else:  # Default to ResNet‑50 (pretrained on ImageNet)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config["CLASSES"])

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)




## === cell 4
train_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.2, 1.0)),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.2, 1.0)),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)




## === cell 5
class CassavaDataset(Dataset):
    """
    Dataset that pre‑loads raw images into memory for O(1) disk access.
    Transformations remain applied on‑the‑fly to preserve randomness.
    """

    def __init__(self, images_dir, csv_path, transform):
        self.transform = transform
        df = pd.read_csv(csv_path)
        self.image_ids = df["image_id"].values
        self.labels = df["label"].astype(int).values
        self.raw_images = [
            np.array(Image.open(os.path.join(images_dir, img_id)).convert("RGB"))
            for img_id in self.image_ids
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_image = self.raw_images[idx]
        image = self.transform(image=raw_image)["image"]
        label = int(self.labels[idx])
        return image, label


train_dataset = CassavaDataset(
    images_dir=config["DATA"]["IMAGES"],
    csv_path=config["DATA"]["LABELS"],
    transform=train_aug,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["TRAIN_BATCH_SIZE"],
    shuffle=True,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["SGD"]["LR"],
    momentum=config["SGD"]["MOMENTUM"],
    weight_decay=config["SGD"]["WEIGHT_DECAY"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["NUM_EPOCHS"],
    eta_min=config["COS_ANN_LR"]["ETA_MIN"],
)

scaler = torch.cuda.amp.GradScaler()

if not os.path.isfile(config["MODEL_PATH"]):
    model.train()
    for epoch in range(config["NUM_EPOCHS"]):
        epoch_loss = 0.0
        for imgs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['NUM_EPOCHS']}"
        ):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config["MODEL_PATH"])
else:
    model.load_state_dict(torch.load(config["MODEL_PATH"], map_location=device))

model.eval()


class TestDataset(Dataset):
    """Loads raw test images lazily."""

    def __init__(self, images_dir, image_ids):
        self.images_dir = images_dir
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        raw = np.array(Image.open(os.path.join(self.images_dir, img_id)).convert("RGB"))
        return raw, img_id


def test_collate_fn(batch):
    """
    For each image in the batch, generate `tta_count` augmented tensors.
    Returns a stacked tensor of shape (batch*tta, C, H, W) and the list of ids.
    """
    images, ids = zip(*batch)
    aug_tensors = []
    for raw_img in images:
        for _ in range(10):  # tta_count = 10 (hard‑coded to keep original logic)
            aug = sub_aug(image=raw_img)["image"]
            aug_tensors.append(aug)
    batch_tensor = torch.stack(aug_tensors)
    return batch_tensor, list(ids)


test_ids = pd.read_csv(sample_sub_path)["image_id"].values
test_dataset = TestDataset(test_images_path, test_ids)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    collate_fn=test_collate_fn,
)



## === cell 6
predictions = []

with torch.no_grad():
    for aug_batch, batch_ids in tqdm(test_loader, desc="Predicting"):
        aug_batch = aug_batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(aug_batch)  # (batch*tta, classes)

        outputs = outputs.view(len(batch_ids), 10, -1).mean(dim=1)

        _, pred_labels = torch.max(outputs, dim=1)
        for img_id, pred in zip(batch_ids, pred_labels.tolist()):
            predictions.append([img_id, int(pred)])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv(config["DATA"]["SUB_OUTPUT"], index=False)
print("Submission saved to:", config["DATA"]["SUB_OUTPUT"])
print(sub_df.head())
