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

3.11

# 3. Installed packages

albumentations==2.0.8
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
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
timm==1.0.19
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
transformers==4.53.3

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
import sys
import os
import random
import json
import gc
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from functools import partial
from albumentations import (
    Compose,
    Resize,
    Normalize,
    HorizontalFlip,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

base_path = "/kaggle/input/cassava-leaf-disease-classification/"
train_image_path = os.path.join(base_path, "train_images")
test_image_path = os.path.join(base_path, "test_images")

train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
image_files = sorted(
    [
        f
        for f in os.listdir(test_image_path)
        if os.path.splitext(f)[1].lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
)
submission_df = pd.DataFrame({"image_id": image_files, "label": 0})




## === cell 1
used_models_pytorch = {}
fallback_to_imagenet = True




## === cell 2
class CustomResNext(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(num_classes=5)
        else:
            if hasattr(self.model, "fc"):
                n_features = self.model.fc.in_features
                self.model.fc = nn.Linear(n_features, 5)
            elif hasattr(self.model, "classifier"):
                if isinstance(self.model.classifier, nn.Linear):
                    n_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(n_features, 5)
                else:
                    modules = list(self.model.classifier.children())
                    for i in reversed(range(len(modules))):
                        if isinstance(modules[i], nn.Linear):
                            n_features = modules[i].in_features
                            modules[i] = nn.Linear(n_features, 5)
                            break
                    self.model.classifier = nn.Sequential(*modules)
            elif hasattr(self.model, "head"):
                n_features = self.model.head.in_features
                self.model.head = nn.Linear(n_features, 5)
            else:
                raise AttributeError("Unable to locate classifier layer to replace.")

        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)


class TestDataset(Dataset):
    """Loads all test images once to avoid repeated I/O during TTA."""

    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.file_names = df["image_id"].values
        self.transform = transform
        self.cached_images = []
        for fn in self.file_names:
            img_path = os.path.join(self.image_dir, fn)
            try:
                with Image.open(img_path) as img:
                    arr = np.array(img.convert("RGB"))
            except Exception:
                arr = np.zeros((512, 512, 3), dtype=np.uint8)
            self.cached_images.append(arr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        if self.transform:
            aug = self.transform(image=image)
            image = aug["image"]
        return image


class TrainDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.file_names = df["image_id"].values
        self.labels = df["label"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, file_name)
        try:
            with Image.open(img_path) as img:
                image = np.array(img.convert("RGB"))
        except Exception:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = int(self.labels[idx])
        return image, label




## === cell 3
def get_transforms(flip_type=None):
    """Transforms for inference (optional deterministic flip)."""
    transforms = [
        Resize(512, 512),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    if flip_type == "h":
        transforms.insert(0, HorizontalFlip(p=1.0))
    elif flip_type == "v":
        transforms.insert(0, VerticalFlip(p=1.0))
    return Compose(transforms)


def get_train_transforms():
    """Light augmentation for training."""
    return Compose(
        [
            Resize(512, 512),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def load_checkpoint_state(model, ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        print(f"Warning: could not load checkpoint {ckpt_path}: {e}")
        return False


def inference_one_pass(model, ckpt_paths, test_loader, device):
    model.to(device)
    model.eval()
    for ckpt in ckpt_paths:
        if ckpt is not None and load_checkpoint_state(model, ckpt):
            break
    probs = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type):
                logits = model(images)
                prob = logits.softmax(dim=1).cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0)


def inference_with_tta(model, ckpt_paths, test_loader, device):
    probs_orig = inference_one_pass(model, ckpt_paths, test_loader, device)

    hflip_dataset = TestDataset(
        submission_df, test_image_path, transform=get_transforms(flip_type="h")
    )
    hflip_loader = DataLoader(
        hflip_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    probs_hflip = inference_one_pass(model, ckpt_paths, hflip_loader, device)

    vflip_dataset = TestDataset(
        submission_df, test_image_path, transform=get_transforms(flip_type="v")
    )
    vflip_loader = DataLoader(
        vflip_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    probs_vflip = inference_one_pass(model, ckpt_paths, vflip_loader, device)

    return (probs_orig + probs_hflip + probs_vflip) / 3.0




## === cell 4
def train_one_epoch(model, loader, device, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device.type):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)


def fine_tune_model(model, train_loader, device, epochs=1, lr=1e-4):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(epochs):
        if scaler:
            loss = train_one_epoch(
                model, train_loader, device, optimizer, criterion, scaler
            )
        else:
            loss = train_one_epoch(
                model, train_loader, device, optimizer, criterion, scaler=None
            )
        print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")
    return model




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32

model_names = ["resnext50_32x4d", "tf_efficientnet_b0_ns", "convnext_tiny"]
ckpt_paths = []  # No external checkpoints; use ImageNet weights.

train_dataset = TrainDataset(
    train_df, train_image_path, transform=get_train_transforms()
)
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

test_dataset = TestDataset(submission_df, test_image_path, transform=get_transforms())
test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

all_probs = []
for name in model_names:
    print(f"\n--- Processing model: {name} ---")
    model = CustomResNext(name, pretrained=True)
    model = fine_tune_model(model, train_loader, device, epochs=3, lr=1e-4)
    probs = inference_with_tta(model, ckpt_paths, test_loader, device)
    all_probs.append(probs)
    torch.cuda.empty_cache()
    del model
    gc.collect()

pred_probs = np.mean(np.stack(all_probs), axis=0)
submission_df["label"] = np.argmax(pred_probs, axis=1)




## === cell 6
submission_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
