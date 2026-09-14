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

3.13

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

0.8909035962526443

# 6. Current score

0.72272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50934) has done: 'I fix the Albumentations `RandomResizedCrop` API break (it now expects `size=(h,w)`), which is currently stopping execution and causing `tta_transform_*` to never be defined. I also remove dependencies on missing external weight files by switching to torchvision’s built-in pretrained weights (same architectures) so inference can run end-to-end in this environment. Finally, I ensure the submission rows exactly match `sample_submission.csv` order/length by collecting `image_id` strings correctly (not nested lists/tuples) and writing predictions back into the loaded sample submission frame before saving `submission.csv`.'
- What this solution (achieved 0.51532) has done: 'Your current score is low because all your models have randomly initialized 5-class classification heads (you replaced the ImageNet heads but never load cassava-trained weights), so predictions are effectively noise. To move the score toward the 0.8909 target with minimal changes and without altering your modeling approach, I (1) load cassava fine-tuned weights if they exist in the dataset directory, and only fall back to the current behavior if not found, and (2) fix the `image_names.append(str(img_name))` bug that currently creates keys like `"('xxx.jpg',)"` which breaks the `map()` alignment and forces many labels to be filled with 0. These two changes preserve your ensemble+TTA logic while making the submission correctly aligned and (when weights are available) meaningfully accurate. The script still run end-to-end and always write `submission.csv`.'
- What this solution (achieved 0.72272) has done: 'The timeout is dominated by per-image, per-TTA model calls: you run 2 models × 5 TTA × 2676 images with batch_size=1, and each TTA does its own transform+GPU forward, causing huge Python/IO overhead and poor GPU utilization. I keep identical TTA logic and models, but vectorize inference by batching multiple images and multiple TTA copies together, so each model runs far fewer forward passes while producing the same averaged probabilities. I also speed up the dataloader (more workers, pinned memory, persistent workers) without changing what images are read, and use inference_mode + channels_last + TF32 (on Ampere+) for faster but numerically negligible inference. No training logic, transforms, architectures, or weights handling are changed.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import copy
from tqdm import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)



## === cell 3
test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_transform_384 = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_transform_224 = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name


class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]["image_id"]
        label = int(self.dataframe.iloc[idx]["label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"]
        return image, label




## === cell 6
tta_transform = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)



## === cell 7
common_transforms = [
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.7),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
]

tta_transform_efficientnet = A.Compose(
    common_transforms
    + [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

tta_transform_mobilenet = A.Compose(
    common_transforms
    + [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 8
def tta_predict_batch(model, images_np, tta_transform, device, n_tta=5):
    model.eval()
    if not isinstance(images_np, (list, tuple)):
        raise TypeError("images_np must be a list/tuple of HxWx3 numpy arrays")

    b = len(images_np)
    if b == 0:
        return torch.empty((0, 5), device=device)

    aug_list = []
    for img in images_np:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Image must have shape (H, W, 3). Got {img.shape}")
        for _ in range(n_tta):
            aug_list.append(tta_transform(image=img)["image"])

    xb = torch.stack(aug_list, dim=0)  # (b*n_tta, C, H, W)
    xb = xb.to(device, non_blocking=True)
    if device.type == "cuda":
        xb = xb.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        logits = model(xb)
        probs = F.softmax(logits, dim=1)

    probs = probs.view(b, n_tta, -1).mean(dim=1)  # (b, num_classes)
    return probs


def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    model.eval()
    tta_predictions = []

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Image must have shape (H, W, 3). Got {image.shape}")

    with torch.no_grad():
        for _ in range(n_tta):
            augmented = tta_transform(image=image)["image"]
            augmented = augmented.unsqueeze(0).to(device)

            output = model(augmented)
            probs = F.softmax(output, dim=1)
            tta_predictions.append(probs)

    avg_probs = torch.mean(torch.stack(tta_predictions, dim=0), dim=0)
    return avg_probs




## === cell 9
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_bs = 16 if torch.cuda.is_available() else 4
_nw = min(4, os.cpu_count() or 2) if torch.cuda.is_available() else 0
test_loader = DataLoader(
    test_dataset,
    batch_size=_bs,
    shuffle=False,
    num_workers=_nw,
    collate_fn=identity_collate,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_nw > 0),
    prefetch_factor=2 if _nw > 0 else None,
)



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 11
def _find_first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def load_checkpoint_flexible(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded weights: {ckpt_path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {ckpt_path}")


CKPT_CANDIDATES = {
    "efficientnet_v2_s": [
        "/kaggle/input/cassava-leaf-disease-classification/efficientnet_v2_s.pth",
        "/kaggle/input/cassava-leaf-disease-classification/efficientnet_v2_s.pt",
        "/kaggle/input/cassava-leaf-disease-classification/efficientnet.pth",
        "/kaggle/input/cassava-leaf-disease-classification/effnet.pth",
        "/kaggle/input/cassava-leaf-disease-classification/model.pth",
        "/kaggle/input/cassava-leaf-disease-classification/weights.pth",
    ],
    "mobilenet_v3_large": [
        "/kaggle/input/cassava-leaf-disease-classification/mobilenet_v3_large.pth",
        "/kaggle/input/cassava-leaf-disease-classification/mobilenet_v3_large.pt",
        "/kaggle/input/cassava-leaf-disease-classification/mobilenet.pth",
        "/kaggle/input/cassava-leaf-disease-classification/mobile.pth",
    ],
    "resnet50": [
        "/kaggle/input/cassava-leaf-disease-classification/resnet50.pth",
        "/kaggle/input/cassava-leaf-disease-classification/resnet50.pt",
        "/kaggle/input/cassava-leaf-disease-classification/resnet.pth",
    ],
}




## === cell 12
def freeze_backbone_resnet(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def freeze_backbone_effnet(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def freeze_backbone_mobilenet(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def train_head_one_epoch(model, train_loader, device, lr=3e-3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    total = 0
    correct = 0
    running_loss = 0.0
    for xb, yb in tqdm(train_loader, desc="Training head (1 epoch)"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu()) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().detach().cpu())
        total += int(xb.size(0))

    return running_loss / max(1, total), correct / max(1, total)


train_df = pd.read_csv(train_csv_path)
train_df.head()



## === cell 13
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model = resnet_model.to(device)

efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_7 = efficientnet_model_7.to(device)

mobile_model = models.mobilenet_v3_large(
    weights=models.MobileNet_V3_Large_Weights.DEFAULT
)
num_features = mobile_model.classifier[0].in_features
hidden_dim = mobile_model.classifier[0].out_features
mobile_model.classifier = nn.Sequential(
    nn.Linear(num_features, hidden_dim),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.7),
    nn.Linear(hidden_dim, 5),
)
mobile_model = mobile_model.to(device)

if device.type == "cuda":
    resnet_model = resnet_model.to(memory_format=torch.channels_last)
    efficientnet_model_7 = efficientnet_model_7.to(memory_format=torch.channels_last)
    mobile_model = mobile_model.to(memory_format=torch.channels_last)



## === cell 14
resnet_ckpt = _find_first_existing(CKPT_CANDIDATES["resnet50"])
eff_ckpt = _find_first_existing(CKPT_CANDIDATES["efficientnet_v2_s"])
mobile_ckpt = _find_first_existing(CKPT_CANDIDATES["mobilenet_v3_large"])

if resnet_ckpt:
    load_checkpoint_flexible(resnet_model, resnet_ckpt, device)
if eff_ckpt:
    load_checkpoint_flexible(efficientnet_model_7, eff_ckpt, device)
if mobile_ckpt:
    load_checkpoint_flexible(mobile_model, mobile_ckpt, device)



## === cell 15
need_train = (resnet_ckpt is None) and (eff_ckpt is None) and (mobile_ckpt is None)
print(
    f"Found ckpts? resnet={bool(resnet_ckpt)} effnet={bool(eff_ckpt)} mobile={bool(mobile_ckpt)}"
)
print(f"Will train heads: {need_train}")

if need_train:
    batch_size_train = 32 if torch.cuda.is_available() else 16
    num_workers_train = 2 if torch.cuda.is_available() else 0

    eff_train_ds = CassavaTrainDataset(
        train_df, train_image_dir, transform=train_transform_384
    )
    mob_train_ds = CassavaTrainDataset(
        train_df, train_image_dir, transform=train_transform_224
    )

    eff_train_loader = DataLoader(
        eff_train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
    )
    mob_train_loader = DataLoader(
        mob_train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
    )

    freeze_backbone_effnet(efficientnet_model_7)
    loss_eff, acc_eff = train_head_one_epoch(
        efficientnet_model_7, eff_train_loader, device, lr=3e-3
    )
    print(f"EffNet head train loss={loss_eff:.4f} acc={acc_eff:.4f}")

    freeze_backbone_mobilenet(mobile_model)
    loss_mob, acc_mob = train_head_one_epoch(
        mobile_model, mob_train_loader, device, lr=3e-3
    )
    print(f"MobileNet head train loss={loss_mob:.4f} acc={acc_mob:.4f}")

resnet_model.eval()
efficientnet_model_7.eval()
mobile_model.eval()



## === cell 16
weight_efficientnet7 = 0.88
weight_mobilenet = 0.84

total_weight = weight_efficientnet7 + weight_mobilenet
weight_efficientnet7 /= total_weight
weight_mobilenet /= total_weight

ensemble_predictions = []
image_names = []

with torch.inference_mode():
    for batch in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
        images = [x[0] for x in batch]
        names = [x[1] for x in batch]

        probs_efficientnet7 = tta_predict_batch(
            efficientnet_model_7,
            images,
            tta_transform_efficientnet,
            device,
            n_tta=num_tta,
        )
        probs_mobilenet = tta_predict_batch(
            mobile_model,
            images,
            tta_transform_mobilenet,
            device,
            n_tta=num_tta,
        )

        combined_probs = (weight_efficientnet7 * probs_efficientnet7) + (
            weight_mobilenet * probs_mobilenet
        )
        final_preds = combined_probs.argmax(dim=1).detach().cpu().numpy().astype(int)

        ensemble_predictions.extend(final_preds.tolist())
        image_names.extend([str(n) for n in names])

len(ensemble_predictions), len(image_names), test_df.shape



## === cell 17
pred_map = dict(zip(image_names, ensemble_predictions))

submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map)

submission_df["label"] = submission_df["label"].fillna(0).astype(int)

assert len(submission_df) == len(test_df), "Submission length mismatch."
assert submission_df.columns.tolist() == [
    "image_id",
    "label",
], "Submission columns mismatch."

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Saved submission to: {submission_path}")
submission_df.head()
