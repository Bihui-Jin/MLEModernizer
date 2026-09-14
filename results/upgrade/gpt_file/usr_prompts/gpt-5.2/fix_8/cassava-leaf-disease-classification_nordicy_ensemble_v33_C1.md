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

# 5. Code solution

## === cell 0
import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
from PIL import Image  # kept (may be unused, but part of original imports)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



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
_tta_aug_only = A.Compose(common_transforms)

_post_eff = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

_post_mob = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def tta_predict_batch(model, images_np, tta_transform, device, n_tta=5):
    """
    Kept signature/behavior. Internally routes to cached two-stage TTA for the known transforms
    to reduce repeated work; otherwise falls back to original logic.
    """
    model.eval()
    if not isinstance(images_np, (list, tuple)):
        raise TypeError("images_np must be a list/tuple of HxWx3 numpy arrays")

    b = len(images_np)
    if b == 0:
        return torch.empty((0, 5), device=device)

    if tta_transform is tta_transform_efficientnet:
        post = _post_eff
    elif tta_transform is tta_transform_mobilenet:
        post = _post_mob
    else:
        post = None

    if post is not None:
        aug_list = []
        aug_list_extend = aug_list.extend  # minor Python overhead reduction
        for img in images_np:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"Image must have shape (H, W, 3). Got {img.shape}")

            tmp = []
            for _ in range(n_tta):
                a = _tta_aug_only(image=img)["image"]
                tmp.append(post(image=a)["image"])
            aug_list_extend(tmp)

        xb = torch.stack(aug_list, dim=0)  # (b*n_tta, C, H, W)
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            logits = model(xb)
            probs = F.softmax(logits, dim=1)

        return probs.view(b, n_tta, -1).mean(dim=1)

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

_bs = 32 if torch.cuda.is_available() else 4
_nw = min(8, (os.cpu_count() or 2)) if torch.cuda.is_available() else 0

test_loader = DataLoader(
    test_dataset,
    batch_size=_bs,
    shuffle=False,
    num_workers=_nw,
    collate_fn=identity_collate,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_nw > 0),
    prefetch_factor=4 if _nw > 0 else None,
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


def _make_class_weights(train_labels, num_classes=5):
    counts = np.bincount(
        np.asarray(train_labels, dtype=np.int64), minlength=num_classes
    )
    counts = np.maximum(counts, 1)
    w = counts.sum() / counts  # inverse frequency
    w = w / w.mean()  # normalize for stability
    return torch.tensor(w, dtype=torch.float32)


def train_head_epochs(
    model, train_loader, val_loader, device, class_weights=None, lr=3e-3, epochs=6
):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Training head (epoch {ep}/{epochs})"):
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

        scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        v_total = 0
        v_correct = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                v_correct += int((preds == yb).sum().detach().cpu())
                v_total += int(xb.size(0))
        val_acc = v_correct / max(1, v_total)

        print(
            f"epoch={ep} lr={optimizer.param_groups[0]['lr']:.2e} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_val_acc


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
    from sklearn.model_selection import StratifiedShuffleSplit

    y = train_df["label"].astype(int).values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    tr_idx, va_idx = next(splitter.split(train_df, y))
    train_df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
    train_df_va = train_df.iloc[va_idx].reset_index(drop=True)

    class_w = _make_class_weights(train_df_tr["label"].values, num_classes=5)

    batch_size_train = 64 if torch.cuda.is_available() else 16
    num_workers_train = min(4, os.cpu_count() or 2) if torch.cuda.is_available() else 0

    eff_train_ds = CassavaTrainDataset(
        train_df_tr, train_image_dir, transform=train_transform_384
    )
    eff_val_ds = CassavaTrainDataset(
        train_df_va, train_image_dir, transform=efficientnet_transforms
    )

    mob_train_ds = CassavaTrainDataset(
        train_df_tr, train_image_dir, transform=train_transform_224
    )
    mobile_val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    mob_val_ds = CassavaTrainDataset(
        train_df_va, train_image_dir, transform=mobile_val_transform
    )

    res_train_ds = CassavaTrainDataset(
        train_df_tr, train_image_dir, transform=train_transform_224
    )
    res_val_ds = CassavaTrainDataset(
        train_df_va, train_image_dir, transform=mobile_val_transform
    )

    eff_train_loader = DataLoader(
        eff_train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_train > 0),
    )
    eff_val_loader = DataLoader(
        eff_val_ds,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_train > 0),
    )

    mob_train_loader = DataLoader(
        mob_train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_train > 0),
    )
    mob_val_loader = DataLoader(
        mob_val_ds,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_train > 0),
    )

    res_train_loader = DataLoader(
        res_train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_train > 0),
    )
    res_val_loader = DataLoader(
        res_val_ds,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=num_workers_train,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers_train > 0),
    )

    freeze_backbone_effnet(efficientnet_model_7)
    best_val_eff = train_head_epochs(
        efficientnet_model_7,
        eff_train_loader,
        eff_val_loader,
        device,
        class_weights=class_w,
        lr=3e-3,
        epochs=6,
    )
    print(f"EffNet best_val_acc={best_val_eff:.4f}")

    freeze_backbone_mobilenet(mobile_model)
    best_val_mob = train_head_epochs(
        mobile_model,
        mob_train_loader,
        mob_val_loader,
        device,
        class_weights=class_w,
        lr=3e-3,
        epochs=6,
    )
    print(f"MobileNet best_val_acc={best_val_mob:.4f}")

    freeze_backbone_resnet(resnet_model)
    best_val_res = train_head_epochs(
        resnet_model,
        res_train_loader,
        res_val_loader,
        device,
        class_weights=class_w,
        lr=3e-3,
        epochs=6,
    )
    print(f"ResNet best_val_acc={best_val_res:.4f}")

resnet_model.eval()
efficientnet_model_7.eval()
mobile_model.eval()



## === cell 16
weight_efficientnet7 = 0.72
weight_mobilenet = 0.23
weight_resnet = 0.05

total_weight = weight_efficientnet7 + weight_mobilenet + weight_resnet
weight_efficientnet7 /= total_weight
weight_mobilenet /= total_weight
weight_resnet /= total_weight

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
        probs_resnet = tta_predict_batch(
            resnet_model,
            images,
            tta_transform_mobilenet,
            device,
            n_tta=num_tta,
        )

        combined_probs = (
            (weight_efficientnet7 * probs_efficientnet7)
            + (weight_mobilenet * probs_mobilenet)
            + (weight_resnet * probs_resnet)
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
