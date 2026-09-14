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
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

_NUM_WORKERS = min(4, os.cpu_count() or 0)
_PIN_MEMORY = torch.cuda.is_available()



## === cell 2
num_tta = 5



## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"



## === cell 4
sample_path_candidates = [
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
]
for _p in sample_path_candidates:
    if os.path.exists(_p):
        sample_path = _p
        break
else:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle paths."
    )

test_df = pd.read_csv(sample_path)
test_df.head()



## === cell 5
train_path_candidates = [
    "../input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
]
for _p in train_path_candidates:
    if os.path.exists(_p):
        train_path = _p
        break
else:
    raise FileNotFoundError("Could not find train.csv in expected Kaggle paths.")

train_df = pd.read_csv(train_path)
train_df.head()



## === cell 6
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)



## === cell 7
train_transform_384 = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_transform_224 = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

valid_transform_384 = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

valid_transform_224 = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 8
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 9
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)




## === cell 10
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



## === cell 11
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




## === cell 12
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    model.eval()

    if image.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (H, W, 3)")

    augmented_list = []
    for _ in range(n_tta):
        augmented = tta_transform(image=image)["image"]  # CHW tensor
        augmented_list.append(augmented)

    batch = torch.stack(augmented_list, dim=0).to(device, non_blocking=True)

    with torch.inference_mode():
        output = model(batch)  # (n_tta, num_classes)
        probs = F.softmax(output, dim=1)
        avg_probs = probs.mean(dim=0, keepdim=True)  # (1, num_classes)
    return avg_probs




## === cell 13
test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=g,
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)



## === cell 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 15
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 16
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## === cell 17
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## === cell 18
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.DEFAULT
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()



## === cell 19
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
mobile_model.eval()



## === cell 20
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, valid_idx = next(splitter.split(train_df["image_id"], train_df["label"]))
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[valid_idx].reset_index(drop=True)

train_ds_eff = CassavaTrainDataset(
    tr_df, train_image_dir, transform=train_transform_384
)
valid_ds_eff = CassavaTrainDataset(
    va_df, train_image_dir, transform=valid_transform_384
)

train_ds_mob = CassavaTrainDataset(
    tr_df, train_image_dir, transform=train_transform_224
)
valid_ds_mob = CassavaTrainDataset(
    va_df, train_image_dir, transform=valid_transform_224
)

train_loader_eff = DataLoader(
    train_ds_eff,
    batch_size=16,
    shuffle=True,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=g,
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)
valid_loader_eff = DataLoader(
    valid_ds_eff,
    batch_size=32,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=g,
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)

train_loader_mob = DataLoader(
    train_ds_mob,
    batch_size=32,
    shuffle=True,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=g,
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)
valid_loader_mob = DataLoader(
    valid_ds_mob,
    batch_size=64,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_PIN_MEMORY,
    persistent_workers=(_NUM_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=g,
    prefetch_factor=2 if _NUM_WORKERS > 0 else None,
)




## === cell 21
def _set_trainable(model, trainable: bool):
    for p in model.parameters():
        p.requires_grad = trainable


def fine_tune_model(
    model, train_loader, valid_loader, device, epochs=2, lr=3e-4, backbone_lr=1e-4
):
    model.train()

    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            ("classifier" in name)
            or (name.startswith("fc."))
            or (name == "fc.weight")
            or (name == "fc.bias")
        ):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()

        for xb, yb in tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in tqdm(
                valid_loader, desc=f"valid ep{ep+1}/{epochs}", leave=False
            ):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc




## === cell 22
eff_valid_acc = fine_tune_model(
    efficientnet_model_7,
    train_loader_eff,
    valid_loader_eff,
    device,
    epochs=2,
    lr=3e-4,
    backbone_lr=1e-4,
)
mob_valid_acc = fine_tune_model(
    mobile_model,
    train_loader_mob,
    valid_loader_mob,
    device,
    epochs=2,
    lr=5e-4,
    backbone_lr=2e-4,
)

print(f"Validation acc (efficientnet7): {eff_valid_acc:.4f}")
print(f"Validation acc (mobilenet):     {mob_valid_acc:.4f}")



## === cell 23
weight_efficientnet7 = 0.88
weight_mobilenet = 0.84

total_weight = weight_efficientnet7 + weight_mobilenet
weight_efficientnet7 /= total_weight
weight_mobilenet /= total_weight

ensemble_predictions = []
image_names = []

with torch.inference_mode():
    for images, img_names in tqdm(test_loader, total=len(test_loader)):
        image = images[0].numpy() if isinstance(images, torch.Tensor) else images[0]
        img_name = img_names[0]

        probs_efficientnet7 = tta_predict_single_model(
            efficientnet_model_7,
            image,
            tta_transform_efficientnet,
            device,
            n_tta=num_tta,
        )
        probs_mobilenet = tta_predict_single_model(
            mobile_model, image, tta_transform_mobilenet, device, n_tta=num_tta
        )

        combined_probs = (weight_efficientnet7 * probs_efficientnet7) + (
            weight_mobilenet * probs_mobilenet
        )
        final_pred = combined_probs.argmax(dim=1).cpu().item()

        ensemble_predictions.append(int(final_pred))
        image_names.append(str(img_name))



## === cell 24
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map)

submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(
    f"Saved {submission_path} with shape={submission_df.shape} and columns={list(submission_df.columns)}"
)
print(submission_df.head())
print("Any missing labels?", submission_df["label"].isna().any())
print("Rowcount matches sample?", len(submission_df) == len(test_df))
