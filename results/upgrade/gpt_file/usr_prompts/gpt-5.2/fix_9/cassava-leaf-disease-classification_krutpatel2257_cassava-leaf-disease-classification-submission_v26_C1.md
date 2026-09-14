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
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from PIL import Image

from collections import OrderedDict

try:
    from torchvision.io import read_image, ImageReadMode  # type: ignore

    _TV_READ_IMAGE = True
except Exception:
    _TV_READ_IMAGE = False



## === cell 1
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
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
DATA_ROOT = Path("../input/cassava-leaf-disease-classification")

train_csv_path = DATA_ROOT / "train.csv"
train_images_path = DATA_ROOT / "train_images"

sample_sub_path = DATA_ROOT / "sample_submission.csv"
test_images_path = DATA_ROOT / "test_images"

submission_path = config["DATA"]["SUB_OUTPUT"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

print("device:", device)
print("train_csv_path exists:", train_csv_path.exists())
print("train_images_path exists:", train_images_path.exists())
print("sample_sub_path exists:", sample_sub_path.exists())
print("test_images_path exists:", test_images_path.exists())



## === cell 3
train_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(p=0.3),
        A.CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

base_aug = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)




## === cell 4
def _read_rgb_hwc_uint8(img_path: str) -> np.ndarray:
    if _TV_READ_IMAGE:
        t = read_image(img_path, mode=ImageReadMode.RGB)  # CHW uint8
        return t.permute(1, 2, 0).contiguous().numpy()
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


class _LRUImageCacheMixin:
    def _get_cache(self):
        if getattr(self, "_lru", None) is None:
            self._lru = OrderedDict()
        return self._lru

    def _load_base_image(self, img_path: str):
        lru = self._get_cache()
        img = lru.get(img_path, None)
        if img is not None:
            lru.move_to_end(img_path, last=True)
            return img
        img = _read_rgb_hwc_uint8(img_path)
        lru[img_path] = img
        if len(lru) > self.cache_size:
            lru.popitem(last=False)
        return img


class CassavaTrainDataset(Dataset, _LRUImageCacheMixin):
    def __init__(
        self,
        image_paths: np.ndarray,
        labels: np.ndarray,
        transform=None,
        cache_size: int = 2048,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache_size = int(cache_size)
        self._lru = None

    def __len__(self):
        return int(self.image_paths.shape[0])

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = self._load_base_image(img_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = int(self.labels[idx])
        return image, label


class CassavaTestTTACachedDataset(Dataset, _LRUImageCacheMixin):
    def __init__(
        self,
        image_paths: np.ndarray,
        transform=None,
        tta_count: int = 5,
        cache_size: int = 2048,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.tta_count = int(tta_count)
        self.cache_size = int(cache_size)
        self._lru = None

    def __len__(self):
        return int(self.image_paths.shape[0]) * self.tta_count

    def __getitem__(self, idx: int):
        base_idx = idx // self.tta_count
        img_path = self.image_paths[base_idx]
        image = self._load_base_image(img_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, int(base_idx)




## === cell 5
if config["MODEL_TYPE"] == "RESNET_50":
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, config["CLASSES"])
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {config['MODEL_TYPE']}")

model = model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile unavailable, continuing. Reason:", repr(e))

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(np.int64)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
train_idx, val_idx = next(skf.split(train_df["image_id"], train_df["label"]))
trn_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

trn_paths = (train_images_path / trn_df["image_id"]).astype(str).to_numpy()
trn_labels = trn_df["label"].to_numpy(np.int64)

val_paths = (train_images_path / val_df["image_id"]).astype(str).to_numpy()
val_labels = val_df["label"].to_numpy(np.int64)

train_ds = CassavaTrainDataset(
    trn_paths, trn_labels, transform=train_aug, cache_size=4096
)
val_ds = CassavaTrainDataset(val_paths, val_labels, transform=base_aug, cache_size=4096)


def seed_worker(worker_id: int):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

_cpu = os.cpu_count() or 2
_num_workers = min(8, max(2, _cpu // 2))
pin = torch.cuda.is_available()
pin_device = "cuda" if pin else ""
_prefetch = 2 if _num_workers > 0 else None

train_loader = DataLoader(
    train_ds,
    batch_size=config["TRAIN_BATCH_SIZE"],
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=pin,
    pin_memory_device=pin_device if pin else "",
    drop_last=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=seed_worker if _num_workers > 0 else None,
    generator=g,
)
val_loader = DataLoader(
    val_ds,
    batch_size=config["VAL_BATCH_SIZE"],
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=pin,
    pin_memory_device=pin_device if pin else "",
    drop_last=False,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=seed_worker if _num_workers > 0 else None,
    generator=g,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["SGD"]["LR"],
    momentum=config["SGD"]["MOMENTUM"],
    weight_decay=config["SGD"]["WEIGHT_DECAY"],
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["NUM_EPOCHS"] * len(train_loader),
    eta_min=config["COS_ANN_LR"]["ETA_MIN"],
)



## === cell 6
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=1.0),
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
        A.CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)



## === cell 7
best_val_acc = -1.0
best_state = None

model_train = model.train
model_eval = model.eval
to_device = device
is_cuda = device.type == "cuda"
memfmt = torch.channels_last

for epoch in range(config["NUM_EPOCHS"]):
    model_train()
    train_loss = 0.0
    train_correct = 0
    n_train = 0

    for xb, yb in train_loader:
        xb = xb.to(to_device, non_blocking=True)
        if is_cuda and xb.is_contiguous(memory_format=memfmt) is False:
            xb = xb.contiguous(memory_format=memfmt)
        yb = yb.to(to_device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = xb.size(0)
        train_loss += loss.item() * bs
        train_correct += (logits.detach().argmax(1) == yb).sum().item()
        n_train += bs

    model_eval()
    val_loss = 0.0
    val_correct = 0
    n_val = 0

    with torch.inference_mode():
        for xb, yb in val_loader:
            xb = xb.to(to_device, non_blocking=True)
            if is_cuda and xb.is_contiguous(memory_format=memfmt) is False:
                xb = xb.contiguous(memory_format=memfmt)
            yb = yb.to(to_device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            val_loss += loss.item() * bs
            val_correct += (logits.argmax(1) == yb).sum().item()
            n_val += bs

    train_loss /= max(1, n_train)
    train_acc = train_correct / max(1, n_train)
    val_loss /= max(1, n_val)
    val_acc = val_correct / max(1, n_val)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    print(
        f"Epoch {epoch+1}/{config['NUM_EPOCHS']} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

if best_state is not None:
    torch.save(best_state, config["MODEL_PATH"])
    model.load_state_dict(best_state)
else:
    torch.save(model.state_dict(), config["MODEL_PATH"])

print("Best val acc:", best_val_acc)



## === cell 8
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 5

image_ids = sample_sub["image_id"].values
test_paths = (test_images_path / sample_sub["image_id"]).astype(str).to_numpy()

test_ds_tta = CassavaTestTTACachedDataset(
    image_paths=test_paths,
    transform=sub_aug,
    tta_count=tta_count,
    cache_size=8192,
)

test_loader_tta = DataLoader(
    test_ds_tta,
    batch_size=96,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=pin,
    pin_memory_device=pin_device if pin else "",
    drop_last=False,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=seed_worker if _num_workers > 0 else None,
    generator=g,
)

model.eval()

all_logits = torch.zeros((len(image_ids), config["CLASSES"]), dtype=torch.float32)

with torch.inference_mode():
    for xb, base_idx in test_loader_tta:
        xb = xb.to(device, non_blocking=True)
        if (
            device.type == "cuda"
            and xb.is_contiguous(memory_format=torch.channels_last) is False
        ):
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb).to(dtype=torch.float32).cpu()
        all_logits.index_add_(0, base_idx.to(torch.long), logits)

all_logits /= float(tta_count)
pred_labels = torch.argmax(all_logits, dim=1).numpy().astype(int)

sub_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
sub_df.to_csv(submission_path, index=False)
print(sub_df.head())
print("Wrote:", submission_path, "rows:", len(sub_df))
