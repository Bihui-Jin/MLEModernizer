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
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"



## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
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
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
tta_transform = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)



## === cell 7
_PRE_RESIZE_CLAHE = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
    ]
)
_PRE_NORM_TOTENSOR = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def _pre_resize_clahe_np(image: np.ndarray) -> np.ndarray:
    return _PRE_RESIZE_CLAHE(image=image)["image"]


_PRE_CACHE = {}


def tta_predict_single_model(
    model, image, base_transform, tta_transform, device, n_tta=5, cache_key=None
):
    model.eval()

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Image must have shape (H, W, 3)")

    if cache_key is not None and cache_key in _PRE_CACHE:
        pre_np = _PRE_CACHE[cache_key]
    else:
        pre_np = _pre_resize_clahe_np(image)
        if cache_key is not None:
            _PRE_CACHE[cache_key] = pre_np

    augmented_list = [tta_transform(image=pre_np)["image"] for _ in range(n_tta)]
    batch = torch.stack(augmented_list, dim=0)

    batch = batch.to(device, non_blocking=True).contiguous(
        memory_format=torch.channels_last
    )

    with torch.no_grad():
        output = model(batch)
        probs = F.softmax(output, dim=1)
        avg_probs = probs.mean(dim=0, keepdim=True)
    return avg_probs


def project_imagenet_probs_to_5(
    imagenet_probs_1000: torch.Tensor, proto_5x1000: torch.Tensor
) -> torch.Tensor:
    logits5 = imagenet_probs_1000 @ proto_5x1000.t()
    return F.softmax(logits5, dim=1)




## === cell 8
def _stable_seed_from_key(key: str, base_seed: int = SEED) -> int:
    h = 2166136261
    for b in key.encode("utf-8", "ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return (h ^ base_seed) & 0xFFFFFFFF


def _make_tta_batch_from_pre_np_deterministic(
    pre_np: np.ndarray, img_key: str, n_tta: int
) -> torch.Tensor:
    s = _stable_seed_from_key(img_key, SEED)
    py_rng = random.Random(s)
    np_rng = np.random.default_rng(s)

    st_py = random.getstate()
    st_np = np.random.get_state()
    random.seed(py_rng.randint(0, 2**32 - 1))
    np.random.seed(int(np_rng.integers(0, 2**32 - 1)))
    try:
        augmented_list = [tta_transform(image=pre_np)["image"] for _ in range(n_tta)]
    finally:
        random.setstate(st_py)
        np.random.set_state(st_np)

    return torch.stack(augmented_list, dim=0)


class CassavaTestTTADataset(Dataset):
    def __init__(self, dataframe, image_dir, n_tta: int):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.n_tta = int(n_tta)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(self.dataframe.iloc[idx, 0])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pre_np = _pre_resize_clahe_np(image)
        tta_batch = _make_tta_batch_from_pre_np_deterministic(
            pre_np, img_name, self.n_tta
        )  # CPU tensor [T,3,384,384]
        return tta_batch, img_name


def identity_collate(batch):
    return batch


test_dataset = CassavaTestTTADataset(test_df, test_image_dir, num_tta)

_num_workers = min(8, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    collate_fn=identity_collate,
)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 10
def _maybe_load_ckpt(model: nn.Module, ckpt_path: str, device: torch.device) -> bool:
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)
        return True
    return False


def build_efficientnet_v2_s(pretrained_backbone=True):
    weights = (
        models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    return models.efficientnet_v2_s(weights=weights)


def build_resnet50(pretrained_backbone=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    return models.resnet50(weights=weights)


efficientnet_model_1 = build_efficientnet_v2_s(pretrained_backbone=True)
efficientnet_model_7 = build_efficientnet_v2_s(pretrained_backbone=True)
efficientnet_model_8 = build_efficientnet_v2_s(pretrained_backbone=True)
resnet_model = build_resnet50(pretrained_backbone=True)

_loaded = {}
_loaded["resnet"] = _maybe_load_ckpt(
    resnet_model,
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    device,
)
_loaded["eff1"] = _maybe_load_ckpt(
    efficientnet_model_1,
    "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth",
    device,
)
_loaded["eff7"] = _maybe_load_ckpt(
    efficientnet_model_7,
    "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth",
    device,
)
_loaded["eff8"] = _maybe_load_ckpt(
    efficientnet_model_8,
    "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth",
    device,
)

for m in (
    efficientnet_model_1,
    efficientnet_model_7,
    efficientnet_model_8,
    resnet_model,
):
    m.to(device)
    m.eval()
    if device.type == "cuda":
        m.to(memory_format=torch.channels_last)

print("Device:", device)
print("Loaded checkpoints:", _loaded)



## === cell 11
train_df = pd.read_csv(train_csv_path)


class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, train_transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.t = train_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        y = int(self.df.iloc[idx]["label"])
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = self.t(image=image)["image"]
        return x, y


train_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomResizedCrop(size=(384, 384), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.10, rotate_limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def _set_effnet_head_5(m: nn.Module):
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, 5)
        return
    raise RuntimeError("Unexpected EfficientNetV2 classifier structure")


def _set_resnet_head_5(m: nn.Module):
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 5)
        return
    raise RuntimeError("Unexpected ResNet classifier structure")


def _needs_finetune_any(_loaded_dict):
    return not (
        _loaded_dict.get("eff1", False)
        and _loaded_dict.get("eff7", False)
        and _loaded_dict.get("eff8", False)
        and _loaded_dict.get("resnet", False)
    )


def finetune_models_if_needed(max_epochs=1, train_bs=24, lr=2e-4):
    need = _needs_finetune_any(_loaded)
    if not need:
        print("All competition checkpoints present; skipping finetuning.")
        return

    to_train = []
    if not _loaded.get("eff1", False):
        _set_effnet_head_5(efficientnet_model_1)
        to_train.append(("eff1", efficientnet_model_1))
    if not _loaded.get("eff7", False):
        _set_effnet_head_5(efficientnet_model_7)
        to_train.append(("eff7", efficientnet_model_7))
    if not _loaded.get("eff8", False):
        _set_effnet_head_5(efficientnet_model_8)
        to_train.append(("eff8", efficientnet_model_8))
    if not _loaded.get("resnet", False):
        _set_resnet_head_5(resnet_model)
        to_train.append(("resnet", resnet_model))

    if len(to_train) == 0:
        print("No models require finetuning.")
        return

    train_ds = CassavaTrainDataset(train_df, train_image_dir, train_transform)
    w = torch.tensor(
        train_df["label"].value_counts().sort_index().values, dtype=torch.float32
    )
    w = (w.sum() / w).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    _nw = min(8, (os.cpu_count() or 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=_nw,
        pin_memory=True,
        persistent_workers=(_nw > 0),
        prefetch_factor=4 if _nw > 0 else None,
    )

    for key, model in to_train:
        model.to(device)
        model.train()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        for epoch in range(max_epochs):
            running = 0.0
            n = 0
            for xb, yb in tqdm(
                train_loader,
                desc=f"Finetune {key} epoch {epoch+1}/{max_epochs}",
                leave=False,
            ):
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running += float(loss.detach().cpu().item()) * xb.size(0)
                n += xb.size(0)

            print(f"{key}: epoch {epoch+1} train loss: {running / max(n,1):.4f}")

        model.eval()

    for key, _m in to_train:
        _loaded[key] = True
    print("Finetuning complete. Updated loaded flags:", _loaded)


finetune_models_if_needed(max_epochs=1, train_bs=24, lr=2e-4)



## === cell 12
_train_ids_by_class = {
    cls: train_df.loc[train_df["label"].values == cls, "image_id"].tolist()
    for cls in range(5)
}


_DUMMY_1x3x384x384 = torch.zeros(1, 3, 384, 384, device=device).contiguous(
    memory_format=torch.channels_last
)


def _collect_prototypes_for_model(
    model: nn.Module, max_per_class: int = 16, proto_bs: int = 16
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(_DUMMY_1x3x384x384)
        d = int(out.shape[1])

    if d == 5:
        proto = torch.eye(5, device=device)
        proto = proto / proto.sum(dim=1, keepdim=True)
        return proto

    proto_sum = torch.zeros(5, d, device=device)
    proto_cnt = torch.zeros(5, device=device)

    for cls in range(5):
        cls_ids = _train_ids_by_class[cls][:max_per_class]
        if len(cls_ids) == 0:
            continue

        xs = []
        valid = 0
        for img_id in cls_ids:
            img_path = os.path.join(train_image_dir, img_id)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pre_np = _pre_resize_clahe_np(img)
            x = _PRE_NORM_TOTENSOR(image=pre_np)["image"]
            xs.append(x)
            valid += 1

        if valid == 0:
            continue

        for i in range(0, len(xs), proto_bs):
            xb = torch.stack(xs[i : i + proto_bs], dim=0).to(device, non_blocking=True)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.no_grad():
                p = F.softmax(model(xb), dim=1)  # [B, d]
            proto_sum[cls] += p.sum(dim=0)
            proto_cnt[cls] += p.shape[0]

    proto = proto_sum / proto_cnt.clamp_min(1.0).unsqueeze(1)
    proto = proto / proto.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return proto


protos = {}
with torch.no_grad():
    for k, m in [
        ("eff1", efficientnet_model_1),
        ("eff7", efficientnet_model_7),
        ("eff8", efficientnet_model_8),
        ("resnet", resnet_model),
    ]:
        d = int(m(_DUMMY_1x3x384x384).shape[1])
        if d != 5:
            protos[k] = _collect_prototypes_for_model(m, max_per_class=16, proto_bs=16)

print("Prototype dims:", {k: tuple(v.shape) for k, v in protos.items()})




## === cell 13
def predict_5class_probs_from_tta_batch(
    model_key: str, model: nn.Module, tta_batch_gpu: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        output = model(tta_batch_gpu)
        probs = F.softmax(output, dim=1)
        avg_probs = probs.mean(dim=0, keepdim=True)
    C = int(avg_probs.shape[1])
    if C == 5:
        return avg_probs
    proto = protos[model_key]
    return project_imagenet_probs_to_5(avg_probs, proto)


ensemble_predictions = {}

with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        tta_batch_cpu, img_name = batch[0]
        if isinstance(img_name, (list, tuple)):
            img_name = img_name[0]
        img_name = str(img_name)

        tta_batch_gpu = tta_batch_cpu.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )

        probs1 = predict_5class_probs_from_tta_batch(
            "eff1", efficientnet_model_1, tta_batch_gpu
        )
        probs7 = predict_5class_probs_from_tta_batch(
            "eff7", efficientnet_model_7, tta_batch_gpu
        )
        probs8 = predict_5class_probs_from_tta_batch(
            "eff8", efficientnet_model_8, tta_batch_gpu
        )
        probs_res = predict_5class_probs_from_tta_batch(
            "resnet", resnet_model, tta_batch_gpu
        )

        combined_probs = (probs1 + probs7 + probs8 + probs_res) / 4.0
        pred = int(combined_probs.argmax(dim=1).cpu().item())
        ensemble_predictions[img_name] = pred



## === cell 14
ordered_preds = test_df["image_id"].map(ensemble_predictions)
ordered_preds = ordered_preds.fillna(0).astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_preds.values}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Saved submission to: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(test_df))
print("Label distribution:", submission_df["label"].value_counts().to_dict())
