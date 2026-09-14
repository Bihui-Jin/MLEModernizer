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
import glob
import random
from pathlib import Path

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models



## === cell 1
model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"


def _pick_first_existing_file(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _pick_first_existing_dir(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


sample_sub_path = _pick_first_existing_file(
    [
        sample_sub_path,
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

test_images_path = _pick_first_existing_dir(
    [
        test_images_path,
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/test_images",
    ]
)

train_csv_path = _pick_first_existing_file(
    [
        train_csv_path,
        "/kaggle/input/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/train.csv",
    ]
)

train_images_path = _pick_first_existing_dir(
    [
        train_images_path,
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/train_images",
    ]
)

print("sample_sub_path:", sample_sub_path)
print("test_images_path:", test_images_path)
print("train_csv_path:", train_csv_path)
print("train_images_path:", train_images_path)

if sample_sub_path is None or test_images_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv and/or test_images directory in this environment."
    )
if train_csv_path is None or train_images_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv and/or train_images directory in this environment."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 2
def _try_load_state_dict(path: str, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception:
        return None

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        return None

    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ("model.", "module.", "net.", "backbone.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v
    return new_state


def _find_5class_head_keypair(state: dict):
    if state is None:
        return None
    direct_prefixes = ("fc", "classifier", "head", "last_linear", "logits", "linear")
    for prefix in direct_prefixes:
        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"
        if w_key in state and b_key in state:
            w = state[w_key]
            b = state[b_key]
            try:
                if int(w.shape[0]) == 5 and int(b.shape[0]) == 5:
                    return (w_key, b_key)
            except Exception:
                pass
    keys = list(state.keys())
    for w_suf, b_suf in [
        (".fc.weight", ".fc.bias"),
        (".classifier.weight", ".classifier.bias"),
        (".head.weight", ".head.bias"),
        (".last_linear.weight", ".last_linear.bias"),
        (".logits.weight", ".logits.bias"),
        (".linear.weight", ".linear.bias"),
    ]:
        for k in keys:
            if k.endswith(w_suf):
                bkey = k[: -len(w_suf)] + b_suf
                if bkey in state:
                    w = state[k]
                    b = state[bkey]
                    try:
                        if int(w.shape[0]) == 5 and int(b.shape[0]) == 5:
                            return (k, bkey)
                    except Exception:
                        pass
    return None


def _looks_like_resnext_backbone(state: dict) -> bool:
    if state is None or not isinstance(state, dict) or len(state) < 50:
        return False
    required_any = [
        "conv1.weight",
        "bn1.weight",
        "bn1.bias",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
        "layer4.0.conv1.weight",
    ]
    present = sum([1 for k in required_any if k in state])
    return present >= 4


def _looks_like_5class_resnext_state(state: dict) -> bool:
    return _looks_like_resnext_backbone(state) and (
        _find_5class_head_keypair(state) is not None
    )


def resolve_checkpoint_path(primary_path: str, device: torch.device) -> str:
    known_first = [
        "/kaggle/input/cassava-resnext50-32x4d-f9f6d0/resnext50_32x4d-f9f6d0.pth",
        "../input/cassava-resnext50-32x4d-f9f6d0/resnext50_32x4d-f9f6d0.pth",
        "/kaggle/data/cassava-resnext50-32x4d-f9f6d0/resnext50_32x4d-f9f6d0.pth",
        primary_path,
    ]
    for p in known_first:
        if p and os.path.exists(p):
            st = _try_load_state_dict(p, device)
            if _looks_like_5class_resnext_state(st):
                return p
    return primary_path


resolved_model_path = resolve_checkpoint_path(model_path, device)
print("Resolved model checkpoint:", resolved_model_path)
print("Checkpoint exists:", os.path.exists(resolved_model_path))



## === cell 3
torch.backends.cudnn.benchmark = True

model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, 5)
model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

loaded_from_ckpt = False
if resolved_model_path and os.path.exists(resolved_model_path):
    state = _try_load_state_dict(resolved_model_path, device)
    if _looks_like_5class_resnext_state(state):
        head_pair = _find_5class_head_keypair(state)
        if head_pair is not None:
            w_key, b_key = head_pair
            if (w_key, b_key) != ("fc.weight", "fc.bias"):
                state = dict(state)
                state["fc.weight"] = state[w_key]
                state["fc.bias"] = state[b_key]
                print(
                    f"Remapped checkpoint head '{w_key[:-7]}.*' -> 'fc.*' for loading."
                )

        try:
            model.load_state_dict(state, strict=True)
            loaded_from_ckpt = True
            print("Loaded checkpoint with strict=True")
        except RuntimeError as e:
            print(
                "Warning: strict=True load failed, retrying strict=False. Error:",
                str(e)[:300],
            )
            model.load_state_dict(state, strict=False)
            loaded_from_ckpt = True
            print("Loaded checkpoint with strict=False")
    else:
        print(
            "Found checkpoint file but it didn't validate as 5-class ResNeXt; will train instead."
        )



## === cell 4
sub_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

train_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(size=(512, 512), scale=(0.6, 1.0)),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.2),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.Resize(512, 512),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


def hwc_float_to_chw_tensor(x_hwc: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(x_hwc.transpose(2, 0, 1))  # CHW
    return torch.from_numpy(x).float()




## === cell 5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(
    False
)  # keep original behavior (cudnn.benchmark True); do not force slow deterministic kernels

train_df = pd.read_csv(train_csv_path)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_images_path, x)
)

if not loaded_from_ckpt:
    idx = np.arange(len(train_df))
    np.random.shuffle(idx)
    val_size = int(0.1 * len(idx))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)

    class CassavaDataset(Dataset):
        def __init__(self, df, aug):
            self.df = df
            self.aug = aug

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            img = np.array(Image.open(row.image_path).convert("RGB"))
            img = self.aug(image=img)["image"]
            x = hwc_float_to_chw_tensor(img)
            y = int(row.label)
            return x, y

    train_ds = CassavaDataset(tr_df, train_aug)
    valid_ds = CassavaDataset(va_df, valid_aug)

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 3

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_correct += int((logits.argmax(1) == yb).sum().item())
            tr_total += xb.size(0)

        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if device.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += float(loss.item()) * xb.size(0)
                va_correct += int((logits.argmax(1) == yb).sum().item())
                va_total += xb.size(0)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {tr_loss/tr_total:.4f} acc {tr_correct/tr_total:.4f} | "
            f"valid loss {va_loss/va_total:.4f} acc {va_correct/va_total:.4f}"
        )

model.eval()



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 10
image_ids = sample_sub["image_id"].to_numpy()


class TestImageDataset(Dataset):
    def __init__(self, image_ids, images_dir):
        self.image_ids = image_ids
        self.images_dir = images_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        img_path = os.path.join(self.images_dir, image_id)
        image_np = np.array(Image.open(img_path).convert("RGB"))
        return image_np, image_id


def _collate_np_and_ids(batch):
    imgs, ids = zip(*batch)
    return list(imgs), list(ids)


_num_workers = min(4, (os.cpu_count() or 2))
test_ds = TestImageDataset(image_ids, test_images_path)

test_loader = DataLoader(
    test_ds,
    batch_size=8 if device.type == "cuda" else 2,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=False,  # pinning doesn't help for numpy lists; we pin after tensor creation
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    collate_fn=_collate_np_and_ids,
)

predictions = []
model.eval()

with torch.inference_mode():
    for images_np, batch_ids in test_loader:
        tta_tensors = []
        for img_np in images_np:
            for _ in range(tta_count):
                aug_img = sub_aug(image=img_np)["image"]
                tta_tensors.append(hwc_float_to_chw_tensor(aug_img))
        flat = torch.stack(tta_tensors, dim=0)

        flat = flat.to(device, non_blocking=True)
        if device.type == "cuda":
            flat = flat.contiguous(memory_format=torch.channels_last)

        outputs = model(flat)  # [B*T, 5]
        b = len(batch_ids)
        outputs = outputs.view(b, tta_count, -1).mean(dim=1)  # [B, 5]
        pred_labels = outputs.argmax(dim=1).detach().cpu().tolist()

        for image_id, lbl in zip(batch_ids, pred_labels):
            predictions.append([image_id, int(lbl)])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
print("submission.csv exists:", os.path.exists("submission.csv"))
