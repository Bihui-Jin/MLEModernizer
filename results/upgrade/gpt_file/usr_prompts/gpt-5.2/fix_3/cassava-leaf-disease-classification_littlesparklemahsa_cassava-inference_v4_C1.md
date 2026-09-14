# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.14

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class CFG:
    img_size = 384
    batch_size = 64
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = "/kaggle/input/cassava-leaf-disease-classification"
    test_dir = f"{root}/test_images"
    train_csv = f"{root}/train.csv"
    train_dir = f"{root}/train_images"
    sample_sub_path = f"{root}/sample_submission.csv"

    model_paths = [
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold0.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold1.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold2.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold3.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold4.pth",
    ]

    epochs = 2
    lr = 2e-4


seed_everything(42)

test_tfms = A.Compose(
    [
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

train_tfms = A.Compose(
    [
        A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.75, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([str(p) for p in Path(folder).glob("*.jpg")])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_tfms(image=img)["image"]
        return img, os.path.basename(img_path)


class TrainDataset(Dataset):
    def __init__(self, df, folder):
        self.df = df.reset_index(drop=True)
        self.folder = folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.folder, image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = train_tfms(image=img)["image"]
        return img, label


class CassavaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=5
        )

    def forward(self, x):
        return self.backbone(x)


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "network"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(
            "Checkpoint format not understood (expected dict-like state_dict)."
        )

    new_sd = {}
    for k, v in ckpt.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


def discover_checkpoints():
    """
    Bugfix: the originally referenced dataset may not exist in this environment.
    We'll keep the original paths if present; otherwise search for any .pth/.pt under /kaggle/input.
    """
    found = []
    for p in CFG.model_paths:
        if os.path.isfile(p):
            found.append(p)

    if found:
        return found

    search_roots = ["/kaggle/input", "/kaggle/working"]
    exts = (".pth", ".pt", ".bin")
    keywords = ("cassava", "convnext", "best", "fold")

    for root in search_roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for p in rootp.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                name = p.name.lower()
                if any(k in name for k in keywords):
                    found.append(str(p))

    uniq = []
    seen = set()
    for p in found:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3711698872.py in <cell line: 0>()
     67 train_tfms = A.Compose(
     68     [
---> 69         A.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.75, 1.0)),
     70         A.HorizontalFlip(p=0.5),
     71         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 1
def train_fallback_single_model():
    """
    Fallback path when no external checkpoints are available.
    Trains the same ConvNeXt-Tiny model briefly on train_images to enable a valid submission.
    """
    if not os.path.isfile(CFG.train_csv):
        raise FileNotFoundError(f"Missing train.csv at {CFG.train_csv}")
    if not os.path.isdir(CFG.train_dir):
        raise FileNotFoundError(f"Missing train_images directory at {CFG.train_dir}")

    df = pd.read_csv(CFG.train_csv)

    idx = np.arange(len(df))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(0.95 * len(df))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df, va_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(
        drop=True
    )

    tr_ds = TrainDataset(tr_df, CFG.train_dir)
    va_ds = TrainDataset(va_df, CFG.train_dir)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
        drop_last=True,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
        drop_last=False,
    )

    model = CassavaModel().to(CFG.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-2)

    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.device == "cuda"))

    for epoch in range(CFG.epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for imgs, labels in tqdm(
            tr_loader, desc=f"Train epoch {epoch+1}/{CFG.epochs}", leave=False
        ):
            imgs = imgs.to(CFG.device, non_blocking=True)
            labels = torch.as_tensor(labels, device=CFG.device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            if CFG.device == "cuda":
                with autocast(device_type="cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            tr_loss += loss.item() * bs
            tr_n += bs

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(
                va_loader, desc=f"Valid epoch {epoch+1}/{CFG.epochs}", leave=False
            ):
                imgs = imgs.to(CFG.device, non_blocking=True)
                labels = torch.as_tensor(labels, device=CFG.device, dtype=torch.long)
                if CFG.device == "cuda":
                    with autocast(device_type="cuda"):
                        logits = model(imgs)
                else:
                    logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        print(
            f"Epoch {epoch+1}/{CFG.epochs} - train_loss={tr_loss/max(tr_n,1):.4f} - val_acc={correct/max(total,1):.4f}"
        )

    return model




## === cell 2
@torch.no_grad()
def inference():
    print(f"Using device: {CFG.device}")

    dataset = TestDataset(CFG.test_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No test images found in {CFG.test_dir}")

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
    )

    ckpt_paths = discover_checkpoints()
    if len(ckpt_paths) == 0:
        print(
            "No checkpoints found. Falling back to quick training to generate a valid submission."
        )
        model = train_fallback_single_model()
        model.eval()

        all_probs = []
        for imgs, _ in tqdm(loader, leave=False, desc="Inference (fallback model)"):
            imgs = imgs.to(CFG.device, non_blocking=True)
            if CFG.device == "cuda":
                with autocast(device_type="cuda"):
                    p1 = torch.softmax(model(imgs), dim=1)
                    p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            else:
                p1 = torch.softmax(model(imgs), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            all_probs.append(((p1 + p2) / 2).cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        final_labels = np.argmax(probs, axis=1).astype(int)

    else:
        ckpt_paths = ckpt_paths[:5]
        print(f"Found {len(ckpt_paths)} checkpoint(s). Running ensemble inference.")

        ensemble_preds = None
        for fold, path in enumerate(ckpt_paths):
            print(f"Loading fold {fold} → {path}")

            model = CassavaModel().to(CFG.device)
            ckpt = torch.load(path, map_location=CFG.device)
            state_dict = _unwrap_state_dict(ckpt)
            model.load_state_dict(
                state_dict, strict=False
            )  # allow minor key mismatches across formats
            model.eval()

            fold_preds = []
            for imgs, _ in tqdm(loader, leave=False, desc=f"Fold {fold} TTA"):
                imgs = imgs.to(CFG.device, non_blocking=True)

                if CFG.device == "cuda":
                    with autocast(device_type="cuda"):
                        p1 = torch.softmax(model(imgs), dim=1)
                        p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
                else:
                    p1 = torch.softmax(model(imgs), dim=1)
                    p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)

                fold_preds.append(((p1 + p2) / 2).cpu().numpy())

            fold_preds = np.concatenate(fold_preds, axis=0)
            ensemble_preds = (
                fold_preds if ensemble_preds is None else ensemble_preds + fold_preds
            )

            del model, ckpt, state_dict
            if CFG.device == "cuda":
                torch.cuda.empty_cache()

        final_labels = np.argmax(ensemble_preds / len(ckpt_paths), axis=1).astype(int)

    sample = pd.read_csv(CFG.sample_sub_path)
    pred_df = pd.DataFrame(
        {
            "image_id": [os.path.basename(p) for p in dataset.paths],
            "label": final_labels,
        }
    )
    sub = sample[["image_id"]].merge(pred_df, on="image_id", how="left")

    if sub["label"].isna().any():
        missing = sub[sub["label"].isna()]["image_id"].head(10).tolist()
        raise RuntimeError(f"Missing predictions for some images (e.g. {missing}).")

    sub["label"] = sub["label"].astype(int)
    sub.to_csv("submission.csv", index=False)

    print(f"\nSUBMISSION READY → {len(sub)} predictions → submission.csv")
    print(sub.head())
    return sub


inference()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2213657621.py in <cell line: 0>()
    101 
    102 
--> 103 inference()

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/2213657621.py in inference()
      3     print(f"Using device: {CFG.device}")
      4 
----> 5     dataset = TestDataset(CFG.test_dir)
      6     if len(dataset) == 0:
      7         raise RuntimeError(f"No test images found in {CFG.test_dir}")

NameError: name 'TestDataset' is not defined
