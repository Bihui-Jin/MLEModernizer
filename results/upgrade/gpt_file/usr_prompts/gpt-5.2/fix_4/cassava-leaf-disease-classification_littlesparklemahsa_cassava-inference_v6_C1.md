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

0.61061

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.61061) has done: 'I fix the runtime error by ensuring the fallback training routine runs with gradients enabled (it was accidentally executed under `@torch.no_grad()` because `inference()` is decorated). I also make checkpoint discovery safer by only accepting `.pth` files that look like model weights and skipping unrelated `.pth` files found under `/kaggle/input`, preventing silent bad loads. These changes keep the same model/augmentations/inference semantics and only affect correctness/stability so the notebook runs end-to-end and writes a valid `submission.csv`. If valid pretrained checkpoints are present, the fallback training path won’t be used and score behavior remains consistent.'

# 9. Code solution

## === cell 0
import os
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import timm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class CFG:
    img_size = 384
    batch_size = 64
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "/kaggle/input/cassava-leaf-disease-classification"
    train_csv = f"{data_root}/train.csv"
    sample_sub = f"{data_root}/sample_submission.csv"
    train_dir = f"{data_root}/train_images"
    test_dir = f"{data_root}/test_images"

    model_paths = [
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold0.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold1.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold2.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold3.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold4.pth",
    ]

    epochs = 1
    lr = 2e-4


test_tfms = A.Compose(
    [
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

train_tfms = A.Compose(
    [
        A.Resize(CFG.img_size, CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, folder, sample_submission_path=None):
        self.folder = folder
        if sample_submission_path and os.path.exists(sample_submission_path):
            ss = pd.read_csv(sample_submission_path)
            self.image_ids = ss["image_id"].tolist()
            self.paths = [str(Path(folder) / img_id) for img_id in self.image_ids]
        else:
            self.paths = sorted([str(p) for p in Path(folder).glob("*.jpg")])
            self.image_ids = [os.path.basename(p) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_tfms(image=img)["image"]
        return img, self.image_ids[idx]


class TrainDataset(Dataset):
    def __init__(self, df, folder):
        self.df = df.reset_index(drop=True)
        self.folder = folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(Path(self.folder) / row["image_id"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = train_tfms(image=img)["image"]
        y = int(row["label"])
        return img, y


class CassavaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=5
        )

    def forward(self, x):
        return self.backbone(x)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "net", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    new_sd = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_sd[nk] = v
    return new_sd


def _looks_like_state_dict_file(path: str) -> bool:
    try:
        ckpt = torch.load(path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        head_keys = [
            k for k in sd.keys() if k.endswith("head.weight") or k.endswith("head.bias")
        ]
        if not head_keys:
            head_keys = [
                k
                for k in sd.keys()
                if k.endswith("classifier.weight") or k.endswith("classifier.bias")
            ]
        if not head_keys:
            return False
        for k in head_keys:
            v = sd.get(k, None)
            if hasattr(v, "shape") and len(v.shape) >= 1 and int(v.shape[0]) == 5:
                return True
        return False
    except Exception:
        return False


def discover_checkpoints(preferred_paths):
    existing = [p for p in preferred_paths if os.path.exists(p)]
    if existing:
        return existing

    found = []
    for p in Path("/kaggle/input").rglob("*.pth"):
        sp = str(p)
        found.append(sp)
    found = sorted(found)

    filtered = [p for p in found if _looks_like_state_dict_file(p)]
    return filtered




## === cell 1
def train_fallback_model():
    with torch.enable_grad():
        if not os.path.exists(CFG.train_csv):
            raise FileNotFoundError(f"Missing train.csv at {CFG.train_csv}")
        if not os.path.isdir(CFG.train_dir):
            raise FileNotFoundError(f"Missing train_images dir at {CFG.train_dir}")

        df = pd.read_csv(CFG.train_csv)

        ds = TrainDataset(df, CFG.train_dir)
        loader = DataLoader(
            ds,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=(CFG.device == "cuda"),
            drop_last=True,
        )

        model = CassavaModel().to(CFG.device)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
        criterion = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler(enabled=(CFG.device == "cuda"))

        autocast_ctx = (
            torch.autocast(device_type="cuda")
            if CFG.device == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )

        for epoch in range(CFG.epochs):
            pbar = tqdm(loader, desc=f"Fallback training epoch {epoch+1}/{CFG.epochs}")
            running = 0.0
            for imgs, y in pbar:
                imgs = imgs.to(CFG.device, non_blocking=(CFG.device == "cuda"))
                y = y.to(CFG.device, non_blocking=(CFG.device == "cuda"))

                opt.zero_grad(set_to_none=True)
                with autocast_ctx:
                    logits = model(imgs)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                running += float(loss.detach().cpu())
                pbar.set_postfix(loss=running / max(1, (pbar.n + 1)))

        model.eval()
        return model




## === cell 2
@torch.no_grad()
def inference():
    model_paths = discover_checkpoints(CFG.model_paths)

    dataset = TestDataset(CFG.test_dir, sample_submission_path=CFG.sample_sub)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda")
        if CFG.device == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    if len(model_paths) == 0:
        print(
            "No suitable .pth checkpoints found under /kaggle/input; using fallback training."
        )
        model = train_fallback_model()

        preds = []
        for imgs, _ in tqdm(loader, leave=False, desc="Inference (fallback model)"):
            imgs = imgs.to(CFG.device, non_blocking=(CFG.device == "cuda"))
            with autocast_ctx:
                p1 = torch.softmax(model(imgs), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            preds.append(((p1 + p2) / 2).detach().cpu().numpy())
        ensemble_preds = np.concatenate(preds, axis=0)

    else:
        print(f"Using {len(model_paths)} checkpoint(s) for ensemble inference.")
        ensemble_preds = None

        for fold, path in enumerate(model_paths):
            print(f"Loading ckpt {fold} → {path} ({CFG.device})")
            model = CassavaModel().to(CFG.device)

            ckpt = torch.load(path, map_location=CFG.device)
            state_dict = _extract_state_dict(ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(
                    f"  load_state_dict notes: missing={len(missing)}, unexpected={len(unexpected)}"
                )

            model.eval()

            fold_preds = []
            for imgs, _ in tqdm(loader, leave=False, desc=f"Model {fold} inference"):
                imgs = imgs.to(CFG.device, non_blocking=(CFG.device == "cuda"))
                with autocast_ctx:
                    p1 = torch.softmax(model(imgs), dim=1)
                    p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
                fold_preds.append(((p1 + p2) / 2).detach().cpu().numpy())

            fold_preds = np.concatenate(fold_preds, axis=0)
            ensemble_preds = (
                fold_preds if ensemble_preds is None else (ensemble_preds + fold_preds)
            )

            del model, ckpt, state_dict
            if CFG.device == "cuda":
                torch.cuda.empty_cache()

        ensemble_preds = ensemble_preds / len(model_paths)

    final_labels = np.argmax(ensemble_preds, axis=1)

    sub = pd.DataFrame({"image_id": dataset.image_ids, "label": final_labels})
    if os.path.exists(CFG.sample_sub):
        ss = pd.read_csv(CFG.sample_sub)[["image_id"]]
        sub = ss.merge(sub, on="image_id", how="left")
    sub["label"] = sub["label"].astype(int)

    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSUBMISSION READY → {len(sub)} predictions → {out_path}")
    print(sub.head())
    return sub


inference()
