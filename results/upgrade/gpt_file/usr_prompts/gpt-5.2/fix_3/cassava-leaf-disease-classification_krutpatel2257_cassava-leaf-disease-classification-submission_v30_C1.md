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

import numpy as np
import pandas as pd
from PIL import Image

import albumentations as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    from efficientnet_pytorch import EfficientNet  # noqa: F401
except Exception:
    EfficientNet = None



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
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

model_path = "../input/rn-tta-calr-clahe/model(21).pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(sample_sub_path):
    alt = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    if os.path.exists(alt):
        sample_sub_path = alt

if not os.path.exists(test_images_path):
    alt = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    if os.path.exists(alt):
        test_images_path = alt

if not os.path.exists(train_csv_path):
    alt = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    if os.path.exists(alt):
        train_csv_path = alt

if not os.path.exists(train_images_path):
    alt = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if os.path.exists(alt):
        train_images_path = alt

print("device:", device)
print("sample_sub_path:", sample_sub_path)
print("test_images_path exists:", os.path.exists(test_images_path))
print("train_csv_path exists:", os.path.exists(train_csv_path))
print("train_images_path exists:", os.path.exists(train_images_path))




## === cell 3
def build_model():
    if config["MODEL_TYPE"] == "EFFICIENT_NET_B4":
        if EfficientNet is None:
            raise ImportError(
                "efficientnet_pytorch is not available, but MODEL_TYPE requests EfficientNet."
            )
        model_ = EfficientNet.from_pretrained(
            "efficientnet-b4", num_classes=config["CLASSES"]
        )
    elif config["MODEL_TYPE"] == "RESNET_50":
        model_ = models.resnext50_32x4d(pretrained=False)
        model_.fc = nn.Linear(2048, config["CLASSES"])
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config['MODEL_TYPE']}")
    return model_


def find_existing_weight_file():
    if os.path.exists(model_path):
        return model_path

    likely_dirs = [
        "../input",
        "/kaggle/input",
    ]
    likely_patterns = [
        "*/**/*.pth",  # dataset subdir weights
        "*.pth",  # direct weights
    ]

    candidates = []
    for base in likely_dirs:
        if not os.path.exists(base):
            continue
        for pat in likely_patterns:
            candidates.extend(glob.glob(os.path.join(base, pat), recursive=True))

    if not candidates:
        return None

    def score(p):
        name = p.lower()
        s = 0
        if "cassava" in name:
            s += 3
        if "resnext" in name or "resnet" in name or "rn" in name:
            s += 2
        if "model" in name:
            s += 1
        return s

    candidates = sorted(
        set(candidates), key=lambda p: (score(p), os.path.getsize(p)), reverse=True
    )
    return candidates[0] if candidates else None


def load_weights_if_available(model_):
    weight_path = find_existing_weight_file()
    if weight_path is None:
        print("No pretrained .pth weights found in input; will train from scratch.")
        return False

    try:
        state = torch.load(weight_path, map_location=device)
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state
        missing, unexpected = model_.load_state_dict(state, strict=False)
        print(f"Loaded weights from: {weight_path}")
        if missing:
            print(f"Missing keys (showing up to 10): {missing[:10]}")
        if unexpected:
            print(f"Unexpected keys (showing up to 10): {unexpected[:10]}")
        return True
    except Exception as e:
        print(f"Found a .pth but failed to load ({weight_path}): {e}")
        print("Will train from scratch.")
        return False




## === cell 4
model = build_model().to(device)

has_weights = load_weights_if_available(model)
model.eval()



## === cell 5
torch.backends.cudnn.benchmark = True

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

train_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.6, 1.0), interpolation=1, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.3
        ),
        A.CLAHE(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

val_aug = A.Compose(
    [
        A.Resize(512, 512, interpolation=1, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

to_tensor = transforms.ToTensor()


class CassavaDataset(Dataset):
    def __init__(self, df, images_dir, aug):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_id"])
        img = np.array(Image.open(img_path).convert("RGB"))
        img = self.aug(image=img)["image"]
        x = to_tensor(img).float()
        y = int(row["label"])
        return x, y


def train_one_epoch(model_, loader, optimizer, criterion):
    model_.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        out = model_(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_one_epoch(model_, loader, criterion):
    model_.eval()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        out = model_(x)
        loss = criterion(out, y)

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return running_loss / max(total, 1), correct / max(total, 1)


if not has_weights:
    df = pd.read_csv(train_csv_path)
    val_frac = 0.1
    val_idx = []
    for lbl, g in df.groupby("label"):
        n_val = max(1, int(len(g) * val_frac))
        val_idx.extend(g.sample(n=n_val, random_state=seed).index.tolist())
    val_df = df.loc[val_idx].reset_index(drop=True)
    trn_df = df.drop(index=val_idx).reset_index(drop=True)

    train_ds = CassavaDataset(trn_df, train_images_path, train_aug)
    val_ds = CassavaDataset(val_df, train_images_path, val_aug)

    nw = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["VAL_BATCH_SIZE"],
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
        drop_last=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["SGD"]["LR"],
        momentum=config["SGD"]["MOMENTUM"],
        weight_decay=config["SGD"]["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["NUM_EPOCHS"], eta_min=config["COS_ANN_LR"]["ETA_MIN"]
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    for epoch in range(config["NUM_EPOCHS"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion)
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch+1:02d}/{config['NUM_EPOCHS']} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        torch.save(model.state_dict(), config["MODEL_PATH"])
        print(
            f"Saved trained model to: {config['MODEL_PATH']} (best val_acc={best_acc:.4f})"
        )

    model.eval()



## === cell 6
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.2, 1.0), interpolation=1, p=1.0),
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
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 7
@torch.no_grad()
def predict_tta(model_, image_path, tta_count=10):
    img0 = np.array(Image.open(image_path).convert("RGB"))
    logits_sum = None
    for _ in range(tta_count):
        img = sub_aug(image=img0)["image"]
        x = to_tensor(img).float().to(device)
        out = model_(x.unsqueeze(0))
        logits_sum = out if logits_sum is None else (logits_sum + out)
    logits_avg = logits_sum / float(tta_count)
    return int(logits_avg.argmax(dim=1).item())


@torch.no_grad()
def predict_tta_batched(model_, img0_np, tta_count=10, batch_size=10):
    logits_sum = None
    done = 0
    while done < tta_count:
        bs = min(batch_size, tta_count - done)
        xs = []
        for _ in range(bs):
            img = sub_aug(image=img0_np)["image"]
            xs.append(to_tensor(img).float())
        x = torch.stack(xs, dim=0).to(device, non_blocking=True)
        out = model_(x)  # [bs, C]
        s = out.sum(dim=0, keepdim=True)  # [1, C]
        logits_sum = s if logits_sum is None else (logits_sum + s)
        done += bs
    logits_avg = logits_sum / float(tta_count)
    return int(logits_avg.argmax(dim=1).item())




## === cell 8
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 10

predictions = []
model.eval()

tta_batch_size = 10  # keep identical TTA count; just process all views at once
for image_id in sample_sub["image_id"].values:
    img_path = os.path.join(test_images_path, image_id)
    img0 = np.array(Image.open(img_path).convert("RGB"))
    pred_label = predict_tta_batched(
        model, img0, tta_count=tta_count, batch_size=tta_batch_size
    )
    predictions.append([image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
out_path = config["DATA"]["SUB_OUTPUT"]
sub_df.to_csv(out_path, index=False)

print(sub_df.head())
print("Wrote submission to:", out_path)
print("Rows:", len(sub_df), "Cols:", list(sub_df.columns))
