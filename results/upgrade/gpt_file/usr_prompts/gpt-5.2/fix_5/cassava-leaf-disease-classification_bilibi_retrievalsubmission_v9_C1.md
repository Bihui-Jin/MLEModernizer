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

3.9

# 3. Installed packages

geopandas==0.14.4
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

# 5. Target score

0.8573587186461167

# 6. Current score

0.6988

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'Your notebook fails because it relies on a missing external Kaggle Dataset (`baseline-infer`) that provided the `data/`, `model/`, `configs/`, etc. python packages, so the imports and `build_transforms`/`net` construction never work. To keep the same end-to-end intent (load a ResNet50-based model and run test-time inference to create `submission.csv`), I replace those missing modules with a minimal, equivalent PyTorch ResNet50 classifier head and standard image transforms, and make inference follow the exact row order in `sample_submission.csv` so the submission length/order always matches. I also fix pathing to use the provided `/kaggle/input/...` structure, ensure all required imports exist in the cells that use them, and write a valid `submission.csv` with exactly 2676 rows.'
- What this solution (achieved 0.1065) has done: 'Your current score is low because the notebook is almost certainly running with random weights (the `baseline-weights` dataset path isn’t present in the provided file tree), so predictions are effectively random. To move the score toward your target with minimal change, I switch the model to use ImageNet-pretrained ResNet50 weights when the competition checkpoint isn’t found, which preserves the same architecture and inference flow but gives a strong, legitimate baseline. I also apply standard torchvision ImageNet preprocessing (Resize/CenterCrop/Normalize) instead of manual OpenCV resizing to better match the pretrained model’s expected input distribution. Everything else (submission order, format, no extra training) stays the same.'
- What this solution (achieved 0.73468) has done: 'Your current score (0.1065) is far below the target (0.8574), and the main reason is that you’re effectively using a randomly-initialized 5-class head when the competition checkpoint is missing, so predictions are near-random. To move the score up with minimal core-logic changes, I keep the same ResNet50 inference pipeline but (1) replace the random head with a cheap, legitimate linear probe trained on the provided `train.csv` images using the frozen ImageNet backbone, and (2) ensure the exact same ImageNet preprocessing is used for both train and test. This preserves the architecture (ResNet50 + linear head), loss (cross-entropy), and straightforward training loop, while producing a much stronger submission within the time limit. If the external weights file is present, the code still load it and skip training.'
- What this solution (achieved 0.6988) has done: 'Your current score (0.73468) is below the target (0.85736), so we should improve accuracy with the smallest legitimate changes while keeping the same ResNet50 + linear-head training/inference logic. The biggest low-risk gain is to train the same frozen-backbone linear head on the full train set (instead of a 6000-image subset) and to use the standard ImageNet augmentation for training only (RandomResizedCrop + RandomHorizontalFlip) while keeping test preprocessing unchanged. This preserves the architecture, loss (cross-entropy), and training loop, but usually yields a sizable accuracy bump for this competition. I also add a small weighted sampler to reduce class-imbalance harm without changing the model or objective.'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
import random

import numpy as np
import pandas as pd

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as T


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir: {TRAIN_IMG_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"
assert os.path.isfile(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
cfg = {
    "img_size": 512,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "batch_size": 64,
    "num_workers": 2,
    "num_classes": 5,
    "lr": 3e-3,
    "epochs": 4,
}

weights_path = "../input/baseline-weights/epoch_22.pth"

candidate_weights = [
    weights_path,
    "/kaggle/input/baseline-weights/epoch_22.pth",
    "/kaggle/input/baseline-weights/epoch_22.pth.tar",
]

weights_path = next((p for p in candidate_weights if os.path.isfile(p)), None)
weights_path



## === cell 2
train_aug_imagenet = T.Compose(
    [
        T.ToPILImage(),
        T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]
)

test_imagenet = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]
)

large_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(
            (cfg["img_size"], cfg["img_size"]),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]
)

use_imagenet_preproc = weights_path is None
train_transform = train_aug_imagenet if use_imagenet_preproc else large_transform
test_transform = test_imagenet if use_imagenet_preproc else large_transform


class TrainSet(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transform):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        path = os.path.join(self.img_dir, image_id)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class InferSet(Dataset):
    def __init__(self, img_dir: str, image_ids, transform):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.img_dir, image_id)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        return x, image_id


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"
image_ids = sample_sub["image_id"].tolist()

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"

train_df_sub = train_df.reset_index(drop=True)

train_set = TrainSet(TRAIN_IMG_DIR, train_df_sub, train_transform)

label_counts = train_df_sub["label"].value_counts().to_dict()
weights_per_class = {
    c: 1.0 / max(label_counts.get(c, 1), 1) for c in range(cfg["num_classes"])
}
sample_weights = train_df_sub["label"].map(weights_per_class).values.astype(np.float64)
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_set,
    batch_size=cfg["batch_size"],
    shuffle=False,
    sampler=sampler,
    num_workers=cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

infer_set = InferSet(TEST_IMG_DIR, image_ids, test_transform)
infer_loader = DataLoader(
    infer_set,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

xb, yb = next(iter(train_loader))
xb.shape, yb.shape



## === cell 3
if weights_path is None:
    backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
else:
    backbone = models.resnet50(weights=None)

in_features = backbone.fc.in_features
backbone.fc = nn.Linear(in_features, cfg["num_classes"])

if weights_path is not None:
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        new_state[kk] = v

    missing, unexpected = backbone.load_state_dict(new_state, strict=False)
    print(f"Loaded weights from: {weights_path}")
    print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
else:
    print(
        "No competition weights found; using ImageNet backbone and training only the 5-class head on train.csv."
    )

model = backbone.to(device)



## === cell 4
if weights_path is None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=cfg["lr"], weight_decay=1e-4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(cfg["epochs"]):
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu()) * xb.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            running_correct += int((preds == yb).sum().detach().cpu())
            running_n += xb.size(0)

        print(
            f"epoch {epoch+1}/{cfg['epochs']} - loss {running_loss/max(running_n,1):.4f} - acc {running_correct/max(running_n,1):.4f}"
        )

model.eval()



## === cell 5
out_path = "./submission.csv"

preds_all = []
ids_all = []

with torch.no_grad():
    model.eval()
    for xb, fns in infer_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        preds_all.extend(preds)
        ids_all.extend(list(fns))

assert len(ids_all) == len(
    sample_sub
), f"Pred length {len(ids_all)} != sample {len(sample_sub)}"
assert (
    ids_all == image_ids
), "Order mismatch vs sample_submission (should not happen with this DataLoader)."

sub = pd.DataFrame({"image_id": ids_all, "label": preds_all})
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {sub.shape}")
sub.head()



## === cell 6
check = pd.read_csv("./submission.csv")
print(check.shape)
print(check.dtypes)
print(check.head())
assert check.shape[0] == sample_sub.shape[0], "Invalid submission: wrong number of rows"
assert list(check.columns) == ["image_id", "label"], "Invalid submission: wrong columns"
assert check["image_id"].isna().sum() == 0, "Invalid submission: missing image_id"
assert check["label"].isna().sum() == 0, "Invalid submission: missing labels"
assert check["label"].between(0, 4).all(), "Invalid submission: labels must be in [0,4]"
