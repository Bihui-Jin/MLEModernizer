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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 2
num_tta = 5  # kept (not used); preserve original semantics



## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset root in candidates: {DATA_ROOT_CANDIDATES}"
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_image_dir = os.path.join(DATA_ROOT, "train_images")
test_image_dir = os.path.join(DATA_ROOT, "test_images")

print("DATA_ROOT:", DATA_ROOT)
print("train_csv_path exists:", os.path.exists(train_csv_path))
print("train_image_dir exists:", os.path.exists(train_image_dir))
print("test_image_dir exists:", os.path.exists(test_image_dir))



## === cell 4
test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 5
efficientnet_transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

efficientnet_train_transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_train_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name


class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, torch.tensor(y, dtype=torch.long)




## === cell 7
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader_eff = DataLoader(
    test_dataset_eff, batch_size=32, shuffle=False, num_workers=0
)

test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)
test_loader_res = DataLoader(
    test_dataset_res, batch_size=32, shuffle=False, num_workers=0
)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 9
def find_checkpoints_any(
    keywords_any=None, keywords_all=None, search_root="/kaggle/input"
):
    if keywords_any is None:
        keywords_any = []
    if keywords_all is None:
        keywords_all = []
    keywords_any = [k.lower() for k in keywords_any]
    keywords_all = [k.lower() for k in keywords_all]

    matches = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if not fn.lower().endswith((".pth", ".pt", ".bin")):
                continue
            fn_l = fn.lower()
            if keywords_all and not all(k in fn_l for k in keywords_all):
                continue
            if keywords_any and not any(k in fn_l for k in keywords_any):
                continue
            matches.append(os.path.join(root, fn))

    def _score(p):
        pl = p.lower()
        bonus = 0
        if "best" in pl:
            bonus -= 20
        if "final" in pl:
            bonus -= 10
        if "checkpoint" in pl or "ckpt" in pl:
            bonus -= 5
        if "fold" in pl:
            bonus -= 2
        if "epoch" in pl:
            bonus -= 1
        depth = p.count(os.sep)
        return (bonus, depth, len(p), p)

    matches = sorted(matches, key=_score)
    return matches


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _clean_state_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("net."):
            k = k[len("net.") :]
        new_state[k] = v
    return new_state


def ckpt_looks_compatible(ckpt_path, arch, device, num_classes=5):
    try:
        state = torch.load(ckpt_path, map_location=device)
        state = _extract_state_dict(state)
        if not isinstance(state, dict):
            return False
        state = _clean_state_keys(state)

        if arch == "resnet50":
            w = state.get("fc.weight", None)
            b = state.get("fc.bias", None)
            if w is None or b is None:
                return False
            if hasattr(w, "shape") and int(w.shape[0]) != num_classes:
                return False
            if hasattr(b, "shape") and int(b.shape[0]) != num_classes:
                return False
            return True

        if arch == "efficientnet_v2_s":
            w = state.get("classifier.1.weight", None)
            b = state.get("classifier.1.bias", None)
            if w is None or b is None:
                return False
            if hasattr(w, "shape") and int(w.shape[0]) != num_classes:
                return False
            if hasattr(b, "shape") and int(b.shape[0]) != num_classes:
                return False
            return True

        return False
    except Exception:
        return False


def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None:
        print("No checkpoint path provided; using randomly initialized weights.")
        return False
    if not os.path.exists(ckpt_path):
        print(
            f"Checkpoint not found: {ckpt_path} -> using randomly initialized weights."
        )
        return False

    state = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(state)

    if not isinstance(state, dict):
        print(f"Checkpoint format not understood for: {ckpt_path} -> skipping.")
        return False

    state = _clean_state_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")
    head_missing = any(
        k in missing
        for k in [
            "fc.weight",
            "fc.bias",
            "classifier.1.weight",
            "classifier.1.bias",
        ]
    )
    if head_missing:
        print(
            "  WARNING: classifier head weights were missing -> treating as NOT loaded."
        )
        return False
    return True


def pick_first_compatible(candidates, arch, device, exclude=None):
    exclude = set(exclude or [])
    for p in candidates:
        if p in exclude:
            continue
        if ckpt_looks_compatible(p, arch=arch, device=device, num_classes=5):
            return p
    return None


CHECKPOINT_SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/working",
    "/kaggle/data",
]


def find_ckpts_multi_root(keywords_any=None, keywords_all=None):
    out = []
    for r in CHECKPOINT_SEARCH_ROOTS:
        if os.path.exists(r):
            out.extend(
                find_checkpoints_any(
                    keywords_any=keywords_any, keywords_all=keywords_all, search_root=r
                )
            )
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq




## === cell 10
from sklearn.model_selection import StratifiedShuffleSplit

train_df_full = pd.read_csv(train_csv_path)
labels = train_df_full["label"].astype(int).values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df_full, labels))
train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(train_df), "Val size:", len(val_df))
print(
    "Train label dist:",
    train_df["label"].value_counts(normalize=True).round(3).to_dict(),
)
print(
    "Val label dist:", val_df["label"].value_counts(normalize=True).round(3).to_dict()
)


def train_one_model(
    model,
    arch_name,
    train_loader,
    val_loader,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path="best.pth",
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        tr_loss /= max(1, tr_n)
        val_loss /= max(1, total)
        val_acc = correct / max(1, total)
        print(
            f"[{arch_name}] epoch {ep}/{epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        torch.save(best_state, out_path)
        print(
            f"[{arch_name}] saved best checkpoint to: {out_path}  best_val_acc={best_acc:.4f}"
        )

    return best_acc




## === cell 11
train_ds_eff = CassavaTrainDataset(
    train_df, train_image_dir, transform=efficientnet_train_transforms
)
val_ds_eff = CassavaTrainDataset(
    val_df, train_image_dir, transform=efficientnet_transforms
)

train_dl_eff = DataLoader(
    train_ds_eff,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
val_dl_eff = DataLoader(
    val_ds_eff,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

train_ds_res = CassavaTrainDataset(
    train_df, train_image_dir, transform=resnet_train_transforms
)
val_ds_res = CassavaTrainDataset(val_df, train_image_dir, transform=resnet_transforms)

train_dl_res = DataLoader(
    train_ds_res,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
val_dl_res = DataLoader(
    val_ds_res,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)



## === cell 12
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

resnet_ckpt_local = "/kaggle/working/resnet50_best.pth"
eff1_ckpt_local = "/kaggle/working/effv2s_1_best.pth"
eff7_ckpt_local = "/kaggle/working/effv2s_7_best.pth"
eff8_ckpt_local = "/kaggle/working/effv2s_8_best.pth"

_ = train_one_model(
    resnet_model,
    "resnet50",
    train_dl_res,
    val_dl_res,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=resnet_ckpt_local,
)

_ = train_one_model(
    efficientnet_model_1,
    "effv2s_1",
    train_dl_eff,
    val_dl_eff,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=eff1_ckpt_local,
)

torch.manual_seed(SEED + 7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + 7)
_ = train_one_model(
    efficientnet_model_7,
    "effv2s_7",
    train_dl_eff,
    val_dl_eff,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=eff7_ckpt_local,
)

torch.manual_seed(SEED + 8)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + 8)
_ = train_one_model(
    efficientnet_model_8,
    "effv2s_8",
    train_dl_eff,
    val_dl_eff,
    device,
    epochs=1,
    lr=3e-4,
    weight_decay=1e-4,
    out_path=eff8_ckpt_local,
)



## === cell 13
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_candidates = [resnet_ckpt_local]
resnet_candidates += find_ckpts_multi_root(
    keywords_any=["resnet", "resnet50"], keywords_all=["cassava"]
)
resnet_candidates += find_ckpts_multi_root(
    keywords_any=["resnet", "resnet50", "cassava"], keywords_all=[]
)
resnet_candidates += find_ckpts_multi_root(
    keywords_any=["resnet", "resnet50"], keywords_all=[]
)

resnet_ckpt = pick_first_compatible(resnet_candidates, arch="resnet50", device=device)
if resnet_ckpt is None:
    print("No compatible ResNet50 (5-class) checkpoint found; ResNet will be random.")
loaded_resnet = safe_load_state_dict(resnet_model, resnet_ckpt, device)
resnet_model = resnet_model.to(device)
resnet_model.eval()

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff1_candidates = [eff1_ckpt_local]
eff1_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "efficientnetv2", "v2s", "eff"],
    keywords_all=["cassava"],
)
eff1_candidates += find_ckpts_multi_root(
    keywords_any=["eff", "fold", "best", "cassava"], keywords_all=[]
)
eff1_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "eff"], keywords_all=[]
)

eff1_ckpt = pick_first_compatible(
    eff1_candidates, arch="efficientnet_v2_s", device=device
)
if eff1_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_1; it will be random."
    )
loaded_eff1 = safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()

efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_candidates = [eff7_ckpt_local]
eff7_candidates += find_ckpts_multi_root(
    keywords_any=["eff", "fold", "best", "checkpoint", "ckpt", "cassava"],
    keywords_all=[],
)
eff7_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "efficientnetv2", "v2s", "eff"],
    keywords_all=["cassava"],
)
eff7_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "eff"], keywords_all=[]
)

eff7_ckpt = pick_first_compatible(
    eff7_candidates, arch="efficientnet_v2_s", device=device, exclude=[eff1_ckpt]
)
if eff7_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_7; it will be random."
    )
loaded_eff7 = safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()

efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff8_candidates = [eff8_ckpt_local]
eff8_candidates += find_ckpts_multi_root(
    keywords_any=["eff", "fold", "best", "checkpoint", "ckpt", "cassava"],
    keywords_all=[],
)
eff8_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "efficientnetv2", "v2s", "eff"],
    keywords_all=["cassava"],
)
eff8_candidates += find_ckpts_multi_root(
    keywords_any=["efficientnet", "effnet", "eff"], keywords_all=[]
)

eff8_ckpt = pick_first_compatible(
    eff8_candidates,
    arch="efficientnet_v2_s",
    device=device,
    exclude=[eff1_ckpt, eff7_ckpt],
)
if eff8_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_8; it will be random."
    )
loaded_eff8 = safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()

weight_efficientnet = 0.7
weight_resnet = 0.3

print("Using checkpoints:")
print("  ResNet50:", resnet_ckpt, "loaded:", loaded_resnet)
print("  EfficientNetV2-S #1:", eff1_ckpt, "loaded:", loaded_eff1)
print("  EfficientNetV2-S #7:", eff7_ckpt, "loaded:", loaded_eff7)
print("  EfficientNetV2-S #8:", eff8_ckpt, "loaded:", loaded_eff8)

probs_eff_total = []
image_names_eff = []

with torch.no_grad():
    for images, img_names in test_loader_eff:
        images = images.to(device)

        outputs_efficientnet1 = efficientnet_model_1(images)
        probs_efficientnet1 = F.softmax(outputs_efficientnet1, dim=1)

        outputs_efficientnet7 = efficientnet_model_7(images)
        probs_efficientnet7 = F.softmax(outputs_efficientnet7, dim=1)

        outputs_efficientnet8 = efficientnet_model_8(images)
        probs_efficientnet8 = F.softmax(outputs_efficientnet8, dim=1)

        combined_probs_eff = (
            probs_efficientnet1 + probs_efficientnet7 + probs_efficientnet8
        ) / 3.0

        probs_eff_total.append(combined_probs_eff.detach().cpu())
        image_names_eff.extend(list(img_names))

probs_eff_total = torch.cat(probs_eff_total, dim=0).numpy()
pred_map_eff = {k: v for k, v in zip(image_names_eff, probs_eff_total)}
probs_eff_ordered = np.stack(
    [pred_map_eff[iid] for iid in test_df["image_id"].tolist()], axis=0
)

probs_res_total = []
image_names_res = []

with torch.no_grad():
    for images, img_names in test_loader_res:
        images = images.to(device)
        outputs_resnet = resnet_model(images)
        probs_resnet = F.softmax(outputs_resnet, dim=1)
        probs_res_total.append(probs_resnet.detach().cpu())
        image_names_res.extend(list(img_names))

probs_res_total = torch.cat(probs_res_total, dim=0).numpy()
pred_map_res = {k: v for k, v in zip(image_names_res, probs_res_total)}
probs_res_ordered = np.stack(
    [pred_map_res[iid] for iid in test_df["image_id"].tolist()], axis=0
)

probs_ens = weight_efficientnet * probs_eff_ordered + weight_resnet * probs_res_ordered
ordered_labels = probs_ens.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_labels}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(test_df))
print("Unique predicted labels:", submission_df["label"].value_counts().to_dict())
