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

No external packages required in the script and installed.

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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import vit_h_14, efficientnet_v2_l
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

torch.cuda.empty_cache()



## === cell 1


def invert_square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    t = torch.roll(t, shifts=(h // 2, w // 2), dims=(1, 2))
    img2 = transforms.functional.to_pil_image(t)

    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = (max_side - w) - pad_left
    pad_bottom = (max_side - h) - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return transforms.functional.pad(img2, padding, padding_mode="reflect")


torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_EfficientNet = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((480, 480)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2

NUM_CLASSES = 5


class Head(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


model_vit = vit_h_14(weights="DEFAULT")
vit_in = model_vit.heads.head.in_features
model_vit.heads.head = Head(vit_in, NUM_CLASSES)
model_vit.to(device).eval()

model_eff = efficientnet_v2_l(weights="DEFAULT")
eff_in = model_eff.classifier[-1].in_features
model_eff.classifier[-1] = nn.Linear(eff_in, NUM_CLASSES)
model_eff.to(device).eval()

print("Backbones ready:", type(model_vit).__name__, type(model_eff).__name__)




## === cell 3
class CassavaTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tfm_vit, tfm_eff):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm_vit = tfm_vit
        self.tfm_eff = tfm_eff

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image_id = self.df.loc[i, "image_id"]
        label = int(self.df.loc[i, "label"])
        path = os.path.join(self.img_dir, image_id)
        img = Image.open(path).convert("RGB")
        x_vit = self.tfm_vit(img)
        x_eff = self.tfm_eff(img)
        return x_vit, x_eff, label


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, img_dir: str, tfm_vit, tfm_eff):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.tfm_vit = tfm_vit
        self.tfm_eff = tfm_eff

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        path = os.path.join(self.img_dir, image_id)
        img = Image.open(path).convert("RGB")
        x_vit = self.tfm_vit(img)
        x_eff = self.tfm_eff(img)
        return x_vit, x_eff, image_id


@torch.no_grad()
def extract_logits(dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module):
    feats = []
    labels = []
    for batch in dataloader:
        if len(batch) == 3:
            x_vit, x_eff, y = batch
        else:
            raise ValueError("Unexpected batch structure")
        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        out_v = model_vit(x_vit)
        out_e = model_eff(x_eff)
        comb = torch.cat([out_e, out_v], dim=1)  # (B, 10)
        feats.append(comb.cpu().numpy().astype(np.float32))
        labels.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def extract_logits_test(
    dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module
):
    feats = []
    ids = []
    for x_vit, x_eff, image_id in dataloader:
        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        out_v = model_vit(x_vit)
        out_e = model_eff(x_eff)
        comb = torch.cat([out_e, out_v], dim=1)
        feats.append(comb.cpu().numpy().astype(np.float32))
        ids.extend(list(image_id))
    return np.concatenate(feats, axis=0), ids




## === cell 4

train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) == {"image_id", "label"}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_feats = np.zeros((len(train_df), 2 * NUM_CLASSES), dtype=np.float32)
oof_labels = train_df["label"].to_numpy(dtype=np.int64)

BATCH = 16 if torch.cuda.is_available() else 8
NUM_WORKERS = 2

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df["image_id"], train_df["label"]), start=1
):
    va_df = train_df.iloc[va_idx].copy()
    va_ds = CassavaTrainDataset(
        va_df, TRAIN_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    feats_va, y_va = extract_logits(va_loader, model_vit, model_eff)
    oof_feats[va_idx] = feats_va
    print(f"Fold {fold}: extracted {feats_va.shape} for val")

decision_tree = RandomForestClassifier(
    n_estimators=30, criterion="gini", max_depth=8, random_state=SEED, n_jobs=-1
)
decision_tree.fit(oof_feats, oof_labels)
print("Meta-model trained on OOF features:", oof_feats.shape)



## === cell 5

sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = CassavaTestDataset(
    test_image_ids, TEST_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_feats, out_ids = extract_logits_test(test_loader, model_vit, model_eff)
assert (
    out_ids == test_image_ids
), "Test ID order mismatch; submission would be misaligned."

prediction = decision_tree.predict(test_feats).astype(int)

submission = pd.DataFrame({"image_id": out_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))
