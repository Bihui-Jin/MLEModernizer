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
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

SEED = 42
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

NUM_CLASSES = 5

_resnet_weights = ResNet18_Weights.IMAGENET1K_V1
_vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
torch_transforms_512 = _resnet_weights.transforms()
torch_transforms_vit = _vit_weights.transforms()

torch_transforms_512_train_aug = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02
        ),
        torch_transforms_512,
    ]
)

torch_transforms_vit_train_aug = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02
        ),
        torch_transforms_vit,
    ]
)




## === cell 1
def build_model1():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval()
    return m


def build_model2():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    return m


def build_model3():
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.eval()
    return m


model1 = build_model1().to(device).eval()
model2 = build_model2().to(device).eval()
model3 = build_model3().to(device).eval()


@torch.no_grad()
def resnet18_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    x = m.avgpool(x)
    x = torch.flatten(x, 1)
    return x


@torch.no_grad()
def vit_b16_embed(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(m, "forward_features"):
        feats = m.forward_features(x)
        if isinstance(feats, torch.Tensor):
            if feats.ndim == 3:
                return feats[:, 0]
            return feats

    x = m._process_input(x)  # [B, num_patches, hidden_dim]
    n = x.shape[0]
    batch_class_token = m.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = m.encoder(x)

    if hasattr(m, "encoder") and hasattr(m.encoder, "ln"):
        x = m.encoder.ln(x)
    elif hasattr(m, "ln"):
        x = m.ln(x)

    return x[:, 0]




## === cell 2
class CassavaImageDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        path = os.path.join(self.img_dir, image_id)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)

        y = row["label"] if "label" in self.df.columns else -1
        return image_id, x, int(y)


@torch.no_grad()
def extract_features(df, img_dir, batch_size=32, train_mode=False):
    t512 = torch_transforms_512
    tvit = torch_transforms_vit

    ds_512 = CassavaImageDataset(df, img_dir, t512)
    ds_vit = CassavaImageDataset(df, img_dir, tvit)

    loader_512 = DataLoader(
        ds_512,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    loader_vit = DataLoader(
        ds_vit,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    feats = []
    image_ids = []

    for (ids1, x512, _), (ids3, xvit, _) in zip(loader_512, loader_vit):
        if list(ids1) != list(ids3):
            raise RuntimeError(
                "Dataset ordering mismatch between 512 and VIT transforms."
            )
        image_ids.extend(list(ids1))

        x512 = x512.to(device, non_blocking=True)
        xvit = xvit.to(device, non_blocking=True)

        p1 = model1(x512).detach().cpu().numpy().astype(np.float32)  # [B, 1000]
        p2 = (
            resnet18_embed(model2, x512).detach().cpu().numpy().astype(np.float32)
        )  # [B, 512]
        p3 = (
            vit_b16_embed(model3, xvit).detach().cpu().numpy().astype(np.float32)
        )  # [B, 768]

        feats.append(np.concatenate([p1, p2, p3], axis=1))

    out_dim = 1000 + 512 + 768
    feats = np.vstack(feats) if len(feats) else np.zeros((0, out_dim), dtype=np.float32)
    return image_ids, feats




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)

train_ids, train_feats = extract_features(
    train_df[["image_id", "label"]], TRAIN_IMG_DIR, batch_size=32, train_mode=True
)
train_labels = train_df["label"].to_numpy()

if len(train_feats) != len(train_labels):
    raise RuntimeError(
        f"Feature/label length mismatch: feats={len(train_feats)} labels={len(train_labels)}"
    )


def _acc(y_true, y_pred):
    return float((y_true == y_pred).mean())


depth_grid = [5, 7, 9, 11, 13, 15, 17]
min_leaf_grid = [1, 2, 4, 8]
min_split_grid = [4, 9, 16]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

best_params = None
best_cv_acc = -1.0
eps = 1e-6

for depth in depth_grid:
    for min_leaf in min_leaf_grid:
        for min_split in min_split_grid:
            if min_split < 2 * min_leaf:
                continue

            fold_accs = []
            for tr_idx, va_idx in skf.split(train_feats, train_labels):
                X_tr, y_tr = train_feats[tr_idx], train_labels[tr_idx]
                X_va, y_va = train_feats[va_idx], train_labels[va_idx]

                scaler = StandardScaler(with_mean=True, with_std=True)
                X_tr_s = scaler.fit_transform(X_tr).astype(np.float32, copy=False)
                X_va_s = scaler.transform(X_va).astype(np.float32, copy=False)

                counts_tr = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
                class_w_tr = (
                    len(y_tr) / (NUM_CLASSES * np.maximum(counts_tr, 1.0))
                ).astype(np.float64)
                sample_weight_tr = class_w_tr[y_tr].astype(np.float64)

                dt = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=depth,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    random_state=SEED,
                    class_weight=None,
                )
                dt.fit(X_tr_s, y_tr, sample_weight=sample_weight_tr)
                pred_va = dt.predict(X_va_s).astype(int)
                fold_accs.append(_acc(y_va, pred_va))

            cv_acc = float(np.mean(fold_accs))

            if best_params is None:
                take = True
            else:
                prev_depth, prev_leaf, prev_split = best_params
                take = (cv_acc > best_cv_acc + eps) or (
                    abs(cv_acc - best_cv_acc) <= eps
                    and (
                        depth < prev_depth
                        or (depth == prev_depth and min_leaf > prev_leaf)
                        or (
                            depth == prev_depth
                            and min_leaf == prev_leaf
                            and min_split > prev_split
                        )
                    )
                )

            if take:
                best_cv_acc = cv_acc
                best_params = (depth, min_leaf, min_split)

best_depth, best_min_leaf, best_min_split = best_params
print(
    f"CV selection: max_depth={best_depth}, min_samples_leaf={best_min_leaf}, "
    f"min_samples_split={best_min_split} with cv_acc={best_cv_acc:.4f}"
)

scaler = StandardScaler(with_mean=True, with_std=True)
train_feats_scaled = scaler.fit_transform(train_feats).astype(np.float32, copy=False)

counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
class_w = (len(train_labels) / (NUM_CLASSES * np.maximum(counts, 1.0))).astype(
    np.float64
)
sample_weight = class_w[train_labels].astype(np.float64)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_depth,
    min_samples_split=best_min_split,
    min_samples_leaf=best_min_leaf,
    random_state=SEED,
    class_weight=None,
)
decision_tree.fit(train_feats_scaled, train_labels, sample_weight=sample_weight)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["image_id"]].copy()
test_df["label"] = -1  # placeholder for dataset API

test_ids, test_feats = extract_features(
    test_df[["image_id", "label"]], TEST_IMG_DIR, batch_size=32, train_mode=False
)

id_to_row = {img_id: i for i, img_id in enumerate(test_ids)}
order_idx = []
missing = []
for img_id in sample_sub["image_id"].tolist():
    if img_id in id_to_row:
        order_idx.append(id_to_row[img_id])
    else:
        missing.append(img_id)

if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images on disk. First few: {missing[:5]}"
    )

test_feats_ordered = test_feats[np.array(order_idx)]
test_feats_scaled = scaler.transform(test_feats_ordered).astype(np.float32, copy=False)

prediction = decision_tree.predict(test_feats_scaled).astype(int)

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": prediction}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} to {os.path.abspath('submission.csv')}"
)
