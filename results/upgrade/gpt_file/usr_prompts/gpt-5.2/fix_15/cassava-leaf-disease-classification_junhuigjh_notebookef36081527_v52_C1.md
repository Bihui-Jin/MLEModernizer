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
from torchvision import transforms, models

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # keep as in original (fast for fixed 224x224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_DIR = f"{DATA_ROOT}/train_images"
TEST_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

use_model1 = False
model1 = None

common_tfms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 1
@torch.inference_mode()
def pil_to_tensor_common(img_pil: Image.Image) -> torch.Tensor:
    return common_tfms(img_pil)


def build_feature_extractor_resnet50() -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
    except Exception as e:
        print(
            f"WARNING: Could not load pretrained ResNet50 weights; using random init. Root error: {repr(e)}"
        )
        backbone = models.resnet50(weights=None)

    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


def build_feature_extractor_efficientnet_b0() -> nn.Module:
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
    except Exception as e:
        print(
            f"WARNING: Could not load pretrained EfficientNet-B0 weights; using random init. Root error: {repr(e)}"
        )
        backbone = models.efficientnet_b0(weights=None)

    backbone.classifier = nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


model2 = build_feature_extractor_resnet50()
model3 = build_feature_extractor_efficientnet_b0()

if torch.cuda.is_available():
    model2 = model2.to(memory_format=torch.channels_last)
    model3 = model3.to(memory_format=torch.channels_last)

try:
    model2 = torch.compile(model2, mode="reduce-overhead")
    model3 = torch.compile(model3, mode="reduce-overhead")
except Exception:
    pass


def get_features_for_image(img_path: str) -> np.ndarray:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        x = pil_to_tensor_common(img).unsqueeze(0).to(device)

    if torch.cuda.is_available():
        x = x.contiguous(memory_format=torch.channels_last)

    emb50 = model2(x).detach().float().cpu().numpy()[0].astype(np.float32)  # (2048,)
    embef = model3(x).detach().float().cpu().numpy()[0].astype(np.float32)  # (1280,)

    if use_model1:
        p1 = np.array(model1.predict(None), dtype=np.float32)  # unreachable
    else:
        p1 = np.zeros((5,), dtype=np.float32)

    return np.concatenate([p1, emb50, embef], axis=0)  # (5 + 2048 + 1280,)




## === cell 2
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.io import decode_image
    import torchvision.transforms.functional as TF

    _HAS_TV_DECODE = True
except Exception:
    _HAS_TV_DECODE = False
    TF = None


_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _center_crop_tensor(img_chw: torch.Tensor, out_hw: int = 224) -> torch.Tensor:
    h = int(img_chw.shape[1])
    w = int(img_chw.shape[2])
    top = (h - out_hw) // 2
    left = (w - out_hw) // 2
    return img_chw[:, top : top + out_hw, left : left + out_hw]


class _CassavaPathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        fp = self.paths[idx]
        if _HAS_TV_DECODE:
            data = torch.frombuffer(open(fp, "rb").read(), dtype=torch.uint8)
            img = decode_image(data, mode="RGB")  # uint8, 3xHxW

            h, w = int(img.shape[1]), int(img.shape[2])
            if h < w:
                new_h = 256
                new_w = int(round(w * (256.0 / h)))
            else:
                new_w = 256
                new_h = int(round(h * (256.0 / w)))

            img = img.to(dtype=torch.float32).div_(255.0)  # ToTensor equivalent
            img = TF.resize(
                img,
                [new_h, new_w],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            img = _center_crop_tensor(img, 224)
            img = (img - _MEAN) / _STD  # Normalize equivalent
            return img
        else:
            with Image.open(fp) as img:
                img = img.convert("RGB")
                x = pil_to_tensor_common(img)
            return x


@torch.inference_mode()
def get_features_for_paths(paths, batch_size: int = 128) -> np.ndarray:
    paths = list(paths)
    n = len(paths)
    out = np.empty((n, 5 + 2048 + 1280), dtype=np.float32)

    cpu_cnt = os.cpu_count() or 2
    num_workers = min(8, max(2, cpu_cnt // 2))
    pin = torch.cuda.is_available()

    loader = DataLoader(
        _CassavaPathDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    write_pos = 0
    for x in loader:
        b = x.shape[0]
        x = x.to(device, non_blocking=pin)

        if torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)

        emb50_t = model2(x).float().cpu()
        embef_t = model3(x).float().cpu()

        sl = slice(write_pos, write_pos + b)
        out[sl, 0:5] = 0.0
        out[sl, 5 : 5 + 2048] = np.asarray(emb50_t, dtype=np.float32)
        out[sl, 5 + 2048 :] = np.asarray(embef_t, dtype=np.float32)

        write_pos += b

    return out


train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)

train_df["filepath"] = (
    train_df["image_id"].astype(str).map(lambda x: f"{TRAIN_DIR}/{x}")
)
fps = train_df["filepath"].to_numpy()

exists_mask = np.fromiter((os.path.exists(p) for p in fps), count=len(fps), dtype=bool)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

train_df = train_df.sort_values("image_id").reset_index(drop=True)

train_labels = train_df["label"].astype(int).to_numpy()
train_paths = train_df["filepath"].tolist()

train_features = get_features_for_paths(train_paths, batch_size=128)

X_tr, X_va, y_tr, y_va = train_test_split(
    train_features,
    train_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=train_labels,
)

base_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
)
path = base_tree.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = np.unique(path.ccp_alphas)

if ccp_alphas.shape[0] > 25:
    idx = np.linspace(0, ccp_alphas.shape[0] - 1, 25).round().astype(int)
    ccp_grid = ccp_alphas[idx]
else:
    ccp_grid = ccp_alphas

best_alpha = float(ccp_grid[0])
best_acc = -1.0
for a in ccp_grid:
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        ccp_alpha=float(a),
    )
    clf.fit(X_tr, y_tr)
    acc = float((clf.predict(X_va) == y_va).mean())
    if acc > best_acc:
        best_acc = acc
        best_alpha = float(a)

print(f"Selected ccp_alpha={best_alpha:.8g} (holdout acc={best_acc:.5f})")

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
    ccp_alpha=best_alpha,
)
decision_tree.fit(train_features, train_labels)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
assert "image_id" in sample_sub.columns

test_image_ids = sample_sub["image_id"].tolist()
test_filepaths = [os.path.join(TEST_DIR, iid) for iid in test_image_ids]

missing = [fp for fp in test_filepaths if not os.path.exists(fp)]
assert len(missing) == 0, f"Missing {len(missing)} test images; example: {missing[0]}"

test_features = get_features_for_paths(test_filepaths, batch_size=128)

prediction = decision_tree.predict(test_features).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
assert submission.shape[0] == len(sample_sub), "Submission row count mismatch"
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch"

submission.to_csv("submission.csv", index=False)
submission.head(10)
