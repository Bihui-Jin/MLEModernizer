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

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

N_CLASSES = 5

torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

torch_transforms_VIT = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

_HAS_CUDA = torch.cuda.is_available()
PIN_MEMORY = _HAS_CUDA


def _default_num_workers():
    cpu = os.cpu_count() or 2
    return 2 if not _HAS_CUDA else min(4, max(2, cpu // 2))


NUM_WORKERS = _default_num_workers()
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

_DL_GENERATOR = torch.Generator()
_DL_GENERATOR.manual_seed(42)


def _seed_worker(worker_id: int):
    base_seed = 42
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


_TENSOR_CACHE = {}  # (img_dir, image_id, transform_key) -> torch.Tensor


def _transform_key(transform) -> str:
    if transform is torch_transforms:
        return "512"
    if transform is torch_transforms_VIT:
        return "518"
    return str(id(transform))




## === cell 1
class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self._tkey = _transform_key(transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        tcache_key = (self.img_dir, image_id, self._tkey)
        x = _TENSOR_CACHE.get(tcache_key, None)
        if x is None:
            img_path = os.path.join(self.img_dir, image_id)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                x = self.transform(img)
            _TENSOR_CACHE[tcache_key] = x

        if "label" in self.df.columns:
            y = int(row["label"])
            return x, y
        return x


class CassavaIndexDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform, indices, labels=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self._tkey = _transform_key(transform)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, j: int):
        idx = int(self.indices[j])
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        tcache_key = (self.img_dir, image_id, self._tkey)
        x = _TENSOR_CACHE.get(tcache_key, None)
        if x is None:
            img_path = os.path.join(self.img_dir, image_id)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                x = self.transform(img)
            _TENSOR_CACHE[tcache_key] = x

        if self.labels is None:
            return x
        return x, int(self.labels[idx])


class TinyCNN(nn.Module):
    """
    Minimal PyTorch replacement for the missing external pretrained models.
    Outputs logits for 5 classes.
    """

    def __init__(self, in_ch=3, num_classes=5, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def train_one_model(model, train_loader, val_loader, epochs=2, lr=2e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def predict_proba(model, loader):
    model.eval()
    all_probs = []
    for batch in loader:
        xb = batch[0]
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
assert set(["image_id", "label"]).issubset(train_df.columns)
train_df["label"] = train_df["label"].astype(int)

train_labels_all = train_df["label"].to_numpy()

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_features = np.zeros((len(train_df), N_CLASSES * 3), dtype=np.float32)
oof_labels = train_labels_all

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df["image_id"], train_df["label"])
):
    ds_tr_512 = CassavaIndexDataset(
        train_df, TRAIN_IMG_DIR, torch_transforms, tr_idx, labels=train_labels_all
    )
    ds_va_512 = CassavaIndexDataset(
        train_df, TRAIN_IMG_DIR, torch_transforms, va_idx, labels=train_labels_all
    )

    tr_loader_512 = DataLoader(
        ds_tr_512,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker,
        generator=_DL_GENERATOR,
    )
    va_loader_512 = DataLoader(
        ds_va_512,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker,
    )

    ds_tr_518 = CassavaIndexDataset(
        train_df, TRAIN_IMG_DIR, torch_transforms_VIT, tr_idx, labels=train_labels_all
    )
    ds_va_518 = CassavaIndexDataset(
        train_df, TRAIN_IMG_DIR, torch_transforms_VIT, va_idx, labels=train_labels_all
    )

    tr_loader_518 = DataLoader(
        ds_tr_518,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker,
        generator=_DL_GENERATOR,
    )
    va_loader_518 = DataLoader(
        ds_va_518,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker,
    )

    model1 = TinyCNN(width=32, num_classes=N_CLASSES)
    model2 = TinyCNN(width=40, num_classes=N_CLASSES)
    model3 = TinyCNN(width=48, num_classes=N_CLASSES)

    model1 = train_one_model(model1, tr_loader_512, va_loader_512, epochs=2, lr=2e-3)
    model2 = train_one_model(model2, tr_loader_512, va_loader_512, epochs=2, lr=2e-3)
    model3 = train_one_model(model3, tr_loader_518, va_loader_518, epochs=2, lr=2e-3)

    p1 = predict_proba(model1, va_loader_512)
    p2 = predict_proba(model2, va_loader_512)
    p3 = predict_proba(model3, va_loader_518)

    feats = np.concatenate([p1, p2, p3], axis=1)
    oof_features[va_idx] = feats.astype(np.float32)

    del model1, model2, model3, ds_tr_512, ds_va_512, ds_tr_518, ds_va_518
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=8,
    min_samples_split=9,
    random_state=42,
)
decision_tree.fit(oof_features, oof_labels)




## === cell 3
full_idx = np.arange(len(train_df), dtype=np.int64)

full_ds_512 = CassavaIndexDataset(
    train_df, TRAIN_IMG_DIR, torch_transforms, full_idx, labels=train_labels_all
)
full_loader_512 = DataLoader(
    full_ds_512,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
    generator=_DL_GENERATOR,
)

full_ds_518 = CassavaIndexDataset(
    train_df, TRAIN_IMG_DIR, torch_transforms_VIT, full_idx, labels=train_labels_all
)
full_loader_518 = DataLoader(
    full_ds_518,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
    generator=_DL_GENERATOR,
)

final_model1 = TinyCNN(width=32, num_classes=N_CLASSES)
final_model2 = TinyCNN(width=40, num_classes=N_CLASSES)
final_model3 = TinyCNN(width=48, num_classes=N_CLASSES)

final_model1 = train_one_model(final_model1, full_loader_512, None, epochs=2, lr=2e-3)
final_model2 = train_one_model(final_model2, full_loader_512, None, epochs=2, lr=2e-3)
final_model3 = train_one_model(final_model3, full_loader_518, None, epochs=2, lr=2e-3)




## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()
test_df = pd.DataFrame({"image_id": test_ids})
test_idx = np.arange(len(test_df), dtype=np.int64)

test_ds_512 = CassavaIndexDataset(
    test_df, TEST_IMG_DIR, torch_transforms, test_idx, labels=None
)
test_loader_512 = DataLoader(
    test_ds_512,
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
)

test_ds_518 = CassavaIndexDataset(
    test_df, TEST_IMG_DIR, torch_transforms_VIT, test_idx, labels=None
)
test_loader_518 = DataLoader(
    test_ds_518,
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker,
)

p1_test = predict_proba(final_model1, test_loader_512)
p2_test = predict_proba(final_model2, test_loader_512)
p3_test = predict_proba(final_model3, test_loader_518)

combined_output = np.concatenate([p1_test, p2_test, p3_test], axis=1)
prediction = decision_tree.predict(combined_output).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
submission.head()
