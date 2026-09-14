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
import math
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(42)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

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
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return min(4, max(2, cpu // 2))


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


def _list_tfrecords(dir_path: str):
    if not os.path.isdir(dir_path):
        return []
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tfrec")
    ]
    return sorted(files)


TRAIN_TFRECS = _list_tfrecords(TRAIN_TFREC_DIR)
TEST_TFRECS = _list_tfrecords(TEST_TFREC_DIR)


def _can_use_tfrecords():
    return False


def _maybe_compile(model: nn.Module) -> nn.Module:
    return model


CACHE_ROOT = "/kaggle/working/cassava_tensor_cache_v1"


def _cache_path(image_id: str, variant: str) -> str:
    safe = image_id.replace("/", "_")
    return os.path.join(CACHE_ROOT, f"{variant}__{safe}.pt")


def _ensure_cache_dir():
    os.makedirs(CACHE_ROOT, exist_ok=True)




## === cell 1
class CassavaIndexDataset(Dataset):
    def __init__(
        self,
        image_ids,
        img_dir: str,
        transform,
        indices,
        labels=None,
        cache=None,
        cache_variant: str | None = None,
        enable_disk_cache: bool = False,
    ):
        self.image_ids = np.asarray(image_ids)
        self.img_dir = img_dir
        self.transform = transform
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels)
        self.cache = cache  # dict: image_id -> torch.FloatTensor (C,H,W), CPU
        self.cache_variant = cache_variant
        self.enable_disk_cache = enable_disk_cache
        if self.enable_disk_cache:
            _ensure_cache_dir()

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, j: int):
        idx = int(self.indices[j])
        image_id = self.image_ids[idx]

        x = None
        if self.cache is not None:
            x = self.cache.get(image_id, None)

        if x is None and self.enable_disk_cache and (self.cache_variant is not None):
            p = _cache_path(image_id, self.cache_variant)
            if os.path.exists(p):
                x = torch.load(p, map_location="cpu")

        if x is None:
            img_path = os.path.join(self.img_dir, image_id)
            with Image.open(img_path) as im:
                img = im.convert("RGB")
            x = self.transform(img)
            x = x.contiguous()
            if self.cache is not None:
                self.cache[image_id] = x
            if self.enable_disk_cache and (self.cache_variant is not None):
                p = _cache_path(image_id, self.cache_variant)
                try:
                    torch.save(x, p)
                except Exception:
                    pass

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


def _to_channels_last(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available():
        return model.to(memory_format=torch.channels_last)
    return model


def train_one_model(model, train_loader, epochs=2, lr=2e-3):
    model = model.to(DEVICE)
    model = _to_channels_last(model)
    model = _maybe_compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def predict_proba(model, loader):
    model = model.to(DEVICE)
    model = _to_channels_last(model)
    model.eval()
    all_probs = []
    for batch in loader:
        xb = batch[0] if isinstance(batch, (list, tuple)) else batch
        xb = xb.to(DEVICE, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
assert set(["image_id", "label"]).issubset(train_df.columns)
train_df["label"] = train_df["label"].astype(int)

train_image_ids = train_df["image_id"].to_numpy()
train_labels_all = train_df["label"].to_numpy()

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_features = np.zeros((len(train_df), N_CLASSES * 3), dtype=np.float32)
oof_labels = train_labels_all

fold_models = []  # list of (m1, m2, m3) trained on each fold's train split

cache_512_train = None
cache_518_train = None

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_image_ids, train_labels_all)):
    ds_tr_512 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms,
        tr_idx,
        labels=train_labels_all,
        cache=cache_512_train,
        cache_variant="512",
        enable_disk_cache=True,
    )
    ds_va_512 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms,
        va_idx,
        labels=train_labels_all,
        cache=cache_512_train,
        cache_variant="512",
        enable_disk_cache=True,
    )

    tr_loader_512 = DataLoader(
        ds_tr_512,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
        generator=_DL_GENERATOR,
    )
    va_loader_512 = DataLoader(
        ds_va_512,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    )

    ds_tr_518 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms_VIT,
        tr_idx,
        labels=train_labels_all,
        cache=cache_518_train,
        cache_variant="518",
        enable_disk_cache=True,
    )
    ds_va_518 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms_VIT,
        va_idx,
        labels=train_labels_all,
        cache=cache_518_train,
        cache_variant="518",
        enable_disk_cache=True,
    )

    tr_loader_518 = DataLoader(
        ds_tr_518,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
        generator=_DL_GENERATOR,
    )
    va_loader_518 = DataLoader(
        ds_va_518,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    )

    model1 = TinyCNN(width=32, num_classes=N_CLASSES)
    model2 = TinyCNN(width=40, num_classes=N_CLASSES)
    model3 = TinyCNN(width=48, num_classes=N_CLASSES)

    model1 = train_one_model(model1, tr_loader_512, epochs=2, lr=2e-3)
    model2 = train_one_model(model2, tr_loader_512, epochs=2, lr=2e-3)
    model3 = train_one_model(model3, tr_loader_518, epochs=2, lr=2e-3)

    p1 = predict_proba(model1, va_loader_512)
    p2 = predict_proba(model2, va_loader_512)
    p3 = predict_proba(model3, va_loader_518)

    feats = np.concatenate([p1, p2, p3], axis=1)
    oof_features[va_idx] = feats.astype(np.float32)

    fold_models.append((model1, model2, model3))

    del (
        ds_tr_512,
        ds_va_512,
        ds_tr_518,
        ds_va_518,
        tr_loader_512,
        va_loader_512,
        tr_loader_518,
        va_loader_518,
    )
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
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].to_numpy()
test_idx = np.arange(len(test_ids), dtype=np.int64)

cache_512_test = None
cache_518_test = None

test_ds_512 = CassavaIndexDataset(
    test_ids,
    TEST_IMG_DIR,
    torch_transforms,
    test_idx,
    labels=None,
    cache=cache_512_test,
    cache_variant="512",
    enable_disk_cache=True,
)
test_loader_512 = DataLoader(
    test_ds_512,
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
)

test_ds_518 = CassavaIndexDataset(
    test_ids,
    TEST_IMG_DIR,
    torch_transforms_VIT,
    test_idx,
    labels=None,
    cache=cache_518_test,
    cache_variant="518",
    enable_disk_cache=True,
)
test_loader_518 = DataLoader(
    test_ds_518,
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
)

if len(fold_models) == 0:
    prediction = np.zeros(len(test_ids), dtype=int)
else:
    p1_sum = None
    p2_sum = None
    p3_sum = None
    for m1, m2, m3 in fold_models:
        p1 = predict_proba(m1, test_loader_512)
        p2 = predict_proba(m2, test_loader_512)
        p3 = predict_proba(m3, test_loader_518)
        if p1_sum is None:
            p1_sum = p1
            p2_sum = p2
            p3_sum = p3
        else:
            p1_sum += p1
            p2_sum += p2
            p3_sum += p3

    k = float(len(fold_models))
    p1_test = p1_sum / k
    p2_test = p2_sum / k
    p3_test = p3_sum / k

    combined_output = np.concatenate([p1_test, p2_test, p3_test], axis=1)
    prediction = decision_tree.predict(combined_output).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
submission.head()
