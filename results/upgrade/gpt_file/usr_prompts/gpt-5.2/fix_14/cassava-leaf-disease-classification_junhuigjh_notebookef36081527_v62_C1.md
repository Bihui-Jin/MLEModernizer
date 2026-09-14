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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet50
from torchvision.io import read_image, ImageReadMode

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

NUM_CLASSES = train_df["label"].nunique()
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"

try:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except Exception:
    pass


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


class _TensorCache:
    def __init__(self):
        self._d = {}

    def get(self, key):
        return self._d.get(key, None)

    def set(self, key, value):
        self._d[key] = value


_CACHE_512_TRAIN = _TensorCache()
_CACHE_518_TRAIN = _TensorCache()
_CACHE_512_TEST = _TensorCache()
_CACHE_518_TEST = _TensorCache()
_CACHE_RGB_TRAIN = _TensorCache()
_CACHE_RGB_TEST = _TensorCache()

try:
    import torchvision

    if hasattr(torchvision, "set_image_backend"):
        torchvision.set_image_backend("torchvision")
except Exception:
    pass


def _read_rgb_tensor(path: str) -> torch.Tensor:
    return read_image(path, mode=ImageReadMode.RGB)


torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

torch_transforms_VIT = transforms.Compose(
    [
        transforms.Resize((518, 518), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class CassavaDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        transform,
        train_mode=True,
        cache: _TensorCache | None = None,
        rgb_cache: _TensorCache | None = None,
    ):
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].astype(int).tolist() if train_mode else None
        self.image_dir = image_dir
        self.transform = transform
        self.train_mode = train_mode
        self.cache = cache
        self.rgb_cache = rgb_cache

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)

        x = None
        if self.cache is not None:
            x = self.cache.get(image_id)
        if x is None:
            img = None
            if self.rgb_cache is not None:
                img = self.rgb_cache.get(image_id)

            if img is None:
                img = _read_rgb_tensor(img_path)  # uint8 RGB
                if self.rgb_cache is not None:
                    self.rgb_cache.set(image_id, img)

            x = self.transform(img)
            if self.cache is not None:
                self.cache.set(image_id, x)

        if self.train_mode:
            y = int(self.labels[idx])
            return x, y
        return x, image_id


class SmallCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def make_model2_resnet50(num_classes=5):
    m = resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class MediumCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


model1 = SmallCNN(NUM_CLASSES).to(device)
model2 = make_model2_resnet50(NUM_CLASSES).to(device)
model3 = MediumCNN(NUM_CLASSES).to(device)

if device.type == "cuda":
    model1 = model1.to(memory_format=torch.channels_last)
    model2 = model2.to(memory_format=torch.channels_last)
    model3 = model3.to(memory_format=torch.channels_last)


def _maybe_compile(m: nn.Module) -> nn.Module:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return m
    try:
        return compile_fn(m, mode="reduce-overhead", fullgraph=False)
    except Exception:
        return m


model1 = _maybe_compile(model1)
model2 = _maybe_compile(model2)
model3 = _maybe_compile(model3)



## === cell 1
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

if device.type == "cuda":
    BATCH_SIZE = 64
else:
    BATCH_SIZE = 32

EPOCHS = 2  # unchanged

CPU_COUNT = os.cpu_count() or 1

if device.type == "cuda":
    NUM_WORKERS = min(4, CPU_COUNT)
    PIN_MEMORY = True
    PERSISTENT = True if NUM_WORKERS > 0 else False
    PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None
else:
    NUM_WORKERS = min(2, CPU_COUNT)
    PIN_MEMORY = False
    PERSISTENT = True if NUM_WORKERS > 0 else False
    PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None


tr_ds_512 = CassavaDataset(
    tr_df,
    TRAIN_IMG_DIR,
    torch_transforms,
    train_mode=True,
    cache=_CACHE_512_TRAIN,
    rgb_cache=_CACHE_RGB_TRAIN,
)
va_ds_512 = CassavaDataset(
    va_df,
    TRAIN_IMG_DIR,
    torch_transforms,
    train_mode=True,
    cache=_CACHE_512_TRAIN,
    rgb_cache=_CACHE_RGB_TRAIN,
)
tr_ds_518 = CassavaDataset(
    tr_df,
    TRAIN_IMG_DIR,
    torch_transforms_VIT,
    train_mode=True,
    cache=_CACHE_518_TRAIN,
    rgb_cache=_CACHE_RGB_TRAIN,
)
va_ds_518 = CassavaDataset(
    va_df,
    TRAIN_IMG_DIR,
    torch_transforms_VIT,
    train_mode=True,
    cache=_CACHE_518_TRAIN,
    rgb_cache=_CACHE_RGB_TRAIN,
)

tr_loader_512 = DataLoader(
    tr_ds_512,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)
va_loader_512 = DataLoader(
    va_ds_512,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)

tr_loader_518 = DataLoader(
    tr_ds_518,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)
va_loader_518 = DataLoader(
    va_ds_518,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)


def train_one_model(
    model, tr_loader, va_loader, epochs=2, lr=1e-3, collect_val_logits: bool = False
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    collected = None  # (logits_np, labels_np) for va set, in loader order
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0

        want_collect = collect_val_logits and (ep == epochs - 1)
        if want_collect:
            num_samples = len(va_loader.dataset)
            num_classes = NUM_CLASSES
            out = np.empty((num_samples, num_classes), dtype=np.float32)
            ys = np.empty((num_samples,), dtype=np.int64)
            i = 0

        with torch.no_grad():
            for xb, yb in va_loader:
                bs = xb.shape[0]
                xb = xb.to(device, non_blocking=True)
                if device.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                yb_dev = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb_dev).sum().item()
                total += yb_dev.size(0)

                if want_collect:
                    logits_cpu = logits.detach().to("cpu", non_blocking=True)
                    out[i : i + bs] = logits_cpu.numpy()
                    ys[i : i + bs] = yb.numpy()
                    i += bs

        print(f"epoch {ep+1}/{epochs} val_acc={correct/total:.4f}")
        if want_collect:
            collected = (out, ys)

    return model, collected


model1, val_pack_1 = train_one_model(
    model1,
    tr_loader_512,
    va_loader_512,
    epochs=EPOCHS,
    lr=1e-3,
    collect_val_logits=True,
)
model2, val_pack_2 = train_one_model(
    model2,
    tr_loader_512,
    va_loader_512,
    epochs=EPOCHS,
    lr=1e-4,
    collect_val_logits=True,
)  # unchanged
model3, val_pack_3 = train_one_model(
    model3,
    tr_loader_518,
    va_loader_518,
    epochs=EPOCHS,
    lr=1e-3,
    collect_val_logits=True,
)

assert val_pack_1 is not None and val_pack_2 is not None and val_pack_3 is not None



## === cell 2
log1, y = val_pack_1
log2, y2 = val_pack_2
log3, y3 = val_pack_3
assert np.array_equal(y, y2) and np.array_equal(
    y, y3
), "Label mismatch across collected val packs"

train_features = np.concatenate([log1, log2, log3], axis=1)

decision_tree = DecisionTreeClassifier(
    criterion="gini", max_depth=7, min_samples_split=9, random_state=SEED
)
decision_tree.fit(train_features, y)



## === cell 3
test_image_ids = sample_df["image_id"].tolist()


class CassavaTestDataset(Dataset):
    def __init__(
        self,
        image_ids,
        image_dir,
        transform,
        cache: _TensorCache | None = None,
        rgb_cache: _TensorCache | None = None,
    ):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.transform = transform
        self.cache = cache
        self.rgb_cache = rgb_cache

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        x = None
        if self.cache is not None:
            x = self.cache.get(image_id)
        if x is None:
            img = None
            if self.rgb_cache is not None:
                img = self.rgb_cache.get(image_id)
            if img is None:
                img_path = os.path.join(self.image_dir, image_id)
                img = _read_rgb_tensor(img_path)
                if self.rgb_cache is not None:
                    self.rgb_cache.set(image_id, img)
            x = self.transform(img)
            if self.cache is not None:
                self.cache.set(image_id, x)
        return x, image_id


def collect_logits_test_from_loader(model, loader, num_samples: int, num_classes: int):
    model.eval()
    out = np.empty((num_samples, num_classes), dtype=np.float32)
    ids = [None] * num_samples
    i = 0
    with torch.inference_mode():
        for xb, image_ids_batch in loader:
            bs = xb.shape[0]
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            logits = model(xb).detach().to("cpu", non_blocking=True)
            out[i : i + bs] = logits.numpy()
            ids[i : i + bs] = list(image_ids_batch)
            i += bs
    return out, ids


test_ds_512 = CassavaTestDataset(
    test_image_ids,
    TEST_IMG_DIR,
    torch_transforms,
    cache=_CACHE_512_TEST,
    rgb_cache=_CACHE_RGB_TEST,
)
test_ds_518 = CassavaTestDataset(
    test_image_ids,
    TEST_IMG_DIR,
    torch_transforms_VIT,
    cache=_CACHE_518_TEST,
    rgb_cache=_CACHE_RGB_TEST,
)

test_loader_512 = DataLoader(
    test_ds_512,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)
test_loader_518 = DataLoader(
    test_ds_518,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH_FACTOR,
    worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)

n_test = len(test_ds_512)
test_log1, ids1 = collect_logits_test_from_loader(
    model1, test_loader_512, n_test, NUM_CLASSES
)
test_log2, ids2 = collect_logits_test_from_loader(
    model2, test_loader_512, n_test, NUM_CLASSES
)
test_log3, ids3 = collect_logits_test_from_loader(
    model3, test_loader_518, n_test, NUM_CLASSES
)

assert ids1 == test_image_ids, "Test ordering mismatch for model1"
assert ids2 == test_image_ids, "Test ordering mismatch for model2"
assert ids3 == test_image_ids, "Test ordering mismatch for model3"

test_features = np.concatenate([test_log1, test_log2, test_log3], axis=1)
prediction = decision_tree.predict(test_features).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert os.path.exists("submission.csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_df)
