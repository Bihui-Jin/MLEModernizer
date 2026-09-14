# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8724690238742823

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
    cpu = os.cpu_count() or 2
    if not _HAS_CUDA:
        return min(4, max(2, cpu // 2))
    return min(8, max(2, cpu))  # GPU benefits from faster CPU-side decode/resize


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


_PIL_RGB_CACHE = {}
_TENSOR_CACHE = {}


def _transform_key(transform) -> str:
    if transform is torch_transforms:
        return "512"
    if transform is torch_transforms_VIT:
        return "518"
    return str(id(transform))


def _get_pil_rgb(img_dir: str, image_id: str) -> Image.Image:
    key = (img_dir, image_id)
    img = _PIL_RGB_CACHE.get(key)
    if img is None:
        img_path = os.path.join(img_dir, image_id)
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        _PIL_RGB_CACHE[key] = img
    return img


def preload_tensors(image_ids, img_dir: str, transform):
    tkey = _transform_key(transform)
    for image_id in image_ids:
        tcache_key = (img_dir, image_id, tkey)
        if tcache_key in _TENSOR_CACHE:
            continue
        img = _get_pil_rgb(img_dir, image_id)
        _TENSOR_CACHE[tcache_key] = transform(img)


_HAS_TF = False


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




## === cell 1
class CassavaIndexDataset(Dataset):
    def __init__(self, image_ids, img_dir: str, transform, indices, labels=None):
        self.image_ids = np.asarray(image_ids)
        self.img_dir = img_dir
        self.transform = transform
        self._tkey = _transform_key(transform)
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, j: int):
        idx = int(self.indices[j])
        image_id = self.image_ids[idx]

        tcache_key = (self.img_dir, image_id, self._tkey)
        x = _TENSOR_CACHE.get(tcache_key)
        if x is None:
            img = _get_pil_rgb(self.img_dir, image_id)
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


def _to_channels_last(model: nn.Module) -> nn.Module:
    if torch.cuda.is_available():
        return model.to(memory_format=torch.channels_last)
    return model


def train_one_model(model, train_loader, val_loader, epochs=2, lr=2e-3):
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

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_image_ids, train_labels_all)):
    preload_tensors(train_image_ids[va_idx], TRAIN_IMG_DIR, torch_transforms)
    preload_tensors(train_image_ids[va_idx], TRAIN_IMG_DIR, torch_transforms_VIT)

    ds_tr_512 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms,
        tr_idx,
        labels=train_labels_all,
    )
    ds_va_512 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms,
        va_idx,
        labels=train_labels_all,
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
        batch_size=128,  # Runtime optimization (equivalent): larger eval batch reduces overhead, no semantic change.
        shuffle=False,
        num_workers=0,  # Runtime optimization: val uses cache; 0 avoids multiprocessing overhead.
        pin_memory=PIN_MEMORY,
    )

    ds_tr_518 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms_VIT,
        tr_idx,
        labels=train_labels_all,
    )
    ds_va_518 = CassavaIndexDataset(
        train_image_ids,
        TRAIN_IMG_DIR,
        torch_transforms_VIT,
        va_idx,
        labels=train_labels_all,
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
        batch_size=128,  # Runtime optimization (equivalent)
        shuffle=False,
        num_workers=0,  # Runtime optimization: val uses cache
        pin_memory=PIN_MEMORY,
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/1104481058.py in <cell line: 0>()
     93     model1 = train_one_model(model1, tr_loader_512, va_loader_512, epochs=2, lr=2e-3)
     94     model2 = train_one_model(model2, tr_loader_512, va_loader_512, epochs=2, lr=2e-3)
---> 95     model3 = train_one_model(model3, tr_loader_518, va_loader_518, epochs=2, lr=2e-3)
     96 
     97     p1 = predict_proba(model1, va_loader_512)

/tmp/ipykernel_56/3823481126.py in train_one_model(model, train_loader, val_loader, epochs, lr)
     72     for _ in range(epochs):
     73         model.train()
---> 74         for xb, yb in train_loader:
     75             xb = xb.to(DEVICE, non_blocking=True)
     76             if torch.cuda.is_available():

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1249         #   (bool: whether successfully get data, any: data if successful else None)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)
   1253         except Exception as e:

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    178                     if remaining <= 0.0:
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()
    182             self.not_full.notify()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    329             else:
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:
    333                     gotit = waiter.acquire(False)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     71         # This following call uses `waitid` with WNOHANG from C side. Therefore,
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:
     75             assert callable(previous_handler)

RuntimeError: DataLoader worker (pid 102) is killed by signal: Killed. 

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].to_numpy()
test_df = pd.DataFrame({"image_id": test_ids})
test_idx = np.arange(len(test_ids), dtype=np.int64)

preload_tensors(test_ids, TEST_IMG_DIR, torch_transforms)
preload_tensors(test_ids, TEST_IMG_DIR, torch_transforms_VIT)

test_ds_512 = CassavaIndexDataset(
    test_ids, TEST_IMG_DIR, torch_transforms, test_idx, labels=None
)
test_loader_512 = DataLoader(
    test_ds_512,
    batch_size=256,  # Runtime optimization (equivalent): fewer iterations in inference.
    shuffle=False,
    num_workers=0,  # Runtime optimization: cache already populated
    pin_memory=PIN_MEMORY,
)

test_ds_518 = CassavaIndexDataset(
    test_ids, TEST_IMG_DIR, torch_transforms_VIT, test_idx, labels=None
)
test_loader_518 = DataLoader(
    test_ds_518,
    batch_size=256,  # Runtime optimization (equivalent)
    shuffle=False,
    num_workers=0,  # Runtime optimization: cache already populated
    pin_memory=PIN_MEMORY,
)

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

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1352577570.py in <cell line: 0>()
     47 
     48 k = float(len(fold_models))
---> 49 p1_test = p1_sum / k
     50 p2_test = p2_sum / k
     51 p3_test = p3_sum / k

TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'
