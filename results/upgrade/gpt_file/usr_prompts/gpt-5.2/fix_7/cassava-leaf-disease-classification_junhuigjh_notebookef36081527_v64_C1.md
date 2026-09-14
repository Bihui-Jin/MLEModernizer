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


_TENSOR_CACHE = (
    {}
)  # unchanged API; will no longer be the primary path when TFRecords are available.


def _transform_key(transform) -> str:
    if transform is torch_transforms:
        return "512"
    if transform is torch_transforms_VIT:
        return "518"
    return str(id(transform))


_HAS_TF = False
try:
    import tensorflow as tf  # present on Kaggle CPU/GPU images for this competition

    _HAS_TF = True
except Exception:
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
    return _HAS_TF and len(TRAIN_TFRECS) > 0 and len(TEST_TFRECS) > 0


def _maybe_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile"):
        try:
            return torch.compile(
                model
            )  # preserves semantics (same ops), only changes execution strategy
        except Exception:
            return model
    return model




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    model = _maybe_compile(model)
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


if _can_use_tfrecords():
    _TF_FEATURES = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }

    def _tf_decode(example_proto, img_size: int, labeled: bool):
        ex = tf.io.parse_single_example(example_proto, _TF_FEATURES)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(
            img, [img_size, img_size], method="bilinear", antialias=True
        )
        img = tf.cast(img, tf.float32) / 255.0
        img = img * 2.0 - 1.0
        if labeled:
            return img, tf.cast(ex["target"], tf.int32)
        return img

    def _make_tf_dataset(
        tfrecs, img_size: int, batch_size: int, labeled: bool, shuffle: bool, seed: int
    ):
        ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x: _tf_decode(x, img_size, labeled),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    class _TorchFromTFDataset:
        def __init__(self, tf_ds, labeled: bool):
            self.tf_ds = tf_ds
            self.labeled = labeled

        def __iter__(self):
            for batch in self.tf_ds:
                if self.labeled:
                    xb, yb = batch
                    xb = torch.from_numpy(xb.numpy()).permute(0, 3, 1, 2).contiguous()
                    yb = torch.from_numpy(yb.numpy()).long()
                    yield xb, yb
                else:
                    xb = batch
                    xb = torch.from_numpy(xb.numpy()).permute(0, 3, 1, 2).contiguous()
                    yield (xb,)

        def __len__(self):
            return 0




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
    if _can_use_tfrecords():
        use_tfrecords_for_fold = False
    else:
        use_tfrecords_for_fold = False

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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/3170901049.py in <cell line: 0>()
     86     model3 = TinyCNN(width=48, num_classes=N_CLASSES)
     87 
---> 88     model1 = train_one_model(model1, tr_loader_512, va_loader_512, epochs=2, lr=2e-3)
     89     model2 = train_one_model(model2, tr_loader_512, va_loader_512, epochs=2, lr=2e-3)
     90     model3 = train_one_model(model3, tr_loader_518, va_loader_518, epochs=2, lr=2e-3)

/tmp/ipykernel_55/599415343.py in train_one_model(model, train_loader, val_loader, epochs, lr)
     99             yb = yb.to(DEVICE, non_blocking=True)
    100             opt.zero_grad(set_to_none=True)
--> 101             logits = model(xb)
    102             loss = crit(logits, yb)
    103             loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_autograd(flat_fn, flat_args, aot_config, fw_metadata)
    447             if fake_mode is not None and fake_mode.shape_env is not None:
    448                 tensorify_python_scalars(fx_g, fake_mode.shape_env, fake_mode)
--> 449             fw_module, bw_module = aot_config.partition_fn(
    450                 fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
    451             )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in partition_fn(gm, joint_inputs, **kwargs)
   1777             cuda_context = get_cuda_device_context(gm)
   1778             with cuda_context:
-> 1779                 _recursive_joint_graph_passes(gm)
   1780             return min_cut_rematerialization_partition(
   1781                 gm, joint_inputs, **kwargs, compiler="inductor"

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _recursive_joint_graph_passes(gm)
    320             subgraph = getattr(gm, subgraph_name)
    321             _recursive_joint_graph_passes(subgraph)
--> 322         joint_graph_passes(gm)
    323 
    324 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/joint_graph.py in joint_graph_passes(graph)
    466             maybe_count = GraphTransformObserver(
    467                 graph, f"pass_pattern_{i}"
--> 468             ).apply_graph_pass(patterns.apply)
    469             count += maybe_count if maybe_count is not None else 0
    470 

/usr/local/lib/python3.11/dist-packages/torch/fx/passes/graph_transform_observer.py in apply_graph_pass(self, pass_fn)
     68         with self:
     69             if not self._check_disable_pass():
---> 70                 return pass_fn(self.gm.graph)
     71 
     72         return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in apply(self, gm)
   1771                     if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
   1772                         log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
-> 1773                     if is_match(m) and entry.extra_check(m):
   1774                         count += 1
   1775                         entry.apply(m, graph, node)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in check_fn(match)
   1350             specific_pattern_match = specific_pattern.match(node)
   1351 
-> 1352             if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
   1353                 # trace the pattern using the shapes from the user program
   1354                 match.replacement_graph = trace_fn(replace_fn, args)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_mm(match)
    706 def should_pad_mm(match: Match) -> bool:
    707     mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
--> 708     return should_pad_common(mat1, mat2) and should_pad_bench(
    709         match, mat1, mat2, torch.ops.aten.mm
    710     )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_bench(*args, **kwargs)
    384 def should_pad_bench(*args, **kwargs):
    385     with dynamo_timed("pad_mm_benchmark"):
--> 386         return _should_pad_bench(*args, **kwargs)
    387 
    388 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in _should_pad_bench(match, mat1, mat2, op, input)
    592 
    593         if ori_time is None:
--> 594             ori_time = do_bench(orig_bench_fn)
    595             set_cached_base_mm_benchmark_time(ori_time_key, ori_time)
    596 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in wrapper(self, *args, **kwargs)
     64             "benchmarking." + self.__class__.__name__ + "." + fn.__name__
     65         ] += 1
---> 66         return fn(self, *args, **kwargs)
     67 
     68     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in benchmark_gpu(self, _callable, **kwargs)
    200         elif "return_mode" in kwargs:
    201             return self.triton_do_bench(_callable, **kwargs)
--> 202         return self.triton_do_bench(_callable, **kwargs, return_mode="median")
    203 
    204 

/usr/local/lib/python3.11/dist-packages/triton/testing.py in do_bench(fn, warmup, rep, grad_to_none, quantiles, return_mode)
    115     di = runtime.driver.active.get_device_interface()
    116 
--> 117     fn()
    118     di.synchronize()
    119 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in orig_bench_fn()
    561         def orig_bench_fn():
    562             if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
--> 563                 op(mat1, mat2)
    564             else:
    565                 op(input, mat1, mat2)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
   1121         if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
   1122             return _call_overload_packet_from_python(self, args, kwargs)
-> 1123         return self._op(*args, **(kwargs or {}))
   1124 
   1125     # TODO: use this to make a __dir__

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 3

full_idx = np.arange(len(train_df), dtype=np.int64)

if _can_use_tfrecords():
    tf_train_512 = _make_tf_dataset(
        TRAIN_TFRECS, img_size=512, batch_size=32, labeled=True, shuffle=True, seed=42
    )
    tf_train_518 = _make_tf_dataset(
        TRAIN_TFRECS, img_size=518, batch_size=32, labeled=True, shuffle=True, seed=42
    )
    full_loader_512 = _TorchFromTFDataset(tf_train_512, labeled=True)
    full_loader_518 = _TorchFromTFDataset(tf_train_518, labeled=True)
else:
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/1616673194.py in <cell line: 0>()
     52 final_model3 = TinyCNN(width=48, num_classes=N_CLASSES)
     53 
---> 54 final_model1 = train_one_model(final_model1, full_loader_512, None, epochs=2, lr=2e-3)
     55 final_model2 = train_one_model(final_model2, full_loader_512, None, epochs=2, lr=2e-3)
     56 final_model3 = train_one_model(final_model3, full_loader_518, None, epochs=2, lr=2e-3)

/tmp/ipykernel_55/599415343.py in train_one_model(model, train_loader, val_loader, epochs, lr)
     99             yb = yb.to(DEVICE, non_blocking=True)
    100             opt.zero_grad(set_to_none=True)
--> 101             logits = model(xb)
    102             loss = crit(logits, yb)
    103             loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_autograd(flat_fn, flat_args, aot_config, fw_metadata)
    447             if fake_mode is not None and fake_mode.shape_env is not None:
    448                 tensorify_python_scalars(fx_g, fake_mode.shape_env, fake_mode)
--> 449             fw_module, bw_module = aot_config.partition_fn(
    450                 fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
    451             )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in partition_fn(gm, joint_inputs, **kwargs)
   1777             cuda_context = get_cuda_device_context(gm)
   1778             with cuda_context:
-> 1779                 _recursive_joint_graph_passes(gm)
   1780             return min_cut_rematerialization_partition(
   1781                 gm, joint_inputs, **kwargs, compiler="inductor"

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _recursive_joint_graph_passes(gm)
    320             subgraph = getattr(gm, subgraph_name)
    321             _recursive_joint_graph_passes(subgraph)
--> 322         joint_graph_passes(gm)
    323 
    324 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/joint_graph.py in joint_graph_passes(graph)
    466             maybe_count = GraphTransformObserver(
    467                 graph, f"pass_pattern_{i}"
--> 468             ).apply_graph_pass(patterns.apply)
    469             count += maybe_count if maybe_count is not None else 0
    470 

/usr/local/lib/python3.11/dist-packages/torch/fx/passes/graph_transform_observer.py in apply_graph_pass(self, pass_fn)
     68         with self:
     69             if not self._check_disable_pass():
---> 70                 return pass_fn(self.gm.graph)
     71 
     72         return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in apply(self, gm)
   1771                     if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
   1772                         log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
-> 1773                     if is_match(m) and entry.extra_check(m):
   1774                         count += 1
   1775                         entry.apply(m, graph, node)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in check_fn(match)
   1350             specific_pattern_match = specific_pattern.match(node)
   1351 
-> 1352             if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
   1353                 # trace the pattern using the shapes from the user program
   1354                 match.replacement_graph = trace_fn(replace_fn, args)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_mm(match)
    706 def should_pad_mm(match: Match) -> bool:
    707     mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
--> 708     return should_pad_common(mat1, mat2) and should_pad_bench(
    709         match, mat1, mat2, torch.ops.aten.mm
    710     )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_bench(*args, **kwargs)
    384 def should_pad_bench(*args, **kwargs):
    385     with dynamo_timed("pad_mm_benchmark"):
--> 386         return _should_pad_bench(*args, **kwargs)
    387 
    388 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in _should_pad_bench(match, mat1, mat2, op, input)
    592 
    593         if ori_time is None:
--> 594             ori_time = do_bench(orig_bench_fn)
    595             set_cached_base_mm_benchmark_time(ori_time_key, ori_time)
    596 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in wrapper(self, *args, **kwargs)
     64             "benchmarking." + self.__class__.__name__ + "." + fn.__name__
     65         ] += 1
---> 66         return fn(self, *args, **kwargs)
     67 
     68     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in benchmark_gpu(self, _callable, **kwargs)
    200         elif "return_mode" in kwargs:
    201             return self.triton_do_bench(_callable, **kwargs)
--> 202         return self.triton_do_bench(_callable, **kwargs, return_mode="median")
    203 
    204 

/usr/local/lib/python3.11/dist-packages/triton/testing.py in do_bench(fn, warmup, rep, grad_to_none, quantiles, return_mode)
    115     di = runtime.driver.active.get_device_interface()
    116 
--> 117     fn()
    118     di.synchronize()
    119 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in orig_bench_fn()
    561         def orig_bench_fn():
    562             if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
--> 563                 op(mat1, mat2)
    564             else:
    565                 op(input, mat1, mat2)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
   1121         if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
   1122             return _call_overload_packet_from_python(self, args, kwargs)
-> 1123         return self._op(*args, **(kwargs or {}))
   1124 
   1125     # TODO: use this to make a __dir__

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()
test_df = pd.DataFrame({"image_id": test_ids})
test_idx = np.arange(len(test_df), dtype=np.int64)

if _can_use_tfrecords():
    tf_test_512 = _make_tf_dataset(
        TEST_TFRECS, img_size=512, batch_size=64, labeled=False, shuffle=False, seed=42
    )
    tf_test_518 = _make_tf_dataset(
        TEST_TFRECS, img_size=518, batch_size=64, labeled=False, shuffle=False, seed=42
    )
    test_loader_512 = _TorchFromTFDataset(tf_test_512, labeled=False)
    test_loader_518 = _TorchFromTFDataset(tf_test_518, labeled=False)
else:
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

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2280582174.py in <cell line: 0>()
     44 
     45 p1_test = predict_proba(final_model1, test_loader_512)
---> 46 p2_test = predict_proba(final_model2, test_loader_512)
     47 p3_test = predict_proba(final_model3, test_loader_518)
     48 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/599415343.py in predict_proba(model, loader)
    113         xb = batch[0]
    114         xb = xb.to(DEVICE, non_blocking=True)
--> 115         logits = model(xb)
    116         probs = torch.softmax(logits, dim=1).cpu().numpy()
    117         all_probs.append(probs)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/599415343.py in forward(self, x)
     82 
     83     def forward(self, x):
---> 84         x = self.features(x)
     85         x = x.flatten(1)
     86         return self.classifier(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
