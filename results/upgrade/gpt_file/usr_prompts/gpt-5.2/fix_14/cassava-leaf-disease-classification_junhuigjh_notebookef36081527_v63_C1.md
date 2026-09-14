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
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)



## === cell 1
W_DENSE = DenseNet121_Weights.IMAGENET1K_V1
W_RESNET = ResNet50_Weights.IMAGENET1K_V2
W_VIT = ViT_B_16_Weights.IMAGENET1K_V1

from torchvision.io import read_image, ImageReadMode
from collections import OrderedDict

torch_transforms_dense = W_DENSE.transforms()
torch_transforms_resnet = W_RESNET.transforms()
torch_transforms_vit = W_VIT.transforms()


class _LRUCache:
    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, k):
        v = self._d.get(k, None)
        if v is not None:
            self._d.move_to_end(k)
        return v

    def put(self, k, v):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


_RAW_CACHE_MAX = 6000
_XFORM_CACHE_MAX = 12000
_GLOBAL_RAW_IMG_CACHE = _LRUCache(_RAW_CACHE_MAX)  # (img_dir|image_id) -> uint8 CHW
_GLOBAL_XFORM_IMG_CACHE = _LRUCache(
    _XFORM_CACHE_MAX
)  # (img_dir|image_id|transform_tag) -> float tensor CHW


def _raw_key(img_dir: str, image_id: str) -> str:
    return img_dir + "|" + image_id


def _xform_key(img_dir: str, image_id: str, transform_tag: str) -> str:
    return img_dir + "|" + image_id + "|" + transform_tag


class CassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform,
        transform_tag: str,
        use_global_cache: bool = True,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.transform_tag = transform_tag
        self.use_global_cache = use_global_cache

        self.image_ids = df["image_id"].astype(str).to_numpy()
        self.has_label = "label" in df.columns
        self.labels = (
            df["label"].astype(np.int64).to_numpy() if self.has_label else None
        )

    def __len__(self):
        return self.image_ids.shape[0]

    def _load_raw(self, image_id: str):
        k = _raw_key(self.img_dir, image_id)
        if self.use_global_cache:
            cached = _GLOBAL_RAW_IMG_CACHE.get(k)
            if cached is not None:
                return cached
        img_path = os.path.join(self.img_dir, image_id)
        img = read_image(img_path, mode=ImageReadMode.RGB)
        if self.use_global_cache:
            _GLOBAL_RAW_IMG_CACHE.put(k, img)
        return img

    def _load_transformed(self, image_id: str):
        if not self.use_global_cache:
            x = self.transform(self._load_raw(image_id))
            return x.contiguous()

        k = _xform_key(self.img_dir, image_id, self.transform_tag)
        cached = _GLOBAL_XFORM_IMG_CACHE.get(k)
        if cached is not None:
            return cached

        x = self.transform(self._load_raw(image_id)).contiguous()
        _GLOBAL_XFORM_IMG_CACHE.put(k, x)
        return x

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        x = self._load_transformed(image_id)
        if self.has_label:
            y = int(self.labels[idx])
            return x, y, image_id
        return x, image_id


@torch.inference_mode()
def predict_proba(model, loader, num_classes=5):
    model.eval()
    all_probs = []
    all_ids = []
    for batch in loader:
        if len(batch) == 3:
            x, _, image_ids = batch
        else:
            x, image_ids = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(image_ids)  # already list
    all_probs = np.concatenate(all_probs, axis=0)
    assert all_probs.shape[1] == num_classes
    return all_probs, all_ids


@torch.inference_mode()
def predict_proba_ensemble_sum(models, loader, num_classes=5):
    for m in models:
        m.eval()
    all_probs_sum = []
    all_ids = []
    for batch in loader:
        if len(batch) == 3:
            x, _, image_ids = batch
        else:
            x, image_ids = batch
        x = x.to(device, non_blocking=True)
        probs_sum = None
        for m in models:
            logits = m(x)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)
        all_probs_sum.append(probs_sum.detach().cpu().numpy())
        all_ids.extend(image_ids)  # already list
    all_probs_sum = np.concatenate(all_probs_sum, axis=0)
    assert all_probs_sum.shape[1] == num_classes
    return all_probs_sum, all_ids


def set_trainable_classifier_only(model, arch_name: str):
    for p in model.parameters():
        p.requires_grad = False

    if arch_name == "densenet":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif arch_name == "resnet":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif arch_name == "vit":
        for p in model.heads.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown arch_name: {arch_name}")


def train_one_epoch_classifier_head(model, loader, arch_name: str, lr: float = 1e-3):
    set_trainable_classifier_only(model, arch_name)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    total = 0
    running_loss = 0.0
    correct_gpu = torch.zeros((), device=device, dtype=torch.long)

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        running_loss += float(loss.detach().cpu().item()) * x.size(0)
        total += x.size(0)
        correct_gpu += (logits.argmax(1) == y).sum()

    correct = int(correct_gpu.detach().cpu().item())
    return running_loss / max(total, 1), correct / max(total, 1)


def build_models():
    m1 = densenet121(weights=W_DENSE)
    m1.classifier = nn.Linear(m1.classifier.in_features, 5)
    m1 = m1.to(device)

    m2 = resnet50(weights=W_RESNET)
    m2.fc = nn.Linear(m2.fc.in_features, 5)
    m2 = m2.to(device)

    m3 = vit_b_16(weights=W_VIT)
    m3.heads.head = nn.Linear(m3.heads.head.in_features, 5)
    m3 = m3.to(device)

    if torch.cuda.is_available():
        m1 = m1.to(memory_format=torch.channels_last)
        m2 = m2.to(memory_format=torch.channels_last)
        m3 = m3.to(memory_format=torch.channels_last)

    return m1, m2, m3




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
assert set(["image_id", "label"]).issubset(train_df.columns)
y = train_df["label"].astype(int).values

n_classes = 5
oof_p1 = np.zeros((len(train_df), n_classes), dtype=np.float32)
oof_p2 = np.zeros((len(train_df), n_classes), dtype=np.float32)
oof_p3 = np.zeros((len(train_df), n_classes), dtype=np.float32)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

BATCH_CNN = 32 if torch.cuda.is_available() else 8
BATCH_VIT = 16 if torch.cuda.is_available() else 4

cpu_cnt = os.cpu_count() or 2
if torch.cuda.is_available():
    NUM_WORKERS = min(4, cpu_cnt)  # was up to 8
    PREFETCH = 4  # was 8
else:
    NUM_WORKERS = min(2, cpu_cnt)
    PREFETCH = 2

pin = torch.cuda.is_available()


def _collate_train(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), list(ids)


def _collate_test(batch):
    xs, ids = zip(*batch)
    return torch.stack(xs, 0), list(ids)


def make_loader(ds, batch_size, shuffle, generator=None, is_train: bool = True):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        collate_fn=_collate_train if is_train else _collate_test,
        drop_last=False,
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH
    if generator is not None and shuffle:
        kwargs["generator"] = generator
    return DataLoader(ds, **kwargs)


fold_models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
    model1, model2, model3 = build_models()

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    tr_ds_dense = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        torch_transforms_dense,
        transform_tag="dense",
        use_global_cache=True,
    )
    tr_loader_dense = make_loader(
        tr_ds_dense,
        batch_size=BATCH_CNN,
        shuffle=True,
        generator=dl_generator,
        is_train=True,
    )
    va_ds_dense = CassavaDataset(
        va_df,
        TRAIN_IMG_DIR,
        torch_transforms_dense,
        transform_tag="dense",
        use_global_cache=True,
    )
    va_loader_dense = make_loader(
        va_ds_dense, batch_size=BATCH_CNN, shuffle=False, is_train=True
    )

    tr_ds_resnet = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        torch_transforms_resnet,
        transform_tag="resnet",
        use_global_cache=True,
    )
    tr_loader_resnet = make_loader(
        tr_ds_resnet,
        batch_size=BATCH_CNN,
        shuffle=True,
        generator=dl_generator,
        is_train=True,
    )
    va_ds_resnet = CassavaDataset(
        va_df,
        TRAIN_IMG_DIR,
        torch_transforms_resnet,
        transform_tag="resnet",
        use_global_cache=True,
    )
    va_loader_resnet = make_loader(
        va_ds_resnet, batch_size=BATCH_CNN, shuffle=False, is_train=True
    )

    tr_ds_vit = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        torch_transforms_vit,
        transform_tag="vit",
        use_global_cache=True,
    )
    tr_loader_vit = make_loader(
        tr_ds_vit,
        batch_size=BATCH_VIT,
        shuffle=True,
        generator=dl_generator,
        is_train=True,
    )
    va_ds_vit = CassavaDataset(
        va_df,
        TRAIN_IMG_DIR,
        torch_transforms_vit,
        transform_tag="vit",
        use_global_cache=True,
    )
    va_loader_vit = make_loader(
        va_ds_vit, batch_size=BATCH_VIT, shuffle=False, is_train=True
    )

    l1, a1 = train_one_epoch_classifier_head(
        model1, tr_loader_dense, "densenet", lr=1e-3
    )
    l2, a2 = train_one_epoch_classifier_head(
        model2, tr_loader_resnet, "resnet", lr=1e-3
    )
    l3, a3 = train_one_epoch_classifier_head(model3, tr_loader_vit, "vit", lr=5e-4)

    p1, ids1 = predict_proba(model1, va_loader_dense, num_classes=n_classes)
    p2, ids2 = predict_proba(model2, va_loader_resnet, num_classes=n_classes)
    p3, ids3 = predict_proba(model3, va_loader_vit, num_classes=n_classes)

    assert ids1 == list(va_df["image_id"].values)
    assert ids2 == list(va_df["image_id"].values)
    assert ids3 == list(va_df["image_id"].values)

    oof_p1[va_idx] = p1
    oof_p2[va_idx] = p2
    oof_p3[va_idx] = p3

    fold_models.append((model1, model2, model3))

    print(
        f"OOF fold {fold}/3 done: {len(va_idx)} val samples | "
        f"dense head train loss/acc={l1:.4f}/{a1:.4f} | "
        f"resnet head train loss/acc={l2:.4f}/{a2:.4f} | "
        f"vit head train loss/acc={l3:.4f}/{a3:.4f}"
    )

train_meta_X = np.concatenate([oof_p1, oof_p2, oof_p3], axis=1)
train_meta_y = y

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=9,
    random_state=SEED,
)
decision_tree.fit(train_meta_X, train_meta_y)

final_model1, final_model2, final_model3 = build_models()

full_ds_dense = CassavaDataset(
    train_df,
    TRAIN_IMG_DIR,
    torch_transforms_dense,
    transform_tag="dense",
    use_global_cache=True,
)
full_loader_dense = make_loader(
    full_ds_dense, batch_size=BATCH_CNN, shuffle=False, generator=None, is_train=True
)

full_ds_resnet = CassavaDataset(
    train_df,
    TRAIN_IMG_DIR,
    torch_transforms_resnet,
    transform_tag="resnet",
    use_global_cache=True,
)
full_loader_resnet = make_loader(
    full_ds_resnet, batch_size=BATCH_CNN, shuffle=False, generator=None, is_train=True
)

full_ds_vit = CassavaDataset(
    train_df,
    TRAIN_IMG_DIR,
    torch_transforms_vit,
    transform_tag="vit",
    use_global_cache=True,
)
full_loader_vit = make_loader(
    full_ds_vit, batch_size=BATCH_VIT, shuffle=False, generator=None, is_train=True
)

fl1, fa1 = train_one_epoch_classifier_head(
    final_model1, full_loader_dense, "densenet", lr=1e-3
)
fl2, fa2 = train_one_epoch_classifier_head(
    final_model2, full_loader_resnet, "resnet", lr=1e-3
)
fl3, fa3 = train_one_epoch_classifier_head(
    final_model3, full_loader_vit, "vit", lr=5e-4
)
print(
    f"Final full-data head training done | "
    f"dense loss/acc={fl1:.4f}/{fa1:.4f} | "
    f"resnet loss/acc={fl2:.4f}/{fa2:.4f} | "
    f"vit loss/acc={fl3:.4f}/{fa3:.4f}"
)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()
test_df = pd.DataFrame({"image_id": test_ids})

test_ds_dense = CassavaDataset(
    test_df,
    TEST_IMG_DIR,
    torch_transforms_dense,
    transform_tag="dense",
    use_global_cache=True,
)
test_loader_dense = make_loader(
    test_ds_dense, batch_size=BATCH_CNN, shuffle=False, is_train=False
)

test_ds_resnet = CassavaDataset(
    test_df,
    TEST_IMG_DIR,
    torch_transforms_resnet,
    transform_tag="resnet",
    use_global_cache=True,
)
test_loader_resnet = make_loader(
    test_ds_resnet, batch_size=BATCH_CNN, shuffle=False, is_train=False
)

test_ds_vit = CassavaDataset(
    test_df,
    TEST_IMG_DIR,
    torch_transforms_vit,
    transform_tag="vit",
    use_global_cache=True,
)
test_loader_vit = make_loader(
    test_ds_vit, batch_size=BATCH_VIT, shuffle=False, is_train=False
)

dense_models = [m1 for (m1, _, _) in fold_models]
resnet_models = [m2 for (_, m2, _) in fold_models]
vit_models = [m3 for (_, _, m3) in fold_models]

test_p1_sum, ids1 = predict_proba_ensemble_sum(
    dense_models, test_loader_dense, num_classes=n_classes
)
test_p2_sum, ids2 = predict_proba_ensemble_sum(
    resnet_models, test_loader_resnet, num_classes=n_classes
)
test_p3_sum, ids3 = predict_proba_ensemble_sum(
    vit_models, test_loader_vit, num_classes=n_classes
)

assert ids1 == test_ids
assert ids2 == test_ids
assert ids3 == test_ids

test_p1 = test_p1_sum / len(fold_models)
test_p2 = test_p2_sum / len(fold_models)
test_p3 = test_p3_sum / len(fold_models)

test_meta_X = np.concatenate([test_p1, test_p2, test_p3], axis=1)
test_pred = decision_tree.predict(test_meta_X).astype(int)

submission = pd.DataFrame({"image_id": test_ids, "label": test_pred})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
