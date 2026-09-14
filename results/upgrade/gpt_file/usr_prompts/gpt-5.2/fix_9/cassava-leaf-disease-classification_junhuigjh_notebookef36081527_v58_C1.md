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
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import vit_h_14, ViT_H_14_Weights


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

train_df.head(), sample_sub.head()



## === cell 2
train_torch_transforms = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

val_torch_transforms = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

test_torch_transforms = val_torch_transforms

try:
    from torchvision.io import decode_jpeg, ImageReadMode, read_image
    import torchvision.transforms.functional as F

    _HAS_TV_DECODE_JPEG = True
except Exception:
    _HAS_TV_DECODE_JPEG = False

_TFREC_AVAILABLE = os.path.isdir(TRAIN_TFREC_DIR) and os.path.isdir(TEST_TFREC_DIR)


def _list_tfrec_files(tfrec_dir: str):
    files = []
    if os.path.isdir(tfrec_dir):
        for fn in os.listdir(tfrec_dir):
            if (
                fn.endswith(".tfrec")
                or fn.endswith(".tfrecord")
                or fn.endswith(".tfrecords")
            ):
                files.append(os.path.join(tfrec_dir, fn))
    return sorted(files)


import struct


def _iter_tfrecord(path: str):
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) != 12:
                return
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            if len(data) != length:
                return
            f.read(4)  # crc of data
            yield data


def _read_varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _read_len_delimited(buf, i):
    ln, i = _read_varint(buf, i)
    j = i + ln
    return buf[i:j], j


def _skip_field(buf, i, wire_type):
    if wire_type == 0:  # varint
        _, i = _read_varint(buf, i)
        return i
    if wire_type == 1:  # 64-bit
        return i + 8
    if wire_type == 2:  # len-delimited
        _, i = _read_len_delimited(buf, i)
        return i
    if wire_type == 5:  # 32-bit
        return i + 4
    raise ValueError(f"Unsupported wire type {wire_type}")


def _parse_feature_value(feature_msg: bytes):
    i = 0
    out = {"bytes_list": None, "int64_list": None}
    while i < len(feature_msg):
        key, i = _read_varint(feature_msg, i)
        field = key >> 3
        wire = key & 7
        if field in (1, 2, 3) and wire == 2:
            payload, i = _read_len_delimited(feature_msg, i)
            if field == 1:  # bytes_list
                j = 0
                vals = []
                while j < len(payload):
                    k2, j = _read_varint(payload, j)
                    f2 = k2 >> 3
                    w2 = k2 & 7
                    if f2 == 1 and w2 == 2:
                        bts, j = _read_len_delimited(payload, j)
                        vals.append(bts)
                    else:
                        j = _skip_field(payload, j, w2)
                out["bytes_list"] = vals
            elif field == 3:  # int64_list
                j = 0
                vals = []
                while j < len(payload):
                    k2, j = _read_varint(payload, j)
                    f2 = k2 >> 3
                    w2 = k2 & 7
                    if f2 == 1 and w2 == 2:
                        packed, j = _read_len_delimited(payload, j)
                        t = 0
                        while t < len(packed):
                            v, t = _read_varint(packed, t)
                            vals.append(int(v))
                    elif f2 == 1 and w2 == 0:
                        v, j = _read_varint(payload, j)
                        vals.append(int(v))
                    else:
                        j = _skip_field(payload, j, w2)
                out["int64_list"] = vals
            else:
                pass
        else:
            i = _skip_field(feature_msg, i, wire)
    return out


def _parse_example(example_msg: bytes):
    i = 0
    features_blob = None
    while i < len(example_msg):
        key, i = _read_varint(example_msg, i)
        field = key >> 3
        wire = key & 7
        if field == 1 and wire == 2:
            features_blob, i = _read_len_delimited(example_msg, i)
        else:
            i = _skip_field(example_msg, i, wire)
    if features_blob is None:
        return {}

    feats = {}
    j = 0
    while j < len(features_blob):
        key, j = _read_varint(features_blob, j)
        field = key >> 3
        wire = key & 7
        if field == 1 and wire == 2:
            entry, j = _read_len_delimited(features_blob, j)
            k = 0
            map_key = None
            map_val = None
            while k < len(entry):
                k2, k = _read_varint(entry, k)
                f2 = k2 >> 3
                w2 = k2 & 7
                if f2 == 1 and w2 == 2:
                    kb, k = _read_len_delimited(entry, k)
                    map_key = kb.decode("utf-8")
                elif f2 == 2 and w2 == 2:
                    vb, k = _read_len_delimited(entry, k)
                    map_val = _parse_feature_value(vb)
                else:
                    k = _skip_field(entry, k, w2)
            if map_key is not None and map_val is not None:
                feats[map_key] = map_val
        else:
            j = _skip_field(features_blob, j, wire)
    return feats


def _decode_to_pil(jpeg_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(jpeg_bytes)).convert("RGB")


class CassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform=None,
        has_labels: bool = True,
        cache: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.cache = bool(cache)

        self.image_ids = self.df["image_id"].astype(str).values
        self.img_paths = np.array(
            [os.path.join(self.img_dir, x) for x in self.image_ids], dtype=object
        )
        self.labels = None
        if self.has_labels:
            self.labels = self.df["label"].astype(np.int64).values

        self._cache = [None] * len(self.img_paths) if self.cache else None

        self._fast_path = False
        if _HAS_TV_DECODE_JPEG and isinstance(transform, transforms.Compose):
            ts = transform.transforms
            if (
                len(ts) == 3
                and isinstance(ts[0], transforms.Resize)
                and isinstance(ts[1], transforms.ToTensor)
                and isinstance(ts[2], transforms.Normalize)
            ):
                self._resize_size = ts[0].size
                self._mean = torch.tensor(ts[2].mean, dtype=torch.float32).view(3, 1, 1)
                self._std = torch.tensor(ts[2].std, dtype=torch.float32).view(3, 1, 1)
                self._fast_path = True

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            cached = self._cache[idx]
            if cached is not None:
                return cached

        img_path = self.img_paths[idx]

        if self._fast_path:
            img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 CHW
            img = F.resize(img, self._resize_size, antialias=True)  # uint8 CHW
            img = img.to(dtype=torch.float32).div_(255.0)  # float CHW in [0,1]
            img = (img - self._mean) / self._std
        else:
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        if self.has_labels:
            out = (img, int(self.labels[idx]))
        else:
            out = (img, self.image_ids[idx])

        if self._cache is not None:
            self._cache[idx] = out
        return out


from torch.utils.data import IterableDataset, get_worker_info


class CassavaTFRecordDataset(IterableDataset):
    def __init__(
        self, tfrec_files, transform=None, has_labels=True, seed=42, shuffle=False
    ):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.transform = transform
        self.has_labels = has_labels
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

        self._fast_path = False
        if _HAS_TV_DECODE_JPEG and isinstance(transform, transforms.Compose):
            ts = transform.transforms
            if (
                len(ts) == 3
                and isinstance(ts[0], transforms.Resize)
                and isinstance(ts[1], transforms.ToTensor)
                and isinstance(ts[2], transforms.Normalize)
            ):
                self._resize_size = ts[0].size
                self._mean = torch.tensor(ts[2].mean, dtype=torch.float32).view(3, 1, 1)
                self._std = torch.tensor(ts[2].std, dtype=torch.float32).view(3, 1, 1)
                self._fast_path = True

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        wi = get_worker_info()
        if wi is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = wi.id, wi.num_workers

        files = self.tfrec_files[worker_id::num_workers]

        epoch = getattr(self, "_epoch", 0)
        rng = np.random.default_rng(self.seed + epoch)

        for fp in files:
            records = list(_iter_tfrecord(fp))
            if self.shuffle:
                order = rng.permutation(len(records))
            else:
                order = range(len(records))

            for i in order:
                feats = _parse_example(records[i])

                img_bytes = None
                for k in ("image", "img", "jpeg", "image_bytes"):
                    if k in feats and feats[k].get("bytes_list"):
                        img_bytes = feats[k]["bytes_list"][0]
                        break
                if img_bytes is None:
                    continue

                image_id = None
                for k in ("image_name", "image_id", "filename"):
                    if k in feats and feats[k].get("bytes_list"):
                        image_id = feats[k]["bytes_list"][0].decode("utf-8")
                        break

                label = None
                if self.has_labels:
                    for k in ("label", "target", "class"):
                        if k in feats and feats[k].get("int64_list"):
                            label = int(feats[k]["int64_list"][0])
                            break
                    if label is None:
                        continue

                if self._fast_path:
                    buf = torch.frombuffer(img_bytes, dtype=torch.uint8)
                    img = decode_jpeg(buf, mode=ImageReadMode.RGB)  # uint8 CHW
                    img = F.resize(img, self._resize_size, antialias=True)  # uint8 CHW
                    img = img.to(dtype=torch.float32).div_(255.0)
                    img = (img - self._mean) / self._std
                else:
                    img = _decode_to_pil(img_bytes)
                    if self.transform is not None:
                        img = self.transform(img)

                if self.has_labels:
                    yield img, label
                else:
                    yield img, (image_id if image_id is not None else "")




## === cell 3
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.10,
    random_state=42,
    shuffle=True,
    stratify=train_df["label"].values,
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

BATCH_SIZE = 16  # keep core logic unchanged

_cpu = os.cpu_count() or 2

if torch.cuda.is_available():
    NUM_WORKERS = min(8, max(2, _cpu // 2))
else:
    NUM_WORKERS = min(8, max(2, _cpu - 1))

USE_CACHE = False

train_tfrec_files = _list_tfrec_files(TRAIN_TFREC_DIR)
test_tfrec_files = _list_tfrec_files(TEST_TFREC_DIR)

if _TFREC_AVAILABLE and len(train_tfrec_files) > 0:
    train_ds = CassavaTFRecordDataset(
        train_tfrec_files,
        transform=train_torch_transforms,
        has_labels=True,
        seed=42,
        shuffle=True,
    )
    val_ds = CassavaTFRecordDataset(
        train_tfrec_files,
        transform=val_torch_transforms,
        has_labels=True,
        seed=42,
        shuffle=False,
    )
    train_ds = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        transform=train_torch_transforms,
        has_labels=True,
        cache=USE_CACHE,
    )
    val_ds = CassavaDataset(
        va_df,
        TRAIN_IMG_DIR,
        transform=val_torch_transforms,
        has_labels=True,
        cache=USE_CACHE,
    )
else:
    train_ds = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        transform=train_torch_transforms,
        has_labels=True,
        cache=USE_CACHE,
    )
    val_ds = CassavaDataset(
        va_df,
        TRAIN_IMG_DIR,
        transform=val_torch_transforms,
        has_labels=True,
        cache=USE_CACHE,
    )

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=(
        16 if NUM_WORKERS > 0 else None
    ),  # higher prefetch reduces GPU stalls
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=16 if NUM_WORKERS > 0 else None,
)

len(train_loader), len(val_loader)



## === cell 4
num_classes = 5

try:
    weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    backbone = vit_h_14(weights=weights)
except Exception:
    backbone = vit_h_14(weights=None)

in_features = backbone.heads.head.in_features
backbone.heads.head = nn.Linear(in_features, num_classes)

model = backbone.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


def correct_count_from_logits(logits, targets):
    return (logits.argmax(dim=1) == targets).sum().item()


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available() and hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        pass

model



## === cell 5
EPOCHS = 2  # keep unchanged

for epoch in range(1, EPOCHS + 1):
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)

    model.train()
    tr_loss = 0.0
    tr_correct = 0
    n_tr = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        if torch.cuda.is_available():
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        tr_loss += loss.item() * bs
        tr_correct += correct_count_from_logits(logits.detach(), labels)
        n_tr += bs

    model.eval()
    va_loss = 0.0
    va_correct = 0
    n_va = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            if torch.cuda.is_available():
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)
            bs = labels.size(0)
            va_loss += loss.item() * bs
            va_correct += correct_count_from_logits(logits, labels)
            n_va += bs

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss {tr_loss/n_tr:.4f} acc {tr_correct/n_tr:.4f} | "
        f"val loss {va_loss/n_va:.4f} acc {va_correct/n_va:.4f}"
    )



## === cell 6
test_df = sample_sub[["image_id"]].copy()

if _TFREC_AVAILABLE and len(test_tfrec_files) > 0:
    test_ds = CassavaTFRecordDataset(
        test_tfrec_files,
        transform=test_torch_transforms,
        has_labels=False,
        seed=42,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=16 if NUM_WORKERS > 0 else None,
    )
    _need_reorder = True
else:
    test_ds = CassavaDataset(
        test_df,
        TEST_IMG_DIR,
        transform=test_torch_transforms,
        has_labels=False,
        cache=USE_CACHE,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=16 if NUM_WORKERS > 0 else None,
    )
    _need_reorder = False

pred_labels = []
pred_image_ids = []

model.eval()
with torch.no_grad():
    for imgs, image_ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        if torch.cuda.is_available():
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()
        pred_labels.extend(preds)
        if isinstance(image_ids, (list, tuple)):
            pred_image_ids.extend(list(image_ids))
        else:
            pred_image_ids.extend(list(image_ids))

if _need_reorder:
    pred_map = dict(zip(pred_image_ids, pred_labels))
    pred_labels = [int(pred_map[iid]) for iid in test_df["image_id"].values]

len(pred_labels), len(test_df)



## === cell 7
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample_submission."
assert list(submission.columns) == ["image_id", "label"]

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 8
print("Saved:", os.path.abspath("submission.csv"))
print("Rows:", len(submission))
print(submission["label"].value_counts().sort_index())
