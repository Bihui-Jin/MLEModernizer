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
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"


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

from torchvision.io import decode_jpeg

torch_transforms_dense = W_DENSE.transforms()
torch_transforms_resnet = W_RESNET.transforms()
torch_transforms_vit = W_VIT.transforms()

import glob
import struct
from typing import Optional, Tuple, Dict, List


def _list_tfrec_files(tfrec_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(tfrec_dir, "*.tfrec")))
    assert len(files) > 0, f"No tfrecord files found in {tfrec_dir}"
    return files


def _read_tfrec_records(path: str):
    with open(path, "rb") as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes:
                break
            (length,) = struct.unpack("<Q", len_bytes)
            f.read(4)  # skip length crc
            data = f.read(length)
            f.read(4)  # skip data crc
            yield data


def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _read_length_delimited(buf: bytes, i: int) -> Tuple[bytes, int]:
    ln, i = _read_varint(buf, i)
    out = buf[i : i + ln]
    return out, i + ln


def _skip_field(wire_type: int, buf: bytes, i: int) -> int:
    if wire_type == 0:  # varint
        _, i = _read_varint(buf, i)
        return i
    if wire_type == 1:  # 64-bit
        return i + 8
    if wire_type == 2:  # len-delimited
        _, i = _read_length_delimited(buf, i)
        return i
    if wire_type == 5:  # 32-bit
        return i + 4
    raise ValueError(f"Unsupported wire type: {wire_type}")


def _parse_example_for_keys(
    example: bytes, want_label: bool
) -> Tuple[Optional[bytes], Optional[bytes], Optional[int]]:
    i = 0
    image_name = None
    image = None
    label = None

    while i < len(example):
        key, i = _read_varint(example, i)
        field = key >> 3
        wire = key & 0x7
        if field != 1 or wire != 2:
            i = _skip_field(wire, example, i)
            continue
        features_msg, _ = _read_length_delimited(example, i)
        j = 0
        while j < len(features_msg):
            k2, j = _read_varint(features_msg, j)
            f2 = k2 >> 3
            w2 = k2 & 0x7
            if f2 != 1 or w2 != 2:
                j = _skip_field(w2, features_msg, j)
                continue
            entry, j = _read_length_delimited(features_msg, j)
            ek = 0
            name = None
            feat_msg = None
            while ek < len(entry):
                k3, ek = _read_varint(entry, ek)
                f3 = k3 >> 3
                w3 = k3 & 0x7
                if f3 == 1 and w3 == 2:
                    key_bytes, ek = _read_length_delimited(entry, ek)
                    name = key_bytes.decode("utf-8")
                elif f3 == 2 and w3 == 2:
                    feat_msg, ek = _read_length_delimited(entry, ek)
                else:
                    ek = _skip_field(w3, entry, ek)

            if name is None or feat_msg is None:
                continue

            if name not in ("image_name", "image", "label"):
                continue
            if name == "label" and not want_label:
                continue

            fk = 0
            while fk < len(feat_msg):
                k4, fk = _read_varint(feat_msg, fk)
                f4 = k4 >> 3
                w4 = k4 & 0x7
                if w4 != 2:
                    fk = _skip_field(w4, feat_msg, fk)
                    continue
                kind_msg, fk = _read_length_delimited(feat_msg, fk)

                if name in ("image_name", "image") and f4 == 1:
                    bk = 0
                    while bk < len(kind_msg):
                        k5, bk = _read_varint(kind_msg, bk)
                        f5 = k5 >> 3
                        w5 = k5 & 0x7
                        if f5 == 1 and w5 == 2:
                            val, bk = _read_length_delimited(kind_msg, bk)
                            if name == "image_name":
                                image_name = val
                            else:
                                image = val
                            break
                        bk = _skip_field(w5, kind_msg, bk)
                elif name == "label" and f4 == 3:
                    ik = 0
                    while ik < len(kind_msg):
                        k5, ik = _read_varint(kind_msg, ik)
                        f5 = k5 >> 3
                        w5 = k5 & 0x7
                        if f5 == 1 and w5 == 0:
                            v, ik = _read_varint(kind_msg, ik)
                            label = int(v)
                            break
                        ik = _skip_field(w5, kind_msg, ik)

        break  # only one features field

    return image_name, image, label


class CassavaTFRecordIndex:
    def __init__(self, tfrec_dir: str, want_label: bool):
        self.files = _list_tfrec_files(tfrec_dir)
        self.want_label = want_label
        self.loc: Dict[str, Tuple[int, int]] = {}
        for fi, path in enumerate(self.files):
            for ri, ex in enumerate(_read_tfrec_records(path)):
                name_b, _, _ = _parse_example_for_keys(ex, want_label=want_label)
                if name_b is None:
                    continue
                image_id = name_b.decode("utf-8")
                self.loc[image_id] = (fi, ri)

    def fetch(self, image_id: str) -> Tuple[bytes, Optional[int]]:
        fi, ri = self.loc[image_id]
        path = self.files[fi]
        for idx, ex in enumerate(_read_tfrec_records(path)):
            if idx == ri:
                name_b, img_b, lab = _parse_example_for_keys(
                    ex, want_label=self.want_label
                )
                assert name_b is not None and img_b is not None
                return img_b, lab
        raise KeyError(image_id)


_TRAIN_TFREC_INDEX = CassavaTFRecordIndex(TRAIN_TFREC_DIR, want_label=True)
_TEST_TFREC_INDEX = CassavaTFRecordIndex(TEST_TFREC_DIR, want_label=False)


class CassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,  # kept for path-compatibility; TFRecords are used
        transform,
        transform_tag: str,
        use_global_cache: bool = True,  # kept for API compatibility; not used
        tfrec_index: Optional[CassavaTFRecordIndex] = None,
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

        self.tfrec_index = (
            tfrec_index
            if tfrec_index is not None
            else (_TRAIN_TFREC_INDEX if self.has_label else _TEST_TFREC_INDEX)
        )

    def __len__(self):
        return self.image_ids.shape[0]

    def _load_transformed(self, image_id: str) -> torch.Tensor:
        img_bytes, _ = self.tfrec_index.fetch(image_id)
        img_u8 = decode_jpeg(
            torch.frombuffer(memoryview(img_bytes), dtype=torch.uint8), device="cpu"
        )  # CHW uint8
        x = self.transform(img_u8).contiguous()
        return x

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        x = self._load_transformed(image_id)
        if self.has_label:
            y = int(self.labels[idx])
            return x, y, image_id
        return x, image_id


@torch.inference_mode()
def predict_proba(model, loader, num_classes=5, use_channels_last: bool = False):
    model.eval()
    all_probs = []
    all_ids = []
    for batch in loader:
        if len(batch) == 3:
            x, _, image_ids = batch
        else:
            x, image_ids = batch
        x = x.to(device, non_blocking=True)
        if use_channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_ids.extend(image_ids)
    all_probs = np.concatenate(all_probs, axis=0)
    assert all_probs.shape[1] == num_classes
    return all_probs, all_ids


@torch.inference_mode()
def predict_proba_ensemble_sum(
    models, loader, num_classes=5, use_channels_last: bool = False
):
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
        if use_channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        probs_sum = None
        for m in models:
            logits = m(x)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)
        all_probs_sum.append(probs_sum.detach().cpu().numpy())
        all_ids.extend(image_ids)
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

    use_channels_last = arch_name in ("densenet", "resnet")

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        if use_channels_last and torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
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


def _maybe_compile(m: torch.nn.Module) -> torch.nn.Module:
    try:
        if hasattr(torch, "compile"):
            return torch.compile(m, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass
    return m


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

    m1 = _maybe_compile(m1)
    m2 = _maybe_compile(m2)
    m3 = _maybe_compile(m3)

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
    NUM_WORKERS = min(4, cpu_cnt)
    PREFETCH = 4
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
        use_global_cache=False,
        tfrec_index=_TRAIN_TFREC_INDEX,
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
        use_global_cache=False,
        tfrec_index=_TRAIN_TFREC_INDEX,
    )
    va_loader_dense = make_loader(
        va_ds_dense, batch_size=BATCH_CNN, shuffle=False, is_train=True
    )

    tr_ds_resnet = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        torch_transforms_resnet,
        transform_tag="resnet",
        use_global_cache=False,
        tfrec_index=_TRAIN_TFREC_INDEX,
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
        use_global_cache=False,
        tfrec_index=_TRAIN_TFREC_INDEX,
    )
    va_loader_resnet = make_loader(
        va_ds_resnet, batch_size=BATCH_CNN, shuffle=False, is_train=True
    )

    tr_ds_vit = CassavaDataset(
        tr_df,
        TRAIN_IMG_DIR,
        torch_transforms_vit,
        transform_tag="vit",
        use_global_cache=False,
        tfrec_index=_TRAIN_TFREC_INDEX,
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
        use_global_cache=False,
        tfrec_index=_TRAIN_TFREC_INDEX,
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

    p1, ids1 = predict_proba(
        model1, va_loader_dense, num_classes=n_classes, use_channels_last=True
    )
    p2, ids2 = predict_proba(
        model2, va_loader_resnet, num_classes=n_classes, use_channels_last=True
    )
    p3, ids3 = predict_proba(
        model3, va_loader_vit, num_classes=n_classes, use_channels_last=False
    )

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
    use_global_cache=False,
    tfrec_index=_TRAIN_TFREC_INDEX,
)
full_loader_dense = make_loader(
    full_ds_dense, batch_size=BATCH_CNN, shuffle=False, generator=None, is_train=True
)

full_ds_resnet = CassavaDataset(
    train_df,
    TRAIN_IMG_DIR,
    torch_transforms_resnet,
    transform_tag="resnet",
    use_global_cache=False,
    tfrec_index=_TRAIN_TFREC_INDEX,
)
full_loader_resnet = make_loader(
    full_ds_resnet, batch_size=BATCH_CNN, shuffle=False, generator=None, is_train=True
)

full_ds_vit = CassavaDataset(
    train_df,
    TRAIN_IMG_DIR,
    torch_transforms_vit,
    transform_tag="vit",
    use_global_cache=False,
    tfrec_index=_TRAIN_TFREC_INDEX,
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
    use_global_cache=False,
    tfrec_index=_TEST_TFREC_INDEX,
)
test_loader_dense = make_loader(
    test_ds_dense, batch_size=BATCH_CNN, shuffle=False, is_train=False
)

test_ds_resnet = CassavaDataset(
    test_df,
    TEST_IMG_DIR,
    torch_transforms_resnet,
    transform_tag="resnet",
    use_global_cache=False,
    tfrec_index=_TEST_TFREC_INDEX,
)
test_loader_resnet = make_loader(
    test_ds_resnet, batch_size=BATCH_CNN, shuffle=False, is_train=False
)

test_ds_vit = CassavaDataset(
    test_df,
    TEST_IMG_DIR,
    torch_transforms_vit,
    transform_tag="vit",
    use_global_cache=False,
    tfrec_index=_TEST_TFREC_INDEX,
)
test_loader_vit = make_loader(
    test_ds_vit, batch_size=BATCH_VIT, shuffle=False, is_train=False
)

dense_models = [m1 for (m1, _, _) in fold_models]
resnet_models = [m2 for (_, m2, _) in fold_models]
vit_models = [m3 for (_, _, m3) in fold_models]

test_p1_sum, ids1 = predict_proba_ensemble_sum(
    dense_models, test_loader_dense, num_classes=n_classes, use_channels_last=True
)
test_p2_sum, ids2 = predict_proba_ensemble_sum(
    resnet_models, test_loader_resnet, num_classes=n_classes, use_channels_last=True
)
test_p3_sum, ids3 = predict_proba_ensemble_sum(
    vit_models, test_loader_vit, num_classes=n_classes, use_channels_last=False
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
