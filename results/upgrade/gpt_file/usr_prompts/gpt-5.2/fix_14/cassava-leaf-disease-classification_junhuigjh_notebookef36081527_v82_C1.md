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
import torchvision  # needed because __getitem__ references torchvision.io.ImageReadMode
from torchvision import transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights


SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), "Missing train_images directory"
assert os.path.isdir(TEST_IMG_DIR), "Missing test_images directory"

IMG_SIZE = 480

BATCH_SIZE = 8 if torch.cuda.is_available() else 16

NUM_CLASSES = 5
EPOCHS = 2
LR = 3e-4

if torch.cuda.is_available():
    NUM_WORKERS = min(8, (os.cpu_count() or 4))
else:
    NUM_WORKERS = min(4, (os.cpu_count() or 2))

PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None

_HAS_TV_V2 = False
try:
    from torchvision.transforms import v2 as tvv2  # torchvision>=0.15 typically

    _HAS_TV_V2 = True
except Exception:
    _HAS_TV_V2 = False

if _HAS_TV_V2:
    train_tfms = tvv2.Compose(
        [
            tvv2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            tvv2.RandomHorizontalFlip(p=0.5),
            tvv2.ToDtype(torch.float32, scale=True),
            tvv2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_tfms = tvv2.Compose(
        [
            tvv2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            tvv2.ToDtype(torch.float32, scale=True),
            tvv2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
else:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

_HAS_TV_IO_FAST = False
try:
    from torchvision.io import read_file, decode_jpeg

    _HAS_TV_IO_FAST = True
except Exception:
    _HAS_TV_IO_FAST = False

try:
    if NUM_WORKERS > 0:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except Exception:
    pass


_ENABLE_RAM_CACHE = False


class CassavaImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_labels=True, enable_cache=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].astype(int).tolist() if has_labels else None
        self._paths = [os.path.join(self.img_dir, img_id) for img_id in self.image_ids]
        self.enable_cache = bool(enable_cache)
        self._cache = {} if self.enable_cache else None  # idx -> CHW uint8 tensor

    def __len__(self):
        return len(self.image_ids)

    def _load_uint8_chw(self, idx: int):
        if self.enable_cache:
            cached = self._cache.get(idx, None)
            if cached is not None:
                return cached

        img_path = self._paths[idx]

        if _HAS_TV_IO_FAST:
            data = read_file(img_path)  # 1D uint8 tensor
            x = decode_jpeg(data)  # HWC uint8
            if x.ndim == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1).contiguous()  # CHW uint8
            else:
                x = x.contiguous()
                if x.ndim == 3 and x.shape[0] != 3 and x.shape[-1] == 3:
                    x = x.permute(2, 0, 1).contiguous()

            if x.shape[0] == 1:
                x = x.expand(3, -1, -1)
            elif x.shape[0] == 4:
                x = x[:3]
        else:
            img = Image.open(img_path).convert("RGB")
            x = transforms.functional.pil_to_tensor(img)  # CHW uint8

        if self.enable_cache:
            self._cache[idx] = x
        return x

    def __getitem__(self, idx):
        x = self._load_uint8_chw(idx)

        if _HAS_TV_V2:
            x = self.transform(x)
        else:
            img = transforms.functional.to_pil_image(x)
            x = self.transform(img)

        if self.has_labels:
            y = int(self.labels[idx])
            return x, y
        return x, self.image_ids[idx]


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)

_USE_TFRECORDS = False
print("TFRecords enabled:", _USE_TFRECORDS)


def _collate_train(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def _collate_test(batch):
    xs, ids = zip(*batch)
    return torch.stack(xs, 0), list(ids)




## === cell 1
weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
backbone = efficientnet_v2_l(weights=weights)

in_features = backbone.classifier[1].in_features
backbone.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = backbone.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

_ENABLE_TORCH_COMPILE = False
if _ENABLE_TORCH_COMPILE:
    try:
        if hasattr(torch, "compile"):
            model = torch.compile(
                model
            )  # preserves results up to negligible fp differences
            print("torch.compile enabled")
    except Exception as e:
        print("torch.compile skipped:", repr(e))

print("Model ready. AMP enabled:", use_amp)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) >= {"image_id", "label"}

perm = np.random.RandomState(SEED).permutation(len(train_df))
split = int(0.9 * len(train_df))
tr_idx, va_idx = perm[:split], perm[split:]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

common_loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=PERSISTENT_WORKERS,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=dl_generator,
)
if torch.cuda.is_available():
    common_loader_kwargs["pin_memory_device"] = "cuda"
if PREFETCH_FACTOR is not None:
    common_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

try:
    common_loader_kwargs["in_order"] = True
except Exception:
    pass

train_loader = DataLoader(
    CassavaImageDataset(
        tr_df,
        TRAIN_IMG_DIR,
        train_tfms,
        has_labels=True,
        enable_cache=_ENABLE_RAM_CACHE,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    collate_fn=_collate_train,
    **common_loader_kwargs,
)

val_loader = DataLoader(
    CassavaImageDataset(
        va_df, TRAIN_IMG_DIR, test_tfms, has_labels=True, enable_cache=_ENABLE_RAM_CACHE
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=_collate_train,
    **common_loader_kwargs,
)

len_train = len(tr_df)
len_val = len(va_df)
print(f"Train/Val sizes: {len_train}/{len_val}")
print(f"Batch size: {BATCH_SIZE}, workers: {NUM_WORKERS}")



## === cell 3
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0
    seen = 0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu().item())
        correct += (logits.detach().argmax(dim=1) == yb).sum().item()
        seen += yb.numel()
        n_batches += 1

    model.eval()
    val_correct = 0
    val_seen = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_seen += yb.numel()
    model.train()

    train_acc = correct / max(seen, 1)
    val_acc = val_correct / max(val_seen, 1)
    print(
        f"Epoch {epoch}/{EPOCHS} - train_loss={total_loss/max(n_batches,1):.4f} "
        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
    )

if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()

test_df = pd.DataFrame({"image_id": test_ids})

test_loader = DataLoader(
    CassavaImageDataset(
        test_df,
        TEST_IMG_DIR,
        test_tfms,
        has_labels=False,
        enable_cache=_ENABLE_RAM_CACHE,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=_collate_test,
    **common_loader_kwargs,
)

model.eval()
all_ids = []
all_preds = []
with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
        all_preds.extend(preds)
        all_ids.extend(list(ids))

if all_ids != test_ids:
    id_to_pred = {i: p for i, p in zip(all_ids, all_preds)}
    all_ids = test_ids
    all_preds = [int(id_to_pred[i]) for i in test_ids]

assert len(all_ids) == len(
    test_ids
), f"Pred length mismatch: {len(all_ids)} vs {len(test_ids)}"
assert (
    all_ids[0] == test_ids[0] and all_ids[-1] == test_ids[-1]
), "Test ID order mismatch."

submission = pd.DataFrame({"image_id": all_ids, "label": all_preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
