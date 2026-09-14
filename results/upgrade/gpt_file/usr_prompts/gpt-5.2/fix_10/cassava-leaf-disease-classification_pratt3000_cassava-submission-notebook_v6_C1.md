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

3.9

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
import json
import math
import random
import struct
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as tvio

from tqdm.auto import tqdm


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




## === cell 1
def get_image(path: str) -> torch.Tensor:
    img = tvio.read_image(path, mode=tvio.ImageReadMode.RGB)  # uint8, CHW
    return img




## === cell 2
class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

        self.image_ids = df["image_id"].astype(str).values
        self.labels = None
        if output_label:
            self.labels = df["label"].astype(np.int64).values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        path = f"{self.data_root}/{self.image_ids[index]}"
        img = get_image(path)  # torch.uint8 CHW

        if self.transforms:
            img = self.transforms(img)

        if self.output_label:
            label = int(self.labels[index])
            return img, label
        else:
            return img




## === cell 3
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        try:
            return len(self.dl)
        except TypeError:
            raise TypeError(
                "Length is not defined for this DataLoader (IterableDataset)."
            )

    def try_len(self):
        try:
            return len(self.dl)
        except TypeError:
            return None


device = get_device()
print("device:", device)




## === cell 4
def accuracy(out, labels):
    preds = out.argmax(dim=1)
    return (preds == labels).float().mean()


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print(
            "Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["lrs"][-1] if len(result["lrs"]) else float("nan"),
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 5
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnext50_32x4d(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True




## === cell 6
MODEL_PATH = "../input/cassava-leaf-disease-detection/mod.pth"

model = None
if os.path.exists(MODEL_PATH):
    try:
        obj = torch.load(MODEL_PATH, map_location=device)
        if isinstance(obj, nn.Module):
            model = obj
        elif isinstance(obj, dict):
            tmp = Classifier()
            tmp.load_state_dict(obj, strict=False)
            model = tmp
        print("Loaded model from:", MODEL_PATH)
    except Exception as e:
        print(
            "Failed loading external model, falling back to fresh pretrained backbone. Error:",
            repr(e),
        )

if model is None:
    model = Classifier()
    print(
        "Using fallback model: ResNeXt50_32x4d pretrained backbone + 5-class head (will be trained briefly)."
    )

model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)



## === cell 7
BATCH_SIZE = 32  # keep identical
IMG_SIZE = 512
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_TFREC_DIR = "../input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "../input/cassava-leaf-disease-classification/test_tfrecords"

train_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df_full = pd.read_csv(TRAIN_CSV_PATH)

labels = train_df_full["label"].values
idxs = np.arange(len(train_df_full))
rng = np.random.RandomState(SEED)

train_idx = []
val_idx = []
val_frac = 0.1
for c in np.unique(labels):
    c_idx = idxs[labels == c]
    rng.shuffle(c_idx)
    n_val = max(1, int(len(c_idx) * val_frac))
    val_idx.extend(c_idx[:n_val].tolist())
    train_idx.extend(c_idx[n_val:].tolist())

train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

NUM_WORKERS = min(8, (os.cpu_count() or 2))


def _seed_worker(worker_id: int):
    seed = (SEED + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(SEED)


def _list_tfrecords(dir_path: str):
    if not os.path.exists(dir_path):
        return []
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tfrec")
    ]
    return sorted(files)


TRAIN_TFRECS = _list_tfrecords(TRAIN_TFREC_DIR)
USE_TFRECORDS = len(TRAIN_TFRECS) > 0

print("TFRecord train files:", len(TRAIN_TFRECS), "USE_TFRECORDS:", USE_TFRECORDS)




## === cell 8
class TFRecordMapDataset(Dataset):
    def __init__(
        self,
        tfrec_files: List[str],
        transforms=None,
        output_label: bool = True,
        shuffle_files: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.transforms = transforms
        self.output_label = output_label
        self.shuffle_files = shuffle_files
        self.seed = int(seed)

        import tensorflow as tf  # only for gfile + Example parsing (fast, no tf.data)

        self._tf = tf

        files = self.tfrec_files.copy()
        if self.shuffle_files:
            r = random.Random(self.seed)
            r.shuffle(files)
        self.tfrec_files = files

        self.index: List[Tuple[int, int]] = []  # (file_idx, record_offset)
        for fi, fp in enumerate(self.tfrec_files):
            self.index.extend([(fi, off) for off in self._build_offsets(fp)])

    def _build_offsets(self, fp: str) -> List[int]:
        tf = self._tf
        offsets = []
        with tf.io.gfile.GFile(fp, "rb") as f:
            off = 0
            while True:
                header = f.read(12)
                if len(header) < 12:
                    break
                (length,) = struct.unpack("<Q", header[:8])
                offsets.append(off)
                f.seek(int(length) + 4, 1)  # skip data + data_crc
                off = f.tell()
        return offsets

    def __len__(self):
        return len(self.index)

    def _read_record(self, fp: str, offset: int) -> bytes:
        tf = self._tf
        with tf.io.gfile.GFile(fp, "rb") as f:
            f.seek(offset)
            header = f.read(12)
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(int(length))
            _ = f.read(4)  # data crc
        return data

    def __getitem__(self, idx: int):
        tf = self._tf
        fi, off = self.index[idx]
        fp = self.tfrec_files[fi]
        rec = self._read_record(fp, off)

        ex = tf.train.Example.FromString(rec)
        feats = ex.features.feature

        img_bytes = feats["image"].bytes_list.value[0]
        if self.output_label:
            if "target" in feats:
                y = int(feats["target"].int64_list.value[0])
            else:
                y = int(feats["label"].int64_list.value[0])

        img = tvio.decode_jpeg(
            torch.frombuffer(img_bytes, dtype=torch.uint8), mode=tvio.ImageReadMode.RGB
        )

        if self.transforms:
            img = self.transforms(img)

        if self.output_label:
            return img, y
        return img


if USE_TFRECORDS:
    train_ds = TFRecordMapDataset(
        TRAIN_TFRECS,
        transforms=train_transforms,
        output_label=True,
        shuffle_files=True,  # preserves shuffling intent (previously ds.shuffle)
        seed=SEED,
    )
    val_ds = GetDataset(
        val_df, TRAIN_DIR, transforms=valid_transforms, output_label=True
    )
else:
    train_ds = GetDataset(
        train_df, TRAIN_DIR, transforms=train_transforms, output_label=True
    )
    val_ds = GetDataset(
        val_df, TRAIN_DIR, transforms=valid_transforms, output_label=True
    )

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)
val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

print(
    "train/val sizes:",
    len(train_df_full) if USE_TFRECORDS else len(train_df),
    len(val_df),
)
print("num_workers:", NUM_WORKERS, "train_num_workers:", NUM_WORKERS)




## === cell 9
def evaluate(model, val_loader):
    model.eval()
    n_batches = 0
    loss_sum = 0.0
    acc_sum = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            out = model.validation_step(batch)
            loss_sum += float(out["val_loss"].item())
            acc_sum += float(out["val_acc"].item())
            n_batches += 1
    return {
        "val_loss": loss_sum / max(1, n_batches),
        "val_acc": acc_sum / max(1, n_batches),
    }


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        lrs = []

        total = train_loader.try_len() if hasattr(train_loader, "try_len") else None
        for images, labels in tqdm(
            train_loader, total=total, desc=f"train epoch {epoch}"
        ):
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            loss = model.training_step((images, labels))
            train_loss_sum += float(loss.detach().item())
            n_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lrs.append(lr)

        result = evaluate(model, val_loader)
        result["train_loss"] = train_loss_sum / max(1, n_batches)
        result["lrs"] = lrs
        model.epoch_end(epoch, epochs, result)
        history.append(result)
    return history


model.freeze()
_ = fit(epochs=2, lr=1e-3, model=model, train_loader=train_dl, val_loader=val_dl)

model.unfreeze()
_ = fit(epochs=1, lr=1e-4, model=model, train_loader=train_dl, val_loader=val_dl)

model.eval()



## === cell 10
BATCH_SIZE_TEST = 128
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"

test_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TEST_TFRECS = _list_tfrecords(TEST_TFREC_DIR)
USE_TEST_TFRECORDS = len(TEST_TFRECS) > 0
print(
    "TFRecord test files:", len(TEST_TFRECS), "USE_TEST_TFRECORDS:", USE_TEST_TFRECORDS
)

if USE_TEST_TFRECORDS:
    sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
    test_csv = pd.read_csv(sample_path)[["image_id"]]

    test_ds = TFRecordMapDataset(
        TEST_TFRECS,
        transforms=test_transforms,
        output_label=False,
        shuffle_files=False,
        seed=SEED,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_TEST,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    )
else:
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    test_images = []
    for f in os.listdir(TEST_DIR):
        fp = os.path.join(TEST_DIR, f)
        if os.path.isfile(fp) and f.lower().endswith(valid_ext):
            test_images.append(f)
    test_images = sorted(test_images)

    test_csv = pd.DataFrame({"image_id": test_images})
    test_ds = GetDataset(
        test_csv,
        TEST_DIR,
        transforms=test_transforms,
        output_label=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_TEST,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    )

test_dl = DeviceDataLoader(test_dl, device)

print("test rows:", len(test_csv))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))




## === cell 11
def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    probs = []
    total = test_loader.try_len() if hasattr(test_loader, "try_len") else None
    tk0 = tqdm(test_loader, total=total, desc="infer")

    with torch.inference_mode():
        for images in tk0:
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            y_preds = model(images)
            batch_probs = y_preds.softmax(1).detach().cpu().numpy()  # (B,5)
            probs.append(batch_probs)

    probs = (
        np.concatenate(probs, axis=0)
        if len(probs)
        else np.zeros((0, 5), dtype=np.float32)
    )
    print("predictions shape:", probs.shape)
    return probs


predictions = inference(model, test_dl, device)



## === cell 12
test_csv = test_csv.copy()
test_csv["label"] = predictions.argmax(1).astype(int)

sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    pred_df = test_csv[["image_id", "label"]]
    merged = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
    merged["label"] = merged["label"].fillna(0).astype(int)
    submission = merged[["image_id", "label"]]
else:
    submission = test_csv[["image_id", "label"]]

sub_path = "./submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.head())
print("submission rows:", len(submission))
