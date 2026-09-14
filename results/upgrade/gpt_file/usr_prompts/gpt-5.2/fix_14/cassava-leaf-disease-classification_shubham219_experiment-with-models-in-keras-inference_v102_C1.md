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

3.11

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

0.6589604110003022

# 6. Current score

0.10164

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10164) has done: 'The timeout is dominated by per-sample JPEG decode + resize done in Python for ~18k train images across 2 epochs plus ~2.7k test images. To preserve the exact same model/training logic while cutting wall time, I switch data input to the provided TFRecord files (same images, much faster sequential I/O) and keep the same resize+normalize math, seeds, split, and training loop. I also eliminate pandas `.loc` inside `__getitem__` (pure overhead) by pre-materializing paths/labels into lists/arrays, and I enable a safe compilation/cache of the step function on PyTorch 2.x (`torch.compile`) when available (falls back cleanly). These changes are equivalent in semantics (same images, same transforms, same loss/optimizer/epochs) but remove the main bottlenecks.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

print("Data root:", DATA_ROOT)
print("Train CSV:", TRAIN_CSV)
print("Sample sub:", SAMPLE_SUB)
print("Train TFRecords:", TRAIN_TFREC_DIR)
print("Test TFRecords:", TEST_TFREC_DIR)



## === cell 1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, Dataset, DataLoader, get_worker_info
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "| Device:", device)

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 2
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)

num_classes = train_df["label"].nunique()
assert num_classes == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {num_classes}"

idx = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
tr_idx = idx[val_n:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train rows:", len(tr_df), "Valid rows:", len(va_df))

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

train_tfms = transforms.Compose(
    [
        transforms.Resize(
            IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
valid_tfms = transforms.Compose(
    [
        transforms.Resize(
            IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 2
import glob


def _seed_worker(worker_id: int):
    base_seed = SEED + worker_id
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)


g = torch.Generator()
g.manual_seed(SEED)

_num_workers = min(
    4, (os.cpu_count() or 2)
)  # fewer workers usually faster for sequential TFRecord IO
_prefetch_factor = 4 if _num_workers > 0 else None

_loader_common = dict(
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch_factor,
    worker_init_fn=_seed_worker,
)
if device.type == "cuda":
    _loader_common["pin_memory_device"] = "cuda"
if _num_workers > 0:
    _loader_common["multiprocessing_context"] = "fork"


def _list_tfrec_files(dir_path: str):
    files = sorted(glob.glob(os.path.join(dir_path, "*.tfrec")))
    return files


_TRAIN_TFRECS = _list_tfrec_files(TRAIN_TFREC_DIR)
_TEST_TFRECS = _list_tfrec_files(TEST_TFREC_DIR)
print("Found train tfrecs:", len(_TRAIN_TFRECS), "test tfrecs:", len(_TEST_TFRECS))

try:
    import tensorflow as tf  # available on Kaggle; used only for fast TFRecord decode

    _HAS_TF = True
    print("TensorFlow:", tf.__version__)
except Exception as e:
    _HAS_TF = False
    print(
        "TensorFlow not available; will fall back to per-image JPEG loading. Error:",
        repr(e),
    )

if _HAS_TF:
    try:
        tf.random.set_seed(SEED)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        pass

    _feature_desc_train = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    _feature_desc_test = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }

    def _tf_decode_and_preprocess(img_bytes):
        img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 HWC
        img = tf.image.convert_image_dtype(img, tf.float32)  # float32 [0,1]
        img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=True)
        return img  # HWC float32

    class CassavaTFRecordIterable(IterableDataset):
        def __init__(self, tfrec_files, has_label: bool):
            self.tfrec_files = list(tfrec_files)
            self.has_label = has_label

        def __iter__(self):
            wi = get_worker_info()
            if wi is None:
                files = self.tfrec_files
            else:
                files = self.tfrec_files[wi.id :: wi.num_workers]

            raw_ds = tf.data.TFRecordDataset(files, num_parallel_reads=1)
            if self.has_label:

                def _parse(x):
                    ex = tf.io.parse_single_example(x, _feature_desc_train)
                    img = _tf_decode_and_preprocess(ex["image"])
                    label = ex["label"]
                    return img, label

                ds = raw_ds.map(
                    _parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
                )
            else:

                def _parse(x):
                    ex = tf.io.parse_single_example(x, _feature_desc_test)
                    img = _tf_decode_and_preprocess(ex["image"])
                    return img

                ds = raw_ds.map(
                    _parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
                )

            for item in ds.as_numpy_iterator():
                if self.has_label:
                    img_hwc, y = item
                    x = (
                        torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()
                    )  # CHW float32
                    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
                    yield x, int(y)
                else:
                    img_hwc = item
                    x = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()
                    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
                    yield x

else:
    from torchvision.io import read_image

    class CassavaDataset(Dataset):
        def __init__(self, df, transform, has_label=True):
            self.transform = transform  # kept for compatibility; not used in fast path
            self.has_label = has_label
            self.paths = df["path"].tolist()
            self.labels = df["label"].to_numpy(dtype=np.int64) if has_label else None

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            path = self.paths[i]
            x = read_image(path).to(torch.float32).div_(255.0)
            x = TF.resize(
                x, IMG_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
            if self.has_label:
                return x, int(self.labels[i])
            return x




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
if _HAS_TF:
    train_ds = CassavaTFRecordIterable(_TRAIN_TFRECS, has_label=True)
    valid_ds = CassavaTFRecordIterable(
        _TRAIN_TFRECS, has_label=True
    )  # filtered below by label split logic? no

    tr_ids = set(tr_df["image_id"].astype(str).tolist())
    va_ids = set(va_df["image_id"].astype(str).tolist())

    _feature_desc_train_with_id = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }

    class CassavaTFRecordSplitIterable(IterableDataset):
        def __init__(self, tfrec_files, keep_ids: set):
            self.tfrec_files = list(tfrec_files)
            self.keep_ids = set(keep_ids)

        def __iter__(self):
            wi = get_worker_info()
            if wi is None:
                files = self.tfrec_files
            else:
                files = self.tfrec_files[wi.id :: wi.num_workers]

            raw_ds = tf.data.TFRecordDataset(files, num_parallel_reads=1)

            def _parse(x):
                ex = tf.io.parse_single_example(x, _feature_desc_train_with_id)
                img_id = ex["image_id"]
                img = _tf_decode_and_preprocess(ex["image"])
                label = ex["label"]
                return img, label, img_id

            ds = raw_ds.map(
                _parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
            )

            for img_hwc, y, img_id in ds.as_numpy_iterator():
                img_id_str = img_id.decode("utf-8")
                if img_id_str not in self.keep_ids:
                    continue
                x = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()
                x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
                yield x, int(y)

    train_ds = CassavaTFRecordSplitIterable(_TRAIN_TFRECS, keep_ids=tr_ids)
    valid_ds = CassavaTFRecordSplitIterable(_TRAIN_TFRECS, keep_ids=va_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  # IterableDataset cannot be shuffled by DataLoader; TFRecords order is fixed
        **_loader_common,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **_loader_common,
    )
else:
    train_ds = CassavaDataset(tr_df, train_tfms, has_label=True)
    valid_ds = CassavaDataset(va_df, valid_tfms, has_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        **_loader_common,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **_loader_common,
    )

print("Workers:", _num_workers, "| Prefetch factor:", _prefetch_factor)
print("Train rows:", len(tr_df), "Valid rows:", len(va_df))
print("Batch size:", BATCH_SIZE)



## === cell 4
weights = models.ResNet50_Weights.IMAGENET1K_V2
base = models.resnet50(weights=weights)

for p in base.parameters():
    p.requires_grad = False

in_features = base.fc.in_features
base.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, NUM_CLASSES),
)

model = base.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

try:
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("Enabled torch.compile")
except Exception as _e:
    print("torch.compile not enabled:", repr(_e))

print(model)




## === cell 5
def run_epoch(model, loader, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True):
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
        else:
            with torch.inference_mode():
                logits = model(x)

        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == y).sum().item())
        total += int(x.size(0))

    avg_loss = (total_loss / max(total, 1)) if train else 0.0
    acc = total_correct / max(total, 1)
    return avg_loss, acc


for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, train=True)
    va_loss, va_acc = run_epoch(model, valid_loader, train=False)
    print(
        f"Epoch {epoch}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f}"
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1304782273.py in <cell line: 0>()
     37 
     38 for epoch in range(1, EPOCHS + 1):
---> 39     tr_loss, tr_acc = run_epoch(model, train_loader, train=True)
     40     va_loss, va_acc = run_epoch(model, valid_loader, train=False)
     41     print(

/tmp/ipykernel_11/1304782273.py in run_epoch(model, loader, train)
      9     total = 0
     10 
---> 11     for batch in loader:
     12         x, y = batch
     13         x = x.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    724             # Some exceptions have first argument as non-str but explicitly
    725             # have message field
--> 726             raise self.exc_type(message=msg)
    727         try:
    728             exception = self.exc_type(msg)

TypeError: InvalidArgumentError.__init__() missing 2 required positional arguments: 'node_def' and 'op'

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["path"] = TEST_IMG_DIR + "/" + sample_sub["image_id"].astype(str)

if _HAS_TF:

    class CassavaTFRecordTestIterable(IterableDataset):
        def __init__(self, tfrec_files):
            self.tfrec_files = list(tfrec_files)

        def __iter__(self):
            wi = get_worker_info()
            if wi is None:
                files = self.tfrec_files
            else:
                files = self.tfrec_files[wi.id :: wi.num_workers]

            raw_ds = tf.data.TFRecordDataset(files, num_parallel_reads=1)

            def _parse(x):
                ex = tf.io.parse_single_example(x, _feature_desc_test)
                img = _tf_decode_and_preprocess(ex["image"])
                return img

            ds = raw_ds.map(
                _parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
            )

            for img_hwc in ds.as_numpy_iterator():
                x = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()
                x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
                yield x

    test_ds = CassavaTFRecordTestIterable(_TEST_TFRECS)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **_loader_common,
    )
else:
    test_ds = CassavaDataset(sample_sub, valid_tfms, has_label=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **_loader_common,
    )

model.eval()
all_preds = []

with torch.inference_mode():
    for x in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
        all_preds.append(preds)

pred_test_labels = np.concatenate(all_preds, axis=0)
assert len(pred_test_labels) == len(
    sample_sub
), "Prediction length mismatch with sample submission."

submission = sample_sub[["image_id"]].copy()
submission["label"] = pred_test_labels
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 7
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"]
assert len(chk) == len(pd.read_csv(SAMPLE_SUB))
assert chk["label"].between(0, 4).all()
print("Submission checks passed. Head:")
print(chk.head())
