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

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8915080084617709

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.15022) has done: 'I fix the model instantiation error by keeping ViT-B/16 at its pretrained-required `image_size=224` and aligning the preprocessing `img_size` to 224 so inference runs. Then I address the missing-predictions issue by filtering the test dataset file list to only valid image files (and matching the sample submission IDs), preventing mismatches caused by stray entries (e.g., nested folders) and ensuring every `image_id` gets a prediction. Finally, I make the TTA deterministic per-image index so results are stable under `cudnn.deterministic=True`, and the script always write a valid `submission.csv`.'
- What this solution (achieved 0.67564) has done: 'Your current score is low because the model head is randomly initialized and you never load any cassava-trained weights, so predictions are effectively near-random. To move the score toward the target with minimal change to the core logic, I add a standard training step on `train.csv` + `train_images` using the same ViT-B/16 architecture, CrossEntropy loss, and a simple train/val split to sanity-check accuracy. Then I run the same (existing) TTA inference on the test set, but using the trained weights, and write the same `submission.csv` format. I also ensure paths resolve in your provided filesystem and keep image size at 224 to match the pretrained backbone.'

# 9. Code solution

## === cell 0
import os
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision

try:
    import torchvision.io as tvio

    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

cudnn.deterministic = True
cudnn.benchmark = (
    True  # safe with fixed (C,H,W) each step; avoids slow kernel selection each time
)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

_CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
]
data_root = None
for r in _CANDIDATE_ROOTS:
    if os.path.isdir(r):
        data_root = r
        break
if data_root is None:
    raise FileNotFoundError(
        f"Could not find dataset root in candidates: {_CANDIDATE_ROOTS}"
    )
print("Using data_root:", data_root)

train_csv_path = os.path.join(data_root, "train.csv")
sample_path = os.path.join(data_root, "sample_submission.csv")
train_dir = os.path.join(data_root, "train_images")
test_dir = os.path.join(data_root, "test_images")

train_tfrec_dir = os.path.join(data_root, "train_tfrecords")
test_tfrec_dir = os.path.join(data_root, "test_tfrecords")

for p in [train_csv_path, sample_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")
for d in [train_dir, test_dir]:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing required directory: {d}")

img_size = 224
batch_size = 32

num_workers = min(8, os.cpu_count() or 4)

num_classes = 5

epochs = 10

lr = 3e-4
weight_decay = 0.05
label_smoothing = 0.1
grad_clip_norm = 1.0
val_frac = 0.1


def _seed_worker(worker_id: int):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_dl_generator = torch.Generator()
_dl_generator.manual_seed(seed)

_h2d_stream = torch.cuda.Stream() if device.type == "cuda" else None



## === cell 1
vit_model = torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
)
in_features = vit_model.heads.head.in_features
vit_model.heads.head = torch.nn.Linear(in_features, num_classes)
vit_model.to(device)

if device.type == "cuda":
    vit_model = vit_model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile") and device.type == "cuda":
    vit_model = torch.compile(vit_model, mode="reduce-overhead")



## === cell 2
import glob
import tensorflow as tf  # available on Kaggle images for this competition; used only for TFRecord parsing


_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}

_TEST_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_SPEC)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    return img, ex["target"], ex["image_id"]


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURE_SPEC)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    return img, ex["image_id"]


def _list_tfrecords(tfrec_dir, prefix):
    files = sorted(glob.glob(os.path.join(tfrec_dir, f"{prefix}*.tfrec")))
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found under {tfrec_dir} with prefix {prefix}"
        )
    return files


train_tfrecs = _list_tfrecords(train_tfrec_dir, "ld_train")
test_tfrecs = _list_tfrecords(test_tfrec_dir, "ld_test")


class CassavaTFRecordTrainDataset(IterableDataset):
    def __init__(self, tfrecord_files, transform, shuffle, seed, take_ids_set=None):
        super().__init__()
        self.tfrecord_files = list(tfrecord_files)
        self.transform = transform
        self.shuffle = shuffle
        self.seed = int(seed)
        self.take_ids_set = (
            take_ids_set  # set of image_id strings to include (train/val split)
        )

    def __iter__(self):
        ds = tf.data.TFRecordDataset(
            self.tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        if self.shuffle:
            ds = ds.shuffle(8192, seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.map(_parse_train_example, num_parallel_calls=tf.data.AUTOTUNE)

        for img, target, image_id in ds.as_numpy_iterator():
            if self.take_ids_set is not None:
                iid = image_id.decode("utf-8")
                if iid not in self.take_ids_set:
                    continue
            x = torch.from_numpy(img).permute(2, 0, 1)  # uint8
            y = int(target)
            if self.transform is not None:
                x = self.transform(x)
            yield x, y


class CassavaTFRecordTestDataset(IterableDataset):
    def __init__(self, tfrecord_files, transform, ttas, seed):
        super().__init__()
        self.tfrecord_files = list(tfrecord_files)
        self.transform = transform
        self.ttas = ttas or []
        self.seed = int(seed)

    def __iter__(self):
        ds = tf.data.TFRecordDataset(
            self.tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        ds = ds.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            torch.manual_seed(self.seed + worker_info.id)
            np.random.seed(self.seed + worker_info.id)
            random.seed(self.seed + worker_info.id)
        else:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        for img, image_id in ds.as_numpy_iterator():
            iid = image_id.decode("utf-8")
            x0 = torch.from_numpy(img).permute(2, 0, 1)  # uint8 CHW
            if len(self.ttas) == 0:
                views = [self.transform(x0)]
            else:
                views = [self.transform(t(x0)) for t in self.ttas]
            yield torch.stack(views, dim=0), iid




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(
            (img_size, img_size),
            scale=(0.7, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df = pd.read_csv(train_csv_path)
if list(train_df.columns) != ["image_id", "label"]:
    raise ValueError(f"Unexpected train.csv columns: {train_df.columns.tolist()}")

from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
train_idx, val_idx = next(splitter.split(train_df["image_id"], train_df["label"]))
df_tr = train_df.iloc[train_idx].reset_index(drop=True)
df_va = train_df.iloc[val_idx].reset_index(drop=True)

tr_ids = set(df_tr["image_id"].astype(str).tolist())
va_ids = set(df_va["image_id"].astype(str).tolist())

ds_tr = CassavaTFRecordTrainDataset(
    train_tfrecs,
    transform=train_transforms,
    shuffle=True,
    seed=seed,
    take_ids_set=tr_ids,
)
ds_va = CassavaTFRecordTrainDataset(
    train_tfrecs,
    transform=val_transforms,
    shuffle=False,
    seed=seed,
    take_ids_set=va_ids,
)

_prefetch = 4 if num_workers > 0 else None

dl_tr = DataLoader(
    ds_tr,
    batch_size=batch_size,
    shuffle=False,  # IterableDataset handles shuffle internally (same semantics: shuffled each epoch)
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
    generator=_dl_generator,
)
dl_va = DataLoader(
    ds_va,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
    generator=_dl_generator,
)



## === cell 4
criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(vit_model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=lr * 0.05
)


@torch.no_grad()
def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


vit_model.train()
for ep in range(1, epochs + 1):
    vit_model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    n_tr = 0

    for x, y in dl_tr:
        if device.type == "cuda":
            with torch.cuda.stream(_h2d_stream):
                x = x.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                y = y.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(_h2d_stream)
        else:
            x = x.to(device)
            y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = vit_model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vit_model.parameters(), grad_clip_norm)
        optimizer.step()

        bs = x.size(0)
        tr_loss += loss.item() * bs
        tr_acc += accuracy_from_logits(logits.detach(), y) * bs
        n_tr += bs

    vit_model.eval()
    va_loss = 0.0
    va_acc = 0.0
    n_va = 0
    with torch.no_grad():
        for x, y in dl_va:
            if device.type == "cuda":
                with torch.cuda.stream(_h2d_stream):
                    x = x.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                    y = y.to(device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(_h2d_stream)
            else:
                x = x.to(device)
                y = y.to(device)

            logits = vit_model(x)
            loss = criterion(logits, y)

            bs = x.size(0)
            va_loss += loss.item() * bs
            va_acc += accuracy_from_logits(logits, y) * bs
            n_va += bs

    scheduler.step()

    print(
        f"Epoch {ep}/{epochs} | "
        f"lr={scheduler.get_last_lr()[0]:.6g} | "
        f"train_loss={tr_loss/max(n_tr,1):.4f} train_acc={tr_acc/max(n_tr,1):.4f} | "
        f"val_loss={va_loss/max(n_va,1):.4f} val_acc={va_acc/max(n_va,1):.4f}"
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2371163434.py in <cell line: 0>()
     21     n_tr = 0
     22 
---> 23     for x, y in dl_tr:
     24         if device.type == "cuda":
     25             with torch.cuda.stream(_h2d_stream):

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

## === cell 5
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ttas = [
    v2.RandomResizedCrop(
        (img_size, img_size), scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC
    ),
    v2.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
    v2.RandomVerticalFlip(p=1.0),
    v2.RandomAffine(degrees=180, interpolation=InterpolationMode.BILINEAR),
    v2.RandomPerspective(
        distortion_scale=0.5, p=1.0, interpolation=InterpolationMode.BILINEAR
    ),
]

sample_sub = pd.read_csv(sample_path)
if list(sample_sub.columns) != ["image_id", "label"]:
    raise ValueError(
        f"Unexpected sample_submission columns: {sample_sub.columns.tolist()}"
    )

test_dataset = CassavaTFRecordTestDataset(
    test_tfrecs,
    transform=test_transforms,
    ttas=ttas,
    seed=seed,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
    generator=_dl_generator,
)



## === cell 6
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (tta_views, filenames) in enumerate(test_loader):
        if device.type == "cuda":
            with torch.cuda.stream(_h2d_stream):
                tta_views = tta_views.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(_h2d_stream)
        else:
            tta_views = tta_views.to(device)

        B, T, C, H, W = tta_views.shape
        inputs = tta_views.view(B * T, C, H, W)
        if device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        logits = vit_model(inputs)  # (B*T, num_classes)
        probs = torch.softmax(logits, dim=1).view(B, T, num_classes).mean(dim=1)
        pred_labels = probs.argmax(dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2877305131.py in <cell line: 0>()
      6 # WHY: Keep identical TTA math but reduce overhead by avoiding extra Python work and using non_blocking copies.
      7 with torch.no_grad():
----> 8     for batch_idx, (tta_views, filenames) in enumerate(test_loader):
      9         if device.type == "cuda":
     10             with torch.cuda.stream(_h2d_stream):

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

## === cell 7
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_sub[["image_id"]].merge(my_submission, on="image_id", how="left")
if my_submission["label"].isna().any():
    missing = (
        my_submission.loc[my_submission["label"].isna(), "image_id"].head(20).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some images (showing up to 20): {missing}. "
        f"Check test_dir={test_dir} and dataset file filtering."
    )

my_submission["label"] = my_submission["label"].astype(int)

assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(
    sample_sub
), f"Submission length {len(my_submission)} != expected {len(sample_sub)}"

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(my_submission), "rows")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2348146733.py in <cell line: 0>()
      6         my_submission.loc[my_submission["label"].isna(), "image_id"].head(20).tolist()
      7     )
----> 8     raise RuntimeError(
      9         f"Missing predictions for some images (showing up to 20): {missing}. "
     10         f"Check test_dir={test_dir} and dataset file filtering."

RuntimeError: Missing predictions for some images (showing up to 20): ['1234294272.jpg', '1234332763.jpg', '1234375577.jpg', '1234555380.jpg', '1234571117.jpg', '123464878.jpg', '1234924764.jpg', '1234931385.jpg', '1235142158.jpg', '12351712.jpg', '1235188286.jpg', '1236036550.jpg', '1236127980.jpg', '1236328100.jpg', '1236376745.jpg', '1236546384.jpg', '1236840149.jpg', '1236952675.jpg', '1237665250.jpg', '12377617.jpg']. Check test_dir=/kaggle/input/cassava-leaf-disease-classification/test_images and dataset file filtering.

## === cell 8
my_submission.head()
