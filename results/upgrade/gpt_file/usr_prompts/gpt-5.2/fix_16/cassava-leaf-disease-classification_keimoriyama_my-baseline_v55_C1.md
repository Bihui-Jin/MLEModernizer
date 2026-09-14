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

# 5. Target score

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.40022) has done: 'The timeout is dominated by the 5-fold cross-validation loop training *both* models for 3 epochs each (15 epochs per model over ~18k images), plus slow per-sample PIL decoding/augmentations in single-process loading. To keep the exact same training logic/epochs/models while making it fit in 600s, I (1) switch the data source to the provided TFRecords (same images/labels) to remove PIL/JPEG overhead, (2) keep the same torchvision-style augmentations but implement them in TensorFlow decode pipeline (equivalent transforms), and (3) maximize input throughput (parallel reads, caching, prefetch) while preserving determinism via fixed seeds and deterministic shuffle. I also keep AMP logic and avoid any changes to model architecture, losses, optimizers, schedulers, folds, or epoch counts; the training loop semantics remain identical.'
- What this solution (achieved 0.88117) has done: 'The timeout is dominated by doing 5-fold cross-validation for 3 epochs on two large models (ResNet50 + EfficientNet-B2), which multiplies the training cost by ~10x and is unlikely to finish in 600s. To preserve the core training loop and model logic while cutting redundant work, the refactor keeps the exact same dataset/augmentations/optimizers/schedulers/loss, but trains each model on the single existing train/valid split already created (cells 5–9) instead of additionally looping over 5 folds. In addition, it removes compile overhead (which can be slower for short trainings), enables fast DataLoader settings consistently (persistent workers, pinned memory, channels_last), and avoids repeatedly rebuilding validation datasets/loaders. These changes are equivalence-preserving w.r.t. the algorithm (same models, same epochs, same objective), but eliminate the major multiplicative cost that causes the timeout.'
- What this solution (achieved 0.40022) has done: 'The timeout is dominated by repeatedly decoding large JPEGs at 512px for two separate 3‑epoch trainings (resnet50 + efficientnet_b2) with heavy CPU-side augmentation, plus redundant validation dataset construction inside `calc_correction`. I keep the exact same models, transforms, epochs, losses, and training loops, but reduce overhead by (1) using the provided TFRecords instead of per-file JPEG opens (equivalent image content, far faster input pipeline), (2) enabling `torch.backends.cudnn.benchmark=True` (safe here because input shape is fixed at 512×512), and (3) reusing a single cached validation loader/dataset for corrections to avoid re-decoding images. I also tune dataloader knobs for throughput (more workers where available, persistent workers, larger prefetch) without changing batch size, steps, or math. All changes are deterministic (explicit seeds) and do not change evaluation semantics beyond negligible floating-point differences.'

# 9. Code solution

## === cell 0
import os
import random
import time
import glob
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw

from sklearn import model_selection

import timm

import tensorflow as tf

Image.MAX_IMAGE_PIXELS = None

_HAS_TORCH_COMPILE = hasattr(torch, "compile")

os.environ.setdefault("PYTHONHASHSEED", "42")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

torch.backends.cudnn.benchmark = True

_CPU_COUNT = os.cpu_count() or 2
_DATALOADER_WORKERS = min(8, max(2, _CPU_COUNT // 2))
_DATALOADER_PREFETCH = 4
_PERSISTENT_WORKERS = True

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _CPU_COUNT // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, _CPU_COUNT // 4))
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)[:10]




## === cell 2
df = pd.read_csv(path + "/train.csv")




## === cell 3
df.head()




## === cell 4
df["path"] = path + "/train_images/" + df["image_id"].astype(str)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)




## === cell 5
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)




## === cell 6
try:
    ax = train_df.label.value_counts().plot(kind="bar", title="train label counts")
except Exception:
    pass




## === cell 7
try:
    ax = valid_df.label.value_counts().plot(kind="bar", title="valid label counts")
except Exception:
    pass




## === cell 8
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()




## === cell 9
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()




## === cell 10
im = Image.open(train_df["path"][0])
im.size




## === cell 11
try:
    im
except Exception:
    pass




## === cell 12
import matplotlib.image as img  # noqa: F401




## === cell 13
pass




## === cell 14
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None, cache_images=False):
        super().__init__()
        self.paths = dataframe["path"].values
        self.labels = dataframe["label"].values.astype(np.int64)
        self.transform = transform
        self.cache_images = bool(cache_images)
        self._cache = {} if self.cache_images else None

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path_):
        if self._cache is None:
            return Image.open(path_).convert("RGB")
        im = self._cache.get(path_)
        if im is None:
            src = Image.open(path_).convert("RGB")
            self._cache[path_] = src.copy()
            src.close()
            im = self._cache[path_]
        return im.copy()

    def __getitem__(self, index):
        path_ = self.paths[index]
        label = int(self.labels[index])
        image = self._load_image(path_)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 15
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)




## === cell 16
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            ms = min(self.mask_size, max(1, width - 1), max(1, height - 1))
            for _ in range(10):
                start_width.append(random.randrange(0, max(1, width - ms)))
                start_height.append(random.randrange(0, max(1, height - ms)))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + ms, y + ms), fill=(0, 0, 0), outline=(0, 0, 0)
                )
        return image




## === cell 17
image_size = 512
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.3, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 18
pass




## === cell 19
_TFREC_TRAIN_GLOB = os.path.join(path, "train_tfrecords", "*.tfrec")
_TFREC_TEST_GLOB = os.path.join(path, "test_tfrecords", "*.tfrec")

train_tfrecs = sorted(glob.glob(_TFREC_TRAIN_GLOB))
test_tfrecs = sorted(glob.glob(_TFREC_TEST_GLOB))

if not train_tfrecs:
    raise FileNotFoundError(f"No train tfrecords found at: {_TFREC_TRAIN_GLOB}")
if not test_tfrecs:
    raise FileNotFoundError(f"No test tfrecords found at: {_TFREC_TEST_GLOB}")

_train_paths_set = set(train_df["path"].values.tolist())
_valid_paths_set = set(valid_df["path"].values.tolist())
_train_ids_set = set(os.path.basename(p) for p in _train_paths_set)
_valid_ids_set = set(os.path.basename(p) for p in _valid_paths_set)

_label_by_id = {}
for pth, lab in zip(df["path"].values.tolist(), df["label"].values.tolist()):
    _label_by_id[os.path.basename(pth)] = int(lab)

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    return ex["image"], ex["image_name"], ex["label"]


def _build_tfrecord_index(tfrecs):
    return tf.data.TFRecordDataset(tfrecs, num_parallel_reads=tf.data.AUTOTUNE)


class _TFRecordCassavaIterable(Dataset):
    def __init__(
        self,
        tfrecs,
        id_set,
        transform,
        repeat_for_epochs=1,
        shuffle_buffer=2048,
        is_train=False,
    ):
        super().__init__()
        self.tfrecs = list(tfrecs)
        self.id_set = set(id_set)
        self.transform = transform
        self.repeat_for_epochs = int(repeat_for_epochs)
        self.shuffle_buffer = int(shuffle_buffer)
        self.is_train = bool(is_train)

    def __iter__(self):
        ds = _build_tfrecord_index(self.tfrecs)
        if self.is_train:
            ds = ds.shuffle(self.shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat(self.repeat_for_epochs)
        for img_bytes, img_name, label in ds:
            name = img_name.numpy().decode("utf-8")
            if name not in self.id_set:
                continue
            b = img_bytes.numpy()
            im = Image.open(tf.io.gfile.GFile(io.BytesIO(b), "rb")).convert("RGB")

            if self.transform is not None:
                im = self.transform(im)

            yield im, int(label.numpy())

    def __len__(self):
        return len(self.id_set)


import io  # placed here to avoid changing earlier cell structure




## === cell 20
train_dataset_full = _TFRecordCassavaIterable(
    train_tfrecs,
    _train_ids_set,
    transform=train_transform,
    repeat_for_epochs=1,
    shuffle_buffer=4096,
    is_train=True,
)
valid_dataset_full = _TFRecordCassavaIterable(
    train_tfrecs,
    _valid_ids_set,
    transform=valid_transform,
    repeat_for_epochs=1,
    shuffle_buffer=1,
    is_train=False,
)




## === cell 21
pass




## === cell 22
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device




## === cell 23
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)




## === cell 24
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)




## === cell 25
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)




## === cell 26
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)

criterion = nn.CrossEntropyLoss()




## === cell 27
_eval_cache = {}


def _get_eval_loader(ds_key, dataset_obj):
    if ds_key in _eval_cache:
        return _eval_cache[ds_key]
    is_iterable = (
        isinstance(dataset_obj, torch.utils.data.IterableDataset)
        or hasattr(dataset_obj, "__iter__")
        and not hasattr(dataset_obj, "__getitem__")
    )
    nw = 0 if is_iterable else (_DATALOADER_WORKERS if _PERSISTENT_WORKERS else 0)
    eval_loader = DataLoader(
        dataset_obj,
        batch_size=64,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    _eval_cache[ds_key] = (dataset_obj, eval_loader)
    return _eval_cache[ds_key]


def calc_correction(model, valid_dataset_obj):
    model.eval()
    eval_ds, eval_loader = _get_eval_loader("valid", valid_dataset_obj)
    correct = 0
    pred_counts = torch.zeros(5, dtype=torch.long)
    with torch.inference_mode():
        for data, target in eval_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if torch.cuda.is_available():
                data = data.contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(data)
            preds = out.argmax(1)
            correct += (preds == target).sum().item()
            pred_counts += torch.bincount(preds.detach().cpu(), minlength=5)
    percent = correct / len(eval_ds)
    return percent, pred_counts.tolist()




## === cell 28
from matplotlib import pyplot as plt


def plot_losses(epoch_, title, train_losses, valid_losses):
    try:
        y = list(range(len(train_losses)))
        train_loss = plt.plot(y, train_losses)
        valid_loss = plt.plot(y, valid_losses)
        plt.title(title)
        plt.ylabel("loss")
        plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
        plt.show()
    except Exception:
        pass




## === cell 29
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _maybe_compile(model):
    if _HAS_TORCH_COMPILE and torch.cuda.is_available():
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception:
            return model
    return model


def train_model(
    model,
    train_ds,
    valid_ds,
    batch_size_,
    optimizer,
    criterion_,
    scheduler,
    epoch_,
    model_title,
):
    train_losses, valid_losses = [], []

    is_train_iterable = isinstance(train_ds, torch.utils.data.IterableDataset) or (
        hasattr(train_ds, "__iter__") and not hasattr(train_ds, "__getitem__")
    )
    is_valid_iterable = isinstance(valid_ds, torch.utils.data.IterableDataset) or (
        hasattr(valid_ds, "__iter__") and not hasattr(valid_ds, "__getitem__")
    )

    nw_train = 0 if is_train_iterable else _DATALOADER_WORKERS
    nw_valid = 0 if is_valid_iterable else _DATALOADER_WORKERS
    pin = torch.cuda.is_available()

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_,
        shuffle=not is_train_iterable,
        generator=g if (not is_train_iterable) else None,
        num_workers=nw_train,
        pin_memory=pin,
        persistent_workers=_PERSISTENT_WORKERS and (nw_train > 0),
        prefetch_factor=_DATALOADER_PREFETCH if (nw_train > 0) else None,
        worker_init_fn=_seed_worker if (nw_train > 0) else None,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size_,
        shuffle=False,
        num_workers=nw_valid,
        pin_memory=pin,
        persistent_workers=_PERSISTENT_WORKERS and (nw_valid > 0),
        prefetch_factor=_DATALOADER_PREFETCH if (nw_valid > 0) else None,
        worker_init_fn=_seed_worker if (nw_valid > 0) else None,
    )

    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch = -1
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = 0

    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    model = _maybe_compile(model)

    best_ckpt_path = model_title + ".best_tmp"
    if os.path.exists(best_ckpt_path):
        try:
            os.remove(best_ckpt_path)
        except Exception:
            pass

    best_loss_global = float("inf")

    for ep in range(1, epoch_ + 1):
        epoch_start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        seen_train = 0
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if torch.cuda.is_available():
                data = data.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(data)
                loss = criterion_(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = target.numel()
            seen_train += bs
            train_loss += loss.item() * bs

            if is_train_iterable and seen_train >= len(train_ds):
                break

        train_loss = train_loss / len(train_ds)
        train_losses.append(train_loss)

        correct = 0
        total = 0
        model.eval()
        seen_valid = 0
        with torch.inference_mode():
            for data, target in valid_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if torch.cuda.is_available():
                    data = data.contiguous(memory_format=torch.channels_last)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(data)
                    loss = criterion_(output, target)

                preds = output.argmax(1)
                correct += (preds == target).sum().item()
                total += target.numel()

                bs = target.numel()
                seen_valid += bs
                valid_loss += loss.item() * bs

                if is_valid_iterable and seen_valid >= len(valid_ds):
                    break

        avg_valid_loss = valid_loss / len(valid_ds)

        if avg_valid_loss < best_loss_global:
            best_loss_global = avg_valid_loss
            torch.save(model.state_dict(), best_ckpt_path)

        scheduler.step()

        collection = (correct / total) if total else 0.0
        valid_losses.append(avg_valid_loss)
        print(
            "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                time.time() - epoch_start_time,
                ep,
                train_loss,
                avg_valid_loss,
                collection,
            )
        )

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(best_state)
        torch.save(best_state, model_title)
        try:
            os.remove(best_ckpt_path)
        except Exception:
            pass
    else:
        torch.save(model.state_dict(), model_title)

    return model, train_losses, valid_losses




## === cell 30
def train_models():
    global resNet, ef_model
    res_ckpt = "./res_model.pth"
    ef_ckpt = "./ef_model.pth"

    if os.path.exists(res_ckpt):
        resNet.load_state_dict(torch.load(res_ckpt, map_location=device))
        resNet = resNet.to(device).eval()
        if torch.cuda.is_available():
            resNet = resNet.to(memory_format=torch.channels_last)
        print("Loaded existing resNet checkpoint; skipped training.")
    else:
        model_title = res_ckpt
        resNet_trained, train_losses, valid_losses = train_model(
            resNet,
            train_dataset_full,
            valid_dataset_full,
            batch_size,
            resNet_optimizer,
            criterion,
            resNet_scheduler,
            epoch,
            model_title,
        )
        resNet = resNet_trained
        print("resNet valid correction:", calc_correction(resNet, valid_dataset_full))
        plot_losses(epoch, "resNet losses", train_losses, valid_losses)

    if os.path.exists(ef_ckpt):
        ef_model.load_state_dict(torch.load(ef_ckpt, map_location=device))
        ef_model = ef_model.to(device).eval()
        if torch.cuda.is_available():
            ef_model = ef_model.to(memory_format=torch.channels_last)
        print("Loaded existing efficientnet checkpoint; skipped training.")
    else:
        model_title = ef_ckpt
        ef_trained, train_losses, valid_losses = train_model(
            ef_model,
            train_dataset_full,
            valid_dataset_full,
            batch_size,
            ef_optimizer,
            criterion,
            ef_scheduler,
            epoch,
            model_title,
        )
        ef_model = ef_trained
        print(
            "ef_model valid correction:", calc_correction(ef_model, valid_dataset_full)
        )
        plot_losses(epoch, "ef losses", train_losses, valid_losses)




## === cell 31
train_models()




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_55/3028619008.py in <cell line: 0>()
----> 1 train_models()
      2 
      3 

/tmp/ipykernel_55/4197112828.py in train_models()
     12     else:
     13         model_title = res_ckpt
---> 14         resNet_trained, train_losses, valid_losses = train_model(
     15             resNet,
     16             train_dataset_full,

/tmp/ipykernel_55/684019734.py in train_model(model, train_ds, valid_ds, batch_size_, optimizer, criterion_, scheduler, epoch_, model_title)
     90         model.train()
     91         seen_train = 0
---> 92         for data, target in train_loader:
     93             data = data.to(device, non_blocking=True)
     94             target = target.to(device, non_blocking=True)

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
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

NotImplementedError: Caught NotImplementedError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py", line 63, in __getitem__
    raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
NotImplementedError: Subclasses of Dataset should implement __getitem__.


## === cell 32
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))

ef_model = ef_model.to(device).eval()
resNet = resNet.to(device).eval()
if torch.cuda.is_available():
    ef_model = ef_model.to(memory_format=torch.channels_last)
    resNet = resNet.to(memory_format=torch.channels_last)




## === cell 33
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.3 * x1 + 0.7 * x2

    def test(self, x, rate):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        p = rate * x1 + (1 - rate) * x2
        return p




## === cell 34
classifier = CassaveClassifier(resNet, ef_model).to(device).eval()
if torch.cuda.is_available():
    classifier = classifier.to(memory_format=torch.channels_last)

classifier = _maybe_compile(classifier)




## === cell 35
def test_rate():
    classifier.eval()
    for rate in range(1, 10):
        paths = valid_df["path"].values
        labels = valid_df["label"].values.astype(int)
        count = 0
        with torch.inference_mode():
            for image_path, image_label in zip(paths, labels):
                image = Image.open(image_path).convert("RGB")
                image = valid_transform(image).unsqueeze(0).to(device)
                if torch.cuda.is_available():
                    image = image.contiguous(memory_format=torch.channels_last)
                pred = classifier.test(image, rate / 10).argmax(1).item()
                if pred == int(image_label):
                    count += 1
        percent = count / len(paths)
        print("rate: ", rate / 10, "percent: ", percent)




## === cell 36
pass




## === cell 37
test_images_dir = "../input/cassava-leaf-disease-classification/test_images/"




## === cell 38
sample_sub = pd.read_csv(path + "/sample_submission.csv")
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(test_images_dir, fn) for fn in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test image files. Example: {missing[0]}"
    )




## === cell 39
_TEST_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_TFREC_FEATURES)
    return ex["image"], ex["image_name"]


class _TFRecordTestIterable(Dataset):
    def __init__(self, tfrecs, transform, name_order):
        super().__init__()
        self.tfrecs = list(tfrecs)
        self.transform = transform
        self.name_order = list(name_order)
        self._order_set = set(self.name_order)

    def __iter__(self):
        ds = _build_tfrecord_index(self.tfrecs).map(
            _parse_test_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )
        for img_bytes, img_name in ds:
            name = img_name.numpy().decode("utf-8")
            if name not in self._order_set:
                continue
            b = img_bytes.numpy()
            im = Image.open(tf.io.gfile.GFile(io.BytesIO(b), "rb")).convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
            yield name, im

    def __len__(self):
        return len(self.name_order)


test_ds = _TFRecordTestIterable(test_tfrecs, valid_transform, image_id)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

pred_by_name = {}
classifier.eval()
with torch.inference_mode():
    for batch in test_loader:
        names, data = batch
        data = data.to(device, non_blocking=True)
        if torch.cuda.is_available():
            data = data.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = classifier(data)
        preds = out.argmax(1).detach().cpu().numpy().astype(np.int64)
        for n, p in zip(list(names), preds.tolist()):
            pred_by_name[str(n)] = int(p)

pred = [pred_by_name[iid] for iid in image_id]
len(pred), len(image_id)




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_55/3849318713.py in <cell line: 0>()
     53 classifier.eval()
     54 with torch.inference_mode():
---> 55     for batch in test_loader:
     56         names, data = batch
     57         data = data.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py in __getitem__(self, index)
     61 
     62     def __getitem__(self, index) -> _T_co:
---> 63         raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
     64 
     65     # def __getitems__(self, indices: List) -> List[_T_co]:

NotImplementedError: Subclasses of Dataset should implement __getitem__.

## === cell 40
pred[:10]




## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/608936516.py in <cell line: 0>()
----> 1 pred[:10]
      2 
      3 

NameError: name 'pred' is not defined

## === cell 41
sub = pd.DataFrame({"image_id": image_id, "label": np.asarray(pred, dtype=np.int64)})




## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1353685709.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"image_id": image_id, "label": np.asarray(pred, dtype=np.int64)})
      2 
      3 

NameError: name 'pred' is not defined

## === cell 42
sub.head()




## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/962398646.py in <cell line: 0>()
----> 1 sub.head()
      2 
      3 

NameError: name 'sub' is not defined

## === cell 43
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
print(sub.iloc[:3])

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3374770408.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print("submission.csv columns:", list(sub.columns))
      4 print(sub.iloc[:3])

NameError: name 'sub' is not defined
