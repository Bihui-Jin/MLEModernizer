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

0.40022

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.40022) has done: 'The timeout is dominated by the 5-fold cross-validation loop training *both* models for 3 epochs each (15 epochs per model over ~18k images), plus slow per-sample PIL decoding/augmentations in single-process loading. To keep the exact same training logic/epochs/models while making it fit in 600s, I (1) switch the data source to the provided TFRecords (same images/labels) to remove PIL/JPEG overhead, (2) keep the same torchvision-style augmentations but implement them in TensorFlow decode pipeline (equivalent transforms), and (3) maximize input throughput (parallel reads, caching, prefetch) while preserving determinism via fixed seeds and deterministic shuffle. I also keep AMP logic and avoid any changes to model architecture, losses, optimizers, schedulers, folds, or epoch counts; the training loop semantics remain identical.'

# 9. Code solution

## === cell 0
import os
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from PIL import Image, ImageDraw

from sklearn import model_selection
from sklearn.model_selection import KFold

import timm

import tensorflow as tf

Image.MAX_IMAGE_PIXELS = None



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
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.paths = dataframe["path"].values
        self.labels = dataframe["label"].values.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path_ = self.paths[index]
        label = int(self.labels[index])
        with open(path_, "rb") as f:
            image = Image.open(f).convert("RGB")
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
    torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)


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
_TFREC_DIR = os.path.join(path, "train_tfrecords")
_TFREC_FILES = sorted(
    [
        os.path.join(_TFREC_DIR, f)
        for f in os.listdir(_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
if not _TFREC_FILES:
    raise FileNotFoundError(f"No TFRecord files found under: {_TFREC_DIR}")

_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8 [H,W,3]
    label = tf.cast(ex["target"], tf.int64)
    return img, label


def _random_resized_crop(img, size, seed):
    img = tf.image.stateless_random_crop(
        img,
        size=[tf.shape(img)[0], tf.shape(img)[1], 3],
        seed=seed,  # no-op crop; we use sample_distorted_bounding_box below
    )
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], shape=[1, 1, 4], dtype=tf.float32)
    begin, crop_size, _ = tf.image.stateless_sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=bbox,
        seed=seed,
        min_object_covered=0.0,
        aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
        area_range=(0.08, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True,
    )
    cropped = tf.slice(img, begin, crop_size)
    cropped = tf.image.resize(cropped, [size, size], method="bilinear")
    return tf.cast(cropped, tf.uint8)


def _make_mask(img, p, mask_size, seed):
    r = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0)

    def _apply():
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        ms = tf.minimum(
            tf.cast(mask_size, tf.int32),
            tf.minimum(tf.maximum(1, w - 1), tf.maximum(1, h - 1)),
        )
        seeds = tf.random.experimental.stateless_split(seed, 11)
        xs = tf.random.stateless_uniform(
            [10], seed=seeds[1], minval=0, maxval=tf.maximum(1, w - ms), dtype=tf.int32
        )
        ys = tf.random.stateless_uniform(
            [10], seed=seeds[2], minval=0, maxval=tf.maximum(1, h - ms), dtype=tf.int32
        )

        yy = tf.range(h)[:, None]
        xx = tf.range(w)[None, :]
        yy = tf.cast(yy, tf.int32)
        xx = tf.cast(xx, tf.int32)

        x0 = xs[:, None, None]
        y0 = ys[:, None, None]
        inside = (
            (xx[None, :, :] >= x0)
            & (xx[None, :, :] < (x0 + ms))
            & (yy[None, :, :] >= y0)
            & (yy[None, :, :] < (y0 + ms))
        )
        inside_any = tf.reduce_any(inside, axis=0)  # [h,w]
        inside_any = inside_any[:, :, None]  # [h,w,1]
        zeros = tf.zeros_like(img)
        return tf.where(inside_any, zeros, img)

    return tf.cond(r < p, _apply, lambda: img)


def _train_augment(img, label, seed):
    seeds = tf.random.experimental.stateless_split(seed, 5)

    img = tf.image.stateless_random_flip_left_right(img, seed=seeds[0])
    img = tf.image.stateless_random_flip_up_down(img, seed=seeds[1])

    img = _random_resized_crop(img, image_size, seed=seeds[2])

    img = _make_mask(img, p=0.3, mask_size=50, seed=seeds[3])

    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = (img - _MEAN) / _STD
    return img, label


def _valid_preprocess(img, label):
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - _MEAN) / _STD
    return img, label


class _TorchFromTFRecordDataset(Dataset):
    def __init__(self, tfrec_files, length, training, seed=42):
        self.tfrec_files = list(tfrec_files)
        self.length = int(length)
        self.training = bool(training)
        self.seed = int(seed)

        options = tf.data.Options()
        options.experimental_deterministic = True

        ds = tf.data.TFRecordDataset(
            self.tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        ds = ds.with_options(options)
        ds = ds.map(_decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        if self.training:
            ds = ds.shuffle(8192, seed=self.seed, reshuffle_each_iteration=True)

            ds = ds.enumerate()

            def _aug(i, x):
                img, label = x
                s = tf.stack([tf.cast(self.seed, tf.int64), tf.cast(i, tf.int64)])
                img, label = _train_augment(img, label, seed=s)
                return img, label

            ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(_valid_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.prefetch(tf.data.AUTOTUNE)
        self._ds = ds

        raw_ds = tf.data.TFRecordDataset(
            self.tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
        ).with_options(options)
        raw_ds = raw_ds.take(self.length)
        self._serialized = [b.numpy() for b in raw_ds]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ex = tf.io.parse_single_example(self._serialized[int(idx)], _TFREC_FEATURES)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        label = tf.cast(ex["target"], tf.int64)

        if self.training:
            s = tf.stack([tf.cast(self.seed, tf.int64), tf.cast(idx, tf.int64)])
            img, label = _train_augment(img, label, seed=s)
        else:
            img, label = _valid_preprocess(img, label)

        img = torch.from_numpy(img.numpy()).permute(2, 0, 1).contiguous()
        return img, int(label.numpy())


dataset = _TorchFromTFRecordDataset(
    _TFREC_FILES, length=len(df), training=True, seed=42
)



## === cell 20
pass



## === cell 21
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 22
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)



## === cell 23
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)



## === cell 24
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)



## === cell 25
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)

criterion = nn.CrossEntropyLoss()




## === cell 26
class _EvalDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.paths = dataframe["path"].values
        self.labels = dataframe["label"].values.astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            im = Image.open(f).convert("RGB")
        im = self.transform(im)
        return im, int(self.labels[idx])


def calc_correction(model, df_):
    model.eval()
    eval_ds = _EvalDataset(df_, valid_transform)

    nw = min(4, os.cpu_count() or 2)
    persistent = bool(nw > 0)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=64,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=4 if nw > 0 else None,
    )
    correct = 0
    pred_counts = torch.zeros(5, dtype=torch.long)
    with torch.inference_mode():
        for data, target in eval_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(data)
            preds = out.argmax(1)
            correct += (preds == target).sum().item()
            pred_counts += torch.bincount(preds.detach().cpu(), minlength=5)
    percent = correct / len(eval_ds)
    return percent, pred_counts.tolist()




## === cell 27
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




## === cell 28
_KF_CACHE = {}


def _get_kfold_indices(n, n_splits=5, seed=42):
    key = (n, n_splits, seed)
    if key in _KF_CACHE:
        return _KF_CACHE[key]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = [(tr, va) for tr, va in kf.split(np.arange(n))]
    _KF_CACHE[key] = splits
    return splits


def train_model(
    model, dataset_, batch_size_, optimizer, criterion_, scheduler, epoch_, model_title
):
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    nw = min(8, os.cpu_count() or 2)
    persistent = bool(nw > 0)
    pin = torch.cuda.is_available()

    best_ckpt_path = model_title + ".best_tmp"
    splits = _get_kfold_indices(len(dataset_), n_splits=5, seed=42)

    for fold, (train_index, valid_index) in enumerate(splits):
        print("fold: ", fold)
        train_dataset = Subset(dataset_, train_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size_,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=4 if nw > 0 else None,
        )
        valid_dataset = Subset(dataset_, valid_index)

        valid_base = _TorchFromTFRecordDataset(
            _TFREC_FILES, length=len(df), training=False, seed=42
        )
        valid_dataset = Subset(valid_base, valid_index)

        valid_loader = DataLoader(
            valid_dataset,
            batch_size_,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=4 if nw > 0 else None,
        )

        for ep in range(1, epoch_ + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(data)
                    loss = criterion_(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_loader.sampler)
            train_losses.append(train_loss)

            model.eval()
            with torch.inference_mode():
                for data, target in valid_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output = model(data)
                        loss = criterion_(output, target)
                    pred = output.argmax(1) == target
                    acc.append((pred.float().mean()).item())
                    valid_loss += loss.item() * len(data)

            avg_valid_loss = valid_loss / len(valid_loader.sampler)
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                torch.save(model.state_dict(), best_ckpt_path)

            scheduler.step()

            collection = float(np.mean(acc)) if len(acc) else 0.0
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




## === cell 29
def train_models():
    global resNet, ef_model
    res_ckpt = "./res_model.pth"
    ef_ckpt = "./ef_model.pth"

    if os.path.exists(res_ckpt):
        resNet.load_state_dict(torch.load(res_ckpt, map_location=device))
        resNet = resNet.to(device).eval()
        print("Loaded existing resNet checkpoint; skipped training.")
    else:
        model_title = res_ckpt
        resNet, train_losses, valid_losses = train_model(
            resNet,
            dataset,
            batch_size,
            resNet_optimizer,
            criterion,
            resNet_scheduler,
            epoch,
            model_title,
        )
        print("resNet valid correction:", calc_correction(resNet, valid_df))
        plot_losses(epoch, "resNet losses", train_losses, valid_losses)

    if os.path.exists(ef_ckpt):
        ef_model.load_state_dict(torch.load(ef_ckpt, map_location=device))
        ef_model = ef_model.to(device).eval()
        print("Loaded existing efficientnet checkpoint; skipped training.")
    else:
        model_title = ef_ckpt
        ef_model, train_losses, valid_losses = train_model(
            ef_model,
            dataset,
            batch_size,
            ef_optimizer,
            criterion,
            ef_scheduler,
            epoch,
            model_title,
        )
        print("ef_model valid correction:", calc_correction(ef_model, valid_df))
        plot_losses(epoch, "ef losses", train_losses, valid_losses)




## === cell 30
train_models()



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 229) is killed by signal: Aborted. 

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4028810499.py in <cell line: 0>()
----> 1 train_models()
      2 

/tmp/ipykernel_55/4256372930.py in train_models()
     10     else:
     11         model_title = res_ckpt
---> 12         resNet, train_losses, valid_losses = train_model(
     13             resNet,
     14             dataset,

/tmp/ipykernel_55/1002717239.py in train_model(model, dataset_, batch_size_, optimizer, criterion_, scheduler, epoch_, model_title)
     66 
     67             model.train()
---> 68             for data, target in train_loader:
     69                 data = data.to(device, non_blocking=True)
     70                 target = target.to(device, non_blocking=True)

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
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 229) exited unexpectedly

## === cell 31
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))

ef_model = ef_model.to(device).eval()
resNet = resNet.to(device).eval()




## === cell 32
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




## === cell 33
classifier = CassaveClassifier(resNet, ef_model).to(device).eval()




## === cell 34
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
                pred = classifier.test(image, rate / 10).argmax(1).item()
                if pred == int(image_label):
                    count += 1
        percent = count / len(paths)
        print("rate: ", rate / 10, "percent: ", percent)




## === cell 35
pass



## === cell 36
test_images_dir = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 37
sample_sub = pd.read_csv(path + "/sample_submission.csv")
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(test_images_dir, fn) for fn in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test image files. Example: {missing[0]}"
    )




## === cell 38
class _TestDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            im = Image.open(f).convert("RGB")
        im = self.transform(im)
        return im


test_ds = _TestDataset(image_path, valid_transform)

nw = min(8, os.cpu_count() or 2)
persistent = bool(nw > 0)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent,
    prefetch_factor=4 if nw > 0 else None,
)

pred = []
classifier.eval()
with torch.inference_mode():
    for data in test_loader:
        data = data.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = classifier(data)
        preds = out.argmax(1).detach().cpu().tolist()
        pred.extend(int(p) for p in preds)

len(pred), len(image_id)



## === cell 39
pred[:10]



## === cell 40
sub = pd.DataFrame({"image_id": image_id, "label": np.asarray(pred, dtype=np.int64)})



## === cell 41
sub.head()



## === cell 42
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
print(sub.iloc[:3])
