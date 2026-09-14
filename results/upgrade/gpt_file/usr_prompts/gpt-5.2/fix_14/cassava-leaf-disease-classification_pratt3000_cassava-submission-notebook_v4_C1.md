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

0.8822907222725899

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09492) has done: 'I fix the early crash caused by the notebook `tqdm.notebook` import (it triggers a protobuf `MessageFactory` error in Kaggle) by switching to standard `tqdm`, and I remove the directory-walk printing that isn’t needed. I also make the imports and cell ordering consistent so `torch`, `Dataset`, `nn`, and `T` are defined before use, which resolves the cascade of `NameError`s. The provided model path points to a different/non-existent dataset, so I keep the same core inference pipeline but instantiate the defined WideResNet model with ImageNet weights (no training) to ensure end-to-end execution and a valid `submission.csv`. Finally, I fix inference concatenation/argmax shape logic and ensure predictions align with `sample_submission.csv` ordering.'
- What this solution (achieved 0.77018) has done: 'Your current score is low because the model is a pretrained ImageNet WideResNet with a randomly initialized 5-class head that is never trained, so predictions are essentially random. To move the accuracy toward the target, the smallest change that preserves your core architecture and loss is to actually train only the final `fc` layer on `train.csv` for a small number of epochs, then run the same inference code. I keep your dataset class, transforms, model definition, and cross-entropy objective; I add a simple train/valid split plus a standard PyTorch training loop (no early stopping or approximations). I also match transforms between train/valid/test and ensure the submission stays aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.71076) has done: 'Your gap to target is ~0.112 (0.77018 → 0.88229), so we should improve accuracy with the smallest changes that keep your architecture and training semantics intact. The main issue is you only train the randomly initialized 5-class head for 3 epochs at 512px, which underfits; we keep “train only fc” but (1) use a more standard ImageNet input size (224) to let the pretrained backbone features transfer better, (2) train the fc head longer with a slightly lower LR, and (3) add lightweight label smoothing in the same cross-entropy objective to stabilize generalization. We also make runs deterministic and ensure the submission remains aligned to `sample_submission.csv` order and format. These are minimal, safe tweaks that should move the score upward toward your target without changing the core model/training approach.'
- What this solution (achieved 0.71114) has done: 'Your score is far below the target (0.71076 vs 0.88229), so we should improve generalization with minimal changes while keeping the same model (WideResNet101_2), same “train only fc” approach, and same cross-entropy objective. The smallest high-impact fix is to correct the training augmentation: currently you only do horizontal flip, which is too weak; adding standard small geometric and color jitter augmentations (still at 224 and same normalization) typically improves transfer performance without changing core logic. I also add a simple LR schedule (CosineAnnealingLR) while keeping the same optimizer and number of epochs; this usually improves convergence of the fc head without changing semantics or adding early stopping. Finally, I make DataLoader workers/persistent settings slightly more stable and keep the submission writing exactly the same.'
- What this solution (achieved 0.72123) has done: 'Your current score (0.71114) is well below the target (0.88229), so we should improve accuracy with small, safe changes that keep your exact core model (WideResNet101_2), “train only fc” approach, and cross-entropy objective. The biggest likely issue is distribution shift from applying augmentations only during training: we use simple test-time augmentation (TTA) by running inference on a horizontally flipped version too and averaging probabilities, which often gives a meaningful bump without changing training. We also switch `Resize` to `RandomResizedCrop` (train) and `Resize+CenterCrop` (valid/test) to better match ImageNet-style preprocessing while keeping the same 224 resolution and normalization. Finally, we keep everything deterministic and preserve your submission alignment to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import math
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # fixed image size => faster

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TEST_DIR), f"Missing test directory: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing train directory: {TRAIN_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

_HAS_TFRECORDS = os.path.isdir(TRAIN_TFREC_DIR) and os.path.isdir(TEST_TFREC_DIR)
if not _HAS_TFRECORDS:
    raise RuntimeError(
        "Expected TFRecords directories to exist for fast I/O, but they were not found."
    )




## === cell 1
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
device




## === cell 2
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


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
            "Epoch: [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 3
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.wide_resnet101_2(pretrained=True)
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




## === cell 4
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert "image_id" in train_df.columns and "label" in train_df.columns
train_df["label"] = train_df["label"].astype(int)

perm = np.random.RandomState(42).permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

BATCH_SIZE = 64
_cpu = os.cpu_count() or 2
num_workers = min(4, max(2, _cpu // 2))

import tensorflow as tf

tf.random.set_seed(42)

TRAIN_TFRECS = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
TEST_TFRECS = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(TRAIN_TFRECS) > 0, "No train TFRecords found"
assert len(TEST_TFRECS) > 0, "No test TFRecords found"

_trn_ids = set(trn_df["image_id"].tolist())
_val_ids = set(val_df["image_id"].tolist())

_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


def _decode_and_resize_train(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 HWC
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    area = tf.cast(h * w, tf.float32)

    scale = tf.random.uniform([], 0.8, 1.0, dtype=tf.float32)
    target_area = area * scale
    log_ratio = tf.random.uniform(
        [], tf.math.log(0.9), tf.math.log(1.1), dtype=tf.float32
    )
    ratio = tf.exp(log_ratio)

    crop_w = tf.cast(tf.round(tf.sqrt(target_area * ratio)), tf.int32)
    crop_h = tf.cast(tf.round(tf.sqrt(target_area / ratio)), tf.int32)
    crop_w = tf.clip_by_value(crop_w, 1, w)
    crop_h = tf.clip_by_value(crop_h, 1, h)

    offset_h = tf.cond(
        h > crop_h,
        lambda: tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32),
        lambda: tf.constant(0, dtype=tf.int32),
    )
    offset_w = tf.cond(
        w > crop_w,
        lambda: tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32),
        lambda: tf.constant(0, dtype=tf.int32),
    )

    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_h, crop_w)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=True)

    img = tf.image.random_flip_left_right(img)

    def _jitter(x):
        x = tf.image.random_brightness(x, max_delta=0.2)
        x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        x = tf.image.random_saturation(x, lower=0.8, upper=1.2)
        x = tf.image.random_hue(x, max_delta=0.05)
        return tf.clip_by_value(x, 0.0, 1.0)

    img = tf.cond(tf.random.uniform([]) < 0.5, lambda: _jitter(img), lambda: img)

    angle = tf.random.uniform([], -10.0, 10.0, dtype=tf.float32) * (math.pi / 180.0)
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    img = (img - mean) / std

    img = tf.transpose(img, [2, 0, 1])
    return img


def _decode_and_resize_valid(img_bytes, resize_longer):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    scale = tf.cast(resize_longer, tf.float32) / tf.minimum(h, w)
    nh = tf.cast(tf.round(h * scale), tf.int32)
    nw = tf.cast(tf.round(w * scale), tf.int32)
    img = tf.image.resize(img, [nh, nw], method="bilinear", antialias=True)

    offset_h = tf.maximum(0, (nh - IMG_SIZE) // 2)
    offset_w = tf.maximum(0, (nw - IMG_SIZE) // 2)
    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, IMG_SIZE, IMG_SIZE)

    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    img = (img - mean) / std
    img = tf.transpose(img, [2, 0, 1])
    return img


def _parse_and_filter(example_proto, want_split):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    image_name = ex["image_name"]
    label = ex["label"]
    name_str = tf.compat.as_str_any(image_name)

    def _in_split(x):
        s = x.decode("utf-8")
        if want_split == 0:
            return np.bool_(s in _trn_ids)
        else:
            return np.bool_(s in _val_ids)

    keep = tf.numpy_function(_in_split, [image_name], Tout=tf.bool)
    keep.set_shape([])
    return keep, ex


def _build_train_ds():
    ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

    def _filt_map(x):
        keep, ex = _parse_and_filter(x, want_split=0)
        return keep, ex

    ds = ds.map(_filt_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda keep, ex: keep)
    ds = ds.map(
        lambda keep, ex: (
            _decode_and_resize_train(ex["image"]),
            tf.cast(ex["label"], tf.int64),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_val_ds():
    ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=tf.data.AUTOTUNE)

    def _filt_map(x):
        keep, ex = _parse_and_filter(x, want_split=1)
        return keep, ex

    ds = ds.map(_filt_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda keep, ex: keep)
    ds = ds.map(
        lambda keep, ex: (
            _decode_and_resize_valid(ex["image"], int(IMG_SIZE * 256 / 224)),
            tf.cast(ex["label"], tf.int64),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_tfds = _build_train_ds()
val_tfds = _build_val_ds()


class TorchFromTFDS:
    def __init__(self, tfds, device):
        self.tfds = tfds
        self.device = device

    def __iter__(self):
        for x_np, y_np in self.tfds.as_numpy_iterator():
            x = (
                torch.from_numpy(x_np)
                .to(self.device, non_blocking=True)
                .contiguous(memory_format=torch.channels_last)
            )
            y = torch.from_numpy(y_np).to(self.device, non_blocking=True).long()
            yield x, y

    def __len__(self):
        return (
            int(math.ceil(len(trn_df) / BATCH_SIZE))
            if self.tfds is train_tfds
            else int(math.ceil(len(val_df) / BATCH_SIZE))
        )


trn_loader = TorchFromTFDS(train_tfds, device)
val_loader = TorchFromTFDS(val_tfds, device)

model = Classifier()
model.freeze()
model = model.to(device).to(memory_format=torch.channels_last)

_DO_COMPILE = False
if _DO_COMPILE and hasattr(torch, "compile") and torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        pass




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
def evaluate(model, val_loader):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            out = model(images)
            loss = F.cross_entropy(out, labels)
            preds = out.argmax(dim=1)
            acc = (preds == labels).float().mean()
            loss_sum += float(loss)
            acc_sum += float(acc)
            n_batches += 1
    return {"val_loss": loss_sum / n_batches, "val_acc": acc_sum / n_batches}


def fit_fc(epochs, lr, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(
        model.network.fc.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for images, labels in train_loader:
            out = model(images)
            loss = F.cross_entropy(out, labels)
            train_loss_sum += float(loss.detach())
            n_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        result = evaluate(model, val_loader)
        result["train_loss"] = train_loss_sum / n_batches
        model.epoch_end(epoch, epochs, result)

    return model


EPOCHS = 16
LR = 3e-4
model = fit_fc(EPOCHS, LR, model, trn_loader, val_loader)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/357602905.py in <cell line: 0>()
     48 EPOCHS = 16
     49 LR = 3e-4
---> 50 model = fit_fc(EPOCHS, LR, model, trn_loader, val_loader)
     51 
     52 

/tmp/ipykernel_55/357602905.py in fit_fc(epochs, lr, model, train_loader, val_loader)
     30             out = model(images)
     31             loss = F.cross_entropy(out, labels)
---> 32             train_loss_sum += float(loss.detach())
     33             n_batches += 1
     34 

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## === cell 6
test_csv = pd.read_csv(SAMPLE_SUB_PATH)
assert "image_id" in test_csv.columns and "label" in test_csv.columns

_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _build_test_ds(resize_longer):
    ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=tf.data.AUTOTUNE)

    def _parse(ex_proto):
        ex = tf.io.parse_single_example(ex_proto, _TEST_FEATURE_DESC)
        img = _decode_and_resize_valid(ex["image"], resize_longer)
        return ex["image_name"], img

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(256, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_tfds_a = _build_test_ds(int(IMG_SIZE * 256 / 224))
test_tfds_b = _build_test_ds(int(IMG_SIZE * 288 / 224))


class TorchTestPairFromTFDS:
    def __init__(self, tfds_a, tfds_b, device):
        self.a = tfds_a
        self.b = tfds_b
        self.device = device

    def __iter__(self):
        ita = self.a.as_numpy_iterator()
        itb = self.b.as_numpy_iterator()
        for (names_a, xa), (names_b, xb) in zip(ita, itb):
            xa_t = (
                torch.from_numpy(xa)
                .to(self.device, non_blocking=True)
                .contiguous(memory_format=torch.channels_last)
            )
            xb_t = (
                torch.from_numpy(xb)
                .to(self.device, non_blocking=True)
                .contiguous(memory_format=torch.channels_last)
            )
            yield names_a, xa_t, xb_t

    def __len__(self):
        return int(math.ceil(len(test_csv) / 256))


test_loader = TorchTestPairFromTFDS(test_tfds_a, test_tfds_b, device)

len(test_csv)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InternalError                             Traceback (most recent call last)
/tmp/ipykernel_55/588939190.py in <cell line: 0>()
     24 
     25 
---> 26 test_tfds_a = _build_test_ds(int(IMG_SIZE * 256 / 224))
     27 test_tfds_b = _build_test_ds(int(IMG_SIZE * 288 / 224))
     28 

/tmp/ipykernel_55/588939190.py in _build_test_ds(resize_longer)
     19 
     20     ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
---> 21     ds = ds.batch(256, drop_remainder=False)
     22     ds = ds.prefetch(tf.data.AUTOTUNE)
     23     return ds

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in batch(self, batch_size, drop_remainder, num_parallel_calls, deterministic, name)
   1913     # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
   1914     from tensorflow.python.data.ops import batch_op
-> 1915     return batch_op._batch(self, batch_size, drop_remainder, num_parallel_calls,
   1916                            deterministic, name)
   1917     # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/batch_op.py in _batch(input_dataset, batch_size, drop_remainder, num_parallel_calls, deterministic, name)
     37       warnings.warn("The `deterministic` argument has no effect unless the "
     38                     "`num_parallel_calls` argument is specified.")
---> 39     return _BatchDataset(input_dataset, batch_size, drop_remainder, name=name)
     40   else:
     41     return _ParallelBatchDataset(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/batch_op.py in __init__(self, input_dataset, batch_size, drop_remainder, name)
     54     """See `Dataset.batch()` for details."""
     55     self._input_dataset = input_dataset
---> 56     self._batch_size = ops.convert_to_tensor(
     57         batch_size, dtype=dtypes.int64, name="batch_size")
     58     self._drop_remainder = ops.convert_to_tensor(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0 in order to run _EagerConst: CUDA error: Error recording CUDA event: CUDA_ERROR_ASSERT: device-side assert triggered

## === cell 7
def inference_tta_hflip_pair(model, test_loader, device):
    model.to(device)
    model.eval()

    probs_a = []
    probs_b = []
    names_all = []
    with torch.inference_mode():
        for names, images_a, images_b in test_loader:
            bs = images_a.shape[0]
            x = torch.cat(
                [
                    images_a,
                    torch.flip(images_a, dims=[3]),
                    images_b,
                    torch.flip(images_b, dims=[3]),
                ],
                dim=0,
            )
            logits = model(x).softmax(1)  # (4*bs, 5)
            p1a, p2a, p1b, p2b = logits.split(bs, dim=0)
            probs_a.append(((p1a + p2a) * 0.5).detach().cpu().numpy())
            probs_b.append(((p1b + p2b) * 0.5).detach().cpu().numpy())
            names_all.append(names)

    probs_a = np.concatenate(probs_a, axis=0)
    probs_b = np.concatenate(probs_b, axis=0)
    names_all = np.concatenate(names_all, axis=0).astype("U")
    print("predictions shape A:", probs_a.shape, "B:", probs_b.shape)
    return names_all, probs_a, probs_b


names_all, pred_a, pred_b = inference_tta_hflip_pair(model, test_loader, device)
predictions = (pred_a + pred_b) * 0.5

pred_map = {n: p for n, p in zip(names_all.tolist(), predictions)}
ordered = np.stack([pred_map[i] for i in test_csv["image_id"].tolist()], axis=0)

assert ordered.shape[0] == len(test_csv), (ordered.shape, len(test_csv))

test_csv["label"] = ordered.argmax(axis=1).astype(int)
submission_path = "./submission.csv"
test_csv[["image_id", "label"]].to_csv(submission_path, index=False)

print(test_csv.head())
print("Wrote:", submission_path, "rows:", len(test_csv))
test_csv

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3562218305.py in <cell line: 0>()
     31 
     32 
---> 33 names_all, pred_a, pred_b = inference_tta_hflip_pair(model, test_loader, device)
     34 predictions = (pred_a + pred_b) * 0.5
     35 

NameError: name 'test_loader' is not defined
