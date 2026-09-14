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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
tqdm==4.67.1

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

0.8650649743124811

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'The timeout is dominated by (1) running EfficientNet-B4 training for 20 epochs on ~90% of 18.7k images, (2) slow Python-side PIL+albumentations decoding per sample, and (3) an extremely expensive validation pass due to one-hot + per-batch bookkeeping overhead. The changes below keep the same model, loss, epochs, batch size, and data split semantics, but make the input pipeline and validation much faster: we switch to torchvision’s fast `read_image` backend, enable PyTorch compilation and TF32 (numerically negligible), make DataLoader seeding deterministic with a generator, and remove unnecessary tensor constructions in the LR schedule and validation inner loop. TTA is also made equivalent but cheaper by batching the repeated forward passes into one stacked batch. These are provably equivalent (same computations/order aside from negligible FP differences) and targeted only at runtime.'
- What this solution (achieved 0.11024) has done: 'We fix the crash by ensuring the classifier parameters are not included in both optimizer parameter groups (the current code accidentally duplicates them, triggering the “parameters appear in more than one parameter group” error). This change is minimal and keeps the same model, loss, epochs, and LR schedule semantics, but allows training to actually run end-to-end (which should move score dramatically toward the target vs. the current near-random 0.11024). We also make the parameter-name exclusion robust to EfficientNet-B4’s actual classifier indexing so it works consistently with/without DataParallel. Finally, we keep the submission writing logic intact to produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import random
import glob
import struct
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image




## === cell 1
def _pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


BASE_INPUT = _pick_existing_path(
    [
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification",
    ]
)

TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(BASE_INPUT, "train_images")
TEST_IMAGE_PATH = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_INPUT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_INPUT, "test_tfrecords")



## === cell 2
INPUT_PATH = "../input/mymodelparam"  # kept, but not relied on (may not exist)
SUBMISSION_PATH = "submission.csv"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
if len(DEVICES) == 0:
    DEVICES = [torch.device("cpu")]

OUT_FEATURES = 5
NUM_EPOCHS = 20
BATCH_SIZE = 16
IMAGE_SIZE = 224
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 3

BEST_MODEL_PATH = "best_model.pth"  # Fix: consistent filename (no stray space)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True




## === cell 3
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor).to(device=y_hat.device, dtype=torch.float32)

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")

    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1.0 - y_true) * (1.0 - p)

    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




## === cell 4
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (math.cos(decay_factor) + 1.0) / 2.0
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr




## === cell 5
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333),
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0
        ),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(IMAGE_SIZE // 20, IMAGE_SIZE // 5),
            hole_width_range=(IMAGE_SIZE // 20, IMAGE_SIZE // 5),
            fill=0,
            p=0.5,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 6
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    scale=(0.85, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 7
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)



## === cell 8
from torchvision.io import decode_jpeg

_MASK_TYPE_MAP = {
    "byte": 0x00,
    "float": 0x01,
    "int": 0x02,
}


def _read_varint(buf: memoryview, pos: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _parse_example(serialized: bytes) -> dict:
    buf = memoryview(serialized)
    pos = 0

    msg_len, pos = _read_varint(buf, pos)
    end = pos + msg_len

    features = {}

    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire = tag & 7
        if field_num == 1 and wire == 2:
            flen, pos = _read_varint(buf, pos)
            f_end = pos + flen

            while pos < f_end:
                tag2, pos = _read_varint(buf, pos)
                f_field = tag2 >> 3
                f_wire = tag2 & 7
                if f_field == 1 and f_wire == 2:
                    mlen, pos = _read_varint(buf, pos)
                    m_end = pos + mlen

                    key = None
                    value = None

                    while pos < m_end:
                        tag3, pos = _read_varint(buf, pos)
                        mf = tag3 >> 3
                        mw = tag3 & 7
                        if mf == 1 and mw == 2:  # key
                            klen, pos = _read_varint(buf, pos)
                            key = bytes(buf[pos : pos + klen]).decode("utf-8")
                            pos += klen
                        elif mf == 2 and mw == 2:  # value (Feature message)
                            vlen, pos = _read_varint(buf, pos)
                            v_end = pos + vlen

                            kind = None
                            vals = None

                            while pos < v_end:
                                tag4, pos = _read_varint(buf, pos)
                                ff = tag4 >> 3
                                fw = tag4 & 7
                                if ff == 1 and fw == 2:  # bytes_list
                                    blen, pos = _read_varint(buf, pos)
                                    b_end = pos + blen
                                    out = []
                                    while pos < b_end:
                                        t5, pos = _read_varint(buf, pos)
                                        f5 = t5 >> 3
                                        w5 = t5 & 7
                                        if f5 == 1 and w5 == 2:
                                            sl, pos = _read_varint(buf, pos)
                                            out.append(bytes(buf[pos : pos + sl]))
                                            pos += sl
                                        else:
                                            if w5 == 0:
                                                _, pos = _read_varint(buf, pos)
                                            elif w5 == 2:
                                                l, pos = _read_varint(buf, pos)
                                                pos += l
                                            else:
                                                raise ValueError("Unsupported wire")
                                    kind = "bytes"
                                    vals = out
                                elif ff == 2 and fw == 2:  # float_list
                                    flen2, pos = _read_varint(buf, pos)
                                    f2_end = pos + flen2
                                    out = []
                                    while pos < f2_end:
                                        t5, pos = _read_varint(buf, pos)
                                        f5 = t5 >> 3
                                        w5 = t5 & 7
                                        if f5 == 1 and w5 == 5:
                                            out.append(
                                                struct.unpack_from("<f", buf, pos)[0]
                                            )
                                            pos += 4
                                        else:
                                            if w5 == 0:
                                                _, pos = _read_varint(buf, pos)
                                            elif w5 == 2:
                                                l, pos = _read_varint(buf, pos)
                                                pos += l
                                            elif w5 == 5:
                                                pos += 4
                                            else:
                                                raise ValueError("Unsupported wire")
                                    kind = "float"
                                    vals = out
                                elif ff == 3 and fw == 2:  # int64_list
                                    ilen, pos = _read_varint(buf, pos)
                                    i_end = pos + ilen
                                    out = []
                                    while pos < i_end:
                                        t5, pos = _read_varint(buf, pos)
                                        f5 = t5 >> 3
                                        w5 = t5 & 7
                                        if f5 == 1 and w5 == 0:
                                            v, pos = _read_varint(buf, pos)
                                            out.append(int(v))
                                        else:
                                            if w5 == 0:
                                                _, pos = _read_varint(buf, pos)
                                            elif w5 == 2:
                                                l, pos = _read_varint(buf, pos)
                                                pos += l
                                            else:
                                                raise ValueError("Unsupported wire")
                                    kind = "int"
                                    vals = out
                                else:
                                    if fw == 0:
                                        _, pos = _read_varint(buf, pos)
                                    elif fw == 2:
                                        l, pos = _read_varint(buf, pos)
                                        pos += l
                                    elif fw == 5:
                                        pos += 4
                                    else:
                                        raise ValueError("Unsupported wire")
                            value = (kind, vals)
                            pos = v_end
                        else:
                            if mw == 0:
                                _, pos = _read_varint(buf, pos)
                            elif mw == 2:
                                l, pos = _read_varint(buf, pos)
                                pos += l
                            else:
                                raise ValueError("Unsupported wire")
                    pos = m_end
                    if key is not None and value is not None:
                        features[key] = value
                else:
                    if f_wire == 0:
                        _, pos = _read_varint(buf, pos)
                    elif f_wire == 2:
                        l, pos = _read_varint(buf, pos)
                        pos += l
                    else:
                        raise ValueError("Unsupported wire")
            pos = f_end
        else:
            if wire == 0:
                _, pos = _read_varint(buf, pos)
            elif wire == 2:
                l, pos = _read_varint(buf, pos)
                pos += l
            else:
                raise ValueError("Unsupported wire")
    return features


def _masked_crc32c_to_uint32(v: int) -> int:
    return int(v) & 0xFFFFFFFF


def _unmask_crc32c(masked: int) -> int:
    masked = _masked_crc32c_to_uint32(masked)
    rot = ((masked >> 17) | ((masked << 15) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return _masked_crc32c_to_uint32(rot - 0xA282EAD8)


class TFRecordIndex:
    def __init__(self, tfrec_paths: List[str]):
        self.tfrec_paths = tfrec_paths
        self._offsets: List[Tuple[int, int]] = []
        self._build()

    def _build(self):
        offsets: List[Tuple[int, int]] = []
        for fi, path in enumerate(self.tfrec_paths):
            fsize = os.path.getsize(path)
            with open(path, "rb") as f:
                while True:
                    record_start = f.tell()
                    header = f.read(12)
                    if len(header) < 12:
                        break

                    length = struct.unpack("<Q", header[:8])[0]
                    masked_len_crc = struct.unpack("<I", header[8:12])[0]

                    if length > fsize or length < 0:
                        break

                    offsets.append((fi, record_start))

                    try:
                        f.seek(int(length) + 4, os.SEEK_CUR)
                    except (ValueError, OSError):
                        break
        self._offsets = offsets

    def __len__(self):
        return len(self._offsets)

    def get(self, idx: int) -> Tuple[int, int]:
        return self._offsets[idx]


class TFRecordReader:
    def __init__(self, tfrec_paths: List[str]):
        self.tfrec_paths = tfrec_paths
        self._fps: List[Optional[object]] = [None] * len(tfrec_paths)

    def _fp(self, fi: int):
        fp = self._fps[fi]
        if fp is None:
            fp = open(self.tfrec_paths[fi], "rb")
            self._fps[fi] = fp
        return fp

    def read_example_at(self, fi: int, offset: int) -> bytes:
        fp = self._fp(fi)
        fp.seek(int(offset))
        header = fp.read(12)
        if len(header) < 12:
            raise EOFError("Unexpected EOF while reading TFRecord header")
        length = struct.unpack("<Q", header[:8])[0]
        data = fp.read(int(length))
        fp.read(4)  # data crc
        return data


def _as_bytes_first(features: dict, key: str) -> bytes:
    kind, vals = features[key]
    if kind != "bytes":
        raise ValueError(f"{key} is not bytes_list")
    return vals[0]


def _as_int_first(features: dict, key: str) -> int:
    kind, vals = features[key]
    if kind != "int":
        raise ValueError(f"{key} is not int64_list")
    return int(vals[0])


class MyCassavaLeafDataset(Dataset):
    @staticmethod
    def generate_index(num_total, ratio):
        k = int(max(1, round(ratio * 10)))
        valid_index = np.arange(0, num_total, k, dtype=np.int64)
        mask = np.ones(num_total, dtype=bool)
        mask[valid_index] = False
        train_index = np.nonzero(mask)[0].astype(np.int64)
        return train_index, valid_index

    def __init__(
        self,
        csv_path=None,
        images_path=None,  # kept for path compatibility; unused in TFRecord mode
        transform=None,
        mode="train",
        train_ratio=0.5,
    ):
        super().__init__()
        self.transform = transform
        self.mode = mode

        self.data_info = pd.read_csv(csv_path)
        self.data_len = self.data_info.shape[0]

        if self.mode == "train":
            train_index, _ = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.index_arr = train_index
        elif self.mode == "valid":
            _, valid_index = MyCassavaLeafDataset.generate_index(
                self.data_len, train_ratio
            )
            self.index_arr = valid_index
        else:
            raise ValueError("mode must be 'train' or 'valid' for MyCassavaLeafDataset")

        self.real_len = len(self.index_arr)

        tfrec_paths = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
        if not tfrec_paths:
            raise FileNotFoundError(f"No TFRecords found in {TRAIN_TFREC_DIR}")
        self._tfr_index = TFRecordIndex(tfrec_paths)

        if len(self._tfr_index) < self.data_len:
            raise RuntimeError(
                f"TFRecord index has {len(self._tfr_index)} records but train.csv has {self.data_len} rows."
            )

        self._tfr_reader = TFRecordReader(tfrec_paths)

    def __len__(self):
        return self.real_len

    def __getitem__(self, i):
        global_idx = int(self.index_arr[i])

        fi, off = self._tfr_index.get(global_idx)
        ex = self._tfr_reader.read_example_at(fi, off)
        feats = _parse_example(ex)

        img_bytes = _as_bytes_first(feats, "image")
        label = _as_int_first(feats, "target")

        img_t = decode_jpeg(
            torch.frombuffer(img_bytes, dtype=torch.uint8), device="cpu"
        )
        img = img_t.permute(1, 2, 0).numpy()  # HWC uint8

        return self.transform(image=img)["image"], int(label)


class MyCassavaLeafTestDataset(Dataset):
    def __init__(self, tfrec_dir, transform):
        self.transform = transform
        self.tfrec_paths = sorted(glob.glob(os.path.join(tfrec_dir, "*.tfrec")))
        if not self.tfrec_paths:
            raise FileNotFoundError(f"No TFRecords found in {tfrec_dir}")
        self._tfr_index = TFRecordIndex(self.tfrec_paths)
        self._tfr_reader = TFRecordReader(self.tfrec_paths)

    def __len__(self):
        return len(self._tfr_index)

    def __getitem__(self, idx):
        fi, off = self._tfr_index.get(int(idx))
        ex = self._tfr_reader.read_example_at(fi, off)
        feats = _parse_example(ex)

        img_bytes = _as_bytes_first(feats, "image")
        image_id = _as_bytes_first(feats, "image_name").decode("utf-8")

        img_t = decode_jpeg(
            torch.frombuffer(img_bytes, dtype=torch.uint8), device="cpu"
        )
        img = img_t.permute(1, 2, 0).numpy()

        x = self.transform(image=img)["image"]
        return x, image_id




## === cell 9
train_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=train_augs,
    mode="train",
)
valid_set = MyCassavaLeafDataset(
    csv_path=TRAIN_CSV_PATH,
    images_path=TRAIN_IMAGE_PATH,
    transform=valid_augs,
    mode="valid",
)

_NUM_WORKERS = min(8, (os.cpu_count() or 2))

_g = torch.Generator()
_g.manual_seed(SEED)


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_loader_kwargs = dict(
    num_workers=_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=4 if _NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if _NUM_WORKERS > 0 else None,
    generator=_g,
)

my_train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)
my_valid_dataloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)



## === cell 10
my_model = torchvision.models.efficientnet_b4(weights=None)
my_model.classifier[-1] = nn.Linear(my_model.classifier[-1].in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model.classifier[-1].weight)




## === cell 11
class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_hat_idx = y_hat.argmax(dim=1)
        y_true_idx = y_true.argmax(dim=1)
        return float((y_hat_idx == y_true_idx).sum().item())

    @staticmethod
    def _to_one_hot(y_int, num_classes, device):
        y_int = y_int.to(device=device, dtype=torch.long)
        return F.one_hot(y_int, num_classes=num_classes).to(dtype=torch.float32)

    @staticmethod
    def calc_valid_acc(model, valid_dataloader):
        model.eval()
        device = next(iter(model.parameters())).device
        test_num = 0
        correct = 0
        with torch.inference_mode():
            for x, y_true in valid_dataloader:
                x = x.to(device, non_blocking=True)
                y_true = y_true.to(device, non_blocking=True)
                if torch.cuda.is_available():
                    x = x.to(memory_format=torch.channels_last)
                pred = model(x).argmax(dim=1)
                correct += (pred == y_true).sum().item()
                test_num += y_true.numel()
        return correct / test_num

    def __init__(
        self,
        optimizer,
        model,
        criterion,
        train_dataloader,
        valid_dataloader,
        param_group=True,
        learning_rate=lr_tune,
        num_epochs=NUM_EPOCHS,
        devices=DEVICES,
    ):
        self.optimizer_class = optimizer
        self.model = model
        self.criterion = criterion
        self.devices = devices
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.param_group = param_group
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = None

    def _build_optimizer_once(self):
        if isinstance(self.model, nn.DataParallel):
            module = self.model.module
        else:
            module = self.model

        classifier_params = list(module.classifier.parameters())
        classifier_param_ids = {id(p) for p in classifier_params}

        base_params = [
            p for p in module.parameters() if id(p) not in classifier_param_ids
        ]

        self.optimizer = self.optimizer_class(
            [
                {"params": base_params},
                {"params": classifier_params, "lr": self.learning_rate(0) * 10.0},
            ],
            lr=self.learning_rate(0),
            weight_decay=0.001,
        )

    def _set_epoch_lrs(self, epoch):
        lr = float(self.learning_rate(epoch))
        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.param_groups[1]["lr"] = lr * 10.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        train_num = 0
        train_acc_num = 0.0
        dl = self.train_dataloader

        if self.optimizer is None:
            self._build_optimizer_once()
        self._set_epoch_lrs(epoch)

        device0 = self.devices[0]
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        to_one_hot = MyTrainer._to_one_hot
        accurate_count = MyTrainer.accurate_count

        print(f"epoch{epoch + 1} begins:")

        for x, y_true in dl:
            x = x.to(device0, non_blocking=True)
            if torch.cuda.is_available():
                x = x.to(memory_format=torch.channels_last)
            y_true_oh = to_one_hot(y_true, OUT_FEATURES, device0)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x)
            loss = criterion(y_hat, y_true_oh)
            loss_sum = loss.sum()
            loss_sum.backward()
            optimizer.step()

            bs = y_true_oh.shape[0]
            total_loss += float(loss_sum.detach().item())
            train_num += bs
            train_acc_num += accurate_count(y_hat.detach(), y_true_oh.detach())

        return total_loss / train_num, train_acc_num / train_num

    def train(self):
        best_valid_acc = 0

        if torch.cuda.is_available() and len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
                self.devices[0]
            )
        else:
            self.model = self.model.to(self.devices[0])

        if torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)

        for epoch in range(self.num_epochs):
            train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
            valid_acc = MyTrainer.calc_valid_acc(self.model, self.valid_dataloader)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
            print(
                f"epoch{epoch + 1}:train_loss:{train_loss}, train_acc:{train_acc}, valid_acc:{valid_acc}"
            )




## === cell 12
torch.cuda.empty_cache()

trainer = MyTrainer(
    optimizer=OPTIMIZER,
    model=my_model,
    criterion=sigmoid_focal_cross_entropy,
    train_dataloader=my_train_dataloader,
    valid_dataloader=my_valid_dataloader,
)
trainer.train()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1634245456.py in <cell line: 0>()
      8     valid_dataloader=my_valid_dataloader,
      9 )
---> 10 trainer.train()
     11 

/tmp/ipykernel_55/1817266856.py in train(self)
    132 
    133         for epoch in range(self.num_epochs):
--> 134             train_loss, train_acc = MyTrainer.train_epoch(self, epoch)
    135             valid_acc = MyTrainer.calc_valid_acc(self.model, self.valid_dataloader)
    136             if valid_acc > best_valid_acc:

/tmp/ipykernel_55/1817266856.py in train_epoch(self, epoch)
     98         print(f"epoch{epoch + 1} begins:")
     99 
--> 100         for x, y_true in dl:
    101             x = x.to(device0, non_blocking=True)
    102             if torch.cuda.is_available():

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

KeyError: Caught KeyError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/2638044700.py", line 335, in __getitem__
    img_bytes = _as_bytes_first(feats, "image")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2638044700.py", line 260, in _as_bytes_first
    kind, vals = features[key]
                 ~~~~~~~~^^^^^
KeyError: 'image'


## === cell 13
device = DEVICES[0]
my_model = my_model.to(device)
my_model.eval()

state_dict = None
if os.path.exists(BEST_MODEL_PATH):
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
elif os.path.exists(os.path.join(INPUT_PATH, "best_model.pth")):
    state_dict = torch.load(
        os.path.join(INPUT_PATH, "best_model.pth"), map_location=device
    )
elif os.path.exists(os.path.join(INPUT_PATH, "best_model .pth")):  # legacy typo path
    state_dict = torch.load(
        os.path.join(INPUT_PATH, "best_model .pth"), map_location=device
    )

if state_dict is not None:
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    my_model.load_state_dict(state_dict, strict=True)

if torch.cuda.is_available():
    my_model = my_model.to(memory_format=torch.channels_last)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()
id_to_row = {img_id: i for i, img_id in enumerate(test_image_ids)}

test_ds = MyCassavaLeafTestDataset(TEST_TFREC_DIR, test_augs)

try:
    _lk = {k: v for k, v in _loader_kwargs.items() if v is not None}
except NameError:
    _NUM_WORKERS = min(8, (os.cpu_count() or 2))
    _g = torch.Generator()
    _g.manual_seed(SEED)

    def _seed_worker(worker_id):
        worker_seed = (SEED + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    _lk = dict(
        num_workers=_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=4 if _NUM_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if _NUM_WORKERS > 0 else None,
        generator=_g,
    )
    _lk = {k: v for k, v in _lk.items() if v is not None}

test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    **_lk,
)

pred_labels = np.empty(len(test_image_ids), dtype=np.int64)

with torch.inference_mode():
    for x, image_ids in tqdm(test_dl, total=len(test_dl)):
        x = x.to(device, non_blocking=True)
        if torch.cuda.is_available():
            x = x.to(memory_format=torch.channels_last)

        if TTA > 1:
            x_rep = x.repeat_interleave(TTA, dim=0)
            logits = my_model(x_rep).view(TTA, x.size(0), OUT_FEATURES).sum(dim=0)
        else:
            logits = my_model(x)

        pred = logits.argmax(dim=1).detach().cpu().numpy()
        for img_id, p in zip(image_ids, pred):
            if img_id in id_to_row:
                pred_labels[id_to_row[img_id]] = int(p)

submission = sample_sub.copy()
submission["label"] = pred_labels.astype(int)
submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Wrote {SUBMISSION_PATH} with shape {submission.shape} and columns {list(submission.columns)}"
)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2239359400.py in <cell line: 0>()
     65 
     66 with torch.inference_mode():
---> 67     for x, image_ids in tqdm(test_dl, total=len(test_dl)):
     68         x = x.to(device, non_blocking=True)
     69         if torch.cuda.is_available():

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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

ValueError: Caught ValueError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/2638044700.py", line 361, in __getitem__
    feats = _parse_example(ex)
            ^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2638044700.py", line 176, in _parse_example
    raise ValueError("Unsupported wire")
ValueError: Unsupported wire
