# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os, sys, random, warnings, math

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain

import cv2
from skimage.transform import (
    resize,
)  # kept to preserve import surface; no longer used in hot loops
from skimage.morphology import label  # unused but kept to preserve imports/compat

try:
    from google.protobuf import message_factory as _mf

    _MessageFactory = _mf.MessageFactory

    if not hasattr(_MessageFactory, "GetPrototype"):
        if hasattr(_MessageFactory, "GetMessageClass"):
            _MessageFactory.GetPrototype = _MessageFactory.GetMessageClass
        elif hasattr(_MessageFactory, "GetMessages"):

            def _get_prototype_fallback(self, descriptor):
                msgs = self.GetMessages([descriptor.file])
                return msgs.get(descriptor.full_name)

            _MessageFactory.GetPrototype = _get_prototype_fallback
        else:

            def _missing_getprototype(self, descriptor):
                raise AttributeError(
                    "protobuf MessageFactory has no GetPrototype/GetMessageClass; incompatible protobuf version."
                )

            _MessageFactory.GetPrototype = _missing_getprototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils, optimizers

random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(4, os.cpu_count() or 4))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

warnings.filterwarnings("ignore")




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
if not (os.path.isdir("train") and os.path.isdir("test")):
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

if not (
    os.path.isdir("train/images")
    or any(os.path.basename(r) == "images" for r, _, _ in os.walk("train"))
):
    get_ipython().run_line_magic(
        "bash",
        "unzip -q ../input/tgs-salt-identification-challenge/train.zip -d train/",
    )
if not (
    os.path.isdir("test/images")
    or any(os.path.basename(r) == "images" for r, _, _ in os.walk("test"))
):
    get_ipython().run_line_magic(
        "bash", "unzip -q ../input/tgs-salt-identification-challenge/test.zip -d test/"
    )




## === cell 3
pass




## === cell 4
def _base_dir_from_images_dir(images_dir: str) -> str:
    base = os.path.dirname(images_dir)
    return base if base.endswith(os.sep) else base + os.sep


_train_images_dir = "train/images"
_train_masks_dir = "train/masks"
if not (os.path.isdir(_train_images_dir) and os.path.isdir(_train_masks_dir)):
    _train_images_dir, _train_masks_dir = None, None
    for root, dirs, files in os.walk("train"):
        if os.path.basename(root) == "images" and os.path.isdir(
            os.path.join(os.path.dirname(root), "masks")
        ):
            _train_images_dir = root
            _train_masks_dir = os.path.join(os.path.dirname(root), "masks")
            break
    if _train_images_dir is None or _train_masks_dir is None:
        raise FileNotFoundError(
            "Could not locate extracted train images/masks directories under 'train/'. "
            "Expected either 'train/images'+'train/masks' or a nested structure like 'train/**/images'+'train/**/masks'."
        )

train_images_dir = _train_images_dir

test_images_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(test_images_dir):
    test_images_dir = None
    for root, dirs, files in os.walk(config.path_test):
        if os.path.basename(root) == "images":
            test_images_dir = root
            break
    if test_images_dir is None:
        raise FileNotFoundError(
            "Could not locate extracted test images directory under 'test/'. "
            "Expected either 'test/images' or a nested structure like 'test/**/images'."
        )

config.path_train = _base_dir_from_images_dir(train_images_dir)
config.path_test = _base_dir_from_images_dir(test_images_dir)

train_ids = sorted(os.listdir(os.path.join(config.path_train, "images")))
test_ids = sorted(os.listdir(os.path.join(config.path_test, "images")))




## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3640187948.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     17[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m     [0;32mif[0m [0m_train_images_dir[0m [0;32mis[0m [0;32mNone[0m [0;32mor[0m [0m_train_masks_dir[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m         raise FileNotFoundError(
[0m[1;32m     20[0m             [0;34m"Could not locate extracted train images/masks directories under 'train/'. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m             [0;34m"Expected either 'train/images'+'train/masks' or a nested structure like 'train/**/images'+'train/**/masks'."[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: Could not locate extracted train images/masks directories under 'train/'. Expected either 'train/images'+'train/masks' or a nested structure like 'train/**/images'+'train/**/masks'.

## === cell 5
X = np.empty(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=np.bool_)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

train_img_dir = os.path.join(config.path_train, "images")
train_mask_dir = os.path.join(config.path_train, "masks")

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img_path = os.path.join(train_img_dir, id_)
    msk_path = os.path.join(train_mask_dir, id_)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    X[n, ..., 0] = cv2.resize(
        img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )

    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise FileNotFoundError(f"Failed to read mask: {msk_path}")
    msk_r = cv2.resize(
        msk, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST
    )
    Y[n, ..., 0] = msk_r > 127

print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
