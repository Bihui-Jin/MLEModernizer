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

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 3. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 4. Code solution

## === cell 0
import os
import sys

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print(
        "WARNING: TensorFlow failed to import due to an environment/protobuf mismatch."
    )
    print("Import error:", repr(e))

import cv2
import zipfile
import matplotlib.pyplot as plt

print("Python:", sys.version)
print("TensorFlow:", tf.__version__ if tf is not None else "unavailable")


## === cell 1
TEST_DIR = "../input/dogs-vs-cats-redux-kernels-edition/test.zip"
TRAIN_DIR = "../input/dogs-vs-cats-redux-kernels-edition/train.zip"



## === cell 2
EXTRACT_ROOT = "./extracted"
TRAIN_EXTRACT_DIR = os.path.join(EXTRACT_ROOT, "trainzip")
TEST_EXTRACT_DIR = os.path.join(EXTRACT_ROOT, "testzip")

os.makedirs(TRAIN_EXTRACT_DIR, exist_ok=True)
os.makedirs(TEST_EXTRACT_DIR, exist_ok=True)

train_marker = os.path.join(TRAIN_EXTRACT_DIR, ".done")
test_marker = os.path.join(TEST_EXTRACT_DIR, ".done")

if not os.path.exists(train_marker):
    with zipfile.ZipFile(TRAIN_DIR, "r") as z:
        z.extractall(TRAIN_EXTRACT_DIR)
    with open(train_marker, "w") as f:
        f.write("ok")

if not os.path.exists(test_marker):
    with zipfile.ZipFile(TEST_DIR, "r") as z:
        z.extractall(TEST_EXTRACT_DIR)
    with open(test_marker, "w") as f:
        f.write("ok")

print("Extracted train.zip to:", TRAIN_EXTRACT_DIR)
print("Extracted test.zip to:", TEST_EXTRACT_DIR)



## === cell 3
print("Top-level folders under extracted root:")
for p in sorted(os.listdir(EXTRACT_ROOT)):
    print(" -", p)



## === cell 4
testdir = "test/"
traindir = "train/"




## === cell 5
def _find_image_dir(candidates, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    for d in candidates:
        if os.path.isdir(d):
            try:
                files = os.listdir(d)
            except Exception:
                continue
            if any(f.lower().endswith(exts) for f in files):
                return d if d.endswith("/") else (d + "/")

    for base in candidates:
        root = (
            base.rstrip("/").split("/", 1)[0]
            if "/" in base.rstrip("/")
            else base.rstrip("/")
        )
        if not root:
            root = "."
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if any(fn.lower().endswith(exts) for fn in filenames):
                    return dirpath if dirpath.endswith("/") else (dirpath + "/")
    return None


_test_img_dir = _find_image_dir(
    [
        os.path.join(TEST_EXTRACT_DIR, "test"),
        os.path.join(TEST_EXTRACT_DIR, "test/"),
        os.path.join(TEST_EXTRACT_DIR, "test/test"),
        os.path.join(TEST_EXTRACT_DIR, "test/test/unknown"),
        testdir,
        "test/test",
        "test/test/unknown",
        "dogs-vs-cats-redux-kernels-edition/test",
        "dogs-vs-cats-redux-kernels-edition/test/test",
        "dogs-vs-cats-redux-kernels-edition/test/test/unknown",
    ]
)
_train_img_dir = _find_image_dir(
    [
        os.path.join(TRAIN_EXTRACT_DIR, "train"),
        os.path.join(TRAIN_EXTRACT_DIR, "train/"),
        os.path.join(TRAIN_EXTRACT_DIR, "train/train"),
        traindir,
        "train/train",
        "train/cat",
        "train/dog",
        "dogs-vs-cats-redux-kernels-edition/train",
        "dogs-vs-cats-redux-kernels-edition/train/train",
        "dogs-vs-cats-redux-kernels-edition/train/cat",
        "dogs-vs-cats-redux-kernels-edition/train/dog",
    ]
)

if _test_img_dir is None:
    raise FileNotFoundError(
        f"Could not locate extracted test images directory. Searched extracted test at {TEST_EXTRACT_DIR!r} and common nested paths."
    )
if _train_img_dir is None:
    raise FileNotFoundError(
        f"Could not locate extracted train images directory. Searched extracted train at {TRAIN_EXTRACT_DIR!r} and common nested paths."
    )

test_images = [
    os.path.join(_test_img_dir, i)
    for i in os.listdir(_test_img_dir)
    if i.lower().endswith(".jpg")
]

all_images = []
if any(
    os.path.isdir(os.path.join(_train_img_dir, d)) for d in os.listdir(_train_img_dir)
):
    for dirpath, dirnames, filenames in os.walk(_train_img_dir):
        for fn in filenames:
            if fn.lower().endswith(".jpg"):
                all_images.append(os.path.join(dirpath, fn))
else:
    all_images = [
        os.path.join(_train_img_dir, i)
        for i in os.listdir(_train_img_dir)
        if i.lower().endswith(".jpg")
    ]

all_images = sorted(all_images)
test_images = sorted(test_images)

rng = np.random.default_rng(42)
perm = rng.permutation(len(all_images))
all_images = [all_images[i] for i in perm]

limit = int(0.8 * len(all_images))
train_images = all_images[0:limit]
validation_images = all_images[limit:]

print("Train images:", len(train_images))
print("Val images:", len(validation_images))
print("Test images:", len(test_images))
print("Train dir used:", _train_img_dir)
print("Test dir used:", _test_img_dir)

if len(test_images) != 12500:
    print(
        "WARNING: Expected 12500 test images for this competition, but found:",
        len(test_images),
    )
if len(all_images) != 25000:
    print(
        "WARNING: Expected 25000 train images for this competition, but found:",
        len(all_images),
    )



## === cell 6
img = cv2.imread(train_images[1])
plt.imshow(img[:, :, ::-1])  # BGR->RGB for display
plt.axis("off")



## === cell 7
rows, columns = 160, 160



## === cell 8
import os as _os

if tf is None:
    raise ImportError(
        "TensorFlow is unavailable (failed to import in an earlier cell); cannot build tf.data pipelines."
    )

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

from tensorflow.keras.applications import resnet101


def _label_from_path(p):
    b = os.path.basename(p).lower()
    return 1 if "dog" in b else 0


train_labels = np.array([_label_from_path(p) for p in train_images], dtype=np.int32)
val_labels = np.array([_label_from_path(p) for p in validation_images], dtype=np.int32)


def _load_preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [rows, columns], method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cast(img, tf.uint8)  # keep uint8-like values before model preprocessing
    return img


def _train_map(path, y):
    img = _load_preprocess_image(path)
    img = resnet101.preprocess_input(tf.cast(img, tf.float32))
    y = tf.cast(y, tf.float32)
    return img, y


def _test_map(path):
    img = _load_preprocess_image(path)
    img = resnet101.preprocess_input(tf.cast(img, tf.float32))
    return img


train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(
    buffer_size=min(len(train_images), 4096), seed=42, reshuffle_each_iteration=True
)
train_ds = (
    train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((validation_images, val_labels))
val_ds = (
    val_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = tf.data.Dataset.from_tensor_slices(test_images)
test_ds = (
    test_ds.map(_test_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Prepared tf.data datasets.")


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2015533490.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      6[0m [0;31m# Avoid changing protobuf implementation and avoid a second TensorFlow import.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mif[0m [0mtf[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m     raise ImportError(
[0m[1;32m      9[0m         [0;34m"TensorFlow is unavailable (failed to import in an earlier cell); cannot build tf.data pipelines."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     )

[0;31mImportError[0m: TensorFlow is unavailable (failed to import in an earlier cell); cannot build tf.data pipelines.

## === cell 9
for xb, yb in train_ds.take(1):
    print("train batch x:", xb.shape, xb.dtype, "y:", yb.shape, yb.dtype)
