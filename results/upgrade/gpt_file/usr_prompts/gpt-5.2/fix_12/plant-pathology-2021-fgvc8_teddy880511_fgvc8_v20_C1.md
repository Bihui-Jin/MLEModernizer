# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
            "disable_model_pruning": False,
        }
    )
except Exception:
    pass

keras.utils.set_random_seed(42)

print("Python OK; TF:", tf.__version__, "Keras:", keras.__version__)




## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
train_imgpath = os.path.join(BASE_PATH, "train_images")
train_csvpath = os.path.join(BASE_PATH, "train.csv")
test_imgpath = os.path.join(BASE_PATH, "test_images")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(train_imgpath), f"Missing: {train_imgpath}"
assert os.path.exists(train_csvpath), f"Missing: {train_csvpath}"
assert os.path.exists(test_imgpath), f"Missing: {test_imgpath}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"

train_df = pd.read_csv(train_csvpath)
print(train_df.shape)
train_df.head()




## === cell 2
label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
    "scab frog_eye_leaf_spot complex",
    "frog_eye_leaf_spot complex",
    "rust frog_eye_leaf_spot",
    "rust complex",
    "powdery_mildew complex",
]
label_to_idx = {l: i for i, l in enumerate(label_class)}

train_df["label_num"] = train_df["labels"].map(label_to_idx).fillna(11).astype(int)
train_df["labels"].value_counts().head(10)




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = 160, 240
NUM_CLASSES = 12

imgfiles = sorted(os.listdir(train_imgpath))
label_map = dict(
    zip(train_df["image"].values.tolist(), train_df["label_num"].values.tolist())
)
label_keys = set(label_map.keys())
imgfiles = [f for f in imgfiles if f in label_keys]

y_idx = np.fromiter(
    (label_map[f] for f in imgfiles), dtype=np.int32, count=len(imgfiles)
)
train_paths = np.array(
    [os.path.join(train_imgpath, f) for f in imgfiles], dtype=np.str_
)


@tf.function(reduce_retracing=True)
def _load_and_preprocess(path, y):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([IMG_H, IMG_W, 3])
    y = tf.cast(y, tf.int32)
    y.set_shape([])
    return img, y


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_idx))

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = 0  # let TF decide
    options.experimental_optimization.autotune_ram_budget = 0  # let TF decide
    options.experimental_slack = True
except Exception:
    pass
train_ds = train_ds.with_options(options)

cache_path = os.path.join("/kaggle/working", "train_cache_160x240")
train_ds = (
    train_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    .apply(tf.data.experimental.ignore_errors())
    .cache(cache_path)
)

BATCH_SIZE = 64
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

print("Train samples:", len(imgfiles), "y_idx:", y_idx.shape, y_idx.dtype)




## === cell 4
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(160, 240, 3),
    pooling=None,
    classes=12,
)

model.compile(
    optimizer="SGD",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=64,
)

history = model.fit(train_ds, epochs=35, verbose=2)




## === cell 5
test_files = sorted(os.listdir(test_imgpath))
test_paths = np.array(
    [os.path.join(test_imgpath, f) for f in test_files], dtype=np.str_
)


@tf.function(reduce_retracing=True)
def _load_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([IMG_H, IMG_W, 3])
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = 0
    options.experimental_optimization.autotune_ram_budget = 0
    options.experimental_slack = True
except Exception:
    pass
test_ds = test_ds.with_options(options)

test_cache_path = os.path.join("/kaggle/working", "test_cache_160x240")
test_ds = (
    test_ds.map(_load_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .apply(tf.data.experimental.ignore_errors())
    .cache(test_cache_path)
    .batch(64, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

pred = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [label_class[i] for i in pred_idx]

sub_pred = pd.DataFrame({"image": test_files, "labels": pred_labels})

sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["image"]].merge(sub_pred, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
