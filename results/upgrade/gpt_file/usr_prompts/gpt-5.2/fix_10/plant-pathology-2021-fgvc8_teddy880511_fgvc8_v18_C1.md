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
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.isfile(SAMPLE_SUB))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print(
            "GPU available:",
            gpus,
            "| mixed_precision policy:",
            mixed_precision.global_policy(),
        )
    except Exception as e:
        print("GPU setup warning:", repr(e))
else:
    print("No GPU detected; running on CPU.")

try:
    tf.config.experimental.enable_op_determinism(False)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA/JIT setup warning:", repr(e))

cpu = os.cpu_count() or 2
tf.config.threading.set_inter_op_parallelism_threads(max(1, cpu // 2))
tf.config.threading.set_intra_op_parallelism_threads(max(1, cpu))

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
print("CACHE_DIR:", CACHE_DIR)




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
]

label_to_idx = {lab: i for i, lab in enumerate(label_class)}
default_idx = 6

train_df["label_num"] = (
    train_df["labels"].map(label_to_idx).fillna(default_idx).astype(int)
)
y_train = tf.keras.utils.to_categorical(
    train_df["label_num"].values, num_classes=len(label_class)
)

print("Train rows:", len(train_df))
print("Label distribution (mapped):")
print(train_df["label_num"].value_counts().sort_index())




## === cell 2
IMG_H, IMG_W = 160, 240
AUTOTUNE = tf.data.AUTOTUNE

train_paths = (TRAIN_IMG_DIR + os.sep + train_df["image"].values).astype(str)
train_labels = y_train.astype(np.float32)


@tf.function(reduce_retracing=True)
def _read_resize_f32(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method="bilinear", antialias=False
    )  # float32
    img = tf.reverse(img, axis=[-1])  # RGB -> BGR
    img.set_shape([IMG_H, IMG_W, 3])
    label = tf.ensure_shape(label, [len(label_class)])
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
try:
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

train_ds = train_ds.with_options(options)

TRAIN_CACHE = os.path.join(CACHE_DIR, "train_cache")
train_ds = train_ds.map(_read_resize_f32, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache(TRAIN_CACHE)

train_ds = train_ds.batch(20, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

if tf.config.list_physical_devices("GPU"):
    try:
        train_ds = train_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        print("Enabled prefetch_to_device(/GPU:0) for train_ds.")
    except Exception as e:
        print("prefetch_to_device unavailable:", repr(e))

print(
    "Prepared train_ds with disk cache + parallel decode/resize + batching + prefetch."
)




## === cell 3
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(IMG_H, IMG_W, 3),
    pooling=None,
    classes=len(label_class),
)

if tf.keras.mixed_precision.global_policy().compute_dtype == "float16":
    x = model.output
    x = tf.keras.layers.Activation("linear", dtype="float32", name="fp32_logits")(x)
    model = tf.keras.Model(model.input, x)

model.compile(
    optimizer="SGD",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

model.fit(train_ds, epochs=35, verbose=2)




## === cell 4
sub_df = pd.read_csv(SAMPLE_SUB)
test_images = sub_df["image"].values
test_paths = (TEST_IMG_DIR + os.sep + test_images).astype(str)


@tf.function(reduce_retracing=True)
def _read_resize_f32_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method="bilinear", antialias=False
    )  # float32
    img = tf.reverse(img, axis=[-1])  # RGB -> BGR
    img.set_shape([IMG_H, IMG_W, 3])
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(options)

TEST_CACHE = os.path.join(CACHE_DIR, "test_cache")
test_ds = test_ds.map(_read_resize_f32_test, num_parallel_calls=AUTOTUNE).cache(
    TEST_CACHE
)

test_ds = test_ds.batch(32, drop_remainder=False).prefetch(AUTOTUNE)

if tf.config.list_physical_devices("GPU"):
    try:
        test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        print("Enabled prefetch_to_device(/GPU:0) for test_ds.")
    except Exception as e:
        print("prefetch_to_device unavailable:", repr(e))

pred = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [label_class[j] for j in pred_idx]

submission = pd.DataFrame({"image": test_images, "labels": pred_labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
