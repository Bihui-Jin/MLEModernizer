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
from tensorflow import keras

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_IMG_DIR), f"Missing train image dir: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test image dir: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV), f"Missing train.csv: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv: {SAMPLE_SUB}"

IMG_SIZE = 64

label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
]
label_to_idx = {l: i for i, l in enumerate(label_class)}



## === cell 1
y_train_csv = pd.read_csv(TRAIN_CSV)

labels_arr = y_train_csv["labels"].to_numpy()
label_num = np.fromiter(
    (label_to_idx.get(x, label_to_idx["complex"]) for x in labels_arr),
    dtype=np.int32,
    count=len(labels_arr),
)
y_train = keras.utils.to_categorical(label_num, num_classes=7).astype(np.float32)

train_files = y_train_csv["image"].to_numpy()

train_dir_t = tf.constant(TRAIN_IMG_DIR)


@tf.function
def _load_and_preprocess_from_name(fname, y):
    path = tf.strings.join([train_dir_t, fname], separator=os.sep)
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, y


n = len(train_files)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = max(1, int(0.1 * n))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_files = train_files[tr_idx]
tr_y = y_train[tr_idx]
val_files = train_files[val_idx]
val_y = y_train[val_idx]

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.map_and_batch_fusion = True
options.experimental_optimization.parallel_batch = True

cache_train_path = os.path.join("/kaggle/working", "cache_train_img64")
cache_val_path = os.path.join("/kaggle/working", "cache_val_img64")

train_ds = tf.data.Dataset.from_tensor_slices((tr_files, tr_y)).with_options(options)
train_ds = train_ds.map(_load_and_preprocess_from_name, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache(cache_train_path)
train_ds = train_ds.batch(100, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_y)).with_options(options)
val_ds = val_ds.map(_load_and_preprocess_from_name, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache(cache_val_path)
val_ds = val_ds.batch(128, drop_remainder=False).prefetch(AUTOTUNE)

try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.data.experimental import prefetch_to_device

        train_ds = train_ds.apply(prefetch_to_device("/GPU:0"))
        val_ds = val_ds.apply(prefetch_to_device("/GPU:0"))
except Exception:
    pass



## === cell 2
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=7,
)

model.compile(
    optimizer="SGD",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, epochs=12, verbose=2)




## === cell 3
def f1_mean_per_class_from_counts(tp, fp, fn):
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
    return f1.mean(axis=-1)


val_prob = model.predict(val_ds, verbose=0)
val_true = val_y.astype(np.int32)

grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
thresholds = np.full((7,), 0.5, dtype=np.float32)

val_true_bool = val_true.astype(bool)
P = val_prob  # (N,7)

for c in range(7):
    fixed_pred = P >= thresholds[None, :]  # (N,7) bool
    pred_all = np.repeat(fixed_pred[None, :, :], grid.shape[0], axis=0)  # (G,N,7)
    pred_all[:, :, c] = P[:, c][None, :] >= grid[:, None]

    y_pred = pred_all.astype(np.int32)
    y_true_i = val_true[None, :, :].astype(np.int32)

    tp = (y_pred * y_true_i).sum(axis=1).astype(np.float64)  # (G,7)
    fp = (y_pred * (1 - y_true_i)).sum(axis=1).astype(np.float64)
    fn = ((1 - y_pred) * y_true_i).sum(axis=1).astype(np.float64)

    scores = f1_mean_per_class_from_counts(tp, fp, fn)  # (G,)
    best_idx = int(np.argmax(scores))
    thresholds[c] = float(grid[best_idx])

print("Calibrated thresholds:", thresholds)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_files = sample_sub["image"].to_numpy()

test_dir_t = tf.constant(TEST_IMG_DIR)


@tf.function
def _load_test_from_name(fname):
    path = tf.strings.join([test_dir_t, fname], separator=os.sep)
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


test_options = tf.data.Options()
test_options.experimental_deterministic = True
test_options.experimental_optimization.apply_default_optimizations = True
test_options.experimental_optimization.map_parallelization = True
test_options.experimental_optimization.map_and_batch_fusion = True
test_options.experimental_optimization.parallel_batch = True

cache_test_path = os.path.join("/kaggle/working", "cache_test_img64")

test_ds = tf.data.Dataset.from_tensor_slices(test_files).with_options(test_options)
test_ds = test_ds.map(_load_test_from_name, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache(cache_test_path)
test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)

try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.data.experimental import prefetch_to_device

        test_ds = test_ds.apply(prefetch_to_device("/GPU:0"))
except Exception:
    pass

pred = model.predict(test_ds, verbose=2)  # (N,7) probs

pred_bin = pred >= thresholds[None, :]
label_class_arr = np.array(label_class, dtype=object)

pred_labels = []
for row_bin, row_prob in zip(pred_bin, pred):
    if row_bin.any():
        pred_labels.append(" ".join(label_class_arr[row_bin].tolist()))
    else:
        pred_labels.append(label_class[int(np.argmax(row_prob))])

sub = pd.DataFrame({"image": test_files, "labels": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
