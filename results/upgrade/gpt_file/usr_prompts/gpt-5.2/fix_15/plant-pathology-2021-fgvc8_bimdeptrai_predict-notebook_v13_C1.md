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
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not enabled due to environment incompatibility:", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print("train:", train.shape, "submissions:", submissions.shape)
print(train.head())
print(submissions.head())




## === cell 2
h_target = 512
w_target = 512
batch_size = 32

N_WORKERS = max(2, (os.cpu_count() or 4) - 1)
USE_MULTIPROCESSING = True
MAX_QUEUE_SIZE = 16

AUTOTUNE = tf.data.AUTOTUNE




## === cell 3
label_split = train["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
num_classes = len(class_names)

print("num_classes:", num_classes)
print("classes:", class_names)

train_df = train.copy()
for i, c in enumerate(class_names):
    train_df[c] = y[:, i].astype(np.float32)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print("tr_df:", tr_df.shape, "va_df:", va_df.shape)




## === cell 4
def _build_paths(df, directory):
    return np.asarray(
        [os.path.join(directory, f) for f in df["image"].values], dtype=object
    )


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img


@tf.function(reduce_retracing=True)
def _rotate_via_projective(img, angle_rad):
    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


@tf.function(reduce_retracing=True)
def _augment_stateless_no_tfa(img, seed):
    seed = tf.cast(seed, tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + 1, minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    img = _rotate_via_projective(img, angle)

    zoom = tf.random.stateless_uniform([], seed=seed + 2, minval=0.90, maxval=1.10)
    dx = tf.random.stateless_uniform(
        [], seed=seed + 3, minval=-0.05, maxval=0.05
    ) * tf.cast(w_target, tf.float32)
    dy = tf.random.stateless_uniform(
        [], seed=seed + 4, minval=-0.05, maxval=0.05
    ) * tf.cast(h_target, tf.float32)

    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0
    inv_zoom = 1.0 / zoom
    a0 = inv_zoom
    a1 = 0.0
    a2 = cx - inv_zoom * cx - dx
    b0 = 0.0
    b1 = inv_zoom
    b2 = cy - inv_zoom * cy - dy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


def _with_fast_options(ds, deterministic=True):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic

    try:
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    try:
        options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    except Exception:
        pass
    try:
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    return ds.with_options(options)


tr_paths = _build_paths(tr_df, TRAIN_DIR)
tr_labels = tr_df[class_names].to_numpy(dtype=np.float32, copy=False)

va_paths = _build_paths(va_df, TRAIN_DIR)
va_labels = va_df[class_names].to_numpy(dtype=np.float32, copy=False)

te_paths = _build_paths(submissions, TEST_DIR)


def make_train_ds_correct_from_arrays(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_decode(path, y):
        return _decode_resize(path), y

    ds = ds.map(_map_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # in-memory cache of (image, label) after decode+resize

    shuffle_buf = min(len(paths), 4096)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    @tf.function(reduce_retracing=True)
    def _map_aug(i, x):
        img, y = x
        seed = tf.stack([tf.cast(SEED, tf.int64), tf.cast(i, tf.int64)])
        img = _augment_stateless_no_tfa(img, tf.cast(seed, tf.int32))
        return img, y

    ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE)

    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_options(ds, deterministic=True)
    return ds


def make_valid_ds_from_arrays(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path, y):
        return _decode_resize(path), y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _with_fast_options(ds, deterministic=True)
    return ds


def make_test_ds_from_arrays(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        return _decode_resize(path)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _with_fast_options(ds, deterministic=True)
    return ds


train_ds = make_train_ds_correct_from_arrays(tr_paths, tr_labels)
valid_ds = make_valid_ds_from_arrays(va_paths, va_labels)
test_ds = make_test_ds_from_arrays(te_paths)

train_steps = int(np.ceil(len(tr_df) / batch_size))
valid_steps = int(np.ceil(len(va_df) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))

print("steps:", {"train": train_steps, "valid": valid_steps, "test": test_steps})




## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
x = base.output
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = tf.keras.Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=valid_ds,
    validation_steps=valid_steps,
    epochs=3,
    verbose=1,
)




## === cell 6
def _f1_binary_vec(y_true_01, y_pred_01, eps=1e-12):
    tp = np.sum((y_true_01 == 1) & (y_pred_01 == 1))
    fp = np.sum((y_true_01 == 0) & (y_pred_01 == 1))
    fn = np.sum((y_true_01 == 1) & (y_pred_01 == 0))
    return (2.0 * tp) / (2.0 * tp + fp + fn + eps)


y_val_true = va_df[class_names].values.astype(np.int32)

val_preds = model.predict(
    valid_ds,
    steps=valid_steps,
    verbose=1,
)
print("val_preds shape:", val_preds.shape, "y_val_true shape:", y_val_true.shape)

th_grid = np.round(np.linspace(0.10, 0.60, 11), 2).astype(np.float32)

best_th = np.full(num_classes, 0.30, dtype=np.float32)
best_f1 = np.full(num_classes, -1.0, dtype=np.float32)

pred_ge = val_preds[None, :, :] >= th_grid[:, None, None]

for j in range(num_classes):
    yj = y_val_true[:, j].astype(bool)
    if yj.sum() < 3:
        continue

    pj = pred_ge[:, :, j]
    tp = np.sum(pj & yj[None, :], axis=1).astype(np.float64)
    fp = np.sum(pj & (~yj[None, :]), axis=1).astype(np.float64)
    fn = np.sum((~pj) & yj[None, :], axis=1).astype(np.float64)

    f1s = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
    k = int(np.argmax(f1s))
    best_f1[j] = f1s[k].astype(np.float32)
    best_th[j] = th_grid[k]

print("Per-class thresholds (first 10):", list(zip(class_names[:10], best_th[:10])))
print("Per-class val F1 (first 10):", list(zip(class_names[:10], best_f1[:10])))




## === cell 7
preds = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
print("preds shape:", preds.shape)

pred_df = pd.DataFrame(preds, columns=class_names)
print(pred_df.head())




## === cell 8
if "pred_df" not in globals() or pred_df is None:
    print(
        "WARNING: pred_df missing; creating zero predictions fallback to produce a valid submission.csv."
    )
    pred_df = pd.DataFrame(
        np.zeros((len(submissions), num_classes), dtype=np.float32), columns=class_names
    )

if "best_th" not in globals() or best_th is None:
    best_th = np.full(num_classes, 0.30, dtype=np.float32)

best_th_arr = best_th.astype(np.float32)
preds_arr = pred_df.values.astype(np.float32)

mask = preds_arr >= best_th_arr[None, :]

row_counts = mask.sum(axis=1)
need_argmax = row_counts == 0
if np.any(need_argmax):
    argm = np.argmax(preds_arr[need_argmax], axis=1)
    mask[need_argmax, :] = False
    mask[need_argmax, argm] = True

if "healthy" in class_names:
    healthy_idx = class_names.index("healthy")
    has_healthy = mask[:, healthy_idx]
    has_other = mask.sum(axis=1) > 1
    drop_healthy = has_healthy & has_other
    if np.any(drop_healthy):
        mask[drop_healthy, healthy_idx] = False

out_labels = []
for i in range(mask.shape[0]):
    idxs = np.flatnonzero(mask[i])
    out_labels.append(" ".join([class_names[j] for j in idxs]))

submissions.loc[:, "labels"] = out_labels
print(submissions.head())

submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head(10).to_string(index=False))
