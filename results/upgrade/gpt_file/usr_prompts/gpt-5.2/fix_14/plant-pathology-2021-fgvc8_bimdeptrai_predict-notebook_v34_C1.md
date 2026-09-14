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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
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
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    options = None




## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
train.head()




## === cell 2
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)
labels = pd.DataFrame(y, columns=classes)
labels.head()




## === cell 3
h_target = 256
w_target = 256
batch_size = 32

idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

train_df[classes] = labels.loc[tr_idx, classes].to_numpy()
val_df[classes] = labels.loc[val_idx, classes].to_numpy()

train_df.head()




## === cell 4
def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        (h_target, w_target),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, seed2):
    angle = tf.random.stateless_uniform([], seed=seed2, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0
    a0 = c
    a1 = -s
    a2 = cx - c * cx + s * cy
    b0 = s
    b1 = c
    b2 = cy - s * cx - c * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    max_dx = tf.cast(tf.round(0.05 * tf.cast(w_target, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h_target, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([0, 1], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    scale = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([2, 2], tf.int32), minval=0.9, maxval=1.1
    )
    new_h = tf.cast(tf.round(scale * tf.cast(h_target, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(w_target, tf.float32)), tf.int32)
    img2 = tf.image.resize(img, (new_h, new_w), method="bilinear", antialias=False)
    img2 = tf.image.resize_with_crop_or_pad(img2, h_target, w_target)

    do_flip = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([3, 3], tf.int32), minval=0.0, maxval=1.0
    )
    img2 = tf.cond(do_flip < 0.5, lambda: img2, lambda: tf.image.flip_left_right(img2))
    return img2


USE_MEMORY_CACHE = False
CACHE_DIR = "./tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    _tfdata_threads = max(2, (os.cpu_count() or 4) // 2)
except Exception:
    _tfdata_threads = 4

if options is None:
    options = tf.data.Options()
    options.deterministic = True

try:
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

options.threading.private_threadpool_size = _tfdata_threads
options.threading.max_intra_op_parallelism = 1

_PARALLEL_CALLS = _tfdata_threads


def _cache_ds(ds, cache_path=None):
    if USE_MEMORY_CACHE:
        return ds.cache()
    if cache_path is None:
        return ds
    return ds.cache(cache_path)


@tf.function(jit_compile=False)
def _decode_only_map(path, y):
    return _read_decode_resize(path), y


@tf.function(jit_compile=False)
def _read_only_map(path):
    return _read_decode_resize(path)


@tf.function(jit_compile=False)
def _aug_map(i, xy):
    img, y = xy
    seed2 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
    img = _augment(img, seed2)
    return img, y


def make_ds(df, img_dir, training, cache_name):
    img_dir2 = img_dir.rstrip("/") + "/"

    img_names = df["image"].to_numpy(dtype=str, copy=False)
    paths_np = np.char.add(img_dir2, img_names).astype(str, copy=False)
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)

    ys_np = df[classes].to_numpy(dtype=np.float32, copy=False)
    ys = tf.convert_to_tensor(ys_np, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    ds = ds.with_options(options)

    ds = ds.map(
        _decode_only_map, num_parallel_calls=_PARALLEL_CALLS, deterministic=True
    )
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.cache")
    ds = _cache_ds(ds, cache_path=cache_path)

    if training:
        shuffle_buf = min(len(df), 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.enumerate()
        ds = ds.map(_aug_map, num_parallel_calls=_PARALLEL_CALLS, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=bool(training)).prefetch(_PARALLEL_CALLS)
    return ds


train_gen = make_ds(train_df, TRAIN_IMG_DIR, training=True, cache_name="train_decoded")
val_gen = make_ds(val_df, TRAIN_IMG_DIR, training=False, cache_name="val_decoded")




## === cell 5
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
)
base.trainable = False  # keep training stable and fast under 600s

inp = keras.Input(shape=(h_target, w_target, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 6
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    verbose=1,
)




## === cell 7
submissions = pd.read_csv(SAMPLE_SUB)
submissions.head()




## === cell 8
test_dir2 = TEST_IMG_DIR.rstrip("/") + "/"

test_img_names = submissions["image"].to_numpy(dtype=str, copy=False)
test_paths_np = np.char.add(test_dir2, test_img_names).astype(str, copy=False)
test_paths = tf.convert_to_tensor(test_paths_np, dtype=tf.string)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(options)

test_ds = test_ds.map(
    _read_only_map, num_parallel_calls=_PARALLEL_CALLS, deterministic=True
)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(_PARALLEL_CALLS)

preds = model.predict(test_ds, verbose=1)
preds.shape




## === cell 9
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.7,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.5,
    "scab": 0.5,
}

thr_vec = np.array([thresh[c] for c in classes], dtype=np.float32)




## === cell 10
pred_bin = preds >= thr_vec[None, :]

healthy_idx = classes.index("healthy") if "healthy" in classes else None

row_max_idx = np.argmax(preds, axis=1)
if healthy_idx is not None:
    is_healthy_max = row_max_idx == healthy_idx
else:
    is_healthy_max = np.zeros((preds.shape[0],), dtype=bool)

chosen_mask = pred_bin.copy()

empty = chosen_mask.sum(axis=1) == 0
if healthy_idx is not None:
    includes_healthy = chosen_mask[:, healthy_idx]
else:
    includes_healthy = np.zeros((preds.shape[0],), dtype=bool)

need_fallback = empty | includes_healthy
row_max = preds.max(axis=1, keepdims=True)
fallback_mask = preds >= row_max  # includes ties

final_mask = np.where(need_fallback[:, None], fallback_mask, chosen_mask)

classes_arr = np.asarray(classes, dtype=object)

pos_indices = [np.flatnonzero(row) for row in final_mask]
pred_labels = []
for i, inds in enumerate(pos_indices):
    if is_healthy_max[i]:
        pred_labels.append("healthy")
        continue
    if inds.size == 0:
        pred_labels.append("healthy")
    else:
        s = " ".join(classes_arr[inds].tolist()).strip()
        pred_labels.append(s if s != "" else "healthy")

submissions = submissions.copy()
submissions["labels"] = pred_labels
submissions[["image", "labels"]].to_csv("submission.csv", index=False)

submissions.head()




## === cell 11
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
print(check.head())
print("Unique label strings (sample):", check["labels"].head(10).tolist())
