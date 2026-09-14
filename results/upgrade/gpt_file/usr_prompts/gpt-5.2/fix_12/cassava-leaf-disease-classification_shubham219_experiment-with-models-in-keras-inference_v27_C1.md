# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
def _pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


BASE = _pick_existing_path(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)

if BASE is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder in expected locations."
    )

train_csv_path = _pick_existing_path(
    [
        os.path.join(BASE, "train.csv"),
        "../input/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)

train_img_dir = _pick_existing_path(
    [
        os.path.join(BASE, "train_images"),
        "../input/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/cassava-leaf-disease-classification/train_images",
    ]
)

test_img_dir = _pick_existing_path(
    [
        os.path.join(BASE, "test_images"),
        "../input/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
    ]
)

sample_sub_path = _pick_existing_path(
    [
        os.path.join(BASE, "sample_submission.csv"),
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

if (
    train_csv_path is None
    or train_img_dir is None
    or test_img_dir is None
    or sample_sub_path is None
):
    raise FileNotFoundError(
        f"Missing required files/dirs. train_csv={train_csv_path}, train_img_dir={train_img_dir}, "
        f"test_img_dir={test_img_dir}, sample_sub={sample_sub_path}"
    )

train_df = pd.read_csv(train_csv_path)
if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df["path"] = train_img_dir.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(np.int32)
num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "num_classes:", num_classes)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 16
EPOCHS = 2 if not DEBUG else 1



## === cell 2
AUTO = tf.data.AUTOTUNE

tr_paths = tr_df["path"].to_numpy(dtype=object)
tr_labels = tr_df["label"].to_numpy(dtype=np.int32)
va_paths = va_df["path"].to_numpy(dtype=object)
va_labels = va_df["label"].to_numpy(dtype=np.int32)

_seed0 = tf.constant([SEED, 0], dtype=tf.int32)
_pi_over_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_img_h = tf.constant(IMG_SIZE[0], dtype=tf.int32)
_img_w = tf.constant(IMG_SIZE[1], dtype=tf.int32)
_img_h_f = tf.cast(_img_h, tf.float32)
_img_w_f = tf.cast(_img_w, tf.float32)
_out_shape = tf.stack([_img_h, _img_w])


@tf.function
def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


@tf.function
def _path_to_stateless_seed(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    h = tf.cast(h, tf.int32)
    return _seed0 + tf.stack([h, 1])


@tf.function
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([1, 0], tf.int32), minval=-15.0, maxval=15.0
        )
        * _pi_over_180
    )

    tx = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
        )
        * _img_w_f
    )
    ty = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
        )
        * _img_h_f
    )

    z = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )

    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    cx = (_img_w_f - 1.0) / 2.0
    cy = (_img_h_f - 1.0) / 2.0

    invz = 1.0 / z
    a0 = c * invz
    a1 = -s * invz
    b0 = s * invz
    b1 = c * invz

    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=_out_shape,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


@tf.function
def _decode_onehot(path, label):
    img = _read_decode_resize(path)
    y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return img, y


@tf.function
def _decode_seed_label(path, label):
    img = _read_decode_resize(path)
    seed = _path_to_stateless_seed(path)
    return img, seed, label


@tf.function
def _augment_and_onehot(img, seed, label):
    img = _augment(img, seed)
    y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return img, y


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_slack = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.threading.private_threadpool_size = max(4, min(16, (os.cpu_count() or 8)))
    options.threading.max_intra_op_parallelism = 0  # let TF decide
    return options


def _make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    shuffle_buf = int(min(len(paths), 8192))
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_seed_label, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()  # in-memory cache (faster than /kaggle/working file cache)
    ds = ds.map(_augment_and_onehot, num_parallel_calls=AUTO, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.with_options(_dataset_options()).prefetch(AUTO)
    return ds


def _make_valid_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode_onehot, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()  # in-memory cache (no correctness change)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.with_options(_dataset_options()).prefetch(AUTO)
    return ds


train_ds = _make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
valid_ds = _make_valid_ds(va_paths, va_labels, BATCH_SIZE)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3, seed=SEED)(x)
outputs = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

train_steps = int(np.ceil(len(tr_paths) / BATCH_SIZE))
valid_steps = int(np.ceil(len(va_paths) / BATCH_SIZE))

my_model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 3
test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found under: {test_img_dir}")

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_ds(batch_size=128):
    paths = df_test["path"].to_numpy(dtype=object)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.with_options(_dataset_options()).prefetch(AUTO)
    return ds




## === cell 4
test_ds = make_test_ds(batch_size=128)
pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].map(os.path.basename)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

sample_sub = pd.read_csv(sample_sub_path)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
if final_csv["label"].isna().any():
    mode_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(mode_label).astype(int)

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with rows:", len(final_csv))



## === cell 5
final_csv.head()
