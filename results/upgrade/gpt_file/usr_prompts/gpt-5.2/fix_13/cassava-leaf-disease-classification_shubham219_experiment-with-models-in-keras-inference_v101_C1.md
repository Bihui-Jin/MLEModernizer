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

3.11

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
import os, random, glob, sys

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory. Tried:\n"
        + "\n".join(DATA_ROOT_CANDIDATES)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"train.csv not found at: {TRAIN_CSV}")
if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"train_images directory not found at: {TRAIN_IMG_DIR}")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"test_images directory not found at: {TEST_IMG_DIR}")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)



## === cell 1
import matplotlib.pyplot as plt  # kept (even if unused) to preserve original intent
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

preprocess = tf.keras.applications.resnet50.preprocess_input



## === cell 2
num_classes = 5

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    pooling="avg",
)

base.trainable = False

x = tfl.Dropout(0.2)(base.output)
out = tfl.Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

my_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

my_model.summary()



## === cell 3
df_train = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(df_train.columns):
    raise ValueError(f"Unexpected train.csv columns: {df_train.columns.tolist()}")

df_train["label"] = df_train["label"].astype(np.int32)
df_train["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/") + df_train["image_id"].astype(str)

train_df, val_df = train_test_split(
    df_train[["path", "label"]],
    test_size=0.1,
    random_state=SEED,
    stratify=df_train["label"],
)

BATCH_SIZE = 16
EPOCHS = 3
IMG_SIZE = (256, 256)
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img


@tf.function
def _augment(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    max_dx = tf.cast(0.05 * w, tf.int32)
    max_dy = tf.cast(0.05 * h, tf.int32)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([11, 17], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([19, 23], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    z = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([29, 31], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    z = tf.clip_by_value(z, 0.9, 1.1)
    new_h = tf.cast(tf.round(h / z), tf.int32)
    new_w = tf.cast(tf.round(w / z), tf.int32)
    new_h = tf.minimum(new_h, tf.shape(img)[0])
    new_w = tf.minimum(new_w, tf.shape(img)[1])

    img = tf.image.stateless_random_crop(
        img, size=[new_h, new_w, 3], seed=seed2 + tf.constant([37, 41], tf.int32)
    )
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img


@tf.function
def _preprocess_img(img):
    img = tf.cast(img, tf.float32)
    img = preprocess(img)
    return img


@tf.function
def _train_map_from_img(img, path, label, epoch_tensor):
    seed1 = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed2 = tf.stack(
        [
            tf.cast(seed1, tf.int32),
            tf.cast(SEED, tf.int32) + tf.cast(epoch_tensor, tf.int32),
        ],
        axis=0,
    )
    img = _augment(img, seed2)
    img = _preprocess_img(img)
    return img, label


@tf.function
def _eval_map_from_img(img, label):
    img = _preprocess_img(img)
    return img, label


@tf.function
def _test_map_from_img(img):
    img = _preprocess_img(img)
    return img


def _with_fast_deterministic_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    except Exception:
        pass
    try:
        opts.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    return ds.with_options(opts)


path_np_full = train_df["path"].to_numpy(dtype=object)
label_np_full = train_df["label"].to_numpy(dtype=np.int32)

base_train_ds = tf.data.Dataset.from_tensor_slices((path_np_full, label_np_full))
base_train_ds = _with_fast_deterministic_options(base_train_ds)

train_img_cache_ds = base_train_ds.map(
    lambda p, y: (_read_decode_resize(p), p, y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache()
train_img_cache_ds = _with_fast_deterministic_options(train_img_cache_ds)


def make_train_ds(epoch):
    ds = train_img_cache_ds
    shuffle_buf = min(len(train_df), 8192)
    ds = ds.shuffle(
        buffer_size=shuffle_buf,
        seed=SEED + int(epoch),
        reshuffle_each_iteration=True,
    )
    epoch_tensor = tf.constant(int(epoch), dtype=tf.int32)
    ds = ds.map(
        lambda img, p, y: _train_map_from_img(img, p, y, epoch_tensor),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


def make_val_ds(df):
    path_np = df["path"].to_numpy(dtype=object)
    label_np = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((path_np, label_np))
    ds = ds.map(
        lambda p, y: (_read_decode_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).cache()
    ds = ds.map(_eval_map_from_img, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


val_ds = make_val_ds(val_df)

for epoch in range(EPOCHS):
    train_ds = make_train_ds(epoch=epoch)
    my_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch + 1,
        initial_epoch=epoch,
        verbose=1,
    )



## === cell 4
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found under: {TEST_IMG_DIR}")

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_ds(paths, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(paths, dtype=object))
    ds = ds.map(
        _read_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True
    ).cache()
    ds = ds.map(_test_map_from_img, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds




## === cell 5
test_ds = make_test_ds(df_test["path"].to_numpy(dtype=object), batch_size=16)
pred_test = my_model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1)

image_ids = df_test["path"].map(os.path.basename).to_numpy(dtype=object)

final_csv = pd.DataFrame({"image_id": image_ids, "label": pred_test_labels.astype(int)})
final_csv.to_csv("submission.csv", index=False)

final_csv.head()



## === cell 6
print("Submission shape:", final_csv.shape)
print("Saved to:", os.path.abspath("submission.csv"))
print(final_csv.dtypes)
final_csv.head()
