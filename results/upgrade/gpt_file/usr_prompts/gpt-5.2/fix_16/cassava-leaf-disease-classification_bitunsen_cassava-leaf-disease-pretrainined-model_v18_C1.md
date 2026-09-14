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
import json
import random
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

assert os.path.isdir(BASE_DIR), f"BASE_DIR not found: {BASE_DIR}"
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"



## === cell 2
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false"
)

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"



## === cell 4
pass



## === cell 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16

PRE_TRAINED_MODEL = "../input/unionmodelv05/Cassava_Best_UnitedModel_V05.hdf5"
print("Pretrained model exists?:", os.path.exists(PRE_TRAINED_MODEL))

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS



## === cell 6
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(int)
train_df["filepath"] = (TRAIN_DIR + train_df["image_id"].astype(str)).astype(str)

fps = train_df["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.exists(p) for p in fps), dtype=bool, count=fps.shape[0]
)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

print(
    "Train rows:",
    len(train_df),
    "Unique labels:",
    sorted(train_df["label"].unique().tolist()),
)



## === cell 7
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_df = sample_sub[["image_id"]].copy()
test_df["filepath"] = (TEST_DIR + test_df["image_id"].astype(str)).astype(str)

fps = test_df["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.exists(p) for p in fps), dtype=bool, count=fps.shape[0]
)
test_df = test_df.loc[exists_mask].reset_index(drop=True)

print("Test rows:", len(test_df))




## === cell 8
@tf.function
def decode_and_resize(path, label=None, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.12)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def stratified_split_df(df, label_col="label", test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx = []
    valid_idx = []
    for lab, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_valid = int(np.floor(len(idx) * test_size))
        valid_idx.extend(idx[:n_valid].tolist())
        train_idx.extend(idx[n_valid:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[valid_idx].reset_index(
        drop=True
    )


train_part, valid_part = stratified_split_df(
    train_df, label_col="label", test_size=0.2, seed=42
)

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
except Exception:
    pass

TFREC_TRAIN_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(BASE_DIR, "test_tfrecords")
assert os.path.isdir(TFREC_TRAIN_DIR), f"train_tfrecords not found: {TFREC_TRAIN_DIR}"
assert os.path.isdir(TFREC_TEST_DIR), f"test_tfrecords not found: {TFREC_TEST_DIR}"

train_tfrecs = sorted(
    [
        os.path.join(TFREC_TRAIN_DIR, f)
        for f in os.listdir(TFREC_TRAIN_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TFREC_TEST_DIR, f)
        for f in os.listdir(TFREC_TEST_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"

FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "class": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}
FEATURES_TEST = {"image": tf.io.FixedLenFeature([], tf.string)}


@tf.function
def _parse_train_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES_TRAIN)
    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    y_target = tf.cast(x["target"], tf.int32)
    y_class = tf.cast(x["class"], tf.int32)
    y_label = tf.cast(x["label"], tf.int32)

    y = tf.where(y_target >= 0, y_target, tf.where(y_class >= 0, y_class, y_label))

    y = tf.where(y >= 0, y, tf.zeros_like(y))
    return img, y


@tf.function
def _parse_test_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES_TEST)
    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment_img(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.12)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, y


train_paths = train_part["filepath"].astype(str).to_numpy()
train_labels = train_part["label"].astype(np.int32).to_numpy()
valid_paths = valid_part["filepath"].astype(str).to_numpy()
valid_labels = valid_part["label"].astype(np.int32).to_numpy()

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    options
)
ds_train = ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)
ds_train = ds_train.map(
    lambda p, y: decode_and_resize(p, y, training=True), num_parallel_calls=AUTOTUNE
)
ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

ds_valid = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).with_options(
    options
)
ds_valid = ds_valid.map(
    lambda p, y: decode_and_resize(p, y, training=False), num_parallel_calls=AUTOTUNE
)
ds_valid = ds_valid.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

ds_test_tfr = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(options)
ds_test_tfr = ds_test_tfr.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
ds_test_tfr = ds_test_tfr.cache()

try:
    n_test_tfr = int(tf.data.experimental.cardinality(ds_test_tfr).numpy())
    print(
        "TFRecord test count:", n_test_tfr, "Sample submission count:", len(sample_sub)
    )
except Exception as e:
    print("Could not compute TFRecord test count:", repr(e))



## === cell 9
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
EPOCHS = 8
history = model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, verbose=2)



## === cell 11
GLOBAL_SEED = 42
N_AUG = 5
N_VIEWS = 1 + N_AUG


@tf.function
def _tta_views_batch(imgs, idxs):
    imgs = tf.convert_to_tensor(imgs, tf.float32)
    idxs = tf.cast(idxs, tf.int32)
    b = tf.shape(imgs)[0]

    views0 = imgs[:, None, :, :, :]  # [B,1,H,W,3]
    out = tf.broadcast_to(imgs[:, None, :, :, :], [b, N_AUG, IMG_HEIGHT, IMG_WIDTH, 3])

    js = tf.range(N_AUG, dtype=tf.int32)[None, :]  # [1, N_AUG]
    idxs2 = idxs[:, None]  # [B,1]
    seed_second = idxs2 * 1000 + js  # [B,N_AUG] unique per (image, aug)

    def _stateless_uniform(shape, seed_add):
        seed0 = tf.fill([b, N_AUG], tf.cast(GLOBAL_SEED, tf.int32)) + tf.cast(
            seed_add, tf.int32
        )
        seed1 = seed_second + tf.cast(seed_add * 997, tf.int32)
        seeds = tf.stack([seed0, seed1], axis=-1)  # [B,N_AUG,2]
        seeds = tf.reshape(seeds, [b * N_AUG, 2])  # [B*N_AUG,2]
        r = tf.random.stateless_uniform([b * N_AUG], seed=seeds, dtype=tf.float32)
        return tf.reshape(r, [b, N_AUG])

    r = _stateless_uniform([b, N_AUG], 11)
    do = r < 0.5
    out = tf.where(do[:, :, None, None, None], tf.reverse(out, axis=[3]), out)

    r = _stateless_uniform([b, N_AUG], 22)
    do = r < 0.2
    out = tf.where(do[:, :, None, None, None], tf.reverse(out, axis=[2]), out)

    r = _stateless_uniform([b, N_AUG], 33)
    do = r < 0.5
    r_delta = _stateless_uniform([b, N_AUG], 44)
    delta = (r_delta * 0.24) - 0.12
    out = tf.where(
        do[:, :, None, None, None],
        tf.clip_by_value(out + delta[:, :, None, None, None], 0.0, 1.0),
        out,
    )

    r = _stateless_uniform([b, N_AUG], 55)
    do = r < 0.5
    r_c = _stateless_uniform([b, N_AUG], 66)
    c = 0.85 + r_c * (1.15 - 0.85)
    mean = tf.reduce_mean(out, axis=[2, 3], keepdims=True)  # [B,N_AUG,1,1,3]
    out_contrast = tf.clip_by_value(
        (out - mean) * c[:, :, None, None, None] + mean, 0.0, 1.0
    )
    out = tf.where(do[:, :, None, None, None], out_contrast, out)

    views = tf.concat([views0, out], axis=1)  # [B,N_VIEWS,H,W,3]
    return tf.reshape(views, [b * N_VIEWS, IMG_HEIGHT, IMG_WIDTH, 3])


def _collect_test_images_from_tfrecord(ds, n_expected):
    imgs = []
    for img in ds:
        imgs.append(img.numpy())
    assert len(imgs) == n_expected, (len(imgs), n_expected)
    return np.stack(imgs, axis=0).astype(np.float32, copy=False)


test_imgs_np = _collect_test_images_from_tfrecord(ds_test_tfr, len(test_df))
pred_ids = test_df["image_id"].astype(str).tolist()

ds_test_imgs = tf.data.Dataset.from_tensor_slices(test_imgs_np).with_options(options)
ds_test_imgs = ds_test_imgs.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
ds_test_imgs = ds_test_imgs.enumerate()  # (batch_idx, img_batch)


@tf.function
def _batch_img_to_views(batch_idx, imgs):
    b = tf.shape(imgs)[0]
    start = tf.cast(batch_idx, tf.int32) * tf.cast(batch_size, tf.int32)
    idxs = start + tf.range(b, dtype=tf.int32)  # global per-image indices
    views_flat = _tta_views_batch(imgs, idxs)  # [B*N_VIEWS,H,W,3]
    return views_flat


ds_views_flat = ds_test_imgs.map(
    lambda bi, imgs: _batch_img_to_views(bi, imgs), num_parallel_calls=AUTOTUNE
)
ds_views_flat = ds_views_flat.prefetch(AUTOTUNE)

preds_flat = model.predict(ds_views_flat, verbose=0)

n_images = len(pred_ids)
assert preds_flat.shape[0] == n_images * N_VIEWS, (preds_flat.shape, n_images, N_VIEWS)

preds = preds_flat.reshape(n_images, N_VIEWS, NUM_CLASSES)
preds_mean = preds.mean(axis=1)
pred_labels = preds_mean.argmax(axis=1).astype(np.int32, copy=False).tolist()

submission = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notnull().all(), "Some test image_ids were not predicted."
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print(submission.shape)



## === cell 12
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"], chk.columns
assert len(chk) == len(sample_sub), (len(chk), len(sample_sub))
assert (
    chk["image_id"].astype(str).tolist() == sample_sub["image_id"].astype(str).tolist()
)
assert chk["label"].between(0, NUM_CLASSES - 1).all()
print(chk.head(3))
print("Submission OK:", chk.shape)
