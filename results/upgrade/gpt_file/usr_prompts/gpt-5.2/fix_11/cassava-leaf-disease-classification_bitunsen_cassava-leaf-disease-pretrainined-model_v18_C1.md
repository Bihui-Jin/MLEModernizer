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
from PIL import Image

import tensorflow as tf
import keras
from keras import layers

keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



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

x_tr = train_part["filepath"].values
y_tr = train_part["label"].values
x_va = valid_part["filepath"].values
y_va = valid_part["label"].values

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
except Exception:
    pass


@tf.function
def _decode_only(p, y):
    img = decode_and_resize(p, label=None, training=False)
    return img, tf.cast(y, tf.int32)


@tf.function
def _augment_only(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.12)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, y


ds_train = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
ds_train = ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)
ds_train = ds_train.with_options(options)
ds_train = ds_train.map(_decode_only, num_parallel_calls=AUTOTUNE)

ds_train = ds_train.cache()

ds_train = ds_train.map(_augment_only, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(batch_size, drop_remainder=False)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_valid = tf.data.Dataset.from_tensor_slices((x_va, y_va))
ds_valid = ds_valid.with_options(options)
ds_valid = ds_valid.map(
    lambda p, y: decode_and_resize(p, y, training=False), num_parallel_calls=AUTOTUNE
)

ds_valid = ds_valid.cache()

ds_valid = ds_valid.batch(batch_size, drop_remainder=False)
ds_valid = ds_valid.prefetch(AUTOTUNE)



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
def _tta_views_tf(img, idx):
    img = tf.convert_to_tensor(img, tf.float32)  # [H,W,3]
    views0 = tf.expand_dims(img, axis=0)  # [1,H,W,3]

    js = tf.range(N_AUG, dtype=tf.int32)  # [5]
    idx = tf.cast(idx, tf.int32)
    seed_second = idx * 1000 + js  # [5]
    seed_base = tf.stack(
        [tf.fill([N_AUG], tf.cast(GLOBAL_SEED, tf.int32)), seed_second], axis=1
    )  # [5,2]

    out = tf.broadcast_to(img, [N_AUG, IMG_HEIGHT, IMG_WIDTH, 3])  # [5,H,W,3]

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([1, 11], tf.int32)
    )
    do = r < 0.5
    out = tf.where(do[:, None, None, None], tf.reverse(out, axis=[2]), out)

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([2, 22], tf.int32)
    )
    do = r < 0.2
    out = tf.where(do[:, None, None, None], tf.reverse(out, axis=[1]), out)

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([3, 33], tf.int32)
    )
    do = r < 0.5
    delta = tf.random.stateless_uniform(
        [N_AUG],
        seed=seed_base + tf.constant([4, 44], tf.int32),
        minval=-0.12,
        maxval=0.12,
    )
    out = tf.where(
        do[:, None, None, None],
        tf.clip_by_value(out + delta[:, None, None, None], 0.0, 1.0),
        out,
    )

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([5, 55], tf.int32)
    )
    do = r < 0.5
    c = tf.random.stateless_uniform(
        [N_AUG],
        seed=seed_base + tf.constant([6, 66], tf.int32),
        minval=0.85,
        maxval=1.15,
    )
    mean = tf.reduce_mean(out, axis=[1, 2], keepdims=True)  # [5,1,1,3]
    out_contrast = tf.clip_by_value(
        (out - mean) * c[:, None, None, None] + mean, 0.0, 1.0
    )
    out = tf.where(do[:, None, None, None], out_contrast, out)

    return tf.concat([views0, out], axis=0)  # [6,H,W,3]


@tf.function
def _tta_views_batch(imgs, idxs):
    views = tf.map_fn(
        lambda x: _tta_views_tf(x[0], x[1]),
        (imgs, idxs),
        fn_output_signature=tf.float32,
        parallel_iterations=16,
    )  # [B, N_VIEWS, H, W, 3]
    b = tf.shape(imgs)[0]
    return tf.reshape(views, [b * N_VIEWS, IMG_HEIGHT, IMG_WIDTH, 3])


test_image_ids = test_df["image_id"].values
test_paths = test_df["filepath"].values

ds_test = tf.data.Dataset.from_tensor_slices((test_image_ids, test_paths))
ds_test = ds_test.with_options(options)


@tf.function
def _load_only(image_id, path):
    img = decode_and_resize(path, label=None, training=False)  # [H,W,3]
    return image_id, img


ds_test = ds_test.map(_load_only, num_parallel_calls=AUTOTUNE)

ds_test = ds_test.cache()

ds_test = ds_test.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
ds_test = ds_test.enumerate()  # (batch_idx, (image_id_batch, img_batch))


@tf.function
def _batch_to_views(batch_idx, pair):
    image_ids, imgs = pair  # image_ids: [B], imgs: [B,H,W,3]
    b = tf.shape(imgs)[0]
    start = tf.cast(batch_idx, tf.int32) * tf.cast(batch_size, tf.int32)
    idxs = start + tf.range(b, dtype=tf.int32)  # global per-image indices
    views_flat = _tta_views_batch(imgs, idxs)  # [B*N_VIEWS,H,W,3]
    return image_ids, views_flat


ds_test_views = ds_test.map(_batch_to_views, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

ds_views_flat = ds_test_views.map(
    lambda image_ids, views_flat: views_flat, num_parallel_calls=AUTOTUNE
).unbatch()
ds_views_flat = ds_views_flat.batch(
    batch_size * N_VIEWS, drop_remainder=False
).prefetch(AUTOTUNE)

preds_flat = model.predict(
    ds_views_flat, verbose=0
)  # [N_images*N_VIEWS, NUM_CLASSES] in-order

pred_ids = test_df["image_id"].astype(str).tolist()
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
chk.head(3)
