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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"



## === cell 2
import cv2  # kept to preserve environment parity; no longer used in the optimized train pipeline



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))

label_list = [int(key) for key in map_classes.keys()]
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 4
print("TRAIN_DIR exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR exists:", os.path.isdir(TEST_DIR))



## === cell 5
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv5/Cassava_Best_Xception_Model_V04.hdf5"



## === cell 6
pass



## === cell 7
pass



## === cell 8
import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


def _tf_decode_center_crop_to_float(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.convert_image_dtype(img, tf.float32)  # float32 in [0,1]
    img = tf.image.resize_with_crop_or_pad(
        img, IMG_HEIGHT, IMG_WIDTH
    )  # center crop/pad
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return img


def _center_crop_or_pad(img):
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return img


def _rand_uniform(seed2):
    return tf.random.stateless_uniform([], seed2, dtype=tf.float32)


def _rand_bool(seed2, p):
    return _rand_uniform(seed2) < tf.constant(p, tf.float32)


@tf.function
def _augment_train_tf(img, label, idx):
    s0 = tf.cast(SEED, tf.int32)
    i = tf.cast(idx, tf.int32)

    use_train_aug = _rand_bool(tf.stack([s0, i * 9973 + 1]), 0.5)

    def _train_branch():
        x = img

        do_flip = _rand_bool(tf.stack([s0, i * 9973 + 2]), 0.5)
        x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)

        do_bc = _rand_bool(tf.stack([s0, i * 9973 + 3]), 0.5)

        def _apply_bc():
            delta = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, i * 9973 + 4]), minval=-0.2, maxval=0.2
            )
            y = tf.image.adjust_brightness(x, delta)
            cf = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, i * 9973 + 5]), minval=0.8, maxval=1.2
            )
            y = tf.image.adjust_contrast(y, cf)
            return tf.clip_by_value(y, 0.0, 1.0)

        x = tf.cond(do_bc, _apply_bc, lambda: x)

        x = _center_crop_or_pad(x)

        do_ssr = _rand_bool(tf.stack([s0, i * 9973 + 6]), 0.5)

        def _apply_ssr():
            scale = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, i * 9973 + 7]), minval=0.5, maxval=1.5
            )
            new_h = tf.cast(tf.round(scale * IMG_HEIGHT), tf.int32)
            new_w = tf.cast(tf.round(scale * IMG_WIDTH), tf.int32)
            y = tf.image.resize(
                x, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR
            )
            y = tf.image.resize_with_crop_or_pad(y, IMG_HEIGHT, IMG_WIDTH)

            angle_deg = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, i * 9973 + 8]), minval=-15.0, maxval=15.0
            )
            angle_rad = angle_deg * (np.pi / 180.0)
            try:
                y = tf.image.rotate(y, angle_rad, interpolation="BILINEAR")
            except Exception:
                y = y

            y = tf.clip_by_value(y, 0.0, 1.0)
            y.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
            return y

        x = tf.cond(do_ssr, _apply_ssr, lambda: x)
        x.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        return x, label

    def _test_branch():
        x = _center_crop_or_pad(img)
        return x, label

    out_img, out_label = tf.cond(use_train_aug, _train_branch, _test_branch)
    out_label = tf.cast(out_label, tf.int64)
    return out_img, out_label


def _tf_map_test_from_path(path):
    img = _tf_decode_center_crop_to_float(path)
    return img


_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = True
try:
    _DATA_OPTIONS.threading.private_threadpool_size = max(
        1, min(8, (os.cpu_count() or 2))
    )
    _DATA_OPTIONS.threading.max_intra_op_parallelism = 1
except Exception:
    pass


def make_train_ds(image_paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.with_options(_DATA_OPTIONS)

    ds = ds.map(
        lambda p, y: (_tf_decode_center_crop_to_float(p), tf.cast(y, tf.int64)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.enumerate()  # (idx, (img,label))
    ds = ds.map(
        lambda idx, t: _augment_train_tf(t[0], t[1], idx),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(image_paths_or_images, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(image_paths_or_images)
    ds = ds.with_options(_DATA_OPTIONS)

    spec = tf.nest.map_structure(lambda x: x, ds.element_spec)
    if isinstance(spec, tf.TensorSpec) and spec.dtype == tf.string:
        ds = ds.map(
            _tf_decode_center_crop_to_float,
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 9
test_df = pd.read_csv(SAMPLE_SUB_CSV)
test_df["image_id"] = test_df["image_id"].astype(str)
test_samples = test_df.shape[0]
print("test_samples:", test_samples)

test_paths = (TEST_DIR + test_df["image_id"]).values.astype(str)
test_ds = make_test_ds(test_paths, batch_size=batch_size)



## === cell 10
train_df = pd.read_csv(TRAIN_CSV)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)


def stratified_split(df, label_col="label", val_frac=0.1, seed=SEED):
    rng = np.random.RandomState(seed)
    val_idx = []
    for _, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
    val_idx = np.concatenate(val_idx)
    train_idx = df.index.difference(val_idx)
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(
        drop=True
    )


train_split_df, val_split_df = stratified_split(train_df, val_frac=0.1, seed=SEED)
print("train_split:", train_split_df.shape, "val_split:", val_split_df.shape)

train_paths = (TRAIN_DIR + train_split_df["image_id"]).values.astype(str)
train_labels = train_split_df["label"].values.astype(np.int64)

val_paths = (TRAIN_DIR + val_split_df["image_id"]).values.astype(str)
val_labels = val_split_df["label"].values.astype(np.int64)

train_ds = make_train_ds(train_paths, train_labels, batch_size=batch_size)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.with_options(_DATA_OPTIONS)
val_ds = val_ds.map(
    lambda p, y: (_tf_decode_center_crop_to_float(p), tf.cast(y, tf.int64)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_ds = val_ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTOTUNE)

train_count = 0
for _ in train_ds.take(1):
    train_count += 1
print("train_ds non-empty check (batches):", train_count)



## === cell 11
from tensorflow.keras import layers, models

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 12
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 13
pred_probs = model.predict(
    test_ds,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)
print("pred_labels shape:", pred_labels.shape)



## === cell 14
submission = pd.read_csv(SAMPLE_SUB_CSV)
submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = pred_labels
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head(5))



## === cell 15
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sub_check.columns.tolist()}"
assert len(sub_check) == len(
    pd.read_csv(SAMPLE_SUB_CSV)
), "Row count mismatch vs sample_submission"
assert sub_check["label"].between(0, NUM_CLASSES - 1).all(), "Labels out of range"
print(sub_check.head(3))
