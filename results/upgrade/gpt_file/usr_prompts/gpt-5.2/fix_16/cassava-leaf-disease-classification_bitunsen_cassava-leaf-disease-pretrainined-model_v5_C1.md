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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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

TRAIN_TFRECORD_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFRECORD_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.isdir(TRAIN_TFRECORD_DIR), f"Missing: {TRAIN_TFRECORD_DIR}"
assert os.path.isdir(TEST_TFRECORD_DIR), f"Missing: {TEST_TFRECORD_DIR}"



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


def _parse_tfrecord(example_proto, labeled=True):
    if labeled:
        feature_spec = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        feature_spec = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8
    img = tf.image.convert_image_dtype(img, tf.float32)  # float32 in [0,1]
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    if labeled:
        return img, tf.cast(ex["target"], tf.int64)
    else:
        return img, ex["image_name"]


def _center_crop_or_pad(img):
    img = tf.image.resize_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return img


def _rand_uniform(seed2):
    return tf.random.stateless_uniform([], seed2, dtype=tf.float32)


def _rand_bool(seed2, p):
    return _rand_uniform(seed2) < tf.constant(p, tf.float32)


@tf.function
def _augment_train_tf(img, label, idx, epoch):
    s0 = tf.cast(SEED, tf.int32)
    i = tf.cast(idx, tf.int32)
    e = tf.cast(epoch, tf.int32)

    base = i * 9973 + e * 1000003

    use_train_aug = _rand_bool(tf.stack([s0, base + 1]), 0.5)

    def _train_branch():
        x = img

        do_flip = _rand_bool(tf.stack([s0, base + 2]), 0.5)
        x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)

        do_bc = _rand_bool(tf.stack([s0, base + 3]), 0.5)

        def _apply_bc():
            delta = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, base + 4]), minval=-0.2, maxval=0.2
            )
            y = tf.image.adjust_brightness(x, delta)
            cf = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, base + 5]), minval=0.8, maxval=1.2
            )
            y = tf.image.adjust_contrast(y, cf)
            return tf.clip_by_value(y, 0.0, 1.0)

        x = tf.cond(do_bc, _apply_bc, lambda: x)

        x = _center_crop_or_pad(x)

        do_ssr = _rand_bool(tf.stack([s0, base + 6]), 0.5)

        def _apply_ssr():
            scale = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, base + 7]), minval=0.5, maxval=1.5
            )
            new_h = tf.cast(tf.round(scale * IMG_HEIGHT), tf.int32)
            new_w = tf.cast(tf.round(scale * IMG_WIDTH), tf.int32)
            y = tf.image.resize(
                x, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR
            )
            y = tf.image.resize_with_crop_or_pad(y, IMG_HEIGHT, IMG_WIDTH)

            angle_deg = tf.random.stateless_uniform(
                [], seed=tf.stack([s0, base + 8]), minval=-15.0, maxval=15.0
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


_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = True
try:
    _DATA_OPTIONS.threading.private_threadpool_size = max(
        1, min(8, (os.cpu_count() or 2))
    )
    _DATA_OPTIONS.threading.max_intra_op_parallelism = 1
except Exception:
    pass
try:
    _DATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _DATA_OPTIONS.experimental_optimization.map_parallelization = True
    _DATA_OPTIONS.experimental_optimization.autotune_buffers = True
except Exception:
    pass


def _list_tfrec_files(tfrec_dir, prefix):
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecords found in {tfrec_dir} with prefix {prefix}"
        )
    return files


def make_base_train_ds_from_tfrecords(tfrec_files, cache_path):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DATA_OPTIONS)
    ds = ds.map(
        lambda ex: _parse_tfrecord(ex, labeled=True),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache(cache_path)
    return ds


def make_train_ds_for_epoch(base_ds, batch_size, epoch):
    ds = base_ds.enumerate()
    ds = ds.map(
        lambda idx, t: _augment_train_tf(t[0], t[1], idx, tf.cast(epoch, tf.int64)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(tfrec_files, batch_size, cache_path):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DATA_OPTIONS)
    ds = ds.map(
        lambda ex: _parse_tfrecord(ex, labeled=False),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache(cache_path)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 9
test_df = pd.read_csv(SAMPLE_SUB_CSV)
test_df["image_id"] = test_df["image_id"].astype(str)
test_samples = test_df.shape[0]
print("test_samples:", test_samples)

test_tfrec_files = _list_tfrec_files(TEST_TFRECORD_DIR, "ld_test")
test_cache = os.path.join(
    "/kaggle/working", f"tfrec_cache_test_{IMG_HEIGHT}x{IMG_WIDTH}.cache"
)

test_ds_with_names = make_test_ds_from_tfrecords(
    test_tfrec_files, batch_size=batch_size, cache_path=test_cache
)

print("test_ds prepared from TFRecords:", len(test_tfrec_files), "files")



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

train_names_set = set(train_split_df["image_id"].tolist())
val_names_set = set(val_split_df["image_id"].tolist())

train_names_tf = tf.constant(sorted(list(train_names_set)))
val_names_tf = tf.constant(sorted(list(val_names_set)))


def _build_name_lookup(names_tensor):
    init = tf.lookup.KeyValueTensorInitializer(
        keys=names_tensor,
        values=tf.ones(tf.shape(names_tensor)[0], dtype=tf.int64),
    )
    table = tf.lookup.StaticHashTable(init, default_value=tf.constant(0, tf.int64))
    return table


train_name_table = _build_name_lookup(train_names_tf)
val_name_table = _build_name_lookup(val_names_tf)

train_tfrec_files = _list_tfrec_files(TRAIN_TFRECORD_DIR, "ld_train")

base_all = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
base_all = base_all.with_options(_DATA_OPTIONS)
base_all = base_all.map(
    lambda ex: (
        tf.image.resize_with_crop_or_pad(
            tf.image.convert_image_dtype(
                tf.image.decode_jpeg(
                    tf.io.parse_single_example(
                        ex,
                        {
                            "image": tf.io.FixedLenFeature([], tf.string),
                            "target": tf.io.FixedLenFeature([], tf.int64),
                            "image_name": tf.io.FixedLenFeature([], tf.string),
                        },
                    )["image"],
                    channels=3,
                ),
                tf.float32,
            ),
            IMG_HEIGHT,
            IMG_WIDTH,
        ),
        tf.cast(
            tf.io.parse_single_example(
                ex,
                {
                    "image": tf.io.FixedLenFeature([], tf.string),
                    "target": tf.io.FixedLenFeature([], tf.int64),
                    "image_name": tf.io.FixedLenFeature([], tf.string),
                },
            )["target"],
            tf.int64,
        ),
        tf.io.parse_single_example(
            ex,
            {
                "image": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.int64),
                "image_name": tf.io.FixedLenFeature([], tf.string),
            },
        )["image_name"],
    ),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)


def _fix_shape(img, label, name):
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, 3))
    return img, label, name


base_all = base_all.map(_fix_shape, num_parallel_calls=AUTOTUNE, deterministic=True)

base_cache = os.path.join(
    "/kaggle/working", f"tfrec_cache_train_base_{IMG_HEIGHT}x{IMG_WIDTH}.cache"
)
base_all = base_all.cache(base_cache)

base_train = base_all.filter(
    lambda img, y, n: tf.equal(train_name_table.lookup(n), 1)
).map(lambda img, y, n: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True)
base_val = base_all.filter(lambda img, y, n: tf.equal(val_name_table.lookup(n), 1)).map(
    lambda img, y, n: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True
)

train_cache = os.path.join(
    "/kaggle/working", f"tfrec_cache_train_split_{IMG_HEIGHT}x{IMG_WIDTH}.cache"
)
val_cache = os.path.join(
    "/kaggle/working", f"tfrec_cache_val_split_{IMG_HEIGHT}x{IMG_WIDTH}.cache"
)
base_train = base_train.cache(train_cache)
base_val = base_val.cache(val_cache)

val_ds = base_val.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = int(np.ceil(len(train_split_df) / batch_size))
validation_steps = int(np.ceil(len(val_split_df) / batch_size))

print("train/val base datasets prepared from TFRecords.")
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



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

histories = []
for epoch in range(EPOCHS):
    train_ds = make_train_ds_for_epoch(base_train, batch_size=batch_size, epoch=epoch)
    h = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )
    histories.append(h.history)

history = histories



## === cell 13
pred_probs_all = []
pred_names_all = []

for batch_imgs, batch_names in test_ds_with_names:
    probs = model(batch_imgs, training=False).numpy()
    pred_probs_all.append(probs)
    pred_names_all.append(batch_names.numpy())

pred_probs_all = np.concatenate(pred_probs_all, axis=0)
pred_names_all = np.concatenate(pred_names_all, axis=0).astype("U")

name_to_idx = {n: i for i, n in enumerate(pred_names_all)}
order_idx = np.array(
    [name_to_idx[n] for n in test_df["image_id"].values], dtype=np.int64
)

pred_probs = pred_probs_all[order_idx]
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
