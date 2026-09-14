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
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

ROOT_DIR = "../input/cassava-leaf-disease-classification/"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR listing (head):", os.listdir(ROOT_DIR)[:10])




## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF version:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception as e:
    print("set_random_seed not available:", repr(e))

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("enable_op_determinism not available:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("thread config not available:", repr(e))




## === cell 2
from PIL import Image  # kept (even if unused) to preserve original environment intent

TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("Train shape:", train_df.shape)
print("Sample sub shape:", sample_sub.shape)
print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))
print("Train tfrecords dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test tfrecords dir exists:", os.path.exists(TEST_TFREC_DIR))




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # unchanged
NUM_CLASSES = 5

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def decode_resize_norm(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # ensure RGB
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def load_image_from_path(path, label):
    img_bytes = tf.io.read_file(path)
    img = decode_resize_norm(img_bytes)
    return img, tf.cast(label, tf.int32)


@tf.function
def load_image_from_path_no_label(path):
    img_bytes = tf.io.read_file(path)
    img = decode_resize_norm(img_bytes)
    return img


_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURES_TRAIN_WITH_NAME = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    img = decode_resize_norm(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def parse_train_example_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN_WITH_NAME)
    img = decode_resize_norm(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_name = ex["image_name"]
    return img, label, image_name


@tf.function
def parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TEST)
    img = decode_resize_norm(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


def _dataset_options(deterministic: bool = True):
    opt = tf.data.Options()
    opt.experimental_deterministic = deterministic
    opt.experimental_optimization.apply_default_optimizations = True
    opt.experimental_optimization.map_fusion = True
    opt.experimental_optimization.map_parallelization = True
    return opt


train_image_ids = train_df["image_id"].astype(str).to_numpy()
train_labels = train_df["label"].to_numpy(dtype=np.int32)

idx = np.arange(len(train_image_ids))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_ids = train_image_ids[tr_idx]
va_ids = train_image_ids[va_idx]

train_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
use_tfrecords_for_train = bool(train_tfrec_files)

if use_tfrecords_for_train:
    raw = tf.data.TFRecordDataset(
        train_tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(_dataset_options(deterministic=True))

    parsed = raw.map(
        parse_train_example_with_name, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    tr_keys = tf.constant(tr_ids, dtype=tf.string)
    va_keys = tf.constant(va_ids, dtype=tf.string)

    tr_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tr_keys, values=tf.ones([tf.shape(tr_keys)[0]], dtype=tf.int32)
        ),
        default_value=0,
    )
    va_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=va_keys, values=tf.ones([tf.shape(va_keys)[0]], dtype=tf.int32)
        ),
        default_value=0,
    )

    @tf.function
    def _is_in_train(img, label, name):
        return tf.not_equal(tr_table.lookup(name), 0)

    @tf.function
    def _is_in_val(img, label, name):
        return tf.not_equal(va_table.lookup(name), 0)

    @tf.function
    def _drop_name(img, label, name):
        return img, label

    train_ds = (
        parsed.filter(_is_in_train)
        .map(_drop_name, num_parallel_calls=AUTOTUNE, deterministic=True)
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        parsed.filter(_is_in_val)
        .map(_drop_name, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
else:
    train_paths = np.array(
        [os.path.join(TRAIN_DIR, img_id) for img_id in train_image_ids], dtype=np.str_
    )
    tr_paths, va_paths = train_paths[tr_idx], train_paths[va_idx]
    tr_labels, va_labels = train_labels[tr_idx], train_labels[va_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels)).with_options(
        _dataset_options(deterministic=True)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_labels)).with_options(
        _dataset_options(deterministic=True)
    )

    train_ds = (
        train_ds.map(
            load_image_from_path, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds.map(
            load_image_from_path, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

print("Using TFRecords for train/val:", use_tfrecords_for_train)
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())




## === cell 4
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ],
    name="aug",
)

model = keras.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3, seed=SEED),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)




## === cell 6
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

if test_tfrec_files:
    test_raw = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(_dataset_options(deterministic=True))

    test_ds_named = (
        test_raw.map(
            parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    probs = model.predict(
        test_ds_named.map(lambda x, n: x, num_parallel_calls=AUTOTUNE), verbose=0
    )

    test_images = []
    for _, n in test_ds_named:
        test_images.extend(n.numpy().astype("U").tolist())
else:
    test_paths = tf.io.gfile.glob(os.path.join(TEST_DIR, "*.jpg"))
    test_paths = sorted(test_paths)
    test_images = [os.path.basename(p) for p in test_paths]

    test_ds = (
        tf.data.Dataset.from_tensor_slices(np.array(test_paths, dtype=np.str_))
        .with_options(_dataset_options(deterministic=True))
        .map(
            load_image_from_path_no_label,
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    probs = model.predict(test_ds, verbose=1)

preds = probs.argmax(axis=1).astype(int)

print("Num test images:", len(test_images))
print("Num preds:", len(preds))




## === cell 7
sub = pd.DataFrame({"image_id": test_images, "label": preds})

sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
missing = int(sub["label"].isna().sum())
if missing:
    sub["label"] = sub["label"].fillna(0).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

print(sub.head())
print("Submission shape:", sub.shape, "missing labels:", missing)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", os.path.getsize("submission.csv"), "bytes")




## === cell 8
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"]
assert len(chk) == len(sample_sub)
assert chk["label"].between(0, 4).all()
print(chk.head())
