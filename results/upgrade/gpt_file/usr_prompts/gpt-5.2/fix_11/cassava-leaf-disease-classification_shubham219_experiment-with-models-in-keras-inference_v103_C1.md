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
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_GLOB = os.path.join(DATA_DIR, "test_images", "*.jpg")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"

AUTOTUNE = tf.data.AUTOTUNE

preprocess = tf.keras.applications.resnet50.preprocess_input

NUM_CLASSES = 5
IMG_SIZE = (256, 256)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # minimal, stable training and faster runtime

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = tfl.Lambda(preprocess, name="preprocess")(x)
x = base(x, training=False)
x = tfl.GlobalAveragePooling2D()(x)
x = tfl.Dropout(0.2, seed=SEED)(x)
outputs = tfl.Dense(NUM_CLASSES, activation="softmax")(x)

my_model = Model(inputs, outputs)
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()



## === cell 1
df = pd.read_csv(TRAIN_CSV)

paths = df["image_id"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))
exists_mask = paths.map(os.path.exists).values
df = df.loc[exists_mask].reset_index(drop=True)
assert len(df) > 0, "No training images found after filtering by file existence."
df["path"] = paths.loc[exists_mask].values

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=SEED,
    stratify=df["label"],
)

BATCH_SIZE = 16

TRAIN_TFREC_GLOB = os.path.join(DATA_DIR, "train_tfrecords", "*.tfrec")
TEST_TFREC_GLOB = os.path.join(DATA_DIR, "test_tfrecords", "*.tfrec")
train_tfrecs = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrecs = sorted(glob.glob(TEST_TFREC_GLOB))
assert len(train_tfrecs) > 0, f"No train tfrecords found at {TRAIN_TFREC_GLOB}"
assert len(test_tfrecs) > 0, f"No test tfrecords found at {TEST_TFREC_GLOB}"

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def _projective(img, transform):
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform[None, ...],
        output_shape=[IMG_SIZE[0], IMG_SIZE[1]],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]


@tf.function
def _augment_projective(img, seed):
    seed = tf.cast(seed, tf.int32)

    seed_flip = seed + tf.constant([1, 0], tf.int32)
    seed_tx = seed + tf.constant([2, 0], tf.int32)
    seed_ty = seed + tf.constant([3, 0], tf.int32)
    seed_zoom = seed + tf.constant([4, 0], tf.int32)
    seed_ang = seed + tf.constant([5, 0], tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_flip)

    tx = tf.random.stateless_uniform(
        [], minval=-0.05, maxval=0.05, seed=seed_tx
    ) * tf.cast(IMG_SIZE[1], tf.float32)
    ty = tf.random.stateless_uniform(
        [], minval=-0.05, maxval=0.05, seed=seed_ty
    ) * tf.cast(IMG_SIZE[0], tf.float32)

    zoom = tf.random.stateless_uniform([], minval=0.9, maxval=1.1, seed=seed_zoom)

    ang = tf.random.stateless_uniform([], minval=-15.0, maxval=15.0, seed=seed_ang) * (
        np.pi / 180.0
    )

    cx = (tf.cast(IMG_SIZE[1], tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_SIZE[0], tf.float32) - 1.0) / 2.0

    cos_a = tf.math.cos(ang)
    sin_a = tf.math.sin(ang)

    inv_scale = 1.0 / zoom
    a0 = inv_scale * cos_a
    a1 = inv_scale * sin_a
    b0 = inv_scale * -sin_a
    b1 = inv_scale * cos_a

    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    img = _projective(img, transform)
    return img


@tf.function
def _prep_train_from_img(img, label, seed2):
    img = _augment_projective(img, seed2)
    label = tf.cast(label, tf.int32)
    return img, label


@tf.function
def _prep_val_from_img(img, label):
    label = tf.cast(label, tf.int32)
    return img, label


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    seed2 = tf.stack(
        [
            tf.cast(tf.strings.to_hash_bucket_fast(name, 2**31 - 1), tf.int32),
            tf.constant(SEED, tf.int32),
        ],
        axis=0,
    )
    img, label = _prep_train_from_img(img, label, seed2)
    return img, label


@tf.function
def _parse_val_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    img, label = _prep_val_from_img(img, label)
    return img, label


_split_cache_dir = "/kaggle/working/cassava_split_cache"
os.makedirs(_split_cache_dir, exist_ok=True)
train_split_tfrec = os.path.join(_split_cache_dir, f"train_split_seed{SEED}.tfrec")
val_split_tfrec = os.path.join(_split_cache_dir, f"val_split_seed{SEED}.tfrec")

train_names_set = set(train_df["image_id"].astype(str).tolist())
val_names_set = set(val_df["image_id"].astype(str).tolist())


def _write_split_tfrecords_if_needed(src_tfrecs, out_train_path, out_val_path):
    if tf.io.gfile.exists(out_train_path) and tf.io.gfile.exists(out_val_path):
        return

    name_feature = {"image_name": tf.io.FixedLenFeature([], tf.string)}

    def _get_name_py(raw_bytes: bytes) -> bytes:
        ex = tf.io.parse_single_example(raw_bytes, name_feature)
        return ex["image_name"].numpy()

    ds = tf.data.TFRecordDataset(
        src_tfrecs,
        num_parallel_reads=1,  # deterministic source order
        buffer_size=32
        * 1024
        * 1024,  # larger sequential read buffer reduces I/O overhead
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    train_writer = tf.io.TFRecordWriter(out_train_path)
    val_writer = tf.io.TFRecordWriter(out_val_path)

    for raw_bytes in ds.as_numpy_iterator():
        name_b = _get_name_py(raw_bytes)
        name_s = name_b.decode("utf-8")
        if name_s in train_names_set:
            train_writer.write(raw_bytes)
        elif name_s in val_names_set:
            val_writer.write(raw_bytes)

    train_writer.close()
    val_writer.close()


_write_split_tfrecords_if_needed(train_tfrecs, train_split_tfrec, val_split_tfrec)


def _tfrec_dataset_options_deterministic():
    options = tf.data.Options()
    options.experimental_deterministic = True
    return options


def make_train_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=32 * 1024 * 1024
    )
    ds = ds.with_options(_tfrec_dataset_options_deterministic())
    ds = ds.apply(tf.data.experimental.ignore_errors())

    shuffle_buf = 4096  # deterministic constant; avoids counting pass
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=32 * 1024 * 1024
    )
    ds = ds.with_options(_tfrec_dataset_options_deterministic())
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.map(_parse_val_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds_from_tfrecords([train_split_tfrec], BATCH_SIZE)
val_ds = make_val_ds_from_tfrecords([val_split_tfrec], BATCH_SIZE)



## === cell 2
EPOCHS = 2
_ = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_resize_from_bytes(ex["image"])
    return img, ex["image_name"]


def make_test_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=32 * 1024 * 1024
    )
    ds = ds.with_options(_tfrec_dataset_options_deterministic())

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds_from_tfrecords(test_tfrecs, BATCH_SIZE)

test_names_parts = []
for _, n in test_ds:
    test_names_parts.append(n.numpy())
test_names = np.concatenate(test_names_parts, axis=0).astype("U")

test_img_ds = test_ds.map(
    lambda x, n: x, num_parallel_calls=AUTOTUNE, deterministic=True
)
probs = my_model.predict(test_img_ds, verbose=0)
pred_test_labels = np.argmax(probs, axis=-1).astype(int)

final_csv = pd.DataFrame({"image_id": test_names, "label": pred_test_labels})
final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)



## === cell 3
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.columns.tolist(), "rows:", len(sub))
assert sub.columns.tolist() == ["image_id", "label"]
assert sub["label"].between(0, 4).all()
assert sub["image_id"].str.endswith(".jpg").all()
print("submission.csv looks valid.")
