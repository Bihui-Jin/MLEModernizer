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

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

print("TensorFlow:", tf.__version__)
print("Train CSV:", TRAIN_CSV)
print("Test dir:", TEST_IMG_DIR)

AUTOTUNE = tf.data.AUTOTUNE
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.data.experimental.enable_debug_mode = False




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)  # keep same semantics as original

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32  # keep same

label_strs = sorted(trn_df["label"].unique().tolist())
NUM_CLASSES = len(label_strs)
label_to_index = {s: i for i, s in enumerate(label_strs)}
print("Detected classes:", NUM_CLASSES, "mapping:", label_to_index)

trn_paths = trn_df["path"].to_numpy()
trn_labels = trn_df["label"].map(label_to_index).to_numpy(np.int32)

val_paths = val_df["path"].to_numpy()
val_labels = val_df["label"].map(label_to_index).to_numpy(np.int32)

CACHE_DIR = "/kaggle/working/tf_cache_cassava_b3_300"
os.makedirs(CACHE_DIR, exist_ok=True)


@tf.function
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function
def _augment(img, seed_pair):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 0], tf.int32),
        minval=-10.0,
        maxval=10.0,
        dtype=tf.float32,
    )
    angle = angle * (np.pi / 180.0)
    if hasattr(tf.image, "rotate"):
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
    else:
        img = img

    max_dx = 0.05 * float(IMG_SIZE[1])
    max_dy = 0.05 * float(IMG_SIZE[0])
    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([3, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy,
        dtype=tf.float32,
    )
    img = tf.pad(img, [[2, 2], [2, 2], [0, 0]], mode="REFLECT")
    img = tf.roll(img, shift=tf.cast(tf.round(dy), tf.int32), axis=0)
    img = tf.roll(img, shift=tf.cast(tf.round(dx), tf.int32), axis=1)
    img = img[2:-2, 2:-2, :]

    zoom = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([4, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(zoom * IMG_SIZE[0]), tf.int32)
    new_w = tf.cast(tf.round(zoom * IMG_SIZE[1]), tf.int32)
    resized = tf.image.resize(
        img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR
    )

    def _center_crop(im, th, tw):
        h = tf.shape(im)[0]
        w = tf.shape(im)[1]
        offset_y = tf.maximum(0, (h - th) // 2)
        offset_x = tf.maximum(0, (w - tw) // 2)
        return tf.image.crop_to_bounding_box(im, offset_y, offset_x, th, tw)

    def _pad_to(im, th, tw):
        h = tf.shape(im)[0]
        w = tf.shape(im)[1]
        pad_y = tf.maximum(0, th - h)
        pad_x = tf.maximum(0, tw - w)
        paddings = [
            [pad_y // 2, pad_y - pad_y // 2],
            [pad_x // 2, pad_x - pad_x // 2],
            [0, 0],
        ]
        im = tf.pad(im, paddings, mode="REFLECT")
        return _center_crop(im, th, tw)

    resized = tf.cond(
        tf.logical_and(new_h >= IMG_SIZE[0], new_w >= IMG_SIZE[1]),
        lambda: _center_crop(resized, IMG_SIZE[0], IMG_SIZE[1]),
        lambda: _pad_to(resized, IMG_SIZE[0], IMG_SIZE[1]),
    )
    resized = tf.ensure_shape(resized, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return resized


_DATA_OPTS = tf.data.Options()
_DATA_OPTS.experimental_deterministic = False
try:
    _DATA_OPTS.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    _DATA_OPTS.threading.max_intra_op_parallelism = max(1, (os.cpu_count() or 8) // 2)
except Exception:
    pass
try:
    _DATA_OPTS.experimental_optimization.map_vectorization.enabled = True
    _DATA_OPTS.experimental_optimization.map_parallelization = True
    _DATA_OPTS.experimental_slack = True
except Exception:
    pass

_SHUFFLE_BUFFER = 8192  # keep as-is


def make_train_ds(paths, labels, batch_size):
    n = len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DATA_OPTS)

    @tf.function
    def _decode_and_label(path, y):
        img = _decode_and_resize(path)
        y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_decode_and_label, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache()

    ds = ds.shuffle(
        buffer_size=min(_SHUFFLE_BUFFER, n),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    counter = tf.data.experimental.Counter(start=0, step=1, dtype=tf.int64)
    ds = tf.data.Dataset.zip((counter, ds))

    @tf.function
    def _aug_one(i, xy):
        img, y = xy
        seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)], axis=0)
        img = _augment(img, seed_pair)
        img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
        return img, y

    ds = ds.map(_aug_one, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DATA_OPTS)

    @tf.function
    def _process(path, y):
        img = _decode_and_resize(path)
        y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_process, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_paths, trn_labels, BATCH_SIZE)
val_ds = make_val_ds(val_paths, val_labels, BATCH_SIZE)

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inp)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inp, outputs=out)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FROZEN = 2 if not DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FT = 1 if not DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    verbose=1,
)




## === cell 2
test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = pd.Series(test_images).str.rsplit("/", n=1).str[-1].values

_TFREC_GLOB = os.path.join(TEST_TFREC_DIR, "*.tfrec")
test_tfrecs = tf.io.gfile.glob(_TFREC_GLOB)


def _read_tfrecord(example):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feature_spec)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, ex["image_name"]


def make_test_ds_from_tfrecords(tfrec_files, batch_size=128):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DATA_OPTS)
    ds = ds.map(_read_tfrecord, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_paths(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_DATA_OPTS)

    @tf.function
    def _process(path):
        img = _decode_and_resize(path)
        return img

    ds = ds.map(_process, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 3
if "my_model" not in globals():
    raise RuntimeError(
        "Model (my_model) was not created. Check training cell execution for errors."
    )

if len(test_tfrecs) > 0:
    test_ds_named = make_test_ds_from_tfrecords(sorted(test_tfrecs), batch_size=128)

    names_ds = test_ds_named.map(lambda imgs, names: names, num_parallel_calls=AUTOTUNE)
    names_bytes = b"".join(list(names_ds.unbatch().as_numpy_iterator()))
    names_arr = np.fromiter(
        (
            n.decode("utf-8")
            for n in test_ds_named.unbatch()
            .map(lambda img, name: name)
            .as_numpy_iterator()
        ),
        dtype=object,
    )

    imgs_ds = test_ds_named.map(lambda imgs, names: imgs, num_parallel_calls=AUTOTUNE)
    pred_test = my_model.predict(imgs_ds, verbose=1)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

    pred_df = pd.DataFrame({"image_id": names_arr, "label": pred_test_labels})
else:
    test_ds = make_test_ds_from_paths(df_test["path"].to_numpy(), batch_size=128)
    pred_test = my_model.predict(test_ds, verbose=1)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)
    pred_df = pd.DataFrame(
        {"image_id": df_test["image_id"].values, "label": pred_test_labels}
    )

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if final_csv["label"].isna().any():
    most_freq = int(trn_df["label"].value_counts().idxmax())
    final_csv["label"] = final_csv["label"].fillna(most_freq).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)




## === cell 4
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(pd.read_csv(SAMPLE_SUB))
assert sub["label"].between(0, 4).all()
sub.head()
