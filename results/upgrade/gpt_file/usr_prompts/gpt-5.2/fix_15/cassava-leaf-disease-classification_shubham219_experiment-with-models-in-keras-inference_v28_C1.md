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
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass


def _resolve_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../data/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob(
        "../input/**/train.csv", recursive=True
    )
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("Could not resolve cassava dataset base directory.")


BASE_DIR = _resolve_base_dir()
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"

print("BASE_DIR:", BASE_DIR)

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = train_df["image_id"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))
train_df["label_int"] = train_df["label"].astype(np.int32)

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
NUM_CLASSES = 5

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True
try:
    DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
    DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
    DATASET_OPTIONS.experimental_optimization.autotune_cpu_budget = True
except Exception:
    pass


@tf.function
def _augment(img, seed_pair):
    s1, s2 = tf.unstack(seed_pair)
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [], seed=[s1, s2 + 1], minval=-15.0, maxval=15.0
    ) * (np.pi / 180.0)
    try:
        import tensorflow_addons as tfa  # type: ignore

        img = tfa.image.rotate(
            img, angles=angle, interpolation="BILINEAR", fill_mode="reflect"
        )
    except Exception:
        img = img

    h, w = IMG_SIZE
    dh = int(0.05 * h)
    dw = int(0.05 * w)
    img_pad = tf.image.pad_to_bounding_box(img, dh, dw, h + 2 * dh, w + 2 * dw)
    offset_y = tf.random.stateless_uniform(
        [], seed=[s1 + 2, s2 + 2], minval=0, maxval=2 * dh + 1, dtype=tf.int32
    )
    offset_x = tf.random.stateless_uniform(
        [], seed=[s1 + 3, s2 + 3], minval=0, maxval=2 * dw + 1, dtype=tf.int32
    )
    img = tf.image.crop_to_bounding_box(img_pad, offset_y, offset_x, h, w)

    zoom = tf.random.stateless_uniform(
        [], seed=[s1 + 4, s2 + 4], minval=0.9, maxval=1.0
    )
    ch = tf.cast(tf.round(zoom * tf.cast(h, tf.float32)), tf.int32)
    cw = tf.cast(tf.round(zoom * tf.cast(w, tf.float32)), tf.int32)
    oy_max = h - ch + 1
    ox_max = w - cw + 1
    oy = tf.random.stateless_uniform(
        [], seed=[s1 + 5, s2 + 5], minval=0, maxval=oy_max, dtype=tf.int32
    )
    ox = tf.random.stateless_uniform(
        [], seed=[s1 + 6, s2 + 6], minval=0, maxval=ox_max, dtype=tf.int32
    )
    img = tf.image.crop_to_bounding_box(img, oy, ox, ch, cw)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img


def _list_tfrecords(dir_path: str):
    files = tf.io.gfile.glob(os.path.join(dir_path, "*.tfrec")) + tf.io.gfile.glob(
        os.path.join(dir_path, "*.tfrecord")
    )
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecords found in: {dir_path}")
    return files


_TRAIN_TFRECS = _list_tfrecords(TRAIN_TFREC_DIR)
_TEST_TFRECS = _list_tfrecords(TEST_TFREC_DIR)


def _tfrec_feature_spec(training: bool):
    if training:
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }


@tf.function
def _decode_resize_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


rng_files = np.random.RandomState(SEED)
perm = rng_files.permutation(len(_TRAIN_TFRECS))
split_files = max(1, int(0.9 * len(_TRAIN_TFRECS)))
_TRAIN_TFRECS_TR = [_TRAIN_TFRECS[i] for i in perm[:split_files]]
_TRAIN_TFRECS_VA = [_TRAIN_TFRECS[i] for i in perm[split_files:]]
if not _TRAIN_TFRECS_VA:
    _TRAIN_TFRECS_VA = _TRAIN_TFRECS_TR[-1:]
    _TRAIN_TFRECS_TR = _TRAIN_TFRECS_TR[:-1]


@tf.function
def _make_seed_pair_from_example(seed_scalar, label_i, img_bytes):
    h = tf.strings.to_hash_bucket_fast(img_bytes, 2**31 - 1)
    s1 = tf.cast(seed_scalar, tf.int32)
    s2 = tf.cast(h, tf.int32) ^ (tf.cast(label_i, tf.int32) * 1103515245)
    return tf.stack([s1, s2])


def _make_ds_from_tfrecords_split(tfrecord_files, training: bool):
    ds = tf.data.TFRecordDataset(
        tfrecord_files, num_parallel_reads=AUTOTUNE
    ).with_options(DATASET_OPTIONS)

    if training:
        spec = _tfrec_feature_spec(training=True)

        @tf.function
        def _parse_decode_aug_onehot(ex):
            x = tf.io.parse_single_example(ex, spec)
            img_bytes = x["image"]
            img = _decode_resize_bytes(img_bytes)
            label_i = tf.cast(x["target"], tf.int32)

            seed_pair = _make_seed_pair_from_example(SEED, label_i, img_bytes)
            img = _augment(img, seed_pair)

            y = tf.one_hot(label_i, NUM_CLASSES, dtype=tf.float32)
            return img, y

        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.shuffle(buffer_size=4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            _parse_decode_aug_onehot, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    else:
        spec = _tfrec_feature_spec(training=True)

        @tf.function
        def _parse_decode_onehot(ex):
            x = tf.io.parse_single_example(ex, spec)
            img = _decode_resize_bytes(x["image"])
            label_i = tf.cast(x["target"], tf.int32)
            y = tf.one_hot(label_i, NUM_CLASSES, dtype=tf.float32)
            return img, y

        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.map(
            _parse_decode_onehot, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.cache()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds


train_ds = _make_ds_from_tfrecords_split(_TRAIN_TFRECS_TR, training=True)
valid_ds = _make_ds_from_tfrecords_split(_TRAIN_TFRECS_VA, training=False)

base = EfficientNetB3(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base.trainable = False  # keep it light + stable in <=600s

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = models.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

EPOCHS = 3 if not DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 1
sample_sub = pd.read_csv(SAMPLE_SUB)
test_paths = (TEST_IMG_DIR + "/" + sample_sub["image_id"]).to_numpy(dtype=str)


def make_test_ds_from_tfrecords(tfrecord_files, batch_size=512):
    ds = tf.data.TFRecordDataset(
        tfrecord_files, num_parallel_reads=AUTOTUNE
    ).with_options(DATASET_OPTIONS)
    spec = _tfrec_feature_spec(training=False)

    @tf.function
    def _parse_decode(ex):
        x = tf.io.parse_single_example(ex, spec)
        img = _decode_resize_bytes(x["image"])
        return x["image_name"], img

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_parse_decode, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds_from_tfrecords(_TEST_TFRECS, batch_size=512)

names_ds = test_ds.map(lambda n, x: n, num_parallel_calls=AUTOTUNE, deterministic=True)
imgs_ds = test_ds.map(lambda n, x: x, num_parallel_calls=AUTOTUNE, deterministic=True)

all_names = np.concatenate([b.numpy().astype("U") for b in names_ds], axis=0)
probs = my_model.predict(imgs_ds, verbose=0)
all_labels = np.argmax(probs, axis=-1).astype(np.int32)

name_to_pred = dict(zip(all_names.tolist(), all_labels.astype(int).tolist()))

out = sample_sub.copy()
out["label"] = out["image_id"].map(name_to_pred).astype(int)
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())



## === cell 2
assert os.path.exists("submission.csv")
sub_df = pd.read_csv("submission.csv")
print(sub_df.dtypes)
print(sub_df.head())
print("Unique labels:", sorted(sub_df["label"].unique().tolist()))
