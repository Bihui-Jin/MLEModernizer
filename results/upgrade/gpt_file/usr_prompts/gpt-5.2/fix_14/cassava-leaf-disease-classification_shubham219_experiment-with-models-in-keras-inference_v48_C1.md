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

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def resolve_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../data/cassava-leaf-disease-classification",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )


BASE_DIR = resolve_base_dir()
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

print("BASE_DIR:", BASE_DIR)
print("Train CSV exists:", os.path.isfile(TRAIN_CSV))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Train TFRecords dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.isdir(TEST_TFREC_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))



## === cell 1
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
N_CLASSES = 5
EPOCHS = 2  # unchanged core training loop length

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)

df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
n_val = int(len(df_train) * val_frac)
df_val = df_train.iloc[:n_val].copy()
df_tr = df_train.iloc[n_val:].copy()

AUTOTUNE = tf.data.AUTOTUNE

tr_paths = df_tr["path"].to_numpy()
tr_labels = df_tr["label"].to_numpy(dtype=np.int32)
val_paths = df_val["path"].to_numpy()
val_labels = df_val["label"].to_numpy(dtype=np.int32)


@tf.function
def _decode_resize_rescale_from_bytes(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)  # cassava images are jpeg
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    return _decode_resize_rescale_from_bytes(img)


augment_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=10.0 / 360.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augment",
)

_ = augment_layers(
    tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], tf.float32), training=True
)


@tf.function
def _apply_aug_batch(img_batch):
    return augment_layers(img_batch, training=True)


@tf.function
def _train_decode_only(path, label):
    img = _decode_resize_rescale(path)
    return img, tf.cast(label, tf.int32)


@tf.function
def _train_aug_batch(imgs, labels):
    imgs = _apply_aug_batch(imgs)
    imgs = tf.image.random_flip_left_right(imgs, seed=SEED)
    return imgs, labels


@tf.function
def _val_map(path, label):
    img = _decode_resize_rescale(path)
    return img, tf.cast(label, tf.int32)


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


@tf.function
def _parse_example_decode_only(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    return img, tf.cast(ex["target"], tf.int32)


@tf.function
def _parse_val_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    return img, tf.cast(ex["target"], tf.int32)


_SHUFFLE_BUFFER = min(len(df_tr), 8192)


def _list_tfrecs(tfrecord_dir, pattern="*.tfrec"):
    return sorted(glob.glob(os.path.join(tfrecord_dir, pattern)))


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = True
    opt = options.experimental_optimization
    opt.map_parallelization = True
    opt.map_and_batch_fusion = True
    opt.parallel_batch = True
    if hasattr(opt, "autotune_buffers"):
        opt.autotune_buffers = True
    return options


def make_train_val_ds_from_tfrecords(tfrecord_dir, batch_size, n_val, shuffle_buffer):
    tfrecs = _list_tfrecs(tfrecord_dir)
    if not tfrecs:
        raise FileNotFoundError(f"No TFRecord files found in: {tfrecord_dir}")

    base = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE).with_options(
        _dataset_options()
    )

    base = base.shuffle(
        buffer_size=shuffle_buffer, seed=SEED, reshuffle_each_iteration=True
    )

    base = base.map(
        _parse_example_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    base = base.apply(tf.data.experimental.ignore_errors())
    base = base.enumerate()

    val_ds = base.filter(lambda i, xy: i < n_val).map(
        lambda i, xy: xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    tr_ds = base.filter(lambda i, xy: i >= n_val).map(
        lambda i, xy: xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    tr_ds = tr_ds.batch(batch_size, drop_remainder=False)
    tr_ds = tr_ds.map(_train_aug_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    tr_ds = tr_ds.prefetch(AUTOTUNE)

    val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return tr_ds, val_ds


def make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _dataset_options()
    )
    ds = ds.shuffle(
        buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_train_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(_train_aug_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _dataset_options()
    )
    ds = ds.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if os.path.isdir(TRAIN_TFREC_DIR) and len(_list_tfrecs(TRAIN_TFREC_DIR)) > 0:
    train_ds, val_ds = make_train_val_ds_from_tfrecords(
        TRAIN_TFREC_DIR, BATCH_SIZE, n_val=n_val, shuffle_buffer=_SHUFFLE_BUFFER
    )
else:
    train_ds = make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
    val_ds = make_val_ds(val_paths, val_labels, BATCH_SIZE)

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
base.trainable = False
x = tf.keras.layers.Dropout(0.2)(base.output)
outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
my_model = tf.keras.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

if DEBUG:
    my_model.summary()

steps_per_epoch = int(np.ceil(len(df_tr) / BATCH_SIZE))
validation_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 2
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found in: {TEST_IMG_DIR}")

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = df_test["path"].map(os.path.basename)


@tf.function
def _test_map(path):
    img = _decode_resize_rescale(path)
    return img


def make_test_ds(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_dataset_options())
    ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(df_test["path"].to_numpy(), batch_size=128)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

if os.path.isfile(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        mode_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(mode_label).astype(int)

final_csv.to_csv("submission.csv", index=False)
final_csv.head()



## === cell 3
print("submission.csv written:", os.path.isfile("submission.csv"))
print("Rows:", len(final_csv), "Columns:", list(final_csv.columns))
print(final_csv["label"].value_counts(dropna=False).sort_index())
final_csv.tail()
