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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

TRAIN_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TEST_IMG = (
    "/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
)
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
MODELS_WEIGHTS = "/kaggle/input/cassavaeffentb7models/content/Models"

print("TRAIN_IMG_LOC exists:", os.path.isdir(TRAIN_IMG_LOC))
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("SAMPLE_CSV exists:", os.path.isfile(SAMPLE_CSV))




## === cell 1
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")




## === cell 2
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

IMG_SIZE = (300, 300)  # matches original inference resize
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 3  # keep unchanged

train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)
train_df = train_df.reset_index(drop=True)
train_df["label"] = train_df["label"].astype(np.int32)

from sklearn.model_selection import train_test_split

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
HAVE_TFRECORDS = (
    os.path.isdir(TRAIN_TFREC_DIR)
    and len([f for f in os.listdir(TRAIN_TFREC_DIR) if f.endswith(".tfrec")]) > 0
)
print("TFRecords available:", HAVE_TFRECORDS)

opts_fast = tf.data.Options()
opts_fast.experimental_deterministic = False
opts_fast.experimental_slack = True
opts_fast.experimental_optimization.apply_default_optimizations = True
opts_fast.experimental_optimization.map_parallelization = True
opts_fast.experimental_optimization.parallel_batch = True


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


_aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(
            10.0 / 180.0, fill_mode="reflect", interpolation="bilinear", seed=SEED
        ),
        layers.RandomTranslation(
            0.05, 0.05, fill_mode="reflect", interpolation="bilinear", seed=SEED
        ),
        layers.RandomZoom(
            0.1, 0.1, fill_mode="reflect", interpolation="bilinear", seed=SEED
        ),
    ],
    name="augmentation",
)


@tf.function
def _augment(img):
    return _aug(img, training=True)


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    y = tf.cast(ex["target"], tf.int32)
    return img, y


def _make_train_ds_from_tfrecords(tfrecord_files, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
    )
    ds = ds.with_options(opts_fast)
    ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    def _aug_and_onehot(img, y):
        img = _augment(img)
        y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
        return img, y

    ds = ds.map(_aug_and_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels, batch_size, cache_path=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(opts_fast)

    def _map_fn(p, y):
        img = _decode_resize_preprocess(p)
        y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(opts_fast)
    ds = ds.shuffle(
        buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
    )

    def _decode_only(p, y):
        img = _decode_resize_preprocess(p)
        return img, tf.cast(y, tf.int32)

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE)
    ds = (
        ds.cache()
    )  # in-memory cache; fits comfortably for ~16.8k images at 300x300 float32 (~17-18GB) would NOT fit,

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts_fast)
    ds = ds.shuffle(
        buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(
        "/kaggle/working/train_cache"
    )  # disk cache; avoids re-decoding each epoch

    def _aug_and_onehot(img, y):
        img = _augment(img)
        y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
        return img, y

    ds = ds.map(_aug_and_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


trn_paths = trn_df["filepath"].to_numpy(dtype=str)
trn_labels = trn_df["label"].to_numpy(dtype=np.int32)
val_paths = val_df["filepath"].to_numpy(dtype=str)
val_labels = val_df["label"].to_numpy(dtype=np.int32)

if HAVE_TFRECORDS:
    train_tfrecord_files = sorted(
        [
            os.path.join(TRAIN_TFREC_DIR, f)
            for f in os.listdir(TRAIN_TFREC_DIR)
            if f.endswith(".tfrec")
        ]
    )
    train_ds = _make_train_ds_from_tfrecords(train_tfrecord_files, BATCH_SIZE)
    val_ds = _make_val_ds(
        val_paths, val_labels, BATCH_SIZE, cache_path="/kaggle/working/val_cache"
    )
else:
    train_ds = _make_train_ds(trn_paths, trn_labels, BATCH_SIZE)
    val_ds = _make_val_ds(
        val_paths, val_labels, BATCH_SIZE, cache_path="/kaggle/working/val_cache"
    )

steps_per_epoch = int(np.ceil(len(trn_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

base = EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # unchanged

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = "/kaggle/working/effnetb4_cassava_best.keras"
ckpt = ModelCheckpoint(
    ckpt_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[ckpt],
    verbose=1,
)

if os.path.isfile(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)

print("Model ready. Checkpoint exists:", os.path.isfile(ckpt_path))




## === cell 3
print("Skipping plot_model to save time.")




## === cell 4
print("Skipping test image visualization to save time.")




## === cell 5
if "_decode_resize_preprocess" not in globals():

    @tf.function
    def _decode_resize_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img


if "model" not in globals():
    print(
        "WARNING: `model` was not created during training. Building an untrained model to allow submission generation."
    )
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)

ss = pd.read_csv(SAMPLE_CSV)

test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_paths = (test_img_dir + "/" + ss["image_id"].astype(str)).to_numpy(dtype=str)

paths_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths, dtype=tf.string))


@tf.function
def _load_test(path):
    return _decode_resize_preprocess(path)


img_ds = (
    paths_ds.with_options(opts_fast)
    .map(_load_test, num_parallel_calls=AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

prob = model.predict(img_ds, verbose=0)
preds = np.argmax(prob, axis=1).astype(int)

if len(preds) != len(ss):
    raise RuntimeError(
        f"Prediction count mismatch: preds={len(preds)} vs sample_submission={len(ss)}"
    )

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote submission to:", out_path)
print(my_submission.head())
print("Submission shape:", my_submission.shape)
