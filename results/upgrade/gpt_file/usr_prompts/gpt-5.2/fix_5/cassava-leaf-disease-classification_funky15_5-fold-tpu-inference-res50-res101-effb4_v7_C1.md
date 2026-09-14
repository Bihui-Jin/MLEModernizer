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
import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("TensorFlow:", tf.__version__)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
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



## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_ROOT = first_existing(DATA_ROOT_CANDIDATES)
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset root in any of: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required path: {p}")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print("Train:", train_df.shape, "Sample:", sample_df.shape)
print("Train label counts:\n", train_df["label"].value_counts().sort_index())

HAS_TFRECORDS = os.path.isdir(TRAIN_TFREC_DIR) and (
    len(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))) > 0
)
HAS_TEST_TFRECORDS = os.path.isdir(TEST_TFREC_DIR) and (
    len(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))) > 0
)
print("TFRecords available - train:", HAS_TFRECORDS, "test:", HAS_TEST_TFRECORDS)




## === cell 2
def build_model(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0)(inputs)

    backbone = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    backbone.trainable = False

    x = backbone.output
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()



## === cell 3
from sklearn.model_selection import train_test_split

train_paths = [os.path.join(TRAIN_DIR, fn) for fn in train_df["image_id"].values]
train_labels = train_df["label"].astype(np.int32).values

X_train, X_val, y_train, y_val = train_test_split(
    train_paths, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
)

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def load_image(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # forces RGB
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.uint8)  # keep uint8; normalization happens in model
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


@tf.function
def _tfrecord_parse_train_with_name(example):
    feature_desc = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feature_desc)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.uint8)
    lbl = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, lbl, name


@tf.function
def _tfrecord_parse_test_with_name(example):
    feature_desc = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, feature_desc)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
    img = tf.cast(img, tf.uint8)
    return img, ex["image_name"]


def _build_train_val_from_tfrecords():
    tfrec_files = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if not tfrec_files:
        raise FileNotFoundError(f"No TFRecord files found in {TRAIN_TFREC_DIR}")

    train_names = np.array([os.path.basename(p) for p in X_train], dtype=object)
    val_names = np.array([os.path.basename(p) for p in X_val], dtype=object)

    keys = tf.constant(
        np.concatenate([train_names, val_names]).tolist(), dtype=tf.string
    )
    vals = tf.concat(
        [
            tf.ones([len(train_names)], dtype=tf.int32),
            tf.fill([len(val_names)], tf.constant(2, dtype=tf.int32)),
        ],
        axis=0,
    )
    split_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    opts = tf.data.Options()
    opts.experimental_deterministic = True

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds = ds.map(_tfrecord_parse_train_with_name, num_parallel_calls=AUTOTUNE)

    @tf.function
    def _attach_split(img, lbl, name):
        sid = split_table.lookup(name)
        return img, lbl, sid

    ds = ds.map(_attach_split, num_parallel_calls=AUTOTUNE)

    ds_train_local = ds.filter(lambda img, lbl, sid: tf.equal(sid, 1)).map(
        lambda img, lbl, sid: (img, lbl), num_parallel_calls=AUTOTUNE
    )
    ds_val_local = ds.filter(lambda img, lbl, sid: tf.equal(sid, 2)).map(
        lambda img, lbl, sid: (img, lbl), num_parallel_calls=AUTOTUNE
    )

    ds_train_local = ds_train_local.shuffle(
        4096, seed=SEED, reshuffle_each_iteration=True
    )

    ds_train_local = ds_train_local.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        AUTOTUNE
    )
    ds_val_local = ds_val_local.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        AUTOTUNE
    )

    opt2 = tf.data.Options()
    opt2.experimental_optimization.apply_default_optimizations = True
    opt2.experimental_deterministic = True
    ds_train_local = ds_train_local.with_options(opt2)
    ds_val_local = ds_val_local.with_options(opt2)
    return ds_train_local, ds_val_local


if HAS_TFRECORDS:
    ds_train, ds_val = _build_train_val_from_tfrecords()
else:
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_train = ds_train.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds_train = ds_train.map(load_image, num_parallel_calls=AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    ds_val = ds_val.map(load_image, num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)



## === cell 4
history1 = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)

backbone = None
for layer in model.layers:
    if isinstance(layer, keras.Model) and layer.name.startswith("resnet"):
        backbone = layer
        break
if backbone is None:
    for layer in model.layers:
        if "resnet50" in layer.name.lower():
            backbone = layer
            break

if backbone is not None:
    backbone.trainable = True
    for l in backbone.layers:
        if isinstance(l, keras.layers.BatchNormalization):
            l.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)




## === cell 5
def get_preds_from_sample(sample_df, image_dir, model_obj):
    img_ids = sample_df["image_id"].tolist()

    if HAS_TEST_TFRECORDS:
        tfrec_files = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
        if not tfrec_files:
            raise FileNotFoundError(f"No TFRecord files found in {TEST_TFREC_DIR}")

        opts = tf.data.Options()
        opts.experimental_deterministic = True

        ds_test = tf.data.TFRecordDataset(
            tfrec_files, num_parallel_reads=AUTOTUNE
        ).with_options(opts)
        ds_test = ds_test.map(
            _tfrecord_parse_test_with_name, num_parallel_calls=AUTOTUNE
        )

        keys = tf.constant(img_ids, dtype=tf.string)
        vals = tf.range(len(img_ids), dtype=tf.int32)
        index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
            default_value=-1,
        )

        ds_test = ds_test.map(
            lambda img, name: (img, index_table.lookup(name)),
            num_parallel_calls=AUTOTUNE,
        )
        ds_test = ds_test.filter(lambda img, idx: idx >= 0)
        ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

        out = np.empty((len(img_ids),), dtype=np.int64)
        for batch_imgs, batch_idxs in ds_test.as_numpy_iterator():
            probs = model_obj.predict_on_batch(batch_imgs)
            preds = np.argmax(probs, axis=1).astype(np.int64)
            out[batch_idxs.astype(np.int64)] = preds

        return pd.DataFrame({"image_id": img_ids, "label": out.astype(int)})

    paths = [os.path.join(image_dir, x) for x in img_ids]
    for p in paths[:5]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Test image not found: {p}")

    ds_test = tf.data.Dataset.from_tensor_slices(paths)
    ds_test = ds_test.map(lambda p: load_image(p, None), num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    probs = model_obj.predict(ds_test, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)

    return pd.DataFrame({"image_id": img_ids, "label": preds})


predict_df = get_preds_from_sample(sample_df, TEST_DIR, model)
predict_df.to_csv("submission.csv", index=False)

print(predict_df.head())
print("Wrote submission.csv with shape:", predict_df.shape)
assert list(predict_df.columns) == ["image_id", "label"]
assert len(predict_df) == len(sample_df)
assert predict_df["label"].between(0, NUM_CLASSES - 1).all()
assert os.path.exists("submission.csv")
