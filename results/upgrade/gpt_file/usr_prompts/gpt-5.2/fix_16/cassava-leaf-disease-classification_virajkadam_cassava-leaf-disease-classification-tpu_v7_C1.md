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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

keras = tf.keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

print("TensorFlow:", tf.__version__)
print("tf.keras:", tf.keras.__name__)

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = os.path.join(BASE_DIR, "train_images")
test_dir = os.path.join(BASE_DIR, "test_images")
train_tfrecord_dir = os.path.join(BASE_DIR, "train_tfrecords")
test_tfrecord_dir = os.path.join(BASE_DIR, "test_tfrecords")

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train.head())
print(sample_sub.head())
print("Train size:", len(train), " Test size:", len(sample_sub))



## === cell 1
NUM_CLASSES = 5
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep conservative for memory
AUTOTUNE = tf.data.AUTOTUNE

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.optimizer.set_jit(True)  # XLA (same math, faster graph execution)
except Exception:
    pass


def build_model(backbone_name: str, input_shape=(512, 512, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if backbone_name == "inceptionresnetv2":
        backbone = keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "efficientnetv2b0":
        backbone = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError("Unknown backbone")

    backbone.trainable = True  # preserve original intent (fine-tuning)
    x = backbone(x, training=True)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs)
    return model


model1 = build_model(
    "inceptionresnetv2", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)
model2 = build_model(
    "efficientnetv2b0", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)

opt1 = keras.optimizers.Adam(learning_rate=1e-4)
opt2 = keras.optimizers.Adam(learning_rate=1e-4)

model1.compile(
    optimizer=opt1,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)
model2.compile(
    optimizer=opt2,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

print("Model1 params:", model1.count_params())
print("Model2 params:", model2.count_params())




## === cell 2
def _make_tfdata_options(deterministic=True):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    options.experimental_slack = not deterministic
    return options


TFDATA_OPTS_DETERMINISTIC = _make_tfdata_options(deterministic=True)
TFDATA_OPTS_NOND = _make_tfdata_options(deterministic=False)

TRAIN_TFRECS = []
TEST_TFRECS = []
if os.path.isdir(train_tfrecord_dir):
    TRAIN_TFRECS = sorted(
        [
            os.path.join(train_tfrecord_dir, f)
            for f in os.listdir(train_tfrecord_dir)
            if f.endswith(".tfrec")
        ]
    )
if os.path.isdir(test_tfrecord_dir):
    TEST_TFRECS = sorted(
        [
            os.path.join(test_tfrecord_dir, f)
            for f in os.listdir(test_tfrecord_dir)
            if f.endswith(".tfrec")
        ]
    )

FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_and_resize(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=True)
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment_train(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    return x, y


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES_TRAIN)
    x = _decode_and_resize(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return x, y


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES_TEST)
    x = _decode_and_resize(ex["image"])
    return x


def _detect_tfrecord_compression(tfrecs):
    """Return '' (no compression) or 'GZIP' by probing first file."""
    if not tfrecs:
        return ""
    probe = tfrecs[0]
    try:
        ds = tf.data.TFRecordDataset([probe], compression_type="")
        next(iter(ds.take(1)))
        return ""
    except Exception:
        pass
    try:
        ds = tf.data.TFRecordDataset([probe], compression_type="GZIP")
        next(iter(ds.take(1)))
        return "GZIP"
    except Exception:
        return ""


TFREC_COMPRESSION = _detect_tfrecord_compression(TRAIN_TFRECS or TEST_TFRECS)
print("Detected TFRecord compression_type:", repr(TFREC_COMPRESSION))

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def make_train_ds_from_tfrecords(tfrecs, training=True, cache_tag="train"):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, compression_type=TFREC_COMPRESSION
    )
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=not training
    )

    if not training:
        ds = ds.cache(os.path.join(CACHE_DIR, f"val_{cache_tag}.cache"))
    else:
        ds = ds.cache(os.path.join(CACHE_DIR, f"trn_{cache_tag}.cache"))

    if training:
        ds = ds.map(_augment_train, num_parallel_calls=AUTOTUNE, deterministic=False)

    ds = ds.with_options(TFDATA_OPTS_DETERMINISTIC if training else TFDATA_OPTS_NOND)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(tfrecs, batch_size, cache_tag="test"):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, compression_type=TFREC_COMPRESSION
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache(os.path.join(CACHE_DIR, f"tst_{cache_tag}.cache"))
    ds = ds.with_options(TFDATA_OPTS_NOND)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_train_ds_from_images(df, training=True, cache_tag="img"):
    paths = (BASE_DIR + "/train_images/" + df["image_id"]).values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _read(path, y):
        img_bytes = tf.io.read_file(path)
        x = _decode_and_resize(img_bytes)
        return x, y

    ds = ds.map(_read, num_parallel_calls=AUTOTUNE, deterministic=not training)

    if not training:
        ds = ds.cache(os.path.join(CACHE_DIR, f"val_{cache_tag}.cache"))
    else:
        ds = ds.cache(os.path.join(CACHE_DIR, f"trn_{cache_tag}.cache"))

    if training:
        ds = ds.map(_augment_train, num_parallel_calls=AUTOTUNE, deterministic=False)

    ds = ds.with_options(TFDATA_OPTS_DETERMINISTIC if training else TFDATA_OPTS_NOND)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_images(df, batch_size, cache_tag="imgtest"):
    paths = (BASE_DIR + "/test_images/" + df["image_id"]).values
    ds = tf.data.Dataset.from_tensor_slices(paths)

    @tf.function
    def _read(path):
        img_bytes = tf.io.read_file(path)
        x = _decode_and_resize(img_bytes)
        return x

    ds = ds.map(_read, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache(os.path.join(CACHE_DIR, f"tst_{cache_tag}.cache"))
    ds = ds.with_options(TFDATA_OPTS_NOND)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


use_tfrecords = (len(TRAIN_TFRECS) > 0) and (len(TEST_TFRECS) > 0)

if use_tfrecords:
    num_train_files = len(TRAIN_TFRECS)
    val_frac = 0.1
    val_files = max(1, int(round(num_train_files * val_frac)))
    train_files = num_train_files - val_files
    assert train_files > 0 and val_files > 0

    TRN_TFRECS = TRAIN_TFRECS[val_files:]
    VAL_TFRECS = TRAIN_TFRECS[:val_files]

    try:
        train_ds = make_train_ds_from_tfrecords(
            TRN_TFRECS, training=True, cache_tag="tfrec"
        )
        val_ds = make_train_ds_from_tfrecords(
            VAL_TFRECS, training=False, cache_tag="tfrec"
        )
        _ = next(iter(train_ds.take(1)))
        _ = next(iter(val_ds.take(1)))
        print(
            "Using TFRecords. Train tfrecs:",
            len(TRN_TFRECS),
            " Val tfrecs:",
            len(VAL_TFRECS),
        )
    except Exception as e:
        print("TFRecord pipeline failed; falling back to image files. Error:", repr(e))
        use_tfrecords = False

if not use_tfrecords:
    from sklearn.model_selection import train_test_split

    trn_df, val_df = train_test_split(
        train,
        test_size=0.1,
        random_state=SEED,
        stratify=train["label"],
    )
    train_ds = make_train_ds_from_images(trn_df, training=True, cache_tag="imgs")
    val_ds = make_train_ds_from_images(val_df, training=False, cache_tag="imgs")
    print("Using image files. Train rows:", len(trn_df), " Val rows:", len(val_df))

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 3
EPOCHS = 2  # preserve original training plan

history1 = model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
history2 = model2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 4
x50, y50 = (
    next(iter(val_ds.unbatch().batch(50))) if False else next(iter(val_ds.take(1)))
)
x50 = x50[:50]
y50 = y50[:50]

probs1_s = model1.predict(x50, verbose=0)
probs2_s = model2.predict(x50, verbose=0)
probs_s = 0.5 * probs1_s + 0.5 * probs2_s
preds = np.argmax(probs_s, axis=1).astype(int).tolist()

y_true = y50.numpy().astype(np.int32)
acc = (np.array(preds) == y_true).mean()
print("Sample accuracy (<=50 imgs):", acc)



## === cell 5
sample_test = pd.DataFrame({"Prediction": preds, "Actual": y_true})
print(sample_test.head(30))



## === cell 6
PRED_BATCH_SIZE = 32

if use_tfrecords:
    test_ds_pred = make_test_ds_from_tfrecords(
        TEST_TFRECS, batch_size=PRED_BATCH_SIZE, cache_tag="tfrec"
    )
else:
    test_ds_pred = make_test_ds_from_images(
        sample_sub, batch_size=PRED_BATCH_SIZE, cache_tag="imgs"
    )

print("Test batches:", tf.data.experimental.cardinality(test_ds_pred).numpy())


@tf.function
def _ensemble_predict_step(x):
    p1 = model1(x, training=False)
    p2 = model2(x, training=False)
    return 0.5 * p1 + 0.5 * p2


num_test = int(sample_sub.shape[0])
predictions = np.empty((num_test,), dtype=np.int64)

idx = 0
for xb in test_ds_pred:
    pb = _ensemble_predict_step(xb)
    pred_b = tf.argmax(pb, axis=1, output_type=tf.int64).numpy()
    bs = pred_b.shape[0]
    predictions[idx : idx + bs] = pred_b
    idx += bs

assert idx == num_test, (idx, num_test)

print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions.astype(int)}
)
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

print(submission.head())
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission))
