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

3.12

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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(WORK_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

print("TensorFlow:", tf.__version__)
print("WORK_DIR:", WORK_DIR)




## === cell 1
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}



## === cell 2
MODEL_PATH = "/kaggle/input/finalmodel/myfinal_model.h5"
IMG_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 3  # kept as-is (core logic)


def build_model(img_size=224, num_classes=5):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = efficientnet_v2.preprocess_input(inputs)
    base = efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_tensor=x
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def _make_data_options():
    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    try:
        opts.threading.private_threadpool_size = max(
            2, min(8, (os.cpu_count() or 8) - 1)
        )
        opts.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    return opts


DATA_OPTIONS = _make_data_options()

_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}

_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


@tf.function
def _decode_resize_from_jpeg_bytes(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3, fancy_upscaling=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    lbl = tf.where(ex["target"] >= 0, ex["target"], ex["label"])
    return img, tf.cast(lbl, tf.int32)


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = _decode_resize_from_jpeg_bytes(ex["image"])
    return img


def _list_tfrec_files(directory, prefix):
    if not os.path.isdir(directory):
        return []
    files = tf.io.gfile.glob(os.path.join(directory, f"{prefix}*.tfrec"))
    return sorted(files)


def _maybe_cache(ds, cache_path=None):
    if cache_path:
        return ds.cache(cache_path)
    return ds


def _robust_load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    load_errors = []
    for kwargs in (
        dict(custom_objects=custom_objects, compile=False),
        dict(custom_objects=custom_objects, compile=False, safe_mode=False),
        dict(compile=False),
        dict(compile=False, safe_mode=False),
    ):
        try:
            return load_model(model_path, **kwargs)
        except Exception as e:
            load_errors.append(repr(e))
    raise RuntimeError(
        "Model file exists but could not be loaded.\n"
        f"MODEL_PATH={model_path}\nErrors:\n" + "\n".join(load_errors)
    )


def _to_one_hot(img, lbl):
    lbl_oh = tf.one_hot(tf.cast(lbl, tf.int32), NUM_CLASSES)
    return img, tf.cast(lbl_oh, tf.float32)


def _make_train_val_datasets(train_tfrec_files, train_csv_path, batch_size, seed=42):
    df = pd.read_csv(train_csv_path)
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)

    rng = np.random.RandomState(seed)
    val_frac = 0.1
    val_ids = []
    for c in range(NUM_CLASSES):
        ids_c = df.loc[df["label"] == c, "image_id"].values
        rng.shuffle(ids_c)
        n_val = int(round(len(ids_c) * val_frac))
        val_ids.extend(ids_c[:n_val].tolist())
    val_ids_tf = tf.constant(np.array(val_ids, dtype=np.str_), dtype=tf.string)
    keys = val_ids_tf
    values = tf.ones_like(keys, dtype=tf.bool)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=False,
    )

    @tf.function
    def _is_val_tf(image_name):
        return table.lookup(image_name)

    @tf.function
    def _parse_with_name(serialized):
        ex = tf.io.parse_single_example(serialized, _TRAIN_FEATURES)
        img = _decode_resize_from_jpeg_bytes(ex["image"])
        lbl = tf.where(ex["target"] >= 0, ex["target"], ex["label"])
        return img, tf.cast(lbl, tf.int32), ex["image_name"]

    raw = tf.data.TFRecordDataset(
        train_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(DATA_OPTIONS)
    raw = raw.map(_parse_with_name, num_parallel_calls=tf.data.AUTOTUNE)
    raw = raw.apply(tf.data.experimental.ignore_errors())

    train_ds = raw.filter(lambda img, lbl, name: tf.logical_not(_is_val_tf(name)))
    val_ds = raw.filter(lambda img, lbl, name: _is_val_tf(name))

    train_ds = train_ds.map(
        lambda img, lbl, name: (img, lbl), num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda img, lbl, name: (img, lbl), num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


model = _robust_load_model(MODEL_PATH)
if model is not None:
    print("Loaded existing model:", MODEL_PATH)
else:
    print(
        f"MODEL_PATH not found ({MODEL_PATH}); building EfficientNetV2B0(ImageNet) and training for {EPOCHS} epochs."
    )
    model = build_model(IMG_SIZE, NUM_CLASSES)

    train_tfrec_files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
    if len(train_tfrec_files) == 0:
        raise RuntimeError(f"No train tfrecords found in {TRAIN_TFREC_DIR}")

    train_ds, val_ds = _make_train_val_datasets(
        train_tfrec_files, TRAIN_CSV, BATCH_SIZE, seed=SEED
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
    print(
        "Training complete. Last val_accuracy:",
        history.history.get("val_accuracy", [None])[-1],
    )



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)

test_tfrec_files = _list_tfrec_files(TEST_TFREC_DIR, "ld_test")
use_test_tfrecords = len(test_tfrec_files) > 0

if use_test_tfrecords:
    test_ds = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(DATA_OPTIONS)
    test_ds = test_ds.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = _maybe_cache(test_ds, cache_path=None)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
else:
    test_paths = (
        TEST_IMG_DIR.rstrip("/") + "/" + sample_sub["image_id"].astype(str)
    ).to_numpy(dtype=object)

    @tf.function
    def load_test_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3, fancy_upscaling=False)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False
        )
        img = tf.cast(img, tf.float32)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(DATA_OPTIONS)
    test_ds = test_ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = _maybe_cache(test_ds, cache_path=None)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

assert model is not None, "Model is None; cannot run inference."

preds = model.predict(test_ds, verbose=1)
labels = np.argmax(preds, axis=1).astype(int)

if len(labels) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: preds={len(labels)} vs sample_sub={len(sample_sub)}"
    )

submission = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert os.path.exists("submission.csv")
assert submission.columns.tolist() == ["image_id", "label"]
assert submission["image_id"].isna().sum() == 0
assert submission["label"].between(0, NUM_CLASSES - 1).all()
