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
import sys
import warnings
import subprocess
import pandas as pd
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def _safe_import_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        err = repr(e)
        if "GetPrototype" in err or "google.protobuf" in err or "protobuf" in err:
            warnings.warn(
                "TensorFlow import failed due to a protobuf-related error. "
                "Attempting a local/offline-compatible pin to protobuf==3.20.* and retrying.\n"
                f"Original error: {err}"
            )
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"]
                )
            except Exception as pip_e:
                warnings.warn(
                    f"pip install protobuf==3.20.* failed (may be offline/no wheel). "
                    f"Will retry TF import anyway. pip error: {repr(pip_e)}"
                )
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import importlib

            importlib.invalidate_caches()
            import tensorflow as tf  # noqa: F401

            return tf
        raise


tf = _safe_import_tensorflow()
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (
    img_to_array,
)  # kept to preserve original imports

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
sample_sub_path = f"{INPUT_DIR}/sample_submission.csv"
train_csv_path = f"{INPUT_DIR}/train.csv"
train_img_dir = f"{INPUT_DIR}/train_images/"
test_img_dir = f"{INPUT_DIR}/test_images/"

sample_sub = pd.read_csv(sample_sub_path)

model_candidates = [
    "../input/resnet50-ver-2/ResNet50_ver_1.h5",
    "/kaggle/input/resnet50-ver-2/ResNet50_ver_1.h5",
    "/kaggle/input/resnet50-ver-2/ResNet50_ver_1.h5".replace("//", "/"),
]

exact_name = "ResNet50_ver_1.h5"


def _find_exact_h5_fast(root, target_name, max_dirs=2000):
    if not os.path.isdir(root):
        return None
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        scanned += 1
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
        if scanned >= max_dirs:
            break
    return None


for root in ("/kaggle/input/resnet50-ver-2", "/kaggle/input"):
    p = _find_exact_h5_fast(root, exact_name)
    if p is not None:
        model_candidates.append(p)

seen = set()
model_candidates = [p for p in model_candidates if not (p in seen or seen.add(p))]

model_path = next(
    (p for p in model_candidates if isinstance(p, str) and os.path.exists(p)), None
)

model1 = None
if model_path is not None:
    try:
        try:
            model1 = tf.keras.models.load_model(
                model_path, compile=False, safe_mode=False
            )
        except TypeError:
            model1 = tf.keras.models.load_model(model_path, compile=False)
        print("Loaded external model:", model_path)
    except Exception as e:
        warnings.warn(
            f"Failed to load external model at {model_path} due to: {repr(e)}\n"
            "Will fall back to training a small baseline model from train_images."
        )
        model1 = None
else:
    warnings.warn(
        "Could not find any .h5 model in /kaggle/input (including ResNet50_ver_1.h5). "
        "Will fall back to training a small baseline model from train_images."
    )




## === cell 1
IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_CLASSES = 5


def build_baseline_model(input_shape=(512, 512, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_train_dataset(train_df):
    paths = tf.strings.join(
        [tf.constant(train_img_dir), tf.constant(train_df["image_id"].values)]
    )
    labels = tf.constant(train_df["label"].values.astype(np.int64))
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve determinism
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    ds = ds.shuffle(min(len(train_df), 4096), seed=42, reshuffle_each_iteration=True)

    @tf.function
    def _load_one(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if model1 is None:
    train_df = pd.read_csv(train_csv_path)
    train_ds = make_train_dataset(train_df)

    model1 = build_baseline_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES
    )

    steps_per_epoch = len(train_df) // BATCH_SIZE
    model1.fit(train_ds, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)




## === cell 2
def make_test_dataset(image_ids):
    image_ids_t = tf.constant(np.asarray(image_ids))
    paths = tf.strings.join([tf.constant(test_img_dir), image_ids_t])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    @tf.function
    def _load_one(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        return img

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ids = sample_sub.image_id.tolist()
test_ds = make_test_dataset(test_ids)

prediction1 = model1.predict(test_ds, verbose=0)
preds = np.argmax(prediction1, axis=-1).astype(int)

if len(preds) != len(test_ids):
    warnings.warn(
        f"Predictions length ({len(preds)}) != test_ids length ({len(test_ids)}). "
        "Padding/truncating to match submission length."
    )
    if len(preds) < len(test_ids):
        pad_val = (
            int(np.bincount(preds, minlength=NUM_CLASSES).argmax()) if len(preds) else 0
        )
        preds = np.concatenate(
            [preds, np.full((len(test_ids) - len(preds),), pad_val, dtype=int)]
        )
    else:
        preds = preds[: len(test_ids)]

preds = np.clip(preds, 0, NUM_CLASSES - 1).astype(int)

my_submission = pd.DataFrame({"image_id": test_ids, "label": preds.tolist()})
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(my_submission.head())
print("Rows:", len(my_submission))
print("Unique labels:", sorted(my_submission["label"].unique().tolist()))
