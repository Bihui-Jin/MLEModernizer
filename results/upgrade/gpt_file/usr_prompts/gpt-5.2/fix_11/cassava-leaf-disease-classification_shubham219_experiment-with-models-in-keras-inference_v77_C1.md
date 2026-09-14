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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
DEBUG = False

np.random.seed(SEED)

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import tensorflow as tf

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
candidate_paths = [
    "/kaggle/input/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    "../input/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    "/kaggle/data/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    "/kaggle/input/fineTuned_v0.59.h5",
    "/kaggle/input/cassava-leaf-disease-classification/fineTuned_v0.59.h5",
    "/kaggle/data/cassava-leaf-disease-classification/fineTuned_v0.59.h5",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/fineTuned_v0.59.h5",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/fineTuned_v0.59.h5",
]

if True:
    extra_roots = ["/kaggle/input", "../input", "/kaggle/data", "../data"]
    for r in extra_roots:
        try:
            candidate_paths.extend(
                sorted(
                    glob.glob(
                        os.path.join(r, "**", "fineTuned_v0.59.h5"), recursive=True
                    )
                )[:20]
            )
        except Exception:
            pass

weight_path = None
for p in candidate_paths:
    if p and os.path.exists(p):
        weight_path = p
        break

my_model = None
using_fallback = False

if weight_path is not None:
    from tensorflow.keras.models import load_model

    custom_objects = {}
    try:
        my_model = load_model(weight_path, compile=False, custom_objects=custom_objects)
    except Exception:
        my_model = load_model(weight_path, custom_objects=custom_objects)
    print("Loaded model from:", weight_path)
else:
    print(
        "WARNING: fineTuned_v0.59.h5 not found; using EfficientNetB0(ImageNet) + trained head fallback."
    )
    using_fallback = True

    inputs = tf.keras.Input(shape=(256, 256, 3))
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    base.trainable = False

    x = base.output
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs, outputs)

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=16,
    )




## === cell 2
train_csv_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "../input/cassava-leaf-disease-classification/train.csv",
    "../data/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]

train_csv_path = None
for p in train_csv_candidates:
    if os.path.exists(p):
        train_csv_path = p
        break

if train_csv_path is None and using_fallback:
    raise FileNotFoundError(
        "train.csv not found for fallback training. Tried:\n"
        + "\n".join(train_csv_candidates)
    )

df_train = None
if train_csv_path is not None:
    df_train = pd.read_csv(train_csv_path)
    print("Loaded train.csv:", train_csv_path, "shape:", df_train.shape)
    print(df_train.head())

train_img_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
    "../data/cassava-leaf-disease-classification/train_images",
]
train_img_dir = None
for d in train_img_dir_candidates:
    if os.path.isdir(d):
        train_img_dir = d
        break

if using_fallback and train_img_dir is None:
    raise FileNotFoundError(
        "train_images directory not found for fallback training. Tried:\n"
        + "\n".join(train_img_dir_candidates)
    )




## === cell 3
from functools import partial

AUTOTUNE = tf.data.AUTOTUNE

_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True

try:
    _DATASET_OPTIONS.experimental_slack = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
except Exception:
    pass


@tf.function
def decode_resize_preprocess(path, label=None, img_size=(256, 256)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)  # [0,255]
    img = tf.keras.applications.efficientnet.preprocess_input(img)  # same as before
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


@tf.function
def apply_train_aug(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    return img, label


def make_train_val_ds(df, img_dir, batch_size=32, img_size=(256, 256), val_frac=0.1):
    paths = (img_dir.rstrip("/") + "/" + df["image_id"].astype(str)).values
    labels = df["label"].values.astype(np.int32)

    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)

    val_n = int(n * val_frac)
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    tr_paths, tr_labels = paths[tr_idx], labels[tr_idx]
    val_paths, val_labels = paths[val_idx], labels[val_idx]

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
    tr_ds = tr_ds.with_options(_DATASET_OPTIONS)

    tr_ds = tr_ds.map(
        partial(decode_resize_preprocess, img_size=img_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    tr_ds = tr_ds.cache()

    tr_ds = tr_ds.shuffle(
        min(len(tr_paths), 8192), seed=SEED, reshuffle_each_iteration=True
    )
    tr_ds = tr_ds.map(
        apply_train_aug,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    tr_ds = tr_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.with_options(_DATASET_OPTIONS)
    val_ds = val_ds.map(
        partial(decode_resize_preprocess, img_size=img_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return tr_ds, val_ds


def make_test_ds(df_paths, batch_size=64, img_size=(256, 256)):
    paths = df_paths.values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_DATASET_OPTIONS)
    ds = ds.map(
        partial(decode_resize_preprocess, label=None, img_size=img_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 4
if using_fallback:
    batch_size = 32
    img_size = (256, 256)

    train_ds, val_ds = make_train_val_ds(
        df_train, train_img_dir, batch_size=batch_size, img_size=img_size, val_frac=0.1
    )

    print("Training fallback head (base frozen)...")
    my_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

    print("Fine-tuning last layers (partial unfreeze)...")
    base_model = None
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
            base_model = layer
            break
    if base_model is None:
        try:
            base_model = my_model.get_layer("efficientnetb0")
        except Exception:
            base_model = None

    if base_model is not None:
        base_model.trainable = True
        for l in base_model.layers[:-30]:
            l.trainable = False

        my_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            steps_per_execution=16,
        )
        my_model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1)
    else:
        print(
            "Could not locate EfficientNet base model layer for partial unfreeze; skipping fine-tune."
        )




## === cell 5
test_glob_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/*.jpg",
    "../input/cassava-leaf-disease-classification/test_images/*.jpg",
    "../data/cassava-leaf-disease-classification/test_images/*.jpg",
]

test_images = []
for g in test_glob_candidates:
    test_images = sorted(glob.glob(g))
    if len(test_images) > 0:
        break

if len(test_images) == 0:
    raise FileNotFoundError(
        "No test images found. Tried:\n" + "\n".join(test_glob_candidates)
    )

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = df_test["path"].str.rsplit("/", n=1).str[-1]

print("Found test images:", len(df_test))
print(df_test.head())




## === cell 6
test_ds = make_test_ds(df_test["path"], batch_size=256, img_size=(256, 256))

pred_test = my_model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

final_csv = final_csv[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 7
assert os.path.exists("submission.csv"), "submission.csv was not created."
assert list(final_csv.columns) == ["image_id", "label"], "Wrong submission columns."
assert len(final_csv) == len(df_test), "Submission row count mismatch."
print(final_csv.sample(5, random_state=SEED))
