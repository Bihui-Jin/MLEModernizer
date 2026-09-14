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
import os, warnings, json, math, re

warnings.simplefilter("ignore")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, Sequential, load_model

tf.keras.backend.clear_session()

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

_RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
IMG_SIZE = (512, 512)
AUTO = tf.data.AUTOTUNE


@tf.function
def _decode_resize_norm(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=_RESIZE_METHOD, antialias=False)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _decode_only(p):
    return _decode_resize_norm(p)


@tf.function
def _decode_with_label(p, y):
    return _decode_resize_norm(p), y


@tf.function
def _augment_with_label(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, y




## === cell 1
LOAD_DIR_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "../input/cassava-leaf-disease-classification/",
]

load_dir = None
for cand in LOAD_DIR_CANDIDATES:
    if os.path.exists(os.path.join(cand, "sample_submission.csv")):
        load_dir = cand
        break

if load_dir is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv under expected Kaggle input/data paths. "
        f"Tried: {LOAD_DIR_CANDIDATES}"
    )

BATCH_SIZE = 32

strategy = tf.distribute.get_strategy()

sub_df = pd.read_csv(os.path.join(load_dir, "sample_submission.csv"))
sub_df["paths"] = (
    os.path.join(load_dir, "test_images") + "/" + sub_df["image_id"].astype(str)
)

options_det = tf.data.Options()
options_det.experimental_deterministic = True
options_det.experimental_optimization.apply_default_optimizations = True
options_det.experimental_optimization.map_parallelization = True
options_det.experimental_optimization.parallel_batch = True
options_det.experimental_slack = True

options_nondet = tf.data.Options()
options_nondet.experimental_deterministic = False
options_nondet.experimental_optimization.apply_default_optimizations = True
options_nondet.experimental_optimization.map_parallelization = True
options_nondet.experimental_optimization.parallel_batch = True
options_nondet.experimental_slack = True

test_dataset = tf.data.Dataset.from_tensor_slices(sub_df["paths"].values)
test_dataset = test_dataset.with_options(options_nondet)
test_dataset = test_dataset.map(
    _decode_only, num_parallel_calls=AUTO, deterministic=False
)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())
test_dataset = test_dataset.prefetch(AUTO)

train_csv_path = os.path.join(load_dir, "train.csv")
train_images_dir = os.path.join(load_dir, "train_images")

if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"Missing train.csv at {train_csv_path}")
if not os.path.exists(train_images_dir):
    raise FileNotFoundError(f"Missing train_images dir at {train_images_dir}")




## === cell 2
PRETRAINED_MODEL_PATH_CANDIDATES = [
    "../input/building-a-cnn-base-model-with-keras-gpu/Model_3.h5",
    "/kaggle/input/building-a-cnn-base-model-with-keras-gpu/Model_3.h5",
    "/kaggle/working/building-a-cnn-base-model-with-keras-gpu/Model_3.h5",
]
PRETRAINED_MODEL_PATH = None
for p in PRETRAINED_MODEL_PATH_CANDIDATES:
    if os.path.exists(p):
        PRETRAINED_MODEL_PATH = p
        break


def build_fallback_model(input_shape=(512, 512, 3), num_classes=5):
    model = Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def make_train_val_datasets(
    train_csv_path,
    train_images_dir,
    img_size=(512, 512),
    batch_size=32,
    val_frac=0.1,
    seed=42,
):
    df = pd.read_csv(train_csv_path)
    df["paths"] = os.path.join(train_images_dir, "") + df["image_id"].astype(str)
    labels = df["label"].astype(np.int32).values

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    val_size = int(len(df) * val_frac)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    tr_paths = df.loc[tr_idx, "paths"].values
    tr_labels = labels[tr_idx]
    va_paths = df.loc[val_idx, "paths"].values
    va_labels = labels[val_idx]

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
    tr_ds = tr_ds.with_options(options_det)

    tr_ds = tr_ds.map(_decode_with_label, num_parallel_calls=AUTO, deterministic=True)
    tr_ds = tr_ds.apply(tf.data.experimental.ignore_errors())

    tr_ds = tr_ds.map(_augment_with_label, num_parallel_calls=AUTO, deterministic=True)
    tr_ds = tr_ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)

    tr_ds = tr_ds.batch(batch_size, drop_remainder=True)
    tr_ds = tr_ds.prefetch(AUTO)

    va_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_labels))
    va_ds = va_ds.with_options(options_det)
    va_ds = va_ds.map(_decode_with_label, num_parallel_calls=AUTO, deterministic=True)
    va_ds = va_ds.apply(tf.data.experimental.ignore_errors())
    va_ds = va_ds.batch(batch_size, drop_remainder=False)
    va_ds = va_ds.prefetch(AUTO)

    return tr_ds, va_ds, len(tr_paths), len(va_paths)




## === cell 3
with strategy.scope():
    if PRETRAINED_MODEL_PATH is not None:
        model = load_model(PRETRAINED_MODEL_PATH)

        if not getattr(model, "compiled_loss", None):
            try:
                model.compile()
            except Exception:
                pass
    else:
        model = build_fallback_model(input_shape=(*IMG_SIZE, 3), num_classes=5)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )

        tr_ds, va_ds, n_tr, n_va = make_train_val_datasets(
            train_csv_path=train_csv_path,
            train_images_dir=train_images_dir,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            val_frac=0.1,
            seed=SEED,
        )

        steps_per_epoch = (n_tr + BATCH_SIZE - 1) // BATCH_SIZE
        validation_steps = (n_va + BATCH_SIZE - 1) // BATCH_SIZE

        model.fit(
            tr_ds,
            validation_data=va_ds,
            epochs=2,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2,
        )

    model.summary()




## === cell 4
preds = model.predict(test_dataset, verbose=1)
if preds.ndim != 2 or preds.shape[0] != len(sub_df):
    raise ValueError(
        f"Unexpected prediction shape {preds.shape}, expected (n_test, n_classes)."
    )

sub_df["label"] = preds.argmax(axis=1).astype(int)

submission = sub_df[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
