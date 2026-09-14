# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.080386823813841

# 6. Current score

0.58408

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.58408) has done: 'The timeout is almost certainly caused by the fallback training path: it decodes and resizes ~18k JPEGs to 512×512 for 2 epochs on-the-fly, which is far too slow within 600 seconds. The fastest equivalent fix is to ensure we always use the provided TFRecords (which avoid per-file Python/OS overhead) and to remove the accidental creation-and-then-nullification of TFRecord datasets in the training block. For prediction, we keep TFRecords but increase input pipeline throughput with non-semantic changes (dataset options + deterministic settings preserved) and remove redundant step-counting where Keras can infer it, while keeping identical batching and evaluation semantics. No model architecture, loss, image size, epochs, or data content is changed.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf

print("Tensorflow version " + tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

IMG_SIZE = 512  # preserve original core logic: resize to 512x512
BATCH_SIZE = 16  # conservative to avoid OOM at 512x512 on Kaggle CPU/GPU




## === cell 2
@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(
        img_bytes,
        channels=3,
        fancy_upscaling=False,
        dct_method="INTEGER_FAST",
    )
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1], float32
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    return img


@tf.function
def _decode_and_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    return _decode_and_resize_from_bytes(img_bytes)


def _list_tfrecord_files(tfrecord_dir):
    if not os.path.isdir(tfrecord_dir):
        return []
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrec"))
    files.sort()
    return files


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_and_resize_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_and_resize_from_bytes(ex["image"])
    return img


def make_image_ds_from_ids(
    image_dir,
    ids,
    labels=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache=False,
    drop_remainder=False,
):
    ids = np.asarray(ids).astype(str)
    paths = np.char.add(np.char.add(image_dir, os.sep), ids)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(ids), seed=SEED, reshuffle_each_iteration=True
            )
        ds = ds.map(
            _decode_and_resize_from_path,
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        labels = np.asarray(labels, dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(ids), seed=SEED, reshuffle_each_iteration=True
            )

        @tf.function
        def _map_path_label(p, y):
            img = _decode_and_resize_from_path(p)
            return img, y

        ds = ds.map(_map_path_label, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)
    return ds


def make_train_ds_from_tfrecords(
    tfrecord_files,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache=False,
    drop_remainder=False,
):
    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=not shuffle
    )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(
    tfrecord_files,
    batch_size=BATCH_SIZE,
    cache=False,
    drop_remainder=False,
):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(
        tfrecord_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
    return ds


def _steps(num_examples, batch_size, drop_remainder=False):
    if drop_remainder:
        return num_examples // batch_size
    return (num_examples + batch_size - 1) // batch_size




## === cell 3
submission = pd.read_csv(SAMPLE_SUB)
test_ids = submission["image_id"].values
print("Test images:", len(test_ids))




## === cell 4
MODEL_PATH = "../input/model-trained/my_model.h5"


def _candidate_model_paths():
    candidates = [MODEL_PATH]
    candidates += [
        "/kaggle/input/model-trained/my_model.h5",
        "/kaggle/input/model-trained/my_model",
        "../input/model-trained/my_model",
    ]
    base = "/kaggle/input"
    if os.path.isdir(base):
        for d in os.listdir(base):
            p1 = os.path.join(base, d, "my_model.h5")
            p2 = os.path.join(base, d, "my_model")
            candidates.extend([p1, p2])
    seen = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _load_model_if_exists():
    for p in _candidate_model_paths():
        if os.path.isfile(p) or os.path.isdir(p):
            try:
                m = tf.keras.models.load_model(p)
                print(f"Loaded model from: {p}")
                return m
            except Exception as e:
                print(
                    f"Found model at {p} but failed to load ({type(e).__name__}: {e}). Trying next..."
                )
                continue
    return None


model = _load_model_if_exists()

if model is None:
    print(
        f"Pretrained model not found at {MODEL_PATH} (or alternate locations). Training a fallback model from train_tfrecords/train.csv..."
    )

    train_df = pd.read_csv(TRAIN_CSV)
    train_ids = train_df["image_id"].values
    train_labels = train_df["label"].values.astype(np.int32)

    idx = np.arange(len(train_ids))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_ids, tr_y = train_ids[tr_idx], train_labels[tr_idx]
    va_ids, va_y = train_ids[va_idx], train_labels[va_idx]

    tfrec_files = _list_tfrecord_files(TRAIN_TFREC_DIR)

    if tfrec_files:
        ds_train = make_train_ds_from_tfrecords(
            tfrec_files,
            batch_size=BATCH_SIZE,
            shuffle=True,
            cache=False,  # keep as original intent
            drop_remainder=False,
        )
        ds_val = make_train_ds_from_tfrecords(
            tfrec_files,
            batch_size=BATCH_SIZE,
            shuffle=False,
            cache=False,
            drop_remainder=False,
        )
    else:
        def make_image_ds_from_ids_fast_train(image_dir, ids, labels, batch_size):
            ids = np.asarray(ids).astype(str)
            paths = np.char.add(np.char.add(image_dir, os.sep), ids)
            labels = np.asarray(labels, dtype=np.int32)

            options = tf.data.Options()
            options.experimental_deterministic = False
            try:
                options.experimental_optimization.apply_default_optimizations = True
                options.experimental_optimization.map_parallelization = True
                options.experimental_optimization.map_and_batch_fusion = True
                options.experimental_optimization.parallel_batch = True
            except Exception:
                pass

            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            ds = ds.shuffle(
                buffer_size=len(ids), seed=SEED, reshuffle_each_iteration=True
            )

            @tf.function
            def _map_path_label(p, y):
                img = _decode_and_resize_from_path(p)
                return img, y

            ds = ds.with_options(options)
            ds = ds.map(
                _map_path_label, num_parallel_calls=AUTOTUNE, deterministic=False
            )
            ds = ds.apply(tf.data.experimental.ignore_errors())
            ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
            return ds

        def make_image_ds_from_ids_fast_eval(image_dir, ids, labels, batch_size):
            return make_image_ds_from_ids(
                image_dir,
                ids,
                labels,
                batch_size=batch_size,
                shuffle=False,
                cache=False,
                drop_remainder=False,
            )

        ds_train = make_image_ds_from_ids_fast_train(
            TRAIN_IMG_DIR, tr_ids, tr_y, BATCH_SIZE
        )
        ds_val = make_image_ds_from_ids_fast_eval(
            TRAIN_IMG_DIR, va_ids, va_y, BATCH_SIZE
        )

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_steps = _steps(len(tr_ids), BATCH_SIZE, drop_remainder=False)
    val_steps = _steps(len(va_ids), BATCH_SIZE, drop_remainder=False)

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=2,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=2,
    )

if model is None:
    raise RuntimeError("Model was not created/loaded; cannot proceed to prediction.")




## === cell 5
test_tfrec_files = _list_tfrecord_files(TEST_TFREC_DIR)
if test_tfrec_files:
    ds_test = make_test_ds_from_tfrecords(
        test_tfrec_files,
        batch_size=BATCH_SIZE,
        cache=False,
        drop_remainder=False,
    )
else:
    ds_test = make_image_ds_from_ids(
        TEST_IMG_DIR,
        test_ids,
        labels=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
        cache=False,
        drop_remainder=False,
    )

preds = model.predict(ds_test, verbose=1)
pred_labels = preds.argmax(axis=1).astype(np.int32)

pred_labels = pred_labels[: len(submission)]

submission["label"] = pred_labels
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)




## === cell 6
preds, submission.head()
