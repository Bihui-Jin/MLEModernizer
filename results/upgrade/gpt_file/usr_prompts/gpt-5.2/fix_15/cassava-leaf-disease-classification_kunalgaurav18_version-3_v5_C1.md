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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

IMG_SIZE = 512
NB_CHANNELS = 3
BATCH_SIZE = 32

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## === cell 1
layers = tf.keras.layers
Sequential = tf.keras.models.Sequential
EfficientNetB0 = tf.keras.applications.EfficientNetB0
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

cnn_base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, NB_CHANNELS),
)
cnn_base.trainable = False

cnn = Sequential(
    [
        cnn_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(5, activation="softmax"),
    ]
)
cnn.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"],
)



## === cell 2
CANDIDATE_WEIGHT_PATHS = [
    "/kaggle/input/weights/EffNetB0_512_8_best_weights.h5",
    "/kaggle/input/cassava-leaf-disease-classification/EffNetB0_512_8_best_weights.h5",
    "/kaggle/input/EffNetB0_512_8_best_weights.h5",
    "../input/weights/EffNetB0_512_8_best_weights.h5",
]

found_path = None
for p in CANDIDATE_WEIGHT_PATHS:
    if tf.io.gfile.exists(p):
        found_path = p
        break

if found_path is None:
    top_level = tf.io.gfile.glob("/kaggle/input/*/EffNetB0_512_8_best_weights.h5")
    if top_level:
        top_level.sort()
        found_path = top_level[0]
    else:
        matches = tf.io.gfile.glob("/kaggle/input/**/EffNetB0_512_8_best_weights.h5")
        if matches:
            matches.sort()
            found_path = matches[0]

if found_path is not None:
    cnn.load_weights(found_path)
    print(f"Loaded custom weights from: {found_path}")
else:
    print(
        "Custom weights not found. Will train the Dense(5) head on train.csv with EfficientNetB0 base frozen "
        "so predictions are not near-random."
    )




## === cell 3
@tf.function
def _decode_resize_preprocess(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)  # [0,255]
    img = preprocess_input(img)  # EfficientNet preprocessing
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])  # static shape prevents retracing slowdowns
    return img


def _list_tfrecords(tfrecord_dir, pattern="*.tfrec"):
    if not tf.io.gfile.exists(tfrecord_dir):
        return []
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, pattern))
    files.sort()
    return files


def _parse_tfrecord_example(example_proto, labeled=True, with_id=False):
    feature_spec = {"image": tf.io.FixedLenFeature([], tf.string)}
    if with_id:
        feature_spec["image_id"] = tf.io.FixedLenFeature([], tf.string)
    if labeled:
        feature_spec["target"] = tf.io.FixedLenFeature([], tf.int64, default_value=-1)
        feature_spec["label"] = tf.io.FixedLenFeature([], tf.int64, default_value=-1)

    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img = _decode_resize_preprocess(ex["image"])

    if labeled:
        y = ex["target"]
        y = tf.where(y >= 0, y, ex["label"])
        y = tf.cast(y, tf.int32)
        if with_id:
            return img, y, ex["image_id"]
        return img, y

    if with_id:
        return img, ex["image_id"]
    return img


def make_dataset_from_tfrecords(
    tfrec_files,
    shuffle_files=False,
    shuffle_buffer=4096,
    seed=0,
    labeled=True,
    with_id=False,
):
    if not tfrec_files:
        raise FileNotFoundError("No TFRecord files provided.")

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve determinism
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True

    ds_files = tf.data.Dataset.from_tensor_slices(tfrec_files)
    if shuffle_files:
        ds_files = ds_files.shuffle(
            len(tfrec_files), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
        cycle_length=min(8, len(tfrec_files)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    ds = ds.with_options(options)

    if labeled:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda x: _parse_tfrecord_example(x, labeled=labeled, with_id=with_id),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_dataset(df, img_dir, shuffle=False, seed=0, cache_in_memory=False):
    img_dir = img_dir.rstrip("/")
    paths = (img_dir + "/" + df["image_id"].astype(str).values).astype("U")
    labels = df["label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_preprocess(img_bytes)
        return img, label

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve determinism
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if found_path is None:
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_tfrec_files = _list_tfrecords(TRAIN_TFREC_DIR, pattern="*.tfrec")
    if train_tfrec_files:
        tr_ds = make_dataset_from_tfrecords(
            train_tfrec_files, shuffle_files=False, seed=0, labeled=True, with_id=False
        )
        va_ds = make_dataset(
            va_df, TRAIN_IMG_DIR, shuffle=False, seed=0, cache_in_memory=False
        )
    else:
        tr_ds = make_dataset(
            tr_df, TRAIN_IMG_DIR, shuffle=True, seed=0, cache_in_memory=False
        )
        va_ds = make_dataset(
            va_df, TRAIN_IMG_DIR, shuffle=False, seed=0, cache_in_memory=False
        )

    EPOCHS = 2
    history = cnn.fit(tr_ds, validation_data=va_ds, epochs=EPOCHS, verbose=2)
    print(
        "Finished head training. Last val_accuracy:",
        history.history.get("val_accuracy", [None])[-1],
    )



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()




## === cell 5
def make_test_dataset(image_ids, img_dir):
    img_dir = img_dir.rstrip("/")
    paths = (img_dir + "/" + np.asarray(image_ids, dtype="U")).astype("U")

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_preprocess(img_bytes)
        return img

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(options)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


test_tfrec_files = _list_tfrecords(TEST_TFREC_DIR, pattern="*.tfrec")
if test_tfrec_files:
    test_ds_with_id = make_dataset_from_tfrecords(
        test_tfrec_files, shuffle_files=False, seed=0, labeled=False, with_id=True
    )

    tfrec_ids = []
    for batch in test_ds_with_id:
        _, ids = batch
        tfrec_ids.extend([x.decode("utf-8") for x in ids.numpy().tolist()])

    test_ds = test_ds_with_id.map(
        lambda img, img_id: img, num_parallel_calls=tf.data.AUTOTUNE
    )
    probs = cnn.predict(test_ds, verbose=0)
    preds = np.argmax(probs, axis=1).astype(np.int32)

    if len(preds) != len(tfrec_ids):
        raise RuntimeError(
            f"TFRecord prediction/id mismatch: {len(preds)} preds vs {len(tfrec_ids)} ids"
        )

    pred_map = dict(zip(tfrec_ids, preds.tolist()))
    predictions = np.array([pred_map[i] for i in test_image_ids], dtype=np.int32)

else:
    test_ds = make_test_dataset(test_image_ids, TEST_IMG_DIR)
    steps = (len(test_image_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    probs = cnn.predict(test_ds, steps=steps, verbose=0)
    predictions = np.argmax(probs, axis=1).astype(np.int32)

if len(predictions) != len(test_image_ids):
    raise RuntimeError(
        f"Prediction count mismatch: got {len(predictions)} preds for {len(test_image_ids)} images"
    )

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
