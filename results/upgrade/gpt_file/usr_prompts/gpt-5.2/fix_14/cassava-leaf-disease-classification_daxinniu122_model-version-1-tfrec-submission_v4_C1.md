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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")
SUB_PATH = "/kaggle/working/submission.csv"

NUM_CLASSES = 5
IMG_SIZE = (300, 300)

print("TensorFlow:", tf.__version__)
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("Train CSV exists:", os.path.isfile(TRAIN_CSV_PATH))

try:
    tf.config.experimental.enable_op_determinism()
    print("Enabled TF op determinism.")
except Exception as e:
    print("Could not enable TF op determinism:", repr(e))

try:
    cpu = os.cpu_count() or 0
    tf.config.threading.set_intra_op_parallelism_threads(cpu)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception as e:
    print("Threading config not applied:", repr(e))



## === cell 1
WEIGHTS_FILENAME = "model_ver_3.h5"


def resolve_model_weights_path(filename: str):
    candidates = [
        os.path.join("/kaggle/input", filename),
        os.path.join(INPUT_DIR, filename),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    try:
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)
    except FileNotFoundError:
        pass
    return None


MODEL_WEIGHTS_PATH = resolve_model_weights_path(WEIGHTS_FILENAME)
print("Resolved MODEL_WEIGHTS_PATH:", MODEL_WEIGHTS_PATH)


def build_model(img_size=(300, 300), num_classes=5, imagenet_backbone=False):
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    base = keras.applications.EfficientNetV2B3(
        include_top=False,
        input_tensor=inputs,
        weights="imagenet" if imagenet_backbone else None,
        pooling="avg",
    )
    x = base.output
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


used_custom_weights = False

if MODEL_WEIGHTS_PATH is not None:
    model = build_model(IMG_SIZE, NUM_CLASSES, imagenet_backbone=False)
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("Loaded custom weights successfully from:", MODEL_WEIGHTS_PATH)
        used_custom_weights = True
    except Exception as e:
        print(
            "Direct load_weights failed, trying by_name/skip_mismatch. Error:", repr(e)
        )
        model.load_weights(MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print("Loaded weights by_name/skip_mismatch from:", MODEL_WEIGHTS_PATH)
        used_custom_weights = True
else:
    print(
        f"Custom weights '{WEIGHTS_FILENAME}' not found. "
        "Falling back to ImageNet backbone weights."
    )
    model = build_model(IMG_SIZE, NUM_CLASSES, imagenet_backbone=True)

_ = model(tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32))
print("Model built. used_custom_weights =", used_custom_weights)



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample submission format"
print("Sample submission rows:", len(sample_sub))


@tf.function
def preprocess_image_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet_v2.preprocess_input(img)
    return img


TFREC_TRAIN_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(INPUT_DIR, "test_tfrecords")


def _list_tfrec_files(folder):
    try:
        files = tf.io.gfile.glob(os.path.join(folder, "*.tfrec"))
        return sorted(files)
    except Exception:
        return []


TFREC_TRAIN_FILES = _list_tfrec_files(TFREC_TRAIN_DIR)
TFREC_TEST_FILES = _list_tfrec_files(TFREC_TEST_DIR)

print("Found train tfrecs:", len(TFREC_TRAIN_FILES))
print("Found test tfrecs:", len(TFREC_TEST_FILES))

DATASET_OPTS = tf.data.Options()
DATASET_OPTS.experimental_deterministic = False
DATASET_OPTS.experimental_optimization.apply_default_optimizations = True

try:
    cpu = os.cpu_count() or 4
    DATASET_OPTS.threading.private_threadpool_size = max(4, cpu // 2)
    DATASET_OPTS.threading.max_intra_op_parallelism = cpu
except Exception:
    pass



## === cell 3
train_df = pd.read_csv(TRAIN_CSV_PATH)
assert set(train_df.columns) >= {"image_id", "label"}
print(
    "Train rows:",
    len(train_df),
    "label distribution:",
    train_df["label"].value_counts().to_dict(),
)

train_paths = [os.path.join(TRAIN_IMG_DIR, x) for x in train_df["image_id"].tolist()]
train_labels = train_df["label"].astype(np.int32).values

for p in train_paths[:5]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Train image not found: {p}")

y = tf.convert_to_tensor(train_labels, dtype=tf.int32)

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, y))


@tf.function
def _load_train(path, label):
    img = preprocess_image_from_path(path)
    return img, label


ds_train = ds_train.shuffle(
    buffer_size=min(len(train_df), 8192), seed=SEED, reshuffle_each_iteration=True
)
ds_train = ds_train.map(
    _load_train, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)

BATCH_SIZE_TRAIN = 32
ds_train = ds_train.batch(BATCH_SIZE_TRAIN, drop_remainder=False)
ds_train = ds_train.with_options(DATASET_OPTS).prefetch(tf.data.AUTOTUNE)

if not used_custom_weights:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    EPOCHS = 2
    history = model.fit(ds_train, epochs=EPOCHS, verbose=1)
    print("Finished training. Last epoch acc:", float(history.history["acc"][-1]))
else:
    print(
        "Custom weights provided; skipping training to preserve the original intended inference-only flow."
    )



## === cell 4
_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_and_preprocess_from_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = keras.applications.efficientnet_v2.preprocess_input(img)
    return img, tf.cast(ex["target"], tf.int32), ex["image_name"]


def _make_test_ds_from_tfrecs(tfrec_files, batch_size):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(
        _decode_and_preprocess_from_tfrec,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.with_options(DATASET_OPTS).prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 5
BATCH_SIZE = 128  # inference-only

if TFREC_TEST_FILES:
    ds_test = _make_test_ds_from_tfrecs(TFREC_TEST_FILES, BATCH_SIZE)

    name_chunks = []
    pred_chunks = []

    for img_batch, _, name_batch in ds_test:
        probs_batch = model(img_batch, training=False)
        pred_batch = tf.argmax(probs_batch, axis=1, output_type=tf.int64)
        name_chunks.append(name_batch)  # tf.string tensor
        pred_chunks.append(pred_batch)  # int64 tensor

    names = tf.concat(name_chunks, axis=0).numpy()
    preds_all = tf.concat(pred_chunks, axis=0).numpy().astype(int)

    preds_by_name = dict(zip(names, preds_all))

    sample_ids_bytes = [s.encode("utf-8") for s in sample_sub["image_id"].tolist()]
    missing_keys = [b for b in sample_ids_bytes if b not in preds_by_name]
    if missing_keys:
        raise KeyError(
            f"Missing {len(missing_keys)} test predictions. Example missing: {missing_keys[0]!r}"
        )

    preds = np.fromiter(
        (preds_by_name[b] for b in sample_ids_bytes), dtype=np.int64
    ).astype(int)
else:
    test_paths = [
        os.path.join(TEST_IMG_DIR, img_id) for img_id in sample_sub["image_id"].tolist()
    ]

    missing = [p for p in test_paths[:50] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Some test images were not found. Example missing: {missing[0]}"
        )

    ds = tf.data.Dataset.from_tensor_slices(test_paths)
    ds = ds.map(
        preprocess_image_from_path,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.with_options(DATASET_OPTS).prefetch(tf.data.AUTOTUNE)

    probs = model.predict(ds, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)

print("Preds shape:", preds.shape, "unique:", np.unique(preds))
assert len(preds) == len(
    sample_sub
), f"Pred length {len(preds)} != sample_sub {len(sample_sub)}"



## === cell 6
my_submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})
my_submission.to_csv(SUB_PATH, index=False)

print("Wrote:", SUB_PATH)
print(my_submission.head())
print("Submission rows:", len(my_submission), "columns:", list(my_submission.columns))
assert os.path.isfile(SUB_PATH) and SUB_PATH.endswith(".csv")
assert len(my_submission) == len(sample_sub)
assert set(my_submission.columns) == {"image_id", "label"}
