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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model

TRAIN_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

MODELS_WEIGHTS = "/kaggle/input/cassavaeffentb7models/content/Models"

print("TensorFlow:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())
print("ALL Modules are successfully loaded")

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
_MODEL_PATH_CACHE = {}


def _listdir_safe(path):
    try:
        return tf.io.gfile.listdir(path)
    except Exception:
        try:
            return os.listdir(path)
        except Exception:
            return []


def _isdir(path):
    try:
        return tf.io.gfile.isdir(path)
    except Exception:
        return os.path.isdir(path)


def _exists(path):
    try:
        return tf.io.gfile.exists(path)
    except Exception:
        return os.path.exists(path)


def _join(a, b):
    return a.rstrip("/") + "/" + b


def find_model_path(root_dir: str, max_depth: int = 4):
    from collections import deque

    if root_dir in _MODEL_PATH_CACHE:
        return _MODEL_PATH_CACHE[root_dir]

    if not _exists(root_dir) or not _isdir(root_dir):
        _MODEL_PATH_CACHE[root_dir] = None
        return None

    preferred_names = (
        "best.h5",
        "best.keras",
        "model.h5",
        "model.keras",
        "effnet.h5",
        "effnet.keras",
        "efficientnet.h5",
        "efficientnet.keras",
        "b7.h5",
        "b7.keras",
    )

    queue = deque([(root_dir, 0)])
    candidates = []
    savedmodel_dirs = []

    while queue:
        cur, depth = queue.popleft()
        entries = _listdir_safe(cur)
        if not entries:
            continue

        if "saved_model.pb" in entries:
            savedmodel_dirs.append(cur)

        for name in entries:
            p = _join(cur, name)
            lname = name.lower()

            if lname in preferred_names:
                _MODEL_PATH_CACHE[root_dir] = p
                return p

            if lname.endswith(".h5") or lname.endswith(".keras"):
                candidates.append(p)
                continue

            if depth < max_depth and _isdir(p):
                queue.append((p, depth + 1))

    if candidates:
        preferred = []
        for c in candidates:
            lc = os.path.basename(c).lower()
            if ("best" in lc) or ("eff" in lc) or ("b7" in lc) or ("model" in lc):
                preferred.append(c)
        pick = sorted(preferred or candidates, key=lambda p: (len(p), p))[0]
        _MODEL_PATH_CACHE[root_dir] = pick
        return pick

    if savedmodel_dirs:
        pick = sorted(savedmodel_dirs, key=lambda p: (len(p), p))[0]
        _MODEL_PATH_CACHE[root_dir] = pick
        return pick

    _MODEL_PATH_CACHE[root_dir] = None
    return None


model_path = find_model_path(MODELS_WEIGHTS)

if model_path is None:
    narrow_fallback_roots = [
        "/kaggle/input/cassavaeffentb7models",
        "/kaggle/input/cassava-leaf-disease-classification",
    ]
    for root in narrow_fallback_roots:
        model_path = find_model_path(root)
        if model_path is not None:
            break

print("Found model at:", model_path if model_path is not None else "None")

model = None
if model_path is not None:
    try:
        model = load_model(model_path, compile=False)
        print("Model Loading Complete")
    except Exception as e:
        print(
            "Failed to load discovered model path; will fall back to training a lightweight model."
        )
        print("Load error:", repr(e))
        model = None




## === cell 2
sub = pd.read_csv(SAMPLE_CSV)
assert {"image_id", "label"}.issubset(
    sub.columns
), f"Unexpected columns in sample submission: {sub.columns.tolist()}"
first_img_id = str(sub.loc[0, "image_id"])

test_img_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]
TEST_IMG_DIR = None
for cand in test_img_dir_candidates:
    if os.path.isdir(cand):
        TEST_IMG_DIR = cand
        break
if TEST_IMG_DIR is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {test_img_dir_candidates}"
    )

sample_path = os.path.join(TEST_IMG_DIR, first_img_id)
print("Using TEST_IMG_DIR:", TEST_IMG_DIR)
print("Example test image exists:", os.path.exists(sample_path), sample_path)




## === cell 3
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (380, 380)


@tf.function
def _load_one_tf(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


x = _load_one_tf(tf.constant(sample_path))
x = tf.expand_dims(x, 0).numpy()
print("Single image batch shape:", x.shape)




## === cell 4
NUM_CLASSES = 5


def build_fallback_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    inp = layers.Input(shape=(img_size[0], img_size[1], 3))
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    base.trainable = False
    x = layers.Dropout(0.2)(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def make_tf_dataset_from_df(df, img_dir, batch_size=16, shuffle=True, repeat=False):
    img_ids = df["image_id"].astype(str).to_numpy()
    paths = (pd.Series(img_ids).radd(img_dir.rstrip("/") + "/")).to_numpy()

    labels = df["label"].astype(int).to_numpy() if "label" in df.columns else None

    def _load(path, label=None):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method="bilinear")
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        if label is None:
            return img
        return img, label

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: _load(p, None), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(min(len(df), 4096), seed=42, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if model is None:
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = int(0.1 * len(train_df))
    val_df = train_df.iloc[:val_size].copy()
    trn_df = train_df.iloc[val_size:].copy()

    model = build_fallback_model()
    steps_per_epoch = int(np.ceil(len(trn_df) / 16))
    val_steps = int(np.ceil(len(val_df) / 16))

    trn_ds = make_tf_dataset_from_df(trn_df, TRAIN_IMG_LOC, batch_size=16, shuffle=True)
    val_ds = make_tf_dataset_from_df(
        val_df, TRAIN_IMG_LOC, batch_size=16, shuffle=False
    )

    model.fit(
        trn_ds,
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )

print("Model ready. Output shape:", model.output_shape)

preds = model.predict(x, verbose=0)
print("Single-image preds shape:", preds.shape)
print("Single-image predicted label:", int(np.argmax(preds, axis=1)[0]))




## === cell 5
sub = pd.read_csv(SAMPLE_CSV)

image_ids = sub["image_id"].astype(str).to_numpy()
test_paths = (np.char.add(TEST_IMG_DIR.rstrip("/") + "/", image_ids)).tolist()

if not os.path.exists(test_paths[0]):
    raise FileNotFoundError(f"Test image not found: {test_paths[0]}")

print("Number of test images:", len(test_paths))




## === cell 6
batch_size = 32  # preserves outputs; improves throughput by reducing per-step overhead

paths_tensor = tf.constant(test_paths)


@tf.function
def _load_test(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(
        img_bytes, channels=3
    )  # JPEG-only dataset: faster than decode_image
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(paths_tensor)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
test_ds = test_ds.with_options(options)

test_ds = test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size, drop_remainder=False)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

try:
    if tf.config.list_logical_devices("GPU"):
        test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
except Exception:
    pass

pred_steps = int(np.ceil(len(test_paths) / batch_size))
all_preds = model.predict(test_ds, steps=pred_steps, verbose=0)

labels = np.argmax(all_preds, axis=1).astype(int)

submission = pd.DataFrame({"image_id": image_ids, "label": labels})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## === cell 7
print(submission.tail())
print(
    "submission.csv saved in current working directory:",
    os.path.abspath("submission.csv"),
)
print(
    "Unique predicted labels:",
    submission["label"].value_counts().sort_index().to_dict(),
)
