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

0.1403747355696585

# 6. Current score

0.75448

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07362) has done: 'Main bottlenecks are (1) the extremely expensive TFRecord split logic that re-parses each example multiple times and does a per-element membership test, and (2) non-fused input pipeline ops (decode/resize/augment) running without caching/TF graph compilation. I keep the exact model/training logic intact, but replace the split with a single-pass TFRecord index mapping (`image_name -> tfrecord`) built once, then construct train/val TFRecord file lists (provably equivalent to your predicate) so the dataset no longer scans/filters the full TFRecord stream twice. I also make the `tf.data` pipeline more efficient (deterministic, parallel map, `cache()` after expensive decode for val, and explicit options), and remove slow Python-side loops for `y_true`/test names by collecting labels/names in one pass. Paths, preprocessing, augmentation, model architecture, optimizer/loss/metrics, and training loop semantics are preserved.'
- What this solution (achieved 0.75448) has done: 'I fix the TensorFlow import crash by disabling XLA JIT (it’s triggering a protobuf incompatibility in this environment) and forcing the legacy Keras backend to avoid the `MessageFactory.GetPrototype` error. Then I fix the `model.fit` progress-bar `math domain error`, which happens when Keras can’t infer dataset cardinality (often due to an empty train/val TFRecord file split); I replace the “split TFRecords by file membership” logic with a single-pass, exact per-example split that guarantees non-empty train/val datasets. Finally, I keep your model/optimizer/loss/training loop semantics intact and ensure the submission is produced with the exact required columns and full row count.'
- What this solution (achieved 0.75448) has done: 'You’re crashing immediately on TensorFlow import with a protobuf `MessageFactory.GetPrototype` incompatibility, so I add a safe “preflight” that forces the pure-Python protobuf implementation before importing TensorFlow (this is the most reliable fix in Kaggle notebooks for that exact error). I keep your model, TFRecord parsing, splitting-by-lookup, and training loop intact; the only functional tweaks are to ensure the TF lookup keys match the TFRecord `image_name` dtype and to avoid any accidental empty splits/cardinality issues. Finally, I keep the same submission construction but make the test name collection fully aligned with predictions to guarantee the CSV is valid and complete. Since your current score (0.75448) is already far above the very low target, I not make any score-improving changes beyond correctness/stability.'
- What this solution (achieved 0.75448) has done: 'I fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by setting one additional environment flag to force TensorFlow to use the pure-Python protobuf implementation consistently, before importing TensorFlow. I keep your model, TFRecord parsing/splitting, training loop, and submission creation identical to avoid changing the already far-above-target score. I also add a small safety fallback that re-attempts the TF import with the safe protobuf setting if the first import still fails in this environment. No score-tuning changes are introduced; the goal is simply to run end-to-end and reliably write `submission.csv`.'
- What this solution (achieved 0.75448) has done: 'Your run is failing before any training because the TensorFlow import exception is not an `AttributeError` in this environment (it typically raises `ImportError`/`TypeError` while protobuf is initializing), so the current `except AttributeError` never catches it. I make the protobuf “pure python” setting happen unconditionally before importing TensorFlow and broaden the import retry to catch the correct exception types, which fixes the `MessageFactory.GetPrototype` crash reliably. I also remove the unused image-folder assertions (you train/infer from TFRecords anyway) so the notebook doesn’t fail if only TFRecords are present. No model/training/pipeline logic is changed, so the score behavior should remain essentially the same (already far above the provided target).'
- What this solution (achieved 0.12033) has done: 'I fix the TensorFlow import crash by setting the protobuf-related environment variables before any TensorFlow/Keras import and by importing TensorFlow only once (the current try/except still allows the `MessageFactory.GetPrototype` failure to escape). I keep your exact data pipeline/model/training logic the same, only making the import step robust so the notebook can run end-to-end. I also add a small safety check to ensure the TF lookup table is initialized before dataset iteration (prevents rare runtime lookup init issues). No score-tuning changes are introduced since your current score is already far above the provided target; the goal is stability and guaranteed `submission.csv` creation.'
- What this solution (achieved 0.75448) has done: 'You’re failing before training because TensorFlow import triggers a protobuf `MessageFactory.GetPrototype` crash, so I make the protobuf “pure-Python” environment settings unconditional and avoid any second/late TF imports that can bypass them. Then your pipeline fails in TF 2.x because `tf.lookup.experimental.initialize_tables()` doesn’t exist; I remove that call and instead ensure lookup tables are initialized in a TF2-compatible way (and the dataset split logic remains identical). With those two fixes, `train_ds/val_ds` are created so model training/evaluation/inference runs end-to-end and writes a valid `submission.csv`. I also keep everything else (model, loss, optimizer, TFRecord parsing, split rule, and training loop semantics) unchanged to nudge score upward only by enabling proper training rather than altering modeling.'
- What this solution (achieved 0.75448) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by moving the protobuf environment settings to the very top and forcing a single, clean TensorFlow import with a broad retry that catches the actual exception types raised in this environment. This is a correctness/stability change only: it does not alter the model, TFRecord parsing, split logic, training loop, or prediction logic, so the score should remain essentially unchanged (and already far above your low target). I also ensure no Keras/TensorFlow submodules are imported before the environment is set, which is the common hidden cause of this crash. Finally, the script still write a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.75448) has done: 'We fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation unconditionally at the very top and by preventing any early TensorFlow/Keras/protobuf-related imports before those environment variables are set. The existing try/except currently still lets that AttributeError escape during TensorFlow’s internal import path, so we broaden the retry and keep it single-path and robust. All model architecture, TFRecord parsing, split logic, training loop, and submission formatting remain the same so the score should stay essentially unchanged (and already far above the provided target). We also add a small safety check to ensure the TFRecord file lists are non-empty before building datasets, preventing silent empty pipelines.'

# 9. Code solution

## === cell 0
import os
import json
import random
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_PARSER"] = "0"

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C_PARSER"] = "0"
    import importlib
    import sys

    if "tensorflow" in sys.modules:
        del sys.modules["tensorflow"]
    tf = importlib.import_module("tensorflow")

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Could not set XLA JIT flag:", e)

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
work_dir = "../input/cassava-leaf-disease-classification/"
train_img_dir = os.path.join(work_dir, "train_images")
test_img_dir = os.path.join(work_dir, "test_images")

assert os.path.exists(os.path.join(work_dir, "train.csv")), "train.csv not found"

if not os.path.isdir(train_img_dir):
    print("Warning: train_images directory not found (OK: using TFRecords).")
if not os.path.isdir(test_img_dir):
    print("Warning: test_images directory not found (OK: using TFRecords).")

print("work_dir contents sample:", os.listdir(work_dir)[:10])




## === cell 2
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed = 21
seed_everything(seed)



## === cell 3
data = pd.read_csv(os.path.join(work_dir, "train.csv"))
print("Train rows:", len(data))
print(data["label"].value_counts().sort_index())



## === cell 4
with open(os.path.join(work_dir, "label_num_to_disease_map.json"), "r") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}

data["class_name"] = data["label"].map(real_labels)
print("Example label map:", real_labels)



## === cell 5
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    data,
    test_size=0.05,
    random_state=42,
    stratify=data["class_name"],
)

print("Train/Val sizes:", len(train_df), len(val_df))



## === cell 6
IMG_SIZE = 456
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 15

train_tfrecord_dir = os.path.join(work_dir, "train_tfrecords")
test_tfrecord_dir = os.path.join(work_dir, "test_tfrecords")
assert os.path.isdir(train_tfrecord_dir), "train_tfrecords directory not found"
assert os.path.isdir(test_tfrecord_dir), "test_tfrecords directory not found"

train_tfrec_files = sorted(
    [
        os.path.join(train_tfrecord_dir, f)
        for f in os.listdir(train_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(test_tfrecord_dir, f)
        for f in os.listdir(test_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)

print(
    "Train TFRecords:", len(train_tfrec_files), "Test TFRecords:", len(test_tfrec_files)
)

assert len(train_tfrec_files) > 0, "No train .tfrec files found"
assert len(test_tfrec_files) > 0, "No test .tfrec files found"



## === cell 7
FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=seed)
    img = tf.image.random_flip_up_down(img, seed=seed)
    return img


@tf.function
def _preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


@tf.function
def _parse_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    img = _decode_and_resize(ex["image"])
    img = _augment(img)
    img = _preprocess(img)
    label = tf.cast(ex["target"], tf.int32)
    label = tf.one_hot(label, depth=n_CLASS)
    return img, label


@tf.function
def _parse_val(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    label = tf.cast(ex["target"], tf.int32)
    label = tf.one_hot(label, depth=n_CLASS)
    return img, label


@tf.function
def _parse_test(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    return img, ex["image_name"]


val_ids = tf.constant(val_df["image_id"].astype(str).tolist(), dtype=tf.string)

val_ids_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=val_ids,
        values=tf.ones_like(val_ids, dtype=tf.int32),
    ),
    default_value=0,
)

_ = val_ids_table.lookup(tf.constant(["__init__"], dtype=tf.string))

options = tf.data.Options()
options.experimental_deterministic = True

raw_all_train = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)


@tf.function
def _is_val(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {"image_name": tf.io.FixedLenFeature([], tf.string)},
    )
    name = ex["image_name"]
    return tf.equal(val_ids_table.lookup(name), 1)


raw_val_ds = raw_all_train.filter(_is_val)
raw_tr_ds = raw_all_train.filter(lambda x: tf.logical_not(_is_val(x)))

train_ds = (
    raw_tr_ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
    .map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    raw_val_ds.map(_parse_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_count = int(
    raw_tr_ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()
)
val_count = int(
    raw_val_ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()
)
print("TFRecord example counts -> train:", train_count, "val:", val_count)
assert (
    train_count > 0 and val_count > 0
), "Empty train/val split after TFRecord filtering; cannot train."

steps_per_epoch = int(np.ceil(train_count / BATCH_SIZE))
val_steps = int(np.ceil(val_count / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)
print("Built tf.data train/val datasets from TFRecords.")




## === cell 8
def create_model():
    model = models.Sequential()
    backbone = EfficientNetB3(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False  # keep as-is
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            256,
            activation="relu",
            bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_CLASS, activation="softmax"))
    return model


leaf_model = create_model()
leaf_model.summary()



## === cell 9
EPOCHS = 20

leaf_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, verbose=1, min_lr=1e-7
    ),
    EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

history = leaf_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)



## === cell 10
if os.path.exists("best_model.keras"):
    loaded_model = tf.keras.models.load_model("best_model.keras", compile=False)
else:
    loaded_model = leaf_model

loaded_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

val_loss, val_acc = loaded_model.evaluate(val_ds, steps=val_steps, verbose=1)
print("Validation accuracy:", val_acc)



## === cell 11
from sklearn.metrics import confusion_matrix, classification_report

val_probs = loaded_model.predict(val_ds, steps=val_steps, verbose=1)
val_pred_idx = np.argmax(val_probs, axis=1)

y_true_batches = []
for _, y in val_ds.as_numpy_iterator():
    y_true_batches.append(np.argmax(y, axis=1))
y_true = np.concatenate(y_true_batches, axis=0)[: len(val_pred_idx)]

print("Confusion Matrix:\n", confusion_matrix(y_true, val_pred_idx))

target_names = [real_labels[i] for i in range(n_CLASS)]
print(classification_report(y_true, val_pred_idx, target_names=target_names))



## === cell 12
ss = pd.read_csv(os.path.join(work_dir, "sample_submission.csv"))

raw_test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds = (
    raw_test_ds.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_names = []
for _, names in test_ds.as_numpy_iterator():
    if isinstance(names, np.ndarray):
        test_names.extend([n.decode("utf-8") for n in names.tolist()])
    else:
        test_names.extend([n.decode("utf-8") for n in list(names)])

test_probs = loaded_model.predict(test_ds, verbose=1)
test_pred_idx = np.argmax(test_probs, axis=1)

test_names = test_names[: len(test_pred_idx)]
assert len(test_names) == len(test_pred_idx), "Test names/predictions length mismatch."

submission = pd.DataFrame({"image_id": test_names, "label": test_pred_idx.astype(int)})
submission = ss[["image_id"]].merge(submission, on="image_id", how="left")

if submission["label"].isna().any():
    fallback = int(data["label"].mode()[0])
    submission["label"] = submission["label"].fillna(fallback)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert len(submission) == len(
    ss
), "Submission row count mismatch vs sample_submission.csv"
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch"
