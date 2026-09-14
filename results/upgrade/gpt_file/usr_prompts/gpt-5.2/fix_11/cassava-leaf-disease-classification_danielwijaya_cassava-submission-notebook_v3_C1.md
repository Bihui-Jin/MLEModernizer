# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8919613176186159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.11024) has done: 'I remove the failing imports (`kaggle_datasets`, old protobuf-dependent code, and missing `keras_applications/efficientnet` addons) and replace the unavailable pretrained-model loading with a small in-notebook EfficientNet model built from `tf.keras.applications` so the pipeline runs end-to-end. I also fix the TFRecord parsing bug where `dataset.with_options(...)` wasn’t assigned (so ordering/determinism settings were ignored), and make submission generation robust by reading IDs directly from the ordered test dataset and writing a proper `submission.csv`. These changes keep the overall approach (TFRecords → EfficientNet classifier → argmax → CSV) while ensuring it executes in this environment and produces a valid submission file.'

# 9. Code solution

## === cell 0
import os
import re
import random
import subprocess
import sys

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    if int(_pb_ver.split(".")[0]) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception as e:
    print("protobuf compatibility pre-check warning:", repr(e))

import tensorflow as tf

tf.random.set_seed(SEED)

import matplotlib.pyplot as plt

INPUT_ROOT = "/kaggle/input/cassava-leaf-disease-classification"

print("Listing a few input files:")
shown = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if shown < 25:
            print(os.path.join(dirname, filename))
            shown += 1
        else:
            break
    if shown >= 25:
        break

print("TensorFlow:", tf.__version__)



## === cell 1
AUTOTUNE = tf.data.AUTOTUNE
GCS_PATH = INPUT_ROOT

BATCH_SIZE = 32
IMAGE_SIZE = [224, 224]
NUM_CLASSES = 5


def dataset_sizes(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")

NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)
NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)

print(
    "TFRecords:",
    len(TRAIN_FILENAMES),
    "train files;",
    len(TEST_FILENAMES),
    "test files",
)
print("Counts:", NUM_TRAIN_IMAGES, "train images;", NUM_TEST_IMAGES, "test images")

sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")
test_df = pd.read_csv(sample_sub_path)
print("sample_submission.csv shape:", test_df.shape)
print(test_df.head())

train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
train_df = pd.read_csv(train_csv_path)
print("train.csv shape:", train_df.shape)
print(train_df["label"].value_counts().sort_index())


def normalize_image_id_py(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    if not s.lower().endswith(".jpg"):
        s = s + ".jpg"
    return s


train_df["image_id_norm"] = train_df["image_id"].map(normalize_image_id_py)
test_df["image_id_norm"] = test_df["image_id"].map(normalize_image_id_py)

from sklearn.model_selection import train_test_split

train_ids, val_ids = train_test_split(
    train_df["image_id_norm"].values,
    test_size=0.10,
    random_state=SEED,
    stratify=train_df["label"].values,
)

TRAIN_ID_SET = set(train_ids.tolist())
VAL_ID_SET = set(val_ids.tolist())

print("Split sizes:", len(TRAIN_ID_SET), "train;", len(VAL_ID_SET), "val")



## === cell 2
PREPROCESS = tf.keras.applications.efficientnet.preprocess_input


def to_float32(image, label_or_id):
    return tf.cast(image, tf.float32), label_or_id


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    img = PREPROCESS(img)  # expects 0..255 float input
    return img


def _normalize_image_id(image_id_bytes):
    s = tf.ensure_shape(image_id_bytes, [])
    s = tf.io.decode_utf8(s)
    s = tf.strings.strip(s)
    parts = tf.strings.split(s, sep="/")
    s = parts[-1]
    parts2 = tf.strings.split(s, sep="\\")
    s = parts2[-1]
    lower = tf.strings.lower(s)
    has_jpg = tf.strings.regex_full_match(lower, r".*\.jpg")
    s = tf.where(has_jpg, s, tf.strings.join([s, ".jpg"]))
    return s


def read_tfrecord(example, labeled):
    if labeled:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }
    else:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }

    ex = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(ex["image"])

    image_id_raw = tf.where(
        tf.strings.length(ex["image_id"]) > 0, ex["image_id"], ex["image_name"]
    )
    image_id = _normalize_image_id(image_id_raw)

    if labeled:
        label_raw = tf.where(ex["label"] >= 0, ex["label"], ex["target"])
        label = tf.cast(label_raw, tf.int32)
        return img, label, image_id
    else:
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    opts.experimental_deterministic = True if ordered else False

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(opts)
    ds = ds.map(
        lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return ds


_train_ids_tf = tf.constant(sorted(list(TRAIN_ID_SET)), dtype=tf.string)
_val_ids_tf = tf.constant(sorted(list(VAL_ID_SET)), dtype=tf.string)
_train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=_train_ids_tf, values=tf.ones_like(_train_ids_tf, dtype=tf.int32)
    ),
    default_value=0,
)
_val_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=_val_ids_tf, values=tf.ones_like(_val_ids_tf, dtype=tf.int32)
    ),
    default_value=0,
)


def _keep_if_train(img, label, image_id):
    return tf.equal(_train_id_table.lookup(image_id), 1)


def _keep_if_val(img, label, image_id):
    return tf.equal(_val_id_table.lookup(image_id), 1)


def _augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.10, seed=SEED)
    img = tf.image.random_contrast(img, lower=0.90, upper=1.10, seed=SEED)
    return img, label


def get_train_data():
    ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
    ds = ds.filter(_keep_if_train)
    ds = ds.map(lambda img, label, image_id: (img, label), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_val_data():
    ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=True)
    ds = ds.filter(_keep_if_val)
    ds = ds.map(lambda img, label, image_id: (img, label), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


_train_count = int(
    get_train_data().unbatch().take(5000).reduce(0, lambda c, _: c + 1).numpy()
)
_val_count = int(
    get_val_data().unbatch().take(5000).reduce(0, lambda c, _: c + 1).numpy()
)
print(
    "Sanity check (first up to 5000 elems):", _train_count, "train;", _val_count, "val"
)
if _train_count < 1000 or _val_count < 100:
    raise RuntimeError(
        "Filtered train/val datasets look too small. This usually means image_id normalization "
        "still doesn't match TFRecords; training would be ineffective."
    )




## === cell 3
def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


model = build_model()
model.summary()



## === cell 4
train_ds = get_train_data().map(to_float32)
val_ds = get_val_data().map(to_float32)

NUM_TRAIN_SPLIT = len(TRAIN_ID_SET)
NUM_VAL_SPLIT = len(VAL_ID_SET)
STEPS_PER_EPOCH = max(1, NUM_TRAIN_SPLIT // BATCH_SIZE)
VAL_STEPS = max(1, int(np.ceil(NUM_VAL_SPLIT / BATCH_SIZE)))

EPOCHS_HEAD = 5
EPOCHS_FINETUNE = 3

print("Training (frozen base)...")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    verbose=1,
)

print("Fine-tuning (unfreeze top of base at low LR)...")
base = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
        base = layer
        break
if base is None:
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base = layer
            break

if base is not None:
    base.trainable = True
    n_layers = len(base.layers)
    unfreeze_from = max(0, n_layers - 60)
    for l in base.layers[:unfreeze_from]:
        l.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_HEAD,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    verbose=1,
)



## === cell 5
test_ds = get_test_data(ordered=True).map(to_float32)

print("Computing predictions + collecting ids (single ordered pass)...")
all_ids = []
all_preds = []

for x_batch, id_batch in test_ds:
    probs_batch = model.predict(x_batch, verbose=0)
    preds_batch = np.argmax(probs_batch, axis=-1).astype(np.int64)
    all_preds.append(preds_batch)
    all_ids.extend(id_batch.numpy().astype("U").tolist())

predictions = np.concatenate(all_preds, axis=0)
test_ids = np.asarray(all_ids, dtype="U")

print("Predictions shape:", predictions.shape)
print("IDs shape:", test_ids.shape)
assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

print("Generating submission.csv file (strict sample_submission order)...")
pred_df = pd.DataFrame({"image_id_norm": test_ids, "label": predictions})
pred_df = pred_df.drop_duplicates(subset=["image_id_norm"], keep="first")

submission = test_df[["image_id", "image_id_norm"]].merge(
    pred_df, on="image_id_norm", how="left"
)

missing = submission["label"].isna().sum()
if missing != 0:
    missing_examples = (
        submission.loc[submission["label"].isna(), "image_id"].head(10).tolist()
    )
    raise ValueError(
        f"Missing predictions for {missing} test rows (examples: {missing_examples}). "
        f"Check TFRecord id parsing/normalization."
    )

submission["label"] = submission["label"].astype(int)
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())
