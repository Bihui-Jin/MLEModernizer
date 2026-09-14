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

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import re, math, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Thread config skipped:", e)

AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 16
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5
CLASSES = ["0", "1", "2", "3", "4"]

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")

print("Found train.csv:", os.path.exists(TRAIN_CSV))
print("Found sample_submission.csv:", os.path.exists(SAMPLE_SUB))
print("Found train_images dir:", os.path.isdir(TRAIN_IMG_DIR))
print("Found train_tfrecords dir:", os.path.isdir(TRAIN_TFREC_DIR))
print("Found test_tfrecords dir:", os.path.isdir(TEST_TFREC_DIR))


def swish_activation(x):
    return K.sigmoid(x) * x


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.keras.backend.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)


@tf.function
def decode_image(image):
    image = tf.io.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(
        image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function
def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.deterministic = bool(ordered)
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.noop_elimination = True

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "ld_test*.tfrec"))
TRAIN_FILENAMES = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "ld_train*.tfrec"))


def count_data_items(filenames):
    if not filenames:
        return 0
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
NUM_TRAIN_IMAGES = count_data_items(TRAIN_FILENAMES)
print("Num train TFRecords:", len(TRAIN_FILENAMES))
print("Num train images (from TFRecord filenames):", NUM_TRAIN_IMAGES)
print("Num test TFRecords:", len(TEST_FILENAMES))
print("Num test images (from TFRecord filenames):", NUM_TEST_IMAGES)


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return dataset


print("Test data shapes (first batch):")
for image, idnum in get_test_dataset().take(1):
    print(image.shape, idnum.shape)
    try:
        ids_preview = idnum.numpy()[:5]
        if ids_preview.dtype.kind in ("S", "O"):
            ids_preview = ids_preview.astype("U")
        print("Test data IDs sample:", ids_preview)
    except Exception as e:
        print("Could not preview ids:", e)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print("Train df:", train_df.shape, train_df.columns.tolist())

train_df["filepath"] = (TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)).values

missing = (~train_df["filepath"].map(os.path.exists)).sum()
print("Missing train image files:", int(missing))

tr_df, va_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)


def _set_ds_options(ds, ordered: bool):
    options = tf.data.Options()
    options.deterministic = bool(ordered)
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.noop_elimination = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    return ds.with_options(options)


def _build_tfrecord_shard_index(train_filenames):
    shard_pat = re.compile(r"ld_train(\d+)-\d+\.tfrec$")
    shard_to_file = {}
    for fn in train_filenames:
        m = shard_pat.search(os.path.basename(fn))
        if m:
            shard_to_file[int(m.group(1))] = fn
    return shard_to_file


def _image_ids_to_shards(image_ids):
    shards = set()
    for s in image_ids:
        base = os.path.splitext(str(s))[0]
        try:
            shards.add(int(base) % 100)
        except Exception:
            shards.add(abs(hash(base)) % 100)
    return shards


def _select_filenames_for_ids(train_filenames, image_ids):
    shard_to_file = _build_tfrecord_shard_index(train_filenames)
    needed_shards = _image_ids_to_shards(image_ids)
    selected = [shard_to_file[i] for i in sorted(needed_shards) if i in shard_to_file]
    if not selected:
        return list(train_filenames)
    return selected


train_split_filenames = _select_filenames_for_ids(
    TRAIN_FILENAMES, tr_df["image_id"].astype(str).values
)
valid_split_filenames = _select_filenames_for_ids(
    TRAIN_FILENAMES, va_df["image_id"].astype(str).values
)

print(
    "Selected train TFRecord shards:",
    len(train_split_filenames),
    "of",
    len(TRAIN_FILENAMES),
)
print(
    "Selected valid TFRecord shards:",
    len(valid_split_filenames),
    "of",
    len(TRAIN_FILENAMES),
)


def _make_labeled_ds_direct(filenames, ordered: bool, augment: bool, shuffle: bool):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = _set_ds_options(ds, ordered=ordered)
    ds = ds.map(lambda x: read_tfrecord(x, labeled=True), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(data_augment, num_parallel_calls=AUTOTUNE)

    if ordered and (not shuffle) and (not augment):
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_labeled_ds_direct(
    train_split_filenames, ordered=False, augment=True, shuffle=True
)
valid_ds = _make_labeled_ds_direct(
    valid_split_filenames, ordered=True, augment=False, shuffle=False
)

steps_per_epoch = int(math.ceil(len(tr_df) / BATCH_SIZE))
validation_steps = int(math.ceil(len(va_df) / BATCH_SIZE))

print(
    "steps_per_epoch:",
    steps_per_epoch,
    "validation_steps:",
    validation_steps,
    "| train/valid sizes:",
    len(tr_df),
    len(va_df),
)

inputs = Input(shape=(*IMAGE_SIZE, 3), name="image")
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_ft = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 2
@tf.function
def to_float32(image, label_or_id):
    return tf.cast(image, tf.float32), label_or_id


print("Computing predictions...")

test_ds_ordered = get_test_dataset(ordered=True)

all_ids = []
all_preds = []

for batch_images, batch_ids in test_ds_ordered:
    probs = model(batch_images, training=False)
    preds = tf.argmax(probs, axis=-1, output_type=tf.int64)

    id_arr = batch_ids.numpy()
    if id_arr.dtype.kind in ("S", "O"):
        id_arr = id_arr.astype("U")
    all_ids.append(id_arr.astype("U"))
    all_preds.append(preds.numpy())

all_ids = np.concatenate(all_ids, axis=0).astype("U")
predictions = np.concatenate(all_preds, axis=0).astype(np.int64)

print("Preds:", predictions.shape, "IDs:", all_ids.shape)

sub = pd.read_csv(SAMPLE_SUB)
sub_ids = sub["image_id"].values.astype(str)

pred_df = pd.DataFrame({"image_id": all_ids, "label": predictions})
pred_df = pred_df.set_index("image_id").reindex(sub_ids)

if pred_df["label"].isna().any():
    fill_val = int(pd.Series(predictions).mode().iloc[0])
    pred_df["label"] = pred_df["label"].fillna(fill_val).astype(int)
else:
    pred_df["label"] = pred_df["label"].astype(int)

pred_df = pred_df.reset_index()

print("Generating submission.csv file...")
pred_df.to_csv("submission.csv", index=False)

print(pred_df.head())

print(
    "Wrote:",
    os.path.abspath("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)
print("Columns:", pred_df.columns.tolist(), "rows:", len(pred_df))
