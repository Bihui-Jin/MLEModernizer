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

0.8738289513448172

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'The timeout is most likely coming from (1) decoding/resizing JPEGs for every epoch without caching, (2) using `steps_per_epoch`/`validation_steps` computed from the full split sizes even though you only load a subset of TFRecord shards (so you can end up iterating with `repeat()`-like behavior and excess work), and (3) Python-side prediction loop that materializes all probabilities/ids batch-by-batch. The changes below keep the same model, losses, epochs, and datasets, but make input pipelines cheaper by caching the decoded validation set, computing steps from the actually-loaded datasets, and using `model.predict` to run inference in graph/batched mode without Python accumulation overhead. All changes are runtime-focused and preserve evaluation semantics (same data, same augmentations, same training schedule), with only negligible floating-point differences possible.'

# 9. Code solution

## === cell 0
import os, re, math, json, random
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
    print("Test data IDs sample:", idnum.numpy()[:5].astype("U"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


tr_id_to_label = dict(
    zip(tr_df["image_id"].astype(str).values, tr_df["label"].astype(np.int32).values)
)
va_id_to_label = dict(
    zip(va_df["image_id"].astype(str).values, va_df["label"].astype(np.int32).values)
)

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


@tf.function
def read_tfrecord_image_and_name(example):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(ex["image"])
    name = ex["image_name"]
    return image, name


def _make_labeled_ds_from_shards(
    filenames, id_to_label: dict, ordered: bool, augment: bool, shuffle: bool
):
    keys = tf.constant(list(id_to_label.keys()), dtype=tf.string)
    vals = tf.constant(list(id_to_label.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(-1, tf.int32),
    )

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = _set_ds_options(ds, ordered=ordered)
    ds = ds.map(read_tfrecord_image_and_name, num_parallel_calls=AUTOTUNE)

    def attach_label(image, name):
        label = table.lookup(name)
        return image, label

    ds = ds.map(attach_label, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda im, lb: lb >= 0)

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(data_augment, num_parallel_calls=AUTOTUNE)

    if ordered and (not shuffle) and (not augment):
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_labeled_ds_from_shards(
    train_split_filenames, tr_id_to_label, ordered=False, augment=True, shuffle=True
)
valid_ds = _make_labeled_ds_from_shards(
    valid_split_filenames, va_id_to_label, ordered=True, augment=False, shuffle=False
)


def _cardinality_to_int(card):
    v = int(card.numpy())
    return v


train_card = tf.data.experimental.cardinality(train_ds)
valid_card = tf.data.experimental.cardinality(valid_ds)
steps_per_epoch = _cardinality_to_int(train_card)
validation_steps = _cardinality_to_int(valid_card)
print(
    "steps_per_epoch (dataset):",
    steps_per_epoch,
    "validation_steps (dataset):",
    validation_steps,
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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3656702771.py in <cell line: 0>()
    176 print(model.summary())
    177 
--> 178 history = model.fit(
    179     train_ds,
    180     validation_data=valid_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 2
@tf.function
def to_float32(image, label_or_id):
    return tf.cast(image, tf.float32), label_or_id


print("Computing predictions...")

test_ds_ordered = get_test_dataset(ordered=True).map(
    to_float32, num_parallel_calls=AUTOTUNE
)

id_list = []
for _, batch_ids in test_ds_ordered:
    id_list.append(batch_ids.numpy())
all_ids = np.concatenate(id_list, axis=0).astype("U")

test_ds_for_pred = get_test_dataset(ordered=True).map(
    to_float32, num_parallel_calls=AUTOTUNE
)
probabilities = model.predict(test_ds_for_pred, verbose=0)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

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
