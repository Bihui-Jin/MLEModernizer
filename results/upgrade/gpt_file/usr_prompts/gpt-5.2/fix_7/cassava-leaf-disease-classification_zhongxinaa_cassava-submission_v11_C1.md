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

3.12

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

0.8496524629797522

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import efficientnet_v2

print("TF version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(WORK_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)

train_df["filepath"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
)

assert len(train_df) > 0, "No training rows found."

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print(
    "Label distribution (train):\n",
    tr_df["label"].value_counts(normalize=True).sort_index(),
)
print(
    "Label distribution (val):\n",
    va_df["label"].value_counts(normalize=True).sort_index(),
)



## === cell 4
aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ]
)


@tf.function
def _decode_resize_preprocess(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    return img


_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION)
    img = _decode_resize_preprocess(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES)
    return img, label


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION)
    img = _decode_resize_preprocess(ex["image"])
    image_id = ex["image_id"]
    return img, image_id


def _list_tfrecords(dir_path, prefix):
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecords found under {dir_path} with prefix {prefix}"
        )
    return files


def make_train_val_ds_from_tfrecords(training=False, cache_path=None):
    files = _list_tfrecords(TRAIN_TFREC_DIR, "ld_train")
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    if training:
        ds_files = ds_files.shuffle(
            len(files), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )

    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_deterministic = not training
    options.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
    options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)

    if cache_path is None:
        cache_path = (
            f"/kaggle/working/tfdata_cache_img_{'train' if training else 'val'}"
        )
    ds = ds.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=not training
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


tr_ids = tf.constant(tr_df["image_id"].astype(str).values)
va_ids = tf.constant(va_df["image_id"].astype(str).values)
tr_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tr_ids, tf.ones_like(tr_ids, dtype=tf.int32)),
    default_value=0,
)
va_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(va_ids, tf.ones_like(va_ids, dtype=tf.int32)),
    default_value=0,
)


def make_filtered_ds(id_table, training, cache_path):
    files = _list_tfrecords(TRAIN_TFREC_DIR, "ld_train")
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    if training:
        ds_files = ds_files.shuffle(
            len(files), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds_files.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_deterministic = not training
    options.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
    options.threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)

    @tf.function
    def _parse_with_id(serialized):
        ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION)
        return ex["image"], ex["image_id"], ex["target"]

    ds = ds.map(_parse_with_id, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.filter(lambda img, image_id, target: tf.equal(id_table.lookup(image_id), 1))
    ds = ds.map(
        lambda img, image_id, target: (
            _decode_resize_preprocess(img),
            tf.one_hot(tf.cast(target, tf.int32), NUM_CLASSES),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = make_filtered_ds(
    tr_id_table, training=True, cache_path="/kaggle/working/tfdata_cache_img_train"
)
val_ds = make_filtered_ds(
    va_id_table, training=False, cache_path="/kaggle/working/tfdata_cache_img_val"
)



## === cell 5
base = efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # start with frozen backbone for stability/speed

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=SigmoidFocalCrossEntropy(from_logits=False),
    metrics=["accuracy"],
)

model.summary()



## === cell 6
ckpt_path = "best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=6, callbacks=callbacks, verbose=1
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=SigmoidFocalCrossEntropy(from_logits=False),
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds, validation_data=val_ds, epochs=3, callbacks=callbacks, verbose=1
)

model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3864057310.py in <cell line: 0>()
      9 ]
     10 
---> 11 history = model.fit(
     12     train_ds, validation_data=val_ds, epochs=6, callbacks=callbacks, verbose=1
     13 )

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

## === cell 7
sub = pd.read_csv(SAMPLE_SUB)
sub["filepath"] = TEST_IMG_DIR.rstrip("/") + "/" + sub["image_id"].astype(str)

test_files = _list_tfrecords(TEST_TFREC_DIR, "ld_test")

ds_files = tf.data.Dataset.from_tensor_slices(test_files)
test_ds = ds_files.interleave(
    lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
    cycle_length=AUTOTUNE,
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
options.threading.max_intra_op_parallelism = 1
test_ds = test_ds.with_options(options)

test_ds = test_ds.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.cache("/kaggle/working/tfdata_cache_img_test")

test_ids_ds = test_ds.map(
    lambda x, image_id: image_id, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_img_ds = test_ds.map(
    lambda x, image_id: x, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
test_ids = np.concatenate(list(test_ids_ds.batch(4096).as_numpy_iterator())).astype(str)

probs = model.predict(test_img_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

if len(test_ids) == len(sub) and np.array_equal(
    test_ids, sub["image_id"].astype(str).values
):
    ordered_preds = preds
else:
    id_to_pred = dict(zip(test_ids, preds))
    ordered_preds = sub["image_id"].astype(str).map(id_to_pred).to_numpy(dtype=int)

sub_out = pd.DataFrame({"image_id": sub["image_id"].values, "label": ordered_preds})
sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("Unique labels:", np.unique(ordered_preds))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3111117050.py in <cell line: 0>()
     38 
     39 test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
---> 40 test_ids = np.concatenate(list(test_ids_ds.batch(4096).as_numpy_iterator())).astype(str)
     41 
     42 probs = model.predict(test_img_ds, verbose=1)

ValueError: need at least one array to concatenate
