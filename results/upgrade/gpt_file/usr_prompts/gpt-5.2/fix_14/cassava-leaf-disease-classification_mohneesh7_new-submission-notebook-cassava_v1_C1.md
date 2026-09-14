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

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import warnings

warnings.filterwarnings("ignore")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick good defaults
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

print("TensorFlow:", tf.__version__)
print("Train CSV rows:", sum(1 for _ in open(TRAIN_CSV)) - 1)
print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
print("Test images:", len(os.listdir(TEST_IMG_DIR)))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

train_df["image_path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)

image_size = 224
batch_size = 32
num_classes = train_df["label"].nunique()

AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = (image_size, image_size)

val_frac = 0.1
n_total = len(train_df)
n_val = int(round(n_total * val_frac))
n_train = n_total - n_val

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
train_df_train = train_df.iloc[:n_train].reset_index(drop=True)
train_df_val = train_df.iloc[n_train:].reset_index(drop=True)

class_names = sorted(train_df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(class_names)}
idx_to_label = {i: int(c) for c, i in class_to_index.items()}
print("class_indices:", class_to_index)

IMG_SIZE_I32 = tf.constant([image_size, image_size], dtype=tf.int32)
CX = tf.constant((float(image_size) - 1.0) / 2.0, dtype=tf.float32)
CY = tf.constant((float(image_size) - 1.0) / 2.0, dtype=tf.float32)
PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
MAX_DX = tf.constant(0.05 * image_size, dtype=tf.float32)
MAX_DY = tf.constant(0.05 * image_size, dtype=tf.float32)


@tf.function
def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR)
    return img


@tf.function
def _translate(img, dx, dy):
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])
    transform = tf.expand_dims(transform, axis=0)
    img = tf.expand_dims(img, axis=0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=IMG_SIZE_I32,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


@tf.function
def _rotate(img, angle_rad):
    angle = tf.cast(angle_rad, tf.float32)
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)

    a0 = c
    a1 = -s
    a3 = s
    a4 = c
    a2 = CX - a0 * CX - a1 * CY
    a5 = CY - a3 * CX - a4 * CY

    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])
    transform = tf.expand_dims(transform, axis=0)

    img_b = tf.expand_dims(img, axis=0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_b,
        transforms=transform,
        output_shape=IMG_SIZE_I32,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


@tf.function
def _random_zoom(img, zoom_range=0.1):
    z = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range)
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    new_h = tf.cast(tf.round(h / z), tf.int32)
    new_w = tf.cast(tf.round(w / z), tf.int32)

    def zoom_in():
        offset_h = tf.random.uniform(
            [],
            0,
            tf.maximum(1, tf.shape(img)[0] - new_h + 1),
            dtype=tf.int32,
        )
        offset_w = tf.random.uniform(
            [],
            0,
            tf.maximum(1, tf.shape(img)[1] - new_w + 1),
            dtype=tf.int32,
        )
        cropped = tf.image.crop_to_bounding_box(img, offset_h, offset_w, new_h, new_w)
        return tf.image.resize(
            cropped, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR
        )

    def zoom_out():
        pad_h = tf.maximum(0, new_h - tf.shape(img)[0])
        pad_w = tf.maximum(0, new_w - tf.shape(img)[1])
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = tf.pad(
            img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT"
        )
        padded = tf.image.resize(
            padded, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR
        )
        return padded

    return tf.cond(z >= 1.0, zoom_in, zoom_out)


@tf.function
def _augment(img):
    angle = tf.random.uniform([], -10.0, 10.0) * PI_OVER_180
    img = _rotate(img, angle)

    dx = tf.random.uniform([], -MAX_DX, MAX_DX)
    dy = tf.random.uniform([], -MAX_DY, MAX_DY)
    img = _translate(img, dx, dy)

    img = _random_zoom(img, zoom_range=0.1)
    img = tf.image.random_flip_left_right(
        img
    )  # deterministic with global seed+determinism
    return img


@tf.function
def _preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


def _labels_to_indices(labels_str_arr):
    cat = pd.Categorical(labels_str_arr, categories=class_names, ordered=True)
    return cat.codes.astype(np.int32)


TRAIN_TFRECS = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(TRAIN_TFRECS) > 0, "No train tfrecords found"
assert len(TEST_TFRECS) > 0, "No test tfrecords found"


def _make_trainval_id_sets(df_train, df_val):
    train_ids = df_train["image_id"].astype(str).tolist()
    val_ids = df_val["image_id"].astype(str).tolist()
    return train_ids, val_ids


train_ids, val_ids = _make_trainval_id_sets(train_df_train, train_df_val)

train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_ids),
        values=tf.ones([len(train_ids)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)
val_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(val_ids),
        values=tf.ones([len(val_ids)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)


@tf.function
def _parse_train_example(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR)
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_id"]
    return img, label, image_id


def _make_ds_from_tfrecords(training: bool):
    ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.filter(
            lambda img, label, image_id: tf.equal(train_id_table.lookup(image_id), 1)
        )
    else:
        ds = ds.filter(
            lambda img, label, image_id: tf.equal(val_id_table.lookup(image_id), 1)
        )

    @tf.function
    def _aug_pre_onehot(img, label, image_id):
        if training:
            img = _augment(img)
        img = _preprocess(img)
        y = tf.one_hot(label, num_classes, dtype=tf.float32)
        return img, y

    if training:
        ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_aug_pre_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    if not training:
        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds_from_tfrecords(training=True)
val_ds = _make_ds_from_tfrecords(training=False)

base = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(
    x
)  # keep output float32
model = Model(inputs=base.input, outputs=out)

base.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

epochs = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3805530443.py in <cell line: 0>()
    289 
    290 epochs = 3
--> 291 history = model.fit(
    292     train_ds,
    293     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Shuffle::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNextAsOptional]] [Op:__inference_multi_step_on_iterator_53724]

## === cell 2
sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sub["image_id"].astype(str).tolist()


@tf.function
def _parse_test_example(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR)
    img = _preprocess(img)
    return ex["image_id"], img


def _make_test_ds_from_tfrecords():
    ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds_from_tfrecords()

all_ids = []
all_probs = []

for batch in test_ds:
    b_ids, b_imgs = batch
    b_probs = model(b_imgs, training=False)
    all_ids.append(b_ids.numpy())
    all_probs.append(b_probs.numpy())

all_ids = np.concatenate(all_ids).astype("U")
probs = np.concatenate(all_probs)

pred_idx = np.argmax(probs, axis=1)
pred_labels = np.array([idx_to_label[int(i)] for i in pred_idx], dtype=np.int64)

id_to_pred = dict(zip(all_ids.tolist(), pred_labels.tolist()))
ordered_preds = [id_to_pred[i] for i in test_image_ids]

submission = pd.DataFrame({"image_id": test_image_ids, "label": ordered_preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2241466352.py in <cell line: 0>()
     42 all_probs = []
     43 
---> 44 for batch in test_ds:
     45     b_ids, b_imgs = batch
     46     b_probs = model(b_imgs, training=False)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__parse_test_example_53764}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:
