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

0.8570565125415534

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing dir: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing dir: {TEST_TFREC_DIR}"



## === cell 2
with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)

print(json.dumps(map_classes, indent=2))
label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected submission columns"
assert set(train_df.columns) == {"image_id", "label"}, "Unexpected train.csv columns"

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))



## === cell 4
IMG_HEIGHT = 400
IMG_WIDTH = 400
BATCH_SIZE = 32

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def load_single_image_from_dir(image_dir, image_id):
    image_path = os.path.join(image_dir, image_id)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
        arr = np.asarray(im, dtype=np.float32)
    return arr




## === cell 5
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        RandomBrightness,
        RandomContrast,
        ToFloat,
        ShiftScaleRotate,
        CenterCrop,
    )

    _ALBU_OK = True
except Exception as e:
    print(
        "Albumentations not available or failed to import; falling back to no-aug. Error:",
        repr(e),
    )
    _ALBU_OK = False

if _ALBU_OK:
    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255.0)])
else:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None




## === cell 6
def _np_albu_apply(image_f32_0_255, do_apply, is_train):
    img = np.asarray(image_f32_0_255, dtype=np.float32)
    if is_train:
        if AUGMENTATIONS_TRAIN is not None and bool(do_apply):
            out = AUGMENTATIONS_TRAIN(image=img)["image"]
        else:
            out = img / 255.0
    else:
        if AUGMENTATIONS_TEST is not None:
            out = AUGMENTATIONS_TEST(image=img)["image"]
        else:
            out = img / 255.0
    return np.asarray(out, dtype=np.float32)


def _tf_apply_train_aug(img_f32_0_255, label):
    if AUGMENTATIONS_TRAIN is None:
        return (img_f32_0_255 / 255.0), label

    do_apply = tf.random.uniform((), seed=SEED) > 0.5
    out = tf.numpy_function(
        func=_np_albu_apply,
        inp=[img_f32_0_255, do_apply, True],
        Tout=tf.float32,
    )
    out.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return out, label


def _tf_apply_eval_aug(img_f32_0_255, label=None):
    if AUGMENTATIONS_TEST is None:
        out = img_f32_0_255 / 255.0
    else:
        out = tf.numpy_function(
            func=_np_albu_apply,
            inp=[img_f32_0_255, False, False],
            Tout=tf.float32,
        )
        out.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    if label is None:
        return out
    return out, label


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _decode_and_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32)  # 0..255 float32
    return img


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = _decode_and_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int64)
    image_id = ex["image_id"]
    return image_id, img, label


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = _decode_and_resize_from_bytes(ex["image"])
    image_id = ex["image_id"]
    return image_id, img


def _list_tfrec_files(tfrec_dir, prefix):
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecords found in {tfrec_dir} with prefix {prefix}"
        )
    return files


def make_train_ds(image_ids, labels, batch_size):
    id_to_label = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(image_ids, dtype=tf.string),
            values=tf.constant(labels, dtype=tf.int64),
        ),
        default_value=tf.constant(-1, dtype=tf.int64),
    )

    files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(
        lambda iid, img, y: tf.not_equal(
            id_to_label.lookup(iid), tf.constant(-1, tf.int64)
        )
    )
    ds = ds.map(
        lambda iid, img, y: (img, id_to_label.lookup(iid)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.shuffle(
        buffer_size=len(image_ids), seed=SEED, reshuffle_each_iteration=True
    )

    ds = ds.map(_tf_apply_train_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(opts)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(image_ids, labels, batch_size):
    id_to_label = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(image_ids, dtype=tf.string),
            values=tf.constant(labels, dtype=tf.int64),
        ),
        default_value=tf.constant(-1, dtype=tf.int64),
    )

    files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(
        lambda iid, img, y: tf.not_equal(
            id_to_label.lookup(iid), tf.constant(-1, tf.int64)
        )
    )
    ds = ds.map(
        lambda iid, img, y: (img, id_to_label.lookup(iid)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.map(
        lambda x, y: _tf_apply_eval_aug(x, y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(opts)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(image_ids, batch_size):
    keys = tf.constant(image_ids, dtype=tf.string)
    vals = tf.ones_like(keys, dtype=tf.int64)
    in_test = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
        default_value=tf.constant(0, dtype=tf.int64),
    )

    files = _list_tfrec_files(TEST_TFREC_DIR, "ld_test")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(
        lambda iid, img: tf.equal(in_test.lookup(iid), tf.constant(1, tf.int64))
    )
    ds = ds.map(lambda iid, img: img, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.map(_tf_apply_eval_aug, num_parallel_calls=AUTOTUNE, deterministic=True)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(opts)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 7
from tensorflow.keras import layers, models

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    pooling="avg",
)
base_model.trainable = False  # phase 1: fast/stable

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.xception.preprocess_input(inputs * 255.0)
x = base_model(x, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 8
def stratified_split_indices(y, test_size=0.1, seed=SEED):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * test_size)))
        val_idx.append(idx_c[:n_val])
        train_idx.append(idx_c[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


train_idx, val_idx = stratified_split_indices(
    train_df["label"].values, test_size=0.1, seed=SEED
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = make_train_ds(
    tr_df["image_id"].values.astype(str),
    tr_df["label"].values.astype(np.int64),
    BATCH_SIZE,
)
val_ds = make_val_ds(
    va_df["image_id"].values.astype(str),
    va_df["label"].values.astype(np.int64),
    BATCH_SIZE,
)



## === cell 9
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

FINE_TUNE_EPOCHS = 1
UNFREEZE_LAST_N_LAYERS = 30

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, verbose=1
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3142440873.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 
      4 FINE_TUNE_EPOCHS = 1
      5 UNFREEZE_LAST_N_LAYERS = 30

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
Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Shuffle::Map::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_8280]

## === cell 10
test_ds = make_test_ds(sample_sub["image_id"].values.astype(str), BATCH_SIZE)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": preds})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert os.path.exists(out_path) and out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4053808396.py in <cell line: 0>()
      1 test_ds = make_test_ds(sample_sub["image_id"].values.astype(str), BATCH_SIZE)
      2 
----> 3 probs = model.predict(test_ds, verbose=1)
      4 preds = np.argmax(probs, axis=1).astype(int)
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:23 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Map::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:
