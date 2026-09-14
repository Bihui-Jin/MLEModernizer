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

0.8875793291024479

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import json
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
label_map_path = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

assert os.path.exists(train_csv_path), train_csv_path
assert os.path.exists(sample_sub_path), sample_sub_path
assert os.path.exists(label_map_path), label_map_path



## === cell 2
with open(label_map_path) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 3
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 4
train_df_count = pd.read_csv(train_csv_path, usecols=["image_id"]).shape[0]
print(f"Number of train images (from train.csv): {train_df_count}")



## === cell 5
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16
PRE_TRAINED_MODEL = "../input/xceptionv8/Cassava_Best_Xception_Model_V08.hdf5"

NUM_CLASSES = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 6
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        RandomBrightnessContrast,
        CenterCrop,
        ShiftScaleRotate,
        ToFloat,
    )

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=0.0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255.0)])
    _AUG_AVAILABLE = True
except Exception as e:
    print(
        "albumentations not available; using no-op augmentations. Import error:",
        repr(e),
    )

    class _NoOpAug:
        def __call__(self, image):
            return {"image": image.astype(np.float32) / 255.0}

    AUGMENTATIONS_TRAIN = _NoOpAug()
    AUGMENTATIONS_TEST = _NoOpAug()
    _AUG_AVAILABLE = False




## === cell 7
@tf.function(reduce_retracing=True)
def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    side = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, side, side)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return img


def _make_ds(paths, labels=None, training=False, augmentations=None, cache_id=""):
    paths = tf.convert_to_tensor(paths)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        labels = tf.convert_to_tensor(labels)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(8192, int(paths.shape[0])),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    options = tf.data.Options()
    options.deterministic = False
    try:
        options.autotune.enabled = True
        options.autotune.cpu_budget = 0
    except Exception:
        pass
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_and_batch_fusion = True
    except Exception:
        pass
    ds = ds.with_options(options)

    if labels is None:
        ds = ds.map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    else:

        def _map_xy(p, y):
            return _decode_resize_normalize(p), y

        ds = ds.map(_map_xy, num_parallel_calls=AUTOTUNE)

    if cache_id:
        ds = ds.cache()

    if training and augmentations is not None and _AUG_AVAILABLE:

        def _aug_np(img_np):
            out = augmentations(image=img_np)["image"].astype(np.float32)
            return out

        def _aug_tf(img, y):
            img2 = tf.numpy_function(_aug_np, [img], Tout=tf.float32)
            img2.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
            return img2, y

        ds = ds.map(_aug_tf, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_df = pd.read_csv(train_csv_path)
assert set(["image_id", "label"]).issubset(train_df.columns)

NEED_FALLBACK_TRAINING = not os.path.exists(PRE_TRAINED_MODEL)
train_ds = None
val_ds = None
if NEED_FALLBACK_TRAINING:
    perm = np.random.RandomState(SEED).permutation(len(train_df))
    val_size = int(0.2 * len(train_df))
    val_idx = perm[:val_size]
    trn_idx = perm[val_size:]

    trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    trn_paths = np.array(
        [os.path.join(TRAIN_DIR, f) for f in trn_df["image_id"].values], dtype=object
    )
    val_paths = np.array(
        [os.path.join(TRAIN_DIR, f) for f in val_df["image_id"].values], dtype=object
    )

    trn_labels = trn_df["label"].values.astype(np.int64)
    val_labels = val_df["label"].values.astype(np.int64)

    train_ds = _make_ds(
        trn_paths,
        trn_labels,
        training=True,
        augmentations=AUGMENTATIONS_TRAIN,
        cache_id="train",
    )
    val_ds = _make_ds(
        val_paths,
        val_labels,
        training=False,
        augmentations=None,  # decode already yields float32 [0,1], ToFloat would be redundant
        cache_id="val",
    )



## === cell 8
test_files = tf.io.gfile.glob(os.path.join(TEST_DIR, "*.jpg"))
test_files = sorted(test_files)
test_paths = np.asarray(test_files, dtype=object)
test_filenames = [os.path.basename(p) for p in test_files]

test_samples = len(test_filenames)
print("Test samples:", test_samples)

test_ds = _make_ds(
    test_paths,
    labels=None,
    training=False,
    augmentations=None,  # decode already yields float32 [0,1], avoid extra map/overhead
    cache_id="",
)

test_df = pd.DataFrame({"image_id": test_filenames})



## === cell 9
from tensorflow.keras.models import load_model


def build_fallback_model():
    base = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    x_in = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.applications.xception.preprocess_input(x_in * 255.0)
    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(x_in, x)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL, compile=False)
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception:
        pass
    print("Loaded pre-trained model:", PRE_TRAINED_MODEL)
else:
    raise FileNotFoundError(
        f"Pre-trained model not found at {PRE_TRAINED_MODEL}. "
        "Fallback training is disabled to ensure the notebook finishes within the 600-second limit."
    )

model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/750403693.py in <cell line: 0>()
     37     print("Loaded pre-trained model:", PRE_TRAINED_MODEL)
     38 else:
---> 39     raise FileNotFoundError(
     40         f"Pre-trained model not found at {PRE_TRAINED_MODEL}. "
     41         "Fallback training is disabled to ensure the notebook finishes within the 600-second limit."

FileNotFoundError: Pre-trained model not found at ../input/xceptionv8/Cassava_Best_Xception_Model_V08.hdf5. Fallback training is disabled to ensure the notebook finishes within the 600-second limit.

## === cell 10
steps = int(np.ceil(test_samples / batch_size))
probs = model.predict(
    test_ds,
    verbose=1,
    steps=steps,
)
preds = np.argmax(probs, axis=1).astype(int)

sub = pd.DataFrame({"image_id": test_df["image_id"].values, "label": preds})

sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")

if sub["label"].isna().any():
    fill_label = int(train_df["label"].mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head(3))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4188958634.py in <cell line: 0>()
      2 # Correctness: identical predictions; just an execution hint.
      3 steps = int(np.ceil(test_samples / batch_size))
----> 4 probs = model.predict(
      5     test_ds,
      6     verbose=1,

NameError: name 'model' is not defined

## === cell 11
submission = pd.read_csv("submission.csv")
assert submission.shape[0] == pd.read_csv(sample_sub_path).shape[0]
assert list(submission.columns) == ["image_id", "label"]
print(submission.head(5))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1420496673.py in <cell line: 0>()
----> 1 submission = pd.read_csv("submission.csv")
      2 assert submission.shape[0] == pd.read_csv(sample_sub_path).shape[0]
      3 assert list(submission.columns) == ["image_id", "label"]
      4 print(submission.head(5))

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
