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

0.8667271078875793

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

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

random.seed(42)
np.random.seed(42)



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")



## === cell 2
import matplotlib.pyplot as plt
from PIL import Image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS



## === cell 3
with open(MAP_JSON) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 4
train_df_tmp = pd.read_csv(TRAIN_CSV, usecols=["image_id", "label"])
print(f"Number of train images (from train.csv): {len(train_df_tmp)}")
del train_df_tmp



## === cell 5
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv4/Cassava_Best_Xception_Model_V03.hdf5"




## === cell 6
def _resolve_image_dir(dir_path: str) -> str:
    """
    Some Kaggle datasets have nested directories like test_images/test_images.
    Return the deepest directory that actually contains .jpg files.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    entries = []
    try:
        entries = os.listdir(dir_path)
    except Exception:
        pass

    if any(e.lower().endswith(".jpg") for e in entries):
        return dir_path

    subdirs = [
        os.path.join(dir_path, e)
        for e in entries
        if os.path.isdir(os.path.join(dir_path, e))
    ]
    if len(subdirs) == 1:
        return _resolve_image_dir(subdirs[0])

    for sd in subdirs:
        try:
            sub_entries = os.listdir(sd)
        except Exception:
            continue
        if any(e.lower().endswith(".jpg") for e in sub_entries):
            return sd

    raise FileNotFoundError(f"Could not find .jpg files under: {dir_path}")


TRAIN_DIR = _resolve_image_dir(TRAIN_DIR)
TEST_DIR = _resolve_image_dir(TEST_DIR)

print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR:", TEST_DIR)



## === cell 7
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTRAOP_THREADS", "2"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTEROP_THREADS", "2"))
    )
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE


@tf.function(jit_compile=True)
def _load_one_tf(image_path, label):
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img, label


@tf.function(jit_compile=True)
def _load_one_tf_nolabel(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def make_dataset_from_df(
    df, images_dir, batch_size, training: bool, cache_name: str = None
):
    """
    Speed fixes while preserving semantics:
    - Avoid in-memory caching of full decoded 400x400 float tensors (val/test .cache()) which can exceed RAM
      and cause severe slowdown/timeout due to paging/GC. The dataset remains identical; it just doesn't
      store the entire tensor set in memory.
    - Increase input pipeline throughput with explicit parallel reads and non-blocking prefetch.
    - Keep deterministic=True and same shuffle seed/behavior to preserve stability.
    """
    paths = (images_dir.rstrip(os.sep) + os.sep + df["image_id"].values).astype(str)
    image_paths = tf.convert_to_tensor(paths, dtype=tf.string)

    has_labels = "label" in df.columns
    if not has_labels:
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(
            _load_one_tf_nolabel, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        labels = tf.convert_to_tensor(df["label"].values, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(_load_one_tf, num_parallel_calls=AUTOTUNE, deterministic=True)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_name:
        ds = ds.cache(cache_name)

    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
train_df = pd.read_csv(TRAIN_CSV)
print(train_df.shape)
print(train_df.head())

idx = np.arange(len(train_df))
rng = np.random.RandomState(42)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print("Train/Val sizes:", tr_df.shape, va_df.shape)

train_ds = make_dataset_from_df(
    tr_df,
    TRAIN_DIR,
    batch_size=batch_size,
    training=True,
    cache_name=None,
)

val_ds = make_dataset_from_df(
    va_df,
    TRAIN_DIR,
    batch_size=batch_size,
    training=False,
    cache_name=None,
)



## === cell 9
if os.path.exists(PRE_TRAINED_MODEL):
    model = tf.keras.models.load_model(PRE_TRAINED_MODEL)
    print("Loaded pretrained model:", PRE_TRAINED_MODEL)
else:
    base = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs
    x = base(x, training=False)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    EPOCHS = 2
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/844993183.py in <cell line: 0>()
     24 
     25     EPOCHS = 2
---> 26     model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
     27 
     28 model.summary()

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

## === cell 10
test_paths = tf.io.gfile.glob(os.path.join(TEST_DIR, "*.jpg"))
test_filenames = sorted([os.path.basename(p) for p in test_paths])

test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
print("Test samples:", test_samples)

test_ds = make_dataset_from_df(
    test_df,
    TEST_DIR,
    batch_size=batch_size,
    training=False,
    cache_name=None,
)

pred_steps = int(np.ceil(test_samples / batch_size))
pred_proba = model.predict(test_ds, verbose=1, steps=pred_steps)
pred_labels = np.argmax(pred_proba, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head(3))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_11/3688077509.py in <cell line: 0>()
     19 # - Provide explicit steps to avoid any extra end-of-dataset bookkeeping; predictions are identical.
     20 pred_steps = int(np.ceil(test_samples / batch_size))
---> 21 pred_proba = model.predict(test_ds, verbose=1, steps=pred_steps)
     22 pred_labels = np.argmax(pred_proba, axis=1).astype(int)
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value

## === cell 11
chk = pd.read_csv("submission.csv")
print(chk.shape)
print(chk.columns.tolist())
print(chk.head())
assert chk.shape[0] == pd.read_csv(SAMPLE_SUB_CSV).shape[0]
assert chk.columns.tolist() == ["image_id", "label"]
assert chk["label"].between(0, NUM_CLASSES - 1).all()
print("Submission sanity checks passed.")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1564158809.py in <cell line: 0>()
----> 1 chk = pd.read_csv("submission.csv")
      2 print(chk.shape)
      3 print(chk.columns.tolist())
      4 print(chk.head())
      5 assert chk.shape[0] == pd.read_csv(SAMPLE_SUB_CSV).shape[0]

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
