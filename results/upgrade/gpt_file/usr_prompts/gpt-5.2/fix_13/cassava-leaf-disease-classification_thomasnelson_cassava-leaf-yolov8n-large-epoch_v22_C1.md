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

3.13

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

0.8851616802659413

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SUB_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Train dir exists:", TRAIN_IMG_DIR)
print("Test dir exists:", TEST_IMG_DIR)
print("GPUs:", tf.config.list_physical_devices("GPU"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

required_train_cols = {"image_id", "label"}
required_sub_cols = {"image_id", "label"}
assert required_train_cols.issubset(
    train_df.columns
), f"train.csv columns: {train_df.columns}"
assert required_sub_cols.issubset(
    sample_df.columns
), f"sample_submission.csv columns: {sample_df.columns}"

train_df["label"] = train_df["label"].astype(int)
num_classes = int(train_df["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

train_df["filepath"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
).astype(str)

print("Train rows:", len(train_df))
print(train_df.head())




## === cell 2
def stratified_split(df, label_col="label", val_frac=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby(label_col).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_mask[idx[:n_val]] = True
    val_df = df.loc[val_mask].reset_index(drop=True)
    trn_df = df.loc[~val_mask].reset_index(drop=True)
    return trn_df, val_df


trn_df, val_df = stratified_split(train_df, val_frac=0.1, seed=SEED)
print("Train split:", len(trn_df), "Val split:", len(val_df))
print(
    "Train label distribution:\n",
    trn_df["label"].value_counts(normalize=True).sort_index(),
)
print(
    "Val label distribution:\n",
    val_df["label"].value_counts(normalize=True).sort_index(),
)




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_decoded.cache")
VAL_CACHE = os.path.join(CACHE_DIR, "val_decoded.cache")
TEST_CACHE = os.path.join(CACHE_DIR, "test_decoded.cache")

_decode_resize_spec = [tf.TensorSpec(shape=(), dtype=tf.string)]


@tf.function(input_signature=_decode_resize_spec)
def _decode_resize_only(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function(
    input_signature=[tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)]
)
def _augment_only(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    return img


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def _decode_with_label(path, y):
    return _decode_resize_only(path), tf.cast(y, tf.int32)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def _augment_with_label(x, y):
    return _augment_only(x), y


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_and_batch_fusion = True
    except Exception:
        pass
    try:
        options.experimental_slack = True
    except Exception:
        pass
    try:
        options.threading.private_threadpool_size = 0
        options.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    return options


def _maybe_prefetch_to_device(ds):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            return ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            return ds.prefetch(AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


USE_FILE_CACHE = (
    False  # deterministic; does not change outputs, only performance characteristics
)


def make_decoded_ds(df, cache_path):
    paths = df["filepath"].astype(str).to_numpy()
    labels = df["label"].astype(np.int32).to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_dataset_options())
    ds = ds.map(_decode_with_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(cache_path if USE_FILE_CACHE else None)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def make_train_ds_from_decoded(decoded_ds, n_items):
    ds = decoded_ds.shuffle(
        buffer_size=min(int(n_items), 4096), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_augment_with_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_eval_ds_from_decoded(decoded_ds):
    ds = decoded_ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = _maybe_prefetch_to_device(ds)
    return ds


train_decoded = make_decoded_ds(trn_df, TRAIN_CACHE)
val_decoded = make_decoded_ds(val_df, VAL_CACHE)

train_ds = make_train_ds_from_decoded(train_decoded, n_items=len(trn_df))
val_ds = make_eval_ds_from_decoded(val_decoded)

train_steps = len(trn_df) // BATCH_SIZE  # drop_remainder=True matches original
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

print("train_steps:", train_steps, "val_steps:", val_steps)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1877099220.py in <cell line: 0>()
    123 
    124 
--> 125 train_decoded = make_decoded_ds(trn_df, TRAIN_CACHE)
    126 val_decoded = make_decoded_ds(val_df, VAL_CACHE)
    127 

/tmp/ipykernel_11/1877099220.py in make_decoded_ds(df, cache_path)
    102     ds = ds.map(_decode_with_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    103     # cache AFTER decode so repeated epochs don't re-read/decode JPEGs
--> 104     ds = ds.cache(cache_path if USE_FILE_CACHE else None)
    105     ds = ds.apply(tf.data.experimental.ignore_errors())
    106     return ds

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in cache(self, filename, name)
   1566     # pylint: disable=g-import-not-at-top,protected-access
   1567     from tensorflow.python.data.ops import cache_op
-> 1568     return cache_op._cache(self, filename, name)
   1569     # pylint: enable=g-import-not-at-top,protected-access
   1570 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in _cache(input_dataset, filename, name)
     24 
     25 def _cache(input_dataset, filename, name):  # pylint: disable=unused-private-name
---> 26   return CacheDataset(input_dataset, filename, name)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in __init__(self, input_dataset, filename, name)
     33     """See `Dataset.cache()` for details."""
     34     self._input_dataset = input_dataset
---> 35     self._filename = ops.convert_to_tensor(
     36         filename, dtype=dtypes.string, name="filename")
     37     self._name = name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.

## === cell 4
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # keep stable and within time budget

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

model.summary()




## === cell 5
EPOCHS = 6  # keep as-is (core logic / training schedule)

history = model.fit(
    train_ds.repeat(),
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=2,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

history_ft = model.fit(
    train_ds.repeat(),
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3039260727.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds.repeat(),
      5     validation_data=val_ds,
      6     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 6
test_paths = (
    TEST_IMG_DIR.rstrip("/") + "/" + sample_df["image_id"].astype(str)
).to_numpy()

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(_dataset_options())
test_ds = test_ds.map(
    _decode_resize_only, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.cache(TEST_CACHE if USE_FILE_CACHE else None)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = _maybe_prefetch_to_device(test_ds)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

sub = sample_df.copy()
sub["label"] = preds
sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH)
print(sub.head())

with open(SUB_PATH, "r", encoding="utf-8") as f:
    for _ in range(6):
        print(f.readline().rstrip("\n"))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1153257898.py in <cell line: 0>()
     10 # CHANGE (timeout fix, correctness-preserving):
     11 # Same rationale as train/val: avoid slow disk cache writes; in-memory cache (or no cache) is faster.
---> 12 test_ds = test_ds.cache(TEST_CACHE if USE_FILE_CACHE else None)
     13 test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
     14 test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in cache(self, filename, name)
   1566     # pylint: disable=g-import-not-at-top,protected-access
   1567     from tensorflow.python.data.ops import cache_op
-> 1568     return cache_op._cache(self, filename, name)
   1569     # pylint: enable=g-import-not-at-top,protected-access
   1570 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in _cache(input_dataset, filename, name)
     24 
     25 def _cache(input_dataset, filename, name):  # pylint: disable=unused-private-name
---> 26   return CacheDataset(input_dataset, filename, name)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in __init__(self, input_dataset, filename, name)
     33     """See `Dataset.cache()` for details."""
     34     self._input_dataset = input_dataset
---> 35     self._filename = ops.convert_to_tensor(
     36         filename, dtype=dtypes.string, name="filename")
     37     self._name = name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.
