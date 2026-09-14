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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.2095106186518933

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, math, re
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

print("TF:", tf.__version__)

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

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

print(train.shape, sub.shape)



## === cell 2
AUTO = tf.data.AUTOTUNE



## === cell 3
pass



## === cell 4
import pathlib



## === cell 5
train_dir = os.path.join(path, "train_images")
test_dir = os.path.join(path, "test_images")

train_images = train["image"].to_numpy()
test_images = sub["image"].to_numpy()

train_paths = [os.path.join(train_dir, fn) for fn in train_images]
test_paths = [os.path.join(test_dir, fn) for fn in test_images]

assert len(train_paths) == len(train)
assert len(test_paths) == len(sub)



## === cell 6
all_classes = sorted({c for s in train["labels"].astype(str).values for c in s.split()})
print("Classes:", all_classes)



## === cell 7
labels_split = train["labels"].astype(str).str.get_dummies(sep=" ")
labels_split = labels_split.reindex(columns=all_classes, fill_value=0).astype(
    np.float32
)

new_train = labels_split.copy()
new_train.insert(0, "image", train["image"].values)



## === cell 8
new_train



## === cell 9
IMAGE_SIZE = (512, 512)
_IMAGE_SIZE_T = tf.constant([IMAGE_SIZE[0], IMAGE_SIZE[1]], dtype=tf.int32)


@tf.function(reduce_retracing=True, input_signature=[tf.TensorSpec([], tf.string)])
def _decode_image_center_square_no_label(filename):
    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]
    side = tf.minimum(h, w)

    cropped = tf.image.resize_with_crop_or_pad(image, side, side)

    cropped = tf.image.resize(
        cropped, _IMAGE_SIZE_T, method="bilinear", antialias=False
    )
    return cropped


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([None], tf.float32),
    ],
)
def _decode_image_center_square_with_label(filename, label):
    cropped = _decode_image_center_square_no_label(filename)
    return cropped, label


def decode_image_test(filename, image_size=(512, 512)):
    return _decode_image_center_square_no_label(filename)


def decode_image_train(filename, label, image_size=(512, 512)):
    return _decode_image_center_square_with_label(filename, label)




## === cell 10
test_paths[:5], len(test_paths)



## === cell 11
_has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
BATCH_SIZE = 32 if _has_gpu else 16
print("GPU:", _has_gpu, "BATCH_SIZE:", BATCH_SIZE)




## === cell 12
def _configure_dataset(ds: tf.data.Dataset, deterministic: bool) -> tf.data.Dataset:
    opts = tf.data.Options()
    opts.deterministic = deterministic
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.threading.private_threadpool_size = 16
        opts.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    try:
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return ds.with_options(opts)


def map_batch_prefetch(
    ds,
    map_fn,
    batch_size,
    deterministic=True,
    cache=None,
    shuffle=None,
    seed=None,
    repeat=False,
    drop_remainder=False,
):
    if shuffle is not None:
        ds = ds.shuffle(shuffle, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(map_fn, num_parallel_calls=AUTO, deterministic=deterministic)

    if cache:
        ds = ds.cache(cache)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)
    ds = _configure_dataset(ds, deterministic=deterministic)
    return ds




## === cell 13
test_paths_tf = tf.constant(np.asarray(test_paths, dtype=np.str_))
test_dataset = tf.data.Dataset.from_tensor_slices(test_paths_tf)
test_dataset = map_batch_prefetch(
    test_dataset,
    lambda f: decode_image_test(f, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=None,
    repeat=False,
    drop_remainder=False,
)
test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())



## === cell 14
from tensorflow import keras

NUM_CLASSES = len(all_classes)

idx = np.arange(len(new_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_paths_np = np.asarray(train_paths, dtype=np.str_)
x_train = train_paths_np[tr_idx]
y_train = new_train.iloc[tr_idx][all_classes].to_numpy(dtype=np.float32, copy=False)

x_val = train_paths_np[va_idx]
y_val = new_train.iloc[va_idx][all_classes].to_numpy(dtype=np.float32, copy=False)

x_train_tf = tf.constant(x_train)
x_val_tf = tf.constant(x_val)

train_ds = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train))
train_ds = map_batch_prefetch(
    train_ds,
    lambda f, y: decode_image_train(f, y, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=False,
    cache=None,
    shuffle=2048,
    seed=SEED,
    repeat=True,
    drop_remainder=True,  # fixed shapes for steps_per_epoch; avoids retracing overhead
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

val_ds = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val))
val_ds = map_batch_prefetch(
    val_ds,
    lambda f, y: decode_image_train(f, y, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=True,  # in-memory cache
    shuffle=None,
    seed=None,
    repeat=False,
    drop_remainder=False,
)
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

model.summary()

steps_per_epoch = int(math.ceil(len(x_train) / BATCH_SIZE))
validation_steps = int(math.ceil(len(x_val) / BATCH_SIZE))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2084996392.py in <cell line: 0>()
     39 # This preserves exact validation semantics (same data, same preprocessing).
     40 val_ds = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val))
---> 41 val_ds = map_batch_prefetch(
     42     val_ds,
     43     lambda f, y: decode_image_train(f, y, image_size=IMAGE_SIZE),

/tmp/ipykernel_11/3189824894.py in map_batch_prefetch(ds, map_fn, batch_size, deterministic, cache, shuffle, seed, repeat, drop_remainder)
     34 
     35     if cache:
---> 36         ds = ds.cache(cache)
     37 
     38     if repeat:

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

TypeError: Cannot convert True to EagerTensor of dtype string

## === cell 15
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4107178584.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(
      3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS,

NameError: name 'model' is not defined

## === cell 16
test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
probs = model.predict(test_dataset, steps=test_steps, verbose=1)
print("probs:", probs.shape, probs.dtype)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1722729026.py in <cell line: 0>()
      1 test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
----> 2 probs = model.predict(test_dataset, steps=test_steps, verbose=1)
      3 print("probs:", probs.shape, probs.dtype)
      4 

NameError: name 'model' is not defined

## === cell 17
class_to_idx = {c: i for i, c in enumerate(all_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

default_thr = 0.30
threshold_by_class = {
    "scab": 0.20,
    "frog_eye_leaf_spot": 0.30,
    "complex": 0.15,
    "rust": 0.30,
    "powdery_mildew": 0.35,
    "healthy": 0.50,  # only chosen when nothing else triggers
}
thr = np.array(
    [threshold_by_class.get(c, default_thr) for c in all_classes], dtype=np.float32
)

probs_np = np.asarray(probs, dtype=np.float32)

healthy_idx = class_to_idx.get("healthy", None)
non_healthy_mask = np.ones(NUM_CLASSES, dtype=bool)
if healthy_idx is not None:
    non_healthy_mask[healthy_idx] = False

hits_mat = probs_np[:, non_healthy_mask] > thr[non_healthy_mask]
hit_counts = hits_mat.sum(axis=1)
is_complex = hit_counts >= 3
is_healthy = hit_counts == 0

non_healthy_classes = np.array([c for c in all_classes if c != "healthy"], dtype=object)

pred_string = []
for i in range(probs_np.shape[0]):
    if is_complex[i]:
        pred_string.append("complex")
    elif is_healthy[i]:
        pred_string.append("healthy")
    else:
        pred_string.append(" ".join(non_healthy_classes[hits_mat[i]].tolist()))

submission = sub.copy()
submission["labels"] = pred_string
submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2184079257.py in <cell line: 0>()
     15 )
     16 
---> 17 probs_np = np.asarray(probs, dtype=np.float32)
     18 
     19 healthy_idx = class_to_idx.get("healthy", None)

NameError: name 'probs' is not defined
