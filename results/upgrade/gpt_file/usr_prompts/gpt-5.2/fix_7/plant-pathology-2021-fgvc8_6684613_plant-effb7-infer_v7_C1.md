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

0.78016620498615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

os.environ.setdefault("PYTHONHASHSEED", "42")
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as _:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    TPU if available, otherwise default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not available, using default strategy. Reason:", repr(e))
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original choice: 600

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num classes (unique label strings):", n_labels)
print("First 10 classes:", class_name[:10])



## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = 32

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image"]].copy()
assert test_df["image"].nunique() == len(
    test_df
), "Duplicate images in sample submission?"

print("Test rows:", len(test_df))
print("Test dir exists:", os.path.isdir(test_dir))



## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input

test_paths = tf.constant(
    [os.path.join(test_dir, fn) for fn in test_df["image"].tolist()]
)

AUTOTUNE = tf.data.AUTOTUNE

_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.deterministic = True


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _apply_tta(img, tta_index):
    t = tf.math.mod(tta_index, 4)
    img = tf.cond(t >= 2, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(
        tf.logical_or(t == 1, t == 3), lambda: tf.image.flip_up_down(img), lambda: img
    )
    return img


def make_base_test_ds_cached():
    ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(_DATA_OPTIONS)
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    cache_path = os.path.join("/kaggle/working", f"test_cache_{im_size}.tfdata")
    ds = ds.cache(cache_path)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function
def _tta_map(img, tta_index):
    return _apply_tta(img, tta_index)


def make_test_ds_from_base(base_ds, tta_index):
    tta_index = tf.constant(tta_index, dtype=tf.int32)
    ds = base_ds.with_options(_DATA_OPTIONS)
    ds = ds.map(
        lambda x: _tta_map(x, tta_index),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_tta_test_ds(base_ds, tta):
    ds_img = base_ds.repeat(tta)
    n = tf.shape(test_paths)[0]
    ds_tta = tf.data.Dataset.range(n * tta).map(
        lambda k: tf.cast(tf.math.mod(k, tta), tf.int32),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = tf.data.Dataset.zip((ds_img, ds_tta)).with_options(_DATA_OPTIONS)
    ds = ds.map(
        lambda x, ti: _apply_tta(x, ti),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalMaxPooling2D(),
            Dense(n_labels, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights not found:", weights_path)
    print(
        "Proceeding with randomly initialized model to generate a valid submission.csv."
    )

try:
    model.run_eagerly = False
except Exception:
    pass



## === cell 7
TTA = 6

base_ds = make_base_test_ds_cached()

tta_ds = make_tta_test_ds(base_ds, TTA)
p_all = model.predict(tta_ds, verbose=1)

n_test = len(test_df)
expected = n_test * TTA
if p_all.shape[0] != expected:
    raise RuntimeError(
        f"TTA prediction length mismatch: got {p_all.shape[0]} preds, expected {expected}"
    )

pred = p_all.reshape(n_test, TTA, -1).mean(axis=1)
argpred = np.argmax(pred, axis=1)

if len(argpred) != len(test_df):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(argpred)} preds, expected {len(test_df)}"
    )

test_df["labels"] = argpred.astype(np.int32)
test_df["labels"] = np.take(
    np.array(class_name, dtype=object), test_df["labels"].values
)

submission = test_df[["image", "labels"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2723085180.py in <cell line: 0>()
      5 # CHANGE (timeout fix, correctness-preserving):
      6 # Predict all TTA-augmented images in a single pass, then reshape and average.
----> 7 tta_ds = make_tta_test_ds(base_ds, TTA)
      8 p_all = model.predict(tta_ds, verbose=1)
      9 

/tmp/ipykernel_11/3176462148.py in make_tta_test_ds(base_ds, tta)
     75     ds_img = base_ds.repeat(tta)
     76     n = tf.shape(test_paths)[0]
---> 77     ds_tta = tf.data.Dataset.range(n * tta).map(
     78         lambda k: tf.cast(tf.math.mod(k, tta), tf.int32),
     79         num_parallel_calls=AUTOTUNE,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in range(*args, **kwargs)
   1020     # pylint: disable=g-import-not-at-top,protected-access
   1021     from tensorflow.python.data.ops import range_op
-> 1022     return range_op._range(*args, **kwargs)
   1023     # pylint: enable=g-import-not-at-top,protected-access
   1024 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/range_op.py in _range(*args, **kwargs)
     23 
     24 def _range(*args, **kwargs):  # pylint: disable=unused-private-name
---> 25   return _RangeDataset(*args, **kwargs)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/range_op.py in __init__(self, *args, **kwargs)
     31   def __init__(self, *args, **kwargs):
     32     """See `Dataset.range()` for details."""
---> 33     self._parse_args(*args, **kwargs)
     34     self._structure = tensor_spec.TensorSpec([], self._output_type)
     35     variant_tensor = gen_dataset_ops.range_dataset(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/range_op.py in _parse_args(self, *args, **kwargs)
     44     if len(args) == 1:
     45       self._start = self._build_tensor(0, "start")
---> 46       self._stop = self._build_tensor(args[0], "stop")
     47       self._step = self._build_tensor(1, "step")
     48     elif len(args) == 2:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/range_op.py in _build_tensor(self, int64_value, name)
     64 
     65   def _build_tensor(self, int64_value, name):
---> 66     return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)
     67 
     68   @property

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
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in __tf_tensor__(self, dtype, name)
    625                 name=name))
    626       return graph.capture(self, name=name)
--> 627     return super().__tf_tensor__(dtype, name)
    628 
    629   def _capture_as_const(self, name) -> Optional[tensor_lib.Tensor]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __tf_tensor__(self, dtype, name)
    759       ) -> "Tensor":
    760     if dtype is not None and not dtype.is_compatible_with(self.dtype):
--> 761       raise ValueError(
    762           _add_error_prefix(
    763               f"Tensor conversion requested dtype {dtype.name} "

ValueError: stop: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=22362>
