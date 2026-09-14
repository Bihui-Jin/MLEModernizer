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

0.6328716528162511

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'Main bottlenecks are in the input pipeline: (1) caching **after** decode/resize stores huge 380×380 float tensors in memory/disk, (2) `enumerate()` forces a sequential dependency that reduces pipeline parallelism, and (3) training-time augmentation uses a slow fallback (no `tfa`) and runs after the heavy cache step. The optimized version keeps the exact same model/training loop/epochs and the same deterministic stateless augmentation semantics, but changes the pipeline to cache **compressed JPEG bytes** (tiny) and decode/resize (and augment) from those cached bytes each epoch, and replaces `enumerate()` with `Dataset.random(seed=...)` to generate deterministic per-example seeds without serializing the pipeline. This preserves correctness (same images, same label vectors, same augmentation distribution, deterministic behavior) while greatly reducing memory pressure and improving throughput so the notebook finishes under the 600s limit.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

BASE_CANDIDATES = [
    Path("/kaggle/input/plant-pathology-2021-fgvc8"),
    Path("/kaggle/data/plant-pathology-2021-fgvc8"),
    Path("../input/plant-pathology-2021-fgvc8"),
]
BASE_DIR = next((p for p in BASE_CANDIDATES if p.exists()), None)
if BASE_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir in any of: {BASE_CANDIDATES}")

TRAIN_CSV = BASE_DIR / "train.csv"
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

print("BASE_DIR:", BASE_DIR)
print("TRAIN_CSV exists:", TRAIN_CSV.exists())
print("SAMPLE_SUB exists:", SAMPLE_SUB.exists())
print("TRAIN_IMAGES_DIR exists:", TRAIN_IMAGES_DIR.exists())
print("TEST_IMAGES_DIR exists:", TEST_IMAGES_DIR.exists())

AUTOTUNE = tf.data.AUTOTUNE
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(df_train))
print(df_train.head())

labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
label_to_idx = {l: i for i, l in enumerate(labels)}


def labels_to_vec(s: str):
    s = str(s).strip()
    vec = np.zeros(len(labels), dtype=np.float32)
    if not s:
        return vec
    parts = s.split()
    for p in parts:
        if p in label_to_idx:
            vec[label_to_idx[p]] = 1.0
    return vec


y = np.stack([labels_to_vec(s) for s in df_train["labels"].values], axis=0)
df_train = df_train.copy()
df_train["filepath"] = TRAIN_IMAGES_DIR.as_posix() + "/" + df_train["image"].astype(str)

paths_np = df_train["filepath"].to_numpy(dtype=str)

for p in paths_np[:5]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(
            f"Training image referenced in train.csv does not exist: {p}"
        )

for i, l in enumerate(labels):
    df_train[l] = y[:, i].astype(np.float32)

df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_frac = 0.1
n_val = int(len(df_train) * val_frac)
df_val = df_train.iloc[:n_val].reset_index(drop=True)
df_trn = df_train.iloc[n_val:].reset_index(drop=True)

print("Train split:", len(df_trn), "Val split:", len(df_val))



## === cell 2
IMG_SIZE = (380, 380)
BATCH_SIZE = 32
EPOCHS = 3

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True
try:
    DATA_OPTS.threading.private_threadpool_size = max(8, (os.cpu_count() or 8) - 1)
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

try:
    import tensorflow_addons as tfa  # noqa: F401
except Exception:
    tfa = None


def decode_and_resize(path_or_bytes, is_bytes=False):
    img_bytes = path_or_bytes if is_bytes else tf.io.read_file(path_or_bytes)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


def augment(img, seed):
    seed = tf.cast(seed, tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-15.0, maxval=15.0
    ) * (np.pi / 180.0)
    if tfa is not None:
        img = tfa.image.rotate(img, angles=angle, fill_mode="reflect")
    else:
        img = tf.image.rot90(img, k=tf.cast(tf.round(angle / (np.pi / 2.0)), tf.int32))

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE[0], tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE[1], tf.float32)
    if tfa is not None:
        img = tfa.image.translate(img, translations=[ty, tx], fill_mode="reflect")

    z = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * z), tf.int32)
    img2 = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])
    img = img2

    return img


def make_train_ds(df, shuffle=True):
    paths = np.asarray(df["filepath"].values, dtype=np.str_)
    ys = np.asarray(df[labels].values, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    ds = ds.with_options(DATA_OPTS)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=42, reshuffle_each_iteration=True
        )

    @tf.function
    def _read_bytes(path, y):
        return tf.io.read_file(path), y

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE).cache()

    seed_stream = tf.data.Dataset.random(seed=42, rerandomize_each_iteration=True).map(
        lambda v: tf.cast(v * 2147483647.0, tf.int32), num_parallel_calls=AUTOTUNE
    )
    ds = tf.data.Dataset.zip((ds, seed_stream))

    @tf.function
    def _decode_aug(data, s):
        img_bytes, y = data
        img = decode_and_resize(img_bytes, is_bytes=True)
        seed = tf.stack([tf.cast(42, tf.int32), s])
        img = augment(img, seed)
        return img, y

    ds = ds.map(_decode_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df):
    paths = np.asarray(df["filepath"].values, dtype=np.str_)
    ys = np.asarray(df[labels].values, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    ds = ds.with_options(DATA_OPTS)

    @tf.function
    def _read_bytes(path, y):
        return tf.io.read_file(path), y

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE).cache()

    @tf.function
    def _decode(img_bytes, y):
        img = decode_and_resize(img_bytes, is_bytes=True)
        return img, y

    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(df_trn, shuffle=True)
val_ds = make_val_ds(df_val)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2864740344.py in <cell line: 0>()
    139 
    140 
--> 141 train_ds = make_train_ds(df_trn, shuffle=True)
    142 val_ds = make_val_ds(df_val)
    143 

/tmp/ipykernel_11/2864740344.py in make_train_ds(df, shuffle)
     96     # Runtime fix: avoid enumerate() (introduces serial dependency) by using Dataset.random
     97     # to produce deterministic per-example seeds in parallel.
---> 98     seed_stream = tf.data.Dataset.random(seed=42, rerandomize_each_iteration=True).map(
     99         lambda v: tf.cast(v * 2147483647.0, tf.int32), num_parallel_calls=AUTOTUNE
    100     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_file4ceb1k2w.py in <lambda>(v)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda v: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.cast, (v * 2147483647.0, tf.int32), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file4ceb1k2w.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda v: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.cast, (v * 2147483647.0, tf.int32), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    559           raise err
    560         else:
--> 561           raise TypeError(
    562               f"Expected {dtypes.as_dtype(dtype).name} passed to parameter "
    563               f"'{input_arg.name}' of op '{op_type_name}', got "

TypeError: in user code:

    File "/tmp/ipykernel_11/2864740344.py", line 99, in None  *
        lambda v: tf.cast(v * 2147483647.0, tf.int32)

    TypeError: Expected int64 passed to parameter 'y' of op 'Mul', got 2147483647.0 of type 'float' instead. Error: Expected int64, but got 2147483647.0 of type 'float'.


## === cell 3
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3115365910.py in <cell line: 0>()
     17 
     18 history = model.fit(
---> 19     train_ds,
     20     validation_data=val_ds,
     21     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
test_paths = np.asarray(
    sorted(tf.io.gfile.glob((TEST_IMAGES_DIR / "*.jpg").as_posix())), dtype=np.str_
)


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DATA_OPTS)

    @tf.function
    def _read_bytes(path):
        return tf.io.read_file(path)

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE).cache()

    @tf.function
    def _decode(img_bytes):
        img = decode_and_resize(img_bytes, is_bytes=True)
        return img

    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(128, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_paths)
print("Test images:", len(test_paths))



## === cell 5
x = model.predict(test_ds, verbose=1)
print("Pred shape:", x.shape)

threshold = 0.7
z = x > threshold

label_arr = np.array(labels, dtype=object)
idxs = [np.flatnonzero(row) for row in z]
predictions_str = [" ".join(label_arr[ii]) if len(ii) else "healthy" for ii in idxs]

filenames = [Path(p).name for p in test_paths]
df_pred = pd.DataFrame({"image": filenames, "labels": predictions_str})

sample = pd.read_csv(SAMPLE_SUB)
df = sample[["image"]].merge(df_pred, on="image", how="left")
df["labels"] = df["labels"].fillna("healthy")

print(df.head())
print("Rows:", len(df), "Unique images:", df["image"].nunique())



## === cell 6
out_path = Path("submission.csv")
df.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print("Submission columns:", list(df.columns))
print("Any missing labels:", df["labels"].isna().sum())
print(
    "Non-string labels:", (df["labels"].apply(lambda v: not isinstance(v, str))).sum()
)
