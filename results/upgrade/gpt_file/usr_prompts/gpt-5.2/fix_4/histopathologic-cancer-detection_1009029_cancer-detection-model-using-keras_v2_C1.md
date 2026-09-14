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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.7793147687630658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from glob import glob
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from skimage.io import imread

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential

print("TF version:", tf.__version__)
print("Listing ../input:", os.listdir("../input")[:20])


def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these paths exist: " + str(paths))


DATA_ROOT = pick_existing(
    "../input/histopathologic-cancer-detection",
    "../input",
    "/kaggle/input/histopathologic-cancer-detection",
    "/kaggle/input",
)

TRAIN_LABELS_CSV = pick_existing(
    os.path.join(DATA_ROOT, "train_labels.csv"),
    "../input/train_labels.csv",
)
SAMPLE_SUB_CSV = pick_existing(
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    "../input/sample_submission.csv",
)

TRAIN_DIR = pick_existing(
    os.path.join(DATA_ROOT, "train"),
    "../input/train",
)
TEST_DIR = pick_existing(
    os.path.join(DATA_ROOT, "test"),
    "../input/test",
)

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_LABELS_CSV:", TRAIN_LABELS_CSV)
print("SAMPLE_SUB_CSV:", SAMPLE_SUB_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    import tensorflow_io as tfio  # type: ignore
except Exception as e:
    tfio = None
    print(
        "WARNING: tensorflow_io not available; will fall back to skimage.imdecode (slower). Error:",
        repr(e),
    )


def _decode_tiff_tf(path_bytes):
    """Decode a single TIFF from a filename (tf.string) to uint8 tensor [H,W,3]."""
    img_bytes = tf.io.read_file(path_bytes)
    img = tfio.experimental.image.decode_tiff(img_bytes)  # [H,W,C] uint8
    img = tf.ensure_shape(img, [96, 96, 3])
    return img


def _decode_tiff_py(path_bytes):
    """Fallback: decode via skimage in a py_function; correct but slower."""

    def _read(p):
        p = p.decode("utf-8")
        return imread(p).astype(np.uint8)

    img = tf.py_function(_read, [path_bytes], Tout=tf.uint8)
    img = tf.ensure_shape(img, [96, 96, 3])
    return img


def make_image_ds(paths, batch_size, shuffle=False, seed=42):
    """Build a deterministic tf.data pipeline that outputs float32 images normalized to [0,1]."""
    ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(paths, dtype=tf.string)
    )
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=False
        )
    if tfio is not None:
        decode = _decode_tiff_tf
        num_parallel = tf.data.AUTOTUNE
    else:
        decode = _decode_tiff_py
        num_parallel = 1  # py_function isn't parallel-safe for speed/determinism
    ds = ds.map(decode, num_parallel_calls=num_parallel)
    ds = ds.map(
        lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_LABELS_CSV)
train_df.head()



## === cell 2
train_df.label.unique()



## === cell 3
distribution = train_df.label.value_counts()
print(distribution)
p = distribution[1] / distribution.sum()
print("Percentage of cancer affected cells are {}".format(p))



## === cell 4
label_counts = train_df["label"].value_counts()
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation=90)



## === cell 5
base_tile_dir = TRAIN_DIR
df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})

df["id"] = df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

labels = pd.read_csv(TRAIN_LABELS_CSV)
df = df.merge(labels, on="id")
df.head(10)



## === cell 6
df0 = df[df.label == 0].sample(5000, random_state=42)
df1 = df[df.label == 1].sample(5000, random_state=42)
df = pd.concat([df0, df1], ignore_index=True).reset_index(drop=True)
df = df[["path", "id", "label"]]
df.sample(10)



## === cell 7
train_paths = df["path"].values.astype(str)
train_ds_for_np = make_image_ds(train_paths, batch_size=512, shuffle=False, seed=42)

input_images = np.empty((len(df), 96, 96, 3), dtype=np.float32)
offset = 0
for batch in train_ds_for_np:
    b = batch.numpy()
    input_images[offset : offset + b.shape[0]] = b
    offset += b.shape[0]
del train_ds_for_np
gc.collect()

df["image"] = list((input_images * 255.0).astype(np.uint8))
df.sample(3)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2852859244.py in <cell line: 0>()
      2 # Correctness: we still load the exact same files and keep uint8 pixels; images are used later only to build input_images.
      3 train_paths = df["path"].values.astype(str)
----> 4 train_ds_for_np = make_image_ds(train_paths, batch_size=512, shuffle=False, seed=42)
      5 
      6 # Materialize into a single numpy array (keeps the same later logic: x is a dense array).

/tmp/ipykernel_11/3201460225.py in make_image_ds(paths, batch_size, shuffle, seed)
    119         decode = _decode_tiff_py
    120         num_parallel = 1  # py_function isn't parallel-safe for speed/determinism
--> 121     ds = ds.map(decode, num_parallel_calls=num_parallel)
    122     ds = ds.map(
    123         lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE

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

/tmp/__autograph_generated_fileu1nyam5d.py in tf___decode_tiff_tf(path_bytes)
     10                 retval_ = ag__.UndefinedReturnValue()
     11                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path_bytes),), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img_bytes),), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(img), [96, 96, 3]), None, fscope)
     14                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    432     if is_autograph_strict_conversion_mode():
    433       raise
--> 434     return _fall_back_unconverted(f, args, kwargs, options, e)
    435 
    436   with StackTraceMapper(converted_f), tf_stack.CurrentModuleFilter():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _fall_back_unconverted(f, args, kwargs, options, exc)
    483     logging.warning(warning_template, f, file_bug_message, exc)
    484 
--> 485   return _call_unconverted(f, args, kwargs, options)
    486 
    487 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py in decode_tiff(contents, index, name)
     85       A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
     86     """
---> 87     return core_ops.io_decode_tiff(contents, index, name=name)
     88 
     89 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in __getattr__(self, attrb)
     86 
     87     def __getattr__(self, attrb):
---> 88         return getattr(self._load(), attrb)
     89 
     90     def __dir__(self):

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load(self)
     82     def _load(self):
     83         if self._mod is None:
---> 84             self._mod = _load_library(self._library)
     85         return self._mod
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load_library(filename, lib)
     67         except (tf.errors.NotFoundError, OSError) as e:
     68             errs.append(str(e))
---> 69     raise NotImplementedError(
     70         "unable to open file: "
     71         + f"{filename}, from paths: {filenames}\ncaused by: {errs}"

NotImplementedError: in user code:

    File "/tmp/ipykernel_11/3201460225.py", line 88, in _decode_tiff_tf  *
        img = tfio.experimental.image.decode_tiff(img_bytes)  # [H,W,C] uint8
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py", line 87, in decode_tiff  **
        return core_ops.io_decode_tiff(contents, index, name=name)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 8
images = [
    (df["image"].iloc[0], df["label"].iloc[0]),
    (df["image"].iloc[1], df["label"].iloc[1]),
    (df["image"].iloc[2], df["label"].iloc[2]),
    (df["image"].iloc[5000], df["label"].iloc[5000]),
    (df["image"].iloc[5001], df["label"].iloc[5001]),
    (df["image"].iloc[5002], df["label"].iloc[5002]),
]

fig, m_axs = plt.subplots(1, len(images), figsize=(20, 2))
for ii, c_ax in enumerate(m_axs):
    c_ax.imshow(images[ii][0])
    c_ax.set_title(images[ii][1])
    c_ax.axis("off")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'image'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1366520383.py in <cell line: 0>()
      1 images = [
----> 2     (df["image"].iloc[0], df["label"].iloc[0]),
      3     (df["image"].iloc[1], df["label"].iloc[1]),
      4     (df["image"].iloc[2], df["label"].iloc[2]),
      5     (df["image"].iloc[5000], df["label"].iloc[5000]),

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'image'

## === cell 9
input_images.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3390723135.py in <cell line: 0>()
      1 # input_images already built and normalized to float32 [0,1].
----> 2 input_images.shape
      3 

NameError: name 'input_images' is not defined

## === cell 10
x = input_images
y = df["label"].astype(np.int32).values

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.10, random_state=101, stratify=y
)

train_y.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1035940290.py in <cell line: 0>()
----> 1 x = input_images
      2 y = df["label"].astype(np.int32).values
      3 
      4 train_x, test_x, train_y, test_y = train_test_split(
      5     x, y, test_size=0.10, random_state=101, stratify=y

NameError: name 'input_images' is not defined

## === cell 11
np.random.seed(42)
tf.random.set_seed(42)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=False
)

checkpointer = ModelCheckpoint(filepath="weights.keras", verbose=1, save_best_only=True)

model = Sequential()
model.add(
    Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=(96, 96, 3),
    )
)
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="elu"))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="elu"))
model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="elu"))

model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.summary()



## === cell 12
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

epochs = 15
history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    epochs=epochs,
    batch_size=80,
    verbose=1,
    callbacks=[early_stopping, checkpointer],
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1447901090.py in <cell line: 0>()
      3 epochs = 15
      4 history = model.fit(
----> 5     train_x,
      6     train_y,
      7     validation_data=(test_x, test_y),

NameError: name 'train_x' is not defined

## === cell 13
model = tf.keras.models.load_model("weights.keras")

val_pred = model.predict(test_x, batch_size=256, verbose=0).ravel()
val_auc = roc_auc_score(test_y, val_pred)
print("Validation ROC-AUC:", val_auc)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2588993654.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model("weights.keras")
      2 
      3 val_pred = model.predict(test_x, batch_size=256, verbose=0).ravel()
      4 val_auc = roc_auc_score(test_y, val_pred)
      5 print("Validation ROC-AUC:", val_auc)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=weights.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 14
base_tile_dir = TEST_DIR
test_df = pd.DataFrame({"path": glob(os.path.join(base_tile_dir, "*.tif"))})
test_df["id"] = test_df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])

sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_df = sample_sub[["id"]].merge(test_df, on="id", how="left")

missing = test_df["path"].isna().sum()
print("Test rows:", len(test_df), "Missing paths after merge:", missing)
test_df.head()



## === cell 15
n = len(test_df)
preds = np.full(n, 0.5, dtype=np.float32)

valid_mask = test_df["path"].notna().values
valid_paths = test_df.loc[valid_mask, "path"].values.astype(str)

if len(valid_paths) > 0:
    infer_bs = 512
    test_ds = make_image_ds(valid_paths, batch_size=infer_bs, shuffle=False, seed=42)
    valid_preds = model.predict(test_ds, verbose=0).ravel().astype(np.float32)
    preds[valid_mask] = valid_preds
    del test_ds, valid_preds
    gc.collect()

test_df["label"] = preds
test_df[["id", "label"]].head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/3465283489.py in <cell line: 0>()
     11     # Larger batch improves throughput; model.predict will internally handle last partial batch.
     12     infer_bs = 512
---> 13     test_ds = make_image_ds(valid_paths, batch_size=infer_bs, shuffle=False, seed=42)
     14     valid_preds = model.predict(test_ds, verbose=0).ravel().astype(np.float32)
     15     preds[valid_mask] = valid_preds

/tmp/ipykernel_11/3201460225.py in make_image_ds(paths, batch_size, shuffle, seed)
    119         decode = _decode_tiff_py
    120         num_parallel = 1  # py_function isn't parallel-safe for speed/determinism
--> 121     ds = ds.map(decode, num_parallel_calls=num_parallel)
    122     ds = ds.map(
    123         lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE

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

/tmp/__autograph_generated_fileu1nyam5d.py in tf___decode_tiff_tf(path_bytes)
     10                 retval_ = ag__.UndefinedReturnValue()
     11                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path_bytes),), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img_bytes),), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(img), [96, 96, 3]), None, fscope)
     14                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    329   if conversion.is_in_allowlist_cache(f, options):
    330     logging.log(2, 'Allowlisted %s: from cache', f)
--> 331     return _call_unconverted(f, args, kwargs, options, False)
    332 
    333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py in decode_tiff(contents, index, name)
     85       A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
     86     """
---> 87     return core_ops.io_decode_tiff(contents, index, name=name)
     88 
     89 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in __getattr__(self, attrb)
     86 
     87     def __getattr__(self, attrb):
---> 88         return getattr(self._load(), attrb)
     89 
     90     def __dir__(self):

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load(self)
     82     def _load(self):
     83         if self._mod is None:
---> 84             self._mod = _load_library(self._library)
     85         return self._mod
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load_library(filename, lib)
     67         except (tf.errors.NotFoundError, OSError) as e:
     68             errs.append(str(e))
---> 69     raise NotImplementedError(
     70         "unable to open file: "
     71         + f"{filename}, from paths: {filenames}\ncaused by: {errs}"

NotImplementedError: in user code:

    File "/tmp/ipykernel_11/3201460225.py", line 88, in _decode_tiff_tf  *
        img = tfio.experimental.image.decode_tiff(img_bytes)  # [H,W,C] uint8
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py", line 87, in decode_tiff  **
        return core_ops.io_decode_tiff(contents, index, name=name)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 16
submission = test_df[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False, header=True)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3498513362.py in <cell line: 0>()
----> 1 submission = test_df[["id", "label"]].copy()
      2 submission.to_csv("submission.csv", index=False, header=True)
      3 
      4 print("Wrote submission.csv with shape:", submission.shape)
      5 print(submission.head())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['label'] not in index"
