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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.15789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

print("Input root:", os.listdir("/kaggle/input")[:10])



## === cell 1
try:
    import google.protobuf  # noqa: F401
    import pkgutil
    import importlib.metadata as importlib_metadata

    pb_ver = importlib_metadata.version("protobuf")
    print("Detected protobuf:", pb_ver)
    major = int(pb_ver.split(".")[0])
    if major >= 5:
        print("Downgrading protobuf to 4.25.3 for TensorFlow compatibility...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf==4.25.3",
            ]
        )
        import importlib

        importlib.invalidate_caches()
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf") or m == "protobuf":
                sys.modules.pop(m, None)
        pb_ver2 = importlib_metadata.version("protobuf")
        print("protobuf after install attempt:", pb_ver2)
except Exception as e:
    print("protobuf pin step skipped/failed (continuing):", repr(e))



## === cell 2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

print("TensorFlow:", tf.__version__)



## === cell 3
train_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
print("Train shape:", train_df.shape, "Test(submission template) shape:", test_df.shape)
print(train_df.head())



## === cell 4
from glob import glob

datapath = glob("/kaggle/input/plant-pathology-2021-fgvc8/train_images/*")
print("Number of train images found:", len(datapath))




## === cell 5
def add_link(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + str(path)




## === cell 6
def add_link_test(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + str(path)




## === cell 7
train_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + train_df[
    "image"
].astype(str)
print(train_df.head())



## === cell 8
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
test_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + test_df[
    "image"
].astype(str)
print(test_df.head())



## === cell 9
print("Unique label strings:", train_df["labels"].nunique())
print(train_df["labels"].value_counts().head(10))



## === cell 10
unique_list = np.unique(train_df["labels"])
print("Num unique label strings:", len(unique_list))



## === cell 11
all_tokens = sorted(
    {tok for s in train_df["labels"].astype(str).tolist() for tok in s.split() if tok}
)
print("Num classes (tokens):", len(all_tokens))
print("Classes:", all_tokens)




## === cell 12
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, channels=3, dtype=tf.float32)
    return image


def get_label(path):
    return train_df.loc[train_df["image"] == path, "labels"].tolist()


def get_label_image(path):
    label = get_label(path)
    image = read_image(path)
    return label, image




## === cell 13
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32
CLASSES = len(all_tokens)

print("INPUT_SIZE:", INPUT_SIZE, "BATCH_SIZE:", BATCH_SIZE, "CLASSES:", CLASSES)



## === cell 14
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True
)
print("Train Data Shape: ", train_data.shape)
print("Validation Data Shape: ", val_data.shape)



## === cell 15
AUTO = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(all_tokens)}
keys = tf.constant(list(class_to_idx.keys()), dtype=tf.string)
vals = tf.constant(list(class_to_idx.values()), dtype=tf.int64)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1
)

BASE_SEED = tf.constant([42, 4242], dtype=tf.int32)


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _labels_to_onehot(label_str):
    tokens = tf.strings.split(label_str)  # RaggedTensor [num_tokens]
    idx = table.lookup(tokens)  # RaggedTensor[int64]
    idx = tf.ragged.boolean_mask(idx, idx >= 0)

    idx = tf.ragged.map_flat_values(tf.cast, idx, tf.int32)  # RaggedTensor[int32]
    flat_idx = idx.flat_values  # now safe (RaggedTensor)
    onehot = tf.reduce_sum(
        tf.one_hot(flat_idx, depth=CLASSES, dtype=tf.float32), axis=0
    )
    onehot = tf.clip_by_value(onehot, 0.0, 1.0)
    return onehot


def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    max_dx = tf.cast(tf.round(0.3 * tf.cast(INPUT_SIZE[1], tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        (),
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    if max_dx > 0:
        pad = tf.abs(dx)
        img = tf.pad(img, [[0, 0], [pad, pad], [0, 0]], mode="REFLECT")
        start_x = pad + dx
        img = tf.image.crop_to_bounding_box(
            img, 0, start_x, INPUT_SIZE[0], INPUT_SIZE[1]
        )
    zoom = tf.random.stateless_uniform(
        (),
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=0.8,
        maxval=1.0,
        dtype=tf.float32,
    )
    crop_h = tf.cast(tf.round(zoom * INPUT_SIZE[0]), tf.int32)
    crop_w = tf.cast(tf.round(zoom * INPUT_SIZE[1]), tf.int32)
    crop_h = tf.maximum(crop_h, 1)
    crop_w = tf.maximum(crop_w, 1)
    img = tf.image.stateless_random_crop(
        img, size=[crop_h, crop_w, 3], seed=seed + tf.constant([3, 0], tf.int32)
    )
    img = tf.image.resize(img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR)
    return img


def make_train_ds(df):
    paths = df["image"].astype(str).values
    labels = df["labels"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    def _map(path, label_str):
        img = _decode_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        ex_seed = tf.stack([BASE_SEED[0], tf.cast(h, tf.int32)])
        img = _augment(img, ex_seed)
        y = _labels_to_onehot(label_str)
        return img, y

    ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_val_ds(df):
    paths = df["image"].astype(str).values
    labels = df["labels"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path, label_str):
        img = _decode_resize(path)
        y = _labels_to_onehot(label_str)
        return img, y

    ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_test_ds(df):
    paths = df["image"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        img = _decode_resize(path)
        return img

    ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_generator = make_train_ds(train_data)
val_generator = make_val_ds(val_data)
test_generator = make_test_ds(test_df)

class_indices = {c: i for i, c in enumerate(all_tokens)}
print(
    "Datasets built. Example batch shapes:",
    next(iter(train_generator))[0].shape,
    next(iter(train_generator))[1].shape,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1705884575.py in <cell line: 0>()
    121 
    122 
--> 123 train_generator = make_train_ds(train_data)
    124 val_generator = make_val_ds(val_data)
    125 test_generator = make_test_ds(test_df)

/tmp/ipykernel_11/1705884575.py in make_train_ds(df)
     85         return img, y
     86 
---> 87     ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
     88     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
     89     ds = ds.prefetch(AUTO)

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

/tmp/__autograph_generated_filekr1k3hcn.py in tf___map(path, label_str)
     12                 ex_seed = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(BASE_SEED)[0], ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope)],), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(ex_seed)), None, fscope)
---> 14                 y = ag__.converted_call(ag__.ld(_labels_to_onehot), (ag__.ld(label_str),), None, fscope)
     15                 try:
     16                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filedgjf4vfb.py in tf___labels_to_onehot(label_str)
     12                 idx = ag__.converted_call(ag__.ld(tf).ragged.boolean_mask, (ag__.ld(idx), ag__.ld(idx) >= 0), None, fscope)
     13                 idx = ag__.converted_call(ag__.ld(tf).ragged.map_flat_values, (ag__.ld(tf).cast, ag__.ld(idx), ag__.ld(tf).int32), None, fscope)
---> 14                 flat_idx = ag__.ld(idx).flat_values
     15                 onehot = ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(flat_idx),), dict(depth=ag__.ld(CLASSES), dtype=ag__.ld(tf).float32), fscope),), dict(axis=0), fscope)
     16                 onehot = ag__.converted_call(ag__.ld(tf).clip_by_value, (ag__.ld(onehot), 0.0, 1.0), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __getattr__(self, name)
    258         tf.experimental.numpy.experimental_enable_numpy_behavior()
    259       """)
--> 260     self.__getattribute__(name)
    261 
    262   @property

AttributeError: in user code:

    File "/tmp/ipykernel_11/1705884575.py", line 84, in _map  *
        y = _labels_to_onehot(label_str)
    File "/tmp/ipykernel_11/1705884575.py", line 30, in _labels_to_onehot  *
        flat_idx = idx.flat_values  # now safe (RaggedTensor)

    AttributeError: 'SymbolicTensor' object has no attribute 'flat_values'


## === cell 16
pre_model = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SIZE)
pre_model.trainable = False  # preserve transfer-learning intent and stability
print("Backbone output shape:", pre_model.output_shape)



## === cell 17
model = tf.keras.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(CLASSES, activation="softmax"))



## === cell 18
callback = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=3, min_lr=1e-5)



## === cell 19
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 20
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[callback],
    verbose=1,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3618540624.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_generator,
      3     epochs=25,
      4     validation_data=val_generator,
      5     callbacks=[callback],

NameError: name 'train_generator' is not defined

## === cell 21
preds = model.predict(test_generator, verbose=1)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/302912489.py in <cell line: 0>()
----> 1 preds = model.predict(test_generator, verbose=1)
      2 

NameError: name 'test_generator' is not defined

## === cell 22
top1_idx = np.argmax(preds, axis=1)

idx_to_class = {v: k for k, v in class_indices.items()}
pred_labels = [idx_to_class[int(i)] for i in top1_idx]

submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
submission["labels"] = pred_labels

assert list(submission.columns) == ["image", "labels"]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3789215058.py in <cell line: 0>()
----> 1 top1_idx = np.argmax(preds, axis=1)
      2 
      3 idx_to_class = {v: k for k, v in class_indices.items()}
      4 pred_labels = [idx_to_class[int(i)] for i in top1_idx]
      5 

NameError: name 'preds' is not defined

## === cell 23
print("Files in CWD:", [f for f in os.listdir(".") if f.endswith(".csv")])
