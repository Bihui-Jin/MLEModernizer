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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9438

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
import random

random.seed(42)
np.random.seed(42)

print("Listing a few input files:")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
from zipfile import ZipFile



## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
train_csv_path = os.path.join(path, "train.csv")
train_zip_path = os.path.join(path, "train.zip")
test_zip_path = os.path.join(path, "test.zip")
sample_sub_path = os.path.join(path, "sample_submission.csv")

files_dataframe = pd.read_csv(train_csv_path, dtype={"id": str, "has_cactus": str})
files_dataframe.head()



## === cell 3
os.makedirs("./training", exist_ok=True)
os.makedirs("./test", exist_ok=True)

train_extracted_dir = "./training/train"
test_extracted_dir = "./test/test"


def _dir_has_files(d):
    try:
        it = os.scandir(d)
    except FileNotFoundError:
        return False
    with it:
        for _ in it:
            return True
    return False


if not _dir_has_files(train_extracted_dir):
    with ZipFile(train_zip_path, "r") as zipper:
        zipper.extractall("./training")
else:
    print("Train already extracted; skipping unzip.")

if not _dir_has_files(test_extracted_dir):
    with ZipFile(test_zip_path, "r") as zipper:
        zipper.extractall("./test")
else:
    print("Test already extracted; skipping unzip.")

print("Train dir exists:", os.path.isdir("./training/train"))
print("Test dir exists:", os.path.isdir("./test/test"))



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()

total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts["1"])
no_cactus_weight = total_samples / (2 * class_reparts["0"])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 5
import tensorflow as tf
import tf_keras

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
def _tf_percentile_rescale_3_97(img_u8):
    img = tf.cast(img_u8, tf.float32)  # 0..255
    flat = tf.reshape(img, [-1])

    n = tf.size(flat)
    n_f = tf.cast(n, tf.float32)

    k3 = tf.cast(tf.floor(0.03 * (n_f - 1.0)), tf.int32)
    k97 = tf.cast(tf.floor(0.97 * (n_f - 1.0)), tf.int32)

    neg = -flat
    k3_top = tf.maximum(1, n - k3)
    k97_top = tf.maximum(1, n - k97)

    p2 = -tf.nn.top_k(neg, k=k3_top, sorted=True).values[-1]
    p98 = -tf.nn.top_k(neg, k=k97_top, sorted=True).values[-1]

    denom = tf.maximum(p98 - p2, 1e-6)
    img = (img - p2) / denom
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img * 255.0
    return img


def _tf_samplewise_center_std(img_f32):
    mean = tf.reduce_mean(img_f32)
    std = tf.math.reduce_std(img_f32)
    std = tf.maximum(std, 1e-6)
    return (img_f32 - mean) / std




## === cell 7
BATCH_SIZE = 128
IMG_SIZE = (32, 32)


def _decode_jpeg(path):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=False)  # float32
    img = tf.cast(img, tf.uint8)  # keep percentile semantics on 0..255 like original
    return img


def _augment(img_u8, seed):
    seed = tf.cast(seed, tf.int32)
    img = tf.image.stateless_random_flip_left_right(
        img_u8, seed=seed + tf.constant([1, 0], tf.int32)
    )
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed + tf.constant([2, 0], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-45.0, maxval=45.0
    )
    rad = angle * (np.pi / 180.0)
    img = tf.image.rotate(
        tf.cast(img, tf.float32), rad, fill_mode="reflect", interpolation="bilinear"
    )
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)

    shear_deg = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=-10.0, maxval=10.0
    )
    shear = tf.math.tan(shear_deg * (np.pi / 180.0))
    a0 = 1.0
    a1 = shear
    a2 = 0.0
    b0 = 0.0
    b1 = 1.0
    b2 = 0.0
    c0 = 0.0
    c1 = 0.0
    transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[tf.newaxis, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.cast(img, tf.float32)[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant(IMG_SIZE, dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
    return img


def _preprocess(img_u8):
    img = _tf_percentile_rescale_3_97(img_u8)
    img = _tf_samplewise_center_std(img)
    return img


def _make_dataset(filepaths, labels, training, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.enumerate()  # (idx, (path,label))

    def _map_fn(idx, data):
        path, y = data
        img_u8 = _decode_jpeg(path)
        if training:
            img_u8 = _augment(
                img_u8, seed=tf.stack([tf.cast(seed, tf.int32), tf.cast(idx, tf.int32)])
            )
        x = _preprocess(img_u8)
        y_oh = tf.one_hot(tf.cast(y, tf.int32), depth=2, dtype=tf.float32)
        return x, y_oh

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 8
files_dataframe["has_cactus"] = files_dataframe["has_cactus"].astype(str)

train_dir = "./training/train/"
all_paths = np.array(
    [os.path.join(train_dir, fn) for fn in files_dataframe["id"].values], dtype=object
)
all_labels = files_dataframe["has_cactus"].astype(int).values

n = len(all_paths)
val_frac = 0.25
n_val = int(np.floor(n * val_frac))
rng = np.random.RandomState(42)
perm = rng.permutation(n)

val_idx = perm[:n_val]
train_idx = perm[n_val:]

train_paths, train_labels = all_paths[train_idx], all_labels[train_idx]
val_paths, val_labels = all_paths[val_idx], all_labels[val_idx]

training_ds = _make_dataset(train_paths, train_labels, training=True, seed=42)
validation_ds = _make_dataset(val_paths, val_labels, training=False, seed=42)

class_indices = {"0": 0, "1": 1}
print("Class indices (label -> column):", class_indices)
print("Train samples:", len(train_paths), "Val samples:", len(val_paths))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/968781966.py in <cell line: 0>()
     22 val_paths, val_labels = all_paths[val_idx], all_labels[val_idx]
     23 
---> 24 training_ds = _make_dataset(train_paths, train_labels, training=True, seed=42)
     25 validation_ds = _make_dataset(val_paths, val_labels, training=False, seed=42)
     26 

/tmp/ipykernel_11/3753517691.py in _make_dataset(filepaths, labels, training, seed)
     97         return x, y_oh
     98 
---> 99     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    100     # Cache decoded+preprocessed tensors to avoid re-decoding overhead across epochs.
    101     ds = ds.cache()

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

/tmp/__autograph_generated_file070vg3_v.py in tf___map_fn(idx, data)
     27                     nonlocal img_u8
     28                     pass
---> 29                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img_u8',), 1)
     30                 x = ag__.converted_call(ag__.ld(_preprocess), (ag__.ld(img_u8),), None, fscope)
     31                 y_oh = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(y), ag__.ld(tf).int32), None, fscope),), dict(depth=2, dtype=ag__.ld(tf).float32), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_file070vg3_v.py in if_body()
     22                 def if_body():
     23                     nonlocal img_u8
---> 24                     img_u8 = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img_u8),), dict(seed=ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(seed), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(idx), ag__.ld(tf).int32), None, fscope)],), None, fscope)), fscope)
     25 
     26                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filenfv8tukk.py in tf___augment(img_u8, seed)
     13                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([3, 0], ag__.ld(tf).int32), None, fscope), minval=-45.0, maxval=45.0), fscope)
     14                 rad = ag__.ld(angle) * (ag__.ld(np).pi / 180.0)
---> 15                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope), ag__.ld(rad)), dict(fill_mode='reflect', interpolation='bilinear'), fscope)
     16                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).clip_by_value, (ag__.ld(img), 0.0, 255.0), None, fscope), ag__.ld(tf).uint8), None, fscope)
     17                 shear_deg = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([4, 0], ag__.ld(tf).int32), None, fscope), minval=-10.0, maxval=10.0), fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/3753517691.py", line 91, in _map_fn  *
        img_u8 = _augment(
    File "/tmp/ipykernel_11/3753517691.py", line 33, in _augment  *
        img = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 9
from tf_keras import layers, models

model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 10
from tf_keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/tmp/checkpoint.keras"
save_best_model = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)



## === cell 11
history = model.fit(
    training_ds,
    validation_data=validation_ds,
    verbose=1,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1130452243.py in <cell line: 0>()
      1 # NOTE (speed): tf.data pipeline handles parallelism; Keras workers args are for Python generators.
      2 history = model.fit(
----> 3     training_ds,
      4     validation_data=validation_ds,
      5     verbose=1,

NameError: name 'training_ds' is not defined

## === cell 12
test_dir = "./test/test"
test_ids = sorted([fn for fn in os.listdir(test_dir) if fn.lower().endswith(".jpg")])
test_paths = np.array([os.path.join(test_dir, fn) for fn in test_ids], dtype=object)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _preprocess(_decode_jpeg(p)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/219394315.py in <cell line: 0>()
      1 # NOTE (speed/correctness): fast tf.data test input pipeline (no augmentation), shuffle=False like original.
      2 test_dir = "./test/test"
----> 3 test_ids = sorted([fn for fn in os.listdir(test_dir) if fn.lower().endswith(".jpg")])
      4 test_paths = np.array([os.path.join(test_dir, fn) for fn in test_ids], dtype=object)
      5 

FileNotFoundError: [Errno 2] No such file or directory: './test/test'

## === cell 13
from tf_keras.models import load_model

model = load_model(checkpoint_path)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3017079564.py in <cell line: 0>()
      1 from tf_keras.models import load_model
      2 
----> 3 model = load_model(checkpoint_path)
      4 

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode, **kwargs)
    260 
    261     # Legacy case.
--> 262     return legacy_sm_saving_lib.load_model(
    263         filepath, custom_objects=custom_objects, compile=compile, **kwargs
    264     )

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/legacy/save.py in load_model(filepath, custom_objects, compile, options)
    231                     if isinstance(filepath_str, str):
    232                         if not tf.io.gfile.exists(filepath_str):
--> 233                             raise IOError(
    234                                 f"No file or directory found at {filepath_str}"
    235                             )

OSError: No file or directory found at /tmp/checkpoint.keras

## === cell 14
probs = model.predict(test_ds, verbose=1)

idx_pos = class_indices.get("1", 1)
has_cactus_prob = probs[:, idx_pos].astype(float)

sample_sub = pd.read_csv(sample_sub_path, dtype={"id": str})

pred_df = pd.DataFrame({"id": test_ids, "has_cactus": has_cactus_prob})
pred_df = sample_sub[["id"]].merge(pred_df, on="id", how="left")
pred_df["has_cactus"] = pred_df["has_cactus"].fillna(0.5)

print("Pred rows:", len(pred_df), "Sample rows:", len(sample_sub))
print("Any missing predictions filled with 0.5:", (pred_df["has_cactus"] == 0.5).sum())

pred_df.to_csv("submission.csv", index=False)
print(pred_df.head())
print("Wrote submission.csv with shape:", pred_df.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1051632181.py in <cell line: 0>()
----> 1 probs = model.predict(test_ds, verbose=1)
      2 
      3 idx_pos = class_indices.get("1", 1)
      4 has_cactus_prob = probs[:, idx_pos].astype(float)
      5 

NameError: name 'test_ds' is not defined

## === cell 15
import shutil

try:
    shutil.rmtree("test")
except OSError:
    print("Test files already erased or not present")
try:
    shutil.rmtree("training")
except OSError:
    print("Training files already erased or not present")
