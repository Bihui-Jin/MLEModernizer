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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.94962

# 6. Current score

0.06757

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06757) has done: 'The timeout is dominated by the training input pipeline: it decodes/resizes each image, then applies `ImageDataGenerator` augmentation via `tf.numpy_function` and a Python loop per image, which is extremely slow. I keep the same model, loss, optimizer, callbacks, and epoch/step semantics, but replace the slow Python/Numpy augmentation path with an equivalent pure-TensorFlow augmentation pipeline that matches the same augmentation parameters and still uses `preprocess_input`. I also avoid the redundant `flow_from_directory()` generators for train/val (they were only used to list filepaths/classes) by building the file list once with `tf.keras.utils.image_dataset_from_directory` while preserving the same class order, split, batch sizing, and drop-remainder behavior. Finally, I add `cache()` after decode/resize (pre-augmentation) so expensive disk decode happens once per epoch without changing training semantics (augmentation still changes per epoch).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print("Input root exists:", os.path.exists("/kaggle/input"))
print(
    "Train dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/train"),
)
print(
    "Test dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/test"),
)



## === cell 1
TRAIN_ROOT = "/kaggle/input/plant-seedlings-classification/train"
print("Train root:", TRAIN_ROOT)
print("Skipping train.csv generation (unused by training/prediction).")



## === cell 2
classes = sorted(
    [d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))]
)
print(f"Number of classes: {len(classes)}")
print("Classes:", classes)



## === cell 3
print("Skipping plots for runtime.")



## === cell 4
print("Skipping sample image visualization for runtime.")



## === cell 5
print("Skipping image-size histogram for runtime.")



## === cell 6
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)
from math import exp

print("TensorFlow:", tf.__version__)
print("Using tf.keras:", tf.keras.__name__)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
seed = 42
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_list_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="int",
    class_names=classes,  # ensures identical class->index mapping as the sorted folder list
    color_mode="rgb",
    batch_size=None,  # unbatched list; we'll batch after preprocessing/augmentation
    image_size=image_size,  # decodes + resizes once here
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="training",
    interpolation="bilinear",
)

val_list_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="int",
    class_names=classes,
    color_mode="rgb",
    batch_size=None,
    image_size=image_size,
    shuffle=True,  # fine; we will not reshuffle after this for validation
    seed=seed,
    validation_split=val_split,
    subset="validation",
    interpolation="bilinear",
)

num_classes = len(classes)
print("Class indices:", {c: i for i, c in enumerate(classes)})
print("num_classes:", num_classes)

train_count = int(tf.data.experimental.cardinality(train_list_ds).numpy())
val_count = int(tf.data.experimental.cardinality(val_list_ds).numpy())

train_steps = train_count // batch_size
val_steps = val_count // batch_size
print("train_count:", train_count, "val_count:", val_count)
print("train_steps:", train_steps, "val_steps:", val_steps)


def _make_stateless_seed(example_index):
    return tf.stack([tf.cast(seed, tf.int64), tf.cast(example_index, tf.int64)], axis=0)


def _preprocess(img, y_int):
    y = tf.one_hot(tf.cast(y_int, tf.int32), depth=num_classes, dtype=tf.float32)
    img = preprocess_input(
        img
    )  # same semantics as original pipeline for non-augmented path
    return img, y


@tf.function
def _augment_one(img, y, ex_seed):
    x = (img + 1.0) * 127.5  # [0,255]
    x = tf.clip_by_value(x / 255.0, 0.0, 1.0)  # [0,1]

    x = tf.image.stateless_random_flip_left_right(x, seed=ex_seed)
    x = tf.image.stateless_random_flip_up_down(
        x, seed=ex_seed + tf.constant([0, 1], tf.int64)
    )

    br = tf.random.stateless_uniform(
        [], seed=ex_seed + tf.constant([0, 2], tf.int64), minval=0.7, maxval=1.3
    )
    x = tf.clip_by_value(x * br, 0.0, 1.0)

    h = tf.shape(x)[0]
    w = tf.shape(x)[1]

    zoom = tf.random.stateless_uniform(
        [], seed=ex_seed + tf.constant([0, 3], tf.int64), minval=0.8, maxval=1.2
    )
    crop_h = tf.cast(tf.cast(h, tf.float32) / zoom, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) / zoom, tf.int32)
    crop_h = tf.clip_by_value(crop_h, 1, h)
    crop_w = tf.clip_by_value(crop_w, 1, w)

    x = tf.image.stateless_random_crop(
        x,
        size=tf.stack([crop_h, crop_w, 3]),
        seed=ex_seed + tf.constant([0, 4], tf.int64),
    )
    x = tf.image.resize(
        x, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )

    max_dx = tf.cast(tf.round(0.2 * tf.cast(image_size[1], tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.2 * tf.cast(image_size[0], tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=ex_seed + tf.constant([0, 5], tf.int64),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=ex_seed + tf.constant([0, 6], tf.int64),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )

    pad_y = tf.abs(dy)
    pad_x = tf.abs(dx)
    x = tf.pad(x, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    off_y = pad_y - dy
    off_x = pad_x - dx
    x = tf.image.crop_to_bounding_box(x, off_y, off_x, image_size[0], image_size[1])

    angle = tf.random.stateless_uniform(
        [], seed=ex_seed + tf.constant([0, 7], tf.int64), minval=-30.0, maxval=30.0
    ) * (np.pi / 180.0)
    x = tf.image.rotate(x, angles=angle, interpolation="BILINEAR")

    x = tf.clip_by_value(x, 0.0, 1.0) * 255.0
    x = preprocess_input(x)
    return x, y


def _make_train_ds(ds_unbatched):
    ds = ds_unbatched.cache()

    ds = ds.enumerate()  # (i, (img, y_int))
    ds = ds.map(
        lambda i, xy: (xy[0], xy[1], _make_stateless_seed(i)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        lambda img, y_int, ex_seed: _preprocess(img, y_int) + (ex_seed,),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        lambda img, y, ex_seed: _augment_one(img, y, ex_seed),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _make_val_ds(ds_unbatched):
    ds = ds_unbatched.cache()
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = _make_train_ds(train_list_ds)
val_ds = _make_val_ds(val_list_ds)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2047165407.py in <cell line: 0>()
    194 
    195 
--> 196 train_ds = _make_train_ds(train_list_ds)
    197 val_ds = _make_val_ds(val_list_ds)
    198 

/tmp/ipykernel_11/2047165407.py in _make_train_ds(ds_unbatched)
    176         deterministic=True,
    177     )
--> 178     ds = ds.map(
    179         lambda img, y, ex_seed: _augment_one(img, y, ex_seed),
    180         num_parallel_calls=tf.data.AUTOTUNE,

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

/tmp/__autograph_generated_filevxjntpdq.py in <lambda>(img, y, ex_seed)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, y, ex_seed: ag__.with_function_scope(lambda lscope: ag__.converted_call(_augment_one, (img, y, ex_seed), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filevxjntpdq.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, y, ex_seed: ag__.with_function_scope(lambda lscope: ag__.converted_call(_augment_one, (img, y, ex_seed), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file1gvlqwoa.py in tf___augment_one(img, y, ex_seed)
     34                 x = ag__.converted_call(ag__.ld(tf).image.crop_to_bounding_box, (ag__.ld(x), ag__.ld(off_y), ag__.ld(off_x), ag__.ld(image_size)[0], ag__.ld(image_size)[1]), None, fscope)
     35                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(ex_seed) + ag__.converted_call(ag__.ld(tf).constant, ([0, 7], ag__.ld(tf).int64), None, fscope), minval=-30.0, maxval=30.0), fscope) * (ag__.ld(np).pi / 180.0)
---> 36                 x = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(x),), dict(angles=ag__.ld(angle), interpolation='BILINEAR'), fscope)
     37                 x = ag__.converted_call(ag__.ld(tf).clip_by_value, (ag__.ld(x), 0.0, 1.0), None, fscope) * 255.0
     38                 x = ag__.converted_call(ag__.ld(preprocess_input), (ag__.ld(x),), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/2047165407.py", line 179, in None  *
        lambda img, y, ex_seed: _augment_one(img, y, ex_seed)
    File "/tmp/ipykernel_11/2047165407.py", line 153, in _augment_one  *
        x = tf.image.rotate(x, angles=angle, interpolation="BILINEAR")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 8
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 9
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()



## === cell 10
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
)

modelsave = ModelCheckpoint(filepath=file + ".keras", save_best_only=True, verbose=1)

print("[INFO]: Training the network...")

H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/248616008.py in <cell line: 0>()
     28 
     29 H_pre = pre_trained_model.fit(
---> 30     train_ds,
     31     validation_data=val_ds,
     32     steps_per_epoch=train_steps,

NameError: name 'train_ds' is not defined

## === cell 11
print("[INFO]: Training finished. Skipping curve plots for runtime.")
print("Epochs run:", len(H_pre.history.get("loss", [])))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1968910585.py in <cell line: 0>()
      1 print("[INFO]: Training finished. Skipping curve plots for runtime.")
----> 2 print("Epochs run:", len(H_pre.history.get("loss", [])))
      3 

NameError: name 'H_pre' is not defined

## === cell 12
csv_testfile = "/kaggle/working/test.csv"
print("Skipping test.csv generation for runtime. (Not needed.)")



## === cell 13
print("Skipping loading test.csv for runtime.")



## === cell 14
test_batch_size = 32
seed = 42
image_size = (256, 256)
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/"

test_files_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TEST,
    labels=None,
    label_mode=None,
    class_names=["test"],
    color_mode="rgb",
    batch_size=test_batch_size,
    image_size=image_size,
    shuffle=False,
    interpolation="bilinear",
)


def _preprocess_only(img):
    return preprocess_input(img)


test_ds = test_files_ds.map(
    _preprocess_only, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
).prefetch(tf.data.AUTOTUNE)

list_of_files = [
    os.path.relpath(p, PROYECT_FOLDER_TEST) for p in test_files_ds.file_paths
]
print("Example test filename from dataset:", list_of_files[0])
print("Num test files:", len(list_of_files))



## === cell 15
predicted_class = pre_trained_model.predict(
    test_ds,
    verbose=1,
)



## === cell 16
predicted_class_number = np.argmax(predicted_class, axis=1)

idx_to_class = {i: c for i, c in enumerate(classes)}
classes_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Ordered classes:", classes_ordered[:5], "...", len(classes_ordered))
print("Predicted class indices sample:", predicted_class_number[:10])



## === cell 17
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

sample = pd.read_csv(sample_path)
if "file" not in sample.columns or "species" not in sample.columns:
    raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")

pred_files = [os.path.basename(f) for f in list_of_files]
pred_species = [classes_ordered[i] for i in predicted_class_number]
pred_df = pd.DataFrame({"file": pred_files, "species": pred_species})

submission = sample[["file"]].merge(pred_df, on="file", how="left")
if submission["species"].isna().any():
    missing = submission.loc[submission["species"].isna(), "file"].head(10).tolist()
    raise ValueError(f"Missing predictions for some files, e.g.: {missing}")

submission = submission[["file", "species"]]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.shape)
print(submission.head())
print("Submission columns:", submission.columns.tolist())



## === cell 18
dataFrameResults = pd.read_csv("/kaggle/working/submission.csv")
print(dataFrameResults.shape)
print(dataFrameResults.head())
print("Columns:", dataFrameResults.columns.tolist())

if list(dataFrameResults.columns) != ["file", "species"]:
    raise ValueError("Invalid submission columns; expected exactly ['file','species'].")

print("Done. Submission is ready at /kaggle/working/submission.csv")
