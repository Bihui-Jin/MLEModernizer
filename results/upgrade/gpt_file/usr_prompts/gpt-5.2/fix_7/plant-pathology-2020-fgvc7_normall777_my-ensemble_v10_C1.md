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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9728760982073276

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, math, random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

print("Tensorflow version " + tf.__version__)

tf.random.set_seed(2020)
np.random.seed(2020)
random.seed(2020)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

tf.config.run_functions_eagerly(False)
try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

try:
    from tensorflow.keras import mixed_precision

    if tpu is not None:
        mixed_precision.set_global_policy("mixed_bfloat16")
    else:
        mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())
except Exception as e:
    print("Mixed precision not enabled:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def format_path(st):
    return os.path.join(IMAGES_DIR, st + ".jpg")


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET_COLS = [c for c in sub.columns if c != "image_id"]

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train[TARGET_COLS].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020
)

print("Train:", len(train_paths), "Valid:", len(valid_paths), "Test:", len(test_paths))
print("Targets:", TARGET_COLS)




## === cell 2
img_size = 768


@tf.function
def decode_image(filename, label, image_size):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image.set_shape([None, None, 3])
    image = tf.ensure_shape(image, [image_size[0], image_size[1], 3])
    return image, label


@tf.function
def decode_image_nolabel(filename, image_size):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image.set_shape([None, None, 3])
    image = tf.ensure_shape(image, [image_size[0], image_size[1], 3])
    return image


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image, seed=2020)
    image = tf.image.random_flip_up_down(image, seed=2020)
    return image, label




## === cell 3
def _dataset_options_fast():
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return opts


def _maybe_prefetch_to_device(ds):
    if tpu is not None:
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/device:TPU:0"))
        except Exception:
            pass
    return ds


CACHE_DIR = "/kaggle/working/tf_cache_pp2020"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, f"train_decoded_resized_{img_size}.cache")
VALID_CACHE = os.path.join(CACHE_DIR, f"valid_decoded_resized_{img_size}.cache")
TEST_CACHE = os.path.join(CACHE_DIR, f"test_decoded_resized_{img_size}.cache")

IMAGE_SIZE_T = tf.constant([img_size, img_size], dtype=tf.int32)


@tf.function
def _decode_train(path, label):
    return decode_image(path, label, IMAGE_SIZE_T)


@tf.function
def _decode_valid(path, label):
    return decode_image(path, label, IMAGE_SIZE_T)


@tf.function
def _decode_test(path):
    return decode_image_nolabel(path, IMAGE_SIZE_T)


@tf.function
def _augment(image, label):
    return data_augment(image, label)


train_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(_dataset_options_fast())
    .map(_decode_train, num_parallel_calls=AUTO)
    .cache(TRAIN_CACHE)
)

train_dataset = (
    train_base.shuffle(512, seed=2020, reshuffle_each_iteration=True)
    .map(_augment, num_parallel_calls=AUTO)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)
train_dataset = _maybe_prefetch_to_device(train_dataset)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .with_options(_dataset_options_fast())
    .map(_decode_valid, num_parallel_calls=AUTO)
    .cache(VALID_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
valid_dataset = _maybe_prefetch_to_device(valid_dataset)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(_dataset_options_fast())
    .map(_decode_test, num_parallel_calls=AUTO)
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
test_dataset = _maybe_prefetch_to_device(test_dataset)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3555567766.py in <cell line: 0>()
     52     tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
     53     .with_options(_dataset_options_fast())
---> 54     .map(_decode_train, num_parallel_calls=AUTO)
     55     .cache(TRAIN_CACHE)
     56 )

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file2tswxvwq.py in tf___decode_train(path, label)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf___decode_train

/tmp/__autograph_generated_filemxhklq41.py in tf__decode_image(filename, label, image_size)
     13                 image = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), ag__.ld(image_size)), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR, antialias=False), fscope)
     14                 ag__.converted_call(ag__.ld(image).set_shape, ([None, None, 3],), None, fscope)
---> 15                 image = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(image), [ag__.ld(image_size)[0], ag__.ld(image_size)[1], 3]), None, fscope)
     16                 try:
     17                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_11/3555567766.py", line 32, in _decode_train  *
        return decode_image(path, label, IMAGE_SIZE_T)
    File "/tmp/ipykernel_11/490237477.py", line 16, in decode_image  *
        image = tf.ensure_shape(image, [image_size[0], image_size[1], 3])

    TypeError: Dimension value must be integer or None or have an __index__ method, got value '<tf.Tensor 'strided_slice:0' shape=() dtype=int32>' with type '<class 'tensorflow.python.framework.ops.SymbolicTensor'>'


## === cell 4
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import DenseNet201


def get_model(use_model):
    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    outputs = Dense(train_labels.shape[1], activation="sigmoid", dtype="float32")(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[],
        run_eagerly=False,
        steps_per_execution=16,
    )
    return model


with strategy.scope():
    model1 = get_model(EfficientNetB7)
with strategy.scope():
    model2 = get_model(DenseNet201)




## === cell 5
steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
validation_steps = math.ceil(len(valid_paths) / BATCH_SIZE)

history1 = model1.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    verbose=2,
)

history2 = model2.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1022240407.py in <cell line: 0>()
      3 
      4 history1 = model1.fit(
----> 5     train_dataset,
      6     epochs=EPOCHS,
      7     steps_per_epoch=steps_per_epoch,

NameError: name 'train_dataset' is not defined

## === cell 6
best_alpha = 0.75
bad_alpha = 0.30

print("Вычисляем предсказания...")
probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2
probabilities = np.clip(probabilities, 0.0, 1.0)

sub[TARGET_COLS] = probabilities
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/54881495.py in <cell line: 0>()
      3 
      4 print("Вычисляем предсказания...")
----> 5 probabilities1 = model1.predict(test_dataset, verbose=1)
      6 probabilities2 = model2.predict(test_dataset, verbose=1)
      7 

NameError: name 'test_dataset' is not defined
