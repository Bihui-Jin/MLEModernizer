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

3.8

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

0.7953659417082056

# 6. Current score

0.43316

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43635) has done: 'The timeout is dominated by Python-side augmentation inside `tf.data` (`tf.numpy_function` + `ImageDataGenerator.random_transform`), which prevents graph optimizations and keeps the input pipeline on the CPU with high per-step overhead. To preserve the exact training semantics while making it fast, I keep the same augmentation logic but switch to Keras’ built-in `ImageDataGenerator.flow(...)`, which performs the same transforms in optimized C/NumPy code and feeds batches directly to `model.fit` without `tf.numpy_function`. I also remove expensive “shuffle the entire dataset buffer” in `tf.data` and avoid repeated dtype conversions by filling arrays as `float32` once and scaling in-place. These changes keep the same model, loss, metrics, callbacks, and augmentation parameters, but drastically reduce per-epoch overhead so the run fits within 600 seconds.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by (1) slow Python-loop image loading/resizing with PIL and (2) long training caused by feeding a fully materialized NumPy array through `ImageDataGenerator`. To preserve the exact model and training semantics, I keep the same architecture, optimizer, loss, and augmentation policy, but move image decoding/resizing into an efficient, parallel `tf.data` pipeline and keep using the same `ImageDataGenerator` transforms via `tf.numpy_function` (so the augmentation behavior remains the same). I also remove non-essential display cells, add caching/prefetching, and switch prediction to a streaming dataset so we never hold all test images in RAM. These changes reduce wall time substantially without changing what the model learns (only negligible float-level differences are possible).'
- What this solution (achieved 0.43316) has done: 'I fix the two root-cause runtime issues preventing any training/inference: (1) TensorFlow import crashing due to an incompatible protobuf version, and (2) `np.char.add` failing because `image_id` arrays are `object` dtype. The safest minimal fix for (1) in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow; this avoids the `MessageFactory.GetPrototype` crash without changing model logic. For (2) I build image paths using plain Python string concatenation on `astype(str)` IDs, which is stable across NumPy versions. With those fixes, the pipeline should run end-to-end and write a valid `submission.csv` in the required format; score should improve from 0.5 simply because the model actually train and predict properly.'
- What this solution (achieved 0.43316) has done: 'The timeout is dominated by the input pipeline: every training step calls `tf.numpy_function` which executes Python/Numpy augmentation (`ImageDataGenerator.random_transform`) and reseeds global RNG, preventing TensorFlow graph optimizations and making mapping single-thread/CPU-bound. To preserve the same model and training loop semantics, I keep the architecture, optimizer/loss, epochs, callbacks, and augmentation *concept*, but replace the Python-based `ImageDataGenerator` augmentation with an equivalent pure-TensorFlow augmentation pipeline (same transform types and ranges), fully vectorized and parallelizable in `tf.data`. I also add safe dataset options (deterministic, ignore errors) and enable caching of decoded+resized images (not augmented images) to avoid repeated JPEG decode/resize across epochs. These changes are provably equivalent at the level of logic (decode→resize→normalize→random augment→batch→fit) while removing the main Python bottleneck so training completes within the 600s budget.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["PYTHONHASHSEED"] = "0"

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import random

random.seed(0)
np.random.seed(0)

import tensorflow as tf

tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")




## === cell 2
_ = train.head()




## === cell 3
_ = train.loc[:, "healthy":"scab"].sum(axis=0) / (train.shape[0] / 100)




## === cell 4
_ = test.head()




## === cell 5
_ = submission.head()




## === cell 6
from tensorflow.keras.utils import load_img, img_to_array


def load_resize_image_float32(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # PIL under the hood
    arr = img_to_array(img)  # float32, shape (H,W,3)
    return arr


IMG_SIZE = (224, 224)
IMG_DIR = "/kaggle/input/plant-pathology-2020-fgvc7/images"

train_ids = train["image_id"].astype(str).values
test_ids = test["image_id"].astype(str).values

train_paths = np.array([f"{IMG_DIR}/{iid}.jpg" for iid in train_ids], dtype=object)
test_paths = np.array([f"{IMG_DIR}/{iid}.jpg" for iid in test_ids], dtype=object)

missing = [
    p
    for p in (train_paths[:5].tolist() + test_paths[:5].tolist())
    if not os.path.exists(p)
]
if missing:
    raise FileNotFoundError(f"Could not find image(s), e.g.: {missing[0]}")


def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img




## === cell 7
train_label = train.loc[:, "healthy":"scab"]




## === cell 8
train_label = np.asarray(train_label, dtype=np.float32)
train_label = train_label / np.clip(train_label.sum(axis=1, keepdims=True), 1.0, None)




## === cell 9
print("n_train:", len(train_paths))
print("n_test :", len(test_paths))
print("train_label shape:", train_label.shape)




## === cell 10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
)




## === cell 11
pass




## === cell 12
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint




## === cell 13
base_model = DenseNet201(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)

model = Sequential()
model.add(base_model)

model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation="softmax"))

froze = True  # keep original behavior
if froze is True:
    base_model.trainable = False
else:
    for layer in base_model.layers[: -int(froze)]:
        layer.trainable = False

reduce_learning_rate = ReduceLROnPlateau(
    monitor="categorical_accuracy",
    factor=0.1,
    patience=2,
    cooldown=2,
    min_lr=0.0000001,
    verbose=1,
)
early_stopping = EarlyStopping(monitor="categorical_accuracy", patience=5)

check_point = ModelCheckpoint(
    filepath="resnet_50.h5", monitor="categorical_accuracy", save_best_only=True
)

callbacks = [reduce_learning_rate, early_stopping, check_point]

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)




## === cell 14
model.summary()




## === cell 15

BATCH_SIZE = 32
steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))


def _stateless_rand_uniform(shape, seed, minval, maxval, dtype=tf.float32):
    return tf.random.stateless_uniform(
        shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype
    )


def _augment_tf(idx, img, label):
    idx = tf.cast(idx, tf.int32)
    base_seed = tf.stack([0, idx])  # (2,)

    flip_p = _stateless_rand_uniform(
        [], base_seed + tf.constant([1, 0], tf.int32), 0.0, 1.0
    )
    img = tf.cond(flip_p < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    angle_deg = _stateless_rand_uniform(
        [], base_seed + tf.constant([2, 0], tf.int32), -15.0, 15.0
    )
    angle_rad = angle_deg * (np.pi / 180.0)
    img = tf.image.rotate(
        img, angles=angle_rad, fill_mode="reflect", interpolation="bilinear"
    )

    zoom = _stateless_rand_uniform(
        [], base_seed + tf.constant([3, 0], tf.int32), 0.75, 1.25
    )
    new_h = tf.cast(tf.round(zoom * IMG_SIZE[0]), tf.int32)
    new_w = tf.cast(tf.round(zoom * IMG_SIZE[1]), tf.int32)
    img_zoom = tf.image.resize(img, [new_h, new_w], method="bilinear")
    img = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE[0], IMG_SIZE[1])

    max_dx = int(round(0.2 * IMG_SIZE[1]))
    max_dy = int(round(0.2 * IMG_SIZE[0]))
    dx = tf.cast(
        tf.round(
            _stateless_rand_uniform(
                [], base_seed + tf.constant([4, 0], tf.int32), -max_dx, max_dx
            )
        ),
        tf.int32,
    )
    dy = tf.cast(
        tf.round(
            _stateless_rand_uniform(
                [], base_seed + tf.constant([5, 0], tf.int32), -max_dy, max_dy
            )
        ),
        tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, label


options = tf.data.Options()
options.experimental_deterministic = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_label))
    .shuffle(buffer_size=len(train_paths), seed=0, reshuffle_each_iteration=True)
    .map(lambda p, y: (_decode_resize_normalize(p), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .enumerate()
    .map(lambda i, xy: _augment_tf(i, xy[0], xy[1]), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

start = dt.now()
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    callbacks=callbacks,
    verbose=1,
)
print(
    "Время работы модели: {}. Количество эпох: {}.".format(
        dt.now() - start, len(history.epoch)
    )
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/964379674.py in <cell line: 0>()
     84     .cache()
     85     .enumerate()
---> 86     .map(lambda i, xy: _augment_tf(i, xy[0], xy[1]), num_parallel_calls=AUTOTUNE)
     87     .batch(BATCH_SIZE, drop_remainder=False)
     88     .prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_fileu7zu4_fv.py in <lambda>(i, xy)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda i, xy: ag__.with_function_scope(lambda lscope: ag__.converted_call(_augment_tf, (i, xy[0], xy[1]), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileu7zu4_fv.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda i, xy: ag__.with_function_scope(lambda lscope: ag__.converted_call(_augment_tf, (i, xy[0], xy[1]), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_fileehb0sin5.py in tf___augment_tf(idx, img, label)
     14                 angle_deg = ag__.converted_call(ag__.ld(_stateless_rand_uniform), ([], ag__.ld(base_seed) + ag__.converted_call(ag__.ld(tf).constant, ([2, 0], ag__.ld(tf).int32), None, fscope), -15.0, 15.0), None, fscope)
     15                 angle_rad = ag__.ld(angle_deg) * (ag__.ld(np).pi / 180.0)
---> 16                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle_rad), fill_mode='reflect', interpolation='bilinear'), fscope)
     17                 zoom = ag__.converted_call(ag__.ld(_stateless_rand_uniform), ([], ag__.ld(base_seed) + ag__.converted_call(ag__.ld(tf).constant, ([3, 0], ag__.ld(tf).int32), None, fscope), 0.75, 1.25), None, fscope)
     18                 new_h = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.ld(zoom) * ag__.ld(IMG_SIZE)[0],), None, fscope), ag__.ld(tf).int32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/964379674.py", line 86, in None  *
        lambda i, xy: _augment_tf(i, xy[0], xy[1])
    File "/tmp/ipykernel_11/964379674.py", line 37, in _augment_tf  *
        img = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 16
import gc

del train_label
gc.collect()




## === cell 17
def plot_loss(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history["loss"], label="train_loss")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.close()


def plot_acc(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, epoch),
        his.history.get("categorical_accuracy", []),
        label="categorical_accuracy",
    )
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.close()




## === cell 18
if "history" in globals():
    plot_loss(history, "Training Dataset")
    plot_acc(history, "Training Dataset")




## === cell 19
ds_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    .batch(64)
    .prefetch(AUTOTUNE)
)
y_pred = model.predict(ds_test, verbose=1)
print(y_pred)




## === cell 20
target_cols = [c for c in submission.columns if c != "image_id"]
if y_pred.shape[1] != len(target_cols):
    raise ValueError(
        f"Prediction shape {y_pred.shape} does not match submission targets {len(target_cols)}: {target_cols}"
    )
submission.loc[:, target_cols] = y_pred




## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
