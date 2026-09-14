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

0.1578947368421052

# 6. Current score

0.28627

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28627) has done: 'Your code likely fails to yield a Kaggle score because the training generator uses a “dummy index label” (`__idx__`) but `flow_from_dataframe(..., class_mode="raw")` returns that column as float values (and can be out-of-sync when shuffled), which can lead to incorrect/invalid indexing into `y_array` (or silently wrong labels), producing a broken model and sometimes runtime errors. I make the smallest change that preserves your architecture/training loop: store the true global row index in `__idx__`, make it an integer label via `class_mode="sparse"`, and disable shuffling at the Keras generator level while keeping randomness via the DataFrame shuffle. This keeps the same core logic (same CNN, same loss, same epochs) but fixes label/target alignment so training is meaningful and a valid submission is reliably created. I also keep the submission writing exactly as required (`submission.csv` with `image,labels`).'
- What this solution (achieved 0.28627) has done: 'The timeout is dominated by slow Python-based image augmentation/loading in `ImageDataGenerator.flow_from_dataframe` plus per-batch Python `next()`/`yield` overhead in the wrapper; this keeps the GPU/CPU underutilized and burns wall time. I keep the same train/val split, model, loss, optimizer, and epochs, but replace the generators with an equivalent `tf.data` pipeline that performs the same decoding, resizing, rescaling, and augmentations using TensorFlow ops, with parallel mapping, caching, and prefetching. I also remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it forces a much slower protobuf backend) while preserving determinism via explicit seeds and deterministic dataset options. Finally, I speed up submission label formatting by vectorizing the thresholding and top-1 fallback without changing prediction semantics.'

# 9. Code solution

## === cell 0
import os


os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB_CSV)

print(train.shape, submissions.shape)
train.head()




## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

print("Num classes:", len(classes))
print("Classes:", classes[:20], "..." if len(classes) > 20 else "")




## === cell 3
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_df = train.iloc[tr_idx].reset_index(drop=True).copy()
val_df = train.iloc[va_idx].reset_index(drop=True).copy()

train_y = y[tr_idx]
val_y = y[va_idx]

train_df["__idx__"] = np.arange(len(train_df), dtype=np.int32)
val_df["__idx__"] = np.arange(len(val_df), dtype=np.int32)

train_df.head()




## === cell 4
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

AUTO = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True

train_paths = (TRAIN_IMG_DIR + "/" + train_df["image"].values).astype(str)
val_paths = (TRAIN_IMG_DIR + "/" + val_df["image"].values).astype(str)

train_indices = train_df["__idx__"].values.astype(np.int32)
val_indices = val_df["__idx__"].values.astype(np.int32)

train_y_tf = tf.constant(train_y.astype(np.float32))
val_y_tf = tf.constant(val_y.astype(np.float32))


def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # dataset is jpg
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    angle = tf.random.stateless_uniform(
        shape=[],
        seed=[SEED, 1],
        minval=-10.0 * np.pi / 180.0,
        maxval=10.0 * np.pi / 180.0,
        dtype=tf.float32,
    )
    img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    zoom = tf.random.stateless_uniform(
        shape=[], seed=[SEED, 2], minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * zoom), tf.int32)
    img_zoom = tf.image.resize(
        img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR
    )

    img = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE[0], IMG_SIZE[1])

    max_dx = int(round(0.05 * IMG_SIZE[1]))
    max_dy = int(round(0.05 * IMG_SIZE[0]))
    dx = tf.random.stateless_uniform(
        shape=[], seed=[SEED, 3], minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        shape=[], seed=[SEED, 4], minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    return img


def _make_ds(paths, indices, y_tf, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, indices))
    ds = ds.with_options(options)

    def _load(path, idx):
        img = _decode_resize_rescale(path)
        if training:
            img = _augment(img)
        label = tf.gather(y_tf, idx)
        return img, label

    ds = ds.map(_load, num_parallel_calls=AUTO)
    ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = _make_ds(train_paths, train_indices, train_y_tf, training=True)
val_ds = _make_ds(val_paths, val_indices, val_y_tf, training=False)

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

steps_per_epoch, val_steps




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3557769485.py in <cell line: 0>()
     96 
     97 
---> 98 train_ds = _make_ds(train_paths, train_indices, train_y_tf, training=True)
     99 val_ds = _make_ds(val_paths, val_indices, val_y_tf, training=False)
    100 

/tmp/ipykernel_11/3557769485.py in _make_ds(paths, indices, y_tf, training)
     86 
     87     # Parallel map + cache decoded/resized images to reduce repeated JPEG decode across epochs
---> 88     ds = ds.map(_load, num_parallel_calls=AUTO)
     89     ds = ds.cache()
     90 

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

/tmp/__autograph_generated_fileh3vn6gz9.py in tf___load(path, idx)
     26                     nonlocal img
     27                     pass
---> 28                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     29                 label = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(y_tf), ag__.ld(idx)), None, fscope)
     30                 try:

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

/tmp/__autograph_generated_fileh3vn6gz9.py in if_body()
     21                 def if_body():
     22                     nonlocal img
---> 23                     img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img),), None, fscope)
     24 
     25                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filej_gqm7go.py in tf___augment(img)
     10                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(SEED)), fscope)
     11                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=[ag__.ld(SEED), 1], minval=-10.0 * ag__.ld(np).pi / 180.0, maxval=10.0 * ag__.ld(np).pi / 180.0, dtype=ag__.ld(tf).float32), fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)
     13                 zoom = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=[ag__.ld(SEED), 2], minval=0.9, maxval=1.1, dtype=ag__.ld(tf).float32), fscope)
     14                 new_h = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE)[0], ag__.ld(tf).float32), None, fscope) * ag__.ld(zoom),), None, fscope), ag__.ld(tf).int32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/3557769485.py", line 83, in _load  *
        img = _augment(img)
    File "/tmp/ipykernel_11/3557769485.py", line 47, in _augment  *
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 5
num_classes = len(classes)

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 6
EPOCHS = 3  # keep identical budget/loop

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1707448098.py in <cell line: 0>()
      3 
      4 history = model.fit(
----> 5     train_ds,
      6     validation_data=val_ds,
      7     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 7
test_paths = (TEST_IMG_DIR + "/" + submissions["image"].values).astype(str)


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(options)
    ds = ds.map(_decode_resize_rescale, num_parallel_calls=AUTO)
    ds = ds.batch(32, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = _make_test_ds(test_paths)

preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)




## === cell 8
thresh = 0.5

preds_np = np.asarray(preds)
mask = preds_np >= thresh
top1 = np.argmax(preds_np, axis=1)

classes_arr = np.array(classes, dtype=object)

pred_labels = []
for i in range(preds_np.shape[0]):
    chosen = classes_arr[mask[i]]
    if chosen.size == 0:
        chosen = np.array([classes_arr[top1[i]]], dtype=object)
    pred_labels.append(" ".join(chosen.tolist()))

submissions["labels"] = pred_labels
submissions.to_csv("submission.csv", index=False)

submissions.head()




## === cell 9
assert os.path.exists("submission.csv"), "submission.csv was not created"
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image", "labels"], "Submission columns mismatch"
assert len(check) == len(submissions), "Submission row count mismatch"

print("Wrote submission.csv with", len(check), "rows")
print(check.head())
