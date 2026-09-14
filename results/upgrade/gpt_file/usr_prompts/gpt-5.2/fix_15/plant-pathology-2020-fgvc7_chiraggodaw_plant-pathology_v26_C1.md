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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.83563

# 6. Current score

0.44266

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49742) has done: 'I fix the runtime/import error by removing incompatible standalone `keras` imports and using `tf.keras` consistently (this avoids the protobuf-related crash in your environment). I fix OpenCV headless errors by removing `cv2.destroyAllWindows()` and also fix broken variables caused by earlier cell failures. I make the training/prediction pipeline produce a valid submission by reading `test.csv` for the correct 183 image_ids, loading those exact files, and writing `submission.csv` with the required column order. Finally, I switch deprecated `fit_generator` to `fit` and ensure labels are correctly one-hot encoded for 4 classes so training runs end-to-end.'
- What this solution (achieved 0.44266) has done: 'The timeout is almost certainly dominated by slow input pipeline (JPEG decode/resize done repeatedly while training) plus expensive per-step augmentation and XLA compile overhead; we keep the exact same model, preprocessing, augmentation, and training semantics while reducing wasted work. The main speedups are: cache decoded+resized tensors once to disk, then cache augmented training batches to RAM per-epoch (augmentation remains stochastic per-epoch due to reshuffle) and move batching earlier to reduce map-call overhead; also fix an out-of-range validation slice (equivalent data, but avoids extra/empty steps issues). Finally, we remove redundant prefetch layers and use `tf.io.gfile.exists` checks to avoid re-creating caches if already present in the working directory.'

# 9. Code solution

## === cell 0
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

INPUT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(INPUT_DIR, "images")

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
IMG_W, IMG_H = (410, 273)  # keep identical resize target (width=410, height=273)

CACHE_DIR = "/kaggle/working/tf_cache_pp2020"
os.makedirs(CACHE_DIR, exist_ok=True)

tf.get_logger().setLevel("ERROR")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gc.collect()

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(c in train_df.columns for c in ["image_id"] + target_cols)
assert list(sample_sub.columns) == ["image_id"] + target_cols

train_x_images = train_df["image_id"].values
train_y_multi = train_df[target_cols].values.astype(np.int32)
training_y = np.argmax(train_y_multi, axis=1).astype(np.int32)




## === cell 2
gc.collect()

train_ids_all = train_x_images  # shuffled already
train_labels_all = training_y

split_idx = 1120
train_ids = train_ids_all[:split_idx]
train_y = train_labels_all[:split_idx]

val_ids = train_ids_all[split_idx:]
val_y = train_labels_all[split_idx:]

y_binary_train = to_categorical(train_y, num_classes=4).astype(np.float32)
y_binary_val = to_categorical(val_y, num_classes=4).astype(np.float32)


def _build_paths(ids_np):
    ids_np = ids_np.astype(str)
    return np.char.add(np.char.add(IMG_DIR + os.sep, ids_np), ".jpg")


train_paths = _build_paths(train_ids)
val_paths = _build_paths(val_ids)


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess(path, y):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA, antialias=True
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # identical preprocessing
    img.set_shape([IMG_H, IMG_W, 3])
    return img, y




## === cell 3
gc.collect()

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=1.0, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=42
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=42,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    ],
    name="augmenter",
)


@tf.function(reduce_retracing=True)
def _augment_only(img, y):
    img = augmenter(img, training=True)
    img.set_shape([IMG_H, IMG_W, 3])
    return img, y


train_opts = tf.data.Options()
train_opts.deterministic = True
train_opts.experimental_optimization.apply_default_optimizations = True
train_opts.experimental_slack = True

val_opts = tf.data.Options()
val_opts.deterministic = True
val_opts.experimental_optimization.apply_default_optimizations = True
val_opts.experimental_slack = True

train_cache_path = os.path.join(CACHE_DIR, "train_decoded.cache")
val_cache_path = os.path.join(CACHE_DIR, "val_decoded.cache")

train_decoded = (
    tf.data.Dataset.from_tensor_slices((train_paths, y_binary_train))
    .with_options(train_opts)
    .shuffle(buffer_size=len(train_paths), seed=42, reshuffle_each_iteration=True)
    .map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
    .cache(train_cache_path)
)

train_ds = (
    train_decoded.batch(BATCH_SIZE, drop_remainder=False)
    .map(_augment_only, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, y_binary_val))
    .with_options(val_opts)
    .map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
    .cache(val_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2164533662.py in <cell line: 0>()
     52 train_ds = (
     53     train_decoded.batch(BATCH_SIZE, drop_remainder=False)
---> 54     .map(_augment_only, num_parallel_calls=AUTOTUNE)
     55     # Speed: cache augmented batches in RAM per-epoch iteration order (keeps stochasticity due to reshuffle).
     56     # This avoids re-running augmentation within the same epoch when Keras may re-iterate due to steps_per_epoch.

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

/tmp/__autograph_generated_fileh9i69daa.py in tf___augment_only(img, y)
     10                 retval_ = ag__.UndefinedReturnValue()
     11                 img = ag__.converted_call(ag__.ld(augmenter), (ag__.ld(img),), dict(training=True), fscope)
---> 12                 ag__.converted_call(ag__.ld(img).set_shape, ([ag__.ld(IMG_H), ag__.ld(IMG_W), 3],), None, fscope)
     13                 try:
     14                     do_return = True

ValueError: in user code:

    File "/tmp/ipykernel_11/2164533662.py", line 24, in _augment_only  *
        img.set_shape([IMG_H, IMG_W, 3])

    ValueError: Shapes must be equal rank, but are 4 and 3


## === cell 4
gc.collect()

model = tf.keras.Sequential(
    [
        tf.keras.applications.Xception(
            weights="imagenet", include_top=False, input_shape=(IMG_H, IMG_W, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax),
    ]
)

model.layers[0].trainable = False

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

spe = 64 if steps_per_epoch >= 64 else steps_per_epoch

model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
    steps_per_execution=spe,
)

annealer = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=5, verbose=1, min_lr=1e-5
)

checkpoint = ModelCheckpoint(
    "model.h5", verbose=1, save_best_only=True, monitor="val_accuracy", mode="max"
)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("val_accuracy", 0) > 0.99:
            print("\nReached 99% validation accuracy so cancelling training!")
            self.model.stop_training = True




## === cell 5
gc.collect()

MAX_EPOCHS_CAP = 40

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS_CAP,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[annealer, checkpoint, myCallback()],
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2927146462.py in <cell line: 0>()
      4 
      5 history = model.fit(
----> 6     train_ds,
      7     validation_data=val_ds,
      8     epochs=MAX_EPOCHS_CAP,

NameError: name 'train_ds' is not defined

## === cell 6
print("Skipping plots for performance.")




## === cell 7
print("Skipping loss plot for performance.")




## === cell 8
gc.collect()

test_ids = test_df["image_id"].values
test_paths = _build_paths(test_ids)


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess_x(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA, antialias=True
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([IMG_H, IMG_W, 3])
    return img


test_opts = tf.data.Options()
test_opts.deterministic = True
test_opts.experimental_optimization.apply_default_optimizations = True
test_opts.experimental_slack = True

test_cache_path = os.path.join(CACHE_DIR, "test_decoded.cache")

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(test_opts)
    .map(_decode_resize_preprocess_x, num_parallel_calls=AUTOTUNE)
    .cache(test_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 9
gc.collect()

results = model.predict(test_ds, verbose=1)




## === cell 10
df = pd.DataFrame(results, columns=target_cols)
df.insert(0, "image_id", test_ids)
df = df[["image_id"] + target_cols]




## === cell 11
out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path} with shape {df.shape}")




## === cell 12
df.head(10)




## === cell 13
assert (
    df.shape[0] == sample_sub.shape[0]
), f"Row mismatch: {df.shape[0]} vs {sample_sub.shape[0]}"
assert list(df.columns) == list(
    sample_sub.columns
), f"Column mismatch: {df.columns} vs {sample_sub.columns}"
df.describe(include="all")
