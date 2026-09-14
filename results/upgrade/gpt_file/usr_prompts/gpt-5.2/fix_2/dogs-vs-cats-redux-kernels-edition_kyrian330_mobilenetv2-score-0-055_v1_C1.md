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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.0575566125141965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, zipfile, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers, callbacks


SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("✅ Imports OK")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

extract_dir = "/kaggle/working/"

train_marker = os.path.join(extract_dir, ".train_unzipped")
test_marker = os.path.join(extract_dir, ".test_unzipped")

if not os.path.exists(train_marker):
    with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    open(train_marker, "w").close()

if not os.path.exists(test_marker):
    with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    open(test_marker, "w").close()

print("✅ Unzip OK")



## === cell 2
IMG_SIZE = 256
BATCH_SIZE = 32

TRAIN_GLOB_CANDIDATES = [
    "/kaggle/working/train/*.jpg",
    "/kaggle/working/train/train/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train/*.jpg",
]
TEST_GLOB_CANDIDATES = [
    "/kaggle/working/test/*.jpg",
    "/kaggle/working/test/test/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/*.jpg",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/test/*.jpg",
]


def first_nonempty_glob(patterns):
    for pat in patterns:
        files = glob.glob(pat)
        if len(files) > 0:
            return pat, files
    return None, []


train_pat, all_images = first_nonempty_glob(TRAIN_GLOB_CANDIDATES)
test_pat, _test_images = first_nonempty_glob(TEST_GLOB_CANDIDATES)

print(f"Resolved train pattern: {train_pat} ({len(all_images)} files)")
print(f"Resolved test pattern:  {test_pat} ({len(_test_images)} files)")

if len(all_images) == 0:
    raise FileNotFoundError(
        "No training images found after unzip. Checked patterns:\n"
        + "\n".join(TRAIN_GLOB_CANDIDATES)
    )
if len(_test_images) == 0:
    raise FileNotFoundError(
        "No test images found after unzip. Checked patterns:\n"
        + "\n".join(TEST_GLOB_CANDIDATES)
    )

print("✅ Paths OK")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4067620043.py in <cell line: 0>()
     33 
     34 if len(all_images) == 0:
---> 35     raise FileNotFoundError(
     36         "No training images found after unzip. Checked patterns:\n"
     37         + "\n".join(TRAIN_GLOB_CANDIDATES)

FileNotFoundError: No training images found after unzip. Checked patterns:
/kaggle/working/train/*.jpg
/kaggle/working/train/train/*.jpg
/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/*.jpg
/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train/*.jpg

## === cell 3
labels = [1 if "dog" in os.path.basename(p) else 0 for p in all_images]

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, labels, test_size=0.15, stratify=labels, random_state=SEED
)


def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, tf.cast(label, tf.float32)


def build_dataset(paths, labels, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = build_dataset(train_paths, train_labels, is_train=True)
val_ds = build_dataset(val_paths, val_labels, is_train=False)

print("✅ Datasets OK")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1508310290.py in <cell line: 0>()
      2 labels = [1 if "dog" in os.path.basename(p) else 0 for p in all_images]
      3 
----> 4 train_paths, val_paths, train_labels, val_labels = train_test_split(
      5     all_images, labels, test_size=0.15, stratify=labels, random_state=SEED
      6 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])
-> 2562     n_train, n_test = _validate_shuffle_split(
   2563         n_samples, test_size, train_size, default_test_size=0.25
   2564     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
   2234 
   2235     if n_train == 0:
-> 2236         raise ValueError(
   2237             "With n_samples={}, test_size={} and train_size={}, the "
   2238             "resulting train set will be empty. Adjust any of the "

ValueError: With n_samples=0, test_size=0.15 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.

## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)

base_model.trainable = False
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="MobileNetV2")(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(2e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

print("✅ Model OK")



## === cell 5
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=1, restore_best_weights=True
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=40, callbacks=[early_stop], verbose=2
)

print("✅ Train OK")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/749910425.py in <cell line: 0>()
      4 
      5 history = model.fit(
----> 6     train_ds, validation_data=val_ds, epochs=40, callbacks=[early_stop], verbose=2
      7 )
      8 

NameError: name 'train_ds' is not defined

## === cell 6
model.save("/kaggle/working/MobileNetV2.keras")
print("✅ Saved model")



## === cell 7
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.show()

print("✅ Plots OK")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3972581181.py in <cell line: 0>()
      1 plt.figure(figsize=(12, 5))
      2 plt.subplot(1, 2, 1)
----> 3 plt.plot(history.history["accuracy"], label="Train Acc")
      4 plt.plot(history.history["val_accuracy"], label="Val Acc")
      5 plt.title("Accuracy")

NameError: name 'history' is not defined

## === cell 8
test_paths = sorted(
    _test_images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)


def build_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_img(p, 0.0)[0], num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


test_ds = build_test_ds(test_paths)

preds = model.predict(test_ds, verbose=0).ravel()
preds = preds.clip(min=0.005, max=0.995)

print("✅ Predict OK")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1069274569.py in <cell line: 0>()
     11 
     12 
---> 13 test_ds = build_test_ds(test_paths)
     14 
     15 preds = model.predict(test_ds, verbose=0).ravel()

/tmp/ipykernel_11/1069274569.py in build_test_ds(paths)
      7 def build_test_ds(paths):
      8     ds = tf.data.Dataset.from_tensor_slices(paths)
----> 9     ds = ds.map(lambda p: decode_img(p, 0.0)[0], num_parallel_calls=tf.data.AUTOTUNE)
     10     return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
     11 

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

/tmp/__autograph_generated_filehwf758it.py in <lambda>(p)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_img, (p, 0.0), None, lscope)[0], 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filehwf758it.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_img, (p, 0.0), None, lscope)[0], 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

NameError: in user code:

    File "/tmp/ipykernel_11/1069274569.py", line 9, in None  *
        lambda p: decode_img(p, 0.0)[0]

    NameError: name 'decode_img' is not defined


## === cell 9
print(f"预测最大值：{preds.max():.4f}")
print(f"预测最小值：{preds.min():.4f}")
print(f"预测均值：{preds.mean():.4f}")
plt.hist(preds, bins=100)
plt.title("Test Prediction Distribution")
plt.show()

print("✅ Pred stats OK")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2630027730.py in <cell line: 0>()
----> 1 print(f"预测最大值：{preds.max():.4f}")
      2 print(f"预测最小值：{preds.min():.4f}")
      3 print(f"预测均值：{preds.mean():.4f}")
      4 plt.hist(preds, bins=100)
      5 plt.title("Test Prediction Distribution")

NameError: name 'preds' is not defined

## === cell 10
submission = pd.DataFrame(
    {
        "id": [int(os.path.splitext(os.path.basename(p))[0]) for p in test_paths],
        "label": preds,
    }
)

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"✅ Wrote submission: {out_path}  shape={submission.shape}")
print(submission.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2412338786.py in <cell line: 0>()
      3     {
      4         "id": [int(os.path.splitext(os.path.basename(p))[0]) for p in test_paths],
----> 5         "label": preds,
      6     }
      7 )

NameError: name 'preds' is not defined

## === cell 11
import shutil


def safe_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Skip removing {path}, reason: {e}")
    else:
        print(f"Not found, skip: {path}")


safe_rmtree("/kaggle/working/train")
safe_rmtree("/kaggle/working/test")

print("✅ Cleanup OK")
