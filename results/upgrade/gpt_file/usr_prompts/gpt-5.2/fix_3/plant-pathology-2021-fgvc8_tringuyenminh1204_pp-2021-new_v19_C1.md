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

0.1662788550323177

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, re, math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(
        True
    )  # XLA can speed up ResNet training/inference on GPU/CPU
except Exception as e:
    print("Could not enable XLA:", e)

try:
    tf.data.experimental.enable_debug_mode  # attribute exists in some TF versions
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "sample_submission.csv")
sub = pd.read_csv(path + "sample_submission.csv")

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, test.shape)
print(train.head(2))




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE




## === cell 4
import pathlib




## === cell 5
def _list_images_fast(directory):
    patterns = [
        os.path.join(directory, "**", "*.jpg"),
        os.path.join(directory, "**", "*.jpeg"),
        os.path.join(directory, "**", "*.png"),
        os.path.join(directory, "**", "*.JPG"),
        os.path.join(directory, "**", "*.JPEG"),
        os.path.join(directory, "**", "*.PNG"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(tf.io.gfile.glob(pat))
    return sorted(set(paths))


train_paths = _list_images_fast(train_images_dir)
test_paths = _list_images_fast(test_images_dir)

print("n_train_images:", len(train_paths))
print("n_test_images:", len(test_paths))
print("example test path:", test_paths[0] if test_paths else None)




## === cell 6
kind = np.unique(train["labels"])
kind




## === cell 7
all_classes = sorted(
    {c for s in train["labels"].astype(str).tolist() for c in s.split()}
)
print("classes:", all_classes)

mlb = pd.Series(train["labels"].astype(str)).str.get_dummies(sep=" ")
labels_onehot_features = mlb[all_classes].astype(np.float32)

new_train = pd.concat([train[["image"]], labels_onehot_features], axis=1).iloc[:]
new_train.head()




## === cell 8
new_train




## === cell 9
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_paths[:5]




## === cell 11
BATCH_SIZE = 64




## === cell 13
import tensorflow as tf
from tensorflow import keras




## === cell 14
IMG_SIZE = (512, 512)
N_CLASSES = len(all_classes)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = base.output
out = Dense(N_CLASSES, activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

print(model.output_shape)




## === cell 15
image_to_path = {os.path.basename(p): p for p in train_paths}

df = new_train.copy()
df["path"] = df["image"].map(image_to_path)
df = df.dropna(subset=["path"]).reset_index(drop=True)

idx = np.arange(len(df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(df))
train_idx, val_idx = idx[:split], idx[split:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)

y_cols = all_classes


def make_dataset(frame, training=True):
    paths = frame["path"].values
    labels = frame[y_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img, y = decode_image(p, y, image_size=IMG_SIZE)
        img = tf.keras.applications.resnet.preprocess_input(img * 255.0)
        return img, y

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    ds = (
        ds.map(_map_fn, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return ds


train_ds = make_dataset(df_train, training=True)
val_ds = make_dataset(df_val, training=False)

print("train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("val batches:", tf.data.experimental.cardinality(val_ds).numpy())




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/382118859.py in <cell line: 0>()
     46 
     47 
---> 48 train_ds = make_dataset(df_train, training=True)
     49 val_ds = make_dataset(df_val, training=False)
     50 

/tmp/ipykernel_11/382118859.py in make_dataset(frame, training)
     38     # cache() after map ensures we cache the fully processed tensors.
     39     ds = (
---> 40         ds.map(_map_fn, num_parallel_calls=AUTO)
     41         .cache()
     42         .batch(BATCH_SIZE)

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

/tmp/__autograph_generated_filev94r5jcz.py in tf___map_fn(p, y)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img, y = ag__.converted_call(ag__.ld(decode_image), (ag__.ld(p), ag__.ld(y)), dict(image_size=ag__.ld(IMG_SIZE)), fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).keras.applications.resnet.preprocess_input, (ag__.ld(img) * 255.0,), None, fscope)
     12                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filexglm4gtj.py in tf__decode_image(filename, label, image_size)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 bits = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(filename),), None, fscope)
     11                 image = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(bits),), dict(channels=3), fscope)
     12                 image = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image), ag__.ld(tf).float32), None, fscope) / 255.0

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    586       pass  # Add nodes to the TensorFlow graph.
    587   # Add nodes to the TensorFlow graph.
--> 588   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    589         "ReadFile", filename=filename, name=name)
    590   _result = _outputs[:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    576                   (input_name, op_type_name, observed))
    577         if input_arg.type != types_pb2.DT_INVALID:
--> 578           raise TypeError(f"{prefix} expected type of "
    579                           f"{dtypes.as_dtype(input_arg.type).name}.")
    580         else:

TypeError: in user code:

    File "/tmp/ipykernel_11/382118859.py", line 30, in _map_fn  *
        img, y = decode_image(p, y, image_size=IMG_SIZE)
    File "/tmp/ipykernel_11/3140041662.py", line 2, in decode_image  *
        bits = tf.io.read_file(filename)

    TypeError: Input 'filename' of 'ReadFile' Op has type float64 that does not match expected type of string.


## === cell 16
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1368826698.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 
      4 

NameError: name 'train_ds' is not defined

## === cell 17
def make_test_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(p):
        img = decode_image(p, label=None, image_size=IMG_SIZE)
        img = tf.keras.applications.resnet.preprocess_input(img * 255.0)
        return img

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    ds = (
        ds.map(_map_fn, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    return ds






## === cell 18
test_image_to_path = {os.path.basename(p): p for p in test_paths}
ordered_test_paths = [test_image_to_path[i] for i in test["image"].values]
ordered_test_ds = make_test_dataset(ordered_test_paths)

probs = model.predict(ordered_test_ds, verbose=1)
print("probs shape:", probs.shape)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2615875035.py in <cell line: 0>()
      2 # then predicted again in submission order). We build ordered paths once and predict once.
      3 test_image_to_path = {os.path.basename(p): p for p in test_paths}
----> 4 ordered_test_paths = [test_image_to_path[i] for i in test["image"].values]
      5 ordered_test_ds = make_test_dataset(ordered_test_paths)
      6 

/tmp/ipykernel_11/2615875035.py in <listcomp>(.0)
      2 # then predicted again in submission order). We build ordered paths once and predict once.
      3 test_image_to_path = {os.path.basename(p): p for p in test_paths}
----> 4 ordered_test_paths = [test_image_to_path[i] for i in test["image"].values]
      5 ordered_test_ds = make_test_dataset(ordered_test_paths)
      6 

KeyError: 'ca6a50c5d2adb8ae.jpg'

## === cell 19
probs[:2]




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/724140116.py in <cell line: 0>()
----> 1 probs[:2]
      2 
      3 

NameError: name 'probs' is not defined

## === cell 20
temp_probs = probs




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2953550481.py in <cell line: 0>()
----> 1 temp_probs = probs
      2 
      3 

NameError: name 'probs' is not defined

## === cell 21
temp_probs.shape




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/251966503.py in <cell line: 0>()
----> 1 temp_probs.shape
      2 
      3 

NameError: name 'temp_probs' is not defined

## === cell 22
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.27, 1: 0.5, 2: 0.3, 3: 0.5, 4: 0.5, 5: 0.5}

class_to_idx = {c: i for i, c in enumerate(all_classes)}

pred_string = []
for line in temp_probs:
    chosen = []
    for cls_name, thr in [
        ("scab", threshold[0]),
        ("frog_eye_leaf_spot", threshold[1]),
        ("complex", threshold[2]),
        ("rust", threshold[3]),
        ("powdery_mildew", threshold[4]),
    ]:
        if cls_name in class_to_idx:
            j = class_to_idx[cls_name]
            if float(line[j]) > float(thr):
                chosen.append(cls_name)

    if len(chosen) == 0:
        chosen = ["healthy"]

    pred_string.append(" ".join(chosen))

test["labels"] = pred_string

submission = test[["image", "labels"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3607528516.py in <cell line: 0>()
     13 
     14 pred_string = []
---> 15 for line in temp_probs:
     16     chosen = []
     17     for cls_name, thr in [

NameError: name 'temp_probs' is not defined
