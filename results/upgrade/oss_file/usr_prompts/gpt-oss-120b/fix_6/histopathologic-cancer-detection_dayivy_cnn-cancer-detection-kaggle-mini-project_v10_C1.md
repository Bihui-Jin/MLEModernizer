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

3.10

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

0.7995

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

train_labels = pd.read_csv(
    "../input/histopathologic-cancer-detection/train_labels.csv", dtype=str
)
print("Train labels shape:", train_labels.shape)




## === cell 1
train_labels["label"] = train_labels["label"].astype(int)

train_labels_pos = train_labels[train_labels["label"] == 1]
train_labels_neg = train_labels[train_labels["label"] == 0].sample(
    n=train_labels_pos.shape[0], random_state=12345
)

train_labels_balanced = (
    pd.concat([train_labels_neg, train_labels_pos])
    .sample(frac=1, random_state=12345)
    .reset_index(drop=True)
)
print("Balanced dataset shape:", train_labels_balanced.shape)
print(train_labels_balanced["label"].value_counts())




## === cell 2
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    train_labels_balanced,
    test_size=0.25,
    random_state=1234,
    stratify=train_labels_balanced["label"],
)

train_df["id"] = train_df["id"] + ".tif"
valid_df["id"] = valid_df["id"] + ".tif"

train_df["label"] = train_df["label"].astype(str)
valid_df["label"] = valid_df["label"].astype(str)




## === cell 3
import tensorflow as tf
import tensorflow_io as tfio  # for TIFF decoding
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dropout,
    Dense,
    PReLU,
)
from tensorflow.keras import initializers

BATCH_SIZE = 64
TRAIN_DIR = "../input/histopathologic-cancer-detection/train/"


def _decode_tiff(path):
    img = tf.io.read_file(path)
    img = tfio.experimental.image.decode_tiff(img)  # decode .tif
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale to [0,1]
    img = tf.image.resize(img, [96, 96])
    return img


def _process_path_label(path, label):
    img = _decode_tiff(path)
    return img, tf.cast(tf.strings.to_number(label), tf.float32)


def make_dataset(df, shuffle=False):
    paths = tf.data.Dataset.from_tensor_slices(
        tf.strings.join([TRAIN_DIR, df["id"].values])
    )
    labels = tf.data.Dataset.from_tensor_slices(df["label"].values)
    ds = tf.data.Dataset.zip((paths, labels))
    ds = ds.map(_process_path_label, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, shuffle=True)
valid_dataset = make_dataset(valid_df, shuffle=False)

STEP_SIZE_TRAIN = math.ceil(train_df.shape[0] / BATCH_SIZE)
STEP_SIZE_VALID = math.ceil(valid_df.shape[0] / BATCH_SIZE)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
model4 = Sequential()
model4.add(Conv2D(32, (3, 3), padding="same", input_shape=(96, 96, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(32, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())

model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(64, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())

model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(Conv2D(128, (3, 3)))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(BatchNormalization())

model4.add(Flatten())
model4.add(Dropout(0.25))
model4.add(Dense(512))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))

model4.add(Dropout(0.25))
model4.add(Dense(256))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))

model4.add(Dropout(0.25))
model4.add(Dense(64))
model4.add(PReLU(alpha_initializer=initializers.Constant(value=0.25)))

model4.add(Dropout(0.25))
model4.add(Dense(1, activation="sigmoid"))

opt = tf.keras.optimizers.RMSprop(0.001)
model4.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])




## === cell 5
history4 = model4.fit(
    train_dataset,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_dataset,
    validation_steps=STEP_SIZE_VALID,
    epochs=5,
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/210618359.py in <cell line: 0>()
      1 history4 = model4.fit(
----> 2     train_dataset,
      3     steps_per_epoch=STEP_SIZE_TRAIN,
      4     validation_data=valid_dataset,
      5     validation_steps=STEP_SIZE_VALID,

NameError: name 'train_dataset' is not defined

## === cell 6
test_files = os.listdir("../input/histopathologic-cancer-detection/test/")
test_df = pd.DataFrame(test_files, columns=["id"])
print("Test set size:", test_df.shape[0])




## === cell 7
TEST_DIR = "../input/histopathologic-cancer-detection/test/"

test_paths = tf.data.Dataset.list_files(os.path.join(TEST_DIR, "*.tif"), shuffle=False)


def _load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tfio.experimental.image.decode_tiff(img)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale to [0,1]
    img = tf.image.resize(img, [96, 96])
    return img


test_ds = test_paths.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(256).prefetch(tf.data.AUTOTUNE)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_55/89325870.py in <cell line: 0>()
     12 
     13 
---> 14 test_ds = test_paths.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
     15 test_ds = test_ds.batch(256).prefetch(tf.data.AUTOTUNE)
     16 

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

/tmp/__autograph_generated_filenpjgraaw.py in tf___load_and_preprocess(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img),), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.convert_image_dtype, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [96, 96]), None, fscope)

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

    File "/tmp/ipykernel_55/89325870.py", line 8, in _load_and_preprocess  *
        img = tfio.experimental.image.decode_tiff(img)
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
preds = model4.predict(test_ds, verbose=1)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3591668481.py in <cell line: 0>()
----> 1 preds = model4.predict(test_ds, verbose=1)
      2 
      3 

NameError: name 'test_ds' is not defined

## === cell 9
predictions = preds.squeeze().tolist()
print("Predictions sample:", predictions[:5])




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1677432433.py in <cell line: 0>()
----> 1 predictions = preds.squeeze().tolist()
      2 print("Predictions sample:", predictions[:5])
      3 
      4 

NameError: name 'preds' is not defined

## === cell 10
submission = test_df.copy()
submission["id"] = submission["id"].str[:-4]  # remove .tif extension
submission["label"] = predictions
submission.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1157099104.py in <cell line: 0>()
      1 submission = test_df.copy()
      2 submission["id"] = submission["id"].str[:-4]  # remove .tif extension
----> 3 submission["label"] = predictions
      4 submission.head()
      5 

NameError: name 'predictions' is not defined

## === cell 11
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: Submission should have a label column
