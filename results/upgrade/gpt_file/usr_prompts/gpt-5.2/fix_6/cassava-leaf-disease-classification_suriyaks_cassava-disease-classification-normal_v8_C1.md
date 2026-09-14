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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.0959

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the first error by removing the out-of-bounds loop and replacing labels using a safe vectorized map (while keeping labels numeric, which `flow_from_dataframe` expects for categorical class mode). I also fix the Keras/TensorFlow import/protobuf issues by consistently using `tf.keras` APIs (and not standalone `keras.preprocessing`, which is removed in Keras 3), so the generators, model, and prediction code work in this environment. The TFHub section fails due to Keras 3 incompatibility and internet access; I keep the core training/inference logic with EfficientNetB7 (already in your code) and make sure it actually trains (uncomment fit) so the pipeline produces a valid `submission.csv`. These changes are directly aimed at making the notebook run end-to-end and produce a proper submission; score may improve vs. the currently broken training flow.'
- What this solution (achieved 0.61099) has done: 'The timeout is dominated by (1) extremely heavy training compute/IO from EfficientNetB7 at 512×512 with a Python `ImageDataGenerator`, and (2) per-image test inference in a Python loop (2676 separate `model.predict` calls). To keep the same core model/training semantics, I switch data input to a `tf.data` pipeline with parallel decode/resize/prefetch (same rescale + flips + split) and I keep the same model and epochs/callbacks. For inference, I replace the per-image loop with a batched `tf.data` pipeline and a single `model.predict` over the full dataset, which is exactly equivalent but dramatically faster. I also remove unnecessary directory listings/printing and force protobuf to use the default (faster) C++ implementation.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

train = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
_ = path  # keep variable to preserve notebook structure without extra I/O




## === cell 2
_ = train.head()




## === cell 3
_ = (train.shape, train.dtypes.to_dict())




## === cell 4
_ = train["label"].unique()




## === cell 5
train_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"




## === cell 6
file = open(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
json_data = json.load(file)
file.close()




## === cell 7
train["label"] = train["label"].astype(np.int32)

label_map = {int(k): v for k, v in json_data.items()}
_ = train.head()




## === cell 8
size = 512
bat_size = 16
split = 0.33
epoch = 5

AUTOTUNE = tf.data.AUTOTUNE

rng = np.random.RandomState(42)
perm = rng.permutation(len(train))
val_size = int(np.floor(len(train) * split))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)


_resizer = keras.layers.Resizing(
    size, size, interpolation="nearest", crop_to_aspect_ratio=False, name="resize_nn"
)
_rescaler = keras.layers.Rescaling(1.0 / 255.0, name="rescale_255")


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = _resizer(img)
    img = _rescaler(img)
    return img


def _parse_train_example(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TRAIN)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), depth=5, dtype=tf.float32)
    return img, label


def _parse_test_example(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TEST)
    img = _decode_resize_from_bytes(ex["image"])
    return img, ex["image_name"]


train_ids = train["image_id"].values
id_to_pos = {img_id: i for i, img_id in enumerate(train_ids)}


def _df_to_shard_indices(df):
    pos = np.fromiter(
        (id_to_pos[x] for x in df["image_id"].values), dtype=np.int64, count=len(df)
    )
    shard = (pos // 1338).astype(np.int32)
    within = (pos % 1338).astype(np.int32)
    return shard, within


train_shard, train_within = _df_to_shard_indices(train_df)
val_shard, val_within = _df_to_shard_indices(val_df)


def _make_train_tfrecord_files_for_df(df_shards):
    shards = np.unique(df_shards).tolist()
    return [
        f"/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/ld_train{s:02d}-1338.tfrec"
        for s in shards
    ]


def _make_ds_from_tfrecords(df, df_shards, df_within, training: bool):
    files = _make_train_tfrecord_files_for_df(df_shards)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)

    shard_list = np.unique(df_shards)
    shard_to_order = {int(s): i for i, s in enumerate(shard_list.tolist())}
    global_keep = np.array(
        [shard_to_order[int(s)] * 1338 + int(w) for s, w in zip(df_shards, df_within)],
        dtype=np.int64,
    )
    global_keep_tf = tf.constant(np.sort(global_keep), dtype=tf.int64)

    @tf.function
    def _keep_by_index(i, x):
        j = tf.searchsorted(global_keep_tf, i, side="left")
        keep = tf.logical_and(
            j < tf.shape(global_keep_tf)[0], tf.equal(global_keep_tf[j], i)
        )
        return keep

    ds = ds.enumerate()
    ds = ds.filter(_keep_by_index)
    ds = ds.map(lambda i, x: x, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    opts = tf.data.Options()
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=42, reshuffle_each_iteration=True
        )
        ds = ds.map(
            lambda x, y: (_augment(x), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = ds.cache()

    ds = ds.batch(bat_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_generator = _make_ds_from_tfrecords(
    train_df, train_shard, train_within, training=True
)
validation_generator = _make_ds_from_tfrecords(
    val_df, val_shard, val_within, training=False
)

class_names = ["0", "1", "2", "3", "4"]
class_indices = {c: i for i, c in enumerate(class_names)}




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2312217534.py in <cell line: 0>()
    161 
    162 
--> 163 train_generator = _make_ds_from_tfrecords(
    164     train_df, train_shard, train_within, training=True
    165 )

/tmp/ipykernel_11/2312217534.py in _make_ds_from_tfrecords(df, df_shards, df_within, training)
    130 
    131     ds = ds.enumerate()
--> 132     ds = ds.filter(_keep_by_index)
    133     ds = ds.map(lambda i, x: x, num_parallel_calls=AUTOTUNE, deterministic=True)
    134 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in filter(self, predicate, name)
   2561     # pylint: disable=g-import-not-at-top,protected-access
   2562     from tensorflow.python.data.ops import filter_op
-> 2563     return filter_op._filter(self, predicate, name)
   2564     # pylint: enable=g-import-not-at-top,protected-access
   2565 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in _filter(input_dataset, predicate, name)
     23 
     24 def _filter(input_dataset, predicate, name=None):  # pylint: disable=redefined-builtin
---> 25   return _FilterDataset(input_dataset, predicate, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in __init__(self, input_dataset, predicate, use_legacy_function, name)
     36     """See `Dataset.filter` for details."""
     37     self._input_dataset = input_dataset
---> 38     wrapped_func = structured_function.StructuredFunctionWrapper(
     39         predicate,
     40         self._transformation_name(),

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

/tmp/__autograph_generated_file1joq8299.py in tf___keep_by_index(i, x)
      9                 do_return = False
     10                 retval_ = ag__.UndefinedReturnValue()
---> 11                 j = ag__.converted_call(ag__.ld(tf).searchsorted, (ag__.ld(global_keep_tf), ag__.ld(i)), dict(side='left'), fscope)
     12                 keep = ag__.converted_call(ag__.ld(tf).logical_and, (ag__.ld(j) < ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(global_keep_tf),), None, fscope)[0], ag__.converted_call(ag__.ld(tf).equal, (ag__.ld(global_keep_tf)[ag__.ld(j)], ag__.ld(i)), None, fscope)), None, fscope)
     13                 try:

ValueError: in user code:

    File "/tmp/ipykernel_11/2312217534.py", line 125, in _keep_by_index  *
        j = tf.searchsorted(global_keep_tf, i, side="left")

    ValueError: slice index -1 of dimension 0 out of bounds. for '{{node strided_slice_1}} = StridedSlice[Index=DT_INT32, T=DT_INT32, begin_mask=0, ellipsis_mask=0, end_mask=0, new_axis_mask=0, shrink_axis_mask=1](Shape_1/shape_as_tensor, strided_slice_1/stack, strided_slice_1/stack_1, strided_slice_1/stack_2)' with input shapes: [0], [1], [1], [1] and with computed input tensors: input[1] = <-1>, input[2] = <0>, input[3] = <1>.


## === cell 9
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)




## === cell 10
from tensorflow.keras.applications import EfficientNetB7

backbone = EfficientNetB7(
    include_top=False, weights="imagenet", input_shape=(512, 512, 3)
)
backbone.trainable = False

inputs = keras.Input(shape=(512, 512, 3))
x = backbone(inputs, training=False)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

model.summary()




## === cell 11
history = model.fit(
    train_generator,
    epochs=epoch,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[callback],
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4224867440.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_generator,
      3     epochs=epoch,
      4     validation_data=validation_generator,
      5     verbose=1,

NameError: name 'train_generator' is not defined

## === cell 12
_ = len(os.listdir(test_path))




## === cell 13
ss = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

inv_class_indices = {v: int(k) for k, v in class_indices.items()}


test_files = [
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test00-1338.tfrec",
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test01-1338.tfrec",
]
test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_ds = test_ds.batch(bat_size, drop_remainder=False).prefetch(AUTOTUNE)

all_probs = []
all_names = []
for batch_imgs, batch_names in test_ds:
    all_probs.append(model(batch_imgs, training=False).numpy())
    all_names.append(batch_names.numpy())

probs = np.concatenate(all_probs, axis=0)
names = np.concatenate(all_names, axis=0).astype("U")  # bytes->str

cls_idx = np.argmax(probs, axis=1).astype(int)
preds = np.array([inv_class_indices[i] for i in cls_idx.tolist()], dtype=np.int32)

pred_df = pd.DataFrame({"image_id": names, "label": preds})
my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission.to_csv("submission.csv", index=False)
_ = my_submission.head()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/141015140.py in <cell line: 0>()
      3 )
      4 
----> 5 inv_class_indices = {v: int(k) for k, v in class_indices.items()}
      6 
      7 # --------------------------------------------------------------------------------------

NameError: name 'class_indices' is not defined

## === cell 14
model.save("model.keras")
