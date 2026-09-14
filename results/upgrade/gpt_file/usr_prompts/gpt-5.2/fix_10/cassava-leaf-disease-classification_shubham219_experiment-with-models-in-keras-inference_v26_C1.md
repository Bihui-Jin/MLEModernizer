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

No external packages required in the script and installed.

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

0.703838017527954

# 6. Current score

0.13976

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.15807) has done: 'I fix the initial TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in many Kaggle images. Then I remove the dependency on a missing external `.h5` file by instantiating the same kind of Keras classifier locally (EfficientNet backbone + softmax head) so the notebook runs end-to-end. Finally, I make the test generator robust (correct file paths, ensure prediction length matches test rows) and always write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.13976) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in many Kaggle runtimes. Then I fix the TFRecord parsing logic: the cassava TFRecords don’t contain `image_id`, so I remove it from the required feature spec and avoid train/val filtering by `image_id` when using TFRecords (while preserving the same model/training loop via a simple `validation_split` on the dataframe generator fallback). Finally, I make test inference always produce a correctly aligned `submission.csv` by using `sample_submission.csv` for ordering and predicting from either TFRecords (without `image_id`) or the JPEG directory, ensuring the file is always written.'
- What this solution (achieved 0.13976) has done: 'The timeout is dominated by slow input pipelines: `ImageDataGenerator.flow_from_dataframe` + wrapping it with `tf.data.Dataset.from_generator` forces Python to sit in the training loop and starves the model, and you also force the slow pure-Python protobuf implementation. I keep the same model, loss, optimizer, epochs, and the same data augmentation semantics, but move training/validation loading to a fully-TensorFlow `tf.data` pipeline that does decode/resize/preprocess + the exact same augmentations on-device with parallel map and prefetch. For test, I keep TFRecords but optimize decoding (fused jpeg decode+resize) and batching/prefetch. These changes remove Python from the hot path and typically cut epoch time dramatically while preserving the algorithm and accuracy.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import glob
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

np.random.seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_TRAIN_CSV = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "../input/cassava-leaf-disease-classification/train.csv",
    "../data/cassava-leaf-disease-classification/train.csv",
]
CANDIDATE_TRAIN_IMG_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
    "../data/cassava-leaf-disease-classification/train_images",
]
CANDIDATE_TRAIN_TFREC_GLOBS = [
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "/kaggle/data/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "../data/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
]

train_csv_path = None
for p in CANDIDATE_TRAIN_CSV:
    if os.path.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(
        "Could not find train.csv. Tried:\n" + "\n".join(CANDIDATE_TRAIN_CSV)
    )

train_img_dir = None
for d in CANDIDATE_TRAIN_IMG_DIRS:
    if os.path.isdir(d):
        train_img_dir = d
        break
if train_img_dir is None:
    raise FileNotFoundError(
        "Could not find train_images dir. Tried:\n"
        + "\n".join(CANDIDATE_TRAIN_IMG_DIRS)
    )

train_tfrec_files = []
for pat in CANDIDATE_TRAIN_TFREC_GLOBS:
    train_tfrec_files = sorted(glob.glob(pat))
    if len(train_tfrec_files) > 0:
        break

train_df = pd.read_csv(train_csv_path)
if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)

print("Train rows:", len(train_df), "train_img_dir:", train_img_dir)
print("Found train tfrecords:", len(train_tfrec_files))



## === cell 2
NUM_CLASSES = 5
IMG_SIZE = (300, 300)

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = keras.layers.Dropout(0.2)(base.output)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

my_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

my_model.summary()



## === cell 3
val_frac = 0.1
train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:n_val].copy()
trn_df = train_df_shuf.iloc[n_val:].copy()

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

batch_size = 32 if not DEBUG else 16
EPOCHS = 2 if DEBUG else 3

steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
validation_steps = int(np.ceil(len(val_df) / batch_size))

AUTOTUNE = tf.data.AUTOTUNE

tf_data_opts = tf.data.Options()
tf_data_opts.deterministic = True

use_tfrecs_for_train = False


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img


@tf.function
def _augment_like_idg(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    angle = tf.random.uniform([], -15.0, 15.0, seed=SEED) * (np.pi / 180.0)
    img = tf.image.rotate(img, angle, interpolation="BILINEAR")

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    pad_h = tf.cast(tf.cast(h, tf.float32) * 0.05, tf.int32)
    pad_w = tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32)
    img = tf.pad(img, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, size=[h, w, 3], seed=SEED)

    scale = tf.random.uniform([], 0.9, 1.1, seed=SEED)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, h, w)
    return img2


def _make_path_ds(df, shuffle, augment):
    paths = (
        (df["image_id"].apply(lambda x: os.path.join(train_img_dir, x)))
        .astype(str)
        .values
    )
    labels = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.with_options(tf_data_opts)

    def _load(path, label):
        img = _decode_resize_preprocess(path)
        if augment:
            img = _augment_like_idg(img)
        return img, tf.cast(label, tf.float32)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


if use_tfrecs_for_train:
    raise RuntimeError(
        "TFRecord training disabled because train TFRecords lack labels."
    )
else:
    train_ds = _make_path_ds(trn_df, shuffle=True, augment=True)
    val_ds = _make_path_ds(val_df, shuffle=False, augment=False)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3540777391.py in <cell line: 0>()
     91     )
     92 else:
---> 93     train_ds = _make_path_ds(trn_df, shuffle=True, augment=True)
     94     val_ds = _make_path_ds(val_df, shuffle=False, augment=False)
     95 

/tmp/ipykernel_11/3540777391.py in _make_path_ds(df, shuffle, augment)
     80         return img, tf.cast(label, tf.float32)
     81 
---> 82     ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
     83     ds = ds.batch(batch_size, drop_remainder=False)
     84     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_filejn4h__8v.py in tf___load(path, label)
     25                     nonlocal img
     26                     pass
---> 27                 ag__.if_stmt(ag__.ld(augment), if_body, else_body, get_state, set_state, ('img',), 1)
     28                 try:
     29                     do_return = True

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

/tmp/__autograph_generated_filejn4h__8v.py in if_body()
     20                 def if_body():
     21                     nonlocal img
---> 22                     img = ag__.converted_call(ag__.ld(_augment_like_idg), (ag__.ld(img),), None, fscope)
     23 
     24                 def else_body():

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

/tmp/__autograph_generated_fileh6581hq7.py in tf___augment_like_idg(img)
     10                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(SEED)), fscope)
     11                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -15.0, 15.0), dict(seed=ag__.ld(SEED)), fscope) * (ag__.ld(np).pi / 180.0)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img), ag__.ld(angle)), dict(interpolation='BILINEAR'), fscope)
     13                 h = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[0]
     14                 w = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[1]

AttributeError: in user code:

    File "/tmp/ipykernel_11/3540777391.py", line 79, in _load  *
        img = _augment_like_idg(img)
    File "/tmp/ipykernel_11/3540777391.py", line 45, in _augment_like_idg  *
        img = tf.image.rotate(img, angle, interpolation="BILINEAR")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 4
CANDIDATE_TEST_TFREC_GLOBS = [
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "/kaggle/data/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "../data/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
]
test_tfrec_files = []
for pat in CANDIDATE_TEST_TFREC_GLOBS:
    test_tfrec_files = sorted(glob.glob(pat))
    if len(test_tfrec_files) > 0:
        break


def _decode_and_resize(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


_FEATURE_DESCRIPTION_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION_TEST)
    img = _decode_and_resize(ex["image"])
    img = preprocess_fn(img)
    return img


if len(test_tfrec_files) > 0:
    test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
    test_ds = test_ds.with_options(tf_data_opts)
    test_ds = test_ds.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)
    df_test = None  # will be taken from sample_submission
else:
    CANDIDATE_TEST_GLOBS = [
        "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
        "/kaggle/data/cassava-leaf-disease-classification/test_images/*.jpg",
        "../input/cassava-leaf-disease-classification/test_images/*.jpg",
        "../data/cassava-leaf-disease-classification/test_images/*.jpg",
    ]
    test_images = []
    for pat in CANDIDATE_TEST_GLOBS:
        test_images = glob.glob(pat)
        if len(test_images) > 0:
            break

    if len(test_images) == 0:
        raise FileNotFoundError(
            "Could not find test images. Tried:\n" + "\n".join(CANDIDATE_TEST_GLOBS)
        )

    df_test = pd.DataFrame({"path": test_images})
    df_test["image_id"] = df_test["path"].astype(str).str.split("/").str[-1]

    test_paths = df_test["path"].astype(str).values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.with_options(tf_data_opts)

    def _load_test(path):
        return _decode_resize_preprocess(path)

    test_ds = test_ds.map(_load_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 5
CANDIDATE_SAMPLE = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "../data/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in CANDIDATE_SAMPLE:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv. Tried:\n" + "\n".join(CANDIDATE_SAMPLE)
    )

sample = pd.read_csv(sample_path)
sample["image_id"] = sample["image_id"].astype(str)
print("sample_submission rows:", len(sample))

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(np.int64)

if df_test is None:
    n = len(sample)
    if len(pred_test_labels) != n:
        if len(pred_test_labels) > n:
            pred_test_labels = pred_test_labels[:n]
        else:
            pred_test_labels = np.pad(
                pred_test_labels, (0, n - len(pred_test_labels)), constant_values=0
            )
    final_csv = sample[["image_id"]].copy()
    final_csv["label"] = pred_test_labels.astype(int)
else:
    tmp = df_test[["image_id"]].copy()
    tmp["label"] = pred_test_labels[: len(tmp)].astype(int)
    final_csv = sample[["image_id"]].merge(tmp, on="image_id", how="left")
    final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")
print("Saved: submission.csv")
