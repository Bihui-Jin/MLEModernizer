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

0.8798730734360835

# 6. Current score

0.09043

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.09043) has done: 'I remove the pure-Python protobuf override (it makes TFRecord/protobuf parsing significantly slower) and enable TensorFlow graph execution for the input pipeline and inference to cut Python overhead. I also switch training data loading from `ImageDataGenerator.flow_from_dataframe` (Python/keras-preprocessing bottleneck) to an equivalent `tf.data` pipeline that performs the same decode/resize + EfficientNet preprocess and applies the same augmentations using TensorFlow ops, keeping model/optimizer/loss/epochs unchanged. I add caching/prefetching and set `steps_per_epoch`/`validation_steps` explicitly to avoid generator bookkeeping overhead while preserving the same number of samples per epoch. Finally, I speed up submission inference by using the same optimized `tf.data` pipeline with `ignore_errors()` and graph-compiled map functions.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TRAIN_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TEST_IMG = (
    "/kaggle/input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
)
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
MODELS_WEIGHTS = "/kaggle/input/cassavaeffentb7models/content/Models"

print("TRAIN_IMG_LOC exists:", os.path.isdir(TRAIN_IMG_LOC))
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("SAMPLE_CSV exists:", os.path.isfile(SAMPLE_CSV))



## === cell 1
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)
try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

IMG_SIZE = (300, 300)  # matches original inference resize
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 3  # keep unchanged

train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)
train_df = train_df[train_df["filepath"].map(os.path.isfile)].reset_index(drop=True)
train_df["label"] = train_df["label"].astype(np.int32)

from sklearn.model_selection import train_test_split

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img)

    angle = tf.random.uniform([], minval=-10.0, maxval=10.0, dtype=tf.float32) * (
        np.pi / 180.0
    )
    img = tf.image.rotate(img, angle, fill_mode="reflect", interpolation="BILINEAR")

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    max_dx = 0.05 * w
    max_dy = 0.05 * h
    dx = tf.random.uniform([], -max_dx, max_dx, dtype=tf.float32)
    dy = tf.random.uniform([], -max_dy, max_dy, dtype=tf.float32)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(
            tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]), 0
        ),
        output_shape=tf.shape(img)[:2],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]

    scale = tf.random.uniform([], 0.9, 1.1, dtype=tf.float32)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)
    img2 = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)

    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])
    return img2


def _make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(
        buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
    )

    def _map_fn(p, y):
        img = _decode_resize_preprocess(p)
        img = _augment(img)
        y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(p, y):
        img = _decode_resize_preprocess(p)
        y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


trn_paths = trn_df["filepath"].to_numpy(dtype=str)
trn_labels = trn_df["label"].to_numpy(dtype=np.int32)
val_paths = val_df["filepath"].to_numpy(dtype=str)
val_labels = val_df["label"].to_numpy(dtype=np.int32)

train_ds = _make_train_ds(trn_paths, trn_labels, BATCH_SIZE)
val_ds = _make_val_ds(val_paths, val_labels, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(trn_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

base = EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # unchanged

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = "/kaggle/working/effnetb4_cassava_best.keras"
ckpt = ModelCheckpoint(
    ckpt_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[ckpt],
    verbose=1,
)

if os.path.isfile(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)

print("Model ready. Checkpoint exists:", os.path.isfile(ckpt_path))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3461205999.py in <cell line: 0>()
    133 val_labels = val_df["label"].to_numpy(dtype=np.int32)
    134 
--> 135 train_ds = _make_train_ds(trn_paths, trn_labels, BATCH_SIZE)
    136 val_ds = _make_val_ds(val_paths, val_labels, BATCH_SIZE)
    137 

/tmp/ipykernel_11/3461205999.py in _make_train_ds(paths, labels, batch_size)
    105         return img, y
    106 
--> 107     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    108     # Cache decoded+augmented pipeline is not correct (would freeze augmentation), so do not cache train.
    109     ds = ds.batch(batch_size, drop_remainder=False)

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

/tmp/__autograph_generated_filev0oek2kv.py in tf___map_fn(p, y)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img = ag__.converted_call(ag__.ld(_decode_resize_preprocess), (ag__.ld(p),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img),), None, fscope)
     12                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(y), ag__.ld(tf).int32), None, fscope), ag__.ld(NUM_CLASSES)), None, fscope)
     13                 try:

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

/tmp/__autograph_generated_filep0y_b2cj.py in tf___augment(img)
     10                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img),), None, fscope)
     11                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([],), dict(minval=-10.0, maxval=10.0, dtype=ag__.ld(tf).float32), fscope) * (ag__.ld(np).pi / 180.0)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img), ag__.ld(angle)), dict(fill_mode='reflect', interpolation='BILINEAR'), fscope)
     13                 h = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[0], ag__.ld(tf).float32), None, fscope)
     14                 w = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[1], ag__.ld(tf).float32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/3461205999.py", line 103, in _map_fn  *
        img = _augment(img)
    File "/tmp/ipykernel_11/3461205999.py", line 63, in _augment  *
        img = tf.image.rotate(img, angle, fill_mode="reflect", interpolation="BILINEAR")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 3
print("Skipping plot_model to save time.")



## === cell 4
print("Skipping test image visualization to save time.")



## === cell 5
if "model" not in globals():
    print(
        "WARNING: `model` was not created during training. Building an untrained model to allow submission generation."
    )
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)

ss = pd.read_csv(SAMPLE_CSV)

test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_paths = (test_img_dir + "/" + ss["image_id"].astype(str)).to_numpy(dtype=str)

paths_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths, dtype=tf.string))


@tf.function
def _load_test(path):
    img = _decode_resize_preprocess(path)
    return img


img_ds = (
    paths_ds.map(_load_test, num_parallel_calls=AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

prob = model.predict(img_ds, verbose=0)
preds = np.argmax(prob, axis=1).astype(int)

if len(preds) != len(ss):
    raise RuntimeError(
        f"Prediction count mismatch: preds={len(preds)} vs sample_submission={len(ss)}"
    )

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote submission to:", out_path)
print(my_submission.head())
print("Submission shape:", my_submission.shape)
