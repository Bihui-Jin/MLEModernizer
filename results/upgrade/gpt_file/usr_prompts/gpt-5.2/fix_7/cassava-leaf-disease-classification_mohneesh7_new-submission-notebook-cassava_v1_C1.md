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

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import warnings

warnings.filterwarnings("ignore")

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick good defaults
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

print("TensorFlow:", tf.__version__)
print("Train CSV rows:", sum(1 for _ in open(TRAIN_CSV)) - 1)
print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
print("Test images:", len(os.listdir(TEST_IMG_DIR)))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

train_df["image_path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)

train_df["label"] = train_df["label"].astype(str)

image_size = 224
batch_size = 32
num_classes = train_df["label"].nunique()

AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = (image_size, image_size)

val_frac = 0.1
n_total = len(train_df)
n_val = int(round(n_total * val_frac))
n_train = n_total - n_val

train_df_train = train_df.iloc[:n_train].reset_index(drop=True)
train_df_val = train_df.iloc[n_train:].reset_index(drop=True)

class_names = sorted(train_df["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(class_names)}
idx_to_label = {i: int(c) for c, i in class_to_index.items()}
print("class_indices:", class_to_index)


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR)
    return img


def _translate(img, dx, dy):
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])
    transform = tf.expand_dims(transform, axis=0)
    img = tf.expand_dims(img, axis=0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img,
        transforms=transform,
        output_shape=tf.constant([image_size, image_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


def _random_zoom(img, zoom_range=0.1):
    z = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range, seed=SEED)
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    new_h = tf.cast(tf.round(h / z), tf.int32)
    new_w = tf.cast(tf.round(w / z), tf.int32)

    def zoom_in():
        offset_h = tf.random.uniform(
            [],
            0,
            tf.maximum(1, tf.shape(img)[0] - new_h + 1),
            dtype=tf.int32,
            seed=SEED,
        )
        offset_w = tf.random.uniform(
            [],
            0,
            tf.maximum(1, tf.shape(img)[1] - new_w + 1),
            dtype=tf.int32,
            seed=SEED,
        )
        cropped = tf.image.crop_to_bounding_box(img, offset_h, offset_w, new_h, new_w)
        return tf.image.resize(
            cropped, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR
        )

    def zoom_out():
        pad_h = tf.maximum(0, new_h - tf.shape(img)[0])
        pad_w = tf.maximum(0, new_w - tf.shape(img)[1])
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = tf.pad(
            img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT"
        )
        padded = tf.image.resize(
            padded, IMG_SHAPE, method=tf.image.ResizeMethod.BILINEAR
        )
        return padded

    return tf.cond(z >= 1.0, zoom_in, zoom_out)


def _augment(img):
    angle = tf.random.uniform([], -10.0, 10.0, seed=SEED) * (np.pi / 180.0)
    img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    max_dx = 0.05 * image_size
    max_dy = 0.05 * image_size
    dx = tf.random.uniform([], -max_dx, max_dx, seed=SEED)
    dy = tf.random.uniform([], -max_dy, max_dy, seed=SEED)
    img = _translate(img, dx, dy)

    img = _random_zoom(img, zoom_range=0.1)

    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img


def _preprocess(img):
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _make_ds(df, training):
    paths = df["image_path"].astype(str).values
    labels_str = df["label"].values.astype(str)
    labels = np.array([class_to_index[s] for s in labels_str], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(path, label):
        img = _read_decode_resize(path)
        if training:
            img = _augment(img)
        img = _preprocess(img)
        y = tf.one_hot(label, num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(train_df_train, training=True)
val_ds = _make_ds(train_df_val, training=False)

base = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(
    x
)  # keep output float32 under mixed precision
model = Model(inputs=base.input, outputs=out)

base.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

epochs = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2563349877.py in <cell line: 0>()
    176 
    177 
--> 178 train_ds = _make_ds(train_df_train, training=True)
    179 val_ds = _make_ds(train_df_val, training=False)
    180 

/tmp/ipykernel_11/2563349877.py in _make_ds(df, training)
    168         return img, y
    169 
--> 170     ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    171     # Cache decoded/preprocessed images to avoid repeated disk IO across epochs (big speedup, preserves semantics).
    172     ds = ds.cache()

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

/tmp/__autograph_generated_filenfl_l05g.py in tf___load(path, label)
     25                     nonlocal img
     26                     pass
---> 27                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     28                 img = ag__.converted_call(ag__.ld(_preprocess), (ag__.ld(img),), None, fscope)
     29                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(label), ag__.ld(num_classes)), dict(dtype=ag__.ld(tf).float32), fscope)

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

/tmp/__autograph_generated_filenfl_l05g.py in if_body()
     20                 def if_body():
     21                     nonlocal img
---> 22                     img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img),), None, fscope)
     23 
     24                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_fileu9v9xhj4.py in tf___augment(img)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 angle = ag__.converted_call(ag__.ld(tf).random.uniform, ([], -10.0, 10.0), dict(seed=ag__.ld(SEED)), fscope) * (ag__.ld(np).pi / 180.0)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)
     12                 max_dx = 0.05 * ag__.ld(image_size)
     13                 max_dy = 0.05 * ag__.ld(image_size)

AttributeError: in user code:

    File "/tmp/ipykernel_11/2563349877.py", line 165, in _load  *
        img = _augment(img)
    File "/tmp/ipykernel_11/2563349877.py", line 123, in _augment  *
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 2
sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sub["image_id"].tolist()
test_paths = (TEST_IMG_DIR + "/" + sub["image_id"].astype(str)).values.astype(str)


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    def _load(path):
        img = _read_decode_resize(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds(test_paths)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)
pred_labels = [idx_to_label[int(i)] for i in pred_idx]

submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/876635026.py in <cell line: 0>()
     24 test_ds = _make_test_ds(test_paths)
     25 
---> 26 probs = model.predict(test_ds, verbose=1)
     27 pred_idx = np.argmax(probs, axis=1)
     28 pred_labels = [idx_to_label[int(i)] for i in pred_idx]

NameError: name 'model' is not defined
