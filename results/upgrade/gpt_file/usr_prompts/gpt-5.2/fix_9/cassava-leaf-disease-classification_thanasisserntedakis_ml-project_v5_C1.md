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

0.6158960411000303

# 6. Current score

0.14425

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14425) has done: 'The timeout is dominated by slow Python-based JPEG decoding/augmentation via `ImageDataGenerator.flow_from_dataframe` plus oversized `IMG_SIZE=300`, causing the CPU to spend most time in Python generators rather than TensorFlow. Without changing the model or training loop semantics, I switch the input pipeline to a `tf.data` pipeline that uses TensorFlow ops for JPEG decode/resize/rescale and applies the same augmentations (rotation/shift/zoom/flip) inside the graph, with caching/prefetching and parallel mapping to remove Python overhead. I also ensure deterministic behavior via seeds and deterministic dataset options, and I keep the same split, labels, loss, optimizer, epochs, and prediction semantics (argmax of softmax). Paths remain unchanged and the submission format is identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("TF version:", tf.__version__)


def _fast_jpg_count(img_dir: str) -> int:
    try:
        return sum(1 for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    except Exception:
        return -1


print("Train images:", _fast_jpg_count(TRAIN_IMG_DIR))
print("Test images:", _fast_jpg_count(TEST_IMG_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)

train_files_set = set(os.listdir(TRAIN_IMG_DIR))
exists_mask = df["image_id"].isin(train_files_set)
missing = int((~exists_mask).sum())
print("Missing train files:", missing)
if missing:
    df = df.loc[exists_mask].reset_index(drop=True)

df["path"] = (TRAIN_IMG_DIR + os.sep + df["image_id"].astype(str)).astype(str)
df["label"] = df["label"].astype(str)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)

IMG_SIZE = 300
BATCH_SIZE = 32

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

class_names = sorted(df["label"].unique().tolist())
NUM_CLASSES = len(class_names)
print("Num classes:", NUM_CLASSES)
print("Class names:", class_names)
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"

label_to_index = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(class_names),
        values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int64),
    ),
    default_value=-1,
)


def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-10.0,
        maxval=10.0,
        dtype=tf.float32,
    )
    angle = angle * (np.pi / 180.0)
    img = tfa_image_rotate(img, angle)

    max_dx = tf.cast(tf.round(0.05 * tf.cast(IMG_SIZE, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(IMG_SIZE, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([3, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tfa_image_translate(img, tf.cast([dx, dy], tf.float32))

    scale = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([4, 0], tf.int32),
        minval=0.95,
        maxval=1.05,
        dtype=tf.float32,
    )
    img = zoom_image(img, scale)
    return img


def tfa_image_rotate(image, angle):
    if hasattr(tf.image, "rotate"):
        return tf.image.rotate(image, angle, interpolation="BILINEAR")
    raise AttributeError("tf.image.rotate not available in this TF build")


def tfa_image_translate(image, translations):
    dx = translations[0]
    dy = translations[1]
    transforms = tf.convert_to_tensor(
        [[1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]], dtype=tf.float32
    )
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=transforms,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]


def zoom_image(image, scale):
    if scale < 1.0:
        new_size = tf.cast(tf.round(scale * tf.cast(IMG_SIZE, tf.float32)), tf.int32)
        new_size = tf.maximum(1, new_size)
        image = tf.image.random_crop(image, size=[new_size, new_size, 3], seed=SEED)
        image = tf.image.resize(
            image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
        )
        return image
    elif scale > 1.0:
        pad = tf.cast(
            tf.round((scale - 1.0) * tf.cast(IMG_SIZE, tf.float32) / 2.0), tf.int32
        )
        image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
        return image
    else:
        return image


AUTOTUNE = tf.data.AUTOTUNE


def make_ds(paths, labels=None, training=False, batch_size=32, shuffle=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices((paths,))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    if labels is None:

        def _map_fn(path):
            img = _decode_resize_rescale(path)
            return img

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.enumerate()

        def _map_fn(i, pl):
            path, label_str = pl
            img = _decode_resize_rescale(path)
            if training:
                seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
                img = _augment(img, seed)
            y = label_to_index.lookup(label_str)
            y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
            return img, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_paths = train_df["path"].values.astype(str)
val_paths = val_df["path"].values.astype(str)
train_labels = train_df["label"].values.astype(str)
val_labels = val_df["label"].values.astype(str)

train_ds = make_ds(
    train_paths, train_labels, training=True, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = make_ds(
    val_paths, val_labels, training=False, batch_size=BATCH_SIZE, shuffle=False
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/380732562.py in <cell line: 0>()
    211 val_labels = val_df["label"].values.astype(str)
    212 
--> 213 train_ds = make_ds(
    214     train_paths, train_labels, training=True, batch_size=BATCH_SIZE, shuffle=True
    215 )

/tmp/ipykernel_11/380732562.py in make_ds(paths, labels, training, batch_size, shuffle)
    196             return img, y
    197 
--> 198         ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    199 
    200     # Cache decoded/resized images to avoid repeating expensive JPEG decode across epochs.

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

/tmp/__autograph_generated_fileq6j41hkd.py in tf___map_fn(i, pl)
     28                     pass
     29                 seed = ag__.Undefined('seed')
---> 30                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     31                 y = ag__.converted_call(ag__.ld(label_to_index).lookup, (ag__.ld(label_str),), None, fscope)
     32                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(y), ag__.ld(tf).int32), None, fscope), ag__.ld(NUM_CLASSES)), None, fscope)

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

/tmp/__autograph_generated_fileq6j41hkd.py in if_body()
     22                     nonlocal img
     23                     seed = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(i), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 24                     img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed)), None, fscope)
     25 
     26                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filenf9jc269.py in tf___augment(img, seed)
     11                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(seed) + ag__.converted_call(ag__.ld(tf).constant, ([1, 0], ag__.ld(tf).int32), None, fscope), minval=-10.0, maxval=10.0, dtype=ag__.ld(tf).float32), fscope)
     12                 angle = ag__.ld(angle) * (ag__.ld(np).pi / 180.0)
---> 13                 img = ag__.converted_call(ag__.ld(tfa_image_rotate), (ag__.ld(img), ag__.ld(angle)), None, fscope)
     14                 max_dx = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (0.05 * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope),), None, fscope), ag__.ld(tf).int32), None, fscope)
     15                 max_dy = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (0.05 * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope),), None, fscope), ag__.ld(tf).int32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file__vj1nf3.py in tf__tfa_image_rotate(image, angle)
     28                     nonlocal retval_, do_return
     29                     raise ag__.converted_call(ag__.ld(AttributeError), ('tf.image.rotate not available in this TF build',), None, fscope)
---> 30                 ag__.if_stmt(ag__.converted_call(ag__.ld(hasattr), (ag__.ld(tf).image, 'rotate'), None, fscope), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     31                 return fscope.ret(retval_, do_return)
     32         return tf__tfa_image_rotate

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

/tmp/__autograph_generated_file__vj1nf3.py in else_body()
     27                 def else_body():
     28                     nonlocal retval_, do_return
---> 29                     raise ag__.converted_call(ag__.ld(AttributeError), ('tf.image.rotate not available in this TF build',), None, fscope)
     30                 ag__.if_stmt(ag__.converted_call(ag__.ld(hasattr), (ag__.ld(tf).image, 'rotate'), None, fscope), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     31                 return fscope.ret(retval_, do_return)

AttributeError: in user code:

    File "/tmp/ipykernel_11/380732562.py", line 193, in _map_fn  *
        img = _augment(img, seed)
    File "/tmp/ipykernel_11/380732562.py", line 74, in _augment  *
        img = tfa_image_rotate(img, angle)
    File "/tmp/ipykernel_11/380732562.py", line 116, in tfa_image_rotate  *
        raise AttributeError("tf.image.rotate not available in this TF build")

    AttributeError: tf.image.rotate not available in this TF build


## === cell 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)

model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/770287392.py in <cell line: 0>()
     35 # Speed: use tf.data datasets (prefetch, parallel map, cache) instead of Python generator.
     36 history = model.fit(
---> 37     train_ds,
     38     validation_data=val_ds,
     39     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 3
sub = pd.read_csv(SAMPLE_SUB)

test_files_set = set(os.listdir(TEST_IMG_DIR))
exists_mask = sub["image_id"].isin(test_files_set)
missing_test = int((~exists_mask).sum())
print("Missing test files from sample_submission paths:", missing_test)

if missing_test:
    test_images = [
        os.path.join(TEST_IMG_DIR, f)
        for f in test_files_set
        if f.lower().endswith(".jpg")
    ]
    df_test = pd.DataFrame({"path": test_images})
    df_test["image_id"] = df_test["path"].map(os.path.basename)
    sub = sub[["image_id"]].merge(df_test, on="image_id", how="left")
    assert sub["path"].notna().all(), "Could not resolve all test image paths"
else:
    sub["path"] = (TEST_IMG_DIR + os.sep + sub["image_id"].astype(str)).astype(str)

test_paths = sub["path"].values.astype(str)
test_ds = make_ds(test_paths, labels=None, training=False, batch_size=64, shuffle=False)

pred_test = model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

submission = pd.DataFrame(
    {"image_id": sub["image_id"].values, "label": pred_test_labels}
)
assert list(submission.columns) == ["image_id", "label"]
assert submission.shape[0] == sub.shape[0]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
