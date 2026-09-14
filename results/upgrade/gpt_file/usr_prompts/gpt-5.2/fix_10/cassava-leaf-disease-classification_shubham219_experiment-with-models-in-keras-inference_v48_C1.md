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

0.6533695980658809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def resolve_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../data/cassava-leaf-disease-classification",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )


BASE_DIR = resolve_base_dir()
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

print("BASE_DIR:", BASE_DIR)
print("Train CSV exists:", os.path.isfile(TRAIN_CSV))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Train TFRecords dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.isdir(TEST_TFREC_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
N_CLASSES = 5
EPOCHS = 2  # unchanged core training loop length

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)

df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
n_val = int(len(df_train) * val_frac)
df_val = df_train.iloc[:n_val].copy()
df_tr = df_train.iloc[n_val:].copy()

AUTOTUNE = tf.data.AUTOTUNE

tr_paths = df_tr["path"].to_numpy()
tr_labels = df_tr["label"].to_numpy(dtype=np.int32)
val_paths = df_val["path"].to_numpy()
val_labels = df_val["label"].to_numpy(dtype=np.int32)


@tf.function
def _decode_resize_rescale_from_bytes(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)  # cassava images are jpeg
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    return _decode_resize_rescale_from_bytes(img)


augment_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=10.0 / 360.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augment",
)

_ = augment_layers(
    tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], tf.float32), training=True
)


@tf.function
def _train_map(path, label):
    img = _decode_resize_rescale(path)
    img = augment_layers(img, training=True)
    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img, tf.cast(label, tf.int32)


@tf.function
def _val_map(path, label):
    img = _decode_resize_rescale(path)
    return img, tf.cast(label, tf.int32)


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    img = augment_layers(img, training=True)
    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img, tf.cast(ex["target"], tf.int32)


@tf.function
def _parse_val_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    return img, tf.cast(ex["target"], tf.int32)


_SHUFFLE_BUFFER = min(len(df_tr), 8192)


def _list_tfrecs(tfrecord_dir, pattern="*.tfrec"):
    return sorted(glob.glob(os.path.join(tfrecord_dir, pattern)))


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = True

    opt = options.experimental_optimization
    opt.map_parallelization = True
    opt.map_and_batch_fusion = True
    opt.parallel_batch = True

    if hasattr(opt, "autotune_buffers"):
        opt.autotune_buffers = True

    return options


def make_train_ds(paths, labels, batch_size):
    tfrec_files = (
        _list_tfrecs(TRAIN_TFREC_DIR, "*.tfrec")
        if os.path.isdir(TRAIN_TFREC_DIR)
        else []
    )

    if tfrec_files:
        ds = tf.data.Dataset.from_tensor_slices(tfrec_files).with_options(
            _dataset_options()
        )
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
            cycle_length=min(8, len(tfrec_files)),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        ds = ds.shuffle(
            buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.map(
            _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _dataset_options()
    )
    ds = ds.shuffle(
        buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_train_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    tfrec_files = (
        _list_tfrecs(TRAIN_TFREC_DIR, "*.tfrec")
        if os.path.isdir(TRAIN_TFREC_DIR)
        else []
    )

    if tfrec_files:
        ds = tf.data.Dataset.from_tensor_slices(tfrec_files).with_options(
            _dataset_options()
        )
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
            cycle_length=min(8, len(tfrec_files)),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        ds = ds.map(_parse_val_example, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _dataset_options()
    )
    ds = ds.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
val_ds = make_val_ds(val_paths, val_labels, BATCH_SIZE)

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
base.trainable = False
x = tf.keras.layers.Dropout(0.2)(base.output)
outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
my_model = tf.keras.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

if DEBUG:
    my_model.summary()

steps_per_epoch = int(np.ceil(len(df_tr) / BATCH_SIZE))
validation_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3560621330.py in <cell line: 0>()
    193 
    194 
--> 195 train_ds = make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
    196 val_ds = make_val_ds(val_paths, val_labels, BATCH_SIZE)
    197 

/tmp/ipykernel_11/3560621330.py in make_train_ds(paths, labels, batch_size)
    141             buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    142         )
--> 143         ds = ds.map(
    144             _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
    145         )

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

/tmp/__autograph_generated_fileoitms5eu.py in tf___parse_train_example(example_proto)
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example_proto), ag__.ld(_FEATURES)), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(_decode_resize_rescale_from_bytes), (ag__.ld(ex)['image'],), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(augment_layers), (ag__.ld(img),), dict(training=True), fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).image.random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(SEED)), fscope)
     14                 try:

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: in user code:

    File "/tmp/ipykernel_11/3560621330.py", line 87, in _parse_train_example  *
        img = augment_layers(img, training=True)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py", line 272, in _adjust_input_rank
        raise ValueError(

    ValueError: Exception encountered when calling Sequential.call().
    
    Invalid input shape for input Tensor("PartitionedCall:0", shape=(300, 300, 3), dtype=float32). Expected shape (1, 300, 300, 3), but input has incompatible shape (300, 300, 3)
    
    Arguments received by Sequential.call():
      • inputs=tf.Tensor(shape=(300, 300, 3), dtype=float32)
      • training=True
      • mask=None


## === cell 2
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found in: {TEST_IMG_DIR}")

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = df_test["path"].map(os.path.basename)


@tf.function
def _test_map(path):
    img = _decode_resize_rescale(path)
    return img


def make_test_ds(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_dataset_options())
    ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(df_test["path"].to_numpy(), batch_size=128)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

if os.path.isfile(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        mode_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(mode_label).astype(int)

final_csv.to_csv("submission.csv", index=False)
final_csv.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/113064347.py in <cell line: 0>()
     23 test_ds = make_test_ds(df_test["path"].to_numpy(), batch_size=128)
     24 
---> 25 pred_test = my_model.predict(
     26     test_ds,
     27     verbose=1,

NameError: name 'my_model' is not defined

## === cell 3
print("submission.csv written:", os.path.isfile("submission.csv"))
print("Rows:", len(final_csv), "Columns:", list(final_csv.columns))
print(final_csv["label"].value_counts(dropna=False).sort_index())
final_csv.tail()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2178964416.py in <cell line: 0>()
      1 print("submission.csv written:", os.path.isfile("submission.csv"))
----> 2 print("Rows:", len(final_csv), "Columns:", list(final_csv.columns))
      3 print(final_csv["label"].value_counts(dropna=False).sort_index())
      4 final_csv.tail()

NameError: name 'final_csv' is not defined
