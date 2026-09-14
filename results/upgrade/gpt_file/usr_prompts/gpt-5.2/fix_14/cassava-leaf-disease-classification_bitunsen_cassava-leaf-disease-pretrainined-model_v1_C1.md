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

0.7355696585071019

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

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import json

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 32

PRE_TRAINED_MODEL_CANDIDATES = [
    "../input/xceptionv1/Cassava_Best_Model_XceptionV3_V01.hdf5",
    "/kaggle/input/xceptionv1/Cassava_Best_Model_XceptionV3_V01.hdf5",
]
PRE_TRAINED_MODEL = next(
    (p for p in PRE_TRAINED_MODEL_CANDIDATES if os.path.exists(p)), None
)
print("PRE_TRAINED_MODEL:", PRE_TRAINED_MODEL)



## === cell 7
pass



## === cell 8
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
test_samples




## === cell 9
def make_test_dataset(image_ids, batch_size):
    image_ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    base = tf.constant(TEST_DIR, dtype=tf.string)

    def _path_from_id(image_id):
        return tf.strings.join([base, image_id])

    def _load_and_preprocess_from_path(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # RGB
        img = tf.image.resize(
            img,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.AREA,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(image_ids)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_path_from_id, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.interleave(
        lambda p: tf.data.Dataset.from_tensors(p).map(
            _load_and_preprocess_from_path, num_parallel_calls=tf.data.AUTOTUNE
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_dataset(test_df["image_id"].values, batch_size=batch_size)



## === cell 10
import random
from keras.models import load_model


def build_fallback_model(img_h=IMG_HEIGHT, img_w=IMG_WIDTH, n_classes=5):
    base = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(img_h, img_w, 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = keras.applications.xception.preprocess_input(inputs * 255.0)  # inputs are 0..1
    x = base(x, training=False)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass


def make_train_valid_datasets(
    tr_image_ids, tr_labels, va_image_ids, va_labels, batch_size
):
    tr_image_ids = tf.convert_to_tensor(tr_image_ids, dtype=tf.string)
    va_image_ids = tf.convert_to_tensor(va_image_ids, dtype=tf.string)

    tr_labels = tf.convert_to_tensor(tr_labels, dtype=tf.int64)
    va_labels = tf.convert_to_tensor(va_labels, dtype=tf.int64)

    base = tf.constant(TRAIN_DIR, dtype=tf.string)

    def _path_from_id(image_id):
        return tf.strings.join([base, image_id])

    def _decode_resize_from_path(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.AREA,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _augment_stateless(img, idx):
        r0 = tf.random.stateless_uniform(
            [], seed=[SEED, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )
        img = tf.cond(r0 < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

        r1 = tf.random.stateless_uniform(
            [], seed=[SEED + 1, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )

        def _cb():
            alpha = tf.random.stateless_uniform(
                [], seed=[SEED + 2, tf.cast(idx, tf.int32)], minval=0.8, maxval=1.2
            )
            beta = tf.random.stateless_uniform(
                [], seed=[SEED + 3, tf.cast(idx, tf.int32)], minval=-0.2, maxval=0.2
            )
            x = tf.clip_by_value(img * alpha + beta, 0.0, 1.0)
            return x

        img = tf.cond(r1 < 0.5, _cb, lambda: img)

        r2 = tf.random.stateless_uniform(
            [], seed=[SEED + 4, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )

        def _rot():
            angle_deg = tf.random.stateless_uniform(
                [], seed=[SEED + 5, tf.cast(idx, tf.int32)], minval=-15.0, maxval=15.0
            )
            angle = angle_deg * (np.pi / 180.0)
            try:
                return tf.image.rotate(
                    img,
                    angles=angle,
                    interpolation="BILINEAR",
                    fill_mode="CONSTANT",
                    fill_value=0.0,
                )
            except Exception:
                return img

        img = tf.cond(r2 < 0.2, _rot, lambda: img)
        return img

    def _train_map(idx, path, y):
        img = _decode_resize_from_path(path)
        r = tf.random.stateless_uniform(
            [], seed=[SEED + 6, tf.cast(idx, tf.int32)], minval=0.0, maxval=1.0
        )
        img = tf.cond(r > 0.5, lambda: _augment_stateless(img, idx), lambda: img)
        return img, y

    def _valid_map(path, y):
        img = _decode_resize_from_path(path)
        return img, y

    tr_ds = tf.data.Dataset.from_tensor_slices((tr_image_ids, tr_labels))
    tr_ds = tr_ds.map(
        lambda img_id, y: (_path_from_id(img_id), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    va_ds = tf.data.Dataset.from_tensor_slices((va_image_ids, va_labels))
    va_ds = va_ds.map(
        lambda img_id, y: (_path_from_id(img_id), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    tr_ds = tr_ds.with_options(opts)
    va_ds = va_ds.with_options(opts)

    tr_ds = tr_ds.shuffle(
        buffer_size=min(8192, int(tr_labels.shape[0])),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    tr_ds = tr_ds.enumerate()

    tr_ds = tr_ds.interleave(
        lambda idx, py: tf.data.Dataset.from_tensors((idx, py[0], py[1])).map(
            lambda i, p, y: _train_map(i, p, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    tr_ds = tr_ds.apply(tf.data.experimental.ignore_errors())
    tr_ds = tr_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    va_ds = va_ds.interleave(
        lambda py: tf.data.Dataset.from_tensors((py[0], py[1])).map(
            _valid_map, num_parallel_calls=tf.data.AUTOTUNE
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    va_ds = va_ds.apply(tf.data.experimental.ignore_errors())
    va_ds = va_ds.cache()  # validation is deterministic/static
    va_ds = va_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return tr_ds, va_ds


model = None
if PRE_TRAINED_MODEL is not None and os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
else:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_df))
    split = int(0.9 * len(train_df))
    tr_idx = perm[:split]
    va_idx = perm[split:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds, valid_ds = make_train_valid_datasets(
        tr_df["image_id"].values,
        tr_df["label"].values,
        va_df["image_id"].values,
        va_df["label"].values,
        batch_size=batch_size,
    )

    model = build_fallback_model()
    model.fit(train_ds, validation_data=valid_ds, epochs=1, verbose=1)

model.summary()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4076674483.py in <cell line: 0>()
    201     va_df = train_df.iloc[va_idx].reset_index(drop=True)
    202 
--> 203     train_ds, valid_ds = make_train_valid_datasets(
    204         tr_df["image_id"].values,
    205         tr_df["label"].values,

/tmp/ipykernel_11/4076674483.py in make_train_valid_datasets(tr_image_ids, tr_labels, va_image_ids, va_labels, batch_size)
    170     tr_ds = tr_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    171 
--> 172     va_ds = va_ds.interleave(
    173         lambda py: tf.data.Dataset.from_tensors((py[0], py[1])).map(
    174             _valid_map, num_parallel_calls=tf.data.AUTOTUNE

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in interleave(self, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)
   2532     # pylint: disable=g-import-not-at-top,protected-access
   2533     from tensorflow.python.data.ops import interleave_op
-> 2534     return interleave_op._interleave(self, map_func, cycle_length, block_length,
   2535                                      num_parallel_calls, deterministic, name)
   2536     # pylint: enable=g-import-not-at-top,protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/interleave_op.py in _interleave(input_dataset, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)
     47         input_dataset, map_func, cycle_length, block_length, name=name)
     48   else:
---> 49     return _ParallelInterleaveDataset(
     50         input_dataset,
     51         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/interleave_op.py in __init__(self, input_dataset, map_func, cycle_length, block_length, num_parallel_calls, buffer_output_elements, prefetch_input_elements, deterministic, name)
    117     """See `Dataset.interleave()` for details."""
    118     self._input_dataset = input_dataset
--> 119     self._map_func = structured_function.StructuredFunctionWrapper(
    120         map_func, self._transformation_name(), dataset=input_dataset)
    121     if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):

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

TypeError: in user code:


    TypeError: outer_factory.<locals>.inner_factory.<locals>.<lambda>() takes 1 positional argument but 2 were given


## === cell 11
predict = model.predict(
    test_ds,
    steps=int(np.ceil(test_samples / batch_size)),
    verbose=1,
)
predict.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2325267668.py in <cell line: 0>()
----> 1 predict = model.predict(
      2     test_ds,
      3     steps=int(np.ceil(test_samples / batch_size)),
      4     verbose=1,
      5 )

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 12
test_pred_labels = np.argmax(predict, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": test_pred_labels}
)
submission.to_csv("submission.csv", index=False)
submission.head(3)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1137289259.py in <cell line: 0>()
----> 1 test_pred_labels = np.argmax(predict, axis=1).astype(int)
      2 
      3 submission = pd.DataFrame(
      4     {"image_id": test_df["image_id"].values, "label": test_pred_labels}
      5 )

NameError: name 'predict' is not defined

## === cell 13
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
check.head(3)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3593788507.py in <cell line: 0>()
----> 1 check = pd.read_csv("submission.csv")
      2 print(check.shape)
      3 print(check.columns.tolist())
      4 check.head(3)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
