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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
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

0.8851616802659413

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.64387) has done: 'I fix the runtime failure that prevents the model from loading by switching protobuf to the pure-Python implementation *before* importing TensorFlow/tf_keras/keras, which addresses the `MessageFactory.GetPrototype` error in this environment. I also make the model path robust by falling back to running an in-notebook small CNN if the external `../input/my-model/...` file is not present, so the notebook always produces a valid `submission.csv`. To nudge accuracy upward (since you currently have no score at all), the fallback model train on the provided `train_images` with a simple stratified split and standard normalization, while preserving the original “predict then argmax” inference semantics. Finally, I ensure the submission format matches `sample_submission.csv` exactly and that the output file has a `.csv` suffix.'
- What this solution (achieved 0.63976) has done: 'I fix the protobuf/TensorFlow import ordering bug that’s still triggering `MessageFactory.GetPrototype` by moving the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting to the very first cell, before any TensorFlow/tf_keras (or anything that may transitively load protobuf) is imported. I also switch `BASE_DIR` to the correct Kaggle dataset mount (`/kaggle/input/...`) so the notebook can reliably find the images/CSVs in this environment. To improve the score toward your target with minimal core-logic disruption, I keep the same small CNN and training loop but train a bit longer (still a simple fit loop) and add lightweight in-model augmentation layers (doesn’t change inference semantics: still `predict -> argmax`). The script still always write a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"



## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}

print(json.dumps(map_classes, indent=4))



## === cell 2
import pandas as pd
import cv2



## === cell 3
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 4
img_shapes = {}
for image_name in os.listdir(os.path.join(BASE_DIR, "train_images"))[:300]:
    image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_name))
    if image is None:
        continue
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)



## === cell 5
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 6
df_train["image_id"] = df_train["image_id"].astype("str")
df_train["label"] = df_train["label"].astype("str")



## === cell 7
df_train["label"].value_counts()



## === cell 8
import numpy as np

import tensorflow as tf
import tf_keras

try:
    import keras  # Keras 3.x (optional)
except Exception as e:
    keras = None
    print(
        "Warning: keras (Keras 3) import failed; will rely on tf_keras. Error:", repr(e)
    )

print("tf version:", tf.__version__)
print("tf_keras version:", getattr(tf_keras, "__version__", "unknown"))

MODEL_PATH = "../input/my-model/Cassava_best_model.h5"
final_model = None
load_errors = []

if os.path.exists(MODEL_PATH):
    try:
        final_model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        load_errors.append(("tf_keras.models.load_model", repr(e)))

    if final_model is None and keras is not None:
        try:
            final_model = keras.saving.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            load_errors.append(("keras.saving.load_model", repr(e)))

    if final_model is None:
        print(
            "Warning: Model could not be loaded from external path; will train fallback model.\n"
            + "\n".join([f"- {src}: {err}" for src, err in load_errors])
        )
else:
    print("External model not found at:", MODEL_PATH, "-> will train fallback model.")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
try:
    if final_model is not None:
        try:
            final_model.summary()
        except Exception as e:
            print("Warning: final_model.summary() failed (non-fatal). Error:", repr(e))
except Exception:
    pass


def _infer_hw_from_model(m):
    for attr in ("input_shape", "inputs"):
        try:
            val = getattr(m, attr, None)
        except Exception:
            val = None

        if attr == "input_shape" and isinstance(val, tuple) and len(val) == 4:
            _, h, w, c = val
            return h, w, c

        if attr == "inputs" and val is not None:
            try:
                t = val[0] if isinstance(val, (list, tuple)) else val
                shp = tuple(getattr(t, "shape", ()))
                if len(shp) == 4:
                    _, h, w, c = shp
                    return h, w, c
            except Exception:
                pass

    return None, None, None


H, W, C = (None, None, None)
if final_model is not None:
    H, W, C = _infer_hw_from_model(final_model)

if H is None or W is None:
    H, W = 224, 224  # keep original fallback size
if C not in (1, 3, None):
    print(f"Warning: unexpected channel dimension C={C}; will feed RGB (3 channels).")
    C = 3

print("Using inference input size H,W,C =", H, W, C)



## === cell 10
from sklearn.model_selection import train_test_split

if final_model is None:
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=SEED,
        stratify=df["label"],
    )

    train_ids = tf.constant(train_df["image_id"].tolist())
    train_labels = tf.constant(train_df["label"].tolist(), dtype=tf.int64)
    val_ids = tf.constant(val_df["image_id"].tolist())
    val_labels = tf.constant(val_df["label"].tolist(), dtype=tf.int64)

    train_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(train_ids, train_labels),
        default_value=tf.constant(-1, dtype=tf.int64),
    )
    val_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(val_ids, val_labels),
        default_value=tf.constant(-1, dtype=tf.int64),
    )

    tfrec_train_dir = os.path.join(BASE_DIR, "train_tfrecords")
    tfrec_files = sorted(
        [
            os.path.join(tfrec_train_dir, f)
            for f in os.listdir(tfrec_train_dir)
            if f.endswith(".tfrec")
        ]
    )
    if len(tfrec_files) == 0:
        raise FileNotFoundError(f"No .tfrec files found in {tfrec_train_dir}")

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_example(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_description)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(img, [int(H), int(W)], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        image_id = tf.strings.decode(ex["image_name"], "UTF-8")
        return img, image_id

    def _attach_label(image, image_id, table):
        lbl = table.lookup(image_id)
        return image, lbl

    def _filter_labeled(image, label):
        return tf.not_equal(label, tf.constant(-1, dtype=tf.int64))

    def _make_ds_from_tfrecords(table, training):
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(
            lambda img, iid: _attach_label(img, iid, table),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.filter(_filter_labeled)

        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = _make_ds_from_tfrecords(train_table, training=True)
    ds_val = _make_ds_from_tfrecords(val_table, training=False)

    inputs = tf_keras.layers.Input(shape=(int(H), int(W), 3))
    x = tf_keras.layers.RandomFlip("horizontal")(inputs)
    x = tf_keras.layers.RandomRotation(0.05)(x)
    x = tf_keras.layers.RandomZoom(0.1)(x)

    x = tf_keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.MaxPooling2D()(x)
    x = tf_keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.MaxPooling2D()(x)
    x = tf_keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.GlobalAveragePooling2D()(x)
    x = tf_keras.layers.Dropout(0.2)(x)
    outputs = tf_keras.layers.Dense(5, activation="softmax")(x)

    final_model = tf_keras.Model(inputs, outputs)

    final_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    final_model.fit(ds_train, validation_data=ds_val, epochs=12, verbose=1)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1937563437.py in <cell line: 0>()
     86         return ds
     87 
---> 88     ds_train = _make_ds_from_tfrecords(train_table, training=True)
     89     ds_val = _make_ds_from_tfrecords(val_table, training=False)
     90 

/tmp/ipykernel_55/1937563437.py in _make_ds_from_tfrecords(table, training)
     73     def _make_ds_from_tfrecords(table, training):
     74         ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
---> 75         ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
     76         ds = ds.map(
     77             lambda img, iid: _attach_label(img, iid, table),

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

/tmp/__autograph_generated_file4gk5sy7f.py in tf___parse_example(example_proto)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.converted_call(ag__.ld(int), (ag__.ld(H),), None, fscope), ag__.converted_call(ag__.ld(int), (ag__.ld(W),), None, fscope)]), dict(method='bilinear'), fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0
---> 14                 image_id = ag__.converted_call(ag__.ld(tf).strings.decode, (ag__.ld(ex)['image_name'], 'UTF-8'), None, fscope)
     15                 try:
     16                     do_return = True

AttributeError: in user code:

    File "/tmp/ipykernel_55/1937563437.py", line 63, in _parse_example  *
        image_id = tf.strings.decode(ex["image_name"], "UTF-8")

    AttributeError: module 'tensorflow._api.v2.strings' has no attribute 'decode'


## === cell 11
sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_images = sub["image_id"].astype(str).tolist()

tfrec_test_dir = os.path.join(BASE_DIR, "test_tfrecords")
test_tfrec_files = sorted(
    [
        os.path.join(tfrec_test_dir, f)
        for f in os.listdir(tfrec_test_dir)
        if f.endswith(".tfrec")
    ]
)
if len(test_tfrec_files) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {tfrec_test_dir}")

test_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, test_feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [int(H), int(W)], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    image_id = tf.strings.decode(ex["image_name"], "UTF-8")
    return img, image_id


ds_test = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
ds_test = ds_test.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)

all_ids = []
all_preds = []

for batch_imgs, batch_ids in ds_test:
    probs = final_model.predict(batch_imgs, verbose=0)
    probs = np.asarray(probs)
    pred = np.argmax(probs, axis=-1).astype(int)
    all_preds.append(pred)
    all_ids.extend([x.decode("utf-8") for x in batch_ids.numpy().tolist()])

predictions = np.concatenate(all_preds).astype(int).tolist()

print("n_test_images (sample_submission):", len(test_images))
print("n_test_from_tfrecords:", len(all_ids))
print("n_predictions:", len(predictions))

pred_map = {iid: p for iid, p in zip(all_ids, predictions)}
ordered_preds = [int(pred_map[iid]) for iid in test_images]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/281904006.py in <cell line: 0>()
     31 
     32 ds_test = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
---> 33 ds_test = ds_test.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
     34 ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)
     35 

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

/tmp/__autograph_generated_file4v3n5gdf.py in tf___parse_test_example(example_proto)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.converted_call(ag__.ld(int), (ag__.ld(H),), None, fscope), ag__.converted_call(ag__.ld(int), (ag__.ld(W),), None, fscope)]), dict(method='bilinear'), fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0
---> 14                 image_id = ag__.converted_call(ag__.ld(tf).strings.decode, (ag__.ld(ex)['image_name'], 'UTF-8'), None, fscope)
     15                 try:
     16                     do_return = True

AttributeError: in user code:

    File "/tmp/ipykernel_55/281904006.py", line 28, in _parse_test_example  *
        image_id = tf.strings.decode(ex["image_name"], "UTF-8")

    AttributeError: module 'tensorflow._api.v2.strings' has no attribute 'decode'


## === cell 12
if len(ordered_preds) != len(test_images):
    raise ValueError(
        f"Prediction length mismatch: {len(ordered_preds)} vs {len(test_images)}"
    )

submission = pd.DataFrame({"image_id": test_images, "label": ordered_preds})
submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1952193121.py in <cell line: 0>()
----> 1 if len(ordered_preds) != len(test_images):
      2     raise ValueError(
      3         f"Prediction length mismatch: {len(ordered_preds)} vs {len(test_images)}"
      4     )
      5 

NameError: name 'ordered_preds' is not defined
