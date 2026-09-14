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

3.10

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

0.6024478694469628

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18572) has done: 'The changes keep the same model, data splits, and training schedule, but eliminate an unnecessary extra scaling step in the image generators and enable multiprocessing data loading, which markedly reduces the I/O‑and‑preprocessing overhead that caused the 10‑minute timeout.  Removing the duplicate `rescale=1.0/255` does not alter the final pixel values because `tf.keras.applications.inception_v3.preprocess_input` already performs the required scaling.  Adding `workers` and `use_multiprocessing=True` to `model.fit` speeds up batch preparation without changing any training logic or accuracy.'
- What this solution (achieved 0.13378) has done: 'The changes add parallel data loading by setting worker processes for the ImageDataGenerator flows and for model fitting, which removes the main thread bottleneck of reading and preprocessing images from disk. This keeps the exact same model, training schedule, and augmentation logic, so results remain unchanged while reducing total runtime.'
- What this solution (achieved 0.24402) has done: 'The changes increase data‑loading parallelism (more workers, enable multiprocessing) and explicitly set `steps_per_epoch` and `validation_steps` so TensorFlow does not waste time iterating over empty batches. This keeps the same model, augmentations, training loops, and early‑stopping logic, but speeds up each epoch by reducing I/O bottlenecks.'

# 9. Code solution

## === cell 0
import os, warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
tf.random.set_seed(2020)

try:
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUBMIT_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(int)

BATCH_SIZE = 64
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
EPOCHS = 30
N_SPLITS = 1




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTOTUNE = tf.data.AUTOTUNE


def _parse_image_base(filename, label=None):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    if label is None:
        return img
    else:
        return img, label


def _augment(img_label):
    img, label = img_label
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    return img, label


def make_dataset(df, shuffle=False, augment=False):
    filepaths = [os.path.join(TRAIN_IMG_DIR, img) for img in df["image_id"]]
    int_labels = tf.constant(df["label"].values, dtype=tf.int32)
    one_hot = tf.one_hot(int_labels, depth=5, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((filepaths, one_hot))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=2020, reshuffle_each_iteration=False)
    ds = ds.map(_parse_image_base, num_parallel_calls=AUTOTUNE).cache()
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def make_test_dataset(df):
    filepaths = [os.path.join(TEST_IMG_DIR, img) for img in df["image_id"]]
    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = ds.map(_parse_image_base, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds




## === cell 2
def create_inception():
    base = InceptionV3(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    inputs = Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(5, activation="softmax", dtype="float32")(x)
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model




## === cell 3
split = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.1, random_state=2020)
for train_idx, val_idx in split.split(df_train["image_id"], df_train["label"]):
    train_df = df_train.iloc[train_idx]
    val_df = df_train.iloc[val_idx]

    train_ds = make_dataset(train_df, shuffle=True, augment=True)
    val_ds = make_dataset(val_df, shuffle=False, augment=False)

    steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

    model = create_inception()
    checkpoint_path = "/kaggle/working/inception_best.h5"
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )
    model = load_model(checkpoint_path)
    break  # only one split




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1250922967.py in <cell line: 0>()
      4     val_df = df_train.iloc[val_idx]
      5 
----> 6     train_ds = make_dataset(train_df, shuffle=True, augment=True)
      7     val_ds = make_dataset(val_df, shuffle=False, augment=False)
      8 

/tmp/ipykernel_11/882643743.py in make_dataset(df, shuffle, augment)
     33     ds = ds.map(_parse_image_base, num_parallel_calls=AUTOTUNE).cache()
     34     if augment:
---> 35         ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
     36     ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
     37     return ds

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

TypeError: in user code:


    TypeError: outer_factory.<locals>.inner_factory.<locals>.tf___augment() takes 1 positional argument but 2 were given


## === cell 4
sample_submit = pd.read_csv(SAMPLE_SUBMIT_CSV)

test_ds = make_test_dataset(sample_submit)

pred_probs = model.predict(test_ds, verbose=1)

pred_labels = np.argmax(pred_probs, axis=1)

sample_submit["label"] = pred_labels
output_path = os.path.join("/kaggle/working", "submission.csv")
sample_submit.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1450435250.py in <cell line: 0>()
      3 test_ds = make_test_dataset(sample_submit)
      4 
----> 5 pred_probs = model.predict(test_ds, verbose=1)
      6 
      7 pred_labels = np.argmax(pred_probs, axis=1)

NameError: name 'model' is not defined
