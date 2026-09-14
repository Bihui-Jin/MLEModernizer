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

3.13

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

0.8899969779389544

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow imports and model‑loading code, replace it with a simple baseline that predicts the most frequent class from the training labels for every test image, and ensure the script always writes a correctly formatted `submission.csv`. This fixes the import error, handles missing model files, and guarantees a valid submission file.'
- What this solution (achieved 0.61099) has done: 'I replace the baseline “most‑common class” logic with a lightweight transfer‑learning model: a frozen EfficientNetB0 backbone extracts 1280‑dimensional features from resized images, a single trainable dense layer maps them to the five disease classes, and the model is trained for a few epochs on a random subset of the training data to boost accuracy toward the target. If TensorFlow cannot be imported, the script falls back to the original most‑common‑class prediction to still produce a valid `submission.csv`. The updated pipeline also ensures the submission file has the correct format.'
- What this solution (achieved 0.61099) has done: 'The fix adds an environment variable before importing TensorFlow to avoid the protobuf `MessageFactory` AttributeError, allowing the TensorFlow‑based EfficientNet model to run. All other logic remains unchanged, so the script can now train the lightweight model and produce a properly formatted `submission.csv` with improved accuracy.'
- What this solution (achieved 0.61099) has done: 'I fix the shape argument for the Input layer (it must be a tuple, not an int) so the model can be built, which also resolves the later “model not defined” error. No other logic changes are needed; the rest of the pipeline and fallback logic remain intact, ensuring a valid `submission.csv` is produced and the score can improve toward the target.'
- What this solution (achieved 0.61099) has done: 'The update raises the training epochs from 5 to 15 and expands the classifier by adding a hidden 256‑unit ReLU layer before the final softmax. These minimal adjustments keep the same EfficientNet‑B0 feature extractor and overall pipeline while allowing the model to learn richer patterns, which should raise the validation accuracy and move the score closer to the target 0.8899. All other logic, fallback handling, and submission creation remain unchanged.'
- What this solution (achieved 0.61099) has done: 'I increase the batch size to halve the number of iteration steps and remove the in‑memory `.cache()` call, which unnecessarily loads all images at once and can cause swapping. Both changes keep the exact model architecture, training loops, and data splits, so the predictions remain identical while reducing I/O and CPU overhead, allowing the whole pipeline to finish well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

import pandas as pd
import numpy as np
import random
import json

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf_available = True
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

except Exception:
    tf_available = False




## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_path = "/kaggle/working/submission.csv"

train_df = pd.read_csv(train_csv_path)
most_common_label = train_df["label"].mode()[0]  # fallback label

train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_image_dir, x)
)




## === cell 2
if tf_available:
    IMG_SIZE = 224
    BATCH_SIZE = 64
    AUTOTUNE = tf.data.AUTOTUNE
    SUBSET_SIZE = len(train_df)  # full set
    VALID_SPLIT = 0.1
    EPOCHS = 40  # a bit longer training

    subset_df = train_df.sample(n=SUBSET_SIZE, random_state=42)

    val_frac = int(len(subset_df) * VALID_SPLIT)
    val_df = subset_df.iloc[:val_frac]
    train_subset_df = subset_df.iloc[val_frac:]

    def decode_and_resize(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image / 255.0
        return image, label

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
        ]
    )

    def make_dataset(df, augment=False):
        paths = df["image_path"].values
        labels = df["label"].values.astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(
                lambda img, lbl: (augment(img, training=True), lbl),
                num_parallel_calls=AUTOTUNE,
            )
        ds = ds.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_ds = make_dataset(train_subset_df, augment=True)
    val_ds = make_dataset(val_df, augment=False)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    def images_only(ds):
        return ds.map(lambda img, lbl: img, num_parallel_calls=AUTOTUNE)

    train_features = base_model.predict(images_only(train_ds), verbose=0)
    val_features = base_model.predict(images_only(val_ds), verbose=0)

    train_labels = train_subset_df["label"].values.astype(np.int32)
    val_labels = val_df["label"].values.astype(np.int32)

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(train_features.shape[1],)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.1),  # reduced dropout for better fitting
            layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_feat_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_feat_ds = train_feat_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_feat_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    val_feat_ds = val_feat_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    model.fit(train_feat_ds, validation_data=val_feat_ds, epochs=EPOCHS, verbose=2)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1826134001.py in <cell line: 0>()
     41         return ds
     42 
---> 43     train_ds = make_dataset(train_subset_df, augment=True)
     44     val_ds = make_dataset(val_df, augment=False)
     45 

/tmp/ipykernel_11/1826134001.py in make_dataset(df, augment)
     34         ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
     35         if augment:
---> 36             ds = ds.map(
     37                 lambda img, lbl: (augment(img, training=True), lbl),
     38                 num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_fileb4zufbz7.py in <lambda>(img, lbl)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda img, lbl: ag__.with_function_scope(lambda lscope: (ag__.converted_call(augment, (img,), dict(training=True), lscope), lbl), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileb4zufbz7.py in <lambda>(lscope)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda img, lbl: ag__.with_function_scope(lambda lscope: (ag__.converted_call(augment, (img,), dict(training=True), lscope), lbl), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    367       return py_builtins.locals_in_original_context(caller_fn_scope)
    368     if kwargs:
--> 369       return py_builtins.overload_of(f)(*args, **kwargs)
    370     else:
    371       return py_builtins.overload_of(f)(*args)

TypeError: in user code:

    File "/tmp/ipykernel_11/1826134001.py", line 37, in None  *
        lambda img, lbl: (augment(img, training=True), lbl)

    TypeError: 'bool' object is not callable


## === cell 3
if tf_available:
    test_files = [
        f
        for f in os.listdir(test_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    test_paths = [os.path.join(test_image_dir, f) for f in test_files]

    def preprocess_test(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image / 255.0
        return image

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(preprocess_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_features = base_model.predict(test_ds, verbose=0)

    preds = model.predict(test_features, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    submission_df = pd.DataFrame(
        {"image_id": test_files, "label": pred_labels.astype(int)}
    )
else:
    image_predictions = []
    for image_name in os.listdir(test_image_dir):
        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_predictions.append(
                {"image_id": image_name, "label": int(most_common_label)}
            )
    if not image_predictions:  # fallback to sample submission IDs
        sample_df = pd.read_csv(sample_submission_path)
        image_predictions = [
            {"image_id": row["image_id"], "label": int(most_common_label)}
            for _, row in sample_df.iterrows()
        ]
    submission_df = pd.DataFrame(image_predictions)

submission_df.to_csv(submission_path, index=False)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2229686351.py in <cell line: 0>()
     18     test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
     19 
---> 20     test_features = base_model.predict(test_ds, verbose=0)
     21 
     22     preds = model.predict(test_features, verbose=0)

NameError: name 'base_model' is not defined

## === cell 4
submission_df.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/923863559.py in <cell line: 0>()
----> 1 submission_df.head()

NameError: name 'submission_df' is not defined
