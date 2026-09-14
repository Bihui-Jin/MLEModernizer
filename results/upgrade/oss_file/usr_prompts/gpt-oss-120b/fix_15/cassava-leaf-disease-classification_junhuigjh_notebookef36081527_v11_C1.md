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

0.804472650347537

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The changes replace the slow `ImageDataGenerator` + multiprocessing pipeline with a native `tf.data` pipeline that uses TensorFlow’s built‑in data loading and augmentation layers.  This keeps the exact same ResNet‑50 architecture, loss, optimizer, early‑stopping, batch size, and number of epochs, but removes the heavy Python‑level image‑reading overhead that caused the 10‑minute timeout.  By loading the validation set once into memory and using `prefetch`/`AUTOTUNE`, the training loop runs much faster while preserving deterministic results.'
- What this solution (achieved 0.11584) has done: 'I fix the protobuf import issue by setting the appropriate environment variable before importing TensorFlow, convert label strings to integers, move the data‑augmentation layers out of the `map` call so they are created only once, and unfreeze the ResNet‑50 backbone to let the model learn better features. These changes resolve the runtime errors and should raise the validation accuracy toward the target while keeping the overall architecture unchanged.'
- What this solution (achieved 0.11584) has done: 'Implemented a robust import order fix for the protobuf‑TensorFlow clash and added a safe fallback that predicts the most frequent class if TensorFlow cannot be loaded. The environment variable is now set before **any** imports, and TensorFlow loading is wrapped in a try/except. When TensorFlow is unavailable, the script skips model training and directly creates a submission using the majority label from the training set, guaranteeing a valid `submission.csv`. This resolves the runtime error and ensures the pipeline always produces a correctly formatted output, moving the solution toward the target score while preserving the original logic when TensorFlow runs successfully.'
- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow import failure by providing a robust fallback that trains a lightweight scikit‑learn logistic‑regression model on down‑scaled images when TensorFlow cannot be loaded. This keeps the original TensorFlow pipeline unchanged for environments where it works, while the fallback dramatically improves prediction quality over the previous majority‑class guess, moving the score toward the target. I also add the necessary imports and image‑loading utilities.'
- What this solution (achieved 0.61435) has done: 'I disable the TensorFlow path (which raises a protobuf error) and replace the fallback logistic‑regression model with a stronger RandomForest classifier that works on resized image pixels. This keeps the overall pipeline structure but uses a more capable scikit‑learn model, improving validation accuracy and ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed or unavailable:", e)
    TF_AVAILABLE = False

if not TF_AVAILABLE:
    TF_AVAILABLE = False

if TF_AVAILABLE:
    tf.random.set_seed(42)
np.random.seed(42)

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "sample_submission.csv")  # contains test image IDs

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["label"] = train_df["label"].astype(int)

BATCH_SIZE = 128
AUTOTUNE = None  # not needed for the sklearn fallback

IMG_SIZE = (128, 128)


def load_and_preprocess(df, img_dir):
    """Load images, resize, normalize, and flatten into feature vectors."""
    images = []
    for img_name in df["image_id"]:
        path = os.path.join(img_dir, img_name)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
            images.append(arr.flatten())
        except Exception:
            images.append(np.zeros(IMG_SIZE[0] * IMG_SIZE[1] * 3, dtype=np.float32))
    return np.stack(images)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_split = int(0.9 * len(train_df))
train_split_df = train_df.iloc[:val_split]
val_split_df = train_df.iloc[val_split:]

if TF_AVAILABLE:

    def df_to_dataset(df, shuffle=False, augment=False):
        file_paths = (
            df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x)).values
        )
        labels = df["label"].values

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        def _load_image(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = img / 255.0
            label_one_hot = tf.one_hot(tf.cast(label, tf.int32), depth=5)
            return img, label_one_hot

        ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=1024, seed=42)

        if augment:
            ds = ds.map(
                lambda img, lbl: (augmentation_layer(img, training=True), lbl),
                num_parallel_calls=AUTOTUNE,
            )

        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_dataset = df_to_dataset(train_split_df, shuffle=True, augment=True)
    val_dataset = df_to_dataset(val_split_df, shuffle=False, augment=False)
else:
    X_train = load_and_preprocess(train_split_df, TRAIN_IMG_DIR)
    y_train = train_split_df["label"].values

    X_val = load_and_preprocess(val_split_df, TRAIN_IMG_DIR)
    y_val = val_split_df["label"].values

    mlp_pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=30,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
            verbose=False,
        ),
    )

    mlp_pipe.fit(X_train, y_train)

    val_score = mlp_pipe.score(X_val, y_val)
    print(f"Validation accuracy (MLP pipeline): {val_score:.4f}")

    model = mlp_pipe  # keep variable name consistent for downstream inference



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2491558547.py in <cell line: 0>()
     36         return ds
     37 
---> 38     train_dataset = df_to_dataset(train_split_df, shuffle=True, augment=True)
     39     val_dataset = df_to_dataset(val_split_df, shuffle=False, augment=False)
     40 else:

/tmp/ipykernel_55/2491558547.py in df_to_dataset(df, shuffle, augment)
     28 
     29         if augment:
---> 30             ds = ds.map(
     31                 lambda img, lbl: (augmentation_layer(img, training=True), lbl),
     32                 num_parallel_calls=AUTOTUNE,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     41           "`num_parallel_calls` argument is specified."
     42       )
---> 43     return _MapDataset(
     44         input_dataset,
     45         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, force_synchronous, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, name)
    155     self._use_inter_op_parallelism = use_inter_op_parallelism
    156     self._preserve_cardinality = preserve_cardinality
--> 157     self._map_func = structured_function.StructuredFunctionWrapper(
    158         map_func,
    159         self._transformation_name(),

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

/tmp/__autograph_generated_file0a1gj2_9.py in <lambda>(img, lbl)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, lbl: ag__.with_function_scope(lambda lscope: (ag__.converted_call(augmentation_layer, (img,), dict(training=True), lscope), lbl), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file0a1gj2_9.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, lbl: ag__.with_function_scope(lambda lscope: (ag__.converted_call(augmentation_layer, (img,), dict(training=True), lscope), lbl), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

NameError: in user code:

    File "/tmp/ipykernel_55/2491558547.py", line 31, in None  *
        lambda img, lbl: (augmentation_layer(img, training=True), lbl)

    NameError: name 'augmentation_layer' is not defined


## === cell 2
if TF_AVAILABLE:
    test_paths = (
        test_df["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x)).values
    )
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def _load_test_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        return img

    test_ds = test_ds.map(_load_test_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    pred_probs = model.predict(test_ds, verbose=1)
    pred_labels = np.argmax(pred_probs, axis=1)
else:
    X_test = load_and_preprocess(test_df, TEST_IMG_DIR)
    pred_labels = model.predict(X_test)

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3621322958.py in <cell line: 0>()
     15     test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
     16 
---> 17     pred_probs = model.predict(test_ds, verbose=1)
     18     pred_labels = np.argmax(pred_probs, axis=1)
     19 else:

NameError: name 'model' is not defined
