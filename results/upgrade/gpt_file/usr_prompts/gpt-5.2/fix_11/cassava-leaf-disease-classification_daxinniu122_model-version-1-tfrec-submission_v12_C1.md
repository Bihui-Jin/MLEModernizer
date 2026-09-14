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

0.8541855545482019

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Main runtime is being wasted by (1) per-batch `model.predict()` overhead inside a Python loop, (2) slow JPEG decode/resize in the input pipeline without caching/deterministic options, and (3) Python-side byte-decoding/list appends. The core logic (same TFRecord parsing, same preprocessing, same ResNet50 model, same argmax predictions, same merge with sample_submission) is preserved, but inference is executed as a single graph `model(test_ds)` pass with `tf.data` optimizations (deterministic, map fusion, caching) and vectorized ID decoding at the end. These changes reduce Python overhead and maximize pipeline throughput while keeping outputs equivalent up to negligible floating-point differences. File paths and submission semantics remain unchanged.'
- What this solution (achieved 0.60762) has done: 'The timeout is dominated by decoding/resizing 2676 images to 512×512 and running ResNet50 inference, plus an extra full dataset pass just to re-collect image ids. I remove the redundant second pass by having the dataset yield deterministic integer indices and then map predictions back to `sample_submission.image_id` directly (same ordering as TFRecords), which preserves evaluation semantics while cutting runtime almost in half. I also switch dataset caching from in-memory (`cache()`) to no-cache (for a single pass it only adds overhead/memory pressure) and enable standard TF data optimizations while keeping deterministic ordering and identical preprocessing/model behavior. Model architecture/inference remains unchanged; only the input pipeline and post-processing are optimized.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
OUT_PATH = "/kaggle/working/submission.csv"

TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"
MODEL_PATH = "/kaggle/input/resnet50-ver-1/ResNet50_ver_1.h5"

IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # inference batch size

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Sample submission path exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Test tfrecord dir exists:", os.path.isdir(TEST_TFREC_DIR))
print("Model path exists:", os.path.exists(MODEL_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "Unexpected sample_submission columns"

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

preprocess_input = tf.keras.applications.resnet50.preprocess_input


def _decode_and_preprocess(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.image.decode_and_resize_jpeg(
        ex["image"], img_size=IMG_SIZE, channels=3, method="bilinear"
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # ResNet50 preprocess expects float32 0..255 RGB
    image_id = ex["image_name"]
    return img, image_id


def make_test_dataset_with_index(tfrecord_paths, batch_size):
    options = tf.data.Options()
    options.deterministic = True

    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True

    ds = tf.data.TFRecordDataset(
        tfrecord_paths,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=64 * 1024 * 1024,
    ).with_options(options)

    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_tfrecords = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(test_tfrecords) > 0, f"No TFRecords found in {TEST_TFREC_DIR}"

test_ds = make_test_dataset_with_index(test_tfrecords, BATCH_SIZE)

print("Found TFRecords:", len(test_tfrecords))
print("Sample submission rows:", len(sample_sub))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3522171211.py in <cell line: 0>()
     52 assert len(test_tfrecords) > 0, f"No TFRecords found in {TEST_TFREC_DIR}"
     53 
---> 54 test_ds = make_test_dataset_with_index(test_tfrecords, BATCH_SIZE)
     55 
     56 print("Found TFRecords:", len(test_tfrecords))

/tmp/ipykernel_11/3522171211.py in make_test_dataset_with_index(tfrecord_paths, batch_size)
     43     ).with_options(options)
     44 
---> 45     ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
     46     ds = ds.batch(batch_size, drop_remainder=False)
     47     ds = ds.prefetch(tf.data.AUTOTUNE)

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

/tmp/__autograph_generated_filepfiz8res.py in tf___decode_and_preprocess(example_proto)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example_proto), ag__.ld(_FEATURES)), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_resize_jpeg, (ag__.ld(ex)['image'],), dict(img_size=ag__.ld(IMG_SIZE), channels=3, method='bilinear'), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(preprocess_input), (ag__.ld(img),), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/3522171211.py", line 19, in _decode_and_preprocess  *
        img = tf.image.decode_and_resize_jpeg(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'decode_and_resize_jpeg'


## === cell 2
def build_resnet50_head(input_shape=(512, 512, 3), num_classes=5):
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        pooling="avg",
    )
    x = keras.layers.Dropout(0.2)(base.output)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=out)
    return model


def load_resnet50_model_safely(model_path: str) -> tf.keras.Model:
    if model_path and os.path.exists(model_path):
        try:
            m = tf.keras.models.load_model(model_path, compile=False)
            print("Loaded model via tf.keras.models.load_model:", model_path)
            return m
        except Exception as e:
            print("load_model failed, attempting weights-only load. Error:", repr(e))
            m = build_resnet50_head(
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=5
            )
            m.load_weights(model_path)
            print("Loaded weights into fallback architecture:", model_path)
            return m
    else:
        print(
            f"WARNING: model file not found at {model_path}. Using untrained fallback model so pipeline completes."
        )
        return build_resnet50_head(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=5
        )


model = load_resnet50_model_safely(MODEL_PATH)



## === cell 3
num_test = len(sample_sub)
steps = (num_test + BATCH_SIZE - 1) // BATCH_SIZE

pred_probs = model.predict(test_ds, steps=steps, verbose=0)
pred_probs = np.asarray(pred_probs)[:num_test]
preds = np.argmax(pred_probs, axis=1).astype(np.int32)

id_list = []
seen = 0
for _imgs, batch_ids in test_ds:
    ids = batch_ids.numpy().tolist()
    id_list.extend(
        [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in ids
        ]
    )
    seen += len(ids)
    if seen >= num_test:
        break
id_list = id_list[:num_test]

assert len(preds) == num_test, "Prediction count must match sample_submission rows"
assert len(id_list) == num_test, "ID count must match sample_submission rows"

sub_df = pd.DataFrame({"image_id": id_list, "label": preds.astype(int)})

assert len(sub_df) == len(sample_sub), "Submission length mismatch vs sample_submission"
assert sub_df["label"].between(0, 4).all(), "Predicted labels must be in [0,4]"

sub_df.to_csv(OUT_PATH, index=False)

print("Wrote:", OUT_PATH)
print(sub_df.head())
print("Rows:", len(sub_df))
print("Label distribution:\n", sub_df["label"].value_counts().sort_index())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1084083876.py in <cell line: 0>()
      4 # Speed-critical: use model.predict over the dataset pipeline to reduce Python overhead.
      5 # To preserve ordering and avoid extra memory, stream IDs in the same dataset order and trim to num_test.
----> 6 pred_probs = model.predict(test_ds, steps=steps, verbose=0)
      7 pred_probs = np.asarray(pred_probs)[:num_test]
      8 preds = np.argmax(pred_probs, axis=1).astype(np.int32)

NameError: name 'test_ds' is not defined
