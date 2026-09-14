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

0.7858869749168933

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix two blockers that prevent any submission from being generated: (1) an environment-level protobuf/TensorFlow import crash that happens before your code runs, and (2) the missing external weight file that makes `load_model()` fail, which then cascades into `my_model` not being defined. To keep core inference logic intact (ImageDataGenerator → model.predict → argmax → submission), I add a safe fallback that builds a simple Keras model only when the provided `.h5` path is unavailable, so the notebook always runs end-to-end and writes `submission.csv`. I also make the code robust to either of the duplicated dataset directory layouts in your environment by selecting the first existing DATA_ROOT. These changes are strictly to unblock execution and produce a valid submission; they don’t change your prediction pipeline semantics when the weight file is actually present.'
- What this solution (achieved 0.10762) has done: 'I remove the hard failure on the missing external `.h5` weight file and instead search for it; if it’s still not available, I fall back to a small deterministic Keras model so `my_model` is always defined and the pipeline can generate `submission.csv` end-to-end. This directly fixes the `FileNotFoundError`/`NameError` chain and guarantees a valid submission file in the correct format. I also make the model input shape align with your generator’s `target_size=(512,512)` to avoid shape/runtime issues during `predict`. These changes preserve your core inference semantics (ImageDataGenerator → model.predict → argmax → submission) and are only to unblock execution when the external weights aren’t attached.'
- What this solution (achieved 0.1932) has done: 'Your current score (0.10762) is far below the target (0.7859), and the biggest reason is that when the `.h5` weights aren’t found you fall back to an untrained tiny CNN, which produce near-random predictions. To move toward the target while preserving your core inference semantics (ImageDataGenerator → model.predict → argmax → submission), I keep the exact pipeline but replace the fallback with a pretrained ImageNet backbone (EfficientNetB0) and a deterministic 5-class head; this is a minimal change that dramatically improves accuracy without changing your loops or post-processing. I also automatically adapt `target_size` and the model’s input shape to whatever the loaded model expects (so if the `.h5` is present, nothing changes), and I add the standard EfficientNet `preprocess_input` in the generator only when using the pretrained fallback. These changes keep runtime within limits and still write a valid `submission.csv`.'
- What this solution (achieved 0.30157) has done: 'Your current gap to the target is large (0.1932 → 0.7859), and the biggest limiter is that the EfficientNetB0 “fallback” is being used for inference without any cassava-specific training, so accuracy stays near chance. To move the score upward while keeping your inference pipeline identical (ImageDataGenerator → model.predict → argmax → submission), I add a minimal training step *only when the external `.h5` isn’t found*: train a small softmax head on top of the frozen ImageNet EfficientNetB0 using `train.csv` and the provided `train_images/`. I also use the same EfficientNet preprocessing for both training and test generators in fallback mode, and keep everything deterministic via fixed seeds. If the `.h5` is present, behavior remains the same as your current script (no retraining, same loading/inference).'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
DEBUG = False

np.random.seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def _pick_data_root(candidates):
    for p in candidates:
        if os.path.exists(os.path.join(p, "sample_submission.csv")) and os.path.exists(
            os.path.join(p, "test_images")
        ):
            return p
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


DATA_ROOT = _pick_data_root(DATA_ROOT_CANDIDATES)

TEST_IMG_GLOB = os.path.join(DATA_ROOT, "test_images", "*.jpg")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

WEIGHT_PATH = (
    "/kaggle/input/experiment-with-models-using-keras-with-updates/model_v0.25.h5"
)

print("Python:", sys.version)
print("DATA_ROOT:", DATA_ROOT, "exists:", os.path.exists(DATA_ROOT))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Train csv exists:", os.path.exists(TRAIN_CSV_PATH))
print("Train image dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test images glob example:", TEST_IMG_GLOB)
print("Weight path exists:", os.path.exists(WEIGHT_PATH))




## === cell 1
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF decide
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TensorFlow:", tf.__version__)

custom_objects = {}

try:
    from tensorflow.keras.layers import DepthwiseConv2D

    custom_objects["DepthwiseConv2D"] = DepthwiseConv2D
except Exception:
    pass

try:
    custom_objects["swish"] = tf.nn.swish
except Exception:
    pass


class FixedDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


custom_objects["FixedDropout"] = FixedDropout


def _find_weight_file(preferred_path: str):
    """Try to locate the .h5 if the originally referenced dataset isn't attached."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    base_dir = "/kaggle/input"
    target_name = "model_v0.25.h5"
    if os.path.isdir(base_dir):
        try:
            for d1 in os.listdir(base_dir):
                p1 = os.path.join(base_dir, d1)
                if not os.path.isdir(p1):
                    continue
                cand = os.path.join(p1, target_name)
                if os.path.exists(cand):
                    return cand
                try:
                    for d2 in os.listdir(p1):
                        p2 = os.path.join(p1, d2)
                        if not os.path.isdir(p2):
                            continue
                        cand2 = os.path.join(p2, target_name)
                        if os.path.exists(cand2):
                            return cand2
                except Exception:
                    continue
        except Exception:
            pass

    hits = glob.glob(os.path.join(base_dir, "*", "*.h5"))
    for h in hits:
        if os.path.basename(h) == target_name:
            return h
    return None


resolved_weight_path = _find_weight_file(WEIGHT_PATH)
print("Resolved weight path:", resolved_weight_path)

if resolved_weight_path is None or (not os.path.exists(resolved_weight_path)):
    raise FileNotFoundError(
        "Pretrained weight file model_v0.25.h5 not found. "
        "Fallback training is disabled to guarantee the 600s timeout without changing "
        "core logic/accuracy. Please attach the dataset containing the weights or "
        f"place the file at {WEIGHT_PATH}."
    )

my_model = load_model(
    resolved_weight_path, custom_objects=custom_objects, compile=False
)
print("Model loaded. Output shape:", my_model.output_shape)

MODEL_TARGET_SIZE = (512, 512)  # overridden if model expects a different size
MODEL_INPUT_SHAPE = (512, 512, 3)

try:
    inferred = my_model.input_shape
    if isinstance(inferred, list):
        inferred = inferred[0]
    if inferred is not None and len(inferred) == 4:
        h, w, c = inferred[1], inferred[2], inferred[3]
        if (h is not None) and (w is not None) and (c == 3):
            MODEL_TARGET_SIZE = (int(h), int(w))
            MODEL_INPUT_SHAPE = (int(h), int(w), 3)
except Exception as e:
    print("WARNING: could not infer model input shape; using default 512x512.", e)

USE_EFFNET_FALLBACK = False
IDX_TO_LABEL = None

print(
    "Final MODEL_TARGET_SIZE:",
    MODEL_TARGET_SIZE,
    "MODEL_INPUT_SHAPE:",
    MODEL_INPUT_SHAPE,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _train_head_if_needed():
    return


_train_head_if_needed()




## === cell 3
test_images = sorted(glob.glob(TEST_IMG_GLOB))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found via glob: {TEST_IMG_GLOB}")

df_test = pd.DataFrame({"path": test_images})
df_test["image_id"] = [os.path.basename(p) for p in df_test["path"].values]

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    df_test = sample_sub[["image_id"]].merge(
        df_test[["image_id", "path"]], on="image_id", how="left"
    )
    if df_test["path"].isna().any():
        missing = df_test[df_test["path"].isna()]["image_id"].head(5).tolist()
        raise FileNotFoundError(
            f"Some sample_submission image_ids not found in test_images. Example missing: {missing}"
        )


def make_test_gen(batch_size=64):
    options = tf.data.Options()
    options.experimental_deterministic = True

    def _decode_resize(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img, MODEL_TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR
        )
        return img

    paths = df_test["path"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


print("Test images:", len(df_test))
print("Example test row:", df_test.head(1).to_dict(orient="records")[0])




## === cell 4
pred_list = []
for _ in range(1):
    test_gen = make_test_gen(batch_size=128)

    pred_test = my_model.predict(
        test_gen,
        verbose=1,
    )
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_idx = np.argmax(pred_test, axis=-1).astype(int)

pred_test_labels = pred_test_idx

if pred_test_labels.shape[0] != len(df_test):
    raise ValueError(
        f"Predictions count {pred_test_labels.shape[0]} != test rows {len(df_test)}"
    )

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

final_csv["image_id"] = final_csv["image_id"].astype(str)
final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/902640763.py in <cell line: 0>()
      1 pred_list = []
      2 for _ in range(1):
----> 3     test_gen = make_test_gen(batch_size=128)
      4 
      5     pred_test = my_model.predict(

/tmp/ipykernel_11/1194592092.py in make_test_gen(batch_size)
     36     paths = df_test["path"].astype(str).values
     37     ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
---> 38     ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
     39     # Safe in-memory cache: test set is ~2.7k images; caching resized tensors fits typical RAM.
     40     ds = ds.cache()

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

/tmp/__autograph_generated_file5ra6jl6d.py in tf___decode_resize(path)
     10                 img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(img),), dict(channels=3), fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(MODEL_TARGET_SIZE)), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)
     13                 try:
     14                     do_return = True

NameError: in user code:

    File "/tmp/ipykernel_11/1194592092.py", line 31, in _decode_resize  *
        img = tf.image.resize(

    NameError: name 'MODEL_TARGET_SIZE' is not defined


## === cell 5
print(final_csv["label"].value_counts().sort_index())
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv size (bytes):", os.path.getsize("submission.csv"))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/698219458.py in <cell line: 0>()
----> 1 print(final_csv["label"].value_counts().sort_index())
      2 print("submission.csv exists:", os.path.exists("submission.csv"))
      3 print("submission.csv size (bytes):", os.path.getsize("submission.csv"))

NameError: name 'final_csv' is not defined
