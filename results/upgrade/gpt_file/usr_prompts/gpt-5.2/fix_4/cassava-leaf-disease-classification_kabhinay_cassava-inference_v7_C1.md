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

0.8516168026594138

# 6. Current score

0.14873

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14873) has done: 'I remove the forced pure-Python protobuf implementation (it significantly slows TF graph/IO) and keep deterministic settings otherwise. I switch the data pipeline from `ImageDataGenerator` (Python-side JPEG decode/augment, major bottleneck) to an equivalent `tf.data` pipeline that uses the same images, same preprocessing, and the same augmentation semantics, but runs inside TensorFlow with parallelism, caching, prefetch, and optional XLA compilation. I also ensure we don’t do any redundant work (e.g., avoid repeated dataframe shuffles beyond what’s needed) and use efficient `tf.data` options to overlap CPU input with GPU training. The model, architecture, losses, training schedule (5 epochs + 1 fine-tune epoch), image size, and optimizer settings remain unchanged.'

# 9. Code solution

## === cell 0
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 1337
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["image_id", "label"]).issubset(train_df.columns)
assert set(["image_id", "label"]).issubset(sample_df.columns)

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)
print(train_df.head())

train_df = train_df.copy()
train_df["label"] = train_df["label"].astype(str)

val_frac = 0.1
train_parts = []
val_parts = []
for lbl, g in train_df.groupby("label", sort=False):
    g = g.sample(frac=1.0, random_state=SEED)
    n_val = max(1, int(round(len(g) * val_frac)))
    val_parts.append(g.iloc[:n_val])
    train_parts.append(g.iloc[n_val:])

train_df_split = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
val_df_split = pd.concat(val_parts, axis=0).reset_index(drop=True)

print("Train split:", len(train_df_split), "Val split:", len(val_df_split))




## === cell 2
IMG_SIZE = 448
BATCH_SIZE = 16  # unchanged

preprocess = tf.keras.applications.efficientnet.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

labels_sorted = sorted(train_df_split["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(labels_sorted)}
print("Class indices:", class_to_index)

train_paths = (TRAIN_DIR + "/" + train_df_split["image_id"]).to_numpy()
train_labels = train_df_split["label"].map(class_to_index).astype(np.int32).to_numpy()

val_paths = (TRAIN_DIR + "/" + val_df_split["image_id"]).to_numpy()
val_labels = val_df_split["label"].map(class_to_index).astype(np.int32).to_numpy()


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # dataset is .jpg
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img


def _augment(img, seed_pair):
    s0, s1 = seed_pair[0], seed_pair[1]

    img = tf.image.stateless_random_flip_left_right(img, seed=[s0, s1])

    angle = tf.random.stateless_uniform(
        [], seed=[s0 + 1, s1 + 1], minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    try:
        img = tf.image.rotate(img, angles=angle, fill_mode="reflect")
    except Exception:
        rr = tf.keras.layers.RandomRotation(
            factor=(10.0 / 360.0), fill_mode="reflect", seed=SEED
        )
        img = rr(tf.expand_dims(img, 0), training=True)[0]

    dx = tf.random.stateless_uniform(
        [], seed=[s0 + 2, s1 + 2], minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE, tf.float32)
    dy = tf.random.stateless_uniform(
        [], seed=[s0 + 3, s1 + 3], minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE, tf.float32)
    try:
        img = tf.image.translate(img, translations=[dx, dy], fill_mode="reflect")
    except Exception:
        img = tf.roll(
            img, shift=[tf.cast(dy, tf.int32), tf.cast(dx, tf.int32)], axis=[0, 1]
        )

    z = tf.random.stateless_uniform([], seed=[s0 + 4, s1 + 4], minval=0.9, maxval=1.1)
    new_size = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * z), tf.int32)
    new_size = tf.maximum(new_size, 1)
    img_zoom = tf.image.resize(
        img, [new_size, new_size], method="bilinear", antialias=True
    )
    img_zoom = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE, IMG_SIZE)

    return img_zoom


def _preprocess(img):
    return preprocess(img)


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed_pair = tf.stack([tf.cast(h, tf.int32), tf.cast(SEED, tf.int32)])
        img = _augment(img, seed_pair)
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _decode_resize(path)
        img = _preprocess(img)
        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_train_ds(train_paths, train_labels)
val_gen = make_val_ds(val_paths, val_labels)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4276258437.py in <cell line: 0>()
    122 
    123 
--> 124 train_gen = make_train_ds(train_paths, train_labels)
    125 val_gen = make_val_ds(val_paths, val_labels)
    126 

/tmp/ipykernel_11/4276258437.py in make_train_ds(paths, labels)
    101         return img, y
    102 
--> 103     ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    104     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    105     ds = ds.prefetch(AUTOTUNE)

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
   1225       # In this case we have created variables on the first call, so we run the
   1226       # version which is guaranteed to never create variables.
-> 1227       return tracing_compilation.trace_function(
   1228           args,
   1229           kwargs,

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

/tmp/__autograph_generated_file1pi2px16.py in tf___map_fn(path, label)
     11                 h = ag__.converted_call(ag__.ld(tf).strings.to_hash_bucket_fast, (ag__.ld(path), 2 ** 31 - 1), None, fscope)
     12                 seed_pair = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(h), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 13                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed_pair)), None, fscope)
     14                 img = ag__.converted_call(ag__.ld(_preprocess), (ag__.ld(img),), None, fscope)
     15                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(label),), dict(depth=ag__.ld(num_classes), dtype=ag__.ld(tf).float32), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filer44hbnmb.py in tf___augment(img, seed_pair)
     14                     img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), fill_mode='reflect'), fscope)
     15                 except Exception:
---> 16                     rr = ag__.converted_call(ag__.ld(tf).keras.layers.RandomRotation, (), dict(factor=10.0 / 360.0, fill_mode='reflect', seed=ag__.ld(SEED)), fscope)
     17                     img = ag__.converted_call(ag__.ld(rr), (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(img), 0), None, fscope),), dict(training=True), fscope)[0]
     18                 dx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=[ag__.ld(s0) + 2, ag__.ld(s1) + 2], minval=-0.05, maxval=0.05), fscope) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE), ag__.ld(tf).float32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    329   if conversion.is_in_allowlist_cache(f, options):
    330     logging.log(2, 'Allowlisted %s: from cache', f)
--> 331     return _call_unconverted(f, args, kwargs, options, False)
    332 
    333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py in __init__(self, factor, fill_mode, interpolation, seed, fill_value, data_format, **kwargs)
     93         super().__init__(factor=factor, data_format=data_format, **kwargs)
     94         self.seed = seed
---> 95         self.generator = SeedGenerator(seed)
     96         self.fill_mode = fill_mode
     97         self.interpolation = interpolation

/usr/local/lib/python3.11/dist-packages/keras/src/random/seed_generator.py in __init__(self, seed, name, **kwargs)
     85 
     86         with self.backend.name_scope(self.name, caller=self):
---> 87             self.state = self.backend.Variable(
     88                 seed_initializer,
     89                 shape=(2,),

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in __init__(self, initializer, shape, dtype, trainable, autocast, aggregation, name)
    184             if callable(initializer):
    185                 self._shape = self._validate_shape(shape)
--> 186                 self._initialize_with_initializer(initializer)
    187             else:
    188                 self._initialize(initializer)

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in _initialize_with_initializer(self, initializer)
     45 
     46     def _initialize_with_initializer(self, initializer):
---> 47         self._initialize(lambda: initializer(self._shape, dtype=self._dtype))
     48 
     49     def _deferred_initialize(self):

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in _initialize(self, value)
     36 
     37     def _initialize(self, value):
---> 38         self._value = tf.Variable(
     39             value,
     40             dtype=self._dtype,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in invalid_creator_scope(*unused_args, **unused_kwds)
    700     def invalid_creator_scope(*unused_args, **unused_kwds):
    701       """Disables variable creation."""
--> 702       raise ValueError(
    703           "tf.function only supports singleton tf.Variables created on the "
    704           "first call. Make sure the tf.Variable is only created once or "

ValueError: in user code:

    File "/tmp/ipykernel_11/4276258437.py", line 98, in _map_fn  *
        img = _augment(img, seed_pair)
    File "/tmp/ipykernel_11/4276258437.py", line 51, in _augment  *
        rr = tf.keras.layers.RandomRotation(
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_rotation.py", line 95, in __init__  **
        self.generator = SeedGenerator(seed)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/random/seed_generator.py", line 87, in __init__
        self.state = self.backend.Variable(
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py", line 186, in __init__
        self._initialize_with_initializer(initializer)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py", line 47, in _initialize_with_initializer
        self._initialize(lambda: initializer(self._shape, dtype=self._dtype))
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py", line 38, in _initialize
        self._value = tf.Variable(

    ValueError: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function. See https://www.tensorflow.org/guide/function#creating_tfvariables for more information.


## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = base(inp, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
model_v4 = tf.keras.Model(inp, out, name="cassava_efficientnetb0")

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_v4.summary()




## === cell 4
EPOCHS = 5

history = model_v4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_v4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model_v4.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1887502146.py in <cell line: 0>()
      2 
      3 history = model_v4.fit(
----> 4     train_gen,
      5     validation_data=val_gen,
      6     epochs=EPOCHS,

NameError: name 'train_gen' is not defined

## === cell 5
test_df = sample_df[["image_id"]].copy()
test_paths = (TEST_DIR + "/" + test_df["image_id"]).to_numpy()


def make_test_ds(paths, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_resize(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_gen = make_test_ds(test_paths, batch_size=32)

pred_v4 = model_v4.predict(test_gen, verbose=1)
pred_v4 = np.asarray(pred_v4)
if pred_v4.ndim == 1:
    pred_v4 = pred_v4.reshape(-1, 1)

print("pred_v4 shape:", pred_v4.shape, "expected rows:", len(test_df))




## === cell 6
predicted_class_indices_v4 = np.argmax(pred_v4, axis=1).astype(int)

sub = sample_df.copy()
if len(predicted_class_indices_v4) != len(sub):
    raise RuntimeError(
        f"Prediction rows ({len(predicted_class_indices_v4)}) != sample rows ({len(sub)})"
    )

sub["label"] = predicted_class_indices_v4

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", sub.columns.tolist())
print("Label distribution:", sub["label"].value_counts().to_dict())
