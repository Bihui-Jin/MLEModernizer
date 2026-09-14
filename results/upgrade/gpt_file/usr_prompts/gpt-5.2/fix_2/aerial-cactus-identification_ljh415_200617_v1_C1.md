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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9991

# 6. Current score

0.33364

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.33364) has done: 'I fix the environment/runtime blockers first: the protobuf/TensorFlow import crash (by forcing the safe pure-Python protobuf implementation before importing TF), and the missing `train/*.jpg` / `test/*.jpg` issue (by using the already-extracted image folders under `/kaggle/input/aerial-cactus-identification/` instead of relying on zip extraction). Then I fix the `decode_image` unknown-shape pipeline that causes `TensorShape` errors by switching to `decode_jpeg` and explicitly setting shapes and dtypes (keeping the same CNN and training loop). Finally, I ensure inference reads the correct test paths and writes a valid `submission.csv` with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

base_input = "/kaggle/input"
for dirname, _, filenames in os.walk(base_input):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DATA_ROOT = "/kaggle/input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(TRAIN_CSV), f"Missing train.csv: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv: {SAMPLE_SUB}"

print("Found:", TRAIN_DIR, TEST_DIR)



## === cell 3
df = pd.read_csv(TRAIN_CSV)
print(df.head())

file_list = df["id"].astype(str).values
has_cactus = df["has_cactus"].astype(np.int64).values
print(len(file_list), len(has_cactus))



## === cell 4
test_df = pd.read_csv(SAMPLE_SUB)
print(test_df.head())

test_fnames = test_df["id"].astype(str).values
print(len(test_fnames))



## === cell 5
data_paths_glob = glob(os.path.join(TRAIN_DIR, "*.jpg"))
test_paths_glob = glob(os.path.join(TEST_DIR, "*.jpg"))
print("Train images:", len(data_paths_glob), "Test images:", len(test_paths_glob))



## === cell 6
pa = data_paths_glob[0]
g = tf.io.read_file(pa)
im = tf.io.decode_jpeg(g, channels=3)
print(im.shape)
plt.imshow(im.numpy())
plt.axis("off")
plt.show()



## === cell 7
input_shape = (32, 32, 3)
batch_size = 32



## === cell 8
data_paths = []
for fname, label in zip(file_list, has_cactus):
    data_paths.append((os.path.join(TRAIN_DIR, fname), int(label)))

data_paths[:3], len(data_paths)



## === cell 9
data_paths[:10]




## === cell 10
def tmp_func(path_name):
    return path_name




## === cell 11
def read_data(path_name):
    img_path = path_name[0]
    label = tf.cast(path_name[1], tf.int64)

    gfile = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(gfile, channels=3)
    image = tf.ensure_shape(image, input_shape)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]

    return image, label




## === cell 12
a = tf.data.Dataset.from_tensor_slices(np.array(data_paths[:2], dtype=object))
a = a.map(tmp_func)
p = next(iter(a))
p[0].numpy().decode("utf-8"), p[1].numpy()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3556521962.py in <cell line: 0>()
----> 1 a = tf.data.Dataset.from_tensor_slices(np.array(data_paths[:2], dtype=object))
      2 a = a.map(tmp_func)
      3 p = next(iter(a))
      4 p[0].numpy().decode("utf-8"), p[1].numpy()
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in from_tensor_slices(tensors, name)
    825     # pylint: disable=g-import-not-at-top,protected-access
    826     from tensorflow.python.data.ops import from_tensor_slices_op
--> 827     return from_tensor_slices_op._from_tensor_slices(tensors, name)
    828     # pylint: enable=g-import-not-at-top,protected-access
    829 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in _from_tensor_slices(tensors, name)
     23 
     24 def _from_tensor_slices(tensors, name=None):
---> 25   return _TensorSliceDataset(tensors, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in __init__(self, element, is_files, name)
     31   def __init__(self, element, is_files=False, name=None):
     32     """See `Dataset.from_tensor_slices` for details."""
---> 33     element = structure.normalize_element(element)
     34     batched_spec = structure.type_spec_from_value(element)
     35     self._tensors = structure.to_batched_tensor_list(batched_spec, element)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    132           dtype = getattr(spec, "dtype", None)
    133           normalized_components.append(
--> 134               ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
    135   return nest.pack_sequence_as(pack_as, normalized_components)
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

## === cell 13
train_ratio = 0.8
rng = np.random.RandomState(42)
idx = np.arange(len(data_paths))
rng.shuffle(idx)

split = int(train_ratio * len(idx))
train_idx = idx[:split]
valid_idx = idx[split:]

train_paths = [data_paths[i] for i in train_idx]
valid_paths = [data_paths[i] for i in valid_idx]

len(train_paths), len(valid_paths)



## === cell 14
train_ds = tf.data.Dataset.from_tensor_slices(np.array(train_paths, dtype=object))
train_ds = train_ds.map(read_data, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(len(train_paths), reshuffle_each_iteration=True)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.repeat()
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1115906563.py in <cell line: 0>()
----> 1 train_ds = tf.data.Dataset.from_tensor_slices(np.array(train_paths, dtype=object))
      2 train_ds = train_ds.map(read_data, num_parallel_calls=tf.data.AUTOTUNE)
      3 train_ds = train_ds.shuffle(len(train_paths), reshuffle_each_iteration=True)
      4 train_ds = train_ds.batch(batch_size)
      5 train_ds = train_ds.repeat()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in from_tensor_slices(tensors, name)
    825     # pylint: disable=g-import-not-at-top,protected-access
    826     from tensorflow.python.data.ops import from_tensor_slices_op
--> 827     return from_tensor_slices_op._from_tensor_slices(tensors, name)
    828     # pylint: enable=g-import-not-at-top,protected-access
    829 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in _from_tensor_slices(tensors, name)
     23 
     24 def _from_tensor_slices(tensors, name=None):
---> 25   return _TensorSliceDataset(tensors, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in __init__(self, element, is_files, name)
     31   def __init__(self, element, is_files=False, name=None):
     32     """See `Dataset.from_tensor_slices` for details."""
---> 33     element = structure.normalize_element(element)
     34     batched_spec = structure.type_spec_from_value(element)
     35     self._tensors = structure.to_batched_tensor_list(batched_spec, element)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    132           dtype = getattr(spec, "dtype", None)
    133           normalized_components.append(
--> 134               ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
    135   return nest.pack_sequence_as(pack_as, normalized_components)
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

## === cell 15
valid_ds = tf.data.Dataset.from_tensor_slices(np.array(valid_paths, dtype=object))
valid_ds = valid_ds.map(read_data, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.batch(batch_size)
valid_ds = valid_ds.repeat()
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/245323528.py in <cell line: 0>()
----> 1 valid_ds = tf.data.Dataset.from_tensor_slices(np.array(valid_paths, dtype=object))
      2 valid_ds = valid_ds.map(read_data, num_parallel_calls=tf.data.AUTOTUNE)
      3 valid_ds = valid_ds.batch(batch_size)
      4 valid_ds = valid_ds.repeat()
      5 valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in from_tensor_slices(tensors, name)
    825     # pylint: disable=g-import-not-at-top,protected-access
    826     from tensorflow.python.data.ops import from_tensor_slices_op
--> 827     return from_tensor_slices_op._from_tensor_slices(tensors, name)
    828     # pylint: enable=g-import-not-at-top,protected-access
    829 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in _from_tensor_slices(tensors, name)
     23 
     24 def _from_tensor_slices(tensors, name=None):
---> 25   return _TensorSliceDataset(tensors, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_tensor_slices_op.py in __init__(self, element, is_files, name)
     31   def __init__(self, element, is_files=False, name=None):
     32     """See `Dataset.from_tensor_slices` for details."""
---> 33     element = structure.normalize_element(element)
     34     batched_spec = structure.type_spec_from_value(element)
     35     self._tensors = structure.to_batched_tensor_list(batched_spec, element)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/structure.py in normalize_element(element, element_signature)
    132           dtype = getattr(spec, "dtype", None)
    133           normalized_components.append(
--> 134               ops.convert_to_tensor(t, name="component_%d" % i, dtype=dtype))
    135   return nest.pack_sequence_as(pack_as, normalized_components)
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    232 
    233     if ret is None:
--> 234       ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
    235 
    236     if ret is NotImplemented:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_tensor_conversion.py in _constant_tensor_conversion_function(v, dtype, name, as_ref)
     27 
     28   _ = as_ref
---> 29   return constant_op.constant(v, dtype=dtype, name=name)
     30 
     31 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
    140   def wrapper(*args, **kwargs):
    141     if not ops.is_auto_dtype_conversion_enabled():
--> 142       return op(*args, **kwargs)
    143     bound_arguments = signature.bind(*args, **kwargs)
    144     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in constant(value, dtype, shape, name)
    274     ValueError: if called on a symbolic tensor.
    275   """
--> 276   return _constant_impl(value, dtype, shape, name, verify_shape=False,
    277                         allow_broadcast=True)
    278 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
    287       with trace.Trace("tf.constant"):
    288         return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
--> 289     return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    290 
    291   const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    299 ) -> ops._EagerTensorBase:
    300   """Creates a constant on the current device."""
--> 301   t = convert_to_eager_tensor(value, ctx, dtype)
    302   if shape is None:
    303     return t

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py in convert_to_eager_tensor(value, ctx, dtype)
    106       dtype = dtypes.as_dtype(dtype).as_datatype_enum
    107   ctx.ensure_initialized()
--> 108   return ops.EagerTensor(value, ctx.device_name, dtype)
    109 
    110 

ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

## === cell 16
inputs = layers.Input(input_shape)

net = layers.Conv2D(32, 3, 1, "SAME")(inputs)
net = layers.Activation("relu")(net)
net = layers.Conv2D(32, 3, 1, "SAME")(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Conv2D(64, 3, 1, "SAME")(net)
net = layers.Activation("relu")(net)
net = layers.Conv2D(64, 3, 1, "SAME")(net)
net = layers.Activation("relu")(net)
net = layers.MaxPooling2D((2, 2))(net)
net = layers.Dropout(0.5)(net)

net = layers.Flatten()(net)
net = layers.Dense(512)(net)
net = layers.Activation("relu")(net)
net = layers.Dropout(0.5)(net)
net = layers.Dense(1)(net)
net = layers.Activation("sigmoid")(net)

model = tf.keras.Model(inputs=inputs, outputs=net, name="cactus_cnn")



## === cell 17
model.summary()



## === cell 18
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)



## === cell 19
steps_per_epoch = len(train_paths) // batch_size
validation_steps = max(1, len(valid_paths) // batch_size)
steps_per_epoch, validation_steps



## === cell 20
hist = model.fit(
    train_ds,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1645712403.py in <cell line: 0>()
      1 hist = model.fit(
----> 2     train_ds,
      3     validation_data=valid_ds,
      4     validation_steps=validation_steps,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 21
test_df.head()



## === cell 22
test_fnames[:5]



## === cell 23
test_df["has_cactus"].head()



## === cell 24
eval_paths = [os.path.join(TEST_DIR, fname) for fname in test_fnames]
print(eval_paths[:5])

missing = [p for p in eval_paths[:50] if not os.path.exists(p)]
print("Missing in first 50:", len(missing))




## === cell 25
def image_read(path):
    g = tf.io.read_file(path)
    im = tf.io.decode_jpeg(g, channels=3)
    im = tf.ensure_shape(im, input_shape)
    im = tf.image.convert_image_dtype(im, tf.float32)
    return im




## === cell 26
from tqdm import tqdm



## === cell 27
for p in tqdm(eval_paths[:10]):
    _ = image_read(p)



## === cell 28
test_ds = tf.data.Dataset.from_tensor_slices(np.array(eval_paths, dtype=object))
test_ds = test_ds.map(image_read, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)



## === cell 29
pred = model.predict(test_ds, verbose=1)
pred.shape



## === cell 30
pred.shape



## === cell 31
pred = pred.reshape((-1,))
len(pred), pred[:5]



## === cell 32
submit_df = pd.read_csv(SAMPLE_SUB)
submit_df["has_cactus"] = pred.astype(np.float32)

submit_df.to_csv("submission.csv", index=False)
print(submit_df.head())
print("Wrote submission.csv with shape:", submit_df.shape)
