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

0.6104563312178906

# 6. Current score

0.16256

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I remove the problematic `kaggle_datasets` import, replace the missing model load with a lightweight EfficientNetB0 model that is trained on the provided training images, and use Keras `flow_from_dataframe` for both training/validation and test data. This fixes all NameError/Filename errors, generates a proper `submission.csv` with the required columns, and provides a modest model that should achieve an accuracy near the target without altering the core competition logic.'
- What this solution (achieved 0.16256) has done: 'The changes keep the model architecture, loss, and training loops unchanged while removing unnecessary XLA and mixed‑precision overhead on CPU, adding deterministic‑False options to the datasets, and using a disk‑based cache for the training set to avoid huge memory pressure. These tweaks speed up image loading, decoding and training without altering any semantics or final predictions.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import applications, layers, models, optimizers
from tensorflow.keras.preprocessing import image as keras_image

tf.config.optimizer.set_jit(False)

tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
tf.config.threading.set_inter_op_parallelism_threads(max(1, os.cpu_count() // 2))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


tf.random.set_seed(42)
np.random.seed(42)

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train_images")
TEST_IMG_PATH = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
submission = pd.read_csv(SAMPLE_SUB)

train_df["label"] = train_df["label"].astype(str)

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (128, 128)
BATCH_SIZE = 128


def _process_path(path, label=None):
    """Read image file, decode, resize, and scale to [0,1] efficiently."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if label is None:
        return image
    else:
        return image, label


def make_dataset(df, folder, training=False):
    paths = df["image_id"].apply(lambda x: os.path.join(folder, x)).values
    labels = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    ds = ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    cache_file = "/tmp/train_cache" if training else None
    ds = ds.cache(cache_file)
    if training:
        ds = ds.shuffle(buffer_size=10000, seed=42)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, TRAIN_IMG_PATH, training=True)
val_ds = make_dataset(val_df, TRAIN_IMG_PATH, training=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2017887352.py in <cell line: 0>()
     35 
     36 train_ds = make_dataset(train_df, TRAIN_IMG_PATH, training=True)
---> 37 val_ds = make_dataset(val_df, TRAIN_IMG_PATH, training=False)
     38 

/tmp/ipykernel_11/2017887352.py in make_dataset(df, folder, training)
     26     # Use a disk‑based cache to avoid huge RAM usage
     27     cache_file = "/tmp/train_cache" if training else None
---> 28     ds = ds.cache(cache_file)
     29     if training:
     30         ds = ds.shuffle(buffer_size=10000, seed=42)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in cache(self, filename, name)
   1566     # pylint: disable=g-import-not-at-top,protected-access
   1567     from tensorflow.python.data.ops import cache_op
-> 1568     return cache_op._cache(self, filename, name)
   1569     # pylint: enable=g-import-not-at-top,protected-access
   1570 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in _cache(input_dataset, filename, name)
     24 
     25 def _cache(input_dataset, filename, name):  # pylint: disable=unused-private-name
---> 26   return CacheDataset(input_dataset, filename, name)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/cache_op.py in __init__(self, input_dataset, filename, name)
     33     """See `Dataset.cache()` for details."""
     34     self._input_dataset = input_dataset
---> 35     self._filename = ops.convert_to_tensor(
     36         filename, dtype=dtypes.string, name="filename")
     37     self._name = name

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

ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.

## === cell 2
base_model = applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,),
    weights="imagenet",
    include_top=False,
    pooling="avg",
)

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
outputs = layers.Dense(5, activation="softmax", dtype="float32")(
    x
)  # keep output in float32
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 3
EPOCHS = 10
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2401657839.py in <cell line: 0>()
      3     train_ds,
      4     epochs=EPOCHS,
----> 5     validation_data=val_ds,
      6     verbose=1,
      7 )

NameError: name 'val_ds' is not defined

## === cell 4
test_paths = (
    submission["image_id"].apply(lambda x: os.path.join(TEST_IMG_PATH, x)).values
)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
options = tf.data.Options()
options.experimental_deterministic = False
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(
    lambda p: _process_path(p), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=1)
preds = np.argmax(pred_probs, axis=-1)

submission["label"] = preds.astype(int)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
