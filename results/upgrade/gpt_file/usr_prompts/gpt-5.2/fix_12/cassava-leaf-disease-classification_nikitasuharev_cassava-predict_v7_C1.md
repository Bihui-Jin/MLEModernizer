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

0.7417648836506497

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, random, math, re

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

print("Tensorflow version " + tf.__version__)

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 2
NUM_CLASSES = 5
IMG_SIZE = 512


def build_pretrained_app(
    app_ctor,
    preprocess_fn,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
):
    inputs = keras.Input(shape=input_shape)
    x = preprocess_fn(inputs)
    base = app_ctor(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(
        base.output
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


dense201 = build_pretrained_app(
    keras.applications.DenseNet201, keras.applications.densenet.preprocess_input
)
inception = build_pretrained_app(
    keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input
)
efficient_net = build_pretrained_app(
    keras.applications.EfficientNetB3, keras.applications.efficientnet.preprocess_input
)

_GPUS = tf.config.list_logical_devices("GPU")
if len(_GPUS) > 1:
    STRATEGY = tf.distribute.MirroredStrategy()
else:
    STRATEGY = tf.distribute.get_strategy()



## === cell 3
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
JPEG_PATH = "../input/cassava-leaf-disease-classification/test_images"


def _build_test_dataset_from_tfrecords(tfrecord_glob, batch_size):
    files = tf.io.gfile.glob(tfrecord_glob)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for glob: {tfrecord_glob}")

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_example(example_proto):
        x = tf.io.parse_single_example(example_proto, feature_description)
        img = tf.io.decode_jpeg(
            x["image"], channels=3, fancy_upscaling=False, dct_method="INTEGER_FAST"
        )
        img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1] float32
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False
        )
        return img, x["image_id"]

    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_threading.private_threadpool_size = 0
        opts.experimental_threading.max_intra_op_parallelism = 0
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=tf.data.AUTOTUNE, compression_type=None
    )
    ds = ds.with_options(opts)
    ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_test_dataset_from_files(jpeg_path, image_ids, batch_size):
    paths = tf.strings.join([tf.constant(jpeg_path + "/"), tf.constant(image_ids)])
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _decode_resize(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(
            img_bytes, channels=3, fancy_upscaling=False, dct_method="INTEGER_FAST"
        )
        img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0,1] float32
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False
        )
        return img

    opts = tf.data.Options()
    opts.deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_threading.private_threadpool_size = 0
        opts.experimental_threading.max_intra_op_parallelism = 0
    except Exception:
        pass

    ds = ds.with_options(opts)
    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_image_ids = np.sort(submission.image_id.values)




## === cell 5
def vote_in_ensemble(v1, v2, v3):
    if v1 == v2:
        return v1
    if v2 == v3:
        return v2
    if v1 == v3:
        return v3
    return v1




## === cell 6
BATCH_SIZE = 64

ds_test = _build_test_dataset_from_tfrecords(TEST_TFREC_GLOB, batch_size=BATCH_SIZE)

n = len(test_image_ids)


@tf.function(
    jit_compile=True,
    input_signature=[
        tf.TensorSpec(shape=[None, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
    ],
)
def _ensemble_argmax_stacked(batch):
    d = dense201(batch, training=False)
    i = inception(batch, training=False)
    e = efficient_net(batch, training=False)
    d = tf.argmax(d, axis=-1, output_type=tf.int32)
    i = tf.argmax(i, axis=-1, output_type=tf.int32)
    e = tf.argmax(e, axis=-1, output_type=tf.int32)
    return tf.stack([d, i, e], axis=0)  # [3, B]


preds_by_id = {}

if isinstance(STRATEGY, tf.distribute.MirroredStrategy):
    dist_ds = STRATEGY.experimental_distribute_dataset(ds_test)

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[None, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
        ],
    )
    def _dist_step(x):
        stacked = STRATEGY.run(_ensemble_argmax_stacked, args=(x,))
        stacked = STRATEGY.gather(stacked, axis=1)  # -> [3, B_total]
        return stacked

    for batch_imgs, batch_ids in dist_ds:
        stacked = _dist_step(batch_imgs).numpy()
        a = stacked[0]
        b = stacked[1]
        c = stacked[2]
        voted = np.where(a == b, a, np.where(b == c, b, np.where(a == c, c, a))).astype(
            np.int32
        )
        ids_np = (
            tf.concat(batch_ids.values, axis=0).numpy()
            if isinstance(batch_ids, tf.distribute.DistributedValues)
            else batch_ids.numpy()
        )
        for _id, _p in zip(ids_np, voted.tolist()):
            preds_by_id[_id.decode("utf-8")] = int(_p)
else:
    for batch_imgs, batch_ids in ds_test:
        stacked = _ensemble_argmax_stacked(batch_imgs).numpy()
        a = stacked[0]
        b = stacked[1]
        c = stacked[2]
        voted = np.where(a == b, a, np.where(b == c, b, np.where(a == c, c, a))).astype(
            np.int32
        )
        ids_np = batch_ids.numpy()
        for _id, _p in zip(ids_np, voted.tolist()):
            preds_by_id[_id.decode("utf-8")] = int(_p)

result = [preds_by_id[iid] for iid in submission["image_id"].tolist()]

del ds_test
gc.collect()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2454993278.py in <cell line: 0>()
     62             preds_by_id[_id.decode("utf-8")] = int(_p)
     63 else:
---> 64     for batch_imgs, batch_ids in ds_test:
     65         stacked = _ensemble_argmax_stacked(batch_imgs).numpy()
     66         a = stacked[0]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 7
pred_map = dict(zip(submission["image_id"].tolist(), result))
submission["label"] = submission["image_id"].map(pred_map).astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == 2676 and list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3591122170.py in <cell line: 0>()
----> 1 pred_map = dict(zip(submission["image_id"].tolist(), result))
      2 submission["label"] = submission["image_id"].map(pred_map).astype(int)
      3 
      4 submission.to_csv("submission.csv", index=False)
      5 

NameError: name 'result' is not defined
