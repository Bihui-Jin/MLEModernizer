# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, gc, random, math, re

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
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

        image_id = tf.where(
            tf.strings.length(x["image_id"]) > 0, x["image_id"], x["id"]
        )
        return img, image_id

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

test_image_ids = submission["image_id"].values  # keep exact submission order




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


def _predict_with_ids_from_ds(ds):
    preds_by_id_local = {}

    if isinstance(STRATEGY, tf.distribute.MirroredStrategy):
        dist_ds = STRATEGY.experimental_distribute_dataset(ds)

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
            a, b, c = stacked[0], stacked[1], stacked[2]
            voted = np.where(
                a == b, a, np.where(b == c, b, np.where(a == c, c, a))
            ).astype(np.int32)

            ids_np = (
                tf.concat(batch_ids.values, axis=0).numpy()
                if isinstance(batch_ids, tf.distribute.DistributedValues)
                else batch_ids.numpy()
            )
            for _id, _p in zip(ids_np, voted.tolist()):
                _id = _id.decode("utf-8")
                preds_by_id_local[_id] = int(_p)
    else:
        for batch_imgs, batch_ids in ds:
            stacked = _ensemble_argmax_stacked(batch_imgs).numpy()
            a, b, c = stacked[0], stacked[1], stacked[2]
            voted = np.where(
                a == b, a, np.where(b == c, b, np.where(a == c, c, a))
            ).astype(np.int32)
            ids_np = batch_ids.numpy()
            for _id, _p in zip(ids_np, voted.tolist()):
                _id = _id.decode("utf-8")
                preds_by_id_local[_id] = int(_p)

    return preds_by_id_local


preds_by_id = _predict_with_ids_from_ds(ds_test)

need_fallback = (len(preds_by_id) == 0) or any(
    (k is None) or (k == "") for k in preds_by_id.keys()
)
if need_fallback:
    ds_test_files = _build_test_dataset_from_files(
        JPEG_PATH, test_image_ids, batch_size=BATCH_SIZE
    )
    preds = []
    for batch_imgs in ds_test_files:
        stacked = _ensemble_argmax_stacked(batch_imgs).numpy()
        a, b, c = stacked[0], stacked[1], stacked[2]
        voted = np.where(a == b, a, np.where(b == c, b, np.where(a == c, c, a))).astype(
            np.int32
        )
        preds.extend(voted.tolist())
    result = preds[: len(test_image_ids)]
else:
    result = [preds_by_id[iid] for iid in submission["image_id"].tolist()]

del ds_test
gc.collect()




## === cell 7
pred_map = dict(zip(submission["image_id"].tolist(), result))
submission["label"] = submission["image_id"].map(pred_map).astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == 2676 and list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")
