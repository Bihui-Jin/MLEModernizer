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
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import gc, random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
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
IMG_SIZE = (512, 512)
N_CLASSES = 5


def build_densenet201():
    base = tf.keras.applications.DenseNet201(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    x = base.output
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="cassava_head")(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=x, name="densenet201_imagenet_to_5"
    )
    return model


def build_inceptionv3():
    base = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    x = base.output
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="cassava_head")(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=x, name="inceptionv3_imagenet_to_5"
    )
    return model


def build_efficientnetb3():
    base = tf.keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3), pooling="avg"
    )
    x = base.output
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="cassava_head")(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=x, name="efficientnetb3_imagenet_to_5"
    )
    return model




## === cell 3
JPEG_PATH = "../input/cassava-leaf-disease-classification/test_images"
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_normalize_image_only(image_id):
    path = tf.strings.join([JPEG_PATH, "/", image_id])
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # RGB
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_test_images_dataset(image_ids, batch_size, cache_path=None):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)

    options = tf.data.Options()
    options.deterministic = True
    options.autotune.enabled = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    try:
        options.threading.private_threadpool_size = 0
        options.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(
        _decode_resize_normalize_image_only,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    if cache_path is None:
        ds = ds.cache()
    else:
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_ids = submission["image_id"].values
n_test = len(test_image_ids)
print("Test images:", n_test)
print("First test id:", test_image_ids[0])




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

cache_path = os.path.join(
    "/kaggle/working", f"cassava_test_cache_{IMG_SIZE[0]}x{IMG_SIZE[1]}"
)

ds_images = make_test_images_dataset(
    test_image_ids, batch_size=BATCH_SIZE, cache_path=cache_path
)


def _maybe_load_weights(model, candidate_paths):
    for p in candidate_paths:
        if p and tf.io.gfile.exists(p):
            model.load_weights(p)
            print(f"Loaded weights: {p}")
            return True
    print("No external weights found; using ImageNet backbone + random head as-is.")
    return False


def predict_labels_for_model(build_fn, ds_images_only, n_items, weight_paths=()):
    model = build_fn()

    _maybe_load_weights(model, weight_paths)

    probs = model.predict(ds_images_only, verbose=0)
    preds = np.argmax(probs, axis=-1).astype(np.int64, copy=False)

    if preds.shape[0] != n_items:
        preds = preds[:n_items]

    del model
    gc.collect()
    return preds


dense_weight_candidates = (
    "../input/cassava-leaf-disease-classification/densenet201.h5",
    "../input/cassava-leaf-disease-classification/densenet201_weights.h5",
    "../input/cassava-leaf-disease-classification/densenet201.ckpt",
)
inception_weight_candidates = (
    "../input/cassava-leaf-disease-classification/inceptionv3.h5",
    "../input/cassava-leaf-disease-classification/inceptionv3_weights.h5",
    "../input/cassava-leaf-disease-classification/inceptionv3.ckpt",
)
effb3_weight_candidates = (
    "../input/cassava-leaf-disease-classification/efficientnetb3.h5",
    "../input/cassava-leaf-disease-classification/efficientnetb3_weights.h5",
    "../input/cassava-leaf-disease-classification/efficientnetb3.ckpt",
)

dense_preds = predict_labels_for_model(
    build_densenet201, ds_images, n_test, dense_weight_candidates
)
inception_preds = predict_labels_for_model(
    build_inceptionv3, ds_images, n_test, inception_weight_candidates
)
efficient_net_preds = predict_labels_for_model(
    build_efficientnetb3, ds_images, n_test, effb3_weight_candidates
)

d = dense_preds
i = inception_preds
e = efficient_net_preds

result = np.where(d == i, d, np.where(i == e, i, np.where(d == e, d, d))).astype(
    np.int64
)

try:
    tf.keras.backend.clear_session()
except Exception:
    pass




## === cell 7
submission["label"] = np.asarray(result, dtype=np.int64)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Label value counts:\n", submission["label"].value_counts(dropna=False).sort_index()
)
