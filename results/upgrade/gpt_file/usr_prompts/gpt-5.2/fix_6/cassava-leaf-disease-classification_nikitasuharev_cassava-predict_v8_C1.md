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

0.8806285886974917

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import gc, random, math, re
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


dense201 = build_densenet201()
inception = build_inceptionv3()
efficient_net = build_efficientnetb3()

for _m in (dense201, inception, efficient_net):
    _m.compile(jit_compile=True)




## === cell 3
JPEG_PATH = "../input/cassava-leaf-disease-classification/test_images"

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_normalize(image_id):
    path = tf.strings.join([JPEG_PATH, "/", image_id])
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_test_dataset(image_ids, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    ds = ds.map(
        _decode_resize_normalize, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.cache()  # in-memory cache of preprocessed test set
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
BATCH_SIZE = 32

ds_test = make_test_dataset(test_image_ids, batch_size=BATCH_SIZE)


def predict_for_pretrained(model, ds, n_items):
    out = np.empty((n_items,), dtype=np.int64)
    idx = 0
    for batch in ds:
        probs = model.predict_on_batch(batch).numpy()
        bs = probs.shape[0]
        out[idx : idx + bs] = np.argmax(probs, axis=-1).astype(np.int64, copy=False)
        idx += bs
    return out


dense_preds = predict_for_pretrained(dense201, ds_test, n_test)
inception_preds = predict_for_pretrained(inception, ds_test, n_test)
efficient_net_preds = predict_for_pretrained(efficient_net, ds_test, n_test)

d = dense_preds
i = inception_preds
e = efficient_net_preds

result = np.where(d == i, d, np.where(i == e, i, np.where(d == e, d, d))).astype(
    np.int64
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2735926271.py in <cell line: 0>()
     21 
     22 
---> 23 dense_preds = predict_for_pretrained(dense201, ds_test, n_test)
     24 inception_preds = predict_for_pretrained(inception, ds_test, n_test)
     25 efficient_net_preds = predict_for_pretrained(efficient_net, ds_test, n_test)

/tmp/ipykernel_11/2735926271.py in predict_for_pretrained(model, ds, n_items)
     14     idx = 0
     15     for batch in ds:
---> 16         probs = model.predict_on_batch(batch).numpy()
     17         bs = probs.shape[0]
     18         out[idx : idx + bs] = np.argmax(probs, axis=-1).astype(np.int64, copy=False)

AttributeError: 'numpy.ndarray' object has no attribute 'numpy'

## === cell 7
submission["label"] = np.asarray(result, dtype=np.int64)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Label value counts:\n", submission["label"].value_counts(dropna=False).sort_index()
)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3700847984.py in <cell line: 0>()
----> 1 submission["label"] = np.asarray(result, dtype=np.int64)
      2 submission.to_csv("submission.csv", index=False)
      3 
      4 print(submission.head())
      5 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'result' is not defined
