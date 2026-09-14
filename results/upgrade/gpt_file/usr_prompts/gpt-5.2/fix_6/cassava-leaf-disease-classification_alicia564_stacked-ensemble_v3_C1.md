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

0.8951344817165306

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

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # matches ImageDataGenerator preprocessing_function
    return image, label


label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"].astype(str))

train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

train_paths = train["path"].values
train_labels_np = train["label_encoded"].values.astype(np.int64, copy=False)
valid_paths = valid["path"].values
valid_labels_np = valid["label_encoded"].values.astype(np.int64, copy=False)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_np))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels_np))

options = tf.data.Options()
options.experimental_deterministic = True

BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224


def _parse_tfrec(example_proto, labeled=True):
    if labeled:
        feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        feature_desc = {"image": tf.io.FixedLenFeature([], tf.string)}
    ex = tf.io.parse_single_example(example_proto, feature_desc)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    if labeled:
        return img, tf.cast(ex["label"], tf.int64)
    return img


def _tfrec_files(pattern):
    return tf.io.gfile.glob(pattern)


train_tfrec_files = sorted(
    _tfrec_files(
        "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
    )
)
test_tfrec_files = sorted(
    _tfrec_files(
        "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
    )
)

USE_TFRECORDS = (len(train_tfrec_files) > 0) and (len(test_tfrec_files) > 0)
print(
    "Using TFRecords:",
    USE_TFRECORDS,
    "| train tfrecs:",
    len(train_tfrec_files),
    "| test tfrecs:",
    len(test_tfrec_files),
)

if USE_TFRECORDS:
    train_ds = (
        tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
        .map(lambda x: _parse_tfrec(x, labeled=True), num_parallel_calls=AUTOTUNE)
        .shuffle(2048, seed=42, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    valid_ds = (
        tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE).map(
            lambda x: _parse_tfrec(x, labeled=True), num_parallel_calls=AUTOTUNE
        )
    )
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels_np))
    valid_ds = (
        valid_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
else:
    train_ds = (
        train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    valid_ds = (
        valid_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )

print("Train/Valid sizes:", len(train), len(valid))
print("Num classes:", len(le.classes_))



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 3
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2


NUM_CLASSES = len(le.classes_)
INPUT_SHAPE = (224, 224, 3)


def build_base_model(backbone_fn, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=input_shape)
    base = backbone_fn(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    x = layers.Dropout(0.2)(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


cropnet_model = build_base_model(
    EfficientNetB2
)  # strongest -> will be weighted highest
densenet_model = build_base_model(EfficientNetB1)  # mid
efficientnet_model = build_base_model(EfficientNetB0)  # light

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.1

print(
    "Built base models:",
    [m.name for m in [cropnet_model, densenet_model, efficientnet_model]],
)



## === cell 4
history = cropnet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=8,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

_ = densenet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)
_ = efficientnet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1537506910.py in <cell line: 0>()
----> 1 history = cropnet_model.fit(
      2     train_ds,
      3     validation_data=valid_ds,
      4     epochs=8,
      5     callbacks=[early_stopping, learning_rate_reduction],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_77046]

## === cell 5
from sklearn.linear_model import LogisticRegression


def to_probs(pred_output):
    """
    Keras model output is typically an ndarray; keep compatibility with the original dict/tensor cases.
    Return a numpy array of probabilities/logits as (batch, num_classes).
    """
    if isinstance(pred_output, dict):
        if "output_0" in pred_output:
            arr = pred_output["output_0"]
        else:
            arr = next(iter(pred_output.values()))
    else:
        arr = pred_output
    return np.asarray(arr)


ensemble_predictor = tf.keras.Model(
    inputs=cropnet_model.input,
    outputs=[cropnet_model.output, densenet_model.output, efficientnet_model.output],
)

if USE_TFRECORDS:
    train_images_ds = (
        tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
        .map(lambda x: _parse_tfrec(x, labeled=True)[0], num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
else:
    train_images_ds = (
        tf.data.Dataset.from_tensor_slices(train_paths)
        .map(
            lambda p: load_and_preprocess_image_nolabel(p), num_parallel_calls=AUTOTUNE
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )

cropnet_probs_all, densenet_probs_all, efficientnet_probs_all = [
    to_probs(x) for x in ensemble_predictor.predict(train_images_ds, verbose=0)
]

soft_voting_features = (
    cropnet_weight * cropnet_probs_all
    + densenet_weight * densenet_probs_all
    + efficientnet_weight * efficientnet_probs_all
)

soft_voting_labels = train_labels_np

meta_model = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)
meta_model.fit(soft_voting_features, soft_voting_labels)

print("Meta-model trained on features:", soft_voting_features.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1088420265.py in <cell line: 0>()
     20 # This is exactly equivalent to calling predict() three times on the same images, but it avoids repeating
     21 # dataset iteration/transfer overhead 3x and reduces Python<->TF boundary overhead.
---> 22 ensemble_predictor = tf.keras.Model(
     23     inputs=cropnet_model.input,
     24     outputs=[cropnet_model.output, densenet_model.output, efficientnet_model.output],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/tracking.py in wrapper(*args, **kwargs)
     24     def wrapper(*args, **kwargs):
     25         with DotNotTrackScope():
---> 26             return fn(*args, **kwargs)
     27 
     28     return wrapper

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in __init__(self, inputs, outputs, name, **kwargs)
    133             inputs, outputs = clone_graph_nodes(inputs, outputs)
    134 
--> 135         Function.__init__(self, inputs, outputs, name=name)
    136 
    137         if trainable is not None:

/usr/local/lib/python3.11/dist-packages/keras/src/ops/function.py in __init__(self, inputs, outputs, name)
     75             self._self_setattr_tracking = _self_setattr_tracking
     76 
---> 77         (nodes, nodes_by_depth, operations, operations_by_depth) = map_graph(
     78             self._inputs, self._outputs
     79         )

/usr/local/lib/python3.11/dist-packages/keras/src/ops/function.py in map_graph(inputs, outputs)
    327     for name in all_names:
    328         if all_names.count(name) != 1:
--> 329             raise ValueError(
    330                 f'The name "{name}" is used {all_names.count(name)} '
    331                 "times in the model. All operation names should be unique."

ValueError: The name "stem_conv_pad" is used 3 times in the model. All operation names should be unique.

## === cell 6
from sklearn.metrics import accuracy_score

meta_model_predictions = meta_model.predict(soft_voting_features)
accuracy = accuracy_score(soft_voting_labels, meta_model_predictions)
print("Meta-model (Logistic Regression) Train Accuracy:", accuracy)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2551020032.py in <cell line: 0>()
      1 from sklearn.metrics import accuracy_score
      2 
----> 3 meta_model_predictions = meta_model.predict(soft_voting_features)
      4 accuracy = accuracy_score(soft_voting_labels, meta_model_predictions)
      5 print("Meta-model (Logistic Regression) Train Accuracy:", accuracy)

NameError: name 'meta_model' is not defined

## === cell 7
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)

image_names = sample_sub["image_id"].tolist()
test_paths = [os.path.join(image_dir, fn) for fn in image_names]


def load_and_preprocess_image_nolabel(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image


if USE_TFRECORDS:
    test_ds = (
        tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
        .map(lambda x: _parse_tfrec(x, labeled=False), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
else:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = (
        test_ds.map(load_and_preprocess_image_nolabel, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )

cropnet_test, densenet_test, efficientnet_test = [
    to_probs(x) for x in ensemble_predictor.predict(test_ds, verbose=0)
]

soft_voting_test = (
    cropnet_weight * cropnet_test
    + densenet_weight * densenet_test
    + efficientnet_weight * efficientnet_test
)

predictions = meta_model.predict(soft_voting_test).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)

assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns must be image_id,label."
assert out_path.endswith(".csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1495966808.py in <cell line: 0>()
     38 # Use the same one-pass multi-output predictor to avoid iterating test_ds 3 times.
     39 cropnet_test, densenet_test, efficientnet_test = [
---> 40     to_probs(x) for x in ensemble_predictor.predict(test_ds, verbose=0)
     41 ]
     42 

NameError: name 'ensemble_predictor' is not defined
