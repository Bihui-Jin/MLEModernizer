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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import glob
import numpy as np
import pandas as pd



## === cell 1
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

tf.random.set_seed(23)
np.random.seed(23)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass



## === cell 2
IMG_SIZE_incres = 320
IMG_SIZE_effnet = 333

BATCH_SZ = 256



## === cell 3
effnet_path = "../input/effnetpp/eff.h5"
incRes_path = "../input/inceptionresnet/inceptionResNetv2_Sun_6pm.h5"


def _safe_load_or_build(model_path: str, builder_fn):
    if model_path is not None and tf.io.gfile.exists(model_path):
        try:
            m = load_model(model_path, compile=False)
            return m
        except Exception as e:
            print(
                f"Warning: failed to load {model_path}: {e}\nFalling back to application model."
            )
    return builder_fn()


def build_effnet_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE_effnet, IMG_SIZE_effnet, 3),
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(base.input, out)


def build_incres_model():
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE_incres, IMG_SIZE_incres, 3),
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(base.input, out)


effModel = _safe_load_or_build(effnet_path, build_effnet_model)
incResModel = _safe_load_or_build(incRes_path, build_incres_model)

effModel.trainable = False
incResModel.trainable = False

EFF_PREPROCESS = tf.keras.applications.efficientnet.preprocess_input
INRES_PREPROCESS = tf.keras.applications.inception_resnet_v2.preprocess_input



## === cell 4
train = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train["label"] = train["label"].astype("string")

diseases = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
print(train.head())
print(diseases)




## === cell 5
def make_incres_generators(train_df: pd.DataFrame):
    datagen_incres = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.2, 1.5],
        shear_range=25,
        zoom_range=0.3,
        channel_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=INRES_PREPROCESS,
        validation_split=0.15,
    )

    val_datagen_incres = ImageDataGenerator(
        preprocessing_function=INRES_PREPROCESS, validation_split=0.2
    )

    train_generator_incres = datagen_incres.flow_from_dataframe(
        dataframe=train_df,
        directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
        batch_size=32,
        subset="training",
        shuffle=True,
        class_mode="categorical",
    )

    val_generator_incres = val_datagen_incres.flow_from_dataframe(
        dataframe=train_df,
        directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
        batch_size=32,
        subset="validation",
        class_mode="categorical",
        shuffle=True,
    )
    return train_generator_incres, val_generator_incres


def make_effnet_generators(train_df: pd.DataFrame):
    datagen_effnet = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.2, 1.5],
        shear_range=25,
        zoom_range=0.3,
        channel_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=EFF_PREPROCESS,
        validation_split=0.15,
    )

    val_datagen_effnet = ImageDataGenerator(
        preprocessing_function=EFF_PREPROCESS, validation_split=0.2
    )

    train_generator_effnet = datagen_effnet.flow_from_dataframe(
        dataframe=train_df,
        directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
        batch_size=32,
        subset="training",
        shuffle=True,
        class_mode="categorical",
    )

    val_generator_effnet = val_datagen_effnet.flow_from_dataframe(
        dataframe=train_df,
        directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
        batch_size=32,
        subset="validation",
        class_mode="categorical",
        shuffle=True,
    )
    return train_generator_effnet, val_generator_effnet




## === cell 6
def _is_fallback(path: str) -> bool:
    return not (path is not None and tf.io.gfile.exists(path))


if _is_fallback(effnet_path):
    train_generator_effnet, val_generator_effnet = make_effnet_generators(train)

    effModel.trainable = True
    for layer in effModel.layers:
        layer.trainable = False
    effModel.layers[-1].trainable = True  # Dense(5)

    effModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    effModel.fit(
        train_generator_effnet,
        validation_data=val_generator_effnet,
        epochs=2,
        verbose=1,
    )
    effModel.trainable = False

if _is_fallback(incRes_path):
    train_generator_incres, val_generator_incres = make_incres_generators(train)

    incResModel.trainable = True
    for layer in incResModel.layers:
        layer.trainable = False
    incResModel.layers[-1].trainable = True  # Dense(5)

    incResModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    incResModel.fit(
        train_generator_incres,
        validation_data=val_generator_incres,
        epochs=2,
        verbose=1,
    )
    incResModel.trainable = False



## === cell 7
ss = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

assert {"image_id", "label"}.issubset(
    ss.columns
), "sample_submission.csv must have image_id and label columns"
print(ss.head(), len(ss))



## === cell 8
AUTO = tf.data.AUTOTUNE

_TEST_DIR_BASE = tf.constant(test_dir + os.sep, dtype=tf.string)

_DS_OPTS = tf.data.Options()
_DS_OPTS.deterministic = True
try:
    _DS_OPTS.experimental_optimization.map_parallelization = True
    _DS_OPTS.experimental_optimization.parallel_batch = True
    _DS_OPTS.experimental_optimization.autotune_buffers = True
except Exception:
    pass


@tf.function
def _to_path(img_id):
    return tf.strings.join([_TEST_DIR_BASE, img_id])


@tf.function
def _decode_and_make_two_sizes_fast(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, ratio=2, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0

    img_inc = tf.image.resize(
        img, [IMG_SIZE_incres, IMG_SIZE_incres], method="bilinear"
    )
    img_eff = tf.image.resize(
        img, [IMG_SIZE_effnet, IMG_SIZE_effnet], method="bilinear"
    )

    img_inc = INRES_PREPROCESS(img_inc)
    img_eff = EFF_PREPROCESS(img_eff)
    return img_inc, img_eff


@tf.function
def _ensemble_predict_batch(x_inc, x_eff):
    p_inc = incResModel(x_inc, training=False)
    p_eff = effModel(x_eff, training=False)
    avg_p = (p_inc + p_eff) / 2.0
    return tf.argmax(avg_p, axis=1, output_type=tf.int64)


def predict_ensemble(image_ids, batch_size=BATCH_SZ):
    image_ids_t = tf.constant(image_ids, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices(image_ids_t).with_options(_DS_OPTS)
    ds = ds.map(_to_path, num_parallel_calls=AUTO)
    ds = ds.map(_decode_and_make_two_sizes_fast, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    preds_out = np.empty((len(image_ids),), dtype=np.int64)
    offset = 0

    for x_inc, x_eff in ds:
        batch_preds = _ensemble_predict_batch(x_inc, x_eff).numpy()
        bsz = batch_preds.shape[0]
        preds_out[offset : offset + bsz] = batch_preds
        offset += bsz

    return preds_out


preds = predict_ensemble(ss["image_id"].tolist(), batch_size=BATCH_SZ)
print(preds[:10], preds.shape)



## === cell 9
submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
