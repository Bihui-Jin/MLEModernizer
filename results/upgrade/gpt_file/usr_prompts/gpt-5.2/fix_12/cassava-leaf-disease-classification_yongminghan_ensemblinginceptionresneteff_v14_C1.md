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

0.8389241462677546

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07324) has done: 'I fix the TensorFlow/Keras import stack to avoid the protobuf `MessageFactory.GetPrototype` crash by using `tf.keras` consistently and removing legacy standalone `keras` imports that trigger the error in this environment. I also fix the missing `load_model`/`image` symbols and make the model-loading robust by falling back to lightweight ImageNet `tf.keras.applications` models when the referenced `.h5` files aren’t available, so the notebook always runs end-to-end. Then I correct the inference pipeline to read from the real test set (not the train/validation generators) and generate predictions efficiently in batches, preserving the same ensemble-averaging logic as your original code. Finally, I ensure a valid `submission.csv` is written with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.10239) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is the standard Kaggle workaround for this exact incompatibility. Then I correct the Keras preprocessing imports (ImageDataGenerator lives under `tensorflow.keras.preprocessing.image`, not under the `image` module alias you used), so the script runs end-to-end. Finally, to move accuracy substantially toward the target (your current 0.073 suggests essentially-random predictions from untrained fallback heads), I keep your same two-backbone ensemble logic but load the official ImageNet preprocessors for each backbone so inference matches each model’s expected input normalization (a minimal, metric-aligned change that usually yields a large gain without changing the model architecture or training loop).'
- What this solution (achieved 0.12033) has done: 'The timeout is most likely coming from slow Python-side JPEG decoding + resizing for every test image and from using `ImageDataGenerator` without multi-worker prefetching during any fallback training. I keep the exact ensemble logic and preprocessing, but speed up inference by switching to a single `tf.data` pipeline that (a) uses `tf.image.decode_jpeg(..., dct_method="INTEGER_FAST")`, (b) fuses mapping/batching/prefetching with deterministic settings, and (c) avoids per-batch `.numpy()` roundtrips by using `model.predict_on_batch` and `tf.argmax` on-device before a single final `.numpy()`. I also enable `workers`/`use_multiprocessing` in the (rare) fallback `.fit()` calls to prevent generator bottlenecks, without changing epochs, augmentations, or model structure. All changes are deterministic-equivalent and preserve evaluation semantics (same averaging of the two model softmax outputs, same argmax labels).'
- What this solution (achieved 0.11921) has done: 'The timeout is most likely coming from the fallback training path (when the external `.h5` weights aren’t found) plus slow input pipeline settings in `ImageDataGenerator` and repeated JPEG decode/resize work. I keep the exact same model architectures and training loops, but make the data input for both training and inference faster and more parallel: enable generator multiprocessing/workers, ensure validation split is consistent (so we don’t accidentally run extra work), and tune the `tf.data` pipeline (parallel interleave, deterministic options, avoid extra graph retracing, and keep batching/prefetching efficient). These changes are runtime-only optimizations; they do not alter the model math, loss, or evaluation semantics (only negligible floating point ordering differences may occur). Paths and outputs remain identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import google.protobuf  # noqa: F401

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

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
        include_top=True,
        weights="imagenet",
        input_shape=(IMG_SIZE_effnet, IMG_SIZE_effnet, 3),
    )
    x = base.output
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(base.input, out)


def build_incres_model():
    base = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_shape=(IMG_SIZE_incres, IMG_SIZE_incres, 3),
    )
    x = base.output
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="softmax")(x)
    return tf.keras.Model(base.input, out)


effModel = _safe_load_or_build(effnet_path, build_effnet_model)
incResModel = _safe_load_or_build(incRes_path, build_incres_model)

effModel.trainable = False
incResModel.trainable = False

EFF_PREPROCESS = tf.keras.applications.efficientnet.preprocess_input
INRES_PREPROCESS = tf.keras.applications.inception_resnet_v2.preprocess_input



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/588103418.py in <cell line: 0>()
     43 
     44 
---> 45 effModel = _safe_load_or_build(effnet_path, build_effnet_model)
     46 incResModel = _safe_load_or_build(incRes_path, build_incres_model)
     47 

/tmp/ipykernel_11/588103418.py in _safe_load_or_build(model_path, builder_fn)
     12                 f"Warning: failed to load {model_path}: {e}\nFalling back to application model."
     13             )
---> 14     return builder_fn()
     15 
     16 

/tmp/ipykernel_11/588103418.py in build_effnet_model()
     20 # (EffNet + IncRes) and softmax averaging identical, but avoids near-random predictions when .h5 isn't available.
     21 def build_effnet_model():
---> 22     base = tf.keras.applications.EfficientNetB0(
     23         include_top=True,
     24         weights="imagenet",

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    569     name="efficientnetb0",
    570 ):
--> 571     return EfficientNet(
    572         1.0,
    573         1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)
    287 
    288     # Determine proper input shape
--> 289     input_shape = imagenet_utils.obtain_input_shape(
    290         input_shape,
    291         default_size=default_size,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/imagenet_utils.py in obtain_input_shape(input_shape, default_size, min_size, data_format, require_flatten, weights)
    347         if input_shape is not None:
    348             if input_shape != default_shape:
--> 349                 raise ValueError(
    350                     "When setting `include_top=True` "
    351                     "and loading `imagenet` weights, "

ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(333, 333, 3)

## === cell 4
train = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train["label"] = train["label"].astype("int32")

diseases = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
print(train.head())
print(diseases)



## === cell 5
_VAL_SPLIT = 0.15


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
        validation_split=_VAL_SPLIT,
    )

    val_datagen_incres = ImageDataGenerator(
        preprocessing_function=INRES_PREPROCESS, validation_split=_VAL_SPLIT
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
        class_mode="raw",  # yields integer labels for sparse loss
    )

    val_generator_incres = val_datagen_incres.flow_from_dataframe(
        dataframe=train_df,
        directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
        batch_size=32,
        subset="validation",
        class_mode="raw",
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
        validation_split=_VAL_SPLIT,
    )

    val_datagen_effnet = ImageDataGenerator(
        preprocessing_function=EFF_PREPROCESS, validation_split=_VAL_SPLIT
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
        class_mode="raw",
    )

    val_generator_effnet = val_datagen_effnet.flow_from_dataframe(
        dataframe=train_df,
        directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
        x_col="image_id",
        y_col="label",
        target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
        batch_size=32,
        subset="validation",
        class_mode="raw",
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
        loss="sparse_categorical_crossentropy",
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
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    incResModel.fit(
        train_generator_incres,
        validation_data=val_generator_incres,
        epochs=2,
        verbose=1,
    )
    incResModel.trainable = False



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/628012656.py in <cell line: 0>()
      6 # Keep same epochs/loop; just remove unsupported kwargs so it runs end-to-end.
      7 if _is_fallback(effnet_path):
----> 8     train_generator_effnet, val_generator_effnet = make_effnet_generators(train)
      9 
     10     effModel.trainable = True

/tmp/ipykernel_11/490530989.py in make_effnet_generators(train_df)
     58         horizontal_flip=True,
     59         vertical_flip=True,
---> 60         preprocessing_function=EFF_PREPROCESS,
     61         validation_split=_VAL_SPLIT,
     62     )

NameError: name 'EFF_PREPROCESS' is not defined

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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3269634752.py in <cell line: 0>()
     65 
     66 
---> 67 preds = predict_ensemble(ss["image_id"].tolist(), batch_size=BATCH_SZ)
     68 print(preds[:10], preds.shape)
     69 

/tmp/ipykernel_11/3269634752.py in predict_ensemble(image_ids, batch_size)
     49     ds = tf.data.Dataset.from_tensor_slices(image_ids_t).with_options(_DS_OPTS)
     50     ds = ds.map(_to_path, num_parallel_calls=AUTO)
---> 51     ds = ds.map(_decode_and_make_two_sizes_fast, num_parallel_calls=AUTO)
     52     ds = ds.batch(batch_size, drop_remainder=False)
     53     ds = ds.prefetch(AUTO)

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file32gp37wp.py in tf___decode_and_make_two_sizes_fast(path)
     13                 img_inc = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(IMG_SIZE_incres), ag__.ld(IMG_SIZE_incres)]), dict(method='bilinear'), fscope)
     14                 img_eff = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(IMG_SIZE_effnet), ag__.ld(IMG_SIZE_effnet)]), dict(method='bilinear'), fscope)
---> 15                 img_inc = ag__.converted_call(ag__.ld(INRES_PREPROCESS), (ag__.ld(img_inc),), None, fscope)
     16                 img_eff = ag__.converted_call(ag__.ld(EFF_PREPROCESS), (ag__.ld(img_eff),), None, fscope)
     17                 try:

NameError: in user code:

    File "/tmp/ipykernel_11/3269634752.py", line 33, in _decode_and_make_two_sizes_fast  *
        img_inc = INRES_PREPROCESS(img_inc)

    NameError: name 'INRES_PREPROCESS' is not defined


## === cell 9
submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds.astype(int)})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1778851100.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds.astype(int)})
      2 submission.to_csv("submission.csv", index=False)
      3 print(submission.head())
      4 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'preds' is not defined
