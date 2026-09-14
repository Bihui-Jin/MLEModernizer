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

0.12033

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07324) has done: 'I fix the TensorFlow/Keras import stack to avoid the protobuf `MessageFactory.GetPrototype` crash by using `tf.keras` consistently and removing legacy standalone `keras` imports that trigger the error in this environment. I also fix the missing `load_model`/`image` symbols and make the model-loading robust by falling back to lightweight ImageNet `tf.keras.applications` models when the referenced `.h5` files aren’t available, so the notebook always runs end-to-end. Then I correct the inference pipeline to read from the real test set (not the train/validation generators) and generate predictions efficiently in batches, preserving the same ensemble-averaging logic as your original code. Finally, I ensure a valid `submission.csv` is written with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.10239) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is the standard Kaggle workaround for this exact incompatibility. Then I correct the Keras preprocessing imports (ImageDataGenerator lives under `tensorflow.keras.preprocessing.image`, not under the `image` module alias you used), so the script runs end-to-end. Finally, to move accuracy substantially toward the target (your current 0.073 suggests essentially-random predictions from untrained fallback heads), I keep your same two-backbone ensemble logic but load the official ImageNet preprocessors for each backbone so inference matches each model’s expected input normalization (a minimal, metric-aligned change that usually yields a large gain without changing the model architecture or training loop).'
- What this solution (achieved 0.12033) has done: 'The timeout is most likely coming from slow Python-side JPEG decoding + resizing for every test image and from using `ImageDataGenerator` without multi-worker prefetching during any fallback training. I keep the exact ensemble logic and preprocessing, but speed up inference by switching to a single `tf.data` pipeline that (a) uses `tf.image.decode_jpeg(..., dct_method="INTEGER_FAST")`, (b) fuses mapping/batching/prefetching with deterministic settings, and (c) avoids per-batch `.numpy()` roundtrips by using `model.predict_on_batch` and `tf.argmax` on-device before a single final `.numpy()`. I also enable `workers`/`use_multiprocessing` in the (rare) fallback `.fit()` calls to prevent generator bottlenecks, without changing epochs, augmentations, or model structure. All changes are deterministic-equivalent and preserve evaluation semantics (same averaging of the two model softmax outputs, same argmax labels).'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
IMG_SIZE_incres = 320
IMG_SIZE_effnet = 333
BATCH_SZ = 64  # keep reasonable for memory; original BATCH_SZ=320 is not required for correctness



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
    dataframe=train,
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
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
    batch_size=32,
    subset="validation",
    class_mode="categorical",
    shuffle=True,
)



## === cell 6
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
    dataframe=train,
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
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
    batch_size=32,
    subset="validation",
    class_mode="categorical",
    shuffle=True,
)




## === cell 7
def _is_fallback(path: str) -> bool:
    return not (path is not None and tf.io.gfile.exists(path))


_FIT_WORKERS = max(1, (os.cpu_count() or 2) - 1)

if _is_fallback(effnet_path):
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
        workers=_FIT_WORKERS,
        use_multiprocessing=True,
        max_queue_size=32,
    )
    effModel.trainable = False

if _is_fallback(incRes_path):
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
        workers=_FIT_WORKERS,
        use_multiprocessing=True,
        max_queue_size=32,
    )
    incResModel.trainable = False



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3028631810.py in <cell line: 0>()
     18         metrics=["accuracy"],
     19     )
---> 20     effModel.fit(
     21         train_generator_effnet,
     22         validation_data=val_generator_effnet,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 8
ss = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

assert {"image_id", "label"}.issubset(
    ss.columns
), "sample_submission.csv must have image_id and label columns"
print(ss.head(), len(ss))



## === cell 9
AUTO = tf.data.AUTOTUNE


def _decode_and_make_two_sizes_fast(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
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


def predict_ensemble(image_ids, batch_size=BATCH_SZ):
    paths = tf.constant([os.path.join(test_dir, x) for x in image_ids])

    opts = tf.data.Options()
    opts.deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
    ds = ds.map(_decode_and_make_two_sizes_fast, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    preds_out = np.empty((len(image_ids),), dtype=np.int64)
    offset = 0

    for x_inc, x_eff in ds:
        p_inc = incResModel.predict_on_batch(x_inc)
        p_eff = effModel.predict_on_batch(x_eff)

        avg_p = (p_inc + p_eff) / 2.0
        batch_preds = tf.argmax(avg_p, axis=1, output_type=tf.int64).numpy()

        bsz = batch_preds.shape[0]
        preds_out[offset : offset + bsz] = batch_preds
        offset += bsz

    return preds_out


preds = predict_ensemble(ss["image_id"].tolist(), batch_size=BATCH_SZ)
preds[:10], preds.shape



## === cell 10
submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
