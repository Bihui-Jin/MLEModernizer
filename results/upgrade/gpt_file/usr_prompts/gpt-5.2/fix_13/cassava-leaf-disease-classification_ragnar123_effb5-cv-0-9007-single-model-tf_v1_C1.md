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
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
from tensorflow.keras import backend as K
import glob

from tensorflow.keras.applications import EfficientNetB5

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KaggleDatasets = None




## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 20
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
LR = 0.0001
TTA = 10
VERBOSE = 2
N_CLASSES = 5

TEST_FILENAMES = "../input/cassava-leaf-disease-classification/test_images/*.jpg"




## === cell 3
_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_slack = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass


@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function
def read_image(file_path):
    image_name = tf.strings.regex_replace(file_path, r"^.*[\\/]", "")
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return image, image_name


TEST_FILEPATHS = tf.io.gfile.glob(TEST_FILENAMES)
TEST_FILEPATHS.sort()
NUM_TESTING_IMAGES = len(TEST_FILEPATHS)
print("NUM_TESTING_IMAGES =", NUM_TESTING_IMAGES)




## === cell 4
def get_model(weights_mode="cassava_or_imagenet"):
    """
    Architecture and compile settings unchanged: EfficientNetB5 backbone + GAP + Dropout + Dense softmax,
    Adam LR, CategoricalCrossentropy(label_smoothing=0.4), CategoricalAccuracy.
    """
    with strategy.scope():
        inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))

        effnet_weights = None
        if weights_mode == "imagenet":
            effnet_weights = "imagenet"

        x = EfficientNetB5(weights=effnet_weights, include_top=False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=[inp], outputs=[output])

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.4)],
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return model


def _predict_tta_mean_single_call(model, tta_image_ds, steps_per_pass, tta):
    total_steps = int(tta) * int(steps_per_pass)
    probs_all = model.predict(tta_image_ds, steps=total_steps, verbose=0)
    probs_all = probs_all[: int(tta) * NUM_TESTING_IMAGES]
    probs_all = probs_all.reshape((int(tta), NUM_TESTING_IMAGES, N_CLASSES))
    return probs_all.mean(axis=0)


def inference(model_paths):
    prediction = np.zeros((NUM_TESTING_IMAGES, N_CLASSES), dtype=np.float32)
    steps_per_pass = int(math.ceil(NUM_TESTING_IMAGES / BATCH_SIZE))

    test_filepaths = list(TEST_FILEPATHS)
    image_name = np.asarray([os.path.basename(p) for p in test_filepaths], dtype="U")

    print("Building cached decoded test dataset (single decode pass)...")
    decoded_cached = tf.data.Dataset.from_tensor_slices(test_filepaths).with_options(
        _DATASET_OPTIONS
    )
    decoded_cached = decoded_cached.map(read_image, num_parallel_calls=AUTO).cache()

    @tf.function
    def _augment_batch_vectorized(images, names):
        batch_size = tf.shape(images)[0]

        p_spatial = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_rotate = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_pixel_1 = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_pixel_2 = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_pixel_3 = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)
        p_crop = tf.random.uniform([batch_size], 0, 1.0, dtype=tf.float32)

        x = tf.image.random_flip_left_right(images)
        x = tf.image.random_flip_up_down(x)

        x_t = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.where(p_spatial[:, None, None, None] > 0.75, x_t, x)

        r3 = tf.image.rot90(x, k=3)
        r2 = tf.image.rot90(x, k=2)
        r1 = tf.image.rot90(x, k=1)
        x = tf.where(p_rotate[:, None, None, None] > 0.75, r3, x)
        x = tf.where(
            tf.logical_and(p_rotate > 0.5, p_rotate <= 0.75)[:, None, None, None], r2, x
        )
        x = tf.where(
            tf.logical_and(p_rotate > 0.25, p_rotate <= 0.5)[:, None, None, None], r1, x
        )

        sat = tf.image.random_saturation(x, lower=0.7, upper=1.3)
        x = tf.where(p_pixel_1[:, None, None, None] >= 0.4, sat, x)

        con = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        x = tf.where(p_pixel_2[:, None, None, None] >= 0.4, con, x)

        bri = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.where(p_pixel_3[:, None, None, None] >= 0.4, bri, x)

        def _central_crop(frac):
            y = tf.image.central_crop(x, central_fraction=frac)
            y = tf.image.resize(y, size=IMAGE_SIZE)
            y = tf.reshape(y, [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
            return y

        seeds = tf.random.uniform(
            [batch_size, 2], minval=0, maxval=2**31 - 1, dtype=tf.int32
        )

        def _one_random_crop(args):
            img, seed = args
            hh = tf.shape(img)[0]
            minsz = tf.cast(tf.round(tf.cast(hh, tf.float32) * 0.8), tf.int32)
            maxsz = tf.cast(hh, tf.int32)

            crop_size = tf.random.stateless_uniform(
                shape=[], seed=seed, minval=minsz, maxval=maxsz, dtype=tf.int32
            )
            max_off = tf.maximum(hh - crop_size, 0)
            off_seed = seed + tf.constant([11, 23], dtype=tf.int32)
            off_y = tf.random.stateless_uniform(
                shape=[], seed=off_seed, minval=0, maxval=max_off + 1, dtype=tf.int32
            )
            off_x = tf.random.stateless_uniform(
                shape=[],
                seed=off_seed + 1,
                minval=0,
                maxval=max_off + 1,
                dtype=tf.int32,
            )

            cropped = tf.image.crop_to_bounding_box(
                img, off_y, off_x, crop_size, crop_size
            )
            cropped = tf.image.resize(cropped, IMAGE_SIZE)
            cropped = tf.reshape(cropped, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
            return cropped

        x0 = tf.image.resize(x, size=IMAGE_SIZE)
        x0 = tf.reshape(x0, [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

        x07 = _central_crop(0.7)
        x08 = _central_crop(0.8)
        x09 = _central_crop(0.9)

        xrc = tf.map_fn(
            _one_random_crop,
            (x, seeds),
            fn_output_signature=tf.float32,
            parallel_iterations=32,
        )
        xrc = tf.reshape(xrc, [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

        out = x0
        out = tf.where((p_crop > 0.9)[:, None, None, None], x07, out)
        out = tf.where(
            (tf.logical_and(p_crop > 0.8, p_crop <= 0.9))[:, None, None, None], x08, out
        )
        out = tf.where(
            (tf.logical_and(p_crop > 0.7, p_crop <= 0.8))[:, None, None, None], x09, out
        )
        out = tf.where(
            (tf.logical_and(p_crop > 0.4, p_crop <= 0.7))[:, None, None, None], xrc, out
        )

        return out, names

    tta_batched = decoded_cached.batch(BATCH_SIZE, drop_remainder=False)
    tta_batched = tta_batched.map(
        _augment_batch_vectorized, num_parallel_calls=AUTO
    ).repeat()
    tta_image = tta_batched.map(
        lambda image, image_name: image, num_parallel_calls=AUTO
    ).prefetch(AUTO)

    if not model_paths:
        print(
            "WARNING: No model .h5 files found in ../input/cassava-models/*.h5. "
            "Falling back to EfficientNetB5(weights='imagenet') to generate a valid submission."
        )
        K.clear_session()
        model = get_model(weights_mode="imagenet")
        prediction += _predict_tta_mean_single_call(
            model, tta_image, steps_per_pass=steps_per_pass, tta=TTA
        )
    else:
        inv_n_models = 1.0 / len(model_paths)
        for fold, model_path in enumerate(model_paths):
            print("\n" + "-" * 50)
            print(f"Predicting fold {fold + 1} | loading: {model_path}")
            K.clear_session()
            model = get_model(weights_mode="cassava_or_imagenet")
            model.load_weights(model_path)

            prediction += (
                _predict_tta_mean_single_call(
                    model, tta_image, steps_per_pass=steps_per_pass, tta=TTA
                )
                * inv_n_models
            )

    sub = pd.DataFrame(
        {
            "image_id": image_name,
            "label": np.argmax(prediction, axis=-1).astype(np.int64),
        }
    )
    sub.to_csv("submission.csv", index=False)
    return image_name, prediction, sub


model_paths = sorted(glob.glob("../input/cassava-models/*.h5"))

image_name, prediction, sub = inference(model_paths)
sub.head()
