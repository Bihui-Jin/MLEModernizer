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

0.8834995466908432

# 6. Current score

0.11547

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The timeout is dominated by Python-side per-image TTA generation (`ImageDataGenerator.flow` + `next` in nested loops) and repeated PIL decode/resize for each augmentation view. I keep the exact same model and TTA semantics (original + 5 augmented views, same `ImageDataGenerator` params), but reduce overhead by (1) decoding/resizing each test image once into a contiguous float32 array cache, (2) generating all TTA views for a chunk using a single `flow` call per image (instead of creating/advancing an iterator per augmentation), and (3) preallocating arrays and minimizing Python loop work. These changes are provably equivalent in terms of what gets predicted (same generator, same number of views, same averaging), with only negligible floating-point ordering differences.'
- What this solution (achieved 0.05531) has done: 'I target the current bottleneck: Python-side TTA generation (`ImageDataGenerator.random_transform`) inside nested loops, which dominates runtime. The core logic (same pretrained model, same TTA policy/count, same mean-aggregation, same evaluation semantics) is preserved by moving augmentation and decoding into a fully-vectorized `tf.data` + TensorFlow image ops pipeline, eliminating per-image Python overhead and enabling parallelism/prefetch. I also keep determinism by using stateless RNG seeds derived from the global seed, image index, and TTA index so each view is reproducible. Finally, I avoid repeated dataset construction and reduce conversions/copies while keeping identical input scaling and prediction averaging.'
- What this solution (achieved 0.11547) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf environment setting that’s incompatible with the Kaggle TF/protobuf build. Then, since the provided pretrained weight file (`unionmodelv04/Cassava_Best_UnitedModel_V04.hdf5`) is not available in your environment, I keep the same inference/TTA pipeline but add a minimal fallback: train a small CNN quickly on the provided `train_images` via the existing `AugmentedImageSequence`, so a model is always defined and a valid `submission.csv` is produced. I also make the data paths robust by falling back to `/kaggle/input/` if the competition subfolder path isn’t present. These changes unblock execution end-to-end and should improve score versus a broken pipeline (though likely below the original pretrained-model score).'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/input/",
]
BASE_DIR = next(
    (p for p in BASE_DIR_CANDIDATES if os.path.exists(p)), BASE_DIR_CANDIDATES[0]
)

TRAIN_DIR = os.path.join(BASE_DIR, "train_images/")
TEST_DIR = os.path.join(BASE_DIR, "test_images/")

if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    nested = os.path.join(BASE_DIR, "cassava-leaf-disease-classification/")
    if os.path.exists(nested):
        BASE_DIR = nested
        TRAIN_DIR = os.path.join(BASE_DIR, "train_images/")
        TEST_DIR = os.path.join(BASE_DIR, "test_images/")

print("BASE_DIR:", BASE_DIR)
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))



## === cell 2
import matplotlib.pyplot as plt
import json
from PIL import Image

import tensorflow as tf
from tensorflow import keras

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16

_PRETRAINED_CANDIDATES = [
    "../input/unionmodelv04/Cassava_Best_UnitedModel_V04.hdf5",
    "/kaggle/input/unionmodelv04/Cassava_Best_UnitedModel_V04.hdf5",
]
PRE_TRAINED_MODEL = next(
    (p for p in _PRETRAINED_CANDIDATES if os.path.exists(p)), _PRETRAINED_CANDIDATES[0]
)

if hasattr(Image, "Resampling"):
    PIL_RESAMPLE = Image.Resampling.LANCZOS
else:
    PIL_RESAMPLE = Image.LANCZOS

print("Pretrained candidate selected:", PRE_TRAINED_MODEL)
print("Pretrained exists:", os.path.exists(PRE_TRAINED_MODEL))



## === cell 7
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        CLAHE,
        HueSaturationValue,
        CenterCrop,
        RandomBrightness,
        RandomContrast,
        RandomGamma,
        Cutout,
        ToFloat,
        ShiftScaleRotate,
    )

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                always_apply=False,
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255),
        ]
    )

    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255)])
    print("Albumentations available: using AUGMENTATIONS_TEST/TRAIN.")
except Exception as e:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None
    print(
        f"Albumentations not usable ({e}); proceeding without albumentations augmentations."
    )



## === cell 8
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 9
from tensorflow.keras.utils import Sequence
import random


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img_data = Image.open(image_path).convert("RGB")
    img_data = np.array(img_data.resize((IMG_HEIGHT, IMG_WIDTH), PIL_RESAMPLE))
    return img_data


class AugmentedImageSequence(Sequence):
    def __init__(
        self, mode, data_set_type, x_set, y_set, batch_size, augmentations, seed=42
    ):
        self.mode = mode
        self.data_type = data_set_type
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations
        self._seed = int(seed)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        rng = random.Random(self._seed + int(idx))

        img_list = []
        for x in batch_x:
            img = load_single_image(self.data_type, x)

            if self.data_type == "TRAIN_DATA":
                random_num = rng.uniform(0, 1)
                if random_num > 0.5 and self.augment is not None:
                    img = self.augment(image=img)["image"]
            else:
                if self.data_type == "VALIDATE_DATA" and self.augment is not None:
                    img = self.augment(image=img)["image"]

            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0

            img_list.append(img)

        img_array = np.stack(img_list, axis=0)
        return img_array, np.array(batch_y)




## === cell 10
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
test_df = pd.read_csv(sample_sub_path)[["image_id"]]
test_samples = test_df.shape[0]
print("Test samples:", test_samples)



## === cell 11
test_gen = None
print(
    "Skipped building slow Python Sequence test_gen; using batched TTA prediction instead."
)



## === cell 12
from tensorflow.keras.models import load_model

if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
    print(f"Loaded pretrained model from: {PRE_TRAINED_MODEL}")
else:
    print(
        "Pretrained model file not found; training a small fallback CNN on train_images "
        "to enable end-to-end execution and a valid submission."
    )

    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    perm = np.random.RandomState(42).permutation(len(train_df))
    split = int(0.9 * len(train_df))
    tr_idx, va_idx = perm[:split], perm[split:]

    x_train = train_df.iloc[tr_idx]["image_id"].values
    y_train = train_df.iloc[tr_idx]["label"].values.astype(np.int32)

    x_val = train_df.iloc[va_idx]["image_id"].values
    y_val = train_df.iloc[va_idx]["label"].values.astype(np.int32)

    train_seq = AugmentedImageSequence(
        mode="TRAIN",
        data_set_type="TRAIN_DATA",
        x_set=x_train,
        y_set=y_train,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
        seed=42,
    )
    val_seq = AugmentedImageSequence(
        mode="VAL",
        data_set_type="VALIDATE_DATA",
        x_set=x_val,
        y_set=y_val,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TEST,
        seed=4242,
    )

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=2,
        verbose=2,
        workers=1,
        use_multiprocessing=False,
    )

model.summary()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2249725975.py in <cell line: 0>()
     63 
     64     # Keep runtime reasonable; no early stopping introduced (fixed epochs).
---> 65     model.fit(
     66         train_seq,
     67         validation_data=val_seq,

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

## === cell 13
TTA_N_AUG = 5  # keep exactly the same number of generated augmentations as original

_TTA_DATAGEN = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_float255_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32)  # [0,255]
    return img


@tf.function
def _tta_augment_tf(img255, seed2):

    s1 = tf.random.stateless_uniform(
        [2], seed=seed2, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    do_lr = tf.random.stateless_uniform([], seed=s1, minval=0.0, maxval=1.0) < 0.5
    img = tf.cond(do_lr, lambda: tf.image.flip_left_right(img255), lambda: img255)
    s2 = tf.random.stateless_uniform(
        [2], seed=s1, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    do_ud = tf.random.stateless_uniform([], seed=s2, minval=0.0, maxval=1.0) < 0.5
    img = tf.cond(do_ud, lambda: tf.image.flip_up_down(img), lambda: img)

    s3 = tf.random.stateless_uniform(
        [2], seed=s2, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    angle = tf.random.stateless_uniform([], seed=s3, minval=-45.0, maxval=45.0) * (
        np.pi / 180.0
    )

    s4 = tf.random.stateless_uniform(
        [2], seed=s3, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    zoom = tf.random.stateless_uniform([], seed=s4, minval=0.6, maxval=1.4)

    s5 = tf.random.stateless_uniform(
        [2], seed=s4, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    shear = tf.random.stateless_uniform([], seed=s5, minval=-0.1, maxval=0.1)

    s6 = tf.random.stateless_uniform(
        [2], seed=s5, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    tx = tf.random.stateless_uniform([], seed=s6, minval=-0.1, maxval=0.1) * tf.cast(
        IMG_HEIGHT, tf.float32
    )

    s7 = tf.random.stateless_uniform(
        [2], seed=s6, minval=0, maxval=2**31 - 1, dtype=tf.int32
    )
    ty = tf.random.stateless_uniform([], seed=s7, minval=-0.1, maxval=0.1) * tf.cast(
        IMG_WIDTH, tf.float32
    )

    c = tf.cos(angle)
    s = tf.sin(angle)

    sh = tf.tan(shear)

    a0 = (c + s * sh) / zoom
    a1 = (-s + c * sh) / zoom
    b0 = s / zoom
    b1 = c / zoom

    cx = (tf.cast(IMG_WIDTH, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_HEIGHT, tf.float32) - 1.0) / 2.0

    t0 = cx - (a0 * cx + a1 * cy) + ty
    t1 = cy - (b0 * cx + b1 * cy) + tx

    transform = tf.stack([a0, a1, t0, b0, b1, t1, 0.0, 0.0], axis=0)
    transform = tf.expand_dims(transform, axis=0)  # [1,8]

    img4 = tf.expand_dims(img, axis=0)  # [1,H,W,3]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return tf.squeeze(out, axis=0)


def predict_with_tta_batched(
    model, image_ids, tta_n_aug=5, predict_batch_size=256, seed=42
):
    image_ids = np.asarray(image_ids)
    n = int(image_ids.shape[0])
    n_views = 1 + int(tta_n_aug)

    paths = np.array([os.path.join(TEST_DIR, x) for x in image_ids], dtype=object)

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.range(n, dtype=tf.int32))
    )

    def _make_views(path, idx):
        base = _decode_resize_float255_from_path(path)  # float32 [0,255]
        base_view = base * (1.0 / 255.0)

        if tta_n_aug <= 0:
            return base_view, idx

        views = [base_view]
        for k in range(tta_n_aug):
            seed2 = tf.stack(
                [tf.cast(seed, tf.int32) ^ idx, tf.cast(10007 * (k + 1), tf.int32)],
                axis=0,
            )
            aug = _tta_augment_tf(base, seed2) * (1.0 / 255.0)
            views.append(aug)
        return tf.stack(views, axis=0), idx  # (n_views,H,W,3), idx

    ds = ds.map(_make_views, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(predict_batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    preds_accum = np.zeros((n, 5), dtype=np.float32)

    for views_b, idx_b in ds:
        b = int(views_b.shape[0])
        views_flat = tf.reshape(views_b, [b * n_views, IMG_HEIGHT, IMG_WIDTH, 3])

        preds_flat = model.predict(views_flat, batch_size=256, verbose=0)
        preds = preds_flat.reshape((b, n_views, 5)).mean(axis=1)

        idx_np = idx_b.numpy()
        preds_accum[idx_np] = preds.astype(np.float32, copy=False)

    return preds_accum


image_ids = test_df["image_id"].values
mean_preds = predict_with_tta_batched(
    model,
    image_ids,
    tta_n_aug=TTA_N_AUG,
    predict_batch_size=128,
    seed=42,
)

labels = np.argmax(mean_preds, axis=1).astype(int)
test_results_df = pd.DataFrame({"image_id": image_ids, "label": labels})

submission = pd.read_csv(sample_sub_path)[["image_id"]].merge(
    test_results_df, on="image_id", how="left"
)
assert submission["label"].isna().sum() == 0, "Some test images missing predictions."
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head(5))



## === cell 14
check = pd.read_csv("submission.csv")
print(check.columns.tolist(), check.shape)
print(check.head(3))
assert list(check.columns) == ["image_id", "label"]
assert check.shape[0] == test_df.shape[0]
print("Submission looks valid.")
