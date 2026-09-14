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
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import json
import cv2
from PIL import Image

try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16

PRE_TRAINED_MODEL = "../input/unionmodelv07/Cassava_Best_UnitedModel_V07.hdf5"



## === cell 6
from albumentations import (
    Compose,
    HorizontalFlip,
    CenterCrop,
    ToFloat,
    ShiftScaleRotate,
    RandomBrightnessContrast,
)



## === cell 7
AUGMENTATIONS_TRAIN = Compose(
    [
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
        ShiftScaleRotate(
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



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass



## === cell 9
from tensorflow.keras.utils import Sequence
import random


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        img_bgr = cv2.resize(
            img_bgr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4
        )
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_data = Image.open(image_path).convert("RGB")
    img_data = np.array(img_data.resize((IMG_HEIGHT, IMG_WIDTH), RESAMPLE))
    return img_data


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x = np.array(list(x_set))
        self.y = None if y_set is None else np.array(list(y_set))
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        img_list = []
        for x in batch_x:
            if self.data_type == "TRAIN_DATA":
                random_num = random.uniform(0, 1)
                if random_num > 0.5:
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x)
            else:
                if self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x)

            img_list.append(img_data)

        img_array = np.stack(img_list, axis=0).astype(np.float32)
        return img_array, np.array(batch_y)




## === cell 10
from tensorflow.keras.models import load_model

candidate_paths = [
    PRE_TRAINED_MODEL,
    "/kaggle/input/unionmodelv07/Cassava_Best_UnitedModel_V07.hdf5",
    "/kaggle/input/unionmodelv07/cassava_best_unitedmodel_v07.hdf5",
]

model = None
found_path = None
for p in candidate_paths:
    if os.path.exists(p):
        found_path = p
        break

if found_path is not None:
    model = load_model(found_path, compile=False)
    print("Loaded pretrained model:", found_path)
else:
    print(
        "Pretrained model not found. Training a fallback model to produce a valid submission..."
    )



## === cell 11
if model is None:
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_frac = 0.2
    n_val = int(len(train_df) * val_frac)
    val_df = train_df.iloc[:n_val].reset_index(drop=True)
    trn_df = train_df.iloc[n_val:].reset_index(drop=True)

    train_gen = AugmentedImageSequence(
        "TRAIN",
        "TRAIN_DATA",
        trn_df["image_id"],
        trn_df["label"],
        batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    val_gen = AugmentedImageSequence(
        "VALID",
        "VALIDATE_DATA",
        val_df["image_id"],
        val_df["label"],
        batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
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
        train_gen,
        validation_data=val_gen,
        epochs=3,
        verbose=1,
    )



## === cell 12
model.summary()



## === cell 13
from tensorflow.keras.preprocessing.image import apply_affine_transform

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

_TTA_SEED = 12345
_tta_rng = np.random.RandomState(_TTA_SEED)


def _standardize_transform_params(params):
    params = dict(params)
    params.setdefault("theta", 0)
    params.setdefault("tx", 0)
    params.setdefault("ty", 0)
    params.setdefault("shear", 0)
    params.setdefault("zx", 1)
    params.setdefault("zy", 1)
    params.setdefault("flip_horizontal", False)
    params.setdefault("flip_vertical", False)
    params.setdefault("channel_shift_intensity", None)
    params.setdefault("brightness", None)
    return params


_tta_params_list = []
_dummy_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
for _ in range(5):
    p = _TTA_DATAGEN.get_random_transform(
        _dummy_shape, seed=int(_tta_rng.randint(0, 2**31 - 1))
    )
    _tta_params_list.append(_standardize_transform_params(p))


def _apply_tta_params(image_rgb_uint8, params, out_float32):
    x = image_rgb_uint8.astype(np.float32, copy=False)

    if params.get("brightness", None) is not None:
        pass

    x = apply_affine_transform(
        x,
        theta=params["theta"],
        tx=params["tx"],
        ty=params["ty"],
        shear=params["shear"],
        zx=params["zx"],
        zy=params["zy"],
        row_axis=0,
        col_axis=1,
        channel_axis=2,
        fill_mode=_TTA_DATAGEN.fill_mode,
        cval=_TTA_DATAGEN.cval,
        order=1,
    )

    if params.get("flip_horizontal", False):
        x = x[:, ::-1, :]
    if params.get("flip_vertical", False):
        x = x[::-1, :, :]

    out_float32[:] = x
    return out_float32


def get_augmented_images_to_buffer(image_rgb_uint8, out_buf_float32):
    out_buf_float32[0] = image_rgb_uint8.astype(np.float32, copy=False)
    for j, params in enumerate(_tta_params_list, start=1):
        _apply_tta_params(image_rgb_uint8, params, out_buf_float32[j])
    return out_buf_float32




## === cell 14
def _model_has_rescaling(m):
    try:
        first = m.layers[0]
        if isinstance(first, keras.Model) and len(first.layers) > 0:
            first = first.layers[0]
        return isinstance(first, keras.layers.Rescaling)
    except Exception:
        return False


_NEEDS_MANUAL_SCALE = not _model_has_rescaling(model)
print("Manual /255 scaling applied at inference:", _NEEDS_MANUAL_SCALE)

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
image_ids = sample_sub["image_id"].values

tta_views = 6  # identity + 5 transforms

theta = tf.constant([p["theta"] for p in _tta_params_list], dtype=tf.float32) * (
    np.pi / 180.0
)
tx = tf.constant([p["tx"] for p in _tta_params_list], dtype=tf.float32)
ty = tf.constant([p["ty"] for p in _tta_params_list], dtype=tf.float32)
shear = tf.constant([p["shear"] for p in _tta_params_list], dtype=tf.float32) * (
    np.pi / 180.0
)
zx = tf.constant([p["zx"] for p in _tta_params_list], dtype=tf.float32)
zy = tf.constant([p["zy"] for p in _tta_params_list], dtype=tf.float32)
flip_h = tf.constant(
    [bool(p.get("flip_horizontal", False)) for p in _tta_params_list], dtype=tf.bool
)
flip_v = tf.constant(
    [bool(p.get("flip_vertical", False)) for p in _tta_params_list], dtype=tf.bool
)

_FILL_MODE = "NEAREST"


def _tf_load_image_uint8(image_id):
    path = tf.strings.join([tf.constant(TEST_DIR), image_id])
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(tf.round(img), tf.uint8)
    return img, image_id


def _transform_to_projective(theta, tx, ty, shear, zx, zy):
    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)

    rot = tf.stack(
        [
            tf.stack([cos_t, -sin_t, 0.0]),
            tf.stack([sin_t, cos_t, 0.0]),
            tf.stack([0.0, 0.0, 1.0]),
        ]
    )

    sh = tf.stack(
        [
            tf.stack([1.0, -tf.sin(shear), 0.0]),
            tf.stack([0.0, tf.cos(shear), 0.0]),
            tf.stack([0.0, 0.0, 1.0]),
        ]
    )

    zm = tf.stack(
        [
            tf.stack([1.0 / zx, 0.0, 0.0]),
            tf.stack([0.0, 1.0 / zy, 0.0]),
            tf.stack([0.0, 0.0, 1.0]),
        ]
    )

    tr = tf.stack(
        [tf.stack([1.0, 0.0, tx]), tf.stack([0.0, 1.0, ty]), tf.stack([0.0, 0.0, 1.0])]
    )

    m = tf.linalg.matmul(tr, tf.linalg.matmul(rot, tf.linalg.matmul(sh, zm)))

    cx = (tf.cast(IMG_WIDTH, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_HEIGHT, tf.float32) - 1.0) / 2.0
    c = tf.stack(
        [tf.stack([1.0, 0.0, cx]), tf.stack([0.0, 1.0, cy]), tf.stack([0.0, 0.0, 1.0])]
    )
    c_inv = tf.stack(
        [
            tf.stack([1.0, 0.0, -cx]),
            tf.stack([0.0, 1.0, -cy]),
            tf.stack([0.0, 0.0, 1.0]),
        ]
    )

    m = tf.linalg.matmul(c, tf.linalg.matmul(m, c_inv))

    m_inv = tf.linalg.inv(m)
    a0 = m_inv[0, 0]
    a1 = m_inv[0, 1]
    a2 = m_inv[0, 2]
    a3 = m_inv[1, 0]
    a4 = m_inv[1, 1]
    a5 = m_inv[1, 2]
    a6 = m_inv[2, 0]
    a7 = m_inv[2, 1]
    return tf.stack([a0, a1, a2, a3, a4, a5, a6, a7])


proj = tf.stack(
    [
        _transform_to_projective(theta[i], tx[i], ty[i], shear[i], zx[i], zy[i])
        for i in range(5)
    ],
    axis=0,
)


def _apply_tta_all_views(batch_uint8):
    batch_uint8 = tf.convert_to_tensor(batch_uint8)
    bsz = tf.shape(batch_uint8)[0]

    x0 = tf.cast(batch_uint8, tf.float32)  # identity view

    def _one_view(i):
        t = proj[i]
        y = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.cast(batch_uint8, tf.float32),
            transforms=tf.repeat(t[None, :], bsz, axis=0),
            output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode=_FILL_MODE,
            fill_value=0.0,
        )
        y = tf.where(flip_h[i], tf.reverse(y, axis=[2]), y)
        y = tf.where(flip_v[i], tf.reverse(y, axis=[1]), y)
        return y

    views_5 = tf.map_fn(
        _one_view,
        tf.range(5, dtype=tf.int32),
        fn_output_signature=tf.float32,
        parallel_iterations=5,
    )
    views = tf.concat([x0[None, ...], views_5], axis=0)

    if _NEEDS_MANUAL_SCALE:
        views = views / 255.0
    return views


@tf.function(reduce_retracing=True, jit_compile=True)
def _predict_probs_tta_mean(batch_uint8):
    views = _apply_tta_all_views(batch_uint8)  # [6, B, H, W, C]
    v = tf.shape(views)[0]
    b = tf.shape(views)[1]

    flat = tf.reshape(views, (v * b, IMG_HEIGHT, IMG_WIDTH, 3))
    probs = model(flat, training=False)  # [6*B, num_classes]
    probs = tf.reshape(probs, (v, b, -1))  # [6, B, num_classes]
    return tf.reduce_mean(probs, axis=0)  # [B, num_classes]


infer_batch = 128
infer_batch = min(infer_batch, len(image_ids))

ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_ids))
ds = ds.map(_tf_load_image_uint8, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(infer_batch, drop_remainder=False)
ds = ds.prefetch(tf.data.AUTOTUNE)

pred_labels = np.empty(len(image_ids), dtype=np.int64)

offset = 0
for batch_uint8, batch_ids in ds:
    bsz = int(batch_uint8.shape[0])
    probs_mean = _predict_probs_tta_mean(batch_uint8)
    pred_labels[offset : offset + bsz] = (
        tf.argmax(probs_mean, axis=1).numpy().astype(np.int64)
    )
    offset += bsz

submission = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 15
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id", "label"]
assert len(sub_check) == len(
    pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
)
sub_check.head(3)
