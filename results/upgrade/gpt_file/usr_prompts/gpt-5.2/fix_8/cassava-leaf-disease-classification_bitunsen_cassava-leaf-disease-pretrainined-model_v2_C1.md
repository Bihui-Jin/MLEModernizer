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

0.8833484436385615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the `albumentations` import errors by switching to the current transform names (`RandomBrightnessContrast` instead of removed `RandomBrightness/RandomContrast`) while keeping the same augmentation intent. I also fix the TensorFlow/protobuf runtime crash (`MessageFactory.GetPrototype`) by removing the unnecessary `albumentations` dependency entirely and implementing equivalent augmentations using `tf.image` inside the `Sequence` (this preserves the same training approach and semantics: random flips/brightness/contrast + resize + normalization). Then I ensure generators are defined before training/prediction so later cells don’t fail with `NameError`, and I keep the submission format identical to `sample_submission.csv` while always writing `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'We fix the protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the known incompatibility in some Kaggle TF/protobuf builds. Next, we fix the Keras 3 API break where `model.fit()`/`model.predict()` no longer accept `workers/use_multiprocessing/max_queue_size` by removing those arguments (keeping the same generators, epochs, and model logic). Finally, we keep the submission formatting identical to `sample_submission.csv` and always write `submission.csv` successfully.'

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"




## === cell 2
import matplotlib.pyplot as plt
from PIL import Image

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)

print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("Labels:", label_list, "NUM_CLASSES:", NUM_CLASSES)

input_files = os.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")




## === cell 3
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv2/Cassava_Model_V06.hdf5"

if hasattr(Image, "Resampling"):
    PIL_RESAMPLE = Image.Resampling.LANCZOS
else:
    PIL_RESAMPLE = Image.LANCZOS

AUGMENTATIONS_TRAIN = "tf_image_aug_v1"
AUGMENTATIONS_TEST = "tf_image_noaug_v1"




## === cell 4
import tensorflow as tf
from tensorflow.keras.utils import Sequence

tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

_IMAGE_CACHE = {}  # key: ("TRAIN_DATA"/"TEST_DATA", image_id) -> np.uint8(H,W,3)


def load_single_image(data_type, image_id):
    key = (data_type, image_id)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    if cv2 is not None:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), resample=PIL_RESAMPLE)
            arr = np.array(img, dtype=np.uint8)
            _IMAGE_CACHE[key] = arr
            return arr
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(
            img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4
        )
        arr = img_rgb.astype(np.uint8, copy=False)
        _IMAGE_CACHE[key] = arr
        return arr

    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), resample=PIL_RESAMPLE)
    arr = np.array(img, dtype=np.uint8)
    _IMAGE_CACHE[key] = arr
    return arr


@tf.function
def _augment_batch_train(images_uint8, seed2_int32):
    x = tf.image.convert_image_dtype(images_uint8, tf.float32)

    seed2_int32 = tf.convert_to_tensor(seed2_int32, dtype=tf.int32)
    seed0 = seed2_int32 + tf.constant([0, 0], dtype=tf.int32)
    seed1 = seed2_int32 + tf.constant([0, 1], dtype=tf.int32)
    seed2 = seed2_int32 + tf.constant([0, 2], dtype=tf.int32)
    seed3 = seed2_int32 + tf.constant([0, 3], dtype=tf.int32)

    x = tf.image.stateless_random_flip_left_right(x, seed=seed0)
    x = tf.image.stateless_random_flip_up_down(x, seed=seed1)
    x = tf.image.stateless_random_brightness(x, max_delta=0.2, seed=seed2)
    x = tf.image.stateless_random_contrast(x, lower=0.8, upper=1.2, seed=seed3)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x


@tf.function
def _preprocess_batch_test(images_uint8):
    return tf.image.convert_image_dtype(images_uint8, tf.float32)


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x = list(x_set)
        self.y = None if y_set is None else list(y_set)
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = None
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        b = len(batch_x)
        images_uint8 = np.empty((b, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        _load = load_single_image
        _dt = self.data_type
        for j, x_id in enumerate(batch_x):
            images_uint8[j] = _load(_dt, x_id)

        if self.data_type == "TRAIN_DATA":
            do_aug = np.fromiter(
                (random.uniform(0, 1) > 0.5 for _ in range(b)), dtype=np.bool_, count=b
            )
            if do_aug.any():
                seed2 = np.array(
                    [SEED + idx * 1000, SEED + idx * 1000 + 7], dtype=np.int32
                )
                aug_batch = _augment_batch_train(
                    tf.convert_to_tensor(images_uint8), tf.convert_to_tensor(seed2)
                )
                out = images_uint8.astype(np.float32) / 255.0
                out_aug = aug_batch.numpy()
                out[do_aug] = out_aug[do_aug]
                img_array = out.astype(np.float32, copy=False)
            else:
                img_array = (images_uint8.astype(np.float32) / 255.0).astype(
                    np.float32, copy=False
                )
        else:
            img_array = (
                _preprocess_batch_test(tf.convert_to_tensor(images_uint8))
                .numpy()
                .astype(np.float32, copy=False)
            )

        if self.mode == "TEST":
            return img_array, np.zeros((b,), dtype=np.int32)

        return img_array, np.array(batch_y, dtype=np.int32)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2832067877.py in <cell line: 0>()
----> 1 import tensorflow as tf
      2 from tensorflow.keras.utils import Sequence
      3 
      4 tf.random.set_seed(SEED)
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

filepaths = TRAIN_DIR + train_df["image_id"].astype(str)
exists_mask = filepaths.map(os.path.exists).to_numpy()
train_df = train_df.loc[exists_mask].reset_index(drop=True)

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"].values,
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_gen = AugmentedImageSequence(
    mode="TRAIN",
    data_set_type="TRAIN_DATA",
    x_set=tr_df["image_id"],
    y_set=tr_df["label"],
    batch_size=batch_size,
    augmentations=AUGMENTATIONS_TRAIN,
)
val_gen = AugmentedImageSequence(
    mode="VALIDATE",
    data_set_type="VALIDATE_DATA",
    x_set=va_df["image_id"],
    y_set=va_df["label"],
    batch_size=batch_size,
    augmentations=AUGMENTATIONS_TEST,
)

test_gen = AugmentedImageSequence(
    mode="TEST",
    data_set_type="TEST_DATA",
    x_set=sample_sub["image_id"],
    y_set=None,
    batch_size=batch_size,
    augmentations=AUGMENTATIONS_TEST,
)

print(
    "Train batches:",
    len(train_gen),
    "Val batches:",
    len(val_gen),
    "Test batches:",
    len(test_gen),
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1903297743.py in <cell line: 0>()
     19 va_df = train_df.iloc[val_idx].reset_index(drop=True)
     20 
---> 21 train_gen = AugmentedImageSequence(
     22     mode="TRAIN",
     23     data_set_type="TRAIN_DATA",

NameError: name 'AugmentedImageSequence' is not defined

## === cell 6
from tensorflow.keras import layers, models


def build_fallback_model(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES
):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


if os.path.exists(PRE_TRAINED_MODEL):
    from tensorflow.keras.models import load_model

    model = load_model(PRE_TRAINED_MODEL)
    print("Loaded pretrained model:", PRE_TRAINED_MODEL)
else:
    print(
        f"Pretrained model not found at {PRE_TRAINED_MODEL}. Training fallback model instead."
    )
    model = build_fallback_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        verbose=1,
        workers=max(1, (os.cpu_count() or 2) - 1),
        use_multiprocessing=True,
        max_queue_size=16,
    )

model.summary()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1091415952.py in <cell line: 0>()
----> 1 from tensorflow.keras import layers, models
      2 
      3 
      4 def build_fallback_model(
      5     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 7
pred_probs = model.predict(
    test_gen,
    verbose=1,
    workers=max(1, (os.cpu_count() or 2) - 1),
    use_multiprocessing=True,
    max_queue_size=16,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

submission = sample_sub.copy()
submission["label"] = pred_labels

submission = submission[["image_id", "label"]]
assert len(submission) == len(sample_sub), "Submission row count mismatch."
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3610774254.py in <cell line: 0>()
      1 # --- Speed fix: enable multi-worker prefetching during prediction as well.
----> 2 pred_probs = model.predict(
      3     test_gen,
      4     verbose=1,
      5     workers=max(1, (os.cpu_count() or 2) - 1),

NameError: name 'model' is not defined
