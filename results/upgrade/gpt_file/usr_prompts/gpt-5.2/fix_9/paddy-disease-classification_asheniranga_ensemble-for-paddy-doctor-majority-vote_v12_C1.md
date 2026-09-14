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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.9815668202764976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17948) has done: 'I fix the environment-crashing import issue in the first cell by removing/deferring optional imports that trigger protobuf/TensorFlow incompatibilities, while keeping the core workflow unchanged. Then I fix the tf.data augmentation bug by applying random crop per-image (not per-batch) so the dataset pipeline can be built and used for training/inference. I also make the label mapping consistent with the directory-inferred class order (instead of a hardcoded map) so predicted indices translate to correct class names and submission accuracy improves legitimately. Finally, I ensure the fallback model always runs if no external ensemble models are found, and that a valid `submission.csv` is always written.'
- What this solution (achieved 0.17948) has done: 'I fix the TensorFlow/protobuf crash by guarding TensorFlow import and forcing a compatible protobuf implementation before importing it, which unblocks the whole pipeline. Then I correct the dataset path issue where an extra nested `train_images/` folder is being interpreted as a class, by automatically resolving the real image root directories. Next I fix the class-name mapping logic so it never KeyErrors and always maps directory-inferred classes to the canonical label set from `train.csv`, while keeping the same training/prediction semantics. Finally, I ensure at least one model always produces predictions (fallback model path), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.17948) has done: 'I fix the TensorFlow/protobuf crash that stops execution by setting the protobuf implementation/version-related env vars before importing TensorFlow and by falling back to a safe import path. Then I correct a label/probability reindexing bug: your current `_reindex_proba_to_canonical` applies the wrong permutation, which severely scrambles class probabilities and explains the very low accuracy. I also make the directory→canonical mapping robust (use train.csv class order but aligned to directory class names), and ensure the submission always aligns exactly to `sample_submission.csv` image order. These changes keep the same model/training approach while legitimately moving accuracy toward the target.'
- What this solution (achieved 0.17832) has done: 'I fix the TensorFlow/protobuf crash in the very first cell by setting the protobuf env vars earlier and forcing the pure-Python protobuf runtime before importing TensorFlow, which removes the `MessageFactory.GetPrototype` failure. Then I fix a major label/probability reindexing bug: your probabilities are already in canonical order after `tf.gather(y, dir_to_canonical, axis=1)`, but `_reindex_proba_to_canonical` permutes them again, scrambling predictions and causing the very low accuracy. I make prediction reindexing a no-op (or only apply when it’s truly needed) so model outputs align with `inverse_map` class names. These changes preserve the same model/training/ensemble logic and should legitimately move accuracy sharply toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf

hub = None
tfa = None
ss = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(False)

DATA_DIR = "../input/paddy-disease-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

print("TF version:", tf.__version__)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("Train CSV exists:", os.path.isfile(TRAIN_CSV_PATH))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3467688566.py in <cell line: 0>()
     12 import pandas as pd
     13 
---> 14 import tensorflow as tf
     15 
     16 hub = None

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

## === cell 1
def random_cut_out(images):
    if tfa is None:
        return images
    return tfa.image.random_cutout(images, (32, 32), constant_values=0)


@tf.function
def center_crop_and_random_augmentations_tf(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (256, 256, 3))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image.numpy()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2245382327.py in <cell line: 0>()
      5 
      6 
----> 7 @tf.function
      8 def center_crop_and_random_augmentations_tf(image):
      9     image = tf.convert_to_tensor(image)

NameError: name 'tf' is not defined

## === cell 2
def _resolve_image_root(dir_path: str) -> str:
    if not os.path.isdir(dir_path):
        return dir_path
    base = os.path.basename(os.path.normpath(dir_path))
    nested = os.path.join(dir_path, base)
    if os.path.isdir(nested):
        try:
            if len(os.listdir(nested)) > 0:
                return nested
        except Exception:
            pass
    return dir_path


TRAIN_DIR = _resolve_image_root(TRAIN_DIR)
TEST_DIR = _resolve_image_root(TEST_DIR)

print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR:", TEST_DIR)

BATCH_SIZE = 16
IMG_SIZE_256 = (256, 256)
IMG_SIZE_300 = (300, 300)
VAL_SPLIT = 0.2

AUTOTUNE = tf.data.AUTOTUNE

train_ds_256 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_256,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds_256 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_256,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

train_ds_300 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_300,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

valid_ds_300 = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE_300,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

num_classes = len(train_ds_256.class_names)

train_df = pd.read_csv(TRAIN_CSV_PATH)

csv_class_names = pd.Index(train_df["label"]).unique().tolist()
dir_class_names = list(train_ds_256.class_names)

if len(csv_class_names) != num_classes:
    print("WARNING: class count mismatch between train.csv and directory inference.")
    print("train.csv classes:", len(csv_class_names), csv_class_names)
    print("dir inferred classes:", len(dir_class_names), dir_class_names)

csv_set = set(csv_class_names)
dir_set = set(dir_class_names)

if dir_set == csv_set:
    class_names = csv_class_names  # canonical
else:
    print("WARNING: Using directory-inferred class order as canonical due to mismatch.")
    unknown = sorted(list(dir_set - csv_set))
    missing = sorted(list(csv_set - dir_set))
    if unknown:
        print("Unknown dir classes vs train.csv:", unknown)
    if missing:
        print("Missing dir classes vs train.csv:", missing)
    class_names = dir_class_names

inverse_map = {i: name for i, name in enumerate(class_names)}
name_to_canonical_idx = {name: i for i, name in enumerate(class_names)}

_dir_norm = {n.strip().lower(): n for n in dir_class_names}
_can_norm = {n.strip().lower(): n for n in class_names}

dir_to_canonical_list = []
for dn in dir_class_names:
    key = dn.strip().lower()
    if key in _can_norm:
        dir_to_canonical_list.append(name_to_canonical_idx[_can_norm[key]])
    else:
        dir_to_canonical_list.append(0)

dir_to_canonical = np.array(dir_to_canonical_list, dtype=np.int64)

canonical_to_dir = np.empty_like(dir_to_canonical)
canonical_to_dir[dir_to_canonical] = np.arange(len(dir_to_canonical), dtype=np.int64)


def _prep_train_256(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.map_fn(
        center_crop_and_random_augmentations_tf, x, fn_output_signature=tf.float32
    )
    y = tf.gather(y, dir_to_canonical, axis=1)
    return x, y


def _prep_valid(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.gather(y, dir_to_canonical, axis=1)
    return x, y


def _prep_train_300(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.map_fn(
        center_crop_and_random_augmentations_tf, x, fn_output_signature=tf.float32
    )
    y = tf.gather(y, dir_to_canonical, axis=1)
    return x, y


train_datagen = train_ds_256.map(_prep_train_256, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)
valid_datagen = valid_ds_256.map(_prep_valid, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

train_datagen_300 = train_ds_300.map(
    _prep_train_300, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)
valid_datagen_300 = valid_ds_300.map(_prep_valid, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)

print("Directory-inferred classes:", dir_class_names)
print("Canonical classes:", class_names)
print("num_classes:", num_classes)
print("dir_to_canonical:", dir_to_canonical.tolist())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3473959196.py in <cell line: 0>()
     13 
     14 
---> 15 TRAIN_DIR = _resolve_image_root(TRAIN_DIR)
     16 TEST_DIR = _resolve_image_root(TEST_DIR)
     17 

NameError: name 'TRAIN_DIR' is not defined

## === cell 3
def _list_test_files(test_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in os.listdir(test_dir):
        if name.lower().endswith(exts):
            files.append(os.path.join(test_dir, name))
    files.sort()
    return files


test_files = _list_test_files(TEST_DIR)
print("Num test files:", len(test_files))
if len(test_files) == 0:
    raise RuntimeError("No test images found in TEST_DIR")


def _decode_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def _make_test_ds(target_hw):
    ds = tf.data.Dataset.from_tensor_slices(test_files)
    ds = ds.map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda x: tf.image.resize(x, target_hw, method="bilinear", antialias=False),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


test_data_256 = _make_test_ds(IMG_SIZE_256)
test_data_300 = _make_test_ds(IMG_SIZE_300)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4291355578.py in <cell line: 0>()
      9 
     10 
---> 11 test_files = _list_test_files(TEST_DIR)
     12 print("Num test files:", len(test_files))
     13 if len(test_files) == 0:

NameError: name 'TEST_DIR' is not defined

## === cell 4
def _custom_objects_for_load():
    if hub is not None:
        return {"KerasLayer": hub.KerasLayer}
    return {}


def safe_load_model(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(
                path, custom_objects=_custom_objects_for_load()
            )
        except Exception as e:
            print(f"Failed to load model at {path}: {repr(e)}")
            return None
    return None


m1 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_b3.hdf5"
)
m2 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/effnet_v2_m.hdf5"
)
m3 = safe_load_model(
    "../input/paddy-doc-ensemble-models/ensemble_estimators/xception.hdf5"
)
m4 = safe_load_model("../input/paddydocoutputs/model_resnet150.hdf5")
m5 = safe_load_model(
    "../input/k/kasunpramodya/paddy-doctor-training/model_effnet_s.hdf5"
)
m6 = safe_load_model("../input/notebooka9ca40495e/model_effnet_s.hdf5")

loaded = [m is not None for m in [m1, m2, m3, m4, m5, m6]]
print("Loaded ensemble models:", loaded, "count:", sum(loaded))

model_outputs_canonical = {}

if sum(loaded) == 0:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(256, 256, 3),
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(256, 256, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    fallback_model = tf.keras.Model(inp, out)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    EPOCHS = 3
    fallback_model.fit(
        train_datagen,
        validation_data=valid_datagen,
        epochs=EPOCHS,
        verbose=1,
    )

    m1, m2, m3, m4, m5, m6 = fallback_model, None, None, None, None, None
    model_outputs_canonical[id(m1)] = True
else:
    for mm in [m1, m2, m3, m4, m5, m6]:
        if mm is not None:
            model_outputs_canonical[id(mm)] = False




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4020736082.py in <cell line: 0>()
     39 
     40 if sum(loaded) == 0:
---> 41     base = tf.keras.applications.EfficientNetB0(
     42         include_top=False,
     43         weights="imagenet",

NameError: name 'tf' is not defined

## === cell 5
def _needs_300(model):
    try:
        ish = model.input_shape
        if isinstance(ish, list):
            ish = ish[0]
        return ish[1] == 300 and ish[2] == 300
    except Exception:
        return False


model_train_test_score = []
for model in [m1, m2, m3, m4, m5, m6]:
    if model is None:
        model_train_test_score.append(None)
        continue
    ds = valid_datagen_300 if _needs_300(model) else valid_datagen
    try:
        model_train_test_score.append(model.evaluate(ds, verbose=0))
    except Exception as e:
        print("Evaluation failed for a model; skipping. Error:", repr(e))
        model_train_test_score.append(None)

model_train_test_score




## === cell 6
def _reindex_proba_to_canonical(p, assume_dir_order: bool):
    """
    Fix (score): external models are likely outputting probabilities in directory class order.
    Our submission labels are in canonical `class_names` order. If model output is in dir order,
    convert to canonical order by gathering with dir_to_canonical along class axis.
    """
    if p is None:
        return None
    if not assume_dir_order:
        return p
    return p[:, dir_to_canonical]


def predict_model(model, ds256, ds300):
    if model is None:
        return None
    ds = ds300 if _needs_300(model) else ds256
    try:
        p = model.predict(ds, verbose=1)
        assume_dir_order = not model_outputs_canonical.get(id(model), False)
        return _reindex_proba_to_canonical(p, assume_dir_order=assume_dir_order)
    except Exception as e:
        print("Prediction failed for a model; skipping. Error:", repr(e))
        return None


m1_p = predict_model(m1, test_data_256, test_data_300)
m2_p = predict_model(m2, test_data_256, test_data_300)
m3_p = predict_model(m3, test_data_256, test_data_300)
m4_p = predict_model(m4, test_data_256, test_data_300)
m5_p = predict_model(m5, test_data_256, test_data_300)
m6_p = predict_model(m6, test_data_256, test_data_300)

pred_probas = [p for p in [m1_p, m2_p, m3_p, m4_p, m5_p, m6_p] if p is not None]
print("Num prediction arrays:", len(pred_probas))
print("Prediction shape example:", pred_probas[0].shape if pred_probas else None)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145026524.py in <cell line: 0>()
     26 
     27 
---> 28 m1_p = predict_model(m1, test_data_256, test_data_300)
     29 m2_p = predict_model(m2, test_data_256, test_data_300)
     30 m3_p = predict_model(m3, test_data_256, test_data_300)

NameError: name 'test_data_256' is not defined

## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
image_ids = sample_sub["image_id"].astype(str).tolist()

fallback_label = (
    "normal"
    if "normal" in class_names
    else (class_names[0] if class_names else "normal")
)

if len(pred_probas) == 0:
    print(
        "WARNING: No model predictions available; writing fallback labels for all test images."
    )
    pred_df = pd.DataFrame(
        {"image_id": image_ids, "label": [fallback_label] * len(image_ids)}
    )
else:
    avg_proba = np.mean(np.stack(pred_probas, axis=0), axis=0)
    pred_idx = np.argmax(avg_proba, axis=1)
    pred_label = [inverse_map[int(i)] for i in pred_idx]

    pred_from_files = pd.DataFrame(
        {"image_id": [os.path.basename(p) for p in test_files], "label": pred_label}
    )
    pred_df = sample_sub[["image_id"]].merge(pred_from_files, on="image_id", how="left")
    pred_df["label"] = pred_df["label"].fillna(fallback_label)

print(pred_df.head())
print("Submission shape:", pred_df.shape)
print("Missing labels:", pred_df["label"].isna().sum())
print("Unique predicted labels (sample):", pred_df["label"].value_counts().head(10))

submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print("Columns:", pred_df.columns.tolist())
print(pred_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1501908764.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
      2 image_ids = sample_sub["image_id"].astype(str).tolist()
      3 
      4 fallback_label = (
      5     "normal"

NameError: name 'SAMPLE_SUB_PATH' is not defined
