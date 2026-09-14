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

3.12

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.82373

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.72175) has done: 'I remove notebook-only magics and the problematic unused imports that trigger the protobuf `MessageFactory` crash, so the script runs in Kaggle’s Python environment. I also fix the broken visualization cell paths (those images don’t exist) by making it robust/optional, without affecting training/inference. Then I correct two accuracy-killing logic bugs in the training pipeline: the validation dataset uses `(img_height, img_height)` instead of `(img_height, img_width)`, and the loss is incorrectly configured with `from_logits=True` despite a `softmax` output. Finally, I make test dataset loading work with the flat `test_images/` directory by creating a temporary “dummy class” folder with symlinks/copies so `image_dataset_from_directory()` can read all test images and produce a valid `submission.csv`.'
- What this solution (achieved 0.69908) has done: 'I fix the immediate runtime crash caused by protobuf/keras import side effects by removing the unused visualization-heavy imports (matplotlib/seaborn) and `EarlyStopping` import from standalone `keras`, and instead import `EarlyStopping` from `tensorflow.keras` after TensorFlow is loaded. Then I keep your model, data loading, loss, and inference logic identical, only making the visualization cells no-ops so the pipeline runs fully headless and reliably in Kaggle. Finally, I keep the test-directory “dummy class” workaround but make it a bit more robust (handle both possible test_images locations) to guarantee a valid `submission.csv` is always written.'
- What this solution (achieved 0.17487) has done: 'I fix the protobuf `MessageFactory` crash by setting a safe protobuf implementation before TensorFlow is imported (this is a known issue in some Kaggle images with TF 2.18 + protobuf 6). I keep your model/training loop intact, but I add deterministic seeding and ensure the dataset pipeline uses the normalization you intended (currently `normalization_layer` is created but not applied to `train_ds`/`val_ds`), which should improve accuracy toward the target without changing the architecture or loss. I also keep the “dummy class folder” test loader, but make sure the submission order exactly matches `sample_submission.csv` (stable mapping) and always writes `submission.csv`. These are minimal, directly relevant changes: they unblock execution and nudge score upward via correct preprocessing and determinism.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/962227439.py in <cell line: 0>()
     13 import numpy as np
     14 import pandas as pd
---> 15 import tensorflow as tf
     16 
     17 from tensorflow.keras.callbacks import EarlyStopping

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
data = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")
data.head()



## === cell 2
data.shape



## === cell 3
data["label"].unique().tolist()



## === cell 4
data["variety"].unique().tolist()



## === cell 5
data.age.describe()



## === cell 6
print(
    "Skipping visualization cell (variety histogram) to keep runtime stable/headless."
)



## === cell 7
print("Skipping visualization cell (label histogram) to keep runtime stable/headless.")



## === cell 8
normal = data[data["label"] == "normal"]
normal = normal[normal["variety"] == "ADT45"]
five_normals = normal.image_id[:5].values
five_normals.tolist()



## === cell 9
dead = data[data["label"] == "dead_heart"]
dead = dead[dead["variety"] == "ADT45"]
five_deads = dead.image_id[:5].values
five_deads.tolist()



## === cell 10
print("Skipping example image grid visualization.")



## === cell 11
print("Skipping fixed example image visualization (paths may not exist).")



## === cell 12
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data_enc = data.copy()
data_enc["label"] = encoder.fit_transform(data_enc["label"])
data_enc["variety"] = encoder.fit_transform(data_enc["variety"])
data_enc.head()



## === cell 13
batch_size = 32
img_height = 224
img_width = 224



## === cell 14
path = "/kaggle/input/paddy-disease-classification/train_images/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3091988843.py in <cell line: 0>()
      1 path = "/kaggle/input/paddy-disease-classification/train_images/"
      2 
----> 3 train_ds = tf.keras.utils.image_dataset_from_directory(
      4     directory=path,
      5     validation_split=0.2,

NameError: name 'tf' is not defined

## === cell 15
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2056292807.py in <cell line: 0>()
----> 1 val_ds = tf.keras.utils.image_dataset_from_directory(
      2     directory=path,
      3     validation_split=0.2,
      4     subset="validation",
      5     seed=SEED,

NameError: name 'tf' is not defined

## === cell 16
class_names = train_ds.class_names
print(class_names)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4019458612.py in <cell line: 0>()
----> 1 class_names = train_ds.class_names
      2 print(class_names)
      3 

NameError: name 'train_ds' is not defined

## === cell 17
for image_batch, label_batch in train_ds:
    print(image_batch.shape)
    print(label_batch.shape)
    break



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/742537254.py in <cell line: 0>()
----> 1 for image_batch, label_batch in train_ds:
      2     print(image_batch.shape)
      3     print(label_batch.shape)
      4     break
      5 

NameError: name 'train_ds' is not defined

## === cell 18
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)


def _scale(x, y):
    x = normalization_layer(x)
    return x, y


train_ds = train_ds.map(_scale, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(_scale, num_parallel_calls=tf.data.AUTOTUNE)

image_batch, label_batch = next(iter(train_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3627849729.py in <cell line: 0>()
----> 1 normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
      2 
      3 
      4 def _scale(x, y):
      5     x = normalization_layer(x)

NameError: name 'tf' is not defined

## === cell 19
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3377069980.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
      3 val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
      4 

NameError: name 'tf' is not defined

## === cell 20
num_classes = len(class_names)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1296292671.py in <cell line: 0>()
----> 1 num_classes = len(class_names)
      2 
      3 # Fix accuracy-killing preprocessing bug: images are already scaled in the dataset
      4 # via normalization_layer, so we must not rescale again inside the model.
      5 # This preserves core architecture/training loop and only corrects input scaling.

NameError: name 'class_names' is not defined

## === cell 21
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1250794155.py in <cell line: 0>()
----> 1 model.compile(
      2     optimizer="adam",
      3     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      4     metrics=["accuracy"],
      5 )

NameError: name 'model' is not defined

## === cell 22
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1,
)

val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

print("Skipping training curves plotting.")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1531914927.py in <cell line: 0>()
----> 1 early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
      2 
      3 history = model.fit(
      4     train_ds,
      5     validation_data=val_ds,

NameError: name 'EarlyStopping' is not defined

## === cell 23
model.summary()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1903595429.py in <cell line: 0>()
----> 1 model.summary()
      2 

NameError: name 'model' is not defined

## === cell 24
loss, accu = model.evaluate(val_ds, verbose=0)
print(f"the Testing loss is {loss:.4f}")
print(f"The testing accuracy is {accu*100:.2f}%")



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2924239265.py in <cell line: 0>()
----> 1 loss, accu = model.evaluate(val_ds, verbose=0)
      2 print(f"the Testing loss is {loss:.4f}")
      3 print(f"The testing accuracy is {accu*100:.2f}%")
      4 

NameError: name 'model' is not defined

## === cell 25
test_data_dir_candidates = [
    "/kaggle/input/paddy-disease-classification/test_images/",
    "/kaggle/input/paddy-disease-classification/paddy-disease-classification/test_images/",
]
test_data_dir = None
for c in test_data_dir_candidates:
    if os.path.isdir(c):
        test_data_dir = c
        break
if test_data_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images/ in candidates: {test_data_dir_candidates}"
    )

print("Using test_data_dir:", test_data_dir)



## === cell 26
tmp_test_root = "/kaggle/working/_tmp_test_images"
tmp_test_class_dir = os.path.join(tmp_test_root, "test")

if os.path.exists(tmp_test_root):
    shutil.rmtree(tmp_test_root)
os.makedirs(tmp_test_class_dir, exist_ok=True)

test_image_files = sorted(
    [f for f in os.listdir(test_data_dir) if f.lower().endswith(".jpg")]
)
if len(test_image_files) == 0:
    raise RuntimeError(f"No .jpg files found in {test_data_dir}")

for fname in test_image_files:
    src = os.path.join(test_data_dir, fname)
    dst = os.path.join(tmp_test_class_dir, fname)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

test_ds = tf.keras.utils.image_dataset_from_directory(
    tmp_test_root,
    labels=None,
    label_mode=None,
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
)

test_ds = test_ds.map(lambda x: normalization_layer(x), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3941904877.py in <cell line: 0>()
     20         shutil.copy2(src, dst)
     21 
---> 22 test_ds = tf.keras.utils.image_dataset_from_directory(
     23     tmp_test_root,
     24     labels=None,

NameError: name 'tf' is not defined

## === cell 27
y_pred = model.predict(test_ds, batch_size=batch_size, verbose=1)
y_pred.shape



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1425271991.py in <cell line: 0>()
----> 1 y_pred = model.predict(test_ds, batch_size=batch_size, verbose=1)
      2 y_pred.shape
      3 

NameError: name 'model' is not defined

## === cell 28
y_pred_classes = y_pred.argmax(axis=1)
y_pred_classes.shape



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3569264811.py in <cell line: 0>()
----> 1 y_pred_classes = y_pred.argmax(axis=1)
      2 y_pred_classes.shape
      3 

NameError: name 'y_pred' is not defined

## === cell 29
y_classes_names = [class_names[x] for x in y_pred_classes]



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2546577703.py in <cell line: 0>()
----> 1 y_classes_names = [class_names[x] for x in y_pred_classes]
      2 

NameError: name 'y_pred_classes' is not defined

## === cell 30
sub = pd.read_csv("/kaggle/input/paddy-disease-classification/sample_submission.csv")

pred_df = pd.DataFrame({"image_id": test_image_files, "label": y_classes_names})
pred_map = dict(zip(pred_df["image_id"], pred_df["label"]))

sub["label"] = sub["image_id"].map(pred_map)
sub["label"] = sub["label"].fillna(class_names[0])

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1976674315.py in <cell line: 0>()
      1 sub = pd.read_csv("/kaggle/input/paddy-disease-classification/sample_submission.csv")
      2 
----> 3 pred_df = pd.DataFrame({"image_id": test_image_files, "label": y_classes_names})
      4 pred_map = dict(zip(pred_df["image_id"], pred_df["label"]))
      5 

NameError: name 'y_classes_names' is not defined
