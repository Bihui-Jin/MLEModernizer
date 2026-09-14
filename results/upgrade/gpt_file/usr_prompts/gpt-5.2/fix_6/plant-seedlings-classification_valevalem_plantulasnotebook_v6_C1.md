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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.8879093198992444

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import tensorflow as tf

tf.keras.utils.set_random_seed(123)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/378183446.py in <cell line: 0>()
      9 os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
     10 
---> 11 import tensorflow as tf
     12 
     13 tf.keras.utils.set_random_seed(123)

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
if os.path.exists("/kaggle/input"):
    DATA_ROOT = "/kaggle/input/plant-seedlings-classification"
    print("Estamos en Kaggle")
else:
    DATA_ROOT = "./data"
    print("Estamos en Colab")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR exists:", os.path.isdir(TEST_DIR))
print("SAMPLE_SUB exists:", os.path.isfile(SAMPLE_SUB_PATH))




## === cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import matplotlib.image as mpimg

RUN_EDA = os.environ.get("RUN_EDA", "0") == "1"




## === cell 3
train_dir = TRAIN_DIR
categories = sorted(
    [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
)
print("Num categories found:", len(categories))
print(categories)




## === cell 4
if RUN_EDA:
    plt.figure(figsize=(9, 12))
    shown = 0
    for category in categories:
        files = [
            f
            for f in os.listdir(os.path.join(train_dir, category))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            continue
        img_name = random.choice(files)
        img_path = os.path.join(train_dir, category, img_name)

        shown += 1
        plt.subplot(4, 3, shown)
        plt.imshow(Image.open(img_path))
        plt.title(category)
        plt.axis("off")
        if shown >= 12:
            break
    plt.tight_layout()
    plt.show()




## === cell 5
if RUN_EDA:
    data_stats = []
    for category in categories:
        cat_path = os.path.join(train_dir, category)
        files = [
            f
            for f in os.listdir(cat_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            continue

        sample_img_path = os.path.join(cat_path, files[random.randrange(len(files))])
        with Image.open(sample_img_path) as img:
            width, height = img.size

        data_stats.append(
            {
                "Category": category,
                "Count": len(files),
                "Sample Resolution": f"{width}x{height}",
            }
        )

    df_stats = pd.DataFrame(data_stats)
    print(df_stats)
    print(f"\nNumero total de imagenes: {df_stats['Count'].sum()}")




## === cell 6
if RUN_EDA:
    primera_foto = [
        f
        for f in os.listdir(os.path.join(TRAIN_DIR, "Black-grass"))
        if f.lower().endswith(".png")
    ][0]
    full_path = os.path.join(TRAIN_DIR, "Black-grass", primera_foto)

    img = mpimg.imread(full_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("dimensiones de la foto:", np.shape(img))
    print("Tipo de dato de la foto:", img.dtype)
    print("valor minimo de los pixeles:", img.min())
    print("valor maximo de los pixeles", img.max())




## === cell 7
if RUN_EDA:
    metadata_table = []
    for category in categories:
        files = [
            f
            for f in os.listdir(os.path.join(train_dir, category))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            continue
        img_name = files[0]
        img_path = os.path.join(train_dir, category, img_name)
        img = mpimg.imread(img_path)

        metadata_table.append(
            {
                "Category": category,
                "Dimensiones primera foto": np.shape(img),
                "tipo de datos": img.dtype,
                "valor minimo de los pixeles": float(img.min()),
                "valor maximo de los pixeles": float(img.max()),
            }
        )

    datos = pd.DataFrame(metadata_table)
    print(datos)




## === cell 8
from tensorflow.keras import layers, models




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3312123074.py in <cell line: 0>()
----> 1 from tensorflow.keras import layers, models
      2 
      3 

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

## === cell 9
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

NUM_CLASSES = len(train_ds.class_names)
print("Detected class names:", train_ds.class_names)
print("NUM_CLASSES:", NUM_CLASSES)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1067820901.py in <cell line: 0>()
      2 BATCH_SIZE = 32
      3 
----> 4 train_ds = tf.keras.utils.image_dataset_from_directory(
      5     TRAIN_DIR,
      6     validation_split=0.15,

NameError: name 'tf' is not defined

## === cell 10
AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.autotune.enabled = True
except Exception:
    pass

train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, "train_cache")
val_cache_path = os.path.join(cache_dir, "val_cache")

train_ds = train_ds.cache(train_cache_path).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache(val_cache_path).prefetch(buffer_size=AUTOTUNE)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/798436148.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 
      3 options = tf.data.Options()
      4 options.experimental_deterministic = True
      5 try:

NameError: name 'tf' is not defined

## === cell 11
model_cnn = models.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model_cnn.summary()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3296345405.py in <cell line: 0>()
----> 1 model_cnn = models.Sequential(
      2     [
      3         layers.Input(shape=(224, 224, 3)),
      4         layers.RandomFlip("horizontal_and_vertical"),
      5         layers.RandomRotation(0.2),

NameError: name 'models' is not defined

## === cell 12
model_cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3428913691.py in <cell line: 0>()
      2 # jit_compile compiles the same training step graph for speed; it doesn't alter the model,
      3 # optimizer, loss, or metrics (only negligible FP differences possible).
----> 4 model_cnn.compile(
      5     optimizer="adam",
      6     loss="categorical_crossentropy",

NameError: name 'model_cnn' is not defined

## === cell 13
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_seedling_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2130789908.py in <cell line: 0>()
----> 1 early_stop = tf.keras.callbacks.EarlyStopping(
      2     monitor="val_loss", patience=4, restore_best_weights=True
      3 )
      4 
      5 reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

NameError: name 'tf' is not defined

## === cell 14
history = model_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1925305316.py in <cell line: 0>()
----> 1 history = model_cnn.fit(
      2     train_ds,
      3     validation_data=val_ds,
      4     epochs=20,
      5     callbacks=[early_stop, reduce_lr, checkpoint],

NameError: name 'model_cnn' is not defined

## === cell 15
test_loss, test_acc = model_cnn.evaluate(val_ds)
print("Accuracy de la evaluacion:", test_acc)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4207863511.py in <cell line: 0>()
----> 1 test_loss, test_acc = model_cnn.evaluate(val_ds)
      2 print("Accuracy de la evaluacion:", test_acc)
      3 
      4 

NameError: name 'model_cnn' is not defined

## === cell 16
print("Skipping unused val_ds predictions to save time.")




## === cell 17
def visualize_predictions(model, dataset, class_names, n=18):
    images, labels = next(iter(dataset))
    preds = model.predict(images, verbose=0)

    plt.figure(figsize=(12, 25))
    n = min(n, images.shape[0])
    for i in range(n):
        ax = plt.subplot(6, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        actual_idx = int(np.argmax(labels[i]))
        predict_idx = int(np.argmax(preds[i]))
        color = "green" if actual_idx == predict_idx else "red"

        plt.title(
            f"Real: {class_names[actual_idx]}\nPredicho: {class_names[predict_idx]}",
            color=color,
            fontsize=10,
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if RUN_EDA:
    visualize_predictions(model_cnn, val_ds, train_ds.class_names)




## === cell 18
test_dir = TEST_DIR
test_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

class_names = train_ds.class_names
print(f"Preprocesando {len(test_files)} imagenes...")

test_paths = [os.path.join(test_dir, f) for f in test_files]
path_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _load_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_png(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


test_cache_path = os.path.join(cache_dir, "test_cache")

path_ds = path_ds.map(_load_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds2 = (
    path_ds.with_options(options)
    .batch(BATCH_SIZE)
    .cache(test_cache_path)
    .prefetch(AUTOTUNE)
)

probs = model_cnn.predict(test_ds2, verbose=0)
pred_idx = np.argmax(probs, axis=1)
pred_labels = [class_names[int(i)] for i in pred_idx]

submission_df = pd.DataFrame({"file": test_files, "species": pred_labels})

if os.path.isfile(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df = sample_sub[["file"]].merge(submission_df, on="file", how="left")
    if submission_df["species"].isna().any():
        submission_df["species"] = submission_df["species"].fillna(class_names[0])

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Fichero guardado como: {out_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3678666254.py in <cell line: 0>()
      4 )
      5 
----> 6 class_names = train_ds.class_names
      7 print(f"Preprocesando {len(test_files)} imagenes...")
      8 

NameError: name 'train_ds' is not defined
