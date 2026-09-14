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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

32.44366

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78762) has done: 'I fix the Keras/protobuf import/runtime error by using `tf.keras` consistently (avoiding mixed `keras`/`tensorflow.keras` imports that break in this environment). I update deprecated/removed imports (`keras.layers.convolutional`) and restore `ImageDataGenerator` availability, keeping your CNN architecture and training loop intact. I also fix the training/validation wiring bugs (validation was accidentally using the training set) and ensure preprocessing is consistent for both train and test (rescale), which should also improve log-loss versus the broken pipeline. Finally, I make sure the model checkpoint uses the required `.keras` suffix and that a valid submission CSV is written with columns exactly matching `sample_submission.csv`.'
- What this solution (achieved 4.78689) has done: 'The timeout is dominated by two things: (1) Python-side image loading into lists followed by large conversions/copies, and (2) `ImageDataGenerator(zca_whitening=True)` which forces an expensive ZCA computation (`datagen.fit`) over the whole training set and then applies it per-batch. To preserve the exact model/training logic while making it fast, I (a) switch image ingestion to a preallocated NumPy array (same pixels, same resize, same dtype) to remove Python list growth and extra copies, and (b) cache the ZCA statistics to disk and reuse them on subsequent runs so you don’t pay the ZCA `fit` cost every time (training behavior remains identical once the cache exists). I also add `workers`/`use_multiprocessing` to `model.fit` so the generator preprocessing runs in parallel without changing the data semantics, and I remove display-only `head()` calls that can cost time in notebook environments.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
import cv2

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3964705795.py in <cell line: 0>()
     15     pass
     16 
---> 17 import tensorflow as tf
     18 from tensorflow import keras
     19 from tensorflow.keras.models import Sequential, load_model

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
def gen_graph(history, title):
    plt.plot(history.history.get("categorical_accuracy", []))
    plt.plot(history.history.get("val_categorical_accuracy", []))
    plt.title("Accuracy " + title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()




## === cell 2
df_train = pd.read_csv("../input/dog-breed-identification/labels.csv")
df_test = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
jpg_train = "../input/dog-breed-identification/train/{}.jpg"
jpg_test = "../input/dog-breed-identification/test/{}.jpg"



## === cell 3
pass



## === cell 4
pass



## === cell 5
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)



## === cell 6
one_hot_labels = np.asarray(one_hot).astype(np.float32)



## === cell 7
im_resize = 64  # image size
num_class = 120  # number of classes



## === cell 8
n_train = len(df_train)
n_test = len(df_test)

X_all_train = np.empty((n_train, im_resize, im_resize, 3), dtype=np.float32)
Y_all_train = one_hot_labels  # already aligned with df_train order

train_ids = df_train["id"].values
for i, f in enumerate(tqdm(train_ids, total=n_train)):
    img = cv2.imread(jpg_train.format(f))
    if img is None:
        raise FileNotFoundError(f"Could not read train image: {jpg_train.format(f)}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (im_resize, im_resize), interpolation=cv2.INTER_AREA)
    X_all_train[i] = img_resized



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2582226754.py in <cell line: 0>()
      7 train_ids = df_train["id"].values
      8 for i, f in enumerate(tqdm(train_ids, total=n_train)):
----> 9     img = cv2.imread(jpg_train.format(f))
     10     if img is None:
     11         raise FileNotFoundError(f"Could not read train image: {jpg_train.format(f)}")

NameError: name 'cv2' is not defined

## === cell 9
X_test_raw = np.empty((n_test, im_resize, im_resize, 3), dtype=np.float32)

test_ids = df_test["id"].values
for i, f in enumerate(tqdm(test_ids, total=n_test)):
    img = cv2.imread(jpg_test.format(f))
    if img is None:
        raise FileNotFoundError(f"Could not read test image: {jpg_test.format(f)}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (im_resize, im_resize), interpolation=cv2.INTER_AREA)
    X_test_raw[i] = img_resized



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1195712106.py in <cell line: 0>()
      3 test_ids = df_test["id"].values
      4 for i, f in enumerate(tqdm(test_ids, total=n_test)):
----> 5     img = cv2.imread(jpg_test.format(f))
      6     if img is None:
      7         raise FileNotFoundError(f"Could not read test image: {jpg_test.format(f)}")

NameError: name 'cv2' is not defined

## === cell 10
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_all_train,
    Y_all_train,
    shuffle=True,
    test_size=0.1,
    random_state=42,
    stratify=np.argmax(Y_all_train, axis=1),
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3179697913.py in <cell line: 0>()
----> 1 X_train, X_valid, Y_train, Y_valid = train_test_split(
      2     X_all_train,
      3     Y_all_train,
      4     shuffle=True,
      5     test_size=0.1,

NameError: name 'train_test_split' is not defined

## === cell 11
del X_all_train, Y_all_train, df_train



## === cell 12
datagen = ImageDataGenerator(
    rotation_range=15, rescale=1.0 / 255.0, horizontal_flip=True, zca_whitening=True
)

zca_cache_path = "zca_cache_im64.npy"
if os.path.exists(zca_cache_path):
    cache = np.load(zca_cache_path, allow_pickle=True).item()
    datagen.zca_mean = cache["zca_mean"]
    datagen.zca_whitening_matrix = cache["zca_whitening_matrix"]
else:
    datagen.fit(X_train)
    cache = {
        "zca_mean": getattr(datagen, "zca_mean", None),
        "zca_whitening_matrix": getattr(datagen, "zca_whitening_matrix", None),
    }
    if cache["zca_mean"] is None or cache["zca_whitening_matrix"] is None:
        raise RuntimeError(
            "ZCA whitening attributes were not created after datagen.fit; "
            "cannot proceed with zca_whitening=True."
        )
    np.save(zca_cache_path, cache, allow_pickle=True)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1755624466.py in <cell line: 0>()
----> 1 datagen = ImageDataGenerator(
      2     rotation_range=15, rescale=1.0 / 255.0, horizontal_flip=True, zca_whitening=True
      3 )
      4 
      5 # Fix: Cache and restore the actual attributes used by ImageDataGenerator for ZCA whitening.

NameError: name 'ImageDataGenerator' is not defined

## === cell 13
pass



## === cell 14
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        input_shape=(im_resize, im_resize, 3),
        activation="relu",
    )
)
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(num_class, activation="softmax"))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1082979299.py in <cell line: 0>()
----> 1 model = Sequential()
      2 
      3 model.add(
      4     Conv2D(
      5         32,

NameError: name 'Sequential' is not defined

## === cell 15
print(model.summary())



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518542503.py in <cell line: 0>()
----> 1 print(model.summary())
      2 

NameError: name 'model' is not defined

## === cell 16
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[categorical_crossentropy, categorical_accuracy],
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4262319494.py in <cell line: 0>()
----> 1 model.compile(
      2     optimizer="Adam",
      3     loss="categorical_crossentropy",
      4     metrics=[categorical_crossentropy, categorical_accuracy],
      5 )

NameError: name 'model' is not defined

## === cell 17
batch_size = 256
train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)

X_valid_scaled = X_valid / 255.0



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4197659717.py in <cell line: 0>()
      1 batch_size = 256
----> 2 train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True)
      3 
      4 X_valid_scaled = X_valid / 255.0
      5 

NameError: name 'datagen' is not defined

## === cell 18
earlystop = EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    min_delta=0,
    patience=5,
    restore_best_weights=False,
)

checkpoint_path = "model_best.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1475517687.py in <cell line: 0>()
----> 1 earlystop = EarlyStopping(
      2     monitor="val_categorical_accuracy",
      3     mode="max",
      4     min_delta=0,
      5     patience=5,

NameError: name 'EarlyStopping' is not defined

## === cell 19
Epochs = 100
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

history_rmsprop = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_valid_scaled, Y_valid),
    verbose=1,
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1686968766.py in <cell line: 0>()
      1 Epochs = 100
----> 2 steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
      3 
      4 # Fix: Keras 3 TF trainer doesn't accept workers/use_multiprocessing/max_queue_size in fit().
      5 history_rmsprop = model.fit(

NameError: name 'X_train' is not defined

## === cell 20
gen_graph(history_rmsprop, "график точности")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2338312875.py in <cell line: 0>()
----> 1 gen_graph(history_rmsprop, "график точности")
      2 

NameError: name 'history_rmsprop' is not defined

## === cell 21
if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)
else:
    print(f"Warning: checkpoint {checkpoint_path} not found; using last-epoch model.")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3073658822.py in <cell line: 0>()
----> 1 if os.path.exists(checkpoint_path):
      2     model = load_model(checkpoint_path)
      3 else:
      4     print(f"Warning: checkpoint {checkpoint_path} not found; using last-epoch model.")
      5 

NameError: name 'checkpoint_path' is not defined

## === cell 22
X_test = X_test_raw / 255.0
del X_test_raw
preds = model.predict(X_test, verbose=1, batch_size=256)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/429519876.py in <cell line: 0>()
      1 X_test = X_test_raw / 255.0
      2 del X_test_raw
----> 3 preds = model.predict(X_test, verbose=1, batch_size=256)
      4 

NameError: name 'model' is not defined

## === cell 23
sub = pd.DataFrame(preds, columns=one_hot.columns.values)
sub.insert(0, "id", df_test["id"].values)

sample_cols = list(df_test.columns)
sub = sub.reindex(columns=sample_cols)

if sub.isnull().values.any():
    prob_cols = [c for c in sub.columns if c != "id"]
    sub[prob_cols] = sub[prob_cols].fillna(1.0 / len(prob_cols))

sub.to_csv("output_rmsprop_aug.csv", index=False)
print("Saved submission to output_rmsprop_aug.csv")
print(sub.shape)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3588133739.py in <cell line: 0>()
----> 1 sub = pd.DataFrame(preds, columns=one_hot.columns.values)
      2 sub.insert(0, "id", df_test["id"].values)
      3 
      4 # Ensure the submission has exactly the same columns/order as sample_submission.csv
      5 sample_cols = list(df_test.columns)

NameError: name 'preds' is not defined
