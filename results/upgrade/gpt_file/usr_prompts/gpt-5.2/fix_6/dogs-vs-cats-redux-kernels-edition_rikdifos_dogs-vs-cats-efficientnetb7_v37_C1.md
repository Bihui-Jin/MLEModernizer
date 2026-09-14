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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

1.02519

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.75071) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting, which is causing the `MessageFactory` error in this environment. I also fix the dataset extraction/path logic: your zip extraction puts images directly under `./data/` (not `./data/train` / `./data/test`), so the directory finder must look there to correctly populate `train_images` and `test_images`. To prevent downstream `NameError`s, I keep the original core training/prediction logic but ensure variables are defined by making the earlier cells succeed and by safely filtering for train/test filenames. Finally, I ensure a valid `submission.csv` with columns `id,label` is always written.'
- What this solution (achieved 0.75066) has done: 'The crash happens before any training because TensorFlow 2.18 with protobuf 6.x can hit a known `MessageFactory.GetPrototype` incompatibility at import time in some Kaggle images; the minimal reliable fix is to pin protobuf to the Python implementation *before* importing TensorFlow. Your current code tries to unset the env var, which keeps the crash. I set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early (and keep everything else the same) so the notebook runs end-to-end and still writes `submission.csv` with `id,label`. Since your current score (0.75071) is already better than the target (1.02519) for a “lower is better” metric, I won’t make any modeling changes that would intentionally worsen/improve score; this is a runtime/stability fix only.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import cv2, re, random, time, zipfile, gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam

SEED = 558
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3637689103.py in <cell line: 0>()
     21 from sklearn.model_selection import train_test_split
     22 
---> 23 import tensorflow as tf
     24 from tensorflow import keras
     25 from tensorflow.keras import layers, models

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
PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
train_image_path = os.path.join(PATH, "train.zip")
test_image_path = os.path.join(PATH, "test.zip")

os.makedirs("./data", exist_ok=True)

with zipfile.ZipFile(train_image_path, "r") as z:
    z.extractall("./data")

with zipfile.ZipFile(test_image_path, "r") as z:
    z.extractall("./data")



## === cell 2
start = time.time()


def _find_image_dir(
    root_dir, expected_prefixes=("cat.", "dog."), ext=(".jpg", ".jpeg", ".png")
):
    if not os.path.exists(root_dir):
        return None
    try:
        files = os.listdir(root_dir)
    except Exception:
        files = []
    if any(
        f.lower().endswith(ext) and any(f.startswith(p) for p in expected_prefixes)
        for f in files
    ):
        return root_dir

    best = None
    best_count = 0
    for cur, _, files in os.walk(root_dir):
        cnt = sum(
            1
            for f in files
            if f.lower().endswith(ext)
            and any(f.startswith(p) for p in expected_prefixes)
        )
        if cnt > best_count:
            best_count = cnt
            best = cur
    return best


def _find_test_dir(root_dir, ext=(".jpg", ".jpeg", ".png")):
    if not os.path.exists(root_dir):
        return None
    try:
        files = os.listdir(root_dir)
    except Exception:
        files = []
    if any(f.lower().endswith(ext) and os.path.splitext(f)[0].isdigit() for f in files):
        return root_dir

    best = None
    best_count = 0
    for cur, _, files in os.walk(root_dir):
        cnt = sum(
            1
            for f in files
            if f.lower().endswith(ext) and os.path.splitext(f)[0].isdigit()
        )
        if cnt > best_count:
            best_count = cnt
            best = cur
    return best


TRAIN_DIR = _find_image_dir("./data", expected_prefixes=("cat.", "dog."))
TEST_DIR = _find_test_dir("./data")

if TRAIN_DIR is None or not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(
        f"Could not locate extracted train image directory under ./data. "
        f"Top-level contents: {os.listdir('./data') if os.path.isdir('./data') else 'MISSING'}"
    )
if TEST_DIR is None or not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Could not locate extracted test image directory under ./data. "
        f"Top-level contents: {os.listdir('./data') if os.path.isdir('./data') else 'MISSING'}"
    )

all_files = os.listdir("./data")
train_images = [
    os.path.join("./data", f)
    for f in all_files
    if f.startswith(("cat.", "dog.")) and f.lower().endswith(".jpg")
]
test_images = [
    os.path.join("./data", f)
    for f in all_files
    if os.path.splitext(f)[0].isdigit() and f.lower().endswith(".jpg")
]


def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]


train_images = sorted(train_images, key=natural_keys)
test_images = sorted(test_images, key=natural_keys)

print("Resolved TRAIN_DIR:", TRAIN_DIR)
print("Resolved TEST_DIR :", TEST_DIR)
print("N train images:", len(train_images))
print("N test images :", len(test_images))

if len(train_images) == 0 or len(test_images) == 0:
    raise RuntimeError(
        "Failed to build non-empty train/test file lists after extraction."
    )




## === cell 3
def txt_dig(text):
    """输入字符串，如果是数字则输出数字，如果不是则输出原本字符串"""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """输入字符串，将数字与文字分隔开，将数字串转化为int"""
    return [txt_dig(c) for c in re.split(r"(\d+)", text)]




## === cell 4
if len(train_images) >= 25000:
    train_images = train_images[0:7500] + train_images[17500:25000]  # 抽样
else:
    train_images = train_images[: min(len(train_images), 15000)]

random.seed(558)
random.shuffle(train_images)



## === cell 5
IMG_WIDTH = 128
IMG_HEIGHT = 128

x = []
valid_train_images = []
for img in train_images:
    arr = cv2.imread(img)
    if arr is None:
        continue
    x.append(cv2.resize(arr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))
    valid_train_images.append(img)

test = []
valid_test_images = []
for img in test_images:
    arr = cv2.imread(img)
    if arr is None:
        continue
    test.append(cv2.resize(arr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))
    valid_test_images.append(img)

x = np.array(x)
test = np.array(test)

print("The shape of train data is {}".format(x.shape))
print("The shape of test data is {}".format(test.shape))

plt.rcParams["figure.facecolor"] = "white"

y = []
for i in valid_train_images:
    base = os.path.basename(i)
    if "dog" in base:
        y.append(1)
    elif "cat" in base:
        y.append(0)
y = np.array(y)

if len(y) != len(x):
    raise RuntimeError(f"Label/feature mismatch: len(y)={len(y)} vs len(x)={len(x)}")

print("y shape:", y.shape, "positives:", int(y.sum()), "negatives:", int((1 - y).sum()))
sns.countplot(x=y)



## === cell 6
random.seed(558)
plt.subplots(facecolor="white", figsize=(10, 20))

sample = random.choice(valid_train_images)
image = load_img(sample)
plt.subplot(131)
plt.imshow(image)
plt.axis("off")

sample = random.choice(valid_train_images)
image = load_img(sample)
plt.subplot(132)
plt.imshow(image)
plt.axis("off")

sample = random.choice(valid_train_images)
image = load_img(sample)
plt.subplot(133)
plt.imshow(image)
plt.axis("off")

plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2960047060.py in <cell line: 0>()
      3 
      4 sample = random.choice(valid_train_images)
----> 5 image = load_img(sample)
      6 plt.subplot(131)
      7 plt.imshow(image)

NameError: name 'load_img' is not defined

## === cell 7
plt.subplots(facecolor="white", figsize=(10, 20))
idxs = [1024, 546, 742]
idxs = [min(i, len(x) - 1) for i in idxs if len(x) > 0]

for j, idx in enumerate(idxs[:3], start=1):
    plt.subplot(1, 3, j)
    plt.imshow(cv2.cvtColor(x[idx], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.show()



## === cell 8
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=2020, stratify=y
)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 9
model = models.Sequential()

efnModel = tf.keras.applications.EfficientNetB7(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)
model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation="sigmoid"))

opt1 = RMSprop(learning_rate=1e-5, decay=1e-6)
opt2 = Adam(learning_rate=0.006)

model.compile(loss="binary_crossentropy", optimizer=opt1, metrics=["accuracy"])

model.summary()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2539925000.py in <cell line: 0>()
----> 1 model = models.Sequential()
      2 
      3 efnModel = tf.keras.applications.EfficientNetB7(
      4     weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
      5 )

NameError: name 'models' is not defined

## === cell 10
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3604397579.py in <cell line: 0>()
----> 1 datagen = ImageDataGenerator(
      2     rescale=1.0 / 255,
      3     rotation_range=40,
      4     width_shift_range=0.2,
      5     height_shift_range=0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 11
def plot_gened(train_images, seed=320):
    """plot pictures after processing"""
    df = pd.DataFrame({"filename": train_images})
    np.random.seed(seed)
    vis_df = df.sample(n=1).reset_index(drop=True)
    vis_df["category"] = "0"
    vis_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    vis_gen0 = vis_gen.flow_from_dataframe(
        vis_df,
        x_col="filename",
        y_col="category",
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=16,
        class_mode="raw",
        shuffle=False,
    )
    plt.rcParams["figure.facecolor"] = "white"
    plt.figure(figsize=(8, 8))
    for i in range(0, 9):
        plt.subplot(3, 3, i + 1)
        for X_batch, Y_batch in vis_gen0:
            image = X_batch[0]
            plt.imshow(image)
            plt.axis("off")
            break
    plt.tight_layout()
    plt.show()


plot_gened(valid_train_images)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/338527466.py in <cell line: 0>()
     38 
     39 
---> 40 plot_gened(valid_train_images)
     41 

/tmp/ipykernel_11/338527466.py in plot_gened(train_images, seed)
      5     vis_df = df.sample(n=1).reset_index(drop=True)
      6     vis_df["category"] = "0"
----> 7     vis_gen = ImageDataGenerator(
      8         rescale=1.0 / 255,
      9         rotation_range=40,

NameError: name 'ImageDataGenerator' is not defined

## === cell 12
BATCH_SIZE = 16
datagen_flow = datagen.flow(
    x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
val_datagen_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

earlystop1 = EarlyStopping(patience=5, restore_best_weights=True)
earlystop2 = ReduceLROnPlateau(
    monitor="val_accuracy", min_lr=0.001, patience=5, mode="max", verbose=1
)

history = model.fit(
    datagen_flow,
    steps_per_epoch=45,
    epochs=20,
    validation_data=val_datagen_flow,
    callbacks=[earlystop1, earlystop2],
    validation_steps=25,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1781410338.py in <cell line: 0>()
      1 BATCH_SIZE = 16
----> 2 datagen_flow = datagen.flow(
      3     x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
      4 )
      5 val_datagen_flow = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

NameError: name 'datagen' is not defined

## === cell 13
plt.rcParams["figure.facecolor"] = "white"
model_loss = pd.DataFrame(history.history)
print(model_loss.head())
model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0, 1])
plt.show()
model_loss[["loss", "val_loss"]].plot()
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3011053779.py in <cell line: 0>()
      1 plt.rcParams["figure.facecolor"] = "white"
----> 2 model_loss = pd.DataFrame(history.history)
      3 print(model_loss.head())
      4 model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0, 1])
      5 plt.show()

NameError: name 'history' is not defined

## === cell 14
x_val_scaled = x_val.astype("float32") / 255.0
val_preds = model.predict(x_val_scaled, verbose=0)
val_preds_class = np.where(val_preds.ravel() > 0.5, 1, 0)

print("Out of Fold Accuracy is {:.5}".format(accuracy_score(y_val, val_preds_class)))
print("Out of Fold log loss is {:.5}".format(log_loss(y_val, val_preds.ravel())))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2930148156.py in <cell line: 0>()
      1 x_val_scaled = x_val.astype("float32") / 255.0
----> 2 val_preds = model.predict(x_val_scaled, verbose=0)
      3 val_preds_class = np.where(val_preds.ravel() > 0.5, 1, 0)
      4 
      5 print("Out of Fold Accuracy is {:.5}".format(accuracy_score(y_val, val_preds_class)))

NameError: name 'model' is not defined

## === cell 15
test_scaled = test.astype("float32") / 255.0
test_pred = model.predict(test_scaled, verbose=0)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/982464627.py in <cell line: 0>()
      1 test_scaled = test.astype("float32") / 255.0
----> 2 test_pred = model.predict(test_scaled, verbose=0)
      3 

NameError: name 'model' is not defined

## === cell 16
if len(valid_test_images) != len(test_pred):
    raise RuntimeError(
        f"Mismatch between valid test images ({len(valid_test_images)}) and predictions ({len(test_pred)})."
    )

test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in valid_test_images]

test_pred_safe = np.clip(test_pred.ravel().astype("float64"), 1e-7, 1.0 - 1e-7)

submission = pd.DataFrame({"id": test_ids, "label": test_pred_safe})
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)

print("This program costs {:.2f} seconds".format(time.time() - start))
print(submission.head())
print(submission.tail())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3762399266.py in <cell line: 0>()
----> 1 if len(valid_test_images) != len(test_pred):
      2     raise RuntimeError(
      3         f"Mismatch between valid test images ({len(valid_test_images)}) and predictions ({len(test_pred)})."
      4     )
      5 

NameError: name 'test_pred' is not defined

## === cell 17
import shutil

shutil.rmtree("/kaggle/working/data/", ignore_errors=True)
gc.collect()
