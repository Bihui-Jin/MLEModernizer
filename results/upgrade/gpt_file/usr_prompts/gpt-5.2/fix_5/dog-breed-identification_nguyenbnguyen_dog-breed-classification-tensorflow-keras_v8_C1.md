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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

1.5640095204611175

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "TensorFlow must be available in the runtime environment. "
        "Runtime pip installs were removed to meet the 600s timeout."
    ) from e




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/744991151.py in <cell line: 0>()
     10 try:
---> 11     import tensorflow as tf  # noqa: F401
     12 except Exception as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/744991151.py in <cell line: 0>()
     11     import tensorflow as tf  # noqa: F401
     12 except Exception as e:
---> 13     raise RuntimeError(
     14         "TensorFlow must be available in the runtime environment. "
     15         "Runtime pip installs were removed to meet the 600s timeout."

RuntimeError: TensorFlow must be available in the runtime environment. Runtime pip installs were removed to meet the 600s timeout.

## === cell 1
from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)

KAGGLE_DATA_DIR = Path("/kaggle/input/dog-breed-identification")

labels_df = pd.read_csv(KAGGLE_DATA_DIR / "labels.csv")
filenames = [
    str(KAGGLE_DATA_DIR / f"train/{filename}.jpg") for filename in labels_df["id"]
]




## === cell 2
labels = labels_df["breed"].to_numpy()
len(labels) == len(filenames)




## === cell 3
filenames[:5]




## === cell 4
len(filenames)




## === cell 5
import random
import numpy as np

random_state = 42
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2911420105.py in <cell line: 0>()
      5 random.seed(random_state)
      6 np.random.seed(random_state)
----> 7 tf.random.set_seed(random_state)
      8 
      9 try:

NameError: name 'tf' is not defined

## === cell 6
import matplotlib.pyplot as plt

try:
    from IPython.display import Image  # noqa: F401
except Exception:
    Image = None  # type: ignore




## === cell 7
unique_breeds = np.unique(labels)
unique_breeds[:10]




## === cell 8
labels_df.head()




## === cell 9
labels_df.describe()




## === cell 10
if False:
    labels_df["breed"].value_counts(ascending=True).plot.barh(figsize=(20, 30))
    plt.tight_layout()
    plt.show()




## === cell 11
if False and Image is not None:
    example_dog_breed_name = labels_df[
        labels_df["id"] == "0021f9ceb3235effd7fcde7f7538ed62"
    ]["breed"].values[0]
    print(f"{example_dog_breed_name}")
    example_dog_breed = Image(
        KAGGLE_DATA_DIR / "train/0021f9ceb3235effd7fcde7f7538ed62.jpg"
    )
    example_dog_breed




## === cell 12
if False:
    image = plt.imread(filenames[0])
    image.shape




## === cell 13
if False:
    image[:1]




## === cell 14
print("TF version:", tf.__version__)
if tf.config.list_physical_devices("GPU"):
    print("GPU enabled")
else:
    print("GPU is not available, switch to CPU")




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2264476527.py in <cell line: 0>()
----> 1 print("TF version:", tf.__version__)
      2 if tf.config.list_physical_devices("GPU"):
      3     print("GPU enabled")
      4 else:
      5     print("GPU is not available, switch to CPU")

NameError: name 'tf' is not defined

## === cell 15
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

IMG_WIDTH = 224
IMG_HEIGHT = IMG_WIDTH
IMG_CHANNELS = 3
BATCH_SIZE = 32


@tf.autograph.experimental.do_not_convert
def process_image(image_path: str):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size=[IMG_WIDTH, IMG_HEIGHT])
    return image


@tf.autograph.experimental.do_not_convert
def get_image_label(image_path: str, label):
    image = process_image(image_path)
    return image, label


def create_data_batches(
    X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    options = tf.data.Options()
    options.experimental_deterministic = True

    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        data = data.with_options(options)
        data = data.map(
            process_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        ).cache()
        data_batch = data.batch(batch_size, drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
        return data_batch

    if valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.with_options(options)
        data = data.map(
            get_image_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        ).cache()
        data_batch = data.batch(batch_size, drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
        return data_batch

    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    data = data.with_options(options)

    shuffle_buf = min(len(X), 2048)
    data = data.map(
        get_image_label, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).cache()

    data_batch = (
        data.shuffle(
            buffer_size=shuffle_buf, seed=random_state, reshuffle_each_iteration=True
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    return data_batch


lb = LabelBinarizer()
encoded_labels = lb.fit_transform(labels)
print(f"{encoded_labels[:1] = }")

strat = np.argmax(encoded_labels, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    filenames, encoded_labels, test_size=0.1, random_state=random_state, stratify=strat
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=7, stratify=np.argmax(y_train, axis=1)
)

train_data = create_data_batches(X_train, y_train)
valid_data = create_data_batches(X_valid, y_valid, valid_data=True)
test_data_ = create_data_batches(X_test, y_test, valid_data=True)

train_steps = int(np.ceil(len(X_train) / BATCH_SIZE))
valid_steps = int(np.ceil(len(X_valid) / BATCH_SIZE))
test_steps = int(np.ceil(len(X_test) / BATCH_SIZE))
train_data_rep = train_data.repeat()
valid_data_rep = valid_data.repeat()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3524473693.py in <cell line: 0>()
      8 
      9 
---> 10 @tf.autograph.experimental.do_not_convert
     11 def process_image(image_path: str):
     12     image = tf.io.read_file(image_path)

NameError: name 'tf' is not defined

## === cell 16
train_data.element_spec, valid_data.element_spec




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3988449245.py in <cell line: 0>()
----> 1 train_data.element_spec, valid_data.element_spec
      2 
      3 

NameError: name 'train_data' is not defined

## === cell 17
def show_25_images(images, labels):
    plt.figure(figsize=(15, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[np.argmax(labels[i])])
        plt.axis("off")
    plt.tight_layout()




## === cell 18
if False:
    train_images, train_labels = next(train_data.as_numpy_iterator())
    show_25_images(train_images, train_labels)




## === cell 19
if False:
    valid_images, valid_labels = next(valid_data.as_numpy_iterator())
    show_25_images(valid_images, valid_labels)




## === cell 20
if False:
    test_images, test_labels = next(test_data_.as_numpy_iterator())
    show_25_images(test_images, test_labels)




## === cell 21
import datetime

try:
    import tensorflow_hub as hub  # noqa: F401
except Exception:
    hub = None  # type: ignore

from tensorflow.keras import layers


def save_model(model, file_name, path_folder="./", include_datetime=True):
    suffix = None
    if include_datetime:
        suffix = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    if suffix:
        full_file_name = f"{suffix}-{file_name}.keras"
    else:
        full_file_name = f"{file_name}.keras"

    model_path = Path(path_folder) / full_file_name
    print(f"Saving model to: {model_path}...")
    model.save(model_path)
    return model_path


def load_model(model_path):
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
    num_classes=len(unique_breeds),
    base_model_trainable=False,
    data_augmentation=None,
):
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="max"
    )
    base_model.trainable = base_model_trainable

    inputs = layers.Input(input_shape)
    if data_augmentation:
        X = data_augmentation(inputs)
    else:
        X = inputs
    X = base_model(X, training=base_model_trainable)
    outputs = layers.Dense(num_classes)(X)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_model(model=None, learning_rate=1e-3):
    if model is None:
        model = MobileNetV2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def data_augmenter():
    data_augmentation = tf.keras.models.Sequential()
    data_augmentation.add(layers.RandomFlip("horizontal"))
    data_augmentation.add(layers.RandomZoom(0.5))
    return data_augmentation


lr = 3e-1
model = create_model(MobileNetV2(data_augmentation=data_augmenter()), learning_rate=lr)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=3, min_lr=1e-9
)
val_acc_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", restore_best_weights=True, patience=5, start_from_epoch=3
)
model.summary(show_trainable=True)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2065734447.py in <cell line: 0>()
      6     hub = None  # type: ignore
      7 
----> 8 from tensorflow.keras import layers
      9 
     10 

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

## === cell 22
train_images, train_labels = next(iter(train_data.take(1)))
_tmp_logits = model(train_images[:2], training=False)
_tmp_probs = tf.nn.softmax(_tmp_logits, axis=1)
print("Softmax sum (first sample):", float(tf.reduce_sum(_tmp_probs[0])))




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3325428225.py in <cell line: 0>()
----> 1 train_images, train_labels = next(iter(train_data.take(1)))
      2 _tmp_logits = model(train_images[:2], training=False)
      3 _tmp_probs = tf.nn.softmax(_tmp_logits, axis=1)
      4 print("Softmax sum (first sample):", float(tf.reduce_sum(_tmp_probs[0])))
      5 

NameError: name 'train_data' is not defined

## === cell 23
epochs = 20

history = model.fit(
    train_data_rep,
    validation_data=valid_data_rep,
    callbacks=[reduce_lr, val_acc_stopping],
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/329595099.py in <cell line: 0>()
      1 epochs = 20
      2 
----> 3 history = model.fit(
      4     train_data_rep,
      5     validation_data=valid_data_rep,

NameError: name 'model' is not defined

## === cell 24
print(
    "Last epoch:", history.epoch[-1], "Last val_loss:", history.history["val_loss"][-1]
)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1147738378.py in <cell line: 0>()
      1 print(
----> 2     "Last epoch:", history.epoch[-1], "Last val_loss:", history.history["val_loss"][-1]
      3 )
      4 
      5 

NameError: name 'history' is not defined

## === cell 25
if False:
    acc = [0.0] + history.history["accuracy"]
    val_acc = [0.0] + history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()




## === cell 26
base_model = model.layers[-2]
base_model.trainable = True
model.summary(show_trainable=True)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2166148241.py in <cell line: 0>()
----> 1 base_model = model.layers[-2]
      2 base_model.trainable = True
      3 model.summary(show_trainable=True)
      4 
      5 

NameError: name 'model' is not defined

## === cell 27
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr * 1e-3)
metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy", dtype=np.float32)]
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4212529086.py in <cell line: 0>()
----> 1 loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
      2 optimizer = tf.keras.optimizers.Adam(learning_rate=lr * 1e-3)
      3 metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy", dtype=np.float32)]
      4 model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
      5 

NameError: name 'tf' is not defined

## === cell 28
fine_tune_epochs = 50
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_data_rep,
    validation_data=valid_data_rep,
    epochs=total_epochs,
    callbacks=[reduce_lr, val_acc_stopping],
    initial_epoch=history.epoch[-1],
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1018453564.py in <cell line: 0>()
      2 total_epochs = epochs + fine_tune_epochs
      3 
----> 4 history_fine = model.fit(
      5     train_data_rep,
      6     validation_data=valid_data_rep,

NameError: name 'model' is not defined

## === cell 29
if False:
    acc = [0.0] + history.history["accuracy"]
    val_acc = [0.0] + history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    acc += history_fine.history["accuracy"]
    val_acc += history_fine.history["val_accuracy"]
    loss += history_fine.history["loss"]
    val_loss += history_fine.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.ylim([0, 1])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.ylim([0, 1.0])
    plt.plot([epochs - 1, epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()




## === cell 30
def get_pred_label(prediction_probabilities):
    return unique_breeds[np.argmax(prediction_probabilities)]


preds = model.predict(test_data_, verbose=0)

index = 0
print(f"Max value (probability of prediction): {np.max(tf.nn.softmax(preds[index]))}")
print("Sum:", np.sum(tf.nn.softmax(preds[index])))
print("Max index:", np.argmax(preds[index]))
print("Predicted label:", unique_breeds[np.argmax(preds[index])])
print("Actual label:", unique_breeds[np.argmax(y_test[index])])

pred_label = get_pred_label(preds[7])
print(f"{pred_label = }")




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1115336245.py in <cell line: 0>()
      3 
      4 
----> 5 preds = model.predict(test_data_, verbose=0)
      6 
      7 index = 0

NameError: name 'model' is not defined

## === cell 31
def unbatchify(data):
    images = []
    labels_out = []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels_out.append(unique_breeds[np.argmax(label)])
    return images, labels_out


if False:
    test_images, test_labels = unbatchify(test_data_)
    test_images[0], test_labels[0]




## === cell 32
def plot_pred(prediction_probabilities, labels, images, n=1):
    pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
    pred_label = get_pred_label(pred_prob)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    color = "green" if pred_label == true_label else "red"
    plt.title(
        "{} {:2.0f}% {}".format(pred_label, np.max(pred_prob) * 100, true_label),
        color=color,
    )


if False:
    plot_pred(
        prediction_probabilities=tf.nn.softmax(preds, axis=1),
        labels=test_labels,
        images=test_images,
        n=10,
    )




## === cell 33
def plot_pred_conf(prediction_probabilities, labels, n=1):
    pred_prob, true_label = prediction_probabilities[n], labels[n]
    pred_label = get_pred_label(pred_prob)

    top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]

    top_10_plot = plt.barh(
        np.arange(len(top_10_pred_labels)), top_10_pred_values, color="grey"
    )
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(len(top_10_pred_labels)), labels=top_10_pred_labels)

    if np.isin(true_label, top_10_pred_labels):
        top_10_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")


if False:
    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
    plot_pred_conf(
        prediction_probabilities=prediction_probabilities, labels=test_labels, n=42
    )




## === cell 34
if False:
    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
    i_multiplier = 30
    num_rows = 3
    num_cols = 2
    num_images = num_rows * num_cols
    plt.figure(figsize=(10 * num_cols, 5 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_pred(
            prediction_probabilities=prediction_probabilities,
            labels=test_labels,
            images=test_images,
            n=i + i_multiplier,
        )
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_pred_conf(
            prediction_probabilities=prediction_probabilities,
            labels=test_labels,
            n=i + i_multiplier,
        )
    plt.tight_layout(h_pad=1.0)
    plt.show()




## === cell 35
if False:
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    prediction_probabilities = tf.nn.softmax(preds, axis=1).numpy()
    test_images, test_labels = unbatchify(test_data_)
    cf_matrix = confusion_matrix(
        test_labels, [get_pred_label(pred) for pred in prediction_probabilities]
    )

    plt.figure(figsize=(30, 15))
    ax = sns.heatmap(
        cf_matrix,
        cmap="Reds",
        linewidths=1,
        xticklabels=lb.classes_,
        yticklabels=lb.classes_,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()




## === cell 36
_ = save_model(model, file_name="mobilenetv2-Adam", include_datetime=False)




## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2674122661.py in <cell line: 0>()
----> 1 _ = save_model(model, file_name="mobilenetv2-Adam", include_datetime=False)
      2 
      3 

NameError: name 'save_model' is not defined

## === cell 37
sample_sub_path = KAGGLE_DATA_DIR / "sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
breed_columns = [c for c in sample_sub.columns if c != "id"]

submitted_data_dir = KAGGLE_DATA_DIR / "test"
submitted_img_paths = sorted([str(p) for p in submitted_data_dir.glob("*.jpg")])
submitted_ids = [Path(p).stem for p in submitted_img_paths]

model_class_order = list(unique_breeds)
print(
    "Classes in model:",
    len(model_class_order),
    "Classes in sample:",
    len(breed_columns),
)

submitted_data = create_data_batches(submitted_img_paths, test_data=True)




## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2205112543.py in <cell line: 0>()
     15 )
     16 
---> 17 submitted_data = create_data_batches(submitted_img_paths, test_data=True)
     18 
     19 

NameError: name 'create_data_batches' is not defined

## === cell 38
submit_logits = model.predict(submitted_data, verbose=1)
submit_probs = tf.nn.softmax(submit_logits, axis=1).numpy()

if model_class_order != breed_columns:
    model_idx = {cls: i for i, cls in enumerate(model_class_order)}
    reorder_idx = [model_idx[c] for c in breed_columns]
    submit_probs = submit_probs[:, reorder_idx]

out_df = pd.DataFrame(submit_probs, columns=breed_columns)
out_df.insert(0, "id", submitted_ids)
out_df = out_df.set_index("id").reindex(sample_sub["id"]).reset_index()
out_df.head()




## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3711665739.py in <cell line: 0>()
----> 1 submit_logits = model.predict(submitted_data, verbose=1)
      2 submit_probs = tf.nn.softmax(submit_logits, axis=1).numpy()
      3 
      4 if model_class_order != breed_columns:
      5     model_idx = {cls: i for i, cls in enumerate(model_class_order)}

NameError: name 'model' is not defined

## === cell 39
out_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out_df.shape)
print("Columns:", out_df.columns[:5].tolist(), "...", out_df.columns[-3:].tolist())
print("First id:", out_df.loc[0, "id"])

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1205230309.py in <cell line: 0>()
----> 1 out_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", out_df.shape)
      3 print("Columns:", out_df.columns[:5].tolist(), "...", out_df.columns[-3:].tolist())
      4 print("First id:", out_df.loc[0, "id"])

NameError: name 'out_df' is not defined
