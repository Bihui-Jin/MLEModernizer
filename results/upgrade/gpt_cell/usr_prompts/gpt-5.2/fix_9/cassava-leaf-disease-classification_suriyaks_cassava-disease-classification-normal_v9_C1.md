# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import pandas as pd
train = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')


## === cell 1
import os
path = '/kaggle/input/cassava-leaf-disease-classification/'
for i in os.listdir(path):
    print(i)


## === cell 2
train


## === cell 3
train.info()


## === cell 4
train['label'].unique()


## === cell 5
from PIL import Image  
im = Image.open("/kaggle/input/cassava-leaf-disease-classification/train_images/999616605.jpg")  
print('opened')
im.size


## === cell 6
train_path = '/kaggle/input/cassava-leaf-disease-classification/train_images'
test_path = '/kaggle/input/cassava-leaf-disease-classification/test_images'


## === cell 7
import json

file = open('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json')
json_data = json.load(file)


## === cell 8
json_data


## === cell 9
for i in range(len(train)):
    if train["label"][i] == 0:
        train.replace(
            to_replace=train["label"][i],
            value="Cassava Bacterial Blight (CBB)",
            inplace=True,
        )
    elif train["label"][i] == 1:
        train.replace(
            to_replace=train["label"][i],
            value="Cassava Brown Streak Disease (CBSD)",
            inplace=True,
        )
    elif train["label"][i] == 2:
        train.replace(
            to_replace=train["label"][i],
            value="Cassava Green Mottle (CGM)",
            inplace=True,
        )
    elif train["label"][i] == 3:
        train.replace(
            to_replace=train["label"][i],
            value="Cassava Mosaic Disease (CMD)",
            inplace=True,
        )
    elif train["label"][i] == 4:
        train.replace(to_replace=train["label"][i], value="Healthy", inplace=True)


## === cell 10
train.head()


## === cell 11
import os

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-input", "protobuf==4.25.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

keras = tf.keras  # keep `keras` name available for any later references

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image


## === cell 12
size = 512
bat_size = 16
split = 0.31
epoch = 10


## === cell 13
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=2,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=[0.0, 0.2],
    shear_range=0.0,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip = True,
    vertical_flip = True,
    validation_split=split,
)


## === cell 14
train_generator = train_datagen.flow_from_dataframe(
    train,
    directory=train_path,
    x_col="image_id",
    y_col="label",
    weight_col=None,
    target_size=(size, size),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=bat_size,
    shuffle=True,
    seed=None,
    subset='training',
    interpolation="nearest",
    validate_filenames=True
)


## === cell 15
validation_generator = train_datagen.flow_from_dataframe(
    train,
    directory=train_path,
    x_col="image_id",
    y_col="label",
    weight_col=None,
    target_size=(size, size),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=bat_size,
    subset='validation',
    shuffle=True,
    seed=None,
    validate_filenames=True
)


## === cell 16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
)
import tensorflow as tf

model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(size, size, 3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(
    Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
        activity_regularizer=tf.keras.regularizers.l2(1e-5),
    )
)
model.add(
    Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
        activity_regularizer=tf.keras.regularizers.l2(1e-5),
    )
)
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


## === cell 17
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_acc",
    min_delta=0.80,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)


## === cell 18
'''model.fit_generator(train_generator,
                         epochs = epoch,
                         validation_data = validation_generator,
                         verbose=1,
                         callbacks = [callback])'''


## === cell 19
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB7


model = EfficientNetB7(include_top=False, weights="imagenet", input_shape=(512, 512, 3))

model.trainable = False

inputs = keras.Input(shape=(512, 512, 3))
x = model(inputs, training=False)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(5, activation='softmax')(x)
model = keras.Model(inputs, outputs)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)


## === cell 20
model.summary()


## === cell 21
'''model.fit(train_generator,
         epochs = epoch,
         validation_data = validation_generator,
         verbose=1,
         callbacks = [callback])'''


## === cell 22
!pip install --upgrade tensorflow_hub


## === cell 23
import tensorflow_hub as hub
import tensorflow as tf


## === cell 24
import os

os.environ["TFHUB_USE_TF_KERAS"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub

inputs = tf.keras.Input(shape=(512, 512, 3))
x = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4"
)(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
m = tf.keras.Model(inputs=inputs, outputs=outputs)

m.build([None, 512, 512, 3])  # Batch input shape.


## --- ERROR in cell 24, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1909502458.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     10[0m [0;34m[0m[0m
[1;32m     11[0m [0minputs[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mInput[0m[0;34m([0m[0mshape[0m[0;34m=[0m[0;34m([0m[0;36m512[0m[0;34m,[0m [0;36m512[0m[0;34m,[0m [0;36m3[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m x = hub.KerasLayer(
[0m[1;32m     13[0m     [0;34m"https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m )(inputs)

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m     68[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     69[0m             [0;31m# `tf.debugging.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     72[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py[0m in [0;36mcall[0;34m(self, inputs, training)[0m
[1;32m    248[0m         [0;31m# Behave like BatchNormalization. (Dropout is different, b/181839368.)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0mtraining[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m       result = smart_cond.smart_cond(training,
[0m[1;32m    251[0m                                      [0;32mlambda[0m[0;34m:[0m [0mf[0m[0;34m([0m[0mtraining[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m                                      lambda: f(training=False))

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py[0m in [0;36m<lambda>[0;34m()[0m
[1;32m    250[0m       result = smart_cond.smart_cond(training,
[1;32m    251[0m                                      [0;32mlambda[0m[0;34m:[0m [0mf[0m[0;34m([0m[0mtraining[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 252[0;31m                                      lambda: f(training=False))
[0m[1;32m    253[0m [0;34m[0m[0m
[1;32m    254[0m     [0;31m# Unwrap dicts returned by signatures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py[0m in [0;36mcanonicalize_to_monomorphic[0;34m(args, kwargs, default_values, capture_types, polymorphic_type)[0m
[1;32m    581[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    582[0m       parameters.append(
[0;32m--> 583[0;31m           _make_validated_mono_param(name, arg, poly_parameter.kind,
[0m[1;32m    584[0m                                      [0mtype_context[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    585[0m                                      poly_parameter.type_constraint))

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py[0m in [0;36m_make_validated_mono_param[0;34m(name, value, kind, type_context, poly_type)[0m
[1;32m    520[0m ) -> Parameter:
[1;32m    521[0m   [0;34m"""Generates and validates a parameter for Monomorphic FunctionType."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 522[0;31m   [0mmono_type[0m [0;34m=[0m [0mtrace_type[0m[0;34m.[0m[0mfrom_value[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mtype_context[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    523[0m [0;34m[0m[0m
[1;32m    524[0m   [0;32mif[0m [0mpoly_type[0m [0;32mand[0m [0;32mnot[0m [0mmono_type[0m[0;34m.[0m[0mis_subtype_of[0m[0;34m([0m[0mpoly_type[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py[0m in [0;36mfrom_value[0;34m(value, context)[0m
[1;32m    183[0m [0;34m[0m[0m
[1;32m    184[0m   [0;32mif[0m [0mutil[0m[0;34m.[0m[0mis_np_ndarray[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 185[0;31m     [0mndarray[0m [0;34m=[0m [0mvalue[0m[0;34m.[0m[0m__array__[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    186[0m     [0;32mreturn[0m [0mdefault_types[0m[0;34m.[0m[0mTENSOR[0m[0;34m([0m[0mndarray[0m[0;34m.[0m[0mshape[0m[0;34m,[0m [0mndarray[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    187[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py[0m in [0;36m__array__[0;34m(self)[0m
[1;32m    106[0m [0;34m[0m[0m
[1;32m    107[0m     [0;32mdef[0m [0m__array__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 108[0;31m         raise ValueError(
[0m[1;32m    109[0m             [0;34m"A KerasTensor is symbolic: it's a placeholder for a shape "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    110[0m             [0;34m"an a dtype. It doesn't have any actual numerical value. "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Exception encountered when calling layer 'keras_layer' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'keras_layer' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 512, 512, 3), dtype=float32, sparse=False, name=keras_tensor_1095>
  • training=None

## === cell 25
m.summary()
