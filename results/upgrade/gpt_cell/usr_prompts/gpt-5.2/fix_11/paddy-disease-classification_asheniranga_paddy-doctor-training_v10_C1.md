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

3.10

# 2. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version
    import google.protobuf as _pb

    if Version(_pb.__version__) >= Version("5.0.0"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
        importlib.reload(_pb)
except Exception:
    pass

import random
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import tensorflow_addons as tfa
except Exception:
    tfa = None

import tensorflow_hub as hub
import seaborn as sns
import cv2
import albumentations as A

from albumentations.core.composition import Compose
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add, Activation
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img


## === cell 1
train_meta_data = '../train.csv'
train_data_dir = '../input/paddy-disease-classification/train_images'
epochs = 25
lr = 1e-4
valid_split = 0.2
input_size = 128
batch_size = 16
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy


## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(patience=20,
                                              monitor='val_loss',
                                              restore_best_weights=True,
                                              verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5,
                                                 monitor='val_loss',
                                                 factor=0.5,
                                                 verbose=1)


## === cell 3
src = '../input/paddy-disease-classification/train_images/dead_heart/100008.jpg'
img = img_to_array(load_img(src), dtype='uint8')


## === cell 4
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])

            if (rv not in anchors_x):
                anchors_x.append(rv)

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])

            if (rv not in anchors_y):
                anchors_y.append(rv)

        for x, y in zip(anchors_x, anchors_y):
            image[x:x+patch_size, y:y+patch_size, :] = 0

        return image
    
    else:
        return image

def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7,7), 0)
    
    else:
        return image

def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0,1])
        
        if ax == 0:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            
            return np.row_stack(slices)
        
            
        else:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            
            return np.column_stack(slices)
        
    else:
        return image

def center_crop_and_random_augmentations_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 8, 16)
    image = random_displacment(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1).numpy()
    
    return image

def test_time_augmentation_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    
    return image


## === cell 5
generator = ImageDataGenerator(rescale=1 / 255,
                               rotation_range=5,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               featurewise_center=True,
                               featurewise_std_normalization=True,
                               horizontal_flip=True,
                               vertical_flip=True,
                               validation_split=valid_split,
                               preprocessing_function=center_crop_and_random_augmentations_fn
                              )

train_datagen = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                              target_size=(input_size, input_size),
                                              batch_size=batch_size,
                                              subset='training')

valid_datagen = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                              target_size=(input_size, input_size),
                                              batch_size=batch_size,
                                              subset='validation')


## === cell 6
to_gen_fit = []

train_images_root = "../input/paddy-disease-classification/train_images"

class_dirs = [
    d
    for d in sorted(os.listdir(train_images_root))
    if os.path.isdir(os.path.join(train_images_root, d))
]

files_to_fit = []
for cls in class_dirs:
    cls_dir = os.path.join(train_images_root, cls)
    jpgs = sorted(
        f
        for f in os.listdir(cls_dir)
        if os.path.isfile(os.path.join(cls_dir, f)) and f.lower().endswith(".jpg")
    )
    if jpgs:
        files_to_fit.append(os.path.join(cls_dir, jpgs[0]))

for file in files_to_fit:
    to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))


## === cell 7
generator.fit(to_gen_fit)


## === cell 8
len(next(train_datagen)[0]), len(next(valid_datagen)[0])


## === cell 9
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(next(train_datagen)[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)

plt.show()


## === cell 10
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(next(valid_datagen)[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)

plt.show()


## === cell 11
inputs = tf.keras.Input(shape=(input_size, input_size, 3))
x = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
    trainable=True,
)(inputs)
outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/958760344.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m [0;31m# preserving identical model architecture/semantics.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0minputs[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mInput[0m[0;34m([0m[0mshape[0m[0;34m=[0m[0;34m([0m[0minput_size[0m[0;34m,[0m [0minput_size[0m[0;34m,[0m [0;36m3[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m x = hub.KerasLayer(
[0m[1;32m      7[0m     [0;34m"https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m     [0mtrainable[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

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
  • inputs=<KerasTensor shape=(None, 128, 128, 3), dtype=float32, sparse=False, name=keras_tensor>
  • training=None

## === cell 12


model.build([None, input_size, input_size, 3])

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])
