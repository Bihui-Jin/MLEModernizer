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

0.96313

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import cv2
import albumentations as A

from albumentations.core.composition import Compose
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Add,
    Activation,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    concatenate,
    Dropout,
    Flatten,
)
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = "../train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
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
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)



## === cell 3
src = "../input/paddy-disease-classification/train_images/dead_heart/100008.jpg"
img = img_to_array(load_img(src), dtype="uint8")




## === cell 4
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_y:
                anchors_y.append(rv)
        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0
        return image
    else:
        return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    else:
        return image


def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
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
generator = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
    preprocessing_function=center_crop_and_random_augmentations_fn,
    class_mode="categorical",
)

train_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    shuffle=True,
    seed=42,
)

valid_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    shuffle=False,
    seed=42,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3517474638.py in <cell line: 0>()
----> 1 generator = ImageDataGenerator(
      2     rescale=1 / 255,
      3     rotation_range=5,
      4     width_shift_range=0.1,
      5     height_shift_range=0.1,

TypeError: ImageDataGenerator.__init__() got an unexpected keyword argument 'class_mode'

## === cell 6
to_gen_fit = []
files_to_fit = [
    "../input/paddy-disease-classification/train_images/bacterial_leaf_blight/100049.jpg",
    "../input/paddy-disease-classification/train_images/bacterial_leaf_streak/100042.jpg",
    "../input/paddy-disease-classification/train_images/bacterial_panicle_blight/100068.jpg",
    "../input/paddy-disease-classification/train_images/blast/100012.jpg",
    "../input/paddy-disease-classification/train_images/brown_spot/100022.jpg",
    "../input/paddy-disease-classification/train_images/dead_heart/100020.jpg",
    "../input/paddy-disease-classification/train_images/downy_mildew/100059.jpg",
    "../input/paddy-disease-classification/train_images/hispa/100139.jpg",
    "../input/paddy-disease-classification/train_images/normal/100111.jpg",
    "../input/paddy-disease-classification/train_images/tungro/100134.jpg",
]

for file in files_to_fit:
    to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/248058006.py in <cell line: 0>()
     14 
     15 for file in files_to_fit:
---> 16     to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))
     17 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

FileNotFoundError: [Errno 2] No such file or directory: '../input/paddy-disease-classification/train_images/blast/100012.jpg'

## === cell 7
generator.fit(to_gen_fit)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2490998269.py in <cell line: 0>()
----> 1 generator.fit(to_gen_fit)
      2 

NameError: name 'generator' is not defined

## === cell 8
print(len(train_datagen.next()[0]), len(valid_datagen.next()[0]))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3724952305.py in <cell line: 0>()
----> 1 print(len(train_datagen.next()[0]), len(valid_datagen.next()[0]))
      2 

NameError: name 'train_datagen' is not defined

## === cell 9
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(train_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
plt.show()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4013875886.py in <cell line: 0>()
      1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
----> 3 for i, arr in enumerate(train_datagen.next()[0]):
      4     img = array_to_img(arr)
      5     axes[i].imshow(img)

NameError: name 'train_datagen' is not defined

## === cell 10
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(valid_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2877757462.py in <cell line: 0>()
      1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
----> 3 for i, arr in enumerate(valid_datagen.next()[0]):
      4     img = array_to_img(arr)
      5     axes[i].imshow(img)

NameError: name 'valid_datagen' is not defined

## === cell 11
model = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
            trainable=True,
            input_shape=(input_size, input_size, 3),
        ),
        tf.keras.layers.Dense(classes, activation="softmax"),
    ]
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2921970655.py in <cell line: 0>()
----> 1 model = tf.keras.Sequential(
      2     [
      3         hub.KerasLayer(
      4             "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
      5             trainable=True,

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in __init__(self, layers, trainable, name)
     73         if layers:
     74             for layer in layers:
---> 75                 self.add(layer, rebuild=False)
     76             self._maybe_rebuild()
     77 

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in add(self, layer, rebuild)
     95                 layer = origin_layer
     96         if not isinstance(layer, Layer):
---> 97             raise ValueError(
     98                 "Only instances of `keras.Layer` can be "
     99                 f"added to a Sequential model. Received: {layer} "

ValueError: Only instances of `keras.Layer` can be added to a Sequential model. Received: <tensorflow_hub.keras_layer.KerasLayer object at 0x7f9df816cf90> (of type <class 'tensorflow_hub.keras_layer.KerasLayer'>)

## === cell 12
model.build([None, input_size, input_size, 3])
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3714262588.py in <cell line: 0>()
----> 1 model.build([None, input_size, input_size, 3])
      2 model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
      3 

NameError: name 'model' is not defined

## === cell 13
model.summary()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1903595429.py in <cell line: 0>()
----> 1 model.summary()
      2 

NameError: name 'model' is not defined

## === cell 14
history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3837220834.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_datagen,
      3     validation_data=valid_datagen,
      4     batch_size=batch_size,
      5     epochs=epochs,

NameError: name 'model' is not defined

## === cell 15
tta_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    shuffle=False,
    seed=42,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3993596999.py in <cell line: 0>()
----> 1 tta_datagen = generator.flow_from_directory(
      2     "../input/paddy-disease-classification/train_images/",
      3     target_size=(input_size, input_size),
      4     batch_size=batch_size,
      5     subset="validation",

NameError: name 'generator' is not defined

## === cell 16
eve_encodings = np.zeros((len(tta_datagen.filenames), classes))
for _ in range(5):
    enc = model.predict(tta_datagen, verbose=0)
    eve_encodings += enc
eve_encodings /= 5



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3263901612.py in <cell line: 0>()
----> 1 eve_encodings = np.zeros((len(tta_datagen.filenames), classes))
      2 for _ in range(5):
      3     enc = model.predict(tta_datagen, verbose=0)
      4     eve_encodings += enc
      5 eve_encodings /= 5

NameError: name 'tta_datagen' is not defined

## === cell 17
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_datagen.classes
print("Validation accuracy:", accuracy_score(true_classes, pred_classes))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2048880667.py in <cell line: 0>()
----> 1 pred_classes = np.argmax(eve_encodings, axis=1)
      2 true_classes = tta_datagen.classes
      3 print("Validation accuracy:", accuracy_score(true_classes, pred_classes))
      4 

NameError: name 'eve_encodings' is not defined

## === cell 18
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["accuracy"])),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_accuracy"])),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1781478178.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["accuracy"])),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 19
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3700018787.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
      4 )
      5 sns.lineplot(

NameError: name 'history' is not defined

## === cell 20
pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/820322879.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)
      2 

NameError: name 'history' is not defined

## === cell 21
model.save("model_effnet_b4.hdf5")
model.save_weights("model_effnet_b4_weights.hdf5")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2871281348.py in <cell line: 0>()
----> 1 model.save("model_effnet_b4.hdf5")
      2 model.save_weights("model_effnet_b4_weights.hdf5")
      3 

NameError: name 'model' is not defined

## === cell 22
test_loc = "../input/paddy-disease-classification/test_images"

test_generator = ImageDataGenerator(
    rescale=1 / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn,
)

test_generator.fit(to_gen_fit)



## === cell 23
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_loc,
    labels=None,
    image_size=(input_size, input_size),
    batch_size=batch_size,
    shuffle=False,
)


def normalize_batch(batch):
    batch = tf.cast(batch, tf.float32) / 255.0
    batch = (
        (batch - test_generator.mean) / test_generator.std
        if hasattr(test_generator, "mean")
        else batch
    )
    return batch


test_dataset = test_dataset.map(lambda x: normalize_batch(x))



## === cell 24
test_encodings = np.zeros((len(test_dataset.file_paths), classes))
for _ in range(5):
    enc = model.predict(test_dataset, verbose=0)
    test_encodings += enc
test_encodings /= 5



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1547451762.py in <cell line: 0>()
----> 1 test_encodings = np.zeros((len(test_dataset.file_paths), classes))
      2 for _ in range(5):
      3     enc = model.predict(test_dataset, verbose=0)
      4     test_encodings += enc
      5 test_encodings /= 5

AttributeError: '_MapDataset' object has no attribute 'file_paths'

## === cell 25
predict_max = np.argmax(test_encodings, axis=1)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2194500239.py in <cell line: 0>()
----> 1 predict_max = np.argmax(test_encodings, axis=1)
      2 

NameError: name 'test_encodings' is not defined

## === cell 26
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1018055797.py in <cell line: 0>()
----> 1 inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
      2 predictions = [inverse_map[k] for k in predict_max]
      3 

NameError: name 'train_datagen' is not defined

## === cell 27
files = [os.path.basename(p) for p in test_dataset.file_paths]
submission = pd.DataFrame({"image_id": files, "label": predictions})
submission.to_csv("model_submission_v17.csv", index=False)
print(submission.head())

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/732761671.py in <cell line: 0>()
----> 1 files = [os.path.basename(p) for p in test_dataset.file_paths]
      2 submission = pd.DataFrame({"image_id": files, "label": predictions})
      3 submission.to_csv("model_submission_v17.csv", index=False)
      4 print(submission.head())

AttributeError: '_MapDataset' object has no attribute 'file_paths'
