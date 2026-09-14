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
import tensorflow_addons as tfa
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
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/409132772.py in <cell line: 0>()
      1 src = '../input/paddy-disease-classification/train_images/dead_heart/100008.jpg'
----> 2 img = img_to_array(load_img(src), dtype='uint8')

NameError: name 'img_to_array' is not defined

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


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2856334021.py in <cell line: 0>()
----> 1 generator = ImageDataGenerator(rescale=1 / 255,
      2                                rotation_range=5,
      3                                width_shift_range=0.1,
      4                                height_shift_range=0.1,
      5                                featurewise_center=True,

NameError: name 'ImageDataGenerator' is not defined

## === cell 6
to_gen_fit = []
files_to_fit = ['../input/paddy-disease-classification/train_images/bacterial_leaf_blight/100049.jpg',
                '../input/paddy-disease-classification/train_images/bacterial_leaf_streak/100042.jpg',
                '../input/paddy-disease-classification/train_images/bacterial_panicle_blight/100068.jpg',
                '../input/paddy-disease-classification/train_images/blast/100012.jpg',
                '../input/paddy-disease-classification/train_images/brown_spot/100022.jpg',
                '../input/paddy-disease-classification/train_images/dead_heart/100020.jpg',
                '../input/paddy-disease-classification/train_images/downy_mildew/100059.jpg',
                '../input/paddy-disease-classification/train_images/hispa/100139.jpg',
                '../input/paddy-disease-classification/train_images/normal/100111.jpg',
                '../input/paddy-disease-classification/train_images/tungro/100134.jpg',
               ]

for file in files_to_fit:
    to_gen_fit.append(img_to_array(load_img(file), dtype='uint8'))


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2567020947.py in <cell line: 0>()
     13 
     14 for file in files_to_fit:
---> 15     to_gen_fit.append(img_to_array(load_img(file), dtype='uint8'))

NameError: name 'img_to_array' is not defined

## === cell 7
generator.fit(to_gen_fit)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1493599609.py in <cell line: 0>()
----> 1 generator.fit(to_gen_fit)

NameError: name 'generator' is not defined

## === cell 8
len(train_datagen.next()[0]), len(valid_datagen.next()[0])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/459534848.py in <cell line: 0>()
----> 1 len(train_datagen.next()[0]), len(valid_datagen.next()[0])

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
/tmp/ipykernel_11/3974544208.py in <cell line: 0>()
----> 1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
      3 
      4 for i, arr in enumerate(train_datagen.next()[0]):
      5     img = array_to_img(arr)

NameError: name 'plt' is not defined

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
/tmp/ipykernel_11/2917829699.py in <cell line: 0>()
----> 1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
      3 
      4 for i, arr in enumerate(valid_datagen.next()[0]):
      5     img = array_to_img(arr)

NameError: name 'plt' is not defined

## === cell 11
model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
                                            trainable=True),
                             tf.keras.layers.Dense(classes, activation='softmax')
                            ])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3698878690.py in <cell line: 0>()
----> 1 model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
      2                                             trainable=True),
      3                              tf.keras.layers.Dense(classes, activation='softmax')
      4                             ])

NameError: name 'hub' is not defined

## === cell 12


model.build([None, input_size, input_size, 3])

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2494139886.py in <cell line: 0>()
      6 # model = Model(input_layer,output_layer)
      7 
----> 8 model.build([None, input_size, input_size, 3])
      9 
     10 model.compile(optimizer=optimizer,

NameError: name 'model' is not defined

## === cell 13
model.summary()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3035046171.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 14
history = model.fit(train_datagen,
                    validation_data=valid_datagen,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[early_stop,reduce_lr])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4058437641.py in <cell line: 0>()
----> 1 history = model.fit(train_datagen,
      2                     validation_data=valid_datagen,
      3                     batch_size=batch_size,
      4                     epochs=epochs,
      5                     callbacks=[early_stop,reduce_lr])

NameError: name 'model' is not defined

## === cell 15
tta_datagen = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
                                            target_size=(input_size, input_size),
                                            batch_size=batch_size,
                                            subset='validation',
                                            shuffle=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/296094707.py in <cell line: 0>()
----> 1 tta_datagen = generator.flow_from_directory('../input/paddy-disease-classification/train_images/',
      2                                             target_size=(input_size, input_size),
      3                                             batch_size=batch_size,
      4                                             subset='validation',
      5                                             shuffle=False)

NameError: name 'generator' is not defined

## === cell 16
eve_encodings = np.zeros((2077, 10))

for _ in range(5):
    encodings = model.predict(tta_datagen, verbose=1)
    eve_encodings += encodings
    
eve_encodings /= 5
eve_encodings


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2763246832.py in <cell line: 0>()
      2 
      3 for _ in range(5):
----> 4     encodings = model.predict(tta_datagen, verbose=1)
      5     eve_encodings += encodings
      6 

NameError: name 'model' is not defined

## === cell 17
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_datagen.classes

accuracy_score(true_classes, pred_classes)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/462040950.py in <cell line: 0>()
      1 pred_classes = np.argmax(eve_encodings, axis=1)
----> 2 true_classes = tta_datagen.classes
      3 
      4 accuracy_score(true_classes, pred_classes)

NameError: name 'tta_datagen' is not defined

## === cell 18
plt.figure(figsize=[12,6], dpi=300)
sns.lineplot(x=list(range(len(history.history['accuracy']))),
             y=history.history['accuracy'],
             label='train')
sns.lineplot(x=list(range(len(history.history['val_accuracy']))),
             y=history.history['val_accuracy'],
             label='validation')
plt.show()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/700131097.py in <cell line: 0>()
----> 1 plt.figure(figsize=[12,6], dpi=300)
      2 sns.lineplot(x=list(range(len(history.history['accuracy']))),
      3              y=history.history['accuracy'],
      4              label='train')
      5 sns.lineplot(x=list(range(len(history.history['val_accuracy']))),

NameError: name 'plt' is not defined

## === cell 19
plt.figure(figsize=[12,6], dpi=300)
sns.lineplot(x=list(range(len(history.history['loss']))),
             y=history.history['loss'],
             label='train')
sns.lineplot(x=list(range(len(history.history['val_loss']))),
             y=history.history['val_loss'],
             label='validation')
plt.show()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1528515447.py in <cell line: 0>()
----> 1 plt.figure(figsize=[12,6], dpi=300)
      2 sns.lineplot(x=list(range(len(history.history['loss']))),
      3              y=history.history['loss'],
      4              label='train')
      5 sns.lineplot(x=list(range(len(history.history['val_loss']))),

NameError: name 'plt' is not defined

## === cell 20
temp = pd.DataFrame(history.history)
temp.to_csv('model_effnetb4_history.csv', index=False)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3691212203.py in <cell line: 0>()
----> 1 temp = pd.DataFrame(history.history)
      2 temp.to_csv('model_effnetb4_history.csv', index=False)

NameError: name 'history' is not defined

## === cell 21
model.save('model_effnet_b4.hdf5')


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2922151421.py in <cell line: 0>()
----> 1 model.save('model_effnet_b4.hdf5')

NameError: name 'model' is not defined

## === cell 22
model.save_weights('model_effnet_b4_weights.hdf5')


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4105567460.py in <cell line: 0>()
----> 1 model.save_weights('model_effnet_b4_weights.hdf5')

NameError: name 'model' is not defined

## === cell 23
test_loc = '../input/paddy-disease-classification/test_images'

test_generator = ImageDataGenerator(rescale=1.0/255,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    preprocessing_function=test_time_augmentation_fn)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/396216341.py in <cell line: 0>()
      1 test_loc = '../input/paddy-disease-classification/test_images'
      2 
----> 3 test_generator = ImageDataGenerator(rescale=1.0/255,
      4                                     featurewise_center=True,
      5                                     featurewise_std_normalization=True,

NameError: name 'ImageDataGenerator' is not defined

## === cell 24
test_generator.fit(to_gen_fit)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4278429380.py in <cell line: 0>()
----> 1 test_generator.fit(to_gen_fit)

NameError: name 'test_generator' is not defined

## === cell 25
test_datagen = test_generator.flow_from_directory(directory=test_loc,
                                                  target_size=(input_size, input_size),
                                                  batch_size=batch_size,
                                                  classes=['.'],
                                                  shuffle=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4039627665.py in <cell line: 0>()
----> 1 test_datagen = test_generator.flow_from_directory(directory=test_loc,
      2                                                   target_size=(input_size, input_size),
      3                                                   batch_size=batch_size,
      4                                                   classes=['.'],
      5                                                   shuffle=False)

NameError: name 'test_generator' is not defined

## === cell 26
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(test_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    
plt.show()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/660403308.py in <cell line: 0>()
----> 1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
      3 
      4 for i, arr in enumerate(test_datagen.next()[0]):
      5     img = array_to_img(arr)

NameError: name 'plt' is not defined

## === cell 27
test_encodings = np.zeros((3469, 10))

for _ in range(5):
    encodings = model.predict(test_datagen, verbose=1)
    test_encodings += encodings
    
test_encodings /= 5
test_encodings


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1300850244.py in <cell line: 0>()
      2 
      3 for _ in range(5):
----> 4     encodings = model.predict(test_datagen, verbose=1)
      5     test_encodings += encodings
      6 

NameError: name 'model' is not defined

## === cell 28
predict_max = np.argmax(test_encodings, axis=1)
predict_max


## === cell 29
inverse_map = {v:k for k,v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467284863.py in <cell line: 0>()
----> 1 inverse_map = {v:k for k,v in train_datagen.class_indices.items()}
      2 predictions = [inverse_map[k] for k in predict_max]

NameError: name 'train_datagen' is not defined

## === cell 30
files=test_datagen.filenames

temp = pd.DataFrame({"image_id":files,
                      "label":predictions})

temp.image_id = temp.image_id.str.replace('./', '')
temp.to_csv('model_submission_v17.csv', index=False)
temp


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2876719919.py in <cell line: 0>()
----> 1 files=test_datagen.filenames
      2 
      3 temp = pd.DataFrame({"image_id":files,
      4                       "label":predictions})
      5 

NameError: name 'test_datagen' is not defined

## === cell 31
temp.label.value_counts()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/158377133.py in <cell line: 0>()
----> 1 temp.label.value_counts()

NameError: name 'temp' is not defined
