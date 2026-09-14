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

0.96543

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
import seaborn as sns
import cv2
import albumentations as A

from albumentations.core.composition import Compose
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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
epochs = 100
lr = 1e-4
valid_split = 0.2
input_size = 224
batch_size = 32
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy


## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(patience=15,
                                              monitor='val_loss',
                                              restore_best_weights=True,
                                              verbose=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5,
                                                 monitor='val_loss',
                                                 factor=0.75,
                                                 verbose=1)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_chp.hdf5',
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1950436402.py in <cell line: 0>()
      9                                                  verbose=1)
     10 
---> 11 checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_chp.hdf5',
     12                                                 monitor='val_loss',
     13                                                 verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    192                 self.filepath.endswith(ext) for ext in (".keras", ".h5")
    193             ):
--> 194                 raise ValueError(
    195                     "The filepath provided must end in `.keras` "
    196                     "(Keras model format). Received: "

ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=best_chp.hdf5

## === cell 3
def resize(image, size):
    return tf.image.resize(image, size)


def blur(img, blur_limit):
    return cv2.blur(img, ksize=[blur_limit, blur_limit])


def gaussian_blur(img, blur_limit=(3, 7), sigma_limit=0):
    return cv2.GaussianBlur(img, ksize=blur_limit, sigmaX=sigma_limit)


def motion_blur(img, blur_limit=7):
    kmb = np.zeros((blur_limit, blur_limit))
    kmb[(blur_limit - 1) // 2, :] = np.ones(blur_limit)
    kmb = kmb / blur_limit
    return cv2.filter2D(img, -1, kernel=kmb)


def random_cut_out(images):
    return tfa.image.random_cutout(images, (32, 32), constant_values=0)


def aug_fn(image):
    data = {"image":image}
    aug_data = get_transform(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[224, 224])
    return aug_img

get_transform = Compose([A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)])


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/114927377.py in <cell line: 0>()
     30     return aug_img
     31 
---> 32 get_transform = Compose([A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)])

NameError: name 'Compose' is not defined

## === cell 4
def get_transforms_train(image):
    if np.random.choice([True, False], p=[0.45, 0.55]):
        crop_side = int(224*random.uniform(0.5, 1))
        temp = tf.image.random_crop(image, size=(crop_side, crop_side, 3)).numpy()
        temp = resize(temp, size=(224, 224)).numpy()

        temp = tf.image.random_flip_left_right(temp).numpy()

        if np.random.choice([True, False], p=[0.45, 0.55]):
            if random.choice([True, False]):
                delta = random.uniform(-0.3, 0.3)
                cf = random.uniform(-1.0, 1.0)
                temp = tf.image.adjust_brightness(temp, delta=delta).numpy()
                temp = tf.image.adjust_contrast(temp, contrast_factor=cf).numpy()

        if np.random.choice([True, False], p=[0.25, 0.75]):
            delta = random.uniform(-0.1, 0.2)
            temp = tf.image.adjust_hue(temp, delta=delta).numpy()




        if np.random.choice([True, False], p=[0.3, 0.7]):
            temp = temp.reshape([1,temp.shape[0], temp.shape[1], 3])
            temp = random_cut_out(temp).numpy()

            return tf.convert_to_tensor(temp[0], dtype=tf.float32)

        temp = aug_fn(temp).numpy()

        return tf.convert_to_tensor(temp, dtype=tf.float32)
    else:
        return image


## === cell 5
generator = ImageDataGenerator(rescale=1 / 255,
                                  rotation_range=10,
                                  shear_range=0.25,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  validation_split=valid_split,
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
/tmp/ipykernel_11/2335125942.py in <cell line: 0>()
----> 1 generator = ImageDataGenerator(rescale=1 / 255,
      2                                   rotation_range=10,
      3                                   shear_range=0.25,
      4                                   zoom_range=0.1,
      5                                   horizontal_flip=True,

NameError: name 'ImageDataGenerator' is not defined

## === cell 6
len(train_datagen.next()[0]), len(valid_datagen.next()[0])


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/459534848.py in <cell line: 0>()
----> 1 len(train_datagen.next()[0]), len(valid_datagen.next()[0])

NameError: name 'train_datagen' is not defined

## === cell 7
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(train_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    
plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3864340544.py in <cell line: 0>()
----> 1 fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
      2 axes = axes.ravel()
      3 
      4 for i, arr in enumerate(train_datagen.next()[0]):
      5     img = array_to_img(arr)

NameError: name 'plt' is not defined

## === cell 8
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(valid_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    
plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2709735406.py in <cell line: 0>()
----> 1 fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
      2 axes = axes.ravel()
      3 
      4 for i, arr in enumerate(valid_datagen.next()[0]):
      5     img = array_to_img(arr)

NameError: name 'plt' is not defined

## === cell 9
meta = pd.read_csv('../input/paddy-disease-classification/train.csv')
meta


## === cell 10
plt.figure(figsize=[24,20], dpi=200)
sns.barplot(x='age', y='label', hue='variety', data=meta, palette='OrRd_r')
plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/64722423.py in <cell line: 0>()
----> 1 plt.figure(figsize=[24,20], dpi=200)
      2 sns.barplot(x='age', y='label', hue='variety', data=meta, palette='OrRd_r')
      3 plt.show()

NameError: name 'plt' is not defined

## === cell 11
plt.figure(figsize=[12,6], dpi=200)
sns.barplot(x='age', y='label', hue='variety', 
            data=meta.groupby(by=['age', 'variety'])[['label']].count().reset_index(), 
            palette='OrRd_r')
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2513192799.py in <cell line: 0>()
----> 1 plt.figure(figsize=[12,6], dpi=200)
      2 sns.barplot(x='age', y='label', hue='variety', 
      3             data=meta.groupby(by=['age', 'variety'])[['label']].count().reset_index(),
      4             palette='OrRd_r')
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 12
back_bone = tf.keras.applications.Xception(weights='imagenet', input_shape=(input_size,input_size,3), include_top=False)
back_bone.summary()


## === cell 13
tf.keras.utils.plot_model(back_bone, to_file='xception.png')


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2497802290.py in <cell line: 0>()
----> 1 tf.keras.utils.plot_model(back_bone, to_file='xception.png')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/model_visualization.py in plot_model(model, to_file, show_shapes, show_dtype, show_layer_names, rankdir, expand_nested, dpi, show_layer_activations, show_trainable, **kwargs)
    475         extension = extension[1:]
    476     # Save image to disk.
--> 477     dot.write(to_file, format=extension)
    478     # Return the image as a Jupyter Image object, to be displayed in-line.
    479     # Note that we cannot easily detect whether the code is running in a

/usr/local/lib/python3.11/dist-packages/pydot/core.py in write(self, path, prog, format, encoding)
   1760                 f.write(s)
   1761         else:
-> 1762             s = self.create(prog, format, encoding=encoding)
   1763             with open(path, mode="wb") as f:
   1764                 f.write(s)

/usr/local/lib/python3.11/dist-packages/pydot/core.py in create(self, prog, format, encoding)
   1869             )
   1870 
-> 1871         assert process.returncode == 0, (
   1872             f'"{prog}" with args {arguments} '
   1873             f"returned code: {process.returncode}"

AssertionError: "dot" with args ['-Tpng', '/tmp/tmpsj4ytj8v/tmp5jbohmcb'] returned code: -6

## === cell 14
input_layer = Input(shape=(input_size,input_size,3))
x = back_bone(input_layer)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(input_layer,output_layer)

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/86477090.py in <cell line: 0>()
----> 1 input_layer = Input(shape=(input_size,input_size,3))
      2 x = back_bone(input_layer)
      3 x = GlobalAveragePooling2D()(x)
      4 output_layer = Dense(10, activation='softmax')(x)
      5 

NameError: name 'Input' is not defined

## === cell 15
model.summary()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3035046171.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 16
history = model.fit(train_datagen,
                    validation_data=valid_datagen,
                    batch_size=batch_size,
                    epochs=100,
                    callbacks=[early_stop,reduce_lr])


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/863531842.py in <cell line: 0>()
----> 1 history = model.fit(train_datagen,
      2                     validation_data=valid_datagen,
      3                     batch_size=batch_size,
      4                     epochs=100,
      5                     callbacks=[early_stop,reduce_lr])

NameError: name 'model' is not defined

## === cell 17
model.evaluate(valid_datagen)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2903439416.py in <cell line: 0>()
----> 1 model.evaluate(valid_datagen)

NameError: name 'model' is not defined

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
temp.to_csv('model_xception_history.csv', index=False)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3914904009.py in <cell line: 0>()
----> 1 temp = pd.DataFrame(history.history)
      2 temp.to_csv('model_xception_history.csv', index=False)

NameError: name 'history' is not defined

## === cell 21
model.save('model_xception.hdf5')


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2538191306.py in <cell line: 0>()
----> 1 model.save('model_xception.hdf5')

NameError: name 'model' is not defined

## === cell 22
model.save_weights('model_xception_weights.hdf5')


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3036980216.py in <cell line: 0>()
----> 1 model.save_weights('model_xception_weights.hdf5')

NameError: name 'model' is not defined

## === cell 23
test_loc = '../input/paddy-disease-classification/test_images'

test_data = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
                                                                    target_size=(input_size, input_size),
                                                                    batch_size=batch_size,
                                                                    classes=['.'],
                                                                    shuffle=False,
                                                                   )


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/389829401.py in <cell line: 0>()
      1 test_loc = '../input/paddy-disease-classification/test_images'
      2 
----> 3 test_data = ImageDataGenerator(rescale=1.0/255).flow_from_directory(directory=test_loc,
      4                                                                     target_size=(input_size, input_size),
      5                                                                     batch_size=batch_size,

NameError: name 'ImageDataGenerator' is not defined

## === cell 24
train_datagen.class_indices


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3388287861.py in <cell line: 0>()
----> 1 train_datagen.class_indices

NameError: name 'train_datagen' is not defined

## === cell 25
predict_max = np.argmax(model.predict(test_data, verbose=1),axis=1)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3320596487.py in <cell line: 0>()
----> 1 predict_max = np.argmax(model.predict(test_data, verbose=1),axis=1)

NameError: name 'model' is not defined

## === cell 26
inverse_map = {v:k for k,v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467284863.py in <cell line: 0>()
----> 1 inverse_map = {v:k for k,v in train_datagen.class_indices.items()}
      2 predictions = [inverse_map[k] for k in predict_max]

NameError: name 'train_datagen' is not defined

## === cell 27
files=test_data.filenames

temp = pd.DataFrame({"image_id":files,
                      "label":predictions})

temp.image_id = temp.image_id.str.replace('./', '')
temp.to_csv('model_submission_v5.csv', index=False)
temp


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2135220468.py in <cell line: 0>()
----> 1 files=test_data.filenames
      2 
      3 temp = pd.DataFrame({"image_id":files,
      4                       "label":predictions})
      5 

NameError: name 'test_data' is not defined

## === cell 28
temp.label.value_counts()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/158377133.py in <cell line: 0>()
----> 1 temp.label.value_counts()

NameError: name 'temp' is not defined
