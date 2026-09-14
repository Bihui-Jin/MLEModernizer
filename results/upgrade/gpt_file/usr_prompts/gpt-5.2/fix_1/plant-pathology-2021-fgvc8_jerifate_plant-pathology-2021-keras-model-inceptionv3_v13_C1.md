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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.15789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)



## === cell 1
os.listdir('/kaggle/input/plant-pathology-2021-fgvc8/')


## === cell 2
import pandas as pd
import numpy as np
import os
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

plt.style.use('seaborn')
%matplotlib inline


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_df = pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/train.csv')
test_df = pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv')
print("Dataset Shape: ",train_df.shape)
train_df.head()


## === cell 4
datapath = glob('/kaggle/input/plant-pathology-2021-fgvc8/train_images/*')
print("Image Datasets Shape: ", len(datapath))


## === cell 5
def add_link(path):
    return_path ='/kaggle/input/plant-pathology-2021-fgvc8/train_images/'+str(path)
    return return_path


## === cell 6
def add_link_test(path):
    return_path ='/kaggle/input/plant-pathology-2021-fgvc8/test_images/'+str(path)
    return return_path


## === cell 7
train_df['image'] = train_df['image'].apply(lambda x: add_link(x))
train_df.head()


## === cell 8
test_df['image'] = test_df['image'].apply(lambda x: add_link_test(x))
test_df.head()


## === cell 9
f,ax = plt.subplots(1,1,figsize=(18,8))
ax = sns.countplot(train_df['labels'], order=train_df['labels'].value_counts().sort_values(ascending=False).index)
ax.set_xlim(0,8)
ax.set_title('Train Dataset per labels Count')
plt.show()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1014003019.py in <cell line: 0>()
      1 f,ax = plt.subplots(1,1,figsize=(18,8))
----> 2 ax = sns.countplot(train_df['labels'], order=train_df['labels'].value_counts().sort_values(ascending=False).index)
      3 ax.set_xlim(0,8)
      4 ax.set_title('Train Dataset per labels Count')
      5 plt.show()

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    479                 if order is not None:
    480                     error = "Input data must be a pandas object to reorder"
--> 481                     raise ValueError(error)
    482 
    483                 # The input data is an array

ValueError: Input data must be a pandas object to reorder

## === cell 10
unique_list = np.unique(train_df['labels'])
print(unique_list)
print(train_df['labels'].value_counts().count())


## === cell 11
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype = tf.float32)
    return image


## === cell 12
def get_label(path):
    return_label = train_df[train_df['image']==path]['labels']
    print(return_label)
    return list(return_label)


## === cell 13
def get_label_image(path):
    label = get_label(path)
    image = read_image(path)
    return label, image


## === cell 14
index_list = np.random.randint(128, size=5)
for i in index_list:
    path = datapath[i]
    sample_label, sample_image = get_label_image(path)

    plt.imshow(sample_image)
    plt.title(sample_label)
    plt.show()


## === cell 15
INPUT_SIZE = (224,224,3)
BATCH_SIZE = 32
CLASSES = train_df['labels'].value_counts().count() # 12


## === cell 16
train_data, val_data = train_test_split(train_df, test_size=0.2)
print('Train Data Shape: ',train_data.shape)
print('Validation Data Shape: ',val_data.shape)


## === cell 17
train_datagen = ImageDataGenerator(
    rescale = 1/255.,
    width_shift_range = 0.3, 
    zoom_range=0.2, 
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    rescale = 1/255.
)

val_datagen = ImageDataGenerator(
    rescale = 1/255.
)


## === cell 18
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='image',
    y_col='labels',
    target_size=INPUT_SIZE[:2],
    batch_size=BATCH_SIZE
)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image',
    y_col='labels',
    target_size=INPUT_SIZE[:2]
)
val_generator = val_datagen.flow_from_dataframe(
    val_data,
    x_col='image',
    y_col='labels',
    target_size=INPUT_SIZE[:2],
    batch_size=BATCH_SIZE
)


## === cell 19
pre_model = DenseNet121(include_top=False, weights='imagenet', input_shape=INPUT_SIZE)
pre_model.summary()


## === cell 20
model = tf.keras.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(CLASSES, activation='softmax'))


## === cell 21
callback= ReduceLROnPlateau(monitor='val_loss',
                       factor=.01, 
                       patience=3, 
                       min_lr=1e-5) 


## === cell 22
model.compile(optimizer =keras.optimizers.SGD(lr=0.001,momentum=.9,nesterov=False),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/697129799.py in <cell line: 0>()
----> 1 model.compile(optimizer =keras.optimizers.SGD(lr=0.001,momentum=.9,nesterov=False),
      2                 loss = 'categorical_crossentropy',
      3                 metrics = ['accuracy'])

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/sgd.py in __init__(self, learning_rate, momentum, nesterov, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     58         **kwargs,
     59     ):
---> 60         super().__init__(
     61             learning_rate=learning_rate,
     62             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.001}

## === cell 23
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator)//BATCH_SIZE,
    epochs=25,
    validation_data=val_generator,
    validation_steps=len(val_generator)//BATCH_SIZE,
    callbacks=[callback]
)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1465014845.py in <cell line: 0>()
----> 1 history = model.fit_generator(
      2     train_generator,
      3     steps_per_epoch=len(train_generator)//BATCH_SIZE,
      4     epochs=25,
      5     validation_data=val_generator,

AttributeError: 'Sequential' object has no attribute 'fit_generator'

## === cell 24
f,ax = plt.subplots(2,1,figsize=(20,10))
ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].set_title('Model loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Val'], loc='upper left')

ax[1].plot(history.history['accuracy'])
ax[1].plot(history.history['val_accuracy'])
ax[1].set_title('Model accuracy')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Val'], loc='upper left')

f.tight_layout()
plt.show()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2172471102.py in <cell line: 0>()
      1 f,ax = plt.subplots(2,1,figsize=(20,10))
----> 2 ax[0].plot(history.history['loss'])
      3 ax[0].plot(history.history['val_loss'])
      4 ax[0].set_title('Model loss')
      5 ax[0].set_ylabel('Loss')

NameError: name 'history' is not defined

## === cell 25
submission = pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv')
submission.head()


## === cell 26
preds=model.predict(test_generator)


## === cell 27
test_preds = np.argmax(preds, axis=-1)
submission['labels'] = test_preds


## === cell 28
submission.to_csv('submission.csv',index=False)
