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
pillow==11.3.0
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

0.89976

# 6. Current score

0.02998

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df=pd.read_csv('../input/paddy-disease-classification/train.csv')
train_df.head()


## === cell 2
all_images={}
train_images_path='../input/paddy-disease-classification/train_images/'
for category in os.listdir(train_images_path):
    for img in os.listdir(train_images_path+category):
        all_images[img]=train_images_path+category+'/'+img


## === cell 3
categories=os.listdir(train_images_path)


## === cell 4
label_dict={}
for category in categories:
    label_dict[category]=categories.index(category)
label_dict


## === cell 5
def get_label(label):
    return label_dict[label]


## === cell 6
num_label={
    0:'tungro',
    1:'hispa',
    2:'downy_mildew',
    3:'bacterial_leaf_streak', 
    4:'bacterial_leaf_blight',
    5:'brown_spot',
    6:'blast',
    7:'normal',
    8:'dead_heart',
    9:'bacterial_panicle_blight'
}


## === cell 7
def get_name(x):
    return num_label[x]


## === cell 8
num_label


## === cell 9
img_size=128


## === cell 10
training=[]
for i in range(len(train_df)):
    img_array=cv2.imread(all_images[train_df.iloc[i,0]])
    new_array=cv2.resize(img_array,(img_size,img_size))
    label=get_label(train_df.iloc[i,1])
    training.append([new_array,label])
training[0]


## === cell 11
random.shuffle(training)


## === cell 12
X=[]
y=[]
for features, label in training:
    X.append(features)
    y.append(label)
X=np.array(X).reshape(-1,img_size,img_size,3)


## === cell 13
X=X.astype('float32')
X/=255
from keras.utils import np_utils
Y=np_utils.to_categorical(y,10)
print(Y[100])
print(Y.shape)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3166544615.py in <cell line: 0>()
      1 X=X.astype('float32')
      2 X/=255
----> 3 from keras.utils import np_utils
      4 Y=np_utils.to_categorical(y,10)
      5 print(Y[100])

ImportError: cannot import name 'np_utils' from 'keras.utils' (/usr/local/lib/python3.11/dist-packages/keras/api/utils/__init__.py)

## === cell 14
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1023565936.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)

NameError: name 'Y' is not defined

## === cell 15
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/762717254.py in <cell line: 0>()
      1 from keras.models import Sequential
----> 2 from keras.layers.core import Dense, Activation, Dropout, Flatten
      3 from keras.layers.convolutional import Convolution2D, MaxPooling2D
      4 from tensorflow.keras.optimizers import Adam

ModuleNotFoundError: No module named 'keras.layers.core'

## === cell 16
model = tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(8192, activation='relu'),
tf.keras.layers.Dense(1024, activation='relu'),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])


## === cell 17
model.summary()


## === cell 18
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


## === cell 19
train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))
valid_ds=tf.data.Dataset.from_tensor_slices((X_valid,y_valid))


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/138079174.py in <cell line: 0>()
----> 1 train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))
      2 valid_ds=tf.data.Dataset.from_tensor_slices((X_valid,y_valid))

NameError: name 'X_train' is not defined

## === cell 20
history=model.fit(train_ds.batch(128),
         epochs=30,
         validation_data=valid_ds.batch(128))


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2912941310.py in <cell line: 0>()
----> 1 history=model.fit(train_ds.batch(128),
      2          epochs=30,
      3          validation_data=valid_ds.batch(128))

NameError: name 'train_ds' is not defined

## === cell 21
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training Data')
plt.plot(history.history['val_accuracy'],label='Validation Data')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3335191993.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
----> 2 plt.plot(history.history['accuracy'],label='Training Data')
      3 plt.plot(history.history['val_accuracy'],label='Validation Data')
      4 plt.xlabel('Epochs')
      5 plt.ylabel('Accuracy')

NameError: name 'history' is not defined

## === cell 22
model.evaluate(X_valid,y_valid)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/522423662.py in <cell line: 0>()
----> 1 model.evaluate(X_valid,y_valid)

NameError: name 'X_valid' is not defined

## === cell 23
submission_df=pd.read_csv('../input/paddy-disease-classification/sample_submission.csv')
submission_df


## === cell 24
test_path='../input/paddy-disease-classification/test_images/'
test_images=[]
for i in range(len(submission_df)):
    img_array=cv2.imread(test_path+submission_df.iloc[i,0])
    new_array=cv2.resize(img_array,(img_size,img_size))
    test_images.append(new_array)
test_images[0]


## === cell 25
X_test=[]
for features in test_images:
    X_test.append(features)
X_test=np.array(X_test).reshape(-1,img_size,img_size,3)


## === cell 26
X_test[0].shape


## === cell 27
X_test=X_test.astype('float32')
X_test/=255


## === cell 28
y_pred=model.predict(X_test)


## === cell 29
y_pred


## === cell 30
labels=[]
for i in y_pred:
    i=np.array(i)
    labels.append(np.argmax(i,axis=0))
labels


## === cell 31
submission_df['label']=labels


## === cell 32
submission_df['label']=submission_df['label'].apply(get_name)


## === cell 33
submission_df


## === cell 34
submission_df.to_csv('submission.csv',index=False)


## === cell 35
model.save("paddy_classification.h5")
