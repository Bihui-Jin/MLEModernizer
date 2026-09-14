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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.8831

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Resizing, Rescaling, RandomFlip, RandomRotation, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import EfficientNetB7, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras import Input
from sklearn.preprocessing import LabelEncoder


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_data = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train_data.head()


## === cell 3
dossier = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
train_data['image_name'] = dossier + train_data['image_id']+'.jpg'
train_data.head()


## === cell 5
train_data.info()


## === cell 6
train_data.shape


## === cell 7
labels_cols = (train_data.drop(['image_id', 'image_name'], axis=1)).columns.values
labels_cols


## === cell 8
print(train_data[labels_cols].isnull().sum())


## === cell 9
print(train_data[labels_cols].sum())


## === cell 10
X_cols = train_data[['image_id','image_name']]
X_cols.head()


## === cell 11
y_cols = train_data[labels_cols].values
y_cols


## === cell 12
print(y_cols.sum())


## === cell 13

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_cols, y_cols)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1384118816.py in <cell line: 0>()
      1 # Oversampling pour que chaque catégorie ait la même quantité de données
      2 
----> 3 ros = RandomOverSampler(random_state=42)
      4 X_train, y_train = ros.fit_resample(X_cols, y_cols)

NameError: name 'RandomOverSampler' is not defined

## === cell 14
X_train.head()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1605972052.py in <cell line: 0>()
----> 1 X_train.head()

NameError: name 'X_train' is not defined

## === cell 15
print(y_train.sum())


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/831456268.py in <cell line: 0>()
----> 1 print(y_train.sum())

NameError: name 'y_train' is not defined

## === cell 16
labels_train = pd.DataFrame(y_train, columns=labels_cols)
labels_train.head()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4102428120.py in <cell line: 0>()
----> 1 labels_train = pd.DataFrame(y_train, columns=labels_cols)
      2 labels_train.head()

NameError: name 'y_train' is not defined

## === cell 17
print(labels_train.sum())


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/673540608.py in <cell line: 0>()
----> 1 print(labels_train.sum())

NameError: name 'labels_train' is not defined

## === cell 18
label_names = labels_train[labels_train==1].stack().reset_index()['level_1']
label_names


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1025793409.py in <cell line: 0>()
      1 # Transposition pour avoir une colonne "etat_sante" qui regroupe tous les types
----> 2 label_names = labels_train[labels_train==1].stack().reset_index()['level_1']
      3 label_names

NameError: name 'labels_train' is not defined

## === cell 19
labels_train["state"] = label_names
labels_train.head()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2450044464.py in <cell line: 0>()
----> 1 labels_train["state"] = label_names
      2 labels_train.head()

NameError: name 'label_names' is not defined

## === cell 20
train_data = pd.concat([X_train, labels_train], axis=1)
train_data.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2884083210.py in <cell line: 0>()
      1 # Gather all data
----> 2 train_data = pd.concat([X_train, labels_train], axis=1)
      3 train_data.head()

NameError: name 'X_train' is not defined

## === cell 21
train_data.shape


## === cell 22
IMG_SIZE = 224


## === cell 23
def ici_decode_image(filename, label, image_size=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    label = tf.expand_dims(label, 0)
    
    if label is None:
        return image
    else:
        return image, label

def ici_data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    label = tf.expand_dims(label, 0)
    
    if label is None:
        return image
    else:
        return image, label
    
def ici_decode_image_test(filename, image_size=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return image

def preprocess_img4(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
    return img
   

def flip_and_rotate(img_arr):
    img_flip = tf.image.random_flip_left_right(img_arr)
    img_flip = tf.image.random_flip_up_down(img_flip)
    img_rot = tf.image.rot90(img_flip)
    return img_rot
    

def preprocess_train_img(img_path):
    img_path = tf.keras.backend.get_value(img_path)
    
    img = tf.keras.utils.load_img(img_path)
    img_arr = tf.keras.utils.img_to_array(img)
    img_arr = tf.image.resize(img_arr, [IMG_SIZE,IMG_SIZE])
    img_arr = tf.cast(img_arr, tf.float32) / 255.0
    
    img_flip = tf.image.random_flip_left_right(img_arr)
    img_flip = tf.image.random_flip_up_down(img_flip)
    img_rot = tf.image.rot90(img_flip)
    img_rot = img_rot.set_shape(tf.TensorShape([IMG_SIZE, IMG_SIZE, 3]))
    return img_rot


## === cell 24
label_enc = LabelEncoder()
label_enc.fit(train_data["state"])


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/950144454.py in <cell line: 0>()
----> 1 label_enc = LabelEncoder()
      2 label_enc.fit(train_data["state"])

NameError: name 'LabelEncoder' is not defined

## === cell 25
train_data["state_label"] = label_enc.transform(train_data["state"])


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2199448846.py in <cell line: 0>()
----> 1 train_data["state_label"] = label_enc.transform(train_data["state"])

NameError: name 'label_enc' is not defined

## === cell 26
train_data["state_label"] 


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'state_label'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2772118436.py in <cell line: 0>()
----> 1 train_data["state_label"]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'state_label'

## === cell 27
X = train_data["image_name"]
y = train_data["state_label"]


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'state_label'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2617412951.py in <cell line: 0>()
      1 X = train_data["image_name"]
----> 2 y = train_data["state_label"]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'state_label'

## === cell 28
print(X.shape, y.shape)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2320836809.py in <cell line: 0>()
----> 1 print(X.shape, y.shape)

NameError: name 'y' is not defined

## === cell 29
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/328271290.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

NameError: name 'train_test_split' is not defined

## === cell 30
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1314262932.py in <cell line: 0>()
----> 1 print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

NameError: name 'X_train' is not defined

## === cell 31
X_train = X_train.values
X_val = X_val.values

y_train = y_train.values
y_val = y_val.values


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/265027299.py in <cell line: 0>()
----> 1 X_train = X_train.values
      2 X_val = X_val.values
      3 
      4 y_train = y_train.values
      5 y_val = y_val.values

NameError: name 'X_train' is not defined

## === cell 32
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
AUTOTUNE


## === cell 33
AUTO = tf.data.experimental.AUTOTUNE
AUTO


## === cell 34
print(X_train.shape, y_train.shape)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1798708973.py in <cell line: 0>()
----> 1 print(X_train.shape, y_train.shape)

NameError: name 'X_train' is not defined

## === cell 35
ici_train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(ici_decode_image, num_parallel_calls=AUTO)
    .map(ici_data_augment, num_parallel_calls=AUTO)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3295929959.py in <cell line: 0>()
      1 ici_train_dataset = (
      2     tf.data.Dataset
----> 3     .from_tensor_slices((X_train, y_train))
      4     .map(ici_decode_image, num_parallel_calls=AUTO)
      5     .map(ici_data_augment, num_parallel_calls=AUTO)

NameError: name 'X_train' is not defined

## === cell 36
ici_val_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(ici_decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3738169947.py in <cell line: 0>()
      1 ici_val_dataset = (
      2     tf.data.Dataset
----> 3     .from_tensor_slices((X_val, y_val))
      4     .map(ici_decode_image, num_parallel_calls=AUTO)
      5     .batch(BATCH_SIZE)

NameError: name 'X_val' is not defined

## === cell 37
pretrained_model = DenseNet201(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet", include_top=False)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/518408433.py in <cell line: 0>()
----> 1 pretrained_model = DenseNet201(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet", include_top=False)

NameError: name 'DenseNet201' is not defined

## === cell 38
pretrained_model.trainable = False


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2768991044.py in <cell line: 0>()
----> 1 pretrained_model.trainable = False

NameError: name 'pretrained_model' is not defined

## === cell 39
model = Sequential([   
    pretrained_model,
    GlobalAveragePooling2D(),
    Dense(units=128, activation="relu"),
    Dropout(0.3),
    Dense(units=64, activation="relu"),
    Dense(units=4, activation="softmax"),    
])


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1941274764.py in <cell line: 0>()
----> 1 model = Sequential([   
      2     pretrained_model,
      3     GlobalAveragePooling2D(),
      4     Dense(units=128, activation="relu"),
      5     Dropout(0.3),

NameError: name 'Sequential' is not defined

## === cell 40
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3664776423.py in <cell line: 0>()
----> 1 model.compile(loss="sparse_categorical_crossentropy", 
      2               optimizer=Adam(learning_rate=0.001),
      3               metrics=['accuracy'])

NameError: name 'model' is not defined

## === cell 41
model.summary()


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3035046171.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 42
len(ici_train_dataset)


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3898688919.py in <cell line: 0>()
----> 1 len(ici_train_dataset)

NameError: name 'ici_train_dataset' is not defined

## === cell 43
history = model.fit(ici_train_dataset,
                    validation_data=ici_val_dataset,
                    epochs=20,
                    batch_size = BATCH_SIZE)


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2545391703.py in <cell line: 0>()
----> 1 history = model.fit(ici_train_dataset,
      2                     validation_data=ici_val_dataset,
      3                     epochs=20,
      4                     batch_size = BATCH_SIZE)

NameError: name 'model' is not defined

## === cell 44
test_data = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test_data.head()


## === cell 45
test_data.shape


## === cell 46
dossier = "/kaggle/input/plant-pathology-2020-fgvc7/images/"
test_data['image_name'] = dossier + test_data['image_id']+'.jpg'
test_data.head()


## === cell 47
X_test = test_data["image_name"].values


## === cell 48
ici_test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test))
    .map(ici_decode_image_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


## === cell 49
preds = model.predict(ici_test_dataset)


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/123849543.py in <cell line: 0>()
----> 1 preds = model.predict(ici_test_dataset)
      2 #preds.shape

NameError: name 'model' is not defined

## === cell 50
predictions_df = pd.DataFrame(np.round(preds,2), columns=["healthy", "multiple_diseases", "rust", "scab"])
predictions_df.head()


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3390194072.py in <cell line: 0>()
----> 1 predictions_df = pd.DataFrame(np.round(preds,2), columns=["healthy", "multiple_diseases", "rust", "scab"])
      2 predictions_df.head()

NameError: name 'preds' is not defined

## === cell 51
submission_file = pd.concat([test_data["image_id"], predictions_df], axis=1)

submission_file.head()


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1398748886.py in <cell line: 0>()
----> 1 submission_file = pd.concat([test_data["image_id"], predictions_df], axis=1)
      2 
      3 submission_file.head()

NameError: name 'predictions_df' is not defined

## === cell 52
submission_file.shape


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3131767.py in <cell line: 0>()
----> 1 submission_file.shape

NameError: name 'submission_file' is not defined

## === cell 53
submission_file.to_csv("Submission.csv", index=False)


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/35417577.py in <cell line: 0>()
----> 1 submission_file.to_csv("Submission.csv", index=False)

NameError: name 'submission_file' is not defined
