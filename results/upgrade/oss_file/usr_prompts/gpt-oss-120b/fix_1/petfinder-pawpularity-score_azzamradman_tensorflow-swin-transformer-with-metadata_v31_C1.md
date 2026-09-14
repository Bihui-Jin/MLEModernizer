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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.62399

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, 
                                     ReLU, Flatten, Softmax, GlobalAveragePooling2D, Concatenate, Add)
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
train_df['Id'] = train_df['Id'].apply(lambda x: x + '.jpg')
print(train_df.shape)
dirs_df = pd.DataFrame(columns=['dirs'])
dirs_df['dirs'] = train_df[['Id']]
train_labels = train_df['Pawpularity']


## === cell 2
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
dirs_df['fold'] = -1
dirs_df['label'] = train_labels
train_df['fold'] = -1

for fold, (tr, val) in enumerate(skf.split(dirs_df, train_labels)):
    dirs_df.loc[val, 'fold'] = fold
    train_df.loc[val, 'fold'] = fold


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4051533946.py in <cell line: 0>()
      1 n_splits = 10
----> 2 skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
      3 dirs_df['fold'] = -1
      4 dirs_df['label'] = train_labels
      5 train_df['fold'] = -1

NameError: name 'StratifiedKFold' is not defined

## === cell 3
train_dirs = dirs_df[~dirs_df['fold'].isin([5, 6, 9])].drop(['fold', 'label'], axis=1)
train_labs = dirs_df[~dirs_df['fold'].isin([5, 6, 9])].drop(['fold', 'dirs'], axis=1)
X_train = train_df[~train_df['fold'].isin([5, 6, 9])].drop(['fold', 'Pawpularity', 'Id'], axis=1)
y_train = train_df[~train_df['fold'].isin([5, 6, 9])]['Pawpularity']


valid_dirs = dirs_df[dirs_df['fold'].isin([5, 6])].drop(['fold', 'label'], axis=1)
valid_labs = dirs_df[dirs_df['fold'].isin([5, 6])].drop(['fold', 'dirs'], axis=1)
X_valid = train_df[~train_df['fold'].isin([5, 6, 9])].drop(['fold', 'Pawpularity', 'Id'], axis=1)
y_valid = train_df[~train_df['fold'].isin([5, 6, 9])]['Pawpularity']


test_dirs = dirs_df[dirs_df['fold'] == 9].drop(['fold', 'label'], axis=1)
test_labs = dirs_df[dirs_df['fold'] == 9].drop(['fold', 'dirs'], axis=1)
X_test = train_df[~train_df['fold'].isin([5, 6, 9])].drop(['fold', 'Pawpularity', 'Id'], axis=1)
y_test = train_df[~train_df['fold'].isin([5, 6, 9])]['Pawpularity']


## --- ERROR in cell 3, traceback:
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

KeyError: 'fold'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1280176593.py in <cell line: 0>()
----> 1 train_dirs = dirs_df[~dirs_df['fold'].isin([5, 6, 9])].drop(['fold', 'label'], axis=1)
      2 train_labs = dirs_df[~dirs_df['fold'].isin([5, 6, 9])].drop(['fold', 'dirs'], axis=1)
      3 X_train = train_df[~train_df['fold'].isin([5, 6, 9])].drop(['fold', 'Pawpularity', 'Id'], axis=1)
      4 y_train = train_df[~train_df['fold'].isin([5, 6, 9])]['Pawpularity']
      5 

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

KeyError: 'fold'

## === cell 4
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)

X_valid_tensor = tf.data.Dataset.from_tensor_slices(X_valid)
y_valid_tensor = tf.data.Dataset.from_tensor_slices(y_valid)

X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1530019389.py in <cell line: 0>()
----> 1 X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
      2 y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
      3 
      4 X_valid_tensor = tf.data.Dataset.from_tensor_slices(X_valid)
      5 y_valid_tensor = tf.data.Dataset.from_tensor_slices(y_valid)

NameError: name 'X_train' is not defined

## === cell 5
train_dir = '../input/petfinder-pawpularity-score/train/'
train_dirs = [train_dir+branch for branch in train_dirs.dirs.tolist()]
valid_dirs = [train_dir+branch for branch in valid_dirs.dirs.tolist()]
test_dirs = [train_dir+branch for branch in test_dirs.dirs.tolist()]


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344172913.py in <cell line: 0>()
      1 train_dir = '../input/petfinder-pawpularity-score/train/'
----> 2 train_dirs = [train_dir+branch for branch in train_dirs.dirs.tolist()]
      3 valid_dirs = [train_dir+branch for branch in valid_dirs.dirs.tolist()]
      4 test_dirs = [train_dir+branch for branch in test_dirs.dirs.tolist()]

NameError: name 'train_dirs' is not defined

## === cell 6
sample = pd.read_csv('../input/petfinder-pawpularity-score/sample_submission.csv')


## === cell 7
train_ds = tf.data.Dataset.list_files(train_dirs, shuffle=False)
valid_ds = tf.data.Dataset.list_files(valid_dirs, shuffle=False)
test_ds = tf.data.Dataset.list_files(test_dirs, shuffle=False)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/320060103.py in <cell line: 0>()
----> 1 train_ds = tf.data.Dataset.list_files(train_dirs, shuffle=False)
      2 valid_ds = tf.data.Dataset.list_files(valid_dirs, shuffle=False)
      3 test_ds = tf.data.Dataset.list_files(test_dirs, shuffle=False)

NameError: name 'train_dirs' is not defined

## === cell 8
train_image_counts = len(train_ds)
print("Number of images =", train_image_counts)

valid_image_counts = len(valid_ds)
print("Number of images =", valid_image_counts)

test_image_counts = len(test_ds)
print("Number of images =", test_image_counts)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2696774191.py in <cell line: 0>()
----> 1 train_image_counts = len(train_ds)
      2 print("Number of images =", train_image_counts)
      3 
      4 valid_image_counts = len(valid_ds)
      5 print("Number of images =", valid_image_counts)

NameError: name 'train_ds' is not defined

## === cell 9
for path in train_ds.take(5):
    print(path.numpy())


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3173807161.py in <cell line: 0>()
----> 1 for path in train_ds.take(5):
      2     print(path.numpy())

NameError: name 'train_ds' is not defined

## === cell 10
BATCH_SIZE = 32
EPOCHS = 100
VERBOSE = 1


## === cell 11
def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = img/255.0
    return img


## === cell 12
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds_images = train_ds.map(process_image)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379848611.py in <cell line: 0>()
      1 AUTOTUNE = tf.data.experimental.AUTOTUNE
----> 2 train_ds_images = train_ds.map(process_image)

NameError: name 'train_ds' is not defined

## === cell 13
def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = img/255.0
    return img


## === cell 14
valid_ds_images = valid_ds.map(process_image)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/18790135.py in <cell line: 0>()
----> 1 valid_ds_images = valid_ds.map(process_image)

NameError: name 'valid_ds' is not defined

## === cell 15
def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = img/255.0
    return img


## === cell 16
test_ds_images = test_ds.map(process_image)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2443566153.py in <cell line: 0>()
----> 1 test_ds_images = test_ds.map(process_image)

NameError: name 'test_ds' is not defined

## === cell 17
train_tensor = tf.data.Dataset.zip(({'img': train_ds_images, 'table': X_train_tensor}, 
                                    y_train_tensor)).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

valid_tensor = tf.data.Dataset.zip(({'img': valid_ds_images, 'table': X_valid_tensor}, 
                                    y_valid_tensor)).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_tensor = tf.data.Dataset.zip(({'img': test_ds_images, 'table': X_test_tensor}, 
                                    y_test_tensor)).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/502776249.py in <cell line: 0>()
----> 1 train_tensor = tf.data.Dataset.zip(({'img': train_ds_images, 'table': X_train_tensor}, 
      2                                     y_train_tensor)).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
      3 
      4 valid_tensor = tf.data.Dataset.zip(({'img': valid_ds_images, 'table': X_valid_tensor}, 
      5                                     y_valid_tensor)).shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

NameError: name 'train_ds_images' is not defined

## === cell 18
for epoch in range(5):
    for x, y in train_tensor:
        img = x['img']
        table = x['table']
        print(img.shape)
        print(table.shape)
        print(y.shape)
        break
    break


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3528259741.py in <cell line: 0>()
      1 for epoch in range(5):
----> 2     for x, y in train_tensor:
      3         img = x['img']
      4         table = x['table']
      5         print(img.shape)

NameError: name 'train_tensor' is not defined

## === cell 19
for epoch in range(1):
    for x, y in valid_tensor:
        img = x['img']
        table = x['table']
        print(img.shape)
        print(table.shape)
        print(y.shape)
        break
    break


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3234201280.py in <cell line: 0>()
      1 for epoch in range(1):
----> 2     for x, y in valid_tensor:
      3         img = x['img']
      4         table = x['table']
      5         print(img.shape)

NameError: name 'valid_tensor' is not defined

## === cell 20
for epoch in range(1):
    for x, y in test_tensor:
        img = x['img']
        table = x['table']
        print(img.shape)
        print(table.shape)
        print(y.shape)
        break
    break


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/23993213.py in <cell line: 0>()
      1 for epoch in range(1):
----> 2     for x, y in test_tensor:
      3         img = x['img']
      4         table = x['table']
      5         print(img.shape)

NameError: name 'test_tensor' is not defined

## === cell 21
def test_process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = img/255.0
    return img


## === cell 22
test_images_ds = tf.data.Dataset.list_files('../input/petfinder-pawpularity-score/test/*', shuffle=False)
test_images_ds = test_images_ds.map(test_process_image)


## === cell 23
test_df = pd.read_csv('../input/petfinder-pawpularity-score/test.csv').drop('Id', axis=1)
test_ds = tf.data.Dataset.from_tensor_slices(test_df)


## === cell 24
test_ds_tensor = tf.data.Dataset.zip(({'img': test_images_ds, 'table': test_ds})).batch(BATCH_SIZE)


## === cell 25
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = tpu_strategy.num_replicas_in_sync * 64
    print("Running on TPU:", tpu.master())
    print(f"Batch Size: {BATCH_SIZE}")
    
except ValueError:
    strategy = tf.distribute.get_strategy()
    BATCH_SIZE = BATCH_SIZE
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    print(f"Batch Size: {BATCH_SIZE}")


## === cell 26
input_shape = (224, 224, 3)
patch_size = (4, 4)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 224  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 3e-4
batch_size = 32
num_epochs = 40
validation_split = 0.1
weight_decay = 1e-5
label_smoothing = 0.1


## === cell 27
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3641643121.py in <cell line: 0>()
     23 
     24 
---> 25 class DropPath(layers.Layer):
     26     def __init__(self, drop_prob=None, **kwargs):
     27         super(DropPath, self).__init__(**kwargs)

NameError: name 'layers' is not defined

## === cell 28
class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3650518390.py in <cell line: 0>()
----> 1 class WindowAttention(layers.Layer):
      2     def __init__(
      3         self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
      4     ):
      5         super(WindowAttention, self).__init__(**kwargs)

NameError: name 'layers' is not defined

## === cell 29
class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3119629393.py in <cell line: 0>()
----> 1 class SwinTransformer(layers.Layer):
      2     def __init__(
      3         self,
      4         dim,
      5         num_patch,

NameError: name 'layers' is not defined

## === cell 30
class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1060345445.py in <cell line: 0>()
----> 1 class PatchExtract(layers.Layer):
      2     def __init__(self, patch_size, **kwargs):
      3         super(PatchExtract, self).__init__(**kwargs)
      4         self.patch_size_x = patch_size[0]
      5         self.patch_size_y = patch_size[0]

NameError: name 'layers' is not defined

## === cell 31
def swin_transformer_with_mlp_model():
    
    input_1 = layers.Input(input_shape, name='img')
    x = PatchExtract(patch_size)(input_1)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_1 = tf.keras.layers.LeakyReLU()(x)
    
    input_2 = Input(shape=(12,), name='table')
    x = Dense(256)(input_2)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dense(128)(x)
    x_2 = tf.keras.layers.LeakyReLU()(x)

    concat = Concatenate()([x_1, x_2])

    x = Dense(64)(concat)
    x = tf.keras.layers.LeakyReLU()(x)
    output = layers.Dense(1, activation='relu')(x)
    
    model = Model(inputs=[input_1, input_2], outputs=output)
    return model


## === cell 32
optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
loss_fn = tf.keras.losses.MeanSquaredError()
acc_metric = tf.keras.metrics.RootMeanSquaredError()
acc_metric_valid = tf.keras.metrics.RootMeanSquaredError()


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2250085631.py in <cell line: 0>()
----> 1 optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
      2 loss_fn = tf.keras.losses.MeanSquaredError()
      3 acc_metric = tf.keras.metrics.RootMeanSquaredError()
      4 acc_metric_valid = tf.keras.metrics.RootMeanSquaredError()

NameError: name 'tfa' is not defined

## === cell 33
model = swin_transformer_with_mlp_model()
print(model.summary())


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3064807656.py in <cell line: 0>()
----> 1 model = swin_transformer_with_mlp_model()
      2 print(model.summary())

/tmp/ipykernel_11/707257421.py in swin_transformer_with_mlp_model()
      2 
      3     # Images Swin Transformer
----> 4     input_1 = layers.Input(input_shape, name='img')
      5     x = PatchExtract(patch_size)(input_1)
      6     x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)

NameError: name 'layers' is not defined

## === cell 34
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3369879978.py in <cell line: 0>()
      1 # plot the model
----> 2 tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

NameError: name 'model' is not defined

## === cell 35
num_epochs = 10
for epoch in range(num_epochs):
    print(f"\n Start of Training Epoch {epoch+1}")
    for batch_idx, ((x_batch, y_batch), (x_batch_valid, y_batch_valid)) in enumerate(zip(train_tensor, valid_tensor)):
        img = x_batch['img']
        table = x_batch['table']
        
        img_valid = x_batch_valid['img']
        table_valid = x_batch_valid['table']
        
        with tf.GradientTape() as tape:
            y_pred = model((img, table), training=True)
            loss = loss_fn(y_batch, y_pred)
            
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)
        
        with tf.GradientTape() as tape_valid:
            y_pred_valid = model((img_valid, table_valid), training=False)
            loss_valid = loss_fn(y_batch_valid, y_pred_valid)
            
        acc_metric_valid.update_state(y_batch_valid, y_pred_valid)
        
    train_acc = acc_metric.result()
    valid_acc = acc_metric_valid.result()
    print(f"Train RMSE of epoch {train_acc}")
    print(f"Valid RMSE of epoch {valid_acc}")
    acc_metric.reset_states()
    acc_metric_valid.reset_states()


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3463414426.py in <cell line: 0>()
      2 for epoch in range(num_epochs):
      3     print(f"\n Start of Training Epoch {epoch+1}")
----> 4     for batch_idx, ((x_batch, y_batch), (x_batch_valid, y_batch_valid)) in enumerate(zip(train_tensor, valid_tensor)):
      5         img = x_batch['img']
      6         table = x_batch['table']

NameError: name 'train_tensor' is not defined

## === cell 36
y_test_preds_array = np.zeros(len(sample))
row = 0
for batch_idx, x_batch in enumerate(test_ds_tensor):
    print(f"Batch No. {batch_idx+1}")
    img = x_batch['img']
    table = x_batch['table']
              
    y_test_pred = model((img, table), training=False)
    
    try:            
        y_test_preds_array[row:row+BATCH_SIZE] = y_test_pred.numpy().flatten()
    except:
        y_test_preds_array[row:] = y_test_pred.numpy().flatten()
    
    row += BATCH_SIZE


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/716099456.py in <cell line: 0>()
      7     table = x_batch['table']
      8 
----> 9     y_test_pred = model((img, table), training=False)
     10 
     11     try:

NameError: name 'model' is not defined

## === cell 37
sample.iloc[:, 1] = y_test_preds_array
sample.to_csv('submission.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
