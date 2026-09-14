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
lightgbm==4.6.0
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

20.50836

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1

AUTOTUNE = tf.data.experimental.AUTOTUNE  
img_size = 224
channels = 3
Batch_size = 64


train_dir = "/kaggle/input/petfinder-pawpularity-score/train/"
test_dir = "/kaggle/input/petfinder-pawpularity-score/test/"

def seed_everything():
    np.random.seed(123)
    random.seed(123)
    tf.random.set_seed(123)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    os.environ['PYTHONHASHSEED'] = str(123)

seed_everything()


## === cell 2

df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
df_test = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
Id = df_test["Id"].copy()



df["Id"] = df["Id"].apply(lambda x : "/kaggle/input/petfinder-pawpularity-score/train/" + x + ".jpg")
df_test["Id"] = df_test["Id"].apply(lambda x : "/kaggle/input/petfinder-pawpularity-score/test/" + x + ".jpg")


## === cell 3

def image_preprocess(is_labelled):  
    def augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, 0.95, 1.05)
        image = tf.image.random_contrast(image, 0.95, 1.05)
        return image
    
    def can_be_augmented(img, label):
        return augment(img), label
    
    return can_be_augmented if is_labelled else augment



def image_read(is_labelled):
    def decode(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (img_size, img_size))
        return image
    
    def can_be_decoded(path, label):
        return decode(path), label
    

    return can_be_decoded if is_labelled else decode


def create_dataset(df, batch_size, is_labelled = False, augment = False, shuffle = False):
    image_read_fn = image_read(is_labelled)
    image_preprocess_fn = image_preprocess(is_labelled)
    
    if is_labelled:
        dataset = tf.data.Dataset.from_tensor_slices((df["Id"].values, df["Pawpularity"].values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((df["Id"].values))
    
    dataset = dataset.map(image_read_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(image_preprocess_fn, num_parallel_calls=AUTOTUNE) if augment else dataset
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


## === cell 4

trn = df.iloc[:9000]
val = df.iloc[9001:]

train = create_dataset(trn, Batch_size, is_labelled = True, augment = True, shuffle = True)
validation = create_dataset(val, Batch_size, is_labelled = True, augment = False, shuffle = False)
test = create_dataset(df_test, Batch_size, is_labelled = False, augment = False, shuffle=False)


## === cell 5
train


## === cell 8
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


## === cell 9
def get_model03():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=(224,224,3))

    h = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="Conv_3")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)
    
    
    h = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", name="Conv_5")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)
    

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(512, activation="relu")(h)

    output = keras.layers.Dense(1, activation="relu")(h)

    model = keras.Model(inpt, output)
    
    return model


## === cell 10
model = get_model03()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3866698697.py in <cell line: 0>()
----> 1 model = get_model03()

/tmp/ipykernel_11/659289257.py in get_model03()
      6     inpt = keras.Input(shape=(224,224,3))
      7 
----> 8     h = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inpt)
      9 
     10     h = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)

AttributeError: module 'tensorflow.keras.layers' has no attribute 'experimental'

## === cell 11

early_stopping = EarlyStopping(patience = 5,restore_best_weights=True)


lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100, decay_rate=0.96,
    staircase=True)


## === cell 12

model.compile(loss="mse", 
              optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1355172600.py in <cell line: 0>()
      1 # Compiling and Fitting the model
      2 
----> 3 model.compile(loss="mse", 
      4               optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule),
      5               metrics=[tf.keras.metrics.RootMeanSquaredError()])

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
predictor = model.fit(train,
                      epochs=10, 
                      validation_data = validation,
                      callbacks=[early_stopping])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2155720003.py in <cell line: 0>()
----> 1 predictor = model.fit(train,
      2                       epochs=10,
      3                       validation_data = validation,
      4                       callbacks=[early_stopping])

NameError: name 'model' is not defined

## === cell 15
pred = model.predict(test)

final=pd.DataFrame()
final['Id']=Id
final['Pawpularity']=pred
final.to_csv('submission.csv',index=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3774955184.py in <cell line: 0>()
----> 1 pred = model.predict(test)
      2 
      3 final=pd.DataFrame()
      4 final['Id']=Id
      5 final['Pawpularity']=pred

NameError: name 'model' is not defined
