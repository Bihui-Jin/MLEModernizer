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
tqdm==4.67.1

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

22.28318

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Dense,concatenate, Activation,Add, Dropout, MaxPooling2D, Conv2D, Flatten, Input, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
comp_path = '../input/petfinder-pawpularity-score'
train_csv_path = '../input/petfinder-pawpularity-score/train.csv'
test_csv_path = '../input/petfinder-pawpularity-score/test.csv'


## === cell 2
train_meta = pd.read_csv(train_csv_path)
train_meta.head()


## === cell 3
def test(id):
    return os.path.join(comp_path, 'train', id+'.jpg')
train_meta['Id2'] = train_meta['Id'].apply(test)
train_meta.head()


## === cell 4
def gen_flow_for_two_inputs(datagen, batch, x_train, shuffle=True):
    """
    Args:
        datagen(image.ImageDataGenerator): data generator
        batch(int): batch size 
        x_train: dataframe for input img and metadata
        y_train(np.ndarray): label array for output 
        shuffle(bool): bool to shuffle data
    """
    x_train_2 = x_train.set_index('Id')
    batch = datagen.flow_from_dataframe(x_train, batch_size=batch, shuffle=shuffle, 
                                        x_col='Id2', y_col='Id', class_mode = 'raw',
                                        target_size=(224, 224))
    while True:
        batch_image, batch_index = batch.next()
        yield [batch_image, 
               x_train_2.loc[batch_index, 
                           ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 
                            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']].values], x_train_2.loc[batch_index, 'Pawpularity'].values


## === cell 5
input1 = Input(shape=(224, 224, 3))
input2 = Input(shape=(12,))

x1 = Conv2D(16, kernel_size=3, activation="relu")(input1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(32, kernel_size=3, activation="relu")(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(64, kernel_size=3, activation="relu")(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Dropout(0.5)(x1)
x1 = Flatten()(x1)
x1 = Model(inputs=input1, outputs=x1)

x2 = Model(inputs=input2, outputs=input2)

combined = concatenate([x1.output, x2.output])

z = Dense(64, activation="relu",  kernel_regularizer=regularizers.l2(0.001))(combined)
z = Dropout(0.25)(z)
z = Dense(8, activation="relu",  kernel_regularizer=regularizers.l2(0.001))(z)
z = Dense(1)(z)

model = Model(inputs=[x1.input, x2.input], outputs=z)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()


## === cell 6
train_datagen = image.ImageDataGenerator(
    rescale=1/255,
    )

EPOCH = 10
BATCH = 32

log = model.fit_generator(
    generator = gen_flow_for_two_inputs(train_datagen, BATCH, train_meta),
    steps_per_epoch = 2,
    epochs = EPOCH
    )


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/777359804.py in <cell line: 0>()
      6 BATCH = 32
      7 
----> 8 log = model.fit_generator(
      9     generator = gen_flow_for_two_inputs(train_datagen, BATCH, train_meta),
     10     #steps_per_epoch = np.ceil(train_meta.shape[0] / BATCH),

AttributeError: 'Functional' object has no attribute 'fit_generator'

## === cell 7
plt.plot(log.history['loss'], label='loss')
plt.legend(frameon=False) 
plt.xlabel("epochs")
plt.ylabel("mse")
plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2573717770.py in <cell line: 0>()
      1 # 学習結果のグラフ表示
----> 2 plt.plot(log.history['loss'], label='loss')
      3 #plt.plot(log.history['val_loss'], label='val_loss')
      4 plt.legend(frameon=False)
      5 plt.xlabel("epochs")

NameError: name 'log' is not defined

## === cell 8
test_datagen = image.ImageDataGenerator(
    rescale=1/255,
    )

BATCH = 32

test_meta = pd.read_csv(test_csv_path)
def test2(id):
    return os.path.join(comp_path, 'test', id+'.jpg')
test_meta['Id2'] = test_meta['Id'].apply(test2)

def gen_flow_for_two_inputs_2(datagen, batch, x_train, shuffle=True):
    """
    Args:
        datagen(image.ImageDataGenerator): data generator
        batch(int): batch size 
        x_train: dataframe for input img and metadata
        y_train(np.ndarray): label array for output 
        shuffle(bool): bool to shuffle data
    """
    x_train_2 = x_train.set_index('Id')
    batch = datagen.flow_from_dataframe(x_train, batch_size=batch, shuffle=shuffle, 
                                        x_col='Id2', y_col='Id', class_mode = 'raw',
                                        target_size=(224, 224))
    while True:
        batch_image, batch_index = batch.next()
        yield [batch_image, 
               x_train_2.loc[batch_index, 
                           ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 
                            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']].values], np.zeros(1)

pred = model.predict_generator(
    generator = gen_flow_for_two_inputs_2(test_datagen, BATCH, test_meta, shuffle=False), verbose= 1,
    steps = np.ceil(test_meta.shape[0] / BATCH)
    )


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/142202005.py in <cell line: 0>()
     33                             'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']].values], np.zeros(1)
     34 
---> 35 pred = model.predict_generator(
     36     generator = gen_flow_for_two_inputs_2(test_datagen, BATCH, test_meta, shuffle=False), verbose= 1,
     37     steps = np.ceil(test_meta.shape[0] / BATCH)

AttributeError: 'Functional' object has no attribute 'predict_generator'

## === cell 9
test_meta['Pawpularity'] = pred 
submission_df = test_meta[['Id','Pawpularity']]
submission_df.to_csv("submission.csv", index=False)
submission_df.head()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1827642141.py in <cell line: 0>()
----> 1 test_meta['Pawpularity'] = pred
      2 submission_df = test_meta[['Id','Pawpularity']]
      3 submission_df.to_csv("submission.csv", index=False)
      4 submission_df.head()

NameError: name 'pred' is not defined
