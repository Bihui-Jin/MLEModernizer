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

No external packages required in the script and installed.

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

21.955026738187023

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')
if (len(physical_device) >= 1):
    if (tf.config.experimental.get_memory_growth(physical_device[0]) == 1):
        tf.config.experimental.set_memory_growth(physical_device[0],True)

## === cell 5
dir_csv = '../input/petfinder-pawpularity-score/'

train_df = pd.read_csv(dir_csv+'train.csv')
test_df = pd.read_csv(dir_csv+'test.csv')

print('Train dataset has NaN values: ', train_df.isnull().values.any())
print('Test dataset has NaN values: ', test_df.isnull().values.any())

## === cell 8
train_df.head()

## === cell 10
corr_train_df = train_df.corr()
plt.figure(figsize=(14, 8))
sns.set(font_scale=1)
ax = sns.heatmap(corr_train_df,
        vmin=-1, vmax=1, annot=True, linewidths=.5,
        xticklabels=corr_train_df.columns,
        yticklabels=corr_train_df.columns)
ax.set_ylim(len(corr_train_df.keys()),0)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3879583982.py in <cell line: 0>()
----> 1 corr_train_df = train_df.corr()
      2 plt.figure(figsize=(14, 8))
      3 sns.set(font_scale=1)
      4 ax = sns.heatmap(corr_train_df,
      5         vmin=-1, vmax=1, annot=True, linewidths=.5,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: '1a8795e64a294ed0c95132e18ee198e1'

## === cell 12
corr_train_df = train_df.corr()
plt.figure(figsize=(10, 8))
sns.set(font_scale=1)
ax = sns.heatmap(corr_train_df[['Pawpularity']],
        vmin=-1, vmax=1, annot=True, linewidths=.5,
        xticklabels=['Pawpularity'],
        yticklabels=corr_train_df.columns
        )
ax.set_ylim(len(corr_train_df.keys()),0)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2031753765.py in <cell line: 0>()
----> 1 corr_train_df = train_df.corr()
      2 plt.figure(figsize=(10, 8))
      3 sns.set(font_scale=1)
      4 ax = sns.heatmap(corr_train_df[['Pawpularity']],
      5         vmin=-1, vmax=1, annot=True, linewidths=.5,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: '1a8795e64a294ed0c95132e18ee198e1'

## === cell 15
print('Min value of pawpularity: ', train_df['Pawpularity'].values.min())
print('Max value of pawpularity: ', train_df['Pawpularity'].values.max())

## === cell 17
%matplotlib inline
train_df['Pawpularity'].plot(kind="hist", bins=100)

## === cell 20
train_df['Pawpularity'].value_counts().sort_index()

## === cell 21
train_df['Pawpularity'].value_counts()

## === cell 23
sampled_train_df = pd.DataFrame(columns=train_df.keys())

## === cell 24
max_occ = train_df['Pawpularity'].value_counts().max()

for class_i in range(1,101):
    if(train_df[train_df['Pawpularity'] == class_i]['Pawpularity'].value_counts().values[0] < max_occ):
        ids_class_i = train_df.index[train_df['Pawpularity'] == class_i].tolist()
        sampled_ids_class_i = np.random.choice(ids_class_i, max_occ)
        sampled_train_df = pd.concat([sampled_train_df, train_df.loc[sampled_ids_class_i]])
    else:
        ids_class_i = train_df.index[train_df['Pawpularity'] == class_i].tolist()
        sampled_train_df = pd.concat([sampled_train_df, train_df.loc[ids_class_i]])
        
sampled_train_df = sampled_train_df.reset_index(drop=True)

## === cell 26
sampled_train_df['Pawpularity'].plot(kind="hist", bins=100)

## === cell 27
print('train_df:', train_df.info())
print('sampled_train_df:', sampled_train_df.info())

## === cell 28
for key in sampled_train_df.keys()[1:]:
    sampled_train_df[key] = sampled_train_df[key].astype('int64')
print('sampled_train_df:', sampled_train_df.info())

## === cell 31
class CustomTrainDataGen(Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(250, 250, 3),
                 shuffle=True): 
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.list_IDs = np.arange(len(self.df.index))
        self.indexes = np.arange(len(self.df.index))
        self.shuffle = shuffle 
        self.n = len(self.df)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, target_size):
        if (len(self.df.keys()) == len(train_df.keys())):
            img_path = dir_csv+'train/'+str(path)+'.jpg'
        else:
            print("Generator Image Data Generation Error")
            return -1
        image = tf.keras.preprocessing.image.load_img(img_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_data(self, batches):
        subfocus_batch = self.df.iloc[batches, :]['Subject Focus'].values
        eyes_batch = self.df.iloc[batches, :]['Eyes'].values
        face_batch = self.df.iloc[batches, :]['Face'].values
        near_batch = self.df.iloc[batches, :]['Near'].values
        action_batch = self.df.iloc[batches, :]['Action'].values
        acc_batch = self.df.iloc[batches, :]['Accessory'].values
        group_batch = self.df.iloc[batches, :]['Group'].values
        collage_batch = self.df.iloc[batches, :]['Collage'].values
        human_batch = self.df.iloc[batches, :]['Human'].values
        occlusion_batch = self.df.iloc[batches, :]['Occlusion'].values
        info_batch = self.df.iloc[batches, :]['Info'].values
        blur_batch = self.df.iloc[batches, :]['Blur'].values
        subfocus_batch = np.expand_dims(subfocus_batch, axis=1)
        eyes_batch = np.expand_dims(eyes_batch, axis=1)
        face_batch = np.expand_dims(face_batch, axis=1)
        near_batch = np.expand_dims(near_batch, axis=1)
        action_batch = np.expand_dims(action_batch, axis=1)
        acc_batch = np.expand_dims(acc_batch, axis=1)
        group_batch = np.expand_dims(group_batch, axis=1)
        collage_batch = np.expand_dims(collage_batch, axis=1)
        human_batch = np.expand_dims(human_batch, axis=1)
        occlusion_batch = np.expand_dims(occlusion_batch, axis=1)
        info_batch = np.expand_dims(info_batch, axis=1)
        blur_batch = np.expand_dims(blur_batch, axis=1)

        id_batch = self.df.iloc[batches, :]['Id'].values      
        image_batch = np.asarray([self.__get_input(id, self.input_size) for id\
             in id_batch])
        image_batch = np.reshape(image_batch,(self.batch_size,-1))
        
        pawpularity_batch = self.df.iloc[batches, :]['Pawpularity'].values

        X_batch = np.concatenate((subfocus_batch, eyes_batch, face_batch, near_batch,\
            action_batch, acc_batch, group_batch, collage_batch, human_batch,\
               occlusion_batch, info_batch, blur_batch, image_batch),axis=1)
        y_batch = pawpularity_batch
        return X_batch, y_batch
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]    
        X, y = self.__get_data(indexes)  
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

## === cell 33
class CustomTestDataGen(Sequence):
    
    def __init__(self, df, X_col,
                 batch_size,
                 input_size=(250, 250, 3),
                 shuffle=True): 
        self.df = df.copy()
        self.X_col = X_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.list_IDs = np.arange(len(self.df.index))
        self.indexes = np.arange(len(self.df.index))
        self.shuffle = shuffle 
        self.n = len(self.df)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, target_size):
        if (len(self.df.keys()) == len(test_df.keys())):
            img_path = dir_csv+'test/'+str(path)+'.jpg'
        else:
            print("Generator Image Data Generation Error")
            return -1
        image = tf.keras.preprocessing.image.load_img(img_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_data(self, batches):
        subfocus_batch = self.df.iloc[batches, :]['Subject Focus'].values
        eyes_batch = self.df.iloc[batches, :]['Eyes'].values
        face_batch = self.df.iloc[batches, :]['Face'].values
        near_batch = self.df.iloc[batches, :]['Near'].values
        action_batch = self.df.iloc[batches, :]['Action'].values
        acc_batch = self.df.iloc[batches, :]['Accessory'].values
        group_batch = self.df.iloc[batches, :]['Group'].values
        collage_batch = self.df.iloc[batches, :]['Collage'].values
        human_batch = self.df.iloc[batches, :]['Human'].values
        occlusion_batch = self.df.iloc[batches, :]['Occlusion'].values
        info_batch = self.df.iloc[batches, :]['Info'].values
        blur_batch = self.df.iloc[batches, :]['Blur'].values

        subfocus_batch = np.expand_dims(subfocus_batch, axis=1)
        eyes_batch = np.expand_dims(eyes_batch, axis=1)
        face_batch = np.expand_dims(face_batch, axis=1)
        near_batch = np.expand_dims(near_batch, axis=1)
        action_batch = np.expand_dims(action_batch, axis=1)
        acc_batch = np.expand_dims(acc_batch, axis=1)
        group_batch = np.expand_dims(group_batch, axis=1)
        collage_batch = np.expand_dims(collage_batch, axis=1)
        human_batch = np.expand_dims(human_batch, axis=1)
        occlusion_batch = np.expand_dims(occlusion_batch, axis=1)
        info_batch = np.expand_dims(info_batch, axis=1)
        blur_batch = np.expand_dims(blur_batch, axis=1)
        id_batch = self.df.iloc[batches, :]['Id'].values      
        image_batch = np.asarray([self.__get_input(id, self.input_size) for id\
             in id_batch])
        image_batch = np.reshape(image_batch,(self.batch_size,-1))

        X_batch = np.concatenate((subfocus_batch, eyes_batch, face_batch, near_batch,\
            action_batch, acc_batch, group_batch, collage_batch, human_batch,\
               occlusion_batch, info_batch, blur_batch, image_batch),axis=1)
        return X_batch
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]    
        X = self.__get_data(indexes)  
        return X
    
    def __len__(self):
        return self.n // self.batch_size

## === cell 36
def build_model(nb_annotations, image_shape):

    input_shape = nb_annotations + image_shape[0]*image_shape[1]*image_shape[2]
    inputs = Input(shape=input_shape)

    annotations_input = inputs[:,:nb_annotations]
    img_input = inputs[:,nb_annotations:]
    img_input = tf.reshape(img_input,(tf.shape(inputs)[0],image_shape[0],image_shape[1],image_shape[2]))

    x = Conv2D(16, 3, activation='relu')(img_input)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(32, 3, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(2)(x)

    x = Flatten()(x)

    x = Dense(16, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)

    x = Concatenate()([annotations_input, x])

    attention = Dense(32, activation='relu')(x)
    attention = Dense(28, activation='softmax')(attention)
    x = attention*x

    output = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    
    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

## === cell 39
num_epochs = 100
batch_size = 32
target_size = (250, 250, 3)
annotations = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action',\
             'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info',\
                  'Blur']

## === cell 41
model = build_model(len(annotations), target_size)

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/24384203.py in <cell line: 0>()
      1 # IF CPU or GPU
----> 2 model = build_model(len(annotations), target_size)

/tmp/ipykernel_11/2627233648.py in build_model(nb_annotations, image_shape)
      3     # Our input features of 12 annotations and corresponding image
      4     input_shape = nb_annotations + image_shape[0]*image_shape[1]*image_shape[2]
----> 5     inputs = Input(shape=input_shape)
      6 
      7     annotations_input = inputs[:,:nb_annotations]

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in Input(shape, batch_size, dtype, sparse, batch_shape, name, tensor, optional)
    189     ```
    190     """
--> 191     layer = InputLayer(
    192         shape=shape,
    193         batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/input_layer.py in __init__(self, shape, batch_size, dtype, sparse, batch_shape, input_tensor, optional, name, **kwargs)
     90 
     91             if shape is not None:
---> 92                 shape = backend.standardize_shape(shape)
     93                 batch_shape = (batch_size,) + shape
     94 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in standardize_shape(shape)
    560             raise ValueError("Undefined shapes are not supported.")
    561         if not hasattr(shape, "__iter__"):
--> 562             raise ValueError(f"Cannot convert '{shape}' to a shape.")
    563         if config.backend() == "tensorflow":
    564             if isinstance(shape, tf.TensorShape):

ValueError: Cannot convert '187512' to a shape.

## === cell 43
model.summary()

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3470139634.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 45
tr_df, val_df = train_test_split(sampled_train_df, test_size=0.1, random_state=2)
print(tr_df.shape)
print(val_df.shape)

## === cell 47
traingen = CustomTrainDataGen(tr_df,
                         X_col={'Id':'Id',
                         'Subject Focus':'Subject Focus',
                         'Eyes':'Eyes',
                         'Face':'Face',
                         'Near':'Near',
                         'Action':'Action',
                         'Accessory':'Accessory',
                         'Group':'Group',
                         'Collage':'Collage',
                         'Human':'Human',
                         'Occlusion':'Occlusion',
                         'Info':'Info',
                         'Blur':'Blur'},
                         y_col={'Pawpularity': 'Pawpularity'},
                         batch_size=batch_size,
                         input_size=target_size)
valgen = CustomTrainDataGen(val_df,
                       X_col={'Id':'Id',
                         'Subject Focus':'Subject Focus',
                         'Eyes':'Eyes',
                         'Face':'Face',
                         'Near':'Near',
                         'Action':'Action',
                         'Accessory':'Accessory',
                         'Group':'Group',
                         'Collage':'Collage',
                         'Human':'Human',
                         'Occlusion':'Occlusion',
                         'Info':'Info',
                         'Blur':'Blur'},
                       y_col={'Pawpularity': 'Pawpularity'},
                       batch_size=batch_size,
                       input_size=target_size)

## === cell 49
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

early_stop = EarlyStopping(monitor='val_loss', patience=30)

dir_path_batchtr = './Train_logs'
os.makedirs(dir_path_batchtr, exist_ok=True)

dir_weight_path_batchtr = dir_path_batchtr + '/Weights'
os.makedirs(dir_weight_path_batchtr, exist_ok=True)
checkpoint_name = dir_weight_path_batchtr + '/weights_best.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

dir_hist_path_batchtr = dir_path_batchtr + '/Histories'
os.makedirs(dir_hist_path_batchtr, exist_ok=True)
logger_name = dir_hist_path_batchtr + '/history_log.csv'
logger = CSVLogger(logger_name, append=True, separator=',')

dir_tensorboard_log = "./tensorboard_logs"
os.makedirs(dir_tensorboard_log, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_tensorboard_log)

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1781994533.py in <cell line: 0>()
     10 os.makedirs(dir_weight_path_batchtr, exist_ok=True)
     11 checkpoint_name = dir_weight_path_batchtr + '/weights_best.hdf5'
---> 12 checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
     13 
     14 # Logger for history

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=./Train_logs/Weights/weights_best.hdf5

## === cell 54
checkpoint_name = '../input/weights-final/weights_best.hdf5'
model.load_weights(checkpoint_name)

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1139116785.py in <cell line: 0>()
      1 checkpoint_name = '../input/weights-final/weights_best.hdf5'
----> 2 model.load_weights(checkpoint_name)

NameError: name 'model' is not defined

## === cell 56
testgen = CustomTestDataGen(test_df,
                         X_col={'Id':'Id',
                         'Subject Focus':'Subject Focus',
                         'Eyes':'Eyes',
                         'Face':'Face',
                         'Near':'Near',
                         'Action':'Action',
                         'Accessory':'Accessory',
                         'Group':'Group',
                         'Collage':'Collage',
                         'Human':'Human',
                         'Occlusion':'Occlusion',
                         'Info':'Info',
                         'Blur':'Blur'},
                         batch_size=1,
                         input_size=target_size,
                         shuffle=False)

## === cell 58
predictions = model.predict(testgen)

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/393110479.py in <cell line: 0>()
----> 1 predictions = model.predict(testgen)

NameError: name 'model' is not defined

## === cell 60
pred_df = pd.DataFrame({'Id':test_df['Id']})
pred_df['Pawpularity'] = predictions

pred_df.to_csv('./submission.csv', index=False)

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3822811205.py in <cell line: 0>()
      1 # New dataframe for predictions with the id from test_df
      2 pred_df = pd.DataFrame({'Id':test_df['Id']})
----> 3 pred_df['Pawpularity'] = predictions
      4 
      5 # Save as a CSV file

NameError: name 'predictions' is not defined

## === cell 62
orig_tr_df, orig_val_df = train_test_split(train_df, test_size=0.1, random_state=2)
print(orig_tr_df.shape)
print(orig_val_df.shape)

## === cell 63
valgen_test = CustomTrainDataGen(orig_val_df,
                       X_col={'Id':'Id',
                         'Subject Focus':'Subject Focus',
                         'Eyes':'Eyes',
                         'Face':'Face',
                         'Near':'Near',
                         'Action':'Action',
                         'Accessory':'Accessory',
                         'Group':'Group',
                         'Collage':'Collage',
                         'Human':'Human',
                         'Occlusion':'Occlusion',
                         'Info':'Info',
                         'Blur':'Blur'},
                       y_col={'Pawpularity': 'Pawpularity'},
                       batch_size=1,
                       input_size=target_size,
                       shuffle=False)

## === cell 64
predictions = model.predict(valgen_test)
print(predictions[:10])
print(orig_val_df.iloc[:10]['Pawpularity'])

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1529235663.py in <cell line: 0>()
----> 1 predictions = model.predict(valgen_test)
      2 print(predictions[:10])
      3 print(orig_val_df.iloc[:10]['Pawpularity'])

NameError: name 'model' is not defined
