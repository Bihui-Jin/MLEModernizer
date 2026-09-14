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

20.49045

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 2
df = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
test_csv = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')
submission = pd.read_csv('../input/petfinder-pawpularity-score/sample_submission.csv')


## === cell 3
df.shape


## === cell 4
df.head()


## === cell 5
df.info()


## === cell 6
sns.distplot(df['Pawpularity'])


## === cell 7
import os 
import cv2


## === cell 8
os.chdir('../input/petfinder-pawpularity-score/train')

size_data = pd.DataFrame()
for file in os.listdir():
    imgg = cv2.imread(file)
    w,h,c = imgg.shape
    size_data=size_data.append([[w,h,c,imgg.size/3]])
size_data


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1479723572.py in <cell line: 0>()
      7     imgg = cv2.imread(file)
      8     w,h,c = imgg.shape
----> 9     size_data=size_data.append([[w,h,c,imgg.size/3]])
     10 size_data

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 9
size_data[3].value_counts()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    412             try:
--> 413                 return self._range.index(new_key)
    414             except ValueError as err:

ValueError: 3 is not in range

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/831367834.py in <cell line: 0>()
----> 1 size_data[3].value_counts()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    413                 return self._range.index(new_key)
    414             except ValueError as err:
--> 415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
    417             raise KeyError(key)

KeyError: 3

## === cell 10
train_img = []
for file in os.listdir():
    file = cv2.imread(file)
    file = cv2.resize(file , (64,64) , interpolation = cv2.INTER_AREA)
    train_img.append(file/255)
train_img[:1]


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/3765292176.py in <cell line: 0>()
      2 for file in os.listdir():
      3     file = cv2.imread(file)
----> 4     file = cv2.resize(file , (64,64) , interpolation = cv2.INTER_AREA)
      5     train_img.append(file/255)
      6 train_img[:1]

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 11
train_img_name = []
for i in os.listdir():
    train_img_name.append(i)
train_img_name[:5]


## === cell 12
for name in train_img_name:
    if(name[:-4] == '.jpg'):
        print(name)


## === cell 13
import tensorflow as tf


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
train_csv_data = pd.DataFrame()
for img ,name in zip(train_img , train_img_name):
    name = name[:-4]
    location = df[df['Id']== name].index[0]
    train_csv_data = train_csv_data.append([df.loc[location]])
train_csv_data


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3272130670.py in <cell line: 0>()
      3     name = name[:-4]
      4     location = df[df['Id']== name].index[0]
----> 5     train_csv_data = train_csv_data.append([df.loc[location]])
      6 train_csv_data

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 15
train_csv_data = train_csv_data.reset_index().drop(['index'] , axis = 1)
train_csv_data


## === cell 16
image_1 = cv2.imread('./'+train_csv_data['Id'][0]+'.jpg')
plt.imshow(image_1)


## --- ERROR in cell 16, traceback:
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

KeyError: 'Id'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3496178639.py in <cell line: 0>()
----> 1 image_1 = cv2.imread('./'+train_csv_data['Id'][0]+'.jpg')
      2 plt.imshow(image_1)

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

KeyError: 'Id'

## === cell 17
plt.imshow(train_img[0])


## === cell 18
os.chdir('../test')

for i in os.listdir():
    file = cv2.imread(i)
    print(file.shape)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1647612462.py in <cell line: 0>()
      3 for i in os.listdir():
      4     file = cv2.imread(i)
----> 5     print(file.shape)

AttributeError: 'NoneType' object has no attribute 'shape'

## === cell 19
test_img = []
for i in os.listdir():
    file = cv2.imread(i)
    file = cv2.resize(file , (64,64) , interpolation = cv2.INTER_AREA)
    test_img.append(file/255)
test_img[:1]


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/1851939678.py in <cell line: 0>()
      2 for i in os.listdir():
      3     file = cv2.imread(i)
----> 4     file = cv2.resize(file , (64,64) , interpolation = cv2.INTER_AREA)
      5     test_img.append(file/255)
      6 test_img[:1]

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 20
test_img_name = []
for i in os.listdir():
    test_img_name.append(i)
test_img_name[:1]


## === cell 21
test_csv_data = pd.DataFrame()
for img,name in zip(test_img , test_img_name):
    name = name[:-4]
    location = test_csv[test_csv['Id'] == name].index[0]
    test_csv_data = test_csv_data.append([test_csv.loc[location]])
test_csv_data = test_csv_data.reset_index().drop(['index'] , axis= 1)
test_csv_data[:1]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/619562053.py in <cell line: 0>()
      3     name = name[:-4]
      4     location = test_csv[test_csv['Id'] == name].index[0]
----> 5     test_csv_data = test_csv_data.append([test_csv.loc[location]])
      6 test_csv_data = test_csv_data.reset_index().drop(['index'] , axis= 1)
      7 test_csv_data[:1]

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 22
plt.imshow(test_img[0])


## === cell 23
train_csv_x = train_csv_data.drop(['Id','Pawpularity'],axis=1)
train_y = train_csv_data['Pawpularity']

test_csv_x = test_csv_data.drop(['Id'],axis=1)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1838908785.py in <cell line: 0>()
----> 1 train_csv_x = train_csv_data.drop(['Id','Pawpularity'],axis=1)
      2 train_y = train_csv_data['Pawpularity']
      3 
      4 test_csv_x = test_csv_data.drop(['Id'],axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['Id', 'Pawpularity'] not found in axis"

## === cell 24
train_csv_x.shape


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2670970349.py in <cell line: 0>()
----> 1 train_csv_x.shape

NameError: name 'train_csv_x' is not defined

## === cell 25
csv_input = tf.keras.Input(shape = train_csv_x.shape[1:] , name = 'CSV_Input')
img_input = tf.keras.Input(shape = np.array(train_img).shape[1:] , name = 'IMG_input')
csv_hidden1 = tf.keras.layers.Dense(200, activation= 'elu' , kernel_initializer = 'he_normal' , name = 'CSV_Hidden1')(csv_input)
csv_hidden2 = tf.keras.layers.Dense(200, activation= 'elu' , kernel_initializer = 'he_normal' , name = 'CSV_Hidden2')(csv_hidden1)
csv_hidden3 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden3')(csv_hidden2)
csv_hidden4 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden4')(csv_hidden3)
csv_hidden5 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden5')(csv_hidden4)
csv_hidden6 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden6')(csv_hidden5)
csv_dropout = tf.keras.layers.Dropout(0.5, name ='CSV_Dropout')(csv_hidden6)

img_conv1 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv1')(img_input)
img_conv2 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv2')(img_conv1)
img_pooling1 = tf.keras.layers.MaxPooling2D(4, name= 'IMG_Max1')(img_conv2)

img_conv3 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv3')(img_pooling1)
img_conv4 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv4')(img_conv3)
img_pooling2 = tf.keras.layers.MaxPooling2D(4, name= 'IMG_Max2')(img_conv4)

img_conv5 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv5')(img_pooling2)
img_conv6 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv6')(img_conv5)
img_pooling3 = tf.keras.layers.MaxPooling2D(3, name= 'IMG_Max3')(img_conv6)

img_dropout = tf.keras.layers.Dropout(0.5, name = 'IMG_Dropout')(img_pooling3)
img_conv7 = tf.keras.layers.Conv2D(120, 4, padding = 'same', activation = 'elu', kernel_initializer = 'he_normal',name='IMG_Conv7')(img_dropout)

img_hidden1 = tf.keras.layers.Dense(300, activation='elu', kernel_initializer = 'he_normal',name='IMG_hidden1' ,use_bias=False)(img_conv7)
img_dropout1 = tf.keras.layers.Dropout(0.5, name='IMG_Dropout1')(img_hidden1)

img_hidden2 = tf.keras.layers.Dense(300, activation='elu', kernel_initializer = 'he_normal',name='IMG_hidden2' ,use_bias=False)(img_dropout1)

img_gpool = tf.keras.layers.GlobalAvgPool2D(name = 'IMG_Gpool')(img_hidden2)

img_dropout2 = tf.keras.layers.Dropout(0.5, name='IMG_Dropout2')(img_gpool)



csv_output = tf.keras.layers.Dense(1, name = 'CSV_Output')(csv_dropout)
img_output = tf.keras.layers.Dense(1,name = 'IMG_Output')(img_dropout2)

model = tf.keras.Model(inputs=[csv_input, img_input], outputs=[csv_output, img_output], name='MY_WORK')


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/248109546.py in <cell line: 0>()
----> 1 csv_input = tf.keras.Input(shape = train_csv_x.shape[1:] , name = 'CSV_Input')
      2 img_input = tf.keras.Input(shape = np.array(train_img).shape[1:] , name = 'IMG_input')
      3 csv_hidden1 = tf.keras.layers.Dense(200, activation= 'elu' , kernel_initializer = 'he_normal' , name = 'CSV_Hidden1')(csv_input)
      4 csv_hidden2 = tf.keras.layers.Dense(200, activation= 'elu' , kernel_initializer = 'he_normal' , name = 'CSV_Hidden2')(csv_hidden1)
      5 csv_hidden3 = tf.keras.layers.Dense(200, activation='elu', kernel_initializer = 'he_normal', name='CSV_Hidden3')(csv_hidden2)

NameError: name 'train_csv_x' is not defined

## === cell 26
model.summary()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3035046171.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 27
os.chdir('../')
os.chdir('../')
os.chdir('../')
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1543797672.py in <cell line: 0>()
      3 os.chdir('../')
      4 os.chdir('../')
----> 5 tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

NameError: name 'model' is not defined

## === cell 28
learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate = 0.002,
decay_steps = 10000,
decay_rate = 0.97)

opt = tf.keras.optimizers.Adam(learning_rate = learning_schedule)
model.compile(loss=['mse','mse'], loss_weights=[0.5, 0.5], optimizer = opt, metrics = tf.keras.metrics.RootMeanSquaredError())

epoch_number = 20

check_1 = tf.keras.callbacks.ModelCheckpoint('MY_WORK.h5', save_best_only=True, verbose=2)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3873949869.py in <cell line: 0>()
      5 
      6 opt = tf.keras.optimizers.Adam(learning_rate = learning_schedule)
----> 7 model.compile(loss=['mse','mse'], loss_weights=[0.5, 0.5], optimizer = opt, metrics = tf.keras.metrics.RootMeanSquaredError())
      8 
      9 epoch_number = 20

NameError: name 'model' is not defined

## === cell 29
model.fit( 
    x= [train_csv_x, np.array(train_img)], y = [train_y, train_y], epochs=epoch_number, 
    validation_split=0.2, verbose =2, workers=3, batch_size = 100, validation_batch_size = 100,
    callbacks = [check_1])


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3032895973.py in <cell line: 0>()
----> 1 model.fit( 
      2     x= [train_csv_x, np.array(train_img)], y = [train_y, train_y], epochs=epoch_number,
      3     validation_split=0.2, verbose =2, workers=3, batch_size = 100, validation_batch_size = 100,
      4     callbacks = [check_1])

NameError: name 'model' is not defined

## === cell 30
best_model = tf.keras.models.load_model('MY_WORK.h5')
csv_result, img_result = best_model.predict([test_csv_x, np.array(test_img)])
final_result = pd.DataFrame(0.5 * csv_result + 0.5 * img_result)
final_result.columns =['Pawpularity']
final_result


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/9756850.py in <cell line: 0>()
----> 1 best_model = tf.keras.models.load_model('MY_WORK.h5')
      2 csv_result, img_result = best_model.predict([test_csv_x, np.array(test_img)])
      3 final_result = pd.DataFrame(0.5 * csv_result + 0.5 * img_result)
      4 final_result.columns =['Pawpularity']
      5 final_result

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'MY_WORK.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 31
for ids, paw in zip(test_csv_data['Id'], final_result['Pawpularity']):
    location = submission[submission['Id'] == ids].index[0]
    submission['Pawpularity'].loc[location] = paw
submission
submission.to_csv('./working/submission.csv',index=False)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/925496780.py in <cell line: 0>()
----> 1 for ids, paw in zip(test_csv_data['Id'], final_result['Pawpularity']):
      2     location = submission[submission['Id'] == ids].index[0]
      3     submission['Pawpularity'].loc[location] = paw
      4 submission
      5 submission.to_csv('./working/submission.csv',index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'Id'
