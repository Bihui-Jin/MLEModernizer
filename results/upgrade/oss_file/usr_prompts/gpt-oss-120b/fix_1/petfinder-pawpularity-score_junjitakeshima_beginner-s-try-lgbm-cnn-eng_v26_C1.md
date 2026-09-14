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
missingno==0.5.2
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

20.58389

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error


## === cell 1
train = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
test = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')


## === cell 2
train.head(20)


## === cell 3
train_corr = train.corr()
plt.figure(figsize = (13,13))
sns.heatmap(train_corr, vmax=1, vmin=-1, center=0,annot=True)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4218857193.py in <cell line: 0>()
----> 1 train_corr = train.corr()
      2 plt.figure(figsize = (13,13))
      3 sns.heatmap(train_corr, vmax=1, vmin=-1, center=0,annot=True)

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

## === cell 4
train["Pawpularity"].plot.hist(bins=50)


## === cell 5
def training_exe (train):
    kf = KFold(n_splits = 3)
    models = []
    rmses =[]
    categories = ["Id"]
    
    train["Id"] = train["Id"].astype('category')
    X_train = train.drop(['Pawpularity'], axis=1)
    Y_train = train['Pawpularity']
    
    lgbm_params = {
        "objective":"regression",
        "random_seed":1234
    }
    
    for train_index, val_index in kf.split(X_train):
        XX_train = X_train.iloc[train_index]
        XX_valid = X_train.iloc[val_index]
        
        YY_train = Y_train.iloc[train_index]
        YY_valid = Y_train.iloc[val_index]
        
        lgbm_train = lgbm.Dataset(XX_train, YY_train, categorical_feature = categories)
        lgbm_eval = lgbm.Dataset(XX_valid, YY_valid, categorical_feature = categories, reference=lgbm_train)
        
        model_lgbm = lgbm.train(lgbm_params,
                           lgbm_train,
                           valid_sets = lgbm_eval,
                           num_boost_round = 200,
                           early_stopping_rounds =20,
                           verbose_eval = 10,
                           )
        y_pred = model_lgbm.predict(XX_valid, num_iteration = model_lgbm.best_iteration)
        
        tmp_rmse = np.sqrt(mean_squared_error(YY_valid, y_pred))
        print (tmp_rmse)
        models.append(model_lgbm)
        rmses.append(tmp_rmse)
        
    ave_rmse = sum(rmses)/len(rmses)    
    
    return models, ave_rmse


## === cell 6
def pred_exe (test):
    preds=[]
    test["Id"] = test["Id"].astype('category')

    for model in models:
        pred =model.predict(test)
        preds.append(pred)
    
    preds_array = np.array(preds)
    preds_mean = np.mean(preds_array, axis=0)
    
    return preds_mean


## === cell 7
models, ave_rmse = training_exe (train)
ave_rmse


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1167280548.py in <cell line: 0>()
----> 1 models, ave_rmse = training_exe (train)
      2 ave_rmse

/tmp/ipykernel_11/1829248154.py in training_exe(train)
     24         lgbm_eval = lgbm.Dataset(XX_valid, YY_valid, categorical_feature = categories, reference=lgbm_train)
     25 
---> 26         model_lgbm = lgbm.train(lgbm_params,
     27                            lgbm_train,
     28                            valid_sets = lgbm_eval,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 8
test.head()


## === cell 9
pred = pred_exe(test)

sub = pd.DataFrame()
sub['Id']=test['Id']

sub.head()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2907418988.py in <cell line: 0>()
----> 1 pred = pred_exe(test)
      2 
      3 sub = pd.DataFrame()
      4 sub['Id']=test['Id']
      5 #sub['Pawpularity'] = pred

/tmp/ipykernel_11/609727599.py in pred_exe(test)
      3     test["Id"] = test["Id"].astype('category')
      4 
----> 5     for model in models:
      6         pred =model.predict(test)
      7         preds.append(pred)

NameError: name 'models' is not defined

## === cell 10
def add_feature(df) :
    df["Attractive"] = df["Eyes"] + df["Face"] + df["Near"] + df["Subject Focus"]
    df["Humantic"]   = df["Human"] + df["Collage"]
    df["Addition"]   = df["Accessory"] + df["Info"]
    return df  


## === cell 11
train_df1 = train.copy()
add_feature(train_df1)


## === cell 12
models, ave_rmse = training_exe (train_df1)
ave_rmse


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3910511185.py in <cell line: 0>()
----> 1 models, ave_rmse = training_exe (train_df1)
      2 ave_rmse

/tmp/ipykernel_11/1829248154.py in training_exe(train)
     24         lgbm_eval = lgbm.Dataset(XX_valid, YY_valid, categorical_feature = categories, reference=lgbm_train)
     25 
---> 26         model_lgbm = lgbm.train(lgbm_params,
     27                            lgbm_train,
     28                            valid_sets = lgbm_eval,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 13
test_df1 = test.copy()

add_feature(test)


## === cell 14
pred = pred_exe(test)

sub['Pawpularity'] = pred
sub.head()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2826212274.py in <cell line: 0>()
----> 1 pred = pred_exe(test)
      2 
      3 #sub = pd.DataFrame()
      4 #sub['Id']=test['Id']
      5 sub['Pawpularity'] = pred

/tmp/ipykernel_11/609727599.py in pred_exe(test)
      3     test["Id"] = test["Id"].astype('category')
      4 
----> 5     for model in models:
      6         pred =model.predict(test)
      7         preds.append(pred)

NameError: name 'models' is not defined

## === cell 15
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 16
import missingno as msno

train=pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
test=pd.read_csv('../input/petfinder-pawpularity-score/test.csv')


## === cell 17
train["file_path"] = train["Id"].apply(lambda x: "../input/petfinder-pawpularity-score/train/" + x + ".jpg")
test["file_path"] = test["Id"].apply(lambda x: "../input/petfinder-pawpularity-score/test/" + x + ".jpg")


## === cell 18
%%time

def preprocess(image_url):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (128, 128))
    return image
x_train=[]
for i in train['file_path']:
    x1=preprocess(i)
    x_train.append(x1)


## === cell 19
plt.figure(figsize=(20, 20))
row, col = 5, 4
for i in range(row * col):
    plt.subplot(row, col, i+1)
    image = cv2.imread(train.loc[i, 'file_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target = train.loc[i, 'Pawpularity']
    plt.imshow(image)
    plt.title(f"No: {i}" f"   Pawpularity: {target}")
plt.show()


## === cell 20
test1=[]
for i in test['file_path']:
    x1=preprocess(i)
    test1.append(x1)
test1=np.array(test1)


## === cell 21
x_train=np.array(x_train)
y_train=train['Pawpularity']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2)


## === cell 22
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


## === cell 23
model = Sequential()
input_shape = (128, 128, 3)


## === cell 24
model.add(
    Conv2D(
        filters=32,
        kernel_size = (3,3),
        strides =2,
        padding = 'same',
        activation = 'relu',
        input_shape = input_shape,
    ))

model.add(
    Conv2D(
        filters=64,
        kernel_size = (3,3),
        strides =2,
        padding = 'same',
        activation = 'relu',        
    ))

model.add(
    Conv2D(
        filters=32,
        kernel_size = (3,3),
        strides =2,
        padding = 'same',
        activation = 'relu',        
    ))

model.add(Flatten())

model.add(
    Dense(128, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),    
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"],
    optimizer = tf.keras.optimizers.Adam(1e-3)
)


## === cell 25
total_train = x_train.shape[0]
total_validate = y_train.shape[0]
batch_size = 64


## === cell 26
%%time

import math
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, Callback

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(
        drop,
        math.floor((epoch)/epochs_drop)
    )
    return lrate

lrate = LearningRateScheduler(step_decay)

earstop = EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0,
    patience = 5)

history = model.fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size=64,
    validation_data = (x_test,y_test),
    verbose = 1,
    callbacks = [lrate, earstop]
)


## === cell 27
cnn_pred=model.predict(test1)


## === cell 28
sub['Pawpularity']=cnn_pred
sub


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3089887516.py in <cell line: 0>()
      1 #sub=pd.DataFrame()
      2 #sub['Id']=test['Id']
----> 3 sub['Pawpularity']=cnn_pred
      4 #sub.to_csv('submission.csv',index=False)
      5 sub

NameError: name 'sub' is not defined

## === cell 29
sub['Pawpularity'] = pred*0.5 + cnn_pred*0.5
sub.to_csv('submission.csv',index=False)
sub


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1876633844.py in <cell line: 0>()
----> 1 sub['Pawpularity'] = pred*0.5 + cnn_pred*0.5
      2 sub.to_csv('submission.csv',index=False)
      3 sub

NameError: name 'pred' is not defined
