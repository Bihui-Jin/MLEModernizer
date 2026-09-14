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

20.45862

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



## === cell 2
import os
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset




warnings.filterwarnings("ignore")


## === cell 3
import seaborn as sns
from pathlib import Path


## === cell 4
df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")


## === cell 5
Train_image_dir = '../input/petfinder-pawpularity-score/train/'
Test_image_dir  = '../input/petfinder-pawpularity-score/test/'
Train_f = '../input/petfinder-pawpularity-score/train.csv'
Test_f  = '../input/petfinder-pawpularity-score/test.csv'


## === cell 6
Train_pd = pd.read_csv(Train_f)
Test_pd  = pd.read_csv(Test_f)


## === cell 7
Train_pd.head()


## === cell 8
print('train_f dimensions: ', Train_pd.shape)
print('train_f column names: ', Train_pd.columns.values.tolist())

print('')

print('test_f dimensions: ',Test_pd.shape)
print('test_f column names: ', Test_pd.columns.values.tolist())


## === cell 9
sns.set(rc={'figure.figsize':(15,5)})
fig = plt.figure()
sns.histplot(data=Train_pd, x='Pawpularity', bins=100)
plt.axvline(Train_pd['Pawpularity'].mean(), c='red', ls='-', lw=3, label='Mean Pawpularity')
plt.axvline(Train_pd['Pawpularity'].median(),c='blue',ls='-',lw=3, label='Median Pawpularity')
plt.title('Distribution of Pawpularity Scores', fontsize=20, fontweight='bold')
plt.legend()
plt.show()


## === cell 10
boxplot = Train_pd.boxplot(column='Pawpularity')


## === cell 11
Train_pd[['Pawpularity']].describe()


## === cell 12
fig, ax = plt.subplots(1,2)
sns.boxplot(data=Train_pd, x='Eyes', y='Pawpularity', ax=ax[0])
sns.histplot(Train_pd, x="Pawpularity", hue="Eyes", kde=True, ax=ax[1])
plt.suptitle("Eyes", fontsize=20, fontweight='bold')
fig.show()


## === cell 13

feature_variables = Train_pd.columns.values.tolist()

for variable in feature_variables[1:-1]:
    fig, ax = plt.subplots(1,2)
    sns.boxplot(data=Train_pd, x=variable, y='Pawpularity', ax=ax[0])
    sns.histplot(Train_pd, x="Pawpularity", hue=variable, kde=True, ax=ax[1])
    plt.suptitle(variable, fontsize=20, fontweight='bold')
    fig.show()


## === cell 14
print(Train_pd['Pawpularity'].nunique())
print(Train_pd['Pawpularity'].value_counts(sort = True, ascending = True, bins=10))


## === cell 15
from collections import Counter


## === cell 16
unique_image_files = np.unique(Train_pd.Id.values)
print("The number of unique file names : {}".format(len(unique_image_files)))


## === cell 17
from keras.preprocessing.image import load_img, img_to_array

npic = 25
npix = 224
target_size = (npix,npix,3)

count = 1
fig = plt.figure(figsize=(10,20))
for  img in unique_image_files[:npic]:
    filename = Train_image_dir + img + '.jpg'
    captions = list(Train_pd["Pawpularity"].loc[Train_pd["Id"]==img].values)
    image_load = load_img(filename, target_size=target_size)
    plt.subplot(5, 5, count)
    plt.imshow(image_load)
    plt.axis('off')
    plt.title(f'Pawpularity: {captions}')
    count += 1


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 18
from fastai.vision.all import *


## === cell 19
dataset_path = Path('../input/petfinder-pawpularity-score/')
dataset_path.ls()


## === cell 20
Train_pd['path'] = Train_pd['Id'].map(lambda x:str(dataset_path/'train'/x)+'.jpg')


## === cell 21
Train_pd.head()


## === cell 22
Train_pd['norm_score'] = Train_pd['Pawpularity']/100


## === cell 23
dls = ImageDataLoaders.from_df(Train_pd, #pass in train DataFrame
                               valid_pct=0.2, #80-20 train-validation random split
                               seed=999, #seed
                               fn_col='path', #filename/path is in the second column of the DataFrame
                               label_col='norm_score', #label is in the first column of the DataFrame
                               y_block=RegressionBlock, #The type of target
                               bs=32, #pass in batch size
                               num_workers=8,
                               item_tfms=Resize(224), #pass in item_tfms
                               batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])) #pass in batch_tfms


## === cell 24
dls.show_batch()


## === cell 26
num_bins = int(np.floor(1+np.log2(len(Train_pd))))
num_bins


## === cell 27
Train_pd['bins'] = pd.cut(Train_pd['norm_score'], bins=num_bins, labels=False)
Train_pd['bins'].hist()


## === cell 28
Test_pd = pd.read_csv(Test_f)
Test_pd.head()


## === cell 29
Test_pd['Pawpularity'] = [1]*len(Test_pd)
Test_pd['path'] = Test_pd['Id'].map(lambda x:str(dataset_path/'test'/x)+'.jpg')
Train_pd['norm_score'] = Train_pd['Pawpularity']/100


## === cell 31
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
from tqdm import trange
from colorama import Fore


## === cell 32
en = ElasticNet(random_state=0)
gbr = GradientBoostingRegressor(random_state=0)
VR_model = VotingRegressor([('en', en),('gbr', gbr)], n_jobs=-1)


## === cell 33
r2_train = 0; r2_val = 0
rmse_train = 0; rmse_val = 0


## === cell 34
def create_folds_regression(data, target="target", num_splits = 5): 
    """
    Helper function to create folds
    
    """
    data["kfold"] = -1 
    data = data.sample(frac=1).reset_index(drop=True)
    
    num_bins = int(1 + np.log2(len(data))) 

    data.loc[:, "bins"] = pd.cut(data[target], bins=num_bins, labels=False) 
    
    kf = StratifiedKFold(n_splits=num_splits)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)): 
        data.loc[v_, 'kfold'] = f
        
    data = data.drop(["bins"], axis = 1)         
    return data 


data_df = create_folds_regression(Train_pd, target = 'Pawpularity', num_splits = 5)
data_df.kfold.value_counts()


## === cell 35
req_cols = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
       'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]


## === cell 36
def rmse_score(y_label, y_preds):
    """
    Gives RMSE score
    """
    return np.sqrt(mean_squared_error(y_label, y_preds))


## === cell 37
model = VR_model
for fold in trange(5, desc = "Models are training...", bar_format="{l_bar}%s{bar:50}%s{r_bar}" % (Fore.CYAN, Fore.RESET), position = 0, leave = True):
    print(fold)
    Train_pd = data_df.loc[data_df.kfold != fold].reset_index(drop = True)
    val_df = data_df.loc[data_df.kfold == fold].reset_index(drop = True)

    train_X = Train_pd[req_cols]; train_Y = Train_pd["Pawpularity"]
    val_X = val_df[req_cols]; val_Y = val_df["Pawpularity"]
    val_X.head()
    model.fit(train_X, train_Y)

    Y_train_preds = model.predict(train_X)
    Y_val_preds = model.predict(val_X)

    r2_train += r2_score(train_Y, Y_train_preds)
    r2_val += r2_score(val_Y, Y_val_preds)

    rmse_train += rmse_score(train_Y, Y_train_preds)
    rmse_val += rmse_score(val_Y, Y_val_preds)

print(f"Avg R2 Train Score : {r2_train/5}")
print(f"Avg R2 Val Score : {r2_val/5}")
print(f"Avg RSME Train Score : {rmse_train/5}")
print(f"Avg RSME Val Score : {rmse_val/5}")


## === cell 38
Test_pd


## === cell 39
test_X = Test_pd[req_cols]

model_preds = model.predict(test_X)
Test_pd["Pawpularity"] = model_preds

submission = Test_pd[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index = False)
data_df.to_csv("data.csv", index = False)
Test_pd.to_csv("test.csv", index = False)
submission
