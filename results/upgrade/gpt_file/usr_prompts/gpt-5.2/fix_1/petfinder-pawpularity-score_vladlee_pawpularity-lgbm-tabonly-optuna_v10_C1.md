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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

20.48745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0


from IPython import display as ipd

import gc
import time
import logging
import re, math

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

import random
from tqdm import tqdm

import lightgbm as lgb

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import roc_curve, auc, accuracy_score, cohen_kappa_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix


## === cell 1
def seeding(SEED, use_tf=False):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    if use_tf:
        tf.random.set_seed(SEED)
    print('seeding done!!!')


## === cell 2
RANDOM_SEED = 42
DEBUG = True
HYPER_TUNING = False

DATA_PATH = '/kaggle/input/petfinder-pawpularity-score/'

train = pd.read_csv(DATA_PATH+'train.csv')
test = pd.read_csv(DATA_PATH+'test.csv')
submission = pd.read_csv(DATA_PATH+'sample_submission.csv')

seeding(RANDOM_SEED)
display(train.head(2))


## === cell 3
FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
target = train.Pawpularity
train = train[FEATURES]
test = test[FEATURES]

print('train shape:',train.shape)
print('test shape:',test.shape)
print('target shape:',target.shape)


## === cell 4
CAT_FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
ALL_CAT_FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']

def add_cross_features(df):
    for feature1 in CAT_FEATURES:    
        for feature2 in CAT_FEATURES:
            if feature1 != feature2:
                x2_feature_name = f'{feature1}-{feature2}'
                df[x2_feature_name] = df[feature1].astype(str) + '_' + df[feature2].astype(str)
                ALL_CAT_FEATURES.append(x2_feature_name)
                for feature3 in CAT_FEATURES:
                    if feature3 != feature2 and feature3 != feature1:
                        x3_feature_name = f'{feature1}-{feature2}-{feature3}'
                        df[x3_feature_name] = df[feature1].astype(str) + '_' + df[feature2].astype(str) + '_' + df[feature3].astype(str)
                        ALL_CAT_FEATURES.append(x3_feature_name)
    return df
                
                
train = add_cross_features(train)
test = add_cross_features(test)

print('train shape:',train.shape)
print('test shape:',test.shape)


## === cell 5
print('train shape:',train.shape)
print('test shape:',test.shape)
print('target shape:',target.shape)


## === cell 6
for c in train.columns:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')


## === cell 7
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

pca = PCA(n_components=0.999)
pca.fit_transform(train_scaled)
pca.transform(test_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


## === cell 8

n_components = len(pca.explained_variance_ratio_)
pca = PCA(n_components=n_components)
pca_train = pca.fit_transform(train_scaled)
pca_test = pca.transform(test_scaled)

X = pd.DataFrame(pca_train, columns=['PCA%i' % i for i in range(n_components)], index=train.index)


## === cell 9
def run_train(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds ):
    models = []
    oof_predicted = []
    evals_results = {}  # to record eval results for plotting
    folds = StratifiedKFold(n_splits=splits)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print(f'Fold {fold_n+1} started')
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = lgb.train(
            run_params, valid_names=["train", "valid"], 
            train_set=lgb.Dataset(X_train, y_train ), 
            num_boost_round = num_boost_round,
            valid_sets = [lgb.Dataset(X_valid, y_valid)],
            verbose_eval = verbose_eval,
            evals_result=evals_results,
            early_stopping_rounds = early_stopping_rounds,
        )
            
        oof_predicted.append(model.predict(X_valid))
        models.append(model)
    return models, oof_predicted, evals_results


## === cell 10
LEARNING_RATE = 0.001
MAX_DEPTH = -1
NUM_LEAVES = 31
TOTAL_SPLITS = 4
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50

run_params = {
    'verbose': -1, 
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'rmse'},
    'learning_rate': LEARNING_RATE, 
    'num_leaves': NUM_LEAVES, 
    'max_depth': MAX_DEPTH, 
}

models, oof_predicted, evals_results = run_train(X, target, run_params, TOTAL_SPLITS, 
                                                NUM_BOOST_ROUND, VERBOSE_EVAL, EARLY_STOPPING_ROUNDS)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3264165914.py in <cell line: 0>()
     17 }
     18 
---> 19 models, oof_predicted, evals_results = run_train(X, target, run_params, TOTAL_SPLITS, 
     20                                                 NUM_BOOST_ROUND, VERBOSE_EVAL, EARLY_STOPPING_ROUNDS)

/tmp/ipykernel_11/1193488240.py in run_train(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds)
      8         X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
      9         y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
---> 10         model = lgb.train(
     11             run_params, valid_names=["train", "valid"],
     12             train_set=lgb.Dataset(X_train, y_train ),

TypeError: train() got an unexpected keyword argument 'verbose_eval'

## === cell 11
predicted = []
for model in models:
    predicted.append(model.predict(pca_test))

avg_preds = np.mean(predicted, axis=0)    


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2625863610.py in <cell line: 0>()
      1 predicted = []
----> 2 for model in models:
      3     predicted.append(model.predict(pca_test))
      4 
      5 avg_preds = np.mean(predicted, axis=0)

NameError: name 'models' is not defined

## === cell 12
submission = pd.DataFrame({'Id': submission['Id'], 'Pawpularity': avg_preds})
submission.to_csv('submission.csv', index=False, float_format='%.6f')
submission.head(20)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2755075184.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'Id': submission['Id'], 'Pawpularity': avg_preds})
      2 submission.to_csv('submission.csv', index=False, float_format='%.6f')
      3 submission.head(20)

NameError: name 'avg_preds' is not defined
