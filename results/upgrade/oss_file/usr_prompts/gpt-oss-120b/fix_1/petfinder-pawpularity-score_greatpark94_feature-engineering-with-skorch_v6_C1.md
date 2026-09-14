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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

20.61428634069452

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install ../input/torchnotebook/skorch-0.10.0-py3-none-any.whl

## === cell 1
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import cv2


from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from skorch import NeuralNetRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.inspection import permutation_importance

from torch import optim
from torch.optim.lr_scheduler import CyclicLR

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm

random_seed = 2021
num_folds = 5

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1128470357.py in <cell line: 0>()
     26 
     27 # neuralnet
---> 28 from skorch import NeuralNetRegressor
     29 from sklearn.base import BaseEstimator, TransformerMixin
     30 

ModuleNotFoundError: No module named 'skorch'

## === cell 3
df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
df.head()

## === cell 4
for col in df.columns:
    print(col, df[col].nunique())

## === cell 6
def load_img(img_file_path):
    img = cv2.imread(img_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

## === cell 7
img_file_paths = [f'/kaggle/input/petfinder-pawpularity-score/train/{img_filename}.jpg' for img_filename in df['Id']]

## === cell 9
img_stg1_df = df.copy()

## === cell 10
def get_shape_info(img, img_file_path):
    return pd.Series([img.shape[0], img.shape[1], img.shape[0] / img.shape[1], os.path.getsize(img_file_path)])

def add_shape_info(img_file_path):
    img = load_img(img_file_path)
    return get_shape_info(img, img_file_path)

## === cell 11
shape_cols = ['width', 'height', 'w_h_ratio', 'size']

## === cell 12
img_stg1_df[shape_cols] =  [add_shape_info(img_file_path) for img_file_path in tqdm(img_file_paths)]

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1582862870.py in <cell line: 0>()
----> 1 img_stg1_df[shape_cols] =  [add_shape_info(img_file_path) for img_file_path in tqdm(img_file_paths)]

NameError: name 'tqdm' is not defined

## === cell 14
img_stg2_df = df.copy()

## === cell 15
def get_stats(data):
    return np.min(data), np.max(data), np.mean(data), np.std(data)

def get_rgb_info(img):
    r = img[0]
    g = img[1]
    b = img[2]
    
    r_stats =  get_stats(r)
    g_stats =  get_stats(g)
    b_stats =  get_stats(b)
    
    return np.concatenate([r_stats, g_stats, b_stats])

def add_rgb_info(img_file_path):
    img = load_img(img_file_path)
    return get_rgb_info(img)

## === cell 16
rgb_cols = np.concatenate([[f'{prefix}_{postfix}' for postfix in ['min', 'max', 'mean', 'std']] for prefix in ['r', 'g', 'b']])

## === cell 17
img_stg2_df[rgb_cols] =  [add_rgb_info(img_file_path) for img_file_path in tqdm(img_file_paths)]

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1592378056.py in <cell line: 0>()
----> 1 img_stg2_df[rgb_cols] =  [add_rgb_info(img_file_path) for img_file_path in tqdm(img_file_paths)]

NameError: name 'tqdm' is not defined

## === cell 19
img_stg3_df = df.copy()

## === cell 20
img_stg3_df[shape_cols] = img_stg1_df[shape_cols]
img_stg3_df[rgb_cols] = img_stg2_df[rgb_cols]

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/452232233.py in <cell line: 0>()
----> 1 img_stg3_df[shape_cols] = img_stg1_df[shape_cols]
      2 img_stg3_df[rgb_cols] = img_stg2_df[rgb_cols]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['width', 'height', 'w_h_ratio', 'size'], dtype='object')] are in the [columns]"

## === cell 21
img_stg3_df.head()

## === cell 23
def split_data(df):
    X = df.drop(['Id', 'Pawpularity'], axis=1)
    y = df['Pawpularity']
    tmp_label = [Pawpularity // 4 for Pawpularity in df['Pawpularity']]

    skf = StratifiedKFold(n_splits=num_folds, random_state=random_seed, shuffle=True)

    data_per_fold = dict()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, tmp_label)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        data_per_fold[fold] = {}
        data_per_fold[fold]['X_train'] = X_train
        data_per_fold[fold]['X_test'] = X_test
        data_per_fold[fold]['y_train'] = y_train
        data_per_fold[fold]['y_test'] = y_test
        
    return data_per_fold

## === cell 25
class RegressorModule(nn.Module): 
    def __init__(self, num_input):
        super(RegressorModule, self).__init__()
        
        self.sequence = nn.Sequential(nn.Linear(num_input, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 12),
                                   nn.ReLU(),
                                   nn.Linear(12, 8),
                                   nn.ReLU(),
                                   nn.Linear(8, 1),
                                   )
        
    def forward(self, x):
        return self.sequence(x)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3005485471.py in <cell line: 0>()
----> 1 class RegressorModule(nn.Module):
      2     def __init__(self, num_input):
      3         super(RegressorModule, self).__init__()
      4 
      5         self.sequence = nn.Sequential(nn.Linear(num_input, 16),

NameError: name 'nn' is not defined

## === cell 26
def get_preprocessor(features, num_features, degree):  
    num_features.sort()
    num_transformer = Pipeline(
        steps=[
            ("polynomial", PolynomialFeatures(degree=degree)), 
            ("scaler", RobustScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[('num', num_transformer, num_features)],
        remainder='passthrough'
    )
    
    return preprocessor

## === cell 27
class FloatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, x):
        return np.array(x, dtype=np.float32)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/491959821.py in <cell line: 0>()
----> 1 class FloatTransformer(BaseEstimator, TransformerMixin):
      2     def __init__(self):
      3         pass
      4     def fit(self, X, y=None):
      5         return self

NameError: name 'BaseEstimator' is not defined

## === cell 28
def get_model(method, num_features):
    if method == 'lr':
        model = LinearRegression(fit_intercept=True)
    elif method == 'rf':
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=3,
            min_samples_leaf=2,   
        )
    elif method == 'nn':        
        model = NeuralNetRegressor(
            RegressorModule(num_input=num_features),
            max_epochs=100, verbose=0,
            warm_start=True,
            criterion=RMSELoss,
            optimizer = optim.AdamW,
            optimizer__lr = 0.001
        )
        
    return model

## === cell 29
def build_pipeline(features, method, degree=1, num_features=None):    
    if num_features is None:
        model = get_model(method, len(features))
        return Pipeline(
            [
                ('float64to32', FloatTransformer()),
                ('model', model)
            ]
        )
    else:
        preprocessor = get_preprocessor(features, num_features, degree)
        model = get_model(method, len(features) + 1)

        return Pipeline(
            [
                ('preprocessor', preprocessor),
                ('float64to32', FloatTransformer()),
                ('model', model)
            ]
        )

## === cell 31
def rmse_score(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

## === cell 33
def run(df, num_features=None):
    data_per_fold = split_data(df)
    features = data_per_fold[0]['X_train'].columns.tolist()

    model_lr = build_pipeline(features, 'lr', num_features=num_features)
    model_rf = build_pipeline(features, 'rf', num_features=num_features)
    model_nn = build_pipeline(features, 'nn', num_features=num_features)
    
    for fold in range(num_folds):
        print('fold',fold)

        data = data_per_fold[fold]
        X_train = data_per_fold[fold]['X_train']
        X_test = data_per_fold[fold]['X_test']
        y_train = data_per_fold[fold]['y_train']
        y_test = data_per_fold[fold]['y_test']

        model_lr.fit(X_train, y_train)
        pred_lr = model_lr.predict(X_test)
        rmse_lr = rmse_score(pred_lr, y_test.to_numpy())
        print('lr: ', rmse_lr)

        model_rf.fit(X_train, y_train)
        pred_rf = model_rf.predict(X_test)
        rmse_rf = rmse_score(pred_rf, y_test.to_numpy())
        print('rf: ', rmse_rf)

        model_nn.fit(X_train, y_train.astype(np.float32).values.reshape(-1, 1))
        pred_nn = model_nn.predict(X_test)
        rmse_nn = rmse_score(pred_nn, y_test.to_numpy())
        print('nn: ', rmse_nn)

        rmse_ensemble = rmse_score((pred_lr + pred_rf + pred_nn) / 3, y_test.to_numpy())
        print('ensemble: ', rmse_ensemble)

        pi_lr = permutation_importance(model_lr, X_test, y_test, n_repeats=30, random_state=random_seed)
        pi_rf = permutation_importance(model_rf, X_test, y_test, n_repeats=30, random_state=random_seed)
        pi_nn = permutation_importance(model_nn, X_test, y_test, n_repeats=30, random_state=random_seed)

        fig, axs = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True, sharey=True)

        for ax, pi, title in zip(axs, [pi_lr, pi_rf, pi_nn], ["Linear Reg.", "Random Forest", "Neural Net"]):
            ax.barh(X_test.columns, pi.importances_mean, xerr=pi.importances_std, color="orange")
            ax.invert_yaxis()
            ax.set_xlim(0, )
            ax.set_title(title, pad=16)

        plt.show()

## === cell 39
final_df = df.drop(['Subject Focus', 'Action', 'Collage'], axis=1)
final_df[shape_cols] =  [add_shape_info(img_file_path) for img_file_path in img_file_paths]
final_df = final_df.drop(['width', 'height'], axis=1)

features = final_df.columns.tolist()

## === cell 41
X_train = final_df.drop(['Id', 'Pawpularity'], axis=1)
y_train = final_df['Pawpularity']

features = X_train.columns.tolist()
num_features = ['w_h_ratio', 'size'] 

## === cell 43
test_df = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')

test_img_file_paths = [f'/kaggle/input/petfinder-pawpularity-score/test/{img_filename}.jpg' for img_filename in test_df['Id']]

X_test = test_df.drop(['Id', 'Subject Focus', 'Action', 'Collage'], axis=1)
X_test[shape_cols] =  [add_shape_info(img_file_path) for img_file_path in test_img_file_paths]
X_test = X_test.drop(['width', 'height'], axis=1)

## === cell 44
model_lr = build_pipeline(features, 'lr', num_features=num_features)
model_rf = build_pipeline(features, 'rf', num_features=num_features)
model_nn = build_pipeline(features, 'nn', num_features=num_features)


model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)

model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

model_nn.fit(X_train, y_train.astype(np.float32).values.reshape(-1, 1))
pred_nn = model_nn.predict(X_test).squeeze(1)

pred_ensemble = (pred_lr + pred_rf + pred_nn) / 3

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2533522483.py in <cell line: 0>()
----> 1 model_lr = build_pipeline(features, 'lr', num_features=num_features)
      2 model_rf = build_pipeline(features, 'rf', num_features=num_features)
      3 model_nn = build_pipeline(features, 'nn', num_features=num_features)
      4 
      5 

/tmp/ipykernel_11/308682376.py in build_pipeline(features, method, degree, num_features)
     15             [
     16                 ('preprocessor', preprocessor),
---> 17                 ('float64to32', FloatTransformer()),
     18                 ('model', model)
     19             ]

NameError: name 'FloatTransformer' is not defined

## === cell 45
submission = test_df[['Id']].copy()
submission['Pawpularity'] = pred_ensemble
submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344291232.py in <cell line: 0>()
      1 submission = test_df[['Id']].copy()
----> 2 submission['Pawpularity'] = pred_ensemble
      3 submission.to_csv('submission.csv', index=False)

NameError: name 'pred_ensemble' is not defined

## === cell 46
submission
