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
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

20.479608358664265

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import sys
sys.path.append("../input/pytorch-tabnet-zip")

## === cell 3
import os

import pandas as pd
import numpy as np
import datatable as dt
import warnings
import random
warnings.filterwarnings('ignore')
pd.set_option('max_columns',None)
from sklearn.metrics import mean_squared_error

from time import time
import pprint
import joblib
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pytorch_tabnet.tab_model import TabNetRegressor

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1312207489.py in <cell line: 0>()
      3 import pandas as pd
      4 import numpy as np
----> 5 import datatable as dt
      6 import warnings
      7 import random

ModuleNotFoundError: No module named 'datatable'

## === cell 5
FOLDER = "/kaggle/input/petfinder-pawpularity-score/"
TRAIN_FNAME = os.path.join(FOLDER, "train.csv")
TEST_FNAME = os.path.join(FOLDER, "test.csv")
SUBMISSION_FNAME = os.path.join(FOLDER, "sample_submission.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.1
MAX_EPOCHS_TABNET = 200

## === cell 6
def rmse_fn(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))

## === cell 8
train = pd.read_csv(TRAIN_FNAME)
test = pd.read_csv(TEST_FNAME)
submission = pd.read_csv(SUBMISSION_FNAME)

## === cell 9
train.shape, test.shape, submission.shape

## === cell 10
train.head()

## === cell 11
train.shape

## === cell 13
train = train.rename(columns={"Pawpularity": "target"})

## === cell 14
X_train, X_val, y_train, y_val = train_test_split(
    train.drop(["target", "Id"], axis=1),
    train.target,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=train.target
)
X_train = X_train.values
X_val = X_val.values
y_train = y_train.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3617693174.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(
      2     train.drop(["target", "Id"], axis=1),
      3     train.target,
      4     test_size=TEST_SIZE,
      5     random_state=RANDOM_STATE,

NameError: name 'train_test_split' is not defined

## === cell 16
from pytorch_tabnet.tab_model import TabNetRegressor


def fit_pred_tabnet(X_train, y_train, X_val, random_state=RANDOM_STATE, max_epochs_tabnet=MAX_EPOCHS_TABNET):
    model = TabNetRegressor(verbose=1,seed=random_state)
    print("Fit tabnet")
    model.fit(X_train=X_train, y_train=y_train,
               patience=5,max_epochs=max_epochs_tabnet,batch_size=256,
               eval_metric=['rmse'])
    print("Predict tabnet")
    pred_tabnet = model.predict(X_val)
    pred_tabnet = pred_tabnet.reshape(len(pred_tabnet))
    return pred_tabnet


pred_tabnet = fit_pred_tabnet(X_train, y_train, X_val)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2434191603.py in <cell line: 0>()
----> 1 from pytorch_tabnet.tab_model import TabNetRegressor
      2 
      3 
      4 def fit_pred_tabnet(X_train, y_train, X_val, random_state=RANDOM_STATE, max_epochs_tabnet=MAX_EPOCHS_TABNET):
      5     model = TabNetRegressor(verbose=1,seed=random_state)

ModuleNotFoundError: No module named 'pytorch_tabnet'

## === cell 17
rmse_tabnet = rmse_fn(pred_tabnet, y_val)
rmse_tabnet

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3793726030.py in <cell line: 0>()
----> 1 rmse_tabnet = rmse_fn(pred_tabnet, y_val)
      2 rmse_tabnet

NameError: name 'pred_tabnet' is not defined

## === cell 18
import xgboost as xgb


def fit_pred_xgb(X_train, y_train, X_val, random_state=RANDOM_STATE):
    xgb_regressor = xgb.XGBRegressor(seed=random_state, **{'n_estimators': 10000, 'max_depth': 7, 'learning_rate': 0.0022137388320075573})
    print("Fit XGB")
    xgb_regressor.fit(X_train, y_train)
    print("Predict XGB")
    pred_xgb = xgb_regressor.predict(X_val)
    return pred_xgb

pred_xgb = fit_pred_xgb(X_train, y_train, X_val)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2743407747.py in <cell line: 0>()
     10     return pred_xgb
     11 
---> 12 pred_xgb = fit_pred_xgb(X_train, y_train, X_val)

NameError: name 'X_train' is not defined

## === cell 19
rmse_xgb = rmse_fn(pred_xgb, y_val)
rmse_xgb

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618320324.py in <cell line: 0>()
----> 1 rmse_xgb = rmse_fn(pred_xgb, y_val)
      2 rmse_xgb

NameError: name 'pred_xgb' is not defined

## === cell 20
weights = list(np.arange(0, 1, 0.1))
weights

## === cell 21
def avg_tabnet_xgb(pred_tabnet, pred_xgb, w_tabnet):
    return np.average([pred_tabnet, pred_xgb], weights=[w_tabnet, 1-w_tabnet], axis=0)

## === cell 22
list_rmse = []

for w_tabnet in weights:
    pred_stack = avg_tabnet_xgb(pred_tabnet, pred_xgb, w_tabnet)
    rmse = rmse_fn(pred_stack, y_val)
    list_rmse.append(rmse)

print(list_rmse)
idx_best_w = list_rmse.index(min(list_rmse))
idx_best_w

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4163311329.py in <cell line: 0>()
      2 
      3 for w_tabnet in weights:
----> 4     pred_stack = avg_tabnet_xgb(pred_tabnet, pred_xgb, w_tabnet)
      5     rmse = rmse_fn(pred_stack, y_val)
      6     list_rmse.append(rmse)

NameError: name 'pred_tabnet' is not defined

## === cell 24
print(f"rmse_tabnet={rmse_tabnet}")
print(f"rmse_xgb={rmse_xgb}")

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/170148965.py in <cell line: 0>()
----> 1 print(f"rmse_tabnet={rmse_tabnet}")
      2 print(f"rmse_xgb={rmse_xgb}")

NameError: name 'rmse_tabnet' is not defined

## === cell 25
best_weight = weights[idx_best_w]
pred_stack = avg_tabnet_xgb(pred_tabnet, pred_xgb, best_weight)
rmse_weighted_avg = rmse_fn(pred_stack, y_val)

print(f"best weight is w={best_weight}")
print(f"weighted average ={rmse_weighted_avg}")

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3829511923.py in <cell line: 0>()
----> 1 best_weight = weights[idx_best_w]
      2 pred_stack = avg_tabnet_xgb(pred_tabnet, pred_xgb, best_weight)
      3 rmse_weighted_avg = rmse_fn(pred_stack, y_val)
      4 
      5 print(f"best weight is w={best_weight}")

NameError: name 'idx_best_w' is not defined

## === cell 27
def get_final_pred(X_train, y_train, X_val, w_tabnet):
    pred_tabnet = fit_pred_tabnet(X_train, y_train, X_val)
    pred_xgb = fit_pred_xgb(X_train, y_train, X_val)
    pred_stack = avg_tabnet_xgb(pred_tabnet, pred_xgb, w_tabnet)
    return pred_stack

## === cell 28
X_train = train.drop(["target", "Id"], axis=1).values
y_train = train.target.values.reshape(-1, 1)

X_test = test.drop(["Id"], axis=1).values

## === cell 29
predictions = get_final_pred(X_train=X_train, y_train=y_train, X_val=X_test, w_tabnet=best_weight)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1484166701.py in <cell line: 0>()
----> 1 predictions = get_final_pred(X_train=X_train, y_train=y_train, X_val=X_test, w_tabnet=best_weight)

NameError: name 'best_weight' is not defined

## === cell 30
test["Pawpularity"] = predictions
test[["Id", "Pawpularity"]].to_csv('submission.csv', index=False)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3627067832.py in <cell line: 0>()
----> 1 test["Pawpularity"] = predictions
      2 test[["Id", "Pawpularity"]].to_csv('submission.csv', index=False)

NameError: name 'predictions' is not defined
