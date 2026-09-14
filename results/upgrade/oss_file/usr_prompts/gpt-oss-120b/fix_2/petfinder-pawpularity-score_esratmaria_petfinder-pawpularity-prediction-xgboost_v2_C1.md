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
optuna==4.5.0
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

20.51205

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import optuna
from tqdm import tqdm
from sklearn.model_selection import KFold
from functools import partial
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import lightgbm as lgb

pd.set_option("display.max_colwidth", None)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)




## === cell 2
train.head(2)




## === cell 3
test.head(2)




## === cell 4
sample_submission.head(2)




## === cell 5
print(train.shape, test.shape, sample_submission.shape)




## === cell 6
xgb_params = {
    "eval_metric": "rmse",
    "seed": 1337,
    "eta": 0.0123,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "tree_method": "hist",  # use CPU‑only method for compatibility
    "silent": 1,
}




## === cell 7
N_FOLDS = 4
FOLDS = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)




## === cell 8
import xgboost as xgb


def run_xgb(params, X_train, X_test):
    """
    X_train : DataFrame with columns Id, features..., Pawpularity
    X_test  : DataFrame with columns Id, features...
    """
    kf = FOLDS
    n_splits = N_FOLDS

    verbose_eval = 1000
    num_rounds = 60000
    early_stop = 500

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    X_test_feat = X_test.drop(["Id"], axis=1)

    i = 0
    for train_idx, valid_idx in kf.split(X_train):
        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[valid_idx].copy()

        y_tr = X_tr["Pawpularity"].values
        y_val = X_val["Pawpularity"].values

        X_tr = X_tr.drop(["Pawpularity", "Id"], axis=1)
        X_val = X_val.drop(["Pawpularity", "Id"], axis=1)

        d_train = xgb.DMatrix(
            data=X_tr._get_numeric_data(),
            label=y_tr,
            feature_names=list(X_tr.columns),
            enable_categorical=True,
        )

        d_valid = xgb.DMatrix(
            data=X_val._get_numeric_data(),
            label=y_val,
            feature_names=list(X_val.columns),
            enable_categorical=True,
        )

        watchlist = [(d_train, "train"), (d_valid, "valid")]
        model = xgb.train(
            dtrain=d_train,
            num_boost_round=num_rounds,
            evals=watchlist,
            early_stopping_rounds=early_stop,
            verbose_eval=verbose_eval,
            params=params,
        )

        valid_pred = model.predict(
            xgb.DMatrix(X_val._get_numeric_data(), feature_names=list(X_val.columns)),
            ntree_limit=model.best_ntree_limit,
        )

        test_pred = model.predict(
            xgb.DMatrix(
                X_test_feat._get_numeric_data(), feature_names=list(X_test_feat.columns)
            ),
            ntree_limit=model.best_ntree_limit,
        )

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    return model, oof_train, oof_test


model, oof_train, oof_test = run_xgb(xgb_params, train, test)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2638758511.py in <cell line: 0>()
     76 
     77 
---> 78 model, oof_train, oof_test = run_xgb(xgb_params, train, test)
     79 
     80 

/tmp/ipykernel_55/2638758511.py in run_xgb(params, X_train, X_test)
     58         valid_pred = model.predict(
     59             xgb.DMatrix(X_val._get_numeric_data(), feature_names=list(X_val.columns)),
---> 60             ntree_limit=model.best_ntree_limit,
     61         )
     62 

AttributeError: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 9
xgb_453_train_pred = oof_train
xgb_453_test_pred = np.mean(oof_test, axis=1)
xgb_453_train_pred.shape, xgb_453_test_pred.shape




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3830453354.py in <cell line: 0>()
----> 1 xgb_453_train_pred = oof_train
      2 xgb_453_test_pred = np.mean(oof_test, axis=1)
      3 xgb_453_train_pred.shape, xgb_453_test_pred.shape
      4 
      5 

NameError: name 'oof_train' is not defined

## === cell 10
xgb_453_test_pred




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2368966885.py in <cell line: 0>()
----> 1 xgb_453_test_pred
      2 
      3 

NameError: name 'xgb_453_test_pred' is not defined

## === cell 11
submission = pd.DataFrame(
    {"Id": sample_submission["Id"], "Pawpularity": xgb_453_test_pred}
)
submission.head(10)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1916823236.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"Id": sample_submission["Id"], "Pawpularity": xgb_453_test_pred}
      3 )
      4 submission.head(10)
      5 

NameError: name 'xgb_453_test_pred' is not defined

## === cell 12
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
