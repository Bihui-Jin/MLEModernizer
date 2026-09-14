# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

20.47168

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn import model_selection
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

np.random.seed(42)



## === cell 1
base_input = Path("../input/petfinder-pawpularity-score")
if not base_input.exists():
    base_input = Path("/kaggle/input/petfinder-pawpularity-score")

train_data = pd.read_csv(base_input / "train.csv")
test_data = pd.read_csv(base_input / "test.csv")
sample = pd.read_csv(base_input / "sample_submission.csv")

print("Shape of input data :", train_data.shape, test_data.shape, sample.shape)
print("Identify the Null values :\n", train_data.isnull().sum())
train_data.sample(2)



## === cell 2
train_data["kfold"] = -1
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (_, valid_indicies) in enumerate(kfold.split(X=train_data)):
    train_data.loc[valid_indicies, "kfold"] = fold

print(train_data.kfold.value_counts())
train_data.to_csv("trainfold_10.csv", index=False)



## === cell 3
train = pd.read_csv("./trainfold_10.csv")

heat = train_data.select_dtypes(include=[np.number]).corr().round(5)

mask = np.zeros_like(heat, dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(16, 16))
ax = sns.heatmap(
    heat,
    annot=False,
    mask=mask,
    cmap="RdYlGn",
    annot_kws={"weight": "bold", "fontsize": 13},
)
ax.set_title("Feature correlation heatmap", fontsize=17)
plt.setp(
    ax.get_xticklabels(),
    rotation=90,
    ha="right",
    rotation_mode="anchor",
    weight="normal",
)
plt.setp(
    ax.get_yticklabels(),
    weight="normal",
    rotation_mode="anchor",
    rotation=0,
    ha="right",
)
plt.show()



## === cell 4
train.columns



## === cell 5
final_predictions = []
score = []

useful_features = [c for c in train.columns if c not in ("Id", "Pawpularity", "kfold")]
object_cols = [
    col for col in useful_features
]  # (kept as-is; original logic encodes all useful features)
test = test_data[useful_features]

for fold in range(10):
    xtrain = train[train.kfold != fold].reset_index(drop=True)
    xvalid = train[train.kfold == fold].reset_index(drop=True)
    xtest = test.copy()

    ytrain = xtrain.Pawpularity
    yvalid = xvalid.Pawpularity

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    ordinal_encoder = OrdinalEncoder()
    xtrain.loc[:, object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid.loc[:, object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest.loc[:, object_cols] = ordinal_encoder.transform(xtest[object_cols])

    xgb_params = {
        "learning_rate": 0.2113303692287,
        "subsample": 0.12703520389320402,
        "colsample_bytree": 0.2566392406542389,
        "max_depth": 2,
        "booster": "gbtree",
        "reg_lambda": 0.0005172374569093787,
        "reg_alpha": 0.001273145009879541,
        "random_state": 256,
        "n_estimators": 30000,
    }

    model = XGBRegressor(**xgb_params, tree_method="hist", predictor="auto")

    model.fit(
        xtrain,
        ytrain,
        early_stopping_rounds=100,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
    )

    preds_valid = model.predict(xvalid)
    test_pre = model.predict(xtest)
    final_predictions.append(test_pre)

    rms = mean_squared_error(yvalid, preds_valid, squared=False)
    score.append(rms)
    print(f"fold:{fold},rmse:{rms}")

print(np.mean(score), np.std(score))



## === cell 6
final_predictions = []
score = []

useful_features = [c for c in train.columns if c not in ("Id", "Pawpularity", "kfold")]
object_cols = [col for col in useful_features]
test = test_data[useful_features]

for fold in range(10):
    xtrain = train[train.kfold != fold].reset_index(drop=True)
    xvalid = train[train.kfold == fold].reset_index(drop=True)
    xtest = test.copy()

    ytrain = xtrain.Pawpularity
    yvalid = xvalid.Pawpularity

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    ordinal_encoder = OrdinalEncoder()
    xtrain.loc[:, object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid.loc[:, object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest.loc[:, object_cols] = ordinal_encoder.transform(xtest[object_cols])

    params_lgb = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "subsample": 0.95312,
        "learning_rate": 0.11635,
        "max_depth": 2,
        "feature_fraction": 0.2256038826485174,
        "bagging_fraction": 0.7705303688019942,
        "min_child_samples": 290,
        "reg_alpha": 14.68267919457715,
        "reg_lambda": 66.156,
        "max_bin": 772,
        "min_data_per_group": 177,
        "bagging_freq": 1,
        "cat_smooth": 96,
        "cat_l2": 17,
        "verbosity": -1,
        "seed": 42,
        "num_threads": 4,
        "colsample_bytree": 0.1107,
    }

    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_val = lgb.Dataset(xvalid, yvalid, reference=lgb_train)

    callbacks = [
        lgb.early_stopping(stopping_rounds=300, verbose=False),
        lgb.log_evaluation(period=1000),
    ]

    model = lgb.train(
        params=params_lgb,
        train_set=lgb_train,
        valid_sets=[lgb_val],
        callbacks=callbacks,
        num_boost_round=5000,
    )

    preds_valid = model.predict(xvalid, num_iteration=model.best_iteration)
    test_pre = model.predict(xtest, num_iteration=model.best_iteration)
    final_predictions.append(test_pre)

    rms = mean_squared_error(yvalid, preds_valid, squared=False)
    score.append(rms)
    print(f"fold:{fold},rmse:{rms}")

print(np.mean(score), np.std(score))



## === cell 7
final_predictions = []
score = []

useful_features = [c for c in train.columns if c not in ("Id", "Pawpularity", "kfold")]
object_cols = [col for col in useful_features]
test = test_data[useful_features]

for fold in range(10):
    xtrain = train[train.kfold != fold].reset_index(drop=True)
    xvalid = train[train.kfold == fold].reset_index(drop=True)
    xtest = test.copy()

    ytrain = xtrain.Pawpularity
    yvalid = xvalid.Pawpularity

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    ordinal_encoder = OrdinalEncoder()
    xtrain.loc[:, object_cols] = ordinal_encoder.fit_transform(xtrain[object_cols])
    xvalid.loc[:, object_cols] = ordinal_encoder.transform(xvalid[object_cols])
    xtest.loc[:, object_cols] = ordinal_encoder.transform(xtest[object_cols])

    catpara = {
        "subsample": 0.95312,
        "learning_rate": 0.0011356,
        "max_depth": 6,
        "min_data_in_leaf": 77,
        "random_state": 42,
        "n_estimators": 8000,
        "rsm": 0.5,
        "l2_leaf_reg": 0.02247766515106271,
        "loss_function": "RMSE",
        "verbose": 1000,
    }

    model = CatBoostRegressor(**catpara)

    model.fit(
        xtrain,
        ytrain,
        eval_set=(xvalid, yvalid),
        early_stopping_rounds=100,
        verbose=1000,
    )

    preds_valid = model.predict(xvalid)
    test_pre = model.predict(xtest)
    final_predictions.append(test_pre)

    rms = mean_squared_error(yvalid, preds_valid, squared=False)
    score.append(rms)
    print(f"fold:{fold},rmse:{rms}")

print(np.mean(score), np.std(score))



## === cell 8
preds = np.mean(np.column_stack(final_predictions), axis=1)

preds = np.clip(preds, 0, 100)

sample["Pawpularity"] = preds
sample.to_csv("submission.csv", index=False)

print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
print("success")
