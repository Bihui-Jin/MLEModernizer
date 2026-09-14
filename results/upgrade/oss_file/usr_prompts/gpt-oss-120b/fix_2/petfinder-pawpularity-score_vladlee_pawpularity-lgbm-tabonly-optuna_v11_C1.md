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
optuna==4.5.0
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

20.49309

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from IPython import display as ipd
import gc, time, logging, re, math
import numpy as np
import pandas as pd
import os, warnings, random
from tqdm import tqdm
import lightgbm as lgb

print(lgb.__version__)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import optuna




## === cell 1
def seeding(SEED, use_tf=False):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = str(SEED)
    if use_tf:
        import tensorflow as tf

        tf.random.set_seed(SEED)
    print("seeding done!!!")




## === cell 2
RANDOM_SEED = 42
DEBUG = True
TUNING = False
DATA_PATH = "/kaggle/input/petfinder-pawpularity-score/"

train_df = pd.read_csv(DATA_PATH + "train.csv")
test_df = pd.read_csv(DATA_PATH + "test.csv")
submission_template = pd.read_csv(DATA_PATH + "sample_submission.csv")
seeding(RANDOM_SEED)



## === cell 3
FEATURES = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
target = train_df["Pawpularity"]

train = train_df[FEATURES].copy()
test = test_df[FEATURES].copy()

CAT_FEATURES = FEATURES.copy()


def add_cross_features(df):
    for f1 in FEATURES:
        for f2 in FEATURES:
            if f1 != f2:
                name2 = f"{f1}-{f2}"
                df[name2] = df[f1].astype(str) + "_" + df[f2].astype(str)
                CAT_FEATURES.append(name2)
                for f3 in FEATURES:
                    if f3 not in (f1, f2):
                        name3 = f"{f1}-{f2}-{f3}"
                        df[name3] = (
                            df[f1].astype(str)
                            + "_"
                            + df[f2].astype(str)
                            + "_"
                            + df[f3].astype(str)
                        )
                        CAT_FEATURES.append(name3)
    return df


train = add_cross_features(train)
test = add_cross_features(test)

print("train shape after cross features:", train.shape)
print("test shape after cross features:", test.shape)



## === cell 4
X = train.copy()
print("Feature matrix shape X:", X.shape)



## === cell 5
print("train shape X:", X.shape)
print("test shape:", test.shape)
print("target shape:", target.shape)



## === cell 6
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100


def objective(trial, X, y):
    param_grid = {
        "verbosity": -1,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 50, 2000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 2000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, shuffle=True
    )
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES)
    valid_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_FEATURES)

    model = lgb.train(
        param_grid,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_set],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        callbacks=[lgb.log_evaluation(VERBOSE_EVAL)],
    )

    pred = model.predict(X_val)
    return mean_squared_error(pred, y_val)




## === cell 7
if TUNING:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, target), n_trials=200)
    print("Best trial score:", study.best_value)
    print("Best parameters:", study.best_params)




## === cell 8
def run_train(
    X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds
):
    models = []
    oof_predicted = []
    evals_results = {}
    folds = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
        print(f"Fold {fold_n + 1} started")
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES)
        valid_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_FEATURES)

        model = lgb.train(
            run_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[valid_set],
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[lgb.log_evaluation(verbose_eval)],
        )
        oof_predicted.append(model.predict(X_val))
        models.append(model)
    return models, oof_predicted, evals_results




## === cell 9
TOTAL_SPLITS = 3
NUM_BOOST_ROUND = 4000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

run_params = {
    "verbosity": -1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.09802528890284778,
    "num_leaves": 1200,
    "max_depth": 7,
    "min_data_in_leaf": 200,
    "max_bin": 257,
    "lambda_l1": 100,
    "lambda_l2": 10,
    "feature_fraction": 0.5291871221439385,
    "bagging_fraction": 0.4529000420532695,
    "bagging_freq": 7,
    "min_child_samples": 7,
}

models, oof_predicted, evals_results = run_train(
    X,
    target,
    run_params,
    TOTAL_SPLITS,
    NUM_BOOST_ROUND,
    VERBOSE_EVAL,
    EARLY_STOPPING_ROUNDS,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/897376755.py in <cell line: 0>()
     22 }
     23 
---> 24 models, oof_predicted, evals_results = run_train(
     25     X,
     26     target,

/tmp/ipykernel_11/1513856821.py in run_train(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds)
     14         valid_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_FEATURES)
     15 
---> 16         model = lgb.train(
     17             run_params,
     18             train_set,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 10
predicted = [model.predict(test) for model in models]
avg_preds = np.mean(predicted, axis=0)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2516378980.py in <cell line: 0>()
----> 1 predicted = [model.predict(test) for model in models]
      2 avg_preds = np.mean(predicted, axis=0)
      3 

NameError: name 'models' is not defined

## === cell 11
submission = pd.DataFrame({"Id": submission_template["Id"], "Pawpularity": avg_preds})
submission.to_csv("submission.csv", index=False, float_format="%.6f")
submission.head(20)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1500992916.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"Id": submission_template["Id"], "Pawpularity": avg_preds})
      2 submission.to_csv("submission.csv", index=False, float_format="%.6f")
      3 submission.head(20)

NameError: name 'avg_preds' is not defined
