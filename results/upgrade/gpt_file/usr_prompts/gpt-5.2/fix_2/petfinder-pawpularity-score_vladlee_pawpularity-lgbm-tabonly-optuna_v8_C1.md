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
wandb==0.21.0
ydata-profiling==4.17.0

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

20.49187

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

import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import random
from tqdm import tqdm

import lightgbm as lgb

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GroupKFold

from sklearn.metrics import mean_absolute_error, mean_squared_error




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
WANDB = False

if WANDB:
    import wandb
    from wandb.lightgbm import wandb_callback
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=api_key)



## === cell 3
RANDOM_SEED = 42
DEBUG = True
HYPER_TUNING = False
PROFILE = False

DATA_PATH = "/kaggle/input/petfinder-pawpularity-score/"
IMAGE_SIZE = [128, 128]

seeding(RANDOM_SEED)




## === cell 4
def add_image_info(file_name, dir_name, limit=-1):
    df = pd.read_csv(DATA_PATH + file_name)
    if limit > 0:
        df = df.drop(labels=range(limit, len(df)), axis=0)
    df["image_path"] = DATA_PATH + dir_name + "/" + df.Id + ".jpg"
    tqdm.pandas(desc=dir_name)
    return df


train = add_image_info("train.csv", "train")
display(train.head(2))

test = add_image_info("test.csv", "test")
display(test.head(2))

sample_submission = pd.read_csv(DATA_PATH + "sample_submission.csv")



## === cell 5
print("train_files:", train.shape[0])
print("test_files:", test.shape[0])



## === cell 6
from pandas_profiling import ProfileReport

if PROFILE:
    train_profile = ProfileReport(train, title="Train Data")
    test_profile = ProfileReport(test, title="Test Data")
    display(train_profile)



## === cell 7
train.drop_duplicates(inplace=True)
gc.collect()



## === cell 8
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
target = train.Pawpularity
train = train[FEATURES]
test = test[FEATURES]

print("train shape:", train.shape)
print("test shape:", test.shape)
print("target shape:", target.shape)



## === cell 9
CAT_FEATURES = [
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

for c in train.columns:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")



## === cell 10
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

pca = PCA(n_components=0.95)
_ = pca.fit_transform(train_scaled)
_ = pca.transform(test_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")



## === cell 11
pca = PCA(n_components=2)
X_pca_train = pca.fit_transform(train_scaled)
X_pca_test = pca.transform(test_scaled)

f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax1.scatter(X_pca_train[:, 0], X_pca_train[:, 1], c=target, cmap="rainbow")
ax2.scatter(X_pca_test[:, 0], X_pca_test[:, 1], cmap="rainbow")
plt.show()




## === cell 12
def run_train(
    X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds
):
    models = []
    oof_predicted = []
    evals_results = {}

    folds = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f"Fold {fold_n+1} started")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=verbose_eval),
        ]
        if WANDB:
            callbacks = callbacks + [wandb_callback()]

        model = lgb.train(
            params=run_params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            evals_result=evals_results,
            callbacks=callbacks,
        )

        oof_predicted.append(model.predict(X_valid, num_iteration=model.best_iteration))
        models.append(model)

    return models, oof_predicted, evals_results




## === cell 13
LEARNING_RATE = 0.001
MAX_DEPTH = -1
NUM_LEAVES = 31
TOTAL_SPLITS = 4
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50

run_params = {
    "verbose": -1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": LEARNING_RATE,
    "num_leaves": NUM_LEAVES,
    "max_depth": MAX_DEPTH,
}

if WANDB:
    wandb.init(
        project="Pawpularity-LGBM", settings=wandb.Settings(_save_requirements=False)
    )

models, oof_predicted, evals_results = run_train(
    train,
    target,
    run_params,
    TOTAL_SPLITS,
    NUM_BOOST_ROUND,
    VERBOSE_EVAL,
    EARLY_STOPPING_ROUNDS,
)

if WANDB:
    wandb.finish()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1127657198.py in <cell line: 0>()
     22     )
     23 
---> 24 models, oof_predicted, evals_results = run_train(
     25     train,
     26     target,

/tmp/ipykernel_11/2243529630.py in run_train(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds)
     24             callbacks = callbacks + [wandb_callback()]
     25 
---> 26         model = lgb.train(
     27             params=run_params,
     28             train_set=dtrain,

TypeError: train() got an unexpected keyword argument 'evals_result'

## === cell 14
predicted = []
for model in models:
    predicted.append(model.predict(test, num_iteration=model.best_iteration))

avg_preds = np.mean(predicted, axis=0)

avg_preds = np.clip(avg_preds, 1.0, 100.0)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2388074961.py in <cell line: 0>()
      1 predicted = []
----> 2 for model in models:
      3     predicted.append(model.predict(test, num_iteration=model.best_iteration))
      4 
      5 avg_preds = np.mean(predicted, axis=0)

NameError: name 'models' is not defined

## === cell 15
submission = pd.DataFrame(
    {"Id": sample_submission["Id"].values, "Pawpularity": avg_preds}
)
submission.to_csv("submission.csv", index=False, float_format="%.6f")
print(submission.head(20))
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3039077885.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"Id": sample_submission["Id"].values, "Pawpularity": avg_preds}
      3 )
      4 submission.to_csv("submission.csv", index=False, float_format="%.6f")
      5 print(submission.head(20))

NameError: name 'avg_preds' is not defined
