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
        callbacks=[
            lgb.log_evaluation(VERBOSE_EVAL),
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        ],
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
            callbacks=[
                lgb.log_evaluation(verbose_eval),
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False
                ),
            ],
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
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/897376755.py in <cell line: 0>()
     22 }
     23 
---> 24 models, oof_predicted, evals_results = run_train(
     25     X,
     26     target,

/tmp/ipykernel_11/3011544790.py in run_train(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds)
     14         valid_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CAT_FEATURES)
     15 
---> 16         model = lgb.train(
     17             run_params,
     18             train_set,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    295     # construct booster
    296     try:
--> 297         booster = Booster(params=params, train_set=train_set)
    298         if is_valid_contain_train:
    299             booster.set_train_data_name(train_data_name)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in __init__(self, params, train_set, model_file, model_str)
   3654                 )
   3655             # construct booster object
-> 3656             train_set.construct()
   3657             # copy the parameters from train_set
   3658             params.update(train_set.get_params())

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in construct(self)
   2588             else:
   2589                 # create train
-> 2590                 self._lazy_init(
   2591                     data=self.data,
   2592                     label=self.label,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _lazy_init(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)
   2121             categorical_feature = reference.categorical_feature
   2122         if isinstance(data, pd_DataFrame):
-> 2123             data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
   2124                 data=data,
   2125                 feature_name=feature_name,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    866 
    867     return (
--> 868         _pandas_to_numpy(data, target_dtype=target_dtype),
    869         feature_name,
    870         categorical_feature,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _pandas_to_numpy(data, target_dtype)
    812     target_dtype: "np.typing.DTypeLike",
    813 ) -> np.ndarray:
--> 814     _check_for_bad_pandas_dtypes(data.dtypes)
    815     try:
    816         # most common case (no nullable dtypes)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _check_for_bad_pandas_dtypes(pandas_dtypes_series)
    803     ]
    804     if bad_pandas_dtypes:
--> 805         raise ValueError(
    806             f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
    807         )

ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: Subject Focus-Eyes: object, Subject Focus-Eyes-Face: object, Subject Focus-Eyes-Near: object, Subject Focus-Eyes-Action: object, Subject Focus-Eyes-Accessory: object, Subject Focus-Eyes-Group: object, Subject Focus-Eyes-Collage: object, Subject Focus-Eyes-Human: object, Subject Focus-Eyes-Occlusion: object, Subject Focus-Eyes-Info: object, Subject Focus-Eyes-Blur: object, Subject Focus-Face: object, Subject Focus-Face-Eyes: object, Subject Focus-Face-Near: object, Subject Focus-Face-Action: object, Subject Focus-Face-Accessory: object, Subject Focus-Face-Group: object, Subject Focus-Face-Collage: object, Subject Focus-Face-Human: object, Subject Focus-Face-Occlusion: object, Subject Focus-Face-Info: object, Subject Focus-Face-Blur: object, Subject Focus-Near: object, Subject Focus-Near-Eyes: object, Subject Focus-Near-Face: object, Subject Focus-Near-Action: object, Subject Focus-Near-Accessory: object, Subject Focus-Near-Group: object, Subject Focus-Near-Collage: object, Subject Focus-Near-Human: object, Subject Focus-Near-Occlusion: object, Subject Focus-Near-Info: object, Subject Focus-Near-Blur: object, Subject Focus-Action: object, Subject Focus-Action-Eyes: object, Subject Focus-Action-Face: object, Subject Focus-Action-Near: object, Subject Focus-Action-Accessory: object, Subject Focus-Action-Group: object, Subject Focus-Action-Collage: object, Subject Focus-Action-Human: object, Subject Focus-Action-Occlusion: object, Subject Focus-Action-Info: object, Subject Focus-Action-Blur: object, Subject Focus-Accessory: object, Subject Focus-Accessory-Eyes: object, Subject Focus-Accessory-Face: object, Subject Focus-Accessory-Near: object, Subject Focus-Accessory-Action: object, Subject Focus-Accessory-Group: object, Subject Focus-Accessory-Collage: object, Subject Focus-Accessory-Human: object, Subject Focus-Accessory-Occlusion: object, Subject Focus-Accessory-Info: object, Subject Focus-Accessory-Blur: object, Subject Focus-Group: object, Subject Focus-Group-Eyes: object, Subject Focus-Group-Face: object, Subject Focus-Group-Near: object, Subject Focus-Group-Action: object, Subject Focus-Group-Accessory: object, Subject Focus-Group-Collage: object, Subject Focus-Group-Human: object, Subject Focus-Group-Occlusion: object, Subject Focus-Group-Info: object, Subject Focus-Group-Blur: object, Subject Focus-Collage: object, Subject Focus-Collage-Eyes: object, Subject Focus-Collage-Face: object, Subject Focus-Collage-Near: object, Subject Focus-Collage-Action: object, Subject Focus-Collage-Accessory: object, Subject Focus-Collage-Group: object, Subject Focus-Collage-Human: object, Subject Focus-Collage-Occlusion: object, Subject Focus-Collage-Info: object, Subject Focus-Collage-Blur: object, Subject Focus-Human: object, Subject Focus-Human-Eyes: object, Subject Focus-Human-Face: object, Subject Focus-Human-Near: object, Subject Focus-Human-Action: object, Subject Focus-Human-Accessory: object, Subject Focus-Human-Group: object, Subject Focus-Human-Collage: object, Subject Focus-Human-Occlusion: object, Subject Focus-Human-Info: object, Subject Focus-Human-Blur: object, Subject Focus-Occlusion: object, Subject Focus-Occlusion-Eyes: object, Subject Focus-Occlusion-Face: object, Subject Focus-Occlusion-Near: object, Subject Focus-Occlusion-Action: object, Subject Focus-Occlusion-Accessory: object, Subject Focus-Occlusion-Group: object, Subject Focus-Occlusion-Collage: object, Subject Focus-Occlusion-Human: object, Subject Focus-Occlusion-Info: object, Subject Focus-Occlusion-Blur: object, Subject Focus-Info: object, Subject Focus-Info-Eyes: object, Subject Focus-Info-Face: object, Subject Focus-Info-Near: object, Subject Focus-Info-Action: object, Subject Focus-Info-Accessory: object, Subject Focus-Info-Group: object, Subject Focus-Info-Collage: object, Subject Focus-Info-Human: object, Subject Focus-Info-Occlusion: object, Subject Focus-Info-Blur: object, Subject Focus-Blur: object, Subject Focus-Blur-Eyes: object, Subject Focus-Blur-Face: object, Subject Focus-Blur-Near: object, Subject Focus-Blur-Action: object, Subject Focus-Blur-Accessory: object, Subject Focus-Blur-Group: object, Subject Focus-Blur-Collage: object, Subject Focus-Blur-Human: object, Subject Focus-Blur-Occlusion: object, Subject Focus-Blur-Info: object, Eyes-Subject Focus: object, Eyes-Subject Focus-Face: object, Eyes-Subject Focus-Near: object, Eyes-Subject Focus-Action: object, Eyes-Subject Focus-Accessory: object, Eyes-Subject Focus-Group: object, Eyes-Subject Focus-Collage: object, Eyes-Subject Focus-Human: object, Eyes-Subject Focus-Occlusion: object, Eyes-Subject Focus-Info: object, Eyes-Subject Focus-Blur: object, Eyes-Face: object, Eyes-Face-Subject Focus: object, Eyes-Face-Near: object, Eyes-Face-Action: object, Eyes-Face-Accessory: object, Eyes-Face-Group: object, Eyes-Face-Collage: object, Eyes-Face-Human: object, Eyes-Face-Occlusion: object, Eyes-Face-Info: object, Eyes-Face-Blur: object, Eyes-Near: object, Eyes-Near-Subject Focus: object, Eyes-Near-Face: object, Eyes-Near-Action: object, Eyes-Near-Accessory: object, Eyes-Near-Group: object, Eyes-Near-Collage: object, Eyes-Near-Human: object, Eyes-Near-Occlusion: object, Eyes-Near-Info: object, Eyes-Near-Blur: object, Eyes-Action: object, Eyes-Action-Subject Focus: object, Eyes-Action-Face: object, Eyes-Action-Near: object, Eyes-Action-Accessory: object, Eyes-Action-Group: object, Eyes-Action-Collage: object, Eyes-Action-Human: object, Eyes-Action-Occlusion: object, Eyes-Action-Info: object, Eyes-Action-Blur: object, Eyes-Accessory: object, Eyes-Accessory-Subject Focus: object, Eyes-Accessory-Face: object, Eyes-Accessory-Near: object, Eyes-Accessory-Action: object, Eyes-Accessory-Group: object, Eyes-Accessory-Collage: object, Eyes-Accessory-Human: object, Eyes-Accessory-Occlusion: object, Eyes-Accessory-Info: object, Eyes-Accessory-Blur: object, Eyes-Group: object, Eyes-Group-Subject Focus: object, Eyes-Group-Face: object, Eyes-Group-Near: object, Eyes-Group-Action: object, Eyes-Group-Accessory: object, Eyes-Group-Collage: object, Eyes-Group-Human: object, Eyes-Group-Occlusion: object, Eyes-Group-Info: object, Eyes-Group-Blur: object, Eyes-Collage: object, Eyes-Collage-Subject Focus: object, Eyes-Collage-Face: object, Eyes-Collage-Near: object, Eyes-Collage-Action: object, Eyes-Collage-Accessory: object, Eyes-Collage-Group: object, Eyes-Collage-Human: object, Eyes-Collage-Occlusion: object, Eyes-Collage-Info: object, Eyes-Collage-Blur: object, Eyes-Human: object, Eyes-Human-Subject Focus: object, Eyes-Human-Face: object, Eyes-Human-Near: object, Eyes-Human-Action: object, Eyes-Human-Accessory: object, Eyes-Human-Group: object, Eyes-Human-Collage: object, Eyes-Human-Occlusion: object, Eyes-Human-Info: object, Eyes-Human-Blur: object, Eyes-Occlusion: object, Eyes-Occlusion-Subject Focus: object, Eyes-Occlusion-Face: object, Eyes-Occlusion-Near: object, Eyes-Occlusion-Action: object, Eyes-Occlusion-Accessory: object, Eyes-Occlusion-Group: object, Eyes-Occlusion-Collage: object, Eyes-Occlusion-Human: object, Eyes-Occlusion-Info: object, Eyes-Occlusion-Blur: object, Eyes-Info: object, Eyes-Info-Subject Focus: object, Eyes-Info-Face: object, Eyes-Info-Near: object, Eyes-Info-Action: object, Eyes-Info-Accessory: object, Eyes-Info-Group: object, Eyes-Info-Collage: object, Eyes-Info-Human: object, Eyes-Info-Occlusion: object, Eyes-Info-Blur: object, Eyes-Blur: object, Eyes-Blur-Subject Focus: object, Eyes-Blur-Face: object, Eyes-Blur-Near: object, Eyes-Blur-Action: object, Eyes-Blur-Accessory: object, Eyes-Blur-Group: object, Eyes-Blur-Collage: object, Eyes-Blur-Human: object, Eyes-Blur-Occlusion: object, Eyes-Blur-Info: object, Face-Subject Focus: object, Face-Subject Focus-Eyes: object, Face-Subject Focus-Near: object, Face-Subject Focus-Action: object, Face-Subject Focus-Accessory: object, Face-Subject Focus-Group: object, Face-Subject Focus-Collage: object, Face-Subject Focus-Human: object, Face-Subject Focus-Occlusion: object, Face-Subject Focus-Info: object, Face-Subject Focus-Blur: object, Face-Eyes: object, Face-Eyes-Subject Focus: object, Face-Eyes-Near: object, Face-Eyes-Action: object, Face-Eyes-Accessory: object, Face-Eyes-Group: object, Face-Eyes-Collage: object, Face-Eyes-Human: object, Face-Eyes-Occlusion: object, Face-Eyes-Info: object, Face-Eyes-Blur: object, Face-Near: object, Face-Near-Subject Focus: object, Face-Near-Eyes: object, Face-Near-Action: object, Face-Near-Accessory: object, Face-Near-Group: object, Face-Near-Collage: object, Face-Near-Human: object, Face-Near-Occlusion: object, Face-Near-Info: object, Face-Near-Blur: object, Face-Action: object, Face-Action-Subject Focus: object, Face-Action-Eyes: object, Face-Action-Near: object, Face-Action-Accessory: object, Face-Action-Group: object, Face-Action-Collage: object, Face-Action-Human: object, Face-Action-Occlusion: object, Face-Action-Info: object, Face-Action-Blur: object, Face-Accessory: object, Face-Accessory-Subject Focus: object, Face-Accessory-Eyes: object, Face-Accessory-Near: object, Face-Accessory-Action: object, Face-Accessory-Group: object, Face-Accessory-Collage: object, Face-Accessory-Human: object, Face-Accessory-Occlusion: object, Face-Accessory-Info: object, Face-Accessory-Blur: object, Face-Group: object, Face-Group-Subject Focus: object, Face-Group-Eyes: object, Face-Group-Near: object, Face-Group-Action: object, Face-Group-Accessory: object, Face-Group-Collage: object, Face-Group-Human: object, Face-Group-Occlusion: object, Face-Group-Info: object, Face-Group-Blur: object, Face-Collage: object, Face-Collage-Subject Focus: object, Face-Collage-Eyes: object, Face-Collage-Near: object, Face-Collage-Action: object, Face-Collage-Accessory: object, Face-Collage-Group: object, Face-Collage-Human: object, Face-Collage-Occlusion: object, Face-Collage-Info: object, Face-Collage-Blur: object, Face-Human: object, Face-Human-Subject Focus: object, Face-Human-Eyes: object, Face-Human-Near: object, Face-Human-Action: object, Face-Human-Accessory: object, Face-Human-Group: object, Face-Human-Collage: object, Face-Human-Occlusion: object, Face-Human-Info: object, Face-Human-Blur: object, Face-Occlusion: object, Face-Occlusion-Subject Focus: object, Face-Occlusion-Eyes: object, Face-Occlusion-Near: object, Face-Occlusion-Action: object, Face-Occlusion-Accessory: object, Face-Occlusion-Group: object, Face-Occlusion-Collage: object, Face-Occlusion-Human: object, Face-Occlusion-Info: object, Face-Occlusion-Blur: object, Face-Info: object, Face-Info-Subject Focus: object, Face-Info-Eyes: object, Face-Info-Near: object, Face-Info-Action: object, Face-Info-Accessory: object, Face-Info-Group: object, Face-Info-Collage: object, Face-Info-Human: object, Face-Info-Occlusion: object, Face-Info-Blur: object, Face-Blur: object, Face-Blur-Subject Focus: object, Face-Blur-Eyes: object, Face-Blur-Near: object, Face-Blur-Action: object, Face-Blur-Accessory: object, Face-Blur-Group: object, Face-Blur-Collage: object, Face-Blur-Human: object, Face-Blur-Occlusion: object, Face-Blur-Info: object, Near-Subject Focus: object, Near-Subject Focus-Eyes: object, Near-Subject Focus-Face: object, Near-Subject Focus-Action: object, Near-Subject Focus-Accessory: object, Near-Subject Focus-Group: object, Near-Subject Focus-Collage: object, Near-Subject Focus-Human: object, Near-Subject Focus-Occlusion: object, Near-Subject Focus-Info: object, Near-Subject Focus-Blur: object, Near-Eyes: object, Near-Eyes-Subject Focus: object, Near-Eyes-Face: object, Near-Eyes-Action: object, Near-Eyes-Accessory: object, Near-Eyes-Group: object, Near-Eyes-Collage: object, Near-Eyes-Human: object, Near-Eyes-Occlusion: object, Near-Eyes-Info: object, Near-Eyes-Blur: object, Near-Face: object, Near-Face-Subject Focus: object, Near-Face-Eyes: object, Near-Face-Action: object, Near-Face-Accessory: object, Near-Face-Group: object, Near-Face-Collage: object, Near-Face-Human: object, Near-Face-Occlusion: object, Near-Face-Info: object, Near-Face-Blur: object, Near-Action: object, Near-Action-Subject Focus: object, Near-Action-Eyes: object, Near-Action-Face: object, Near-Action-Accessory: object, Near-Action-Group: object, Near-Action-Collage: object, Near-Action-Human: object, Near-Action-Occlusion: object, Near-Action-Info: object, Near-Action-Blur: object, Near-Accessory: object, Near-Accessory-Subject Focus: object, Near-Accessory-Eyes: object, Near-Accessory-Face: object, Near-Accessory-Action: object, Near-Accessory-Group: object, Near-Accessory-Collage: object, Near-Accessory-Human: object, Near-Accessory-Occlusion: object, Near-Accessory-Info: object, Near-Accessory-Blur: object, Near-Group: object, Near-Group-Subject Focus: object, Near-Group-Eyes: object, Near-Group-Face: object, Near-Group-Action: object, Near-Group-Accessory: object, Near-Group-Collage: object, Near-Group-Human: object, Near-Group-Occlusion: object, Near-Group-Info: object, Near-Group-Blur: object, Near-Collage: object, Near-Collage-Subject Focus: object, Near-Collage-Eyes: object, Near-Collage-Face: object, Near-Collage-Action: object, Near-Collage-Accessory: object, Near-Collage-Group: object, Near-Collage-Human: object, Near-Collage-Occlusion: object, Near-Collage-Info: object, Near-Collage-Blur: object, Near-Human: object, Near-Human-Subject Focus: object, Near-Human-Eyes: object, Near-Human-Face: object, Near-Human-Action: object, Near-Human-Accessory: object, Near-Human-Group: object, Near-Human-Collage: object, Near-Human-Occlusion: object, Near-Human-Info: object, Near-Human-Blur: object, Near-Occlusion: object, Near-Occlusion-Subject Focus: object, Near-Occlusion-Eyes: object, Near-Occlusion-Face: object, Near-Occlusion-Action: object, Near-Occlusion-Accessory: object, Near-Occlusion-Group: object, Near-Occlusion-Collage: object, Near-Occlusion-Human: object, Near-Occlusion-Info: object, Near-Occlusion-Blur: object, Near-Info: object, Near-Info-Subject Focus: object, Near-Info-Eyes: object, Near-Info-Face: object, Near-Info-Action: object, Near-Info-Accessory: object, Near-Info-Group: object, Near-Info-Collage: object, Near-Info-Human: object, Near-Info-Occlusion: object, Near-Info-Blur: object, Near-Blur: object, Near-Blur-Subject Focus: object, Near-Blur-Eyes: object, Near-Blur-Face: object, Near-Blur-Action: object, Near-Blur-Accessory: object, Near-Blur-Group: object, Near-Blur-Collage: object, Near-Blur-Human: object, Near-Blur-Occlusion: object, Near-Blur-Info: object, Action-Subject Focus: object, Action-Subject Focus-Eyes: object, Action-Subject Focus-Face: object, Action-Subject Focus-Near: object, Action-Subject Focus-Accessory: object, Action-Subject Focus-Group: object, Action-Subject Focus-Collage: object, Action-Subject Focus-Human: object, Action-Subject Focus-Occlusion: object, Action-Subject Focus-Info: object, Action-Subject Focus-Blur: object, Action-Eyes: object, Action-Eyes-Subject Focus: object, Action-Eyes-Face: object, Action-Eyes-Near: object, Action-Eyes-Accessory: object, Action-Eyes-Group: object, Action-Eyes-Collage: object, Action-Eyes-Human: object, Action-Eyes-Occlusion: object, Action-Eyes-Info: object, Action-Eyes-Blur: object, Action-Face: object, Action-Face-Subject Focus: object, Action-Face-Eyes: object, Action-Face-Near: object, Action-Face-Accessory: object, Action-Face-Group: object, Action-Face-Collage: object, Action-Face-Human: object, Action-Face-Occlusion: object, Action-Face-Info: object, Action-Face-Blur: object, Action-Near: object, Action-Near-Subject Focus: object, Action-Near-Eyes: object, Action-Near-Face: object, Action-Near-Accessory: object, Action-Near-Group: object, Action-Near-Collage: object, Action-Near-Human: object, Action-Near-Occlusion: object, Action-Near-Info: object, Action-Near-Blur: object, Action-Accessory: object, Action-Accessory-Subject Focus: object, Action-Accessory-Eyes: object, Action-Accessory-Face: object, Action-Accessory-Near: object, Action-Accessory-Group: object, Action-Accessory-Collage: object, Action-Accessory-Human: object, Action-Accessory-Occlusion: object, Action-Accessory-Info: object, Action-Accessory-Blur: object, Action-Group: object, Action-Group-Subject Focus: object, Action-Group-Eyes: object, Action-Group-Face: object, Action-Group-Near: object, Action-Group-Accessory: object, Action-Group-Collage: object, Action-Group-Human: object, Action-Group-Occlusion: object, Action-Group-Info: object, Action-Group-Blur: object, Action-Collage: object, Action-Collage-Subject Focus: object, Action-Collage-Eyes: object, Action-Collage-Face: object, Action-Collage-Near: object, Action-Collage-Accessory: object, Action-Collage-Group: object, Action-Collage-Human: object, Action-Collage-Occlusion: object, Action-Collage-Info: object, Action-Collage-Blur: object, Action-Human: object, Action-Human-Subject Focus: object, Action-Human-Eyes: object, Action-Human-Face: object, Action-Human-Near: object, Action-Human-Accessory: object, Action-Human-Group: object, Action-Human-Collage: object, Action-Human-Occlusion: object, Action-Human-Info: object, Action-Human-Blur: object, Action-Occlusion: object, Action-Occlusion-Subject Focus: object, Action-Occlusion-Eyes: object, Action-Occlusion-Face: object, Action-Occlusion-Near: object, Action-Occlusion-Accessory: object, Action-Occlusion-Group: object, Action-Occlusion-Collage: object, Action-Occlusion-Human: object, Action-Occlusion-Info: object, Action-Occlusion-Blur: object, Action-Info: object, Action-Info-Subject Focus: object, Action-Info-Eyes: object, Action-Info-Face: object, Action-Info-Near: object, Action-Info-Accessory: object, Action-Info-Group: object, Action-Info-Collage: object, Action-Info-Human: object, Action-Info-Occlusion: object, Action-Info-Blur: object, Action-Blur: object, Action-Blur-Subject Focus: object, Action-Blur-Eyes: object, Action-Blur-Face: object, Action-Blur-Near: object, Action-Blur-Accessory: object, Action-Blur-Group: object, Action-Blur-Collage: object, Action-Blur-Human: object, Action-Blur-Occlusion: object, Action-Blur-Info: object, Accessory-Subject Focus: object, Accessory-Subject Focus-Eyes: object, Accessory-Subject Focus-Face: object, Accessory-Subject Focus-Near: object, Accessory-Subject Focus-Action: object, Accessory-Subject Focus-Group: object, Accessory-Subject Focus-Collage: object, Accessory-Subject Focus-Human: object, Accessory-Subject Focus-Occlusion: object, Accessory-Subject Focus-Info: object, Accessory-Subject Focus-Blur: object, Accessory-Eyes: object, Accessory-Eyes-Subject Focus: object, Accessory-Eyes-Face: object, Accessory-Eyes-Near: object, Accessory-Eyes-Action: object, Accessory-Eyes-Group: object, Accessory-Eyes-Collage: object, Accessory-Eyes-Human: object, Accessory-Eyes-Occlusion: object, Accessory-Eyes-Info: object, Accessory-Eyes-Blur: object, Accessory-Face: object, Accessory-Face-Subject Focus: object, Accessory-Face-Eyes: object, Accessory-Face-Near: object, Accessory-Face-Action: object, Accessory-Face-Group: object, Accessory-Face-Collage: object, Accessory-Face-Human: object, Accessory-Face-Occlusion: object, Accessory-Face-Info: object, Accessory-Face-Blur: object, Accessory-Near: object, Accessory-Near-Subject Focus: object, Accessory-Near-Eyes: object, Accessory-Near-Face: object, Accessory-Near-Action: object, Accessory-Near-Group: object, Accessory-Near-Collage: object, Accessory-Near-Human: object, Accessory-Near-Occlusion: object, Accessory-Near-Info: object, Accessory-Near-Blur: object, Accessory-Action: object, Accessory-Action-Subject Focus: object, Accessory-Action-Eyes: object, Accessory-Action-Face: object, Accessory-Action-Near: object, Accessory-Action-Group: object, Accessory-Action-Collage: object, Accessory-Action-Human: object, Accessory-Action-Occlusion: object, Accessory-Action-Info: object, Accessory-Action-Blur: object, Accessory-Group: object, Accessory-Group-Subject Focus: object, Accessory-Group-Eyes: object, Accessory-Group-Face: object, Accessory-Group-Near: object, Accessory-Group-Action: object, Accessory-Group-Collage: object, Accessory-Group-Human: object, Accessory-Group-Occlusion: object, Accessory-Group-Info: object, Accessory-Group-Blur: object, Accessory-Collage: object, Accessory-Collage-Subject Focus: object, Accessory-Collage-Eyes: object, Accessory-Collage-Face: object, Accessory-Collage-Near: object, Accessory-Collage-Action: object, Accessory-Collage-Group: object, Accessory-Collage-Human: object, Accessory-Collage-Occlusion: object, Accessory-Collage-Info: object, Accessory-Collage-Blur: object, Accessory-Human: object, Accessory-Human-Subject Focus: object, Accessory-Human-Eyes: object, Accessory-Human-Face: object, Accessory-Human-Near: object, Accessory-Human-Action: object, Accessory-Human-Group: object, Accessory-Human-Collage: object, Accessory-Human-Occlusion: object, Accessory-Human-Info: object, Accessory-Human-Blur: object, Accessory-Occlusion: object, Accessory-Occlusion-Subject Focus: object, Accessory-Occlusion-Eyes: object, Accessory-Occlusion-Face: object, Accessory-Occlusion-Near: object, Accessory-Occlusion-Action: object, Accessory-Occlusion-Group: object, Accessory-Occlusion-Collage: object, Accessory-Occlusion-Human: object, Accessory-Occlusion-Info: object, Accessory-Occlusion-Blur: object, Accessory-Info: object, Accessory-Info-Subject Focus: object, Accessory-Info-Eyes: object, Accessory-Info-Face: object, Accessory-Info-Near: object, Accessory-Info-Action: object, Accessory-Info-Group: object, Accessory-Info-Collage: object, Accessory-Info-Human: object, Accessory-Info-Occlusion: object, Accessory-Info-Blur: object, Accessory-Blur: object, Accessory-Blur-Subject Focus: object, Accessory-Blur-Eyes: object, Accessory-Blur-Face: object, Accessory-Blur-Near: object, Accessory-Blur-Action: object, Accessory-Blur-Group: object, Accessory-Blur-Collage: object, Accessory-Blur-Human: object, Accessory-Blur-Occlusion: object, Accessory-Blur-Info: object, Group-Subject Focus: object, Group-Subject Focus-Eyes: object, Group-Subject Focus-Face: object, Group-Subject Focus-Near: object, Group-Subject Focus-Action: object, Group-Subject Focus-Accessory: object, Group-Subject Focus-Collage: object, Group-Subject Focus-Human: object, Group-Subject Focus-Occlusion: object, Group-Subject Focus-Info: object, Group-Subject Focus-Blur: object, Group-Eyes: object, Group-Eyes-Subject Focus: object, Group-Eyes-Face: object, Group-Eyes-Near: object, Group-Eyes-Action: object, Group-Eyes-Accessory: object, Group-Eyes-Collage: object, Group-Eyes-Human: object, Group-Eyes-Occlusion: object, Group-Eyes-Info: object, Group-Eyes-Blur: object, Group-Face: object, Group-Face-Subject Focus: object, Group-Face-Eyes: object, Group-Face-Near: object, Group-Face-Action: object, Group-Face-Accessory: object, Group-Face-Collage: object, Group-Face-Human: object, Group-Face-Occlusion: object, Group-Face-Info: object, Group-Face-Blur: object, Group-Near: object, Group-Near-Subject Focus: object, Group-Near-Eyes: object, Group-Near-Face: object, Group-Near-Action: object, Group-Near-Accessory: object, Group-Near-Collage: object, Group-Near-Human: object, Group-Near-Occlusion: object, Group-Near-Info: object, Group-Near-Blur: object, Group-Action: object, Group-Action-Subject Focus: object, Group-Action-Eyes: object, Group-Action-Face: object, Group-Action-Near: object, Group-Action-Accessory: object, Group-Action-Collage: object, Group-Action-Human: object, Group-Action-Occlusion: object, Group-Action-Info: object, Group-Action-Blur: object, Group-Accessory: object, Group-Accessory-Subject Focus: object, Group-Accessory-Eyes: object, Group-Accessory-Face: object, Group-Accessory-Near: object, Group-Accessory-Action: object, Group-Accessory-Collage: object, Group-Accessory-Human: object, Group-Accessory-Occlusion: object, Group-Accessory-Info: object, Group-Accessory-Blur: object, Group-Collage: object, Group-Collage-Subject Focus: object, Group-Collage-Eyes: object, Group-Collage-Face: object, Group-Collage-Near: object, Group-Collage-Action: object, Group-Collage-Accessory: object, Group-Collage-Human: object, Group-Collage-Occlusion: object, Group-Collage-Info: object, Group-Collage-Blur: object, Group-Human: object, Group-Human-Subject Focus: object, Group-Human-Eyes: object, Group-Human-Face: object, Group-Human-Near: object, Group-Human-Action: object, Group-Human-Accessory: object, Group-Human-Collage: object, Group-Human-Occlusion: object, Group-Human-Info: object, Group-Human-Blur: object, Group-Occlusion: object, Group-Occlusion-Subject Focus: object, Group-Occlusion-Eyes: object, Group-Occlusion-Face: object, Group-Occlusion-Near: object, Group-Occlusion-Action: object, Group-Occlusion-Accessory: object, Group-Occlusion-Collage: object, Group-Occlusion-Human: object, Group-Occlusion-Info: object, Group-Occlusion-Blur: object, Group-Info: object, Group-Info-Subject Focus: object, Group-Info-Eyes: object, Group-Info-Face: object, Group-Info-Near: object, Group-Info-Action: object, Group-Info-Accessory: object, Group-Info-Collage: object, Group-Info-Human: object, Group-Info-Occlusion: object, Group-Info-Blur: object, Group-Blur: object, Group-Blur-Subject Focus: object, Group-Blur-Eyes: object, Group-Blur-Face: object, Group-Blur-Near: object, Group-Blur-Action: object, Group-Blur-Accessory: object, Group-Blur-Collage: object, Group-Blur-Human: object, Group-Blur-Occlusion: object, Group-Blur-Info: object, Collage-Subject Focus: object, Collage-Subject Focus-Eyes: object, Collage-Subject Focus-Face: object, Collage-Subject Focus-Near: object, Collage-Subject Focus-Action: object, Collage-Subject Focus-Accessory: object, Collage-Subject Focus-Group: object, Collage-Subject Focus-Human: object, Collage-Subject Focus-Occlusion: object, Collage-Subject Focus-Info: object, Collage-Subject Focus-Blur: object, Collage-Eyes: object, Collage-Eyes-Subject Focus: object, Collage-Eyes-Face: object, Collage-Eyes-Near: object, Collage-Eyes-Action: object, Collage-Eyes-Accessory: object, Collage-Eyes-Group: object, Collage-Eyes-Human: object, Collage-Eyes-Occlusion: object, Collage-Eyes-Info: object, Collage-Eyes-Blur: object, Collage-Face: object, Collage-Face-Subject Focus: object, Collage-Face-Eyes: object, Collage-Face-Near: object, Collage-Face-Action: object, Collage-Face-Accessory: object, Collage-Face-Group: object, Collage-Face-Human: object, Collage-Face-Occlusion: object, Collage-Face-Info: object, Collage-Face-Blur: object, Collage-Near: object, Collage-Near-Subject Focus: object, Collage-Near-Eyes: object, Collage-Near-Face: object, Collage-Near-Action: object, Collage-Near-Accessory: object, Collage-Near-Group: object, Collage-Near-Human: object, Collage-Near-Occlusion: object, Collage-Near-Info: object, Collage-Near-Blur: object, Collage-Action: object, Collage-Action-Subject Focus: object, Collage-Action-Eyes: object, Collage-Action-Face: object, Collage-Action-Near: object, Collage-Action-Accessory: object, Collage-Action-Group: object, Collage-Action-Human: object, Collage-Action-Occlusion: object, Collage-Action-Info: object, Collage-Action-Blur: object, Collage-Accessory: object, Collage-Accessory-Subject Focus: object, Collage-Accessory-Eyes: object, Collage-Accessory-Face: object, Collage-Accessory-Near: object, Collage-Accessory-Action: object, Collage-Accessory-Group: object, Collage-Accessory-Human: object, Collage-Accessory-Occlusion: object, Collage-Accessory-Info: object, Collage-Accessory-Blur: object, Collage-Group: object, Collage-Group-Subject Focus: object, Collage-Group-Eyes: object, Collage-Group-Face: object, Collage-Group-Near: object, Collage-Group-Action: object, Collage-Group-Accessory: object, Collage-Group-Human: object, Collage-Group-Occlusion: object, Collage-Group-Info: object, Collage-Group-Blur: object, Collage-Human: object, Collage-Human-Subject Focus: object, Collage-Human-Eyes: object, Collage-Human-Face: object, Collage-Human-Near: object, Collage-Human-Action: object, Collage-Human-Accessory: object, Collage-Human-Group: object, Collage-Human-Occlusion: object, Collage-Human-Info: object, Collage-Human-Blur: object, Collage-Occlusion: object, Collage-Occlusion-Subject Focus: object, Collage-Occlusion-Eyes: object, Collage-Occlusion-Face: object, Collage-Occlusion-Near: object, Collage-Occlusion-Action: object, Collage-Occlusion-Accessory: object, Collage-Occlusion-Group: object, Collage-Occlusion-Human: object, Collage-Occlusion-Info: object, Collage-Occlusion-Blur: object, Collage-Info: object, Collage-Info-Subject Focus: object, Collage-Info-Eyes: object, Collage-Info-Face: object, Collage-Info-Near: object, Collage-Info-Action: object, Collage-Info-Accessory: object, Collage-Info-Group: object, Collage-Info-Human: object, Collage-Info-Occlusion: object, Collage-Info-Blur: object, Collage-Blur: object, Collage-Blur-Subject Focus: object, Collage-Blur-Eyes: object, Collage-Blur-Face: object, Collage-Blur-Near: object, Collage-Blur-Action: object, Collage-Blur-Accessory: object, Collage-Blur-Group: object, Collage-Blur-Human: object, Collage-Blur-Occlusion: object, Collage-Blur-Info: object, Human-Subject Focus: object, Human-Subject Focus-Eyes: object, Human-Subject Focus-Face: object, Human-Subject Focus-Near: object, Human-Subject Focus-Action: object, Human-Subject Focus-Accessory: object, Human-Subject Focus-Group: object, Human-Subject Focus-Collage: object, Human-Subject Focus-Occlusion: object, Human-Subject Focus-Info: object, Human-Subject Focus-Blur: object, Human-Eyes: object, Human-Eyes-Subject Focus: object, Human-Eyes-Face: object, Human-Eyes-Near: object, Human-Eyes-Action: object, Human-Eyes-Accessory: object, Human-Eyes-Group: object, Human-Eyes-Collage: object, Human-Eyes-Occlusion: object, Human-Eyes-Info: object, Human-Eyes-Blur: object, Human-Face: object, Human-Face-Subject Focus: object, Human-Face-Eyes: object, Human-Face-Near: object, Human-Face-Action: object, Human-Face-Accessory: object, Human-Face-Group: object, Human-Face-Collage: object, Human-Face-Occlusion: object, Human-Face-Info: object, Human-Face-Blur: object, Human-Near: object, Human-Near-Subject Focus: object, Human-Near-Eyes: object, Human-Near-Face: object, Human-Near-Action: object, Human-Near-Accessory: object, Human-Near-Group: object, Human-Near-Collage: object, Human-Near-Occlusion: object, Human-Near-Info: object, Human-Near-Blur: object, Human-Action: object, Human-Action-Subject Focus: object, Human-Action-Eyes: object, Human-Action-Face: object, Human-Action-Near: object, Human-Action-Accessory: object, Human-Action-Group: object, Human-Action-Collage: object, Human-Action-Occlusion: object, Human-Action-Info: object, Human-Action-Blur: object, Human-Accessory: object, Human-Accessory-Subject Focus: object, Human-Accessory-Eyes: object, Human-Accessory-Face: object, Human-Accessory-Near: object, Human-Accessory-Action: object, Human-Accessory-Group: object, Human-Accessory-Collage: object, Human-Accessory-Occlusion: object, Human-Accessory-Info: object, Human-Accessory-Blur: object, Human-Group: object, Human-Group-Subject Focus: object, Human-Group-Eyes: object, Human-Group-Face: object, Human-Group-Near: object, Human-Group-Action: object, Human-Group-Accessory: object, Human-Group-Collage: object, Human-Group-Occlusion: object, Human-Group-Info: object, Human-Group-Blur: object, Human-Collage: object, Human-Collage-Subject Focus: object, Human-Collage-Eyes: object, Human-Collage-Face: object, Human-Collage-Near: object, Human-Collage-Action: object, Human-Collage-Accessory: object, Human-Collage-Group: object, Human-Collage-Occlusion: object, Human-Collage-Info: object, Human-Collage-Blur: object, Human-Occlusion: object, Human-Occlusion-Subject Focus: object, Human-Occlusion-Eyes: object, Human-Occlusion-Face: object, Human-Occlusion-Near: object, Human-Occlusion-Action: object, Human-Occlusion-Accessory: object, Human-Occlusion-Group: object, Human-Occlusion-Collage: object, Human-Occlusion-Info: object, Human-Occlusion-Blur: object, Human-Info: object, Human-Info-Subject Focus: object, Human-Info-Eyes: object, Human-Info-Face: object, Human-Info-Near: object, Human-Info-Action: object, Human-Info-Accessory: object, Human-Info-Group: object, Human-Info-Collage: object, Human-Info-Occlusion: object, Human-Info-Blur: object, Human-Blur: object, Human-Blur-Subject Focus: object, Human-Blur-Eyes: object, Human-Blur-Face: object, Human-Blur-Near: object, Human-Blur-Action: object, Human-Blur-Accessory: object, Human-Blur-Group: object, Human-Blur-Collage: object, Human-Blur-Occlusion: object, Human-Blur-Info: object, Occlusion-Subject Focus: object, Occlusion-Subject Focus-Eyes: object, Occlusion-Subject Focus-Face: object, Occlusion-Subject Focus-Near: object, Occlusion-Subject Focus-Action: object, Occlusion-Subject Focus-Accessory: object, Occlusion-Subject Focus-Group: object, Occlusion-Subject Focus-Collage: object, Occlusion-Subject Focus-Human: object, Occlusion-Subject Focus-Info: object, Occlusion-Subject Focus-Blur: object, Occlusion-Eyes: object, Occlusion-Eyes-Subject Focus: object, Occlusion-Eyes-Face: object, Occlusion-Eyes-Near: object, Occlusion-Eyes-Action: object, Occlusion-Eyes-Accessory: object, Occlusion-Eyes-Group: object, Occlusion-Eyes-Collage: object, Occlusion-Eyes-Human: object, Occlusion-Eyes-Info: object, Occlusion-Eyes-Blur: object, Occlusion-Face: object, Occlusion-Face-Subject Focus: object, Occlusion-Face-Eyes: object, Occlusion-Face-Near: object, Occlusion-Face-Action: object, Occlusion-Face-Accessory: object, Occlusion-Face-Group: object, Occlusion-Face-Collage: object, Occlusion-Face-Human: object, Occlusion-Face-Info: object, Occlusion-Face-Blur: object, Occlusion-Near: object, Occlusion-Near-Subject Focus: object, Occlusion-Near-Eyes: object, Occlusion-Near-Face: object, Occlusion-Near-Action: object, Occlusion-Near-Accessory: object, Occlusion-Near-Group: object, Occlusion-Near-Collage: object, Occlusion-Near-Human: object, Occlusion-Near-Info: object, Occlusion-Near-Blur: object, Occlusion-Action: object, Occlusion-Action-Subject Focus: object, Occlusion-Action-Eyes: object, Occlusion-Action-Face: object, Occlusion-Action-Near: object, Occlusion-Action-Accessory: object, Occlusion-Action-Group: object, Occlusion-Action-Collage: object, Occlusion-Action-Human: object, Occlusion-Action-Info: object, Occlusion-Action-Blur: object, Occlusion-Accessory: object, Occlusion-Accessory-Subject Focus: object, Occlusion-Accessory-Eyes: object, Occlusion-Accessory-Face: object, Occlusion-Accessory-Near: object, Occlusion-Accessory-Action: object, Occlusion-Accessory-Group: object, Occlusion-Accessory-Collage: object, Occlusion-Accessory-Human: object, Occlusion-Accessory-Info: object, Occlusion-Accessory-Blur: object, Occlusion-Group: object, Occlusion-Group-Subject Focus: object, Occlusion-Group-Eyes: object, Occlusion-Group-Face: object, Occlusion-Group-Near: object, Occlusion-Group-Action: object, Occlusion-Group-Accessory: object, Occlusion-Group-Collage: object, Occlusion-Group-Human: object, Occlusion-Group-Info: object, Occlusion-Group-Blur: object, Occlusion-Collage: object, Occlusion-Collage-Subject Focus: object, Occlusion-Collage-Eyes: object, Occlusion-Collage-Face: object, Occlusion-Collage-Near: object, Occlusion-Collage-Action: object, Occlusion-Collage-Accessory: object, Occlusion-Collage-Group: object, Occlusion-Collage-Human: object, Occlusion-Collage-Info: object, Occlusion-Collage-Blur: object, Occlusion-Human: object, Occlusion-Human-Subject Focus: object, Occlusion-Human-Eyes: object, Occlusion-Human-Face: object, Occlusion-Human-Near: object, Occlusion-Human-Action: object, Occlusion-Human-Accessory: object, Occlusion-Human-Group: object, Occlusion-Human-Collage: object, Occlusion-Human-Info: object, Occlusion-Human-Blur: object, Occlusion-Info: object, Occlusion-Info-Subject Focus: object, Occlusion-Info-Eyes: object, Occlusion-Info-Face: object, Occlusion-Info-Near: object, Occlusion-Info-Action: object, Occlusion-Info-Accessory: object, Occlusion-Info-Group: object, Occlusion-Info-Collage: object, Occlusion-Info-Human: object, Occlusion-Info-Blur: object, Occlusion-Blur: object, Occlusion-Blur-Subject Focus: object, Occlusion-Blur-Eyes: object, Occlusion-Blur-Face: object, Occlusion-Blur-Near: object, Occlusion-Blur-Action: object, Occlusion-Blur-Accessory: object, Occlusion-Blur-Group: object, Occlusion-Blur-Collage: object, Occlusion-Blur-Human: object, Occlusion-Blur-Info: object, Info-Subject Focus: object, Info-Subject Focus-Eyes: object, Info-Subject Focus-Face: object, Info-Subject Focus-Near: object, Info-Subject Focus-Action: object, Info-Subject Focus-Accessory: object, Info-Subject Focus-Group: object, Info-Subject Focus-Collage: object, Info-Subject Focus-Human: object, Info-Subject Focus-Occlusion: object, Info-Subject Focus-Blur: object, Info-Eyes: object, Info-Eyes-Subject Focus: object, Info-Eyes-Face: object, Info-Eyes-Near: object, Info-Eyes-Action: object, Info-Eyes-Accessory: object, Info-Eyes-Group: object, Info-Eyes-Collage: object, Info-Eyes-Human: object, Info-Eyes-Occlusion: object, Info-Eyes-Blur: object, Info-Face: object, Info-Face-Subject Focus: object, Info-Face-Eyes: object, Info-Face-Near: object, Info-Face-Action: object, Info-Face-Accessory: object, Info-Face-Group: object, Info-Face-Collage: object, Info-Face-Human: object, Info-Face-Occlusion: object, Info-Face-Blur: object, Info-Near: object, Info-Near-Subject Focus: object, Info-Near-Eyes: object, Info-Near-Face: object, Info-Near-Action: object, Info-Near-Accessory: object, Info-Near-Group: object, Info-Near-Collage: object, Info-Near-Human: object, Info-Near-Occlusion: object, Info-Near-Blur: object, Info-Action: object, Info-Action-Subject Focus: object, Info-Action-Eyes: object, Info-Action-Face: object, Info-Action-Near: object, Info-Action-Accessory: object, Info-Action-Group: object, Info-Action-Collage: object, Info-Action-Human: object, Info-Action-Occlusion: object, Info-Action-Blur: object, Info-Accessory: object, Info-Accessory-Subject Focus: object, Info-Accessory-Eyes: object, Info-Accessory-Face: object, Info-Accessory-Near: object, Info-Accessory-Action: object, Info-Accessory-Group: object, Info-Accessory-Collage: object, Info-Accessory-Human: object, Info-Accessory-Occlusion: object, Info-Accessory-Blur: object, Info-Group: object, Info-Group-Subject Focus: object, Info-Group-Eyes: object, Info-Group-Face: object, Info-Group-Near: object, Info-Group-Action: object, Info-Group-Accessory: object, Info-Group-Collage: object, Info-Group-Human: object, Info-Group-Occlusion: object, Info-Group-Blur: object, Info-Collage: object, Info-Collage-Subject Focus: object, Info-Collage-Eyes: object, Info-Collage-Face: object, Info-Collage-Near: object, Info-Collage-Action: object, Info-Collage-Accessory: object, Info-Collage-Group: object, Info-Collage-Human: object, Info-Collage-Occlusion: object, Info-Collage-Blur: object, Info-Human: object, Info-Human-Subject Focus: object, Info-Human-Eyes: object, Info-Human-Face: object, Info-Human-Near: object, Info-Human-Action: object, Info-Human-Accessory: object, Info-Human-Group: object, Info-Human-Collage: object, Info-Human-Occlusion: object, Info-Human-Blur: object, Info-Occlusion: object, Info-Occlusion-Subject Focus: object, Info-Occlusion-Eyes: object, Info-Occlusion-Face: object, Info-Occlusion-Near: object, Info-Occlusion-Action: object, Info-Occlusion-Accessory: object, Info-Occlusion-Group: object, Info-Occlusion-Collage: object, Info-Occlusion-Human: object, Info-Occlusion-Blur: object, Info-Blur: object, Info-Blur-Subject Focus: object, Info-Blur-Eyes: object, Info-Blur-Face: object, Info-Blur-Near: object, Info-Blur-Action: object, Info-Blur-Accessory: object, Info-Blur-Group: object, Info-Blur-Collage: object, Info-Blur-Human: object, Info-Blur-Occlusion: object, Blur-Subject Focus: object, Blur-Subject Focus-Eyes: object, Blur-Subject Focus-Face: object, Blur-Subject Focus-Near: object, Blur-Subject Focus-Action: object, Blur-Subject Focus-Accessory: object, Blur-Subject Focus-Group: object, Blur-Subject Focus-Collage: object, Blur-Subject Focus-Human: object, Blur-Subject Focus-Occlusion: object, Blur-Subject Focus-Info: object, Blur-Eyes: object, Blur-Eyes-Subject Focus: object, Blur-Eyes-Face: object, Blur-Eyes-Near: object, Blur-Eyes-Action: object, Blur-Eyes-Accessory: object, Blur-Eyes-Group: object, Blur-Eyes-Collage: object, Blur-Eyes-Human: object, Blur-Eyes-Occlusion: object, Blur-Eyes-Info: object, Blur-Face: object, Blur-Face-Subject Focus: object, Blur-Face-Eyes: object, Blur-Face-Near: object, Blur-Face-Action: object, Blur-Face-Accessory: object, Blur-Face-Group: object, Blur-Face-Collage: object, Blur-Face-Human: object, Blur-Face-Occlusion: object, Blur-Face-Info: object, Blur-Near: object, Blur-Near-Subject Focus: object, Blur-Near-Eyes: object, Blur-Near-Face: object, Blur-Near-Action: object, Blur-Near-Accessory: object, Blur-Near-Group: object, Blur-Near-Collage: object, Blur-Near-Human: object, Blur-Near-Occlusion: object, Blur-Near-Info: object, Blur-Action: object, Blur-Action-Subject Focus: object, Blur-Action-Eyes: object, Blur-Action-Face: object, Blur-Action-Near: object, Blur-Action-Accessory: object, Blur-Action-Group: object, Blur-Action-Collage: object, Blur-Action-Human: object, Blur-Action-Occlusion: object, Blur-Action-Info: object, Blur-Accessory: object, Blur-Accessory-Subject Focus: object, Blur-Accessory-Eyes: object, Blur-Accessory-Face: object, Blur-Accessory-Near: object, Blur-Accessory-Action: object, Blur-Accessory-Group: object, Blur-Accessory-Collage: object, Blur-Accessory-Human: object, Blur-Accessory-Occlusion: object, Blur-Accessory-Info: object, Blur-Group: object, Blur-Group-Subject Focus: object, Blur-Group-Eyes: object, Blur-Group-Face: object, Blur-Group-Near: object, Blur-Group-Action: object, Blur-Group-Accessory: object, Blur-Group-Collage: object, Blur-Group-Human: object, Blur-Group-Occlusion: object, Blur-Group-Info: object, Blur-Collage: object, Blur-Collage-Subject Focus: object, Blur-Collage-Eyes: object, Blur-Collage-Face: object, Blur-Collage-Near: object, Blur-Collage-Action: object, Blur-Collage-Accessory: object, Blur-Collage-Group: object, Blur-Collage-Human: object, Blur-Collage-Occlusion: object, Blur-Collage-Info: object, Blur-Human: object, Blur-Human-Subject Focus: object, Blur-Human-Eyes: object, Blur-Human-Face: object, Blur-Human-Near: object, Blur-Human-Action: object, Blur-Human-Accessory: object, Blur-Human-Group: object, Blur-Human-Collage: object, Blur-Human-Occlusion: object, Blur-Human-Info: object, Blur-Occlusion: object, Blur-Occlusion-Subject Focus: object, Blur-Occlusion-Eyes: object, Blur-Occlusion-Face: object, Blur-Occlusion-Near: object, Blur-Occlusion-Action: object, Blur-Occlusion-Accessory: object, Blur-Occlusion-Group: object, Blur-Occlusion-Collage: object, Blur-Occlusion-Human: object, Blur-Occlusion-Info: object, Blur-Info: object, Blur-Info-Subject Focus: object, Blur-Info-Eyes: object, Blur-Info-Face: object, Blur-Info-Near: object, Blur-Info-Action: object, Blur-Info-Accessory: object, Blur-Info-Group: object, Blur-Info-Collage: object, Blur-Info-Human: object, Blur-Info-Occlusion: object

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
