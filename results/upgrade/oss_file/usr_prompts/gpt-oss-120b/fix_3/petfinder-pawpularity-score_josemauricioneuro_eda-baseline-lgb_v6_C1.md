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

3.9

# 3. Installed packages

geopandas==0.14.4
hyperopt==0.2.7
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
pillow==11.3.0
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

20.4976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from PIL import Image
from tqdm import tqdm
import os
import lightgbm as lgb



## === cell 1
sample_sub = "/kaggle/input/petfinder-pawpularity-score/sample_submission.csv"
train_metadata = "/kaggle/input/petfinder-pawpularity-score/train.csv"
test_metadata = "/kaggle/input/petfinder-pawpularity-score/test.csv"




## === cell 2
def create_shape_feature(df):
    width_height_list = []
    file_size_list = []
    for path_ in tqdm(df["img_path"]):
        width_height_list.append(Image.open(path_).size)
        file_size_list.append(os.path.getsize(path_))
    df["width_height"] = width_height_list
    df["file_size"] = file_size_list
    df["width"] = df["width_height"].apply(lambda x: x[0])
    df["height"] = df["width_height"].apply(lambda x: x[1])
    df["area"] = df["width"] * df["height"]
    df["size_per_pixel"] = df["area"] / df["file_size"]
    return df




## === cell 3
df_train = pd.read_csv(train_metadata)
df_test = pd.read_csv(test_metadata)

df_train["img_path"] = df_train["Id"].apply(
    lambda x: f"../input/petfinder-pawpularity-score/train/{str(x)}.jpg"
)
df_test["img_path"] = df_test["Id"].apply(
    lambda x: f"../input/petfinder-pawpularity-score/test/{str(x)}.jpg"
)

df_train = create_shape_feature(df_train)
df_test = create_shape_feature(df_test)

metadata = [
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



## === cell 4
df_train.head()



## === cell 5
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
df_train.hist(ax=ax)
plt.show()



## === cell 6
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
df_test.hist(ax=ax)
plt.show()



## === cell 7
fig, ax = plt.subplots(4, 3, figsize=(15, 18))
i = 0
j = 0
for x in metadata:
    sns.boxplot(x=x, y="Pawpularity", data=df_train, ax=ax[i, j])
    i += 1
    if i > 3:
        i = 0
        j += 1



## === cell 8
numeric_corr = df_train.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(numeric_corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    numeric_corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)
plt.show()



## === cell 9
df_train["Pawpularity_tgt"] = 100 - df_train["Pawpularity"]




## === cell 10
def rmse(y, yhat):
    return np.sqrt(np.mean(np.power(y - yhat, 2)))




## === cell 11
seed = 42


def train_and_optimize_lgb(p):
    print(p)
    params = {
        "objective": "tweedie",
        "boosting_type": "gbdt",
        "max_depth": int(p["max_depth"]),
        "max_bin": int(p["max_bin"]),
        "min_data_in_leaf": int(p["min_data_in_leaf"]),
        "learning_rate": p["learning_rate"],
        "subsample": p["subsample"],
        "subsample_freq": int(p["subsample_freq"]),
        "feature_fraction": p["feature_fraction"],
        "lambda_l1": p["lambda_l1"],
        "lambda_l2": p["lambda_l2"],
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    features = metadata + ["width", "height", "file_size", "area", "size_per_pixel"]
    oof_predictions = np.zeros(df_train.shape[0])
    kfold = KFold(n_splits=5, random_state=seed, shuffle=True)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(df_train)):
        x_train, x_val = (
            df_train.loc[trn_ind, features],
            df_train.loc[val_ind, features],
        )
        y_train, y_val = (
            df_train.loc[trn_ind, "Pawpularity_tgt"],
            df_train.loc[val_ind, "Pawpularity_tgt"],
        )

        train_dataset = lgb.Dataset(x_train, y_train)
        val_dataset = lgb.Dataset(x_val, y_val, reference=train_dataset)

        model = lgb.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=800,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            verbose_eval=False,
        )

        oof_predictions[val_ind] = 100 - model.predict(
            x_val, num_iteration=model.best_iteration
        )
        print(f"Fold {fold} RMSE:", rmse(df_train["Pawpularity"], oof_predictions))

    overall_rmse = rmse(df_train["Pawpularity"], oof_predictions)
    print("OOF RMSE:", overall_rmse)
    return overall_rmse


def make_predictions(p):
    params = {
        "objective": "tweedie",
        "boosting_type": "gbdt",
        "max_depth": int(p["max_depth"]),
        "max_bin": int(p["max_bin"]),
        "min_data_in_leaf": int(p["min_data_in_leaf"]),
        "learning_rate": p["learning_rate"],
        "subsample": p["subsample"],
        "subsample_freq": int(p["subsample_freq"]),
        "feature_fraction": p["feature_fraction"],
        "lambda_l1": p["lambda_l1"],
        "lambda_l2": p["lambda_l2"],
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    features = metadata + ["width", "height", "file_size", "area", "size_per_pixel"]
    x_train = df_train[features]
    y_train = df_train["Pawpularity_tgt"]

    train_dataset = lgb.Dataset(x_train, y_train)

    model = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=200,
        verbose_eval=False,
    )

    test_pred = 100 - model.predict(df_test[features])
    df_test["Pawpularity"] = np.clip(test_pred, 0, 100)
    df_test[["Id", "Pawpularity"]].to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")




## === cell 12
param_space = {
    "max_depth": scope.int(hp.uniform("max_depth", 2, 8)),
    "max_bin": scope.int(hp.uniform("max_bin", 2, 100)),
    "min_data_in_leaf": scope.int(hp.uniform("min_data_in_leaf", 10, 1000)),
    "learning_rate": hp.uniform("learning_rate", 0.001, 0.1),
    "subsample": hp.uniform("subsample", 0.2, 0.9),
    "subsample_freq": scope.int(hp.uniform("subsample_freq", 1, 30)),
    "feature_fraction": hp.uniform("feature_fraction", 0.5, 0.9),
    "lambda_l1": hp.uniform("lambda_l1", 0.1, 3),
    "lambda_l2": hp.uniform("lambda_l2", 0.1, 3),
}

trials = Trials()
hopt = fmin(
    fn=train_and_optimize_lgb,
    space=param_space,
    algo=tpe.suggest,
    max_evals=30,  # reduced for speed
    trials=trials,
    rstate=np.random.RandomState(seed),  # compatible RNG for hyperopt
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2555199225.py in <cell line: 0>()
     12 
     13 trials = Trials()
---> 14 hopt = fmin(
     15     fn=train_and_optimize_lgb,
     16     space=param_space,

/usr/local/lib/python3.11/dist-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, timeout, loss_threshold, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin, points_to_evaluate, max_queue_len, show_progressbar, early_stop_fn, trials_save_file)
    538 
    539     if allow_trials_fmin and hasattr(trials, "fmin"):
--> 540         return trials.fmin(
    541             fn,
    542             space,

/usr/local/lib/python3.11/dist-packages/hyperopt/base.py in fmin(self, fn, space, algo, max_evals, timeout, loss_threshold, max_queue_len, rstate, verbose, pass_expr_memo_ctrl, catch_eval_exceptions, return_argmin, show_progressbar, early_stop_fn, trials_save_file)
    669         from .fmin import fmin
    670 
--> 671         return fmin(
    672             fn,
    673             space,

/usr/local/lib/python3.11/dist-packages/hyperopt/fmin.py in fmin(fn, space, algo, max_evals, timeout, loss_threshold, trials, rstate, allow_trials_fmin, pass_expr_memo_ctrl, catch_eval_exceptions, verbose, return_argmin, points_to_evaluate, max_queue_len, show_progressbar, early_stop_fn, trials_save_file)
    584 
    585     # next line is where the fmin is actually executed
--> 586     rval.exhaust()
    587 
    588     if return_argmin:

/usr/local/lib/python3.11/dist-packages/hyperopt/fmin.py in exhaust(self)
    362     def exhaust(self):
    363         n_done = len(self.trials)
--> 364         self.run(self.max_evals - n_done, block_until_done=self.asynchronous)
    365         self.trials.refresh()
    366         return self

/usr/local/lib/python3.11/dist-packages/hyperopt/fmin.py in run(self, N, block_until_done)
    277                     # processes orchestration
    278                     new_trials = algo(
--> 279                         new_ids, self.domain, trials, self.rstate.integers(2 ** 31 - 1)
    280                     )
    281                     assert len(new_ids) >= len(new_trials)

AttributeError: 'numpy.random.mtrand.RandomState' object has no attribute 'integers'

## === cell 13
print("Best hyperparameters found:", hopt)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/405294777.py in <cell line: 0>()
----> 1 print("Best hyperparameters found:", hopt)
      2 

NameError: name 'hopt' is not defined

## === cell 14
make_predictions(hopt)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2040910712.py in <cell line: 0>()
----> 1 make_predictions(hopt)

NameError: name 'hopt' is not defined
