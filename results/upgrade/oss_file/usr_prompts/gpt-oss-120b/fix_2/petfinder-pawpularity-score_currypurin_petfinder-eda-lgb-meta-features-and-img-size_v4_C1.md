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

20.479513072030347

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm



## === cell 1
BASE = Path("../input/petfinder-pawpularity-score")
TRAIN_IMG_DIR = BASE / "train"
TEST_IMG_DIR = BASE / "test"

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

print(f"train shape: {train.shape}, test shape: {test.shape}")



## === cell 2
train["img_path"] = train["Id"].apply(lambda x: str(TRAIN_IMG_DIR / f"{x}.jpg"))
test["img_path"] = test["Id"].apply(lambda x: str(TEST_IMG_DIR / f"{x}.jpg"))

target_col = "Pawpularity"
metadata_cols = [
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




## === cell 3
def create_shape_feature(df):
    """Add image width, height and file size."""
    width_height_list = []
    file_size_list = []
    for path_ in tqdm(df["img_path"], desc="reading images"):
        with Image.open(path_) as im:
            width_height_list.append(im.size)  # (width, height)
        file_size_list.append(os.path.getsize(path_))
    df["width_height"] = width_height_list
    df["file_size"] = file_size_list
    df["width"] = df["width_height"].apply(lambda x: x[0])
    df["height"] = df["width_height"].apply(lambda x: x[1])
    return df


train = create_shape_feature(train)
test = create_shape_feature(test)



## === cell 4
train["area"] = train["width"] * train["height"]
train["size_per_pixel"] = train["file_size"] / train["area"]

test["area"] = test["width"] * test["height"]
test["size_per_pixel"] = test["file_size"] / test["area"]



## === cell 5
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def calc_model_importance(model, feature_names=None, importance_type="gain"):
    importance_df = pd.DataFrame(
        model.feature_importance(importance_type=importance_type),
        index=feature_names,
        columns=["importance"],
    ).sort_values("importance")
    return importance_df


def calc_mean_importance(importance_df_list):
    mean_vals = np.mean(
        np.stack([df["importance"].values for df in importance_df_list]), axis=0
    )
    mean_df = importance_df_list[0].copy()
    mean_df["importance"] = mean_vals
    return mean_df


def plot_importance(importance_df, title=""):
    importance_df = importance_df.iloc[-50:, :]
    fig, ax = plt.subplots(figsize=(4, 6))
    importance_df.plot.barh(ax=ax)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def do_train(all_feature, params):
    y = all_feature[target_col].values
    X = all_feature.drop(["Id", "img_path", "width_height", target_col], axis=1)

    print(f"features used ({len(X.columns)}): {list(X.columns)}")

    oof = np.zeros(len(X))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    models = []
    scores = []
    gain_imp_list = []
    split_imp_list = []

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, y_tr = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        model = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dvalid],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(30, verbose=False)],
            categorical_feature=metadata_cols,
        )

        pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = pred
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        print(f"Fold {fold} RMSE: {rmse:.3f}")
        scores.append(rmse)
        models.append(model)

        feat_names = X_tr.columns.tolist()
        gain_imp_list.append(calc_model_importance(model, feat_names, "gain"))
        split_imp_list.append(calc_model_importance(model, feat_names, "split"))

    overall_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"CV overall RMSE: {overall_rmse:.3f}")

    gain_imp = calc_mean_importance(gain_imp_list)
    split_imp = calc_mean_importance(split_imp_list)

    return models, gain_imp, split_imp, oof, overall_rmse




## === cell 6
lgb_params = {
    "objective": "regression",
    "max_depth": 3,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "verbosity": -1,
}

models, gain_imp_df, split_imp_df, oof, cv_score = do_train(train, lgb_params)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3681706002.py in <cell line: 0>()
     13 }
     14 
---> 15 models, gain_imp_df, split_imp_df, oof, cv_score = do_train(train, lgb_params)
     16 

/tmp/ipykernel_11/1881290661.py in do_train(all_feature, params)
     57         dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
     58 
---> 59         model = lgb.train(
     60             params=params,
     61             train_set=dtrain,

TypeError: train() got an unexpected keyword argument 'categorical_feature'

## === cell 7
plt.scatter(train[target_col], oof, s=2)
plt.xlabel("True Pawpularity")
plt.ylabel("OOF Prediction")
plt.title(f"CV OOF Scatter (RMSE={cv_score:.3f})")
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2639790195.py in <cell line: 0>()
      1 # optional visual checks
----> 2 plt.scatter(train[target_col], oof, s=2)
      3 plt.xlabel("True Pawpularity")
      4 plt.ylabel("OOF Prediction")
      5 plt.title(f"CV OOF Scatter (RMSE={cv_score:.3f})")

NameError: name 'oof' is not defined

## === cell 8
sample = pd.read_csv(BASE / "sample_submission.csv")
test_features = test.drop(["Id", "img_path", "width_height"], axis=1)

preds = [m.predict(test_features, num_iteration=m.best_iteration) for m in models]
sample[target_col] = np.mean(preds, axis=0)
sample.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3880223682.py in <cell line: 0>()
      3 test_features = test.drop(["Id", "img_path", "width_height"], axis=1)
      4 
----> 5 preds = [m.predict(test_features, num_iteration=m.best_iteration) for m in models]
      6 sample[target_col] = np.mean(preds, axis=0)
      7 sample.to_csv("submission.csv", index=False)

NameError: name 'models' is not defined
