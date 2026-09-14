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

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

20.45681

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb



## === cell 1
N_SPLITS = 10
SEED = 0
EARLY_STOPPING_ROUNDS = 300
VERBOSE = 1000

PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_child_samples": 60,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": -1,
}




## === cell 2
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


set_seed(SEED)



## === cell 3
CANDIDATE_INPUTS = [
    "../input/petfinder-pawpularity-score/",
    "/kaggle/input/petfinder-pawpularity-score/",
    "/kaggle/data/petfinder-pawpularity-score/",
    "/kaggle/data/input/petfinder-pawpularity-score/",
]
INPUT = None
for p in CANDIDATE_INPUTS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT = p
        break
if INPUT is None:
    raise FileNotFoundError(
        "Could not find petfinder-pawpularity-score dataset. Checked: "
        + ", ".join(CANDIDATE_INPUTS)
    )

train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT, "test.csv"))
sample_submission = pd.read_csv(os.path.join(INPUT, "sample_submission.csv"))
print("Using INPUT:", INPUT)
print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_submission shape:",
    sample_submission.shape,
)



## === cell 4
train.head()



## === cell 5
train["collage_and_info"] = train["Collage"] * train["Info"]
train["collage_or_info"] = train["Collage"] + train["Info"]
train["occlusion_and_human"] = train["Occlusion"] * train["Human"]
train["not_blur_and_eyes"] = (1 - train["Blur"]) * train["Eyes"]
train["not_collage_and_info_or_not_blur_or_group_or_accessory"] = (
    (1 - train["Collage"] * train["Info"])
    + (1 - train["Blur"])
    + train["Group"]
    + train["Accessory"]
)

test["collage_and_info"] = test["Collage"] * test["Info"]
test["collage_or_info"] = test["Collage"] + test["Info"]
test["occlusion_and_human"] = test["Occlusion"] * test["Human"]
test["not_blur_and_eyes"] = (1 - test["Blur"]) * test["Eyes"]
test["not_collage_and_info_or_not_blur_or_group_or_accessory"] = (
    (1 - test["Collage"] * test["Info"])
    + (1 - test["Blur"])
    + test["Group"]
    + test["Accessory"]
)



## === cell 6
train.head()



## === cell 7
X_train, X_test = train.drop(["Pawpularity", "Id"], axis=1), test.drop(["Id"], axis=1)
y_train = train["Pawpularity"]



## === cell 8
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_df = pd.DataFrame(
    {
        "Id": train["Id"],
        "pred": np.zeros(train.shape[0]),
        "Pawpularity": train["Pawpularity"],
    }
)
test_preds = np.zeros(X_test.shape[0], dtype=float)

callbacks = [
    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
    lgb.log_evaluation(period=VERBOSE),
]

for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_trn, X_val = X_train.iloc[trn_idx, :], X_train.iloc[val_idx, :]
    y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    clf = LGBMRegressor(**PARAMS, random_state=SEED)
    clf.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    trn_preds = clf.predict(X_trn)
    val_preds = clf.predict(X_val)
    oof_df.loc[val_idx, "pred"] = val_preds

    test_preds += clf.predict(X_test) / N_SPLITS

    print(f"==== Fold {fold} ====")
    print(f"Trn RMSE: {mean_squared_error(y_trn, trn_preds, squared=False):.4f}")
    print(f"Val RMSE: {mean_squared_error(y_val, val_preds, squared=False):.4f}")

print("==== Results ====")
print(
    f"OOF RMSE: {mean_squared_error(oof_df['Pawpularity'], oof_df['pred'], squared=False):.4f}"
)
oof_df.to_csv("oof.csv", index=False)



## === cell 9
test_preds = np.clip(test_preds, 1.0, 100.0)

submission = pd.DataFrame(
    {"Id": test["Id"].values, "Pawpularity": test_preds.astype(float)}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
