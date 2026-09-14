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

18.196449398872364

# 6. Current score

20.25777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.19669) has done: 'I replaced the failing TensorFlow imports and image‑based pipeline with a lightweight metadata‑only model using scikit‑learn. The new script loads the CSV files, builds simple numeric features from the provided columns, trains a RandomForestRegressor, evaluates RMSE on a validation split, retrains on the full data, and writes a correctly‑formatted `submission.csv`. This fixes all import and file‑not‑found errors and guarantees a valid submission file.'
- What this solution (achieved 20.08872) has done: 'I keep the existing metadata‑only pipeline but add a lightweight GradientBoostingRegressor and average its predictions with the RandomForest. This small ensemble often lowers RMSE on tabular data while preserving the core logic. The change touches only the model training/evaluation cells, keeping all I/O unchanged.'
- What this solution (achieved 20.08858) has done: 'I adjust the ensemble to weight the RandomForest and GradientBoosting predictions according to their individual validation RMSEs, giving more influence to the better model. This small change keeps the overall pipeline unchanged while expected to lower the overall validation RMSE and move the score closer to the target. The same weighting is applied when generating the final test predictions, and the rest of the code remains intact.'
- What this solution (achieved 20.06343) has done: 'I replace the simple inverse‑RMSE weighting with a tiny linear‑regression stack that learns the optimal combination of the RandomForest and GradientBoosting predictions on the validation split. This keeps the original models unchanged, adds only a lightweight meta‑learner, and should lower the validation RMSE, moving the score closer to the target. The rest of the pipeline and the submission format remain the same.'
- What this solution (achieved 20.06607) has done: 'I added a lightweight feature “total_flags” that sums all binary metadata columns, increased the capacity of the RandomForest and GradientBoosting models (more trees and a lower learning rate), and replaced the plain linear stack with a Ridge regression meta‑learner to improve generalisation. These minimal, targeted tweaks keep the original pipeline intact while moving the validation RMSE closer to the target. The script now writes a correctly‑formatted `submission.csv` after these adjustments.'
- What this solution (achieved 20.07834) has done: 'I slightly boost model capacity and add a third simple Ridge predictor to the stacking ensemble. By increasing the number of trees for the RandomForest, using more estimators with a lower learning rate for the GradientBoostingRegressor, and including a direct Ridge regression on the raw features, the validation RMSE should drop a bit, moving the score closer to the target while keeping the original pipeline intact.'
- What this solution (achieved 20.07187) has done: 'I add a simple “mean_flag” feature (total flag count divided by the number of flag columns) to give the models a normalized signal, slightly increase model capacity (more trees for the RandomForest and a larger GradientBoostingRegressor with a lower learning rate), and make the ridge‑stack meta‑learner a bit less regularised. These minimal adjustments keep the original pipeline intact while aiming to lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.06237) has done: 'I add the simple flag‑based features (`total_flags` and `mean_flag`) to the meta‑learner’s inputs and use a slightly less regularised Ridge (alpha = 0.1). This keeps the original three base models unchanged while giving the stack a bit more useful signal, which should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.08919) has done: 'I add an ExtraTreesRegressor as a third base model and replace the ridge‑stack meta‑learner with an unregularized LinearRegression. This keeps the overall pipeline intact while giving the ensemble a richer set of predictions, which should modestly lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.25767) has done: 'I keep the overall pipeline unchanged but replace the unregularized LinearRegression stack with a simple inverse‑RMSE weighted blend of the four base regressors, falling back to the linear stack only if it gives a better validation RMSE. This tiny adjustment preserves all core logic while providing a more calibrated ensemble that should lower the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.25736) has done: 'I slightly boost the tree‑based models (a few more estimators and a lower learning rate for GradientBoosting) and replace the unregularized linear stack with a small‑regularisation Ridge meta‑learner. These modest tweaks keep the overall pipeline unchanged while giving the models a bit more capacity and stabilising the final blending, which should lower the validation RMSE and move the score toward the target.'
- What this solution (achieved 20.25777) has done: 'I slightly increase the capacity of the three tree‑based models (more estimators, a bit deeper GradientBoosting) and simplify the weighted blend to use only these stronger models (dropping the simple Ridge base from the blend). This modest change should lower the validation RMSE, moving the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
import warnings

warnings.filterwarnings("ignore")



## === cell 1
train_df_path = "../input/petfinder-pawpularity-score/train.csv"
test_df_path = "../input/petfinder-pawpularity-score/test.csv"

df_train = pd.read_csv(train_df_path)
df_test = pd.read_csv(test_df_path)

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)



## === cell 2
TARGET = "Pawpularity"
SEED = 88
TEST_SIZE = 0.15

BASE_FEATURE_COLS = [c for c in df_train.columns if c not in ["Id", TARGET]]

X_base = df_train[BASE_FEATURE_COLS].copy()
X_base["total_flags"] = X_base.sum(axis=1)

num_flag_cols = len(BASE_FEATURE_COLS)
X_base["mean_flag"] = X_base["total_flags"] / num_flag_cols

test_features_base = df_test[BASE_FEATURE_COLS].copy()
test_features_base["total_flags"] = test_features_base.sum(axis=1)
test_features_base["mean_flag"] = test_features_base["total_flags"] / num_flag_cols

X = X_base
y = df_train[TARGET]



## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True
)

rf = RandomForestRegressor(
    n_estimators=2500,  # increased
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=SEED,
    n_jobs=-1,
)

gbr = GradientBoostingRegressor(
    n_estimators=3000,  # increased
    learning_rate=0.008,  # finer learning
    max_depth=4,  # a touch deeper
    random_state=SEED,
)

extra = ExtraTreesRegressor(
    n_estimators=2500,  # increased
    max_depth=None,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

ridge_base = Ridge(alpha=1.0, random_state=SEED)  # unchanged

rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)
extra.fit(X_train, y_train)
ridge_base.fit(X_train, y_train)

rf_val_pred = rf.predict(X_val)
gbr_val_pred = gbr.predict(X_val)
extra_val_pred = extra.predict(X_val)
ridge_val_pred = ridge_base.predict(X_val)

rf_rmse = mean_squared_error(y_val, rf_val_pred, squared=False)
gbr_rmse = mean_squared_error(y_val, gbr_val_pred, squared=False)
extra_rmse = mean_squared_error(y_val, extra_val_pred, squared=False)
ridge_rmse = mean_squared_error(y_val, ridge_val_pred, squared=False)
print(f"Validation RMSE (RF): {rf_rmse:.4f}")
print(f"Validation RMSE (GBR): {gbr_rmse:.4f}")
print(f"Validation RMSE (ExtraTrees): {extra_rmse:.4f}")
print(f"Validation RMSE (Ridge base): {ridge_rmse:.4f}")

meta_X_val = np.column_stack(
    [
        rf_val_pred,
        gbr_val_pred,
        extra_val_pred,
        ridge_val_pred,
        X_val["total_flags"].values,
        X_val["mean_flag"].values,
    ]
)

linear_meta = Ridge(alpha=0.5, random_state=SEED)
linear_meta.fit(meta_X_val, y_val)

linear_val_pred = linear_meta.predict(meta_X_val)
linear_val_rmse = mean_squared_error(y_val, linear_val_pred, squared=False)
print(f"Validation RMSE (ridge stack with flags): {linear_val_rmse:.4f}")

inv_weights = np.array([1 / rf_rmse, 1 / gbr_rmse, 1 / extra_rmse])
inv_weights /= inv_weights.sum()
weighted_val_pred = (
    rf_val_pred * inv_weights[0]
    + gbr_val_pred * inv_weights[1]
    + extra_val_pred * inv_weights[2]
)
weighted_val_rmse = mean_squared_error(y_val, weighted_val_pred, squared=False)
print(
    f"Validation RMSE (inverse‑RMSE weighted blend of trees): {weighted_val_rmse:.4f}"
)

if weighted_val_rmse < linear_val_rmse:
    use_weighted = True
    best_val_rmse = weighted_val_rmse
    print("Using weighted blend for final predictions.")
else:
    use_weighted = False
    best_val_rmse = linear_val_rmse
    print("Using ridge stack for final predictions.")



## === cell 4
rf.fit(X, y)
gbr.fit(X, y)
extra.fit(X, y)
ridge_base.fit(X, y)

if not use_weighted:
    meta_X_full = np.column_stack(
        [
            rf.predict(X),
            gbr.predict(X),
            extra.predict(X),
            ridge_base.predict(X),
            X["total_flags"].values,
            X["mean_flag"].values,
        ]
    )
    linear_meta.fit(meta_X_full, y)



## === cell 5
test_features = test_features_base

rf_test_pred = rf.predict(test_features)
gbr_test_pred = gbr.predict(test_features)
extra_test_pred = extra.predict(test_features)
ridge_test_pred = ridge_base.predict(test_features)

if use_weighted:
    test_pred = (
        rf_test_pred * inv_weights[0]
        + gbr_test_pred * inv_weights[1]
        + extra_test_pred * inv_weights[2]
    )
else:
    meta_X_test = np.column_stack(
        [
            rf_test_pred,
            gbr_test_pred,
            extra_test_pred,
            ridge_test_pred,
            test_features["total_flags"].values,
            test_features["mean_flag"].values,
        ]
    )
    test_pred = linear_meta.predict(meta_X_test)

submission = pd.DataFrame({"Id": df_test["Id"], "Pawpularity": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
