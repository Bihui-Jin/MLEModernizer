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

20.46829

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna

import matplotlib.pyplot as plt
import seaborn as sns




## === cell 1
TRAIN_DATA_PATH = "../input/petfinder-pawpularity-score/train.csv"
TEST_DATA_PATH = "../input/petfinder-pawpularity-score/test.csv"




## === cell 2
TARGET_NAME = "Pawpularity"
VAL_SIZE = 0.15
SEED = 5
EARLY_ROUNDS = 50




## === cell 3
def set_seed(seed=42):
    """Utility function to use for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_display():
    """Function sets display options for charts and pd.DataFrames."""
    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = 12, 8
    plt.rcParams.update({"font.size": 14})
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.options.display.float_format = "{:.4f}".format


def get_features(df: pd.DataFrame) -> list:
    """Select input features from a DataFrame (exclude Id and target)."""
    return [c for c in df.columns if c not in ("Id", TARGET_NAME)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features based on the global `features` list."""
    df["features_sum"] = df[features].sum(axis=1) / len(features)

    for i in range(len(features) - 1):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            df[f"{f1}_{f2}"] = (df[f1] + df[f2]) / 2

    for i in range(len(features) - 2):
        for j in range(i + 1, len(features) - 1):
            for z in range(j + 1, len(features)):
                f1, f2, f3 = features[i], features[j], features[z]
                df[f"{f1}_{f2}_{f3}"] = (df[f1] + df[f2] + df[f3]) / 3
    return df


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def objective(trial):
    """Optuna objective – kept for completeness (not used in final run)."""
    global model
    params = {
        "tree_method": "hist",
        "predictor": "cpu_predictor",
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 250, 10_000, 250),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100, step=5),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.95),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    fit_params = dict(
        eval_set=[(valid_x, valid_y)],
        eval_metric="rmse",
        early_stopping_rounds=EARLY_ROUNDS,
        verbose=False,
    )

    model = XGBRegressor(**params)
    model.fit(train_x, train_y, **fit_params)
    val_rmse = rmse(valid_y, model.predict(valid_x))
    return val_rmse




## === cell 4
set_seed(SEED)
set_display()




## === cell 5
data_train = pd.read_csv(TRAIN_DATA_PATH)
print(f"Train data shape: {data_train.shape}")
data_train.head()




## === cell 6
data_test = pd.read_csv(TEST_DATA_PATH)
print(f"Test data shape: {data_test.shape}")
data_test.head()




## === cell 7
print(
    f"Target values: {data_train[TARGET_NAME].min()} - {data_train[TARGET_NAME].max()}\n"
    f"Mean value: {data_train[TARGET_NAME].mean()}\n"
    f"Median value: {data_train[TARGET_NAME].median()}\n"
    f"Standard deviation: {data_train[TARGET_NAME].std()}"
)

sns.histplot(data=data_train, x=TARGET_NAME, kde=True)
plt.axvline(data_train[TARGET_NAME].mean(), c="orange", ls="-", lw=3, label="Mean")
plt.axvline(data_train[TARGET_NAME].median(), c="green", ls="-", lw=3, label="Median")
plt.legend()
plt.title("Pawpularity Score")
plt.tight_layout()
plt.show()




## === cell 8
numeric_corr = data_train.drop(columns=["Id"]).corr()
ax = sns.heatmap(numeric_corr, center=0, annot=True, cmap="RdBu_r", fmt="0.3f")
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title("Correlation Matrix")
plt.show()
numeric_corr[TARGET_NAME].sort_values()




## === cell 9
neg_features = numeric_corr[numeric_corr[TARGET_NAME] < 0].index.tolist()
for col in neg_features:
    data_train[col] = (data_train[col] + 1) % 2
    data_test[col] = (data_test[col] + 1) % 2




## === cell 10
features = get_features(data_train)

data_train = add_features(data_train)
data_test = add_features(data_test)




## === cell 11
numeric_corr2 = data_train.drop(columns=["Id"]).corr()
numeric_corr2[TARGET_NAME].sort_values()




## === cell 12
y = data_train[TARGET_NAME]
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED
)
print(f"Train data shape: {train_x.shape}\n" f"Validation data shape: {valid_x.shape}")




## === cell 13
xgb_model = XGBRegressor(
    tree_method="hist",
    predictor="cpu_predictor",
    objective="reg:squarederror",
    booster="gbtree",
)
xgb_model.fit(
    train_x,
    train_y,
    eval_set=[(valid_x, valid_y)],
    eval_metric="rmse",
    early_stopping_rounds=EARLY_ROUNDS,
)




## === cell 14
importance = pd.DataFrame(
    {"features": features, "importance": xgb_model.feature_importances_}
)
importance.sort_values(by="importance", inplace=True)

plt.barh(range(len(importance)), importance["importance"])
plt.title("XGBoost Feature Importance")
plt.show()




## === cell 15
threshold = 0.0
importance = importance[importance["importance"] >= threshold]
plt.figure(figsize=(12, 16))
plt.barh(importance["features"], importance["importance"])
plt.title("XGBoost Feature Importance (filtered)")
plt.savefig("features.png", dpi=300)
plt.show()




## === cell 16
features = importance["features"].to_list()
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED
)
print(f"Re‑split with selected features: {train_x.shape}, {valid_x.shape}")




## === cell 17
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=SEED),
    direction="minimize",
    study_name="xgb_dummy",
)




## === cell 18
pass




## === cell 19
pass




## === cell 20
data_test[TARGET_NAME] = xgb_model.predict(data_test[features])
submission_path = "submission.csv"
data_test[["Id", TARGET_NAME]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
data_test[["Id", TARGET_NAME]].head()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1125863344.py in <cell line: 0>()
      1 # Predict on test set and write submission
----> 2 data_test[TARGET_NAME] = xgb_model.predict(data_test[features])
      3 submission_path = "submission.csv"
      4 data_test[["Id", TARGET_NAME]].to_csv(submission_path, index=False)
      5 print(f"Submission file written to {submission_path}")

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2416             data, fns, _ = _transform_pandas_df(data, enable_categorical)
   2417             if validate_features:
-> 2418                 self._validate_features(fns)
   2419         if _is_list(data) or _is_tuple(data):
   2420             data = np.array(data)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _validate_features(self, feature_names)
   2968                 )
   2969 
-> 2970             raise ValueError(msg.format(self.feature_names, feature_names))
   2971 
   2972     def get_split_value_histogram(

ValueError: feature_names mismatch: ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'] ['Group', 'Human', 'Collage', 'Eyes', 'Accessory', 'Blur', 'Info', 'Occlusion', 'Near', 'Action', 'Subject Focus', 'Face']
