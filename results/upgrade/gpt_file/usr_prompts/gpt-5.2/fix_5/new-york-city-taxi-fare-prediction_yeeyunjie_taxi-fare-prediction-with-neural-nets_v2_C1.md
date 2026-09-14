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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
statsmodels==0.14.5
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

5.07016

# 6. Current score

23.47694

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 954.05739) has done: 'I fix the notebook-breaking issues without changing the core modeling approach: remove Jupyter magics, avoid importing Keras at import-time (it triggers the protobuf `MessageFactory/GetPrototype` crash in this environment), and correct the Pandas `.any(1)` calls to `.any(axis=1)` so the data-cleaning pipeline runs. I also fix the Haversine distance function (it currently applies `math.radians` to Series and fail) by implementing a vectorized numpy version that preserves the same feature intent but runs fast on 1M rows. Finally, I ensure the script trains the same Ridge baseline (already present) and writes a valid `submission.csv` with the exact required columns and a `.csv` suffix in `/kaggle/working/`.'
- What this solution (achieved 22.44773) has done: 'I fix the crash in the coordinate outlier dropping step by making the second `.drop()` ignore already-removed indices (the current KeyError stops the pipeline before training/inference). Then, because your current RMSE (~954) is far from the target (~5.07), I make one minimal, score-critical correction that preserves the same Ridge+scaling core logic: train only on the NYC-bounded, cleaned dataframe (`df1`) instead of the pre-bbox `df2`, so train/test feature distributions match and predictions aren’t wildly out-of-domain. I also ensure the test set gets the same bbox/validity filters (without dropping rows) by clipping coordinates into the training bbox prior to distance computation so submission length/order stays identical. The rest of the model, features, and training approach remain unchanged, and the script still write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 23.2323) has done: 'Your current Ridge+scaled baseline is underperforming mainly because it’s learning on a very noisy label distribution; the biggest RMSE win (without changing the model/feature set) is to remove extreme “distance outlier” trips that survive the NYC bbox but still create huge fares relative to computed distance. I add one robust cleaning step that drops training rows with implausibly large `distance` (and also `distance<=0`) after the distance feature is computed, keeping the exact same Ridge/StandardScaler pipeline and inference. This should pull the RMSE down substantially toward the 5.07 target while staying within your existing logic. The submission format, order, paths, and model remain unchanged.'
- What this solution (achieved 23.47694) has done: 'Your current RMSE (23.23) is far worse than the target (5.07), so we should improve meaningfully without changing the Ridge+StandardScaler core pipeline. The biggest likely issue left is label noise from remaining extreme but “legal” trips; we add one more minimal, competition-standard cleaning step: drop training rows with implausible high fare-per-km ratio (after distance is computed), which preserves your existing features/model but removes influential outliers that blow up RMSE. We also make the train/test processing more consistent by applying the same datetime rounding to test (already done) and ensuring no NaNs/inf can enter the scaler by filtering them out in train only (test rows are kept; any invalid values are safely filled). The submission format/path remain unchanged and a valid `/kaggle/working/submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test_df = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("Train loaded:", df.shape, "Test loaded:", test_df.shape, "Sample:", sample.shape)
print(df.head(2))
print(test_df.head(2))



## === cell 2
print(f"Number of records: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")



## === cell 3
df.info()



## === cell 4
df.describe()



## === cell 5
df.head()



## === cell 6
null_rows = df[df.isnull().any(axis=1)]
print("Rows with any nulls:", null_rows.shape[0])



## === cell 7
df.columns[df.isnull().any()]



## === cell 8
df1 = df[~df.isnull().any(axis=1)].copy()
print("After dropping null rows:", df1.shape)



## === cell 9
incorrect_location = df1[
    ((df1["dropoff_latitude"] < 0) | (df1["pickup_latitude"] < 0))
    & ((df1["dropoff_longitude"] > 0) | (df1["pickup_longitude"] > 0))
].copy()
print("Incorrect-location rows:", incorrect_location.shape)



## === cell 10
if len(incorrect_location) > 0:
    incorrect_location = incorrect_location.rename(
        columns={
            "pickup_latitude": "pickup_longitude",
            "pickup_longitude": "pickup_latitude",
            "dropoff_latitude": "dropoff_longitude",
            "dropoff_longitude": "dropoff_latitude",
        }
    )
    incorrect_location = incorrect_location[df1.columns]

    df1.loc[
        df1.index.isin(incorrect_location.index),
        [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ],
    ] = incorrect_location[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values



## === cell 11
remaining_odd = df1[
    ((df1["dropoff_latitude"] < 0) | (df1["pickup_latitude"] < 0))
    & ((df1["dropoff_longitude"] > 0) | (df1["pickup_longitude"] > 0))
]
print("Remaining odd coord rows (will drop):", remaining_odd.shape)



## === cell 12
todrop = remaining_odd
df1 = df1[~df1.index.isin(todrop.index)].copy()
print("After dropping remaining odd coords:", df1.shape)



## === cell 13
df1 = df1.drop(
    df1[
        (df1["dropoff_latitude"] == 0)
        & (df1["dropoff_longitude"] == 0)
        & (df1["pickup_latitude"] == 0)
        & (df1["pickup_longitude"] == 0)
    ].index
).copy()
print("After dropping (0,0) coords:", df1.shape)



## === cell 14
df1.head()



## === cell 15
todrop_lat = df1[
    (df1["pickup_latitude"].lt(24) | df1["pickup_latitude"].gt(50))
    | (df1["dropoff_latitude"].lt(24) | df1["dropoff_latitude"].gt(50))
]
print("Odd latitude rows:", todrop_lat.shape)



## === cell 16
todrop_lon = df1[
    (df1["pickup_longitude"].lt(-125) | df1["pickup_longitude"].gt(-67))
    | (df1["dropoff_longitude"].lt(-125) | df1["dropoff_longitude"].gt(-67))
]
print("Odd longitude rows:", todrop_lon.shape)



## === cell 17
df1 = (
    df1.drop(todrop_lat.index, errors="ignore")
    .drop(todrop_lon.index, errors="ignore")
    .copy()
)
print("After dropping odd lat/lon:", df1.shape)



## === cell 18
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.boxplot(x=test_df["pickup_latitude"], ax=ax[0])
sns.boxplot(x=test_df["pickup_longitude"], ax=ax[1])
plt.tight_layout()
plt.show()



## === cell 19
test_df.describe()



## === cell 20
df1 = df1[
    ((df1["pickup_longitude"] > -75) & (df1["pickup_longitude"] < -72))
    & ((df1["pickup_latitude"] > 40) & (df1["pickup_latitude"] < 42))
    & ((df1["dropoff_longitude"] > -75) & (df1["dropoff_longitude"] < -72))
    & ((df1["dropoff_latitude"] > 40) & (df1["dropoff_latitude"] < 42))
].copy()
print("After NYC bbox filter:", df1.shape)



## === cell 21
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.boxplot(x=df1["pickup_latitude"], ax=ax[0])
sns.boxplot(x=df1["pickup_longitude"], ax=ax[1])
plt.tight_layout()
plt.show()



## === cell 22
df1 = df1.drop(df1[df1["fare_amount"] <= 0].index).copy()
print("After dropping non-positive fares:", df1.shape)



## === cell 23
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x=df1["passenger_count"])
plt.tight_layout()
plt.show()



## === cell 24
df1[df1["passenger_count"] > 50].head()



## === cell 25
df1 = df1.drop(df1[df1["passenger_count"] > 50].index).copy()
print("After dropping passenger_count>50:", df1.shape)



## === cell 26
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x=df1["fare_amount"])
plt.tight_layout()
plt.show()



## === cell 27
df1 = df1.drop(df1[df1["fare_amount"] > 200].index).copy()
print("After dropping fare_amount>200:", df1.shape)



## === cell 28
pd.to_datetime(
    pd.to_datetime(df1.head()["pickup_datetime"]).dt.strftime("%Y-%m-%d %H:%M")
)



## === cell 29
df1["pickup_datetime"] = pd.to_datetime(
    pd.to_datetime(df1["pickup_datetime"]).dt.strftime("%Y-%m-%d %H:%M")
)
test_df["pickup_datetime"] = pd.to_datetime(
    pd.to_datetime(test_df["pickup_datetime"]).dt.strftime("%Y-%m-%d %H:%M")
)



## === cell 30
df1["year"] = df1["pickup_datetime"].dt.year
df1["month"] = df1["pickup_datetime"].dt.month
df1["day"] = df1["pickup_datetime"].dt.day
df1["weekday"] = df1["pickup_datetime"].dt.weekday
df1["hour"] = df1["pickup_datetime"].dt.hour
df1["min"] = df1["pickup_datetime"].dt.minute

test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["weekday"] = test_df["pickup_datetime"].dt.weekday
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["min"] = test_df["pickup_datetime"].dt.minute




## === cell 31
def haversine_np(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    """
    Vectorized haversine distance (km).
    Inputs are arrays/Series in decimal degrees.
    """
    lon1 = np.radians(pickup_lon.astype(float))
    lat1 = np.radians(pickup_lat.astype(float))
    lon2 = np.radians(dropoff_lon.astype(float))
    lat2 = np.radians(dropoff_lat.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r




## === cell 32
df1["distance"] = haversine_np(
    df1["pickup_longitude"],
    df1["pickup_latitude"],
    df1["dropoff_longitude"],
    df1["dropoff_latitude"],
)
print(df1[["distance"]].describe())



## === cell 33
before = df1.shape[0]
df1 = df1[(df1["distance"] > 0) & (df1["distance"] <= 100)].copy()
print(
    f"After dropping distance outliers (0 or >100km): {df1.shape} (dropped {before - df1.shape[0]})"
)



## === cell 34
df1["fare_per_km"] = df1["fare_amount"] / np.maximum(df1["distance"], 1e-6)
before = df1.shape[0]
df1 = df1[(df1["fare_per_km"] > 0) & (df1["fare_per_km"] <= 50)].copy()
df1 = df1.drop(columns=["fare_per_km"])
print(
    f"After dropping fare_per_km outliers (>50): {df1.shape} (dropped {before - df1.shape[0]})"
)



## === cell 35
df2 = df1.copy()
df2 = df2.drop(
    columns=[
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
)
print("df2 columns:", df2.columns.tolist())



## === cell 36
test_df = test_df.copy()
for col, lo, hi in [
    ("pickup_longitude", -75.0, -72.0),
    ("dropoff_longitude", -75.0, -72.0),
    ("pickup_latitude", 40.0, 42.0),
    ("dropoff_latitude", 40.0, 42.0),
]:
    test_df[col] = test_df[col].clip(lower=lo, upper=hi)

test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)
test_df = test_df.drop(
    columns=[
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
)
print("test_df columns:", test_df.columns.tolist())



## === cell 37
corr = df2.drop(columns=["key"]).corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, ax=ax, annot=True, cmap=cmap, mask=mask, fmt=".2f")
plt.tight_layout()
plt.show()



## === cell 38
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.violinplot(y=df2["fare_amount"], x=df2["year"], ax=ax[0])
sns.violinplot(y=df2["fare_amount"], x=df2["month"], ax=ax[1])
plt.tight_layout()
plt.show()



## === cell 39
fig, ax = plt.subplots(2, figsize=(12, 6))
sns.barplot(y=df2["fare_amount"], x=df2["year"], ax=ax[0], palette="Set2")
sns.barplot(y=df2["fare_amount"], x=df2["month"], ax=ax[1], palette="Set2")
plt.tight_layout()
plt.show()



## === cell 40
pass



## === cell 41
pass



## === cell 42
X = df2.drop(columns=["fare_amount", "key", "pickup_datetime"])
y = df2["fare_amount"].astype(float)



## === cell 43
mask_finite = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
before = X.shape[0]
X = X.loc[mask_finite].copy()
y = y.loc[mask_finite].copy()
print(
    f"After dropping non-finite training rows: {X.shape} (dropped {before - X.shape[0]})"
)



## === cell 44
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=RANDOM_STATE)



## === cell 45
print(f"train: {X_train.shape}")
print(f"train target: {y_train.shape}")
print(f"val: {X_valid.shape}")
print(f"val target: {y_valid.shape}")



## === cell 46
X_train.head()



## === cell 47
ss = StandardScaler()
ss.fit(X_train)
X_train_ss = ss.transform(X_train)
X_valid_ss = ss.transform(X_valid)

ridge_tmp = Ridge(alpha=10)
ridge_tmp.fit(X_train_ss, y_train)
valid_pred = ridge_tmp.predict(X_valid_ss)
rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
print("Validation RMSE (Ridge alpha=10):", rmse)



## === cell 48
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    ElasticNet,
    HuberRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def get_models(models=dict()):
    models["lr"] = LinearRegression()
    models["lasso"] = Lasso()
    models["ridge"] = Ridge()
    models["en"] = ElasticNet()
    models["huber"] = HuberRegressor()
    models["pa"] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    return models


def get_models_nl(models=dict()):
    models["svr"] = SVR()
    n_trees = 100
    models["ada"] = AdaBoostRegressor(n_estimators=n_trees)
    models["bag"] = BaggingRegressor(n_estimators=n_trees)
    models["rf"] = RandomForestRegressor(
        n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1
    )
    models["et"] = ExtraTreesRegressor(
        n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=-1
    )
    models["gbm"] = GradientBoostingRegressor(
        n_estimators=n_trees, random_state=RANDOM_STATE
    )
    return models


def evaluate_models(models, X_train_ss, y_train, X_test_ss, y_test):
    for name, model in models.items():
        model_fit = model.fit(X_train_ss, y_train)
        train_preds = model_fit.predict(X_train_ss)
        test_preds = model_fit.predict(X_test_ss)
        train_mse = mean_squared_error(y_train, train_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        print(f"{name}:")
        print(f"----")
        print(f"Train MSE: {round(train_mse, 2)}")
        print(f"Test MSE: {round(test_mse, 2)}\n")


def params(model):
    if model == "lasso":
        return {"alpha": [0.01, 0.1, 1, 2, 5, 10]}
    elif model == "ridge":
        return {"alpha": [0.01, 0.1, 1, 2, 5, 10]}
    elif model == "en":
        return {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.6]}
    elif model == "svr":
        return {
            "kernel": ["rbf", "linear", "poly"],
            "C": [1, 20, 50, 100],
            "gamma": ["scale", "auto"],
            "epsilon": [0.1, 1, 10],
        }
    elif model == "ada":
        return {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 1]}
    elif model == "bag":
        return {
            "n_estimators": [20, 50, 100, 150],
            "max_features": [2, 4, 6],
            "max_samples": [0.1, 0.2, 0.3, 0.5, 0.7],
            "bootstrap": [True],
        }
    elif model == "rf":
        return {
            "bootstrap": [True],
            "max_depth": [5, 10, 15],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4],
            "min_samples_split": [2, 3, 4],
            "n_estimators": [50, 200, 300],
            "random_state": [RANDOM_STATE],
        }
    elif model == "et":
        return {
            "bootstrap": [True],
            "max_depth": [5, 10, 15],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4],
            "min_samples_split": [2, 3, 4],
            "n_estimators": [50, 200, 300],
            "random_state": [RANDOM_STATE],
        }
    elif model == "gbm":
        return {
            "learning_rate": [0.1, 0.3, 0.6, 1],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [2, 3, 4],
            "max_depth": [3, 5, 8],
        }
    else:
        return {}


def grid_search_rs(model_name, models, X_train, y_train, X_test, y_test):
    pipe_params = params(model_name)
    model = models[model_name]
    if len(pipe_params) == 0:
        raise ValueError(f"No params defined for model '{model_name}'")
    gs = RandomizedSearchCV(
        model,
        param_distributions=pipe_params,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=True,
        n_jobs=8,
        random_state=RANDOM_STATE,
    )
    gs.fit(X_train, y_train)
    train_score = gs.score(X_train, y_train)
    test_score = gs.score(X_test, y_test)

    print(f"Results from: {model_name}")
    print(f"-----------------------------------")
    print(f"Best Hyperparameters: {gs.best_params_}")
    print(f"Mean MSE: {-round(gs.best_score_, 4)}")
    print(f"Train MSE: {-round(train_score, 4)}")
    print(f"Test MSE: {-round(test_score, 4)}")
    print(" ")
    return gs




## === cell 49
models = get_models()
evaluate_models(models, X_train_ss, y_train, X_valid_ss, y_valid)



## === cell 50
pass



## === cell 51
pass



## === cell 52
pass



## === cell 53
pass



## === cell 54
pass



## === cell 55
pass



## === cell 56
ss_full = StandardScaler()
ss_full.fit(X)
X_ss = ss_full.transform(X)

test_features = test_df.drop(columns=["key", "pickup_datetime"])
test_features = test_features[X.columns]

test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

test_df_ss = ss_full.transform(test_features)



## === cell 57
print(f"Shape of X: {X_ss.shape}")
print(f"Shape of y: {y.shape}")
print("Test features shape:", test_df_ss.shape)



## === cell 58
ridge = Ridge(alpha=10)
ridge.fit(X_ss, y)
ridge_preds = ridge.predict(test_df_ss)

ridge_preds = np.clip(ridge_preds, 0.0, None)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": ridge_preds})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())



## === cell 59
check = pd.read_csv("/kaggle/working/submission.csv")
print("Submission loaded back:", check.shape)
print(check.columns.tolist())
print(check.head())



## === cell 60
pass



## === cell 61
pass
