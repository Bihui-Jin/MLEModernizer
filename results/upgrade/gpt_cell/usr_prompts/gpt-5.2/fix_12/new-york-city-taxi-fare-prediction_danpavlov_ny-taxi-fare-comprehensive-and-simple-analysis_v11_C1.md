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

3.7

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
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
xgboost==2.0.3

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

3.91427

# 6. Current score

6.15572

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.80865) has done: 'Diagnosis: Cell 35 crashes because in xgboost==2.0.3 the returned `Booster` no longer always exposes `best_ntree_limit`; early stopping now tracks the best iteration via `best_iteration` (and prediction can use `iteration_range`). The code attempts to access a removed/absent attribute, raising `AttributeError`.  
Patch summary: Update cell 35 to use `best_iteration` when available and pass `iteration_range` to `predict`, while keeping the same training call and early-stopping behavior. If no best iteration exists, fall back to plain prediction.  
Updated cells: Only cell 35 is changed.  
Compatibility notes for cell k+1: `XGBPredictions` is still created as a NumPy array and remains usable unchanged in cell 36.  
Assumptions: The model is trained with early stopping as written, so `best_iteration` is available; if not, the fallback path preserves prior behavior by predicting with all trees.'
- What this solution (achieved 5.81122) has done: 'I make two minimal, score-relevant fixes that usually improve RMSE for this competition without changing your overall modeling approach. First, I replace the very slow per-row `great_circle` loop with a vectorized haversine distance calculation that produces the same kind of distance feature but far more accurately and consistently (and ensures it’s created for both train and test without silent dtype issues). Second, I stop rounding predictions to 2 decimals before submission, because rounding almost always worsens RMSE and is not required by the competition format. Everything else (data loading size, cleaning rules, features used, train/test split, linear model, and XGBoost training with early stopping) stays the same.'
- What this solution (achieved 5.84725) has done: 'You’re currently far above the target RMSE (5.81 vs 3.91; lower is better), so we should improve model generalization with the smallest changes that keep your pipeline and modeling approach intact. The biggest score-relevant issue is that your XGBoost trains only against a tiny parameter set and evaluates on a random split without any randomness control; adding a few standard, conservative XGBoost regression parameters (still the same `xgb.train` workflow and loss/metric) usually yields a large RMSE drop on this competition. I also make the train/test split deterministic (so you can iterate reliably) and ensure we use `reg:squarederror` (modern equivalent of `reg:linear`) while keeping RMSE evaluation and early stopping identical. Finally, I write the submission to a `.csv` filename Kaggle accepts (your current filename lacks the `.csv` extension).'
- What this solution (achieved 6.55446) has done: 'Your current RMSE (5.847) is much worse than the target (3.914), so we should make the smallest changes that reliably improve generalization without changing your overall pipeline. The biggest low-risk gain here is to add a couple of standard NYC-taxi features (absolute deltas and simple Manhattan distance) alongside your existing haversine distance, while keeping the same train/test split, same model types, and same XGBoost training loop. I also add a conservative fare outlier cap (common for this dataset) to reduce RMSE impact from extreme labels, and fix the submission filename to end with `.csv` for Kaggle. Everything else (data sampling size, cleaning filters, LinearRegression baseline, XGBoost with early stopping) stays the same.'
- What this solution (achieved 6.14279) has done: 'Your RMSE (6.55) is far worse than the target (3.91; lower is better), so we need a small but meaningful generalization boost without changing your overall approach (same features, same XGBoost training loop). The most score-relevant minimal fix is to add standard regularization/robustness parameters to your existing `xgb.train` call (gamma, max_delta_step, and a stronger min_child_weight) and increase boosting rounds while keeping early stopping—this usually reduces RMSE on this competition without altering semantics. I also add a conservative post-processing clamp to keep predictions within the same fare range you trained on (0–250), which typically reduces RMSE impact from occasional extreme predictions. Finally, I keep your submission format but change the filename to end with `.csv` (Kaggle expects a CSV).'
- What this solution (achieved 6.2606) has done: 'Your current RMSE (6.14279; lower is better) is still far above the target (3.91427), so we need a small, reliable generalization boost without changing your overall pipeline. The biggest score-relevant gap in your current setup is that you train on only 500k rows and you ignore rich time and location structure; we keep the same XGBoost training loop but slightly increase the training sample size (still feasible) and add a few standard “minimal” taxi-fare features (datetime parts and NYC-airport distance flags) that are derived from existing columns. We also align train/test feature columns explicitly to avoid silent column/order mismatches and keep your existing outlier/geo filters, distance feature, and early-stopping behavior intact. The submission filename remain a valid `.csv`.'
- What this solution (achieved 9.41385) has done: 'Diagnosis: The crash happens inside `haversine_km` in cell 20 because `lon2`/`lat2` are passed as Python floats (e.g., `JFK[0]`) but the function assumes they are pandas Series/NumPy arrays and calls `.astype()` on them. This raises `AttributeError: 'float' object has no attribute 'astype'`.  
Patch summary: Make `haversine_km` robust to scalar `lon2`/`lat2` by converting all inputs to NumPy arrays with `np.asarray(..., dtype="float64")` before applying `np.radians`, preserving the same haversine calculation and output dtype.  
Updated cells: Only cell 20 is changed.  
Compatibility notes for cell k+1: `train` and `test` still receive the same added columns (`*_km`, `is_airport_trip`) with the same names and compatible dtypes, so `test.head()` in cell 21 works unchanged.  
Assumptions: Inputs columns exist and are numeric-compatible; broadcasting scalar airport coordinates across arrays is intended.'
- What this solution (achieved 7.0749) has done: 'Your current RMSE (9.41) is far worse than the target (3.91; lower is better), so we need a small change that improves generalization without changing the modeling approach. The biggest score issue in this script is that the features you feed the model exclude the raw pickup/dropoff coordinates, which are critical for this competition; adding them back is a minimal feature-list change and preserves your existing cleaning, engineered features, and XGBoost training loop. I also ensure any rows with invalid datetimes created by `errors="coerce"` are dropped so time-derived features don’t become NaNs (which can degrade model fit). Everything else (sampling size, feature engineering functions, train/test split, XGBoost params/early stopping, submission format) stays the same.'
- What this solution (achieved 6.25527) has done: 'We’re far above the target RMSE (7.0749 vs 3.9143; lower is better), so we should make a small, legitimate generalization improvement without changing your overall pipeline. The biggest score-relevant issue is your validation setup: you’re using a random split, which is a poor proxy for the leaderboard in this time-dependent dataset; switching to a time-based split keeps the same training loop and objective but typically reduces Kaggle RMSE noticeably. I also add `base_score` to XGBoost using the training mean (a standard calibration tweak that doesn’t change the approach) and ensure we don’t drop any test rows due to datetime parsing (keeping submission row count correct). Everything else (features, cleaning rules, XGBoost training call with early stopping, and submission schema) stays the same.'
- What this solution (achieved 6.15572) has done: 'Your current RMSE (6.255) is far above the target (3.914; lower is better), so we should make a small but high-impact generalization improvement without changing your overall pipeline. The biggest issue is that the model is learning from raw longitude/latitude degrees (and degree-based manhattan distance), which are poorly scaled versus kilometer-based distances; we add two minimal, standard engineered features in kilometers (`delta_lon_km`, `delta_lat_km`) and a kilometer Manhattan distance (`manhattan_km`) while keeping your existing features and XGBoost training loop intact. We also make sure these new features are created for both train and test and automatically included in the existing `X.columns` feature set. Finally, we keep the same submission schema but write to a `.csv` filename that’s clearly valid for Kaggle.'
- What this solution (achieved 6.15572) has done: 'We need to reduce RMSE from 6.15572 toward 3.91427 (lower is better), so the smallest high-impact change is to remove a clear train/test feature mismatch bug: you reindex `test_pred` to `X.columns` (built from the unsorted full train), but you actually train on `X_sorted`, which can have a different column order—this silently degrades predictions. I make `test_pred` reindex to `X_train.columns` (same as training) and also ensure the DMatrix uses the same feature names. As a second minimal, score-relevant fix, I align prediction clipping to a more realistic lower bound (NYC fares can’t be 0; clipping to 2.5 usually helps RMSE slightly without changing the modeling approach). Everything else (sampling size, cleaning, features, time-based split, and XGBoost training loop/params) remains unchanged and it still writes a valid submission CSV.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=2000000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.distplot(train["fare_amount"])



## === cell 9
sns.distplot(train["passenger_count"])



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)]

train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    R = 6371.0088  # mean Earth radius in km
    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = (R * c).astype("float32")




## === cell 15
dist_calc(train)
dist_calc(test)




## === cell 16
def add_geo_features(df):
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )
    df["manhattan_deg"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")

    lat_mean_rad = np.radians(
        (
            (
                df["pickup_latitude"].astype("float64")
                + df["dropoff_latitude"].astype("float64")
            )
            / 2.0
        ).values
    )
    km_per_deg_lat = 111.32  # ~km per degree latitude
    km_per_deg_lon = (111.32 * np.cos(lat_mean_rad)).astype("float64")

    df["delta_lat_km"] = (df["abs_lat_diff"].astype("float64") * km_per_deg_lat).astype(
        "float32"
    )
    df["delta_lon_km"] = (df["abs_lon_diff"].astype("float64") * km_per_deg_lon).astype(
        "float32"
    )
    df["manhattan_km"] = (df["delta_lat_km"] + df["delta_lon_km"]).astype("float32")
    return df


train = add_geo_features(train)
test = add_geo_features(test)



## === cell 17
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 18
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 19
train = train.dropna(subset=["pickup_datetime"]).copy()

_dt_fill = train["pickup_datetime"].median()
test["pickup_datetime"] = test["pickup_datetime"].fillna(_dt_fill)

for df in (train, test):
    df["year"] = df.pickup_datetime.dt.year.astype("int16")
    df["month"] = df.pickup_datetime.dt.month.astype("int8")
    df["dayofweek"] = df.pickup_datetime.dt.dayofweek.astype("int8")
    df["hour"] = df.pickup_datetime.dt.hour.astype("int8")




## === cell 20
def add_airport_features(df):
    JFK = (-73.7781, 40.6413)
    LGA = (-73.8740, 40.7769)
    EWR = (-74.1745, 40.6895)
    MAN = (-73.9857, 40.7484)

    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0088
        lon1 = np.radians(np.asarray(lon1, dtype="float64"))
        lat1 = np.radians(np.asarray(lat1, dtype="float64"))
        lon2 = np.radians(np.asarray(lon2, dtype="float64"))
        lat2 = np.radians(np.asarray(lat2, dtype="float64"))
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return (R * c).astype("float32")

    df["pickup_jfk_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], JFK[0], JFK[1]
    )
    df["dropoff_jfk_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], JFK[0], JFK[1]
    )
    df["pickup_lga_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], LGA[0], LGA[1]
    )
    df["dropoff_lga_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], LGA[0], LGA[1]
    )
    df["pickup_ewr_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], EWR[0], EWR[1]
    )
    df["dropoff_ewr_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], EWR[0], EWR[1]
    )
    df["pickup_man_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], MAN[0], MAN[1]
    )
    df["dropoff_man_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], MAN[0], MAN[1]
    )

    df["is_airport_trip"] = (
        (
            (df["pickup_jfk_km"] < 2.0)
            | (df["dropoff_jfk_km"] < 2.0)
            | (df["pickup_lga_km"] < 2.0)
            | (df["dropoff_lga_km"] < 2.0)
            | (df["pickup_ewr_km"] < 2.0)
            | (df["dropoff_ewr_km"] < 2.0)
        )
    ).astype("uint8")
    return df


train = add_airport_features(train)
test = add_airport_features(test)



## === cell 21
test.head()



## === cell 22
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 23
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
    ],
    axis=1,
)
y = train["fare_amount"]



## === cell 24
X.head()



## === cell 25
y.head()



## === cell 26
train_sorted = train.sort_values("pickup_datetime").reset_index(drop=True)
X_sorted = train_sorted.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y_sorted = train_sorted["fare_amount"]

split_idx = int(len(train_sorted) * 0.8)
X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]



## === cell 27
test_pred = test.drop(
    [
        "key",
        "pickup_datetime",
    ],
    axis=1,
)

test_pred = test_pred.reindex(columns=X_train.columns, fill_value=0)



## === cell 28
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 29
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 30
LinearPredictions = lm.predict(test_pred)
LinearPredictions



## === cell 31
LinearPredictions.size



## === cell 32
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 33
linear_submission.head()




## === cell 34
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    base_score = float(np.mean(y_train))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,  # slightly smaller step; compensated by more rounds + early stopping
        "max_depth": 8,
        "min_child_weight": 5,  # more conservative splits -> often better generalization on NYC taxi
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,  # requires gain to split -> reduces noisy splits
        "max_delta_step": 1,  # stabilizes updates for regression with outliers
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
        "verbosity": 0,
        "base_score": base_score,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,  # more capacity; early stopping prevents over-training
        early_stopping_rounds=50,  # same concept as before; gives a bit more room to settle
        evals=[(dtest, "test")],
    )




## === cell 35
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dtest_pred = xgb.DMatrix(test_pred, feature_names=list(test_pred.columns))

if hasattr(xgbm, "best_iteration") and xgbm.best_iteration is not None:
    XGBPredictions = xgbm.predict(
        dtest_pred, iteration_range=(0, xgbm.best_iteration + 1)
    )
else:
    XGBPredictions = xgbm.predict(dtest_pred)

XGBPredictions = np.clip(XGBPredictions, 2.5, 250.0).astype(np.float32)



## === cell 36
XGBPredictions



## === cell 37
XGBPredictions



## === cell 38
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 39
submission = XGB_submission



## === cell 40
submission.to_csv("XGBSubmission16082018.csv", index=False)
