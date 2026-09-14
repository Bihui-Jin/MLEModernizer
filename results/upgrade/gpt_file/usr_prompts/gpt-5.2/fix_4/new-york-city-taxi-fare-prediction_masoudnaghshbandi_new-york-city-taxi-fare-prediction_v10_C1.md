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

3.10

# 3. Installed packages

geopandas==0.14.4
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

3.77552

# 6. Current score

4.63478

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34119) has done: 'I fix the runtime errors by (1) correcting the undefined variable in RMSE computation, (2) properly using `StandardScaler` (fit on train, transform on val/test) without changing the model, and (3) keeping the `key` column for submission alignment while only dropping rows with missing feature values (not filtering test rows by geo bounds, which caused the length mismatch). I also ensure the submission file has exactly the required columns `key,fare_amount` and preserves the original test row order/length. These changes keep your core approach (basic cleaning + XGBoost regressor) intact while making the pipeline run end-to-end and produce a valid CSV.'
- What this solution (achieved 7.26096) has done: 'Your current RMSE (5.34119) is worse than the target (3.77552), so we should improve performance with minimal, low-risk edits while keeping the same XGBoost regressor approach. The biggest gains here typically come from adding a single, standard distance feature and a couple of simple time features from `pickup_datetime`, plus clipping obviously bad fare outliers in training; this keeps the same model/training loop and loss, but gives the model much more signal. I keep your existing cleaning and scaling pattern intact, just extend the feature set consistently for train/validation/test and keep submission alignment exactly as required. No early stopping, no architecture changes beyond adding informative features and mild target outlier filtering.'
- What this solution (achieved 4.63478) has done: 'Your current gap to the target is large (7.26096 vs 3.77552; lower is better), so we make small, low-risk improvements without changing the core approach (same XGBRegressor, same loss/fit/predict flow). The biggest issue is that you train on a filtered geo-bounded dataset but you do not apply the same bounds (and some other cleanups) to test, creating a distribution mismatch that hurts RMSE; we apply the same feature cleaning and geo-bounds consistently to both train and test (but keep test row count by imputing out-of-bounds rows instead of dropping them). We also add two very standard, lightweight features (`abs_lon_diff`, `abs_lat_diff`) derived from existing columns, keeping everything else the same, and we ensure feature columns are built from the processed training frame to avoid train/test feature drift. Submission writing stays identical and produces `submission1.csv` with the required `key,fare_amount` columns and original test order.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train.shape, test.shape



## === cell 4
train.head()



## === cell 5
train.isnull().sum()



## === cell 6
train = train.dropna(how="any", axis="rows")



## === cell 7
test.isnull().sum()



## === cell 8
train.head()



## === cell 9
train["fare_amount"].describe()



## === cell 10
train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 208].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 11
train.head()



## === cell 12
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()




## === cell 13
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    num_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    dt = pd.to_datetime(out["pickup_datetime"], errors="coerce", utc=True)
    out["pickup_hour"] = dt.dt.hour.astype("float32")
    out["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")

    out["abs_lon_diff"] = (
        (out["dropoff_longitude"] - out["pickup_longitude"]).abs().astype("float32")
    )
    out["abs_lat_diff"] = (
        (out["dropoff_latitude"] - out["pickup_latitude"]).abs().astype("float32")
    )

    R = 6371.0
    lat1 = np.radians(out["pickup_latitude"].astype("float64"))
    lon1 = np.radians(out["pickup_longitude"].astype("float64"))
    lat2 = np.radians(out["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(out["dropoff_longitude"].astype("float64"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    out["haversine_km"] = (2.0 * R * np.arcsin(np.sqrt(a))).astype("float32")

    return out


train = add_features(train)



## === cell 14
train.drop(["key"], axis=1, inplace=True)



## === cell 15
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 16
train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)



## === cell 17
train.head()



## === cell 18
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)



## === cell 19
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)



## === cell 20
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    seed=123,
)



## === cell 21
xgb_r.fit(X_train_s, y_train)



## === cell 22
y_pred = xgb_r.predict(X_test_s)



## === cell 23
rmse = float(np.sqrt(MSE(y_test, y_pred)))
print("RMSE : % f" % (rmse))



## === cell 24
test_for_model = test.copy()



## === cell 25
test_for_model.head()



## === cell 26
test_for_model = add_features(test_for_model)

feature_cols = X.columns.tolist()  # ensures exact train feature set/order is used

test_for_model[feature_cols] = test_for_model[feature_cols].apply(
    pd.to_numeric, errors="coerce"
)

test_for_model[feature_cols] = test_for_model[feature_cols].fillna(
    test_for_model[feature_cols].median(numeric_only=True)
)

pickup_oob = (
    (test_for_model["pickup_longitude"] < -75)
    | (test_for_model["pickup_longitude"] > -72)
    | (test_for_model["pickup_latitude"] < 40)
    | (test_for_model["pickup_latitude"] > 42)
)
dropoff_oob = (
    (test_for_model["dropoff_longitude"] < -75)
    | (test_for_model["dropoff_longitude"] > -72)
    | (test_for_model["dropoff_latitude"] < 40)
    | (test_for_model["dropoff_latitude"] > 42)
)

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
test_for_model.loc[pickup_oob | dropoff_oob, coord_cols] = np.nan

test_for_model[coord_cols] = test_for_model[coord_cols].fillna(
    pd.DataFrame(train[coord_cols]).median(numeric_only=True)
)
test_for_model = add_features(test_for_model)
test_for_model[feature_cols] = test_for_model[feature_cols].apply(
    pd.to_numeric, errors="coerce"
)
test_for_model[feature_cols] = test_for_model[feature_cols].fillna(
    pd.DataFrame(train[feature_cols]).median(numeric_only=True)
)



## === cell 27
test_for_model.head()



## === cell 28
test_X = test_for_model.drop(["key", "pickup_datetime"], axis=1)



## === cell 29
test_X_s = scaler.transform(test_X)



## === cell 30
new_pred = xgb_r.predict(test_X_s)



## === cell 31
submission = pd.DataFrame({"key": test["key"], "fare_amount": new_pred})



## === cell 32
submission.head()



## === cell 33
submission.to_csv("submission1.csv", index=False)



## === cell 34
print("Wrote submission1.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print("Nulls:", submission.isnull().sum().to_dict())
