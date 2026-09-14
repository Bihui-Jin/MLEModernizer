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

3.623

# 6. Current score

4.91848

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.50328) has done: 'I remove the deprecated use of `best_ntree_limit` and the old `reg:linear` objective, updating the XGBoost training function to return a Booster that can predict directly. I also guard the Jupyter magic line so the script runs in a plain Python environment. These fixes eliminate the runtime errors, allow predictions to be generated, and ensure a proper CSV submission file is written.'
- What this solution (achieved 7.39037) has done: 'I keep the overall pipeline unchanged but improve the XGBoost model by using a richer set of hyper‑parameters (deeper trees, learning rate, subsampling) and a slightly higher early‑stopping patience. After predicting I also clip any negative fares to zero. These small tweaks are expected to lower the RMSE toward the target without altering the core logic.'
- What this solution (achieved 9.58233) has done: 'I increase the training sample size, add cyclic hour/weekday/month features (sin / cos), and slightly strengthen the XGBoost model (deeper trees, lower learning rate) so the validation RMSE moves closer to the target while leaving the overall pipeline unchanged.'
- What this solution (achieved 5.16198) has done: 'The changes vectorize the costly haversine loops, replacing `iterrows` with fast NumPy operations for both the pickup‑dropoff distance and distances to the airports/Manhattan. This keeps the exact same feature values while cutting runtime from minutes to seconds. Additionally, the XGBoost parameters now include `tree_method="hist"` which speeds up training without altering the model’s logic or predictions. All other code and paths remain unchanged.'
- What this solution (achieved 5.04284) has done: 'I added robust handling for missing and infinite values after feature engineering, filling NaNs with column medians so both LinearRegression and XGBoost can operate without errors. I also ensured that predicted fares are non‑negative by clipping LinearRegression outputs. No core modeling logic was changed.'
- What this solution (achieved 6.79145) has done: 'I train the XGBoost model directly on the original fare amounts instead of on a log‑transformed target. This small change keeps the overall pipeline unchanged but aligns the loss function with the competition’s RMSE metric, which should reduce the validation error and move the score closer to the target. I also adjust the prediction step to use the raw model output (no exp‑transform) and keep the existing clipping and rounding.'
- What this solution (achieved 5.04284) has done: 'I added the missing imports, ensured that the data is loaded before any processing, and kept the original feature‑engineering and model pipeline unchanged. The script now runs end‑to‑end, creates the predictions with the XGBoost model, and writes a valid `submission.csv` file (named `XGBSubmission23082018_2M.csv`) containing the required `key` and `fare_amount` columns.'
- What this solution (achieved 4.91848) has done: 'We keep the overall workflow and feature engineering unchanged, but limit the maximum number of XGBoost boosting rounds. The original 5000‑round limit forces the library to iterate many times even though early stopping usually stops far earlier; reducing this ceiling to a smaller value (e.g., 1500) removes unnecessary work while preserving the same training logic, early‑stopping behavior, and model architecture.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xgboost as xgb




## === cell 1
print(os.listdir("../input"))




## === cell 2
test = pd.read_csv("../input/test.csv")




## === cell 3
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train = pd.read_csv("../input/train.csv", nrows=5_000_000, dtype=types)




## === cell 4
train.head()




## === cell 5
train.describe()




## === cell 6
train.isnull().sum()




## === cell 7
train.dropna(inplace=True)




## === cell 8
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]




## === cell 9
train.describe()




## === cell 10
def quick_dist_calc(df):
    """Compute haversine distance between pickup and drop‑off.
    Caches radian conversions in temporary columns to avoid recomputation."""
    if "pickup_lat_rad" not in df.columns:
        df["pickup_lat_rad"] = np.radians(df["pickup_latitude"].astype("float32"))
        df["pickup_lon_rad"] = np.radians(df["pickup_longitude"].astype("float32"))
        df["dropoff_lat_rad"] = np.radians(df["dropoff_latitude"].astype("float32"))
        df["dropoff_lon_rad"] = np.radians(df["dropoff_longitude"].astype("float32"))

    R = 6373.0
    dlat = df["dropoff_lat_rad"] - df["pickup_lat_rad"]
    dlon = df["dropoff_lon_rad"] - df["pickup_lon_rad"]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(df["pickup_lat_rad"])
        * np.cos(df["dropoff_lat_rad"])
        * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance"] = R * c


def quick_dist_calc_loc(df, latc, lonc, cname):
    """Compute distance from a fixed location (latc,lonc) to both pickup and drop‑off.
    Uses cached radian columns when available."""
    if "pickup_lat_rad" not in df.columns:
        df["pickup_lat_rad"] = np.radians(df["pickup_latitude"].astype("float32"))
        df["pickup_lon_rad"] = np.radians(df["pickup_longitude"].astype("float32"))
        df["dropoff_lat_rad"] = np.radians(df["dropoff_latitude"].astype("float32"))
        df["dropoff_lon_rad"] = np.radians(df["dropoff_longitude"].astype("float32"))

    R = 6373.0
    latc_rad = np.radians(latc)
    lonc_rad = np.radians(lonc)

    dlat1 = latc_rad - df["pickup_lat_rad"]
    dlon1 = lonc_rad - df["pickup_lon_rad"]
    a1 = (
        np.sin(dlat1 / 2) ** 2
        + np.cos(latc_rad) * np.cos(df["pickup_lat_rad"]) * np.sin(dlon1 / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    dist_pickup = R * c1

    dlat2 = latc_rad - df["dropoff_lat_rad"]
    dlon2 = lonc_rad - df["dropoff_lon_rad"]
    a2 = (
        np.sin(dlat2 / 2) ** 2
        + np.cos(latc_rad) * np.cos(df["dropoff_lat_rad"]) * np.sin(dlon2 / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    dist_dropoff = R * c2

    df[f"{cname}_pickup_dist"] = dist_pickup
    df[f"{cname}_dropoff_dist"] = dist_dropoff




## === cell 11
quick_dist_calc(train)
quick_dist_calc(test)




## === cell 12
jfk_airport = (-73.785193, 40.645972)
laguardia_airport = (-73.872925, 40.773335)
newark_airport = (-74.184156, 40.692764)
manhattan = (-73.983132, 40.759006)

quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(train, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(train, manhattan[1], manhattan[0], "manhattan")

quick_dist_calc_loc(test, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    test, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(test, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(test, manhattan[1], manhattan[0], "manhattan")




## === cell 13
train["jfk_distance"] = np.minimum(
    train["jfk_airport_pickup_dist"], train["jfk_airport_dropoff_dist"]
)
train["laguardia_distance"] = np.minimum(
    train["laguardia_airport_pickup_dist"], train["laguardia_airport_dropoff_dist"]
)
train["newark_distance"] = np.minimum(
    train["newark_airport_pickup_dist"], train["newark_airport_dropoff_dist"]
)
train["manhattan_distance"] = np.minimum(
    train["manhattan_pickup_dist"], train["manhattan_dropoff_dist"]
)

test["jfk_distance"] = np.minimum(
    test["jfk_airport_pickup_dist"], test["jfk_airport_dropoff_dist"]
)
test["laguardia_distance"] = np.minimum(
    test["laguardia_airport_pickup_dist"], test["laguardia_airport_dropoff_dist"]
)
test["newark_distance"] = np.minimum(
    test["newark_airport_pickup_dist"], test["newark_airport_dropoff_dist"]
)
test["manhattan_distance"] = np.minimum(
    test["manhattan_pickup_dist"], test["manhattan_dropoff_dist"]
)




## === cell 14
train.drop(
    [
        "jfk_airport_pickup_dist",
        "jfk_airport_dropoff_dist",
        "laguardia_airport_pickup_dist",
        "laguardia_airport_dropoff_dist",
        "newark_airport_pickup_dist",
        "newark_airport_dropoff_dist",
        "manhattan_pickup_dist",
        "manhattan_dropoff_dist",
    ],
    axis=1,
    inplace=True,
)

test.drop(
    [
        "jfk_airport_pickup_dist",
        "jfk_airport_dropoff_dist",
        "laguardia_airport_pickup_dist",
        "laguardia_airport_dropoff_dist",
        "newark_airport_pickup_dist",
        "newark_airport_dropoff_dist",
        "manhattan_pickup_dist",
        "manhattan_dropoff_dist",
    ],
    axis=1,
    inplace=True,
)




## === cell 15
for df in (train, test):
    rad_cols = [c for c in df.columns if c.endswith("_rad")]
    df.drop(columns=rad_cols, inplace=True)

train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

for df in [train, test]:
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["distance_per_passenger"] = train["distance"] / train["passenger_count"]
test["distance_per_passenger"] = test["distance"] / test["passenger_count"]

train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

median_vals = train.median(numeric_only=True)
train.fillna(median_vals, inplace=True)
test.fillna(median_vals, inplace=True)




## === cell 16
train.head()




## === cell 17
test.head()




## === cell 18
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]




## === cell 19
X.head()




## === cell 20
y.head()




## === cell 21
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## === cell 22
test_pred = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 23
lm = LinearRegression()
lm.fit(X_train, y_train)
print("Linear train R^2:", lm.score(X_train, y_train))
print("Linear val   R^2:", lm.score(X_val, y_val))




## === cell 24
y_pred_lr_val = lm.predict(X_val)
lrmse_val = np.sqrt(metrics.mean_squared_error(y_pred_lr_val, y_val))
print("Linear RMSE on validation set:", lrmse_val)




## === cell 25
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.maximum(LinearPredictions, 0)  # no negative fares
LinearPredictions = np.round(LinearPredictions, 2)




## === cell 26
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)




## === cell 27
def XGBoost(X_train, X_val, y_train, y_val):
    """Train XGBoost with a reduced upper bound on boosting rounds.
    Early stopping will still determine the actual number of rounds,
    so model quality is unchanged while unnecessary iterations are avoided."""
    X_train_f = X_train.astype(np.float32)
    X_val_f = X_val.astype(np.float32)

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    dtrain = xgb.DMatrix(X_train_f, label=y_train_log)
    dval = xgb.DMatrix(X_val_f, label=y_val_log)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "max_depth": 10,
        "eta": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 0,
        "gamma": 0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "nthread": os.cpu_count(),
        "max_bin": 256,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1500,
        early_stopping_rounds=100,
        evals=[(dval, "val")],
        verbose_eval=False,
    )
    return model




## === cell 28
xgbm = XGBoost(X_train, X_val, y_train, y_val)




## === cell 29
test_pred_f = test_pred.astype(np.float32)
XGBPredictions = np.expm1(xgbm.predict(xgb.DMatrix(test_pred_f)))
XGBPredictions = np.maximum(XGBPredictions, 0)
XGBPredictions = np.round(XGBPredictions, 2)




## === cell 30
EnsemblePredictions = np.round(
    np.maximum((LinearPredictions + XGBPredictions) / 2, 0), 2
)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": EnsemblePredictions},
    columns=["key", "fare_amount"],
)




## === cell 31
submission.to_csv("XGBSubmission23082018_2M.csv", index=False)
