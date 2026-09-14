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

3.44303

# 6. Current score

6.33061

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.29803) has done: 'I fix the XGBoost inference bug caused by using the deprecated `ntree_limit/model.best_ntree_limit` API in xgboost 2.x by switching to `iteration_range` (and falling back safely if early stopping isn’t set). I also update the objective from deprecated `reg:linear` to `reg:squarederror` (same regression semantics) to avoid warnings and ensure stable training. Finally, I make sure the script always writes a valid `submission.csv` with the required `key,fare_amount` columns in the working directory using the existing pipeline and features.'
- What this solution (achieved 7.88196) has done: 'You don’t yet have a valid Kaggle score, so the most likely blocker is that `clean_df(test_df)` can drop test rows, producing a submission with fewer rows than `sample_submission.csv` (Kaggle reject it). I keep your exact feature engineering and XGBoost training logic, but change test-time handling to never filter out rows; instead, I clip out-of-range coordinates/passenger counts to safe bounds (so every `key` remains) and fill any remaining missing values. I also align the submission to `sample_submission.csv` keys to guarantee correct order/rowcount, and I make a small, safe training tweak by using more boosting rounds with early stopping (same approach, typically improves RMSE toward your target without changing core logic). These changes should produce a valid `submission.csv` and move RMSE downward from your prior ~6.3 toward the 3.44 target.'
- What this solution (achieved 6.33061) has done: 'Your current RMSE (7.88) is far worse than the target (3.44), so we should make a small, legitimate improvement without changing the overall XGBoost approach. The biggest issue is that the model is likely being trained on untransformed raw coordinates/time without enough structure for NYC taxi fares; we can keep the same feature engineering but add two standard, minimal extensions that usually cut RMSE a lot: (1) add a few simple geospatial/time features (absolute deltas, haversine distance in km already exists, plus a rough Manhattan-distance proxy), and (2) use a slightly more appropriate XGBoost configuration for this problem (still `xgb.train` regression, same loss/metric) by adding conservative tree/regularization defaults and training on a larger sample (still bounded to run under the time limit). We also keep your “never drop test rows” logic and ensure the submission is aligned to `sample_submission.csv` keys and has exactly the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=3_000_000)
train_df.dtypes



## === cell 2
print(train_df.isnull().sum())



## === cell 3
train_df = train_df.dropna(how="any", axis="rows")



## === cell 4
train_df.head()



## === cell 5
try:
    ax1 = train_df.iloc[:1000].plot.scatter("pickup_longitude", "pickup_latitude")
    ax2 = train_df.iloc[:1000].plot.scatter("dropoff_longitude", "dropoff_latitude")
except Exception as e:
    print(f"Plotting skipped due to: {e}")

train_df.describe()




## === cell 6
def clean_df(df):
    base = (
        (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
        & (df.passenger_count > 0)
        & (df.passenger_count < 10)
    )
    return df[base]


train_df = clean_df(train_df)
train_df = train_df[(train_df.fare_amount > 0) & (train_df.fare_amount <= 250)]
print(len(train_df))




## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce"
    )

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    return dataset


def add_geo_features(df):
    df = df.copy()
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    mean_lat = np.radians((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat)

    df["manhattan_km"] = (
        df["abs_lat_diff"] * km_per_deg_lat + df["abs_lon_diff"] * km_per_deg_lon
    )
    return df


train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df = add_datetime_info(train_df)
train_df = add_geo_features(train_df)

train_df.head()



## === cell 8
train_df.drop(columns=["key", "pickup_datetime"], inplace=True)
train_df.head()



## === cell 9
y = train_df["fare_amount"]
train = train_df.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.2
)




## === cell 10
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 0,
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=4000,
        early_stopping_rounds=50,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)




## === cell 11
def sanitize_test_df(df):
    df = df.copy()
    df["pickup_longitude"] = df["pickup_longitude"].clip(-80, -70)
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(-80, -70)
    df["pickup_latitude"] = df["pickup_latitude"].clip(35, 45)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(35, 45)

    df["passenger_count"] = df["passenger_count"].fillna(1)
    df["passenger_count"] = df["passenger_count"].clip(1, 9)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    return df


test_df = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")

test_df = sanitize_test_df(test_df)

test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df = add_datetime_info(test_df)
test_df = add_geo_features(test_df)

for c in ["hour", "day", "month", "weekday"]:
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(0).astype(int)

feat_cols = [c for c in test_df.columns if c not in ["key", "pickup_datetime"]]
test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

test_key = test_df["key"]
x_pred = test_df.drop(columns=["key", "pickup_datetime"])

dm_pred = xgb.DMatrix(x_pred)

best_iter = getattr(model, "best_iteration", None)
if best_iter is not None:
    prediction = model.predict(dm_pred, iteration_range=(0, best_iter + 1))
else:
    prediction = model.predict(dm_pred)

pred_map = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission = sample_sub[["key"]].merge(pred_map, on="key", how="left")

submission["fare_amount"] = submission["fare_amount"].fillna(
    float(np.nanmean(prediction))
)

submission.to_csv("submission.csv", index=False)
submission
