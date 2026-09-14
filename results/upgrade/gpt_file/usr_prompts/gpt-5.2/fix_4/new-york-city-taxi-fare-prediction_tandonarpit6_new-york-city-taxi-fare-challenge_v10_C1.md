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

4.01327

# 6. Current score

8.71901

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.67107) has done: 'I fix the XGBoost prediction call to use the correct signature (remove the invalid `data=` keyword) so inference runs and `Y_pred` is created. I also ensure `y` is a 1D numeric vector (not a single-column DataFrame) and drop rows with missing values so `model.fit` doesn’t crash or silently degrade. Finally, I generate the submission using exactly the required columns `key` and `fare_amount` and write it to a new `.csv` filename to avoid confusing it with the provided sample file. These changes are minimal, preserve your feature logic/model choice, and unblock end-to-end submission generation.'
- What this solution (achieved 8.79006) has done: 'Your current gap is 4.67107 − 4.01327 = 0.6578 (worse; lower RMSE is better), so we should improve the score with minimal, low-risk edits. The biggest issue is that you’re discarding key signal by dropping `passenger_count`, the raw pickup coordinates, and using simple abs deltas instead of a distance proxy; we keep the same XGBoost regressor/fit/predict flow but add a standard haversine distance feature and keep `passenger_count` and coordinates. We also apply a very small, common-sense cleaning filter on the training sample (NYC bounding box + positive reasonable fares) to remove extreme outliers that disproportionately hurt RMSE, without changing the overall approach. Finally, we set a fixed random seed and modest XGBoost parameters (still the same model class) to reduce variance and typically improve RMSE on this competition.'
- What this solution (achieved 8.71901) has done: 'We need to improve RMSE (lower is better) from 8.79006 toward 4.01327; we’re far outside the ±10% band, so we can make a few more minimal, low-risk fixes without changing the core “XGBRegressor on engineered tabular features” approach. The biggest issue in your current code is that you create the NYC bounding-box mask *after* dropping the raw coordinate columns, so the filter is effectively broken (and should error or be skipped in your run), which leaves many outliers that inflate RMSE. I fix this by computing cleaning masks on the raw training frame before dropping columns, then applying the same feature engineering as you already do. I also add two tiny, standard features (log1p(haversine_km) and manhattan_km) and ensure train/test use identical column order, which usually improves stability and RMSE without altering the overall model/training loop.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "../input"
print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

training_data = pd.read_csv(train_path, nrows=2_000_000)
test_data = pd.read_csv(test_path)



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km




## === cell 5
raw = training_data.copy()
raw["pickup_datetime"] = pd.to_datetime(raw["pickup_datetime"], errors="coerce")

y = pd.to_numeric(raw["fare_amount"], errors="coerce")

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
base_mask = (
    y.notna()
    & raw["pickup_datetime"].notna()
    & raw[coord_cols].notna().all(axis=1)
    & raw["passenger_count"].notna()
)

bbox_mask = (
    raw["pickup_longitude"].between(-74.3, -73.6)
    & raw["dropoff_longitude"].between(-74.3, -73.6)
    & raw["pickup_latitude"].between(40.5, 41.0)
    & raw["dropoff_latitude"].between(40.5, 41.0)
)
pc_mask = raw["passenger_count"].between(1, 6)
fare_mask = y.between(2.5, 250.0)

train_mask = base_mask & bbox_mask & pc_mask & fare_mask

raw = raw.loc[train_mask].reset_index(drop=True)
y = y.loc[train_mask].reset_index(drop=True)

print("Train rows after cleaning:", len(raw), " / original:", len(training_data))



## === cell 6
X_train = raw.copy()

X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()

X_train["haversine_km"] = haversine_np(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
)

X_train["manhattan_km"] = (
    X_train["latitude_distance"] + X_train["longitude_distance"]
) * 111.0
X_train["log_haversine"] = np.log1p(X_train["haversine_km"])

X_train = X_train.drop(columns=["key", "fare_amount", "pickup_datetime"])



## === cell 7
X_test = test_data.copy()
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"], errors="coerce")
X_test["hour"] = X_test["pickup_datetime"].dt.hour

X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()

X_test["haversine_km"] = haversine_np(
    X_test["pickup_longitude"].values,
    X_test["pickup_latitude"].values,
    X_test["dropoff_longitude"].values,
    X_test["dropoff_latitude"].values,
)

X_test["manhattan_km"] = (
    X_test["latitude_distance"] + X_test["longitude_distance"]
) * 111.0
X_test["log_haversine"] = np.log1p(X_test["haversine_km"])

X_test = X_test.drop(columns=["key", "pickup_datetime"])

X_test = X_test.reindex(columns=X_train.columns)

X_test = X_test.fillna(0)



## === cell 8
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y)

Y_pred = model.predict(X_test)



## === cell 9
from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model, max_num_features=15)
plt.show()



## === cell 10
submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": Y_pred})

submission["fare_amount"] = pd.to_numeric(
    submission["fare_amount"], errors="coerce"
).fillna(0.0)

submission["fare_amount"] = submission["fare_amount"].clip(lower=0.0)



## === cell 11
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
