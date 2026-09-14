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

4.73089

# 6. Current score

6.07499

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 192.36038) has done: 'Diagnosis: The crash happens because in pandas 2.2 the `.dt.week` and `.dt.weekofyear` accessors were removed from `DatetimeProperties`. The code in cell 4 relies on these deprecated attributes to compute week-based features, so it raises `AttributeError`.  
Patch summary: Replace `.dt.week` and `.dt.weekofyear` with the supported ISO calendar week accessor, keeping the same feature names (`week`, `week_of_year`) and producing integer week numbers. This preserves downstream semantics and keeps compatibility with the next cell, which creates the same columns for `test`.  
Updated cells: Only cell 4 is modified.  
Compatibility notes for cell k+1: Cell 5 still uses `.dt.week`/`.dt.weekofyear` and error next; however per instruction we do not modify non-buggy cells. The updated `train` columns remain `hour/day/week/month/day_of_year/week_of_year` as expected.  
Assumptions: Using ISO week number (1–53) is an acceptable replacement for the removed pandas week fields and matches the intended “week of year” feature.'
- What this solution (achieved 192.36617) has done: 'Diagnosis: Cell 5 crashes because `pandas` 2.2 removed the deprecated `Series.dt.week` and `Series.dt.weekofyear` accessors, so `test['pickup_datetime'].dt.week` raises `AttributeError`.  
Patch summary: Replace the deprecated week accessors in cell 5 with the supported ISO-week accessor `dt.isocalendar().week.astype("int64")`, matching the approach already used in cell 4. Keep all column names and semantics (`week`, `week_of_year`) intact so downstream cells continue to work.  
Updated cells: Only cell 5 is changed.  
Compatibility notes for cell k+1: Cell 6 expects `test` to exist and does not depend on the specific week extraction method; the patched code still creates the same columns with numeric dtype.  
Assumptions: ISO week number is an acceptable substitute for the removed `.dt.week`/`.dt.weekofyear` behavior (and is consistent with the train feature engineering in cell 4).'
- What this solution (achieved 192.45281) has done: 'Your score is far worse than the target (RMSE 192 vs 4.73), which strongly suggests a submission alignment bug rather than model quality. The smallest fix is to build the submission from `test[['key']]` and merge predictions by `key`, instead of assuming the row order in `sample_submission.csv` matches `test.csv`. This preserves your core feature engineering and model ensemble exactly, but ensures every prediction is assigned to the correct `key`, which should move RMSE dramatically toward the target. I also add a tiny safety check to confirm submission row count and key uniqueness before writing the CSV.'
- What this solution (achieved 11.21619) has done: 'Your RMSE is so far from the target that this is almost certainly not “model weakness” but a prediction scale/outlier issue; the simplest legitimate fix is to clip impossible negative/huge fare predictions before writing the submission, which typically improves RMSE drastically on this competition. I keep your exact feature engineering and the same three-model ensemble, and only add a small post-processing step consistent with the training filters you already apply (`0 < fare_amount < 200`). I also make the RandomForest deterministic (fixed `random_state`) to stabilize the score and reduce run-to-run variance without changing the overall approach. The output remain a valid `key,fare_amount` CSV with the same row order as `test`.'
- What this solution (achieved 5.93689) has done: 'Your RMSE (11.216) is still far from the target (4.731), and with only `abs_diff_longitude/latitude` as features the main issue is under-modeling distance/geometry rather than submission formatting. To move toward the target while preserving your core approach (same three regressors + same averaging ensemble), I add a minimal set of standard NYC Taxi engineered features (haversine distance, manhattan distance, and a couple of simple geo interactions) and include them in `feature_names`. I also apply the same basic coordinate/passenger sanity filters to `test` (without dropping rows) by clipping to the training-filter ranges, which typically reduces extreme, high-error predictions. Finally, I keep your prediction clipping and ensure the submission schema remains exactly `key,fare_amount`.'
- What this solution (achieved 5.93706) has done: 'You’re currently worse than the target (RMSE 5.936 vs 4.731; lower is better), so we should make a small, low-risk improvement without changing your model ensemble or training approach. The biggest remaining issue is that KNN and LinearRegression are sensitive to feature scale; adding a `StandardScaler` fitted on train and applied to both train/test typically improves RMSE materially for this competition while keeping the same models and averaging logic. I keep the RandomForest unchanged (it doesn’t need scaling) and only scale the inputs used by LinearRegression and KNN, preserving your existing features, filters, and prediction clipping. The output remains a valid `key,fare_amount` CSV with identical schema and row alignment.'
- What this solution (achieved 6.07499) has done: 'You’re currently worse than the target (RMSE 5.937 vs 4.731; lower is better), so the smallest likely gain is to add a few time features you already compute (hour/day/week/month/day_of_year/week_of_year) into `feature_names` so the same models can learn systematic fare differences by time without changing the modeling approach. This preserves your exact ensemble (LinearRegression + KNN on scaled features, RandomForest on raw features, same averaging) and uses only columns already present in both train/test. I’m also adding a tiny data-cleaning step to replace any inf/NaN values in the expanded feature set (train + test) to avoid silent model degradation. The submission format and file writing remain unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd

train = pd.read_csv("../input/train.csv", nrows=300_000)
test = pd.read_csv("../input/test.csv")



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt



## === cell 4
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype("int64")
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week.astype("int64")



## === cell 5
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype("int64")
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week.astype("int64")



## === cell 6
train.head()
train = train.dropna(how="any", axis="rows")



## === cell 7
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < 75)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < 75)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]
train = train.loc[train["passenger_count"] <= 8]



## === cell 8
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()



## === cell 9
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()



## === cell 10
train.head()



## === cell 11
train.head()



## === cell 12
sns.barplot(data=train, x="passenger_count", y="fare_amount")



## === cell 13
import numpy as np


def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    return R_km * c


train["haversine_km"] = haversine_np(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
train["manhattan_km"] = haversine_np(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["pickup_latitude"].values,
) + haversine_np(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["pickup_longitude"].values,
    train["dropoff_latitude"].values,
)

train["lon_lat_sum_absdiff"] = train["abs_diff_longitude"] + train["abs_diff_latitude"]
train["lon_lat_prod_absdiff"] = train["abs_diff_longitude"] * train["abs_diff_latitude"]

test["haversine_km"] = haversine_np(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)
test["manhattan_km"] = haversine_np(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["pickup_latitude"].values,
) + haversine_np(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["pickup_longitude"].values,
    test["dropoff_latitude"].values,
)
test["lon_lat_sum_absdiff"] = test["abs_diff_longitude"] + test["abs_diff_latitude"]
test["lon_lat_prod_absdiff"] = test["abs_diff_longitude"] * test["abs_diff_latitude"]



## === cell 14
test["pickup_longitude"] = test["pickup_longitude"].clip(-75, 75)
test["dropoff_longitude"] = test["dropoff_longitude"].clip(-75, 75)
test["pickup_latitude"] = test["pickup_latitude"].clip(40, 45)
test["dropoff_latitude"] = test["dropoff_latitude"].clip(40, 45)
test["passenger_count"] = test["passenger_count"].clip(lower=0, upper=8)

for col in [
    "haversine_km",
    "manhattan_km",
    "lon_lat_sum_absdiff",
    "lon_lat_prod_absdiff",
]:
    test[col] = test[col].replace([np.inf, -np.inf], np.nan).fillna(test[col].median())

for col in [
    "haversine_km",
    "manhattan_km",
    "lon_lat_sum_absdiff",
    "lon_lat_prod_absdiff",
]:
    train[col] = (
        train[col].replace([np.inf, -np.inf], np.nan).fillna(train[col].median())
    )



## === cell 15
feature_names = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "manhattan_km",
    "lon_lat_sum_absdiff",
    "lon_lat_prod_absdiff",
    "hour",
    "day",
    "week",
    "month",
    "day_of_year",
    "week_of_year",
]
feature_names



## === cell 16
label_name = "fare_amount"
label_name



## === cell 17
X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]



## === cell 18
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb



## === cell 19
for col in feature_names:
    med = X_train[col].replace([np.inf, -np.inf], np.nan).median()
    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan).fillna(med)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).fillna(med)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



## === cell 20
regr = LinearRegression()
regr.fit(X_train_scaled, y_train)
regr_prediction = regr.predict(X_test_scaled)



## === cell 21
knr = KNeighborsRegressor()
knr.fit(X_train_scaled, y_train)
knr_prediction = knr.predict(X_test_scaled)



## === cell 22
rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)



## === cell 23
predictions = (regr_prediction * 2 + rfr_prediction * 2 + knr_prediction) / 5



## === cell 24
predictions



## === cell 25
predictions = pd.Series(predictions).clip(lower=0.0, upper=200.0).to_numpy()

pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": predictions})
assert len(pred_df) == len(test), "Predictions length must match test length"
assert pred_df["key"].isna().sum() == 0, "Test keys must be non-null"
assert pred_df["key"].nunique() == len(
    pred_df
), "Test keys should be unique for a valid submission"
submission = pred_df



## === cell 26
submission.head()



## === cell 27
submission.to_csv("./simplenewyorktaxi.csv", index=False)
