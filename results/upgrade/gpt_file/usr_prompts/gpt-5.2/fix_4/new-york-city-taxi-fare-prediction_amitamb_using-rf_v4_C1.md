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
sklearn-pandas==2.2.0

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

3.50273

# 6. Current score

6.6708

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12.97472) has done: 'You’re currently far from the target RMSE (4.44 vs 3.50, lower is better), so we make small, safe changes that reliably improve generalization without changing the core model type or feature set. The biggest issue is noisy/outlier training rows (invalid coordinates, unrealistically high fares, etc.) that a RandomForest overfit; adding standard NYC Taxi competition cleaning rules usually drops RMSE substantially while preserving the same features and training loop. I also make the train/validation split deterministic and align the internal evaluation to RMSE (your Kaggle metric) so you can track progress locally, without changing what gets submitted. Submission writing stays the same (creates `RFSubmission.csv` with `key,fare_amount`).'
- What this solution (achieved 9.49367) has done: 'You’re far above the target (RMSE 12.97 vs 3.50; lower is better), so the smallest reliable way to move toward the target is to improve data quality and align features with the metric without changing the model type or feature family. I keep your RandomForest and the same feature extraction pattern, but (1) add the standard NYC taxi “distance” feature (Haversine) while keeping your existing diffs, and (2) apply a couple of common, safe cleaning rules (remove extreme/unphysical distances) that otherwise dominate RMSE. I also clip negative test predictions to 0 (fares can’t be negative), which usually reduces RMSE a bit with no semantic change. Submission writing remains `RFSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.6708) has done: 'You’re currently above the target (9.49 vs 3.50 RMSE; lower is better), so we should make the smallest changes that reliably reduce RMSE without changing your RandomForest approach or feature family. The biggest remaining gap is usually caused by residual noisy/outlier rows; we tighten cleaning with standard NYC Taxi rules that don’t alter core semantics: remove “fare vs distance” inconsistencies, cap extreme coordinate deltas, and filter impossible timestamps. We also add a single robust, competition-standard feature (straight-line “manhattan” distance in km) while keeping your existing diffs and haversine. Finally, we keep submission writing identical (`RFSubmission.csv` with `key,fare_amount`) and preserve determinism.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
n_train = 100_000
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float64,
    "pickup_latitude": np.float64,
    "dropoff_longitude": np.float64,
    "dropoff_latitude": np.float64,
    "passenger_count": np.int8,
}

df = pd.read_csv(
    "../input/train.csv",
    nrows=n_train,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
)
df_test = pd.read_csv(
    "../input/test.csv", parse_dates=["pickup_datetime"], dtype=dtype_map
)



## === cell 2
df.describe()



## === cell 3
df.dtypes




## === cell 4
def add_travel_vector_features(df_):
    df_["abs_diff_longitude"] = (df_.dropoff_longitude - df_.pickup_longitude).abs()
    df_["abs_diff_latitude"] = (df_.dropoff_latitude - df_.pickup_latitude).abs()

    r = 6371.0
    lat1 = np.deg2rad(df_["pickup_latitude"].values)
    lon1 = np.deg2rad(df_["pickup_longitude"].values)
    lat2 = np.deg2rad(df_["dropoff_latitude"].values)
    lon2 = np.deg2rad(df_["dropoff_longitude"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    df_["haversine_km"] = r * c

    mean_lat = np.deg2rad(
        (df_["pickup_latitude"].values + df_["dropoff_latitude"].values) / 2.0
    )
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat)
    df_["manhattan_km"] = (df_["abs_diff_latitude"].values * km_per_deg_lat) + (
        df_["abs_diff_longitude"].values * km_per_deg_lon
    )


add_travel_vector_features(df)
add_travel_vector_features(df_test)

print(df.isnull().sum())
print("Old size: %d" % len(df))
df = df.dropna(how="any", axis="rows")
print("New size: %d" % len(df))




## === cell 5
def clean_train_rows(df_):
    df_ = df_.copy()

    df_ = df_[(df_.fare_amount > 0) & (df_.fare_amount <= 250)]
    df_ = df_[(df_.passenger_count >= 1) & (df_.passenger_count <= 6)]

    coord_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    for c in coord_cols:
        df_ = df_[df_[c].notnull()]

    df_ = df_[~((df_.pickup_longitude == 0) & (df_.pickup_latitude == 0))]
    df_ = df_[~((df_.dropoff_longitude == 0) & (df_.dropoff_latitude == 0))]

    df_ = df_[
        (df_.pickup_longitude.between(-75, -72))
        & (df_.dropoff_longitude.between(-75, -72))
        & (df_.pickup_latitude.between(40, 42))
        & (df_.dropoff_latitude.between(40, 42))
    ]

    df_ = df_[(df_.abs_diff_longitude + df_.abs_diff_latitude) > 0]

    df_ = df_[(df_["haversine_km"] > 0.05) & (df_["haversine_km"] < 100.0)]

    df_ = df_[(df_["abs_diff_longitude"] < 1.0) & (df_["abs_diff_latitude"] < 1.0)]

    min_expected = 2.5 + 0.5 * df_["haversine_km"]
    max_expected = 10.0 + 15.0 * df_["haversine_km"]
    df_ = df_[
        (df_["fare_amount"] >= min_expected) & (df_["fare_amount"] <= max_expected)
    ]

    years = df_["pickup_datetime"].dt.year
    df_ = df_[(years >= 2009) & (years <= 2015)]

    return df_


old_len = len(df)
df = clean_train_rows(df)
print(f"After cleaning: {old_len} -> {len(df)} rows")



## === cell 6
min_year = df.pickup_datetime.dt.year.min()

df["pickup_year"] = df.pickup_datetime.dt.year - min_year
df["pickup_hour"] = df.pickup_datetime.dt.hour
df["pickup_day"] = df.pickup_datetime.dt.dayofyear

df_test["pickup_year"] = df_test.pickup_datetime.dt.year - min_year
df_test["pickup_hour"] = df_test.pickup_datetime.dt.hour
df_test["pickup_day"] = df_test.pickup_datetime.dt.dayofyear



## === cell 7
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
len(df_val)




## === cell 8
def get_input_matrix(df_):
    return np.column_stack(
        (
            df_.abs_diff_longitude,
            df_.abs_diff_latitude,
            df_.haversine_km,
            df_.manhattan_km,
            df_.dropoff_longitude,
            df_.dropoff_latitude,
            df_.pickup_longitude,
            df_.pickup_latitude,
            df_.passenger_count,
            df_.pickup_year,
            df_.pickup_hour,
            df_.pickup_day,
        )
    )


x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)
y_train, y_val = np.array(df_train.fare_amount), np.array(df_val.fare_amount)



## === cell 9
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    max_depth=25,
    n_estimators=500,
    oob_score=True,
    n_jobs=-1,
    min_samples_split=10,
    verbose=1,
    random_state=42,  # deterministic forests for stability
)



## === cell 10
reg.fit(x_train, y_train)



## === cell 11
reg.oob_score_



## === cell 12
from sklearn.metrics import r2_score

y_pred = reg.predict(x_val)



## === cell 13
score = r2_score(y_val, y_pred)
score



## === cell 14
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_val, y_pred, squared=False)
rmse



## === cell 15
mean_squared_error(y_val, y_pred)



## === cell 16
import matplotlib.pyplot as plt

plt.scatter(y_val, y_pred, alpha=0.02)



## === cell 17
reg.score(x_val, y_val)



## === cell 18
x_test = get_input_matrix(df_test)



## === cell 19
predictions = reg.predict(x_test)
predictions = np.clip(predictions, 0, None)

RFSubmission = pd.DataFrame({"key": df_test.key.ravel(), "fare_amount": predictions})
RFSubmission.to_csv("RFSubmission.csv", index=False)
print("Wrote RFSubmission.csv with shape:", RFSubmission.shape)
