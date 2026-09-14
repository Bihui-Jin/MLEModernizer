# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception as e:
    print("sklearnex patch not applied:", repr(e))

np.random.seed(42)

n_train = 1_000_000

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
pass



## === cell 3
pass




## === cell 4
def add_travel_vector_features(df_):
    pickup_lon = df_["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    pickup_lat = df_["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dropoff_lon = df_["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dropoff_lat = df_["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_lon = np.abs(dropoff_lon - pickup_lon)
    abs_diff_lat = np.abs(dropoff_lat - pickup_lat)
    df_["abs_diff_longitude"] = abs_diff_lon
    df_["abs_diff_latitude"] = abs_diff_lat

    r = 6371.0
    lat1 = np.deg2rad(pickup_lat)
    lon1 = np.deg2rad(pickup_lon)
    lat2 = np.deg2rad(dropoff_lat)
    lon2 = np.deg2rad(dropoff_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    df_["haversine_km"] = r * c

    mean_lat = np.deg2rad((pickup_lat + dropoff_lat) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat)
    df_["manhattan_km"] = abs_diff_lat * km_per_deg_lat + abs_diff_lon * km_per_deg_lon

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df_["bearing"] = np.arctan2(y, x)  # radians in [-pi, pi]


add_travel_vector_features(df)
add_travel_vector_features(df_test)

old_size = len(df)
df = df.dropna(how="any", axis="rows")
print("Old size: %d" % old_size)
print("New size: %d" % len(df))




## === cell 5
def clean_train_rows(df_):
    fare = df_["fare_amount"].to_numpy(copy=False)
    pax = df_["passenger_count"].to_numpy(copy=False)

    pu_lon = df_["pickup_longitude"].to_numpy(copy=False)
    pu_lat = df_["pickup_latitude"].to_numpy(copy=False)
    do_lon = df_["dropoff_longitude"].to_numpy(copy=False)
    do_lat = df_["dropoff_latitude"].to_numpy(copy=False)

    abs_dlon = df_["abs_diff_longitude"].to_numpy(copy=False)
    abs_dlat = df_["abs_diff_latitude"].to_numpy(copy=False)
    hav = df_["haversine_km"].to_numpy(copy=False)

    mask = np.ones(len(df_), dtype=bool)

    mask &= (fare > 0) & (fare <= 200)
    mask &= (pax >= 1) & (pax <= 6)

    mask &= (
        np.isfinite(pu_lon)
        & np.isfinite(pu_lat)
        & np.isfinite(do_lon)
        & np.isfinite(do_lat)
    )

    mask &= ~((pu_lon == 0) & (pu_lat == 0))
    mask &= ~((do_lon == 0) & (do_lat == 0))

    mask &= (pu_lon >= -75) & (pu_lon <= -72)
    mask &= (do_lon >= -75) & (do_lon <= -72)
    mask &= (pu_lat >= 40) & (pu_lat <= 42)
    mask &= (do_lat >= 40) & (do_lat <= 42)

    mask &= (abs_dlon + abs_dlat) > 0

    mask &= (hav > 0.05) & (hav < 50.0)
    mask &= (abs_dlon < 1.0) & (abs_dlat < 1.0)

    min_expected = 2.5 + 0.5 * hav
    max_expected = 10.0 + 15.0 * hav
    mask &= (fare >= min_expected) & (fare <= max_expected)

    years = df_["pickup_datetime"].dt.year.to_numpy(copy=False)
    mask &= (years >= 2009) & (years <= 2015)

    fare_per_km = fare / np.clip(hav, 0.1, None)
    mask &= (fare_per_km >= 1.5) & (fare_per_km <= 40.0)

    return df_.loc[mask]


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
feature_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "manhattan_km",
    "bearing",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_longitude",
    "pickup_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_hour",
    "pickup_day",
]


def get_input_matrix(df_):
    X = np.empty((len(df_), len(feature_cols)), dtype=np.float32)
    for j, c in enumerate(feature_cols):
        X[:, j] = df_[c].to_numpy(copy=False)
    return X


x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)
y_train, y_val = np.asarray(df_train.fare_amount, dtype=np.float32), np.asarray(
    df_val.fare_amount, dtype=np.float32
)



## === cell 9
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(
    max_depth=25,
    n_estimators=500,
    oob_score=True,
    n_jobs=-1,
    min_samples_split=10,
    verbose=1,
    random_state=42,
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
pass



## === cell 17
reg.score(x_val, y_val)



## === cell 18
x_full = get_input_matrix(df)
y_full = np.asarray(df.fare_amount, dtype=np.float32)
reg.fit(x_full, y_full)

x_test = get_input_matrix(df_test)

predictions = reg.predict(x_test)
predictions = np.clip(predictions, 0, None)

RFSubmission = pd.DataFrame({"key": df_test.key.ravel(), "fare_amount": predictions})
RFSubmission.to_csv("RFSubmission.csv", index=False)
print("Wrote RFSubmission.csv with shape:", RFSubmission.shape)
