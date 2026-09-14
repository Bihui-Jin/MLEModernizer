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
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
n_train = 1_000_000  # keep identical sample size

train_path = "../input/train.csv"
test_path = "../input/test.csv"

dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float64,
    "pickup_latitude": np.float64,
    "dropoff_longitude": np.float64,
    "dropoff_latitude": np.float64,
    "passenger_count": np.int8,
}



## === cell 2
rng = np.random.RandomState(42)

n_total = pd.read_csv(train_path, usecols=["key"]).shape[0]

if n_train >= n_total:
    skiprows = None
else:
    keep_idx = rng.choice(np.arange(1, n_total + 1), size=n_train, replace=False)
    keep_idx.sort()

    mask = np.ones(n_total + 1, dtype=bool)  # includes position 0 for header
    mask[0] = False  # never skip header
    mask[keep_idx] = False  # do not skip kept rows
    skiprows = np.flatnonzero(mask).tolist()

df = pd.read_csv(
    train_path,
    skiprows=skiprows,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
)

df_test = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
)

print("Loaded train shape:", df.shape, "test shape:", df_test.shape)



## === cell 3
pass



## === cell 4
pass




## === cell 5
def add_travel_vector_features(dfin):
    dfin["abs_diff_longitude"] = (dfin.dropoff_longitude - dfin.pickup_longitude).abs()
    dfin["abs_diff_latitude"] = (dfin.dropoff_latitude - dfin.pickup_latitude).abs()


add_travel_vector_features(df)
add_travel_vector_features(df_test)

print(df.isnull().sum())
print("Old size: %d" % len(df))
df = df.dropna(how="any", axis="rows")
print("New size: %d" % len(df))



## === cell 6
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 250)]
df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

lon_min, lon_max = -74.5, -72.8
lat_min, lat_max = 40.5, 41.8
df = df[
    (df["pickup_longitude"].between(lon_min, lon_max))
    & (df["dropoff_longitude"].between(lon_min, lon_max))
    & (df["pickup_latitude"].between(lat_min, lat_max))
    & (df["dropoff_latitude"].between(lat_min, lat_max))
]

df = df[(df["abs_diff_longitude"] > 0) | (df["abs_diff_latitude"] > 0)]

df_test = df_test.dropna(how="any", axis="rows").copy()
for c in ["pickup_longitude", "dropoff_longitude"]:
    df_test[c] = df_test[c].clip(lon_min, lon_max)
for c in ["pickup_latitude", "dropoff_latitude"]:
    df_test[c] = df_test[c].clip(lat_min, lat_max)

print("After cleaning size: %d" % len(df))
print("After test clip size: %d" % len(df_test))



## === cell 7
min_year = df["pickup_datetime"].dt.year.min()

df["pickup_year"] = (df["pickup_datetime"].dt.year - min_year).astype(np.int16)
df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
df["pickup_day"] = df["pickup_datetime"].dt.dayofyear.astype(np.int16)

df_test["pickup_year"] = (df_test["pickup_datetime"].dt.year - min_year).astype(
    np.int16
)
df_test["pickup_hour"] = df_test["pickup_datetime"].dt.hour.astype(np.int8)
df_test["pickup_day"] = df_test["pickup_datetime"].dt.dayofyear.astype(np.int16)



## === cell 8
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
len(df_val)




## === cell 9
def get_input_matrix(dfin):
    cols = [
        "abs_diff_longitude",
        "abs_diff_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "passenger_count",
        "pickup_year",
        "pickup_hour",
        "pickup_day",
    ]
    return np.column_stack([dfin[c].to_numpy() for c in cols])


x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)
y_train, y_val = np.array(df_train.fare_amount), np.array(df_val.fare_amount)



## === cell 10
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



## === cell 11
reg.fit(x_train, y_train)



## === cell 12
reg.oob_score_



## === cell 13
from sklearn.metrics import r2_score

y_pred = reg.predict(x_val)



## === cell 14
score = r2_score(y_val, y_pred)
score



## === cell 15
pass



## === cell 16
reg.score(x_val, y_val)



## === cell 17
from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_pred)



## === cell 18
x_test = get_input_matrix(df_test)



## === cell 19
predictions = reg.predict(x_test)
predictions = np.clip(predictions, 0, None)

RFSubmission = pd.DataFrame({"key": df_test.key.ravel(), "fare_amount": predictions})
RFSubmission.to_csv("RFSubmission.csv", index=False)
print("Wrote RFSubmission.csv with shape:", RFSubmission.shape)
