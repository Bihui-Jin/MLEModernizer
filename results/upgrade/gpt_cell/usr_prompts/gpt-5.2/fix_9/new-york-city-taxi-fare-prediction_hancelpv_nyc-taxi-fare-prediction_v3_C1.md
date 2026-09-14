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
import time

np.random.seed(42)

print(os.listdir("../input"))



## === cell 1
TRAIN_NROWS = 1500000

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtypes_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "object",
}
dtypes_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "object",
}

_read_csv_kwargs_train = dict(
    filepath_or_buffer="../input/train.csv",
    nrows=TRAIN_NROWS,
    dtype=dtypes_train,
    usecols=train_usecols,
)
_read_csv_kwargs_test = dict(
    filepath_or_buffer="../input/test.csv",
    dtype=dtypes_test,
    usecols=test_usecols,
)
try:
    train = pd.read_csv(engine="pyarrow", **_read_csv_kwargs_train)
    test = pd.read_csv(engine="pyarrow", **_read_csv_kwargs_test)
except Exception:
    train = pd.read_csv(**_read_csv_kwargs_train)
    test = pd.read_csv(**_read_csv_kwargs_test)

samp = pd.read_csv("../input/sample_submission.csv")



## === cell 2
train = train.dropna(how="any", axis="rows")



## === cell 3
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)]

train = train[
    (train["pickup_longitude"].between(-75, -72))
    & (train["dropoff_longitude"].between(-75, -72))
    & (train["pickup_latitude"].between(40, 42))
    & (train["dropoff_latitude"].between(40, 42))
    & (train["passenger_count"].between(1, 6))
]



## === cell 4
test.shape



## === cell 5
y = train.fare_amount.to_numpy(copy=False)
n_train = len(train)
n_test = len(test)
test_id = test.key




## === cell 6
def week_num_from_day(day_series: pd.Series) -> pd.Categorical:
    d = day_series.to_numpy(copy=False)
    out = np.empty(d.shape[0], dtype=object)
    out[d <= 7] = "first"
    m = (d > 7) & (d <= 14)
    out[m] = "second"
    m = (d > 14) & (d <= 21)
    out[m] = "third"
    m = (d > 21) & (d <= 28)
    out[m] = "fourth"
    out[d > 28] = "fifth"
    return pd.Categorical(
        out, categories=["first", "second", "third", "fourth", "fifth"]
    )




## === cell 7
_DAY_NAMES = np.array(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    dtype=object,
)


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(data["pickup_datetime"], cache=True, errors="coerce")

    hour = dt.dt.hour.astype("int16", copy=False)
    dow = dt.dt.dayofweek.astype("int8", copy=False)
    day = dt.dt.day.astype("int8", copy=False)
    month = dt.dt.month.astype("int8", copy=False)
    year = dt.dt.year.astype("int16", copy=False)

    out = pd.DataFrame(index=data.index)
    out["hour"] = pd.Categorical(hour.astype(str))
    out["day_of_week"] = pd.Categorical(_DAY_NAMES[dow.to_numpy(copy=False)])
    out["week_of_month"] = week_num_from_day(day)
    out["month"] = pd.Categorical(month.astype(str))
    out["year"] = pd.Categorical(year.astype(str))
    return out




## === cell 8
def add_geo_features(data: pd.DataFrame) -> pd.DataFrame:
    p_long = data["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    d_long = data["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    p_lat = data["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    d_lat = data["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    abs_diff_long = np.abs(d_long - p_long).astype(np.float32, copy=False)
    abs_diff_lat = np.abs(d_lat - p_lat).astype(np.float32, copy=False)

    out = pd.DataFrame(index=data.index)
    out["abs_diff_longitude"] = abs_diff_long
    out["abs_diff_latitude"] = abs_diff_lat
    out["manhattan_distance"] = (abs_diff_long + abs_diff_lat).astype(
        np.float32, copy=False
    )
    out["euclid_disance"] = np.sqrt(
        abs_diff_long * abs_diff_long + abs_diff_lat * abs_diff_lat
    ).astype(np.float32)
    return out




## === cell 9
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

train_time = add_time_features(train)
test_time = add_time_features(test)

train_geo = add_geo_features(train)
test_geo = add_geo_features(test)

train_fe = pd.concat(
    [
        train[["passenger_count"]].reset_index(drop=True),
        train_time.reset_index(drop=True),
        train_geo.reset_index(drop=True),
    ],
    axis=1,
)
test_fe = pd.concat(
    [
        test[["passenger_count"]].reset_index(drop=True),
        test_time.reset_index(drop=True),
        test_geo.reset_index(drop=True),
    ],
    axis=1,
)

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
for c in cat_cols:
    all_cats = pd.Index(train_fe[c].cat.categories).union(
        pd.Index(test_fe[c].cat.categories)
    )
    train_fe[c] = train_fe[c].cat.set_categories(all_cats)
    test_fe[c] = test_fe[c].cat.set_categories(all_cats)

all_fe = pd.concat([train_fe[features], test_fe[features]], axis=0, ignore_index=True)
all_x_df = pd.get_dummies(all_fe, dtype=np.uint8, sparse=True)

train_x_df = all_x_df.iloc[:n_train]
test_x_df = all_x_df.iloc[n_train:]
x_columns = train_x_df.columns  # identical columns for both by construction

x = train_x_df.to_numpy()
x_test = test_x_df.to_numpy()



## === cell 10
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
    max_samples=1.0,
    oob_score=False,
)



## === cell 11
start_time = time.time()
model.fit(x, y)
print("Fit time (sec):", round(time.time() - start_time, 2))



## === cell 12
feat_imp = pd.DataFrame([model.feature_importances_], columns=list(x_columns)).T
feat_imp.columns = ["imp"]
feat_imp.sort_values("imp", ascending=False, inplace=True)



## === cell 13
test_pred = model.predict(x_test)



## === cell 14
samp.head()



## === cell 15
sub = pd.DataFrame()
sub["key"] = test_id
sub["fare_amount"] = test_pred
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
