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
import math
import os

INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "../input"
print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))

np.random.seed(42)



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    nrows=1000000,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=test_cols,
    parse_dates=["pickup_datetime"],
)
samp = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")



## === cell 3
train = train.dropna(how="any", axis="rows")
m = (
    (train.fare_amount > 0)
    & (train["passenger_count"] <= 6)
    & (train.pickup_latitude > -90)
    & (train.pickup_latitude < 90)
    & (train.dropoff_latitude > -90)
    & (train.dropoff_latitude < 90)
    & (train.pickup_longitude > -180)
    & (train.pickup_longitude < 180)
    & (train.dropoff_longitude > -180)
    & (train.dropoff_longitude < 180)
)
train = train.loc[m]



## === cell 4
y = train.fare_amount.to_numpy(copy=False)
n_train = len(train)
n_test = len(test)
test_id = test.key

train_Xbase = train.drop(columns=["fare_amount"])
test_Xbase = test.drop(columns=["key"])

all_data = pd.concat((train_Xbase, test_Xbase), axis=0, ignore_index=True, copy=False)




## === cell 5
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 6
def add_time_features(data):
    dt = data["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=True)
        data["pickup_datetime"] = dt

    hour = dt.dt.hour.to_numpy()
    month = dt.dt.month.to_numpy()
    year = dt.dt.year.to_numpy()
    dom = dt.dt.day.to_numpy()

    dow = dt.dt.day_name()

    wom = np.empty(dom.shape[0], dtype=object)
    wom[:] = "fifth"
    wom[dom <= 7] = "first"
    wom[(dom > 7) & (dom <= 14)] = "second"
    wom[(dom > 14) & (dom <= 21)] = "third"
    wom[(dom > 21) & (dom <= 28)] = "fourth"
    wom[pd.isna(dom)] = None

    data["hour"] = pd.Categorical(hour.astype("float64", copy=False))
    data["day_of_week"] = pd.Categorical(dow)
    data["week_of_month"] = pd.Categorical(wom)
    data["month"] = pd.Categorical(month.astype("float64", copy=False))
    data["year"] = pd.Categorical(year.astype("float64", copy=False))

    data.drop("pickup_datetime", axis=1, inplace=True)
    return data




## === cell 7
def add_geo_features(data):
    plon = data["pickup_longitude"].to_numpy()
    plat = data["pickup_latitude"].to_numpy()
    dlon = data["dropoff_longitude"].to_numpy()
    dlat = data["dropoff_latitude"].to_numpy()

    abs_diff_long = np.abs(dlon - plon)
    abs_diff_lat = np.abs(dlat - plat)

    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat
    data["manhattan_distance"] = abs_diff_long + abs_diff_lat

    squared_long = abs_diff_long * abs_diff_long
    squared_lat = abs_diff_lat * abs_diff_lat
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat
    data["euclid_disance"] = np.sqrt(squared_long + squared_lat)

    return data




## === cell 8
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)



## === cell 9
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

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
all_data = all_data[features]

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

for c in cat_cols:
    if not pd.api.types.is_categorical_dtype(all_data[c]):
        all_data[c] = all_data[c].astype("category")
    if "nan" not in all_data[c].cat.categories:
        all_data[c] = all_data[c].cat.add_categories(["nan"])
    all_data[c] = all_data[c].fillna("nan")

all_data[num_cols] = all_data[num_cols].fillna(0.0)

x_df = all_data.iloc[:n_train]
x_test_df = all_data.iloc[n_train:]

X_num = x_df[num_cols].to_numpy(dtype=np.float32, copy=False)
X_test_num = x_test_df[num_cols].to_numpy(dtype=np.float32, copy=False)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

X_cat = ohe.fit_transform(x_df[cat_cols])
X_test_cat = ohe.transform(x_test_df[cat_cols])

X = sparse.hstack([sparse.csr_matrix(X_num), X_cat], format="csr")
X_test = sparse.hstack([sparse.csr_matrix(X_test_num), X_test_cat], format="csr")
X.sort_indices()
X_test.sort_indices()



## === cell 10
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    max_bins=256,  # histogram optimization
)



## === cell 11
model.fit(X, y)



## === cell 12
test_pred = model.predict(X_test)

sub = pd.DataFrame()
sub["key"] = test_id.values
sub["fare_amount"] = test_pred.astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
