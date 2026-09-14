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

print(os.listdir("../input"))



## === cell 1
SEED = 42
RNG = np.random.RandomState(SEED)

TRAIN_NROWS_TARGET = 500_000  # was 200_000

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "object",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "object",
}

train_path = "../input/train.csv"

TRAIN_TOTAL_ROWS = (
    55_423_856  # from dataset description; used only to set sampling rate
)
p_keep = min(1.0, TRAIN_NROWS_TARGET / float(TRAIN_TOTAL_ROWS))


def read_train_sampled_csv(path, p, seed=SEED, chunksize=1_000_000):
    rng = np.random.RandomState(seed)
    usecols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    out = []
    kept = 0
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=train_dtypes,
        parse_dates=["pickup_datetime"],
        chunksize=chunksize,
    ):
        m = rng.rand(len(chunk)) < p
        if m.any():
            out.append(chunk.loc[m])
            kept += int(m.sum())
        if kept >= TRAIN_NROWS_TARGET:
            break
    if not out:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(out, ignore_index=True)
    if len(df) > TRAIN_NROWS_TARGET:
        df = df.iloc[:TRAIN_NROWS_TARGET].reset_index(drop=True)
    return df


train = read_train_sampled_csv(train_path, p_keep, seed=SEED, chunksize=1_000_000)

test = pd.read_csv(
    "../input/test.csv",
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)
samp = pd.read_csv("../input/sample_submission.csv")



## === cell 2
train = train.dropna(how="any", axis="rows").reset_index(drop=True)

fare = train["fare_amount"].to_numpy()
pc = train["passenger_count"].to_numpy()

plon = train["pickup_longitude"].to_numpy()
plat = train["pickup_latitude"].to_numpy()
dlon = train["dropoff_longitude"].to_numpy()
dlat = train["dropoff_latitude"].to_numpy()

mask = (fare > 0) & (fare <= 250)
mask &= (pc >= 1) & (pc <= 6)

mask &= (plon >= -180) & (plon <= 180)
mask &= (dlon >= -180) & (dlon <= 180)
mask &= (plat >= -90) & (plat <= 90)
mask &= (dlat >= -90) & (dlat <= 90)

nyc_lon_min, nyc_lon_max = -74.3, -73.7
nyc_lat_min, nyc_lat_max = 40.5, 41.0

mask &= (plon >= nyc_lon_min) & (plon <= nyc_lon_max)
mask &= (dlon >= nyc_lon_min) & (dlon <= nyc_lon_max)
mask &= (plat >= nyc_lat_min) & (plat <= nyc_lat_max)
mask &= (dlat >= nyc_lat_min) & (dlat <= nyc_lat_max)

d_lon = np.abs(dlon - plon)
d_lat = np.abs(dlat - plat)
mask &= (d_lon + d_lat) > 1e-4
mask &= (d_lon <= 0.3) & (d_lat <= 0.3)

train = train.loc[mask].reset_index(drop=True)



## === cell 3
test.shape



## === cell 4
y = train["fare_amount"].to_numpy()
n_train = len(train)
n_test = len(test)
test_id = test["key"].copy()




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

    data["hour"] = dt.dt.hour.astype("int16").astype("category")
    data["day_of_week"] = dt.dt.dayofweek.astype("int16").astype("category")

    dom = dt.dt.day.astype("int16")
    wom = pd.cut(
        dom,
        bins=[0, 7, 14, 21, 28, 31],
        labels=["first", "second", "third", "fourth", "fifth"],
        include_lowest=True,
        right=True,
    )
    data["week_of_month"] = wom.astype("category")

    data["month"] = dt.dt.month.astype("int16").astype("category")
    data["year"] = dt.dt.year.astype("int16").astype("category")

    return data




## === cell 7
def add_geo_features(data):
    p_lo = data["pickup_longitude"].to_numpy()
    p_la = data["pickup_latitude"].to_numpy()
    d_lo = data["dropoff_longitude"].to_numpy()
    d_la = data["dropoff_latitude"].to_numpy()

    dlo = d_lo - p_lo
    dla = d_la - p_la
    abs_dlo = np.abs(dlo)
    abs_dla = np.abs(dla)

    data["abs_diff_longitude"] = abs_dlo
    data["abs_diff_latitude"] = abs_dla
    data["manhattan_distance"] = abs_dlo + abs_dla

    sq_long = abs_dlo * abs_dlo
    sq_lat = abs_dla * abs_dla
    data["squared_long"] = sq_long
    data["squared_lat"] = sq_lat
    data["euclid_disance"] = np.sqrt(sq_long + sq_lat)

    return data




## === cell 8
train_fe = train[
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
].copy()
train_fe = add_time_features(train_fe)
train_fe = add_geo_features(train_fe)

test_fe = test[
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
].copy()
test_fe = add_time_features(test_fe)
test_fe = add_geo_features(test_fe)



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

train_fe = train_fe[features]
test_fe = test_fe[features]

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

for c in cat_cols:
    train_fe[c] = train_fe[c].astype("category")
    test_fe[c] = test_fe[c].astype("category")

pre = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.uint8),
            cat_cols,
        ),
        ("num", "passthrough", num_cols),
    ],
    sparse_threshold=1.0,
)

x = pre.fit_transform(train_fe)
x_test = pre.transform(test_fe)



## === cell 10
if sparse.issparse(x):
    x = x.tocsr(copy=False)
    if x.data.dtype != np.float32:
        x.data = x.data.astype(np.float32, copy=False)
    x.sort_indices()
else:
    x = np.asarray(x, dtype=np.float32, order="C")

if sparse.issparse(x_test):
    x_test = x_test.tocsr(copy=False)
    if x_test.data.dtype != np.float32:
        x_test.data = x_test.data.astype(np.float32, copy=False)
    x_test.sort_indices()
else:
    x_test = np.asarray(x_test, dtype=np.float32, order="C")



## === cell 11
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)



## === cell 12
model.fit(x, y)



## === cell 13
try:
    feat_names = list(pre.get_feature_names_out())
    feat_imp = pd.DataFrame({"imp": model.feature_importances_}, index=feat_names)
    feat_imp.sort_values("imp", ascending=False, inplace=True)
    feat_imp.head(10)
except Exception:
    pass



## === cell 14
test_pred = model.predict(x_test)



## === cell 15
samp.head()



## === cell 16
sub = pd.DataFrame()
sub["key"] = test_id.values
sub["fare_amount"] = test_pred

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
