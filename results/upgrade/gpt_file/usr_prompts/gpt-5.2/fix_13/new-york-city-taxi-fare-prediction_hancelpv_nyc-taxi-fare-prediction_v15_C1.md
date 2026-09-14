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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 1))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(42)

print(os.listdir("../input"))

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



## === cell 1
test = pd.read_csv(
    "../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
    parse_dates=["pickup_datetime"],
    low_memory=False,
)
samp = pd.read_csv("../input/sample_submission.csv")
test_id = test["key"].to_numpy(copy=False)




## === cell 2
def _filter_train_df(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.dropna(how="any", axis="rows")
    train_df = train_df[(train_df.fare_amount > 0) & (train_df["passenger_count"] <= 6)]

    mask = (
        (train_df.pickup_latitude > -90)
        & (train_df.pickup_latitude < 90)
        & (train_df.dropoff_latitude > -90)
        & (train_df.dropoff_latitude < 90)
        & (train_df.pickup_longitude > -180)
        & (train_df.pickup_longitude < 180)
        & (train_df.dropoff_longitude > -180)
        & (train_df.dropoff_longitude < 180)
    )
    return train_df[mask]




## === cell 3
_DOW_CATS = np.array(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    dtype=object,
)
_WOM_CATS = np.array(["first", "second", "third", "fourth", "fifth"], dtype=object)

_HOUR_CATS_LIST = list(range(24))
_MONTH_CATS_LIST = list(range(1, 13))
_DOW_CATS_LIST = list(_DOW_CATS)
_WOM_CATS_LIST = list(_WOM_CATS)


def add_time_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=False)
    if pd.api.types.is_datetime64tz_dtype(dt):
        dt = dt.dt.tz_convert(None)

    hour = dt.dt.hour.astype(np.int16, copy=False)
    month = dt.dt.month.astype(np.int16, copy=False)
    year = dt.dt.year.astype(np.int16, copy=False)

    dow_idx = dt.dt.dayofweek.to_numpy(dtype=np.int8, copy=False)  # Monday=0
    day_of_week = pd.Categorical.from_codes(
        dow_idx, categories=_DOW_CATS_LIST, ordered=False
    )

    dom = dt.dt.day.to_numpy(dtype=np.int16, copy=False)
    wom_code = np.empty(dom.shape[0], dtype=np.int8)
    wom_code[dom <= 7] = 0
    wom_code[(dom > 7) & (dom <= 14)] = 1
    wom_code[(dom > 14) & (dom <= 21)] = 2
    wom_code[(dom > 21) & (dom <= 28)] = 3
    wom_code[dom > 28] = 4
    week_of_month = pd.Categorical.from_codes(
        wom_code, categories=_WOM_CATS_LIST, ordered=False
    )

    df["hour"] = pd.Categorical(hour, categories=_HOUR_CATS_LIST, ordered=False)
    df["month"] = pd.Categorical(month, categories=_MONTH_CATS_LIST, ordered=False)
    df["year"] = pd.Categorical(year, ordered=False)
    df["day_of_week"] = day_of_week
    df["week_of_month"] = week_of_month
    return df




## === cell 4
def add_geo_features(data):
    plon = data["pickup_longitude"].astype(np.float32, copy=False)
    plat = data["pickup_latitude"].astype(np.float32, copy=False)
    dlon = data["dropoff_longitude"].astype(np.float32, copy=False)
    dlat = data["dropoff_latitude"].astype(np.float32, copy=False)

    abs_diff_long = (dlon - plon).abs()
    abs_diff_lat = (dlat - plat).abs()
    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat

    manhattan = abs_diff_long + abs_diff_lat
    data["manhattan_distance"] = manhattan

    squared_long = np.square(abs_diff_long.to_numpy(dtype=np.float32, copy=False))
    squared_lat = np.square(abs_diff_lat.to_numpy(dtype=np.float32, copy=False))
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat

    data["euclid_distance"] = np.sqrt(squared_long + squared_lat).astype(
        np.float32, copy=False
    )
    return data




## === cell 5
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "euclid_distance",
]

test_feat = test.drop(columns=["key"])
test_feat = add_time_features_fast(test_feat)
test_feat = add_geo_features(test_feat)
if "pickup_datetime" in test_feat.columns:
    test_feat.drop(["pickup_datetime"], axis=1, inplace=True)
test_feat = test_feat[features]

test_feat["hour"] = test_feat["hour"].cat.set_categories(_HOUR_CATS_LIST)
test_feat["month"] = test_feat["month"].cat.set_categories(_MONTH_CATS_LIST)
test_feat["day_of_week"] = test_feat["day_of_week"].cat.set_categories(_DOW_CATS_LIST)
test_feat["week_of_month"] = test_feat["week_of_month"].cat.set_categories(
    _WOM_CATS_LIST
)



## === cell 6
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "euclid_distance",
]

pre = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore", sparse_output=True, dtype=np.float32
            ),
            cat_cols,
        ),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)



## === cell 7
TRAIN_NROWS = 1_000_000
CHUNK_SIZE = 200_000  # keep bounded memory but reduce per-chunk overhead

train_iter = pd.read_csv(
    "../input/train.csv",
    nrows=TRAIN_NROWS,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
    low_memory=False,
    chunksize=CHUNK_SIZE,
)

x_blocks = []
y_blocks = []
pre_fitted = False

for chunk in train_iter:
    chunk = _filter_train_df(chunk)
    if len(chunk) == 0:
        continue

    y_blocks.append(chunk["fare_amount"].to_numpy(dtype=np.float32, copy=False))

    feat = chunk.drop(["fare_amount"], axis=1)
    feat = add_time_features_fast(feat)
    feat = add_geo_features(feat)
    if "pickup_datetime" in feat.columns:
        feat.drop(["pickup_datetime"], axis=1, inplace=True)
    feat = feat[features]

    feat["hour"] = feat["hour"].cat.set_categories(_HOUR_CATS_LIST)
    feat["month"] = feat["month"].cat.set_categories(_MONTH_CATS_LIST)
    feat["day_of_week"] = feat["day_of_week"].cat.set_categories(_DOW_CATS_LIST)
    feat["week_of_month"] = feat["week_of_month"].cat.set_categories(_WOM_CATS_LIST)

    if not pre_fitted:
        x0 = pre.fit_transform(feat)
        x_blocks.append(x0)
        pre_fitted = True
    else:
        x_blocks.append(pre.transform(feat))

if not pre_fitted:
    raise RuntimeError("No training rows left after filtering; cannot train.")

y = np.concatenate(y_blocks, axis=0).astype(np.float32, copy=False)

x = (
    sparse.vstack(x_blocks, format="csr", dtype=np.float32)
    if sparse.issparse(x_blocks[0])
    else np.vstack(x_blocks).astype(np.float32, copy=False)
)
x_test = pre.transform(test_feat)
if sparse.issparse(x_test):
    x_test = x_test.tocsr()
    if x_test.dtype != np.float32:
        x_test = x_test.astype(np.float32)
else:
    x_test = x_test.astype(np.float32, copy=False)

USING_SPARSE = sparse.issparse(x)
print(
    "Prepared matrices. Using sparse:",
    USING_SPARSE,
    "Train shape:",
    x.shape,
    "Test shape:",
    x_test.shape,
    "Train rows used after filtering:",
    y.shape[0],
)



## === cell 8
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
    warm_start=False,
    bootstrap=True,
    verbose=0,
)



## === cell 9
x_rf = x.tocsr() if sparse.issparse(x) else x
x_test_rf = x_test.tocsr() if sparse.issparse(x_test) else x_test

model_1.fit(x_rf, y)

model_1_pred = model_1.predict(x_test_rf)
model_1_pred = np.maximum(model_1_pred, 0.0)

sub_1 = pd.DataFrame({"key": test_id, "fare_amount": model_1_pred})
sub_1.to_csv("submission_rf.csv", index=False)

print("Wrote submission_rf.csv with shape:", sub_1.shape)



## === cell 10
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler



## === cell 11
num_features = x.shape[1]
print("Num features:", num_features)



## === cell 12
x_mlp = x.tocsr() if sparse.issparse(x) else x
x_test_mlp = x_test.tocsr() if sparse.issparse(x_test) else x_test

scaler = StandardScaler(with_mean=False)
x_scaled = scaler.fit_transform(x_mlp)
x_test_scaled = scaler.transform(x_test_mlp)

if sparse.issparse(x_scaled) and not sparse.isspmatrix_csr(x_scaled):
    x_scaled = x_scaled.tocsr()
if sparse.issparse(x_test_scaled) and not sparse.isspmatrix_csr(x_test_scaled):
    x_test_scaled = x_test_scaled.tocsr()



## === cell 13
mlp = MLPRegressor(
    hidden_layer_sizes=(30, 15, 7, 3),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=1024,
    learning_rate_init=0.001,
    max_iter=5,  # matches original epochs=5 intent
    shuffle=True,
    random_state=42,
    verbose=True,
)



## === cell 14
mlp.fit(x_scaled, y)

test_pred = mlp.predict(x_test_scaled).reshape(-1)
test_pred = np.maximum(test_pred, 0.0)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission_nn.csv", index=False)

print("Wrote submission_nn.csv with shape:", sub.shape)
print(sub.head())
