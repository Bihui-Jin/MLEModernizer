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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
from fastai.imports import *

import os, random, time, math, re

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

from IPython.display import display

try:
    from fastai.structured import set_rf_samples  # type: ignore
except Exception:
    try:
        from fastai.tabular.all import set_rf_samples  # type: ignore
    except Exception:

        def set_rf_samples(n):
            try:
                from sklearn.ensemble import forest

                forest._generate_sample_indices = (  # type: ignore
                    lambda rs, n_samples: rs.randint(0, n_samples, n)
                )
            except Exception:
                pass




## === cell 1
def add_datepart(df, fldname, drop=True, time=False, errors="raise"):
    """
    Minimal fastai v0.7/v1-style add_datepart implementation used in many NYC taxi notebooks.
    Expands a datetime column into multiple date/time-related columns.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    fld = df[fldname]

    targ_pre = re.sub("[Dd]ate$", "", fldname)
    dt = fld.dt

    df[targ_pre + "Year"] = dt.year.astype(np.int16, copy=False)
    df[targ_pre + "Month"] = dt.month.astype(np.int8, copy=False)
    df[targ_pre + "Week"] = dt.isocalendar().week.astype(np.int16, copy=False)
    df[targ_pre + "Day"] = dt.day.astype(np.int8, copy=False)
    df[targ_pre + "Dayofweek"] = dt.dayofweek.astype(np.int8, copy=False)
    df[targ_pre + "Dayofyear"] = dt.dayofyear.astype(np.int16, copy=False)

    df[targ_pre + "Is_month_end"] = dt.is_month_end
    df[targ_pre + "Is_month_start"] = dt.is_month_start
    df[targ_pre + "Is_quarter_end"] = dt.is_quarter_end
    df[targ_pre + "Is_quarter_start"] = dt.is_quarter_start
    df[targ_pre + "Is_year_end"] = dt.is_year_end
    df[targ_pre + "Is_year_start"] = dt.is_year_start

    if time:
        df[targ_pre + "Hour"] = dt.hour.astype(np.int8, copy=False)
        df[targ_pre + "Minute"] = dt.minute.astype(np.int8, copy=False)
        df[targ_pre + "Second"] = dt.second.astype(np.int8, copy=False)

    df[targ_pre + "Elapsed"] = (fld.view("int64") // 10**9).astype(np.int64, copy=False)

    if drop:
        df.drop(columns=[fldname], inplace=True)




## === cell 2
PATH = "../input"
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
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

read_csv_kwargs = dict(
    nrows=10000000,
    usecols=usecols,
    dtype=dtypes,
    parse_dates=["pickup_datetime"],
    low_memory=False,
    memory_map=True,
)

try:
    df_raw = pd.read_csv(f"{PATH}/train.csv", engine="pyarrow", **read_csv_kwargs)
except Exception:
    df_raw = pd.read_csv(f"{PATH}/train.csv", engine="c", **read_csv_kwargs)

key = df_raw["key"]
df_raw.drop(columns=["key"], inplace=True)




## === cell 3
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)




## === cell 4
pass



## === cell 5
if pd.api.types.is_datetime64tz_dtype(df_raw["pickup_datetime"]):
    df_raw["pickup_datetime"] = df_raw["pickup_datetime"].dt.tz_convert(None)

add_datepart(df_raw, "pickup_datetime", drop=True, time=True)



## === cell 6
pass




## === cell 7
def distance(data):
    dlo = data["dropoff_longitude"].to_numpy(copy=False)
    plo = data["pickup_longitude"].to_numpy(copy=False)
    dla = data["dropoff_latitude"].to_numpy(copy=False)
    pla = data["pickup_latitude"].to_numpy(copy=False)

    data["longitutde_traversed"] = np.abs(dlo - plo).astype(np.float32, copy=False)
    data["latitude_traversed"] = np.abs(dla - pla).astype(np.float32, copy=False)




## === cell 8
distance(df_raw)



## === cell 9
pass



## === cell 10
df_raw.dropna(axis=0, how="any", inplace=True)



## === cell 11
df_raw.shape



## === cell 12
pass



## === cell 13
df_raw.passenger_count.value_counts()



## === cell 14
df_raw = df_raw[(df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)]



## === cell 15
len(df_raw)



## === cell 16
df_raw.reset_index(drop=True, inplace=True)



## === cell 17
num_cols = df_raw.columns  # all should be numeric here
vals2_cols = ["longitutde_traversed", "latitude_traversed"]

arr = df_raw.to_numpy(copy=False)
if arr.dtype != np.float32:
    arr = arr.astype(np.float32, copy=False)

if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)

q1 = np.quantile(arr, 0.25, axis=0, method="linear").astype(np.float32, copy=False)
q3 = np.quantile(arr, 0.75, axis=0, method="linear").astype(np.float32, copy=False)

step = np.float32(2.0) * (q3 - q1)
lower = q1 - step
upper = q3 + step

keep_mask = np.logical_and(arr >= lower, arr <= upper).all(axis=1)

col_to_idx = {c: i for i, c in enumerate(num_cols)}
idxs = np.fromiter(
    (col_to_idx[c] for c in vals2_cols), dtype=np.int32, count=len(vals2_cols)
)

q1_2 = q1[idxs]
q3_2 = q3[idxs]
step2 = np.float32(10.0) * (q3_2 - q1_2)
lower2 = q1_2 - step2
upper2 = q3_2 + step2

arr2 = arr[:, idxs]
keep_mask2 = np.logical_and(arr2 >= lower2, arr2 <= upper2).all(axis=1)



## === cell 18
1.0 - keep_mask.mean()



## === cell 19
1.0 - keep_mask2.mean()



## === cell 20
df_raw = df_raw[keep_mask & keep_mask2].reset_index(drop=True)



## === cell 21
len(df_raw)



## === cell 22
y = df_raw.fare_amount
df_raw.drop("fare_amount", axis=1, inplace=True)



## === cell 23
n = len(df_raw)
valid_size = 10000
rng = np.random.RandomState(42)
perm = rng.permutation(n)
idx_valid = perm[:valid_size].astype(np.int32, copy=False)
idx_train = perm[valid_size:].astype(np.int32, copy=False)

X_all_np = df_raw.to_numpy(copy=False)
if X_all_np.dtype != np.float32:
    X_all_np = X_all_np.astype(np.float32, copy=False)
if not X_all_np.flags["C_CONTIGUOUS"]:
    X_all_np = np.ascontiguousarray(X_all_np)

y_all_np = y.to_numpy(copy=False)
if y_all_np.dtype != np.float32:
    y_all_np = y_all_np.astype(np.float32, copy=False)
if not y_all_np.flags["C_CONTIGUOUS"]:
    y_all_np = np.ascontiguousarray(y_all_np)

X_train_np = X_all_np[idx_train]
X_valid_np = X_all_np[idx_valid]
y_train_np = y_all_np[idx_train]
y_valid_np = y_all_np[idx_valid]




## === cell 24
def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    n = 10000
    Xtr = X_train_np[:n]
    ytr = y_train_np[:n]
    res = [
        rmse(m.predict(Xtr), ytr),
        rmse(m.predict(X_valid_np), y_valid_np),
        m.score(Xtr, ytr),
        m.score(X_valid_np, y_valid_np),
    ]
    print(res)




## === cell 25
set_rf_samples(10000)



## === cell 26
m = RandomForestRegressor(n_jobs=-1, random_state=42)
t0 = time.time()
m.fit(X_train_np, y_train_np)
print(f"fit_seconds={time.time()-t0:.3f}")
print_score(m)



## === cell 27
pass



## === cell 28
pass



## === cell 29
test_read_csv_kwargs = dict(
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
        "key": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
    parse_dates=["pickup_datetime"],
    low_memory=False,
    memory_map=True,
)
try:
    test_set = pd.read_csv(f"{PATH}/test.csv", engine="pyarrow", **test_read_csv_kwargs)
except Exception:
    test_set = pd.read_csv(f"{PATH}/test.csv", engine="c", **test_read_csv_kwargs)



## === cell 30
test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)



## === cell 31
add_datepart(test_set, "pickup_datetime", drop=True, time=True)
distance(test_set)



## === cell 32
X_test = test_set.to_numpy(copy=False)
if X_test.dtype != np.float32:
    X_test = X_test.astype(np.float32, copy=False)
if not X_test.flags["C_CONTIGUOUS"]:
    X_test = np.ascontiguousarray(X_test)

test_predictions = m.predict(X_test)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission.to_csv("submissions.csv", index=False)
