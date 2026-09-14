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

# 5. Target score

3.76038

# 6. Current score

5.07912

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.07912) has done: 'The timeout is dominated by reading/processing 10M training rows and fitting a default `RandomForestRegressor` (very expensive with ~10M samples, even with `set_rf_samples(10000)` because the sklearnex patch may bypass fastai’s sampling hook). I keep the same model and training flow, but make the sampling deterministic and explicit by actually subsetting `X_train/y_train` to the same 10,000 rows before `.fit()` (provably equivalent to the intended `set_rf_samples(10000)` behavior). I also remove redundant NA/outlier checks that scan the full 10M frame, and optimize feature engineering to avoid repeated pandas overhead while preserving identical derived columns. These changes reduce both the asymptotic work and constant factors while keeping predictions/evaluation semantics the same.'
- What this solution (achieved 5.07912) has done: 'The crash happens because `set_rf_samples` is a fastai v0.7-era helper and is not available in fastai v2, so the name is undefined at runtime. Cell 25 already implements the intended behavior (fit the RandomForest on a deterministic 10k subsample), so the safest minimal fix is to remove the obsolete call. To preserve the notebook’s execution flow without changing modeling semantics, cell 24 should become a no-op with a brief comment explaining that sampling is handled later. This keeps the same training approach and results, and avoids introducing any new logic.'

# 9. Code solution

## === cell 0
from fastai.imports import *

from fastai.tabular.all import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display

import os, random

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
PATH = "../input"

usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "fare_amount": "float32",
    "pickup_datetime": "object",  # parse later (faster than parse_dates during read for huge CSV)
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

_read_csv_kwargs = dict(
    nrows=10000000,
    usecols=usecols,
    dtype=dtypes,
    na_filter=True,
)
try:
    df_raw = pd.read_csv(
        f"{PATH}/train.csv",
        engine="pyarrow",
        **_read_csv_kwargs,
    )
except Exception:
    df_raw = pd.read_csv(f"{PATH}/train.csv", **_read_csv_kwargs)

df_raw.dropna(axis=0, how="any", inplace=True)
df_raw = df_raw[(df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)]




## === cell 2
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)




## === cell 3
pass




## === cell 4
def add_datepart_fast(df, field_name="pickup_datetime", drop=True, time=True):
    dt = pd.to_datetime(df[field_name], errors="coerce", utc=False)
    df[field_name + "Year"] = dt.dt.year.astype("int16", copy=False)
    df[field_name + "Month"] = dt.dt.month.astype("int8", copy=False)
    df[field_name + "Week"] = dt.dt.isocalendar().week.astype("int16", copy=False)
    df[field_name + "Day"] = dt.dt.day.astype("int8", copy=False)
    df[field_name + "Dayofweek"] = dt.dt.dayofweek.astype("int8", copy=False)
    df[field_name + "Dayofyear"] = dt.dt.dayofyear.astype("int16", copy=False)
    if time:
        df[field_name + "Hour"] = dt.dt.hour.astype("int8", copy=False)
        df[field_name + "Minute"] = dt.dt.minute.astype("int8", copy=False)
        df[field_name + "Second"] = dt.dt.second.astype("int8", copy=False)
    if drop:
        df.drop(columns=[field_name], inplace=True)


add_datepart_fast(df_raw, "pickup_datetime", drop=True, time=True)



## === cell 5
pass




## === cell 6
def distance(data):
    plo = data["pickup_longitude"].to_numpy(copy=False)
    dlo = data["dropoff_longitude"].to_numpy(copy=False)
    pla = data["pickup_latitude"].to_numpy(copy=False)
    dla = data["dropoff_latitude"].to_numpy(copy=False)

    data["longitutde_traversed"] = np.abs(dlo - plo).astype(np.float32, copy=False)
    data["latitude_traversed"] = np.abs(dla - pla).astype(np.float32, copy=False)


distance(df_raw)



## === cell 7
pass



## === cell 8
pass



## === cell 9
df_raw.dropna(axis=0, how="any", inplace=True)



## === cell 10
_ = df_raw.shape



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
_ = len(df_raw)



## === cell 15
df_raw.reset_index(drop=True, inplace=True)



## === cell 16
outliers = []



## === cell 17
features = ["longitutde_traversed", "latitude_traversed"]

_sample_n = 10000
if len(df_raw) > _sample_n:
    _iqr_idx = np.random.RandomState(0).choice(
        len(df_raw), size=_sample_n, replace=False
    )
    _vals_iqr = (
        df_raw.loc[_iqr_idx, features]
        .to_numpy(copy=False)
        .astype(np.float32, copy=False)
    )
else:
    _vals_iqr = df_raw[features].to_numpy(copy=False).astype(np.float32, copy=False)

q1 = np.percentile(_vals_iqr, 25, axis=0)
q3 = np.percentile(_vals_iqr, 75, axis=0)
step = 10.0 * (q3 - q1)

lower = q1 - step
upper = q3 + step

outliers = []



## === cell 18
_ = len(outliers) / len(df_raw) if len(df_raw) else 0.0



## === cell 19
pass



## === cell 20
pass



## === cell 21
y = df_raw.fare_amount
df_raw.drop("fare_amount", axis=1, inplace=True)



## === cell 22
X_train, X_valid, y_train, y_valid = train_test_split(
    df_raw, y, test_size=10000, random_state=0
)




## === cell 23
def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    if hasattr(m, "rf_idxs"):
        idxs = m.rf_idxs
        X_tr = X_train.iloc[idxs]
        y_tr = y_train.iloc[idxs]
    else:
        X_tr = X_train
        y_tr = y_train

    pred_tr = m.predict(X_tr)
    pred_va = m.predict(X_valid)

    res = [
        rmse(pred_tr, y_tr),
        rmse(pred_va, y_valid),
        m.score(X_tr, y_tr),
        m.score(X_valid, y_valid),
    ]
    print(res)




## === cell 24
pass


## === cell 25
_rs = np.random.RandomState(0)
if len(X_train) > 10000:
    _fit_idx = _rs.choice(len(X_train), size=10000, replace=False)
    X_fit = X_train.iloc[_fit_idx]
    y_fit = y_train.iloc[_fit_idx]
else:
    X_fit = X_train
    y_fit = y_train

m = RandomForestRegressor(n_jobs=-1, random_state=0)
m.fit(X_fit, y_fit)
print_score(m)



## === cell 26
try:
    test_set = pd.read_csv(
        f"{PATH}/test.csv",
        engine="pyarrow",
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
            "pickup_datetime": "object",  # parse later (faster)
            "pickup_longitude": "float32",
            "pickup_latitude": "float32",
            "dropoff_longitude": "float32",
            "dropoff_latitude": "float32",
            "passenger_count": "uint8",
        },
    )
except Exception:
    test_set = pd.read_csv(
        f"{PATH}/test.csv",
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
            "pickup_datetime": "object",
            "pickup_longitude": "float32",
            "pickup_latitude": "float32",
            "dropoff_longitude": "float32",
            "dropoff_latitude": "float32",
            "passenger_count": "uint8",
        },
    )



## === cell 27
test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)



## === cell 28
add_datepart_fast(test_set, "pickup_datetime", drop=True, time=True)
distance(test_set)



## === cell 29
test_predictions = m.predict(test_set)



## === cell 30
submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission.to_csv("submissions.csv", index=False)
