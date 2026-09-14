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
import os
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display

np.random.seed(42)




## === cell 1
def add_datepart(df, fldname, drop=True, time=False, errors="coerce"):
    """
    Bugfix: handle tz-aware datetimes (e.g. datetime64[ns, UTC]) which break np.issubdtype.
    Also ensures the original datetime column is converted to naive datetime64[ns] before extracting parts.
    """
    if fldname not in df.columns:
        raise KeyError(f"{fldname} not found in dataframe columns")

    s = df[fldname]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors=errors, utc=False)
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_convert(None)
    df[fldname] = s

    fld = df[fldname]
    targ_pre = fldname
    attrs = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    if time:
        attrs += ["Hour", "Minute", "Second"]

    for n in attrs:
        if n == "Week":
            df[targ_pre + n] = fld.dt.isocalendar().week.astype("int16")
        else:
            df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + "Elapsed"] = (
        fld.to_numpy().astype("datetime64[ns]").view("int64") // 10**9
    )

    if drop:
        df.drop(columns=[fldname], inplace=True)


def set_rf_samples(n):
    try:
        from sklearn.ensemble import _forest
    except Exception:
        return

    def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
        random_instance = np.random.RandomState(random_state)
        return random_instance.choice(n_samples, n_samples_bootstrap, replace=False)

    _forest._generate_sample_indices = _generate_sample_indices
    _forest._generate_unsampled_indices = None  # not used in our flow




## === cell 2
PATH = "../input"
if not os.path.exists(PATH):
    if os.path.exists("/kaggle/input"):
        PATH = "/kaggle/input"
    elif os.path.exists("/kaggle/data"):
        PATH = "/kaggle/data"
    else:
        PATH = "../input"

train_path = f"{PATH}/train.csv"
test_path = f"{PATH}/test.csv"

NROWS = int(os.environ.get("NROWS", "1000000"))  # default 1M for runtime safety

train_dtypes = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df_raw = pd.read_csv(
    train_path, nrows=NROWS, dtype=train_dtypes, parse_dates=["pickup_datetime"]
)




## === cell 3
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)




## === cell 4
pass



## === cell 5
add_datepart(df_raw, "pickup_datetime", drop=True, time=True)




## === cell 6
def distance(data):
    """
    Bugfix: return the modified dataframe to ensure consistent usage in both train and test flows.
    (Keeps core feature logic identical.)
    """
    data["longitutde_traversed"] = (
        data.dropoff_longitude - data.pickup_longitude
    ).abs()
    data["latitude_traversed"] = (data.dropoff_latitude - data.pickup_latitude).abs()
    return data


df_raw = distance(df_raw)



## === cell 7
pass



## === cell 8
pass



## === cell 9
df_raw.dropna(axis=0, how="any", inplace=True)



## === cell 10
pass



## === cell 11
key = df_raw.key
df_raw.drop("key", axis=1, inplace=True)



## === cell 12
pass



## === cell 13
df_raw = df_raw[(df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)]



## === cell 14
pass



## === cell 15
df_raw.reset_index(drop=True, inplace=True)



## === cell 16
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

Q1 = df_raw[num_cols].quantile(0.25, interpolation="linear")
Q3 = df_raw[num_cols].quantile(0.75, interpolation="linear")
step = 2.0 * (Q3 - Q1)

lower = Q1 - step
upper = Q3 + step

outlier_mask = ((df_raw[num_cols] < lower) | (df_raw[num_cols] > upper)).any(axis=1)



## === cell 17
outlier_mask.mean()



## === cell 18
trav_cols = ["longitutde_traversed", "latitude_traversed"]

Q1_t = df_raw[trav_cols].quantile(0.25, interpolation="linear")
Q3_t = df_raw[trav_cols].quantile(0.75, interpolation="linear")
step_t = 10.0 * (Q3_t - Q1_t)

lower_t = Q1_t - step_t
upper_t = Q3_t + step_t

outlier_mask_t = ((df_raw[trav_cols] < lower_t) | (df_raw[trav_cols] > upper_t)).any(
    axis=1
)



## === cell 19
outlier_mask_t.mean()



## === cell 20
df = df_raw.loc[~outlier_mask_t].reset_index(drop=True)



## === cell 21
len(df)



## === cell 22
y = df.fare_amount
df.drop("fare_amount", axis=1, inplace=True)



## === cell 23
X_train, X_valid, y_train, y_valid = train_test_split(
    df, y, test_size=10000, random_state=42
)




## === cell 24
def rmse(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    res = [
        rmse(m.predict(X_train), y_train),
        rmse(m.predict(X_valid), y_valid),
        m.score(X_train, y_train),
        m.score(X_valid, y_valid),
    ]
    print(res)




## === cell 25
set_rf_samples(10000)



## === cell 26
m = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
m.fit(X_train, y_train)
print_score(m)



## === cell 27
test_dtypes = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_set = pd.read_csv(test_path, dtype=test_dtypes, parse_dates=["pickup_datetime"])



## === cell 28
test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)



## === cell 29
add_datepart(test_set, "pickup_datetime", drop=True, time=True)
test_set = distance(test_set)

missing_cols = [c for c in X_train.columns if c not in test_set.columns]
for c in missing_cols:
    test_set[c] = 0
extra_cols = [c for c in test_set.columns if c not in X_train.columns]
if extra_cols:
    test_set.drop(columns=extra_cols, inplace=True)
test_set = test_set[X_train.columns]



## === cell 30
test_predictions = m.predict(test_set)



## === cell 31
submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
submission.head()
