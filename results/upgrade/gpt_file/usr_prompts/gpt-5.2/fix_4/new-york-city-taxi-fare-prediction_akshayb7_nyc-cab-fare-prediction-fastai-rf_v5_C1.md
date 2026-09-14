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

3.83596

# 6. Current score

5.50976

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.15477) has done: 'I remove the deprecated fastai v0 imports and replace only the missing helper utilities (`add_datepart`, `set_rf_samples`, `reset_rf_samples`, `rf_feat_importance`) with small local equivalents so the pipeline runs in this environment. I fix the outlier loop crashing on non-numeric columns by restricting it to numeric features, and fix a logic bug where you computed outlier-filtered `df` but then trained on `df_raw`/`y` instead. I also ensure the train/test feature engineering produces identical columns (especially the date parts), and that `train_test_split` and `RandomForestRegressor` are correctly imported and executed. Finally, the code always write a valid `submission.csv` with columns `key,fare_amount` in the working directory.'
- What this solution (achieved 5.42484) has done: 'Your RMSE is worse than the target, so we should legitimately improve generalization with minimal changes while keeping the same RandomForest + same feature set and training flow. The biggest safe gain here is to (1) remove obviously invalid training rows (NYC bounding box + positive fare + sane coordinates) which otherwise teach the model nonsense, and (2) use a log1p target transform during training and invert at prediction time, which typically reduces RMSE for skewed fare distributions without changing the model type or loss. I’m also switching the validation split to a proportion (not an absolute 10k) so it scales correctly with your 10M sample and better reflects performance. Finally, I clip negative predictions to 0 to avoid invalid fares hurting RMSE.'
- What this solution (achieved 5.50976) has done: 'Your current RMSE (5.42484) is worse than the target (3.83596), so we should legitimately improve generalization with the smallest safe changes while keeping the same RandomForest + same features/training flow. The biggest issue is that you train on a log1p-transformed target but your validation RMSE is computed in log space, which misleads tuning; we compute RMSE in original fare space (via expm1) without changing training. Next, we replace the overly-aggressive global “outlier over all numeric columns” removal (which can drop many valid rows and distort the target distribution) with only the already-present distance-based outlier filter, preserving the same core cleaning idea but making it less destructive. Finally, we add a minimal derived “manhattan_distance” feature from your existing traversed deltas (same feature family, no model change), which typically improves taxi-fare RF performance.'

# 9. Code solution

## === cell 0
import os
import gc
import math
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display


def add_datepart(df, fldname, drop=True, time=False, errors="coerce"):
    """
    Adds columns relevant to a date in the column `fldname`.
    Mimics the classic fastai v0 structured.add_datepart behavior sufficiently for this notebook.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = pd.to_datetime(fld, errors=errors, utc=False)
    fld = df[fldname]

    prefix = fldname
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
    for attr in attrs:
        if attr == "Week":
            df[prefix + attr] = fld.dt.isocalendar().week.astype("int16")
        else:
            df[prefix + attr] = getattr(fld.dt, attr.lower())
    if time:
        df[prefix + "Hour"] = fld.dt.hour
        df[prefix + "Minute"] = fld.dt.minute
        df[prefix + "Second"] = fld.dt.second

    if drop:
        df.drop(columns=[fldname], inplace=True)
    return df


_old_generate_sample_indices = None


def set_rf_samples(n):
    """
    Limit bootstrap sample size per tree to n (fastai v0 trick).
    """
    global _old_generate_sample_indices
    from sklearn.ensemble import _forest

    if _old_generate_sample_indices is None:
        _old_generate_sample_indices = _forest._generate_sample_indices

    def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
        return _old_generate_sample_indices(random_state, n_samples, n)

    _forest._generate_sample_indices = _generate_sample_indices


def reset_rf_samples():
    """
    Restore sklearn's original bootstrap sampling.
    """
    global _old_generate_sample_indices
    if _old_generate_sample_indices is None:
        return
    from sklearn.ensemble import _forest

    _forest._generate_sample_indices = _old_generate_sample_indices
    _old_generate_sample_indices = None


def rf_feat_importance(m, df):
    """
    Return feature importances DataFrame (fastai v0 style).
    """
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)




## === cell 1
PATH = "/kaggle/input"  # Kaggle canonical path
train_path = f"{PATH}/train.csv"
test_path = f"{PATH}/test.csv"

df_raw = pd.read_csv(train_path, nrows=10_000_000)




## === cell 2
def display_all(df):
    with pd.option_context("display.max_rows", 1000), pd.option_context(
        "display.max_columns", 1000
    ):
        display(df)




## === cell 3
display_all(df_raw.head(5))



## === cell 4
add_datepart(df_raw, "pickup_datetime", drop=True, time=True)



## === cell 5
display_all(df_raw.head(5))




## === cell 6
def distance(data):
    data["longitutde_traversed"] = (
        data.dropoff_longitude - data.pickup_longitude
    ).abs()
    data["latitude_traversed"] = (data.dropoff_latitude - data.pickup_latitude).abs()
    data["manhattan_distance"] = (
        data["longitutde_traversed"] + data["latitude_traversed"]
    )




## === cell 7
distance(df_raw)



## === cell 8
display_all(df_raw.head(2).T)



## === cell 9
df_raw.isnull().sum()



## === cell 10
df_raw.dropna(axis=0, how="any", inplace=True)



## === cell 11
df_raw.shape



## === cell 12
key = df_raw.key
df_raw.drop("key", axis=1, inplace=True)



## === cell 13
df_raw.passenger_count.value_counts()



## === cell 14
df_raw = df_raw[(df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)]



## === cell 15
len(df_raw)



## === cell 16
df_raw.reset_index(drop=True, inplace=True)



## === cell 17
nyc_min_long, nyc_max_long = -74.3, -73.7
nyc_min_lat, nyc_max_lat = 40.5, 41.0

coord_mask = (
    df_raw["pickup_longitude"].between(nyc_min_long, nyc_max_long)
    & df_raw["dropoff_longitude"].between(nyc_min_long, nyc_max_long)
    & df_raw["pickup_latitude"].between(nyc_min_lat, nyc_max_lat)
    & df_raw["dropoff_latitude"].between(nyc_min_lat, nyc_max_lat)
)

fare_mask = (df_raw["fare_amount"] > 0) & (df_raw["fare_amount"] < 250)

dist_mask = (df_raw["longitutde_traversed"] >= 0) & (
    df_raw["longitutde_traversed"] < 2.0
)
dist_mask &= (df_raw["latitude_traversed"] >= 0) & (df_raw["latitude_traversed"] < 2.0)

df_raw = df_raw[coord_mask & fare_mask & dist_mask].reset_index(drop=True)



## === cell 18
outliers = []
for feature in ["longitutde_traversed", "latitude_traversed"]:
    Q1 = np.percentile(df_raw[feature], 25, axis=0)
    Q3 = np.percentile(df_raw[feature], 75, axis=0)
    step = 10 * (Q3 - Q1)
    feature_outlier = df_raw[
        ~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))
    ]
    outliers += feature_outlier.index.tolist()

len(outliers) / len(df_raw)



## === cell 19
df = df_raw.drop(df_raw.index[outliers]).reset_index(drop=True)



## === cell 20
len(df)



## === cell 21
y = np.log1p(df.fare_amount)
df.drop("fare_amount", axis=1, inplace=True)



## === cell 22
X_train, X_valid, y_train, y_valid = train_test_split(
    df, y, test_size=0.01, random_state=42
)




## === cell 23
def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    pred_tr_log = m.predict(X_train)
    pred_va_log = m.predict(X_valid)

    pred_tr = np.expm1(pred_tr_log)
    pred_va = np.expm1(pred_va_log)

    y_tr = np.expm1(y_train)
    y_va = np.expm1(y_valid)

    res = [
        rmse(pred_tr, y_tr),
        rmse(pred_va, y_va),
        m.score(X_train, y_train),
        m.score(X_valid, y_valid),
    ]
    print(res)




## === cell 24
set_rf_samples(10000)



## === cell 25
m = RandomForestRegressor(n_jobs=-1, random_state=42)
m.fit(X_train, y_train)
print_score(m)



## === cell 26
fi = rf_feat_importance(m, X_train)
fi[:10]




## === cell 27
def plot_fi(fi):
    return fi.plot("cols", "imp", "barh", figsize=(12, 8), legend=False)


plot_fi(fi)



## === cell 28
test_set = pd.read_csv(test_path)



## === cell 29
test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)



## === cell 30
add_datepart(test_set, "pickup_datetime", drop=True, time=True)
distance(test_set)



## === cell 31
for c in X_train.columns:
    if c not in test_set.columns:
        test_set[c] = 0
test_set = test_set[X_train.columns]

test_predictions_log = m.predict(test_set)
test_predictions = np.expm1(test_predictions_log)
test_predictions = np.maximum(test_predictions, 0.0)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
