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

No external packages required in the script and installed.

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

3.6375985209102257

# 6. Current score

18.16927

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 18.16927) has done: 'I make the script compatible with the Kaggle runtime by (1) removing the failing TensorFlow import/setup and replacing it with a lightweight, deterministic baseline that runs without external model files, (2) fixing the datetime feature extraction bug (`.dt.week` no longer exists) so `add_datepart` creates the expected columns, and (3) correcting the distance function’s haversine computation. Since your current run yields no valid submission, the priority is to produce a correctly formatted `submission.csv` (`key,fare_amount`) end-to-end; the baseline uses a sensible distance-based fare estimate plus a small base fare, which should score materially better than a constant predictor while remaining simple and stable within the 600s limit.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

print(os.listdir("../input"))



## === cell 1
dnn_path = "../input/dnn-model"
if os.path.exists(dnn_path):
    print("Found dnn-model:", os.listdir(dnn_path))
else:
    print("dnn-model path not found, proceeding without it:", dnn_path)



## === cell 2
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
df_test.head()



## === cell 4
sample_sub = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
print(sample_sub.head())
print("test rows:", len(df_test), "sample rows:", len(sample_sub))




## === cell 5
def add_datepart(df, fldname, drop=True):
    """
    Bugfix: pandas removed .dt.week; use .dt.isocalendar().week (or fallback to strftime).
    Also normalize feature names to match the original expected columns.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, errors="coerce"
        )

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month
    df[targ_pre + "Day"] = fld.dt.day
    df[targ_pre + "Dayofweek"] = fld.dt.dayofweek
    df[targ_pre + "Dayofyear"] = fld.dt.dayofyear
    df[targ_pre + "hour"] = fld.dt.hour

    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int16)
    except Exception:
        df[targ_pre + "Week"] = fld.dt.strftime("%V").astype("int16")

    df[targ_pre + "Is_month_end"] = fld.dt.is_month_end
    df[targ_pre + "Is_month_start"] = fld.dt.is_month_start
    df[targ_pre + "Is_quarter_end"] = fld.dt.is_quarter_end
    df[targ_pre + "Is_quarter_start"] = fld.dt.is_quarter_start
    df[targ_pre + "Is_year_end"] = fld.dt.is_year_end
    df[targ_pre + "Is_year_start"] = fld.dt.is_year_start

    df[targ_pre + "Elapsed"] = (fld.view("int64") // 10**9).astype("float64")
    df.loc[fld.isna(), targ_pre + "Elapsed"] = np.nan

    if drop:
        df.drop(fldname, axis=1, inplace=True)


def distance(data):
    """
    Bugfix: haversine formula should use arctan2(sqrt(a), sqrt(1-a)).
    Input data expected as Nx4 [lon1, lat1, lon2, lat2].
    Returns distance in km.
    """
    radius = 6371.0  # km
    lon1 = data[:, 0]
    lat1 = data[:, 1]
    lon2 = data[:, 2]
    lat2 = data[:, 3]

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)

    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1r) * np.cos(lat2r) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c




## === cell 6
class DNN_Model:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Original TensorFlow DNN model is not supported in this runtime (tf.contrib removed and no checkpoint provided)."
        )




## === cell 7
df_test["Herv_Dist"] = distance(
    df_test[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ]
    .astype("float64")
    .values
)
add_datepart(df_test, "pickup_datetime", drop=True)



## === cell 8
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetimeYear",
    "pickup_datetimeMonth",
    "pickup_datetimeWeek",
    "pickup_datetimeDay",
    "pickup_datetimeDayofweek",
    "pickup_datetimeDayofyear",
    "pickup_datetimehour",
    "pickup_datetimeElapsed",
    "Herv_Dist",
]
missing = [c for c in feature_cols if c not in df_test.columns]
print("Missing feature cols:", missing)
df_test[feature_cols].head()



## === cell 9
dist_km = (
    df_test["Herv_Dist"].astype("float64").fillna(0.0).clip(lower=0.0, upper=200.0)
)
passengers = (
    df_test["passenger_count"].astype("float64").fillna(1.0).clip(lower=1.0, upper=6.0)
)

fare_pred = 2.50 + 1.75 * dist_km + 0.10 * (passengers - 1.0)

fare_pred = fare_pred.clip(lower=2.50, upper=300.0)



## === cell 10
my_submission = pd.DataFrame(
    {
        "key": df_test["key"].astype(str).values,
        "fare_amount": fare_pred.astype("float64").values,
    }
)

if "key" in sample_sub.columns and len(sample_sub) == len(my_submission):
    my_submission = sample_sub[["key"]].merge(my_submission, on="key", how="left")

my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
