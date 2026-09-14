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
scipy==1.15.3
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata  # noqa: F401

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def _resolve_input_path(filename: str) -> str:
    """
    Kaggle typically mounts datasets under /kaggle/input/<dataset-slug>/.
    This notebook's original code used ../input/, so we support both.
    """
    candidates = [
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/input/{filename}",
        f"../input/{filename}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/data/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


print("Listing /kaggle/input (if present):")
if os.path.exists("/kaggle/input"):
    print(os.listdir("/kaggle/input")[:50])
else:
    print("No /kaggle/input directory; will rely on fallback paths if available.")




## === cell 1
train_path = _resolve_input_path("train.csv")

N_TOTAL = 55_423_856  # provided dataset size

N_SAMPLE = 3_000_000

rng = np.random.RandomState(21)

keep = set(rng.choice(np.arange(1, N_TOTAL + 1), size=N_SAMPLE, replace=False))
skip = lambda i: (i != 0) and (i not in keep)

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

df = pd.read_csv(train_path, skiprows=skip, usecols=train_usecols)
print("Loaded sampled train rows:", df.shape)
df.head()




## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]

df = df[
    df.pickup_longitude.between(-75, -72)
    & df.dropoff_longitude.between(-75, -72)
    & df.pickup_latitude.between(40, 42)
    & df.dropoff_latitude.between(40, 42)
]

df.head()




## === cell 3
alpha_ang = 0.506


def distance_travel(df_):
    df_.loc[:, "abs_diff_longitude"] = (
        df_.dropoff_longitude - df_.pickup_longitude
    ).abs() * 50
    df_.loc[:, "abs_diff_latitude"] = (
        df_.dropoff_latitude - df_.pickup_latitude
    ).abs() * 69
    df_.loc[:, "displacement_vector"] = (
        df_.abs_diff_latitude**2 + df_.abs_diff_longitude**2
    ) ** 0.5

    denom = df_["abs_diff_latitude"].replace(0, np.nan)
    angle = np.arctan(df_["abs_diff_longitude"] / denom)

    df_.loc[:, "actual_long"] = (
        df_["displacement_vector"] * np.sin(angle - alpha_ang)
    ).abs()
    df_.loc[:, "actual_lat"] = (
        df_["displacement_vector"] * np.cos(angle - alpha_ang)
    ).abs()
    df_.loc[:, "distance_travel"] = df_["actual_long"] + df_["actual_lat"]

    df_.loc[:, "distance_travel"] = df_["distance_travel"].replace(
        [np.inf, -np.inf], np.nan
    )


distance_travel(df)
df = df[df.distance_travel > 0]
df.head()




## === cell 4
test = df[df.passenger_count == 1]
_ = test.iloc[:20000].plot.scatter("distance_travel", "fare_amount")




## === cell 5
df = df[df.distance_travel < 30]
df = df[df.fare_amount < 100]
_ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
df = df.sample(frac=1.0, random_state=21).reset_index(drop=True)

l = len(df)
print(l)
df_train = df[: int(0.7 * l)]
df_test = df[int(0.7 * l) :]

train_X = np.column_stack(
    (df_train.distance_travel, df_train.passenger_count, np.ones(len(df_train)))
)
test_X = np.column_stack(
    (df_test.distance_travel, df_test.passenger_count, np.ones(len(df_test)))
)
train_y = np.array(df_train.fare_amount)
test_y = np.array(df_test.fare_amount)




## === cell 7
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X = imp.fit_transform(train_X)
test_X = imp.transform(test_X)

regr = GradientBoostingRegressor(random_state=21, n_estimators=400)
regr.fit(train_X, train_y)

pred_val = regr.predict(test_X)
rmse = mean_squared_error(test_y, pred_val, squared=False)
print("Holdout RMSE:", rmse)




## === cell 8
test_path = _resolve_input_path("test.csv")

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
tdf = pd.read_csv(test_path, usecols=test_usecols)

tdf["_row_id"] = np.arange(len(tdf))  # preserve original order for submission alignment

coord_ok = (
    tdf.pickup_longitude.between(-75, -72)
    & tdf.dropoff_longitude.between(-75, -72)
    & tdf.pickup_latitude.between(40, 42)
    & tdf.dropoff_latitude.between(40, 42)
)
for c in [
    "pickup_longitude",
    "dropoff_longitude",
    "pickup_latitude",
    "dropoff_latitude",
]:
    tdf.loc[~coord_ok, c] = np.nan

distance_travel(tdf)

tdf.loc[tdf["passenger_count"] <= 0, "passenger_count"] = np.nan

tdf.loc[tdf["distance_travel"] <= 0, "distance_travel"] = np.nan
tdf.loc[tdf["distance_travel"] >= 30, "distance_travel"] = np.nan

tdf.loc[tdf["passenger_count"] > 6, "passenger_count"] = np.nan

tdf.head()




## === cell 9
tdf_sorted = tdf.sort_values("_row_id").reset_index(drop=True)

ttrain_X = np.column_stack(
    (tdf_sorted.distance_travel, tdf_sorted.passenger_count, np.ones(len(tdf_sorted)))
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.clip(output, 0.0, None)

print(output[:10])




## === cell 10
my_submission = pd.DataFrame({"key": tdf_sorted.key.values, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
