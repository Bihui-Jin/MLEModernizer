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

# 5. Target score

4.07608

# 6. Current score

6.05799

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.44092) has done: 'Diagnosis: Cell 7 crashes because `sklearn.impute.SimpleImputer` (scikit-learn 1.2.2) no longer accepts the `axis` argument, and `missing_values='NaN'` is also incorrect for representing actual NaN values. The rest of the cell doesn’t actually use the imputer output, but removing the invalid parameters is the minimal change to unblock execution while preserving the same model training logic.  
Patch summary: Update the `SimpleImputer` initialization to remove the unsupported `axis` parameter and use `missing_values=np.nan` to correctly denote NaNs. Keep all other logic (fit, model, training, scoring) unchanged.  
Updated cells: Only cell 7 is modified.  
Compatibility notes for cell k+1: All variables created in cell 7 (`imp`, `regr`) remain defined as before; downstream cells are unaffected.  
Assumptions: The intent was to impute NaN values column-wise (default behavior), and training data is numeric so `np.nan` is the correct missing marker.'
- What this solution (achieved 7.25567) has done: 'Your current score is worse than the target (RMSE 5.44092 vs 4.07608), so we should improve predictive accuracy with the smallest changes that don’t alter the core model/training approach. The biggest issue is that the imputer is fit but never actually applied to `train_X`/`test_X`, while it *is* applied to the test features at inference—this train/serve mismatch can hurt score; we apply the same transform to train and validation features. Second, `distance_travel()` can create NaNs/Infs when latitude differences are zero; we make the angle computation numerically safe to reduce bad rows/features without changing the feature definition. Finally, we ensure the submission uses finite, non-negative fares (a legitimate post-processing consistent with the problem) to avoid extreme errors.'
- What this solution (achieved 6.05799) has done: 'We keep your exact feature set and model (GradientBoostingRegressor on distance_travel + passenger_count + bias) but fix two small issues that hurt RMSE: (1) the current train/validation split is order-based (first 70% vs last 30%), which can create distribution shift; we switch to a deterministic random split while keeping the same 70/30 ratio. (2) `distance_travel` is computed on the raw data without removing invalid/NaN coordinates, so extreme/NaN distances can slip into training and test; we add minimal coordinate sanity filtering (NYC bounding box + finite checks) and apply the same cleaning to both train and test before feature building. These are minimal, metric-aligned fixes that typically move RMSE down toward your 4.07608 target without changing your core approach.'

# 9. Code solution

## === cell 0
from sklearn.impute import SimpleImputer as Imputer
import numpy as np  # linear algebra
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import math
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=10_00_000)
df.head()



## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]
df.head()



## === cell 3
alpha_ang = 0.506


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5  # as the crow flies

    angle = np.arctan2(
        df["abs_diff_longitude"].to_numpy(), df["abs_diff_latitude"].to_numpy()
    )
    angle = angle - alpha_ang

    dv = df["displacement_vector"].to_numpy()
    df["actual_long"] = np.abs(dv * np.sin(angle))
    df["actual_lat"] = np.abs(dv * np.cos(angle))
    df["distance_travel"] = df["actual_long"] + df["actual_lat"]


def clean_coords(df):
    cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in cols:
        df = df[np.isfinite(df[c].to_numpy())]

    df = df[
        (df.pickup_longitude.between(-74.3, -73.7))
        & (df.dropoff_longitude.between(-74.3, -73.7))
        & (df.pickup_latitude.between(40.5, 41.0))
        & (df.dropoff_latitude.between(40.5, 41.0))
    ]
    return df


df = clean_coords(df)
distance_travel(df)
df = df[np.isfinite(df.distance_travel.to_numpy())]
df = df[df.distance_travel > 0]
df.head()



## === cell 4
test = df[df.passenger_count == 1]
plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")



## === cell 5
df = df[df.distance_travel < 30]
df = df[df.fare_amount < 100]
plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")



## === cell 6
l = len(df)
print(l)

rng = np.random.RandomState(21)
idx = rng.permutation(l)
cut = int(0.7 * l)
train_idx = idx[:cut]
test_idx = idx[cut:]

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

train_X = np.column_stack(
    (df_train.distance_travel, df_train.passenger_count, np.ones(len(df_train)))
)
test_X = np.column_stack(
    (df_test.distance_travel, df_test.passenger_count, np.ones(len(df_test)))
)
train_y = np.array(df_train.fare_amount)
test_y = np.array(df_test.fare_amount)



## === cell 7
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

imp = Imputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(train_X)
train_X = imp.transform(train_X)
test_X = imp.transform(test_X)

regr = GradientBoostingRegressor(random_state=21, n_estimators=400)
regr.fit(train_X, train_y)
regr.score(test_X, test_y)



## === cell 8
tdf = pd.read_csv("../input/test.csv", nrows=10_00_000)

tdf = clean_coords(tdf)
distance_travel(tdf)
tdf = tdf[np.isfinite(tdf.distance_travel.to_numpy())]
tdf.head()



## === cell 9
ttrain_X = np.column_stack(
    (tdf.distance_travel, tdf.passenger_count, np.ones(len(tdf)))
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.where(np.isfinite(output), output, np.nan)
if np.isnan(output).any():
    output = np.nan_to_num(output, nan=float(np.nanmean(output)))
output = np.maximum(output, 0.0)

print(output)



## === cell 10
sample = pd.read_csv("../input/sample_submission.csv")
pred_df = pd.DataFrame({"key": tdf.key, "fare_amount": output})

my_submission = sample[["key"]].merge(pred_df, on="key", how="left")

if my_submission["fare_amount"].isna().any():
    my_submission["fare_amount"] = my_submission["fare_amount"].fillna(
        float(np.nanmean(output))
    )

my_submission.to_csv("submission.csv", index=False)
my_submission.head()
