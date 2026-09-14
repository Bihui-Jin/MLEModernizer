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

4.08766

# 6. Current score

5.33438

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.55996) has done: 'Diagnosis: The crash happens because `SimpleImputer` in scikit-learn 1.2.2 no longer accepts the `axis` parameter (it was deprecated/removed from the older `Imputer` API). The notebook aliases `SimpleImputer` as `Imputer`, but still calls it with legacy arguments (`axis=0` and `missing_values='NaN'` as a string). To unblock execution without changing the model/training semantics, remove the unsupported `axis` argument and use `missing_values=np.nan` (the correct representation for NaNs in NumPy arrays). The rest of the cell (fit, model training, scoring) remains unchanged.

Patch summary: Update the `Imputer(...)` construction in cell 7 to be compatible with `sklearn.impute.SimpleImputer` by removing `axis` and using `missing_values=np.nan`.

Updated cells: cell 7 only.

Compatibility notes for cell k+1: Cell 8 does not reference `imp`/`regr`; however, this patch preserves `imp` and `regr` variables and keeps `train_X/test_X` usage unchanged, so downstream cells (if any) that rely on them remain compatible.

Assumptions: `train_X` contains numeric data and any missing values are represented as `np.nan` (or none), so using `missing_values=np.nan` is correct and consistent with `SimpleImputer`.'
- What this solution (achieved 5.77525) has done: 'Your current RMSE (5.55996) is worse than the target (4.08766), so we should improve generalization with minimal changes. The biggest issue is that you fit the `SimpleImputer` but never actually transform `train_X`/`test_X` before training/scoring, while you *do* transform the test features at prediction time—this train/test preprocessing mismatch hurts performance. I apply the imputer transform consistently to training and validation features (keeping the same model and features), and I also make the train/test split reproducibly shuffled to reduce distribution shift from slicing the first 70% of rows. Submission format/path stays the same and still writes `submission.csv`.'
- What this solution (achieved 5.38851) has done: 'Your current RMSE (5.77525) is still well above the target (4.08766), so we should improve with very small, low-risk changes that keep the same feature and model family. The main gains here typically come from (1) removing obviously bad coordinate rows (outside NYC bounds / zero coordinates) that inject noise into the `distance_travel` feature, and (2) using a slightly stronger but still identical-core `GradientBoostingRegressor` configuration (more trees + a bit smaller learning rate) to better fit the nonlinearity without changing the overall approach. I also apply the exact same coordinate filtering to test to avoid generating out-of-domain distances. Submission format, filename, and core logic (distance_travel feature + GBDT regressor) remain the same.'
- What this solution (achieved 5.33438) has done: 'To move RMSE down toward the 4.09 target with minimal risk, I keep your exact feature set (distance_travel + passenger_count + bias) and the same GradientBoostingRegressor approach, but fix a train/test mismatch: you filter NYC coordinates for test (dropping rows) but you *don’t* apply your own “distance<30” and “fare<100” style trimming logic to test, and you also never clip/guard predictions. I (1) compute distance on the full test set without dropping rows, using a boolean “valid_nyc” mask and imputing invalid distances, which prevents distribution shift from row-dropping and avoids relying on the merge+fill fallback, and (2) clip negative/too-large predictions to the training target range to reduce RMSE impact from a small number of outliers (this changes only post-processing, not model/training). These are small changes that typically reduce public LB RMSE without changing the core model or features.'

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
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat


def filter_nyc_coords(frame):
    frame = frame[
        (frame.pickup_longitude.between(-74.3, -73.6))
        & (frame.dropoff_longitude.between(-74.3, -73.6))
        & (frame.pickup_latitude.between(40.4, 41.0))
        & (frame.dropoff_latitude.between(40.4, 41.0))
    ]
    frame = frame[
        (frame.pickup_longitude != 0)
        & (frame.dropoff_longitude != 0)
        & (frame.pickup_latitude != 0)
        & (frame.dropoff_latitude != 0)
    ]
    return frame


def valid_nyc_mask(frame):
    m = (
        frame.pickup_longitude.between(-74.3, -73.6)
        & frame.dropoff_longitude.between(-74.3, -73.6)
        & frame.pickup_latitude.between(40.4, 41.0)
        & frame.dropoff_latitude.between(40.4, 41.0)
        & (frame.pickup_longitude != 0)
        & (frame.dropoff_longitude != 0)
        & (frame.pickup_latitude != 0)
        & (frame.dropoff_latitude != 0)
    )
    return m


df = filter_nyc_coords(df)

distance_travel(df)
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
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

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
from sklearn.ensemble import GradientBoostingRegressor

imp = Imputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(train_X)
train_X_imp = imp.transform(train_X)
test_X_imp = imp.transform(test_X)

regr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
regr.fit(train_X_imp, train_y)
regr.score(test_X_imp, test_y)



## === cell 8
tdf = pd.read_csv("../input/test.csv", nrows=10_00_000)

valid_mask = valid_nyc_mask(tdf)

tdf_work = tdf.copy()
tdf_work.loc[~valid_mask, "pickup_longitude"] = np.nan
tdf_work.loc[~valid_mask, "pickup_latitude"] = np.nan
tdf_work.loc[~valid_mask, "dropoff_longitude"] = np.nan
tdf_work.loc[~valid_mask, "dropoff_latitude"] = np.nan

distance_travel(tdf_work)

tdf_work.loc[~valid_mask, "distance_travel"] = np.nan

tdf_work.head()



## === cell 9
ttrain_X = np.column_stack(
    (tdf_work.distance_travel, tdf_work.passenger_count, np.ones(len(tdf_work)))
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

lo = float(np.percentile(train_y, 0.5))
hi = float(np.percentile(train_y, 99.5))
output = np.clip(output, lo, hi)

print(output)



## === cell 10
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})

if len(my_submission) != 9914:
    all_test = pd.read_csv("../input/test.csv", usecols=["key"])
    my_submission = all_test.merge(my_submission, on="key", how="left")
    my_submission["fare_amount"] = my_submission["fare_amount"].fillna(
        np.median(train_y)
    )

my_submission.to_csv("submission.csv", index=False)
my_submission.head()
