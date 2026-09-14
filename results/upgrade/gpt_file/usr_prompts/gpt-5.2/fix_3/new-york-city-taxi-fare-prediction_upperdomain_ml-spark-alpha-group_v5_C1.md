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

5.58616

# 6. Current score

996.7925

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 996.52313) has done: 'I fix the immediate runtime errors by replacing the deprecated `Imputer` with `SimpleImputer`, restoring missing imports by ensuring the first cell runs, and updating file paths to the provided Kaggle `/kaggle/input/...` structure so CSVs can be read. I also make the custom `distance_travel()` feature function numerically safe (avoid divide-by-zero) without changing its intent, and ensure the same preprocessing is applied to both train and test. Finally, I write a valid `submission.csv` with exactly the required columns (`key`, `fare_amount`) and keep the model logic (simple linear regression on engineered distance + passenger count) intact.'
- What this solution (achieved 996.7925) has done: 'Your score is extremely worse than the target (RMSE 996 vs ~5.6), which almost always means the submission rows are misaligned with the `key` order in `test.csv` or you have NaNs/inf in engineered features causing wild predictions. I keep your exact core model (linear regression on `distance_travel`, `passenger_count`, and a constant) but make the feature engineering robust and non-leaky by having `distance_travel()` return a clean series without leaving behind intermediate columns or SettingWithCopy issues. I also enforce strict row-order alignment by building the submission from `sample_submission.csv` (keys in the expected order) and merging predictions by `key`. Finally, I apply the same basic sanity filters to the training data for invalid lat/long ranges to prevent the regression from learning from garbage coordinates, which is a minimal change that typically collapses RMSE dramatically toward the expected band.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn import linear_model

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata  # noqa: F401

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "../input"

print("INPUT_DIR =", INPUT_DIR)
print("Files:", os.listdir(INPUT_DIR)[:20])



## === cell 1
df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"), nrows=1_000_000)
df.head()



## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]

df = df[
    df.pickup_longitude.between(-75, -72)
    & df.dropoff_longitude.between(-75, -72)
    & df.pickup_latitude.between(40, 42)
    & df.dropoff_latitude.between(40, 42)
].copy()

df.head()



## === cell 3
alpha_ang = 0.506


def distance_travel(df_):
    dlon = df_["dropoff_longitude"].to_numpy() - df_["pickup_longitude"].to_numpy()
    dlat = df_["dropoff_latitude"].to_numpy() - df_["pickup_latitude"].to_numpy()

    abs_diff_longitude = np.abs(dlon) * 50.0
    abs_diff_latitude = np.abs(dlat) * 69.0

    displacement_vector = np.sqrt(abs_diff_latitude**2 + abs_diff_longitude**2)

    denom = np.where(abs_diff_latitude == 0, np.finfo(float).eps, abs_diff_latitude)
    angle = np.arctan(abs_diff_longitude / denom)

    actual_long = np.abs(displacement_vector * np.sin(angle - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(angle - alpha_ang))
    dist = actual_long + actual_lat

    dist = np.where(np.isfinite(dist), dist, np.nan)
    return pd.Series(dist, index=df_.index, name="distance_travel")


df["distance_travel"] = distance_travel(df)
df = df[df.distance_travel > 0].copy()
df.head()



## === cell 4
test = df[df.passenger_count == 1]
_ = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")



## === cell 5
df = df[df.distance_travel < 30].copy()
df = df[df.fare_amount < 100].copy()
_ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")



## === cell 6
l = len(df)
print("Filtered train rows:", l)

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

print("train_X shape:", train_X.shape, "train_y shape:", train_y.shape)



## === cell 7
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(train_X)

train_X_imp = imp.transform(train_X)
test_X_imp = imp.transform(test_X)

regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
regr.fit(train_X_imp, train_y)

print("Coefficients:", regr.coef_)
print("Validation R^2:", regr.score(test_X_imp, test_y))



## === cell 8
tdf = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
tdf["distance_travel"] = distance_travel(tdf)
tdf.head()



## === cell 9
ttrain_X = np.column_stack(
    (tdf.distance_travel, tdf.passenger_count, np.ones(len(tdf)))
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.clip(output, 0, None)

print(output[:10])



## === cell 10
sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
pred_df = pd.DataFrame({"key": tdf["key"].values, "fare_amount": output})

sub = sub.drop(columns=["fare_amount"], errors="ignore").merge(
    pred_df, on="key", how="left"
)

sub["fare_amount"] = sub["fare_amount"].fillna(float(np.nanmean(output)))

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()
