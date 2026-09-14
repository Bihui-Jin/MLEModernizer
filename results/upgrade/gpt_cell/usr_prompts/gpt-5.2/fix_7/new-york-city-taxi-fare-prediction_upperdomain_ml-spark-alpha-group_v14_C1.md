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

4.20022

# 6. Current score

5.5341

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.37892) has done: 'Diagnosis: The crash happens in cell 9 when fitting `SimpleImputer`: `missing_values="NaN"` is a string, but `train_X` is a numeric `float64` array, so scikit-learn raises a dtype mismatch error. In modern scikit-learn, missing values in numeric arrays should be specified as `np.nan` (or another numeric sentinel), not the string `"NaN"`. The rest of the pipeline doesn’t actually use `imp.transform(...)`, so the only requirement is to make `.fit(train_X)` succeed without changing training semantics.

Patch summary: Change the `Imputer` initialization to use `missing_values=np.nan` instead of `"NaN"` so the imputer accepts numeric inputs and the loop proceeds unchanged. No other logic (data preprocessing, feature construction, model training/scoring) is modified.

Updated cells: cell 9 only.

Compatibility notes for cell k+1: Cell 10 expects only that cell 9 completes; no variables from cell 9 are required by cell 10. This patch keeps all existing variables (`gen`, `regr`, `imp`, etc.) and behavior intact.

Assumptions: Numeric missing values (if any) are represented as `np.nan` in `train_X`/`test_X` (or there are none), consistent with pandas/numpy defaults.'
- What this solution (achieved 5.41292) has done: 'To move RMSE down toward your 4.20022 target (current 5.37892; lower is better), the smallest meaningful improvement is to make the train/test split consistent across the whole chunk stream instead of re-splitting inside each chunk (which injects noise and hurts fit). I keep your exact feature engineering and the same `GradientBoostingRegressor` setup, but switch to training on all cleaned rows in each chunk (still using your chunk loop). I also make the imputer actually apply to `train_X` (you already apply it to `test_X`), so train/test preprocessing matches and avoids any NaN-related inconsistency. Finally, I cap extreme/invalid predictions at a small positive minimum to avoid RMSE blow-ups from negatives without changing the core model.'
- What this solution (achieved 5.39632) has done: 'To move your RMSE down toward the 4.20022 target (current 5.41292; lower is better) with minimal logic changes, I keep your exact feature set and `GradientBoostingRegressor`, but fix the “fake incremental” training: `warm_start=True` only continues training if `n_estimators` increases, so your loop was effectively re-fitting the same 100-tree model each chunk and discarding earlier chunks. I make the loop actually add trees each chunk (small increase per chunk) so it accumulates signal across many chunks without changing the model family or training approach. I also ensure the test set receives the same `distance_travel` computation deterministically and keep your prediction clipping to avoid negative fares.'
- What this solution (achieved 5.5341) has done: 'Your current RMSE (5.39632) is worse than the 4.20022 target (lower is better), so we make the smallest changes that legitimately improve generalization without changing your core model or features. The biggest issue is that you “fit” the imputer on every chunk and only keep the last chunk’s statistics, then apply that to the whole test set; this mismatch can hurt predictions. I fit the imputer once on the first cleaned chunk and keep it fixed for all subsequent chunks (training still uses the same features and the same warm-start tree-growing loop). I also make `distance_travel()` numerically safer by avoiding division-by-zero in the arctan term, which prevents rare infinities/NaNs from leaking into the model and degrading RMSE.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from sklearn.impute import SimpleImputer as Imputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))




## === cell 1
def chunck_generator(filename, header=False, chunk_size=10**6):
    for chunk in pd.read_csv(
        filename, delimiter=",", iterator=True, chunksize=chunk_size, parse_dates=[1]
    ):
        yield (chunk)




## === cell 2
alpha_ang = 0.506


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5  ### as the crow flies

    denom = df.abs_diff_latitude.replace(0, np.nan)
    angle = np.arctan(df.abs_diff_longitude / denom)
    angle = angle.fillna(0.0)

    df["actual_long"] = (df.displacement_vector * np.sin(angle - alpha_ang)).abs()
    df["actual_lat"] = (df.displacement_vector * np.cos(angle - alpha_ang)).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat
    return df




## === cell 3
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    distance_travel(df)
    df = df[df.distance_travel > 0]
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 60]
    return df




## === cell 5
def graph_present(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def data_preprocessing(df):
    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)
    return df




## === cell 7
df = pd.read_csv("../input/train.csv", nrows=10_00_000)
df = df = distance_travel(df)
df = df[df.passenger_count == 1]
df = df[df.distance_travel < 30]
df.distance_travel.hist(bins=50, figsize=(12, 4))
plt.xlabel("distance miles")
plt.title("Histogram ride distances in miles")
df.distance_travel.describe()



## === cell 8
df = pd.read_csv("../input/train.csv", nrows=1_00_000)
df = df = distance_travel(df)
df = df[df.passenger_count <= 6]
df = df[df.distance_travel < 30]
df = df[df.fare_amount > 0]
df = df[df.fare_amount < 60]
df.fare_amount.hist(bins=50, figsize=(12, 4))
plt.xlabel("fare_amount")
df.fare_amount.describe()



## === cell 9
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=50, warm_start=True, random_state=42)

imp = Imputer(missing_values=np.nan, strategy="mean")

t = 56
trees_per_chunk = 5

imputer_fitted = False

while t > 0:
    print(t)
    df = next(gen)
    df = data_preprocessing(df)

    train_X = np.column_stack(
        (df.distance_travel, df.passenger_count, np.ones(len(df)))
    )
    train_y = np.array(df.fare_amount)

    if not imputer_fitted:
        imp = imp.fit(train_X)
        imputer_fitted = True

    train_X = imp.transform(train_X)

    regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)
    regr = regr.fit(train_X, train_y)

    try:
        print(regr.score(train_X, train_y))
    except Exception as e:
        print("score_failed:", e)

    t -= 1



## === cell 10
test_df = pd.read_csv("../input/test.csv", nrows=10_00_000)
distance_travel(test_df)
test_df.head()



## === cell 11
test_X = np.column_stack(
    (test_df.distance_travel, test_df.passenger_count, np.ones(len(test_df)))
)
test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)

predicted_fare = np.clip(predicted_fare, 0.0, None)

print(predicted_fare)



## === cell 12
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
