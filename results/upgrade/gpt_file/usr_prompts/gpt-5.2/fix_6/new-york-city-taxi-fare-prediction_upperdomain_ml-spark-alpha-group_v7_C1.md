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

5.28637

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.66111) has done: 'I fix the immediate import/runtime errors by replacing the removed `Imputer` with `SimpleImputer`, ensuring `pandas/numpy` are imported before use, and correcting the data paths to the Kaggle-provided `/kaggle/input/...` structure with a safe fallback to `../input` when present. I also make the distance feature computation robust to division-by-zero so the model can fit/predict without producing NaNs/Infs. To keep the core logic unchanged, I preserve the same feature set (distance_travel, passenger_count, bias term) and the same `GradientBoostingRegressor` settings, only ensuring the imputer is actually applied to both train and test matrices. Finally, I guarantee a valid `submission.csv` with exactly `key,fare_amount` is written.'
- What this solution (achieved 6.47829) has done: 'To move RMSE down toward the 4.07608 target without changing the core model/feature set, I make the training split representative by shuffling before the 70/30 split (the current sequential split can be biased by time ordering). I also apply the exact same basic sanity filters used in training to the test feature generation (distance>0, passenger_count>0) and safely handle any remaining NaN/Inf distances so the imputer/model sees consistent inputs. Finally, I keep the same `GradientBoostingRegressor` configuration and feature columns, but ensure predictions are aligned to the original test row order and always produce a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.29348) has done: 'To reduce RMSE toward the 4.07608 target without changing the model or feature set, I keep the same `GradientBoostingRegressor` and the same three input columns, but make the train/test preprocessing consistent by applying the *same* outlier/validity filters to the training data that the distance feature expects (bounding lat/longs and removing extreme coordinate zeros) so the model learns from cleaner, more realistic trips. I also compute `distance_travel` on a copy to avoid chained-assignment issues and ensure missing/invalid values become NaN consistently for the imputer. Finally, I fix a subtle alignment risk in the submission creation by predicting in the same order as `tdf_sorted` (rather than indexing predictions by original DataFrame indices), while still writing `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 5.28674) has done: 'You’re currently worse than the target (RMSE 5.29348 vs 4.07608, lower is better), so the smallest safe move is to improve generalization without changing the model or feature set. The biggest low-risk issue is that the model is being trained on only the first 1,000,000 rows (time-ordered), which is less representative; we instead sample 1,000,000 rows uniformly across the full training file using `skiprows` so it stays fast and within memory. We keep the exact same filters, the exact same 3-feature matrix, and the same `GradientBoostingRegressor` config, but train on a more representative subset. This should reduce public RMSE while preserving your core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 5.28637) has done: 'Your current RMSE (5.28674) is still meaningfully worse than the target (4.07608, lower is better), so we should improve generalization with the smallest changes that don’t alter the core model/features. The biggest low-risk issue is train/serving skew: you filter training to `distance_travel < 30` and `fare_amount < 100`, but you do not apply the analogous `distance_travel < 30` filter to test, so the model is forced to extrapolate on longer trips, hurting RMSE. I keep the exact same feature set (distance_travel, passenger_count, bias) and the same `GradientBoostingRegressor`, but I (1) clamp test `distance_travel` to the same upper bound (set >30 to NaN so the existing imputer handles it) and (2) prevent huge passenger_count values from leaking into inference by setting unrealistic counts (>6) to NaN (again handled by the same imputer). These changes keep evaluation semantics intact, avoid changing the training loop/architecture, and should move the score down toward the target.'

# 9. Code solution

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
N_SAMPLE = 1_000_000
rng = np.random.RandomState(21)

keep = set(rng.choice(np.arange(1, N_TOTAL + 1), size=N_SAMPLE, replace=False))
skip = lambda i: (i != 0) and (i not in keep)

df = pd.read_csv(train_path, skiprows=skip)
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
tdf = pd.read_csv(test_path)
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
