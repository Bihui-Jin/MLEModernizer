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

6.17147

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.71754) has done: 'I fix the immediate runtime blocker by replacing the deprecated `sklearn.preprocessing.Imputer` with `sklearn.impute.SimpleImputer`, and ensure all required imports execute so later cells have `pd/np/df` defined. I also adjust file paths from `../input/...` to the provided Kaggle-style path `/kaggle/input/...` (with a safe fallback) so the CSVs can actually be read in your environment. To preserve the core modeling logic, I keep the same feature construction and `GradientBoostingRegressor` training, but make the missing-value imputation actually applied (it was fit but never used for train/test features). Finally, I guarantee a correctly formatted `submission.csv` with columns `key` and `fare_amount` is written.'
- What this solution (achieved 6.17147) has done: 'You’re currently far from the target (RMSE 5.72 vs 4.09; lower is better), so we should make small, metric-aligned improvements without changing the model family or overall approach. The biggest gap is likely from (1) training/serving skew caused by not applying the same outlier filtering logic to test features, and (2) a non-random split (first 70% of rows) that hurts the model fit and hyperparameter choice even though it doesn’t leak labels. I (a) apply the same basic validity filters to the test set (passenger_count and plausible coordinate ranges) and then clip test `distance_travel` to the training-supported range instead of leaving extreme values unhandled, and (b) shuffle before the 70/30 split to get a more representative training set while keeping the same training loop and model. These are minimal changes that typically reduce RMSE materially for this competition and should move you closer to the 4.09 target without altering your core feature design or estimator.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

INPUT_DIR_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.isdir(d):
        INPUT_DIR = d
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory among known candidates."
    )

print("Using INPUT_DIR:", INPUT_DIR)
print("Top-level INPUT_DIR listing:", os.listdir(INPUT_DIR)[:20])


def _resolve_file(*names):
    """Try resolving a file either directly under INPUT_DIR or at INPUT_DIR/<competition-subdir>/."""
    for name in names:
        p1 = os.path.join(INPUT_DIR, name)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", name)
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(f"Could not resolve any of: {names} under {INPUT_DIR}")


TRAIN_PATH = _resolve_file("train.csv")
TEST_PATH = _resolve_file("test.csv")
SAMPLE_SUB_PATH = _resolve_file("sample_submission.csv")



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
df.head()



## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]
df.head()



## === cell 3
alpha_ang = 0.506


def distance_travel(df_):
    df_["abs_diff_longitude"] = (
        df_.dropoff_longitude - df_.pickup_longitude
    ).abs() * 50
    df_["abs_diff_latitude"] = (df_.dropoff_latitude - df_.pickup_latitude).abs() * 69
    df_["displacement_vector"] = (
        df_.abs_diff_latitude**2 + df_.abs_diff_longitude**2
    ) ** 0.5  # crow flies

    denom = df_.abs_diff_latitude.replace(0, np.nan)
    angle = np.arctan(df_.abs_diff_longitude / denom)

    df_["actual_long"] = (df_.displacement_vector * np.sin(angle - alpha_ang)).abs()
    df_["actual_lat"] = (df_.displacement_vector * np.cos(angle - alpha_ang)).abs()
    df_["distance_travel"] = df_.actual_long + df_.actual_lat


distance_travel(df)
df = df[df.distance_travel > 0]
df.head()



## === cell 4
if plt is not None:
    test = df[df.passenger_count == 1]
    _ = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")



## === cell 5
df = df[df.distance_travel < 30]
df = df[df.fare_amount < 100]
if plt is not None:
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")



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
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X_imp = imp.fit_transform(train_X)
test_X_imp = imp.transform(test_X)

regr = GradientBoostingRegressor(n_estimators=100, random_state=42)
regr.fit(train_X_imp, train_y)

print("Holdout R^2:", regr.score(test_X_imp, test_y))



## === cell 8
tdf = pd.read_csv(TEST_PATH)

tdf = tdf[tdf.passenger_count > 0].copy()
for c in ["pickup_longitude", "dropoff_longitude"]:
    tdf = tdf[tdf[c].between(-75, -72)]
for c in ["pickup_latitude", "dropoff_latitude"]:
    tdf = tdf[tdf[c].between(40, 42)]

distance_travel(tdf)

tdf["distance_travel"] = tdf["distance_travel"].clip(lower=0, upper=30)

tdf.head()



## === cell 9
ttrain_X = np.column_stack(
    (tdf.distance_travel, tdf.passenger_count, np.ones(len(tdf)))
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.clip(output, 0, None)

print(output[:10], " ... ", "n_preds=", len(output))



## === cell 10
full_test = pd.read_csv(TEST_PATH)
if len(tdf) != len(full_test):
    distance_travel(full_test)
    full_test["distance_travel"] = full_test["distance_travel"].clip(lower=0, upper=30)
    full_X = np.column_stack(
        (full_test.distance_travel, full_test.passenger_count, np.ones(len(full_test)))
    )
    full_X = imp.transform(full_X)
    output_full = np.clip(regr.predict(full_X), 0, None)
    my_submission = pd.DataFrame({"key": full_test.key, "fare_amount": output_full})
else:
    my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})

my_submission.to_csv("submission.csv", index=False)
my_submission.head()
