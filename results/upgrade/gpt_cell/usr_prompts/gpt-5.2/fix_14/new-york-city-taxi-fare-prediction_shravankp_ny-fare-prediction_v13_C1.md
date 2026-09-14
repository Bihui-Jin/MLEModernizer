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

eli5==0.13.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

4.57128

# 6. Current score

6.66769

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.87076) has done: 'Diagnosis: The crash in cell 10 occurs because `pd.to_datetime(...)` is producing timezone-aware datetimes (likely due to strings already including a timezone/UTC offset), and calling `.dt.tz_localize('UTC')` on an already tz-aware series raises `TypeError: Already tz-aware, use tz_convert to convert.`. This is a pandas behavior change/strictness that shows up with mixed/aware datetime parsing. The fix is to parse datetimes in a way that guarantees UTC consistently without double-localizing. We keep the same downstream feature extraction semantics (UTC-based hour/weekday/day/year) by using `utc=True` in `pd.to_datetime`, which yields tz-aware UTC datetimes deterministically.

Patch summary: In cell 10, replace the `.dt.tz_localize('UTC')` calls with `pd.to_datetime(..., utc=True)` for both train and test. Leave all subsequent feature extraction lines unchanged to preserve the same columns and types expected later.

Updated cells:'
- What this solution (achieved 6.26036) has done: 'Diagnosis: The crash happens inside `eli5` when it tries to use `protobuf` APIs; the installed `protobuf` version is incompatible with `eli5==0.13.0`, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known breakage from newer `protobuf` releases where ELI5 expects an older API. Since we cannot change the environment packages here, we need a runtime workaround. The safest minimal fix is to force `protobuf` to use the pure-Python implementation, which restores compatibility for ELI5’s import/runtime in many environments.

Patch summary: In cell 44 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) before importing `eli5`, then proceed with the same permutation importance computation and display call unchanged.

Updated cells: Only cell 44 is modified below.

Compatibility notes for cell k+1: Cell 45 depends only on `pred_test` and `test.timestamp_with_key`, which are created earlier; this patch does not alter them or any upstream variables. The ELI5 output remains a display-side effect only, so downstream submission generation is unaffected.

Assumptions: The environment allows setting `os.environ` before importing `eli5` within the notebook session, and the pure-Python protobuf fallback is available (standard with protobuf installs).'
- What this solution (achieved 6.70213) has done: 'Diagnosis: The crash happens when importing/using `eli5` because it depends on an older `protobuf` API; with the currently installed `protobuf` version, `google.protobuf.message_factory.MessageFactory.GetPrototype` no longer exists, leading to `AttributeError`. The `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variables are not sufficient to restore the removed attribute. Since this cell only provides optional model explainability and does not feed any variables into later cells, the minimal safe fix is to guard the `eli5`/`PermutationImportance` import and execution so the notebook can continue even when `protobuf` is incompatible.

Patch summary: Wrap the `eli5` import and permutation-importance computation in a `try/except` block catching the protobuf-related `AttributeError` (and `ImportError`), and print a short message when it cannot run. Keep the existing environment-variable lines and the core logic unchanged otherwise.

Updated cells: cell 44 only.

Compatibility notes for cell k+1: Cell 45 only uses `test.timestamp_with_key` and `pred_test` from earlier cells; cell 44 does not define or modify these. Skipping/guarding `eli5` does not affect the submission generation.

Assumptions: `eli5` explainability output is not required for the rest of the pipeline and can be safely skipped when incompatible with the installed `protobuf`.'
- What this solution (achieved 6.2816) has done: 'The crash is triggered inside the `eli5` import path: `eli5==0.13.0` is incompatible with the protobuf version available in this environment, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import/init. Since cell 44 is only for optional permutation importance visualization and not required for model training or submission generation, the minimal deterministic fix is to broaden the exception handling so the notebook continues even when protobuf/eli5 internals raise unexpected exceptions. This preserves all existing modeling logic and outputs, and keeps `pred_test` intact for cell 45.'
- What this solution (achieved 6.47661) has done: 'Diagnosis: Cell 44 crashes while importing/using `eli5` because `eli5` depends on an older protobuf API (`MessageFactory.GetPrototype`) that is missing in the protobuf version available in this environment. The current try/except does not prevent the crash because the `AttributeError` is being raised during module import/initialization in a way that still aborts the cell. The minimal fix is to avoid importing/using `eli5` entirely in this environment and safely skip permutation importance. This keeps the rest of the notebook (including submission generation in cell 45) unchanged.

Patch summary: Modify only cell 44 to remove the `eli5` import/use and replace it with a deterministic, explicit skip message. Keep the same variables produced earlier (`xgb_train`, `val_x`, `val_y`) untouched so downstream cells remain compatible.

Updated cells: Only cell 44 is changed.

Compatibility notes for cell k+1: Cell 45 only uses `test.timestamp_with_key` and `pred_test`, which are created before cell 44; skipping `eli5` does not affect them.

Assumptions: `eli5` permutation importance is non-essential for producing predictions/submission; the environment’s protobuf version is incompatible with `eli5==0.13.0` and cannot be changed here.'
- What this solution (achieved 6.65951) has done: 'Your current RMSE (6.47661) is worse than the target (4.57128), so we need a small, legitimate improvement without changing the overall approach. The biggest low-risk gain here is to stop rounding predictions to cents before scoring/submitting; that rounding injects avoidable error and typically worsens RMSE. I keep the exact same features, model, and training flow, and only remove the `.round(decimals=2)` calls (optionally rounding only for display, not for evaluation/submission). This should move RMSE down toward the target while keeping runtime and core logic unchanged.'
- What this solution (achieved 9.97596) has done: 'Your current RMSE (6.65951) is worse than the target (4.57128), so we should make a small, legitimate improvement without changing the overall approach. The biggest low-risk issue is that you overwrite `key` with an integer extracted from the string; that destroys the intended grouping/ID semantics and injects noise into both feature engineering and (indirectly) model fit. I keep the same feature set and XGBoost workflow, but preserve the original key string in a separate column and use a stable derived integer id (`key_id`) for grouping/feature selection, while keeping the submission `key` exactly as in `test.csv`. This should reduce RMSE meaningfully while keeping runtime and the rest of the logic essentially unchanged and still producing `submission.csv`.'
- What this solution (achieved 7.08836) has done: 'Your RMSE (9.97596) is much worse than the target (4.57128), so we make the smallest changes that legitimately reduce error without changing the overall feature engineering or XGBoost approach. The main issue is that you normalize geographic coordinates/distances with `Normalizer`, which destroys scale information that the fare model needs; we switch to `StandardScaler` (fit on train, apply to val/test) while keeping the same columns. We also set a fixed `random_state` for the shuffle and XGBoost for stability, and prevent obviously invalid negative fare predictions by clipping to 0 before writing the submission (a harmless post-process for an inherently nonnegative target). These changes should move RMSE substantially down toward the target while preserving the same pipeline structure and producing a valid `submission.csv`.'
- What this solution (achieved 6.73421) has done: 'Your current RMSE (7.08836) is still well above the target (4.57128), so we need a small change that improves generalization without changing your overall feature/model approach. The most impactful minimal fix is to prevent extreme outlier targets from dominating the XGBoost fit by adding a standard NYC Taxi Fare cleaning step that drops unrealistically high fares (this keeps the same features, same model, same training loop). This typically reduces RMSE substantially on this competition with minimal risk and runtime impact. I also make the `pd.cut` binning for test robust by explicitly filling NaNs for the dropoff bins (same semantics as your pickup bins), which avoids silent missing-value issues that can degrade predictions.'
- What this solution (achieved 6.93211) has done: 'Your RMSE (6.73421) is worse than the target (4.57128), so we make the smallest legitimate improvements that usually reduce error on this competition without changing your model/feature set. The biggest low-risk issue is that you compute “cwd_factor” rankings separately on the test set, which makes those engineered features inconsistent with training and can hurt generalization; we compute the factor tables on train only and apply the same mapping to both train and test (filling unseen bins with the train median factor). We also clip extreme outlier distances (very large Haversine distances) that slip past the current filters, because they inject noise into the model fit and worsen RMSE. Everything else (features, XGBRegressor usage, train/val split, scaling, submission format) stays the same.'
- What this solution (achieved 6.71948) has done: 'Your RMSE (6.93211) is still well above the target (4.57128), so we should make a small, legitimate improvement without changing your overall XGBoost approach or feature set. The biggest low-risk lever here is to tune the existing `XGBRegressor` hyperparameters away from defaults: defaults are usually underpowered for this competition and can leave substantial error on the table. I keep the same training flow (single fit on the same engineered features) and simply set conservative, commonly effective parameters (more trees, smaller learning rate, reasonable depth/subsampling) while preserving determinism via `random_state`. I also clip negative predictions (already present) and additionally cap extremely large fares to reduce the impact of occasional outlier predictions on RMSE.'
- What this solution (achieved 6.64648) has done: 'Your current RMSE (6.71948) is worse than the target (4.57128), so we should make a small, safe improvement that preserves your pipeline and model type. The most impactful minimal fix here is to train the same XGBoost regressor with an evaluation set and early stopping, then use the best iteration for predictions; this typically improves generalization materially without changing features or loss/metric semantics. I keep the same engineered features and the same overall training flow, only adding `eval_set`, `eval_metric="rmse"`, and `early_stopping_rounds`, and then use the resulting best model for val/test predictions. The submission format and paths remain unchanged, and we still clip predictions to the same [0, 250] bounds.'
- What this solution (achieved 6.66769) has done: 'To move RMSE down toward your target with minimal disruption, I keep the exact same feature engineering and XGBoost setup, but (1) scale the target by training on `log1p(fare_amount)` and invert with `expm1` at prediction time, which commonly improves RMSE for this competition without changing the model type or features. I also (2) compute validation/test predictions using `best_iteration` from early stopping to ensure we actually use the best found tree count, and (3) clip `fare_amount` only after inverse-transform to keep the log-space training consistent. These are small, localized changes around the training target/prediction post-processing and should improve generalization while preserving the pipeline’s core logic and producing the same submission format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("/kaggle/input"))

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")
train.head()



## === cell 2
train.isnull().sum()



## === cell 3
train = train.dropna(how="any", axis=0)



## === cell 4
train["abs_diff_longitude"] = np.abs(
    train["dropoff_longitude"] - train["pickup_longitude"]
)
train["abs_diff_latitude"] = np.abs(
    train["dropoff_latitude"] - train["pickup_latitude"]
)
test["abs_diff_longitude"] = np.abs(
    test["dropoff_longitude"] - test["pickup_longitude"]
)
test["abs_diff_latitude"] = np.abs(test["dropoff_latitude"] - test["pickup_latitude"])



## === cell 5
train = train.loc[train["fare_amount"] > 0, :]
train = train.loc[(train["passenger_count"] <= 6) & (train["passenger_count"] > 0), :]
train = train.loc[
    (train["abs_diff_latitude"] < 2) & (train["abs_diff_longitude"] < 2), :
]
train = train.loc[
    (train["abs_diff_latitude"] > 0) & (train["abs_diff_longitude"] > 0), :
]

train = train.loc[train["fare_amount"] <= 250, :]



## === cell 6
train.loc[:, "timestamp_with_key"] = train.loc[:, "key"]
test.loc[:, "timestamp_with_key"] = test.loc[:, "key"]

train.loc[:, "key_id"] = train["key"].str.split(".").str[1].astype("int")
test.loc[:, "key_id"] = test["key"].str.split(".").str[1].astype("int")



## === cell 7
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True, utc=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True, utc=True
)

train.loc[:, "hour_no"] = train["pickup_datetime"][:].dt.strftime("%-H").astype("int")
test.loc[:, "hour_no"] = test["pickup_datetime"][:].dt.strftime("%-H").astype("int")
train.loc[:, "weekday_no"] = train["pickup_datetime"][:].dt.strftime("%w").astype("int")
test.loc[:, "weekday_no"] = test["pickup_datetime"][:].dt.strftime("%w").astype("int")
train.loc[:, "day_no"] = train["pickup_datetime"][:].dt.strftime("%-d").astype("int")
test.loc[:, "day_no"] = test["pickup_datetime"][:].dt.strftime("%-d").astype("int")
train.loc[:, "year_no"] = train["pickup_datetime"][:].dt.strftime("%-y").astype("int")
test.loc[:, "year_no"] = test["pickup_datetime"][:].dt.strftime("%-y").astype("int")




## === cell 8
def dist_haversine(x):
    R = 6371  # km
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat - picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon - picklon)

    a = math.sin(latdiff / 2) * math.sin(latdiff / 2) + math.cos(picklat) * math.cos(
        droplat
    ) * math.sin(londiff / 2) * math.sin(londiff / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


train["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            lambda x: dist_haversine(x),
            train[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=train.index,
)

test["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            lambda x: dist_haversine(x),
            test[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=test.index,
)



## === cell 9
train = train.loc[train["dist_haversine_km"] <= 60, :].copy()



## === cell 10
train["fare_per_km"] = train["fare_amount"] / (train["dist_haversine_km"])
train["fare_per_km_passenger"] = train["fare_amount"] / (
    train["dist_haversine_km"] * train["passenger_count"]
)



## === cell 11
train.groupby("key_id").agg(
    {
        "fare_per_km_passenger": "mean",
        "key_id": "count",
        "passenger_count": "mean",
        "fare_amount": "mean",
    }
)



## === cell 12
train.loc[
    train["fare_per_km_passenger"] > 20,
    ["fare_per_km_passenger", "fare_amount", "dist_haversine_km"],
]



## === cell 13
grouped_df = train.groupby("key_id")
count = 0
for key, item in grouped_df:
    count += 1
    if count == 2:  ## to view key = 2
        filtered = (
            grouped_df.get_group(key)["dist_haversine_km"] > 1
        )  # ignoring drives within 1km
        df = pd.DataFrame(
            grouped_df.get_group(key).loc[filtered, :].sort_values(by="pickup_datetime")
        )
        break



## === cell 14
indexes = ["key_id", df["pickup_datetime"].dt.strftime("%a"), "hour_no"]
grouped = (
    df[:][:].groupby(indexes).agg({"fare_per_km_passenger": "mean", "hour_no": "count"})
)
grouped.rename(columns={"hour_no": "count"}, inplace=True)
grouped



## === cell 15
reindexed = grouped.reset_index().drop("key_id", axis=1)
get_max_count = reindexed.groupby(["pickup_datetime"]).agg({"count": "max"})
get_max_count = get_max_count.reindex(reindexed["pickup_datetime"], method="ffill")
reindexed = reindexed.set_index("pickup_datetime")
reindexed.loc[get_max_count["count"] == reindexed["count"], :]



## === cell 16
train = train.loc[~((train["fare_per_km"] < 0.2) & (train["dist_haversine_km"] > 1))]
train = train.loc[~((train["dist_haversine_km"] < 0.01) & (train["fare_per_km"] > 50))]



## === cell 17
print(
    train.shape[0]
    - train.loc[
        train["pickup_latitude"].between(40.5, 41)
        | train["dropoff_latitude"].between(40.5, 41)
        | train["pickup_longitude"].between(-74, -73.9)
        | train["dropoff_longitude"].between(-74, -73.9)
    ].shape[0]
)
old_train = train.copy()
train = train.loc[
    train["pickup_latitude"].between(40.5, 41)
    & train["dropoff_latitude"].between(40.5, 41)
    & train["pickup_longitude"].between(-74, -73.9)
    & train["dropoff_longitude"].between(-74, -73.9)
]



## === cell 18
train.loc[:, "pickuplat_no"], pick_lat_bin = pd.cut(
    train["pickup_latitude"], 100, labels=False, retbins=True
)
train.loc[:, "pickuplong_no"], pick_long_bin = pd.cut(
    train["pickup_longitude"], 100, labels=False, retbins=True
)
train.loc[:, "dropofflat_no"], drop_lat_bin = pd.cut(
    train["dropoff_latitude"], 100, labels=False, retbins=True
)
train.loc[:, "dropofflong_no"], drop_long_bin = pd.cut(
    train["dropoff_longitude"], 100, labels=False, retbins=True
)
test["pickuplat_no"] = pd.cut(
    test["pickup_latitude"], pick_lat_bin, labels=False
).fillna(int(train.loc[:, "pickuplat_no"].mean()))
test["pickuplong_no"] = pd.cut(
    test["pickup_longitude"], pick_long_bin, labels=False
).fillna(int(train.loc[:, "pickuplong_no"].mean()))



## === cell 19
test["dropofflat_no"] = pd.cut(
    test["dropoff_latitude"], drop_lat_bin, labels=False
).fillna(int(train.loc[:, "dropofflat_no"].mean()))
test["dropofflong_no"] = pd.cut(
    test["dropoff_longitude"], drop_long_bin, labels=False
).fillna(int(train.loc[:, "dropofflong_no"].mean()))



## === cell 20
train.loc[:, "pickdrop_lat_diff"] = abs(
    train["pickuplat_no"].astype(int) - train["dropofflat_no"].astype(int)
)
train.loc[:, "pickdrop_long_diff"] = abs(
    train["pickuplong_no"].astype(int) - train["dropofflong_no"].astype(int)
)
train.loc[:, "final_dist_factor"] = train["pickdrop_lat_diff"].astype(int) + train[
    "pickdrop_long_diff"
].astype(int)

test.loc[:, "pickdrop_lat_diff"] = abs(
    test["pickuplat_no"].astype(int) - test["dropofflat_no"].astype(int)
)
test.loc[:, "pickdrop_long_diff"] = abs(
    test["pickuplong_no"].astype(int) - test["dropofflong_no"].astype(int)
)
test.loc[:, "final_dist_factor"] = test["pickdrop_lat_diff"].astype(int) + test[
    "pickdrop_long_diff"
].astype(int)



## === cell 21
print(train.shape, test.shape)



## === cell 22
train = train.drop(["fare_per_km_passenger", "fare_per_km"], axis=1)




## === cell 23
def calc_cwd_factor(df, col):
    new_df = (
        df.groupby(col)["key_id"].count().sort_values(ascending=False).reset_index()
    )
    new_df["cwd_factor"] = 1
    count = 1
    for i in range(1, new_df.shape[0]):
        count += 1
        if new_df.loc[i - 1, "key_id"] == new_df.loc[i, "key_id"]:
            count -= 1
        new_df.loc[i, "cwd_factor"] = count
    new_df.index = new_df[col]
    return new_df


def apply_cwd_from_train(train_df, test_df, col, new_col):
    fact_df = calc_cwd_factor(train_df, col)
    train_df.loc[:, new_col] = list(
        map(lambda x: fact_df.loc[x, "cwd_factor"], train_df[col])
    )
    fallback = int(fact_df["cwd_factor"].median())
    test_df.loc[:, new_col] = (
        test_df[col].map(fact_df["cwd_factor"]).fillna(fallback).astype(int)
    )
    return train_df, test_df


train, test = apply_cwd_from_train(train, test, "pickuplat_no", "pickuplat_cwd_factor")
train, test = apply_cwd_from_train(
    train, test, "pickuplong_no", "pickuplong_cwd_factor"
)
train, test = apply_cwd_from_train(
    train, test, "dropofflat_no", "dropofflat_cwd_factor"
)
train, test = apply_cwd_from_train(
    train, test, "dropofflong_no", "dropofflong_cwd_factor"
)



## === cell 24
print(train.shape, test.shape)



## === cell 25
import sklearn
from sklearn import *
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



## === cell 26
orig_train = train.copy()
orig_test = test.copy()



## === cell 27
train = shuffle(train.iloc[:, :], random_state=42).reset_index(drop=True)
val = train.iloc[int(0.9 * train.shape[0]) :, :].copy()
train = train.iloc[: int(0.9 * train.shape[0]), :].copy()

cols_to_normalize = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
]

scaler = StandardScaler().fit(train.loc[:, cols_to_normalize])
train.loc[:, cols_to_normalize] = scaler.transform(train.loc[:, cols_to_normalize])
val.loc[:, cols_to_normalize] = scaler.transform(val.loc[:, cols_to_normalize])
test.loc[:, cols_to_normalize] = scaler.transform(test.loc[:, cols_to_normalize])



## === cell 28
import seaborn as sns

categorical_cols = [
    i
    for i in train.columns
    if i
    not in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "abs_diff_longitude",
        "abs_diff_latitude",
        "dist_haversine_km",
        "key",  # keep original key string out of features
        "key_id",  # keep derived id out of features (high-cardinality noise)
        "pickup_datetime",
        "timestamp_with_key",
        "fare_amount",
    ]
]
numerical_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "fare_amount",
]
object_cols = ["key", "key_id", "pickup_datetime", "timestamp_with_key"]
cor = train.loc[:, numerical_cols]
f, ax = plt.subplots(1, 1, figsize=(18, 7))
sns.heatmap(cor.corr(), annot=True)



## === cell 29
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

obj = SelectKBest(f_regression, k=10)
obj.fit(train[categorical_cols], train["fare_amount"])



## === cell 30
categories_selected = []
for i in range(len(categorical_cols)):
    if obj.get_support()[i]:
        categories_selected.append(categorical_cols[i])
categories_selected



## === cell 31
train_y = np.log1p(train["fare_amount"].astype(float))
val_y = np.log1p(val["fare_amount"].astype(float))

cols = [
    i
    for i in categories_selected + numerical_cols
    if i not in ["fare_amount"] + object_cols
]
train_x = train.loc[:, cols]
val_x = val.loc[:, cols]
test_x = test.loc[:, cols]



## === cell 32
import xgboost as xgb
from xgboost import XGBRegressor



## === cell 33
xgbr = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
)

xgb_train = xgbr.fit(
    train_x,
    train_y,
    eval_set=[(val_x, val_y)],
    eval_metric="rmse",
    verbose=False,
    early_stopping_rounds=50,
)



## === cell 34
best_iter = getattr(xgbr, "best_iteration", None)
if best_iter is None:
    pred_train_log = xgbr.predict(train_x)
    pred_val_log = xgbr.predict(val_x)
    pred_test_log = xgbr.predict(test_x)
else:
    pred_train_log = xgbr.predict(train_x, iteration_range=(0, best_iter + 1))
    pred_val_log = xgbr.predict(val_x, iteration_range=(0, best_iter + 1))
    pred_test_log = xgbr.predict(test_x, iteration_range=(0, best_iter + 1))

pred_train = np.expm1(pred_train_log)
pred_val = np.expm1(pred_val_log)
pred_test = np.expm1(pred_test_log)



## === cell 35
from sklearn.metrics import mean_squared_error

rmse_train = np.sqrt(mean_squared_error(train["fare_amount"].astype(float), pred_train))
rmse_val = np.sqrt(mean_squared_error(val["fare_amount"].astype(float), pred_val))
print(rmse_train, rmse_val)



## === cell 36
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

print(
    "Skipping eli5 permutation importance due to protobuf incompatibility in this environment."
)



## === cell 37
pred_test = np.clip(pred_test, 0, 250)

final = pd.DataFrame(
    {"key": test["timestamp_with_key"], "fare_amount": pred_test},
    columns=["key", "fare_amount"],
)
final.to_csv("submission.csv", index=False)



## === cell 38
final.head()
