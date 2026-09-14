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
geopy==2.4.1
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

3.86972

# 6. Current score

5.11272

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.77094) has done: 'Diagnosis: The crash occurs in cell 35 because `xgb.train()` returns a `Booster` object, and with xgboost==2.0.3 this object no longer exposes `best_ntree_limit`. The early-stopping result is still available via `best_iteration`, and the supported way to limit trees during prediction is `iteration_range`.  
Patch summary: Update the prediction call in cell 35 to use `iteration_range=(0, best_iteration+1)` when `best_iteration` exists; otherwise fall back to plain `predict`. This preserves the same early-stopping semantics without changing the model/training logic.  
Updated cells: Only cell 35 is modified.  
Compatibility notes for cell k+1: `XGBPredictions` remains a NumPy array and is still defined as expected for cell 36.  
Assumptions: The model was trained with early stopping as shown, so `best_iteration` is available; if not, the fallback path still run.'
- What this solution (achieved 5.74411) has done: 'Your current gap to the target is large (RMSE 5.77 vs 3.87, lower is better), and the biggest driver is a slow, row-wise `great_circle` distance that both hurts feature quality (noise/NaNs risk) and makes it hard to scale training data. I keep the same overall pipeline (same models, same features: distance + time parts + passenger_count, same training approach), but replace the distance computation with a vectorized haversine (same semantic “distance in km”, just computed efficiently and consistently). With that speedup, I increase the training sample size moderately (still from train.csv only, no leakage) to reduce variance and improve generalization toward the target. I also update the XGBoost objective to the modern equivalent (`reg:squarederror`) without changing the RMSE optimization intent, and I write a submission file with a `.csv` extension matching Kaggle expectations.'
- What this solution (achieved 5.74403) has done: 'Your current RMSE (5.74411) is worse than the target (3.86972), so we should make a small, safe improvement without changing the modeling approach. The biggest low-risk gain here is to remove prediction rounding to cents, because Kaggle RMSE is computed on raw floats and rounding adds quantization error. I keep your exact features, split, and XGBoost training settings, but output full-precision predictions and (to be safe) clip negative fares to 0 which matches your train-time constraint `fare_amount > 0`. I also write the submission with a `.csv` suffix (required) while keeping the same filename stem.'
- What this solution (achieved 5.78961) has done: 'Your RMSE is still far above target, so we should make a small, low-risk improvement that preserves your exact feature set and XGBoost training approach. The biggest issue is that the model is trained on raw fares with heavy-tailed outliers, which hurts RMSE; we can keep the same objective and model but clip extreme `fare_amount` values in training to a reasonable upper bound to reduce outlier influence. This does not change your features, architecture, loops, or metric semantics, and it tends to improve generalization on this competition. I also ensure the submission filename ends with `.csv` (it already does) and keep prediction clipping at 0.'
- What this solution (achieved 5.76568) has done: 'We need to move your RMSE down toward 3.86972 (lower is better) from 5.78961, while keeping the same overall feature set and the same XGBoost training approach. The smallest, high-impact fix is to remove the train/validation leakage mismatch by using the held-out set only for early stopping, but then re-train the final model on all cleaned training data for the chosen number of boosting rounds; this preserves your exact model type and objective but typically improves test performance materially. To keep training stable and comparable, we also make the early-stopping split deterministic and compute `best_num_boost_round` explicitly from `best_iteration`. Submission formatting and file path remain unchanged.'
- What this solution (achieved 5.74258) has done: 'Your current RMSE (5.76568) is far worse than the target (3.86972), so we should improve generalization with the smallest possible, low-risk changes while keeping your exact feature set and XGBoost training approach intact. The biggest issue is that the final model is retrained on all data but without shuffling/stratification considerations; we keep that, but reduce variance and overfitting by adding conservative regularization parameters (still same XGBoost model/objective) and enabling subsampling/colsampling, which typically helps this competition without altering core semantics. We also ensure datetime parsing handles fractional seconds in the `key`-derived timestamp format robustly using `infer_datetime_format`-style behavior via `pd.to_datetime` without a fixed format, preventing silent `NaT` drops that can hurt performance. Finally, we keep the same submission structure and ensure the `.csv` is written as before.'
- What this solution (achieved 5.80388) has done: 'To move RMSE down toward your target with minimal disruption, I keep the exact same feature set and XGBoost training approach but fix two high-impact data issues that commonly inflate error in this competition: (1) ensure the `test` dataframe uses the same numeric dtypes as `train` (avoids subtle dtype/NA inconsistencies and keeps distance/time features consistent), and (2) remove unrealistic rides by filtering on a reasonable maximum computed trip distance (this is a standard cleanup that reduces label noise/outliers without changing the model). I also clip XGBoost predictions to the same upper bound used in training (200) to avoid extreme outputs that hurt RMSE. The script still run end-to-end and write a valid `.csv` submission with the required columns.'
- What this solution (achieved 5.78595) has done: 'Your RMSE (5.80388) is still far above the target (3.86972), so we should make a small, low-risk improvement without changing your feature set or model training approach. The biggest remaining issue is that the model is trained on the raw fare scale, which is heavy‑tailed; we can keep the same XGBoost reg:squarederror training, but train on `log1p(fare_amount)` and invert with `expm1` at prediction time—this commonly reduces RMSE on this competition while preserving core semantics. To keep predictions physically plausible and consistent with your training clipping, we clip inverted predictions to `[0, 200]`. Everything else (data loading, cleaning, distance/time features, early-stopping then retrain) remains the same and the script still writes a valid submission CSV.'
- What this solution (achieved 5.78595) has done: 'Your RMSE (5.78595, lower is better) is still far above the target (3.86972), so we should make one small, high-impact improvement without changing the feature set or XGBoost training approach. The biggest likely driver is inconsistent/poor datetime parsing: `pd.to_datetime(..., errors="coerce")` can silently produce many `NaT` values given the known mixed formats in this dataset, and then you drop those rows, which changes the effective training distribution and harms generalization. I switch to a more robust parse by explicitly removing `" UTC"` and enabling `utc=True` (then dropping tz) so fewer rows are lost and the hour/weekday/month/year features become more reliable, while keeping the same features, model, and training logic. Everything else (distance computation, cleaning rules, log1p target, early-stopping then retrain, submission format/path) remains the same.'
- What this solution (achieved 5.77684) has done: 'Your RMSE (5.78595, lower is better) is still far above the target (3.86972), so we should make one minimal, legitimate improvement that preserves your exact feature set and XGBoost training approach. The biggest remaining issue is target/metric mismatch: you train on `log1p(fare)` but still evaluate early stopping using RMSE on the log scale, which can select a suboptimal number of trees for minimizing RMSE on the original dollar scale. I keep the log-target pipeline exactly the same, but change only the early-stopping evaluation metric to a custom function that computes RMSE in the original fare space after `expm1`, so early stopping selects trees that better align with the Kaggle metric. Everything else (data loading/cleaning, features, training/retrain flow, submission format/path) remains the same and still writes a valid `.csv`.'
- What this solution (achieved 6.00099) has done: 'To move RMSE down toward your target (3.86972) from 5.77684 while keeping the same feature set and the same XGBoost training approach, I make two minimal, high-impact data-quality fixes that directly reduce noise/outliers: filter out zero/negative fares and extremely short trips, and remove obviously wrong geolocations (0/0 and out-of-range lat/lon) before computing distance. I also add one standard, very small feature that doesn’t change the modeling approach (still just engineered numeric columns): absolute lat/lon deltas, which helps the model learn distance/fare relationships more robustly than distance alone. Everything else (log1p target, custom early-stopping metric in fare-space, retrain on full data with selected boosting rounds, submission writing) stays the same.'
- What this solution (achieved 5.78288) has done: 'Your current RMSE (6.00099, lower is better) is still far above the target (3.86972), so we need a small change that improves generalization without changing your feature set or XGBoost training approach. The most likely cause of the regression is the strict “trip distance >= 0.1 km” filter (and some other hard filters) removing many legitimate short rides that are common in the test set, causing a train/test distribution mismatch. I keep all core logic intact, but relax only the minimum-distance threshold slightly and (to keep consistency) apply the same basic coordinate sanity filters to the test set (without dropping rows) by setting invalid coordinates to NaN and filling distance with 0 so the submission still has all keys. This should reduce error by better matching the test distribution while preserving the same model, objective, and feature engineering.'
- What this solution (achieved 5.11272) has done: 'We need to move RMSE down (lower is better) from 5.78288 toward 3.86972, so the smallest legitimate improvement is to fix a train/test preprocessing mismatch that likely hurts generalization: you apply strict NYC bounding-box filters only to train, but you keep all test rows (including clearly invalid coordinates), which then get turned into distance/feature noise and degrade predictions. I keep your exact feature set and XGBoost training approach, but apply the same coordinate sanity checks to the test set by setting invalid coordinates to NaN (not dropping rows), recompute distance with NaN-safe handling (invalid → distance=0), and ensure downstream engineered features don’t produce NaNs/Infs. This preserves core semantics while reducing distribution mismatch and noise, which should move RMSE downward toward the target. The submission writing remains unchanged and still produces a valid `.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from geopy.distance import great_circle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv(
    "../input/test.csv",
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=2000000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.distplot(train["fare_amount"])



## === cell 9
sns.distplot(train["passenger_count"])



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)

train = train[train["fare_amount"] > 0]
train = train[
    train["fare_amount"] >= 2.50
]  # NYC minimum fare; reduces ultra-noisy label tail

for col in ["pickup_longitude", "dropoff_longitude"]:
    train = train[train[col].between(-180.0, 180.0)]
for col in ["pickup_latitude", "dropoff_latitude"]:
    train = train[train[col].between(-90.0, 90.0)]

train = train[
    ~(
        (train["pickup_longitude"].abs() < 1e-6)
        & (train["pickup_latitude"].abs() < 1e-6)
    )
]
train = train[
    ~(
        (train["dropoff_longitude"].abs() < 1e-6)
        & (train["dropoff_latitude"].abs() < 1e-6)
    )
]

train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]

train["fare_amount"] = train["fare_amount"].clip(upper=200.0)



## === cell 12
train.describe()




## === cell 13
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 14
def apply_coordinate_sanity_to_test_inplace(df):
    df = df.copy()

    for col in ["pickup_longitude", "dropoff_longitude"]:
        bad = ~df[col].between(-180.0, 180.0)
        df.loc[bad, col] = np.nan
    for col in ["pickup_latitude", "dropoff_latitude"]:
        bad = ~df[col].between(-90.0, 90.0)
        df.loc[bad, col] = np.nan

    bad_pick = (df["pickup_longitude"].abs() < 1e-6) & (
        df["pickup_latitude"].abs() < 1e-6
    )
    df.loc[bad_pick, ["pickup_longitude", "pickup_latitude"]] = np.nan

    bad_drop = (df["dropoff_longitude"].abs() < 1e-6) & (
        df["dropoff_latitude"].abs() < 1e-6
    )
    df.loc[bad_drop, ["dropoff_longitude", "dropoff_latitude"]] = np.nan

    bad_pick_bb = ~(
        (df["pickup_longitude"] < -72)
        & (df["pickup_latitude"] > 40)
        & (df["pickup_latitude"] < 44)
    )
    df.loc[bad_pick_bb, ["pickup_longitude", "pickup_latitude"]] = np.nan

    bad_drop_bb = ~(
        (df["dropoff_longitude"] < -72)
        & (df["dropoff_latitude"] > 40)
        & (df["dropoff_latitude"] < 44)
    )
    df.loc[bad_drop_bb, ["dropoff_longitude", "dropoff_latitude"]] = np.nan

    bad_pc = ~((df["passenger_count"] > 0) & (df["passenger_count"] < 10))
    df.loc[bad_pc, "passenger_count"] = 0

    return df


def dist_calc_vectorized(df):
    R = 6371.0  # Earth radius in km

    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))

    dist = R * c
    dist = np.where(np.isfinite(dist), dist, 0.0).astype("float32")
    df["distance"] = dist


dist_calc_vectorized(train)

test = apply_coordinate_sanity_to_test_inplace(test)
dist_calc_vectorized(test)



## === cell 15
train = train[(train["distance"] >= 0.01) & (train["distance"] <= 100.0)]



## === cell 16
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)



## === cell 17
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)



## === cell 18
train = train.dropna(subset=["pickup_datetime"])

train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year

for c in ["hour", "weekday", "month", "year"]:
    test[c] = test[c].fillna(0).astype("int16")



## === cell 19
test.head()



## === cell 20
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 21
train["abs_lon_diff"] = (
    (train["pickup_longitude"] - train["dropoff_longitude"]).abs().astype("float32")
)
train["abs_lat_diff"] = (
    (train["pickup_latitude"] - train["dropoff_latitude"]).abs().astype("float32")
)

test["abs_lon_diff"] = (
    (test["pickup_longitude"] - test["dropoff_longitude"]).abs().astype("float32")
)
test["abs_lat_diff"] = (
    (test["pickup_latitude"] - test["dropoff_latitude"]).abs().astype("float32")
)

for c in ["abs_lon_diff", "abs_lat_diff"]:
    test[c] = test[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")



## === cell 22
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)
y = train["fare_amount"]



## === cell 23
X.head()



## === cell 24
y.head()



## === cell 25
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 26
test_pred = test.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)

test_pred = test_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)



## === cell 27
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_valid, y_valid))



## === cell 28
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 29
LinearPredictions = lm.predict(test_pred).astype("float32")
LinearPredictions = np.clip(LinearPredictions, 0.0, None)
LinearPredictions



## === cell 30
LinearPredictions.size



## === cell 31
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 32
linear_submission.head()




## === cell 33
def _rmse_fare_space(pred_log, dmatrix):
    y_true_log = dmatrix.get_label()
    pred_fare = np.expm1(pred_log.astype("float64"))
    true_fare = np.expm1(y_true_log.astype("float64"))
    rmse = np.sqrt(np.mean((pred_fare - true_fare) ** 2))
    return "rmse_fare", float(rmse)


def XGBoost(X_train, X_valid, y_train, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "seed": 42,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        early_stopping_rounds=20,
        evals=[(dvalid, "valid")],
        feval=_rmse_fare_space,
        maximize=False,
        verbose_eval=False,
    )




## === cell 34
y_log = np.log1p(y.astype("float64"))
X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

xgbm_es = XGBoost(X_train, X_valid, y_train_log, y_valid_log)

best_iter = getattr(xgbm_es, "best_iteration", None)
if best_iter is None:
    best_num_boost_round = 300
else:
    best_num_boost_round = int(best_iter) + 1

dtrain_all = xgb.DMatrix(X, label=y_log)

final_params = {
    "objective": "reg:squarederror",
    "seed": 42,
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
}

xgbm_final = xgb.train(
    params=final_params,
    dtrain=dtrain_all,
    num_boost_round=best_num_boost_round,
    evals=[(dtrain_all, "train")],
    verbose_eval=False,
)

dtest_pred = xgb.DMatrix(test_pred)
XGBPredictions_log = xgbm_final.predict(dtest_pred)



## === cell 35
XGBPredictions_log



## === cell 36
XGBPredictions = np.expm1(XGBPredictions_log.astype("float64")).astype("float32")
XGBPredictions = np.clip(XGBPredictions, 0.0, 200.0)
XGBPredictions



## === cell 37
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 38
submission = XGB_submission



## === cell 39
submission_path = "XGBSubmission17082018.csv"
submission.to_csv(submission_path, index=False)
print("Wrote submission:", os.path.abspath(submission_path), "rows:", len(submission))
