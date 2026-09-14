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

3.60104

# 6. Current score

6.09152

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.20203) has done: 'Diagnosis: The crash happens in cell 35 because with xgboost==2.0.3 the trained `Booster` returned by `xgb.train()` no longer always exposes `best_ntree_limit` (it was tied to early stopping behavior and older APIs). The model still trains fine, but prediction should use `iteration_range` (newer API) or default to all trees when early stopping info isn’t available. We modify only cell 35 to compute the best iteration safely (when present) and call `predict()` in a version-compatible way.

Patch summary: Replace the direct access to `xgbm.best_ntree_limit` with a robust fallback that checks `xgbm.best_iteration` and uses `iteration_range=(0, best_iteration+1)` when available; otherwise it predicts with the full model.

Updated cells: Only cell 35 is changed.

Compatibility notes for cell k+1: `XGBPredictions` remains a NumPy array of predictions, same as before, so cell 36 continues to work unchanged.

Assumptions: `xgb.train(..., early_stopping_rounds=20, ...)` may or may not set `best_iteration` depending on whether early stopping triggers; in either case the fallback should be correct and deterministic.'
- What this solution (achieved 6.37709) has done: 'Your score (6.20203 RMSE) is worse than the target (3.60104), so we should improve predictive accuracy with the smallest changes that don’t alter the overall approach. The biggest issue is that `dist_calc()` is extremely slow and can lead to incomplete/unstable feature generation; replacing it with a vectorized Haversine distance keeps the same “distance feature” core idea but makes it correct and consistent. Next, we make the XGBoost objective/metric declaration version-correct for xgboost==2.0.3 (`reg:squarederror` instead of deprecated `reg:linear`), and we prevent unnecessary rounding of predictions which increases RMSE. Finally, we make the submission filename end with `.csv` (required) and keep the output schema identical.'
- What this solution (achieved 6.97144) has done: 'Your current script likely didn’t yield a Kaggle score because it writes a filename without the required `.csv` extension for Kaggle submissions, so we fix that first to guarantee a valid submission artifact. To move RMSE down toward the target with minimal core-logic changes, we (1) make the train/test filtering consistent by applying the NYC bounding-box filter only to train (not test) to avoid dropping any test rows, and (2) add two standard “NYC taxi fare” features derived from the same existing coordinates (absolute deltas and a simple Manhattan distance), which keeps the approach identical (tabular features + XGBoost) while improving accuracy. Finally, we ensure the submission row order and count exactly match `sample_submission.csv` by merging on `key`, preventing any accidental misalignment that can silently worsen RMSE.'
- What this solution (achieved 8.05697) has done: 'Your RMSE (6.97) is much worse than the target (3.60), so we should improve accuracy with the smallest safe changes that keep the same overall approach (tabular features + `xgb.train`). The main likely issue is that the current XGBoost setup is severely underfit for this problem (default tree depth/eta and only 300 rounds), so we minimally tune core XGBoost hyperparameters (still same model/training API) to better fit without changing the pipeline. We also add two standard, low-risk geospatial features (Euclidean distance in degrees and bearing) derived only from the same coordinates, which typically reduces RMSE materially without changing evaluation semantics. Finally, we ensure the submission filename ends with `.csv` (Kaggle requirement) while keeping the exact required columns and row alignment with `sample_submission.csv`.'
- What this solution (achieved 6.09152) has done: 'Your current RMSE (8.05697) is far worse than the target (3.60104), so we should improve accuracy with minimal, safe changes that keep the same XGBoost/tabular pipeline. The biggest likely issue is that the model is learning a “global” mapping without the very strong prior that taxi fare is tightly tied to distance; adding a single canonical feature (Haversine distance *in miles*) and a simple interaction (distance × passenger_count) keeps the same feature-extraction style but usually drops RMSE materially. To avoid score degradation from obvious outliers, we also add a minimal, standard training cleanup that removes “zero-distance but high fare” (and vice versa) rows, without touching the test set. Finally, we keep submission alignment identical but ensure the output filename ends with `.csv` (Kaggle requirement).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/kaggle/input"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "../input"



## === cell 1
print(os.listdir(DATA_DIR))



## === cell 2
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))



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
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), nrows=500000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.histplot(train["fare_amount"], kde=True)



## === cell 9
sns.histplot(train["passenger_count"], kde=False)



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)]

train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]


def _nyc_bbox_filter(df):
    return df[
        (df["pickup_longitude"] >= -74.3)
        & (df["pickup_longitude"] <= -72.9)
        & (df["dropoff_longitude"] >= -74.3)
        & (df["dropoff_longitude"] <= -72.9)
        & (df["pickup_latitude"] >= 40.5)
        & (df["pickup_latitude"] <= 41.8)
        & (df["dropoff_latitude"] >= 40.5)
        & (df["dropoff_latitude"] <= 41.8)
    ]


train = _nyc_bbox_filter(train)



## === cell 12
train.describe()




## === cell 13
def dist_calc(df):
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").values)
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").values)
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R_km = 6371.0088
    df["distance"] = (R_km * c).astype("float32")




## === cell 14
dist_calc(train)
dist_calc(test)



## === cell 15
for df in (train, test):
    df["distance_miles"] = (df["distance"] * 0.621371).astype("float32")



## === cell 16
for df in (train, test):
    df["abs_lon_diff"] = np.abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ).astype("float32")
    df["abs_lat_diff"] = np.abs(df["pickup_latitude"] - df["dropoff_latitude"]).astype(
        "float32"
    )
    df["manhattan"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")

    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float32")
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["euclidean_deg"] = np.sqrt(dlon * dlon + dlat * dlat).astype("float32")

    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").values)
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").values)
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").values)
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").values)
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    bearing = np.arctan2(y, x)
    df["bearing_sin"] = np.sin(bearing).astype("float32")
    df["bearing_cos"] = np.cos(bearing).astype("float32")



## === cell 17
for df in (train, test):
    df["dist_x_passengers"] = (
        df["distance_miles"] * df["passenger_count"].astype("float32")
    ).astype("float32")



## === cell 18
train = train[(train["distance"] > 0.0) & (train["distance"] < 200.0)]
train = train[~((train["distance"] < 0.05) & (train["fare_amount"] > 30.0))]



## === cell 19
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "", regex=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 20
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "", regex=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 21
train = train.dropna(subset=["pickup_datetime"])
test = test.dropna(subset=["pickup_datetime"])

train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year



## === cell 22
test.head()



## === cell 23
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(numeric_only=True),
    annot=True,
    fmt=".4f",
)



## === cell 24
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 25
X.head()



## === cell 26
y.head()



## === cell 27
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 28
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 29
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_valid, y_valid))



## === cell 30
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 31
LinearPredictions = lm.predict(test_pred)
LinearPredictions



## === cell 32
LinearPredictions.size



## === cell 33
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 34
linear_submission.head()




## === cell 35
def XGBoost(X_train, X_valid, y_train, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )




## === cell 36
xgbm = XGBoost(X_train, X_valid, y_train, y_valid)

dtest_pred = xgb.DMatrix(test_pred)

best_iter = getattr(xgbm, "best_iteration", None)
if best_iter is not None:
    XGBPredictions = xgbm.predict(dtest_pred, iteration_range=(0, best_iter + 1))
else:
    XGBPredictions = xgbm.predict(dtest_pred)

XGBPredictions = np.maximum(XGBPredictions, 0.0)



## === cell 37
XGBPredictions



## === cell 38
XGBPredictions = XGBPredictions
XGBPredictions



## === cell 39
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 40
submission = XGB_submission



## === cell 41
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
submission = sample_sub[["key"]].merge(submission, on="key", how="left")

if submission["fare_amount"].isnull().any():
    submission["fare_amount"] = submission["fare_amount"].fillna(
        float(np.nanmean(XGBPredictions))
    )



## === cell 42
out_path = "XGBSubmission17082018.csv"
submission.to_csv(out_path, index=False)
print(
    "Wrote submission:",
    out_path,
    "rows:",
    len(submission),
    "cols:",
    list(submission.columns),
)
print("Null fare_amount rows:", int(submission["fare_amount"].isnull().sum()))
print("Example head:\n", submission.head())
