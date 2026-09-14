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

3.8

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

4.41355

# 6. Current score

5.89209

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.94612) has done: 'I fix the XGBoost inference error caused by `best_ntree_limit` being removed in xgboost 2.x by switching to the supported `iteration_range` API, while keeping the same training loop and early stopping behavior. I also make the notebook compatible with a `.py` run by removing the IPython-only `%matplotlib inline` magic and guarding plotting so it won’t break headless execution. Finally, I ensure the submission is always written as `submission.csv` with the required `key,fare_amount` columns even when early stopping is not triggered.'
- What this solution (achieved 5.9365) has done: 'You’re currently above the target RMSE (5.946 > 4.414), so we should make small, legitimate improvements that usually reduce error without changing the overall approach (still XGBoost regression on engineered features). The biggest low-risk win is to fix the XGBoost objective name (`reg:linear` is deprecated/aliased and can behave inconsistently) and to add a couple of standard tree/learning parameters that improve generalization while keeping the same training loop and early stopping. We also compute the datetime-derived features with vectorized `.dt` accessors (same semantics, fewer NaN surprises and faster), and we clip negative predictions to 0 since fares can’t be negative (often improves RMSE slightly). Submission format and path remain unchanged and still writes `submission.csv`.'
- What this solution (achieved 5.94013) has done: 'Your current RMSE (5.9365) is worse than the target (4.41355), so we make small, legitimate improvements that typically reduce error without changing the overall “XGBoost on simple engineered features” approach. The biggest low-risk win here is to fix a feature mismatch: you compute `weekday` for training but don’t include it in `features` (and you don’t compute it for test either), so we add `weekday` consistently to both train/test feature sets. We also add one standard, conservative regularization term (`reg_lambda`) to reduce overfitting slightly, keeping the same training loop, early stopping, and objective/metric semantics. Everything else (data loading, filters, distance function, train/test split, inference API, and writing `submission.csv`) remains intact.'
- What this solution (achieved 5.91503) has done: 'We’re currently worse than the target RMSE (5.94 vs 4.41, lower is better), so the smallest legitimate step toward the target is to improve generalization without changing the core “XGBoost on simple engineered features” approach. I (1) slightly increase training data size (still filtered the same way) because this model is data-hungry and 1M rows is often not enough for this competition, (2) add a couple of standard, conservative XGBoost regularization parameters (`gamma`, `reg_alpha`) while keeping the same objective/metric/training loop/early stopping, and (3) ensure passenger_count is in a sane range (1–6) since extreme values are noise and hurt RMSE. Submission writing remains identical (`submission.csv` with `key,fare_amount`) and inference continues to use `iteration_range` for xgboost 2.x compatibility.'
- What this solution (achieved 5.93903) has done: 'To move your RMSE down toward the 4.41 target while keeping the same “XGBoost on engineered datetime + haversine distance + passenger_count” core logic, I make only small, high-impact data-quality changes that reduce label noise. Specifically, I add standard NYC Taxi fare cleaning filters for latitude/longitude bounds and remove obviously wrong coordinates (still using the same features, model, loss, and training loop). I also ensure train/test dtypes match for passenger_count (int) to avoid subtle split/handling differences. Everything else (feature set, distance function, XGBoost params, early stopping, prediction clipping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 5.90137) has done: 'Your current RMSE (5.939) is far above the target (4.414), so the smallest legitimate move toward the target is to reduce label noise and add one missing high-signal feature while keeping the same “XGBoost on engineered features + early stopping” core logic. I (1) add a standard fare floor filter (remove unrealistic low fares) that typically improves RMSE a lot on this competition, and (2) add a single additional datetime feature (`month`) computed identically for train/test (no architecture/training-loop change). Everything else (distance, existing filters, XGBoost training with the same loop/early stopping, prediction clipping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 5.72286) has done: 'We’re still well above the target RMSE (5.90 vs 4.41, lower is better), so the smallest high-impact step is to improve feature signal without changing the overall “XGBoost regression on engineered features + early stopping” approach. I add two standard, competition-proven engineered features derived from the same inputs you already use: straight-line distance in NYC degrees (L1 “manhattan” distance) and an approximate bearing angle; these typically reduce RMSE materially while keeping the same model/training loop. I also make the validation split deterministic but more representative by using a simple time-based split (train on earlier rides, validate on later rides), which reduces leakage-like effects and usually improves generalization. Submission writing, inference via `iteration_range`, and the rest of your pipeline stay the same.'
- What this solution (achieved 5.89209) has done: 'Your current RMSE (5.72286) is still well above the target (4.41355), so we should make small, legitimate improvements that reduce error without changing the core “XGBoost on engineered datetime + geo features” approach. The lowest-risk gain here is to add a single, competition-standard feature: the distance from pickup/dropoff to a fixed NYC center point (and reuse those two distances as features), which often helps the model learn airport/outer-borough effects without altering the training loop or model type. We also add two conservative XGBoost knobs (`max_depth` slightly reduced and `min_child_weight` increased) to reduce overfitting noise given the added features, keeping the same objective/metric/early-stopping semantics. Submission writing remains identical (`submission.csv` with `key,fare_amount`) and inference continues to use `iteration_range` for xgboost 2.x compatibility.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

DO_PLOTS = False



## === cell 1
df_train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
)
df_train.head()



## === cell 2
df_train.describe()



## === cell 3
df_test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
df_test.head()



## === cell 4
df_test.describe()



## === cell 5
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >= 0]
print("New size: %d" % len(df_train))



## === cell 6
print("Old size: %d" % len(df_train))
df_train = df_train.dropna(how="any", axis="rows")
print("New size: %d" % len(df_train))



## === cell 7
if DO_PLOTS:
    df_train[df_train.fare_amount < 80].fare_amount.hist(bins=100)
    plt.xlabel("fare $USD")



## === cell 8
df_train["diff_long"] = (df_train.dropoff_longitude - df_train.pickup_longitude).abs()
df_train["diff_long"].describe()



## === cell 9
df_train["diff_lat"] = (df_train.dropoff_latitude - df_train.pickup_latitude).abs()
df_train["diff_lat"].describe()



## === cell 10
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.diff_long < 5.0) & (df_train.diff_lat < 5.0)]
print("New size: %d" % len(df_train))



## === cell 11
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.passenger_count >= 1) & (df_train.passenger_count <= 6)]
print("New size: %d" % len(df_train))



## === cell 12
print("Old size: %d" % len(df_train))
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
df_train = df_train[
    (df_train.pickup_longitude.between(-75, -72))
    & (df_train.dropoff_longitude.between(-75, -72))
    & (df_train.pickup_latitude.between(40, 42))
    & (df_train.dropoff_latitude.between(40, 42))
]
df_train = df_train[
    (df_train.pickup_longitude != 0)
    & (df_train.pickup_latitude != 0)
    & (df_train.dropoff_longitude != 0)
    & (df_train.dropoff_latitude != 0)
]
print("New size: %d" % len(df_train))



## === cell 13
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >= 2.5]
print("New size: %d" % len(df_train))



## === cell 14
df_train["year"] = df_train["pickup_datetime"].dt.year
df_train["month"] = df_train["pickup_datetime"].dt.month
df_train["weekday"] = df_train["pickup_datetime"].dt.weekday
df_train["hour"] = df_train["pickup_datetime"].dt.hour



## === cell 15
df_train.describe()



## === cell 16
df_train[["fare_amount", "hour"]].groupby(["hour"], as_index=False).mean().sort_values(
    by="fare_amount", ascending=False
)



## === cell 17
df_train[["fare_amount", "weekday"]].groupby(
    ["weekday"], as_index=False
).mean().sort_values(by="fare_amount", ascending=False)



## === cell 18
df_train[["fare_amount", "year"]].groupby(["year"], as_index=False).mean().sort_values(
    by="fare_amount", ascending=False
)




## === cell 19
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...




## === cell 20
df_train["distance"] = distance(
    df_train.pickup_latitude,
    df_train.pickup_longitude,
    df_train.dropoff_latitude,
    df_train.dropoff_longitude,
)



## === cell 21
if DO_PLOTS:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        df_train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
    )



## === cell 22
if DO_PLOTS:
    _ = df_train.plot.scatter("distance", "fare_amount")



## === cell 23
if DO_PLOTS:
    _ = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter(
        "distance", "fare_amount", alpha=0.1
    )



## === cell 24
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.distance >= 0.1)]
print("New size: %d" % len(df_train))



## === cell 25
if DO_PLOTS:
    _ = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter(
        "distance", "fare_amount", alpha=0.1
    )



## === cell 26
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.distance <= 50)]
print("New size: %d" % len(df_train))



## === cell 27
if DO_PLOTS:
    _ = df_train.plot.scatter("distance", "fare_amount", alpha=0.1)



## === cell 28
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.fare_amount <= 200)]
print("New size: %d" % len(df_train))



## === cell 29
if DO_PLOTS:
    _ = df_train.plot.scatter("distance", "fare_amount", alpha=0.1)




## === cell 30
def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    dlon = df["dropoff_longitude"] - df["pickup_longitude"]
    dlat = df["dropoff_latitude"] - df["pickup_latitude"]
    df["manhattan"] = dlon.abs() + dlat.abs()
    df["bearing"] = np.arctan2(dlat.values, dlon.values)
    return df


df_train = add_geo_features(df_train)

NYC_LAT, NYC_LON = 40.7128, -74.0060
df_train["pickup_to_center"] = distance(
    df_train["pickup_latitude"].values,
    df_train["pickup_longitude"].values,
    NYC_LAT,
    NYC_LON,
)
df_train["dropoff_to_center"] = distance(
    df_train["dropoff_latitude"].values,
    df_train["dropoff_longitude"].values,
    NYC_LAT,
    NYC_LON,
)



## === cell 31
features = [
    "year",
    "month",
    "weekday",
    "hour",
    "distance",
    "manhattan",
    "bearing",
    "pickup_to_center",  # Change: new feature
    "dropoff_to_center",  # Change: new feature
    "passenger_count",
]
X = df_train[features].values
y = df_train["fare_amount"].values



## === cell 32
df_test["year"] = df_test["pickup_datetime"].dt.year
df_test["month"] = df_test["pickup_datetime"].dt.month
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday
df_test["hour"] = df_test["pickup_datetime"].dt.hour
df_test["distance"] = distance(
    df_test.pickup_latitude,
    df_test.pickup_longitude,
    df_test.dropoff_latitude,
    df_test.dropoff_longitude,
)

df_test = add_geo_features(df_test)

df_test["pickup_to_center"] = distance(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    NYC_LAT,
    NYC_LON,
)
df_test["dropoff_to_center"] = distance(
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
    NYC_LAT,
    NYC_LON,
)

df_test["passenger_count"] = df_test["passenger_count"].astype(np.int64)



## === cell 33
X_kaggle_test = df_test[features].values



## === cell 34
from sklearn.model_selection import train_test_split
import xgboost as xgb

df_train_sorted = df_train.sort_values("pickup_datetime").reset_index(drop=True)
split_idx = int(len(df_train_sorted) * 0.7)

X_train = df_train_sorted.iloc[:split_idx][features].values
y_train = df_train_sorted.iloc[:split_idx]["fare_amount"].values
X_test = df_train_sorted.iloc[split_idx:][features].values
y_test = df_train_sorted.iloc[split_idx:]["fare_amount"].values


def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "gamma": 0.1,
        "seed": 10,
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=200,
        early_stopping_rounds=100,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(X_train, X_test, y_train, y_test)

dtest = xgb.DMatrix(X_kaggle_test)
best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    prediction = model.predict(dtest)
else:
    prediction = model.predict(dtest, iteration_range=(0, best_iter + 1))

prediction = np.maximum(prediction, 0.0)



## === cell 35
submission = pd.DataFrame(
    {"key": df_test["key"], "fare_amount": np.round(prediction, 2)}
)

submission.to_csv("submission.csv", index=False)
submission.head()
