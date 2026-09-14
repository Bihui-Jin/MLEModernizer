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

5.85939

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.94612) has done: 'Diagnosis: In xgboost==2.0.3, the `Booster` attribute `best_ntree_limit` no longer exists (it was used in older versions with early stopping). The model still tracks the best iteration, and prediction should use `iteration_range` (or omit the limit) to be compatible with newer XGBoost. The crash occurs specifically at `model.predict(..., ntree_limit=model.best_ntree_limit)` in cell 32.

Patch summary: Update the prediction call to use `iteration_range=(0, model.best_iteration + 1)` when `best_iteration` is available; otherwise fall back to a plain `predict`. This preserves early-stopping semantics while avoiding the removed attribute.

Updated cells: Only cell 32 is modified.

Compatibility notes for cell k+1: The variable `prediction` remains a 1D numpy array of predictions with the same length/order as `df_test`, so cell 33 can build the submission exactly as before.

Assumptions: `best_iteration` is present when early stopping is used in this XGBoost version; if not, predicting without an iteration cap is an acceptable fallback and avoids crashing.'
- What this solution (achieved 5.94612) has done: 'Your current RMSE (5.94612) is worse than the target (4.41355), so we should modestly improve generalization without changing the overall approach (same features, same XGBoost training API, same early-stopping setup). The biggest low-risk gain here is correcting the deprecated XGBoost objective (`reg:linear`) to the proper squared-error objective while keeping the same boosting rounds and early stopping, which typically reduces error substantially on this competition. I’m also keeping the same evaluation semantics and submission format, only ensuring XGBoost uses a modern objective and deterministic behavior. Everything else (data loading, cleaning, feature engineering, training loop structure, and CSV writing) remains the same.'
- What this solution (achieved 5.93617) has done: 'To move your RMSE down toward the target with minimal disruption, I keep the same feature set and XGBoost training API but tune a few conservative parameters that typically improve generalization on this competition (tree depth, learning rate, subsampling, and regularization). I also add a simple and safe post-processing step: clip negative fare predictions to 0 (fares can’t be negative), which usually reduces RMSE slightly without changing the modeling approach. Finally, I make `passenger_count` sanitization consistent between train and test (fill missing and clip to a reasonable range) to avoid distribution mismatch. The script still run end-to-end and produce `submission.csv` with the required columns.'
- What this solution (achieved 5.92236) has done: 'To move your RMSE down toward the target (lower is better) without changing the core modeling approach, I’m keeping the same XGBoost training API, early stopping, and feature set, but I add two safe, competition-standard features derived from the existing columns: `weekday` and `month` from `pickup_datetime`. This is a minimal extension of your current time-based feature extraction (you already compute `weekday` for train but weren’t using it, and test didn’t have it), and it typically reduces error noticeably on this dataset. I also add the same `weekday`/`month` computation to the test set to avoid train/test feature mismatch, and keep your existing clipping and submission formatting intact.'
- What this solution (achieved 5.8876) has done: 'Your RMSE is above the target (lower is better), so we should make small, low-risk improvements that typically reduce error without changing the overall approach (same features + XGBoost + early stopping). The largest gain with minimal disruption is to use the same validation split but make it time-aware by sorting on `pickup_datetime` and disabling shuffle in `train_test_split`, which reduces train/validation leakage from mixing time periods and usually generalizes better on this dataset. I also fix a subtle distribution mismatch by applying the same basic geographic sanity filtering to the test set (only remove clearly invalid coordinates), which prevents extreme distances from blowing up predictions. Finally, I keep your prediction clipping and submission format unchanged, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 5.89289) has done: 'Your current RMSE (5.8876) is still above the target (4.41355), so we should make small, safe improvements that usually reduce error without changing your core approach (same features + XGBoost train API + early stopping). The biggest low-risk fix here is to make your geographic cleaning consistent by applying the same “diff_long/diff_lat” sanity filter to the training data and then *imputing* bad test coordinates (rather than letting them create extreme/NaN distances), which avoids outlier distances that hurt predictions. I also add a minimal, competition-standard feature that doesn’t change the modeling approach: a straight-line Manhattan distance (`abs(dlat)+abs(dlon)`) computed from the same coordinates, and include it in `features` for both train and test. Finally, I keep your time-aware split, objective/metric, and prediction clipping unchanged, and still write a valid `submission.csv`.'
- What this solution (achieved 5.91938) has done: 'To move your RMSE down toward the target with minimal disruption, I’m keeping the exact same modeling approach (XGBoost + early stopping) and the same feature set, but I add one competition-standard feature computed from the same coordinates: the “haversine distance in miles” (your current `distance` is in km). This keeps core logic intact while giving the model a better-calibrated scale that typically improves generalization without changing training loops or loss. I also add a very small, safe cleanup: drop zero/negative passenger counts (and keep the existing clipping) because those rows are usually data errors that add noise. All paths and submission format remain unchanged, and it still writes `submission.csv`.'
- What this solution (achieved 5.85939) has done: 'Your RMSE is still above the target (lower is better), so we should make a small, legitimate improvement that usually reduces error without changing the overall approach (still XGBoost + early stopping + same training loop). The biggest low-risk gain here is adding one more standard distance feature computed from the same coordinates: `euclidean` (straight-line distance in degrees), which complements your existing haversine+manhattan and is cheap to compute. I also add a minimal, consistent coordinate-based cleaning step to the training set (filter clearly invalid lat/lon ranges) to reduce noise; this doesn’t change the model or loss, just removes corrupted rows similarly to what you already guard against in test. Everything else (paths, split logic, objective/metric, prediction clipping, and submission formatting) stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

plt.style.use("seaborn-whitegrid")



## === cell 1
df_train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
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
coord_ok_train = (
    df_train["pickup_longitude"].between(-180, 180)
    & df_train["dropoff_longitude"].between(-180, 180)
    & df_train["pickup_latitude"].between(-90, 90)
    & df_train["dropoff_latitude"].between(-90, 90)
)
df_train = df_train[coord_ok_train].copy()
print("New size: %d" % len(df_train))



## === cell 12
df_train["year"] = df_train.pickup_datetime.apply(lambda t: t.year)
df_train["month"] = df_train.pickup_datetime.apply(lambda t: t.month)
df_train["weekday"] = df_train.pickup_datetime.apply(lambda t: t.weekday())
df_train["hour"] = df_train.pickup_datetime.apply(lambda t: t.hour)



## === cell 13
df_train.describe()



## === cell 14
df_train[["fare_amount", "hour"]].groupby(["hour"], as_index=False).mean().sort_values(
    by="fare_amount", ascending=False
)



## === cell 15
df_train[["fare_amount", "weekday"]].groupby(
    ["weekday"], as_index=False
).mean().sort_values(by="fare_amount", ascending=False)



## === cell 16
df_train[["fare_amount", "year"]].groupby(["year"], as_index=False).mean().sort_values(
    by="fare_amount", ascending=False
)




## === cell 17
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin... (kilometers)




## === cell 18
df_train["distance"] = distance(
    df_train.pickup_latitude,
    df_train.pickup_longitude,
    df_train.dropoff_latitude,
    df_train.dropoff_longitude,
)



## === cell 19
plt.figure(figsize=(15, 8))
sns.heatmap(
    df_train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 20
plot = df_train.plot.scatter("distance", "fare_amount")



## === cell 21
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter(
    "distance", "fare_amount", alpha=0.1
)



## === cell 22
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.distance >= 0.1)]
print("New size: %d" % len(df_train))



## === cell 23
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter(
    "distance", "fare_amount", alpha=0.1
)



## === cell 24
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.distance <= 50)]
print("New size: %d" % len(df_train))



## === cell 25
plot = df_train.plot.scatter("distance", "fare_amount", alpha=0.1)



## === cell 26
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.fare_amount <= 200)]
print("New size: %d" % len(df_train))



## === cell 27
plot = df_train.plot.scatter("distance", "fare_amount", alpha=0.1)



## === cell 28
df_train = df_train[df_train["passenger_count"].fillna(1) > 0].copy()

df_train["distance_miles"] = df_train["distance"] * 0.621371

df_train["manhattan"] = (df_train.dropoff_latitude - df_train.pickup_latitude).abs() + (
    df_train.dropoff_longitude - df_train.pickup_longitude
).abs()

df_train["euclidean"] = np.sqrt(
    (df_train.dropoff_latitude - df_train.pickup_latitude) ** 2
    + (df_train.dropoff_longitude - df_train.pickup_longitude) ** 2
)

df_train["passenger_count"] = (
    df_train["passenger_count"].fillna(1).clip(lower=1, upper=6).astype(np.int16)
)

features = [
    "year",
    "month",
    "weekday",
    "hour",
    "distance",
    "distance_miles",
    "manhattan",
    "euclidean",
    "passenger_count",
]
X = df_train[features].values
y = df_train["fare_amount"].values



## === cell 29
df_test["year"] = df_test.pickup_datetime.apply(lambda t: t.year)
df_test["month"] = df_test.pickup_datetime.apply(lambda t: t.month)
df_test["weekday"] = df_test.pickup_datetime.apply(lambda t: t.weekday())
df_test["hour"] = df_test.pickup_datetime.apply(lambda t: t.hour)

coord_ok = (
    df_test["pickup_longitude"].between(-180, 180)
    & df_test["dropoff_longitude"].between(-180, 180)
    & df_test["pickup_latitude"].between(-90, 90)
    & df_test["dropoff_latitude"].between(-90, 90)
)

train_pickup_lon_med = df_train["pickup_longitude"].median()
train_pickup_lat_med = df_train["pickup_latitude"].median()
train_dropoff_lon_med = df_train["dropoff_longitude"].median()
train_dropoff_lat_med = df_train["dropoff_latitude"].median()

bad_mask = ~coord_ok
df_test.loc[bad_mask, "pickup_longitude"] = train_pickup_lon_med
df_test.loc[bad_mask, "pickup_latitude"] = train_pickup_lat_med
df_test.loc[bad_mask, "dropoff_longitude"] = train_dropoff_lon_med
df_test.loc[bad_mask, "dropoff_latitude"] = train_dropoff_lat_med

df_test["distance"] = distance(
    df_test.pickup_latitude,
    df_test.pickup_longitude,
    df_test.dropoff_latitude,
    df_test.dropoff_longitude,
)

df_test["distance_miles"] = df_test["distance"] * 0.621371

df_test["manhattan"] = (df_test.dropoff_latitude - df_test.pickup_latitude).abs() + (
    df_test.dropoff_longitude - df_test.pickup_longitude
).abs()

df_test["euclidean"] = np.sqrt(
    (df_test.dropoff_latitude - df_test.pickup_latitude) ** 2
    + (df_test.dropoff_longitude - df_test.pickup_longitude) ** 2
)

df_test["distance"] = df_test["distance"].fillna(df_train["distance"].median())
df_test["distance_miles"] = df_test["distance_miles"].fillna(
    df_train["distance_miles"].median()
)
df_test["manhattan"] = df_test["manhattan"].fillna(df_train["manhattan"].median())
df_test["euclidean"] = df_test["euclidean"].fillna(df_train["euclidean"].median())

df_test["passenger_count"] = (
    df_test["passenger_count"].fillna(1).clip(lower=1, upper=6).astype(np.int16)
)



## === cell 30
X_kaggle_test = df_test[features].values



## === cell 31
from sklearn.model_selection import train_test_split
import xgboost as xgb

sort_idx = np.argsort(df_train["pickup_datetime"].values)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_sorted, y_sorted, random_state=10, test_size=0.3, shuffle=False
)


def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 10,
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        dtrain=matrix_train,
        num_boost_round=500,
        early_stopping_rounds=100,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(X_train, X_test, y_train, y_test)

dtest = xgb.DMatrix(X_kaggle_test)
if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
else:
    prediction = model.predict(dtest)

prediction = np.clip(prediction, 0, None)



## === cell 32
submission = pd.DataFrame({"key": df_test["key"], "fare_amount": prediction.round(2)})
submission.to_csv("submission.csv", index=False)
submission
