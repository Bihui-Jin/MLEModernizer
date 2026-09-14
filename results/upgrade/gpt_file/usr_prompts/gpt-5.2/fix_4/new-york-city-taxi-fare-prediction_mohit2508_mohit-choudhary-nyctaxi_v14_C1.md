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

No external packages required in the script and installed.

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

3.617832926497142

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.33207) has done: 'I remove the unavailable `feather` dependency and stop relying on a cached feather file so the notebook can run in the Kaggle environment. I also fix the pandas `.str.split(' ', 1).str` usage that errors on newer pandas by switching to `.str.split(' ', n=1, expand=True)`, and ensure `df` is always defined by loading and cleaning train data in one place. To keep runtime under control while preserving the existing modeling approach, I train on a deterministic 1,000,000-row subset (as your original code intended) and keep the same feature engineering and models (LinearRegression + RandomForestRegressor). Finally, I always write a valid submission CSV named with a `.csv` suffix and the required `key,fare_amount` columns.'
- What this solution (achieved 5.31159) has done: 'You’re currently far above (worse than) the target RMSE, so we should increase predictive accuracy with minimal, low-risk tweaks that keep your overall approach (cleaning + engineered distance/diffs + LR and RF) intact. The biggest gain-per-change here is to (1) make the model robust to outliers by scaling the target via `log1p` for LinearRegression (same model, same features) and (2) nudge the RandomForest slightly toward better generalization by using more trees and enabling `max_features="sqrt"` (still the same RF approach, just safer hyperparameters). I also add a tiny post-processing step to clip negative fares (physically impossible) which typically reduces RMSE a bit without changing evaluation semantics. The script still write `submission.csv` with `key,fare_amount` and validate it against `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-whitegrid")

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

NROWS = 1_000_000

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=NROWS, low_memory=True)

df["pickup_datetime"] = df["pickup_datetime"].astype(str)

print("Train loaded shape:", df.shape)
print(df.head(2))



## === cell 2
df = df[df.passenger_count > 0]

df = df[df.dropoff_latitude != 0]
df = df[df.pickup_longitude != 0]
df = df[df.pickup_latitude != 0]
df = df[df.dropoff_longitude != 0]

df = df[df.fare_amount > 2]
df = df[df.fare_amount < 100]

df = df.dropna()

dt_split = df["pickup_datetime"].str.split(" ", n=1, expand=True)
df["year"] = dt_split[0].str[:4]
df["hour"] = dt_split[1].str[:2]

df = df.dropna(subset=["year", "hour"])
df[["year", "hour"]] = df[["year", "hour"]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["year", "hour"])

print("After basic cleaning shape:", df.shape)
print(df[["pickup_datetime", "year", "hour"]].head(3))




## === cell 3
def select_within_newYork(df_in, loc):
    return (
        (df_in.pickup_longitude >= loc[0])
        & (df_in.pickup_longitude <= loc[1])
        & (df_in.pickup_latitude >= loc[2])
        & (df_in.pickup_latitude <= loc[3])
        & (df_in.dropoff_longitude >= loc[0])
        & (df_in.dropoff_longitude <= loc[1])
        & (df_in.dropoff_latitude >= loc[2])
        & (df_in.dropoff_latitude <= loc[3])
    )


NYC = (-74.5, -72.8, 40.5, 41.8)
df = df[select_within_newYork(df, NYC)]

print("After NYC bounding box shape:", df.shape)




## === cell 4
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6367 * 2 * np.arcsin(np.sqrt(a)) * 0.62137


df["distance"] = haversine_np(
    df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude
)

print(df[["distance"]].describe())



## === cell 5
print("Co-relation b/w Fare and Distance")
print(st.pearsonr(df.distance, df.fare_amount))

df = df[df.distance <= 30]

print("After distance cutoff shape:", df.shape)



## === cell 6
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
con = (
    (df.distance < 30)
    & (df.distance > 0.5)
    & (df.fare_amount > 0)
    & (df.fare_amount < 200)
)
axs[0].scatter(df.loc[con, "fare_amount"], df.loc[con, "distance"], alpha=0.3)
axs[0].set_xlabel("Fare")
axs[0].set_ylabel("Distance")
axs[0].set_title("Distance vs Fare")
plt.close(fig)



## === cell 7
df["fare-bin"] = pd.cut(df["fare_amount"], bins=list(range(0, 50, 5))).astype(str)
df.loc[df["fare-bin"] == "nan", "fare-bin"] = "[45+]"
df.loc[df["fare-bin"] == "(5, 10]", "fare-bin"] = "(05, 10]"

df["diff_long"] = (df.dropoff_longitude - df.pickup_longitude).abs()
df["diff_lat"] = (df.dropoff_latitude - df.pickup_latitude).abs()

print(df[["diff_lat", "diff_long", "distance"]].head(3))



## === cell 8
print("corelation b/w Distance and Time of Day")
print(st.pearsonr(df.distance, df.hour))

print("corelation b/w Fare and Time of Day")
print(st.pearsonr(df.fare_amount, df.hour))



## === cell 9
test = pd.read_csv(TEST_PATH, low_memory=True)
test["pickup_datetime"] = test["pickup_datetime"].astype(str)

test["diff_lat"] = (test.dropoff_latitude - test.pickup_latitude).abs()
test["diff_long"] = (test.dropoff_longitude - test.pickup_longitude).abs()
test["distance"] = haversine_np(
    test.pickup_longitude,
    test.pickup_latitude,
    test.dropoff_longitude,
    test.dropoff_latitude,
)

dt_split_t = test["pickup_datetime"].str.split(" ", n=1, expand=True)
test["year"] = dt_split_t[0].str[:4]
test["hour"] = dt_split_t[1].str[:2]
test[["year", "hour"]] = test[["year", "hour"]].apply(pd.to_numeric, errors="coerce")

test_id = list(test["key"].values)

print("Test loaded shape:", test.shape)
print(test.head(2))



## === cell 10
lr = LinearRegression()

lr_features = [
    "diff_lat",
    "diff_long",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "passenger_count",
    "year",
    "hour",
]

p = lr.fit(df[lr_features], np.log1p(df["fare_amount"]))

print("Intercept", round(lr.intercept_, 4))
print(
    "Lat diff coef: ",
    round(lr.coef_[0], 4),
    "\tLong diff coef:",
    round(lr.coef_[1], 4),
    "\t Pikcup Latitude  coef",
    round(lr.coef_[2], 4),
    "\t Pikcup Longitude  coef",
    round(lr.coef_[3], 4),
    "\t Dropoff Latitude  coef",
    round(lr.coef_[4], 4),
    "\t Dropoff Longitude  coef",
    round(lr.coef_[5], 4),
    "\tDistance coef:",
    round(lr.coef_[6], 4),
)



## === cell 11
preds_lr = np.expm1(lr.predict(test[lr_features]))
preds_lr = np.clip(preds_lr, 0.0, None)

sub_lr = pd.DataFrame({"key": test_id, "fare_amount": preds_lr})
sub_lr.to_csv("output_lr.csv", index=False)

print("Wrote output_lr.csv with shape:", sub_lr.shape)
print(sub_lr.head(2))



## === cell 12
X = df.drop(
    ["key", "fare_amount", "pickup_datetime", "fare-bin"], axis=1, errors="ignore"
)
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

lr_eval = LinearRegression()
lr_eval.fit(X_train[lr_features], np.log1p(y_train))
y_val_pred = np.expm1(lr_eval.predict(X_val[lr_features]))
y_val_pred = np.clip(y_val_pred, 0.0, None)

lrmse = np.sqrt(metrics.mean_squared_error(y_val, y_val_pred))
print("LinearRegression (log1p target) validation RMSE:", lrmse)



## === cell 13
random_forest = rf(
    n_estimators=50,
    max_depth=12,
    max_features="sqrt",
    oob_score=True,
    bootstrap=True,
    verbose=0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf_features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "diff_lat",
    "diff_long",
    "passenger_count",
    "year",
    "hour",
]

random_forest.fit(df[rf_features], df["fare_amount"])
print("RandomForest fitted. OOB score available:", hasattr(random_forest, "oob_score_"))
if hasattr(random_forest, "oob_score_"):
    print("RandomForest oob_score_:", random_forest.oob_score_)



## === cell 14
test_mask_nyc = select_within_newYork(test, NYC)
test_mask_dist = test["distance"].notna() & (test["distance"] <= 30)
test_mask_feat = test[rf_features].notna().all(axis=1)
test_mask = test_mask_nyc & test_mask_dist & test_mask_feat

predictedFare = np.array(preds_lr, copy=True)  # fallback baseline for all rows
pred_rf_part = random_forest.predict(test.loc[test_mask, rf_features])
predictedFare[test_mask.values] = pred_rf_part
predictedFare = np.clip(predictedFare, 0.0, None)

sub_rf = pd.DataFrame({"key": test_id, "fare_amount": predictedFare})
sub_rf.to_csv("output_rf.csv", index=False)

print("Wrote output_rf.csv with shape:", sub_rf.shape)
print("RF used on rows:", int(test_mask.sum()), "out of", test.shape[0])
print(sub_rf.head(2))



## === cell 15
submission = sub_rf.copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)

sample = pd.read_csv(SAMPLE_SUB_PATH)
assert list(submission.columns) == list(
    sample.columns
), "Submission columns do not match sample_submission.csv"
assert (
    submission.shape[0] == sample.shape[0]
), "Submission row count does not match sample_submission.csv"
print("Submission format validated vs sample_submission.csv")
