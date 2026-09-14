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

4.70641

# 6. Current score

5.18808

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 192.47078) has done: 'Diagnosis: The crash happens because `pandas==2.2.3` removed the deprecated `.dt.week` and `.dt.weekofyear` accessors; they no longer exist on `DatetimeProperties`. The intended value is the ISO week number, which is now accessed via `.dt.isocalendar().week`.  
Patch summary: Update the failing feature engineering lines in cell 4 to compute `week` and `week_of_year` using `dt.isocalendar().week` (cast to integer to match prior dtype expectations) while leaving all other logic unchanged.  
Updated cells: Only cell 4 is modified.  
Compatibility notes for cell k+1: This preserves the same column names (`week`, `week_of_year`) and produces integer week numbers compatible with downstream code; cell 5 still uses the old accessors but is not changed per your constraint.  
Assumptions: The notebook expects ISO week numbering (the common behavior of the old accessors) and integer-typed week features.'
- What this solution (achieved 192.47121) has done: 'Diagnosis: Cell 5 crashes because `pandas` 2.x removed the deprecated `.dt.week` and `.dt.weekofyear` accessors, so `DatetimeProperties` no longer has those attributes. The train feature engineering in cell 4 already uses the supported `.dt.isocalendar().week.astype(int)` approach. We should mirror that for test to keep feature columns consistent and deterministic.

Patch summary: In cell 5, replace `test['pickup_datetime'].dt.week` and `.dt.weekofyear` with `test['pickup_datetime'].dt.isocalendar().week.astype(int)`. No other logic changes are made; all other feature columns remain identical.

Updated cells:'
- What this solution (achieved 192.43708) has done: 'Your current RMSE (192) suggests the submission rows are likely misaligned with the `key` IDs (predictions are being written onto `sample_submission`’s key order rather than `test`’s key order), which can catastrophically worsen RMSE even if the model is reasonable. I keep the modeling and features identical, but build the submission by merging predictions back to the sample submission on `key` to guarantee correct alignment. I also ensure the prediction array length matches the test set and clip negative fares to 0 (a valid fare domain constraint) to avoid extreme errors from occasional negative model outputs. These are minimal, evaluation-relevant changes that should move RMSE dramatically down toward your 4.706 target.'
- What this solution (achieved 5.64948) has done: 'Your RMSE is so large that it’s overwhelmingly likely the model is producing extreme outliers (often from bad coordinates / passenger_count in test) rather than just being “a bit weak”. To move toward the 4.706 target with minimal logic change, I (1) apply the exact same basic sanity filters you already use for train to the test set, and (2) fill any filtered-out test rows with a safe baseline (the mean fare from your filtered training data) so the submission keeps all required keys. I also ensure the weighted-ensemble predictions are clipped to a reasonable upper bound (200, matching your train cap) to prevent rare huge predictions from dominating RMSE. This keeps your core features and models identical while directly reducing catastrophic errors.'
- What this solution (achieved 5.55092) has done: 'Your current RMSE (5.64948) is worse than the target (4.70641), so we should cautiously improve without changing the overall approach. The biggest likely gain with minimal risk is to reduce systematic bias by adding a single strong, domain-relevant feature: straight-line (haversine) distance between pickup and dropoff, while keeping the same model types and ensemble scheme. This preserves your existing filtering, training loop structure, and submission construction, but gives all three models a much more predictive signal than raw coordinate deltas alone. I also keep the same clipping/baseline handling so you don’t regress due to outliers or invalid rows.'
- What this solution (achieved 5.2028) has done: 'Your current RMSE (5.55092) is still above the target (4.70641), so we should make a small, low-risk improvement that better matches the metric without changing the overall modeling approach. The biggest minimal win here is to (1) add a deterministic train/validation split to compute optimal nonnegative ensemble weights (still the same three models and same linear averaging idea) and then (2) use those fitted weights on the test predictions; this typically reduces systematic bias vs fixed 2/2/1 weights. I also set `random_state` for the RandomForest to reduce run-to-run variance and keep clipping/baseline logic unchanged to avoid outlier-driven RMSE blowups. All feature engineering and model classes remain the same; we’re only calibrating the ensemble weights using RMSE on a holdout.'
- What this solution (achieved 5.19261) has done: 'We make two minimal, evaluation-relevant improvements to reduce RMSE from 5.2028 toward 4.70641 without changing your model set or overall training approach. First, we standardize features for KNN only (using `StandardScaler` fit on train and applied to test/validation) because KNN is very sensitive to feature scale and this typically yields a small, safe RMSE gain. Second, we ensure there are no infinities from the haversine calculation (rare numeric edge cases) and keep the same clipping/baseline/key-merge submission logic unchanged. Everything else (features list, three models, ensemble-weight fitting via least squares, and CSV output) stays the same.'
- What this solution (achieved 5.19261) has done: 'Your current RMSE (5.19261) is still worse than the target (4.70641), so we should make a small, low-risk improvement without changing the model set, feature set, or training/ensemble logic. The most impactful minimal change here is to train on the same “clean” rows you already define (i.e., drop rows with NaNs/inf in the feature columns after haversine), because your current `X_train`/`y_train` still include NaNs from the haversine step and that can silently degrade or destabilize all three models. This keeps the exact same features, same three models, same holdout-weight fitting, and same submission construction; it just ensures the training matrix is valid and consistent with your test validity criteria. I also recompute the baseline mean on the cleaned training subset to keep fallback predictions consistent with what the models saw.'
- What this solution (achieved 5.18917) has done: 'Your current RMSE (5.19261) is above the target (4.70641), so we should make a small, safe improvement without changing your overall modeling/ensemble approach. The most likely low-risk gain is to add one strong, standard NYC taxi feature—absolute delta in longitude/latitude (signed/unsiged is fine; we keep abs) plus the existing haversine—by also including simple `abs_sum` interaction (Manhattan-like proxy) while keeping the same three models and the same holdout-fitted nonnegative ensemble weights. This is a minimal feature-only change that typically reduces RMSE a bit for linear/RF/KNN without altering training loops, model classes, or loss/metric semantics. Everything else (filters, scaling for KNN only, clipping/baseline handling, key-merge submission writing) remains unchanged.'
- What this solution (achieved 5.18808) has done: 'Your current RMSE (5.18917) is above the target (4.70641), so we should make a small, low-risk improvement that keeps your exact model set, features, and ensemble-weight fitting approach intact. The most likely gain per unit change is to tune only the KNN hyperparameters (still KNN, still same training/prediction flow) because KNN is very sensitive and currently uses defaults that are rarely optimal for this task. I fit `n_neighbors` and `weights` for KNN using the same existing train/validation split you already compute in cell 21, selecting the setting with the best validation RMSE, then use that tuned KNN for both validation weight fitting and test predictions. Everything else—filters, feature engineering, scaling for KNN only, nonnegative least-squares ensemble weights, clipping/baseline fallback, and key-merge submission writing—remains unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv", nrows=300_000)
test = pd.read_csv("../input/test.csv")



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt



## === cell 4
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype(int)
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week.astype(int)



## === cell 5
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype(int)
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week.astype(int)



## === cell 6
train.head()
train = train.dropna(how="any", axis="rows")



## === cell 7
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < 75)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < 75)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]
train = train.loc[train["passenger_count"] <= 8]



## === cell 8
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()

train["abs_sum_manhattan"] = train["abs_diff_longitude"] + train["abs_diff_latitude"]



## === cell 9
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

test["abs_sum_manhattan"] = test["abs_diff_longitude"] + test["abs_diff_latitude"]




## === cell 10
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


train["haversine_km"] = haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
test["haversine_km"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

for df in (train, test):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)



## === cell 11
train.head()



## === cell 12
train.head()



## === cell 13
sns.barplot(data=train, x="passenger_count", y="fare_amount")



## === cell 14
feature_names = [
    "hour",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "abs_sum_manhattan",  # Added minimal interaction feature to nudge RMSE down toward target.
    "haversine_km",
    "passenger_count",
]
feature_names



## === cell 15
label_name = "fare_amount"
label_name



## === cell 16
train_mask = train[feature_names + [label_name]].notna().all(axis=1)
X_train = train.loc[train_mask, feature_names]
y_train = train.loc[train_mask, label_name]

test_mask = (
    (test["pickup_longitude"] > -75)
    & (test["pickup_longitude"] < 75)
    & (test["pickup_latitude"] > 40)
    & (test["pickup_latitude"] < 45)
    & (test["dropoff_longitude"] > -75)
    & (test["dropoff_longitude"] < 75)
    & (test["dropoff_latitude"] > 40)
    & (test["dropoff_latitude"] < 45)
    & (test["passenger_count"] <= 8)
    & (test["passenger_count"] >= 0)
    & test[feature_names].notna().all(axis=1)
)

X_test_valid = test.loc[test_mask, feature_names]
valid_keys = test.loc[test_mask, "key"].values

baseline_fare = float(y_train.mean())



## === cell 17
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb



## === cell 18
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test_valid)



## === cell 19
X_tr, X_va, y_tr, y_va = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

scaler_knn_w = StandardScaler()
X_tr_knn = scaler_knn_w.fit_transform(X_tr)
X_va_knn = scaler_knn_w.transform(X_va)

best_knn_params = None
best_knn_rmse = np.inf
best_p_knn_va = None

for n_neighbors in (7, 9, 11, 15):
    for weights in ("distance", "uniform"):
        knn_tmp = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        knn_tmp.fit(X_tr_knn, y_tr)
        p_tmp = knn_tmp.predict(X_va_knn)
        rmse_tmp = float(np.sqrt(mean_squared_error(y_va.values, p_tmp)))
        if rmse_tmp < best_knn_rmse:
            best_knn_rmse = rmse_tmp
            best_knn_params = (n_neighbors, weights)
            best_p_knn_va = p_tmp

best_n_neighbors, best_weights = best_knn_params



## === cell 20
scaler_knn = StandardScaler()
X_train_knn = scaler_knn.fit_transform(X_train)
X_test_valid_knn = scaler_knn.transform(X_test_valid)

knr = KNeighborsRegressor(n_neighbors=best_n_neighbors, weights=best_weights)
knr.fit(X_train_knn, y_train)
knr_prediction = knr.predict(X_test_valid_knn)



## === cell 21
rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test_valid)



## === cell 22
regr_w = LinearRegression()
regr_w.fit(X_tr, y_tr)
p_lr_va = regr_w.predict(X_va)

rfr_w = RandomForestRegressor(random_state=42, n_jobs=-1)
rfr_w.fit(X_tr, y_tr)
p_rf_va = rfr_w.predict(X_va)

p_knn_va = best_p_knn_va

P = np.vstack([p_lr_va, p_rf_va, p_knn_va]).T.astype(float)
yva = y_va.values.astype(float)

w, *_ = np.linalg.lstsq(P, yva, rcond=None)
w = np.maximum(w, 0)
if float(w.sum()) == 0.0:
    w = np.array([0.4, 0.4, 0.2], dtype=float)  # safe fallback close to original 2/2/1
else:
    w = w / w.sum()

_ens_va = P @ w
_ = np.sqrt(mean_squared_error(yva, _ens_va))



## === cell 23
predictions_valid = (
    regr_prediction * w[0] + rfr_prediction * w[1] + knr_prediction * w[2]
)



## === cell 24
predictions_valid



## === cell 25
pred_df_valid = pd.DataFrame({"key": valid_keys, "fare_amount": predictions_valid})
pred_df_valid["fare_amount"] = pred_df_valid["fare_amount"].clip(lower=0, upper=200)

pred_df_invalid = pd.DataFrame(
    {"key": test.loc[~test_mask, "key"].values, "fare_amount": baseline_fare}
)

pred_df = pd.concat([pred_df_valid, pred_df_invalid], axis=0, ignore_index=True)

pred_df = pred_df.drop_duplicates(subset=["key"], keep="first")

submission = pd.read_csv("../input/sample_submission.csv")
submission = submission.drop(columns=["fare_amount"]).merge(
    pred_df, on="key", how="left"
)

if submission["fare_amount"].isna().any():
    missing = int(submission["fare_amount"].isna().sum())
    raise ValueError(
        f"Submission has {missing} missing predictions after key-merge; check key alignment."
    )



## === cell 26
submission.head()



## === cell 27
submission.to_csv("./simplenewyorktaxi.csv", index=False)
