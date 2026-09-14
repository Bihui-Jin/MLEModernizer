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

3.9611

# 6. Current score

7.86691

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.84879) has done: 'I fix the main score issue caused by inconsistent one-hot encoding between train and test years by fitting the year dummies on the combined year set, so the feature columns align exactly. I also remove the rounding of predictions (rounding adds avoidable error under RMSE) while keeping the same XGBoost training approach and parameters. Finally, I modernize the XGBoost objective to `reg:squarederror` (same regression semantics, avoids deprecated behavior) and keep everything else intact so the pipeline remains stable and produces `finaloutput.csv` in the required format.'
- What this solution (achieved 6.01473) has done: 'Your current score (5.84879 RMSE) is worse than the target (3.9611), so we should make small, safe improvements that don’t change the overall XGBoost approach. The biggest issue is that `train_test_split` is random and you are using early stopping on a single split without a fixed seed, which can land on a suboptimal number of trees; I make the split deterministic and use a fixed `seed` in XGBoost for stability and typically slightly better generalization. I also add a minimal, standard NYC feature-cleaning step to drop trips with zero/near-zero distance and cap extreme distances; this preserves your core features but removes noise that commonly hurts RMSE. Finally, I compute distance and datetime-derived features vectorized (same semantics) to avoid slow Python loops and keep runtime within limits.'
- What this solution (achieved 5.93382) has done: 'Your score is worse than the target (RMSE 6.01 vs 3.96, lower is better), so the smallest reliable improvement is to stop using a random holdout split with early stopping to pick the number of trees (which can underfit and wastes data). I keep the same XGBoost training approach/params, but first determine the best boosting round via a deterministic CV on the training sample, then retrain once on 100% of that cleaned sample with that fixed number of rounds (no early stopping). I also add one minimal, standard cleaning rule that directly reduces RMSE for this competition: remove rows where `passenger_count==0` (noise/outliers), while keeping your existing filters and features unchanged. The output remains `finaloutput.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 5.92147) has done: 'I make two minimal, score-relevant fixes that typically reduce RMSE for this competition without changing your core XGBoost approach: (1) fix a subtle label/feature misalignment bug introduced by filtering `passenger_count>0` after `dropna/drop outliers` (we apply the same row mask to both `feat` and `df` before extracting `fare_amount`), and (2) add standard but small datetime features (`hour`, `weekday`, `month`) that preserve your existing feature extraction style and usually improve generalization. I also keep your deterministic CV-based choice of boosting rounds and the same objective/metric, and ensure the submission CSV is still written as `finaloutput.csv` with `key,fare_amount`. These changes are designed to improve score toward the target (lower RMSE) with minimal disruption and stable runtime.'
- What this solution (achieved 7.23569) has done: 'Your current RMSE (5.92147) is worse than the target (3.9611), so we should make small, safe improvements that usually reduce error without changing the overall XGBoost approach. The biggest win with minimal disruption is to add a couple of standard NYC Taxi engineered features (absolute lat/lon deltas and simple Manhattan distance, plus a “distance per passenger” interaction) while keeping the same model type, objective, and CV-based selection of boosting rounds. I also add a tiny amount of additional, competition-standard data cleaning (remove extreme low fares and cap passenger_count to >=1) to reduce noise/outliers that inflate RMSE. Finally, I ensure train/test feature columns align exactly after feature engineering so the submission remains valid and stable.'
- What this solution (achieved 7.23693) has done: 'We keep your exact XGBoost approach and feature set, but fix two score-critical data issues that inflate RMSE: (1) remove the duplicated `labels.csv`-style leakage/contamination risk by ensuring we only train on clean `train.csv` rows with valid coordinate bounds and realistic fares/passenger counts, and (2) add one standard NYC taxi cleaning rule that improves RMSE without changing modeling logic—dropping obviously-bad trips with identical pickup/dropoff coordinates (near-zero displacement) which otherwise behave like noisy labels even after the haversine filter. We also make XGBoost consume data with `missing=np.nan` explicitly and set a conservative `min_child_weight`/`subsample`/`colsample_bytree` only if needed—but since core logic must be preserved, we not change your hyperparameters beyond adding deterministic defaults already implied by XGBoost; the main gain comes from cleaner training rows. The pipeline still run end-to-end within time, and still write `finaloutput.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 8.2542) has done: 'Your current RMSE (7.23693, lower is better) is far worse than the target (3.9611), and the biggest minimal-change lever in this competition is training-data quality: the model is being trained on many mislabeled/outlier trips that dominate RMSE. I keep your exact XGBoost approach (same model family, same CV-based boosting-round selection, same features) but add a few standard NYC Taxi cleaning filters that remove obvious label noise (e.g., fare vs distance inconsistencies, zero passengers, and extreme per‑km pricing) while keeping runtime under control. I also ensure the same filters are applied consistently before label extraction (to avoid subtle misalignment) and keep your submission writing exactly as `finaloutput.csv` with `key,fare_amount`. These changes are directly aimed at reducing RMSE toward the target without changing the core training/prediction semantics.'
- What this solution (achieved 7.86691) has done: 'You’re far above the target RMSE (8.2542 vs 3.9611; lower is better), so we should make small, standard data-quality fixes that reliably reduce error without changing your core XGBoost setup. The biggest likely issue is that you’re training on a million-row sample that still contains many outliers and “bad geography” (especially airport/long trips and extreme pricing) that your current filters don’t fully remove; tightening these filters modestly usually drops RMSE a lot on this competition. I keep the same features and the same XGBoost training approach (CV to pick boosting rounds, then train once) and only adjust the cleaning thresholds and add one minimal, competition-standard coordinate sanity filter for test as well (to prevent pathological test rows from producing huge errors). Submission writing stays identical as `finaloutput.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
import xgboost
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=1000000)



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
testkey = test.key



## === cell 4
df = df.dropna(how="any", axis="rows")



## === cell 5
len(df)



## === cell 6
df.head()



## === cell 7
l = df[
    (df.pickup_latitude > 42.0)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.0)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index



## === cell 8
df = df.drop(l, axis=0)



## === cell 9
z = df[
    (df.fare_amount > 300.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 0.0)
].index



## === cell 10
df = df.drop(z, axis=0)



## === cell 11
len(df)




## === cell 12
def distlatlong(lon1, lat1, lon2, lat2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2)) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6373.0 * c
    return distance




## === cell 13
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6373.0 * c


df["dist"] = haversine_km(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)
test["dist"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)



## === cell 14
same_loc = (df["pickup_longitude"].sub(df["dropoff_longitude"]).abs() < 1e-6) & (
    df["pickup_latitude"].sub(df["dropoff_latitude"]).abs() < 1e-6
)
df = df.loc[~same_loc].copy()



## === cell 15
df = df[(df["dist"] > 0.0) & (df["dist"] < 60.0)].copy()

test["dist"] = test["dist"].clip(lower=0.0, upper=60.0)



## === cell 16
sns.boxplot(x=df.dist)



## === cell 17
test.head()



## === cell 18
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 19
df.info()



## === cell 20
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(int)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(int)



## === cell 21
df["year"] = df["pickup_datetime"].dt.year
test["year"] = test["pickup_datetime"].dt.year



## === cell 22
df["day"] = df["pickup_datetime"].dt.day
test["day"] = test["pickup_datetime"].dt.day



## === cell 23
df["hour"] = df["pickup_datetime"].dt.hour
test["hour"] = test["pickup_datetime"].dt.hour
df["weekday"] = df["pickup_datetime"].dt.weekday
test["weekday"] = test["pickup_datetime"].dt.weekday
df["month"] = df["pickup_datetime"].dt.month
test["month"] = test["pickup_datetime"].dt.month



## === cell 24
for _df in (df, test):
    _df["abs_lon_diff"] = (_df["pickup_longitude"] - _df["dropoff_longitude"]).abs()
    _df["abs_lat_diff"] = (_df["pickup_latitude"] - _df["dropoff_latitude"]).abs()
    _df["manhattan_approx"] = _df["abs_lon_diff"] + _df["abs_lat_diff"]
    _df["dist_per_passenger"] = _df["dist"] / (
        _df["passenger_count"].clip(lower=1).astype("float64")
    )



## === cell 25
df.head()



## === cell 26
test.head()



## === cell 27
feat = df.drop(
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
test_feat = test.drop(
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



## === cell 28
test_feat.year.unique()



## === cell 29
all_years = pd.Index(
    pd.concat([feat["year"], test_feat["year"]], axis=0).unique()
).sort_values()
year_dummies_train = pd.get_dummies(feat["year"]).reindex(
    columns=all_years, fill_value=0
)
year_dummies_test = pd.get_dummies(test_feat["year"]).reindex(
    columns=all_years, fill_value=0
)

feat = pd.concat([feat.drop("year", axis=1), year_dummies_train], axis=1)
test_feat = pd.concat([test_feat.drop("year", axis=1), year_dummies_test], axis=1)



## === cell 30
feat.head()



## === cell 31
test_feat.head()



## === cell 32
mask = (
    (feat["passenger_count"] >= 1)
    & (feat["fare_amount"] >= 2.5)
    & (feat["fare_amount"] <= 200.0)
)

dist_for_ratio = feat["dist"].clip(lower=0.05).astype("float64")
fare_per_km = feat["fare_amount"].astype("float64") / dist_for_ratio

mask &= (fare_per_km >= 1.0) & (fare_per_km <= 20.0)

mask &= ~((feat["dist"] > 25.0) & (feat["fare_amount"] < 20.0))
mask &= ~((feat["dist"] < 0.3) & (feat["fare_amount"] > 40.0))

mask &= feat["passenger_count"] <= 6

feat = feat.loc[mask].copy()
df = df.loc[mask].copy()



## === cell 33
label = df["fare_amount"].astype("float64")



## === cell 34
feat = feat.drop("fare_amount", axis=1)



## === cell 35
feat, test_feat = feat.align(test_feat, join="left", axis=1, fill_value=0)



## === cell 36
xtr, xts, ytr, yts = train_test_split(feat, label, test_size=0.2, random_state=42)



## === cell 37
xgbfinaltest = xgboost.DMatrix(test_feat, missing=np.nan)



## === cell 38
params = {"eval_metric": "rmse", "objective": "reg:squarederror", "seed": 42}



## === cell 39
dtrain_full = xgboost.DMatrix(feat, label, missing=np.nan)

cv = xgboost.cv(
    params=params,
    dtrain=dtrain_full,
    num_boost_round=150,
    nfold=5,
    seed=42,
    shuffle=True,
    early_stopping_rounds=20,
    verbose_eval=False,
)
best_num_boost_round = int(cv.shape[0])

xgbmodel = xgboost.train(
    params=params,
    dtrain=dtrain_full,
    num_boost_round=best_num_boost_round,
)



## === cell 40
pred = xgbmodel.predict(xgbfinaltest)



## === cell 41
pred = pred



## === cell 42
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## === cell 43
finalset = finalset[["key", "fare_amount"]]



## === cell 44
finalset.head()



## === cell 45
finalset.to_csv("finaloutput.csv", index=False)
