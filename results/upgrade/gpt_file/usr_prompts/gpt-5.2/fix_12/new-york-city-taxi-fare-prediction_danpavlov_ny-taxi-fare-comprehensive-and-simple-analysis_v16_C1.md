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

3.623

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.33786) has done: 'I fix the XGBoost inference crash caused by `best_ntree_limit` no longer existing in xgboost 2.x by switching to the supported `iteration_range` API while keeping the same training approach (early stopping on a held-out set). I also remove notebook-only `%matplotlib inline` so the script runs as a plain Python program, and replace deprecated `sns.distplot` with `sns.histplot` to avoid runtime errors on newer seaborn. Finally, I ensure we always write a valid submission CSV with the required columns (`key,fare_amount`) and a `.csv` suffix.'
- What this solution (achieved 6.51135) has done: 'You’re far from the target (7.34 vs 3.623 RMSE; lower is better), so we need small, legitimate changes that materially improve generalization without changing the overall approach (same feature set idea + XGBoost training + early stopping). The biggest score drag is the extremely slow per-row `iterrows()` distance engineering: it limits you to only 100k rows and also risks inconsistent feature creation; vectorizing the exact same Haversine math lets you train on more rows within the same time budget, which should move RMSE down toward your target. I also keep the same model objective and early-stopping workflow, but add a fixed `random_state` and a slightly stronger parameter set (depth/eta/subsampling) that is standard for this competition and should improve RMSE without changing the “XGBoost regressor with early stopping” core logic. Finally, I stop rounding predictions (rounding hurts RMSE) and keep the submission format identical.'
- What this solution (achieved 5.33913) has done: 'I fix the crash in the distance-to-landmark feature engineering by making `_haversine_km` accept scalar landmark coordinates (floats) as well as pandas Series/arrays, which unblock cells 17–19 and allow the pipeline to run end-to-end. I also make the column drops robust with `errors="ignore"` so the script won’t fail if a column is missing due to upstream changes. To move RMSE down toward your 3.623 target (current 6.51135; lower is better) without changing the modeling approach, I increase the training sample size modestly (same exact features + same XGBoost training/early-stopping workflow) so the model can learn better within the time limit. Finally, I ensure the submission is written as a valid `.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 4.002) has done: 'Your current RMSE (5.339) is still far above the target (3.623), so we should make small, legitimate improvements that keep the same overall pipeline (feature engineering + XGBoost with early stopping) but reduce generalization error. The biggest gain with minimal semantic change is to align the train/test preprocessing: apply the same geographic outlier filters to the test set as you do to train (then predict and merge back by `key` so the submission stays complete), and ensure passenger_count is clean/consistent. Additionally, train XGBoost using both train and validation in `evals` so early stopping actually monitors overfitting (still the same early-stopping workflow), and use the best iteration for predictions (already done). Finally, keep the submission strictly aligned to `sample_submission` keys and fill any filtered-out test rows with a reasonable fallback (training median), ensuring a valid 9914-row CSV.'
- What this solution (achieved 7.1575) has done: 'We’re still above the target (4.002 vs 3.623 RMSE; lower is better), so the smallest reliable way to move closer is to improve data cleanliness and reduce label noise without changing your feature set or model/training workflow. I keep the same engineered features and the same XGBoost-with-early-stopping approach, but add a couple of standard NYC Taxi Fare cleaning filters (drop extreme fares and implausibly long trips) that typically reduce RMSE for this competition. I also ensure `passenger_count` is consistently numeric in both train/test (your train is typed but test isn’t), and I use the already-computed `test_pred` (full test) to predict for all rows rather than filtering test at inference time (filtering test can create a distribution mismatch and relies on fallback values). Submission format, path, and column names remain identical and a valid `.csv` is always written.'
- What this solution (achieved 6.27174) has done: 'The timeout is dominated by reading/processing 8M training rows with pandas plus several expensive EDA plots/correlations that scale poorly, and by extra DataFrame copies created during feature engineering. I keep the same feature set and XGBoost training approach, but speed up I/O by selecting only needed columns, using faster CSV parsing options, and pushing filters earlier to shrink intermediate frames sooner. I also replace slow `pd.concat(...).min(axis=1)` patterns with equivalent vectorized `np.minimum`, avoid repeated `drop` calls, and remove the plotting/correlation cells (they don’t affect predictions). Finally, I set XGBoost threading explicitly and build DMatrix with `nthread` to reduce overhead while preserving identical training semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb

np.random.seed(42)



## === cell 1
print(os.listdir("../input")[:50])



## === cell 2
test_types = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",  # read as float then clean/cast
}
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test = pd.read_csv(
    "../input/test.csv",
    usecols=test_usecols,
    dtype=test_types,
    engine="c",
    low_memory=False,
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
    "passenger_count": "float32",
}



## === cell 5
train_usecols = test_usecols + ["fare_amount"]
train = pd.read_csv(
    "../input/train.csv",
    nrows=8_000_000,
    usecols=train_usecols,
    dtype=types,
    engine="c",
    low_memory=False,
    memory_map=True,
)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
pass



## === cell 9
pass



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(
    subset=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    inplace=True,
)



## === cell 12
train["passenger_count"] = pd.to_numeric(
    train["passenger_count"], errors="coerce"
).fillna(1.0)
train["passenger_count"] = (
    train["passenger_count"].clip(lower=1, upper=9).astype("uint8")
)

plon = train["pickup_longitude"].to_numpy(copy=False)
dlon = train["dropoff_longitude"].to_numpy(copy=False)
plat = train["pickup_latitude"].to_numpy(copy=False)
dlat = train["dropoff_latitude"].to_numpy(copy=False)
fare = train["fare_amount"].to_numpy(copy=False)
pc = train["passenger_count"].to_numpy(copy=False)

m = (
    (fare > 0.0)
    & (fare < 250.0)
    & (plon < -72)
    & (dlon < -72)
    & (plat > 40)
    & (plat < 44)
    & (dlat > 40)
    & (dlat < 44)
    & (pc >= 1)
    & (pc <= 6)
)
same_loc = (plon == dlon) & (plat == dlat)
m = m & (~same_loc)

train = train.loc[m]



## === cell 13
train.describe()




## === cell 14
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def quick_dist_calc(df):
    df["distance"] = _haversine_km(
        df["pickup_latitude"].to_numpy(copy=False),
        df["pickup_longitude"].to_numpy(copy=False),
        df["dropoff_latitude"].to_numpy(copy=False),
        df["dropoff_longitude"].to_numpy(copy=False),
    ).astype("float32")




## === cell 15
def quick_dist_calc_loc(df, c1, c2, cname):
    df[cname + "_pickup_dist"] = _haversine_km(
        df["pickup_latitude"].to_numpy(copy=False),
        df["pickup_longitude"].to_numpy(copy=False),
        c1,
        c2,
    ).astype("float32")
    df[cname + "_dropoff_dist"] = _haversine_km(
        df["dropoff_latitude"].to_numpy(copy=False),
        df["dropoff_longitude"].to_numpy(copy=False),
        c1,
        c2,
    ).astype("float32")




## === cell 16
quick_dist_calc(train)
quick_dist_calc(test)

dist = train["distance"].to_numpy(copy=False)
train = train.loc[(dist > 0.0) & (dist < 60.0)]



## === cell 17
jfk_airport = (-73.785193, 40.645972)
laguardia_airport = (-73.872925, 40.773335)
newark_airport = (-74.184156, 40.692764)
manhattan = (-73.983132, 40.759006)

quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(train, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(train, manhattan[1], manhattan[0], "manhattan")

quick_dist_calc_loc(test, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    test, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(test, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(test, manhattan[1], manhattan[0], "manhattan")



## === cell 18
train_jfk_pu = train["jfk_airport_pickup_dist"].to_numpy(copy=False)
train_jfk_do = train["jfk_airport_dropoff_dist"].to_numpy(copy=False)
train_lga_pu = train["laguardia_airport_pickup_dist"].to_numpy(copy=False)
train_lga_do = train["laguardia_airport_dropoff_dist"].to_numpy(copy=False)
train_ewr_pu = train["newark_airport_pickup_dist"].to_numpy(copy=False)
train_ewr_do = train["newark_airport_dropoff_dist"].to_numpy(copy=False)
train_mht_pu = train["manhattan_pickup_dist"].to_numpy(copy=False)
train_mht_do = train["manhattan_dropoff_dist"].to_numpy(copy=False)

train["jfk_distance"] = np.minimum(train_jfk_pu, train_jfk_do).astype(
    "float32", copy=False
)
train["laguardia_distance"] = np.minimum(train_lga_pu, train_lga_do).astype(
    "float32", copy=False
)
train["newark_distance"] = np.minimum(train_ewr_pu, train_ewr_do).astype(
    "float32", copy=False
)
train["manhattan_distance"] = np.minimum(train_mht_pu, train_mht_do).astype(
    "float32", copy=False
)

test_jfk_pu = test["jfk_airport_pickup_dist"].to_numpy(copy=False)
test_jfk_do = test["jfk_airport_dropoff_dist"].to_numpy(copy=False)
test_lga_pu = test["laguardia_airport_pickup_dist"].to_numpy(copy=False)
test_lga_do = test["laguardia_airport_dropoff_dist"].to_numpy(copy=False)
test_ewr_pu = test["newark_airport_pickup_dist"].to_numpy(copy=False)
test_ewr_do = test["newark_airport_dropoff_dist"].to_numpy(copy=False)
test_mht_pu = test["manhattan_pickup_dist"].to_numpy(copy=False)
test_mht_do = test["manhattan_dropoff_dist"].to_numpy(copy=False)

test["jfk_distance"] = np.minimum(test_jfk_pu, test_jfk_do).astype(
    "float32", copy=False
)
test["laguardia_distance"] = np.minimum(test_lga_pu, test_lga_do).astype(
    "float32", copy=False
)
test["newark_distance"] = np.minimum(test_ewr_pu, test_ewr_do).astype(
    "float32", copy=False
)
test["manhattan_distance"] = np.minimum(test_mht_pu, test_mht_do).astype(
    "float32", copy=False
)



## === cell 19
drop_cols = [
    "jfk_airport_pickup_dist",
    "jfk_airport_dropoff_dist",
    "laguardia_airport_pickup_dist",
    "laguardia_airport_dropoff_dist",
    "newark_airport_pickup_dist",
    "newark_airport_dropoff_dist",
    "manhattan_pickup_dist",
    "manhattan_dropoff_dist",
]
train.drop(columns=drop_cols, inplace=True, errors="ignore")
test.drop(columns=drop_cols, inplace=True, errors="ignore")



## === cell 20
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, errors="coerce", cache=True
).dt.tz_convert(None)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="coerce", cache=True
).dt.tz_convert(None)

train.dropna(subset=["pickup_datetime"], inplace=True)
test.dropna(subset=["pickup_datetime"], inplace=True)



## === cell 21
dt = train["pickup_datetime"].dt
train["hour"] = dt.hour.astype("uint8")
train["weekday"] = dt.weekday.astype("uint8")
train["month"] = dt.month.astype("uint8")
train["year"] = dt.year.astype("uint16")

dt2 = test["pickup_datetime"].dt
test["hour"] = dt2.hour.astype("uint8")
test["weekday"] = dt2.weekday.astype("uint8")
test["month"] = dt2.month.astype("uint8")
test["year"] = dt2.year.astype("uint16")

test["passenger_count"] = pd.to_numeric(
    test["passenger_count"], errors="coerce"
).fillna(1.0)
test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=9).astype("uint8")



## === cell 22
dist64 = train["distance"].to_numpy(dtype="float64", copy=False)
fare64 = train["fare_amount"].to_numpy(dtype="float64", copy=False)
fare_per_km = fare64 / np.clip(dist64, 0.1, None)

m1 = (fare64 >= 2.5) | (dist64 >= 0.2)
m2 = (fare_per_km > 0.2) & (fare_per_km < 80.0)
m3 = ~((dist64 < 0.5) & (fare64 > 50.0))
train = train.loc[m1 & m2 & m3]



## === cell 23
train.head()



## === cell 24
test.head()



## === cell 25
pass



## === cell 26
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 27
X.head()



## === cell 28
y.head()



## === cell 29
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 30
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 31
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 32
pass



## === cell 33
pass



## === cell 34
pass



## === cell 35
pass



## === cell 36
pass




## === cell 37
def XGBoost(X_train, X_test, y_train, y_test):
    Xtr = np.asarray(X_train.to_numpy(copy=False), order="C", dtype=np.float32)
    Xva = np.asarray(X_test.to_numpy(copy=False), order="C", dtype=np.float32)
    ytr = np.asarray(y_train.to_numpy(copy=False), order="C", dtype=np.float32)
    yva = np.asarray(y_test.to_numpy(copy=False), order="C", dtype=np.float32)

    dtrain = xgb.DMatrix(Xtr, label=ytr, nthread=8)
    dvalid = xgb.DMatrix(Xva, label=yva, nthread=8)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "lambda": 1.5,
        "alpha": 0.0,
        "seed": 42,
        "tree_method": "hist",
        "nthread": 8,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4000,
        early_stopping_rounds=100,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )




## === cell 38
xgbm = XGBoost(X_train, X_test, y_train, y_test)



## === cell 39
dtest_submit = xgb.DMatrix(
    np.asarray(test_pred.to_numpy(copy=False), order="C", dtype=np.float32),
    nthread=8,
)

if getattr(xgbm, "best_iteration", None) is not None:
    XGBPredictions = xgbm.predict(
        dtest_submit, iteration_range=(0, xgbm.best_iteration + 1)
    )
else:
    XGBPredictions = xgbm.predict(dtest_submit)



## === cell 40
XGBPredictions



## === cell 41
XGBPredictions = np.clip(XGBPredictions, 0.0, None)

fallback_fare = float(np.median(y.to_numpy(copy=False)))

sample_sub = pd.read_csv("../input/sample_submission.csv")
submission = sample_sub.copy()

pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": XGBPredictions})
submission = submission.merge(pred_df, on="key", how="left", suffixes=("", "_pred"))
submission["fare_amount"] = submission["fare_amount"].astype("float64")
submission["fare_amount"].fillna(fallback_fare, inplace=True)
submission["fare_amount"] = submission["fare_amount"].clip(lower=0.0)

submission_path = "XGBSubmission23082018_2M.csv"
submission.to_csv(submission_path, index=False)
print("Wrote submission:", os.path.abspath(submission_path), "rows:", len(submission))
print("Any NaNs left:", int(submission["fare_amount"].isna().sum()))
