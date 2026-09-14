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

3.8987

# 6. Current score

6.50284

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.1869) has done: 'Diagnosis: Cell 18 crashes because `np.object` was removed in NumPy 1.26; comparing dtypes to `np.object` raises `AttributeError`. The intended behavior is to identify columns with Python `object` dtype, which can be done by comparing to the builtin `object` (or `'object'`) without changing semantics.  
Patch summary: Replace `np.object` with builtin `object` in the dtype comparison; keep variable names and downstream outputs identical.  
Updated cells: Only cell 18 is modified.  
Compatibility notes for cell k+1: `categoricals` remains the same type (an Index of column names) and `train` is unchanged, so cell 19 continues to work as written.  
Assumptions: The goal is specifically to detect `object`-dtype columns (e.g., `pickup_datetime`) as originally intended.'
- What this solution (achieved 5.17415) has done: 'Diagnosis: Cell 52 crashes because `LassoCV(max_iter=5e4, ...)` passes a float (`50000.0`) to `max_iter`, but scikit-learn 1.2 validates `max_iter` must be an integer. This is triggered by using scientific notation (`5e4`) which produces a float in Python.  
Patch summary: In cell 52, change `max_iter=5e4` to `max_iter=int(5e4)` (or `50000`) to satisfy scikit-learn’s parameter type constraint while keeping the same iteration count and identical modeling semantics.  
Updated cells: Only cell 52 is modified.  
Compatibility notes for cell k+1: Variables `la`, `la_rmse`, and printed outputs remain the same types/meaning, so downstream cells are unaffected.  
Assumptions: scikit-learn’s strict parameter validation is enabled (as in the shown traceback) and we should preserve the same numeric value for `max_iter`.'
- What this solution (achieved 5.17798) has done: 'Diagnosis: Cell 53 crashes because scikit-learn 1.2+ enforces parameter validation and `ElasticNetCV(max_iter=1e4)` passes a float (`10000.0`) instead of the required positive integer. This triggers `InvalidParameterError` during `.fit()`.  
Patch summary: In cell 53, cast `max_iter` to an `int` (matching how `LassoCV` is already configured in cell 52) to satisfy the API constraint without changing the model, data, or training semantics.  
Updated cells: Only cell 53 is modified.  
Compatibility notes for cell k+1: The variable `en` is still created as an `ElasticNetCV` instance and `en_rmse` is computed the same way, so downstream cells (including cell 54) are unaffected.  
Assumptions: No other hidden errors exist past this parameter-validation issue; keeping the exact same hyperparameters and CV defaults is acceptable.'
- What this solution (achieved 5.14457) has done: 'Your current pipeline is underperforming mainly because the engineered “harvesine/km” feature is effectively broken (it uses the dropoff longitude twice and swaps the diffs), which weakens the features used by the RandomForest that generates the submission. I make a minimal, targeted fix to the haversine computation (no change to the modeling approach) so the distance feature becomes meaningful, which should reduce RMSE toward your target. I also fix a small bug in `date_extraction` where you accidentally assign `None` by using `inplace=True` during a drop; this preserves the intended semantics while ensuring the function is correct and stable. Finally, I keep the I/O paths and submission format identical and ensure the code runs end-to-end in a script environment (no notebook-only magic).'
- What this solution (achieved 5.13799) has done: 'Your current score (5.14457, lower is better) is still well above the target (3.8987), so we should make a small, legitimate improvement that keeps the same overall pipeline. The biggest minimal gain here is to fix a train/test feature mismatch: you train the RandomForest on `train_1` columns (which still include `key`) but you predict on `test_1` after dropping `key`, meaning the model is trained with an ID-like column that is absent at inference (and also useless/noisy). I align the feature set by dropping `key` from `train_1` as well and (to prevent silent column-order issues) explicitly reindex `test_1` to the exact training column order before predicting; this preserves the same model and training approach while improving correctness and typically reducing RMSE. All paths and the submission format/filename remain unchanged.'
- What this solution (achieved 5.19463) has done: 'Your score (5.13799, lower is better) is still far from the target (3.8987), so we should make a small, legitimate improvement without changing the overall approach. The biggest low-risk gain here is to remove extreme/invalid target outliers and obvious bad geocoordinates from the 1M-row training sample; NYC Taxi Fare baselines typically improve a lot with simple, standard data cleaning while keeping the same features and models. I add a minimal filter for reasonable fare amounts and NYC-like latitude/longitude bounds (applied only to train, not test) and keep everything else (feature engineering, models, submission format) identical. This should reduce RMSE meaningfully while preserving your pipeline and runtime constraints.'
- What this solution (achieved 5.96533) has done: 'Your current score (5.19463, lower is better) is still far from the target (3.8987), so we make a small, legitimate improvement that keeps the same overall pipeline and model. The biggest low-risk gain is to clean the training sample more appropriately: your current “drop any row with a 0 anywhere” removes many valid rides (e.g., midnight hour/month) and also fails to remove common invalid coordinate patterns (0,0 or identical pickup/dropoff). I replace that blanket zero-row removal with standard NYC taxi filters (valid coordinate bounds, nonzero distance, and reasonable fare/passenger ranges) while keeping the same feature engineering and RandomForest training/prediction flow. This should reduce RMSE meaningfully without changing architecture, loss, or I/O paths, and still writes the same submission CSV.'
- What this solution (achieved 5.93334) has done: 'Your current RMSE (5.96533, lower is better) is still far above the target (3.8987), so we should make a small, legitimate change that improves generalization without changing your overall approach (feature set + RandomForest training/prediction flow). The most impactful minimal fix here is to ensure the RandomForest is trained on the full cleaned training sample (not just the 75% split used for evaluation), because right now you fit `rf` on `X_train` only, which unnecessarily reduces training data and typically hurts Kaggle test performance. I keep the train/validation split and RMSE reporting intact, but refit the same `rf` model on all available cleaned data (`X_1, y_1`) right before generating test predictions. This preserves model type, hyperparameters, and evaluation semantics while moving the score downward toward the target.'
- What this solution (achieved 6.26515) has done: 'Diagnosis: The crash happens in `np.issubdtype(data["pickup_datetime"].dtype, np.datetime64)` because NumPy cannot interpret timezone-aware pandas dtypes like `datetime64[ns, UTC]` as a NumPy dtype. This dataset can include UTC-aware timestamps after parsing, so the dtype check must use pandas’ dtype helpers instead of NumPy’s.  
Patch summary: In cell 19, replace the NumPy `issubdtype` check with `pd.api.types.is_datetime64_any_dtype`, which correctly handles both timezone-naive and timezone-aware datetimes, preserving the same feature extraction and column drop behavior.  
Updated cells: Only cell 19 is modified.  
Compatibility notes for cell k+1: `train` still have the same derived columns (`year`, `month`, `weekday`, `hour`) and `pickup_datetime` still be dropped, so `train.head()` in cell 20 remains valid.  
Assumptions: The intent is simply to detect “already-datetime” regardless of timezone; we keep `errors="coerce"` and the same integer dtypes for extracted fields.'
- What this solution (achieved 6.26515) has done: 'Diagnosis: The crash happens inside `LassoCV.fit()` because scikit-learn (patched by `sklearnex.patch_sklearn()`) ends up using a precomputed Gram matrix that fails validation due to floating-point mismatches. This is a known interaction where accelerated/optimized paths can pass a Gram matrix that is not numerically consistent enough for scikit-learn’s strict `_check_precomputed_gram_matrix`. Disabling Gram precomputation for `LassoCV` avoids this validation step while keeping the same model/optimization objective.

Patch summary: In cell 46 only, set `precompute=False` when constructing `LassoCV` so it computes coefficients directly from `X` without using/validating a precomputed Gram matrix. No other training logic, data, metrics, or parameters are changed.

Updated cells: (cell 46 only)

Compatibility notes for cell k+1: The variables `la`, `la_rmse`, and printed outputs remain the same types and meanings; cell 47 continues to use `ElasticNetCV` unchanged.

Assumptions: The error is triggered by Gram precomputation under sklearnex acceleration; forcing `precompute=False` is sufficient to make `LassoCV.fit()` deterministic and pass validation without changing evaluation semantics.'
- What this solution (achieved 6.26515) has done: 'The crash in cell 47 comes from `ElasticNetCV` internally deciding to use a precomputed Gram matrix and then failing its validation check due to floating-point discrepancies introduced by the `sklearnex` patching (accelerated kernels can yield slightly different Gram values). This is not a modeling-logic error in your code, but an interaction bug between `ElasticNetCV` and the patched sklearn backend. The minimal deterministic fix is to force `ElasticNetCV` to not use any precomputed Gram matrix by setting `precompute=False`, matching what you already did for `LassoCV`. This preserves the same model/training semantics while avoiding the invalid Gram-matrix validation path.'
- What this solution (achieved 6.49018) has done: 'Your current RMSE (6.26515, lower is better) is still far above the target (3.8987), so we should make a small, legitimate improvement that keeps your exact modeling approach (same RandomForest, same features) but fixes a high-impact bug in your datetime feature engineering. Right now `weekday` is incorrectly computed from `.dt.day` (day-of-month), which is much noisier than the intended day-of-week signal; switching to `.dt.weekday` (0–6) typically reduces error meaningfully for this competition without changing the pipeline structure. I also fix a small typo in the “nulls in train” print label and (critically) fill missing `harvesine/km` for **test** the same way as train to avoid NaNs propagating into predictions. All I/O paths and submission columns stay identical, and it still writes `NYCtaxiFare_prediction.csv`.'
- What this solution (achieved 6.50005) has done: 'Your current RMSE (6.49018, lower is better) is still far above the target (3.8987), so we should make a small, legitimate improvement that keeps your exact feature set and RandomForest approach intact. The biggest minimal gain now is to replace the non-geographic “lon/lat Euclidean in radians” proxy with a properly scaled equirectangular distance in kilometers while keeping the same feature names and downstream usage, so the model gets a cleaner distance signal without changing the pipeline structure. I also apply the same rounding you do for train to `test_1` (it was missing), to avoid a subtle train/test distribution mismatch on your two core engineered distance features. All paths, model type, train/refit flow, and submission schema/filename remain unchanged.'
- What this solution (achieved 6.50284) has done: 'Your current RMSE (6.50005, lower is better) is still far above the target (3.8987), so we should make a small, legitimate improvement that keeps your exact feature set and RandomForest approach intact. The biggest low-risk gain is to make the distance features more robust by clipping extreme/implausible distances (which are usually caused by remaining bad coordinates) so the RandomForest doesn’t waste capacity modeling rare noisy outliers. This does not change the model type, training loop, or feature definitions—only a light post-processing on two existing engineered columns, applied consistently to train and test. Finally, we keep all paths and the submission schema/filename identical and still refit on all cleaned data before predicting test.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count()))

print(os.listdir("../input"))



## === cell 1
DTYPES_TRAIN = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
    "key": "object",
}
DTYPES_TEST = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
    "key": "object",
}

train = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
test = pd.read_csv(
    "../input/test.csv",
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes.value_counts()



## === cell 6
test.dtypes.value_counts()



## === cell 7
train.isnull().sum()



## === cell 8
train.dropna(inplace=True)
train.isnull().sum()



## === cell 9
_zero_counts = (train == 0).to_numpy(dtype=np.uint8).sum(axis=0)
pd.Series(_zero_counts, index=train.columns)



## === cell 10
train = train



## === cell 11
pd.Series(_zero_counts, index=train.columns)



## === cell 12
train.shape



## === cell 13
train.describe()



## === cell 14
train.describe()



## === cell 15
train.dtypes.value_counts()



## === cell 16
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 17
mask = (
    (train["fare_amount"] > 0)
    & (train["fare_amount"] <= 250)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-74.5, -72.5))
    & (train["dropoff_longitude"].between(-74.5, -72.5))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
)

same_loc = (train["pickup_longitude"] == train["dropoff_longitude"]) & (
    train["pickup_latitude"] == train["dropoff_latitude"]
)
zeroish_pickup = (train["pickup_longitude"] == 0) | (train["pickup_latitude"] == 0)
zeroish_dropoff = (train["dropoff_longitude"] == 0) | (train["dropoff_latitude"] == 0)

mask &= (~same_loc) & (~zeroish_pickup) & (~zeroish_dropoff)

train = train.loc[mask].copy()
train.shape



## === cell 18
train.drop("key", axis=1, inplace=True)
train.head()



## === cell 19
import datetime as dt


def date_extraction(data):
    if not pd.api.types.is_datetime64_any_dtype(data["pickup_datetime"]):
        data["pickup_datetime"] = pd.to_datetime(
            data["pickup_datetime"], errors="coerce"
        )
    data["year"] = data["pickup_datetime"].dt.year.astype("int16")
    data["month"] = data["pickup_datetime"].dt.month.astype("int8")

    data["weekday"] = data["pickup_datetime"].dt.weekday.astype("int8")

    data["hour"] = data["pickup_datetime"].dt.hour.astype("int8")
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


date_extraction(train)



## === cell 20
train.head()



## === cell 21
date_extraction(test)
test.head()




## === cell 22
def long_lat_distance(x):
    R_km = 6371.0  # Earth radius in kilometers

    lat1 = np.radians(x["pickup_latitude"].to_numpy())
    lat2 = np.radians(x["dropoff_latitude"].to_numpy())
    lon1 = np.radians(x["pickup_longitude"].to_numpy())
    lon2 = np.radians(x["dropoff_longitude"].to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    x["Longitude_distance"] = dlon.astype("float32")
    x["Latitude_distance"] = dlat.astype("float32")

    x_mean = (lat1 + lat2) / 2.0
    dist_km = R_km * np.sqrt((dlon * np.cos(x_mean)) ** 2 + dlat**2)

    x["distance_travelled/10e3"] = dist_km.astype("float32")
    return x




## === cell 23
for x in (train, test):
    long_lat_distance(x)

train.head()




## === cell 24
def harvesine(x):
    r = 6371000.0  # meters
    lat1 = np.radians(x["pickup_latitude"].to_numpy())
    lat2 = np.radians(x["dropoff_latitude"].to_numpy())
    lon1 = np.radians(x["pickup_longitude"].to_numpy())
    lon2 = np.radians(x["dropoff_longitude"].to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    x["harvesine/km"] = ((r * c) / 1000.0).astype("float32")
    return x




## === cell 25
for x in (train, test):
    harvesine(x)

train.head()



## === cell 26
train.dtypes.value_counts()



## === cell 27
train.head()



## === cell 28
test.head()



## === cell 29
train.describe()



## === cell 30
print("Are there any nulls\nin the train data: ")
print(train.isnull().sum())

print("\nAre there any nulls\nin the test data: ")
print(test.isnull().sum())



## === cell 31
train["harvesine/km"] = train["harvesine/km"].fillna(train["harvesine/km"].median())
test["harvesine/km"] = test["harvesine/km"].fillna(train["harvesine/km"].median())



## === cell 32
for df in (train, test):
    df["harvesine/km"] = df["harvesine/km"].clip(lower=0.0, upper=100.0)
    df["distance_travelled/10e3"] = df["distance_travelled/10e3"].clip(
        lower=0.0, upper=100.0
    )



## === cell 33
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x != "fare_amount"]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 34
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)

correlations



## === cell 35
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 36
train.head()



## === cell 37
train_1 = train.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

train_1.head()



## === cell 38
train_1["harvesine/km"] = np.round(train_1["harvesine/km"].to_numpy(), 2).astype(
    "float32"
)
train_1["distance_travelled/10e3"] = np.round(
    train_1["distance_travelled/10e3"].to_numpy(), 2
).astype("float32")

train_1.head()



## === cell 39
train_1.describe()



## === cell 40
test_1 = test.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

test_1.head()



## === cell 41
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV




## === cell 42
def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))




## === cell 43
if "key" in train_1.columns:
    train_1.drop("key", axis=1, inplace=True)

feat_cols = [x for x in train_1.columns if x != "fare_amount"]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)



## === cell 44
lr = LinearRegression(n_jobs=-1).fit(X_train, y_train)
lr_rmse = rmse(y_test, lr.predict(X_test))
print(lr_rmse)



## === cell 45
alphas = [0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5]
rr = RidgeCV(alphas=alphas, cv=4).fit(X_train, y_train)
rr_rmse = rmse(y_test, rr.predict(X_test))

print(rr.alpha_, rr_rmse)



## === cell 46
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])

la = LassoCV(alphas=alphas, max_iter=int(5e4), cv=4, precompute=False).fit(
    X_train, y_train
)
la_rmse = rmse(y_test, la.predict(X_test))
print(la.alpha_, la_rmse)



## === cell 47
l1_ratios = np.linspace(0.1, 0.5, 5)
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])

en = ElasticNetCV(
    alphas=alphas, l1_ratio=l1_ratios, max_iter=int(1e4), precompute=False
).fit(X_train, y_train)
en_rmse = rmse(y_test, en.predict(X_test))

print(en.alpha_, en.l1_ratio_, en_rmse)



## === cell 48
rf = RandomForestRegressor(n_estimators=100, max_features=5, n_jobs=-1, random_state=42)
rf = rf.fit(X_train, y_train)



## === cell 49
labels = ["Linear", "lasso", "Ridge", "Elastic-Net"]
models_rmse = [lr_rmse, la_rmse, rr_rmse, en_rmse]
rmse_df = pd.Series(models_rmse, index=labels).to_frame()
rmse_df.rename(columns={0: "Errors"}, inplace=True)
rmse_df



## === cell 50
test.head()



## === cell 51
test_1["harvesine/km"] = np.round(test_1["harvesine/km"].to_numpy(), 2).astype(
    "float32"
)
test_1["distance_travelled/10e3"] = np.round(
    test_1["distance_travelled/10e3"].to_numpy(), 2
).astype("float32")

if "key" in test_1.columns:
    test_1.drop("key", axis=1, inplace=True)
test_1 = test_1.reindex(columns=X_train.columns)

test_1.head()



## === cell 52
rf = RandomForestRegressor(n_estimators=100, max_features=5, n_jobs=-1, random_state=42)
rf = rf.fit(X_1, y_1)

final_prediction = rf.predict(test_1)

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test.key, "fare_amount": final_prediction}
)
NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)



## === cell 53
NYCtaxiFare_submission.head()
