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

3.10

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

5.01424

# 6. Current score

1045.92796

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 915.26946) has done: 'I make the pipeline produce a valid submission reliably by stopping the accidental row filtering on the test set (it can drop rows and break the required submission row count). I also align one-hot encoded columns between train and test so the model always sees identical features at predict time, avoiding silent column-mismatch degradation. Finally, I fix the feature scaling bug where you normalize `abs_diff_longitude` separately for train and test (and only that one column); instead I fit the scaling on train and apply it to test, which usually reduces RMSE without changing the model or loss. These are minimal changes that keep your core feature set and LinearRegression training logic intact.'
- What this solution (achieved 2524.67693) has done: 'Your current RMSE (915) strongly suggests the model is being trained on many invalid/outlier rows (e.g., bogus coordinates, extreme fares) that dominate a plain LinearRegression fit. To move the score toward the target, I keep your exact feature set and LinearRegression training, but add minimal, standard NYC Taxi data cleaning on the training data only: drop non-positive/huge fares, restrict lat/long to plausible NYC bounds, and cap passenger_count to a reasonable range. I also ensure the train/test columns are aligned (including dropping the now-unused `pickup_time` after deriving `Peak_hour`) so the model sees consistent features at predict time. These are small changes that typically reduce RMSE dramatically without changing your core approach.'
- What this solution (achieved 2524.67693) has done: 'Your RMSE is astronomically high because the haversine distance is computed with a sign error (`dlon = lon1 - lon2` instead of `lon2 - lon1`), which makes the distance feature meaningless and destabilizes a plain LinearRegression fit. I fix that distance calculation in both train and test while keeping the exact same feature set and LinearRegression training logic. I also clamp negative fare predictions to 0.0 (validity constraint consistent with training cleanup) to avoid huge squared-error penalties from negative outputs. These changes are minimal, run fast, preserve the overall approach, and should move the score drastically toward your target.'
- What this solution (achieved 1978.67087) has done: 'The timeout is dominated by `pd.read_csv` using a per-row `skiprows` lambda (55M Python callbacks) plus several large Python loops for datetime parsing and feature creation, and an unnecessary scatter plot. I keep the same sampling logic (≈1,000,000 rows expected with the same RNG seed), same features, and the same LinearRegression fit/predict, but implement the sampling at the C/pandas level via chunked reading and vectorized masks. I also replace the Python loops over timestamps with equivalent vectorized pandas string/datetime operations, and remove the plot (it’s not used for modeling) to avoid backend overhead. All changes preserve the model, features, and evaluation semantics; only execution is made faster.'
- What this solution (achieved 1935.12533) has done: 'Your current RMSE is far above the target, so we need a small, legitimate accuracy improvement without changing the model type or feature set. The biggest issue left is that a plain LinearRegression is very sensitive to remaining outliers; we keep the same LinearRegression training but tighten training-only cleaning slightly (still standard for this dataset) to reduce the impact of extreme trips that inflate RMSE. We also add minimal sanity filters for coordinates and distance (training only) and ensure all feature columns are finite before fitting, while keeping the same submission format and end-to-end behavior. These changes should move the score substantially toward the target while preserving your pipeline’s core logic.'
- What this solution (achieved 1260.44072) has done: 'Your current score is far worse than the target, so we should improve legitimately with minimal impact to your pipeline. The biggest remaining accuracy issue is that plain `LinearRegression` is highly sensitive to outliers and heavy-tailed noise; switching to `Ridge` keeps the same linear model family and training approach while adding small L2 regularization that typically stabilizes RMSE a lot on this dataset. I keep all your existing feature engineering and cleaning, but add a tiny extra training-only filter to remove obviously nonsensical “very short distance but high fare” rows that can dominate squared error. Finally, I ensure train/test numeric coercion and column alignment remain identical and still produce a valid `submission.csv`.'
- What this solution (achieved 1253.8277) has done: 'Your RMSE is still far from the target, so the most likely remaining issue is that the linear model is being fit without an intercept and/or with inconsistent scaling across features, making it behave like it’s “calibrating from the origin” and inflating error. I keep your exact feature set and the same Ridge training/predict flow, but (1) explicitly enable `fit_intercept=True` (and remove the unsupported `random_state` arg), and (2) apply a single `StandardScaler` fit on train features and applied to both validation and submission features to stabilize Ridge on mixed-scale inputs. These are minimal, legitimate changes that preserve your model family/logic while typically reducing RMSE substantially. The submission writing and column alignment are kept intact.'
- What this solution (achieved 1267.45372) has done: 'Your score is far worse than the target (lower RMSE is better), so we should make a small, legitimate improvement without changing the core “Ridge on engineered features” approach. The largest remaining accuracy leak is that you scale only `abs_diff_longitude` manually while leaving other mixed-scale numeric features unscaled until the final `StandardScaler`, and you also quantize distances to 2 decimals, which throws away signal for a linear model. I (1) remove the distance rounding so the model keeps full precision, and (2) replace the one-off `abs_diff_longitude` normalization with scaling handled entirely by the single `StandardScaler` fit on training features (consistent train→test application). These are minimal changes that keep your features/model/training loop intact and should reduce RMSE substantially toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved 1267.45589) has done: 'Your RMSE is massively worse than the target, so the most likely cause is a silent-but-severe feature bug rather than “needing a better model.” I make two minimal, directly relevant fixes that preserve your Ridge-on-engineered-features approach: (1) fix a typo in the airport dropoff distance haversine formula that currently uses `a1` instead of `a2` (this corrupts an important feature for every training row), and (2) stop rounding predictions to 2 decimals (rounding adds avoidable error under RMSE). These changes keep the same features, same training flow, same model family, and same submission format, but should move the score sharply toward your target.'
- What this solution (achieved 1267.45589) has done: 'Your RMSE is catastrophically high for this competition, which almost always means the submission `key`→`fare_amount` alignment is broken (predictions are being paired with the wrong keys), not that the linear model is inherently that bad. I make one minimal, directly relevant fix: preserve the original test row order for the `key` column and align `X_submit` to that exact order after all feature engineering, so the predicted fares map to the correct keys. I also add a simple assertion to guarantee the submission row count and key order match the loaded test set, preventing silent misalignment regressions. No model, loss, or feature changes are made—only prediction/key alignment.'
- What this solution (achieved 1267.45589) has done: 'Your RMSE is so far from the target that the most likely remaining cause is still a train/test feature mismatch (the model is trained with one set/order of columns but you’re predicting with another), plus a subtle key/order misalignment risk from one-hot creation before final column reindexing. I make one minimal, high-impact fix: build `feature_cols` from the actual `X` you train on, then enforce that exact column set and order on the processed test frame *before* scaling/predicting (and assert it). I also ensure the same finite-value cleaning applied to train features is applied to `X_submit` (not altering core features/model), to avoid NaNs/Infs silently turning into 0s in a skewed way. These changes preserve your Ridge + StandardScaler + engineered features pipeline, but should move RMSE sharply down toward the target by preventing corrupted inputs at inference.'
- What this solution (achieved 1267.45589) has done: 'Your RMSE is so far from the target that the most likely remaining issue is still a subtle train/test feature mismatch driven by NaNs being filled with 0 in the test set while the model was trained only on rows without NaNs—this can create a distribution shift large enough to explode RMSE. I make one minimal, directly score-relevant change: apply the same “finite + no-missing” constraint to the training features by imputing with a train-fitted median (then applying that same imputer to test), instead of training on complete-cases but predicting on zero-filled data. This keeps your exact feature set, same Ridge+StandardScaler training flow, and same submission semantics, but removes a major inference-time corruption source. I also ensure the imputer is applied before scaling (consistent preprocessing) and keep your key/order assertions intact.'
- What this solution (achieved 1045.92796) has done: 'Your RMSE is still catastrophically far from the target for a Ridge + basic geo features pipeline, which strongly indicates the model is training on misparsed/invalid time-derived features (creating NaNs/garbage that then get median-imputed) and/or missing key NYC baseline geo features that the linear model needs. I keep your exact model family (Ridge), training flow (imputer→scaler→fit), and existing engineered features, but make two minimal, high-impact fixes: (1) parse `pickup_datetime` robustly with `pd.to_datetime` (instead of brittle string slicing) and derive numeric `hour` and `weekday` directly (then one-hot weekday exactly as before), and (2) add the standard `manhattan` and `euclidean` distance-in-degrees features (linear-friendly) without changing the learning algorithm. These changes are directly score-relevant, preserve submission alignment guarantees, and should move RMSE sharply down toward the target band without changing the core approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

print(os.listdir("../input"))

np.random.seed(42)



## === cell 1
train_path = "../input/train.csv"

p = 1_000_000 / 55_423_856
rs = np.random.RandomState(42)

usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype = {
    "key": "string",
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
    "pickup_datetime": "string",
}

target_n = 1_000_000
chunksize = (
    1_000_000  # large chunks reduce overhead; still streaming to stay memory-safe
)
kept = []
kept_n = 0

reader = pd.read_csv(train_path, usecols=usecols, dtype=dtype, chunksize=chunksize)
for chunk in reader:
    mask = rs.rand(len(chunk)) <= p
    if mask.any():
        sub = chunk.loc[mask]
        kept.append(sub)
        kept_n += len(sub)
        if kept_n >= target_n:
            break

train_df = pd.concat(kept, ignore_index=True)
if len(train_df) > target_n:
    train_df = train_df.iloc[:target_n].copy()

train_df.dtypes




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()

    df["manhattan"] = df["abs_diff_longitude"] + df["abs_diff_latitude"]
    df["euclidean"] = np.sqrt(
        df["abs_diff_longitude"] ** 2 + df["abs_diff_latitude"] ** 2
    )


add_travel_vector_features(train_df)



## === cell 3
print(train_df.isnull().sum())



## === cell 4
t = len(train_df)
print(f"Old size {t}")
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 5
pass



## === cell 6
print("Old size: %d" % len(train_df))

train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 250)]

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

nyc_bbox = {
    "pickup_longitude": (-74.3, -72.9),
    "dropoff_longitude": (-74.3, -72.9),
    "pickup_latitude": (40.5, 41.0),
    "dropoff_latitude": (40.5, 41.0),
}
for col, (lo, hi) in nyc_bbox.items():
    train_df = train_df[(train_df[col] >= lo) & (train_df[col] <= hi)]

train_df = train_df.loc[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
train_df = train_df.loc[
    (train_df.abs_diff_longitude + train_df.abs_diff_latitude) > 0.0
]

print("New size: %d" % len(train_df))



## === cell 7
test_df = pd.read_csv(
    "../input/test.csv",
    dtype={
        k: dtype.get(k, None)
        for k in [
            "key",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    },
)

test_keys = test_df["key"].copy()

test_df.dtypes



## === cell 8
add_travel_vector_features(test_df)



## === cell 9
print("Test rows kept:", len(test_df))



## === cell 10
train_df["pickup_datetime"].head()



## === cell 11
test_df["pickup_datetime"].head()



## === cell 12
train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce", utc=True)

train_df["Weekday"] = train_dt.dt.weekday
test_df["Weekday"] = test_dt.dt.weekday

train_df["pickup_time"] = (
    train_dt.dt.hour.astype("Int16") * 100 + train_dt.dt.minute.astype("Int16")
).astype("Int32")
test_df["pickup_time"] = (
    test_dt.dt.hour.astype("Int16") * 100 + test_dt.dt.minute.astype("Int16")
).astype("Int32")



## === cell 13
train_df.head()



## === cell 14
train_df.info()



## === cell 15
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 16
train_df.head()



## === cell 17
train_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 18
train_df.head()



## === cell 19
train_onehot = pd.get_dummies(train_df["Weekday"])
test_onehot = pd.get_dummies(test_df["Weekday"])
train_onehot, test_onehot = train_onehot.align(
    test_onehot, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_onehot], axis=1)
test_df = pd.concat([test_df, test_onehot], axis=1)



## === cell 20
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 21
train_df.head()



## === cell 22
type(train_df["pickup_time"].iloc[0])



## === cell 23
pass



## === cell 24
m = len(train_df)
print(m)



## === cell 25
train_df["pickup_time"].head()



## === cell 26
type(train_df["pickup_time"])



## === cell 27
pt = train_df["pickup_time"].to_numpy()
is_peak = ((pt > 700) & (pt < 1000)) | ((pt > 1600) & (pt < 2000))
train_df["Peak_hour"] = np.where(is_peak, "peak", "not Peak")



## === cell 28
train_df.head()



## === cell 29
pt = test_df["pickup_time"].to_numpy()
is_peak = ((pt > 700) & (pt < 1000)) | ((pt > 1600) & (pt < 2000))
test_df["Peak_hour"] = np.where(is_peak, "peak", "not Peak")



## === cell 30
trainoh = pd.get_dummies(train_df["Peak_hour"])
testoh = pd.get_dummies(test_df["Peak_hour"])
trainoh, testoh = trainoh.align(testoh, join="outer", axis=1, fill_value=0)

train_df = pd.concat([train_df, trainoh], axis=1)
test_df = pd.concat([test_df, testoh], axis=1)



## === cell 31
test_df.tail()



## === cell 32
train_df.drop("Peak_hour", inplace=True, axis=1)
test_df.drop("Peak_hour", inplace=True, axis=1)

train_df.drop("pickup_time", inplace=True, axis=1)
test_df.drop("pickup_time", inplace=True, axis=1)



## === cell 33
train_df.head()



## === cell 34
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlat = lat2 - lat1
dlon = lon2 - lon1  # FIXED
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlat = lat2 - lat1
dlon = lon2 - lon1  # FIXED
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621



## === cell 35
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)

dlat_pickup = lat3 - lat1
dlon_pickup = lon3 - lon1
dlat_dropoff = lat3 - lat2
dlon_dropoff = lon3 - lon2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_df["pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 36
R = 6373.0
lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)

dlat_pickup = lat3 - lat1
dlon_pickup = lon3 - lon1
dlat_dropoff = lat3 - lat2
dlon_dropoff = lon3 - lon2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_df["pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 37
train_df.head()



## === cell 38
train_df.shape



## === cell 39
test_df.shape



## === cell 40
train_df.head()



## === cell 41
from sklearn.model_selection import train_test_split

feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

for c in feature_cols:
    if train_df[c].dtype == "object" or str(train_df[c].dtype).startswith("string"):
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    if test_df[c].dtype == "object" or str(test_df[c].dtype).startswith("string"):
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

train_df = train_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df[(train_df["Distance"] >= 0.0) & (train_df["Distance"] <= 50.0)]
train_df = train_df[
    (train_df["pickup_Distance_airport"] >= 0.0)
    & (train_df["pickup_Distance_airport"] <= 100.0)
]
train_df = train_df[
    (train_df["Dropoff_Distance_airport"] >= 0.0)
    & (train_df["Dropoff_Distance_airport"] <= 100.0)
]

train_df = train_df.loc[
    ~((train_df["Distance"] < 0.05) & (train_df["fare_amount"] > 50.0))
]

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

keep_idx = y.notna()
X = X.loc[keep_idx]
y = y.loc[keep_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=80
)



## === cell 42
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="median")

X_train_i = imputer.fit_transform(X_train)
X_test_i = imputer.transform(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_i)
X_test_s = scaler.transform(X_test_i)



## === cell 43
from sklearn.linear_model import Ridge

lr = Ridge(alpha=1.0, fit_intercept=True)
lr.fit(X_train_s, y_train)



## === cell 44
print(lr.score(X_test_s, y_test))



## === cell 45
test_df = test_df.reset_index(drop=True)
test_keys = test_keys.reset_index(drop=True)

X_submit = test_df.drop("key", axis=1)

for c in X.columns:
    if c in X_submit.columns and (
        X_submit[c].dtype == "object" or str(X_submit[c].dtype).startswith("string")
    ):
        X_submit[c] = pd.to_numeric(X_submit[c], errors="coerce")

X_submit = X_submit.replace([np.inf, -np.inf], np.nan)

X_submit = X_submit.reindex(columns=X.columns, fill_value=np.nan)

assert list(X_submit.columns) == list(
    X.columns
), "X_submit columns do not match training columns exactly"
assert (
    len(X_submit) == len(test_keys) == len(test_df)
), "Prediction/test row count mismatch (pre-impute)"

X_submit_i = imputer.transform(X_submit)
X_submit_s = scaler.transform(X_submit_i)

pred = lr.predict(X_submit_s)
pred = np.clip(pred, 0.0, None)
pred = pred.astype(np.float64)

assert len(pred) == len(test_keys) == len(test_df), "Prediction/test row count mismatch"



## === cell 46
submission = pd.DataFrame({"key": test_keys, "fare_amount": pred})

assert submission["key"].equals(
    test_keys
), "Submission keys are not in the original test order"

submission = submission[["key", "fare_amount"]]



## === cell 47
submission



## === cell 48
submission.to_csv("submission.csv", index=False)

print(
    "Wrote submission.csv with rows:",
    len(submission),
    "and columns:",
    list(submission.columns),
)
print(submission.head())
