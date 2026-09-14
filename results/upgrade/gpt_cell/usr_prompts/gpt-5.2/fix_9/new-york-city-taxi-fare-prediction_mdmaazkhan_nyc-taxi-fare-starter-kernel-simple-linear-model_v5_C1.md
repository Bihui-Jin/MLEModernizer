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

1260.44072

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 915.26946) has done: 'I make the pipeline produce a valid submission reliably by stopping the accidental row filtering on the test set (it can drop rows and break the required submission row count). I also align one-hot encoded columns between train and test so the model always sees identical features at predict time, avoiding silent column-mismatch degradation. Finally, I fix the feature scaling bug where you normalize `abs_diff_longitude` separately for train and test (and only that one column); instead I fit the scaling on train and apply it to test, which usually reduces RMSE without changing the model or loss. These are minimal changes that keep your core feature set and LinearRegression training logic intact.'
- What this solution (achieved 2524.67693) has done: 'Your current RMSE (915) strongly suggests the model is being trained on many invalid/outlier rows (e.g., bogus coordinates, extreme fares) that dominate a plain LinearRegression fit. To move the score toward the target, I keep your exact feature set and LinearRegression training, but add minimal, standard NYC Taxi data cleaning on the training data only: drop non-positive/huge fares, restrict lat/long to plausible NYC bounds, and cap passenger_count to a reasonable range. I also ensure the train/test columns are aligned (including dropping the now-unused `pickup_time` after deriving `Peak_hour`) so the model sees consistent features at predict time. These are small changes that typically reduce RMSE dramatically without changing your core approach.'
- What this solution (achieved 2524.67693) has done: 'Your RMSE is astronomically high because the haversine distance is computed with a sign error (`dlon = lon1 - lon2` instead of `lon2 - lon1`), which makes the distance feature meaningless and destabilizes a plain LinearRegression fit. I fix that distance calculation in both train and test while keeping the exact same feature set and LinearRegression training logic. I also clamp negative fare predictions to 0.0 (validity constraint consistent with training cleanup) to avoid huge squared-error penalties from negative outputs. These changes are minimal, run fast, preserve the overall approach, and should move the score drastically toward your target.'
- What this solution (achieved 1978.67087) has done: 'The timeout is dominated by `pd.read_csv` using a per-row `skiprows` lambda (55M Python callbacks) plus several large Python loops for datetime parsing and feature creation, and an unnecessary scatter plot. I keep the same sampling logic (≈1,000,000 rows expected with the same RNG seed), same features, and the same LinearRegression fit/predict, but implement the sampling at the C/pandas level via chunked reading and vectorized masks. I also replace the Python loops over timestamps with equivalent vectorized pandas string/datetime operations, and remove the plot (it’s not used for modeling) to avoid backend overhead. All changes preserve the model, features, and evaluation semantics; only execution is made faster.'
- What this solution (achieved 1935.12533) has done: 'Your current RMSE is far above the target, so we need a small, legitimate accuracy improvement without changing the model type or feature set. The biggest issue left is that a plain LinearRegression is very sensitive to remaining outliers; we keep the same LinearRegression training but tighten training-only cleaning slightly (still standard for this dataset) to reduce the impact of extreme trips that inflate RMSE. We also add minimal sanity filters for coordinates and distance (training only) and ensure all feature columns are finite before fitting, while keeping the same submission format and end-to-end behavior. These changes should move the score substantially toward the target while preserving your pipeline’s core logic.'
- What this solution (achieved 1260.44072) has done: 'Your current score is far worse than the target, so we should improve legitimately with minimal impact to your pipeline. The biggest remaining accuracy issue is that plain `LinearRegression` is highly sensitive to outliers and heavy-tailed noise; switching to `Ridge` keeps the same linear model family and training approach while adding small L2 regularization that typically stabilizes RMSE a lot on this dataset. I keep all your existing feature engineering and cleaning, but add a tiny extra training-only filter to remove obviously nonsensical “very short distance but high fare” rows that can dominate squared error. Finally, I ensure train/test numeric coercion and column alignment remain identical and still produce a valid `submission.csv`.'

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
train_df["pickup_time"] = train_df["pickup_datetime"].str.slice(11, -7)
test_df["pickup_time"] = test_df["pickup_datetime"].str.slice(11, -7)



## === cell 13
train_dt = pd.to_datetime(train_df["pickup_datetime"].str.slice(0, -4), errors="coerce")
test_dt = pd.to_datetime(test_df["pickup_datetime"].str.slice(0, -4), errors="coerce")
train_df["Weekday"] = train_dt.dt.weekday
test_df["Weekday"] = test_dt.dt.weekday



## === cell 14
train_df.head()



## === cell 15
train_df.info()



## === cell 16
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 17
train_df.head()



## === cell 18
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



## === cell 19
train_df.head()



## === cell 20
train_onehot = pd.get_dummies(train_df["Weekday"])
test_onehot = pd.get_dummies(test_df["Weekday"])
train_onehot, test_onehot = train_onehot.align(
    test_onehot, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_onehot], axis=1)
test_df = pd.concat([test_df, test_onehot], axis=1)



## === cell 21
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 22
train_df.head()



## === cell 23
type(train_df["pickup_time"][0])



## === cell 24
pt_train = train_df["pickup_time"].str.slice(0, 5)
pt_test = test_df["pickup_time"].str.slice(0, 5)

train_df["pickup_time"] = pt_train.str.slice(0, 2).astype(
    "int32"
) * 100 + pt_train.str.slice(3, 5).astype("int32")
test_df["pickup_time"] = pt_test.str.slice(0, 2).astype(
    "int32"
) * 100 + pt_test.str.slice(3, 5).astype("int32")



## === cell 25
m = len(train_df)
print(m)



## === cell 26
train_df["pickup_time"].head()



## === cell 27
type(train_df["pickup_time"])



## === cell 28
pt = train_df["pickup_time"].to_numpy()
is_peak = ((pt > 700) & (pt < 1000)) | ((pt > 1600) & (pt < 2000))
train_df["Peak_hour"] = np.where(is_peak, "peak", "not Peak")



## === cell 29
train_df.head()



## === cell 30
pt = test_df["pickup_time"].to_numpy()
is_peak = ((pt > 700) & (pt < 1000)) | ((pt > 1600) & (pt < 2000))
test_df["Peak_hour"] = np.where(is_peak, "peak", "not Peak")



## === cell 31
trainoh = pd.get_dummies(train_df["Peak_hour"])
testoh = pd.get_dummies(test_df["Peak_hour"])
trainoh, testoh = trainoh.align(testoh, join="outer", axis=1, fill_value=0)

train_df = pd.concat([train_df, trainoh], axis=1)
test_df = pd.concat([test_df, testoh], axis=1)



## === cell 32
test_df.tail()



## === cell 33
train_df.drop("Peak_hour", inplace=True, axis=1)
test_df.drop("Peak_hour", inplace=True, axis=1)

train_df.drop("pickup_time", inplace=True, axis=1)
test_df.drop("pickup_time", inplace=True, axis=1)



## === cell 34
train_df.head()



## === cell 35
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



## === cell 36
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



## === cell 37
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



## === cell 38
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["pickup_Distance_airport"] = np.round(train_df["pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["pickup_Distance_airport"] = np.round(test_df["pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 39
train_df.head()



## === cell 40
_train_mean = float(train_df["abs_diff_longitude"].mean())
_train_var = float(train_df["abs_diff_longitude"].var())
_train_std = (
    float(np.sqrt(_train_var)) if _train_var > 0 and not np.isnan(_train_var) else 1.0
)

train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - _train_mean
) / _train_std
test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - _train_mean
) / _train_std



## === cell 41
train_df.shape



## === cell 42
test_df.shape



## === cell 43
train_df.head()



## === cell 44
from sklearn.model_selection import train_test_split

feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

for c in feature_cols:
    if train_df[c].dtype == "object" or str(train_df[c].dtype).startswith("string"):
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    if test_df[c].dtype == "object" or str(test_df[c].dtype).startswith("string"):
        test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

train_df = train_df.replace([np.inf, -np.inf], np.nan)
train_df = train_df.dropna(subset=feature_cols + ["fare_amount"])

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=80
)



## === cell 45
from sklearn.linear_model import Ridge

lr = Ridge(alpha=1.0, random_state=80)
lr.fit(X_train, y_train)



## === cell 46
print(lr.score(X_test, y_test))



## === cell 47
X_submit = test_df.drop("key", axis=1).reindex(columns=X.columns, fill_value=0)

pred = lr.predict(X_submit)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 48
submission = pd.DataFrame(data=pred, columns=["fare_amount"])
submission["key"] = test_df["key"]
submission = submission[["key", "fare_amount"]]



## === cell 49
submission



## === cell 50
submission.to_csv("submission.csv", index=False)

print(
    "Wrote submission.csv with rows:",
    len(submission),
    "and columns:",
    list(submission.columns),
)
print(submission.head())
