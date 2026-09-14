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

3.9

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

5.6891

# 6. Current score

inf

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.93312) has done: 'I fix the runtime error caused by the removed `normalize` argument in scikit-learn’s `LinearRegression` by switching to an equivalent `Pipeline(StandardScaler(with_mean=False), LinearRegression())`, which preserves the same core linear model while restoring the intended normalization behavior. I also make the weekday one-hot encoding robust by aligning train/test dummy columns so prediction never fails due to mismatched columns. Finally, I ensure the submission file is written with the required `key,fare_amount` columns and a `.csv` suffix. These changes are minimal, unblock execution end-to-end, and should improve RMSE compared to the unnormalized linear regression baseline.'
- What this solution (achieved 752.76254) has done: 'Your huge RMSE is most likely coming from inconsistent feature scaling: you manually “normalize” `abs_diff_longitude/latitude` using each dataset’s own mean/variance (train vs test), which makes those features live on different scales at inference time and can explode prediction error. I keep your feature engineering and linear-regression pipeline intact, but change the normalization to use train statistics for both train and test (a minimal, metric-aligned fix). I also clip negative fare predictions to 0 (fares can’t be negative), which typically reduces RMSE on this competition without changing the core model. Finally, I keep the submission schema identical while writing a `.csv` file as before.'
- What this solution (achieved 869.39611) has done: 'Your RMSE is still extremely high relative to the target, which strongly suggests the model is being trained on many invalid/outlier rows (e.g., negative/zero fares, impossible passenger counts, bad coordinates), causing the linear regression fit to be dominated by noise. I keep your exact feature engineering and LinearRegression+StandardScaler pipeline, but add minimal, standard NYC taxi data cleaning filters on the training set only (target/coords/passenger_count), which should move RMSE dramatically down toward the target band without changing the core logic. I also ensure that the train/test columns are aligned after *all* feature engineering (your current reindex happens too early, before new numeric features are added), preventing silent feature mismatches that can severely degrade predictions. Submission writing remains identical (`key,fare_amount` to `Submission.csv`).'
- What this solution (achieved 869.39611) has done: 'Your current RMSE is still orders of magnitude above the target, which usually means the model is effectively predicting on a mismatched feature set at inference time. The biggest issue here is that you reindex `test_df` to `train_feature_cols` (which *includes* `key`) and then later drop `key` again, leaving the test matrix with one fewer column than the train matrix the model was fit on—this silently breaks the feature alignment and can explode predictions. I make a minimal fix by defining a single, explicit `feature_cols` list (excluding both `key` and `fare_amount`) and aligning both train and test to it right before fitting/predicting, without changing your feature engineering or model. This should dramatically reduce RMSE toward your target while keeping the same core approach.'
- What this solution (achieved inf) has done: 'Your current RMSE is still far from the target, so we should make a small, legitimate improvement that doesn’t change the core model: fix the linear model’s sensitivity to remaining extreme outliers by training on `log1p(fare_amount)` and then `expm1` back at inference (same LinearRegression, same features, just a monotonic target transform). This is a standard minimal adjustment for heavy-tailed regression targets like taxi fares and usually drops RMSE substantially without changing the overall approach. I also add a single, metric-aligned filter to remove obviously “broken” trips where engineered distances are essentially zero but fare is non-trivial, which otherwise can dominate a linear fit. Submission format/paths remain identical and we still write `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved inf) has done: 'Your `inf` RMSE is almost certainly coming from non-finite values (NaN/±inf) getting into the model inputs or predictions—most commonly via division by (near-)zero variance in your manual normalization or by invalid numeric values after feature engineering. I make a minimal, score-relevant hardening: compute scaling using **standard deviation** (not variance) with an epsilon floor, then **replace any remaining non-finite feature values** in both train and test with safe defaults before fitting/predicting. I also ensure the feature matrix is strictly numeric and aligned, and clamp any non-finite predictions back to a reasonable fallback so the submission never contains NaN/inf (which triggers `inf` RMSE). Core feature engineering, model (LinearRegression in a scaler pipeline), and log1p target transform remain unchanged.'
- What this solution (achieved inf) has done: 'I fix the runtime error in the non-finite check by forcing the feature matrices to be real numeric `float64` arrays before calling `np.isfinite`, since `pandas` can still hold object-dtype values after `to_numeric` in some edge cases. I also add a minimal safeguard to drop any remaining non-numeric feature columns (should be none, but this prevents accidental object columns from slipping through) and ensure train/test feature columns are strictly aligned. These changes are score-neutral except that they prevent NaN/inf/object values from poisoning training or submission values (which is the likely cause of `inf` RMSE). The core feature engineering, log1p target transform, and the LinearRegression+StandardScaler pipeline remain unchanged, and the script still write `Submission.csv` with the required `key,fare_amount` columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

if os.path.exists("../input"):
    INPUT_DIR = "../input"
else:
    INPUT_DIR = "/kaggle/input"

print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
train_df = pd.read_csv(train_path, nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_path = os.path.join(INPUT_DIR, "test.csv")
test_df = pd.read_csv(test_path)
test_df.dtypes



## === cell 6
test_df.head()



## === cell 7
test_df.info()



## === cell 8
test_df.shape



## === cell 9
train_df.isna().sum()




## === cell 10
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 11
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 12
print(train_df.isnull().sum())



## === cell 13
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 15
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 16
print("Old size before basic validity filters: %d" % len(train_df))

train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 500)]

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

train_df = train_df[
    (train_df["pickup_longitude"].between(-75, -72))
    & (train_df["dropoff_longitude"].between(-75, -72))
    & (train_df["pickup_latitude"].between(40, 42))
    & (train_df["dropoff_latitude"].between(40, 42))
]

print("New size after basic validity filters: %d" % len(train_df))




## === cell 17
def creating_time(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][11:-7:]
    df["pickuptime"] = ls1


creating_time(train_df)
creating_time(test_df)



## === cell 18
train_df.head()



## === cell 19
test_df.head()




## === cell 20
def creating_weekdays(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][:-4:]
        ls1[i] = pd.Timestamp(ls1[i])
        ls1[i] = ls1[i].weekday()
    df["Weekday"] = ls1


creating_weekdays(train_df)
creating_weekdays(test_df)



## === cell 21
train_df.head()



## === cell 22
test_df.head()



## === cell 23
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 24
def replace_weekday(df):
    df["Weekday"] = df["Weekday"].replace(
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
    )


replace_weekday(train_df)
replace_weekday(test_df)



## === cell 25
train_df.head()



## === cell 26
test_df.head()



## === cell 27
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 28
train_df.head()



## === cell 29
test_df.head()



## === cell 30
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 31
def creating_pickupdate(df):
    ls1 = list(df["pickuptime"])
    for i in range(len(ls1)):
        z = ls1[i].split(":")
        ls1[i] = int(z[0]) * 100 + int(z[1])
    df["pickuptime"] = ls1


creating_pickupdate(train_df)
creating_pickupdate(test_df)



## === cell 32
train_df.head()



## === cell 33
test_df.head()




## === cell 34
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## === cell 35
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)

    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    d_lon_dropoff = lon3 - lon2
    d_lat_dropoff = lat3 - lat2

    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    distance1 = R * c1
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2

    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 36
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)

test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 37
print("Old size before distance-fare consistency filter: %d" % len(train_df))
train_df = train_df[~((train_df["Distance"] < 0.01) & (train_df["fare_amount"] > 5.0))]
print("New size after distance-fare consistency filter: %d" % len(train_df))



## === cell 38
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 39
eps = 1e-6
abs_lon_mean = float(np.mean(train_df["abs_diff_longitude"]))
abs_lon_std = float(np.std(train_df["abs_diff_longitude"]))
abs_lat_mean = float(np.mean(train_df["abs_diff_latitude"]))
abs_lat_std = float(np.std(train_df["abs_diff_latitude"]))

abs_lon_std = max(abs_lon_std, eps)
abs_lat_std = max(abs_lat_std, eps)

train_df["abs_diff_longitude"] = (
    np.abs(train_df["abs_diff_longitude"] - abs_lon_mean) / abs_lon_std
)
train_df["abs_diff_latitude"] = (
    np.abs(train_df["abs_diff_latitude"] - abs_lat_mean) / abs_lat_std
)

test_df["abs_diff_longitude"] = (
    np.abs(test_df["abs_diff_longitude"] - abs_lon_mean) / abs_lon_std
)
test_df["abs_diff_latitude"] = (
    np.abs(test_df["abs_diff_latitude"] - abs_lat_mean) / abs_lat_std
)



## === cell 40
feature_cols = [c for c in train_df.columns if c not in ("fare_amount", "key")]

X_all = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
test_X = test_df.reindex(columns=feature_cols, fill_value=0).apply(
    pd.to_numeric, errors="coerce"
)

X_all = X_all.select_dtypes(include=[np.number])
test_X = test_X.reindex(columns=X_all.columns, fill_value=0).select_dtypes(
    include=[np.number]
)

X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
test_X = test_X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)

print("Train X shape:", X_all.shape)
print("Test  X shape:", test_X.shape)
print(
    "Any non-finite in Train X:", (~np.isfinite(X_all.to_numpy(dtype=np.float64))).any()
)
print(
    "Any non-finite in Test  X:",
    (~np.isfinite(test_X.to_numpy(dtype=np.float64))).any(),
)



## === cell 41
from sklearn.model_selection import train_test_split

y = np.log1p(train_df["fare_amount"].astype(float))

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.01, random_state=80
)



## === cell 42
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LinearRegression()),
    ]
)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 43
pred_log = lr.predict(test_X)
pred = np.expm1(pred_log)

fallback = float(np.median(train_df["fare_amount"].astype(float)))
pred = np.where(np.isfinite(pred), pred, fallback)

pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)
print(pred)



## === cell 44
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_df["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 45
Submission.head()



## === cell 46
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
print(
    "Any non-finite in submission fares:",
    (~np.isfinite(Submission["fare_amount"].to_numpy(dtype=np.float64))).any(),
)
