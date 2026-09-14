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

28.94679

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the runtime error caused by the removed `normalize` argument in scikit-learn’s `LinearRegression` so the model can train and produce predictions. To keep the core modeling logic the same (linear regression on the engineered features), I add a minimal `StandardScaler` preprocessing step via a `Pipeline`, which is the modern equivalent of the old normalization behavior. I also make the one-hot weekday columns consistent between train and test to prevent column-mismatch errors at prediction time. Finally, I ensure the submission is written as a valid `.csv` file with the required `key,fare_amount` columns.'
- What this solution (achieved 28.74177) has done: 'Your score is extremely far from the target (RMSE 936 vs 5.689), which strongly suggests a correctness bug rather than “model quality”. The biggest issue is that you normalize `abs_diff_longitude/latitude` using each dataset’s own mean/variance, so train and test are on different scales and predictions blow up; we change this to use train statistics for both (same core feature, just correct application). We also clip predictions to a reasonable fare range to prevent extreme outliers from dominating RMSE (this is a standard post-processing step that keeps evaluation semantics). Finally, we write the submission as `submission.csv` with the required `key,fare_amount` columns (keep your existing logic otherwise).'
- What this solution (achieved 28.87246) has done: 'Your current RMSE (28.74) is still far above the target (5.689), so we should make small correctness-focused fixes that typically yield a large improvement without changing the model type. I (1) filter obviously invalid training rows (fare/coords/passenger_count) which otherwise corrupt linear regression, (2) ensure feature columns are identical and in the same order for train vs test before fitting/predicting, and (3) compute the “normalization” of `abs_diff_*` as a standard score using train mean/std (your current mean/variance+abs formulation warps the feature). These keep the same overall approach (same engineered features + LinearRegression in a scaling Pipeline) but remove common sources of inflated RMSE. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 28.87043) has done: 'Your RMSE (28.87) is still far above the target (5.6891), so the most likely issue is remaining label noise/outliers and weak time parsing rather than the linear model itself. I keep your exact feature set and LinearRegression pipeline, but make two minimal correctness upgrades: (1) parse `pickup_datetime` with vectorized `pd.to_datetime` to create *hour* and *weekday* reliably (your current string slicing can silently mis-parse), and (2) add the standard NYC Taxi baseline cleaning step that removes rows where pickup==dropoff and extremely tiny trips (these produce near-zero distances with non-zero fares and hurt RMSE a lot). Everything else (distance features, abs diffs, one-hot weekdays, scaling + LinearRegression, clipping, submission format/path) stays the same.'
- What this solution (achieved 28.94679) has done: 'Your RMSE is still far above the target, so the most likely remaining issue is that the linear model is being pulled by noisy/outlier training rows rather than a fundamental modeling limitation. Keeping your exact feature set and the same LinearRegression-in-a-scaled-Pipeline core, I make a minimal but high-impact cleaning adjustment: tighten the training fare range to the commonly used NYC Taxi baseline (<= 250) and remove extreme “airport distance” outliers that typically correspond to bad coordinates but still pass the coarse NYC bounding box. This reduces label/feature noise without changing the model or adding new features, and it should move the score down toward the target. Everything else (feature engineering, one-hot weekdays, scaling, clipping, and submission writing) stays intact.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv("../input/test.csv")
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
def clean_training_rows(df):
    df = df[
        (df["fare_amount"] > 0.0)
        & (df["fare_amount"] <= 250.0)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
    ]
    df = df[
        (df["pickup_longitude"].between(-75, -72))
        & (df["dropoff_longitude"].between(-75, -72))
        & (df["pickup_latitude"].between(40, 42))
        & (df["dropoff_latitude"].between(40, 42))
    ]
    return df


train_df = clean_training_rows(train_df)
print("After cleaning invalid rows:", train_df.shape)




## === cell 17
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["pickuptime"] = dt.dt.hour.astype("float32")
    df["Weekday"] = dt.dt.weekday.astype("float32")


add_time_features(train_df)
add_time_features(test_df)



## === cell 18
train_df.head()



## === cell 19
test_df.head()



## === cell 20
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 21
def replace_weekday(df):
    mapping = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df["Weekday"] = df["Weekday"].replace(mapping)


replace_weekday(train_df)
replace_weekday(test_df)



## === cell 22
train_df.head()



## === cell 23
test_df.head()



## === cell 24
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 25
train_df.head()



## === cell 26
test_df.head()



## === cell 27
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 28
train_df["pickuptime"] = pd.to_numeric(train_df["pickuptime"], errors="coerce").fillna(
    0.0
)
test_df["pickuptime"] = pd.to_numeric(test_df["pickuptime"], errors="coerce").fillna(
    0.0
)



## === cell 29
train_df.head()



## === cell 30
test_df.head()




## === cell 31
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




## === cell 32
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



## === cell 33
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 34
train_df = train_df[
    (train_df["abs_diff_longitude"] > 0) | (train_df["abs_diff_latitude"] > 0)
]
train_df = train_df[train_df["Distance"] >= 0.01]
print("After removing zero/near-zero distance trips:", train_df.shape)



## === cell 35
train_df = train_df[
    (train_df["Pickup_Distance_airport"].between(0.0, 50.0))
    & (train_df["Dropoff_Distance_airport"].between(0.0, 50.0))
]
print("After removing extreme airport-distance outliers:", train_df.shape)



## === cell 36
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



## === cell 37
abs_long_mean = train_df["abs_diff_longitude"].mean()
abs_long_std = train_df["abs_diff_longitude"].std(ddof=0)
abs_lat_mean = train_df["abs_diff_latitude"].mean()
abs_lat_std = train_df["abs_diff_latitude"].std(ddof=0)

if abs_long_std == 0 or np.isnan(abs_long_std):
    abs_long_std = 1.0
if abs_lat_std == 0 or np.isnan(abs_lat_std):
    abs_lat_std = 1.0

train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - abs_long_mean
) / abs_long_std
train_df["abs_diff_latitude"] = (
    train_df["abs_diff_latitude"] - abs_lat_mean
) / abs_lat_std

test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - abs_long_mean
) / abs_long_std
test_df["abs_diff_latitude"] = (
    test_df["abs_diff_latitude"] - abs_lat_mean
) / abs_lat_std



## === cell 38
print(train_df.shape)
print(test_df.shape)



## === cell 39
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 40
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 41
X_test_kaggle = test_df.drop("key", axis=1).copy()
X_test_kaggle = X_test_kaggle.apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_test_kaggle = X_test_kaggle.reindex(columns=X.columns, fill_value=0.0)

pred = lr.predict(X_test_kaggle)

pred = np.clip(pred, 0.0, 500.0)
pred = np.round(pred, 2)
print(pred)



## === cell 42
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})



## === cell 43
Submission.head()



## === cell 44
Submission.to_csv("submission.csv", index=False)
print("Wrote submission to submission.csv with shape:", Submission.shape)
