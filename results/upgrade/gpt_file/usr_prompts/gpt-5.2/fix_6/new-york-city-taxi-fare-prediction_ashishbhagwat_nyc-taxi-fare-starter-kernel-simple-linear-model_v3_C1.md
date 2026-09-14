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

993.4602

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the runtime error by removing the deprecated `normalize` argument from `LinearRegression` (scikit-learn 1.2+) so the model can train and produce predictions. I also fix a likely hidden bug where one-hot weekday columns can differ between train and test by aligning the columns so `lr.predict()` won’t fail due to mismatched feature sets. Finally, I make the submission writing robust and ensure it outputs a valid `Submission.csv` with exactly `key,fare_amount` columns. These changes preserve the existing feature engineering and linear regression core logic while enabling an end-to-end run and a valid Kaggle submission file.'
- What this solution (achieved 993.4602) has done: 'I make two minimal, score-relevant fixes while preserving your exact feature engineering and LinearRegression core logic. First, I remove the `abs()` from your “normalization” step (it currently destroys directionality and harms RMSE), keeping the same mean/variance stats computed on train and applied to test. Second, I filter out obviously invalid training targets/coordinates (negative/huge fares and out-of-range lat/lon) before training; this is a standard NYC Taxi Fare cleanup that usually reduces RMSE without changing the model class or training loop. Everything else (feature creation, model, prediction clamp, submission format/path) stays the same and the script still writes `Submission.csv`.'
- What this solution (achieved 993.4602) has done: 'Your RMSE is catastrophically high relative to the target, which strongly suggests a correctness bug rather than a “needs a better model” issue. The most score-relevant minimal fix is to correct your “normalization” step: you’re dividing by the variance instead of the standard deviation, which badly mis-scales features and can explode errors. I change that to divide by `sqrt(var)` (std) while keeping the exact same features, LinearRegression model, and training flow. I also keep your train/test column alignment and ensure the submission remains `key,fare_amount` written to `Submission.csv`.'
- What this solution (achieved 993.4602) has done: 'Your RMSE is so far from the target that it’s almost certainly a feature/column mismatch bug at inference rather than “model quality”. The biggest minimal fix is to ensure the one-hot weekday columns are identical between train and test by doing `get_dummies` on the concatenated Weekday series (so you never silently drop a weekday column in test). Second, your current `X.align(..., join="outer")` can introduce train-only columns that don’t exist in test (or vice versa) and encourages mismatch; switching to `join="inner"` keeps only shared features and stabilizes inference without changing the model class or training flow. These are small, score-relevant correctness fixes that preserve your feature engineering intent and keep the same LinearRegression training approach while producing a valid `Submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not find Kaggle input directory among: " + str(INPUT_DIR_CANDIDATES)
    )

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:20])




## === cell 1
def resolve_path(filename: str) -> str:
    direct = os.path.join(INPUT_DIR, filename)
    nested = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", filename)
    if os.path.exists(direct):
        return direct
    if os.path.exists(nested):
        return nested
    if os.path.exists(f"../input/{filename}"):
        return f"../input/{filename}"
    raise FileNotFoundError(f"Could not find {filename} in {direct} or {nested}")


TRAIN_PATH = resolve_path("train.csv")
TEST_PATH = resolve_path("test.csv")
SAMPLE_SUB_PATH = resolve_path("sample_submission.csv")

TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH



## === cell 2
train_df = pd.read_csv(TRAIN_PATH, nrows=10_000_000)
train_df.dtypes



## === cell 3
train_df.head()



## === cell 4
train_df.shape



## === cell 5
train_df.info()



## === cell 6
test_df = pd.read_csv(TEST_PATH)
test_df.dtypes



## === cell 7
test_df.head()



## === cell 8
test_df.info()



## === cell 9
test_df.shape



## === cell 10
train_df.isna().sum()




## === cell 11
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 12
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 13
print(train_df.isnull().sum())



## === cell 14
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 15
try:
    plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 16
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 17
print("Old size (pre-clean): %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0.0)
    & (train_df["fare_amount"] < 500.0)
    & (train_df["passenger_count"] >= 1)
    & (train_df["passenger_count"] <= 6)
    & (train_df["pickup_longitude"].between(-75.0, -72.0))
    & (train_df["dropoff_longitude"].between(-75.0, -72.0))
    & (train_df["pickup_latitude"].between(40.0, 42.0))
    & (train_df["dropoff_latitude"].between(40.0, 42.0))
]
print("New size (post-clean): %d" % len(train_df))




## === cell 18
def creating_time(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][11:-7:]
    df["pickuptime"] = ls1


creating_time(train_df)
creating_time(test_df)



## === cell 19
train_df.head()



## === cell 20
test_df.head()




## === cell 21
def creating_weekdays(df):
    ls1 = list(df["pickup_datetime"])
    for i in range(len(ls1)):
        ls1[i] = ls1[i][:-4:]
        ls1[i] = pd.Timestamp(ls1[i])
        ls1[i] = ls1[i].weekday()
    df["Weekday"] = ls1


creating_weekdays(train_df)
creating_weekdays(test_df)



## === cell 22
train_df.head()



## === cell 23
test_df.head()



## === cell 24
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 25
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



## === cell 26
train_df.head()



## === cell 27
test_df.head()



## === cell 28
all_weekdays = pd.concat(
    [train_df["Weekday"].astype(str), test_df["Weekday"].astype(str)], axis=0
)
all_weekdays_oh = pd.get_dummies(all_weekdays)

train_one_hot = all_weekdays_oh.iloc[: len(train_df)].reset_index(drop=True)
test_one_hot = all_weekdays_oh.iloc[len(train_df) :].reset_index(drop=True)

train_df = pd.concat([train_df.reset_index(drop=True), train_one_hot], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_one_hot], axis=1)



## === cell 29
train_df.head()



## === cell 30
test_df.head()



## === cell 31
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 32
def creating_pickupdate(df):
    ls1 = list(df["pickuptime"])
    for i in range(len(ls1)):
        z = ls1[i].split(":")
        ls1[i] = int(z[0]) * 100 + int(z[1])
    df["pickuptime"] = ls1


creating_pickupdate(train_df)
creating_pickupdate(test_df)



## === cell 33
train_df.head()



## === cell 34
test_df.head()




## === cell 35
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




## === cell 36
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



## === cell 37
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



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
def normalize_with_train_stats(train_df, test_df, col):
    mu = float(np.mean(train_df[col]))
    var = float(np.var(train_df[col]))
    std = float(np.sqrt(var)) if var > 0.0 else 1.0
    if std == 0.0:
        std = 1.0
    train_df[col] = (train_df[col] - mu) / std
    test_df[col] = (test_df[col] - mu) / std


normalize_with_train_stats(train_df, test_df, "abs_diff_longitude")
normalize_with_train_stats(train_df, test_df, "abs_diff_latitude")



## === cell 40
print(train_df.shape)
print(test_df.shape)



## === cell 41
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X_test_features = test_df.drop(["key"], axis=1)

X, X_test_features = X.align(X_test_features, join="inner", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 42
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_valid, y_valid))



## === cell 43
pred = lr.predict(X_test_features)

pred = np.maximum(pred, 0.0)

print(pred[:10])



## === cell 44
Submission = pd.DataFrame(
    {"key": test_df["key"].astype(str), "fare_amount": pred.astype(float)}
)

Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 45
out_path = "Submission.csv"
Submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path, "shape:", Submission.shape)
print(Submission.head())
