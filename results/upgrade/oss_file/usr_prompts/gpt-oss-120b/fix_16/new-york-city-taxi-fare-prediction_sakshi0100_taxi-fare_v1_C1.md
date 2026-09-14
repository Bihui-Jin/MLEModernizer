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

5.68932

# 6. Current score

788.17304

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize=True` argument from `LinearRegression` so the model can be instantiated and trained, which also restores the `lr` object needed for later cells. The rest of the pipeline stays unchanged; the script now run end‑to‑end and write a proper `Submission.csv` with the required columns.'
- What this solution (achieved 936.94064) has done: 'I add a standard‑scaler to normalize the numeric features before fitting the same LinearRegression model, and then clip any negative predictions to 0 before rounding. This keeps the core LinearRegression logic unchanged while improving numerical stability, which should lower the RMSE toward the target.'
- What this solution (achieved 752.76915) has done: 'I fix the inconsistent feature scaling that caused the model to see mismatched distributions between train and test (the original code scaled the test set using its own mean/variance). The fix computes the mean and variance from the training data once and applies those exact statistics to both the training and test “Difference_” features. I also remove the unnecessary rounding of predictions before writing the submission, which slightly improves RMSE while keeping the LinearRegression core unchanged.'
- What this solution (achieved inf) has done: 'I keep the overall feature engineering and LinearRegression model but train it on the log‑transformed fare amount (log1p) and then exponentiate the predictions back to the original scale. This usually stabilises the skewed target distribution and reduces RMSE, moving the score much closer to the target. I also clip extreme fare values in the training data to a sensible range (0‑200 $) to avoid outlier distortion. The rest of the pipeline, including scaling and feature handling, remains unchanged.'
- What this solution (achieved 751.45992) has done: 'I remove the unnecessary log‑transform of the target variable, fitting the LinearRegression directly on the fare amount. This avoids the exponentiation step that can produce infinities and gives a finite validation RMSE, moving the score toward the target while keeping the core model and feature engineering unchanged.'
- What this solution (achieved inf) has done: 'I keep the overall pipeline and LinearRegression model unchanged, but apply a log‑transform to the target variable before fitting and reverse it after prediction. This stabilises the heavy‑tailed fare distribution, reduces extreme errors, and moves the RMSE much closer to the target without altering any core logic.'
- What this solution (achieved 751.45992) has done: 'I remove the log‑transform of the target variable, training the LinearRegression directly on the fare amount. This avoids the overflow that produced an infinite RMSE and yields a finite validation score. The prediction steps are also simplified to use the raw model output (clipped at 0) instead of exponentiating. All other feature‑engineering steps and the core LinearRegression model remain unchanged.'
- What this solution (achieved 784.82681) has done: 'I replace the plain LinearRegression with a Ridge regression (still a linear model) and remove the log‑transform of the target so the model directly predicts fares. I also simplify the submission write‑out by not setting the index, ensuring a proper CSV with the required columns. These minimal adjustments keep the overall feature engineering unchanged while improving calibration and should move the RMSE closer to the target.'
- What this solution (achieved 787.84792) has done: 'I remove the log‑transform of the target so the Ridge model predicts the fare amount directly, and I lower the regularisation strength (α = 0.1) which usually reduces bias for this linear model. The validation RMSE is recomputed on the raw values, and the test‑time predictions are clipped at 0 without any exponentiation. These minimal adjustments keep the original feature engineering and model type while moving the score toward the target 5.68932.'
- What this solution (achieved 788.17304) has done: 'The fix updates the file paths to the actual dataset locations, adds a small safety‑check for missing columns, and keeps the original feature‑engineering and Ridge‑regression pipeline unchanged. These minimal changes let the notebook run end‑to‑end and produce a proper `Submission.csv` while preserving the core modeling logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
train_data = pd.read_csv(train_path, nrows=10_000_000)  # sample for speed
train_data.head()




## === cell 2
train_data.shape




## === cell 3
train_data.info()




## === cell 4
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
test_data = pd.read_csv(test_path)
test_data.head()




## === cell 5
test_data.info()




## === cell 6
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)




## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")




## === cell 8
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## === cell 9
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1




## === cell 10
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]  # remove trailing timezone info
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1




## === cell 11
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)




## === cell 12
train_data["Weekday"].replace(
    to_replace=list(range(7)),
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
test_data["Weekday"].replace(
    to_replace=list(range(7)),
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




## === cell 13
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)




## === cell 14
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## === cell 15
def add_time_features(df):
    hours = []
    minute_fracs = []
    for t in df["pickuptime"]:
        h, m = map(int, t.split(":"))
        hours.append(h)
        minute_fracs.append(m / 60.0)
    df["hour"] = hours
    df["minute_frac"] = minute_fracs
    df.drop("pickuptime", axis=1, inplace=True)


add_time_features(train_data)
add_time_features(test_data)




## === cell 16
train_data.head()




## === cell 17
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = distance * 0.621  # convert km to miles

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = distance * 0.621




## === cell 18
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

lat_air = np.radians(40.6413111)  # JFK latitude
lon_air = np.radians(-73.7781391)  # JFK longitude

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_data["Pickup_Distance_airport"] = (R * c1) * 0.621

dlon_drop = lon_air - lon2
dlat_drop = lat_air - lat2
a2 = (
    np.sin(dlat_drop / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_drop / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_data["Dropoff_Distance_airport"] = (R * c2) * 0.621

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
test_data["Pickup_Distance_airport"] = (R * c1) * 0.621

dlon_drop = lon_air - lon2
dlat_drop = lat_air - lat2
a2 = (
    np.sin(dlat_drop / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_drop / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
test_data["Dropoff_Distance_airport"] = (R * c2) * 0.621




## === cell 19
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    train_data[col] = np.round(train_data[col], 2)
    test_data[col] = np.round(test_data[col], 2)




## === cell 20
train_data.shape, test_data.shape




## === cell 21
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_data["fare_amount"] = train_data["fare_amount"].clip(lower=0, upper=200)

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

feature_cols = X.columns

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = Ridge(alpha=0.0, fit_intercept=True)
lr.fit(X_train_scaled, y_train)

val_pred = lr.predict(X_val_scaled)
val_pred = np.clip(val_pred, 0, None)

val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print("Validation RMSE:", val_rmse)




## === cell 22
test_X = test_data.drop("key", axis=1)

missing_cols = set(feature_cols) - set(test_X.columns)
for col in missing_cols:
    test_X[col] = 0
test_X = test_X[feature_cols]

test_features = scaler.transform(test_X)
test_pred = lr.predict(test_features)
pred = np.clip(test_pred, 0, None)




## === cell 23
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission.head()




## === cell 24
Submission.to_csv("Submission.csv", index=False)
