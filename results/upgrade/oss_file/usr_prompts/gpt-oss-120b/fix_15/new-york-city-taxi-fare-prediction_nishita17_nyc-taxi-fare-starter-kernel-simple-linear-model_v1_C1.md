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

5.34463

# 6. Current score

12.45622

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 871.63826) has done: 'Implemented fixes to resolve the LinearRegression initialization error and ensure feature alignment between training and test sets. Removed the deprecated `normalize` argument, added synchronization of test columns with the training feature set, and corrected the downstream prediction and submission creation steps. The script now runs end‑to‑end and produces a valid `Submission.csv` file ready for Kaggle upload.'
- What this solution (achieved 762.64408) has done: 'The changes standardize all numeric features (using train‑set statistics) and correctly convert the pickup‑time string into minutes‑since‑midnight.  Standardization is applied to both train and test data so the LinearRegression model sees comparable feature scales, which markedly reduces the RMSE and moves the score toward the target.  A small safety‑clip prevents negative fare predictions before writing the submission file.'
- What this solution (achieved 751.18326) has done: 'I remove the unnecessary log‑transform of the target and train the Ridge model directly on the fare amount. This keeps the same linear model but aligns the loss with the RMSE metric, which should lower the validation error and move the score toward the target 5.34 while preserving the overall pipeline. The only changes are in the target handling and the corresponding validation‑prediction calculations.'
- What this solution (achieved 750.21375) has done: 'I adjust the target handling to train the Ridge model directly on the fare amount (removing the log‑transform) and compute RMSE on the raw values. I also fix the submission write‑out so the “key” column is included as a regular column (not as the index). These minimal changes keep the overall pipeline intact while aligning the training loss with the competition RMSE metric and ensuring a valid CSV is produced.'
- What this solution (achieved 750.21375) has done: 'I replace the log‑target handling with a direct regression on the raw fare amount, keeping the same feature engineering and Ridge model. This aligns the loss with the RMSE metric, removes unnecessary exponentiation, and should move the validation score much closer to the target value while preserving the original pipeline structure.'
- What this solution (achieved 750.40153) has done: 'I replace the Ridge regression with an un‑regularized LinearRegression, which keeps the same linear‑model pipeline but often yields a lower RMSE for this dataset. The change is limited to the model‑instantiation line, preserving all previous feature engineering and preprocessing steps.'
- What this solution (achieved 12.45688) has done: 'I replace the plain LinearRegression with a lightly‑regularized Ridge model and clip predictions to a realistic fare range (0‑200). This keeps the overall pipeline unchanged while better matching the RMSE metric and should lower the validation error, moving the score toward the target.'
- What this solution (achieved 12.97334) has done: 'I keep the overall pipeline and feature engineering unchanged but modify the target handling: train the Ridge model on the log‑transformed fare amount and back‑transform predictions before computing RMSE and creating the submission. This aligns the loss with the skewed distribution of fares and is expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 12.45622) has done: 'I replace the log‑target Ridge regression with a plain LinearRegression that trains directly on the raw fare amount. This aligns the loss function with the RMSE metric, removes the unnecessary log/expm1 transforms, and should lower the validation error, moving the score closer to the target while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to
import matplotlib.pyplot as plt
import seaborn as sns
import time

for dirname, _, filenames in os.walk("../input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.head()



## === cell 2
train_df.shape



## === cell 3
train_df.info()



## === cell 4
test_data = pd.read_csv("../input/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_df.isna().sum()




## === cell 7
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_data)



## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 9
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 10
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 11
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1


ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 12
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1


ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1



## === cell 13
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 14
train_df["Weekday"].replace(
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



## === cell 15
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 16
train_df.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 17
ls1 = list(train_df["pickuptime"])
for i in range(len(ls1)):
    h, m = ls1[i].split(":")
    ls1[i] = int(h) * 60 + int(m)
train_df["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    h, m = ls1[i].split(":")
    ls1[i] = int(h) * 60 + int(m)
test_data["pickuptime"] = ls1



## === cell 18
R = 6373.0
lat1 = np.radians(train_df["pickup_latitude"])
lon1 = np.radians(train_df["pickup_longitude"])
lat2 = np.radians(train_df["dropoff_latitude"])
lon2 = np.radians(train_df["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.round(distance * 0.621, 2)

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.round(distance * 0.621, 2)



## === cell 19
R = 6373.0
lat1 = np.radians(train_df["pickup_latitude"])
lon1 = np.radians(train_df["pickup_longitude"])
lat2 = np.radians(train_df["dropoff_latitude"])
lon2 = np.radians(train_df["dropoff_longitude"])
lat3 = np.radians(40.6413111)
lon3 = np.radians(-73.7781391)

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_df["Pickup_Distance_airport"] = np.round(R * c1 * 0.621, 2)

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_df["Dropoff_Distance_airport"] = np.round(R * c2 * 0.621, 2)

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
test_data["Pickup_Distance_airport"] = np.round(R * c1 * 0.621, 2)

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
test_data["Dropoff_Distance_airport"] = np.round(R * c2 * 0.621, 2)



## === cell 20
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 21
train_df["Distance_x_Passenger"] = train_df["Distance"] * train_df["passenger_count"]
test_data["Distance_x_Passenger"] = test_data["Distance"] * test_data["passenger_count"]

train_df["pickup_sin"] = np.sin(2 * np.pi * train_df["pickuptime"] / 1440)
train_df["pickup_cos"] = np.cos(2 * np.pi * train_df["pickuptime"] / 1440)
test_data["pickup_sin"] = np.sin(2 * np.pi * test_data["pickuptime"] / 1440)
test_data["pickup_cos"] = np.cos(2 * np.pi * test_data["pickuptime"] / 1440)



## === cell 22
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ["fare_amount"]]

train_means = train_df[feature_cols].mean()
train_stds = train_df[feature_cols].std().replace(0, 1)

train_df[feature_cols] = (train_df[feature_cols] - train_means) / train_stds
test_data[feature_cols] = (test_data[feature_cols] - train_means) / train_stds

train_df = train_df[train_df["fare_amount"] > 0].copy()
train_df = train_df[train_df["fare_amount"] < 200].copy()

from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]  # raw fare amount

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)



## === cell 23
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)

val_pred = linreg.predict(X_val)
val_pred = np.clip(val_pred, 0, 200)

val_rmse = np.sqrt(((y_val - val_pred) ** 2).mean())
print(f"Validation RMSE (raw‑target LinearRegression): {val_rmse:.4f}")

test_features = test_data.reindex(columns=X.columns, fill_value=0)

test_pred = linreg.predict(test_features)
test_pred = np.clip(test_pred, 0, 200)

pred = np.round(test_pred, 2)



## === cell 24
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission.to_csv("Submission.csv", index=False)
