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

164.7255

# 6. Current score

5067.04403

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5067.04403) has done: 'The fix removes the erroneous overwrite of `df`, updates the `LinearRegression` initialization to the current scikit‑learn API, and corrects the test‑set preprocessing (weekday extraction and one‑hot encoding). These changes allow the pipeline to run end‑to‑end and generate a proper `submission.csv` file without altering the original modeling approach.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
df = pd.read_csv(train_path, nrows=10_000_000)  # sample for faster iteration
df.dropna(inplace=True)



## === cell 2
df["manhattan_distance"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs() + (
    df["dropoff_longitude"] - df["pickup_longitude"]
).abs()
df["Difference_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
df["Difference_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()



## === cell 3
pick_time = df["pickup_datetime"].str[11:-7]  # extract HH:MM:SS
pick_time = pick_time.str.split(":").apply(lambda x: int(x[0]) * 100 + int(x[1]))
df["pickuptime"] = pick_time

weekday = pd.to_datetime(df["pickup_datetime"]).dt.weekday.values.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
weekday_onehot = encoder.fit_transform(weekday)
days = ["mon", "tue", "wed", "thr", "fri", "sat", "sun"]
for i, day in enumerate(days):
    df[day] = weekday_onehot[:, i]



## === cell 4
R = 6373.0
lat1 = np.radians(df["pickup_latitude"].values)
lon1 = np.radians(df["pickup_longitude"].values)
lat2 = np.radians(df["dropoff_latitude"].values)
lon2 = np.radians(df["dropoff_longitude"].values)

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
df["Distance"] = (R * c) * 0.621  # convert km to miles

lat_air = np.radians(40.6413111)
lon_air = np.radians(-73.7781391)

a1 = (
    np.sin((lat_air - lat1) / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin((lon_air - lon1) / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
df["Pickup_Distance_airport"] = (R * c1) * 0.621

a2 = (
    np.sin((lat_air - lat2) / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin((lon_air - lon2) / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
df["Dropoff_Distance_airport"] = (R * c2) * 0.621



## === cell 5
for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    df[col] = df[col].round(2)

df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 6
for col in ["Difference_longitude", "Difference_latitude"]:
    df[col] = (df[col] - df[col].mean()).abs()
    df[col] = df[col] / df[col].var()



## === cell 7
feature_cols = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1).columns
X = df[feature_cols]
y = df["fare_amount"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)



## === cell 8
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Validation R^2:", lr.score(X_val, y_val))



## === cell 9
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
test_data = pd.read_csv(test_path)
test_data.dropna(inplace=True)

test_data["manhattan_distance"] = (
    test_data["dropoff_latitude"] - test_data["pickup_latitude"]
).abs() + (test_data["dropoff_longitude"] - test_data["pickup_longitude"]).abs()
test_data["Difference_longitude"] = (
    test_data["dropoff_longitude"] - test_data["pickup_longitude"]
).abs()
test_data["Difference_latitude"] = (
    test_data["dropoff_latitude"] - test_data["pickup_latitude"]
).abs()

pick_time_test = test_data["pickup_datetime"].str[11:-7]
pick_time_test = pick_time_test.str.split(":").apply(
    lambda x: int(x[0]) * 100 + int(x[1])
)
test_data["pickuptime"] = pick_time_test

weekday_test = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday.values.reshape(
    -1, 1
)
weekday_onehot_test = encoder.transform(weekday_test)
for i, day in enumerate(days):
    test_data[day] = weekday_onehot_test[:, i]

lat1_t = np.radians(test_data["pickup_latitude"].values)
lon1_t = np.radians(test_data["pickup_longitude"].values)
lat2_t = np.radians(test_data["dropoff_latitude"].values)
lon2_t = np.radians(test_data["dropoff_longitude"].values)

dlon_t = lon2_t - lon1_t
dlat_t = lat2_t - lat1_t
a_t = (
    np.sin(dlat_t / 2) ** 2 + np.cos(lat1_t) * np.cos(lat2_t) * np.sin(dlon_t / 2) ** 2
)
c_t = 2 * np.arctan2(np.sqrt(a_t), np.sqrt(1 - a_t))
test_data["Distance"] = (R * c_t) * 0.621

a1_t = (
    np.sin((lat_air - lat1_t) / 2) ** 2
    + np.cos(lat1_t) * np.cos(lat_air) * np.sin((lon_air - lon1_t) / 2) ** 2
)
c1_t = 2 * np.arctan2(np.sqrt(a1_t), np.sqrt(1 - a1_t))
test_data["Pickup_Distance_airport"] = (R * c1_t) * 0.621

a2_t = (
    np.sin((lat_air - lat2_t) / 2) ** 2
    + np.cos(lat2_t) * np.cos(lat_air) * np.sin((lon_air - lon2_t) / 2) ** 2
)
c2_t = 2 * np.arctan2(np.sqrt(a2_t), np.sqrt(1 - a2_t))
test_data["Dropoff_Distance_airport"] = (R * c2_t) * 0.621

for col in ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]:
    test_data[col] = test_data[col].round(2)

test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)

for col in ["Difference_longitude", "Difference_latitude"]:
    test_data[col] = (test_data[col] - test_data[col].mean()).abs()
    test_data[col] = test_data[col] / test_data[col].var()



## === cell 10
X_test_final = test_data[feature_cols]  # ensure same column order as training features
pred = np.round(lr.predict(X_test_final), 2)

submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
