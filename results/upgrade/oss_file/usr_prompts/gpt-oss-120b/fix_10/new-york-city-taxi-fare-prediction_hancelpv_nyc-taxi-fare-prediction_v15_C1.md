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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

3.94377

# 6. Current score

4.93511

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.00398) has done: 'We replace the expensive one‑hot encoding with pandas categorical codes, which dramatically reduces the number of feature columns while keeping the same information. By converting the categorical time columns to `category` dtype and using their integer codes, the RandomForest sees far fewer features, cutting both memory use and tree‑building time without altering the model type or its hyper‑parameters. All other steps stay unchanged, preserving the original logic and result accuracy.'
- What this solution (achieved 5.00859) has done: 'The changes keep the exact same preprocessing, feature set, and model type, but eliminate unnecessary data copies and speed up the RandomForest training by using the `max_samples` and `max_features` parameters (still a standard RandomForest) and by passing NumPy arrays directly to the fit method. These tweaks reduce per‑tree work while preserving the algorithm’s semantics, keeping results deterministic with the same random seed.'
- What this solution (achieved 5.07749) has done: 'The changes reduce the amount of data read and the number of trees in the RandomForest, which are the dominant contributors to runtime, while keeping the same preprocessing, feature engineering, and model type so the prediction logic remains unchanged.'
- What this solution (achieved 4.93511) has done: 'The changes speed up the pipeline by eliminating slow Python‑level mapping in the time‑feature generation (using pure pandas vectorized operations for weekday and week‑of‑month) and by lowering the RandomForest tree count from 500 to 200, which keeps the same model type while cutting training time roughly in half. All other logic, feature engineering, and data handling remain unchanged, so the predictions stay deterministic and comparable.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]




## === cell 2
train = pd.read_csv("../input/train.csv", nrows=1_000_000, usecols=cols, dtype=types)
test = pd.read_csv("../input/test.csv")
samp = pd.read_csv("../input/sample_submission.csv")

train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train["passenger_count"] <= 6]
train = train[train.fare_amount < 200]

latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]

longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]




## === cell 3
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(["fare_amount"], axis=1, inplace=True)
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key




## === cell 5
def add_time_features(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

    data["hour"] = data["pickup_datetime"].dt.hour.astype(np.int8)
    data["day_of_week"] = data["pickup_datetime"].dt.weekday.astype(np.int8)  # 0=Mon
    day = data["pickup_datetime"].dt.day.astype(np.int8)

    data["week_of_month"] = ((day - 1) // 7).astype(np.int8)

    data["month"] = data["pickup_datetime"].dt.month.astype(np.int8)
    data["year"] = data["pickup_datetime"].dt.year.astype(np.int16)

    return data




## === cell 6
def add_geo_features(data):
    data["diff_longitude"] = data["dropoff_longitude"] - data["pickup_longitude"]
    data["diff_latitude"] = data["dropoff_latitude"] - data["pickup_latitude"]

    data["abs_diff_longitude"] = data["diff_longitude"].abs()
    data["abs_diff_latitude"] = data["diff_latitude"].abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]
    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)
    data["euclid_distance"] = np.sqrt(data["squared_long"] + data["squared_lat"])

    R = 6371.0  # Earth radius in km
    lat1 = np.radians(data.pickup_latitude)
    lon1 = np.radians(data.pickup_longitude)
    lat2 = np.radians(data.dropoff_latitude)
    lon2 = np.radians(data.dropoff_longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    data["haversine_distance"] = R * 2 * np.arcsin(np.sqrt(a))

    return data




## === cell 7
all_data = add_time_features(all_data)




## === cell 8
all_data = add_geo_features(all_data)




## === cell 9
categorical_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
for col in categorical_cols:
    all_data[col] = all_data[col].astype("category").cat.codes.astype(np.int16)

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "diff_longitude",
    "diff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "euclid_distance",
    "haversine_distance",
]

all_data = all_data[features].astype(np.float32)




## === cell 10
x = all_data[:n_train]
x_test = all_data[n_train:]




## === cell 11
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(
    n_estimators=200,  # reduced from 500 to cut training time
    random_state=42,
    n_jobs=-1,
    max_samples=1.0,
)




## === cell 12
x_np = x.to_numpy(dtype=np.float32)
y_log = np.log1p(y).astype(np.float32)
model_1.fit(x_np, y_log)




## === cell 13
model_1_pred = np.expm1(model_1.predict(x_test.values))
model_1_pred = np.clip(model_1_pred, 0, None)  # fares cannot be negative




## === cell 14
submission = pd.DataFrame({"key": test_id, "fare_amount": model_1_pred})
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written successfully.')
