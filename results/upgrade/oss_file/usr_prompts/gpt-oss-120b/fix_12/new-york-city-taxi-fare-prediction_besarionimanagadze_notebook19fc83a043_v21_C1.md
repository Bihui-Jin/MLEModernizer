# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.12

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

5.54032

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 961.61623) has done: 'I keep the overall pipeline unchanged but remove the log‑transform that was applied to the target, fitting the Ridge model directly on the fare amounts. This typically yields a lower RMSE on the original scale, moving the validation score closer to the target (while still preserving the same features and model type). I also add a simple clipping of negative predictions before writing the submission.'
- What this solution (achieved 961.61623) has done: 'I removed the unnecessary log‑transform of the target so the Ridge model is trained and evaluated directly on the original fare amounts. The training/validation split now uses the raw y values, the validation RMSE is computed without exponentiation, and the test‑time predictions are generated without applying np.expm1. The rest of the pipeline (features, scaling, model) stays unchanged, and the script still writes a valid submission.csv with clipped non‑negative predictions.'
- What this solution (achieved 56.35472) has done: 'I add a simple outlier‑removal step that drops rows with an unrealistically high `fare_amount` (greater than 500 USD). Extremely large fares heavily inflate the RMSE, so eliminating them brings the validation error much closer to the target while keeping the original feature set, model, and pipeline unchanged. The change is tiny – just an extra condition in the existing cleaning cell – and the rest of the script still writes the required `submission.csv`.'
- What this solution (achieved 56.34434) has done: 'I lower the regularisation strength of the Ridge model (change alpha from 20.0 to 0.1). This keeps the same linear‑model pipeline but lets the model fit the data more closely, which should reduce the RMSE and move the score toward the target 5.54032.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra and log transformations
import pandas as pd  # data processing, CSV I/O
import matplotlib.pyplot as plt  # plotting
from sklearn.linear_model import Ridge  # kept for reference but not used
from sklearn.ensemble import GradientBoostingRegressor  # new model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

BASE_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction"

for fname in ["train.csv", "test.csv", "sample_submission.csv"]:
    print(os.path.join(BASE_PATH, fname))



## === cell 1
train_data_set = pd.read_csv(
    os.path.join(BASE_PATH, "train.csv"),
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
)
train_data_set.head(5)



## === cell 2
print(train_data_set.dtypes)
train_data_set.describe()



## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set[
    (train_data_set.fare_amount >= 0.1) & (train_data_set.fare_amount <= 500)
]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} entities from the dataset")
train_data_set.describe()



## === cell 4
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} entities from the dataset")



## === cell 5
train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")
plt.show()




## === cell 6
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} entities from the dataset")




## === cell 7
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)
train_data_set.head(5)



## === cell 8
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = (
    ((train_data_set["hour"] >= 7) & (train_data_set["hour"] <= 10))
    | ((train_data_set["hour"] >= 16) & (train_data_set["hour"] <= 19))
).astype(np.int8)
train_data_set.head(5)



## === cell 9
nyc_down_town = (-74.0063889, 40.7141667)  # NYC downtown coordinates

train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
)
train_data_set.head(5)



## === cell 10
idx = train_data_set["passenger_count"] != 0

features = [
    "hour",
    "year",
    "distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "distance_to_downtown",
]
target = "fare_amount"

X = train_data_set.loc[idx, features].values
y = train_data_set.loc[idx, target].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    presort=False,  # <-- faster training without changing model semantics
)
gb_model.fit(X_train, y_train)

y_val_pred = gb_model.predict(X_val)
val_rmse = np.sqrt(((y_val_pred - y_val) ** 2).mean())
print(f"Validation RMSE (original scale): {val_rmse:.5f}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2110698418.py in <cell line: 0>()
     17 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
     18 
---> 19 gb_model = GradientBoostingRegressor(
     20     n_estimators=300,
     21     learning_rate=0.05,

TypeError: GradientBoostingRegressor.__init__() got an unexpected keyword argument 'presort'

## === cell 11
test_data_set = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)
test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
)

test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = (
    ((test_data_set["hour"] >= 7) & (test_data_set["hour"] <= 10))
    | ((test_data_set["hour"] >= 16) & (test_data_set["hour"] <= 19))
).astype(np.int8)



## === cell 12
XTEST = test_data_set[features].values

y_pred = gb_model.predict(XTEST)

y_pred = np.clip(y_pred, a_min=0, a_max=500)

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2213374544.py in <cell line: 0>()
      1 XTEST = test_data_set[features].values
      2 
----> 3 y_pred = gb_model.predict(XTEST)
      4 
      5 y_pred = np.clip(y_pred, a_min=0, a_max=500)

NameError: name 'gb_model' is not defined
