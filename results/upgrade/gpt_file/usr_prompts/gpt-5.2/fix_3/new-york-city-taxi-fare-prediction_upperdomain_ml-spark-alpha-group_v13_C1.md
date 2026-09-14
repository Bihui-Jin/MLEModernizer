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
scipy==1.15.3
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

4.22828

# 6. Current score

6.38888

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 6.38888) has done: 'The timeout is dominated by repeatedly fitting `GradientBoostingRegressor` on million-row chunks (and re-fitting the imputer each time), plus extra work from computing `distance_travel` twice per chunk and creating many temporary pandas objects. I keep the exact same model, training loop, features, and cleaning rules, but (1) stream the CSV with an explicit `usecols`/`dtype` plan to reduce parsing overhead, (2) compute `distance_travel` once per chunk using NumPy arrays (equivalent math, fewer pandas temporaries), (3) avoid redundant work in `data_clean`, and (4) fit the imputer once (it never changes because the third column is all ones and the other two have no NaNs after parsing) while still applying the same transformation semantics. These changes reduce constant factors substantially without changing the algorithmic logic or training semantics.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from sklearn.impute import SimpleImputer

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

print("Listing /kaggle/input:")
print(os.listdir("/kaggle/input"))

TRAIN_PATH = r"/kaggle/input/train.csv"
TEST_PATH = r"/kaggle/input/test.csv"

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN = {
    "fare_amount": "float64",  # keep exact as used later
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, header=False, chunk_size=10**6):
    if "train" in os.path.basename(filename):
        usecols = TRAIN_USECOLS
        dtypes = DTYPES_TRAIN
    else:
        usecols = TEST_USECOLS
        dtypes = DTYPES_TEST

    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy()
    plat = df["pickup_latitude"].to_numpy()
    dlon = df["dropoff_longitude"].to_numpy()
    dlat = df["dropoff_latitude"].to_numpy()

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0
    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    ang = np.arctan(abs_diff_longitude / abs_diff_latitude) - alpha_ang
    actual_long = np.abs(displacement_vector * np.sin(ang))
    actual_lat = np.abs(displacement_vector * np.cos(ang))
    dist = actual_long + actual_lat

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = dist
    return df




## === cell 2
def data_clean(df):
    df = df[df.passenger_count > 0]
    df["fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)
    df = df[df.fare_amount > 0]
    df = df[df.distance_travel > 0]
    return df




## === cell 3
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 4
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 5
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 6
filename = TRAIN_PATH
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True)

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp_fitted = False

t = 56

while t > 0:
    print("chunk remaining:", t)
    df = next(gen)

    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)

    l = len(df)
    if l < 10:
        t -= 1
        continue

    split = int(0.9 * l)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    train_dist = df_train["distance_travel"].to_numpy()
    train_pass = df_train["passenger_count"].to_numpy(dtype=np.float64, copy=False)
    train_X = np.empty((len(df_train), 3), dtype=np.float64)
    train_X[:, 0] = train_dist
    train_X[:, 1] = train_pass
    train_X[:, 2] = 1.0

    test_dist = df_test["distance_travel"].to_numpy()
    test_pass = df_test["passenger_count"].to_numpy(dtype=np.float64, copy=False)
    test_X = np.empty((len(df_test), 3), dtype=np.float64)
    test_X[:, 0] = test_dist
    test_X[:, 1] = test_pass
    test_X[:, 2] = 1.0

    train_y = df_train["fare_amount"].to_numpy()
    test_y = df_test["fare_amount"].to_numpy()

    if not imp_fitted:
        imp = imp.fit(train_X)
        imp_fitted = True

    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    regr = incremental_training(train_X, train_y, regr)

    print("R^2 on held-out chunk split:", regr.score(test_X, test_y))
    t -= 1



## === cell 7
test_df = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    engine="c",
    low_memory=False,
)

distance_travel(test_df)
test_df.head()



## === cell 8
test_dist = test_df["distance_travel"].to_numpy()
test_pass = test_df["passenger_count"].to_numpy(dtype=np.float64, copy=False)
test_X = np.empty((len(test_df), 3), dtype=np.float64)
test_X[:, 0] = test_dist
test_X[:, 1] = test_pass
test_X[:, 2] = 1.0

test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)

predicted_fare = np.maximum(predicted_fare, 0)

print(predicted_fare[:10])



## === cell 9
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
