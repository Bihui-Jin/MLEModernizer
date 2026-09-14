# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer as Imputer
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))


def chunck_generator(filename, chunk_size=10**6, usecols=None, dtypes=None):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
        usecols=usecols,
        dtype=dtypes,
        engine="c",
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    dlong = df["dropoff_longitude"].to_numpy() - df["pickup_longitude"].to_numpy()
    dlat = df["dropoff_latitude"].to_numpy() - df["pickup_latitude"].to_numpy()

    abs_diff_longitude = np.abs(dlong) * 50.0
    abs_diff_latitude = np.abs(dlat) * 69.0

    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    theta = np.arctan(abs_diff_longitude / abs_diff_latitude)

    actual_long = np.abs(displacement_vector * np.sin(theta - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(theta - alpha_ang))
    distance = actual_long + actual_lat

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = distance
    return df




## === cell 2
def data_clean(df):
    if "fare_amount" in df.columns:
        df["fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)

    m = df["passenger_count"].to_numpy() > 0

    if "fare_amount" in df.columns:
        m &= df["fare_amount"].to_numpy() > 0

    m &= df["distance_travel"].to_numpy() > 0

    return df.loc[m]




## === cell 3
def remove_outliers(df):
    m = (df["distance_travel"].to_numpy() < 30) & (df["fare_amount"].to_numpy() < 100)
    return df.loc[m]




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
filename = r"../input/train.csv"
chunk_size = 10**6

rng = np.random.RandomState(42)


def _fast_count_lines(path):
    try:
        import subprocess

        out = subprocess.check_output(["wc", "-l", path], stderr=subprocess.DEVNULL)
        return int(out.strip().split()[0])
    except Exception:
        n = 0
        with open(path, "rb") as f:
            for buf in iter(lambda: f.read(1024 * 1024), b""):
                n += buf.count(b"\n")
        return n


n_lines = _fast_count_lines(filename)
n_rows = max(n_lines - 1, 0)  # subtract header row
n_chunks_total = int(math.ceil(n_rows / float(chunk_size))) if n_rows else 0

n_chunks_to_use = 30
n_chunks_to_use = min(n_chunks_to_use, n_chunks_total)

chosen_chunks = (
    sorted(
        rng.choice(
            np.arange(n_chunks_total), size=n_chunks_to_use, replace=False
        ).tolist()
    )
    if n_chunks_total > 0 and n_chunks_to_use > 0
    else []
)
chosen_set = set(chosen_chunks)

print("Total chunks available:", n_chunks_total)
print("Chosen chunk indices:", chosen_chunks)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True)

imp = Imputer(missing_values=np.nan, strategy="mean")
final_imp = None

train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_dtypes = {
    "fare_amount": "float64",  # keep exact
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}

used = 0
for chunk_idx, df in enumerate(
    chunck_generator(
        filename, chunk_size=chunk_size, usecols=train_usecols, dtypes=train_dtypes
    )
):
    if chunk_idx not in chosen_set:
        continue

    print("Training on chunk:", chunk_idx)

    if df is None or len(df) == 0:
        used += 1
        continue

    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)

    l = len(df)
    if l < 1000:
        used += 1
        continue

    if "pickup_datetime" in df.columns:
        order = np.argsort(
            df["pickup_datetime"].to_numpy(), kind="mergesort"
        )  # stable, deterministic
        split = int(0.9 * l)
        train_idx = order[:split]
        test_idx = order[split:]
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
    else:
        split = int(0.9 * l)
        df = df.reset_index(drop=True)
        df_train = df.iloc[:split]
        df_test = df.iloc[split:]

    train_X = np.empty((len(df_train), 3), dtype=np.float64)
    train_X[:, 0] = df_train["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    train_X[:, 1] = df_train["passenger_count"].to_numpy(dtype=np.float64, copy=False)
    train_X[:, 2] = 1.0

    test_X = np.empty((len(df_test), 3), dtype=np.float64)
    test_X[:, 0] = df_test["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    test_X[:, 1] = df_test["passenger_count"].to_numpy(dtype=np.float64, copy=False)
    test_X[:, 2] = 1.0

    train_y = df_train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    test_y = df_test["fare_amount"].to_numpy(dtype=np.float64, copy=False)

    imp = imp.fit(train_X)
    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    regr = incremental_training(train_X, train_y, regr)
    print(regr.score(test_X, test_y))

    final_imp = imp
    used += 1

    if used >= len(chosen_chunks):
        break

if final_imp is None:
    final_imp = imp



## === cell 7
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
test_df = pd.read_csv(
    "../input/test.csv",
    usecols=test_usecols,
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
    engine="c",
)

test_df = distance_travel(test_df)
test_df.head()



## === cell 8
test_X = np.empty((len(test_df), 3), dtype=np.float64)
test_X[:, 0] = test_df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
test_X[:, 1] = test_df["passenger_count"].to_numpy(dtype=np.float64, copy=False)
test_X[:, 2] = 1.0

test_X = final_imp.transform(test_X)

predicted_fare = regr.predict(test_X)
predicted_fare = np.clip(predicted_fare, 0.0, None)

print(predicted_fare)



## === cell 9
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
