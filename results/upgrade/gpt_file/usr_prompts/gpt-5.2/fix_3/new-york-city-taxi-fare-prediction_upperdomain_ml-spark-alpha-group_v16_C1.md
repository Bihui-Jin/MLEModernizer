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

5.57567

# 6. Current score

991.63001

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 962.65062) has done: 'I fix the immediate runtime/import errors by replacing the removed `sklearn.preprocessing.Imputer` with `sklearn.impute.SimpleImputer` and by removing notebook-only `%matplotlib inline` so the script runs as a .py in Kaggle. I also correct a feature mismatch bug where `train_X` and `test_X` used different latitude/longitude columns, which would break evaluation and hurt RMSE; this keeps the same linear-regression core logic but makes it consistent. Finally, I make paths robust for both `../input/...` and the provided `/kaggle/input/...` layout, ensure the model and imputer are always defined, and write a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 991.63001) has done: 'Your RMSE is extremely high because the model is almost certainly training on garbage rows (invalid lat/long ranges, swapped coordinates, and other known NYC taxi dataset issues), and because it’s using an inconsistent/weak feature set (only longitudes, not latitudes). To move your score sharply toward the target while keeping the same core logic (same distance feature, same LinearRegression, same imputer, same chunked training loop), I (1) add minimal, standard coordinate-range filters in `data_clean`, (2) include both pickup/dropoff latitude alongside longitude in train/test features, and (3) ensure the same preprocessing and imputation is applied consistently. These are small, legitimate fixes that don’t change the training approach, but they should dramatically reduce RMSE from the current 962 toward the expected single-digit range.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]
INPUT_DIR = next((p for p in INPUT_DIR_CANDIDATES if os.path.exists(p)), "../input")


def _resolve_input_file(fname: str) -> str:
    p1 = os.path.join(INPUT_DIR, fname)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", fname)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join("/kaggle/data", fname)
    if os.path.exists(p3):
        return p3
    p4 = os.path.join("/kaggle/data", "new-york-city-taxi-fare-prediction", fname)
    if os.path.exists(p4):
        return p4
    return os.path.join("../input", fname)


try:
    print("Listing input dir:", INPUT_DIR)
    print(os.listdir(INPUT_DIR)[:50])
except Exception as e:
    print("Could not list input dir:", e)




## === cell 1
def chunck_generator(filename, header=False, chunk_size=10**6):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=[1],
    ):
        yield chunk




## === cell 2
alpha_ang = 0.506


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5
    denom = df.abs_diff_latitude.replace(0, np.nan)
    angle = np.arctan(df.abs_diff_longitude / denom)
    df["actual_long"] = (df.displacement_vector * np.sin(angle - alpha_ang)).abs()
    df["actual_lat"] = (df.displacement_vector * np.cos(angle - alpha_ang)).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat
    df["distance_travel"] = df["distance_travel"].fillna(0.0)
    return df




## === cell 3
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]

    df = df[df.pickup_longitude.between(-180.0, 180.0)]
    df = df[df.dropoff_longitude.between(-180.0, 180.0)]
    df = df[df.pickup_latitude.between(-90.0, 90.0)]
    df = df[df.dropoff_latitude.between(-90.0, 90.0)]

    df = df[df.pickup_longitude.between(-75.0, -72.0)]
    df = df[df.dropoff_longitude.between(-75.0, -72.0)]
    df = df[df.pickup_latitude.between(40.0, 42.0)]
    df = df[df.dropoff_latitude.between(40.0, 42.0)]

    distance_travel(df)
    df = df[df.distance_travel > 0]
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 60]
    return df




## === cell 5
def graph_present(df):
    test = df[df.passenger_count == 1]
    _ = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def data_preprocessing(df):
    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)
    return df




## === cell 7
try:
    train_path = _resolve_input_file("train.csv")
    df = pd.read_csv(train_path, nrows=100_000)
    df = distance_travel(df)
    df = df[df.passenger_count == 1]
    df = df[df.distance_travel < 30]
    df.distance_travel.hist(bins=50, figsize=(12, 4))
    plt.xlabel("distance miles")
    plt.title("Histogram ride distances in miles")
    plt.show()
    print(df.distance_travel.describe())
except Exception as e:
    print("EDA cell skipped due to:", e)



## === cell 8
try:
    ax = plt.axes(projection="3d")
    df = pd.read_csv(train_path, nrows=100_000)
    df = distance_travel(df)
    df = df[df.passenger_count <= 6]
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount > 0]
    df = df[df.fare_amount < 60]
    ax.scatter3D(
        df.passenger_count,
        df.distance_travel,
        df.fare_amount,
        c=df.distance_travel,
        cmap="Greens",
    )
    ax.set_xlabel("passenger_count")
    ax.set_ylabel("distance_travel")
    ax.set_zlabel("fare_amount")
    plt.show()
except Exception as e:
    print("3D plot cell skipped due to:", e)



## === cell 9
try:
    df = pd.read_csv(train_path, nrows=100_000)
    df = distance_travel(df)
    df = df[df.passenger_count <= 6]
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount > 0]
    df = df[df.fare_amount < 60]
    df.fare_amount.hist(bins=50, figsize=(12, 4))
    plt.xlabel("fare_amount")
    plt.show()
    print(df.fare_amount.describe())
except Exception as e:
    print("Fare histogram cell skipped due to:", e)



## === cell 10
filename = _resolve_input_file("train.csv")
gen = chunck_generator(filename=filename)

linear_regr = LinearRegression(copy_X=True, n_jobs=10)
imp = SimpleImputer(missing_values=np.nan, strategy="mean")

t = 1
max_chunks = 56  # preserve original loop count intent

while t <= max_chunks:
    print("Chunk", t)
    df = next(gen)
    df = data_preprocessing(df)

    l = len(df)
    if l < 10:
        print("Chunk too small after cleaning; skipping fit/score")
        t += 1
        continue

    df_train = df.iloc[: int(0.9 * l)]
    df_test = df.iloc[int(0.9 * l) :]

    train_X = np.column_stack(
        (
            df_train.distance_travel.to_numpy(),
            df_train.passenger_count.to_numpy(),
            df_train.pickup_longitude.to_numpy(),
            df_train.pickup_latitude.to_numpy(),
            df_train.dropoff_longitude.to_numpy(),
            df_train.dropoff_latitude.to_numpy(),
            np.ones(len(df_train), dtype=np.float64),
        )
    )
    test_X = np.column_stack(
        (
            df_test.distance_travel.to_numpy(),
            df_test.passenger_count.to_numpy(),
            df_test.pickup_longitude.to_numpy(),
            df_test.pickup_latitude.to_numpy(),
            df_test.dropoff_longitude.to_numpy(),
            df_test.dropoff_latitude.to_numpy(),
            np.ones(len(df_test), dtype=np.float64),
        )
    )

    train_y = df_train.fare_amount.to_numpy(dtype=np.float64)
    test_y = df_test.fare_amount.to_numpy(dtype=np.float64)

    imp = imp.fit(train_X)
    train_X_imp = imp.transform(train_X)
    test_X_imp = imp.transform(test_X)

    linear_regr.fit(train_X_imp, train_y)
    print("LinearRegressor R^2", linear_regr.score(test_X_imp, test_y))

    t += 1

regr = linear_regr



## === cell 11
test_path = _resolve_input_file("test.csv")
test_df = pd.read_csv(test_path, nrows=10_000_000)
distance_travel(test_df)
print(test_df.head())



## === cell 12
test_X = np.column_stack(
    (
        test_df.distance_travel.to_numpy(),
        test_df.passenger_count.to_numpy(),
        test_df.pickup_longitude.to_numpy(),
        test_df.pickup_latitude.to_numpy(),
        test_df.dropoff_longitude.to_numpy(),
        test_df.dropoff_latitude.to_numpy(),
        np.ones(len(test_df), dtype=np.float64),
    )
)
test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)

predicted_fare = np.clip(predicted_fare, 0.0, None)

print(predicted_fare[:10])
print("Mean predicted fare:", float(np.mean(predicted_fare)))



## === cell 13
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
