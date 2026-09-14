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

3.74165

# 6. Current score

4.65163

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.53902) has done: 'I increase the sampled training size to give the model more data, add Euclidean and Manhattan distance features (which are informative for fare prediction), and compute the validation RMSE so we can see the improvement. These changes keep the RandomForest core unchanged while providing richer inputs that should lower the RMSE toward the target.'
- What this solution (achieved 4.52585) has done: 'I add more informative geographic and temporal features (haversine distance, month, weekday), include them in the input matrix, and slightly increase the forest size so the model can better capture patterns and lower the RMSE toward the target.'
- What this solution (achieved 4.36818) has done: 'I increase the training sample size from 50 000 to 100 000 rows to give the model more data, and I make the RandomForest slightly more expressive by using more trees (1 200) and allowing splits with as few as 2 samples. These modest hyper‑parameter tweaks keep the core model unchanged while expectedly lowering the validation RMSE, moving the score closer to the target without risking over‑fitting or large runtime increases.'
- What this solution (achieved 4.36846) has done: 'I increase the training sample to 150 000 rows so the model sees more data, and I make the RandomForest a bit stronger by using all tree depth, more trees (2000), and limiting each split to a sqrt‑sized feature subset. These modest hyper‑parameter tweaks keep the original RandomForest pipeline intact while expectedly lowering the validation RMSE toward the target. I also renumber the cells so they start at 1 as required.'
- What this solution (achieved 4.27882) has done: 'The changes focus on speeding up the RandomForest training, which is the main bottleneck. By lowering the number of trees, using the default “sqrt” feature subset per split, and disabling the costly out‑of‑bag score (which isn’t used later), the model trains far faster while keeping the same feature set and overall algorithmic approach, so the predictions remain effectively unchanged.'
- What this solution (achieved 4.42551) has done: 'The changes add Intel’s sklearnex patch to accelerate the RandomForest training without altering the model or its parameters, and ensure target arrays are stored as float32 for faster computation. These adjustments keep all feature engineering and evaluation steps identical, preserving the original logic and accuracy while cutting runtime.'
- What this solution (achieved 4.65163) has done: 'The changes keep the same feature engineering and RandomForest model but dramatically cut runtime by sampling fewer rows (still a representative subset) and by avoiding costly pandas‑DataFrame copies after imputation—training now uses pure NumPy arrays, which the sklearn forest implementation consumes directly. These tweaks preserve the exact preprocessing logic and model hyper‑parameters, so predictions remain unchanged apart from negligible floating‑point differences, while the script now finishes well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer



## === cell 1
n_train = 200_000  # reduced from 800_000 for faster training
dtype_map = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df = pd.read_csv(
    "../input/train.csv",
    nrows=n_train,
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
    usecols=train_usecols,
)

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
df_test = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    dtype=dtype_map,
    usecols=test_usecols,
)




## === cell 2
def add_features(df):
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)

    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["haversine_km"] = (earth_radius_km * c).astype(np.float32)

    df["euclidean_km"] = np.sqrt(
        (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
        + (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
    ).astype(np.float32)

    df["manhattan_km"] = (
        np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
        + np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
    ).astype(np.float32)

    return df


df = add_features(df)
df_test = add_features(df_test)

df = df.drop(columns=["pickup_datetime"])
df_test = df_test.drop(columns=["pickup_datetime"])



## === cell 3
X = df.drop(columns=["fare_amount", "key"])
y = df["fare_amount"].astype(np.float32).values  # keep as ndarray

imputer = SimpleImputer(strategy="median")
X_imputed_arr = imputer.fit_transform(X)  # already float32

X_test_raw = df_test.drop(columns=["key"])
X_test_imputed_arr = imputer.transform(X_test_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X_imputed_arr, y, test_size=0.2, random_state=42
)



## === cell 4
reg = RandomForestRegressor(
    max_depth=None,
    n_estimators=2000,
    max_features="sqrt",
    min_samples_split=2,
    oob_score=False,
    n_jobs=-1,
    verbose=0,
    random_state=42,
)



## === cell 5
reg.fit(X_train, y_train)

val_pred = reg.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.5f}")



## === cell 6
test_pred = reg.predict(X_test_imputed_arr)
submission = pd.DataFrame({"key": df_test["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
