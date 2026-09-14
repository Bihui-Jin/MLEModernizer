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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.3735

# 6. Current score

8.42807

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.99934) has done: 'The main issues were a TensorFlow import error and a scaling failure caused by leftover string columns. I replaced the TensorFlow model with a scikit‑learn GradientBoostingRegressor, ensured the feature pipeline runs before scaling, and added proper handling for the test set. The script now loads the data, cleans and engineers features, scales them, trains the model, evaluates RMSE on a validation split, makes predictions for the Kaggle test set, and writes a correctly formatted `submissiontry_water.csv` file.'
- What this solution (achieved 5.98109) has done: 'Implemented fixes to resolve file path issues, datatype typo, cleaning logic for test data, and conditional handling of target‑specific filters. Updated paths to point to the correct Kaggle input directory, corrected the dtype dictionary, made the cleaning function safe for datasets without the `fare_amount` column, and avoided cleaning the test set to keep all keys aligned for submission. The script now runs end‑to‑end and outputs a valid CSV submission.'
- What this solution (achieved 8.42807) has done: 'The script was missing all imports and the helper functions used for cleaning, feature engineering, and writing the submission. I added the necessary `pandas`, `numpy`, and `sklearn` imports, defined `clean`, `add_time_features`, `add_coordinate_features`, `add_distances_features`, and `output_submission`, and reordered the steps so the data is loaded, processed, scaled, modeled, evaluated, and finally saved to a correctly‑named CSV file. This restores full functionality and produces a valid submission that can be evaluated toward the target RMSE.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"
DATASET_SIZE = 80000  # subset for quick run




## === cell 2
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    lon_min, lon_max = -80.0, -70.0
    lat_min, lat_max = 40.0, 42.5
    mask = (
        (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] < 500)
    )
    return df.loc[mask].reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_weekday"] = dt.dt.weekday
    df["pickup_month"] = dt.dt.month
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_lat_diff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["pickup_lon_diff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    df["haversine_km"] = haversine_vectorized(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def output_submission(
    raw_test: pd.DataFrame,
    prediction: np.ndarray,
    id_column: str,
    prediction_column: str,
    file_name: str,
):
    submission = raw_test[[id_column]].copy()
    submission[prediction_column] = prediction
    submission.to_csv(file_name, index=False)




## === cell 3
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)
testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## === cell 4
train_df, val_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)



## === cell 5
train_df = clean(train_df)
val_df = clean(val_df)

for df in (train_df, val_df, testKaggle):
    df = add_time_features(df)
    df = add_coordinate_features(df)
    df = add_distances_features(df)




## === cell 6
drop_cols = [
    "key",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
X_train = train_df.drop(columns=drop_cols + ["fare_amount"])
y_train = train_df["fare_amount"].values

X_val = val_df.drop(columns=drop_cols + ["fare_amount"])
y_val = val_df["fare_amount"].values

test_keys = testKaggle["key"].copy()
X_test = testKaggle.drop(columns=drop_cols)



## === cell 7
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



## === cell 8
gbr = GradientBoostingRegressor(
    n_estimators=1200,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    loss="squared_error",
)
gbr.fit(X_train_scaled, y_train)



## === cell 9
val_pred = gbr.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 10
test_pred = gbr.predict(X_test_scaled)



## === cell 11
output_submission(
    raw_test=pd.DataFrame({"key": test_keys}),
    prediction=test_pred,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)
