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

3.8

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

3.30424

# 6. Current score

360.04832

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 461.38722) has done: 'I make the notebook run end-to-end by fixing the `df.corr()` crash (ensure we only correlate numeric columns) and the Keras import crash (use `tf_keras`, which is installed and compatible in this environment). To improve RMSE toward your target without changing the core model/training loop, I fix a major scaling bug: the current code scales train and test independently, which breaks feature consistency and hurts score. I instead fit scaling on the training features only and apply the same parameters to validation and test, keeping the same “scale” approach and the same network architecture/epochs. I also ensure predictions are written as a 1D float column aligned with the sample submission keys.'
- What this solution (achieved 577.39877) has done: 'I fix the crash in the Keras/TensorFlow stack by avoiding the incompatible `tf_keras` import that triggers the protobuf `MessageFactory.GetPrototype` error, and instead use scikit-learn’s `MLPRegressor` to keep the same core “multi-layer MLP trained with MSE” approach and similar layer sizes. I keep your existing feature engineering and filtering intact, and ensure scaling is fit on the training split and applied consistently to validation and test. I also ensure predictions are 1D floats and written to `submission.csv` with the required `key,fare_amount` columns aligned to the sample submission. This should run end-to-end in the Kaggle environment and drastically reduce RMSE toward your target.'
- What this solution (achieved 360.04832) has done: 'Your RMSE is extremely worse than the target, so we need a small but meaningful correction that keeps your same feature set and MLP training approach. The biggest issue here is that `MLPRegressor(max_iter=16)` is **not equivalent** to “16 epochs”; it’s only 16 optimizer iterations and severely underfit, producing terrible predictions. I keep the same model type and hidden layers, but raise `max_iter` to a reasonable value (while still well within the 600s budget for 500k rows) and enable deterministic convergence via `tol`/`n_iter_no_change` (without early stopping). I also clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE on this competition without changing core semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    parse_dates=["pickup_datetime"],
    nrows=500000,
)
df.dropna(inplace=True)



## === cell 2
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
test



## === cell 3
df.describe()



## === cell 4
test.describe()



## === cell 5
nyc_min_longitude = -74.3
nyc_max_longitude = -72
nyc_min_latitude = 40.63
nyc_max_latitude = 42



## === cell 6
for long in ["pickup_longitude", "dropoff_longitude"]:
    df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]
for lat in ["pickup_latitude", "dropoff_latitude"]:
    df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]



## === cell 7
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1



## === cell 8
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)]



## === cell 9
df.describe()




## === cell 10
def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


df["travel_distance"] = euc_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["travel_distance"] = euc_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)



## === cell 11
df.select_dtypes(include=[np.number]).corr()



## === cell 12
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["hour"] = df["pickup_datetime"].dt.hour
df = df.drop(["pickup_datetime"], axis=1)



## === cell 13
df.select_dtypes(include=[np.number]).corr()



## === cell 14
test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["hour"] = test["pickup_datetime"].dt.hour
test = test.drop(["pickup_datetime"], axis=1)



## === cell 15
test_key = test["key"].copy()
test.drop(["key"], axis=1, inplace=True)
df.drop(["key"], axis=1, inplace=True)



## === cell 16
print(df.isnull().sum())
print(test.isnull().sum())



## === cell 17
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor



## === cell 18
X = df.loc[:, df.columns != "fare_amount"]
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)



## === cell 19
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=200,  # was 16; 16 is far too small and leads to catastrophic underfit
    tol=1e-4,  # deterministic convergence criterion
    n_iter_no_change=20,  # used with tol; early_stopping stays False
    shuffle=True,
    random_state=42,
    early_stopping=False,  # do not introduce early stopping (constraint)
    verbose=True,
)
model.fit(X_train_scaled, y_train)

train_pred = model.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_pred = model.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Val RMSE: {:0.2f}".format(val_rmse))
print("------------------------")



## === cell 20
pred = model.predict(test_scaled)



## === cell 21
pred = np.asarray(pred).reshape(-1).astype(float)
pred = np.clip(pred, 0.0, None)
pred.shape



## === cell 22
submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission["fare_amount"] = pred
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
