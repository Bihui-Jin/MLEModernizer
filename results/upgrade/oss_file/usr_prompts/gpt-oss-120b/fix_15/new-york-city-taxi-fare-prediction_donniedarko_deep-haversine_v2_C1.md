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
ipywidgets==8.1.5
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
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
tqdm==4.67.1

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

3.97307

# 6. Current score

4.92658

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.66763) has done: 'The fixes address three critical problems: the removed `weekofyear` attribute (replaced with `isocalendar().week`), incompatibilities between the standalone Keras package and TensorFlow 2 (switching to `tensorflow.keras` imports), and missing variable definitions caused by earlier errors. After these adjustments the script runs end‑to‑end, trains a simple model on a sampled subset, predicts the fares for the test set, and writes a correctly‑formatted `submission.csv` ready for Kaggle.'
- What this solution (achieved 6.31706) has done: 'The fix removes the problematic `ipywidgets` (and unused plotting) imports that caused the protobuf‑related crash, and it expands the training sample from 1 % to 2 % while training a bit longer (80 epochs) so the model learns from more data and reduces RMSE toward the target. No core architecture changes are made; only data loading and training duration are adjusted, and the script now writes a proper `submission.csv`.'
- What this solution (achieved 17.24234) has done: 'The script now sets the protobuf implementation before loading TensorFlow to avoid the import error, and the model’s output layer uses a linear activation (better for regression). No other logic is altered, so the core architecture and training remain the same while fixing the crash and improving the RMSE.'
- What this solution (achieved 5.27484) has done: 'I remove the TensorFlow/Keras parts that cause the protobuf import error and replace them with a lightweight scikit‑learn regression model (HistGradientBoostingRegressor). The data loading and cleaning logic stay the same, and the feature engineering (hour, weekday, day‑of‑year, week‑of‑year, year, passenger count) is kept. I also increase the training sample to 5 % to give the model more data without changing the overall pipeline. The script now runs end‑to‑end and writes a correct `submission.csv` while moving the RMSE toward the target.'
- What this solution (achieved 5.01998) has done: 'I increase the training sample size to 10 % and add a haversine distance feature to the engineered inputs, which usually helps fare‑prediction models. I also raise the boosting iterations slightly (to 300) so the richer data can be better fitted. These modest adjustments keep the original pipeline intact while aiming to lower the RMSE toward the target.'
- What this solution (achieved 5.34388) has done: 'I increase the training sample from 10 % to 20 % to give the model more data and adjust the gradient‑boosting hyper‑parameters (more iterations with a smaller learning‑rate) so the model can fit the richer dataset better. These modest changes keep the overall pipeline and feature set intact while aiming to lower the RMSE toward the target.'
- What this solution (achieved 5.36887) has done: 'I add a modest scaling of the haversine distance (divide by 100 km) to keep its magnitude comparable to the other features, and slightly fine‑tune the gradient‑boosting parameters (more trees, a smaller learning rate and a tiny L2 regularisation). These small adjustments keep the original pipeline intact while aiming to reduce the RMSE toward the target value.'
- What this solution (achieved 4.93695) has done: 'We log‑transform the target during training and invert the transform after prediction, which often improves regression on skewed fare amounts. Additionally, we slightly increase the number of boosting iterations and lower the learning rate to let the model fit the larger sample more accurately. These minimal adjustments keep the original pipeline intact while aiming to reduce RMSE toward the target.'
- What this solution (achieved 5.00906) has done: 'I increase the training data fraction from 20 % to 25 % to give the model more examples, and I let the gradient‑boosting model train a little longer with a smaller learning‑rate and a tiny L2 penalty (max_iter = 1500, learning_rate = 0.01, l2_regularization = 0.01). These modest adjustments keep the original pipeline intact while providing the model with more information and a finer‑grained fit, which should lower the RMSE toward the target.'
- What this solution (achieved 5.30036) has done: 'I replace the heavy chunk‑wise sampling with a direct read of a modest‑size slice of the training file using `nrows`. This avoids parsing the full 1.6 GB CSV while still providing a representative subset for the model, preserving the original preprocessing, feature engineering, and training logic. The rest of the cells remain unchanged, so the model architecture and training schedule are identical.'
- What this solution (achieved 4.92284) has done: 'I increase the training data fraction from 1 % to 3 % to give the model more examples while still keeping memory use reasonable, and I raise the number of boosting iterations to 3000 so the richer data can be learned more fully. These small adjustments keep the original preprocessing, model type, and loss unchanged, but should lower the RMSE toward the target.'
- What this solution (achieved 4.92658) has done: 'I add a small validation split and fit a linear correction → train on 90 % of the sampled data, compute a simple linear calibration on the held‑out 10 % (after converting the log‑target back to fare), then apply this calibration to the test predictions. This keeps the original model architecture while adjusting predictions toward the true scale, which should lower the RMSE toward the target. I also raise the boosting iterations modestly to give the model a bit more capacity.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




## === cell 1
def read_csv_sampled(path, frac, random_state=None):
    """
    Efficiently read a small random‑like subset of a large CSV.
    Instead of streaming the entire file, we read only the first
    ``n_rows`` rows where ``n_rows = int(total_rows * frac)``.
    The exact total size of the Kaggle taxi dataset is known (~55.4 M rows),
    so we use this constant to compute the required slice.
    The resulting DataFrame is then shuffled to mimic random sampling.
    """
    TOTAL_ROWS = 55_423_856
    n_rows = max(1, int(TOTAL_ROWS * frac))

    usecols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    dtypes = {
        "key": "string",
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    }

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["pickup_datetime"],
        nrows=n_rows,
    )

    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        df = df.sample(frac=1).reset_index(drop=True)

    return df




## === cell 2
df_train_sample = read_csv_sampled("../input/train.csv", 0.03, random_state=1989)




## === cell 3
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])




## === cell 4
df_train_sample.dropna(inplace=True)

is_weird = df_train_sample["fare_amount"] < 0
is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)
is_weird |= df_train_sample["passenger_count"] == 0
df_train_sample = df_train_sample[~is_weird]




## === cell 5
def prep_data(df, shuffle=False, random_state=None):
    """
    Returns feature matrix X and target vector y (if present).
    Adds haversine distance as an extra numerical feature.
    """
    X_cat = np.vstack(
        [
            df["pickup_datetime"].dt.hour,  # 0‑23
            df["pickup_datetime"].dt.weekday + 24,  # 24‑30
            df["pickup_datetime"].dt.dayofyear + 30,  # 31‑396
            df["pickup_datetime"].dt.isocalendar().week + 396,  # 397‑449
            df["pickup_datetime"].dt.year - 2009 + 450,  # 450‑456
            df["passenger_count"] + 456,  # 457‑463
        ]
    ).T.astype(np.float32)

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values.astype(np.float32)
    X_deg /= 180.0  # roughly [-1, 1]

    lat1 = np.deg2rad(df["pickup_latitude"].values.astype(np.float32))
    lon1 = np.deg2rad(df["pickup_longitude"].values.astype(np.float32))
    lat2 = np.deg2rad(df["dropoff_latitude"].values.astype(np.float32))
    lon2 = np.deg2rad(df["dropoff_longitude"].values.astype(np.float32))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = 6371.0 * c  # Earth radius in km
    dist = (dist / 100.0).astype(np.float32).reshape(-1, 1)

    X = np.concatenate([X_deg, dist, X_cat], axis=1)

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype(np.float32)
    else:
        y = None

    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        X = X[idx]
        if y is not None:
            y = y[idx]

    return X, y




## === cell 6
np.random.seed(1989)
X_all, y_all = prep_data(df_train_sample, shuffle=True, random_state=1989)
y_all = np.log1p(y_all)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.10, random_state=1989
)




## === cell 7
model = HistGradientBoostingRegressor(
    max_iter=3500,  # a bit more boosting iterations
    learning_rate=0.01,
    max_depth=8,
    random_state=1989,
    l2_regularization=0.01,
    loss="squared_error",
)

model.fit(X_train, y_train)




## === cell 8
val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val)

calibrator = LinearRegression()
calibrator.fit(val_pred.reshape(-1, 1), val_true)




## === cell 9
X_test, _ = prep_data(df_test, shuffle=False)




## === cell 10
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

test_pred_calibrated = calibrator.predict(test_pred.reshape(-1, 1))
y_test_pred = np.clip(test_pred_calibrated, 0, None)  # fare cannot be negative




## === cell 11
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test_pred})
df_sub.to_csv("submission.csv", index=False)
