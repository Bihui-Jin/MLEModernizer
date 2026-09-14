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
seaborn==0.12.2
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

4.12819

# 6. Current score

6.06953

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.6597) has done: 'The fix adds the missing `is_night_time` feature (and fills possible missing drop‑off coordinates) to the test set, prevents NaN‑related errors, and ensures predictions are non‑negative before writing the submission file.'
- What this solution (achieved 7.02461) has done: 'The changes speed up the notebook by removing heavyweight plotting, vectorizing the night‑time flag creation, and feeding NumPy arrays directly to the scikit‑learn models. These adjustments keep the same preprocessing, feature set, and model hyper‑parameters, so the prediction logic and accuracy remain unchanged.'
- What this solution (achieved 6.07375) has done: 'The update speeds up the pipeline by vectorizing the public‑holiday flag, fixing deterministic random seeds, and limiting the training set to a fixed‑size random sample (200 k rows) before fitting the heavy ensemble models. This reduces the amount of data the RandomForest and GradientBoosting regressors process while preserving the exact model architecture, hyper‑parameters and evaluation logic, so the predictions remain comparable but the runtime fits well inside the 600‑second limit.'
- What this solution (achieved 6.18994) has done: 'I add a squared distance feature, increase the number of trees for the RandomForest and GradientBoosting models, and average all three models (including the linear regression) for the final prediction. These changes keep the original preprocessing and model types while giving the model more expressive power and a richer feature set, which should lower the RMSE toward the target.'
- What this solution (achieved 6.13734) has done: 'The changes keep the same preprocessing and model types but give the models a bit more data (train‑test split 80/20 and a larger sample size) and apply a simple mean‑scale correction derived from the validation split to the final predictions, which is expected to lower the RMSE toward the target. The core logic and feature set remain unchanged.'
- What this solution (achieved 6.08303) has done: 'I replace the model‑selection cell with a small weighted‑ensemble step: using the validation set we fit non‑negative linear weights for the three predictors (Linear, RF, GBR), normalize them, and use the blended model if it improves RMSE. This keeps the original preprocessing and models unchanged while likely lowering the validation error, moving the score closer to the target.'
- What this solution (achieved 6.06953) has done: 'I add a simple linear calibration step that fits a regression on the validation predictions and true fares, then applies this correction to the test predictions. This modest post‑processing keeps the original model pipeline unchanged but typically reduces RMSE, moving the score closer to the target. The change replaces the crude mean‑scale factor with a learned linear adjustment and retains clipping of negative values.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 1
data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)




## === cell 2
mean_dropoff_longitude = data["dropoff_longitude"].mean()
mean_dropoff_latitude = data["dropoff_latitude"].mean()
data["dropoff_longitude"] = data["dropoff_longitude"].fillna(mean_dropoff_longitude)
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(mean_dropoff_latitude)




## === cell 3
testData = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")




## === cell 4
data = data[data["fare_amount"] <= 500]  # upper bound already used
data = data[data["fare_amount"] > 0]  # **new** lower bound to avoid log of negatives
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 7)]

ny_lat_min, ny_lat_max = 40.4774, 40.9176
ny_lon_min, ny_lon_max = -74.2591, -73.7004
data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]




## === cell 5
def haversine_distance_vec(lat1, lon1, lat2, lon2):
    """Haversine distance (km) for array‑like inputs."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r


data["haversine_distance"] = haversine_distance_vec(
    data["pickup_latitude"].values,
    data["pickup_longitude"].values,
    data["dropoff_latitude"].values,
    data["dropoff_longitude"].values,
)
data["haversine_distance_sq"] = data["haversine_distance"] ** 2
data["log_haversine_distance"] = np.log1p(data["haversine_distance"])

testData["haversine_distance"] = haversine_distance_vec(
    testData["pickup_latitude"].values,
    testData["pickup_longitude"].values,
    testData["dropoff_latitude"].values,
    testData["dropoff_longitude"].values,
)
testData["haversine_distance_sq"] = testData["haversine_distance"] ** 2
testData["log_haversine_distance"] = np.log1p(testData["haversine_distance"])




## === cell 6
data = (
    data[(data["haversine_distance"] > 0.1) & (data["haversine_distance"] < 500)]
    .dropna(subset=["fare_amount"])
    .reset_index(drop=True)
)




## === cell 7
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
hour = data["pickup_datetime"].dt.hour
data["is_night_time"] = ((hour >= 21) | (hour < 6)).astype(int)
data["year"] = data["pickup_datetime"].dt.year
data["month"] = data["pickup_datetime"].dt.month
data["hour_of_day"] = hour

testData["pickup_datetime"] = pd.to_datetime(testData["pickup_datetime"])
testData["year"] = testData["pickup_datetime"].dt.year
testData["month"] = testData["pickup_datetime"].dt.month
testData["hour_of_day"] = testData["pickup_datetime"].dt.hour

testData["dropoff_longitude"] = testData["dropoff_longitude"].fillna(
    mean_dropoff_longitude
)
testData["dropoff_latitude"] = testData["dropoff_latitude"].fillna(
    mean_dropoff_latitude
)

test_hour = testData["pickup_datetime"].dt.hour
testData["is_night_time"] = ((test_hour >= 21) | (test_hour < 6)).astype(int)




## === cell 8
airports = [(-73.7789, 40.6413), (-73.8740, 40.7769), (-74.1811, 40.6925)]


def compute_near_airport(df, lat_col, lon_col, airports, threshold_km=5):
    lat = df[lat_col].values
    lon = df[lon_col].values
    near = np.zeros(df.shape[0], dtype=int)
    for lon_a, lat_a in airports:
        dist = haversine_distance_vec(
            lat,
            lon,
            np.full(df.shape[0], lat_a),
            np.full(df.shape[0], lon_a),
        )
        near = np.where(dist < threshold_km, 1, near)
    return near


data["pickup_near_airport"] = compute_near_airport(
    data, "pickup_latitude", "pickup_longitude", airports
)
data["dropoff_near_airport"] = compute_near_airport(
    data, "dropoff_latitude", "dropoff_longitude", airports
)

testData["pickup_near_airport"] = compute_near_airport(
    testData, "pickup_latitude", "pickup_longitude", airports
)
testData["dropoff_near_airport"] = compute_near_airport(
    testData, "dropoff_latitude", "dropoff_longitude", airports
)




## === cell 9
public_holidays = {
    (1, 1),
    (1, 15),
    (2, 12),
    (2, 19),
    (5, 27),
    (6, 19),
    (7, 4),
    (9, 2),
    (10, 14),
    (11, 5),
    (11, 11),
    (11, 28),
    (12, 25),
}
md_train = data["pickup_datetime"].dt.month * 100 + data["pickup_datetime"].dt.day
data["is_public_holiday"] = np.isin(
    md_train, [m * 100 + d for m, d in public_holidays]
).astype(int)

md_test = (
    testData["pickup_datetime"].dt.month * 100 + testData["pickup_datetime"].dt.day
)
testData["is_public_holiday"] = np.isin(
    md_test, [m * 100 + d for m, d in public_holidays]
).astype(int)




## === cell 10
features = [
    "passenger_count",
    "haversine_distance",
    "log_haversine_distance",
    "haversine_distance_sq",  # new feature
    "pickup_near_airport",
    "dropoff_near_airport",
    "hour_of_day",
    "year",
    "month",
    "is_public_holiday",
    "is_night_time",
]

data = data.dropna(subset=features).reset_index(drop=True)
testData[features] = testData[features].fillna(0)


X = data[features].values.astype(np.float32)
y = np.log1p(data["fare_amount"].values).astype(np.float32)

np.random.seed(42)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

max_train_samples = 250_000  # increase a bit while staying within the 600 s limit
if X_train.shape[0] > max_train_samples:
    idx = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]




## === cell 11
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear_log = linear_model.predict(X_test)
y_pred_linear = np.expm1(y_pred_linear_log)

rmse_linear = mean_squared_error(np.expm1(y_test), y_pred_linear, squared=False)
print("Linear Regression RMSE:", rmse_linear)




## === cell 12
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(
    n_estimators=400,  # increased for better performance
    max_depth=25,
    random_state=42,
    n_jobs=-1,
)

random_forest_model.fit(X_train, y_train)

y_pred_rf_log = random_forest_model.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)

rmse_rf = mean_squared_error(np.expm1(y_test), y_pred_rf, squared=False)
print("Random Forest RMSE:", rmse_rf)




## === cell 13
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(
    n_estimators=400,  # increased for better performance
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)

gbr_model.fit(X_train, y_train)

y_pred_gbr_log = gbr_model.predict(X_test)
y_pred_gbr = np.expm1(y_pred_gbr_log)

rmse_gbr = mean_squared_error(np.expm1(y_test), y_pred_gbr, squared=False)
print("Gradient Boosting RMSE:", rmse_gbr)




## === cell 14
import numpy as np

pred_matrix = np.column_stack([y_pred_linear, y_pred_rf, y_pred_gbr])
target = np.expm1(y_test)

weights, _, _, _ = np.linalg.lstsq(pred_matrix, target, rcond=None)

weights = np.where(weights < 0, 0, weights)
if weights.sum() == 0:
    weights = np.ones_like(weights) / len(weights)
else:
    weights = weights / weights.sum()

y_pred_blend = pred_matrix @ weights
rmse_blend = mean_squared_error(target, y_pred_blend, squared=False)
print("Blend (weighted) RMSE:", rmse_blend)

candidate_rmses = {
    "Linear Regression": rmse_linear,
    "Random Forest": rmse_rf,
    "Gradient Boosting": rmse_gbr,
    "Blend": rmse_blend,
}
best_name = min(candidate_rmses, key=candidate_rmses.get)
best_rmse = candidate_rmses[best_name]

print(f"Selected model for final prediction: {best_name} (RMSE {best_rmse:.5f})")

if best_name == "Blend":

    class BlendModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights

        def predict(self, X):
            preds = np.column_stack([m.predict(X) for m in self.models])
            preds = np.expm1(preds)
            return preds @ self.weights

    best_model = BlendModel([linear_model, random_forest_model, gbr_model], weights)
else:
    best_model = {
        "Linear Regression": linear_model,
        "Random Forest": random_forest_model,
        "Gradient Boosting": gbr_model,
    }[best_name]

val_pred = best_model.predict(X_test)
val_pred = (
    np.expm1(val_pred) if best_name != "Blend" else val_pred
)  # ensure fare amount
calibrator = LinearRegression()
calibrator.fit(val_pred.reshape(-1, 1), target)

test_pred = best_model.predict(testData[features].values.astype(np.float32))
test_pred = np.expm1(test_pred) if best_name != "Blend" else test_pred
test_pred = calibrator.predict(test_pred.reshape(-1, 1))

test_predictions = np.clip(test_pred, a_min=0, a_max=None)

submission = pd.DataFrame(
    {"key": testData["key"], "fare_amount": test_predictions},
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written.")
