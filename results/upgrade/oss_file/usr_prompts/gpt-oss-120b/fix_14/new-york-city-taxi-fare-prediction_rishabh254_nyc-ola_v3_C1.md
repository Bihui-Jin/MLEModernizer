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

5.92798

# 6. Current score

6.58953

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1143.23147) has done: 'I add a proper distance feature (haversine kilometers) and include passenger count in the linear regression input matrix, keeping the same OLS approach. This gives the model more predictive power without changing its linear‑regression nature, and the predictions be clamped to non‑negative values before rounding, helping the RMSE move toward the target.'
- What this solution (achieved 15.25184) has done: 'I add simple temporal features (hour and weekday) derived from the pickup datetime and include them in the linear‑regression design matrix, and I remove the overly strict geographic‑difference filter so more training data is used. These lightweight changes keep the core OLS approach while giving the model extra predictive signal, which should move the RMSE closer to the target.'
- What this solution (achieved 10.25841) has done: 'Implemented fixes to handle invalid target values and ensure the model trains without NaNs.  
- Clip `fare_amount` to non‑negative values before applying `log1p` so the log‑target contains no NaNs.  
- Re‑compute the log‑target after clipping and proceed with the train/validation split.  
- The rest of the pipeline remains unchanged, now producing a valid `submission.csv` and allowing RMSE calculation.'
- What this solution (achieved 10.28541) has done: 'I add richer engineered features (squared distance and cyclical hour/weekday encodings) and include them in the design matrix, then train the same Ridge regression (with a slightly smaller regularization strength) on the expanded feature set. These changes keep the core linear‑model approach while providing the model more predictive signals, which should lower the RMSE toward the target.'
- What this solution (achieved 10.28541) has done: 'I slightly lessen the regularisation (α = 0.1 instead of 0.5) to give the linear model more flexibility and remove the artificial rounding on the validation predictions, which was inflating the RMSE. The submission predictions keep two‑decimal rounding for Kaggle‑compatible formatting while the validation metric uses the full‑precision values, moving the score closer to the target without altering the core modelling pipeline.'
- What this solution (achieved 6.58955) has done: 'I add a log‑scaled distance feature (`log_distance_km`) to give the linear model a better‑behaved predictor of the log‑target, and raise the Ridge regularisation strength to 1.0 (still keeping the same linear‑regression pipeline). These small tweaks keep the core logic unchanged while providing extra signal and a slightly more robust fit, which should lower the validation RMSE and move the score toward the target.'
- What this solution (achieved 6.58955) has done: 'I slightly lower the Ridge regularization strength (alpha = 0.8 instead of 1.0) so the linear model can fit the data a bit more flexibly. This change keeps the overall pipeline, features and evaluation unchanged, but is expected to reduce the validation RMSE and move the score closer to the target 5.92798.'
- What this solution (achieved 6.58953) has done: 'I clip the fare amounts to a reasonable upper bound (200 USD) before the log‑transform to reduce the influence of extreme outliers, and I increase the Ridge regularisation slightly (α = 1.0). These minimal tweaks keep the original linear‑model pipeline intact while providing a modest bias‑variance adjustment that should lower the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
data = pd.read_csv("../input/train.csv", nrows=20_000_000)




## === cell 2
def add_travel_vector_features(df):
    """Create geometric, distance, and temporal features for the taxi data."""
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()

    lon1 = np.radians(df.pickup_longitude.values)
    lat1 = np.radians(df.pickup_latitude.values)
    lon2 = np.radians(df.dropoff_longitude.values)
    lat2 = np.radians(df.dropoff_latitude.values)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["distance_km"] = earth_radius_km * c

    df["log_distance_km"] = np.log1p(df["distance_km"])

    df["distance_squared"] = df["distance_km"] ** 2

    df["pickup_datetime_parsed"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce"
    )
    df["hour"] = df["pickup_datetime_parsed"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["pickup_datetime_parsed"].dt.weekday.fillna(0).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)


add_travel_vector_features(data)




## === cell 3
print(data.isnull().sum())




## === cell 4
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))




## === cell 5
plot = data.iloc[:400000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")




## === cell 6
print("Size after (optional) filter: %d" % len(data))




## === cell 7
from sklearn.model_selection import train_test_split

y = data.fare_amount.clip(lower=0, upper=200)
y_log = np.log1p(y)

X = data.drop(["fare_amount"], axis=1)
train_df, val_df, train_y_log, val_y_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

val_y_original = np.expm1(val_y_log)




## === cell 8
def get_input_matrix(df):
    """
    Build the design matrix for OLS/Ridge using the expanded feature set:
    [abs_diff_longitude, abs_diff_latitude, distance_km, log_distance_km,
     distance_squared, passenger_count, hour, hour_sin, hour_cos,
     weekday, weekday_sin, weekday_cos, bias]
    """
    return np.column_stack(
        (
            df.abs_diff_longitude.values,
            df.abs_diff_latitude.values,
            df.distance_km.values,
            df.log_distance_km.values,
            df.distance_squared.values,
            df.passenger_count.values,
            df.hour.values,
            df.hour_sin.values,
            df.hour_cos.values,
            df.weekday.values,
            df.weekday_sin.values,
            df.weekday_cos.values,
            np.ones(len(df)),  # bias term
        )
    )


train_X = get_input_matrix(train_df)

print(train_X.shape)
print(train_y_log.shape)




## === cell 9
feat_mean = train_X[:, :-1].mean(axis=0)
feat_std = train_X[:, :-1].std(axis=0)
feat_std[feat_std == 0] = 1.0  # avoid division by zero


def scale_matrix(mat, mean, std):
    """Standardize features (excluding bias) and keep bias column unchanged."""
    scaled_feats = (mat[:, :-1] - mean) / std
    return np.hstack([scaled_feats, np.ones((mat.shape[0], 1))])


train_X_scaled = scale_matrix(train_X, feat_mean, feat_std)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, fit_intercept=False, solver="auto")
ridge.fit(train_X_scaled, train_y_log)
w_log = ridge.coef_

print("Ridge coefficients on log‑target:", w_log)




## === cell 10
test_df = pd.read_csv("../input/test.csv")
add_travel_vector_features(test_df)

test_X = get_input_matrix(test_df)
val_X = get_input_matrix(val_df)

test_X = scale_matrix(test_X, feat_mean, feat_std)
val_X = scale_matrix(val_X, feat_mean, feat_std)

test_y_pred_log = test_X @ w_log
val_y_pred_log = val_X @ w_log

val_y_predictions = np.maximum(0, np.expm1(val_y_pred_log))
test_y_predictions = np.maximum(0, np.expm1(test_y_pred_log)).round(decimals=2)

val_y_predictions = np.nan_to_num(val_y_predictions, nan=0.0, posinf=0.0, neginf=0.0)
test_y_predictions = np.nan_to_num(test_y_predictions, nan=0.0, posinf=0.0, neginf=0.0)
val_y_original = np.nan_to_num(val_y_original, nan=0.0, posinf=0.0, neginf=0.0)

from sklearn.metrics import mean_squared_error

val_rmse = np.sqrt(mean_squared_error(val_y_original, val_y_predictions))
print("Validation RMSE:", val_rmse)

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print("Files in current directory:", os.listdir("."))
