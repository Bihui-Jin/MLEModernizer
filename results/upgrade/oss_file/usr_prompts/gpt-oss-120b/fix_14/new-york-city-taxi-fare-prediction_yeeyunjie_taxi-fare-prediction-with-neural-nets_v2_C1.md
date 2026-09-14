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

3.9

# 3. Installed packages

geopandas==0.14.4
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
seaborn==0.12.2
sklearn-pandas==2.2.0
statsmodels==0.14.5
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

5.07016

# 6. Current score

5.68528

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 926.18453) has done: 'The script was failing because of outdated imports, incorrect pandas usage (e.g., `any(1)`), and a cascade of undefined variables. I replaced the problematic sections with a clean preprocessing pipeline, added proper datetime and distance features, filtered out invalid rows, and trained a simple Ridge regression model (which reliably reaches an RMSE near the target). The final cell writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 926.92549) has done: 'The fix corrects the feature scaling mismatch when training on the full dataset: we now fit a new scaler on all training features, transform both the full training set and the test set with this same scaler, and then train the final Ridge model. This alignment prevents the huge prediction errors that caused the RMSE of ≈ 926, bringing the score much closer to the target. No other logic is changed.'
- What this solution (achieved 91882.92774) has done: 'I add a simple distance‑squared feature and train the Ridge model directly on the fare amount (removing the log‑transform) while using a slightly lower regularisation strength. These minimal changes keep the original pipeline intact but should lower the RMSE toward the target of 5.07.'
- What this solution (achieved 6.97581) has done: 'I add a log‑transform of the fare amount (using log1p) so the Ridge model learns on a smoother target, then convert predictions back with expm1 for both validation and test. This small change restores a common technique that dramatically lowers RMSE, moving the score from the huge 9 e4 range toward the target ≈ 5 while keeping the original pipeline intact.'
- What this solution (achieved 5.70967) has done: 'I add a modest log‑distance feature and lower the Ridge regularisation (alpha = 0.1). These tiny, targeted tweaks keep the original pipeline intact while giving the linear model a richer representation that should lower RMSE toward the target value.'
- What this solution (achieved 5.68602) has done: 'The changes add cyclic hour features (`hour_sin`, `hour_cos`) and a squared passenger count to give the linear model extra expressive power, and slightly reduce the Ridge regularisation (α = 0.05) which helps capture these new signals while staying close to the original pipeline. These minimal tweaks are expected to lower the validation RMSE, moving the score toward the target 5.07016.'
- What this solution (achieved 5.68477) has done: 'I add a simple interaction feature (`distance_passenger = distance * passenger_count`) to give the linear model extra signal, and reduce the Ridge regularisation (α = 0.01) in both the validation and final training steps. These minimal edits keep the original pipeline intact while expectedly lowering the RMSE toward the target value.'
- What this solution (achieved 5.68833) has done: 'I add cyclical weekday features (sin/cos) to give the linear model a bit more temporal signal and slightly reduce the Ridge regularisation (alpha = 0.005) to let the model use these richer features. These minimal adjustments keep the overall pipeline unchanged while aiming to lower the validation RMSE toward the target.'
- What this solution (achieved 5.68833) has done: 'I slightly increase the Ridge regularisation (alpha = 0.01) in both the validation and final training steps. A modest increase in regularisation often reduces over‑fitting on the split we use for validation, which should lower the RMSE and move the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 5.68528) has done: 'I add a lightweight “distance per passenger” feature to give the linear model a bit more signal and set the Ridge regularisation to zero (i.e., an ordinary least‑squares fit). These minimal tweaks keep the original pipeline intact while expectedly lowering the validation RMSE, moving the score closer to the target of 5.07016.'
- What this solution (achieved 5.68528) has done: 'I add a lightweight model‑selection step that tries a few small Ridge regularisation values (including the current 0.0) on the validation split and picks the one with the lowest RMSE. The chosen α is then used for the final model trained on the full data. This small regularisation tweak is expected to reduce over‑fitting and bring the validation RMSE closer to the target 5.07016 while keeping the overall pipeline unchanged.'
- What this solution (achieved 5.68528) has done: 'I add a very small regularisation option (α = 0.001) to the list of candidates examined during validation. This keeps the core pipeline unchanged while giving the model a tiny amount of shrinkage that often improves generalisation just enough to move the RMSE closer to the target 5.07016.'

# 9. Code solution

## === cell 0
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: "%.3f" % x)




## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

df = pd.read_csv(TRAIN_PATH, nrows=500000)
test_df = pd.read_csv(TEST_PATH)




## === cell 2
df = df.dropna()
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 200)]




## === cell 3
lat_cond = (df["pickup_latitude"].between(40, 42)) & (
    df["dropoff_latitude"].between(40, 42)
)
lon_cond = (df["pickup_longitude"].between(-75, -72)) & (
    df["dropoff_longitude"].between(-75, -72)
)
df = df[lat_cond & lon_cond]




## === cell 4
def haversine_vec(lon1, lat1, lon2, lat2):
    """
    Vectorised haversine distance (kilometers) between two sets of lon/lat points.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


df["distance"] = haversine_vec(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)
test_df["distance"] = haversine_vec(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)

df["distance_sq"] = df["distance"] ** 2
test_df["distance_sq"] = test_df["distance"] ** 2

df["log_distance"] = np.log1p(df["distance"])
test_df["log_distance"] = np.log1p(test_df["distance"])

df["distance_passenger"] = df["distance"] * df["passenger_count"]
test_df["distance_passenger"] = test_df["distance"] * test_df["passenger_count"]

df["distance_per_passenger"] = df["distance"] / (df["passenger_count"] + 1e-3)
test_df["distance_per_passenger"] = test_df["distance"] / (
    test_df["passenger_count"] + 1e-3
)




## === cell 5
def add_datetime_features(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["day"] = data["pickup_datetime"].dt.day
    data["hour"] = data["pickup_datetime"].dt.hour
    data["weekday"] = data["pickup_datetime"].dt.weekday
    data["minute"] = data["pickup_datetime"].dt.minute
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
    data["passenger_count_sq"] = data["passenger_count"] ** 2
    return data


df = add_datetime_features(df)
test_df = add_datetime_features(test_df)




## === cell 6
DROP_COLS = [
    "key",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = df.drop(columns=DROP_COLS + ["fare_amount"])
y = df["fare_amount"]

y_log = np.log1p(y)

test_X = test_df.drop(columns=DROP_COLS)




## === cell 7
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_val_ss = scaler.transform(X_val)
test_X_ss = scaler.transform(test_X)




## === cell 8
candidate_alphas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]
best_alpha = None
best_rmse = np.inf

for a in candidate_alphas:
    model = Ridge(alpha=a, random_state=42)
    model.fit(X_train_ss, y_train_log)
    val_pred_log = model.predict(X_val_ss)
    val_pred = np.expm1(val_pred_log)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_val_log), val_pred))
    print(f"Alpha {a:.3f} -> Validation RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = a

print(f"\nChosen alpha: {best_alpha} with Validation RMSE: {best_rmse:.4f}")




## === cell 9
scaler_full = StandardScaler()
X_full_ss = scaler_full.fit_transform(X)
test_X_ss = scaler_full.transform(test_X)

ridge_full = Ridge(alpha=best_alpha, random_state=42)
ridge_full.fit(X_full_ss, y_log)

test_pred_log = ridge_full.predict(test_X_ss)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
