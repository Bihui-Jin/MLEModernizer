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
sklearn-pandas==2.2.0
xgboost==2.0.3

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

5.0139

# 6. Current score

4.31469

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.87376) has done: 'I corrected the boolean filter used for selecting training rows, switched the XGBoost setup to a more reliable `XGBRegressor` with a log‑transform on the target (which is a small, targeted tweak that often lowers RMSE without changing the overall modeling pipeline), and made sure the final predictions are written to the required `submission.csv` file. These adjustments keep the original feature engineering and overall structure while nudging the model’s performance toward the target score.'
- What this solution (achieved 5.77576) has done: 'I increase the training sample size (load 500 k rows instead of 100 k) and relax the training‑row filter by removing the overly strict `distance_to_downtown < 15` condition. A larger, less‑filtered dataset lets the XGBoost model learn more realistic patterns, which should lower the RMSE toward the target while keeping the original modeling pipeline unchanged.'
- What this solution (achieved 997.15509) has done: 'I increase the XGBoost capacity slightly (more trees, a lower learning rate) and combine its predictions with the earlier linear regression output by averaging them. This ensemble often lowers RMSE by reducing variance without changing the overall modeling pipeline, moving the score closer to the target.'
- What this solution (achieved 7.22661) has done: 'I keep the overall preprocessing and modeling pipeline but remove the linear model from the final ensemble (the XGBoost model alone is much stronger) and ensure the XGBoost training also discards rows with a passenger count of 0, matching the earlier filter. This small change eliminates the dilution caused by averaging with a weaker model and should lower the RMSE toward the target while preserving the core logic.'
- What this solution (achieved 8.84954) has done: 'I increase the training sample size from 500 k to 1 M rows and remove the restrictive New York bounding‑box filter, which together give the model more representative data and should lower the RMSE toward the target. The rest of the pipeline (feature engineering, log‑transformed XGBoost, and submission output) stays unchanged.'
- What this solution (achieved 5.8382) has done: 'The update expands the training sample size slightly and blends the XGBoost predictions with the earlier linear‑regression output (a simple average). Using more rows gives the model a richer learning set, while averaging the two models usually lowers RMSE by reducing prediction variance, moving the score closer to the target.'
- What this solution (achieved 8.85435) has done: 'I keep the overall pipeline unchanged but replace the fixed 0.5 / 0.5 blend with a data‑driven weight that minimizes RMSE on the validation split. By computing the optimal linear‑blend coefficient from the validation predictions of the XGBoost and linear models, we give more influence to the stronger model while still keeping the simple averaging logic. This small calibration is expected to lower the final RMSE and move the score closer to the target.'
- What this solution (achieved 4.31469) has done: 'I train the linear model on the log‑transformed fare (using `np.log1p`) and exponentiate its predictions, so both models operate on the same scale. Then I evaluate each model’s RMSE on the validation split and, if XGBoost alone is better, set the blend weight `w_opt` to 1 (i.e., use only XGBoost). This small, targeted change keeps the overall pipeline intact while expectedly lowering the RMSE toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # plotting library
from sklearn.linear_model import LinearRegression  # linear regression model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_data_set = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,  # increased sample size to give the model more data
    parse_dates=["pickup_datetime"],
)
train_data_set.head(5)




## === cell 2
print(train_data_set.dtypes)
train_data_set.describe()




## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} entities from the dataset")
train_data_set.describe()




## === cell 4
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} entities from the dataset")




## === cell 5
train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")
plt.show()




## === cell 6
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

old_len = len(train_data_set)
new_len = len(train_data_set)
print(
    f"Removed {(old_len - new_len)} entities from the dataset (bounding box filter disabled)"
)




## === cell 7
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set.head(5)




## === cell 8
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)

train_data_set.head(5)




## === cell 9
nyc_down_town = (-74.0063889, 40.7141667)  # NYC downtown coordinates (lon, lat)
jfk_airport = (-73.7822222222, 40.6441666667)  # JFK airport coordinates (lon, lat)

train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
)

train_data_set["distance_to_jfk_airport"] = distance_on_the_sphere(
    jfk_airport[1],
    jfk_airport[0],
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
)

train_data_set.head(5)




## === cell 10
idx = train_data_set["passenger_count"] != 0

features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "year",
    "day_of_week",
    "is_rush_hour",
    "distance_to_downtown",
    "distance_to_jfk_airport",
]
target = "fare_amount"

X = train_data_set.loc[idx, features].values
y = train_data_set.loc[idx, target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

linear_model = Pipeline(
    (
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    )
)

linear_model.fit(X_train, y_train_log)




## === cell 11
test_data_set = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)

test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)

test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
)

test_data_set["distance_to_jfk_airport"] = distance_on_the_sphere(
    jfk_airport[1],
    jfk_airport[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
)




## === cell 12
XTEST = test_data_set[features].values
y_pred_linear_log = linear_model.predict(XTEST)
y_pred_linear = np.expm1(y_pred_linear_log)  # inverse of log1p

linear_submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred_linear}
)
linear_submission.to_csv("linear_submission.csv", index=False)




## === cell 13
train_data_set = train_data_set.drop(columns=["key", "pickup_datetime"])

train_data_set = train_data_set[train_data_set["passenger_count"] != 0]

y = train_data_set["fare_amount"]
X = train_data_set.drop(columns=["fare_amount"])

y_log = np.log1p(y)

x_train, x_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.05, random_state=2666
)

xgb_params = {
    "max_depth": 7,
    "subsample": 0.9,
    "learning_rate": 0.03,
    "colsample_bytree": 0.9,
    "random_state": 2666,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "verbosity": 0,
}

xgb_model = xgb.XGBRegressor(**xgb_params)

xgb_model.fit(
    x_train,
    y_train_log,
    eval_set=[(x_valid, y_valid_log)],
    early_stopping_rounds=30,
    verbose=False,
)

y_valid = np.expm1(y_valid_log)

xgb_valid_pred = np.expm1(xgb_model.predict(x_valid))

lin_valid_pred = np.expm1(linear_model.predict(x_valid.values))

numer = np.sum((y_valid - lin_valid_pred) * (xgb_valid_pred - lin_valid_pred))
denom = np.sum((xgb_valid_pred - lin_valid_pred) ** 2)
w_opt = numer / denom if denom != 0 else 0.5
w_opt = np.clip(w_opt, 0.0, 1.0)  # keep within [0,1]

rmse_xgb = np.sqrt(np.mean((y_valid - xgb_valid_pred) ** 2))
rmse_lin = np.sqrt(np.mean((y_valid - lin_valid_pred) ** 2))

if rmse_xgb < rmse_lin:
    w_opt = 1.0
print(f"Optimal blend weight for XGBoost: {w_opt:.4f}")
print(f"Validation RMSE – XGB: {rmse_xgb:.4f}, Linear: {rmse_lin:.4f}")

x_test_features = test_data_set.drop(columns=["key", "pickup_datetime"])

log_pred = xgb_model.predict(x_test_features)
pred_final = np.expm1(log_pred)  # inverse of log1p
pred_final = np.clip(pred_final, a_min=0, a_max=None)

blended_pred = w_opt * pred_final + (1 - w_opt) * y_pred_linear

submission = pd.DataFrame({"key": test_data_set["key"], "fare_amount": blended_pred})
submission.to_csv("submission.csv", index=False)
submission.head()
