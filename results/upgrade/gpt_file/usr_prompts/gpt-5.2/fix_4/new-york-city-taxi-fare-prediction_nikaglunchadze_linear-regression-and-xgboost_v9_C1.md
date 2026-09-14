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
joblib==1.5.2
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

3.6129

# 6. Current score

8.54792

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.27171) has done: 'Your code likely didn’t yield a valid Kaggle score because the test-set cleaning drops rows (and therefore keys), producing a submission with fewer than 9914 rows. I keep your core feature engineering and XGBoost model unchanged, but change the test-time cleaning to *not drop any rows*: instead we impute missing values and clip/filter out-of-range values into valid bounds so every key is retained. This should produce a valid `submission.csv` with the exact required row count and typically improves RMSE versus dropping rows (which Kaggle rejects or scores poorly if misaligned). I also make the train/validation split deterministic to stabilize offline RMSE and avoid run-to-run variation.'
- What this solution (achieved 8.54792) has done: 'Your current RMSE (10.27) is far above the target (3.61), so we need a real-but-minimal accuracy lift without changing the overall approach (same features, same XGBoost regressor). The biggest likely issue is that the model is trained on raw coordinates while predictions are made on clipped/imputed coordinates; we make train-time preprocessing consistent with test-time (clip coordinates/passenger_count and ensure no NaNs) instead of dropping many rows, which typically improves generalization. We also add the standard NYC Taxi sanity filter that removes zero coordinates and a small “reasonable distance” cap, which reduces label noise/outliers while preserving the same feature set and model. Finally, we keep the submission alignment via `test_keys` and still write a valid `submission.csv` with exactly the test row count.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=5_000_000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df.dtypes



## === cell 1
df.describe()



## === cell 2
df.head()




## === cell 3
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

df = df.dropna()

numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in numeric_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

df["pickup_longitude"] = df["pickup_longitude"].clip(ny_longitude_min, ny_longitude_max)
df["pickup_latitude"] = df["pickup_latitude"].clip(ny_latitude_min, ny_latitude_max)
df["dropoff_longitude"] = df["dropoff_longitude"].clip(
    ny_longitude_min, ny_longitude_max
)
df["dropoff_latitude"] = df["dropoff_latitude"].clip(ny_latitude_min, ny_latitude_max)
df["passenger_count"] = df["passenger_count"].clip(1, 6)

for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    df = df[df[c] != 0]



## === cell 4
df[df["fare_amount"] > 200].describe()



## === cell 5
df = filter_column(df, "fare_amount", 1, 200)




## === cell 6
def refactor_datetime(df_):
    df_["pickup_datetime"] = pd.to_datetime(df_["pickup_datetime"])
    df_["year"] = df_["pickup_datetime"].dt.year
    df_["month"] = df_["pickup_datetime"].dt.month
    df_["day"] = df_["pickup_datetime"].dt.day
    df_["weekday"] = df_["pickup_datetime"].dt.weekday
    df_["hour"] = df_["pickup_datetime"].dt.hour
    df_.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## === cell 7
def haversine(p1, p2):
    lat1, lon1, lat2, lon2 = map(np.radians, [p1[0], p1[1], p2[0], p2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * dist
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 8
def insert_haversine_dists(df_, locations):
    for location in locations:
        df_["pickup_dist_to_" + location[0]] = haversine(
            (df_["pickup_latitude"], df_["pickup_longitude"]), location[1]
        )
        df_["dropoff_dist_to_" + location[0]] = haversine(
            (df_["dropoff_latitude"], df_["dropoff_longitude"]), location[1]
        )
    df_["ride_distance"] = haversine(
        (df_["pickup_latitude"], df_["pickup_longitude"]),
        (df_["dropoff_latitude"], df_["dropoff_longitude"]),
    )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)



## === cell 9
df.describe()



## === cell 10
df = df[df["ride_distance"] > 0]
df = df[df["ride_distance"] <= 200]  # generous cap to remove obvious outliers
df.describe()



## === cell 11
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 12
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]
train_features.info()



## === cell 13
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



## === cell 14
from sklearn.model_selection import cross_val_score


def estimate_model(model, df_):
    X = df_[features]
    y = df_[fare_amount]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 15
estimate_model(linear_model, train_df)



## === cell 16
linear_model.fit(train_features, train_fare_amount)



## === cell 17
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)



## === cell 18
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns

learning_rates = [0.1, 0.15, 0.2]
n_estimators = [80, 100, 150]

sample_fraction = 0.1
train_sample = df.sample(frac=sample_fraction, random_state=42)
X = train_sample[features]
y = train_sample[fare_amount]


def cross_val_rmse(lr, ne, X_, y_):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmse = []

    for train_index, val_index in kf.split(X_):
        X_train, X_val = X_.iloc[train_index], X_.iloc[val_index]
        y_train, y_val = y_.iloc[train_index], y_.iloc[val_index]
        model = XGBRegressor(
            objective="reg:squarederror", learning_rate=lr, n_estimators=ne, n_jobs=-1
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        fold_rmse.append(rmse)

    avg_rmse = np.mean(fold_rmse)
    return lr, ne, avg_rmse


results = Parallel(n_jobs=-1)(
    delayed(cross_val_rmse)(lr, ne, X, y)
    for lr in learning_rates
    for ne in n_estimators
)

results_df = pd.DataFrame(results, columns=["learning_rate", "n_estimators", "rmse"])
results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=results_df, x="n_estimators", y="rmse", hue="learning_rate", marker="o"
)
plt.title("RMSE for Different Learning Rates and n_estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("RMSE")
plt.legend(title="Learning Rate")
plt.show()



## === cell 19
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.15,
    n_estimators=150,
    max_depth=6,
    min_child_weight=2,
    n_jobs=-1,
)
xgb_model.fit(train_features, train_fare_amount)
xgb_predictions = xgb_model.predict(validation_features)
mean_squared_error(validation_fare_amount, xgb_predictions, squared=False)



## === cell 20
xgb_predictions = xgb_model.predict(train_features)
mean_squared_error(train_fare_amount, xgb_predictions, squared=False)



## === cell 21
test_keys = test_df["key"].copy()

numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in numeric_cols:
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(test_df[c].median())

test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(
    ny_longitude_min, ny_longitude_max
)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(
    ny_latitude_min, ny_latitude_max
)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(
    ny_longitude_min, ny_longitude_max
)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(
    ny_latitude_min, ny_latitude_max
)

test_df["passenger_count"] = test_df["passenger_count"].clip(1, 6)

test_df["ride_distance"] = test_df["ride_distance"].clip(lower=1e-6)

for c in features:
    if c in test_df.columns and test_df[c].isna().any():
        test_df[c] = test_df[c].fillna(test_df[c].median())

xgb_test_pred = xgb_model.predict(test_df[features])

submission = pd.DataFrame({"key": test_keys, "fare_amount": xgb_test_pred})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(submission))
print("Expected test rows:", len(test_keys))
print(submission.head())
