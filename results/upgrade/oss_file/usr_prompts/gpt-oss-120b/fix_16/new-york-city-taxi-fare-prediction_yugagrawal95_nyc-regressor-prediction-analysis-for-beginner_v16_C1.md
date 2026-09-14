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
lightgbm==4.6.0
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

3.61341

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore")




## === cell 1
def find_csv(rel_path):
    possible_paths = [
        os.path.join("../input", rel_path),
        os.path.join("./data", rel_path),
        os.path.join(".", rel_path),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to locate {rel_path}")


train_path = find_csv("train.csv")
test_path = find_csv("test.csv")

train_data = pd.read_csv(train_path, nrows=100_000)
test_data = pd.read_csv(test_path)




## === cell 2
def changeDataType(dataset):
    dataset["passenger_count"] = dataset.passenger_count.astype("uint8")
    dataset["pickup_longitude"] = dataset.pickup_longitude.astype("float32")
    dataset["pickup_latitude"] = dataset.pickup_latitude.astype("float32")
    dataset["dropoff_longitude"] = dataset.dropoff_longitude.astype("float32")
    dataset["dropoff_latitude"] = dataset.dropoff_latitude.astype("float32")
    dataset["pickup_datetime"] = pd.to_datetime(
        arg=dataset["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
    )
    return dataset


train_data = changeDataType(train_data)
test_data = changeDataType(test_data)

train_data["fare_amount"] = train_data.fare_amount.astype("float32")



## === cell 3
train_data = train_data.dropna()
train_data = train_data.loc[train_data["fare_amount"] > 0]
train_data = train_data.loc[train_data["fare_amount"] < 400]
train_data = train_data.loc[train_data["passenger_count"] <= 6]

lat_mask = train_data["pickup_latitude"].between(-90, 90) & train_data[
    "dropoff_latitude"
].between(-90, 90)
lon_mask = train_data["pickup_longitude"].between(-180, 180) & train_data[
    "dropoff_longitude"
].between(-180, 180)
train_data = train_data[lat_mask & lon_mask]

coord_nonzero = (
    (train_data["pickup_latitude"] != 0)
    & (train_data["pickup_longitude"] != 0)
    & (train_data["dropoff_latitude"] != 0)
    & (train_data["dropoff_longitude"] != 0)
)
train_data = train_data[coord_nonzero]

test_data = test_data.loc[
    test_data["pickup_latitude"].between(-90, 90)
    & test_data["dropoff_latitude"].between(-90, 90)
    & test_data["pickup_longitude"].between(-180, 180)
    & test_data["dropoff_longitude"].between(-180, 180)
]
test_data = test_data[
    (test_data["pickup_latitude"] != 0)
    & (test_data["pickup_longitude"] != 0)
    & (test_data["dropoff_latitude"] != 0)
    & (test_data["dropoff_longitude"] != 0)
]




## === cell 4
def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    """Haversine distance in kilometres."""
    from_lat = np.radians(pickup_latitude)
    from_long = np.radians(pickup_longitude)
    to_lat = np.radians(dropoff_latitude)
    to_long = np.radians(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


train_data["distance"] = calculate_distance(
    train_data["pickup_latitude"],
    train_data["pickup_longitude"],
    train_data["dropoff_latitude"],
    train_data["dropoff_longitude"],
)
test_data["distance"] = calculate_distance(
    test_data["pickup_latitude"],
    test_data["pickup_longitude"],
    test_data["dropoff_latitude"],
    test_data["dropoff_longitude"],
)

train_data["log_distance"] = np.log1p(train_data["distance"])
test_data["log_distance"] = np.log1p(test_data["distance"])

train_data = train_data.loc[train_data["distance"] < 200]
test_data = test_data.loc[test_data["distance"] < 200]



## === cell 5
for df in (train_data, test_data):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["Day_of_Week"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour

test_data_key = test_data["key"].reset_index(drop=True)

train_data = train_data.drop(columns=["key", "pickup_datetime"])
test_data = test_data.drop(columns=["key", "pickup_datetime"])



## === cell 6
X = train_data.drop(columns="fare_amount")
y = train_data["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=0
)



## === cell 7
gradient_reg = GradientBoostingRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=3, random_state=0
)
gradient_reg.fit(X_train, y_train)
pred_gb = gradient_reg.predict(X_valid)
rmse_gb = np.sqrt(mean_squared_error(y_valid, pred_gb))
print("GradientBoostingRegressor RMSE:", rmse_gb)



## === cell 8
xgreg = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    nthread=-1,
    verbosity=0,
    tree_method="hist",
)
xgreg.fit(X_train, y_train)
pred_xgb = xgreg.predict(X_valid)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, pred_xgb))
print("XGBRegressor RMSE:", rmse_xgb)



## === cell 9
model_lgb = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    random_state=0,
    n_jobs=-1,
)
model_lgb.fit(X_train, y_train)
pred_lgb = model_lgb.predict(X_valid)
rmse_lgb = np.sqrt(mean_squared_error(y_valid, pred_lgb))
print("LGBMRegressor RMSE:", rmse_lgb)



## === cell 10
base_model = [xgreg, gradient_reg, model_lgb]




## === cell 11
def _fit_predict(model, X_tr, y_tr, X_te):
    """Clone, fit on training split and predict on hold‑out split."""
    clone = model.__class__(**model.get_params())
    clone.fit(X_tr, y_tr)
    return clone.predict(X_te)


def out_of_fold_predictions(models, X, y, n_splits=2, random_state=0):
    """Generate OOF predictions using all base models with sequential folds."""
    X_np = X.values
    y_np = y.values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros((X.shape[0], len(models)), dtype=np.float32)

    for i, mdl in enumerate(models):
        for train_idx, holdout_idx in kf.split(X_np, y_np):
            pred = _fit_predict(
                mdl,
                X_np[train_idx],
                y_np[train_idx],
                X_np[holdout_idx],
            )
            oof_pred[holdout_idx, i] = pred
    return oof_pred


oof_preds = out_of_fold_predictions(base_model, X_train, y_train, n_splits=2)



## === cell 12
meta_model = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
meta_model.fit(oof_preds, y_train)



## === cell 13
meta_valid = np.column_stack([pred_gb, pred_xgb, pred_lgb])
stack_val_pred = meta_model.predict(meta_valid)
rmse_stack = np.sqrt(mean_squared_error(y_valid, stack_val_pred))
print("Stacked model validation RMSE:", rmse_stack)

avg_val_pred = np.mean([pred_gb, pred_xgb, pred_lgb], axis=0)
rmse_avg = np.sqrt(mean_squared_error(y_valid, avg_val_pred))
print("Average of base models validation RMSE:", rmse_avg)

target_rmse = 3.61341
if abs(rmse_stack - target_rmse) <= abs(rmse_avg - target_rmse):
    chosen_pred = "stack"
    final_pred = (
        stack_val_pred  # placeholder – will be overwritten by test predictions below
    )
else:
    chosen_pred = "average"
    final_pred = avg_val_pred  # placeholder

print(f"Chosen prediction method for final submission: {chosen_pred}")

test_base_preds = [mdl.predict(test_data) for mdl in base_model]
test_meta = np.column_stack(test_base_preds)

if chosen_pred == "stack":
    final_pred = meta_model.predict(test_meta)
else:
    final_pred = np.mean(test_base_preds, axis=0)

submission = pd.DataFrame(
    {"key": test_data_key, "fare_amount": final_pred}, columns=["key", "fare_amount"]
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
