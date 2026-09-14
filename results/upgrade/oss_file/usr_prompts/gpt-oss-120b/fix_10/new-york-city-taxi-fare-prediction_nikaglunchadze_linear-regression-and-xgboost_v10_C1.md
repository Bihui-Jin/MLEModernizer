# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.44252

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.86885) has done: 'I keep the data loading, cleaning, feature engineering, and train/validation split unchanged, but modify the XGBoost training to predict the log‑transformed fare amount (using log1p) and then exponentiate the predictions back to the original scale. This usually reduces the impact of large outliers and brings the RMSE closer to the target. I also slightly increase the number of trees and add modest subsampling for better generalisation while preserving the original model structure.'
- What this solution (achieved 7.42978) has done: 'I keep the overall pipeline unchanged but add a few small feature enrichments (latitude/longitude differences) and train the XGBoost model with a lower learning rate, deeper trees and early‑stopping on the validation set. These modest tweaks are expected to lower the RMSE and move the score toward the target while preserving the original logic.'
- What this solution (achieved 514.68659) has done: 'I keep the original pipeline intact but add a lightweight blending step that combines the XGBoost predictions with the simpler Linear Regression predictions. By averaging these two models (giving more weight to XGBoost), we can often lower the validation RMSE without changing the core model architecture or training process. The final test predictions use the same blended approach, and the script still produce a correctly‑named `submission.csv` file.'
- What this solution (achieved 6.71091) has done: 'I reduce the XGBoost model complexity (fewer trees and shallower depth) to prevent extreme over‑prediction that caused an RMSE of ~514, and I stop blending with the linear model because the linear predictions were adding error. Using a simpler XGBoost on the log‑transformed target should bring the validation RMSE much closer to the target 3.44 while keeping the original pipeline intact.'
- What this solution (achieved 7.16656) has done: 'I keep the overall pipeline unchanged but make the XGBoost model a bit stronger, which is the most direct way to lower the validation RMSE and move the score toward the target. The changes are limited to the hyper‑parameters in the XGBRegressor (more trees, slightly deeper depth, lower learning rate, and a larger early‑stopping window). These tweaks preserve the original logic, still use the log‑target transformation, and keep the final CSV output unchanged.'
- What this solution (achieved 6.81515) has done: 'I tighten the XGBoost model to avoid over‑fitting (reduce max depth) and clip predictions to the realistic fare range [1, 200] before evaluating and saving. These minimal tweaks keep the overall pipeline intact while likely lowering the RMSE toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    dtype=dtype_spec,
)
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
)



## === cell 1
ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

mask = (
    df["pickup_longitude"].between(ny_longitude_min, ny_longitude_max)
    & df["pickup_latitude"].between(ny_latitude_min, ny_latitude_max)
    & df["dropoff_longitude"].between(ny_longitude_min, ny_longitude_max)
    & df["dropoff_latitude"].between(ny_latitude_min, ny_latitude_max)
    & df["passenger_count"].between(1, 6)
)
df = df.dropna()
df = df.loc[mask]



## === cell 2
df = df[(df["fare_amount"] >= 1) & (df["fare_amount"] <= 200)]




## === cell 3
def refactor_datetime(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)




## === cell 4
def haversine_vec(lat1, lon1, lat2, lon2):
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians,
        [
            lat1.astype("float64"),
            lon1.astype("float64"),
            lat2.astype("float64"),
            lon2.astype("float64"),
        ],
    )
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))
locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 5
def insert_haversine_dists(df, locations):
    for name, (lat_ref, lon_ref) in locations:
        df[f"pickup_dist_to_{name}"] = haversine_vec(
            df["pickup_latitude"], df["pickup_longitude"], lat_ref, lon_ref
        )
        df[f"dropoff_dist_to_{name}"] = haversine_vec(
            df["dropoff_latitude"], df["dropoff_longitude"], lat_ref, lon_ref
        )
    df["ride_distance"] = haversine_vec(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["abs_lat_diff"] = df["lat_diff"].abs()
    df["abs_lon_diff"] = df["lon_diff"].abs()


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3207281912.py in <cell line: 0>()
     19 
     20 
---> 21 insert_haversine_dists(df, locs)
     22 insert_haversine_dists(test_df, locs)
     23 

/tmp/ipykernel_11/3207281912.py in insert_haversine_dists(df, locations)
      1 def insert_haversine_dists(df, locations):
      2     for name, (lat_ref, lon_ref) in locations:
----> 3         df[f"pickup_dist_to_{name}"] = haversine_vec(
      4             df["pickup_latitude"], df["pickup_longitude"], lat_ref, lon_ref
      5         )

/tmp/ipykernel_11/3092993806.py in haversine_vec(lat1, lon1, lat2, lon2)
      5             lat1.astype("float64"),
      6             lon1.astype("float64"),
----> 7             lat2.astype("float64"),
      8             lon2.astype("float64"),
      9         ],

AttributeError: 'float' object has no attribute 'astype'

## === cell 6
df = df[df["ride_distance"] > 0]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'ride_distance'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2876886817.py in <cell line: 0>()
----> 1 df = df[df["ride_distance"] > 0]
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'ride_distance'

## === cell 7
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 8
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
    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",
]
features += [f"pickup_dist_to_{x[0]}" for x in locs]
features += [f"dropoff_dist_to_{x[0]}" for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3058022995.py in <cell line: 0>()
     20 fare_amount = "fare_amount"
     21 
---> 22 train_features = train_df[features]
     23 train_fare_amount = train_df[fare_amount]
     24 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['ride_distance', 'lat_diff', 'lon_diff', 'abs_lat_diff', 'abs_lon_diff', 'pickup_dist_to_ny_center', 'pickup_dist_to_jfk_airport', 'pickup_dist_to_lga_airport', 'pickup_dist_to_ewr_airport', 'dropoff_dist_to_ny_center', 'dropoff_dist_to_jfk_airport', 'dropoff_dist_to_lga_airport', 'dropoff_dist_to_ewr_airport'] not in index"

## === cell 9
linear_model = LinearRegression()




## === cell 10
def estimate_model(model, df):
    X = df[features]
    y = df[fare_amount]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 11
estimate_model(linear_model, train_df)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2952885628.py in <cell line: 0>()
----> 1 estimate_model(linear_model, train_df)
      2 

/tmp/ipykernel_11/2159793960.py in estimate_model(model, df)
      1 def estimate_model(model, df):
----> 2     X = df[features]
      3     y = df[fare_amount]
      4     cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
      5     rmse_scores = np.sqrt(-cv_scores)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['ride_distance', 'lat_diff', 'lon_diff', 'abs_lat_diff', 'abs_lon_diff', 'pickup_dist_to_ny_center', 'pickup_dist_to_jfk_airport', 'pickup_dist_to_lga_airport', 'pickup_dist_to_ewr_airport', 'dropoff_dist_to_ny_center', 'dropoff_dist_to_jfk_airport', 'dropoff_dist_to_lga_airport', 'dropoff_dist_to_ewr_airport'] not in index"

## === cell 12
linear_model.fit(train_features, train_fare_amount)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1153972044.py in <cell line: 0>()
----> 1 linear_model.fit(train_features, train_fare_amount)
      2 

NameError: name 'train_features' is not defined

## === cell 13
linear_predictions = linear_model.predict(validation_features)
print(
    "Linear RMSE:",
    mean_squared_error(validation_fare_amount, linear_predictions, squared=False),
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3550626111.py in <cell line: 0>()
----> 1 linear_predictions = linear_model.predict(validation_features)
      2 print(
      3     "Linear RMSE:",
      4     mean_squared_error(validation_fare_amount, linear_predictions, squared=False),
      5 )

NameError: name 'validation_features' is not defined

## === cell 14
log_train_target = np.log1p(train_fare_amount)
log_val_target = np.log1p(validation_fare_amount)

xgb_log_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.03,
    n_estimators=4000,
    max_depth=8,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
)

xgb_log_model.fit(
    train_features,
    log_train_target,
    eval_set=[(validation_features, log_val_target)],
    early_stopping_rounds=200,
    verbose=False,
)

val_log_pred = xgb_log_model.predict(validation_features)
val_pred_raw = np.expm1(val_log_pred)
val_pred_clipped = np.clip(val_pred_raw, 1, 200)

validation_rmse_raw = mean_squared_error(
    validation_fare_amount, val_pred_raw, squared=False
)
validation_rmse_clip = mean_squared_error(
    validation_fare_amount, val_pred_clipped, squared=False
)
print("Validation RMSE (raw, no clipping):", validation_rmse_raw)
print("Validation RMSE (clipped to [1,200]):", validation_rmse_clip)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/984587448.py in <cell line: 0>()
----> 1 log_train_target = np.log1p(train_fare_amount)
      2 log_val_target = np.log1p(validation_fare_amount)
      3 
      4 xgb_log_model = XGBRegressor(
      5     objective="reg:squarederror",

NameError: name 'train_fare_amount' is not defined

## === cell 15
train_log_pred = xgb_log_model.predict(train_features)
train_pred_raw = np.expm1(train_log_pred)
train_pred_clipped = np.clip(train_pred_raw, 1, 200)

train_rmse_raw = mean_squared_error(train_fare_amount, train_pred_raw, squared=False)
train_rmse_clip = mean_squared_error(
    train_fare_amount, train_pred_clipped, squared=False
)
print("Training RMSE (raw):", train_rmse_raw)
print("Training RMSE (clipped):", train_rmse_clip)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1072669164.py in <cell line: 0>()
----> 1 train_log_pred = xgb_log_model.predict(train_features)
      2 train_pred_raw = np.expm1(train_log_pred)
      3 train_pred_clipped = np.clip(train_pred_raw, 1, 200)
      4 
      5 train_rmse_raw = mean_squared_error(train_fare_amount, train_pred_raw, squared=False)

NameError: name 'xgb_log_model' is not defined

## === cell 16
xgb_val_pred = val_pred_clipped
xgb_rmse = mean_squared_error(validation_fare_amount, xgb_val_pred, squared=False)
print("XGBoost Validation RMSE (final clipped predictions):", xgb_rmse)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4269366375.py in <cell line: 0>()
----> 1 xgb_val_pred = val_pred_clipped
      2 xgb_rmse = mean_squared_error(validation_fare_amount, xgb_val_pred, squared=False)
      3 print("XGBoost Validation RMSE (final clipped predictions):", xgb_rmse)
      4 

NameError: name 'val_pred_clipped' is not defined

## === cell 17
test_pred_raw = np.expm1(xgb_log_model.predict(test_df[features]))
test_pred = np.clip(test_pred_raw, 1, 200)

holdout = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
holdout.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/728457817.py in <cell line: 0>()
----> 1 test_pred_raw = np.expm1(xgb_log_model.predict(test_df[features]))
      2 test_pred = np.clip(test_pred_raw, 1, 200)
      3 
      4 holdout = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
      5 holdout.to_csv("submission.csv", index=False)

NameError: name 'xgb_log_model' is not defined
