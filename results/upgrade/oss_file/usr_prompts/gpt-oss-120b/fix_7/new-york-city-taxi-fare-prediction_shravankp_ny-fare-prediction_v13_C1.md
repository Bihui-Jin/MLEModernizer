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

3.7

# 3. Installed packages

eli5==0.13.0
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

4.57128

# 6. Current score

12.87143

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.57254) has done: 'Implemented a robust fix for the feature‑selection step that caused the pipeline to break.  
The code now safely handles cases where there are fewer than 10 categorical columns (or none) by selecting all available columns, ensuring `selected` is always defined. This enables downstream steps (train/validation split, model fitting, prediction, and CSV export) to run without errors and produce a valid `submission.csv`. No other logic was changed, preserving the original modeling approach.'
- What this solution (achieved 8.44315) has done: 'Implemented three key fixes: replaced the L2 Normalizer with a StandardScaler for more appropriate feature scaling, removed premature rounding of predictions during validation to obtain a true RMSE, and wrapped the optional eli5 inspection in a safe try/except that merely skips on failure. These changes keep the original model architecture intact while substantially improving validation performance, moving the RMSE toward the target score. The script now reliably writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 7.63489) has done: 'Implemented three focused fixes:   
1️⃣ Added the time‑based and passenger count features to the numerical set and scaling pipeline so the model can learn from hour, weekday, day, year and passenger count.   
2️⃣ Expanded the feature list used for training to include these new columns.   
3️⃣ Strengthened the XGBoost model by increasing trees, depth and adding early stopping on the validation split for better generalisation, which should lower the RMSE toward the target.'
- What this solution (achieved 12.87143) has done: 'Implemented fixes to ensure the pipeline runs end‑to‑end and creates a valid `submission.csv`:
- Corrected data paths to the Kaggle input directory.
- Added creation of `fare_per_km` and `fare_per_km_passenger` for the test set.
- Simplified categorical handling by skipping feature selection (avoids duplicate column names).
- Made the feature list unique to satisfy XGBoost’s requirement.
- Adjusted column handling and retained original key for submission output.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, math, datetime as dt
import seaborn as sns, matplotlib.pyplot as plt

print(os.listdir("/kaggle/input"))




## === cell 1
train = pd.read_csv("/kaggle/input/train.csv", nrows=1000000)
test = pd.read_csv("/kaggle/input/test.csv")
train.head()




## === cell 2
train.isnull().sum()




## === cell 3
train = train.dropna(how="any", axis=0)




## === cell 4
train["abs_diff_longitude"] = np.abs(
    train["dropoff_longitude"] - train["pickup_longitude"]
)
train["abs_diff_latitude"] = np.abs(
    train["dropoff_latitude"] - train["pickup_latitude"]
)
test["abs_diff_longitude"] = np.abs(
    test["dropoff_longitude"] - test["pickup_longitude"]
)
test["abs_diff_latitude"] = np.abs(test["dropoff_latitude"] - test["pickup_latitude"])




## === cell 5
train = train.loc[train["fare_amount"] > 0, :]
train = train.loc[(train["passenger_count"] <= 6) & (train["passenger_count"] > 0), :]
train = train.loc[
    (train["abs_diff_latitude"] < 2) & (train["abs_diff_longitude"] < 2), :
]
train = train.loc[
    (train["abs_diff_latitude"] > 0) & (train["abs_diff_longitude"] > 0), :
]




## === cell 6
train["timestamp_with_key"] = train["key"]
test["timestamp_with_key"] = test["key"]
train["key"] = train["key"].str.split(".").str[1].astype("int")
test["key"] = test["key"].str.split(".").str[1].astype("int")




## === cell 7
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True, utc=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True, utc=True
)
train["hour_no"] = train["pickup_datetime"].dt.hour.astype("int")
test["hour_no"] = test["pickup_datetime"].dt.hour.astype("int")
train["weekday_no"] = train["pickup_datetime"].dt.weekday.astype("int")
test["weekday_no"] = test["pickup_datetime"].dt.weekday.astype("int")
train["day_no"] = train["pickup_datetime"].dt.day.astype("int")
test["day_no"] = test["pickup_datetime"].dt.day.astype("int")
train["year_no"] = train["pickup_datetime"].dt.year.astype("int")
test["year_no"] = test["pickup_datetime"].dt.year.astype("int")




## === cell 8
def dist_haversine(x):
    R = 6371  # km
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat - picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon - picklon)
    a = (
        math.sin(latdiff / 2) ** 2
        + math.cos(picklat) * math.cos(droplat) * math.sin(londiff / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


train["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            dist_haversine,
            train[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=train.index,
)

test["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            dist_haversine,
            test[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=test.index,
)




## === cell 9
train["fare_per_km"] = train["fare_amount"] / train["dist_haversine_km"]
train["fare_per_km_passenger"] = train["fare_amount"] / (
    train["dist_haversine_km"] * train["passenger_count"]
)

test["fare_per_km"] = 0.0  # placeholder defaults
test["fare_per_km_passenger"] = 0.0
mask = test["dist_haversine_km"] != 0
test.loc[mask, "fare_per_km"] = 0.0  # test set has no fare, keep as 0
test.loc[mask, "fare_per_km_passenger"] = 0.0




## === cell 10
orig_train = train.copy()
orig_test = test.copy()




## === cell 11
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

train = shuffle(train).reset_index(drop=True)
val = train.iloc[int(0.9 * len(train)) :, :].reset_index(drop=True)
train = train.iloc[: int(0.9 * len(train)), :].reset_index(drop=True)

cols_to_normalize = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "passenger_count",
    "hour_no",
    "weekday_no",
    "day_no",
    "year_no",
]

scaler = StandardScaler().fit(train[cols_to_normalize])
train[cols_to_normalize] = scaler.transform(train[cols_to_normalize])
val[cols_to_normalize] = scaler.transform(val[cols_to_normalize])
test[cols_to_normalize] = scaler.transform(test[cols_to_normalize])




## === cell 12
categorical_cols = [
    c
    for c in train.columns
    if c
    not in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "abs_diff_longitude",
        "abs_diff_latitude",
        "dist_haversine_km",
        "passenger_count",
        "hour_no",
        "weekday_no",
        "day_no",
        "year_no",
        "key",
        "pickup_datetime",
        "timestamp_with_key",
        "fare_amount",
        "fare_per_km",
        "fare_per_km_passenger",
    ]
]

numerical_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "passenger_count",
    "hour_no",
    "weekday_no",
    "day_no",
    "year_no",
    "fare_per_km",
    "fare_per_km_passenger",
    "fare_amount",
]

object_cols = ["key", "pickup_datetime", "timestamp_with_key"]

cor = train[numerical_cols]
f, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.heatmap(cor.corr(), annot=True, ax=ax)




## === cell 13
selected = []




## === cell 14
train_y = np.log1p(train["fare_amount"])
val_y = np.log1p(val["fare_amount"])
feature_cols = [
    c for c in (selected + numerical_cols) if c not in object_cols + ["fare_amount"]
]
feature_cols = list(dict.fromkeys(feature_cols))
train_x = train[feature_cols]
val_x = val[feature_cols]
test_x = test[feature_cols]




## === cell 15
import xgboost as xgb
from xgboost import XGBRegressor

xgbr = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
)

xgbr.fit(
    train_x,
    train_y,
    eval_set=[(val_x, val_y)],
    early_stopping_rounds=50,
    verbose=False,
)




## === cell 16
pred_train_log = xgbr.predict(train_x)
pred_val_log = xgbr.predict(val_x)
pred_test_log = xgbr.predict(test_x)

pred_train = np.expm1(pred_train_log)
pred_val = np.expm1(pred_val_log)
pred_test = np.expm1(pred_test_log)




## === cell 17
from sklearn.metrics import mean_squared_error

rmse_train = np.sqrt(mean_squared_error(train["fare_amount"], pred_train))
rmse_val = np.sqrt(mean_squared_error(val["fare_amount"], pred_val))
print("RMSE Train:", rmse_train, "RMSE Val:", rmse_val)




## === cell 18
try:
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(xgbr, random_state=1).fit(val_x, val_y)
    eli5.show_weights(perm, feature_names=val_x.columns.tolist())
except Exception as e:
    print("eli5 inspection skipped:", e)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 19
final = pd.DataFrame({"key": test["timestamp_with_key"], "fare_amount": pred_test})
final.to_csv("submission.csv", index=False)




## === cell 20
final.head()
