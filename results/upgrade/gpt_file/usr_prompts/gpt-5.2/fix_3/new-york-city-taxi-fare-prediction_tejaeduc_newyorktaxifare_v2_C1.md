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

3.22951

# 6. Current score

7.46635

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.3938) has done: 'Your current gap is large (RMSE 6.11 vs target 3.23, lower is better), so we should legitimately improve generalization without changing the overall model approach (still XGBoost on engineered features). The biggest score drag here is the post-processing `pred.round(2)` (it adds quantization error that directly hurts RMSE), plus a subtle train/test feature mismatch risk from one-hot encoding years separately. I (1) remove rounding, (2) make year dummies consistent by concatenating train+test before `get_dummies`, and (3) align XGBoost with modern parameters (`reg:squarederror`) and make the split deterministic; all are minimal and should move RMSE down toward your target without changing the core pipeline. The script still write `finaloutput.csv` with `key,fare_amount`.'
- What this solution (achieved 7.46635) has done: 'We keep your XGBoost-on-engineered-features pipeline intact, but remove early stopping because it can underfit and materially worsens RMSE relative to your target; instead we train the full fixed 150 rounds for a consistent boost in fit without changing the model class or features. We also make the distance/temporal feature engineering vectorized (same semantics) so the notebook reliably finishes within the time limit while still using the same features. Finally, we ensure predictions are valid (finite and non-negative) to avoid rare pathological outputs harming RMSE, and we still write `finaloutput.csv` with the required `key,fare_amount` columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=1000000)



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
testkey = test.key



## === cell 4
df = df.dropna(how="any", axis="rows")



## === cell 5
len(df)



## === cell 6
df.head()



## === cell 7
l = df[
    (df.pickup_latitude > 42.0)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.0)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index



## === cell 8
df = df.drop(l, axis=0)



## === cell 9
z = df[
    (df.fare_amount > 300.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 0.0)
].index



## === cell 10
df = df.drop(z, axis=0)



## === cell 11
len(df)




## === cell 12
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6373.0 * c




## === cell 13
df["dist"] = haversine_km(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
)



## === cell 14
test["dist"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)



## === cell 15
test.head()



## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 17
df.info()



## === cell 18
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype("int8")
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype("int8")



## === cell 19
df["year"] = df["pickup_datetime"].dt.year.astype("int16")
test["year"] = test["pickup_datetime"].dt.year.astype("int16")



## === cell 20
df["day"] = df["pickup_datetime"].dt.day.astype("int8")
test["day"] = test["pickup_datetime"].dt.day.astype("int8")



## === cell 21
df.head()



## === cell 22
test.head()



## === cell 23
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 24
test_feat.year.unique()



## === cell 25
combined = pd.concat(
    [feat.drop(columns=["fare_amount"]), test_feat], axis=0, ignore_index=True
)
combined = pd.get_dummies(combined, columns=["year"])

X = combined.iloc[: len(feat), :].copy()
X_test = combined.iloc[len(feat) :, :].copy()
y = feat["fare_amount"].copy()



## === cell 26
X.head()



## === cell 27
X_test.head()



## === cell 28
xtr, xts, ytr, yts = train_test_split(X, y, test_size=0.25, random_state=42)



## === cell 29
xgbtrain = xgboost.DMatrix(xtr, label=ytr)
xgbvalid = xgboost.DMatrix(xts, label=yts)
xgbfinaltest = xgboost.DMatrix(X_test)



## === cell 30
params = {"eval_metric": "rmse", "objective": "reg:squarederror", "seed": 42}



## === cell 31
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=150,
    evals=[(xgbvalid, "valid")],
    verbose_eval=False,
)



## === cell 32
pred = xgbmodel.predict(xgbfinaltest)



## === cell 33
pred = np.asarray(pred, dtype="float64")
pred = np.nan_to_num(pred, nan=0.0, posinf=300.0, neginf=0.0)
pred = np.clip(pred, 0.0, 300.0)



## === cell 34
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## === cell 35
finalset = finalset[["key", "fare_amount"]]



## === cell 36
finalset.head()



## === cell 37
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", os.path.abspath("finaloutput.csv"), "rows:", len(finalset))
