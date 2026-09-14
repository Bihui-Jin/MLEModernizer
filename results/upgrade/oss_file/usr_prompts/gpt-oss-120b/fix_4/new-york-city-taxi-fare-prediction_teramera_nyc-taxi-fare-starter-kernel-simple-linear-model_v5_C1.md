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

3.8

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

5.45383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 752.47114) has done: 'I preserve the original workflow and simply keep the test row identifiers (“key”) before we drop non‑feature columns. By storing the keys, aligning the feature columns, and then re‑adding the key column, the submission creation step can find `test_df["key"]` and write a valid CSV.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  # needed for the scatter plot

print(os.listdir("../input"))




## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
print(train_df.dtypes)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)




## === cell 3
print(train_df.isnull().sum())




## === cell 4
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))




## === cell 5
_ = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
plt.close()




## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 7
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]  # keep HH:MM
train_df["pickup_time"] = ls1




## === cell 8
train_df["pickup_time"].head(5)




## === cell 9
test_df = pd.read_csv("../input/test.csv")
test_df.head()




## === cell 10
add_travel_vector_features(test_df)




## === cell 11
test_df.shape




## === cell 12
ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickup_time"] = ls1




## === cell 13
def add_weekday_numeric(df):
    ls = list(df["pickup_datetime"])
    for i in range(len(ls)):
        ts = pd.Timestamp(ls[i][:-4:])  # strip the trailing .000Z etc.
        ls[i] = ts.weekday()
    df["weekday"] = ls


add_weekday_numeric(train_df)
add_weekday_numeric(test_df)




## === cell 14
train_df.shape




## === cell 15
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 16
train_one_hot = pd.get_dummies(train_df["weekday"], prefix="weekday")
train_df = pd.concat([train_df, train_one_hot], axis=1)

test_one_hot = pd.get_dummies(test_df["weekday"], prefix="weekday")
test_df = pd.concat([test_df, test_one_hot], axis=1)

train_df.drop("weekday", inplace=True, axis=1)
test_df.drop("weekday", inplace=True, axis=1)




## === cell 17
test_keys = test_df["key"].copy()

for col in train_one_hot.columns:
    if col not in test_df:
        test_df[col] = 0

feature_cols = train_df.drop(["key", "fare_amount"], axis=1).columns
test_df = test_df[feature_cols]

test_df["key"] = test_keys




## === cell 18
def time_to_int(series):
    out = []
    for v in series:
        h, m = v.split(":")
        out.append(int(h) * 100 + int(m))
    return out


train_df["pickup_time"] = time_to_int(train_df["pickup_time"])
test_df["pickup_time"] = time_to_int(test_df["pickup_time"])




## === cell 19
train_df.shape




## === cell 20
R = 6373.0
lat1 = np.radians(train_df["pickup_latitude"])
lon1 = np.radians(train_df["pickup_longitude"])
lat2 = np.radians(train_df["dropoff_latitude"])
lon2 = np.radians(train_df["dropoff_longitude"])
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
train_df["Distance"] = R * c * 0.621  # miles




## === cell 21
R = 6373.0
lat1 = np.radians(test_df["pickup_latitude"])
lon1 = np.radians(test_df["pickup_longitude"])
lat2 = np.radians(test_df["dropoff_latitude"])
lon2 = np.radians(test_df["dropoff_longitude"])
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
test_df["Distance"] = R * c * 0.621




## === cell 22
train_df["Distance"] = np.round(train_df["Distance"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)




## === cell 23
train_df["abs_diff_longitude"] = np.abs(
    train_df["abs_diff_longitude"] - train_df["abs_diff_longitude"].mean()
)
train_df["abs_diff_latitude"] = np.abs(
    train_df["abs_diff_latitude"] - train_df["abs_diff_latitude"].mean()
)

test_df["abs_diff_longitude"] = np.abs(
    test_df["abs_diff_longitude"] - train_df["abs_diff_longitude"].mean()
)
test_df["abs_diff_latitude"] = np.abs(
    test_df["abs_diff_latitude"] - train_df["abs_diff_latitude"].mean()
)




## === cell 24
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    inplace=True,
    axis=1,
)

test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    inplace=True,
    axis=1,
)




## === cell 25
train_df.head()




## === cell 26
from sklearn.model_selection import train_test_split




## === cell 27
X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]




## === cell 28
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.01, random_state=80
)




## === cell 29
X_train.shape




## === cell 30
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

hgb = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
)

hgb.fit(X_train, y_train_log)

valid_pred_log = hgb.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)

rmse = mean_squared_error(y_valid, valid_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2433856423.py in <cell line: 0>()
     14 )
     15 
---> 16 hgb.fit(X_train, y_train_log)
     17 
     18 # Predict on validation set and convert back

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py in fit(self, X, y, sample_weight)
    359         # time spent predicting X for gradient and hessians update
    360         acc_prediction_time = 0.0
--> 361         X, y = self._validate_data(X, y, dtype=[X_DTYPE], force_all_finite=False)
    362         y = self._encode_y(y)
    363         check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1142         estimator_name = _check_estimator_name(estimator)
   1143         y = column_or_1d(y, warn=True)
-> 1144         _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
   1145         _ensure_no_complex_data(y)
   1146     if y_numeric and y.dtype.kind == "O":

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input y contains NaN.

## === cell 31
test_features = test_df[X.columns]
test_pred_log = hgb.predict(test_features)
pred = np.round(np.expm1(test_pred_log), 2)




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3182245195.py in <cell line: 0>()
      1 test_features = test_df[X.columns]
----> 2 test_pred_log = hgb.predict(test_features)
      3 pred = np.round(np.expm1(test_pred_log), 2)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py in predict(self, X)
   1485         # Return inverse link of raw predictions after converting
   1486         # shape (n_samples, 1) to (n_samples,)
-> 1487         return self._loss.link.inverse(self._raw_predict(X).ravel())
   1488 
   1489     def staged_predict(self, X):

AttributeError: 'HistGradientBoostingRegressor' object has no attribute '_loss'

## === cell 32
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2759218550.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
      2 
      3 

NameError: name 'pred' is not defined

## === cell 33
submission.head()




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/960005582.py in <cell line: 0>()
----> 1 submission.head()
      2 
      3 

NameError: name 'submission' is not defined

## === cell 34
submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2949187407.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Saved submission to submission.csv")

NameError: name 'submission' is not defined
