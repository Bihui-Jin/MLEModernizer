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

8.39274

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))




## === cell 1
taxi_ride_train = pd.read_csv(
    "../input/train.csv",
    sep=",",
    index_col="key",
    header=0,
    parse_dates=["pickup_datetime"],
    nrows=99999,
)
taxi_ride_test = pd.read_csv(
    "../input/test.csv",
    sep=",",
    index_col="key",
    header=0,
    parse_dates=["pickup_datetime"],
)
taxi_ride_train.head()




## === cell 2
print("The shape train data are {0}".format(taxi_ride_train.shape))
print("The shape test data are {0}".format(taxi_ride_test.shape))




## === cell 3
taxi_ride_train.info()




## === cell 4
taxi_ride_test.info()




## === cell 5
taxi_ride_train.dtypes.value_counts().reset_index()




## === cell 6
taxi_ride_train.isnull().sum().sum()




## === cell 7
taxi_ride_test.isnull().sum().sum()




## === cell 8
taxi_ride_train = taxi_ride_train.dropna(axis=0)
taxi_ride_test = taxi_ride_test.dropna(axis=0)
print(taxi_ride_train.isnull().sum().sum())
print(taxi_ride_test.isnull().sum().sum())




## === cell 9
def calculate_distance(row):
    R = 6373.0  # approximate radius of earth in km
    lat1 = radians(row[0])
    lon1 = radians(row[1])
    lat2 = radians(row[2])
    lon2 = radians(row[3])
    longitude_distance = lon2 - lon1
    latitude_distance = lat2 - lat1
    a = (
        sin(latitude_distance / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(longitude_distance / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance




## === cell 10
taxi_ride_train["ride_distance_km"] = taxi_ride_train[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].apply(calculate_distance, axis=1)
taxi_ride_test["ride_distance_km"] = taxi_ride_test[
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
].apply(calculate_distance, axis=1)




## === cell 11
taxi_ride_train["ride_distance_km"].describe()




## === cell 12
sns.boxplot(taxi_ride_train["ride_distance_km"])




## === cell 13
IQR = taxi_ride_train.ride_distance_km.quantile(
    0.75
) - taxi_ride_train.ride_distance_km.quantile(0.25)
Lower_fence = taxi_ride_train.ride_distance_km.quantile(0.25) - (IQR * 3)
Upper_fence = taxi_ride_train.ride_distance_km.quantile(0.75) + (IQR * 3)
print(
    "Distance outliers are values < {lowerboundary} or > {upperboundary}".format(
        lowerboundary=Lower_fence, upperboundary=Upper_fence
    )
)




## === cell 14
distance_outlier_train = len(taxi_ride_train[taxi_ride_train["ride_distance_km"] >= 30])
distance_outlier_test = len(taxi_ride_test[taxi_ride_test["ride_distance_km"] >= 30])
print(
    "There are {0} train rows and {1} test rows that have distance value more than 30km".format(
        distance_outlier_train, distance_outlier_test
    )
)




## === cell 15
taxi_ride_train["ride_distance_km"] = np.where(
    taxi_ride_train["ride_distance_km"].astype("float64") <= 30.0,
    taxi_ride_train["ride_distance_km"],
    30.0,
)
taxi_ride_train["ride_distance_km"] = np.where(
    taxi_ride_train["ride_distance_km"].astype("float64") >= 0.0,
    taxi_ride_train["ride_distance_km"],
    0.0,
)

taxi_ride_test["ride_distance_km"] = np.where(
    taxi_ride_test["ride_distance_km"].astype("float64") <= 30.0,
    taxi_ride_test["ride_distance_km"],
    30.0,
)
taxi_ride_test["ride_distance_km"] = np.where(
    taxi_ride_test["ride_distance_km"].astype("float64") >= 0.0,
    taxi_ride_test["ride_distance_km"],
    0.0,
)




## === cell 16
sns.boxplot(taxi_ride_train["ride_distance_km"])




## === cell 17
sns.jointplot(x="ride_distance_km", y="fare_amount", data=taxi_ride_train)




## === cell 18
pick_up_date_train = taxi_ride_train["pickup_datetime"]
pick_up_date_test = taxi_ride_test["pickup_datetime"]

temp_df_train = pd.DataFrame(
    {
        "year": pick_up_date_train.dt.year,
        "month": pick_up_date_train.dt.month,
        "day": pick_up_date_train.dt.day,
        "hour": pick_up_date_train.dt.hour,
        "dayofyear": pick_up_date_train.dt.dayofyear,
        "week": pick_up_date_train.dt.isocalendar().week,  # .dt.week removed in pandas 2.x
        "weekday": pick_up_date_train.dt.weekday,
        "quarter": pick_up_date_train.dt.quarter,
    }
)

temp_df_test = pd.DataFrame(
    {
        "year": pick_up_date_test.dt.year,
        "month": pick_up_date_test.dt.month,
        "day": pick_up_date_test.dt.day,
        "hour": pick_up_date_test.dt.hour,
        "dayofyear": pick_up_date_test.dt.dayofyear,
        "week": pick_up_date_test.dt.isocalendar().week,
        "weekday": pick_up_date_test.dt.weekday,
        "quarter": pick_up_date_test.dt.quarter,
    }
)

taxi_ride_train = pd.concat([taxi_ride_train, temp_df_train], axis=1)
taxi_ride_test = pd.concat([taxi_ride_test, temp_df_test], axis=1)

taxi_ride_train.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_test.drop("pickup_datetime", inplace=True, axis=1)

taxi_ride_train.head()




## === cell 19
taxi_ride_train.dtypes.value_counts().reset_index()




## === cell 20
print(
    "The new dataset contains {0} null entries ".format(
        taxi_ride_train.isnull().sum().sum()
    )
)




## === cell 21
sns.distplot(taxi_ride_train["fare_amount"])




## === cell 22
taxi_ride_train["fare_amount"].describe()




## === cell 23
length_before = len(taxi_ride_train)
taxi_ride_train = taxi_ride_train[taxi_ride_train.fare_amount >= 0.0]
length_after = len(taxi_ride_train)
print("No of rows removed {0}".format(length_before - length_after))




## === cell 24
print("Skewness before transformation {0}".format(taxi_ride_train.fare_amount.skew()))
sns.distplot(np.log(taxi_ride_train["fare_amount"] + 1))
taxi_ride_train["fare_amount"] = np.log(taxi_ride_train.fare_amount + 1)
print("Skewness after transformation {0}".format(taxi_ride_train.fare_amount.skew()))




## === cell 25
Y_train = taxi_ride_train.fare_amount
X_train = taxi_ride_train.drop("fare_amount", axis=1)
X_test = taxi_ride_test
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train, Y_train, test_size=0.33, random_state=42
)
print("Shape of training set is {0}".format(X_train.shape))
print("Shape of Validation set is {0}".format(X_valid.shape))
print("Shape of testing set is {0}".format(X_test.shape))




## === cell 26
discrete_col_list = []
continous_col_list = []
for col in X_train.columns.tolist():
    if (taxi_ride_train[col].value_counts().count() / len(taxi_ride_train)) < 0.1:
        discrete_col_list.append(col)
    else:
        continous_col_list.append(col)
print("The discrete columns are {0}".format(discrete_col_list))
print("The continuous columns are {0}".format(continous_col_list))




## === cell 27
for var in continous_col_list:
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    fig = taxi_ride_train.boxplot(column=var)
    fig.set_title("")
    plt.subplot(1, 2, 2)
    fig = taxi_ride_train[var].hist(bins=20)
    fig.set_xlabel(var)
    plt.show()




## === cell 28
latitude_upper_range = 90.0
latitude_lower_range = -90.0
for var in ["pickup_latitude", "dropoff_latitude"]:
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") <= latitude_upper_range,
        taxi_ride_train[var],
        latitude_upper_range,
    )
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") >= latitude_lower_range,
        taxi_ride_train[var],
        latitude_lower_range,
    )

    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") <= latitude_upper_range,
        taxi_ride_test[var],
        latitude_upper_range,
    )
    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") >= latitude_lower_range,
        taxi_ride_test[var],
        latitude_lower_range,
    )

longitude_upper_range = 180.0
longitude_lower_range = -180.0
for var in ["pickup_longitude", "dropoff_longitude"]:
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") <= longitude_upper_range,
        taxi_ride_train[var],
        longitude_upper_range,
    )
    taxi_ride_train[var] = np.where(
        taxi_ride_train[var].astype("float64") >= longitude_lower_range,
        taxi_ride_train[var],
        longitude_lower_range,
    )

    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") <= longitude_upper_range,
        taxi_ride_test[var],
        longitude_upper_range,
    )
    taxi_ride_test[var] = np.where(
        taxi_ride_test[var].astype("float64") >= longitude_lower_range,
        taxi_ride_test[var],
        longitude_lower_range,
    )




## === cell 29
for var in continous_col_list:
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    fig = taxi_ride_train.boxplot(column=var)
    fig.set_title("")
    plt.subplot(1, 2, 2)
    fig = taxi_ride_train[var].hist(bins=20)
    fig.set_xlabel(var)




## === cell 30
sns.distplot(np.sqrt(taxi_ride_train["ride_distance_km"]))
taxi_ride_train["ride_distance_km"] = np.sqrt(taxi_ride_train["ride_distance_km"])
taxi_ride_test["ride_distance_km"] = np.sqrt(taxi_ride_test["ride_distance_km"])




## === cell 31
for i, var in enumerate(discrete_col_list):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    sns.countplot(taxi_ride_train[var], ax=ax)




## === cell 32
sns.pairplot(
    taxi_ride_train,
    x_vars=continous_col_list,
    y_vars="fare_amount",
    kind="reg",
    height=3,
)




## === cell 33
sns.heatmap(X_train.corr())




## === cell 34
taxi_ride_train.groupby("hour")["fare_amount"].sum().plot()




## === cell 35
taxi_ride_train.groupby("weekday")["fare_amount"].sum().plot()




## === cell 36
taxi_ride_train.groupby("passenger_count")["fare_amount"].sum().plot()




## === cell 37
taxi_ride_train.groupby("month")["fare_amount"].sum().plot()




## === cell 38
taxi_ride_train.groupby("year")["fare_amount"].sum().plot()




## === cell 39
pd.crosstab(
    taxi_ride_train["quarter"], taxi_ride_train["fare_amount"].astype(int), margins=True
)




## === cell 40
constant_features = [
    feat for feat in taxi_ride_train.columns if taxi_ride_train[feat].std() == 0
]
print(constant_features)




## === cell 41
sel_ = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
sel_.fit(X_train, Y_train)
selected_feat = X_train.columns[sel_.get_support()]
print("The most important features are:", list(selected_feat))




## === cell 42
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_features = correlation(X_train, 0.8)
print("Correlated features to drop:", corr_features)
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_valid.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
print(X_train.shape, X_valid.shape, X_test.shape)




## === cell 43
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)




## === cell 44
params = {
    "n_estimators": 700,
    "max_depth": 5,  # deeper trees than the original 2
    "min_samples_split": 2,
    "learning_rate": 0.05,  # higher learning rate for faster learning
    "loss": "ls",
    "random_state": 42,
}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train_scaled, Y_train)
mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
print("MSE (log space): %.4f" % mse)
print("Variance score (R^2): %.2f" % r2_score(Y_valid, clf.predict(X_valid_scaled)))




## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/4233468796.py in <cell line: 0>()
     10 clf = ensemble.GradientBoostingRegressor(**params)
     11 
---> 12 clf.fit(X_train_scaled, Y_train)
     13 mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
     14 print("MSE (log space): %.4f" % mse)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    418             Fitted estimator.
    419         """
--> 420         self._validate_params()
    421 
    422         if not self.warm_start:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'huber', 'absolute_error', 'quantile', 'squared_error'}. Got 'ls' instead.

## === cell 45
pred_fare = np.exp(clf.predict(X_test_scaled))
submission = pd.DataFrame({"key": X_test.index, "fare_amount": pred_fare})
submission.to_csv("my_submission.csv", index=False)
print("Submission file saved as my_submission.csv with shape", submission.shape)




## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2853981213.py in <cell line: 0>()
      1 # Prepare submission: exponentiate predictions to revert the log transform
----> 2 pred_fare = np.exp(clf.predict(X_test_scaled))
      3 submission = pd.DataFrame({"key": X_test.index, "fare_amount": pred_fare})
      4 submission.to_csv("my_submission.csv", index=False)
      5 print("Submission file saved as my_submission.csv with shape", submission.shape)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in predict(self, X)
   1800         )
   1801         # In regression we can directly return the raw value from the trees.
-> 1802         return self._raw_predict(X).ravel()
   1803 
   1804     def staged_predict(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _raw_predict(self, X)
    685     def _raw_predict(self, X):
    686         """Return the sum of the trees raw predictions (+ init estimator)."""
--> 687         raw_predictions = self._raw_predict_init(X)
    688         predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
    689         return raw_predictions

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _raw_predict_init(self, X)
    671     def _raw_predict_init(self, X):
    672         """Check input and compute raw predictions of the init estimator."""
--> 673         self._check_initialized()
    674         X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
    675         if self.init_ == "zero":

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _check_initialized(self)
    380     def _check_initialized(self):
    381         """Check that the estimator is initialized, raising an error if not."""
--> 382         check_is_fitted(self)
    383 
    384     def fit(self, X, y, sample_weight=None, monitor=None):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 46
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()




## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/424864251.py in <cell line: 0>()
----> 1 feature_importance = clf.feature_importances_
      2 feature_importance = 100.0 * (feature_importance / feature_importance.max())
      3 sorted_idx = np.argsort(feature_importance)
      4 pos = np.arange(sorted_idx.shape[0]) + 0.5
      5 plt.subplot(1, 2, 2)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in feature_importances_(self)
    742             array of zeros.
    743         """
--> 744         self._check_initialized()
    745 
    746         relevant_trees = [

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _check_initialized(self)
    380     def _check_initialized(self):
    381         """Check that the estimator is initialized, raising an error if not."""
--> 382         check_is_fitted(self)
    383 
    384     def fit(self, X, y, sample_weight=None, monitor=None):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 47
def generate_residual_plot(label, prediction, title_suffix):
    plt.scatter(prediction, label - prediction, alpha=0.5)
    plt.title(f"Residual plot for {title_suffix}")
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    plt.hlines(
        y=0, xmin=prediction.min(), xmax=prediction.max(), colors="orange", linewidth=2
    )
    plt.tight_layout()
    plt.show()




## === cell 48
def generate_actual_vs_predicted_plot(label, prediction, title_suffix):
    plt.scatter(prediction, label, s=30, c="r", marker="+")
    plt.title(f"Actual vs Predicted for {title_suffix}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.tight_layout()
    plt.show()




## === cell 49
Y_valid_pred = clf.predict(X_valid_scaled)
generate_residual_plot(Y_valid, Y_valid_pred, "Taxi fares (log space)")
generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred, "Taxi fares (log space)")

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/55981112.py in <cell line: 0>()
----> 1 Y_valid_pred = clf.predict(X_valid_scaled)
      2 generate_residual_plot(Y_valid, Y_valid_pred, "Taxi fares (log space)")
      3 generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred, "Taxi fares (log space)")

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in predict(self, X)
   1800         )
   1801         # In regression we can directly return the raw value from the trees.
-> 1802         return self._raw_predict(X).ravel()
   1803 
   1804     def staged_predict(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _raw_predict(self, X)
    685     def _raw_predict(self, X):
    686         """Return the sum of the trees raw predictions (+ init estimator)."""
--> 687         raw_predictions = self._raw_predict_init(X)
    688         predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
    689         return raw_predictions

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _raw_predict_init(self, X)
    671     def _raw_predict_init(self, X):
    672         """Check input and compute raw predictions of the init estimator."""
--> 673         self._check_initialized()
    674         X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
    675         if self.init_ == "zero":

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _check_initialized(self)
    380     def _check_initialized(self):
    381         """Check that the estimator is initialized, raising an error if not."""
--> 382         check_is_fitted(self)
    383 
    384     def fit(self, X, y, sample_weight=None, monitor=None):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
