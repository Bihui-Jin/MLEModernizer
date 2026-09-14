# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 6
test_df.head()



## === cell 7
test_df.info()



## === cell 8
test_df.shape



## === cell 9
train_df.isna().sum()




## === cell 10
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 11
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 12
print(train_df.isnull().sum())



## === cell 13
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 15
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 16
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickuptime"] = (dt.dt.hour * 100 + dt.dt.minute).astype("Int64")
    df["Weekday"] = dt.dt.dayofweek.astype("Int64")

    df["pickup_year"] = dt.dt.year.astype("Int64")
    df["pickup_month"] = dt.dt.month.astype("Int64")
    df["pickup_hour"] = dt.dt.hour.astype("Int64")
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")


add_time_features(train_df)
add_time_features(test_df)

train_df = train_df.dropna(
    subset=[
        "pickuptime",
        "Weekday",
        "pickup_year",
        "pickup_month",
        "pickup_hour",
        "is_weekend",
    ]
).copy()
test_df = test_df.dropna(
    subset=[
        "pickuptime",
        "Weekday",
        "pickup_year",
        "pickup_month",
        "pickup_hour",
        "is_weekend",
    ]
).copy()



## === cell 17
train_df.head()



## === cell 18
test_df.head()




## === cell 19
def replace_weekday(df):
    weekday_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df["Weekday"] = df["Weekday"].map(weekday_map)


replace_weekday(train_df)
replace_weekday(test_df)



## === cell 20
train_df.head()



## === cell 21
test_df.head()



## === cell 22
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 23
WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

train_one_hot = pd.get_dummies(
    pd.Categorical(train_df["Weekday"], categories=WEEKDAY_ORDER, ordered=True)
)
test_one_hot = pd.get_dummies(
    pd.Categorical(test_df["Weekday"], categories=WEEKDAY_ORDER, ordered=True)
)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 24
train_df.head()



## === cell 25
test_df.head()



## === cell 26
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 27
train_df["pickuptime"] = train_df["pickuptime"].fillna(0).astype(int)
test_df["pickuptime"] = test_df["pickuptime"].fillna(0).astype(int)

for c in ["pickup_year", "pickup_month", "pickup_hour", "is_weekend"]:
    train_df[c] = train_df[c].fillna(0).astype(int)
    test_df[c] = test_df[c].fillna(0).astype(int)


## === cell 28
train_df.head()



## === cell 29
test_df.head()




## === cell 30
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## === cell 31
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)
    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    d_lon_dropoff = lon3 - lon2
    d_lat_dropoff = lat3 - lat2
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    distance1 = R * c1
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2

    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 32
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 33
print("Old size (before distance cleaning): %d" % len(train_df))
train_df = train_df[(train_df["Distance"] > 0) & (train_df["Distance"] <= 100)].copy()
print("New size (after distance cleaning): %d" % len(train_df))



## === cell 34
print("Old size (before fare/coord cleaning): %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 500)
].copy()

train_df = train_df[
    (train_df["pickup_longitude"].between(-75, -72))
    & (train_df["dropoff_longitude"].between(-75, -72))
    & (train_df["pickup_latitude"].between(40, 42))
    & (train_df["dropoff_latitude"].between(40, 42))
].copy()

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
].copy()
print("New size (after fare/coord cleaning): %d" % len(train_df))



## === cell 35
print("Old size (before fare-per-mile cleaning): %d" % len(train_df))
_fpm = train_df["fare_amount"] / (train_df["Distance"] + 1e-3)
train_df = train_df[(_fpm >= 1.0) & (_fpm <= 50.0)].copy()
print("New size (after fare-per-mile cleaning): %d" % len(train_df))




## === cell 36
def add_extra_distance_features(df):
    df["Distance_sq"] = df["Distance"] ** 2
    df["abs_diff_longitude_sq"] = df["abs_diff_longitude"] ** 2
    df["abs_diff_latitude_sq"] = df["abs_diff_latitude"] ** 2
    df["near_airport"] = (
        (df["Pickup_Distance_airport"] < 1.0) | (df["Dropoff_Distance_airport"] < 1.0)
    ).astype(int)


add_extra_distance_features(train_df)
add_extra_distance_features(test_df)



## === cell 37
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)




## === cell 38
def zscore_train_apply(train_df, test_df, col):
    m = float(np.mean(train_df[col]))
    s = float(np.std(train_df[col])) + 1e-12
    train_df[col] = (train_df[col] - m) / s
    test_df[col] = (test_df[col] - m) / s
    return m, s


_abs_lon_mean, _abs_lon_std = zscore_train_apply(
    train_df, test_df, "abs_diff_longitude"
)
_abs_lat_mean, _abs_lat_std = zscore_train_apply(train_df, test_df, "abs_diff_latitude")

_distance_mean, _distance_std = zscore_train_apply(train_df, test_df, "Distance")
_distance_sq_mean, _distance_sq_std = zscore_train_apply(
    train_df, test_df, "Distance_sq"
)
_pda_mean, _pda_std = zscore_train_apply(train_df, test_df, "Pickup_Distance_airport")
_dda_mean, _dda_std = zscore_train_apply(train_df, test_df, "Dropoff_Distance_airport")
_adl2_mean, _adl2_std = zscore_train_apply(train_df, test_df, "abs_diff_longitude_sq")
_adt2_mean, _adt2_std = zscore_train_apply(train_df, test_df, "abs_diff_latitude_sq")



## === cell 39
train_feature_cols = [c for c in train_df.columns if c not in ["fare_amount", "key"]]
test_feature_cols = [c for c in test_df.columns if c != "key"]

missing_in_test = sorted(list(set(train_feature_cols) - set(test_feature_cols)))
missing_in_train = sorted(list(set(test_feature_cols) - set(train_feature_cols)))

for c in missing_in_test:
    test_df[c] = 0
for c in missing_in_train:
    train_df[c] = 0
    train_feature_cols.append(c)

train_feature_cols = sorted(train_feature_cols)

X_all = train_df[train_feature_cols]
X_test_all = test_df[["key"] + train_feature_cols]

print("Train shape (with target):", train_df.shape)
print("X_all shape:", X_all.shape)
print("Test shape (with key):", test_df.shape)
print("X_test_all shape:", X_test_all.shape)



## === cell 40
from sklearn.model_selection import train_test_split

X = X_all
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 41
from sklearn.linear_model import Ridge

lr = Ridge(alpha=1.0, fit_intercept=True, random_state=80)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## --- ERROR in cell 41, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1121050217.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mlr[0m [0;34m=[0m [0mRidge[0m[0;34m([0m[0malpha[0m[0;34m=[0m[0;36m1.0[0m[0;34m,[0m [0mfit_intercept[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;36m80[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mlr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mprint[0m[0;34m([0m[0mlr[0m[0;34m.[0m[0mscore[0m[0;34m([0m[0mX_test[0m[0;34m,[0m [0my_test[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight)[0m
[1;32m   1124[0m [0;34m[0m[0m
[1;32m   1125[0m         [0m_accept_sparse[0m [0;34m=[0m [0m_get_valid_accept_sparse[0m[0;34m([0m[0msparse[0m[0;34m.[0m[0missparse[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0msolver[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1126[0;31m         X, y = self._validate_data(
[0m[1;32m   1127[0m             [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1128[0m             [0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_data[0;34m(self, X, y, reset, validate_separately, **check_params)[0m
[1;32m    582[0m                 [0my[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0my[0m[0;34m,[0m [0minput_name[0m[0;34m=[0m[0;34m"y"[0m[0;34m,[0m [0;34m**[0m[0mcheck_y_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    583[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 584[0;31m                 [0mX[0m[0;34m,[0m [0my[0m [0;34m=[0m [0mcheck_X_y[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mcheck_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    585[0m             [0mout[0m [0;34m=[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m[0m[0;34m[0m[0m
[1;32m    586[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_X_y[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)[0m
[1;32m   1104[0m         )
[1;32m   1105[0m [0;34m[0m[0m
[0;32m-> 1106[0;31m     X = check_array(
[0m[1;32m   1107[0m         [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1108[0m         [0maccept_sparse[0m[0;34m=[0m[0maccept_sparse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    919[0m [0;34m[0m[0m
[1;32m    920[0m         [0;32mif[0m [0mforce_all_finite[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 921[0;31m             _assert_all_finite(
[0m[1;32m    922[0m                 [0marray[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    923[0m                 [0minput_name[0m[0;34m=[0m[0minput_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36m_assert_all_finite[0;34m(X, allow_nan, msg_dtype, estimator_name, input_name)[0m
[1;32m    159[0m                 [0;34m"#estimators-that-handle-nan-values"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m             )
[0;32m--> 161[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mmsg_err[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    162[0m [0;34m[0m[0m
[1;32m    163[0m [0;34m[0m[0m

[0;31mValueError[0m: Input X contains NaN.
Ridge does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 42
pred = lr.predict(X_test_all.drop("key", axis=1))
pred = np.asarray(pred, dtype=float)
pred = np.clip(pred, 0.0, 500.0)  # typical competition-safe bound
pred = np.round(pred, 2)
print(pred)
