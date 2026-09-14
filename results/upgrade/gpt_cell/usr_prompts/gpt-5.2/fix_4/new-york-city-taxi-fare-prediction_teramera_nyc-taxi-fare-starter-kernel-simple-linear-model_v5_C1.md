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

3.8

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
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes




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
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 7
print("Old size (pre-sanity-filters): %d" % len(train_df))
train_df = train_df[
    (train_df.fare_amount > 0)
    & (train_df.fare_amount <= 250)
    & (train_df.passenger_count >= 1)
    & (train_df.passenger_count <= 6)
    & (train_df.pickup_longitude.between(-74.5, -72.8))
    & (train_df.dropoff_longitude.between(-74.5, -72.8))
    & (train_df.pickup_latitude.between(40.5, 41.8))
    & (train_df.dropoff_latitude.between(40.5, 41.8))
].copy()
print("New size (post-sanity-filters): %d" % len(train_df))



## === cell 8
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickup_time"] = ls1



## === cell 9
train_df["pickup_time"].head(5)



## === cell 10
test_df = pd.read_csv("../input/test.csv")
test_df.head()



## === cell 11
add_travel_vector_features(test_df)



## === cell 12
test_df.shape



## === cell 13
ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickup_time"] = ls1



## === cell 14
ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["weekday"] = ls1



## === cell 15
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["weekday"] = ls1



## === cell 16
train_df.shape



## === cell 17
train_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 19
train_df["weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 20
test_df["weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 21
train_one_hot = pd.get_dummies(train_df["weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)



## === cell 22
test_one_hot = pd.get_dummies(test_df["weekday"])
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 23
weekday_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))
for c in weekday_cols:
    if c not in train_df.columns:
        train_df[c] = 0
    if c not in test_df.columns:
        test_df[c] = 0
train_df = train_df.reindex(
    columns=[col for col in train_df.columns if col != "weekday"] + [], copy=False
)
test_df = test_df.reindex(
    columns=[col for col in test_df.columns if col != "weekday"] + [], copy=False
)



## === cell 24
train_df.drop("weekday", inplace=True, axis=1, errors="ignore")


## === cell 25
test_df.drop("weekday", inplace=True, axis=1)



## --- ERROR in cell 25, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/379381071.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtest_df[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0;34m"weekday"[0m[0;34m,[0m [0minplace[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mdrop[0;34m(self, labels, axis, index, columns, level, inplace, errors)[0m
[1;32m   5579[0m                 [0mweight[0m  [0;36m1.0[0m     [0;36m0.8[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5580[0m         """
[0;32m-> 5581[0;31m         return super().drop(
[0m[1;32m   5582[0m             [0mlabels[0m[0;34m=[0m[0mlabels[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5583[0m             [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mdrop[0;34m(self, labels, axis, index, columns, level, inplace, errors)[0m
[1;32m   4786[0m         [0;32mfor[0m [0maxis[0m[0;34m,[0m [0mlabels[0m [0;32min[0m [0maxes[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4787[0m             [0;32mif[0m [0mlabels[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4788[0;31m                 [0mobj[0m [0;34m=[0m [0mobj[0m[0;34m.[0m[0m_drop_axis[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mlevel[0m[0;34m=[0m[0mlevel[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0merrors[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4789[0m [0;34m[0m[0m
[1;32m   4790[0m         [0;32mif[0m [0minplace[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_drop_axis[0;34m(self, labels, axis, level, errors, only_slice)[0m
[1;32m   4828[0m                 [0mnew_axis[0m [0;34m=[0m [0maxis[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0mlevel[0m[0;34m=[0m[0mlevel[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0merrors[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4829[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4830[0;31m                 [0mnew_axis[0m [0;34m=[0m [0maxis[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0merrors[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4831[0m             [0mindexer[0m [0;34m=[0m [0maxis[0m[0;34m.[0m[0mget_indexer[0m[0;34m([0m[0mnew_axis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4832[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mdrop[0;34m(self, labels, errors)[0m
[1;32m   7068[0m         [0;32mif[0m [0mmask[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   7069[0m             [0;32mif[0m [0merrors[0m [0;34m!=[0m [0;34m"ignore"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 7070[0;31m                 [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"{labels[mask].tolist()} not found in axis"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   7071[0m             [0mindexer[0m [0;34m=[0m [0mindexer[0m[0;34m[[0m[0;34m~[0m[0mmask[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   7072[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mdelete[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: "['weekday'] not found in axis"

## === cell 26
ls1 = list(train_df["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickup_time"] = ls1
