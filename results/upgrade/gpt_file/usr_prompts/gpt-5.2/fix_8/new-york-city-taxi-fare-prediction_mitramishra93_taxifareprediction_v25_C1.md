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

3.40466

# 6. Current score

6.53814

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.30681) has done: 'I fix the pandas boolean logic bugs introduced by newer pandas (the `.any(1)` and invalid `|` usage) so the cleaning steps run again. Then I ensure the same cleaning is applied consistently to both train and test after feature engineering so the model does not see NaNs at fit/predict time (which caused the RandomForest crash). Finally, I make the submission creation robust by keeping the test `key` separately (since you later drop it for modeling) and writing a valid `key,fare_amount` CSV to `submission_1.csv`. These are execution/stability fixes and should also improve RMSE compared with failing to train at all.'
- What this solution (achieved 6.73406) has done: 'Your score gap is large (10.31 vs target 3.40, lower is better), so the most direct way to move toward the target without changing the core model is to fix two high-impact data issues: (1) your `key` is being coerced to datetime (and then back to string), which can change the ID format and misalign submissions; keep `key` as the original string and only parse `pickup_datetime`. (2) you currently “correct” all trips with `S_Distance > 200` by overwriting distance from fare, which injects target information back into a feature and can badly distort the learned mapping; restrict that correction to a small, clearly erroneous subset (very high distance with implausibly low fare) to preserve intent while avoiding widespread leakage-like distortion. These are minimal changes that keep the same feature set and RandomForest training loop, but should substantially reduce RMSE toward the target. The script still runs end-to-end and writes a valid `submission_1.csv` with the exact required `key,fare_amount` format.'
- What this solution (achieved 6.09921) has done: 'You’re currently far above the target RMSE (6.73 vs 3.40, lower is better), so we should make a small number of high-impact fixes that don’t change the core model/training loop: (1) remove the two places where `S_Distance` is overwritten using `fare_amount` (this injects target information into a feature and can distort learning), and instead just drop those clearly inconsistent rows; (2) make the train/test cleaning symmetric for coordinate validity so train isn’t “cleaner” than test in a way the model can’t generalize; (3) add a minimal, standard NYC bounding-box filter on train only to remove extreme outliers (a common RMSE killer) while keeping the same features and RandomForest. These are minimal semantic changes to your existing pipeline and should move RMSE down toward the target band while still producing `submission_1.csv`.'
- What this solution (achieved 5.95429) has done: 'We make two small, high-impact data-quality fixes that preserve your core RandomForest approach but reduce RMSE toward the 3.40466 target: (1) remove the accidental coordinate filter being applied to `test_df` (it can drop rows and misalign predictions/keys), and instead keep `test_df` intact while applying only train-side outlier filtering; (2) add two standard NYC Taxi Fare cleaning rules on train only—drop extreme `fare_amount` outliers and zero/negative distances (except where both points are identical)—to reduce label noise that disproportionately hurts RMSE. We also ensure `submission_1.csv` always has exactly the same number/order of rows as `test.csv` by never filtering test and by keeping `test_key` from the original test. These changes are minimal, keep your feature set/engineering and model intact, and are strongly expected to lower RMSE from ~6.10 toward the target band.'
- What this solution (achieved 5.9682) has done: 'Your RMSE (5.95) is still well above the target (3.40, lower is better), so we should make a small number of high-impact data-quality fixes without changing your RandomForest approach or feature set. The biggest remaining issue is that many outlier rows are still leaking into training because we only validated pickup latitude/longitude early, but not dropoff longitude (and we didn’t remove “zero coordinates” like (0,0) which create huge distances and label noise). I add a symmetric, train-only coordinate sanity filter (including dropoff longitude) plus a simple “not near (0,0)” rule, and I also actually drop the `non_rush_hour` zero-distance/low-fare rows that your notebook computes but never removes. These are minimal changes that keep the same features/engineering/model, but should reduce label noise and move RMSE down toward the target band while keeping test intact and submission aligned.'
- What this solution (achieved 6.65443) has done: 'We’re still far above the target RMSE (5.9682 vs 3.40466, lower is better), so the safest way to move toward the target without changing your model/training loop is to reduce training label noise/outliers and make distance-based cleaning consistent. I add two standard, train-only filters that strongly reduce RMSE for this competition: remove unrealistically low fares given distance (fare < 2.5 + 0.5 * km) and remove trips with implausibly high average speed computed from `pickup_datetime` vs distance (these are usually timestamp/coord glitches). I keep your existing feature engineering and RandomForestRegressor usage unchanged, and I ensure test rows are never dropped so `key` alignment and submission row count stay correct. These changes are minimal, metric-aligned, and expected to lower RMSE toward your target band.'
- What this solution (achieved 6.53814) has done: 'We need to move RMSE down from 6.65 toward the 3.40 target (lower is better), without changing your core RandomForest + basic datetime/distance feature engineering. The biggest remaining score-killers here are (a) you never compute trip duration (the current “speed” filter is a placeholder and therefore ineffective), and (b) you don’t remove the most common coordinate outliers (dropoff lat/lon invalid, 0/0 coords, and extreme distances) early enough and consistently, which leaves a lot of label noise. I make minimal, train-only data-quality fixes: add proper dropoff coordinate sanity checks, compute trip duration from pickup_datetime for train, then apply a conservative, standard speed filter and distance cap to remove obvious glitches. I also keep test completely unfiltered and preserve submission alignment exactly as you already do.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
import os

try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", repr(e))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=1000000)
test_df = pd.read_csv("../input/test.csv")



## === cell 2
train_df.shape



## === cell 3
train_df.columns



## === cell 4
train_df.head()



## === cell 5
train_df.info()



## === cell 6
train_df.describe()



## === cell 7
test_df.info()



## === cell 8
test_df.describe()



## === cell 9
train_df.isnull().sum()



## === cell 10
train_df = train_df.drop(train_df[train_df.isnull().any(axis=1)].index, axis=0)



## === cell 11
train_df.info()



## === cell 12
Counter(train_df["fare_amount"] < 0)



## === cell 13
train_df = train_df.drop(train_df[train_df["fare_amount"] < 0].index, axis=0)
train_df.shape



## === cell 14
train_df.describe()



## === cell 15
Counter(train_df["passenger_count"] > 6)



## === cell 16
train_df = train_df.drop(train_df[train_df["passenger_count"] > 6].index, axis=0)
train_df.shape



## === cell 17
Counter(train_df["pickup_latitude"] < -90)



## === cell 18
Counter(train_df["pickup_latitude"] > 90)



## === cell 19
mask_bad_lat = (train_df["pickup_latitude"] < -90) | (train_df["pickup_latitude"] > 90)
train_df = train_df.drop(train_df[mask_bad_lat].index, axis=0)



## === cell 20
train_df.shape



## === cell 21
Counter(train_df["pickup_longitude"] < -180)



## === cell 22
Counter(train_df["pickup_longitude"] > 180)



## === cell 23
train_df = train_df.drop((train_df[train_df["pickup_longitude"] < -180]).index, axis=0)



## === cell 24
train_df.shape



## === cell 25
train_df.dtypes



## === cell 26
train_df.head(3)



## === cell 27
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce"
)



## === cell 28
train_df.dtypes



## === cell 29
test_df.dtypes



## === cell 30
train_df.head()



## === cell 31
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")



## === cell 32
test_df.dtypes



## === cell 33
test_df.head()



## === cell 34
train_df.head()



## === cell 35
data = [train_df, test_df]
for i in data:
    i["date"] = i["pickup_datetime"].dt.day
    i["month"] = i["pickup_datetime"].dt.month
    i["day_of_week"] = i["pickup_datetime"].dt.dayofweek
    i["hour"] = i["pickup_datetime"].dt.hour
    i["year"] = i["pickup_datetime"].dt.year



## === cell 36
train_df.head()



## === cell 37
train_df.describe()




## === cell 38
def sphere_distance(lat1, long1, lat2, long2):
    data = [train_df, test_df]
    for i in data:
        R = 6367
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
        delta_phi = np.radians(i[lat2] - i[lat1])
        delta_lambda = np.radians(i[long2] - i[long1])
        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c
        i["S_Distance"] = d
    return d  # in Kilometer




## === cell 39
sphere_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 40
train_df.head()



## === cell 41
plt.hist(train_df["passenger_count"], bins=15)
plt.xlabel("No. of Passengers")
plt.ylabel("Frequency")



## === cell 42
plt.scatter(x=train_df["passenger_count"], y=train_df["fare_amount"], s=2.0)
plt.xlabel("No. of Passengers")
plt.ylabel("Fare")



## === cell 43
plt.scatter(x=train_df["date"], y=train_df["fare_amount"])
plt.xlabel("Date")
plt.ylabel("Fare")



## === cell 44
plt.hist(train_df["hour"], bins=50)
plt.xlabel("Date")
plt.ylabel("Fare")



## === cell 45
plt.hist(train_df["day_of_week"], bins=20)
plt.xlabel("Date")
plt.ylabel("Fare")



## === cell 46
plt.scatter(x=train_df["day_of_week"], y=train_df["fare_amount"])
plt.xlabel("Date of week")
plt.ylabel("Fare")



## === cell 47
len(train_df)



## === cell 48
train_df.sort_values(["S_Distance", "fare_amount"], ascending=False)



## === cell 49
dis_0 = train_df.loc[(train_df["S_Distance"] == 0), ["S_Distance"]]
dis_1 = train_df.loc[
    (train_df["S_Distance"] > 0) & (train_df["S_Distance"] <= 10), ["S_Distance"]
]
dis_2 = train_df.loc[
    (train_df["S_Distance"] > 10) & (train_df["S_Distance"] <= 50), ["S_Distance"]
]
dis_3 = train_df.loc[
    (train_df["S_Distance"] > 50) & (train_df["S_Distance"] <= 100), ["S_Distance"]
]
dis_4 = train_df.loc[
    (train_df["S_Distance"] > 100) & (train_df["S_Distance"] <= 200), ["S_Distance"]
]
dis_5 = train_df.loc[
    (train_df["S_Distance"] > 200) & (train_df["S_Distance"] <= 300), ["S_Distance"]
]
dis_6 = train_df.loc[
    (train_df["S_Distance"] > 300) & (train_df["S_Distance"] <= 500), ["S_Distance"]
]
dis_7 = train_df.loc[(train_df["S_Distance"] > 500), ["S_Distance"]]
dis_0["bins"] = "0"
dis_1["bins"] = "0-10"
dis_2["bins"] = "11-50"
dis_3["bins"] = "51-100"
dis_4["bins"] = "101-200"
dis_5["bins"] = "201-300"
dis_6["bins"] = "301-500"
dis_7["bins"] = ">500"
dis_bin = pd.concat([dis_0, dis_1, dis_2, dis_3, dis_4, dis_5, dis_6, dis_7])
dis_bin



## === cell 50
x = Counter(dis_bin["bins"])
x



## === cell 51
train_df.loc[
    ((train_df["pickup_latitude"] == 0) & (train_df["pickup_longitude"] == 0))
    & ((train_df["dropoff_latitude"] != 0) & (train_df["dropoff_longitude"] != 0))
    & (train_df["fare_amount"] == 0)
]



## === cell 52
train_df.loc[
    ((train_df["pickup_latitude"] == 0) & (train_df["pickup_longitude"] == 0))
    & ((train_df["dropoff_latitude"] != 0) & (train_df["dropoff_longitude"] != 0))
    & (train_df["fare_amount"] == 0)
]



## === cell 53
train_df = train_df.drop(
    train_df.loc[
        ((train_df["pickup_latitude"] == 0) & (train_df["pickup_longitude"] == 0))
        & ((train_df["dropoff_latitude"] != 0) & (train_df["dropoff_longitude"] != 0))
        & (train_df["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 54
train_df.shape



## === cell 55
train_df = train_df.drop(
    train_df.loc[
        ((train_df["pickup_latitude"] == 0) & (train_df["pickup_longitude"] == 0))
        & ((train_df["dropoff_latitude"] != 0) & (train_df["dropoff_longitude"] != 0))
        & (train_df["fare_amount"] == 0)
    ].index,
    axis=0,
)



## === cell 56
train_df.shape



## === cell 57
high_distance = train_df.loc[
    (train_df["S_Distance"] > 200)
    & (train_df["fare_amount"] > 0)
    & (train_df["fare_amount"] < 50)
]



## === cell 58
high_distance



## === cell 59
high_distance.shape



## === cell 60
train_df = train_df.drop(high_distance.index, axis=0)



## === cell 61
train_df



## === cell 62
train_df[train_df["S_Distance"] == 0]



## === cell 63
train_df[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] == 0)]



## === cell 64
train_df = train_df.drop(
    train_df[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] == 0)].index,
    axis=0,
)



## === cell 65
rush_hour = train_df.loc[
    (
        ((train_df["hour"] >= 6) & (train_df["hour"] <= 20))
        & ((train_df["day_of_week"] >= 1) & (train_df["day_of_week"] <= 5))
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 2.5)
    )
]
rush_hour



## === cell 66
train_df = train_df.drop(rush_hour.index, axis=0)



## === cell 67
train_df.shape



## === cell 68
non_rush_hour = train_df.loc[
    (
        ((train_df["hour"] < 6) | (train_df["hour"] > 20))
        & ((train_df["day_of_week"] >= 1) & (train_df["day_of_week"] <= 5))
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 3.0)
    )
]



## === cell 69
non_rush_hour



## === cell 70
non_rush_hour = train_df.loc[
    (
        ((train_df["hour"] < 6) | (train_df["hour"] > 20))
        & ((train_df["day_of_week"] >= 1) & (train_df["day_of_week"] <= 5))
        & (train_df["S_Distance"] == 0)
        & (train_df["fare_amount"] < 3.0)
    )
]
non_rush_hour



## === cell 71
train_df = train_df.drop(non_rush_hour.index, axis=0)



## === cell 72
train_df.loc[(train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)]



## === cell 73
scenario_3 = train_df.loc[
    (train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)
]
scenario_3



## === cell 74
scenario_3 = train_df.loc[
    (train_df["S_Distance"] != 0) & (train_df["fare_amount"] == 0)
]



## === cell 75
scenario_3 = scenario_3.copy()
scenario_3["fare_amount"] = scenario_3.apply(
    lambda row: ((row["S_Distance"] * 1.56) + 2.50), axis=1
)



## === cell 76
scenario_3["fare_amount"]



## === cell 77
train_df.loc[(train_df["S_Distance"] == 0) & (train_df["fare_amount"] != 0)]



## === cell 78
scenario_4 = train_df.loc[
    (train_df["S_Distance"] == 0) & (train_df["fare_amount"] != 0)
]



## === cell 79
scenario_4



## === cell 80
len(scenario_3)



## === cell 81
len(scenario_4)



## === cell 82
scenario_4.loc[(scenario_4["fare_amount"] <= 3.0) & (scenario_4["S_Distance"] == 0)]



## === cell 83
scenario_4.loc[(scenario_4["fare_amount"] > 3.0) & (scenario_4["S_Distance"] == 0)]



## === cell 84
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["S_Distance"] == 0)
]



## === cell 85
len(scenario_4_sub)



## === cell 86
train_df = train_df.drop(scenario_4_sub.index, axis=0)



## === cell 87
len(train_df)



## === cell 88
train_df.columns



## === cell 89
test_df.columns



## === cell 90
bad_drop_lat = (train_df["dropoff_latitude"] < -90) | (
    train_df["dropoff_latitude"] > 90
)
bad_drop_lon = (train_df["dropoff_longitude"] < -180) | (
    train_df["dropoff_longitude"] > 180
)
train_df = train_df.drop(train_df[bad_drop_lat | bad_drop_lon].index, axis=0)



## === cell 91
coord_mask_train = (
    train_df["pickup_latitude"].between(-90, 90)
    & train_df["dropoff_latitude"].between(-90, 90)
    & train_df["pickup_longitude"].between(-180, 180)
    & train_df["dropoff_longitude"].between(-180, 180)
)
train_df = train_df.loc[coord_mask_train].copy()



## === cell 92
not_null_island = ~(
    (train_df["pickup_latitude"].abs() < 0.01)
    & (train_df["pickup_longitude"].abs() < 0.01)
) & ~(
    (train_df["dropoff_latitude"].abs() < 0.01)
    & (train_df["dropoff_longitude"].abs() < 0.01)
)
train_df = train_df.loc[not_null_island].copy()



## === cell 93
train_df = train_df.loc[train_df["fare_amount"].between(2.5, 250)].copy()

same_point = (train_df["pickup_latitude"] == train_df["dropoff_latitude"]) & (
    train_df["pickup_longitude"] == train_df["dropoff_longitude"]
)
train_df = train_df.loc[(train_df["S_Distance"] > 0) | same_point].copy()



## === cell 94
nyc_mask = (
    train_df["pickup_latitude"].between(40.5, 41.0)
    & train_df["dropoff_latitude"].between(40.5, 41.0)
    & train_df["pickup_longitude"].between(-74.5, -73.5)
    & train_df["dropoff_longitude"].between(-74.5, -73.5)
)
train_df = train_df.loc[nyc_mask].copy()



## === cell 95
min_reasonable_fare = 2.5 + 0.5 * train_df["S_Distance"]
train_df = train_df.loc[
    (train_df["fare_amount"] >= min_reasonable_fare) | (train_df["S_Distance"] == 0)
].copy()



## === cell 96
train_df["time_bucket"] = (
    (train_df["year"] - train_df["year"].min()) * 8760
    + train_df["month"] * 744
    + train_df["date"] * 24
    + train_df["hour"]
)
train_df = train_df.loc[train_df["S_Distance"].between(0, 60)].copy()



## === cell 97
bucket_counts = train_df["time_bucket"].value_counts()
valid_buckets = bucket_counts[
    bucket_counts >= 50
].index  # only where comparison is meaningful
tmp = train_df.loc[
    train_df["time_bucket"].isin(valid_buckets), ["time_bucket", "S_Distance"]
].copy()
bucket_med = tmp.groupby("time_bucket")["S_Distance"].median()
train_df = train_df.loc[
    ~train_df["time_bucket"].isin(valid_buckets)
    | (
        train_df["S_Distance"]
        <= (bucket_med.reindex(train_df["time_bucket"]).values + 25.0)
    )
].copy()



## === cell 98
test_key = test_df["key"].copy()

if "time_bucket" in train_df.columns:
    train_df = train_df.drop(["time_bucket"], axis=1)

train_df = train_df.drop(["key", "pickup_datetime"], axis=1)
test_df = test_df.drop(["key", "pickup_datetime"], axis=1)



## === cell 99
train_df.columns



## === cell 100
test_df.columns



## === cell 101
x_train = train_df.iloc[:, train_df.columns != "fare_amount"]
y_train = train_df["fare_amount"].values
x_test = test_df



## === cell 102
x_train.shape



## === cell 103
y_train.shape



## === cell 104
x_train = x_train.replace([np.inf, -np.inf], np.nan)
x_test = x_test.replace([np.inf, -np.inf], np.nan)

x_test = x_test.reindex(columns=x_train.columns)

med = x_train.median(numeric_only=True)
x_train = x_train.fillna(med)
x_test = x_test.fillna(med)

rg = RandomForestRegressor(random_state=42, n_jobs=-1)
rg.fit(x_train, y_train)
y_predict = rg.predict(x_test)
y_predict



## === cell 105
submission = pd.DataFrame({"key": test_key.values, "fare_amount": y_predict})
submission.to_csv("submission_1.csv", index=False)
submission.head(10)
