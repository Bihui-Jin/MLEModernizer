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

4.22488

# 6. Current score

4.89192

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.36438) has done: 'The crash comes from using `data.pickup_datetime.dt.weekday_name`, which was removed in newer pandas; in pandas 2.x the equivalent is `dt.day_name()`. I change only the `day_of_week` feature line inside `add_time_features` to use `day_name()` while keeping all other feature engineering and types exactly the same. This preserves downstream columns/semantics expected by later cells and avoids changing any training/evaluation logic. No other cells need modification.'
- What this solution (achieved 10.6384) has done: 'Diagnosis: The crash happens because `x_test` ends up empty (0 rows), so `RandomForestRegressor.predict` raises “Found array with 0 sample(s)”. This is caused upstream by filtering `all_data` (after concatenating train+test) with `dropna` and range/passenger filters without updating `n_train`; rows from the beginning of `all_data` can be removed, so `all_data.iloc[n_train:]` can become empty. We should not change earlier cells, so the safest minimal fix is to rebuild `x_test` for prediction from the already-loaded `test` dataframe using the same feature engineering steps and then align its dummy columns to the training matrix `x`.

Patch summary: In cell 15, detect when `x_test` is empty and, if so, recreate the test feature matrix by applying `add_time_features`, `add_geo_features`, selecting `features`, `get_dummies`, and then reindexing columns to match `x` (filling missing with 0). Then run `model.predict` on the corrected `x_test`. This preserves the existing model and training logic while ensuring prediction input has valid rows and matching columns.

Updated cells: Only cell 15 is modified.

Compatibility notes for cell k+1: `test_pred` remains a 1D numpy array of length equal to `len(test_id)` (and `test_id` is unchanged), so cell 16 build the submission dataframe exactly as before.

Assumptions: `add_time_features`, `add_geo_features`, and `features` are already defined (from earlier cells), and `test` and `test_id` are already loaded and available.'
- What this solution (achieved 10.6384) has done: 'Your RMSE is far above the target, so we should make a small change that legitimately improves generalization without changing the overall approach (same feature engineering + RandomForest). The biggest issue is that you train on only 10,000 rows; increasing the training sample size (while keeping the same model and features) is the most direct way to reduce RMSE toward ~4.22. I only change the `nrows` argument used to read `train.csv` (and keep all downstream logic identical), staying within the time/memory budget. The existing “rebuild x_test if empty” safeguard remains to ensure a valid submission is always produced.'
- What this solution (achieved 4.89192) has done: 'The timeout is dominated by fitting a 300-tree `RandomForestRegressor` on a 1,000,000-row sparse one-hot matrix; this combination is far too slow in 600s. To preserve the same model/training logic while making it finish, the key fix is to avoid one-hot expansion for the tree model by encoding the time categoricals to stable integer codes (information-equivalent for trees, and avoids huge sparse matrices). Additionally, we keep all feature engineering identical, reduce pandas overhead by operating on NumPy arrays where possible, and ensure the training matrix is a compact contiguous `float32` array to minimize per-tree split cost. The rest of the pipeline (data reading, filtering, same RF hyperparameters, same fit/predict semantics) is preserved.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os

print(os.listdir("../input")[:20])



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
)

test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_types = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test = pd.read_csv(
    "../input/test.csv",
    usecols=test_cols,
    dtype=test_types,
    parse_dates=["pickup_datetime"],
)

samp = pd.read_csv("../input/sample_submission.csv")



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
m = (train.fare_amount > 0) & (train["passenger_count"] <= 6)
m &= (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
m &= (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
m &= (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
m &= (train.dropoff_longitude > -180) & (train.dropoff_longitude < 180)
train = train[m]



## === cell 4
test.dropna(how="any", axis="rows", inplace=True)
m = test["passenger_count"] <= 6
m &= (test.pickup_latitude > -90) & (test.pickup_latitude < 90)
m &= (test.dropoff_latitude > -90) & (test.dropoff_latitude < 90)
m &= (test.pickup_longitude > -180) & (test.pickup_longitude < 180)
m &= (test.dropoff_longitude > -180) & (test.dropoff_longitude < 180)
test = test[m]



## === cell 5
y = train.fare_amount.values
test_id = test.key




## === cell 6
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 7
def add_geo_features(data):
    plon = data["pickup_longitude"].to_numpy()
    plat = data["pickup_latitude"].to_numpy()
    dlon = data["dropoff_longitude"].to_numpy()
    dlat = data["dropoff_latitude"].to_numpy()

    abs_dlong = np.abs(dlon - plon)
    abs_dlat = np.abs(dlat - plat)

    data["abs_diff_longitude"] = abs_dlong
    data["abs_diff_latitude"] = abs_dlat
    data["manhattan_distance"] = abs_dlong + abs_dlat

    data["squared_long"] = abs_dlong * abs_dlong
    data["squared_lat"] = abs_dlat * abs_dlat

    return data




## === cell 8
def add_time_features(data):
    dt = data["pickup_datetime"]

    hour = dt.dt.hour.to_numpy(dtype=np.int16, copy=False)
    month = dt.dt.month.to_numpy(dtype=np.int16, copy=False)
    year = dt.dt.year.to_numpy(dtype=np.int16, copy=False)

    dow = dt.dt.dayofweek.to_numpy(dtype=np.int8, copy=False)
    day_names = np.array(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        dtype=object,
    )
    day_of_week = day_names[dow]

    day = dt.dt.day.to_numpy(dtype=np.int16, copy=False)
    week_of_month = np.where(
        day <= 7,
        "first",
        np.where(
            day <= 14,
            "second",
            np.where(day <= 21, "third", np.where(day <= 28, "fourth", "fifth")),
        ),
    )

    data["hour"] = pd.Categorical(hour.astype(str))
    data["day_of_week"] = pd.Categorical(day_of_week)
    data["week_of_month"] = pd.Categorical(week_of_month)
    data["month"] = pd.Categorical(month.astype(str))
    data["year"] = pd.Categorical(year.astype(str))

    return data




## === cell 9
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
]

train_feat = train[
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
].copy()

test_feat = test[
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
].copy()

train_feat = add_time_features(train_feat)
train_feat = add_geo_features(train_feat)
train_feat = train_feat[features]

test_feat = add_time_features(test_feat)
test_feat = add_geo_features(test_feat)
test_feat = test_feat[features]

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
]

_cat_maps = {}
for c in cat_cols:
    cats = train_feat[c].cat.categories
    _cat_maps[c] = {k: i for i, k in enumerate(cats)}


def _encode_cats(df, cat_maps):
    out = np.empty((df.shape[0], len(cat_cols)), dtype=np.int16)
    for j, c in enumerate(cat_cols):
        s = df[c].astype(object)
        out[:, j] = s.map(cat_maps[c]).fillna(-1).to_numpy(dtype=np.int16, copy=False)
    return out


X_cat = _encode_cats(train_feat, _cat_maps)
X_cat_test = _encode_cats(test_feat, _cat_maps)

X_num = train_feat[num_cols].to_numpy(dtype=np.float32, copy=False)
X_num_test = test_feat[num_cols].to_numpy(dtype=np.float32, copy=False)

x = np.concatenate([X_cat.astype(np.float32, copy=False), X_num], axis=1)
x_test = np.concatenate([X_cat_test.astype(np.float32, copy=False), X_num_test], axis=1)

x = np.asarray(x, dtype=np.float32, order="C")
x_test = np.asarray(x_test, dtype=np.float32, order="C")



## === cell 10
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2,
    bootstrap=True,
    warm_start=True,
    max_bins=256,
)



## === cell 11
n_fit = min(x.shape[0], len(y))
model.fit(x[:n_fit], y[:n_fit])



## === cell 12
test_pred = model.predict(x_test)

sub = pd.DataFrame()
sub["key"] = test_id.values
sub["fare_amount"] = test_pred
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
