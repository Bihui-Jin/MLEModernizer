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

3.99125

# 6. Current score

5.1929

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.00081) has done: 'I fix the pandas datetime API break (`weekday_name` removed) and ensure the engineered time features exist, so the feature selection in your pipeline no longer fails. I also prevent the `pickup_datetime` timestamp column from leaking into the model matrix (it caused the RandomForest `float()`/Timestamp error) by reading only the needed columns for train/test and then selecting only numeric/dummy-encoded features for modeling. Finally, I keep the same core logic (same features, one-hot encoding, RandomForestRegressor fit/predict) and make sure a valid `submission.csv` with columns `key,fare_amount` is always written.'
- What this solution (achieved 4.75609) has done: 'The timeout is dominated by fitting a 300-tree RandomForest on 1,000,000 rows with very wide one-hot features (string categories) and by expensive pandas operations (concat/copies, slow `.map(week_num)`, and `get_dummies` on object columns). The core model/training loop remain identical, but we reduce overhead around it: avoid unnecessary copies/concat, vectorize `week_of_month` creation, convert time parts to pandas `category` (same semantics, faster/leaner dummies), and avoid materializing extra intermediate DataFrames. We also enable Intel scikit-learn acceleration if available (same estimator semantics) and pass `X` as numpy arrays for faster fit/predict without changing results. These changes preserve the same data filters, the same engineered features, and the same RandomForest hyperparameters.'
- What this solution (achieved 5.1929) has done: 'The timeout is almost certainly dominated by fitting a 300-tree `RandomForestRegressor` on a high-dimensional sparse one-hot matrix; scikit-learn’s RandomForest doesn’t efficiently consume sparse CSR, so it densifies/handles it in a very slow path. To preserve the exact same feature engineering and model hyperparameters while making training fast, the key change is to convert the final design matrix to a single dense `float32` array once (after one-hot), which is feasible at your row count and fixed category sizes, and avoids repeated/implicit densification inside each tree. I also remove a couple of costly pandas operations (`.loc` assignments that can trigger copies) by building categoricals and numeric arrays in a copy-safe way, and I ensure we don’t do redundant DataFrame copies during feature creation. These are correctness-preserving (same data, same categories, same model) but reduce overhead enough to fit within the 600s limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "../input"
print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))

np.random.seed(42)



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

test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

TRAIN_NROWS = 300_000  # tuned for <600s in typical Kaggle CPU environments

train = pd.read_csv(
    f"{INPUT_DIR}/train.csv",
    nrows=TRAIN_NROWS,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    f"{INPUT_DIR}/test.csv",
    usecols=test_cols,
    parse_dates=["pickup_datetime"],
)
samp = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")



## === cell 3
train = train.dropna(how="any", axis="rows")
m = (
    (train.fare_amount > 0)
    & (train["passenger_count"] <= 6)
    & (train.pickup_latitude > -90)
    & (train.pickup_latitude < 90)
    & (train.dropoff_latitude > -90)
    & (train.dropoff_latitude < 90)
    & (train.pickup_longitude > -180)
    & (train.pickup_longitude < 180)
    & (train.dropoff_longitude > -180)
    & (train.dropoff_longitude < 180)
)
train = train.loc[m]

MAX_TRAIN_ROWS = 250_000
if len(train) > MAX_TRAIN_ROWS:
    train = train.sample(n=MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)



## === cell 4
y = train.fare_amount.to_numpy(copy=False)
test_id = test.key

train_Xbase = train.drop(columns=["fare_amount"])
test_Xbase = test.drop(columns=["key"])




## === cell 5
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




## === cell 6
def add_time_features(data):
    dt = data["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=True)
        data["pickup_datetime"] = dt

    hour = dt.dt.hour.to_numpy()
    month = dt.dt.month.to_numpy()
    year = dt.dt.year.to_numpy()
    dom = dt.dt.day.to_numpy()
    dow = dt.dt.dayofweek.to_numpy()

    wom = np.full(dom.shape[0], 4, dtype=np.int8)  # default "fifth"
    wom[dom <= 7] = 0
    wom[(dom > 7) & (dom <= 14)] = 1
    wom[(dom > 14) & (dom <= 21)] = 2
    wom[(dom > 21) & (dom <= 28)] = 3

    data["hour"] = pd.Categorical(hour.astype(np.int16, copy=False))
    data["day_of_week"] = pd.Categorical(dow.astype(np.int8, copy=False))
    data["week_of_month"] = pd.Categorical(wom)
    data["month"] = pd.Categorical(month.astype(np.int8, copy=False))
    data["year"] = pd.Categorical(year.astype(np.int16, copy=False))

    data.drop("pickup_datetime", axis=1, inplace=True)
    return data




## === cell 7
def add_geo_features(data):
    plon = data["pickup_longitude"].to_numpy()
    plat = data["pickup_latitude"].to_numpy()
    dlon = data["dropoff_longitude"].to_numpy()
    dlat = data["dropoff_latitude"].to_numpy()

    abs_diff_long = np.abs(dlon - plon)
    abs_diff_lat = np.abs(dlat - plat)

    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat
    data["manhattan_distance"] = abs_diff_long + abs_diff_lat

    squared_long = abs_diff_long * abs_diff_long
    squared_lat = abs_diff_lat * abs_diff_lat
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat
    data["euclid_disance"] = np.sqrt(squared_long + squared_lat)

    return data




## === cell 8
train_feat = add_geo_features(add_time_features(train_Xbase.copy(deep=False)))
test_feat = add_geo_features(add_time_features(test_Xbase.copy(deep=False)))



## === cell 9
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

train_feat = train_feat[features]
test_feat = test_feat[features]

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

fixed_categories = {
    "hour": list(range(0, 24)),
    "day_of_week": list(range(0, 7)),
    "week_of_month": list(range(0, 5)),  # 0..4
    "month": list(range(1, 13)),
    "year": list(range(2009, 2016)),  # NYC taxi dataset years typically 2009-2015
}

for c in cat_cols:
    train_feat[c] = pd.Categorical(train_feat[c], categories=fixed_categories[c])
    test_feat[c] = pd.Categorical(test_feat[c], categories=fixed_categories[c])

train_feat[num_cols] = train_feat[num_cols].fillna(0.0)
test_feat[num_cols] = test_feat[num_cols].fillna(0.0)

X_num = train_feat[num_cols].to_numpy(dtype=np.float32, copy=False)
X_test_num = test_feat[num_cols].to_numpy(dtype=np.float32, copy=False)

try:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
        categories=[fixed_categories[c] for c in cat_cols],
    )
except TypeError:
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True,
        dtype=np.float32,
        categories=[fixed_categories[c] for c in cat_cols],
    )

X_cat = ohe.fit_transform(train_feat[cat_cols])
X_test_cat = ohe.transform(test_feat[cat_cols])

X_sparse = sparse.hstack([sparse.csr_matrix(X_num), X_cat], format="csr")
X_test_sparse = sparse.hstack([sparse.csr_matrix(X_test_num), X_test_cat], format="csr")
X_sparse.sort_indices()
X_test_sparse.sort_indices()

X = X_sparse.toarray().astype(np.float32, copy=False)
X_test = X_test_sparse.toarray().astype(np.float32, copy=False)



## === cell 10
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    max_bins=256,  # histogram optimization
)



## === cell 11
model.fit(X, y)



## === cell 12
test_pred = model.predict(X_test)

sub = pd.DataFrame()
sub["key"] = test_id.values
sub["fare_amount"] = test_pred.astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
