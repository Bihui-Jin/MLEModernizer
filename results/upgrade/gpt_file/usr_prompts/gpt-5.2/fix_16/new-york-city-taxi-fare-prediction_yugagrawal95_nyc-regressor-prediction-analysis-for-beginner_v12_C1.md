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
lightgbm==4.6.0
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

3.60451

# 6. Current score

5.16627

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.98329) has done: 'I fix the boolean filtering bug in the coordinate-cleaning step (cell 20) by building a proper mask with parentheses so pandas doesn’t try to `|` two DataFrames/DatetimeArrays. Then I fix the submission length mismatch by ensuring test rows are not dropped during zero-coordinate filtering; instead, I impute invalid/zero coordinates in the test set and compute predictions for all 9914 keys. Finally, I correct the stacking feature construction bug so the meta-features have the expected shape, without changing the underlying models or training approach, and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 5.06366) has done: 'Your current gap is 4.98329 − 3.60451 = 1.37878 (worse than target; lower RMSE is better), so we should modestly improve generalization without changing the modeling approach. The biggest score drag is a bug in the Haversine distance formula (you’re converting radians twice inside `cos()`), which directly harms the key “distance” feature; fixing it keeps the same feature but makes it correct. Second, your stacking is leaking/wrong because the base models used at test time are not the ones trained on full data (they’re left fitted on the last CV fold); we preserve the same stacking design but refit fresh clones on the full training set after building OOF features, and then use those for test meta-features. Finally, we keep predictions non-negative and ensure the submission stays aligned to all 9914 keys.'
- What this solution (achieved 4.91809) has done: 'The timeout is dominated by repeated model fitting in two separate out-of-fold (OOF) generation passes plus extra pandas indexing overhead inside the stacking loop. I keep the exact ensemble/stacking logic and model set, but eliminate redundant OOF computation by reusing the first OOF matrix for the meta-model training, and train the final base models only once as before. I also make the OOF function strictly NumPy-based (already partly done) and ensure XGBoost/LightGBM use deterministic settings and avoid unnecessary pandas conversions. Finally, I remove expensive no-op EDA calls that still allocate large intermediate objects (e.g., `describe()`, `groupby().count()`) while keeping all filtering/feature engineering/training semantics unchanged.'
- What this solution (achieved 4.96637) has done: 'We’re currently worse than the target (4.91809 vs 3.60451, lower is better), so we should make small, legitimate improvements that reduce RMSE without changing the overall modeling/stacking approach. The biggest low-risk gain here is to make the meta-model training consistent with how its features are generated: train the meta-model on out-of-fold (OOF) predictions for each fold (instead of fitting it once on OOF but predicting with base models trained on all data), then refit the meta-model on full-data base predictions for final test inference. I also fix a subtle but important data mismatch: `_oof_base` is generated from `X_train` but you currently train the meta-model on `y_train` (from a different split) without ensuring alignment; we generate OOF on the exact array used for meta training. These are minimal changes that preserve your exact feature set and model choices while improving generalization toward the target RMSE, and still produce a valid `submission.csv`.'
- What this solution (achieved 4.99703) has done: 'We’re currently worse than the target (4.96637 vs 3.60451 RMSE), so the smallest legitimate improvements should come from better data quality and feature fidelity without changing your model/stacking design. I keep the same feature set (time parts + distance) and the same base models + LightGBM meta-model, but I fix two data issues that commonly inflate RMSE here: (1) passenger_count=0 rows in train (noise) and (2) invalid/zero coordinates in train being dropped rather than imputed consistently with test, which shifts the training distribution. I also make the meta-feature generation for training consistent by using the exact same numeric matrix (numpy) for both OOF creation and final refits (no pandas `.iloc` in stacking path), preserving the same training approach while reducing subtle alignment/type differences.'
- What this solution (achieved 4.8817) has done: 'Your current RMSE (4.99703) is worse than the target (3.60451), so we should make small, legitimate improvements that reduce error without changing your overall approach (distance + time-part features, same base models, LightGBM meta-model, same stacking semantics). The biggest low-risk gain is to stop unintentionally training the meta-model twice on two different feature spaces (OOF features vs full-fit features) and instead keep it consistent: fit meta-model once on OOF, then predict test using base models trained on full train and the same kind of meta-features. In addition, we make sure the test matrix is aligned to the exact training columns (including the same column order) and remove one subtle train/test mismatch by computing medians only from numeric columns actually present. These changes preserve your core logic but should move RMSE down toward the target band while still producing a valid `submission.csv` for all 9914 keys.'
- What this solution (achieved 4.82011) has done: 'Your current RMSE (4.8817) is worse than the target (3.60451), so we should make a small, legitimate improvement that keeps the same overall pipeline (same features, same base models, same stacking idea) but reduces a key source of generalization error. The biggest minimal fix is to generate out-of-fold (OOF) predictions for the meta-model using the *same* KFold scheme and *the same* underlying matrix type (NumPy) that you later use to refit base models and predict test, eliminating subtle train/predict mismatches. Concretely, we reuse your existing `get_oof_predictions` but run it on the full cleaned training set `X, y` (instead of only `X_train, y_train`), then fit the meta-model on those OOF features, refit base models on full `X, y`, and predict test; this keeps semantics identical but typically improves RMSE materially. We also align test columns to `X.columns` (full training columns) rather than `X_train.columns` to avoid any column-order/selection drift from the earlier split.'
- What this solution (achieved 5.5686) has done: 'We should move RMSE down from 4.82011 toward 3.60451 (lower is better) with minimal risk and without changing your overall stacking design. The biggest low-impact gain is to make the cleaning step match what strong baselines do: filter training points to a NYC bounding box and remove unrealistic “fare vs distance” outliers; this keeps the same features/models but reduces label noise that inflates RMSE. To preserve your core logic, we keep the same feature engineering (time parts + Haversine distance) and the same base models and LightGBM meta-model, but we fit the meta-model on OOF predictions and then refit base models on the full cleaned data for test inference as you already do. Finally, we ensure submission alignment stays intact (all 9914 keys, correct columns) and keep predictions non-negative.'
- What this solution (achieved 5.17482) has done: 'The timeout is dominated by repeated model fitting: you train three heavy models (GBR, XGB, LGB) across 5 folds to build OOF predictions, then train them again on full data. To keep identical logic and semantics, the main speed fix is to (1) avoid pandas `.iloc` inside fold loops by using pre-converted NumPy arrays everywhere, and (2) parallelize *across folds/models* with joblib while preventing nested parallelism (so `n_jobs=-1` inside XGB/LGB doesn’t multiply workers). We also remove redundant datetime parsing and make feature creation fully vectorized (single `.assign` / direct dt access) while preserving the same features and filtering. These changes keep the same models, same folds, same training approach, and should cut wall-time dramatically under the 600s limit.'
- What this solution (achieved 5.18399) has done: 'We should move RMSE down from 5.17482 toward 3.60451 (lower is better) with minimal, low-risk changes that keep your exact features (time parts + Haversine distance) and the same stacking setup (GBR + XGB + LGB → LGB meta). The biggest likely score drag still present is that your OOF base predictions are generated with `KFold(random_state=42)` but the meta-model is trained on those OOF features while the (unused) stacking class uses a different split; we make the meta-model’s split scheme explicit and consistent by generating OOF using the same KFold seed we conceptually use for stacking and by fitting the meta-model on OOF produced from that KFold (no architecture change, just consistency). Second, we remove the train/test mismatch caused by filling any missing test columns with `0` during `reindex` (this can create unrealistic coordinates/time parts if a column is missing for any reason); instead we reindex without `fill_value` and then fill with the already-computed `medians` to preserve your imputation semantics. These are minimal, execution-safe changes that should reduce noise and improve generalization without changing models, loops, or feature engineering.'
- What this solution (achieved 5.16627) has done: 'You’re currently worse than the target (5.18399 vs 3.60451 RMSE; lower is better), so we should make a small data-quality improvement that typically gives a clear RMSE gain without changing your feature set or stacking design. The biggest low-risk issue is that you only apply the “reasonable fare per km” filter but not a direct “min fare for non-trivial trips” / “max fare per km” style filter used in strong NYC Taxi baselines; adding a very small set of standard outlier rules reduces label noise while preserving your same distance/time/passenger features and the same models. I also make the OOF split seed consistent (use the same `random_state=156` everywhere) to remove minor train/meta inconsistencies, without altering the approach. Finally, submission alignment stays unchanged (all 9914 keys, correct columns, non-negative fares).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.ioff()


def _noop(*args, **kwargs):
    return None


sns.distplot = _noop
sns.kdeplot = _noop
sns.countplot = _noop
sns.scatterplot = _noop
sns.barplot = _noop


class _NoFacetGrid:
    def __init__(self, *args, **kwargs):
        pass

    def map(self, *args, **kwargs):
        return self


sns.FacetGrid = _NoFacetGrid

np.random.seed(0)



## === cell 1
import os

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

NROWS_TRAIN = 500000  # keep identical core logic

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes_train = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
dtypes_test = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

read_csv_kwargs_train = dict(
    nrows=NROWS_TRAIN,
    usecols=train_usecols,
    dtype=dtypes_train,
    parse_dates=["pickup_datetime"],
)
read_csv_kwargs_test = dict(
    usecols=test_usecols,
    dtype=dtypes_test,
    parse_dates=["pickup_datetime"],
)
try:
    train_data = pd.read_csv(TRAIN_PATH, engine="pyarrow", **read_csv_kwargs_train)
    test_data = pd.read_csv(TEST_PATH, engine="pyarrow", **read_csv_kwargs_test)
except Exception:
    train_data = pd.read_csv(TRAIN_PATH, **read_csv_kwargs_train)
    test_data = pd.read_csv(TEST_PATH, **read_csv_kwargs_test)

train_data.head(), test_data.head()



## === cell 2
pass



## === cell 3
train_data.shape




## === cell 4
def changeDataType(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], utc=True, errors="coerce"
    )


changeDataType(train_data)
print("--" * 40)
changeDataType(test_data)

train_data["fare_amount"] = train_data.fare_amount.astype("float32")



## === cell 5
train_data["pickup_datetime"].head()



## === cell 6
pass



## === cell 7
train_data.isnull().sum()
train_data = train_data.dropna(axis=0)
train_data.isnull().sum()



## === cell 8
pd.set_option("display.float_format", "{:f}".format)
pass



## === cell 9
plt.figure(figsize=(8, 5), dpi=80)
sns.distplot(train_data["fare_amount"], color="red", kde=False)
train_data = train_data.loc[train_data["fare_amount"] > 0]
train_data["fare_amount"]
pass



## === cell 10
sns.distplot(a=train_data.fare_amount, kde=False)



## === cell 11
p = pd.cut(train_data.fare_amount, 3)
p.value_counts()



## === cell 12
train_data = train_data[train_data.fare_amount < 400]



## === cell 13
sns.kdeplot(data=train_data.fare_amount)



## === cell 14
sns.countplot(x=train_data.passenger_count)



## === cell 15
train_data.passenger_count.describe()
train_data = train_data[
    (train_data.passenger_count >= 1) & (train_data.passenger_count <= 6)
]



## === cell 16
sns.distplot(a=train_data.passenger_count, kde=False)



## === cell 17
pass



## === cell 18
mask_bad = (
    (train_data["pickup_latitude"] < -90)
    | (train_data["pickup_latitude"] > 90)
    | (train_data["pickup_longitude"] < -180)
    | (train_data["pickup_longitude"] > 180)
    | (train_data["dropoff_longitude"] < -180)
    | (train_data["dropoff_longitude"] > 180)
    | (train_data["dropoff_latitude"] < -90)
    | (train_data["dropoff_latitude"] > 90)
)
train_data = train_data.loc[~mask_bad].copy()



## === cell 19
NYC_BOUNDS = {
    "pickup_longitude": (-74.3, -72.9),
    "dropoff_longitude": (-74.3, -72.9),
    "pickup_latitude": (40.5, 41.0),
    "dropoff_latitude": (40.5, 41.0),
}

for col, (lo, hi) in NYC_BOUNDS.items():
    train_data = train_data[train_data[col].between(lo, hi)]



## === cell 20
sns.scatterplot(x=train_data.pickup_latitude, y=train_data.pickup_longitude)
sns.scatterplot(x=train_data.dropoff_latitude, y=train_data.dropoff_longitude)




## === cell 21
def degree_to_radion(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 22
zero_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
train_data[zero_cols] = train_data[zero_cols].mask(train_data[zero_cols].eq(0), np.nan)
test_data[zero_cols] = test_data[zero_cols].mask(test_data[zero_cols].eq(0), np.nan)

for col, (lo, hi) in NYC_BOUNDS.items():
    if col in test_data.columns:
        test_data[col] = test_data[col].clip(lower=lo, upper=hi)

coord_medians = train_data[zero_cols].median(numeric_only=True)
train_data[zero_cols] = train_data[zero_cols].fillna(coord_medians)
test_data[zero_cols] = test_data[zero_cols].fillna(coord_medians)

train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
).astype("float32")

test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
).astype("float32")



## === cell 23
pass



## === cell 24
pass



## === cell 25
p = pd.cut(train_data.distance, 10)
p.value_counts()



## === cell 26
train_data = train_data.loc[train_data.distance < 200]  # keep as-is



## === cell 27
pass



## === cell 28
sns.distplot(train_data.distance, kde=False)



## === cell 29
dist = train_data["distance"].astype("float32")
fare = train_data["fare_amount"].astype("float32")

mask_reasonable = (dist > 0.05) & (fare / dist < 40.0)  # existing $/km cap
mask_more = (
    (
        (dist < 1.0)
        | (fare >= 3.0)  # if trip is >=1km, enforce at least a small base fare
    )
    & (
        (
            fare <= 250.0
        )  # extra safety (already <400, but tighter reduces heavy outliers)
    )
    & (
        (
            fare / (dist + 1e-3) <= 60.0
        )  # gentle secondary cap to remove extreme spikes without over-filtering
    )
)

train_data = train_data.loc[mask_reasonable & mask_more].copy()



## === cell 30
train_data = train_data.drop(columns="key")



## === cell 31
pass



## === cell 32
test_data_key = test_data["key"].copy()
test_data = test_data.drop(columns="key")



## === cell 33
test_data.head()



## === cell 34
for df in (train_data, test_data):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Date"] = dt.day
    df["Day of Week"] = dt.dayofweek
    df["Hour"] = dt.hour



## === cell 35
train_data.head()



## === cell 36
test_data.head()



## === cell 37
sns.scatterplot(x=train_data["passenger_count"], y=train_data["fare_amount"])



## === cell 38
sns.scatterplot(x=train_data["distance"], y=train_data["fare_amount"])



## === cell 39
g = sns.FacetGrid(train_data, col="Year")
g.map(sns.scatterplot, "distance", "fare_amount")



## === cell 40
pass



## === cell 41
train_data[(train_data.distance > 100) & (train_data.fare_amount < 50)]



## === cell 42
sns.scatterplot(x=train_data["Year"], y=train_data["fare_amount"])



## === cell 43
pass



## === cell 44
sns.scatterplot(
    x=train_data["Month"], y=train_data["fare_amount"], hue=train_data["Year"]
)



## === cell 45
sns.scatterplot(x=train_data["Month"], y=train_data["fare_amount"])



## === cell 46
w = sns.FacetGrid(train_data, col="Year")
w.map(sns.scatterplot, "Month", "fare_amount")



## === cell 47
sns.barplot(x=train_data["Day of Week"], y=train_data["fare_amount"])



## === cell 48
plt.figure(figsize=(10, 10), dpi=150)
w = sns.FacetGrid(train_data, col="Month")
w.map(sns.barplot, "Day of Week", "fare_amount")



## === cell 49
sns.barplot(x=train_data["Hour"], y=train_data["fare_amount"])



## === cell 50
fill_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "distance",
    "Year",
    "Month",
    "Date",
    "Day of Week",
    "Hour",
]
medians = train_data[fill_cols].median(numeric_only=True)
train_data[fill_cols] = train_data[fill_cols].fillna(medians)
test_data[fill_cols] = test_data[fill_cols].fillna(medians)



## === cell 51
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 52
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 53
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



## === cell 54
pass



## === cell 55
ran_for_reg = RandomForestRegressor(max_depth=400, n_jobs=-1, random_state=0)
y_ranfor_pred = None
error = None
error



## === cell 56
sns.barplot(x=np.array([]), y=getattr(X_test, "columns", []))



## === cell 57
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bagreg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=10,
    bootstrap=True,
    random_state=0,
    n_jobs=-1,
)
y_bagg_pred = None
error = None
error



## === cell 58
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

adareg = AdaBoostRegressor(DecisionTreeRegressor(), random_state=0)
y_adareg_pred = None
error = None
error



## === cell 59
from sklearn.ensemble import GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor(random_state=0)
y_gradient_pred = None
error = None
error



## === cell 60
from xgboost import XGBRegressor

xgreg = XGBRegressor(
    n_jobs=-1,
    random_state=0,
    verbosity=0,
    tree_method="hist",
)
y_xgreg_pred = None
error = None
error



## === cell 61
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(n_jobs=-1, random_state=0)
y_lgb_pred = None
error = None
error



## === cell 62
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



## === cell 63
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=156)
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf
        )
    )
    return rmse




## === cell 64
from sklearn.base import BaseEstimator


class AverageModel(BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)




## === cell 65
from sklearn.base import clone
from joblib import Parallel, delayed
import os as _os

base_model = [gradient_reg, xgreg, model_lgb]


def get_oof_predictions(X_df, y_ser, models, kfold):
    X_values = X_df.to_numpy(dtype=np.float32, copy=False)
    y_values = y_ser.to_numpy(dtype=np.float32, copy=False)

    splits = list(kfold.split(X_values, y_values))
    oof = np.zeros((X_values.shape[0], len(models)), dtype=np.float32)

    _inner_threads = max(1, (_os.cpu_count() or 2) // 2)

    def _fit_predict_one(model_i, split_i):
        tr_idx, ho_idx = splits[split_i]
        m = clone(models[model_i])
        if hasattr(m, "n_jobs"):
            try:
                m.set_params(n_jobs=_inner_threads)
            except Exception:
                pass
        if hasattr(m, "nthread"):
            try:
                m.set_params(nthread=_inner_threads)
            except Exception:
                pass
        m.fit(X_values[tr_idx], y_values[tr_idx])
        pred = m.predict(X_values[ho_idx]).astype(np.float32, copy=False)
        return model_i, ho_idx, pred

    tasks = [(mi, si) for mi in range(len(models)) for si in range(len(splits))]
    results = Parallel(n_jobs=-1, prefer="processes", batch_size=1)(
        delayed(_fit_predict_one)(mi, si) for mi, si in tasks
    )
    for model_i, ho_idx, pred in results:
        oof[ho_idx, model_i] = pred
    return oof




## === cell 66
kf_avg = KFold(n_splits=n_folds, shuffle=True, random_state=156)
_oof_base = get_oof_predictions(X, y, base_model, kf_avg)

rmse_folds = []
y_np = y.to_numpy(dtype=np.float32, copy=False)
X_idx = np.arange(len(y_np))
for _, ho_idx in kf_avg.split(X_idx, y_np):
    pred_avg = _oof_base[ho_idx].mean(axis=1)
    rmse_folds.append(np.sqrt(mean_squared_error(y_np[ho_idx], pred_avg)))
score = np.array(rmse_folds, dtype=np.float32)
score



## === cell 67
score.mean()



## === cell 68
from sklearn.base import RegressorMixin, TransformerMixin, clone


class stackingModel(BaseEstimator):
    def __init__(self, base_model, meta_model, k_fold=5):
        self.base_model = base_model
        self.meta_model = meta_model
        self.k_fold = k_fold

    def fit(self, X, y):
        kfold = KFold(n_splits=self.k_fold, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros(
            (X.shape[0], len(self.base_model)), dtype=np.float32
        )

        for i, model in enumerate(self.base_model):
            for train_index, holdout_index in kfold.split(X, y):
                m = clone(model)
                m.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = m.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.base_model])
        return self.meta_model.predict(meta_features)




## === cell 69
from sklearn.model_selection import KFold

out_of_fold_predictions = _oof_base
out_of_fold_predictions



## === cell 70
meta_model = lgb.LGBMRegressor(n_jobs=-1, random_state=0)
meta_model.fit(out_of_fold_predictions, y_np)

final_base_models = [clone(m) for m in base_model]
X_np_full = X.to_numpy(dtype=np.float32, copy=False)

import os as _os

_inner_threads = max(1, (_os.cpu_count() or 2) // 2)

for m in final_base_models:
    if hasattr(m, "n_jobs"):
        try:
            m.set_params(n_jobs=_inner_threads)
        except Exception:
            pass
    if hasattr(m, "nthread"):
        try:
            m.set_params(nthread=_inner_threads)
        except Exception:
            pass
    m.fit(X_np_full, y_np)

test_data_aligned = test_data.reindex(columns=X.columns)
test_data_aligned = test_data_aligned.fillna(medians.reindex(test_data_aligned.columns))

X_test_submit_np = test_data_aligned.to_numpy(dtype=np.float32, copy=False)

feature_data = np.column_stack(
    [model.predict(X_test_submit_np) for model in final_base_models]
).astype(np.float32, copy=False)

meta_y = meta_model.predict(feature_data)
meta_y = np.maximum(meta_y, 0)

submission = pd.DataFrame(
    {"key": test_data_key, "fare_amount": meta_y}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
submission.head()
