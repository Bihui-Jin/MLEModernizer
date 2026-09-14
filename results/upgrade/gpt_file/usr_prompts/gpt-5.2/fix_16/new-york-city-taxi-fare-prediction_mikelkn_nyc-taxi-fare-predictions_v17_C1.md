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

3.8987

# 6. Current score

5.63669

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.15099) has done: 'I fix the runtime errors by updating deprecated dtype checks (`np.object`), correcting the datetime feature function (it currently returns `None` due to `inplace=True`), and ensuring scikit-learn parameters like `max_iter` are integers. I also fix the Haversine calculation bugs (wrong lambda and swapped diffs) because they directly degrade feature quality and therefore RMSE, while keeping the same overall feature approach and RandomForest training logic. To avoid crashing in notebook-to-script execution, I remove the IPython magic and make the plotting optional. Finally, I ensure the submission is written with the required columns (`key`, `fare_amount`) and a `.csv` suffix.'
- What this solution (achieved 9.92033) has done: 'You don’t currently have a Kaggle score because the script filters the *test* rows and then only submits predictions for the remaining subset, which makes the submission invalid (wrong row count/keys). I keep the same feature engineering and RandomForest training, but remove the test-set filtering so every test key is predicted and the submission matches `sample_submission.csv` exactly. To avoid train/test feature mismatches, I apply the same cleaning only to train, and add a minimal safeguard to reindex test features to the exact training feature columns before predicting. This should produce a valid `.csv` and move RMSE toward your target by using the full intended evaluation set.'
- What this solution (achieved 9.10586) has done: 'Your current leaderboard RMSE (9.92033) is far above the target (3.8987), so we need a modest, legitimate accuracy improvement without changing the overall approach (same features and RandomForest). The biggest issue is that the RandomForest is trained on `X_train` created from `train_test_split` (only 75% of the already-subsampled 1M rows), which unnecessarily weakens the final model; training the exact same RF on all available cleaned training data usually improves RMSE substantially while keeping architecture/logic identical. I keep your feature engineering and the RF hyperparameters unchanged, but refit the RF on full `X_1, y_1` after the model comparison section, then predict as before. I also keep the test-column reindexing and submission writing unchanged to ensure a valid CSV.'
- What this solution (achieved 8.35883) has done: 'Your RMSE (9.10586) is far worse than the target (3.8987), so we should make a small, legitimate improvement without changing the overall “feature engineering + RandomForestRegressor” approach. The biggest accuracy issue left is that the model never sees a key NYC signal: absolute pickup/dropoff location (only distance + time), because you drop lat/long entirely; keeping those coordinates as features is a minimal change that usually reduces RMSE a lot while preserving the same model and training loop. To keep train/test processing consistent, we drop the raw radian-diff helper columns but retain the original coordinate columns in both train and test. Submission writing and row alignment stay unchanged so you still get a valid `key,fare_amount` CSV.'
- What this solution (achieved 5.32318) has done: 'Your current RMSE (8.35883) is far above the target (3.8987), so we need a small but meaningful accuracy improvement while keeping the same overall “feature engineering + RandomForestRegressor” core logic. The biggest remaining issue is that the model is being trained on raw `fare_amount`, which has a long tail; training the exact same RandomForest on `log1p(fare_amount)` and then applying `expm1` at inference typically improves RMSE substantially in this competition without changing the model type, features, or training loop. I also make the datetime parsing robust to mixed formats/UTC by using `errors='coerce'` and ensure any resulting NaNs are filled consistently, preventing silent feature corruption. Submission writing, keys, row counts, and file naming stay the same to guarantee a valid `.csv`.'
- What this solution (achieved 5.32318) has done: 'Your current RMSE (5.32318) is worse than the target (3.8987), so we should make a small, legitimate accuracy improvement without changing the overall “feature engineering + RandomForestRegressor + log1p target” core. The biggest remaining issue is inconsistent missing-value handling: we fill time/haversine NaNs but leave any NaNs created in `distance_travelled/10e3` (or other numeric columns) untouched, and RandomForest cannot handle NaNs—this can silently reduce effective training rows or cause unstable behavior depending on where NaNs appear. I add a minimal, train-driven median imputation for all remaining numeric feature columns (both train and test), preserving your features and model while improving stability/fit quality. I also ensure the `key` column is removed from training features (currently it’s only dropped from `train`, not `test` until later, but we keep semantics the same) and keep the submission format identical.'
- What this solution (achieved 5.38953) has done: 'We need to move RMSE down from 5.32318 toward 3.8987 (lower is better), so we make small, high-impact fixes without changing the overall “engineer a few numeric features + RandomForestRegressor on log1p(target)” approach. The main accuracy issue left is that the model is trained on only 1M rows *without* removing the well-known noisy/outlier region near LaGuardia (a huge source of label noise in this dataset); adding the standard “drop near-airport zero-ish fare anomalies” filter is a minimal data-cleaning change that usually improves RMSE noticeably. We also align the `weekday` feature to the true weekday (0–6) rather than day-of-month (this is a one-line bugfix that keeps the same feature slot but makes it meaningful). Finally, we keep the same submission writing logic but ensure train/test feature columns are aligned and fully imputed as before.'
- What this solution (achieved 5.38953) has done: 'Your current RMSE (5.38953, lower is better) is still well above the target (3.8987), so we should make a small, legitimate accuracy improvement while keeping the same core pipeline (same engineered features + RandomForestRegressor trained on log1p(target)). The most impactful minimal change here is to fix a feature mismatch bug: the RandomForest is trained on `X_1` built from `train_1` (which still contains the `key` column), while test-time you explicitly drop `key`—this silently degrades learning and generalization. I also ensure the train/test feature matrices are aligned explicitly from the same `feature_cols` list before fitting/predicting (no new features, no new models), and keep the submission writing exactly as required.'
- What this solution (achieved 5.70066) has done: 'Your current RMSE (5.38953, lower is better) is still far above the target (3.8987), so we should make a small, legitimate improvement without changing the core “same features + RandomForestRegressor on log1p(target)” pipeline. The biggest remaining accuracy gap is from residual outliers/noisy rows that slip past the current filters; tightening the standard NYC Taxi cleaning (geographic bounds + fare-per-distance sanity + minimum trip distance) typically reduces RMSE materially while keeping the same modeling approach. I also ensure the `key` column is dropped consistently before any feature list is created (so train/test semantics match exactly) and keep your submission writing unchanged (same columns, same row count, `.csv`). All changes are purely data-cleaning / feature-matrix consistency, so the architecture/training loop/loss remain identical.'
- What this solution (achieved 5.70562) has done: 'We need to move RMSE down from 5.70066 toward 3.8987 (lower is better), so the smallest likely gain without changing the core “feature engineering + RandomForestRegressor on log1p(target)” pipeline is to stop throwing away signal in the timestamps. I keep the exact same models and training flow, but fix the datetime parsing to use local NYC time (extract hour/weekday correctly) and add the standard cyclical encoding for hour/weekday (a minimal feature tweak that preserves semantics while helping tree splits). I also ensure the same datetime handling is applied to both train and test deterministically (filling NaTs before extracting), keeping all paths and the submission format unchanged. Everything still trains the same RF and writes `NYCtaxiFare_prediction.csv` with `key,fare_amount`.'
- What this solution (achieved 5.59595) has done: 'Your RMSE (5.70562, lower is better) is still well above the target (3.8987), so we should make the smallest, high-impact accuracy improvement while keeping the exact same “engineered numeric features + RandomForestRegressor on log1p(target)” core. The biggest remaining gap is that `max_features=5` is artificially limiting splits despite having many informative features (coords + haversine + time + cyclic encodings), so we change it to the standard RF default behavior (`max_features=1.0`) to use all features at each split, without changing the model type or training loop. To keep behavior stable and avoid accidental train/test drift, we also (minimally) enforce identical `feature_cols` ordering and apply the same column alignment at prediction time (already present). Everything still runs end-to-end and writes a valid `NYCtaxiFare_prediction.csv` with `key,fare_amount`.'
- What this solution (achieved 5.60948) has done: 'Your current RMSE (5.59595, lower is better) is still far above the target (3.8987), so we make the smallest changes that typically yield a meaningful RMSE drop without changing the core pipeline (same engineered features + RandomForestRegressor on log1p(target)). The biggest remaining systematic error is that `passenger_count` in the test set can include 0 and 9 (common in this competition), but you only trained on 1–6; clipping test passenger_count into the same training domain avoids out-of-distribution splits at inference. Also, the raw `pickup_longitude/latitude` and `dropoff_longitude/latitude` can occasionally be outside the NYC box in test; clipping them to the same bounds you trained on is a minimal, legitimate preprocessing alignment that usually improves leaderboard RMSE. Finally, we keep the exact model/training loop and submission format unchanged, only adding these train-domain alignment steps before feature generation/prediction.'
- What this solution (achieved 5.63669) has done: 'We need to move RMSE down from 5.60948 toward 3.8987 (lower is better), so we make the smallest changes that improve signal without changing your overall pipeline (same feature engineering + RandomForestRegressor trained on log1p(target)). The biggest remaining systematic issue is that the model can output very large fares for rare long-distance/time patterns, and RMSE is extremely sensitive to these outliers; adding a train-derived cap on predictions (winsorization) is a minimal, legitimate post-processing step aligned with the metric. To keep semantics consistent and avoid train/test drift, we compute the cap from the cleaned training target distribution and apply it after `expm1` (and after non-negativity clipping). Everything else (data loading, cleaning, features, model hyperparameters/training, and submission format) remains unchanged and still writes a valid `NYCtaxiFare_prediction.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

CANDIDATE_INPUT_DIRS = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
]

INPUT_DIR = None
for d in CANDIDATE_INPUT_DIRS:
    if (
        os.path.exists(d)
        and os.path.isfile(os.path.join(d, "train.csv"))
        and os.path.isfile(os.path.join(d, "test.csv"))
    ):
        INPUT_DIR = d
        break

if INPUT_DIR is None:
    INPUT_DIR = "../input" if os.path.exists("../input") else "/kaggle/input"
    raise FileNotFoundError(
        f"Could not find train.csv/test.csv in any of: {CANDIDATE_INPUT_DIRS}. "
        f"Last tried INPUT_DIR={INPUT_DIR} with files={os.listdir(INPUT_DIR)[:20] if os.path.exists(INPUT_DIR) else 'MISSING'}"
    )

print("Input dir:", INPUT_DIR)
print("Files:", sorted(os.listdir(INPUT_DIR))[:50])



## === cell 1
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"), nrows=1_000_000)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes.value_counts()



## === cell 6
test.dtypes.value_counts()



## === cell 7
train.isnull().sum()



## === cell 8
train = train.dropna()
train.isnull().sum()



## === cell 9
(train == 0).astype(int).sum()



## === cell 10
train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 500)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-74.3, -73.6))
    & (train["dropoff_longitude"].between(-74.3, -73.6))
    & (train["pickup_latitude"].between(40.4, 41.0))
    & (train["dropoff_latitude"].between(40.4, 41.0))
].copy()

lga_lon, lga_lat = -73.8740, 40.7769
near_lga = (
    (train["pickup_longitude"].between(lga_lon - 0.05, lga_lon + 0.05))
    & (train["pickup_latitude"].between(lga_lat - 0.05, lga_lat + 0.05))
) | (
    (train["dropoff_longitude"].between(lga_lon - 0.05, lga_lon + 0.05))
    & (train["dropoff_latitude"].between(lga_lat - 0.05, lga_lat + 0.05))
)
train = train[~(near_lga & (train["fare_amount"] < 2.5))].copy()

lon1 = np.radians(train["pickup_longitude"].to_numpy())
lat1 = np.radians(train["pickup_latitude"].to_numpy())
lon2 = np.radians(train["dropoff_longitude"].to_numpy())
lat2 = np.radians(train["dropoff_latitude"].to_numpy())
dlat = lat2 - lat1
dlon = lon2 - lon1
latm = 0.5 * (lat1 + lat2)
r_km = 6371.0
approx_km = r_km * np.sqrt((dlat) ** 2 + (np.cos(latm) * dlon) ** 2)

min_km = 0.1
fare_per_km = train["fare_amount"].to_numpy() / np.maximum(approx_km, 1e-3)
mask = (approx_km >= min_km) & (fare_per_km > 0.5) & (fare_per_km < 200.0)
train = train.loc[mask].copy()



## === cell 11
test = test.copy()



## === cell 12
test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=6)
test["pickup_longitude"] = test["pickup_longitude"].clip(lower=-74.3, upper=-73.6)
test["dropoff_longitude"] = test["dropoff_longitude"].clip(lower=-74.3, upper=-73.6)
test["pickup_latitude"] = test["pickup_latitude"].clip(lower=40.4, upper=41.0)
test["dropoff_latitude"] = test["dropoff_latitude"].clip(lower=40.4, upper=41.0)



## === cell 13
(train == 0).astype(int).sum()



## === cell 14
train.shape



## === cell 15
train.describe()



## === cell 16
train.describe()



## === cell 17
train.dtypes.value_counts()



## === cell 18
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 19
train.drop("key", axis=1, inplace=True)
train.head()



## === cell 20
import datetime as dt


def date_extraction(data: pd.DataFrame) -> pd.DataFrame:
    dt_utc = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=True)
    dt_local = dt_utc.dt.tz_convert("America/New_York")
    dt_local = dt_local.fillna(pd.Timestamp("2009-01-01", tz="America/New_York"))

    data["year"] = dt_local.dt.year.astype(np.int16)
    data["month"] = dt_local.dt.month.astype(np.int8)
    data["weekday"] = dt_local.dt.weekday.astype(np.int8)
    data["hour"] = dt_local.dt.hour.astype(np.int8)

    data["hour_sin"] = np.sin(2.0 * np.pi * (data["hour"].to_numpy() / 24.0))
    data["hour_cos"] = np.cos(2.0 * np.pi * (data["hour"].to_numpy() / 24.0))
    data["wday_sin"] = np.sin(2.0 * np.pi * (data["weekday"].to_numpy() / 7.0))
    data["wday_cos"] = np.cos(2.0 * np.pi * (data["weekday"].to_numpy() / 7.0))

    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


date_extraction(train)



## === cell 21
train.head()



## === cell 22
date_extraction(test)
test.head()




## === cell 23
def long_lat_distance(x: pd.DataFrame) -> pd.DataFrame:
    x["Longitude_distance"] = np.radians(x["pickup_longitude"] - x["dropoff_longitude"])
    x["Latitude_distance"] = np.radians(x["pickup_latitude"] - x["dropoff_latitude"])
    x["distance_travelled/10e3"] = (
        (x["Longitude_distance"] ** 2 + x["Latitude_distance"] ** 2) ** 0.5
    ) * 1000
    return x




## === cell 24
for x in [train, test]:
    long_lat_distance(x)

train.head()




## === cell 25
def harvesine(x: pd.DataFrame) -> pd.DataFrame:
    r = 6371000.0
    theta_1 = np.radians(x["dropoff_latitude"])
    theta_2 = np.radians(x["pickup_latitude"])
    lambda_1 = np.radians(x["dropoff_longitude"])
    lambda_2 = np.radians(x["pickup_longitude"])

    theta_diff = theta_2 - theta_1
    lambda_diff = lambda_2 - lambda_1

    a = (
        np.sin(theta_diff / 2.0) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin(lambda_diff / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(a**0.5, (1.0 - a) ** 0.5)
    x["harvesine/km"] = (r * c) / 1000.0
    return x




## === cell 26
for x in [train, test]:
    harvesine(x)

train.head()



## === cell 27
train.dtypes.value_counts()



## === cell 28
train.head()



## === cell 29
test.head()



## === cell 30
train.describe()



## === cell 31
print("Are there any nulls in the train data:")
print(train.isnull().sum())

print("\nAre there any nulls in the test data:")
print(test.isnull().sum())



## === cell 32
time_cols = [
    "year",
    "month",
    "weekday",
    "hour",
    "hour_sin",
    "hour_cos",
    "wday_sin",
    "wday_cos",
]
for c in time_cols:
    if c in train.columns:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        if c in test.columns:
            test[c] = test[c].fillna(med)

train["harvesine/km"] = train["harvesine/km"].fillna(train["harvesine/km"].median())
test["harvesine/km"] = test["harvesine/km"].fillna(train["harvesine/km"].median())

train_num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
for c in train_num_cols:
    if c == "fare_amount":
        continue
    med = train[c].median()
    train[c] = train[c].fillna(med)
    if c in test.columns:
        test[c] = test[c].fillna(med)

print("Remaining NaNs in train (after fill):", int(train.isnull().sum().sum()))
print("Remaining NaNs in test (after fill):", int(test.isnull().sum().sum()))



## === cell 33
from sklearn.ensemble import RandomForestRegressor

feature_cols = [c for c in train.columns if c != "fare_amount"]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 34
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)
correlations



## === cell 35
try:
    import matplotlib.pyplot as plt

    ax = correlations.plot(kind="bar")
    ax.set(ylim=[-1, 100], ylabel="abs(pearson correlation)*100")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 36
train.head()



## === cell 37
train_1 = train.drop(
    [
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)
train_1.head()



## === cell 38
train_1.head()



## === cell 39
train_1.describe()



## === cell 40
test_1 = test.drop(
    [
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)
test_1.head()



## === cell 41
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV


def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))




## === cell 42
if "key" in train_1.columns:
    train_1 = train_1.drop("key", axis=1)

feat_cols = [c for c in train_1.columns if c != "fare_amount"]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)



## === cell 43
lr = LinearRegression().fit(X_train, y_train)
lr_rmse = rmse(y_test, lr.predict(X_test))
print(lr_rmse)



## === cell 44
try:
    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    ax.plot(y_test, lr.predict(X_test), marker="o", ls="")
    lim = (0, y_test.max())
    ax.set(
        xlabel="actual fare amount",
        ylabel="predicted_amount",
        xlim=lim,
        ylim=(0, 100),
        title="Linear regression results",
    )
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 45
alphas = [0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5]
rr = RidgeCV(alphas=alphas, cv=4).fit(X_train, y_train)
rr_rmse = rmse(y_test, rr.predict(X_test))
print(rr.alpha_, rr_rmse)



## === cell 46
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])
la = LassoCV(alphas=alphas, max_iter=int(5e4), cv=4).fit(X_train, y_train)
la_rmse = rmse(y_test, la.predict(X_test))
print(la.alpha_, la_rmse)



## === cell 47
l1_ratios = np.linspace(0.1, 0.5, 5)
alphas = np.array([0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5])

en = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, max_iter=int(1e4), cv=4).fit(
    X_train, y_train
)
en_rmse = rmse(y_test, en.predict(X_test))
print(en.alpha_, en.l1_ratio_, en_rmse)



## === cell 48
rf = RandomForestRegressor(
    n_estimators=100,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)

rf = rf.fit(X_1, np.log1p(y_1))



## === cell 49
labels = ["Linear", "lasso", "Ridge", "Elastic-Net"]
models_rmse = [lr_rmse, la_rmse, rr_rmse, en_rmse]
rmse_df = pd.Series(models_rmse, index=labels).to_frame()
rmse_df.rename(columns={0: "Errors"}, inplace=True)
rmse_df



## === cell 50
test.head()



## === cell 51
test_1 = test_1.copy()
test_keys = test_1["key"].copy()
test_1.drop("key", axis=1, inplace=True)
test_1.head()



## === cell 52
test_1 = test_1.reindex(columns=X_1.columns)

final_prediction = rf.predict(test_1)
final_prediction = np.expm1(final_prediction)

pred_cap = float(np.quantile(y_1.to_numpy(), 0.999))
final_prediction = np.clip(final_prediction, 0.0, pred_cap)

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": final_prediction}
)

NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)

print("Wrote submission:", "NYCtaxiFare_prediction.csv")
print("Submission shape:", NYCtaxiFare_submission.shape)
NYCtaxiFare_submission.head()
