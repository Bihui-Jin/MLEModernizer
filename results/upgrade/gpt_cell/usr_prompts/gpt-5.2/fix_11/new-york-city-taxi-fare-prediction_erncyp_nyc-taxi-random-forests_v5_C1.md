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

4.24009

# 6. Current score

6.5994

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.64917) has done: 'Diagnosis: Cell 29 fails because `X` and `Y` no longer have the same length at that point. Earlier, `X` was reassigned in cell 17 to include multiple feature columns (`distance,year,month,day,hour`), while `Y` remained the original 1D target array from `train_df` before filtering/reindexing; seaborn tries to build a DataFrame from `x` and `y` and raises `ValueError: All arrays must be of the same length`. The plotting in cell 29 is intended to visualize fare vs distance, so it should use the aligned `train_df['distance']` and `train_df['fare_amount']` directly (or the original single-feature `X` before it was overwritten).  

Patch summary: In cell 29, compute local plotting vectors from `train_df` (`distance` and `fare_amount`) and pass those to `sns.jointplot`, avoiding dependence on the later overwritten `X` variable. This keeps the rest of the notebook variables untouched and makes the plot deterministic and length-consistent.  

Updated cells: Only cell 29 is modified.  

Compatibility notes for cell k+1: Cell 30 expects `X` and `Y` to exist and then overwrites `X` with `X.flatten()`; this patch does not change `X`/`Y`, so cell 30 remains compatible and unchanged.  

Assumptions: `train_df` still exists in memory in cell 29 and contains the engineered `distance` column and the `fare_amount` column after filtering steps.'
- What this solution (achieved 6.80843) has done: 'The timeout is dominated by fitting a default `RandomForestRegressor` on ~1M rows (and even after filtering, still very large), plus unnecessary extra full-dataset predictions for printing metrics and two heavy seaborn plots. To keep the same modeling core (RandomForest + same features + same training flow) while making it finish under 600s, the key is to (a) use Intel-optimized scikit-learn (`sklearnex`) already installed, (b) parallelize the forest (`n_jobs=-1`) and make it deterministic, and (c) avoid expensive training-set predictions and plotting that aren’t required to generate `submission.csv`. Additionally, we fix the broadcasting error by keeping separate variables for the 1-feature vs 5-feature matrices so the later mask uses matching shapes.'
- What this solution (achieved 6.20355) has done: 'Your current score (6.80843 RMSE) is worse than the target (4.24009), so we should make small, legitimate improvements that keep the same overall approach (RandomForest on engineered time + distance features) while reducing error. The biggest minimal win is to remove a clear feature-mismatch: you engineer `minute` but you never use it in `X_more`/`X_to_pred`, so the model is missing time resolution that often matters for NYC taxi fares. Second, include `passenger_count` (already loaded) in the feature set, since it is a real driver of fare and this is still the same feature-engineering paradigm. Finally, add a small, standard data cleaning filter to remove clearly invalid fares and impossible passenger counts, which reduces noise without changing the modeling core.'
- What this solution (achieved 5.99047) has done: 'You’re currently worse than the target (6.20355 vs 4.24009, lower is better), so we should make a small, safe improvement without changing the overall approach (RandomForest on engineered distance + time + passenger_count). The most impactful minimal change here is to tune just a few `RandomForestRegressor` hyperparameters to reduce variance and improve generalization (more trees, min leaf size, and max features), while keeping the same model family and training flow. I’m also keeping runtime under 600s by not increasing training rows and by using `n_jobs=-1` plus `sklearnex` when available. The submission writing remains identical, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 5.76229) has done: 'Your current RMSE (5.99047) is worse than the target (4.24009), so we should make a small, legitimate improvement without changing the overall approach (same features + RandomForestRegressor). The biggest minimal gain for this competition is to remove obvious label noise/outliers that the model can’t learn well: filter to NYC bounding box (instead of “within 5 degrees of mean”) and remove unrealistic trip distances. These are standard cleaning steps for this dataset and typically reduce RMSE substantially while keeping the same core logic and training flow. I’m also keeping runtime stable by not increasing `nrows` and leaving the model family/loop unchanged.'
- What this solution (achieved 5.96129) has done: 'I make two minimal, model-preserving changes aimed at reducing RMSE toward your target: (1) apply the same kind of “obvious invalid record” cleaning to the training data that is standard for this competition by removing extreme-coordinate zeros and clipping implausible per-km fares, which reduces label noise without changing features or model type; and (2) lightly constrain the RandomForest to generalize better by adding `max_depth`, keeping the same RandomForestRegressor approach and feature set. These are small, legitimate adjustments that typically move RMSE down from the mid-5s without changing the overall pipeline, and they keep runtime within the 600s limit. The submission format and file path remain unchanged and a valid `submission.csv` still be written.'
- What this solution (achieved 5.96129) has done: 'Your current RMSE (5.96129) is worse than the target (4.24009), so we should make a small improvement that keeps the same overall pipeline (same engineered features + RandomForestRegressor). The most direct, minimal win for this competition is to clip prediction outputs to a realistic fare range, because negative/very large predictions can disproportionately hurt RMSE on outlier cases even if most predictions are good. This does not change the model, training loop, features, or loss; it only applies a standard post-processing step aligned with the fare domain. I also ensure any remaining NaNs/inf in the engineered `distance` are dropped before fitting to avoid subtle training noise.'
- What this solution (achieved 5.94837) has done: 'Your current RMSE (5.96129) is worse than the target (4.24009), so we should make a small, legitimate improvement without changing the overall model family or feature set. The biggest minimal gain for this competition is to stop training on noisy/outlier geographic points by using a slightly tighter NYC bounding box plus a realistic distance cap (many bad rows remain even after your current filters). I also add the standard “Manhattan distance” (L1 in lat/lon) as an extra feature alongside haversine, which keeps the same feature-engineering style and the same RandomForest training loop but typically reduces error. Finally, I apply the exact same feature engineering to test and keep submission writing unchanged.'
- What this solution (achieved 6.5994) has done: 'To move RMSE down toward your target while keeping the same core pipeline (same features + RandomForestRegressor), I make one data-quality tweak that typically yields a clear gain on this competition: remove rows where pickup and dropoff coordinates are identical/near-identical (these create zero/near-zero trips with nontrivial fares and add noise). I also add two very standard geospatial features (absolute lat/lon deltas) that are consistent with your existing distance/Manhattan-style engineering and don’t change the training approach. Finally, I ensure the same feature engineering is applied to test and keep the same submission writing semantics.'
- What this solution (achieved 6.5994) has done: 'Your current RMSE (6.5994) is still far above the target (4.24009), so we should make small, legitimate improvements without changing the core model family or feature set. The biggest minimal gain on this competition is almost always better training-data cleaning: remove trips with impossible “speed” (distance vs time) and remove outliers with implausible fare-per-degree (since we already use degree deltas as features). These filters reduce label noise that a RandomForest cannot generalize from, typically lowering test RMSE while keeping the same architecture/training flow. I implement the filters right after existing cleaning, recompute aligned `Y` after filtering to avoid any index mismatch risk, and keep the same submission writing.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(42)



## === cell 1
TRAIN_PATH = "../input/train.csv"
USECOLS_TRAIN = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES_TRAIN = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}
train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=1_000_000,
    usecols=USECOLS_TRAIN,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"].to_numpy(),
    train_df["pickup_latitude"].to_numpy(),
    train_df["dropoff_longitude"].to_numpy(),
    train_df["dropoff_latitude"].to_numpy(),
)

train_df["abs_lon_delta"] = (
    train_df["pickup_longitude"] - train_df["dropoff_longitude"]
).abs()
train_df["abs_lat_delta"] = (
    train_df["pickup_latitude"] - train_df["dropoff_latitude"]
).abs()

train_df["manhattan_deg"] = train_df["abs_lon_delta"] + train_df["abs_lat_delta"]



## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])



## === cell 5
dt = train_df["pickup_datetime"].dt
train_df["year"] = dt.year
train_df["month"] = dt.month
train_df["day"] = dt.day
train_df["hour"] = dt.hour
train_df["minute"] = dt.minute



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
new_york_lat = 40
new_york_long = -74
train_df.describe()



## === cell 8
nyc_bounds = {
    "lon_min": -74.25,
    "lon_max": -73.70,
    "lat_min": 40.55,
    "lat_max": 40.90,
}
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df["pickup_longitude"].between(nyc_bounds["lon_min"], nyc_bounds["lon_max"]))
    & (
        train_df["dropoff_longitude"].between(
            nyc_bounds["lon_min"], nyc_bounds["lon_max"]
        )
    )
    & (
        train_df["pickup_latitude"].between(
            nyc_bounds["lat_min"], nyc_bounds["lat_max"]
        )
    )
    & (
        train_df["dropoff_latitude"].between(
            nyc_bounds["lat_min"], nyc_bounds["lat_max"]
        )
    )
]
print("New size: %d" % len(train_df))



## === cell 9
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 250)]
train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]
train_df = train_df[(train_df["distance"] > 0) & (train_df["distance"] <= 60)]

coord_eps = 1e-6
train_df = train_df[
    (train_df["pickup_longitude"].abs() > coord_eps)
    & (train_df["pickup_latitude"].abs() > coord_eps)
    & (train_df["dropoff_longitude"].abs() > coord_eps)
    & (train_df["dropoff_latitude"].abs() > coord_eps)
]

same_coord_eps = 1e-5
train_df = train_df[
    (train_df["abs_lon_delta"] > same_coord_eps)
    | (train_df["abs_lat_delta"] > same_coord_eps)
]

fare_per_km = train_df["fare_amount"] / train_df["distance"]
train_df = train_df[(fare_per_km >= 0.5) & (fare_per_km <= 50.0)]

manhattan_deg = train_df["manhattan_deg"].to_numpy()
fare = train_df["fare_amount"].to_numpy()

deg_eps = 1e-6
fare_per_deg = fare / np.maximum(manhattan_deg, deg_eps)
train_df = train_df[(fare_per_deg >= 20.0) & (fare_per_deg <= 200000.0)]

train_df = train_df.reset_index(drop=True)



## === cell 10
train_df.describe()



## === cell 11
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



## === cell 12
regr = LinearRegression()
regr_quad = LinearRegression()



## === cell 13
X_dist = train_df[["distance"]].values
Y = train_df["fare_amount"].values



## === cell 14
regr.fit(X_dist, Y)



## === cell 15
regr_quad.fit(X_dist**2, Y)



## === cell 16
pass



## === cell 17
regr_more = LinearRegression()
X_more = train_df[
    [
        "distance",
        "manhattan_deg",
        "abs_lon_delta",
        "abs_lat_delta",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "passenger_count",
    ]
].values



## === cell 18
regr_more.fit(X_more, Y)

pass



## === cell 19
from sklearn.ensemble import RandomForestRegressor



## === cell 20
rand_regr = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    max_features="sqrt",
    max_depth=25,
    n_jobs=-1,
    random_state=42,
)



## === cell 21
rand_regr.fit(X_more, Y)

pass



## === cell 22
TEST_PATH = "../input/test.csv"
USECOLS_TEST = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES_TEST = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}
test_df = pd.read_csv(
    TEST_PATH,
    usecols=USECOLS_TEST,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
)



## === cell 23
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"].to_numpy(),
    test_df["pickup_latitude"].to_numpy(),
    test_df["dropoff_longitude"].to_numpy(),
    test_df["dropoff_latitude"].to_numpy(),
)

test_df["abs_lon_delta"] = (
    test_df["pickup_longitude"] - test_df["dropoff_longitude"]
).abs()
test_df["abs_lat_delta"] = (
    test_df["pickup_latitude"] - test_df["dropoff_latitude"]
).abs()
test_df["manhattan_deg"] = test_df["abs_lon_delta"] + test_df["abs_lat_delta"]



## === cell 24
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])



## === cell 25
dt_test = test_df["pickup_datetime"].dt
test_df["year"] = dt_test.year
test_df["month"] = dt_test.month
test_df["day"] = dt_test.day
test_df["hour"] = dt_test.hour
test_df["minute"] = dt_test.minute



## === cell 26
X_to_pred = test_df[
    [
        "distance",
        "manhattan_deg",
        "abs_lon_delta",
        "abs_lat_delta",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "passenger_count",
    ]
].values
y_pred = rand_regr.predict(X_to_pred)



## === cell 27
y_pred = np.clip(y_pred, 0.0, 250.0)



## === cell 28
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
