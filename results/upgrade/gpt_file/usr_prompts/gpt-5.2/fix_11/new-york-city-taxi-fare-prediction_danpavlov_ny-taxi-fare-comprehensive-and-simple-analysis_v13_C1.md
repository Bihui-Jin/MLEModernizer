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
geopy==2.4.1
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

3.60104

# 6. Current score

4.29145

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.30812) has done: 'I fix the notebook so it runs in a Kaggle script environment by removing the IPython-only `%matplotlib inline` magic and correcting the `../input` paths to the actual provided `/kaggle/input/...` files. I also fix the XGBoost inference error caused by deprecated `best_ntree_limit`/`ntree_limit` usage in xgboost 2.0.3 by using `iteration_range` based on `best_iteration` when early stopping is used. To keep score improvements legitimate and minimal, I keep the same core feature set and training flow, but I remove early stopping (which currently depends on deprecated attributes) so training proceeds deterministically for the specified number of boosting rounds. Finally, I ensure a valid submission CSV with the required `key,fare_amount` columns is written with a `.csv` suffix.'
- What this solution (achieved 4.75442) has done: 'Your score is far above the target (RMSE 7.308 vs 3.601, lower is better), and the biggest issue is that the current distance feature is computed with a slow Python loop and (more importantly) it uses great-circle distance in km without any scaling/consistency, which often underperforms on this competition. To move the score toward the target with minimal changes and identical training flow, I replace the looped `geopy.great_circle` distance with a vectorized Haversine distance (still a single “distance” feature, but computed correctly and consistently in kilometers). I also apply the exact same geographic filtering to the test set as training to avoid extreme out-of-distribution coordinates that can blow up RMSE, and I clip negative predictions to zero (fares can’t be negative), which usually reduces RMSE without changing the modeling approach. Everything else (same features used/dropped, same LinearRegression and XGBoost training setup, same output format/path) stays the same.'
- What this solution (achieved 4.5115) has done: 'Your current RMSE (4.75442) is worse than the target (3.60104), so we should make small, legitimate improvements that usually reduce error without changing the modeling approach. The biggest safe gain here is to add one standard NYC Taxi feature—Manhattan distance (L1 in lat/lon)—and include it alongside your existing Haversine “distance” while keeping the same XGBoost training loop and parameters. I also apply the same core geographic filtering logic to both train and test consistently (using the same bounds you already use) and ensure any datetime parse failures don’t create NaNs in the time features by filling with medians (avoids silent degradation). The submission format/path remains identical and we still clip negative fares to zero.'
- What this solution (achieved 4.56716) has done: 'Your current RMSE (4.5115) is still worse than the target (3.60104), so we should make small, legitimate improvements without changing the overall approach (same features family + same XGBoost training call). The biggest low-risk gain for this competition is to add two standard, lightweight features derived from your existing columns: (1) a proper “Manhattan distance in km” (scale lat/lon deltas into kilometers instead of raw degrees), and (2) simple directional deltas (`dlat`, `dlon`) that help tree models capture asymmetries. I also tighten XGBoost’s parameters slightly (depth/eta/subsampling/colsampling/min_child_weight) while keeping the same training loop and number of boosting rounds, and keep the same submission format/path. These changes typically reduce RMSE on NYC Taxi while remaining minimal and within Kaggle constraints.'
- What this solution (achieved 4.4474) has done: 'We need to reduce RMSE from 4.567 toward 3.601 (lower is better), so we make small, legitimate feature/cleaning tweaks that usually improve this exact competition without changing the overall training flow (same XGBoost training call, same loss/metric, same split). The biggest low-risk gain is to add two standard features derived from existing columns: a “bearing” (direction of trip) and a “euclidean_km” (Pythagorean distance in km using the same km scaling as your manhattan_km). We also fix the unscaled `manhattan` (in degrees) to a properly-scaled km version while keeping the original column name, so the model isn’t learning from a mis-scaled proxy. Finally, we apply the same basic fare outlier filtering commonly used for this dataset (keep reasonable fares) to reduce noise in the 500k sample, which typically improves RMSE without changing the modeling approach.'
- What this solution (achieved 4.40977) has done: 'We need to reduce RMSE from 4.4474 toward 3.60104 (lower is better), so we make small, legitimate improvements without changing the overall modeling approach (same LinearRegression + XGBoost reg:squarederror, same train/test flow). The biggest safe gain in this competition is to fix a feature bug: you currently create both `manhattan` (km) and `manhattan_km` (also km), but you only keep `manhattan_km` and you drop the km `manhattan`; we keep the km `manhattan` and remove the redundant `manhattan_km` to avoid duplicate/correlated columns that can hurt generalization. We also add two standard minimal datetime features (dayofweek alias is already there via weekday, but we add `day` and `dayofyear`) and ensure datetime-derived NaNs are filled consistently in both train and test (your test is filled, train isn’t), which reduces silent degradation. Finally, we keep the same XGBoost training call but add a small `gamma` regularization (often reduces overfitting on this dataset) while leaving rounds/eta/depth/etc. unchanged.'
- What this solution (achieved 4.30425) has done: 'To move RMSE down from 4.40977 toward the 3.60104 target (lower is better) without changing the overall modeling approach, I make two minimal, high-impact fixes: (1) apply the same cleaning rules to the test set as you already do for train (including the missing longitude > -80 bound and reasonable passenger_count), so the model doesn’t see out-of-distribution coordinates; and (2) add one standard NYC feature—absolute pickup distance to the city center—derived from existing lat/lon, which typically reduces error while keeping the same XGBoost training flow. I also ensure datetime-derived NaNs are filled consistently in both train and test to prevent silent NaN propagation into XGBoost. Everything else (same models, same training loops, same objective/metric, same submission format/path) remains unchanged.'
- What this solution (achieved 4.32105) has done: 'Your current RMSE (4.30425) is still worse than the target (3.60104), so we should make small, legitimate improvements that typically reduce error without changing the overall modeling approach (still XGBoost reg:squarederror on the same feature family). The biggest low-risk gain here is to fix an inconsistency: you filter/clean extreme coordinates in train but only “mask+median-fill” them in test; instead, we apply the exact same bounds-based clipping+median fill consistently to both train and test features to reduce distribution shift. We also add one standard, very lightweight feature from existing columns—`abs_dlat_km` and `abs_dlon_km` (scaled to km like your other km features)—which helps trees learn axis-aligned effects beyond signed deltas without changing the architecture/training loop. Finally, we remove prediction rounding (keep only non-negativity clipping) because rounding to cents usually worsens RMSE slightly and is not required by the competition.'
- What this solution (achieved 4.25026) has done: 'We need to reduce RMSE from 4.32105 toward 3.60104 (lower is better), so we make the smallest changes that typically improve generalization without altering your core pipeline (same feature family, same XGBoost objective, same training flow). The most impactful minimal fix is to make the train/validation split consistent with the competition distribution by splitting on unique taxi rides (`key`) after cleaning (prevents subtle leakage/duplication effects and gives a validation set that better matches public LB behavior). We also add two very standard, lightweight features derived from existing columns—pickup/dropoff proximity to JFK—and keep everything else intact; these features often reduce error in this competition without changing the model type or loop. Finally, we ensure the test NaN-imputation uses training medians computed on the exact training feature columns, avoiding any column-mismatch/median drift.'
- What this solution (achieved 4.29145) has done: 'To move RMSE down from 4.25026 toward the 3.60104 target (lower is better) with minimal disruption, I keep your exact model/feature pipeline but fix two data issues that commonly dominate this competition: (1) remove clearly invalid zero coordinates (0,0) and extreme lat/lon values that slip through your current bounds, and (2) add a standard “scaled passenger_count” (same information, just numeric stability) while leaving all existing features intact. I also make the train/test cleaning fully symmetric by applying the same coordinate validity rules before feature engineering on test (imputing where needed rather than dropping), which reduces train–test distribution shift without changing the training loop or objective. Finally, I keep the submission schema identical and still clip negative fares to 0.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/new-york-city-taxi-fare-prediction"

print("DATA_DIR:", DATA_DIR)
print("Files:", sorted(os.listdir(DATA_DIR))[:20])



## === cell 2
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), nrows=500000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
try:
    sns.histplot(train["fare_amount"], bins=100, kde=True)
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 9
try:
    sns.countplot(x=train["passenger_count"])
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
eps = 1e-6
train = train[
    ~((train["pickup_longitude"].abs() < eps) & (train["pickup_latitude"].abs() < eps))
]
train = train[
    ~(
        (train["dropoff_longitude"].abs() < eps)
        & (train["dropoff_latitude"].abs() < eps)
    )
]

train = train[train["fare_amount"] > 0]
train = train[train["fare_amount"] <= 250]

train = train[(train["pickup_longitude"] < -72) & (train["pickup_longitude"] > -80)]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[(train["dropoff_longitude"] < -72) & (train["dropoff_longitude"] > -80)]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    lat1 = (
        pd.to_numeric(df["pickup_latitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    lon1 = (
        pd.to_numeric(df["pickup_longitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    lat2 = (
        pd.to_numeric(df["dropoff_latitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    lon2 = (
        pd.to_numeric(df["dropoff_longitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )

    r = 6371.0  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["distance"] = (r * c).astype("float32")




## === cell 15
def manhattan_dist_calc(df):
    plat = pd.to_numeric(df["pickup_latitude"], errors="coerce").astype("float64")
    plon = pd.to_numeric(df["pickup_longitude"], errors="coerce").astype("float64")
    dlat = pd.to_numeric(df["dropoff_latitude"], errors="coerce").astype("float64")
    dlon = pd.to_numeric(df["dropoff_longitude"], errors="coerce").astype("float64")

    dlat_deg = (plat - dlat).to_numpy()
    dlon_deg = (plon - dlon).to_numpy()
    mean_lat_rad = np.radians(((plat + dlat) / 2.0).to_numpy())

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)

    manhattan_km = (np.abs(dlat_deg) * km_per_deg_lat) + (
        np.abs(dlon_deg) * km_per_deg_lon
    )
    df["manhattan"] = manhattan_km.astype("float32")




## === cell 16
def add_delta_and_manhattan_km(df):
    plat = pd.to_numeric(df["pickup_latitude"], errors="coerce").astype("float64")
    plon = pd.to_numeric(df["pickup_longitude"], errors="coerce").astype("float64")
    dlat = pd.to_numeric(df["dropoff_latitude"], errors="coerce").astype("float64")
    dlon = pd.to_numeric(df["dropoff_longitude"], errors="coerce").astype("float64")

    dlat_deg = (dlat - plat).to_numpy()
    dlon_deg = (dlon - plon).to_numpy()

    df["dlat"] = dlat_deg.astype("float32")
    df["dlon"] = dlon_deg.astype("float32")

    mean_lat_rad = np.radians(((plat + dlat) / 2.0).to_numpy())
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)

    manhattan_km = (np.abs(dlat_deg) * km_per_deg_lat) + (
        np.abs(dlon_deg) * km_per_deg_lon
    )
    df["manhattan_km"] = manhattan_km.astype("float32")




## === cell 17
def add_bearing_and_euclidean_km(df):
    plat = pd.to_numeric(df["pickup_latitude"], errors="coerce").astype("float64")
    plon = pd.to_numeric(df["pickup_longitude"], errors="coerce").astype("float64")
    dlat = pd.to_numeric(df["dropoff_latitude"], errors="coerce").astype("float64")
    dlon = pd.to_numeric(df["dropoff_longitude"], errors="coerce").astype("float64")

    lat1 = np.radians(plat.to_numpy())
    lat2 = np.radians(dlat.to_numpy())
    dlonr = np.radians((dlon - plon).to_numpy())

    y = np.sin(dlonr) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlonr)
    bearing = np.arctan2(y, x)  # radians in [-pi, pi]
    df["bearing"] = bearing.astype("float32")

    dlat_deg = (dlat - plat).to_numpy()
    dlon_deg = (dlon - plon).to_numpy()
    mean_lat_rad = np.radians(((plat + dlat) / 2.0).to_numpy())
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)

    dy = dlat_deg * km_per_deg_lat
    dx = dlon_deg * km_per_deg_lon
    df["euclidean_km"] = np.sqrt(dx * dx + dy * dy).astype("float32")




## === cell 18
def add_center_dists(df, center_lat=40.7141667, center_lon=-74.0063889):
    plat = (
        pd.to_numeric(df["pickup_latitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    plon = (
        pd.to_numeric(df["pickup_longitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    dlat = (
        pd.to_numeric(df["dropoff_latitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    dlon = (
        pd.to_numeric(df["dropoff_longitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )

    km_per_deg_lat = 111.32
    mean_lat_p = np.radians((plat + center_lat) / 2.0)
    mean_lat_d = np.radians((dlat + center_lat) / 2.0)
    km_per_deg_lon_p = 111.32 * np.cos(mean_lat_p)
    km_per_deg_lon_d = 111.32 * np.cos(mean_lat_d)

    df["pickup_to_center_km"] = np.sqrt(
        ((plat - center_lat) * km_per_deg_lat) ** 2
        + ((plon - center_lon) * km_per_deg_lon_p) ** 2
    ).astype("float32")
    df["dropoff_to_center_km"] = np.sqrt(
        ((dlat - center_lat) * km_per_deg_lat) ** 2
        + ((dlon - center_lon) * km_per_deg_lon_d) ** 2
    ).astype("float32")




## === cell 19
def add_abs_delta_km(df):
    plat = pd.to_numeric(df["pickup_latitude"], errors="coerce").astype("float64")
    plon = pd.to_numeric(df["pickup_longitude"], errors="coerce").astype("float64")
    dlat = pd.to_numeric(df["dropoff_latitude"], errors="coerce").astype("float64")
    dlon = pd.to_numeric(df["dropoff_longitude"], errors="coerce").astype("float64")

    dlat_deg = (dlat - plat).to_numpy()
    dlon_deg = (dlon - plon).to_numpy()
    mean_lat_rad = np.radians(((plat + dlat) / 2.0).to_numpy())

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)

    df["abs_dlat_km"] = (np.abs(dlat_deg) * km_per_deg_lat).astype("float32")
    df["abs_dlon_km"] = (np.abs(dlon_deg) * km_per_deg_lon).astype("float32")




## === cell 20
def add_airport_dists(df, jfk_lat=40.6413, jfk_lon=-73.7781):
    plat = (
        pd.to_numeric(df["pickup_latitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    plon = (
        pd.to_numeric(df["pickup_longitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    dlat = (
        pd.to_numeric(df["dropoff_latitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )
    dlon = (
        pd.to_numeric(df["dropoff_longitude"], errors="coerce")
        .astype("float64")
        .to_numpy()
    )

    km_per_deg_lat = 111.32
    mean_lat_p = np.radians((plat + jfk_lat) / 2.0)
    mean_lat_d = np.radians((dlat + jfk_lat) / 2.0)
    km_per_deg_lon_p = 111.32 * np.cos(mean_lat_p)
    km_per_deg_lon_d = 111.32 * np.cos(mean_lat_d)

    df["pickup_to_jfk_km"] = np.sqrt(
        ((plat - jfk_lat) * km_per_deg_lat) ** 2
        + ((plon - jfk_lon) * km_per_deg_lon_p) ** 2
    ).astype("float32")
    df["dropoff_to_jfk_km"] = np.sqrt(
        ((dlat - jfk_lat) * km_per_deg_lat) ** 2
        + ((dlon - jfk_lon) * km_per_deg_lon_d) ** 2
    ).astype("float32")




## === cell 21
eps = 1e-6
bad_pickup = (test["pickup_longitude"].abs() < eps) & (
    test["pickup_latitude"].abs() < eps
)
bad_dropoff = (test["dropoff_longitude"].abs() < eps) & (
    test["dropoff_latitude"].abs() < eps
)
for col in ["pickup_longitude", "pickup_latitude"]:
    test.loc[bad_pickup, col] = np.nan
for col in ["dropoff_longitude", "dropoff_latitude"]:
    test.loc[bad_dropoff, col] = np.nan

dist_calc(train)
dist_calc(test)
manhattan_dist_calc(train)
manhattan_dist_calc(test)
add_delta_and_manhattan_km(train)
add_delta_and_manhattan_km(test)
add_bearing_and_euclidean_km(train)
add_bearing_and_euclidean_km(test)
add_center_dists(train)
add_center_dists(test)
add_abs_delta_km(train)
add_abs_delta_km(test)
add_airport_dists(train)
add_airport_dists(test)

if "manhattan_km" in train.columns:
    train.drop(columns=["manhattan_km"], inplace=True)
if "manhattan_km" in test.columns:
    test.drop(columns=["manhattan_km"], inplace=True)



## === cell 22
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 23
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 24
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year
train["day"] = train.pickup_datetime.dt.day
train["dayofyear"] = train.pickup_datetime.dt.dayofyear

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year
test["day"] = test.pickup_datetime.dt.day
test["dayofyear"] = test.pickup_datetime.dt.dayofyear

for c in ["hour", "weekday", "month", "year", "day", "dayofyear"]:
    if train[c].isna().any():
        train[c] = train[c].fillna(train[c].median())
    if test[c].isna().any():
        test[c] = test[c].fillna(train[c].median())



## === cell 25
train["passenger_count_scaled"] = (
    train["passenger_count"].astype("float32") / 6.0
).astype("float32")
test["passenger_count_scaled"] = (
    test["passenger_count"].astype("float32") / 6.0
).astype("float32")



## === cell 26
test.head()



## === cell 27
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=False)
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 28
X = train.drop(
    ["key", "fare_amount", "pickup_datetime", "pickup_longitude", "dropoff_longitude"],
    axis=1,
)
y = train["fare_amount"]



## === cell 29
X.head()



## === cell 30
y.head()



## === cell 31
train_keys, valid_keys = train_test_split(
    train["key"].unique(), test_size=0.2, random_state=42
)
is_valid = train["key"].isin(valid_keys)

X_train = X.loc[~is_valid].copy()
y_train = y.loc[~is_valid].copy()
X_test = X.loc[is_valid].copy()
y_test = y.loc[is_valid].copy()

print("Train size:", X_train.shape, "Valid size:", X_test.shape)



## === cell 32
test_pred = test.drop(
    ["key", "pickup_datetime", "pickup_longitude", "dropoff_longitude"], axis=1
)



## === cell 33
test_mask = (
    (test["pickup_longitude"] < -72)
    & (test["pickup_longitude"] > -80)
    & (test["pickup_latitude"] > 40)
    & (test["pickup_latitude"] < 44)
    & (test["dropoff_longitude"] < -72)
    & (test["dropoff_longitude"] > -80)
    & (test["dropoff_latitude"] > 40)
    & (test["dropoff_latitude"] < 44)
    & (test["passenger_count"] > 0)
    & (test["passenger_count"] < 10)
)

test_pred = test_pred.copy()
test_pred.loc[~test_mask.values, :] = np.nan

medians = X_train.median(numeric_only=True)
test_pred = test_pred.fillna(medians)

test_pred = test_pred[X_train.columns]



## === cell 34
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 35
y_pred = lm.predict(X_test)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y_test))
print("Linear RMSE on validation:", lrmse)



## === cell 36
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.clip(LinearPredictions, 0, None)
LinearPredictions



## === cell 37
LinearPredictions.size



## === cell 38
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 39
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.08,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "gamma": 0.1,
        "seed": 42,
        "tree_method": "hist",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    return booster




## === cell 40
xgbm = XGBoost(X_train, X_test, y_train, y_test)

XGBPredictions = xgbm.predict(xgb.DMatrix(test_pred))



## === cell 41
XGBPredictions



## === cell 42
XGBPredictions = np.clip(XGBPredictions, 0, None)
XGBPredictions



## === cell 43
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 44
submission = XGB_submission



## === cell 45
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
