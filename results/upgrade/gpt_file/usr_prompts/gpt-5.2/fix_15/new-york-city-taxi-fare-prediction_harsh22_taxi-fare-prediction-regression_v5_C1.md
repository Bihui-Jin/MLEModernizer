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

3.47816

# 6. Current score

4.89062

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'I fix the pandas boolean-logic bugs caused by newer pandas APIs (e.g., `.any(1)` and using `|` between DataFrames instead of boolean masks), so the data-cleaning steps run without errors. Then I ensure the model training doesn’t crash due to remaining NaNs by applying a minimal numeric imputation (median) consistently to both train and test, which is score-positive and does not change the core RandomForest approach. Finally, I make submission generation robust by carrying `key` through preprocessing and writing a valid `submission.csv` with exactly `key,fare_amount`. All changes are targeted to unblock execution and produce a valid submission file end-to-end.'
- What this solution (achieved 16.77318) has done: 'Your score is far worse than the target (lower RMSE is better), so the smallest score-positive change is to fix a key bug: converting `key` to datetime destroys the required identifier and also risks misalignment, which can heavily hurt Kaggle scoring even if predictions are reasonable. I keep `key` as the original string for submission, only parse `pickup_datetime` for feature engineering, and drop any rows where `pickup_datetime` became NaT to avoid NaN-derived feature corruption. I also add the missing drop for out-of-range `dropoff_longitude` (you currently only display it), which removes invalid geography that can mis-train the model. These are minimal, semantics-preserving fixes that typically move RMSE sharply down toward the target without changing the RandomForest approach.'
- What this solution (achieved 10.01493) has done: 'Your RMSE is far above the target, so we should make small, score-positive fixes that don’t change the core RandomForest approach. The biggest likely issue is that the model is being trained on many obviously wrong NYC taxi points (e.g., coordinates at 0/0 or far outside NYC but still within [-180,180]), which badly hurts generalization; we add a minimal NYC bounding-box filter and remove rows with zero coordinates. We also cap extreme fares in training (a common minimal cleaning step for this competition) to reduce the impact of outliers on RMSE, without changing the model or loss. Finally, we keep the `key` intact and ensure test rows are not dropped so the submission always has the correct row count and alignment.'
- What this solution (achieved 6.56056) has done: 'Your RMSE (10.01) is still far above the target (3.48), so we should make minimal, score-positive fixes without changing the RandomForest approach. The biggest remaining issue is that the model is likely being trained on noisy/outlier fares and implausible trips that slip past current filters; tightening fare bounds and adding a simple “distance > 0” consistency filter typically reduces RMSE substantially for this competition. I also ensure passenger_count is restricted to a realistic range (1–6) and that we clip negative predictions to 0 (fares can’t be negative), which improves RMSE without altering the core model. All changes keep the same feature set (lat/lon, passenger_count, H_Distance, and time parts) and the same training loop, just cleaner training data and safer post-processing.'
- What this solution (achieved 5.38838) has done: 'The timeout is dominated by fitting a 300-tree `RandomForestRegressor` on ~1M rows (very expensive) plus repeated pandas `drop(...index...)` passes that copy the full DataFrame many times. I keep the same model/feature logic, but make data cleaning a single boolean-mask filter (same conditions, one pass) and avoid expensive debug-only operations (like sorting the full `fare_amount` column). I also read only the required columns with explicit dtypes to reduce CSV parse time and memory pressure, and enable Intel-accelerated sklearn (same semantics) when available. These changes preserve the algorithm and predictions (up to negligible float differences) while cutting constant factors enough to fit within the 600s budget.'
- What this solution (achieved 5.56084) has done: 'Your current RMSE (5.388) is still above the target (3.478), so we should make small, score-positive data-quality fixes without changing the RandomForest model or feature set. The biggest likely remaining issue is mislabeled time features due to `pickup_datetime` not being read/parsing reliably (it’s currently not given a dtype and can become object with more coercion), and weakly filtered geographic outliers that are still within the broad NYC box but implausible for taxi trips. I (1) read `pickup_datetime` explicitly as string to stabilize parsing, (2) add two minimal, competition-standard filters: cap extreme Haversine distances and enforce a loose “fare per km” sanity range (removes pathological points that inflate RMSE), and (3) ensure the same imputer/model/submission semantics remain unchanged.'
- What this solution (achieved 5.55711) has done: 'We need to move RMSE down from 5.56084 toward 3.47816 (lower is better), so we make small, score-positive data-quality fixes while keeping the same RandomForest model, features, and training loop. The biggest low-risk gain for this competition is to (1) add a couple standard NYC taxi cleaning filters (remove extreme long trips and very large coordinate deltas) that are likely still contaminating training, and (2) add a minimal, metric-aligned post-processing that floors predictions at the known minimum fare (2.5) instead of 0. These changes don’t alter your architecture or feature engineering; they only improve the training signal and calibrate outputs to the known domain. Submission generation remains identical (`key,fare_amount`) and row-aligned to `test.csv`.'
- What this solution (achieved 5.00195) has done: 'We need to move RMSE down from 5.55711 toward 3.47816 (lower is better), so the smallest score-positive change is to align the training and inference distributions by applying the same lightweight “bad row” masking to the test set features (without dropping any test rows), because currently the model is trained on heavily filtered NYC-like trips but predicts on unfiltered test inputs that can include edge-case coordinates/distances. I add a minimal `clean_test_features_inplace()` that sets obviously invalid/implausible coordinates and derived values to NaN (so the existing median imputer can handle them consistently) and caps Haversine distance similarly to training filters. I also add two common, low-risk time features (minute and weekday_is_weekend) derived from `pickup_datetime`—this does not change the core model (still RandomForest) or training loop, but typically reduces RMSE for this competition with negligible complexity. Submission generation remains identical (`key,fare_amount`) and row-aligned to `test.csv`.'
- What this solution (achieved 4.95193) has done: 'Your current RMSE (5.00195) is still meaningfully worse than the target (3.47816), so we should make a small, score-positive change that preserves the same RandomForest core logic. The biggest likely remaining issue is that the model is trained on 1M rows that are still quite noisy; for this competition a minimal upgrade is to use the same 1M-row budget but sample it uniformly across the full file (instead of taking the first 1M, which are temporally/regionally biased). I keep the exact same feature engineering, cleaning rules, imputer, and RandomForest training loop, but change the training read to a deterministic random sample via `skiprows` so the training distribution better matches the test set. This typically reduces RMSE materially while staying within time/memory constraints and without changing evaluation semantics or the model architecture.'
- What this solution (achieved 5.00518) has done: 'We need to move RMSE down from 4.95193 toward 3.47816 (lower is better), so we should improve generalization with the smallest change that keeps the same RandomForest model and feature set. The biggest remaining low-risk issue is that the current “uniform sample” loader uses `skiprows` with a Python `set` membership check per line, which is slow and can inadvertently bias/under-sample depending on runtime constraints; we replace it with a deterministic, truly uniform random sample via chunked reservoir-style sampling (same 1,000,000 rows) while keeping all downstream cleaning/feature engineering identical. This keeps the core logic (same features, same cleaning rules, same imputer, same RF hyperparameters) but gives a better-representative training distribution, which should reduce RMSE toward the target. We also make parsing consistent by explicitly disabling `infer_datetime_format` (deprecated/ignored) and ensure the submission row order exactly matches `test.csv` by preserving `test_key` before any in-place edits.'
- What this solution (achieved 5.00504) has done: 'Your current RMSE (5.00518) is worse than the target (3.47816), so we should make a small, score-positive change without changing the RandomForest core. The biggest issue is your “uniform sampling” is not actually uniform across the full 55M rows: it concatenates the first few chunks and then randomly takes the remainder from a single later chunk, which leaves a strong temporal/row-order bias and hurts generalization. I replace that loader with a deterministic, truly-uniform chunked sampling that draws a proportional random subset from every chunk (same 1,000,000-row budget, same downstream cleaning/features/model). Everything else (cleaning rules, features, imputer, RF hyperparameters, submission format) stays the same.'
- What this solution (achieved 4.89062) has done: 'We need to move RMSE down from 5.00504 toward 3.47816 (lower is better), so we make the smallest score-positive changes that keep your RandomForest + current feature set intact. The biggest likely issue is that your “uniform sampler” uses a fixed global probability and then early-stops, which under-samples early chunks and biases the 1M training rows; we replace it with a deterministic per-chunk proportional sampler that always draws ~n_rows spread across the entire file without changing downstream logic. Second, we add one very standard, low-risk cleaning constraint that matches this competition: remove passengers_count==0 (you already restrict to 1–6, but this ensures it happens before NaN filtering) and remove unrealistically large “fare per distance” only when distance is non-trivial (avoids penalizing short trips). Everything else—features, imputer, RF hyperparameters, prediction clipping, and submission format—stays the same.'
- What this solution (achieved 4.89062) has done: 'We need to move RMSE down from 4.89062 toward 3.47816 (lower is better), so I make two minimal, score-positive changes that keep your RandomForest core, features, and training loop intact. First, I fix a subtle but important sampling bias: your sampler breaks early after ~1.2M kept, which still over-represents earlier chunks; instead we deterministically take a proportional sample from every chunk and only downsample at the end, keeping the same 1,000,000-row budget. Second, I add one very standard, low-risk data-cleaning rule for this competition: remove rows where pickup and dropoff coordinates are identical (zero-distance “trips”), which are usually label noise/outliers and inflate RMSE. Everything else (feature engineering, imputer, RF hyperparameters, prediction clipping, and submission format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import sklearn  # noqa: F401
import seaborn as sns  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

if os.path.exists("../input"):
    print(os.listdir("../input"))
else:
    print("../input not found; listing /kaggle/input instead if available.")
    if os.path.exists("/kaggle/input"):
        print(os.listdir("/kaggle/input"))

np.random.seed(42)



## === cell 1
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_train = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_datetime": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
dtype_test = {
    "key": "string",
    "pickup_datetime": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train_path = "../input/train.csv"
test_path = "../input/test.csv"
n_train_sample = 1_000_000


def read_uniform_sample_csv(
    path,
    n_rows,
    usecols,
    dtype,
    seed=42,
    chunksize=250_000,
    approx_total_rows=55_423_856,  # only affects per-chunk take size
):
    """
    Change (score-positive, minimal): remove early-break that biases toward early chunks.
    We still take ~proportional samples from every chunk, then downsample once at the end
    to exactly n_rows. This keeps the same downstream logic/model but improves train/test match.
    """
    rng = np.random.RandomState(seed)
    sampled = []

    frac = n_rows / float(approx_total_rows)

    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize):
        m = len(chunk)
        if m == 0:
            continue

        take_n = int(round(m * frac))
        take_n = max(0, min(m, take_n))

        if take_n > 0:
            idx = rng.choice(m, size=take_n, replace=False)
            sampled.append(chunk.iloc[idx])

    if not sampled:
        return pd.read_csv(path, usecols=usecols, dtype=dtype)

    out = pd.concat(sampled, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(out) >= n_rows:
        out = out.iloc[:n_rows].reset_index(drop=True)
    else:
        remaining = n_rows - len(out)
        for chunk in pd.read_csv(
            path, usecols=usecols, dtype=dtype, chunksize=chunksize
        ):
            if remaining <= 0:
                break
            m = len(chunk)
            take_n = min(m, remaining)
            idx = rng.choice(m, size=take_n, replace=False)
            out = pd.concat([out, chunk.iloc[idx]], axis=0, ignore_index=True)
            remaining -= take_n
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out = out.iloc[:n_rows].reset_index(drop=True)

    return out


train = read_uniform_sample_csv(
    train_path,
    n_rows=n_train_sample,
    usecols=usecols_train,
    dtype=dtype_train,
    seed=42,
    chunksize=250_000,
)

test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(10)



## === cell 5
train.describe()



## === cell 6
train.isnull().sum().sort_values(ascending=False)



## === cell 7
pass



## === cell 8
train.shape



## === cell 9
train["fare_amount"].describe()



## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 11
pass



## === cell 12
train.shape



## === cell 13
train["fare_amount"].describe()



## === cell 14
pass



## === cell 15
train["passenger_count"].describe()



## === cell 16
pass



## === cell 17
pass



## === cell 18
train["passenger_count"].describe()



## === cell 19
train["pickup_latitude"].describe()



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
train.shape



## === cell 24
train["pickup_longitude"].describe()



## === cell 25
pass



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
train.dtypes



## === cell 33
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")

train = train.dropna(subset=["pickup_datetime"]).reset_index(drop=True)



## === cell 34
train.dtypes




## === cell 35
def add_haversine(df, lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1v = df[lat1].to_numpy(dtype=np.float64, copy=False)
    lon1v = df[lon1].to_numpy(dtype=np.float64, copy=False)
    lat2v = df[lat2].to_numpy(dtype=np.float64, copy=False)
    lon2v = df[lon2].to_numpy(dtype=np.float64, copy=False)

    phi1 = np.radians(lat1v)
    phi2 = np.radians(lat2v)
    dphi = np.radians(lat2v - lat1v)
    dlambda = np.radians(lon2v - lon1v)

    a = (np.sin(dphi / 2.0) ** 2) + (
        np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2)
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["H_Distance"] = (r * c).astype(np.float32)
    return df


train = add_haversine(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
test = add_haversine(
    test, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 36
train["H_Distance"].head(10)



## === cell 37
for df in (train, test):
    dt = df["pickup_datetime"]
    df["Year"] = dt.dt.year.astype(np.int16)
    df["Month"] = dt.dt.month.astype(np.int8)
    df["Date"] = dt.dt.day.astype(np.int8)
    df["Day of Week"] = dt.dt.dayofweek.astype(np.int8)
    df["Hour"] = dt.dt.hour.astype(np.int8)
    df["Minute"] = dt.dt.minute.astype(np.int8)
    df["IsWeekend"] = (df["Day of Week"] >= 5).astype(np.int8)



## === cell 38
pass



## === cell 39
pass



## === cell 40
train.shape



## === cell 41
pass



## === cell 42
pass



## === cell 43
pass



## === cell 44
pass



## === cell 45
pass



## === cell 46
pass



## === cell 47
pass



## === cell 48
pass



## === cell 49
pass



## === cell 50
pass



## === cell 51
pass



## === cell 52
pass



## === cell 53
pass



## === cell 54
pass



## === cell 55
pass



## === cell 56
pass



## === cell 57
pass



## === cell 58
pass



## === cell 59
pass



## === cell 60
pass



## === cell 61
pass



## === cell 62
pass



## === cell 63
pass



## === cell 64
pass



## === cell 65
pass



## === cell 66
pass



## === cell 67
pass



## === cell 68
pass



## === cell 69
pass



## === cell 70
pass



## === cell 71
pass



## === cell 72
train.columns



## === cell 73
test.columns



## === cell 74
nyc_bounds = {"min_lon": -74.3, "max_lon": -73.6, "min_lat": 40.5, "max_lat": 41.0}

m = np.ones(len(train), dtype=bool)

m &= ~train.isnull().any(axis=1).to_numpy()

m &= train["fare_amount"].to_numpy() >= 0

m &= train["passenger_count"].to_numpy() > 0

m &= train["passenger_count"].to_numpy() != 208

m &= train["pickup_latitude"].between(-90, 90).to_numpy()
m &= train["dropoff_latitude"].between(-90, 90).to_numpy()
m &= train["pickup_longitude"].between(-180, 180).to_numpy()
m &= train["dropoff_longitude"].between(-180, 180).to_numpy()

plon = train["pickup_longitude"].to_numpy()
plat = train["pickup_latitude"].to_numpy()
dlon = train["dropoff_longitude"].to_numpy()
dlat = train["dropoff_latitude"].to_numpy()
fare = train["fare_amount"].to_numpy()
hd = train["H_Distance"].to_numpy()
hour = train["Hour"].to_numpy()
dow = train["Day of Week"].to_numpy()

m &= ~(((plat == 0) & (plon == 0)) & ((dlat != 0) & (dlon != 0)) & (fare == 0))
m &= ~(((plat != 0) & (plon != 0)) & ((dlat == 0) & (dlon == 0)) & (fare == 0))

m &= ~((hd == 0) & (fare == 0))

m &= ~(
    ((hour >= 6) & (hour <= 20)) & ((dow >= 1) & (dow <= 5)) & (hd == 0) & (fare < 2.5)
)

m &= pd.Series(plon).between(nyc_bounds["min_lon"], nyc_bounds["max_lon"]).to_numpy()
m &= pd.Series(dlon).between(nyc_bounds["min_lon"], nyc_bounds["max_lon"]).to_numpy()
m &= pd.Series(plat).between(nyc_bounds["min_lat"], nyc_bounds["max_lat"]).to_numpy()
m &= pd.Series(dlat).between(nyc_bounds["min_lat"], nyc_bounds["max_lat"]).to_numpy()

m &= ~((plon == 0) | (plat == 0) | (dlon == 0) | (dlat == 0))

m &= (fare >= 2.5) & (fare <= 100.0)

m &= train["passenger_count"].between(1, 6).to_numpy()

m &= ~((hd < 0.01) & (fare > 15.0))

m &= hd > 0
m &= hd <= 50.0  # km

fare_per_km = fare / np.maximum(hd, 0.5)
m &= (hd < 0.5) | ((fare_per_km >= 0.5) & (fare_per_km <= 40.0))

m &= (np.abs(plon - dlon) <= 0.4) & (np.abs(plat - dlat) <= 0.4)

m &= hd <= 35.0

m &= ~((plon == dlon) & (plat == dlat))

train = train.loc[m].reset_index(drop=True)




## === cell 75
def clean_test_features_inplace(df, nyc_bounds):
    plon = df["pickup_longitude"].to_numpy()
    plat = df["pickup_latitude"].to_numpy()
    dlon = df["dropoff_longitude"].to_numpy()
    dlat = df["dropoff_latitude"].to_numpy()
    hd = df["H_Distance"].to_numpy()

    valid = np.ones(len(df), dtype=bool)

    valid &= pd.Series(plon).between(-180, 180).to_numpy()
    valid &= pd.Series(dlon).between(-180, 180).to_numpy()
    valid &= pd.Series(plat).between(-90, 90).to_numpy()
    valid &= pd.Series(dlat).between(-90, 90).to_numpy()

    valid &= (
        pd.Series(plon).between(nyc_bounds["min_lon"], nyc_bounds["max_lon"]).to_numpy()
    )
    valid &= (
        pd.Series(dlon).between(nyc_bounds["min_lon"], nyc_bounds["max_lon"]).to_numpy()
    )
    valid &= (
        pd.Series(plat).between(nyc_bounds["min_lat"], nyc_bounds["max_lat"]).to_numpy()
    )
    valid &= (
        pd.Series(dlat).between(nyc_bounds["min_lat"], nyc_bounds["max_lat"]).to_numpy()
    )

    valid &= ~((plon == 0) | (plat == 0) | (dlon == 0) | (dlat == 0))

    valid &= (hd > 0) & (hd <= 35.0)
    valid &= (np.abs(plon - dlon) <= 0.4) & (np.abs(plat - dlat) <= 0.4)

    valid &= ~((plon == dlon) & (plat == dlat))

    pc = df["passenger_count"].to_numpy()
    valid_pc = (pc >= 1) & (pc <= 6)

    bad = ~valid
    if bad.any():
        cols = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "H_Distance",
        ]
        df.loc[bad, cols] = np.nan

    bad_pc = ~valid_pc
    if bad_pc.any():
        df.loc[bad_pc, ["passenger_count"]] = np.nan

    return df


test_key = test["key"].copy()
test = clean_test_features_inplace(test, nyc_bounds)



## === cell 76
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 77
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test



## === cell 78
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
x_train_imp = imputer.fit_transform(x_train)
x_test_imp = imputer.transform(x_test)



## === cell 79
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(x_train_imp, y_train)
rf_predict = rf.predict(x_test_imp)

rf_predict = np.clip(rf_predict, 2.5, None)



## === cell 80
submission = pd.DataFrame({"key": test_key, "fare_amount": rf_predict})
submission.to_csv("submission.csv", index=False)
submission.head(20)
