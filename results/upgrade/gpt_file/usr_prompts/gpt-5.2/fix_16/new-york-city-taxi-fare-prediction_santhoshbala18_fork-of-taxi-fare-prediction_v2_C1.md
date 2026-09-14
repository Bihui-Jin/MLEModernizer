# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.70513

# 6. Current score

9.11303

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.41453) has done: 'I fix the LightGBM API break by removing the deprecated `silent` argument and correcting misspelled parameter keys so training runs in LightGBM 4.6.0. I also make the pipeline deterministic and prevent a common logic bug where the airport longitude/latitude bounds are incorrectly signed (NYC longitudes are negative), applying the same feature logic to both train and test. Finally, I keep the core modeling/training approach the same, but ensure the script runs end-to-end and always writes a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 5.41453) has done: 'We need to move RMSE down from 5.41 toward 3.705, so the smallest meaningful gain is to fix a feature mismatch/omission that currently hurts generalization: you compute `is_night`, `is_airport`, `is_surge` but then drop the datetime-derived columns and never actually use `pickup_datetime` itself, so the flags are the only time signal and they are kept—however, your current feature set also includes raw lat/lon with no sanity filtering on test and no clipping of extreme distances, which can cause the model to extrapolate poorly on the leaderboard. I keep the same LightGBM training approach and parameters, but (1) apply the same passenger_count bounds and bounding-box filtering to *test* rows via safe clipping (not dropping rows, to keep submission row count), and (2) add a single robustification step: clip `distance_km` to a reasonable max based on the filtered training distribution to reduce the impact of outliers. These are minimal, metric-aligned changes that typically reduce RMSE for this competition without changing the core modeling logic. The script still run end-to-end and write `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.36821) has done: 'Your current RMSE (5.41) is far above the target (3.705), so the smallest meaningful improvement is to fix likely train/test distribution issues without changing the model or training loop. I add a minimal but very impactful training filter: remove extreme/unrealistic fares and very long trips from the training sample (a standard NYC Taxi Fare cleanup), while keeping your existing bounding-box + passenger filtering and the same LightGBM setup. I also ensure the same `distance_km` clipping logic remains consistent and non-leaky by computing caps from the already-filtered training data only. This should reduce outlier-driven error and move RMSE down toward the target while preserving your core approach.'
- What this solution (achieved 5.7214) has done: 'Your RMSE (5.37) is substantially worse than the 3.705 target (lower is better), so we should make the smallest changes that improve generalization without changing your model/training loop. The biggest issue is that you compute time-derived flags but then drop the underlying time features (`hour/day/month/weekday`), which removes useful signal; we keep those columns as numeric features while still dropping the raw datetime string. We also add one standard, minimal NYC taxi cleanup filter that removes “zero-distance but non-trivial fare” rows (often meter errors) from the *training* sample only, which tends to reduce RMSE without changing modeling semantics. Everything else (LightGBM, objective, boosting rounds, distance feature, bounding box, and submission writing) stays the same.'
- What this solution (achieved 5.72806) has done: 'Your RMSE (5.7214) is still far from the target (3.70513, lower is better), so we should make a small but high-impact correction that keeps the same LightGBM training approach and features. The biggest remaining issue is that `distance_km` is computed on raw (possibly out-of-bounds) test coordinates, then you later clip test coordinates—this creates an inconsistent feature (distance doesn’t match the final lat/lon). I recompute `distance_km` after the test-set clipping (and keep train untouched) so the model sees consistent geometry at inference. This is minimal, preserves your core logic, and typically reduces leaderboard error without changing the model, loss, or training loop.'
- What this solution (achieved 5.72806) has done: 'Your current RMSE (5.72806, lower is better) is far above the 3.70513 target, so the smallest high-impact move is to fix a remaining train/test mismatch that hurts generalization. Right now you clip test coordinates to the bounding box, but you never clip training coordinates; that makes the model learn on raw coordinates (including residual edge/outlier values that survive the filter) while predicting on clipped coordinates, which can degrade accuracy. I apply the same safe clipping (not row-dropping) to the already-filtered training data before computing `distance_km` caps and training, keeping your LightGBM setup, features, and training loop unchanged. This keeps the core logic identical while reducing distribution shift, which should move RMSE down toward the target band.'
- What this solution (achieved 5.72589) has done: 'Your current RMSE (5.72806, lower is better) is well above the target (3.70513), so we should make a small, metric-aligned improvement without changing the model or training loop. The biggest remaining issue is that the model can output negative fares and very small fares for short trips, which are heavily penalized by RMSE; we add a minimal, standard post-processing step to clip predictions to a realistic lower bound (and a conservative upper bound) for both validation and submission. This keeps identical training semantics and features, but improves evaluation by preventing impossible predictions. We also keep the submission format unchanged and ensure it still writes a valid `submission.csv`.'
- What this solution (achieved 5.6507) has done: 'We need to move RMSE down from 5.72589 toward 3.70513 (lower is better), and we’re far outside the ±10% target band, so a small but meaningful generalization fix is warranted while keeping the same LightGBM training approach and feature set. The biggest remaining issue is that the training split is random, but the competition is effectively time-based; switching to a simple chronological split (train on earlier rides, validate on later rides) reduces leakage and helps the model generalize to the test distribution without changing the model, objective, or features. I also ensure train/test share identical feature columns and dtypes before fitting/predicting (a common silent performance killer when pandas columns differ). Everything else (distance feature, cleaning, flags, LightGBM params, post-prediction clipping, and submission writing) is kept intact.'
- What this solution (achieved 5.63462) has done: 'Your current RMSE (5.6507) is far worse than the target (3.70513, lower is better), so we should make a small but meaningful generalization improvement without changing your LightGBM model/training loop or feature set. The biggest remaining issue is that you train on a 1e6 random head sample without any stratification; for this competition, model quality improves a lot if the training sample better covers the full time span and fare distribution. I keep the same core preprocessing/feature engineering, but switch to a deterministic, time-spread sampling via `skiprows` (so we still read ~1e6 rows fast) and add one standard NYC cleanup that removes obviously invalid coordinate zeros. This should move RMSE down toward the target while preserving your approach and still producing a valid `submission.csv`.'
- What this solution (achieved 5.59568) has done: 'Your RMSE (5.63) is still far above the 3.705 target (lower is better), so we need a small but high-impact generalization improvement without changing your LightGBM training approach, objective, or feature set. The biggest remaining issue is that the training sample is time-spread but still heavily biased by the huge volume of “easy” short urban rides; adding a deterministic, minimal reweighting by distance bins (via LightGBM sample weights) improves fit on longer rides that otherwise dominate RMSE when missed, while preserving the same model/loops. I also make your existing airport flag symmetric (pickup OR dropoff near airport) because airport trips are a distinct regime and this is a tiny feature tweak that usually helps a lot in this competition. Everything else (cleaning, features, params, rounds, and submission writing) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 5.90452) has done: 'We need to move RMSE down from 5.59568 toward the 3.70513 target (lower is better), so we make one small, high-impact data-cleaning adjustment while keeping your LightGBM model/training loop and feature set unchanged. The main issue is remaining label noise/outliers in the training sample: add a standard NYC taxi constraint that removes rows where the fare is implausible given the trip distance (very high $/km or near-zero $/km), which typically reduces RMSE substantially without changing modeling semantics. This is applied only to training (never to test), and we keep your existing bounding-box filters, datetime features, weights, and prediction clipping. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.89738) has done: 'Your current RMSE (5.90452) is far above the target (3.70513, lower is better), so we should make one minimal, high-impact correction that preserves your LightGBM training loop and feature set. The biggest remaining issue is that `pickup_datetime` is parsed without an explicit timezone, but the NYC Taxi Fare dataset timestamps are in UTC; using local NYC time shifts hour/weekday features and can significantly hurt generalization. I convert datetimes from UTC to `America/New_York` before extracting `hour/day/month/weekday`, keeping the same engineered columns, model params, weights, clipping, and submission writing. This change keeps the core logic intact but makes the time-based features align with the real-world regime, which should move RMSE down toward the target.'
- What this solution (achieved 5.92922) has done: 'To move RMSE down toward 3.705 with minimal disruption, I keep your exact LightGBM training loop/params and existing features, but fix two likely high-impact data issues: (1) ensure timezone conversion never produces NaT-driven missing time features (fill those deterministically), and (2) add one standard NYC fare cleanup that removes obviously bogus low-fare rides (fare < 2.5) that add noise and inflate RMSE. I also add a simple Manhattan-distance feature (in addition to your existing haversine `distance_km`) computed from the same clipped coordinates; this is a small feature addition (not an architectural change) that typically reduces error in this competition. Finally, I keep your submission writing unchanged and guaranteed to produce `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.94686) has done: 'We need to move RMSE down from 5.92922 toward 3.70513 (lower is better), and we’re far outside the ±10% band, so a small but high-impact modeling-aligned fix is warranted while keeping your LightGBM training loop, objective, and features intact. The biggest issue is that taxi fare is heavily piecewise-linear in distance with a non-zero intercept; adding a single feature `distance_sq = distance_km^2` is a minimal extension (no architecture/loop change) that commonly reduces RMSE substantially for this competition. To keep semantics stable and avoid train/test mismatch, we compute and clip this feature consistently for both train and test using caps derived from filtered training only. Everything else (data reading, cleaning, datetime handling, weights, LightGBM params/rounds, and submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 9.11303) has done: 'To move RMSE down toward the 3.70513 target with minimal disruption, I’m keeping your exact LightGBM training loop/params and all existing features, but fixing one clear feature/cleaning inconsistency: you filter out implausible `fare_per_km` rows using `distance_km` (haversine), yet you also train on `manhattan_km` and `distance_sq`, so remaining “geometry-inconsistent” outliers can still inject noise. I add the same kind of plausibility filter using `manhattan_km` (computed already) and keep it very conservative so it removes only extreme label noise. I also recompute the distance-derived features after the final clipping step to guarantee internal consistency (important because you clip after computing these features), without changing the model architecture or training semantics. The script still runs end-to-end and writes a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

print(os.listdir("../input"))



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

NROWS = 10**6
p_keep = 0.02  # expected keep ~= 55M * 0.02 ≈ 1.1M
rng = np.random.RandomState(42)
skip = lambda i: i > 0 and (rng.rand() > p_keep)

df = pd.read_csv(train_path, skiprows=skip, low_memory=False)
if len(df) > NROWS:
    df = df.iloc[:NROWS].copy()

test_set = pd.read_csv(test_path, low_memory=False)




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df.pickup_latitude,
    df.pickup_longitude,
    df.dropoff_latitude,
    df.dropoff_longitude,
)



## === cell 3
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df_in, BB):
    return (
        (df_in.pickup_longitude >= BB[0])
        & (df_in.pickup_longitude <= BB[1])
        & (df_in.pickup_latitude >= BB[2])
        & (df_in.pickup_latitude <= BB[3])
        & (df_in.dropoff_longitude >= BB[0])
        & (df_in.dropoff_longitude <= BB[1])
        & (df_in.dropoff_latitude >= BB[2])
        & (df_in.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))

df = df[
    ~(
        (df["pickup_longitude"] == 0)
        | (df["pickup_latitude"] == 0)
        | (df["dropoff_longitude"] == 0)
        | (df["dropoff_latitude"] == 0)
    )
].copy()

df = df[select_within_boundingbox(df, BB)]
df = df[(df.passenger_count > 0) & (df.passenger_count < 10)]
print("New size: %d" % len(df))

df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=9)
for col, lo, hi in [
    ("pickup_longitude", BB[0], BB[1]),
    ("dropoff_longitude", BB[0], BB[1]),
    ("pickup_latitude", BB[2], BB[3]),
    ("dropoff_latitude", BB[2], BB[3]),
]:
    df[col] = df[col].clip(lower=lo, upper=hi)

df["distance_km"] = distance(
    df.pickup_latitude,
    df.pickup_longitude,
    df.dropoff_latitude,
    df.dropoff_longitude,
)

test_set["passenger_count"] = test_set["passenger_count"].clip(lower=1, upper=9)
for col, lo, hi in [
    ("pickup_longitude", BB[0], BB[[1]][0]),
    ("dropoff_longitude", BB[0], BB[1]),
    ("pickup_latitude", BB[2], BB[3]),
    ("dropoff_latitude", BB[2], BB[3]),
]:
    test_set[col] = test_set[col].clip(lower=lo, upper=hi)

test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)

df["manhattan_km"] = distance(
    df.pickup_latitude, df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude
) + distance(
    df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.pickup_longitude
)
test_set["manhattan_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.pickup_latitude,
    test_set.dropoff_longitude,
) + distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.pickup_longitude,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2904365332.py in <cell line: 0>()
     48 test_set["passenger_count"] = test_set["passenger_count"].clip(lower=1, upper=9)
     49 for col, lo, hi in [
---> 50     ("pickup_longitude", BB[0], BB[[1]][0]),
     51     ("dropoff_longitude", BB[0], BB[1]),
     52     ("pickup_latitude", BB[2], BB[3]),

TypeError: tuple indices must be integers or slices, not list

## === cell 4
def add_datetime_info(dataset):
    dt_utc = pd.to_datetime(dataset["pickup_datetime"], utc=True, errors="coerce")
    dt_local = dt_utc.dt.tz_convert("America/New_York")
    dataset["pickup_datetime"] = dt_local

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    dataset["hour"] = dataset["hour"].fillna(12).astype(np.int16)
    dataset["day"] = dataset["day"].fillna(15).astype(np.int16)
    dataset["month"] = dataset["month"].fillna(6).astype(np.int16)
    dataset["weekday"] = dataset["weekday"].fillna(2).astype(np.int16)

    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
def add_custom_flags(dataset):
    dataset["is_night"] = np.where(
        (
            ((dataset["hour"] >= 20) & (dataset["hour"] <= 23))
            | ((dataset["hour"] >= 0) & (dataset["hour"] < 6))
        ),
        1,
        0,
    )

    drop_is_airport = (
        (dataset["dropoff_longitude"] >= -73.79)
        & (dataset["dropoff_longitude"] <= -73.75)
        & (dataset["dropoff_latitude"] >= 40.63)
        & (dataset["dropoff_latitude"] <= 40.66)
    )
    pick_is_airport = (
        (dataset["pickup_longitude"] >= -73.79)
        & (dataset["pickup_longitude"] <= -73.75)
        & (dataset["pickup_latitude"] >= 40.63)
        & (dataset["pickup_latitude"] <= 40.66)
    )
    dataset["is_airport"] = np.where((drop_is_airport | pick_is_airport), 1, 0)

    dataset["is_surge"] = np.where(
        (
            ((dataset["hour"] >= 16) & (dataset["hour"] < 20))
            & ((dataset["weekday"] != 5) & (dataset["weekday"] != 6))
        ),
        1,
        0,
    )
    return dataset


df = add_custom_flags(df)
test_set = add_custom_flags(test_set)



## === cell 6
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)

df = df[df["fare_amount"] >= 2.5].copy()

df = df[df["fare_amount"] <= 250].copy()  # typical competition cleanup
df = df[
    df["distance_km"] <= 200
].copy()  # remove extreme trips within BB that are likely noise
df = df[~((df["distance_km"] < 0.05) & (df["fare_amount"] > 3.0))].copy()

eps = 0.1  # km; avoids exploding ratios near zero distance
fare_per_km = df["fare_amount"] / (df["distance_km"] + eps)
df = df[(fare_per_km <= 100.0) & (fare_per_km >= 0.5)].copy()

fare_per_manh_km = df["fare_amount"] / (df["manhattan_km"] + eps)
df = df[(fare_per_manh_km <= 100.0) & (fare_per_manh_km >= 0.5)].copy()

dist_cap = float(df["distance_km"].quantile(0.999))
df["distance_km"] = df["distance_km"].clip(lower=0.0, upper=dist_cap)
test_set["distance_km"] = test_set["distance_km"].clip(lower=0.0, upper=dist_cap)

man_cap = float(df["manhattan_km"].quantile(0.999))
df["manhattan_km"] = df["manhattan_km"].clip(lower=0.0, upper=man_cap)
test_set["manhattan_km"] = test_set["manhattan_km"].clip(lower=0.0, upper=man_cap)

df["distance_sq"] = (df["distance_km"] ** 2).astype(np.float32)
test_set["distance_sq"] = (test_set["distance_km"] ** 2).astype(np.float32)

dist_sq_cap = float(df["distance_sq"].quantile(0.999))
df["distance_sq"] = df["distance_sq"].clip(lower=0.0, upper=dist_sq_cap)
test_set["distance_sq"] = test_set["distance_sq"].clip(lower=0.0, upper=dist_sq_cap)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'manhattan_km'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3014499091.py in <cell line: 0>()
     15 df = df[(fare_per_km <= 100.0) & (fare_per_km >= 0.5)].copy()
     16 
---> 17 fare_per_manh_km = df["fare_amount"] / (df["manhattan_km"] + eps)
     18 df = df[(fare_per_manh_km <= 100.0) & (fare_per_manh_km >= 0.5)].copy()
     19 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'manhattan_km'

## === cell 7
try:
    plt.scatter(df["is_night"], df["fare_amount"], c="r", s=2, alpha=0.3)
    plt.show()
except Exception as e:
    print("Plot skipped due to:", repr(e))



## === cell 8
df_sorted = df.sort_values("pickup_datetime").reset_index(drop=True)

train_full = df_sorted.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
target_full = df_sorted["fare_amount"]

split_idx = int(len(df_sorted) * 0.9)
X_train = train_full.iloc[:split_idx].copy()
y_train = target_full.iloc[:split_idx].copy()
X_test = train_full.iloc[split_idx:].copy()
y_test = target_full.iloc[split_idx:].copy()

feature_cols = list(X_train.columns)
X_test = X_test.reindex(columns=feature_cols)
test_set_features = test_set.drop(["key", "pickup_datetime"], axis=1).reindex(
    columns=feature_cols
)

for c in feature_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    test_set_features[c] = pd.to_numeric(test_set_features[c], errors="coerce")

med = X_train.median(numeric_only=True)
X_train = X_train.fillna(med)
X_test = X_test.fillna(med)
test_set_features = test_set_features.fillna(med)

dist_train = X_train["distance_km"].astype(float).values
bins = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0], dtype=float)
bin_idx = np.digitize(dist_train, bins, right=False)  # 1..len(bins)
counts = np.bincount(bin_idx, minlength=len(bins) + 2).astype(float)
weights = 1.0 / np.maximum(counts[bin_idx], 1.0)
weights = weights / np.mean(weights)  # normalize to mean 1.0 for stability



## === cell 9
import lightgbm as lgbm

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1.0,
    "reg_lambda": 0.001,
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "verbose": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
}

train_set = lgbm.Dataset(X_train, label=y_train, weight=weights)

model = lgbm.train(params, train_set=train_set, num_boost_round=300)



## === cell 10
predictions = model.predict(X_test, num_iteration=model.current_iteration())



## === cell 11
import math
from sklearn.metrics import mean_squared_error

predictions_pp = np.clip(predictions, 2.5, 250.0)

rmse = math.sqrt(mean_squared_error(y_test, predictions_pp))
print(rmse)



## === cell 12
test_set_key = test_set["key"]

y_pred_final = model.predict(test_set_features, num_iteration=model.current_iteration())
y_pred_final_pp = np.clip(y_pred_final, 2.5, 250.0)

submission = pd.DataFrame(
    {"key": test_set_key, "fare_amount": y_pred_final_pp},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submitted: submission.csv")



## === cell 13
submission.shape
