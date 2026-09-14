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

3.43988

# 6. Current score

6.43473

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.26696) has done: 'The crash is caused by a feature-name mismatch: `test_feat` still contains `fare_amount` because the earlier `align(..., join="left")` kept the train-only column in the test matrix. I fix this by aligning with `join="inner"` (and ensuring `fare_amount` is dropped from both feature matrices) so train/test have identical columns in the same order. I also update the input path handling to work in the provided `/kaggle/input` layout while keeping the same filenames, and ensure a valid `submission.csv` is always written. These changes are score-neutral (they just make inference consistent) and preserve the existing model/training logic.'
- What this solution (achieved 6.76919) has done: 'Your RMSE is far worse than the target, so we should make a small, legitimate improvement without changing the core modeling approach. The biggest likely issue is that the model is being trained on raw coordinates and a few simple features, and it’s very sensitive to outliers; tightening the data cleaning to the standard NYC bounding box plus a “non-zero distance” filter usually gives a large RMSE drop with minimal code change. I also ensure the datetime-derived features don’t create NaNs in test (coerce + fill), and I keep the same XGBoost training API/loop while adding a tiny regularization + subsampling to reduce overfitting on the small 105k sample (still the same model family and objective). These changes should move the score substantially toward ~3.44 while keeping runtime under the limit and producing a valid `submission.csv`.'
- What this solution (achieved 5.74904) has done: 'Your current RMSE (6.77) is far worse than the target (3.44), so we should make a small but high-impact improvement without changing the overall model family or training flow. The biggest issue is that you’re training on an arbitrary first 105k rows, which can be distribution-shifted; switching to a deterministic random sample from a larger chunk (still 105k final rows) usually drops RMSE substantially while keeping runtime low. I also add two standard, minimal feature tweaks that don’t change the core approach: extracting `hour` (and using it instead of only the “latenights” flag), and using `log1p` on distance-like features (which stabilizes XGBoost on heavy-tailed distances). Finally, I ensure the training `DMatrix` uses the best iteration found by early stopping when predicting (more consistent and typically better RMSE).'
- What this solution (achieved 5.63983) has done: 'Your current RMSE (5.749) is still far from the target (3.44), so the smallest high-impact change is to make the training sample more representative without changing the model or training loop: we draw the 105k training rows from a much larger initial chunk (e.g., 5M) rather than just 1M. We also add one more standard, minimal cleaning filter that strongly reduces RMSE for this competition: remove extreme `dist` outliers (very long trips) after computing distance, while keeping the same features and XGBoost objective. Finally, we keep everything else (features, model params, early stopping, submission writing) the same to preserve core logic and semantics.'
- What this solution (achieved 5.86306) has done: 'Your current script likely fails to generate a valid submission for Kaggle because you filter rows out of `test` (dropping NaNs / bounding-box), but still must submit predictions for all 9,914 test `key`s; this creates a row-count mismatch and/or missing keys. I keep your model, features, and training exactly the same, but change inference to never drop test rows: instead we compute features for all test rows, fill any missing feature values safely, predict for every key, and clip negatives as you already do. To preserve score direction (lower RMSE), we also apply the same basic coordinate sanity filtering only to *training* (as you already do) and keep test intact to avoid missing submissions. Finally, we build the submission using the original test `key` order to guarantee the correct shape and alignment.'
- What this solution (achieved 5.88759) has done: 'Your RMSE (5.86) is still far worse than the target (3.44), so we should make one or two small, high-impact quality fixes without changing the model family or training workflow. The biggest low-risk gain for this competition is to add the standard “airport + city center distance” geographic features (JFK/LGA/EWR/NYC-center), which are simple transformations of the same coordinates and typically reduce error a lot. We also make the train/valid split deterministic but more representative by sampling across the loaded chunk via a fixed random permutation (still using the same 5M read and 105k final size), and keep the rest (cleaning, XGBoost params, early stopping, submission writing) unchanged. These changes are directly score-oriented and should move RMSE closer toward ~3.44 while staying within runtime.'
- What this solution (achieved 6.43473) has done: 'I fix the crash in the haversine feature block by making `haversine_km` accept scalar airport coordinates (floats) as well as Series/arrays, which currently fails on `.astype`. I keep all model/training logic unchanged, and only adjust feature computation to be robust and deterministic for both train and test. I also add a tiny safety fill after feature creation to prevent any NaNs/inf from propagating into XGBoost (score-neutral but prevents runtime issues). The script then run end-to-end and always write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
import os

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]
INPUT_DIR = next((p for p in INPUT_DIR_CANDIDATES if os.path.exists(p)), "../input")

print("Using INPUT_DIR:", INPUT_DIR)
print("Top-level contents:", os.listdir(INPUT_DIR)[:20])



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

READ_NROWS = 5_000_000
FINAL_NROWS = 105_000

df_big = pd.read_csv(
    train_path,
    nrows=READ_NROWS,
    dtype={
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)

if len(df_big) > FINAL_NROWS:
    df = df_big.sample(n=FINAL_NROWS, random_state=42).reset_index(drop=True)
else:
    df = df_big

del df_big
print("Train rows used:", len(df))



## === cell 2
test = pd.read_csv(
    test_path,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)



## === cell 3
testkey = test["key"].copy()



## === cell 4
df = df.dropna(how="any", axis="rows")



## === cell 5
len(df)



## === cell 6
df.head()



## === cell 7
df.describe()



## === cell 8
NYC_LON_MIN, NYC_LON_MAX = -74.3, -72.9
NYC_LAT_MIN, NYC_LAT_MAX = 40.5, 41.8

l = df[
    (df.pickup_latitude < NYC_LAT_MIN)
    | (df.pickup_latitude > NYC_LAT_MAX)
    | (df.dropoff_latitude < NYC_LAT_MIN)
    | (df.dropoff_latitude > NYC_LAT_MAX)
    | (df.pickup_longitude < NYC_LON_MIN)
    | (df.pickup_longitude > NYC_LON_MAX)
    | (df.dropoff_longitude < NYC_LON_MIN)
    | (df.dropoff_longitude > NYC_LON_MAX)
].index



## === cell 9
df = df.drop(l, axis=0)



## === cell 10
z = df[
    (df.fare_amount > 350.0)
    | (df.fare_amount < 2.5)
    | (df.passenger_count > 6)
    | (df.passenger_count < 1)
].index



## === cell 11
df = df.drop(z, axis=0)



## === cell 12
len(df)




## === cell 13
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype="float64")
    lat1 = np.asarray(lat1, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * c  # Earth's mean radius in km




## === cell 14
df["dist"] = haversine_km(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
).astype("float32")
test["dist"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
).astype("float32")

df["abs_lon_diff"] = (
    (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
)
df["abs_lat_diff"] = (
    (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
)
test["abs_lon_diff"] = (
    (test["pickup_longitude"] - test["dropoff_longitude"]).abs().astype("float32")
)
test["abs_lat_diff"] = (
    (test["pickup_latitude"] - test["dropoff_latitude"]).abs().astype("float32")
)

df["manhattan_dist"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
test["manhattan_dist"] = (test["abs_lon_diff"] + test["abs_lat_diff"]).astype("float32")

JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895
NYC_LON, NYC_LAT = -73.985428, 40.748817  # midtown-ish

for name, (alon, alat) in {
    "jfk": (JFK_LON, JFK_LAT),
    "lga": (LGA_LON, LGA_LAT),
    "ewr": (EWR_LON, EWR_LAT),
    "nyc": (NYC_LON, NYC_LAT),
}.items():
    df[f"pickup_{name}_km"] = haversine_km(
        df["pickup_longitude"], df["pickup_latitude"], alon, alat
    ).astype("float32")
    df[f"dropoff_{name}_km"] = haversine_km(
        df["dropoff_longitude"], df["dropoff_latitude"], alon, alat
    ).astype("float32")
    test[f"pickup_{name}_km"] = haversine_km(
        test["pickup_longitude"], test["pickup_latitude"], alon, alat
    ).astype("float32")
    test[f"dropoff_{name}_km"] = haversine_km(
        test["dropoff_longitude"], test["dropoff_latitude"], alon, alat
    ).astype("float32")

df = df[df["dist"] > 0.01].copy()
df = df[df["dist"] < 100.0].copy()

dist_like_cols = ["dist", "abs_lon_diff", "abs_lat_diff", "manhattan_dist"] + [
    f"{side}_{name}_km"
    for side in ["pickup", "dropoff"]
    for name in ["jfk", "lga", "ewr", "nyc"]
]
for c in dist_like_cols:
    df[c] = np.log1p(df[c].astype("float64")).astype("float32")
    test[c] = np.log1p(test[c].astype("float64")).astype("float32")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)



## === cell 15
test.head()



## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")



## === cell 17
df = df.dropna(subset=["pickup_datetime"])



## === cell 18
df.info()



## === cell 19
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype("int8")
test_hour = test["pickup_datetime"].dt.hour
test["latenights"] = (test_hour.fillna(12) < 5).astype("int8")

df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
test["hour"] = test_hour.fillna(12).astype("int8")



## === cell 20
df.pickup_datetime.iloc[0].weekday()



## === cell 21
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype("int8")
test_wd = test["pickup_datetime"].dt.weekday
test["weekday"] = (test_wd.fillna(2) > 4).astype("int8")



## === cell 22
df.head()



## === cell 23
df["year"] = df["pickup_datetime"].dt.year.astype("int16")
test_year = test["pickup_datetime"].dt.year
test["year"] = test_year.fillna(
    test_year.mode(dropna=True).iloc[0] if test_year.notna().any() else 2010
).astype("int16")



## === cell 24
df["day"] = df["pickup_datetime"].dt.day.astype("int8")
test_day = test["pickup_datetime"].dt.day
test["day"] = test_day.fillna(
    test_day.mode(dropna=True).iloc[0] if test_day.notna().any() else 15
).astype("int8")



## === cell 25
df.head()



## === cell 26
test.head()



## === cell 27
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 28
test_feat.year.unique()



## === cell 29
feat = pd.get_dummies(feat, columns=["year"], prefix="year")
test_feat = pd.get_dummies(test_feat, columns=["year"], prefix="year")

feat, test_feat = feat.align(test_feat, join="inner", axis=1, fill_value=0)



## === cell 30
feat.head()



## === cell 31
test_feat.head()



## === cell 32
label = df["fare_amount"].astype("float32")

if "fare_amount" in feat.columns:
    feat = feat.drop("fare_amount", axis=1)
if "fare_amount" in test_feat.columns:
    test_feat = test_feat.drop("fare_amount", axis=1)

train_medians = feat.median(numeric_only=True)
feat = feat.fillna(train_medians)
test_feat = test_feat.fillna(train_medians)



## === cell 33
xtr, xts, ytr, yts = train_test_split(feat, label, test_size=0.2, random_state=42)



## === cell 34
xgbtrain = xgboost.DMatrix(xtr, label=ytr)
xgbvalid = xgboost.DMatrix(xts, label=yts)
xgbfinaltest = xgboost.DMatrix(test_feat)



## === cell 35
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "verbosity": 0,
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
}



## === cell 36
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=350,
    early_stopping_rounds=30,
    evals=[(xgbvalid, "valid")],
)



## === cell 37
best_ntree = getattr(xgbmodel, "best_iteration", None)
if best_ntree is None:
    pred = xgbmodel.predict(xgbfinaltest)
else:
    pred = xgbmodel.predict(xgbfinaltest, iteration_range=(0, best_ntree + 1))



## === cell 38
pred = np.clip(pred, 0.0, None)



## === cell 39
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})
finalset = finalset[["key", "fare_amount"]]



## === cell 40
finalset.head()



## === cell 41
finalset.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", finalset.shape)
print(finalset.head())
