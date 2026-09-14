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

5.68499

# 6. Current score

6.43769

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.75045) has done: 'I fix the runtime errors caused by (1) trying to `imread()` a URL without downloading it, (2) pandas groupby multi-column selection syntax, and (3) seaborn API changes for `jointplot()`. I also ensure the notebook-like code runs as a plain Python script in Kaggle by removing the Jupyter magic and by writing the submission CSV to a guaranteed-writable location with the correct `key,fare_amount` columns. To move RMSE from ~1056 down toward the ~5.68 target (lower is better), I add one minimal, standard post-processing step that preserves the linear model core logic: clip predictions to a reasonable non-negative range based on the cleaned training target distribution. Everything else (features, model, training loop/approach) remains the same.'
- What this solution (achieved 6.75133) has done: 'To move RMSE down from 6.75 toward the 5.68 target without changing your core model/feature logic, I (1) make the train/test split deterministic (so the fitted model is stable and avoids unlucky splits) and (2) fit the final model on all available filtered training data (instead of only the 75% split), which typically improves generalization for this competition while preserving the same LinearRegression+StandardScaler pipeline. I also (3) keep your existing prediction clipping but derive bounds from the same filtered training subset actually used to fit, to avoid inconsistent post-processing. These are minimal changes that keep architecture/training approach intact and should reduce the public RMSE gap.'
- What this solution (achieved 6.25719) has done: 'You’re currently above the target (RMSE 6.75 vs 5.68; lower is better), so we should make the smallest legitimate improvement without changing the core LinearRegression+StandardScaler pipeline or your feature set. The biggest remaining score drag in this classic NYC taxi baseline is unfiltered/invalid training rows (extreme fares, implausible passenger_count) that heavily skew a linear model; we add a minimal, standard outlier filter on `fare_amount` and valid passenger counts using only the already-loaded 500k sample. We also make the `distance_to_center` feature consistent (use pickup for both train/test; you were using pickup for train but dropoff for test), which preserves the same feature definitions but fixes a train/test mismatch that hurts RMSE. Finally, we keep your existing prediction clipping, but compute bounds from the filtered training target actually used for fitting.'
- What this solution (achieved 6.27722) has done: 'We’re currently worse than the target (RMSE 6.257 vs 5.685; lower is better), so we make the smallest legitimate improvements that keep your exact LinearRegression+StandardScaler pipeline and the same 4-feature set. The biggest low-risk gain is fixing a remaining train/test mismatch: you filter training rows with `distance_to_center < 40` but you do not apply the same filter logic to the test features (and your model never sees `distance_to_center` at all), so we remove that unused filter to avoid accidentally biasing the training distribution. Next, we add a minimal, standard validity filter for coordinates and passenger_count *before* feature engineering (without changing the features), which reduces noise/outliers that linear regression fits poorly. Finally, we keep your prediction clipping but set the upper bound from the filtered training distribution consistently (and also enforce the known NYC minimum fare of 2.5) to reduce extreme prediction penalties under RMSE.'
- What this solution (achieved 6.26628) has done: 'Your current RMSE (6.277) is above the target (5.685, lower is better), so we should make small, legitimate data-cleaning changes that improve a LinearRegression-on-simple-features baseline without changing the model/feature set. The biggest remaining low-risk gain is to remove obviously-invalid training rows that distort a linear fit: zero/negative fares (already handled), but also “free rides”/tiny fares, extreme fares, and especially bad GPS points (0,0; out-of-range lat/lon) and unrealistic passenger_count. I also apply the same basic validity filtering to the test set (without dropping rows—just fill invalids to safe values) to avoid generating extreme predictions for corrupted test rows. Finally, I keep your existing prediction clipping, but set the upper clip bound from a slightly less aggressive percentile (99.5 instead of 99.9) to reduce occasional large-error outliers under RMSE.'
- What this solution (achieved 6.44519) has done: 'We’re currently above the target (RMSE 6.266 vs 5.685; lower is better), so we make the smallest legitimate improvements that keep your exact feature set and the same `StandardScaler + LinearRegression` pipeline. The biggest low-risk gain for this competition is better target conditioning for a linear model: apply a log1p transform to `fare_amount` during fitting and invert with expm1 at prediction time (same core model, just a monotonic target transform that aligns better with RMSE). We also add one more minimal, standard validity filter: remove rows with implausible `fare_per_km` (extreme price per km values that strongly skew a linear fit). Finally, we keep your existing prediction clipping, but compute `fare_max` from the inverse-transformed training target distribution consistently.'
- What this solution (achieved 6.43958) has done: 'We’re currently worse than the target (RMSE 6.445 vs 5.685; lower is better), so the smallest safe improvement is to reduce noise/outliers that a linear model is sensitive to without changing the model or feature set. I add one additional, standard training-row validity filter that removes trips with implausible coordinates near NYC (on top of the bounding box) and a conservative maximum distance cap, which usually improves RMSE for this competition while preserving your same engineered features and pipeline. I also make the RMSE-resampling diagnostic deterministic to avoid accidental variability, but keep the same approach. Finally, I keep your existing log1p target transform and clipping, only deriving the clip bound from the filtered training target you actually fit on.'
- What this solution (achieved 6.43769) has done: 'We’re currently worse than the target (RMSE 6.43958 vs 5.68499; lower is better), so the smallest likely gain without changing your model/pipeline is to reduce remaining label noise and mismatch in preprocessing. I add one conservative filter to drop rows with extreme pickup years (a common corruption) and remove rows with unrealistically low/high fares-per-km that can still slip through and skew a linear model. I also apply the same bounding-box/coordinate sanity logic to the test set as “repair” (not dropping rows), so out-of-range test points don’t produce extreme distances and predictions. The model, features (`["year","hour","distance_km","passenger_count"]`), log1p target transform, and clipping semantics remain the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

INPUT_DIR_DEFAULT = "/kaggle/input/new-york-city-taxi-fare-prediction"
if os.path.exists(INPUT_DIR_DEFAULT):
    INPUT_DIR = INPUT_DIR_DEFAULT
else:
    INPUT_DIR = "../input"

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")



## === cell 1
df_train = pd.read_csv(TRAIN_PATH, nrows=500000)
df_train.head()



## === cell 2
df_train.dtypes



## === cell 3
df_train.describe()



## === cell 4
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >= 0]
print("New size: %d" % len(df_train))



## === cell 5
df_train[df_train.fare_amount < 100].fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")



## === cell 6
print(df_train.isnull().sum())



## === cell 7
print("Old size: %d" % len(df_train))
df_train = df_train.dropna(how="any", axis="rows")
print("New size: %d" % len(df_train))



## === cell 8
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))




## === cell 9
def plot_on_map(df, BB, nyc_map=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=0.2, c="r")
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title("Pickup locations")
    if nyc_map is not None:
        axs[0].imshow(nyc_map, zorder=0, extent=[-75, -73, 40, 41.5])

    axs[1].scatter(
        df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=0.2, c="r"
    )
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title("Dropoff locations")
    if nyc_map is not None:
        axs[1].imshow(nyc_map, zorder=0, extent=[-75, -73, 40, 41.5])


nyc_map = None
plot_on_map(df_train, BB, nyc_map)




## === cell 10
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df_train["distance_km"] = distance(
    df_train.pickup_latitude,
    df_train.pickup_longitude,
    df_train.dropoff_latitude,
    df_train.dropoff_longitude,
)

df_train.distance_km.hist(bins=50, figsize=(12, 4))
plt.xlabel("distance km")
plt.title("Histogram")
df_train.distance_km.describe()



## === cell 11
df_train.groupby("passenger_count")[["distance_km", "fare_amount"]].mean()



## === cell 12
print(
    "Average $USD/KM : {:0.2f}".format(
        df_train.fare_amount.sum() / df_train.distance_km.sum()
    )
)



## === cell 13
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].scatter(df_train.distance_km, df_train.fare_amount, alpha=0.2)
axs[0].set_xlabel("distance km")
axs[0].set_ylabel("fare $USD")
axs[0].set_title("All data")

idx = (df_train.distance_km < 21) & (df_train.fare_amount < 100)
axs[1].scatter(df_train[idx].distance_km, df_train[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel("distance km")
axs[1].set_ylabel("fare $USD")
axs[1].set_title("Zoom in on distance < 20km, fare < $100")



## === cell 14
idx = df_train.distance_km >= 0.1
print("Old size: %d" % len(df_train))
df_train = df_train[idx]
print("New size: %d" % len(df_train))



## === cell 15
jfk = (-73.7822222222, 40.6441666667)
nyc = (-74.0063889, 40.7141667)

print(
    "Distance JFK airport - NYC center = {} km".format(
        distance(jfk[1], jfk[0], nyc[1], nyc[0])
    )
)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
idx = distance(df_train.pickup_latitude, df_train.pickup_longitude, jfk[1], jfk[0]) < 3
df_train[idx].fare_amount.hist(bins=100, ax=axs[0])
axs[0].set_xlabel("fare")
axs[0].set_title("Histogram pickup location within 3km of JKF Airport")

idx = (
    distance(df_train.dropoff_latitude, df_train.dropoff_longitude, jfk[1], jfk[0]) < 3
)
df_train[idx].fare_amount.hist(bins=100, ax=axs[1])
axs[1].set_xlabel("fare")
axs[1].set_title("Histogram dropoff location within 3km of JKF Airport")



## === cell 16
dt = pd.to_datetime(df_train.pickup_datetime)
df_train["hour"] = dt.dt.hour
df_train["year"] = dt.dt.year
df_train["fare_per_km"] = df_train.fare_amount / df_train.distance_km



## === cell 17
df_train.fare_per_km.describe()



## === cell 18
idx = (df_train.distance_km < 5) & (df_train.fare_amount < 100)
plt.scatter(df_train[idx].distance_km, df_train[idx].fare_per_km)
plt.xlabel("distance km")
plt.ylabel("fare per distance km")

theta = (12.0, 4.0)
x = np.linspace(0.1, 5, 100)
plt.plot(x, theta[0] / x + theta[1], "--", c="r", lw=2)



## === cell 19
df_train.pivot_table("fare_per_km", index="hour", columns="year").plot(figsize=(14, 6))
plt.ylabel("Fare $USD / KM")



## === cell 20
df_train["distance_to_center"] = distance(
    nyc[1], nyc[0], df_train.pickup_latitude, df_train.pickup_longitude
)



## === cell 21
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
im = axs[0].scatter(
    df_train.distance_to_center,
    df_train.distance_km,
    c=np.clip(df_train.fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[0].set_xlabel("pickup distance from NYC center")
axs[0].set_ylabel("distance km")
axs[0].set_title("All data")
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel("fare_amount", rotation=270)

idx = (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(
    df_train[idx].distance_to_center,
    df_train[idx].distance_km,
    c=np.clip(df_train[idx].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[1].set_xlabel("pickup distance from NYC center")
axs[1].set_ylabel("distance km")
axs[1].set_title("Zoom in")
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel("fare_amount", rotation=270)



## === cell 22
df_train["pickup_distance_to_jfk"] = distance(
    jfk[1], jfk[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_jfk"] = distance(
    jfk[1], jfk[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)



## === cell 23
idx = ~((df_train.pickup_distance_to_jfk < 3) | (df_train.dropoff_distance_to_jfk < 3))

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
im = axs[0].scatter(
    df_train[idx].distance_to_center,
    df_train[idx].distance_km,
    c=np.clip(df_train[idx].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[0].set_xlabel("pickup distance from NYC center")
axs[0].set_ylabel("distance km")
axs[0].set_title("All data")
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel("fare_amount", rotation=270)

idx1 = idx & (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(
    df_train[idx1].distance_to_center,
    df_train[idx1].distance_km,
    c=np.clip(df_train[idx1].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[1].set_xlabel("pickup distance from NYC center")
axs[1].set_ylabel("distance km")
axs[1].set_title("Zoom in")
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel("fare_amount", rotation=270)



## === cell 24
idx = (df_train.fare_amount > 80) & (df_train.distance_km < 40)
plot_on_map(df_train[idx], BB, nyc_map)



## === cell 25
ewr = (-74.175, 40.69)  # EWR
df_train["pickup_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)

lgr = (-73.87, 40.77)  # LGA
df_train["pickup_distance_to_lgr"] = distance(
    lgr[1], lgr[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_lgr"] = distance(
    lgr[1], lgr[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)



## === cell 26
idx = ~(
    (df_train.pickup_distance_to_jfk < 3)
    | (df_train.dropoff_distance_to_jfk < 3)
    | (df_train.pickup_distance_to_ewr < 3)
    | (df_train.dropoff_distance_to_ewr < 3)
    | (df_train.pickup_distance_to_lgr < 3)
    | (df_train.dropoff_distance_to_lgr < 3)
)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
im = axs[0].scatter(
    df_train[idx].distance_to_center,
    df_train[idx].distance_km,
    c=np.clip(df_train[idx].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[0].set_xlabel("pickup distance from NYC center")
axs[0].set_ylabel("distance km")
axs[0].set_title("All data")
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.set_ylabel("fare_amount", rotation=270)

idx1 = idx & (df_train.distance_to_center < 21) & (df_train.distance_km < 40)
im = axs[1].scatter(
    df_train[idx1].distance_to_center,
    df_train[idx1].distance_km,
    c=np.clip(df_train[idx1].fare_amount, 0, 100),
    cmap="viridis",
    alpha=1.0,
    s=1,
)
axs[1].set_xlabel("pickup distance from NYC center")
axs[1].set_ylabel("distance km")
axs[1].set_title("Zoom in")
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.set_ylabel("fare_amount", rotation=270)



## === cell 27
df_test = pd.read_csv(TEST_PATH)



## === cell 28
plot_on_map(df_test, BB, nyc_map)



## === cell 29
df_test.passenger_count.hist()



## === cell 30
df_test["distance_km"] = distance(
    df_test.pickup_latitude,
    df_test.pickup_longitude,
    df_test.dropoff_latitude,
    df_test.dropoff_longitude,
)

df_test["distance_to_center"] = distance(
    nyc[1], nyc[0], df_test.pickup_latitude, df_test.pickup_longitude
)

dt_test = pd.to_datetime(df_test.pickup_datetime)
df_test["hour"] = dt_test.dt.hour
df_test["year"] = dt_test.dt.year



## === cell 31
df_test[~select_within_boundingbox(df_test, BB)]



## === cell 32
fare_low, fare_high = 2.5, 60.0
print("Old size (pre outlier/validity filter): %d" % len(df_train))

valid_coords = (
    df_train["pickup_longitude"].between(-180.0, 180.0)
    & df_train["dropoff_longitude"].between(-180.0, 180.0)
    & df_train["pickup_latitude"].between(-90.0, 90.0)
    & df_train["dropoff_latitude"].between(-90.0, 90.0)
    & ~((df_train["pickup_longitude"] == 0.0) & (df_train["pickup_latitude"] == 0.0))
    & ~((df_train["dropoff_longitude"] == 0.0) & (df_train["dropoff_latitude"] == 0.0))
)

fare_per_km_ok = df_train["fare_per_km"].between(1.0, 30.0)

center_ok = df_train["distance_to_center"].between(0.0, 50.0)
dist_ok = df_train["distance_km"].between(0.2, 80.0)

year_ok = df_train["year"].between(2009, 2015)

df_train = df_train[
    valid_coords
    & fare_per_km_ok
    & center_ok
    & dist_ok
    & year_ok
    & (df_train["fare_amount"] >= fare_low)
    & (df_train["fare_amount"] <= fare_high)
    & (df_train["passenger_count"] >= 1)
    & (df_train["passenger_count"] <= 6)
]
print("New size (post outlier/validity filter): %d" % len(df_train))



## === cell 33
features = ["year", "hour", "distance_km", "passenger_count"]
X = df_train[features].values
y = df_train["fare_amount"].values



## === cell 34
X.shape, y.shape



## === cell 35
from sklearn.metrics import mean_squared_error, explained_variance_score


def plot_prediction_analysis(y, y_pred, figsize=(10, 4), title=""):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c="red")
    axs[0].set_xlabel("$y$")
    axs[0].set_ylabel("$\\hat{y}$")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title("rmse = {:.2f}, evs = {:.2f}".format(rmse, evs))

    axs[1].hist(y - y_pred, bins=50)
    avg = np.mean(y - y_pred)
    std = np.std(y - y_pred)
    axs[1].set_xlabel("$y - \\hat{y}$")
    axs[1].set_title(
        "Histrogram prediction error, $\\mu$ = {:.2f}, $\\sigma$ = {:.2f}".format(
            avg, std
        )
    )

    if title != "":
        fig.suptitle(title)




## === cell 36
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)



## === cell 37
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline(
    (
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    )
)

y_train_t = np.log1p(y_train)
y_test_t = np.log1p(y_test)

model_lin.fit(X_train, y_train_t)

y_train_pred = np.expm1(model_lin.predict(X_train))
plot_prediction_analysis(
    y_train, y_train_pred, title="Linear Model (log1p target) - Trainingset"
)

y_test_pred = np.expm1(model_lin.predict(X_test))
plot_prediction_analysis(
    y_test, y_test_pred, title="Linear Model (log1p target) - Testset"
)




## === cell 38
def plot_rmse_analysis(model, X, y, N=400, test_size=0.25, figsize=(10, 4), title=""):
    rmse_train, rmse_test = [], []
    rng = np.random.RandomState(123)
    for i in range(N):
        rs = int(rng.randint(0, 2**31 - 1))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs
        )
        y_train_t = np.log1p(y_train)
        y_test_t = np.log1p(y_test)

        model.fit(X_train, y_train_t)

        y_train_pred = np.expm1(model.predict(X_train))
        y_test_pred = np.expm1(model.predict(X_test))

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    g = sns.jointplot(
        x=np.array(rmse_train), y=np.array(rmse_test), kind="scatter", height=5
    )
    g.set_axis_labels(
        "RMSE training ($\\mu$={:.2f})".format(np.mean(rmse_train)),
        "RMSE test ($\\mu$={:.2f})".format(np.mean(rmse_test)),
    )
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("{} (N={}, test_size={:0.2f})".format(title, N, test_size))




## === cell 39
plot_rmse_analysis(model_lin, X, y, title="Linear model (log1p target)")



## === cell 40
df_test_sane = df_test.copy()
df_test_sane["passenger_count"] = df_test_sane["passenger_count"].clip(1, 6)

df_test_sane["pickup_longitude"] = df_test_sane["pickup_longitude"].clip(BB[0], BB[1])
df_test_sane["dropoff_longitude"] = df_test_sane["dropoff_longitude"].clip(BB[0], BB[1])
df_test_sane["pickup_latitude"] = df_test_sane["pickup_latitude"].clip(BB[2], BB[3])
df_test_sane["dropoff_latitude"] = df_test_sane["dropoff_latitude"].clip(BB[2], BB[3])

df_test_sane["distance_km"] = distance(
    df_test_sane.pickup_latitude,
    df_test_sane.pickup_longitude,
    df_test_sane.dropoff_latitude,
    df_test_sane.dropoff_longitude,
).clip(0.2, 80.0)

XTEST = df_test_sane[features].values

model_lin.fit(X, np.log1p(y))



## === cell 41
y_pred_final = np.expm1(model_lin.predict(XTEST)).astype(float)

fare_min = 2.5
fare_max = float(np.percentile(y, 99.5))
y_pred_final = np.clip(y_pred_final, fare_min, fare_max)

submission = pd.DataFrame(
    {"key": df_test.key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

submission.head()



## === cell 42
submission
