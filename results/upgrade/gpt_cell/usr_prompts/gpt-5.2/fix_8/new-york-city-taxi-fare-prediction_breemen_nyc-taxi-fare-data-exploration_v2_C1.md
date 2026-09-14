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

No external packages required in the script and installed.

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

797.89866

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1055.12907) has done: 'The crash happens because `matplotlib.pyplot.imread` no longer supports reading directly from a URL string; it requires a file path or a file-like object. In cell 9, we fix this by explicitly opening the URL with `urllib.request.urlopen`, then decoding it with Pillow (`PIL.Image.open`), and converting to a NumPy array as suggested by the error message. This keeps the rest of the plotting logic unchanged and preserves the `nyc_map` variable as an image array for `imshow`. No other cells are modified, and `plot_on_map(df_train, BB, nyc_map)` continues to work the same way.'
- What this solution (achieved 1051.73318) has done: 'Diagnosis: Cell 9 crashes because it tries to download a background map image from an external URL that currently returns HTTP 404 (and network access may be restricted), so `urllib.request.urlopen(url)` raises `HTTPError`. This map is only used for visualization and should not block the rest of the pipeline.  
Patch summary: In cell 9, wrap the URL download in a try/except and fall back to `nyc_map = None` when the download fails. Update `plot_on_map` to only call `imshow` if a map image is available, keeping the same plotting semantics otherwise.  
Updated cells: Only cell 9 is modified.  
Compatibility notes for cell k+1: Cell 10 does not depend on `nyc_map` or `plot_on_map`; it only uses `df_train`, which remains unchanged.  
Assumptions: It is acceptable to skip the map background image when it cannot be retrieved; the scatter plots still render and the notebook can proceed deterministically.'
- What this solution (achieved 1050.96904) has done: 'The crash happens because newer pandas versions no longer allow selecting multiple columns from a GroupBy using a tuple-like syntax inside `[...]`. The expression `['distance_km', 'fare_amount']` is being interpreted as a tuple in this context, triggering the `ValueError`. The minimal fix is to pass an explicit list of column names (double brackets) when subsetting after `groupby`. This keeps the same computation and output structure and does not affect downstream cells.'
- What this solution (achieved 1050.54104) has done: 'The crash happens because newer seaborn versions made `jointplot` accept only keyword arguments for `x` and `y`, so passing the two arrays positionally raises a `TypeError`. In cell 38, we should update the `sns.jointplot(...)` call to use explicit `x=` and `y=` while keeping the same plotting semantics. No other logic (data, modeling, or RMSE computation) is changed. This preserves the returned `JointGrid` object `g` and all subsequent uses.'
- What this solution (achieved 1048.84422) has done: 'Diagnosis: The crash happens in cell 38 inside `plot_rmse_analysis()` because newer seaborn versions no longer accept `x` and `y` as positional arguments in `sns.jointplot()`, and also removed/changed older parameters like `stat_func` and `size`. Passing two positional arrays triggers `TypeError: jointplot() takes from 0 to 1 positional arguments but 2 were given`.  
Patch summary: Update only the `sns.jointplot` call in cell 38 to use keyword arguments `x=` and `y=` and replace deprecated parameters with their modern equivalents (`height=` instead of `size=`, remove `stat_func`). This preserves the exact plotting intent and leaves the model training/evaluation logic unchanged.  
Updated cells: Only cell 38 is modified.  
Compatibility notes for cell k+1: Cell 39 relies on `features` and `df_test`, which remain unchanged; the patch only affects plotting and does not modify any variables used later.  
Assumptions: Seaborn is installed and available as imported earlier; the goal is to unblock execution rather than enforce identical plot aesthetics across seaborn versions.'
- What this solution (achieved 839.40934) has done: 'Your current RMSE (~1048) is catastrophically high for this competition, which almost always indicates invalid predictions (NaNs/infs) and/or extreme outliers caused by a feature bug rather than a weak model. The smallest core-logic-preserving fix is to correct the LGA (LaGuardia) distance feature bug (you accidentally compute `pickup_distance_to_lgr` using EWR coordinates), and then use the already-engineered airport/center distance features in training and test consistently. To stabilize RMSE further without changing the model type, we also clip negative fares in the submission to 0 (fares can’t be negative) and ensure datetime parsing is done once with `pd.to_datetime` for consistent hour/year extraction.'
- What this solution (achieved 797.89866) has done: 'Your RMSE is far worse than the target, so we should make the smallest changes that fix likely feature inconsistencies/bugs between train and test and remove obvious label noise, without changing the model type or training loop. The biggest issue is that `distance_to_center` is computed from pickup in train but from dropoff in test, so the model sees a different meaning at inference; we compute it from pickup in both. We also add the exact same minimal cleaning to test (drop NaNs in engineered datetime and fill remaining feature NaNs safely) and add a standard competition filter to remove extreme/outlier fares and coordinates in train, which typically collapses RMSE by orders of magnitude for this baseline. Finally, we make the train/test split deterministic to stabilize the score and ensure reproducibility.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

plt.style.use("seaborn-whitegrid")

RANDOM_STATE = 42



## === cell 1
df_train = pd.read_csv("../input/train.csv", nrows=500000)
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
import urllib.request
from PIL import Image

url = "https://aiblog.nl/download/nyc_-75_40_-73_41.5.png"

try:
    with urllib.request.urlopen(url) as resp:
        nyc_map = np.array(Image.open(resp))
except Exception:
    nyc_map = None


def plot_on_map(df, BB, nyc_map):
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
dt_train = pd.to_datetime(df_train["pickup_datetime"], errors="coerce")
df_train["hour"] = dt_train.dt.hour
df_train["year"] = dt_train.dt.year
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
ewr = (-74.175, 40.69)  # see https://www.travelmath.com/airport/EWR
df_train["pickup_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_train.pickup_latitude, df_train.pickup_longitude
)
df_train["dropoff_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_train.dropoff_latitude, df_train.dropoff_longitude
)

lgr = (-73.87, 40.77)  # see https://www.travelmath.com/airport/LGA

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
df_test = pd.read_csv("../input/test.csv")



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

dt_test = pd.to_datetime(df_test["pickup_datetime"], errors="coerce")
df_test["hour"] = dt_test.dt.hour
df_test["year"] = dt_test.dt.year



## === cell 31
df_test[~select_within_boundingbox(df_test, BB)]



## === cell 32
df_test["pickup_distance_to_jfk"] = distance(
    jfk[1], jfk[0], df_test.pickup_latitude, df_test.pickup_longitude
)
df_test["dropoff_distance_to_jfk"] = distance(
    jfk[1], jfk[0], df_test.dropoff_latitude, df_test.dropoff_longitude
)

df_test["pickup_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_test.pickup_latitude, df_test.pickup_longitude
)
df_test["dropoff_distance_to_ewr"] = distance(
    ewr[1], ewr[0], df_test.dropoff_latitude, df_test.dropoff_longitude
)

df_test["pickup_distance_to_lgr"] = distance(
    lgr[1], lgr[0], df_test.pickup_latitude, df_test.pickup_longitude
)
df_test["dropoff_distance_to_lgr"] = distance(
    lgr[1], lgr[0], df_test.dropoff_latitude, df_test.dropoff_longitude
)



## === cell 33
print("Train size before outlier filter:", len(df_train))
df_train = df_train[
    (df_train["fare_amount"] > 0)
    & (df_train["fare_amount"] <= 250)
    & (df_train["passenger_count"] > 0)
    & (df_train["passenger_count"] <= 6)
    & (df_train["distance_km"] <= 200)
]
print("Train size after outlier filter:", len(df_train))

df_train = df_train.dropna(subset=["year", "hour"])



## === cell 34
idx = (df_train.distance_to_center < 40) & (df_train.passenger_count != 0)

features = [
    "year",
    "hour",
    "distance_km",
    "passenger_count",
    "distance_to_center",
    "pickup_distance_to_jfk",
    "dropoff_distance_to_jfk",
    "pickup_distance_to_ewr",
    "dropoff_distance_to_ewr",
    "pickup_distance_to_lgr",
    "dropoff_distance_to_lgr",
]
X = df_train[idx][features].values
y = df_train[idx]["fare_amount"].values



## === cell 35
X.shape, y.shape



## === cell 36
from sklearn.metrics import mean_squared_error, explained_variance_score


def plot_prediction_analysis(y, y_pred, figsize=(10, 4), title=""):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c="red")
    axs[0].set_xlabel("$y$")
    axs[0].set_ylabel("$\hat{y}$")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title("rmse = {:.2f}, evs = {:.2f}".format(rmse, evs))

    axs[1].hist(y - y_pred, bins=50)
    avg = np.mean(y - y_pred)
    std = np.std(y - y_pred)
    axs[1].set_xlabel("$y - \hat{y}$")
    axs[1].set_title(
        "Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(
            avg, std
        )
    )

    if title != "":
        fig.suptitle(title)




## === cell 37
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)



## === cell 38
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline(
    (
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    )
)
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title="Linear Model - Trainingset")

y_test_pred = model_lin.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title="Linear Model - Testset")




## === cell 39
def plot_rmse_analysis(model, X, y, N=400, test_size=0.25, figsize=(10, 4), title=""):
    rmse_train, rmse_test = [], []
    for i in range(N):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE + i
        )

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    g = sns.jointplot(
        x=np.array(rmse_train), y=np.array(rmse_test), kind="scatter", height=5
    )
    g.set_axis_labels(
        "RMSE training ($\mu$={:.2f})".format(np.mean(rmse_train)),
        "RMSE test ($\mu$={:.2f})".format(np.mean(rmse_test)),
    )
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("{} (N={}, test_size={:0.2f})".format(title, N, test_size))


plot_rmse_analysis(model_lin, X, y, title="Linear model")



## === cell 40
train_feature_medians = df_train[features].median(numeric_only=True)
df_test[features] = df_test[features].fillna(train_feature_medians)

XTEST = df_test[features].values



## === cell 41
import os

filename = "./output/baseline_linear"
os.makedirs(os.path.dirname(filename), exist_ok=True)

y_pred_final = model_lin.predict(XTEST)

y_pred_final = np.clip(y_pred_final, 0, None)

submission = pd.DataFrame(
    {"key": df_test.key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)



## === cell 42
submission
