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
plotly==5.24.1
plotly-express==0.4.1
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

5.69152

# 6. Current score

1057.71162

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1057.71162) has done: 'I fix the runtime errors caused by trying to `plt.imread()` remote URLs (Kaggle internet is typically blocked and newer Matplotlib disallows direct URL reads), by disabling those optional visualization/mask steps while keeping the rest of the pipeline intact. I also correct the data input paths to the actual provided `/kaggle/input/...` structure so the notebook runs in this environment. To move RMSE down dramatically toward the target, I fix the main logic issue causing the huge score: unbounded/negative predictions from linear regression (and potential NaNs/Infs) by applying a minimal, metric-consistent post-processing clamp to `fare_amount >= 0` and cleaning features (finite numeric, passenger_count bounds). Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print("Listing input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR))

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")



## === cell 1
train = pd.read_csv(TRAIN_PATH, nrows=1_000_000, parse_dates=["pickup_datetime"])
print(train.shape)
train.head()



## === cell 2
train.describe()
train.shape



## === cell 3
train = train[train.fare_amount >= 0]
print("Number of rows {:,}".format(len(train)))



## === cell 4
train.isnull().any()



## === cell 5
train = train[~train.dropoff_longitude.isnull()]
train = train[~train.dropoff_latitude.isnull()]
print("Number of rows {:,}".format(len(train)))



## === cell 6
test = pd.read_csv(TEST_PATH, nrows=2_000_000, parse_dates=["pickup_datetime"])
print(test.shape)
test.describe()



## === cell 7
plt.boxplot(train[train.pickup_latitude < 39].pickup_latitude)
plt.show()



## === cell 8
print("Minimus and maximum longitude Test ")
min(test.pickup_longitude.min(), test.dropoff_longitude.min()), max(
    test.pickup_longitude.max(), test.dropoff_longitude.max()
)



## === cell 9
print("Minimus and maximum latitude Test ")
min(test.pickup_latitude.min(), test.dropoff_latitude.min()), max(
    test.pickup_latitude.max(), test.dropoff_latitude.max()
)




## === cell 10
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


BB = (-74.5, -72.8, 40.5, 41.8)
BB_zoom = (-74.3, -73.7, 40.5, 40.9)

nyc_map = None
nyc_map_zoom = None



## === cell 11
train = train[select_within_boundingbox(train, BB)]
print("Number of rows {:,}".format(len(train)))




## === cell 12
def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2):
    if nyc_map is None:
        print("Map image not available; skipping plot_on_map.")
        return
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].scatter(
        df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c="r", s=s
    )
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title("Pickup locations")
    axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(
        df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c="r", s=s
    )
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title("Dropoff locations")
    axs[1].imshow(nyc_map, zorder=0, extent=BB)




## === cell 13
plot_on_map(train, BB, nyc_map, s=1, alpha=0.3)
plot_on_map(train, BB_zoom, nyc_map_zoom, s=1, alpha=0.3)



## === cell 14
nyc_mask = None




## === cell 15
def lonlat_to_xy(longitude, latitude, dx, dy, BB):
    return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
        dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
    ).astype("int")




## === cell 16
if nyc_mask is None:
    print("nyc_mask not available; skipping water-trip diagnostics.")




## === cell 17
def remove_datapoints_from_water(df):
    return df




## === cell 18
train = remove_datapoints_from_water(train)
print("Number of rows {:,}".format(len(train)))



## === cell 19
train["year"] = train.pickup_datetime.apply(lambda t: t.year)
train["weekday"] = train.pickup_datetime.apply(lambda t: t.weekday())
train["hour"] = train.pickup_datetime.apply(lambda t: t.hour)



## === cell 20
n_hours = 24
n_weekdays = 7
n_years = 7
n_bins_lon = 30
n_bins_lat = 30

BB_traffic = (-74.025, -73.925, 40.7, 40.8)


def calculate_trafic_density(df):
    traffic = np.zeros((n_years, n_weekdays, n_hours, n_bins_lat, n_bins_lon))

    bins_lon = np.zeros(n_bins_lon + 1)
    bins_lat = np.zeros(n_bins_lat + 1)

    delta_lon = (BB_traffic[1] - BB_traffic[0]) / n_bins_lon
    delta_lat = (BB_traffic[3] - BB_traffic[2]) / n_bins_lat

    for i in range(n_bins_lon + 1):
        bins_lon[i] = BB_traffic[0] + i * delta_lon
    for j in range(n_bins_lat + 1):
        bins_lat[j] = BB_traffic[2] + j * delta_lat

    for y in range(n_years):
        for d in range(n_weekdays):
            for h in range(n_hours):
                idx = (df.year == (2009 + y)) & (df.weekday == d) & (df.hour == h)

                inds_pickup_lon = np.digitize(df[idx].pickup_longitude, bins_lon)
                inds_pickup_lat = np.digitize(df[idx].pickup_latitude, bins_lat)

                for i in range(n_bins_lon):
                    for j in range(n_bins_lat):
                        traffic[y, d, h, j, i] = traffic[y, d, h, j, i] + np.sum(
                            (inds_pickup_lon == i + 1) & (inds_pickup_lat == j + 1)
                        )
    return traffic


def plot_traffic(traffic, y, d):
    days = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    fig, axs = plt.subplots(3, 8, figsize=(18, 7))
    axs = axs.ravel()
    for h in range(24):
        axs[h].imshow(
            traffic[y - 2009, days[d], h, ::-1, :],
            zorder=1,
            cmap="coolwarm",
            clim=(0, traffic.max()),
        )
        axs[h].get_xaxis().set_visible(False)
        axs[h].get_yaxis().set_visible(False)
        axs[h].set_title("h={}".format(h))
    fig.suptitle(
        "Pickup traffic density, year={}, day={} (max_pickups={})".format(
            y, d, traffic.max()
        )
    )




## === cell 21
traffic = calculate_trafic_density(train)




## === cell 22
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...




## === cell 23
train["distance_miles"] = distance(
    train.pickup_latitude,
    train.pickup_longitude,
    train.dropoff_latitude,
    train.dropoff_longitude,
)

train.distance_miles.hist(bins=50, figsize=(12, 4))
plt.xlabel("distance miles")
plt.title("Histogram ride distances in miles")
train.distance_miles.describe()



## === cell 24
print("Number of rows {:,}".format(len(train)))
train = train[train.distance_miles >= 0.05]
print("Number of rows {:,}".format(len(train)))



## === cell 25
jfk = (-73.7822222222, 40.6441666667)
nyc = (-74.0063889, 40.7141667)


def plot_location_fare(loc, name, range=1.5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    idx = (
        distance(train.pickup_latitude, train.pickup_longitude, loc[1], loc[0]) < range
    )
    train[idx].fare_amount.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel("fare $USD")
    axs[0].set_title(
        "Histogram pickup location within {} miles of {}".format(range, name)
    )

    idx = (
        distance(train.dropoff_latitude, train.dropoff_longitude, loc[1], loc[0])
        < range
    )
    train[idx].fare_amount.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel("fare $USD")
    axs[1].set_title(
        "Histogram dropoff location within {} miles of {}".format(range, name)
    )


plot_location_fare(jfk, "JFK Airport")



## === cell 26
ewr = (-74.175, 40.69)
lgr = (-73.87, 40.77)
plot_location_fare(ewr, "Newark Airport")
plot_location_fare(lgr, "LaGuardia Airport")



## === cell 27
train["fare_per_mile"] = train.fare_amount / train.distance_miles
train["distance_to_center"] = distance(
    nyc[1], nyc[0], train.pickup_latitude, train.pickup_longitude
)



## === cell 28
train["pickup_distance_to_jfk"] = distance(
    jfk[1], jfk[0], train.pickup_latitude, train.pickup_longitude
)
train["dropoff_distance_to_jfk"] = distance(
    jfk[1], jfk[0], train.dropoff_latitude, train.dropoff_longitude
)
train["pickup_distance_to_ewr"] = distance(
    ewr[1], ewr[0], train.pickup_latitude, train.pickup_longitude
)
train["dropoff_distance_to_ewr"] = distance(
    ewr[1], ewr[0], train.dropoff_latitude, train.dropoff_longitude
)
train["pickup_distance_to_lgr"] = distance(
    lgr[1], lgr[0], train.pickup_latitude, train.pickup_longitude
)
train["dropoff_distance_to_lgr"] = distance(
    lgr[1], lgr[0], train.dropoff_latitude, train.dropoff_longitude
)



## === cell 29
test["distance_miles"] = distance(
    test.pickup_latitude,
    test.pickup_longitude,
    test.dropoff_latitude,
    test.dropoff_longitude,
)
test["distance_to_center"] = distance(
    nyc[1], nyc[0], test.dropoff_latitude, test.dropoff_longitude
)
test["hour"] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
test["year"] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)



## === cell 30
for df in (train, test):
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")

train = train[(train.passenger_count >= 1) & (train.passenger_count <= 6)]
test.loc[
    ~((test.passenger_count >= 1) & (test.passenger_count <= 6)), "passenger_count"
] = 1

train = train[
    np.isfinite(train["distance_miles"]) & np.isfinite(train["distance_to_center"])
]

idx = (train.distance_to_center < 15) & (train.passenger_count != 0)
features = ["year", "hour", "distance_miles", "passenger_count"]
X = train[idx][features].astype(float).values
y = train[idx]["fare_amount"].astype(float).values
X.shape



## === cell 31
from sklearn.metrics import mean_squared_error, explained_variance_score


def plot_prediction_analysis(y_values, y_pred, figsize=(10, 4), title=""):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y_values, y_pred, s=2, alpha=0.3)
    mn = min(np.min(y_values), np.min(y_pred))
    mx = max(np.max(y_values), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c="red")
    axs[0].set_xlabel("$y$")
    axs[0].set_ylabel("$\hat{y}$")
    rmse = np.sqrt(mean_squared_error(y_values, y_pred))
    evs = explained_variance_score(y_values, y_pred)
    axs[0].set_title("rmse = {:.2f}, evs = {:.2f}".format(rmse, evs))

    axs[1].hist(y_values - y_pred, bins=50)
    avg = np.mean(y_values - y_pred)
    std = np.std(y_values - y_pred)
    axs[1].set_xlabel("$y - \hat{y}$")
    axs[1].set_title(
        "Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(
            avg, std
        )
    )

    if title != "":
        fig.suptitle(title)




## === cell 32
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)



## === cell 33
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



## === cell 34
XTEST = test[features].astype(float).values
y_pred_final = model_lin.predict(XTEST)

y_pred_final = np.asarray(y_pred_final, dtype=float)
y_pred_final[~np.isfinite(y_pred_final)] = np.nan
fallback = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 11.35
y_pred_final = np.where(np.isfinite(y_pred_final), y_pred_final, fallback)
y_pred_final = np.clip(y_pred_final, 0.0, None)

submission = pd.DataFrame(
    {"key": test.key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
submission.head()



## === cell 35
submission.head()



## === cell 36
try:
    from keras import backend as K
    from keras.models import Sequential
    from keras.layers import Dense

    def rmse_Keras(y_values, pred):
        return K.sqrt(K.mean(K.square(pred - y_values), axis=-1))

    def baseline_model():
        model = Sequential()
        model.add(
            Dense(
                X_train.shape[1],
                input_dim=X_train.shape[1],
                kernel_initializer="uniform",
                activation="softplus",
            )
        )
        model.add(Dense(1, kernel_initializer="uniform", activation="relu"))
        model.compile(loss="mse", optimizer="Adam", metrics=[rmse_Keras])
        return model

except Exception as e:
    print(
        "Skipping Keras model definition due to missing/unsupported keras in environment:",
        repr(e),
    )

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
