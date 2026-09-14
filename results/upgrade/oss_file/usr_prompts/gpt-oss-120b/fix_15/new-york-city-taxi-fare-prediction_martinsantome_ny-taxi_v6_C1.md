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

9.00838

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1072.56608) has done: 'I fixed the image‑loading errors by adding a small utility that reads images from URLs with Pillow, updated the cells that used `plt.imread` to use this helper, and added the needed imports. I also expanded the feature set slightly (including `distance_to_center`) to improve the linear model’s RMSE while keeping the original modeling approach unchanged. Finally, I ensured predictions are non‑negative and that a proper `submission.csv` is written.'
- What this solution (achieved 1188.10869) has done: 'I fixed the fallback data path so that, when the primary filter removes all rows, the script loads a smaller training sample, adds the same engineered features (`year`, `hour`, `distance_miles`, `distance_to_center`) and applies the original filter before building the training matrix. This prevents the KeyError and ensures the scaler and linear model receive a non‑empty array, allowing the pipeline to finish and write a valid `submission.csv`.'
- What this solution (achieved 9.96619) has done: 'I relax the overly strict distance‑to‑center filter, add the missing weekday feature to both training and test data, and adjust the fallback sample creation accordingly. These minimal changes keep the original modeling pipeline intact while providing the model with more relevant information, which should lower the RMSE toward the target.'
- What this solution (achieved 9.96549) has done: 'I fix the TensorFlow import error by removing the failing import block, add a missing `month` feature for better temporal modeling, and filter out extreme fare amounts (e.g., > 200 USD) to reduce outliers that hurt RMSE. These minimal changes keep the original linear pipeline intact while improving data quality and model performance, moving the score closer to the target.'
- What this solution (achieved 8.3624) has done: 'I added the missing airport‑distance features to the test set (cell 26) and extended the feature list used for the linear model to include those distances (cell 27). By giving the model more relevant geographic information it can capture fare variations better, which should lower the RMSE and bring the score closer to the target.'
- What this solution (achieved 6.44997) has done: 'I add a polynomial‑features step to the existing linear pipeline so the model can capture simple interactions without changing the overall linear‑regression framework. This small tweak usually lowers RMSE, moving the score closer to the target while keeping the core logic intact.'
- What this solution (achieved 8.83684) has done: 'I replace the plain LinearRegression with a Ridge regression (small L2 regularization) while keeping the same polynomial‑features pipeline. This minimal change preserves the core modeling approach but often reduces over‑fit, leading to a lower validation RMSE and thus moving the score closer to the target. The rest of the notebook remains unchanged, and the final submission file is still written as before.'
- What this solution (achieved 9.74708) has done: 'I lower model complexity to reduce over‑fitting and bring the RMSE closer to the target. In the modeling cell I change the polynomial degree from 2 to 1 (no interaction terms) and increase the Ridge regularisation α from 0.5 to 5.0. These tiny adjustments keep the original pipeline intact while likely improving validation performance and moving the score toward the target.'
- What this solution (achieved 9.00838) has done: 'I keep the overall pipeline unchanged but give the linear model a bit more expressive power by using quadratic polynomial features (degree 2) and slightly reduce the ridge regularisation (α = 1.0). This small change preserves the core logic while allowing the model to capture simple interactions, which should lower the RMSE and bring the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
import urllib.request
from urllib.error import HTTPError, URLError
from PIL import Image


def load_image(url):
    """Load an image from a URL and return it as a NumPy array.
    If the URL cannot be reached, return a dummy 1×1 black image."""
    try:
        with urllib.request.urlopen(url) as resp:
            img = Image.open(resp)
            return np.array(img)
    except (HTTPError, URLError, OSError):
        return np.zeros((1, 1, 3), dtype=np.uint8)


print(os.listdir("../input"))




## === cell 1
train = pd.read_csv(
    "../input/train.csv", nrows=1_000_000, parse_dates=["pickup_datetime"]
)
print(train.shape)
train.head()




## === cell 2
train = train[train.fare_amount >= 0]
print("Number of rows {:,}".format(len(train)))




## === cell 3
print("Any nulls per column:", train.isnull().any().to_dict())
train = train[~train.dropoff_longitude.isnull()]
train = train[~train.dropoff_latitude.isnull()]
print("Number of rows after dropping missing coordinates {:,}".format(len(train)))




## === cell 4
test = pd.read_csv(
    "../input/test.csv", nrows=2_000_000, parse_dates=["pickup_datetime"]
)
print(test.shape)
test.describe()




## === cell 5
plt.boxplot(train[train.pickup_latitude < 39].pickup_latitude)
plt.show()




## === cell 6
print(
    "Minimus and maximum longitude Test ",
    min(test.pickup_longitude.min(), test.dropoff_longitude.min()),
    max(test.pickup_longitude.max(), test.dropoff_longitude.max()),
)




## === cell 7
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
nyc_map = load_image("https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png")

BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = load_image("https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png")




## === cell 8
train = train[select_within_boundingbox(train, BB)]
print("Number of rows after BB filter {:,}".format(len(train)))




## === cell 9
def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2):
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




## === cell 10
plot_on_map(train, BB, nyc_map, s=1, alpha=0.3)
plot_on_map(train, BB_zoom, nyc_map_zoom, s=1, alpha=0.3)




## === cell 11
nyc_mask = (
    load_image("https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png")[:, :, 0]
    > 0.9
)

plt.figure(figsize=(8, 8))
plt.imshow(nyc_map, zorder=0)
plt.imshow(nyc_mask, zorder=1, alpha=0.7)  # True = land (black), False = water (white)




## === cell 12
def lonlat_to_xy(longitude, latitude, dx, dy, BB):
    return (
        (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"),
        (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype("int"),
    )


try:
    pickup_x, pickup_y = lonlat_to_xy(
        train.pickup_longitude,
        train.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropout_x, dropout_y = lonlat_to_xy(
        train.dropoff_longitude,
        train.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
except Exception:
    pickup_x = pickup_y = dropout_x = dropout_y = np.array([], dtype=int)




## === cell 13
try:
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropout_x]
    print("Number of trips in water: {}".format(np.sum(~idx)))
except Exception:
    idx = np.ones(len(train), dtype=bool)
    print("Water‑mask step skipped due to placeholder mask.")




## === cell 14
def remove_datapoints_from_water(df):
    BB = (-74.5, -72.8, 40.5, 41.8)
    try:
        nyc_mask_local = (
            load_image("https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png")[
                :, :, 0
            ]
            > 0.9
        )
    except Exception:
        return df

    def lonlat_to_xy_local(longitude, latitude, dx, dy, BB):
        return (
            (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"),
            (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype("int"),
        )

    pickup_x, pickup_y = lonlat_to_xy_local(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask_local.shape[1],
        nyc_mask_local.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy_local(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask_local.shape[1],
        nyc_mask_local.shape[0],
        BB,
    )
    idx = nyc_mask_local[pickup_y, pickup_x] & nyc_mask_local[dropoff_y, dropout_x]
    return df[idx]




## === cell 15
train = remove_datapoints_from_water(train)
print("Number of rows after water‑mask removal {:,}".format(len(train)))




## === cell 16
train["year"] = train.pickup_datetime.apply(lambda t: t.year)
train["month"] = train.pickup_datetime.apply(lambda t: t.month)
train["weekday"] = train.pickup_datetime.apply(lambda t: t.weekday())
train["hour"] = train.pickup_datetime.apply(lambda t: t.hour)




## === cell 17
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
                        traffic[y, d, h, j, i] += np.sum(
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
        axs[h].axis("off")
        axs[h].set_title(f"h={h}")
    fig.suptitle(
        f"Pickup traffic density, year={y}, day={d} (max_pickups={traffic.max()})"
    )




## === cell 18
traffic = calculate_trafic_density(train)




## === cell 19
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...




## === cell 20
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




## === cell 21
print("Number of rows before distance filter {:,}".format(len(train)))
train = train[train.distance_miles >= 0.05]
print("Number of rows after distance filter {:,}".format(len(train)))

train = train[train.fare_amount <= 200]
print("Number of rows after fare outlier filter {:,}".format(len(train)))




## === cell 22
jfk = (-73.7822222222, 40.6441666667)
nyc = (-74.0063889, 40.7141667)


def plot_location_fare(loc, name, range=1.5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    idx = (
        distance(train.pickup_latitude, train.pickup_longitude, loc[1], loc[0]) < range
    )
    train[idx].fare_amount.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel("fare $USD")
    axs[0].set_title(f"Histogram pickup location within {range} miles of {name}")

    idx = (
        distance(train.dropoff_latitude, train.dropoff_longitude, loc[1], loc[0])
        < range
    )
    train[idx].fare_amount.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel("fare $USD")
    axs[1].set_title(f"Histogram dropoff location within {range} miles of {name}")


plot_location_fare(jfk, "JFK Airport")




## === cell 23
ewr = (-74.175, 40.69)  # Newark Liberty International Airport
lgr = (-73.87, 40.77)  # LaGuardia Airport
plot_location_fare(ewr, "Newark Airport")
plot_location_fare(lgr, "LaGuardia Airport")




## === cell 24
train["fare_per_mile"] = train.fare_amount / train.distance_miles
train["distance_to_center"] = distance(
    nyc[1], nyc[0], train.pickup_latitude, train.pickup_longitude
)




## === cell 25
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




## === cell 26
test["distance_miles"] = distance(
    test.pickup_latitude,
    test.pickup_longitude,
    test.dropoff_latitude,
    test.dropoff_longitude,
)
test["distance_to_center"] = distance(
    nyc[1], nyc[0], test.pickup_latitude, test.pickup_longitude
)
test["hour"] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
test["year"] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
test["weekday"] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).weekday())
test["month"] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).month)

test["pickup_distance_to_jfk"] = distance(
    jfk[1], jfk[0], test.pickup_latitude, test.pickup_longitude
)
test["dropoff_distance_to_jfk"] = distance(
    jfk[1], jfk[0], test.dropoff_latitude, test.dropoff_longitude
)
test["pickup_distance_to_ewr"] = distance(
    ewr[1], ewr[0], test.pickup_latitude, test.pickup_longitude
)
test["dropoff_distance_to_ewr"] = distance(
    ewr[1], ewr[0], test.dropoff_latitude, test.dropoff_longitude
)
test["pickup_distance_to_lgr"] = distance(
    lgr[1], lgr[0], test.pickup_latitude, test.pickup_longitude
)
test["dropoff_distance_to_lgr"] = distance(
    lgr[1], lgr[0], test.dropoff_latitude, test.dropoff_longitude
)




## === cell 27
idx = train.passenger_count != 0

features = [
    "year",
    "month",  # now reliably present
    "hour",
    "weekday",
    "distance_miles",
    "passenger_count",
    "distance_to_center",
    "pickup_distance_to_jfk",
    "dropoff_distance_to_jfk",
    "pickup_distance_to_ewr",
    "dropoff_distance_to_ewr",
    "pickup_distance_to_lgr",
    "dropoff_distance_to_lgr",
]
features = [f for f in features if f in train.columns]

X = train.loc[idx, features].fillna(0).values
y = train.loc[idx, "fare_amount"].fillna(0).values
print("Training matrix shape after primary filter:", X.shape)

if X.shape[0] == 0:
    print("Primary filter removed all rows – loading a fresh fallback sample.")
    fallback = pd.read_csv(
        "../input/train.csv", nrows=50_000, parse_dates=["pickup_datetime"]
    )
    fallback = fallback[fallback.fare_amount >= 0].copy()
    fallback["year"] = fallback.pickup_datetime.dt.year
    fallback["month"] = fallback.pickup_datetime.dt.month
    fallback["hour"] = fallback.pickup_datetime.dt.hour
    fallback["weekday"] = fallback.pickup_datetime.dt.weekday
    fallback["distance_miles"] = distance(
        fallback.pickup_latitude,
        fallback.pickup_longitude,
        fallback.dropoff_latitude,
        fallback.dropoff_longitude,
    )
    fallback["distance_to_center"] = distance(
        nyc[1], nyc[0], fallback.pickup_latitude, fallback.pickup_longitude
    )
    fallback["pickup_distance_to_jfk"] = distance(
        jfk[1], jfk[0], fallback.pickup_latitude, fallback.pickup_longitude
    )
    fallback["dropoff_distance_to_jfk"] = distance(
        jfk[1], jfk[0], fallback.dropoff_latitude, fallback.dropoff_longitude
    )
    fallback["pickup_distance_to_ewr"] = distance(
        ewr[1], ewr[0], fallback.pickup_latitude, fallback.pickup_longitude
    )
    fallback["dropoff_distance_to_ewr"] = distance(
        ewr[1], ewr[0], fallback.dropoff_latitude, fallback.dropoff_longitude
    )
    fallback["pickup_distance_to_lgr"] = distance(
        lgr[1], lgr[0], fallback.pickup_latitude, fallback.pickup_longitude
    )
    fallback["dropoff_distance_to_lgr"] = distance(
        lgr[1], lgr[0], fallback.dropoff_latitude, fallback.dropoff_longitude
    )
    fallback["passenger_count"] = fallback.passenger_count.fillna(0)
    idx_fallback = fallback.passenger_count != 0
    fallback = fallback[idx_fallback]
    X = fallback[features].fillna(0).values
    y = fallback["fare_amount"].values
    print("Fallback training matrix shape:", X.shape)




## === cell 28
from sklearn.model_selection import train_test_split

if X.shape[0] < 2:
    X_train, X_val, y_train, y_val = X, X, y, y
    print("Very small dataset – using the same data for training and validation.")
else:
    test_size = 0.25 if X.shape[0] > 4 else 0.5
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(
        f"Split {X.shape[0]} samples into {X_train.shape[0]} train and {X_val.shape[0]} val."
    )




## === cell 29
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt


def plot_prediction_analysis(y_true, y_pred, figsize=(10, 4), title=""):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c="red")
    axs[0].set_xlabel("$y$")
    axs[0].set_ylabel(r"$\hat{y}$")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    evs = explained_variance_score(y_true, y_pred)
    axs[0].set_title(f"rmse = {rmse:.2f}, evs = {evs:.2f}")

    axs[1].hist(y_true - y_pred, bins=50, alpha=0.7)
    axs[1].set_xlabel("$y - \\hat{y}$")
    axs[1].set_title(
        f"Histogram error (μ={np.mean(y_true-y_pred):.2f}, σ={np.std(y_true-y_pred):.2f})"
    )
    if title:
        fig.suptitle(title)




## === cell 30
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

model_lin = Pipeline(
    [
        (
            "poly_features",
            PolynomialFeatures(degree=2, include_bias=False),
        ),
        ("standard_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=1.0, random_state=42)),
    ]
)

model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title="Ridge Model – Train")

y_val_pred = model_lin.predict(X_val)
plot_prediction_analysis(y_val, y_val_pred, title="Ridge Model – Validation")




## === cell 31
X_test = test[features].fillna(0).values
y_pred_final = model_lin.predict(X_test)
y_pred_final = np.maximum(y_pred_final, 0)  # fare cannot be negative

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}, shape:", submission.shape)




## === cell 32
print("Keras model skipped – linear model used for predictions.")




## === cell 33
submission.head()
