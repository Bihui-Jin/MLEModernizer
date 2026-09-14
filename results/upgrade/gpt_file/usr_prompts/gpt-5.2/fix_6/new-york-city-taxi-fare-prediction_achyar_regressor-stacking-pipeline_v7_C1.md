# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
joblib==1.5.2
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
mlxtend==0.23.4
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

# 5. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.3f" % x)
RSEED = 2020

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from mlxtend.regressor import StackingCVRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

np.random.seed(RSEED)




## === cell 1
def _resolve_path(preferred: str, fallback: str) -> str:
    return preferred if os.path.exists(preferred) else fallback


train_path = _resolve_path(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "../input/new-york-city-taxi-fare-prediction/train.csv",
)

NROWS_TRAIN = 1_000_000

data = pd.read_csv(train_path, nrows=NROWS_TRAIN, parse_dates=["pickup_datetime"])
data = data.dropna()



## === cell 2
data.head()



## === cell 3
data.describe(include="all")



## === cell 4
data.isna().sum() / data.shape[0] * 100



## === cell 5
try:
    sns.histplot(data["fare_amount"], bins=100)
    plt.title("Fare amount distribution")
    plt.show()
except Exception:
    pass




## === cell 6
def ecdf(x):
    """Empirical cumulative distribution function of a variable"""
    x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, 1) / n
    return x, y




## === cell 7
try:
    xs, ys = ecdf(data["fare_amount"])
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, ".")
    plt.ylabel("Percentile")
    plt.title("ECDF of Fare Amount")
    plt.xlabel("Fare Amount ($)")
    plt.show()
except Exception:
    pass



## === cell 8
try:
    data["passenger_count"].value_counts().plot.bar(color="b", edgecolor="k")
    plt.title("Passenger Counts")
    plt.xlabel("Number of Passengers")
    plt.ylabel("Count")
    plt.show()
except Exception:
    pass



## === cell 9
print(
    "ada "
    + str(data[data["passenger_count"] == 0].shape[0])
    + " transaksi dengan 0 passangger"
)
print(
    "ada "
    + str(data[data["passenger_count"] == 6].shape[0])
    + " transaksi dengan 6 passangger"
)



## === cell 10
data = data.loc[data["pickup_latitude"].between(40, 42)]
data = data.loc[data["pickup_longitude"].between(-75, -72)]
data = data.loc[data["dropoff_latitude"].between(40, 42)]
data = data.loc[data["dropoff_longitude"].between(-75, -72)]



## === cell 11
from sklearn.cluster import KMeans

try:
    kmeans = KMeans(n_clusters=5, random_state=RSEED, n_init=10)
    data["cluster_pickup"] = kmeans.fit_predict(
        data[["pickup_longitude", "pickup_latitude"]]
    )
    data["cluster_dropoff"] = kmeans.fit_predict(
        data[["dropoff_longitude", "dropoff_latitude"]]
    )
except Exception:
    data["cluster_pickup"] = 0
    data["cluster_dropoff"] = 0




## === cell 12
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


BB_zoom = (-74.1, -73.7, 40.6, 40.85)
nyc_map_zoom = None  # no background image available offline




## === cell 13
def plot_on_map(df, BB, nyc_map=None, s=10, alpha=0.2):
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].scatter(
        df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c="r", s=s
    )
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title("Pickup locations")
    axs[0].axis("off")
    if nyc_map is not None:
        axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(
        df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c="b", s=s
    )
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title("Dropoff locations")
    axs[1].axis("off")
    if nyc_map is not None:
        axs[1].imshow(nyc_map, zorder=0, extent=BB)
    plt.show()


try:
    plot_on_map(
        data.sample(min(50_000, len(data)), random_state=RSEED),
        BB_zoom,
        nyc_map_zoom,
        s=0.05,
        alpha=0.05,
    )
except Exception:
    pass



## === cell 14
try:
    palette = sns.color_palette("Paired", 10)
    color_mapping = {
        c: palette[i % len(palette)]
        for i, c in enumerate(pd.Series(data["cluster_pickup"]).unique())
    }
    data["color"] = data["cluster_pickup"].map(color_mapping)
    plot_data = data.sample(min(50_000, len(data)), random_state=RSEED)
except Exception:
    plot_data = data.head(0)



## === cell 15
try:
    BB = BB_zoom
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    if len(plot_data) > 0:
        for b, df_ in plot_data.groupby("cluster_pickup"):
            axs.scatter(
                df_.pickup_longitude,
                df_.pickup_latitude,
                alpha=0.2,
                c=[color_mapping[b]],
                s=10,
                label=f"{b}",
            )
        axs.set_xlim((BB[0], BB[1]))
        axs.set_ylim((BB[2], BB[3]))
        axs.set_title("Pickup locations")
        axs.axis("off")
        axs.legend()
    plt.show()
except Exception:
    pass



## === cell 16
try:
    palette = sns.color_palette("Paired", 10)
    color_mapping = {
        c: palette[i % len(palette)]
        for i, c in enumerate(pd.Series(data["cluster_dropoff"]).unique())
    }
    data["color"] = data["cluster_dropoff"].map(color_mapping)
    plot_data = data.sample(min(50_000, len(data)), random_state=RSEED)
except Exception:
    plot_data = data.head(0)



## === cell 17
try:
    BB = BB_zoom
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    if len(plot_data) > 0:
        for b, df_ in plot_data.groupby("cluster_dropoff"):
            axs.scatter(
                df_.dropoff_longitude,
                df_.dropoff_latitude,
                alpha=0.2,
                c=[color_mapping[b]],
                s=10,
                label=f"{b}",
            )
        axs.set_xlim((BB[0], BB[1]))
        axs.set_ylim((BB[2], BB[3]))
        axs.set_title("Dropoff locations")
        axs.axis("off")
        axs.legend()
    plt.show()
except Exception:
    pass




## === cell 18
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)


R = 6378.0


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km


place = pd.DataFrame(
    {
        "loc": ["jfk", "nyc", "ewr", "lgr"],
        "long": [-73.7822222222, -74.0063889, -74.175, -73.87],
        "lat": [40.6441666667, 40.7141667, 40.69, 40.77],
    }
)


def distance_to_place(df, location, source_long, source_lat):
    selected_place = place[place["loc"] == location].reset_index(drop=True)
    xx = haversine_np(
        df[source_long],
        df[source_lat],
        selected_place.loc[0, "long"],
        selected_place.loc[0, "lat"],
    )
    return xx


def calculate_direction(df):
    d_lon = df["pickup_longitude"] - df["dropoff_longitude"]
    d_lat = df["pickup_latitude"] - df["dropoff_latitude"]
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    l = np.where(l == 0, 1e-12, l)
    result[d_lon > 0] = (180 / np.pi) * np.arcsin(d_lat[d_lon > 0] / l[d_lon > 0])
    idx = (d_lon < 0) & (d_lat > 0)
    result[idx] = 180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
    idx = (d_lon < 0) & (d_lat < 0)
    result[idx] = -180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
    return result




## === cell 19
data["abs_lat_diff"] = (data["dropoff_latitude"] - data["pickup_latitude"]).abs()
data["abs_lon_diff"] = (data["dropoff_longitude"] - data["pickup_longitude"]).abs()
data["manhattan"] = minkowski_distance(
    data["pickup_longitude"],
    data["dropoff_longitude"],
    data["pickup_latitude"],
    data["dropoff_latitude"],
    1,
)
data["euclidean"] = minkowski_distance(
    data["pickup_longitude"],
    data["dropoff_longitude"],
    data["pickup_latitude"],
    data["dropoff_latitude"],
    2,
)
data["haversine"] = haversine_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
)
for i in place["loc"].tolist():
    for j in ["pickup", "dropoff"]:
        data[f"{j}_distance_to{i}"] = distance_to_place(
            data, i, f"{j}_longitude", f"{j}_latitude"
        )
data["direction"] = calculate_direction(data)



## === cell 20
data["haversine"].describe()



## === cell 21
try:
    cek = data[data["haversine"] < 200]
    sns.histplot(cek["haversine"], bins=100)
    plt.show()
except Exception:
    pass



## === cell 22
(data["haversine"] > 25).sum()



## === cell 23
try:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].scatter(data.haversine, data.fare_amount, alpha=0.2)
    axs[0].set_xlabel("distance km")
    axs[0].set_ylabel("fare $USD")
    axs[0].set_title("All data")

    idx = (data.haversine <= 25) & (data.fare_amount < 100)
    axs[1].scatter(data[idx].haversine, data[idx].fare_amount, alpha=0.2)
    axs[1].set_xlabel("distance km")
    axs[1].set_ylabel("fare $USD")
    axs[1].set_title("Zoom in on distance < 25 km, fare < $100")
    plt.show()
except Exception:
    pass



## === cell 24
data[(data["abs_lat_diff"] == 0) & (data["abs_lon_diff"] == 0)].shape[0], data.shape[
    0
], data[(data["abs_lat_diff"] == 0) & (data["abs_lon_diff"] == 0)].shape[
    0
] / data.shape[
    0
]



## === cell 25
try:
    sns.histplot(
        data[(data["abs_lat_diff"] == 0) & (data["abs_lon_diff"] == 0)][
            "passenger_count"
        ]
    )
    plt.show()
    sns.histplot(
        data[(data["abs_lat_diff"] == 0) & (data["abs_lon_diff"] == 0)]["fare_amount"]
    )
    plt.show()
except Exception:
    pass



## === cell 26
try:
    corrs = data.corr(numeric_only=True)
    corrs = corrs.drop("fare_amount", axis=0)
    corrs["fare_amount"].plot.bar(color="b")
    plt.title("Correlation with Fare Amount")
    plt.show()
except Exception:
    corrs = pd.DataFrame()



## === cell 27
cek_cols = corrs.columns.tolist() if not corrs.empty else []



## === cell 28
try:
    cek = data.copy()
    for i in cek_cols:
        if (cek[i] > 0).all():
            cek[i] = np.log(cek[i])
    corrs2 = cek.corr(numeric_only=True)
    corrs2 = corrs2.drop("fare_amount", axis=0)
    corrs2["fare_amount"].plot.bar(color="b")
    plt.title("Correlation with Fare Amount (some log features)")
    plt.show()
except Exception:
    pass




## === cell 29
def extract_dateinfo(
    df,
    date_col,
    drop=True,
    time=False,
    start_ref=pd.Timestamp(1900, 1, 1),
    extra_attr=False,
):
    """
    Extract Date (and time) Information from a DataFrame.
    Adapted from fastai structured.py approach.

    BUGFIX: pandas removed .dt.week; use .dt.isocalendar().week instead.
    """
    df = df.copy()
    fld = df[date_col]

    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    pre = re.sub("[Dd]ate", "", date_col)
    pre = re.sub("[Tt]ime", "", pre)

    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Days_in_month",
        "is_leap_year",
    ]
    if extra_attr:
        attr = attr + [
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ]
    if time:
        attr = attr + ["Hour", "Minute", "Second"]

    for n in attr:
        n_low = n.lower()
        if n_low == "week":
            df[pre + n] = fld.dt.isocalendar().week.astype("int16")
        else:
            df[pre + n] = getattr(fld.dt, n_low)

    df[pre + "Days_in_year"] = df[pre + "is_leap_year"].astype(int) + 365

    if time:
        df[pre + "frac_day"] = (
            (df[pre + "Hour"]) + (df[pre + "Minute"] / 60) + (df[pre + "Second"] / 3600)
        ) / 24
        df[pre + "frac_week"] = (df[pre + "Dayofweek"] + df[pre + "frac_day"]) / 7
        df[pre + "frac_month"] = (df[pre + "Day"] + (df[pre + "frac_day"])) / (
            df[pre + "Days_in_month"] + 1
        )
        df[pre + "frac_year"] = (df[pre + "Dayofyear"] + df[pre + "frac_day"]) / (
            df[pre + "Days_in_year"] + 1
        )

    df[pre + "Elapsed"] = (fld - start_ref).dt.total_seconds()

    if drop:
        df = df.drop(date_col, axis=1)

    return df


try:
    data = extract_dateinfo(
        data,
        "pickup_datetime",
        drop=False,
        time=True,
        start_ref=data["pickup_datetime"].min(),
    )
except Exception:
    pass




## === cell 30
def time_slicer(df, timeframes, value, color="purple"):
    f, ax = plt.subplots(len(timeframes), figsize=[12, 12])
    for i, x in enumerate(timeframes):
        df.loc[:, [x, value]].groupby([x]).mean().plot(ax=ax[i], color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title(
            f"{value.replace('_',' ').title()} by {x.replace('_',' ').title()}"
        )
        ax[i].set_xlabel("")
    ax[len(timeframes) - 1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)




## === cell 31
try:
    time_slicer(
        df=data,
        timeframes=["pickup_Year", "pickup_Month", "pickup_Day", "pickup_Hour"],
        value="fare_amount",
        color="blue",
    )
    plt.show()
except Exception:
    pass



## === cell 32
try:
    holidays = calendar().holidays()
    data["usFedHoliday"] = (
        pd.to_datetime(data["pickup_datetime"].dt.date)
        .astype("datetime64[ns]")
        .isin(holidays)
    )
except Exception:
    data["usFedHoliday"] = False



## === cell 33
try:
    cek = data[
        (data.haversine <= 25) & (data.fare_amount >= 0) & (data.fare_amount <= 50)
    ]
    sns.scatterplot(x="haversine", y="fare_amount", data=cek, hue="usFedHoliday")
    plt.show()
except Exception:
    pass



## === cell 34
pass




## === cell 35
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)


R = 6378.0


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


place = pd.DataFrame(
    {
        "loc": ["jfk", "nyc", "ewr", "lgr"],
        "long": [-73.7822222222, -74.0063889, -74.175, -73.87],
        "lat": [40.6441666667, 40.7141667, 40.69, 40.77],
    }
)


def distance_to_place(df, location, source_long, source_lat):
    selected_place = place[place["loc"] == location].reset_index(drop=True)
    return haversine_np(
        df[source_long],
        df[source_lat],
        selected_place.loc[0, "long"],
        selected_place.loc[0, "lat"],
    )


def calculate_direction(df):
    d_lon = df["pickup_longitude"] - df["dropoff_longitude"]
    d_lat = df["pickup_latitude"] - df["dropoff_latitude"]
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    l = np.where(l == 0, 1e-12, l)
    result[d_lon > 0] = (180 / np.pi) * np.arcsin(d_lat[d_lon > 0] / l[d_lon > 0])
    idx = (d_lon < 0) & (d_lat > 0)
    result[idx] = 180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
    idx = (d_lon < 0) & (d_lat < 0)
    result[idx] = -180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
    return result




## === cell 36
def extract_dateinfo(
    df,
    date_col,
    drop=True,
    time=False,
    start_ref=pd.Timestamp(1900, 1, 1),
    extra_attr=False,
):
    df = df.copy()
    fld = df[date_col]

    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    pre = re.sub("[Dd]ate", "", date_col)
    pre = re.sub("[Tt]ime", "", pre)

    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Days_in_month",
        "is_leap_year",
    ]
    if extra_attr:
        attr = attr + [
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ]
    if time:
        attr = attr + ["Hour", "Minute", "Second"]

    for n in attr:
        n_low = n.lower()
        if n_low == "week":
            df[pre + n] = fld.dt.isocalendar().week.astype("int16")
        else:
            df[pre + n] = getattr(fld.dt, n_low)

    df[pre + "Days_in_year"] = df[pre + "is_leap_year"].astype(int) + 365

    if time:
        df[pre + "frac_day"] = (
            (df[pre + "Hour"]) + (df[pre + "Minute"] / 60) + (df[pre + "Second"] / 3600)
        ) / 24
        df[pre + "frac_week"] = (df[pre + "Dayofweek"] + df[pre + "frac_day"]) / 7
        df[pre + "frac_month"] = (df[pre + "Day"] + (df[pre + "frac_day"])) / (
            df[pre + "Days_in_month"] + 1
        )
        df[pre + "frac_year"] = (df[pre + "Dayofyear"] + df[pre + "frac_day"]) / (
            df[pre + "Days_in_year"] + 1
        )

    df[pre + "Elapsed"] = (fld - start_ref).dt.total_seconds()

    if drop:
        df = df.drop(date_col, axis=1)
    return df




## === cell 37
from sklearn.base import BaseEstimator, TransformerMixin


class data_transform(BaseEstimator, TransformerMixin):
    def __init__(self, num, cat, is_cat):
        self.num = num
        self.cat = cat
        self.is_cat = is_cat

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        num_cols = self.num
        cat_cols = self.cat
        df = X.copy()

        df["passenger_count"] = np.where(
            df["passenger_count"] < 1, 1, df["passenger_count"]
        )
        df["passenger_count"] = np.where(
            df["passenger_count"] > 6, 5, df["passenger_count"]
        )
        df["pickup_latitude"] = np.where(
            df["pickup_latitude"] < 40, 40, df["pickup_latitude"]
        )
        df["pickup_latitude"] = np.where(
            df["pickup_latitude"] > 42, 42, df["pickup_latitude"]
        )
        df["dropoff_latitude"] = np.where(
            df["dropoff_latitude"] < 40, 40, df["dropoff_latitude"]
        )
        df["dropoff_latitude"] = np.where(
            df["dropoff_latitude"] > 42, 42, df["dropoff_latitude"]
        )
        df["pickup_longitude"] = np.where(
            df["pickup_longitude"] < -75, -75, df["pickup_longitude"]
        )
        df["pickup_longitude"] = np.where(
            df["pickup_longitude"] > -72, -72, df["pickup_longitude"]
        )
        df["dropoff_longitude"] = np.where(
            df["dropoff_longitude"] < -75, -75, df["dropoff_longitude"]
        )
        df["dropoff_longitude"] = np.where(
            df["dropoff_longitude"] > -72, -72, df["dropoff_longitude"]
        )

        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

        df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
        df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
        df["manhattan"] = minkowski_distance(
            df["pickup_longitude"],
            df["dropoff_longitude"],
            df["pickup_latitude"],
            df["dropoff_latitude"],
            1,
        )
        df["euclidean"] = minkowski_distance(
            df["pickup_longitude"],
            df["dropoff_longitude"],
            df["pickup_latitude"],
            df["dropoff_latitude"],
            2,
        )
        df["haversine"] = haversine_np(
            df["pickup_longitude"],
            df["pickup_latitude"],
            df["dropoff_longitude"],
            df["dropoff_latitude"],
        )

        df["haversine"] = np.clip(df["haversine"], 0, 50)

        for i in place["loc"].tolist():
            for j in ["pickup", "dropoff"]:
                df[f"{j}_distance_to{i}"] = distance_to_place(
                    df, i, f"{j}_longitude", f"{j}_latitude"
                )
        df["direction"] = calculate_direction(df)

        start_ref = df["pickup_datetime"].min()
        if pd.isna(start_ref):
            start_ref = pd.Timestamp(1900, 1, 1)

        df = extract_dateinfo(
            df,
            "pickup_datetime",
            drop=False,
            time=True,
            start_ref=start_ref,
        )

        holidays = calendar().holidays()
        df["usFedHoliday"] = (
            pd.to_datetime(df["pickup_datetime"].dt.date)
            .astype("datetime64[ns]")
            .isin(holidays)
        )

        df[cat_cols] = df[cat_cols].astype(str)

        if self.is_cat == 1:
            return df[cat_cols]
        elif self.is_cat == 0:
            return df[num_cols]
        else:
            return df




## === cell 38
data_transform_2 = data_transform




## === cell 39
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 40
ori_cols = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 41
num_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan",
    "euclidean",
    "haversine",
    "pickup_distance_tojfk",
    "dropoff_distance_tojfk",
    "pickup_distance_tonyc",
    "dropoff_distance_tonyc",
    "pickup_distance_toewr",
    "dropoff_distance_toewr",
    "pickup_distance_tolgr",
    "dropoff_distance_tolgr",
    "direction",
    "pickup_Year",
    "pickup_Month",
    "pickup_Week",
    "pickup_Day",
    "pickup_Dayofweek",
    "pickup_Dayofyear",
    "pickup_Days_in_month",
    "pickup_Hour",
    "pickup_Minute",
    "pickup_Second",
    "pickup_Days_in_year",
    "pickup_frac_day",
    "pickup_frac_week",
    "pickup_frac_month",
    "pickup_frac_year",
    "pickup_Elapsed",
]
cat_cols = ["pickup_is_leap_year", "usFedHoliday"]
target = "fare_amount"



## === cell 42
data = data.loc[data["pickup_latitude"].between(40, 42)]
data = data.loc[data["pickup_longitude"].between(-75, -72)]
data = data.loc[data["dropoff_latitude"].between(40, 42)]
data = data.loc[data["dropoff_longitude"].between(-75, -72)]
data = data.loc[data["fare_amount"].between(0, 250)]



## === cell 43
data = data.sort_values("pickup_datetime").reset_index(drop=True)
split_idx = int(len(data) * 0.9)
train = data.iloc[:split_idx].copy()
val = data.iloc[split_idx:].copy()



## === cell 44
from sklearn.model_selection import KFold

num_transformer = Pipeline(
    steps=[
        ("dataprep", data_transform(num_cols, cat_cols, is_cat=0)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", PowerTransformer()),
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("dataprep", data_transform(num_cols, cat_cols, is_cat=1)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

transformer = ColumnTransformer(
    transformers=[
        ("num", num_transformer, ori_cols),
        ("cat", cat_transformer, ori_cols),
    ]
)

knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=123)
rf = DecisionTreeRegressor(random_state=123)  # kept as in original (even if redundant)
eln = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=2020)
rg = Ridge(alpha=1.0, random_state=2020)
ls = Lasso(alpha=1.0, random_state=2020)
xgb = XGBRegressor(
    random_state=2020, booster="gbtree", n_estimators=20, tree_method="hist"
)
lgb = LGBMRegressor(
    objective="regression",
    random_state=2020,
    metric="rmse",
    num_leaves=31,
    boosting_type="gbdt",
    max_depth=5,
    learning_rate=0.034,
)
catb = CatBoostRegressor(iterations=2, learning_rate=0.5, depth=3, silent=True)
lr = LinearRegression()

cv_partition = KFold(n_splits=3, shuffle=True, random_state=RSEED)

stack = StackingCVRegressor(
    regressors=(knn, lgb, xgb, catb, dt, rf, eln, rg, ls),
    meta_regressor=lr,
    cv=cv_partition,
)

stack_pipeline = Pipeline(steps=[("transformer", transformer), ("stack", stack)])

main_pipeline = TransformedTargetRegressor(
    regressor=stack_pipeline, transformer=PowerTransformer()
)



## === cell 45
train_X = train[ori_cols]
train_y = train[target]
main_pipeline.fit(train_X, train_y)



## === cell 46
val_pred = main_pipeline.predict(val[ori_cols])
print("RMSE (validation): %.4f" % rmse(val[target], val_pred))



## === cell 47
train_pred = main_pipeline.predict(train_X)
print("RMSE (train): %.4f" % rmse(train_y, train_pred))



## === cell 48
test_path = _resolve_path(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "../input/new-york-city-taxi-fare-prediction/test.csv",
)
test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])



## === cell 49
test_X = test[ori_cols]
predicted_fare = main_pipeline.predict(test_X)
predicted_fare = np.asarray(predicted_fare)

predicted_fare = np.nan_to_num(predicted_fare, nan=0.0, posinf=500.0, neginf=0.0)
predicted_fare = np.clip(predicted_fare, 0, 500)
print(predicted_fare[:10])



## === cell 50
my_submission = pd.DataFrame({"key": test["key"], "fare_amount": predicted_fare})
my_submission.to_csv("submission_v1a.csv", index=False)
print("Wrote submission_v1a.csv with shape:", my_submission.shape)



## === cell 51
import joblib

joblib.dump(main_pipeline, "nyk_taxi_stack.pkl")



## === cell 52
main_pipeline



## === cell 53
dt_ = "2019-06-20 12:26:21"
plong = -73.844
plat = 41.721
dlong = -74.842
dlat = 42.712
pc = 5



## === cell 54
cek = pd.DataFrame(
    {
        "pickup_datetime": [dt_],
        "pickup_longitude": [plong],
        "pickup_latitude": [plat],
        "dropoff_longitude": [dlong],
        "dropoff_latitude": [dlat],
        "passenger_count": [pc],
    }
)
main_pipeline.predict(cek[ori_cols])
