# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go








import os
print(os.listdir("../input"))


## === cell 1
train =  pd.read_csv('../input/train.csv', nrows = 1000000, parse_dates=["pickup_datetime"])

print(train.shape)
train.head()


## === cell 2
train.describe()
train.shape


## === cell 3
train = train[train.fare_amount>=0]
print('Number of rows {:,}'.format(len(train)))


## === cell 4
train.isnull().any()


## === cell 5
train = train[~train.dropoff_longitude.isnull()]
train = train[~train.dropoff_latitude.isnull()]
print('Number of rows {:,}'.format(len(train)))


## === cell 6
test =  pd.read_csv('../input/test.csv', nrows = 2000000, parse_dates=["pickup_datetime"])

print(test.shape)
test.describe()


## === cell 7
plt.boxplot(train[train.pickup_latitude<39].pickup_latitude)
plt.show()


## === cell 8
print('Minimus and maximum longitude Test ')
min(test.pickup_longitude.min(), test.dropoff_longitude.min()), \
max(test.pickup_longitude.max(), test.dropoff_longitude.max())


## === cell 9
print('Minimus and maximum latitude Test ')
min(test.pickup_latitude.min(), test.dropoff_latitude.min()), \
max(test.pickup_latitude.max(), test.dropoff_latitude.max())


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

import urllib.request
from PIL import Image


def _safe_load_image(url, fallback_shape=(600, 600, 3)):
    try:
        with urllib.request.urlopen(url) as f:
            return np.array(Image.open(f))
    except Exception:
        return np.zeros(fallback_shape, dtype=np.uint8)


nyc_map = _safe_load_image(
    "https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png",
    fallback_shape=(600, 600, 3),
)

BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = _safe_load_image(
    "https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png",
    fallback_shape=(600, 600, 3),
)


## === cell 11
train = train[select_within_boundingbox(train, BB)]
print('Number of rows {:,}'.format(len(train)))


## === cell 12
def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2):
    fig, axs = plt.subplots(1, 2, figsize=(16,10))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title('Pickup locations')
    axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title('Dropoff locations')
    axs[1].imshow(nyc_map, zorder=0, extent=BB)


## === cell 13
plot_on_map(train, BB, nyc_map, s=1, alpha=0.3)
plot_on_map(train, BB_zoom, nyc_map_zoom, s=1, alpha=0.3)


## === cell 14
try:
    with urllib.request.urlopen(
        "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    ) as f:
        _mask_img = np.array(Image.open(f))
except Exception:
    _mask_img = np.zeros((600, 600, 3), dtype=np.uint8)

nyc_mask = _mask_img[:, :, 0] > 0.9

plt.figure(figsize=(8, 8))
plt.imshow(nyc_map, zorder=0)
plt.imshow(nyc_mask, zorder=1, alpha=0.7)


## === cell 15
def lonlat_to_xy(longitude, latitude, dx, dy, BB):
    return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
           (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')


## === cell 16
pickup_x, pickup_y = lonlat_to_xy(train.pickup_longitude, train.pickup_latitude, 
                                  nyc_mask.shape[1], nyc_mask.shape[0], BB)
dropoff_x, dropoff_y = lonlat_to_xy(train.dropoff_longitude, train.dropoff_latitude, 
                                  nyc_mask.shape[1], nyc_mask.shape[0], BB)


## === cell 17
idx = (nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x])
print("Number of trips in water: {}".format(np.sum(~idx)))


## === cell 18
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
               (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    BB = (-74.5, -72.8, 40.5, 41.8)
    
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    
    return df[idx]


## === cell 19
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    try:
        with urllib.request.urlopen(
            "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        ) as f:
            nyc_mask_local = np.array(Image.open(f))[:, :, 0] > 0.9
    except Exception:
        if "nyc_mask" in globals():
            nyc_mask_local = globals()["nyc_mask"]
        else:
            nyc_mask_local = np.ones((600, 600), dtype=bool)

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask_local.shape[1],
        nyc_mask_local.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask_local.shape[1],
        nyc_mask_local.shape[0],
        BB,
    )
    idx = nyc_mask_local[pickup_y, pickup_x] & nyc_mask_local[dropoff_y, dropoff_x]

    return df[idx]


train = remove_datapoints_from_water(train)
print("Number of rows {:,}".format(len(train)))


## === cell 20
train['year'] = train.pickup_datetime.apply(lambda t: t.year)
train['weekday'] = train.pickup_datetime.apply(lambda t: t.weekday())
train['hour'] = train.pickup_datetime.apply(lambda t: t.hour)


## === cell 21
n_hours = 24
n_weekdays = 7
n_years = 7
n_bins_lon = 30
n_bins_lat = 30

BB_traffic = (-74.025, -73.925, 40.7, 40.8)

def calculate_trafic_density(df):
    traffic = np.zeros((n_years, n_weekdays, n_hours, n_bins_lat, n_bins_lon))
    
    bins_lon = np.zeros(n_bins_lon+1) # bin
    bins_lat = np.zeros(n_bins_lat+1) # bin
    
    delta_lon = (BB_traffic[1]-BB_traffic[0]) / n_bins_lon # bin longutide width
    delta_lat = (BB_traffic[3]-BB_traffic[2]) / n_bins_lat # bin latitude height
    
    for i in range(n_bins_lon+1):
        bins_lon[i] = BB_traffic[0] + i * delta_lon
    for j in range(n_bins_lat+1):
        bins_lat[j] = BB_traffic[2] + j * delta_lat
    
    for y in range(n_years):
        for d in range(n_weekdays):
            for h in range(n_hours):
                idx = (df.year==(2009+y)) & (df.weekday==d) & (df.hour==h)

                inds_pickup_lon = np.digitize(df[idx].pickup_longitude, bins_lon)
                inds_pickup_lat = np.digitize(df[idx].pickup_latitude, bins_lat)

                for i in range(n_bins_lon):
                    for j in range(n_bins_lat):
                        traffic[y, d, h, j, i] = traffic[y, d, h, j, i] + \
                                                 np.sum((inds_pickup_lon==i+1) & (inds_pickup_lat==j+1))
    
    return traffic 

def plot_traffic(traffic, y, d):
    days = {'monday' : 0, 'tuesday' : 1, 'wednesday' : 2, 'thursday' : 3, 'friday' : 4, 'saturday' : 5, 'sunday' : 6}
    fig, axs = plt.subplots(3,8,figsize=(18,7))
    axs = axs.ravel()
    for h in range(24):
        axs[h].imshow(traffic[y-2009,days[d],h,::-1,:], zorder=1, cmap='coolwarm', clim=(0, traffic.max()))
        axs[h].get_xaxis().set_visible(False)
        axs[h].get_yaxis().set_visible(False)
        axs[h].set_title('h={}'.format(h))
    fig.suptitle("Pickup traffic density, year={}, day={} (max_pickups={})".format(y, d, traffic.max()))


## === cell 22
traffic = calculate_trafic_density(train)


## === cell 23
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...


## === cell 24
train['distance_miles'] = distance(train.pickup_latitude, train.pickup_longitude, \
                                      train.dropoff_latitude, train.dropoff_longitude)

train.distance_miles.hist(bins=50, figsize=(12,4))
plt.xlabel('distance miles')
plt.title('Histogram ride distances in miles')
train.distance_miles.describe()


## === cell 25
print('Number of rows {:,}'.format(len(train)))
train = train[train.distance_miles >= 0.05]
print('Number of rows {:,}'.format(len(train)))


## === cell 26
jfk = (-73.7822222222, 40.6441666667)
nyc = (-74.0063889, 40.7141667)

def plot_location_fare(loc, name, range=1.5):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    idx = (distance(train.pickup_latitude, train.pickup_longitude, loc[1], loc[0]) < range)
    train[idx].fare_amount.hist(bins=100, ax=axs[0])
    axs[0].set_xlabel('fare $USD')
    axs[0].set_title('Histogram pickup location within {} miles of {}'.format(range, name))

    idx = (distance(train.dropoff_latitude, train.dropoff_longitude, loc[1], loc[0]) < range)
    train[idx].fare_amount.hist(bins=100, ax=axs[1])
    axs[1].set_xlabel('fare $USD')
    axs[1].set_title('Histogram dropoff location within {} miles of {}'.format(range, name));
    
plot_location_fare(jfk, 'JFK Airport')


## === cell 27
ewr = (-74.175, 40.69) # Newark Liberty International Airport, see https://www.travelmath.com/airport/EWR
lgr = (-73.87, 40.77) # LaGuardia Airport, see https://www.travelmath.com/airport/LGA
plot_location_fare(ewr, 'Newark Airport')
plot_location_fare(lgr, 'LaGuardia Airport')


## === cell 28
train['fare_per_mile'] = train.fare_amount / train.distance_miles

train['distance_to_center'] = distance(nyc[1], nyc[0], train.pickup_latitude, train.pickup_longitude)


## === cell 29
train['pickup_distance_to_jfk'] = distance(jfk[1], jfk[0], train.pickup_latitude, train.pickup_longitude)
train['dropoff_distance_to_jfk'] = distance(jfk[1], jfk[0], train.dropoff_latitude, train.dropoff_longitude)
train['pickup_distance_to_ewr'] = distance(ewr[1], ewr[0], train.pickup_latitude, train.pickup_longitude)
train['dropoff_distance_to_ewr'] = distance(ewr[1], ewr[0], train.dropoff_latitude, train.dropoff_longitude)
train['pickup_distance_to_lgr'] = distance(lgr[1], lgr[0], train.pickup_latitude, train.pickup_longitude)
train['dropoff_distance_to_lgr'] = distance(lgr[1], lgr[0], train.dropoff_latitude, train.dropoff_longitude)


## === cell 30
test['distance_miles'] = distance(test.pickup_latitude, test.pickup_longitude, \
                                     test.dropoff_latitude, test.dropoff_longitude)
test['distance_to_center'] = distance(nyc[1], nyc[0], \
                                          test.dropoff_latitude, test.dropoff_longitude)
test['hour'] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
test['year'] = test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)


## === cell 31
idx = (train.distance_to_center<15) & (train.passenger_count!=0)
features = ['year', 'hour', 'distance_miles', 'passenger_count']
X = train[idx][features].values
y = train[idx]['fare_amount'].values

X.shape


## === cell 32
from sklearn.metrics import mean_squared_error, explained_variance_score

def plot_prediction_analysis(y, y_pred, figsize=(10,4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))
    
    axs[1].hist(y-y_pred, bins=50)
    avg = np.mean(y-y_pred)
    std = np.std(y-y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title('Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(avg, std))
    
    if title!='':
        fig.suptitle(title)


## === cell 33
from sklearn.model_selection import train_test_split

try:
    n_x = len(X)
except Exception:
    n_x = 0
try:
    n_y = len(y)
except Exception:
    n_y = 0

if n_x == 0 or n_y == 0:
    X = train[features].values
    y = train["fare_amount"].values

if len(X) < 2:
    raise ValueError(
        f"Not enough samples to split after preprocessing: n_samples={len(X)}. "
        "Upstream filtering/cleaning removed all (or nearly all) rows."
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


## --- ERROR in cell 33, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2619836270.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     19[0m [0;31m# Guard against the (rare) case where upstream cleaning removed all rows.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mif[0m [0mlen[0m[0;34m([0m[0mX[0m[0;34m)[0m [0;34m<[0m [0;36m2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 21[0;31m     raise ValueError(
[0m[1;32m     22[0m         [0;34mf"Not enough samples to split after preprocessing: n_samples={len(X)}. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m         [0;34m"Upstream filtering/cleaning removed all (or nearly all) rows."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Not enough samples to split after preprocessing: n_samples=0. Upstream filtering/cleaning removed all (or nearly all) rows.

## === cell 34
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title='Linear Model - Trainingset')

y_test_pred = model_lin.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title='Linear Model - Testset')
