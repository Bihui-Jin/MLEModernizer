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

1061.13216

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            
BB = (-74.5, -72.8, 40.5, 41.8)
nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png')

BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2103626039.py in <cell line: 0>()
      8 # load image of NYC map
      9 BB = (-74.5, -72.8, 40.5, 41.8)
---> 10 nyc_map = plt.imread('https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png')
     11 
     12 # load extra image to zoom in on NYC

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in imread(fname, format)
   2193 @_copy_docstring_and_deprecators(matplotlib.image.imread)
   2194 def imread(fname, format=None):
-> 2195     return matplotlib.image.imread(fname, format)
   2196 
   2197 

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1556     if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
   1557         # Pillow doesn't handle URLs directly.
-> 1558         raise ValueError(
   1559             "Please open the URL for reading and pass the "
   1560             "result to Pillow, e.g. with "

ValueError: Please open the URL for reading and pass the result to Pillow, e.g. with ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.

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


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1384628805.py in <cell line: 0>()
      1 # plot training data on map
----> 2 plot_on_map(train, BB, nyc_map, s=1, alpha=0.3)
      3 plot_on_map(train, BB_zoom, nyc_map_zoom, s=1, alpha=0.3)

NameError: name 'nyc_map' is not defined

## === cell 14
nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9

plt.figure(figsize=(8,8))
plt.imshow(nyc_map, zorder=0)
plt.imshow(nyc_mask, zorder=1, alpha=0.7); # note: True is show in black, False in white.


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3600756278.py in <cell line: 0>()
      1 # read nyc mask and turn into boolean map with
      2 # land = True, water = False
----> 3 nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
      4 
      5 plt.figure(figsize=(8,8))

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in imread(fname, format)
   2193 @_copy_docstring_and_deprecators(matplotlib.image.imread)
   2194 def imread(fname, format=None):
-> 2195     return matplotlib.image.imread(fname, format)
   2196 
   2197 

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1556     if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
   1557         # Pillow doesn't handle URLs directly.
-> 1558         raise ValueError(
   1559             "Please open the URL for reading and pass the "
   1560             "result to Pillow, e.g. with "

ValueError: Please open the URL for reading and pass the result to Pillow, e.g. with ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.

## === cell 15
def lonlat_to_xy(longitude, latitude, dx, dy, BB):
    return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
           (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')


## === cell 16
pickup_x, pickup_y = lonlat_to_xy(train.pickup_longitude, train.pickup_latitude, 
                                  nyc_mask.shape[1], nyc_mask.shape[0], BB)
dropoff_x, dropoff_y = lonlat_to_xy(train.dropoff_longitude, train.dropoff_latitude, 
                                  nyc_mask.shape[1], nyc_mask.shape[0], BB)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3590223791.py in <cell line: 0>()
      1 pickup_x, pickup_y = lonlat_to_xy(train.pickup_longitude, train.pickup_latitude, 
----> 2                                   nyc_mask.shape[1], nyc_mask.shape[0], BB)
      3 dropoff_x, dropoff_y = lonlat_to_xy(train.dropoff_longitude, train.dropoff_latitude, 
      4                                   nyc_mask.shape[1], nyc_mask.shape[0], BB)

NameError: name 'nyc_mask' is not defined

## === cell 17
idx = (nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x])
print("Number of trips in water: {}".format(np.sum(~idx)))


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/725073149.py in <cell line: 0>()
----> 1 idx = (nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x])
      2 print("Number of trips in water: {}".format(np.sum(~idx)))

NameError: name 'nyc_mask' is not defined

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
train = remove_datapoints_from_water(train)
print('Number of rows {:,}'.format(len(train)))


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1399932495.py in <cell line: 0>()
----> 1 train = remove_datapoints_from_water(train)
      2 print('Number of rows {:,}'.format(len(train)))

/tmp/ipykernel_11/805853527.py in remove_datapoints_from_water(df)
      9     # read nyc mask and turn into boolean map with
     10     # land = True, water = False
---> 11     nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
     12 
     13     # calculate for each lon,lat coordinate the xy coordinate in the mask map

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in imread(fname, format)
   2193 @_copy_docstring_and_deprecators(matplotlib.image.imread)
   2194 def imread(fname, format=None):
-> 2195     return matplotlib.image.imread(fname, format)
   2196 
   2197 

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1556     if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
   1557         # Pillow doesn't handle URLs directly.
-> 1558         raise ValueError(
   1559             "Please open the URL for reading and pass the "
   1560             "result to Pillow, e.g. with "

ValueError: Please open the URL for reading and pass the result to Pillow, e.g. with ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


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


## === cell 35
XTEST = test[features].values

filename = './baseline_linear'

y_pred_final = model_lin.predict(XTEST)

submission = pd.DataFrame(
    {'key': test.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)


## === cell 36
submission.head()


## === cell 37
def rmse_Keras(y_values, pred):
    return K.sqrt(K.mean(K.square(pred - y_values), axis=-1)) 

def baseline_model():
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='uniform', activation='softplus'))
    model.add(Dense(1, kernel_initializer='uniform', activation = 'relu'))    
    
    
    model.compile(loss='mse', optimizer='Adam', metrics=[rmse_Keras])
    return model
