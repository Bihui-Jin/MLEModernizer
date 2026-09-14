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

# 5. Target score

3.56643

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.3f' % x)
RSEED = 2020
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 12
import seaborn as sns
palette = sns.color_palette('Paired', 10)
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, PowerTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


## === cell 2
data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 5_000_00, 
                   parse_dates = ['pickup_datetime']).drop(columns = 'key')
data = data.dropna()


## === cell 3
data.head()


## === cell 4
data.describe()


## === cell 5
data.isna().sum()/data.shape[0]*100


## === cell 6
sns.distplot(data['fare_amount'])


## === cell 7
def ecdf(x):
    """Empirical cumulative distribution function of a variable"""
    x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1, 1) / n
    return x, y


## === cell 8
xs, ys = ecdf(data['fare_amount'])
plt.figure(figsize = (8, 6))
plt.plot(xs, ys, '.')
plt.ylabel('Percentile'); plt.title('ECDF of Fare Amount'); plt.xlabel('Fare Amount ($)');


## === cell 9
data['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Passenger Counts'); plt.xlabel('Number of Passengers'); plt.ylabel('Count');


## === cell 10
print('ada '+str(data[data['passenger_count']==0].shape[0])+'transaksi dengan 0 passangger')
print('ada '+str(data[data['passenger_count']==6].shape[0])+'transaksi dengan 6 passangger')


## === cell 11
data = data.loc[data['pickup_latitude'].between(40, 42)]
data = data.loc[data['pickup_longitude'].between(-75, -72)]
data = data.loc[data['dropoff_latitude'].between(40, 42)]
data = data.loc[data['dropoff_longitude'].between(-75, -72)]


## === cell 12
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
cluster_pickup = kmeans.fit_predict(data[['pickup_longitude','pickup_latitude']])
cluster_dropoff = kmeans.fit_predict(data[['dropoff_longitude','dropoff_latitude']])
data['cluster_pickup']=cluster_pickup
data['cluster_dropoff']=cluster_dropoff


## === cell 13
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

BB_zoom = (-74.1, -73.7, 40.6, 40.85)
nyc_map_zoom = plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true')


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2491799034.py in <cell line: 0>()
      8 # load extra image to zoom in on NYC
      9 BB_zoom = (-74.1, -73.7, 40.6, 40.85)
---> 10 nyc_map_zoom = plt.imread('https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/images/nyc_-74.1_-73.7_40.6_40.85.PNG?raw=true')

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

## === cell 14
def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2, color = False):
    fig, axs = plt.subplots(1, 2, figsize=(18, 22))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title('Pickup locations')
    axs[0].axis('off')
    axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c='b', s=s)
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title('Dropoff locations')
    axs[1].axis('off')
    axs[1].imshow(nyc_map, zorder=0, extent=BB)
    
plot_on_map(data.sample(4_000_00, random_state = RSEED), 
            BB_zoom, nyc_map_zoom, s=0.05, alpha=0.05)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1120751736.py in <cell line: 0>()
     18 # plot training data on map zoomed in
     19 plot_on_map(data.sample(4_000_00, random_state = RSEED), 
---> 20             BB_zoom, nyc_map_zoom, s=0.05, alpha=0.05)

NameError: name 'nyc_map_zoom' is not defined

## === cell 15
color_mapping = {cluster_pickup: palette[i] for i, cluster_pickup in enumerate(data['cluster_pickup'].unique())}
data['color'] = data['cluster_pickup'].map(color_mapping)
plot_data = data.sample(4_000_00, random_state = RSEED)


## === cell 16
BB = BB_zoom

fig, axs = plt.subplots(1, 1, figsize=(20, 18))

for b, df in plot_data.groupby('cluster_pickup'):
    axs.scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=0.2, c=df.color, s=30, label = f'{b}')
    axs.set_xlim((BB[0], BB[1]))
    axs.set_ylim((BB[2], BB[3]))
    axs.set_title('Pickup locations', size = 32)
    axs.axis('off')
    
leg = axs.legend(fontsize = 28, markerscale = 3)

for lh in leg.legendHandles: 
    lh.set_alpha(1)

leg.set_title('Cluster Pickup', prop = {'size': 28})

axs.imshow(nyc_map_zoom, zorder=0, extent=BB_zoom);


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2781062376.py in <cell line: 0>()
     22 
     23 # Show map in background (zorder = 0)
---> 24 axs.imshow(nyc_map_zoom, zorder=0, extent=BB_zoom);

NameError: name 'nyc_map_zoom' is not defined

## === cell 17
color_mapping = {cluster_pickup: palette[i] for i, cluster_pickup in enumerate(data['cluster_dropoff'].unique())}
data['color'] = data['cluster_dropoff'].map(color_mapping)
plot_data = data.sample(4_000_00, random_state = RSEED)


## === cell 18
fig, axs = plt.subplots(1, 1, figsize=(20, 18))

for b, df in plot_data.groupby('cluster_dropoff'):
    axs.scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, 
                alpha=0.2, c=df.color, s=30, label = f'{b}')
    axs.set_xlim((BB[0], BB[1]))
    axs.set_ylim((BB[2], BB[3]))
    axs.set_title('cluster_dropoff', size = 32)
    axs.axis('off')
    
leg = axs.legend(fontsize = 28, markerscale = 3)

for lh in leg.legendHandles: 
    lh.set_alpha(1)

leg.set_title('Cluster Dropoff', prop = {'size': 28})

axs.imshow(nyc_map_zoom, zorder=0, extent=BB_zoom);


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3753303555.py in <cell line: 0>()
     20 
     21 # Show map in background (zorder = 0)
---> 22 axs.imshow(nyc_map_zoom, zorder=0, extent=BB_zoom);

NameError: name 'nyc_map_zoom' is not defined

## === cell 19
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
                                                           
R = 6378

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    
    
    source: https://stackoverflow.com/a/29546836

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    
    return km
                                                           
                                                           
place = pd.DataFrame({'loc' : ['jfk','nyc','ewr','lgr'], 'long' : [-73.7822222222,-74.0063889,-74.175,-73.87], 'lat' : [40.6441666667,40.7141667,40.69,40.77]})

def distance_to_place(df,location,source_long,source_lat):
    selected_place = place[place['loc']==location]
    selected_place = selected_place.reset_index()
    xx = haversine_np(df[source_long], df[source_lat], selected_place['long'][0], selected_place['lat'][0])
    
    return xx
                                                           

def calculate_direction(df):
    d_lon = df['pickup_longitude'] - df['dropoff_longitude']
    d_lat = df['pickup_latitude'] - df['dropoff_latitude']
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result


## === cell 20
data['abs_lat_diff'] = (data['dropoff_latitude'] - data['pickup_latitude']).abs()
data['abs_lon_diff'] = (data['dropoff_longitude'] - data['pickup_longitude']).abs()

data['manhattan'] = minkowski_distance(data['pickup_longitude'], data['dropoff_longitude'],
                                       data['pickup_latitude'], data['dropoff_latitude'], 1)

data['euclidean'] = minkowski_distance(data['pickup_longitude'], data['dropoff_longitude'],
                                       data['pickup_latitude'], data['dropoff_latitude'], 2)


data['haversine'] =  haversine_np(data['pickup_longitude'], data['pickup_latitude'],
                         data['dropoff_longitude'], data['dropoff_latitude']) 

for i in place['loc'].tolist():
    for j in ['pickup','dropoff']:
        data[str(j)+'_distance_to'+str(i)] = distance_to_place(data,i,str(j)+'_longitude',str(j)+'_latitude')

data['direction'] = calculate_direction(data)


## === cell 21
data['haversine'].describe()


## === cell 22
cek = data[data['haversine']<200]
sns.distplot(cek['haversine'])


## === cell 23
(data['haversine']>25).sum()


## === cell 24
fig, axs = plt.subplots(1, 2, figsize=(16,6))
axs[0].scatter(data.haversine, data.fare_amount, alpha=0.2)
axs[0].set_xlabel('distance km')
axs[0].set_ylabel('fare $USD')
axs[0].set_title('All data')

idx = (data.haversine <= 25) & (data.fare_amount < 100)
axs[1].scatter(data[idx].haversine, data[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('distance km')
axs[1].set_ylabel('fare $USD')
axs[1].set_title('Zoom in on distance < 15 km, fare < $100');


## === cell 25
data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)].shape[0], data.shape[0], data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)].shape[0]/data.shape[0]


## === cell 26
sns.distplot(data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)]['passenger_count'])
plt.show()
sns.distplot(data[(data['abs_lat_diff']==0)&(data['abs_lon_diff']==0)]['fare_amount'])
plt.show()


## === cell 27
corrs = data.corr()
corrs = corrs.drop('fare_amount',axis=0)
corrs['fare_amount'].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
TypeError: float() argument must be a string or a real number, not 'tuple'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2722159088.py in <cell line: 0>()
----> 1 corrs = data.corr()
      2 corrs = corrs.drop('fare_amount',axis=0)
      3 corrs['fare_amount'].plot.bar(color = 'b');
      4 plt.title('Correlation with Fare Amount');

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: setting an array element with a sequence.

## === cell 28
cek_cols = corrs.columns.tolist()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1636681126.py in <cell line: 0>()
----> 1 cek_cols = corrs.columns.tolist()

NameError: name 'corrs' is not defined

## === cell 29
cek = data.copy()
for i in cek_cols:
    cek[i] = np.log(cek[i])
corrs = cek.corr()
corrs = corrs.drop('fare_amount',axis=0)
corrs['fare_amount'].plot.bar(color = 'b');
plt.title('Correlation with Fare Amount');


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/192350461.py in <cell line: 0>()
      1 cek = data.copy()
----> 2 for i in cek_cols:
      3     cek[i] = np.log(cek[i])
      4 corrs = cek.corr()
      5 corrs = corrs.drop('fare_amount',axis=0)

NameError: name 'cek_cols' is not defined

## === cell 30
import re
def extract_dateinfo(df, date_col, drop=True, time=False, 
                     start_ref = pd.datetime(1900, 1, 1),
                     extra_attr = False):
    """
    Extract Date (and time) Information from a DataFrame
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/structured.py
    """
    df = df.copy()
    
    fld = df[date_col]
    
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    

    pre = re.sub('[Dd]ate', '', date_col)
    pre = re.sub('[Tt]ime', '', pre)
    
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']
    
    if extra_attr:
        attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    for n in attr: 
        df[pre + n] = getattr(fld.dt, n.lower())
        
    df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365
        
    if time:
        df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24
        
        df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[pre + 'frac_day']) / 7
    
        df[pre + 'frac_month'] = (df[pre + 'Day'] + (df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] +  1)
        
        df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)
        
    df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()
    
    if drop: 
        df = df.drop(date_col, axis=1)
        
    return df

data = extract_dateinfo(data, 'pickup_datetime', drop = False,time = True, start_ref = df['pickup_datetime'].min())


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3966204272.py in <cell line: 0>()
      2 import re
      3 def extract_dateinfo(df, date_col, drop=True, time=False, 
----> 4                      start_ref = pd.datetime(1900, 1, 1),
      5                      extra_attr = False):
      6     """

AttributeError: module 'pandas' has no attribute 'datetime'

## === cell 31
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,12])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)


## === cell 32
time_slicer(df=data, timeframes=['pickup_Year', 'pickup_Month', 'pickup_Day','pickup_Hour'], value = "fare_amount", color="blue")


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2674235858.py in <cell line: 0>()
----> 1 time_slicer(df=data, timeframes=['pickup_Year', 'pickup_Month', 'pickup_Day','pickup_Hour'], value = "fare_amount", color="blue")

/tmp/ipykernel_11/957338431.py in time_slicer(df, timeframes, value, color)
      5     f, ax = plt.subplots(len(timeframes), figsize = [12,12])
      6     for i,x in enumerate(timeframes):
----> 7         df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
      8         ax[i].set_ylabel(value.replace("_", " ").title())
      9         ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['pickup_Year'] not in index"

## === cell 33
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
holidays = calendar().holidays()
data["usFedHoliday"] =  data.pickup_datetime.dt.date.astype('datetime64').isin(holidays)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/649215016.py in <cell line: 0>()
      1 from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
      2 holidays = calendar().holidays()
----> 3 data["usFedHoliday"] =  data.pickup_datetime.dt.date.astype('datetime64').isin(holidays)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    108             from pandas.core.arrays import DatetimeArray
    109 
--> 110             dta = DatetimeArray._from_sequence(arr, dtype=dtype)
    111             return dta._ndarray
    112 

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _from_sequence(cls, scalars, dtype, copy)
    325     @classmethod
    326     def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False):
--> 327         return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)
    328 
    329     @classmethod

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _from_sequence_not_strict(cls, data, dtype, copy, tz, freq, dayfirst, yearfirst, ambiguous)
    352             tz = timezones.maybe_get_tz(tz)
    353 
--> 354         dtype = _validate_dt64_dtype(dtype)
    355         # if dtype has an embedded tz, capture it
    356         tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _validate_dt64_dtype(dtype)
   2542                 "Please pass in 'datetime64[ns]' instead."
   2543             )
-> 2544             raise ValueError(msg)
   2545 
   2546         if (

ValueError: Passing in 'datetime64' dtype with no precision is not allowed. Please pass in 'datetime64[ns]' instead.

## === cell 34
cek= data[(data.haversine <= 25) & (data.fare_amount >=0) & (data.fare_amount <= 50)]
sns.scatterplot(x='haversine',y='fare_amount',data=cek,hue='usFedHoliday')


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2482636034.py in <cell line: 0>()
      1 cek= data[(data.haversine <= 25) & (data.fare_amount >=0) & (data.fare_amount <= 50)]
----> 2 sns.scatterplot(x='haversine',y='fare_amount',data=cek,hue='usFedHoliday')

/usr/local/lib/python3.11/dist-packages/seaborn/relational.py in scatterplot(data, x, y, hue, size, style, palette, hue_order, hue_norm, sizes, size_order, size_norm, markers, style_order, legend, ax, **kwargs)
    740 
    741     variables = _ScatterPlotter.get_semantics(locals())
--> 742     p = _ScatterPlotter(data=data, variables=variables, legend=legend)
    743 
    744     p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

/usr/local/lib/python3.11/dist-packages/seaborn/relational.py in __init__(self, data, variables, legend)
    536         )
    537 
--> 538         super().__init__(data=data, variables=variables)
    539 
    540         self.legend = legend

/usr/local/lib/python3.11/dist-packages/seaborn/_oldcore.py in __init__(self, data, variables)
    638         # information for numeric axes would be information about log scales.
    639         self._var_ordered = {"x": False, "y": False}  # alt., used DefaultDict
--> 640         self.assign_variables(data, variables)
    641 
    642         for var, cls in self._semantic_mappings.items():

/usr/local/lib/python3.11/dist-packages/seaborn/_oldcore.py in assign_variables(self, data, variables)
    699         else:
    700             self.input_format = "long"
--> 701             plot_data, variables = self._assign_variables_longform(
    702                 data, **variables,
    703             )

/usr/local/lib/python3.11/dist-packages/seaborn/_oldcore.py in _assign_variables_longform(self, data, **kwargs)
    936 
    937                 err = f"Could not interpret value `{val}` for parameter `{key}`"
--> 938                 raise ValueError(err)
    939 
    940             else:

ValueError: Could not interpret value `usFedHoliday` for parameter `hue`

## === cell 35
data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 3_000_0, 
                   parse_dates = ['pickup_datetime']).drop(columns = 'key')

data = data.dropna()


## === cell 36
def minkowski_distance(x1, x2, y1, y2, p):
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
                                                           
R = 6378

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    
    
    source: https://stackoverflow.com/a/29546836

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    
    return km
                                                           
                                                           
place = pd.DataFrame({'loc' : ['jfk','nyc','ewr','lgr'], 'long' : [-73.7822222222,-74.0063889,-74.175,-73.87], 'lat' : [40.6441666667,40.7141667,40.69,40.77]})

def distance_to_place(df,location,source_long,source_lat):
    selected_place = place[place['loc']==location]
    selected_place = selected_place.reset_index()
    xx = haversine_np(df[source_long], df[source_lat], selected_place['long'][0], selected_place['lat'][0])
    
    return xx
                                                           

def calculate_direction(df):
    d_lon = df['pickup_longitude'] - df['dropoff_longitude']
    d_lat = df['pickup_latitude'] - df['dropoff_latitude']
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result


## === cell 37
import re
def extract_dateinfo(df, date_col, drop=True, time=False, 
                     start_ref = pd.datetime(1900, 1, 1),
                     extra_attr = False):
    """
    Extract Date (and time) Information from a DataFrame
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/structured.py
    """
    df = df.copy()
    
    fld = df[date_col]
    
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    

    pre = re.sub('[Dd]ate', '', date_col)
    pre = re.sub('[Tt]ime', '', pre)
    
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Days_in_month', 'is_leap_year']
    
    if extra_attr:
        attr = attr + ['Is_month_end', 'Is_month_start', 'Is_quarter_end', 
                       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    for n in attr: 
        df[pre + n] = getattr(fld.dt, n.lower())
        
    df[pre + 'Days_in_year'] = df[pre + 'is_leap_year'] + 365
        
    if time:
        df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24
        
        df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[pre + 'frac_day']) / 7
    
        df[pre + 'frac_month'] = (df[pre + 'Day'] + (df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] +  1)
        
        df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)
        
    df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()
    
    if drop: 
        df = df.drop(date_col, axis=1)
        
    return df


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/293203160.py in <cell line: 0>()
      2 import re
      3 def extract_dateinfo(df, date_col, drop=True, time=False, 
----> 4                      start_ref = pd.datetime(1900, 1, 1),
      5                      extra_attr = False):
      6     """

AttributeError: module 'pandas' has no attribute 'datetime'

## === cell 38
class data_transform():
    
    
    def __init__(self, num, cat, is_cat):
        self.num = num
        self.cat = cat
        self.is_cat = is_cat
        
    def fit(self, X):
        return self
    
    def transform(self, X, y = None):
        num_cols = self.num
        cat_cols = self.cat
        df = X.copy()
        df['passenger_count'] = np.where(df['passenger_count']<1,1,df['passenger_count'])
        df['passenger_count'] = np.where(df['passenger_count']>6,5,df['passenger_count'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']<40, 40,df['pickup_latitude'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']>42,42,df['pickup_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']<40, 40,df['dropoff_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']>42, 42,df['dropoff_latitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']<-75, -75,df['pickup_longitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']>-72, -72,df['pickup_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']<-75, -75,df['dropoff_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']>-72, -72,df['dropoff_longitude'])

        kmeans = KMeans(n_clusters=5)
        cluster_pickup = kmeans.fit_predict(data[['pickup_longitude','pickup_latitude']])
        cluster_dropoff = kmeans.fit_predict(data[['dropoff_longitude','dropoff_latitude']])
        data['cluster_pickup']=cluster_pickup
        data['cluster_dropoff']=cluster_dropoff
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
        df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()

        df['manhattan'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 1)

        df['euclidean'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 2)

        df['haversine'] =  haversine_np(df['pickup_longitude'], df['pickup_latitude'],
                                 df['dropoff_longitude'], df['dropoff_latitude']) 

        for i in place['loc'].tolist():
            for j in ['pickup','dropoff']:
                df[str(j)+'_distance_to'+str(i)] = distance_to_place(df,i,str(j)+'_longitude',str(j)+'_latitude')

        df['direction'] = calculate_direction(df)

        df = extract_dateinfo(df, 'pickup_datetime', drop = False, 
                                 time = True, start_ref = df['pickup_datetime'].min())
        
        holidays = calendar().holidays()
        data["usFedHoliday"] =  data.pickup_datetime.dt.date.astype('datetime64').isin(holidays)

        df[cat_cols] = df[cat_cols].astype(str)
        
        
        if self.is_cat==1:
            df = df[cat_cols]
        elif self.is_cat==0:
            df = df[num_cols] 
        else:
            df = df
            
        return df
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)


## === cell 39
class data_transform():
    
    
    def __init__(self, num, cat, is_cat):
        self.num = num
        self.cat = cat
        self.is_cat = is_cat
        
    def fit(self, X):
        return self
    
    def transform(self, X, y = None):
        num_cols = self.num
        cat_cols = self.cat
        df = X.copy()
        df['passenger_count'] = np.where(df['passenger_count']<1,1,df['passenger_count'])
        df['passenger_count'] = np.where(df['passenger_count']>6,5,df['passenger_count'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']<40, 40,df['pickup_latitude'])
        df['pickup_latitude'] = np.where(df['pickup_latitude']>42,42,df['pickup_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']<40, 40,df['dropoff_latitude'])
        df['dropoff_latitude'] = np.where(df['dropoff_latitude']>42, 42,df['dropoff_latitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']<-75, -75,df['pickup_longitude'])
        df['pickup_longitude'] = np.where(df['pickup_longitude']>-72, -72,df['pickup_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']<-75, -75,df['dropoff_longitude'])
        df['dropoff_longitude'] = np.where(df['dropoff_longitude']>-72, -72,df['dropoff_longitude'])
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
        df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()

        df['manhattan'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 1)

        df['euclidean'] = minkowski_distance(df['pickup_longitude'], df['dropoff_longitude'],
                                               df['pickup_latitude'], df['dropoff_latitude'], 2)

        df['haversine'] =  haversine_np(df['pickup_longitude'], df['pickup_latitude'],
                                 df['dropoff_longitude'], df['dropoff_latitude']) 

        for i in place['loc'].tolist():
            for j in ['pickup','dropoff']:
                df[str(j)+'_distance_to'+str(i)] = distance_to_place(df,i,str(j)+'_longitude',str(j)+'_latitude')

        df['direction'] = calculate_direction(df)

        df = extract_dateinfo(df, 'pickup_datetime', drop = False, 
                                 time = True, start_ref = df['pickup_datetime'].min())

        holidays = calendar().holidays()
        df["usFedHoliday"] =  df.pickup_datetime.dt.date.astype('datetime64').isin(holidays)
        df[cat_cols] = df[cat_cols].astype(str)
        
        
        if self.is_cat==1:
            df = df[cat_cols]
        elif self.is_cat==0:
            df = df[num_cols] 
        else:
            df = df
            
        return df
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)


## === cell 40
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    result = np.sqrt(mean_squared_error(y_true, y_pred))
    return result




## === cell 41
ori_cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count']


## === cell 42
num_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'abs_lat_diff', 'abs_lon_diff', 'manhattan', 'euclidean', 'haversine',
       'pickup_distance_tojfk', 'dropoff_distance_tojfk',
       'pickup_distance_tonyc', 'dropoff_distance_tonyc',
       'pickup_distance_toewr', 'dropoff_distance_toewr',
       'pickup_distance_tolgr', 'dropoff_distance_tolgr', 'direction',
       'pickup_Year', 'pickup_Month', 'pickup_Week', 'pickup_Day',
       'pickup_Dayofweek', 'pickup_Dayofyear', 'pickup_Days_in_month',
       'pickup_Hour', 'pickup_Minute', 'pickup_Second',
       'pickup_Days_in_year', 'pickup_frac_day', 'pickup_frac_week',
       'pickup_frac_month', 'pickup_frac_year', 'pickup_Elapsed']
cat_cols = ['pickup_is_leap_year','usFedHoliday']
target = 'fare_amount'


## === cell 45
from sklearn.model_selection import train_test_split
train, val = train_test_split(data,test_size=0.1, random_state=RSEED)


## === cell 46
num_transformer = Pipeline(steps=[
                                ('dataprep', data_transform(num_cols,cat_cols,is_cat=0)),
                                ('imputer', SimpleImputer(strategy = "mean")),
                                ('scaler', PowerTransformer())  #PowerTransformer
                                ])

cat_transformer = Pipeline(steps=[
                                ('dataprep', data_transform(num_cols,cat_cols,is_cat=1)),
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='error')) #, drop = "if_binary"
                                ])

transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ori_cols),
        ('cat', cat_transformer, ori_cols)
    ])

knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=123)
rf = DecisionTreeRegressor(random_state=123)
eln = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=2020)
rg = Ridge(alpha=1.0, random_state=2020)
ls = Lasso(alpha=1.0, random_state=2020)
xgb = XGBRegressor(random_state=2020, booster='gbtree',n_estimators=20, tree_method='hist')
lgb = LGBMRegressor(objective='regression',random_states=2020, metric = 'rmse', num_leaves = 31, boosting_type='gbdt', max_depth=5, learning_rate=0.034)
catb = CatBoostRegressor(iterations=2,learning_rate=0.5,depth=3, silent=True)
lr = LinearRegression()


stack = StackingCVRegressor(regressors=(knn, lgb, xgb, catb, dt, rf, eln, rg, ls), meta_regressor=lr, cv=3)

stack_pipeline = Pipeline(steps=[('transformer', transformer),
                      ('stack', stack)
                      ])

main_pipeline = TransformedTargetRegressor(stack_pipeline,
                                    transformer = PowerTransformer())

params = {  
    
           
    
          }


model = GridSearchCV(estimator=main_pipeline, param_grid=params,  cv=3, n_jobs=-1, scoring='neg_mean_squared_error' ,refit=True)


train_X = train.drop(target,axis=1)
train_y = train[target]

model.fit(train_X, train_y)


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2906711264.py in <cell line: 0>()
     59 train_y = train[target]
     60 
---> 61 model.fit(train_X, train_y)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
    872                 return results
    873 
--> 874             self._run_search(evaluate_candidates)
    875 
    876             # multimetric is determined here because in the case of a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
   1386     def _run_search(self, evaluate_candidates):
   1387         """Search all candidates in param_grid"""
-> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
   1389 
   1390 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
    849                     )
    850 
--> 851                 _warn_or_raise_about_fit_failures(out, self.error_score)
    852 
    853                 # For callable self.scoring, the return type is only know after

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py in _warn_or_raise_about_fit_failures(results, error_score)
    365                 f"Below are more details about the failures:\n{fit_errors_summary}"
    366             )
--> 367             raise ValueError(all_fits_failed_message)
    368 
    369         else:

ValueError: 
All the 3 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
3 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/compose/_target.py", line 262, in fit
    self.regressor_.fit(X, y_trans, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 401, in fit
    Xt = self._fit(X, y, **fit_params_steps)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 359, in _fit
    X, fitted_transformer = fit_transform_one_cached(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/memory.py", line 326, in __call__
    return self.func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 893, in _fit_transform_one
    res = transformer.fit_transform(X, y, **fit_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py", line 140, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py", line 727, in fit_transform
    result = self._fit_transform(X, y, _fit_transform_one)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py", line 658, in _fit_transform
    return Parallel(n_jobs=self.n_jobs)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py", line 63, in __call__
    return super().__call__(iterable_with_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 1986, in __call__
    return output if self.return_generator else list(output)
                                                ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 1914, in _get_sequential_output
    res = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py", line 123, in __call__
    return self.function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 893, in _fit_transform_one
    res = transformer.fit_transform(X, y, **fit_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 437, in fit_transform
    Xt = self._fit(X, y, **fit_params_steps)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 359, in _fit
    X, fitted_transformer = fit_transform_one_cached(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/memory.py", line 326, in __call__
    return self.func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py", line 893, in _fit_transform_one
    res = transformer.fit_transform(X, y, **fit_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1566755723.py", line 67, in fit_transform
  File "/tmp/ipykernel_11/1566755723.py", line 48, in transform
NameError: name 'extract_dateinfo' is not defined


## === cell 47
print('Best parameters cv: %s' % model.best_params_)
print('Best nmse cv: %.2f' % model.best_score_)
rmse_cv = np.sqrt(model.best_score_*-1) 
print('Best rmse cv: %.2f' % rmse_cv)
print('rmse all train data: %.2f' %rmse(model.predict(train_X),train['fare_amount']))
pd.DataFrame(model.cv_results_)


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2102987770.py in <cell line: 0>()
----> 1 print('Best parameters cv: %s' % model.best_params_)
      2 print('Best nmse cv: %.2f' % model.best_score_)
      3 rmse_cv = np.sqrt(model.best_score_*-1)
      4 print('Best rmse cv: %.2f' % rmse_cv)
      5 print('rmse all train data: %.2f' %rmse(model.predict(train_X),train['fare_amount']))

AttributeError: 'GridSearchCV' object has no attribute 'best_params_'

## === cell 48
print('rmse all train data: %.2f' %rmse(model.predict(val.drop(target,axis=1)),val['fare_amount']))


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1042496442.py in <cell line: 0>()
----> 1 print('rmse all train data: %.2f' %rmse(model.predict(val.drop(target,axis=1)),val['fare_amount']))

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 49
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv', 
                   parse_dates = ['pickup_datetime'])


## === cell 50
test_X = test[ori_cols]
predicted_fare = model.predict(test_X)
print(predicted_fare)


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1437061380.py in <cell line: 0>()
      1 test_X = test[ori_cols]
      2 # Use the model to make predictions
----> 3 predicted_fare = model.predict(test_X)
      4 # We will look at the predicted prices to ensure we have something sensible.
      5 print(predicted_fare)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 51
my_submission = pd.DataFrame({'key': test.key, 'fare_amount': predicted_fare})
my_submission.to_csv('submission_v1a.csv', index=False)


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/597315920.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({'key': test.key, 'fare_amount': predicted_fare})
      2 # you could use any filename. We choose submission here
      3 my_submission.to_csv('submission_v1a.csv', index=False)

NameError: name 'predicted_fare' is not defined

## === cell 52
import joblib
joblib.dump(model, f'nyk_taxi_stack.pkl')


## === cell 53
model


## === cell 54
dt    = '2019-06-20 12:26:21'
plong =  -73.844
plat  =  41.721
dlong =  -74.842
dlat  =  42.712
pc    =  5


## === cell 55
cek = pd.DataFrame({'pickup_datetime' : [dt], 'pickup_longitude' : [plong], 'pickup_latitude' : [plat], 'dropoff_longitude' : [dlong], 'dropoff_latitude' : [dlat], 'passenger_count' : [pc]})
model.predict(cek)


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/988928093.py in <cell line: 0>()
      1 cek = pd.DataFrame({'pickup_datetime' : [dt], 'pickup_longitude' : [plong], 'pickup_latitude' : [plat], 'dropoff_longitude' : [dlong], 'dropoff_latitude' : [dlat], 'passenger_count' : [pc]})
----> 2 model.predict(cek)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
