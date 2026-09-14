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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb


## === cell 1
train = pd.read_csv('../input/train.csv', nrows = 1000000)
test = pd.read_csv('../input/test.csv')


## === cell 2
print('Sum of NaN values for each column')
print(train.isnull().sum())

train = train.dropna()
print('Sum of NaN values for each column after dropping NaN')
print(train.isnull().sum())


## === cell 4
pickup_longitude_min = test.pickup_longitude.min()
pickup_longitude_max = test.pickup_latitude.max()
pickup_latitude_min = test.pickup_latitude.min()
pickup_latitude_max = test.pickup_latitude.max()
dropoff_longitude_min = test.dropoff_longitude.min()
dropoff_longitude_max = test.dropoff_longitude.max()
dropoff_latitude_min = test.dropoff_latitude.min()
dropoff_latitude_max = test.dropoff_latitude.max()


## === cell 5


train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 300)]
train = train.loc[(train['pickup_longitude'] > pickup_longitude_min) & (train['pickup_longitude'] < pickup_longitude_max)]
train = train.loc[(train['pickup_latitude'] > pickup_latitude_min) & (train['pickup_latitude'] < pickup_latitude_max)]
train = train.loc[(train['dropoff_longitude'] > dropoff_longitude_min) & (train['dropoff_longitude'] < dropoff_longitude_max)]
train = train.loc[(train['dropoff_latitude'] > dropoff_latitude_min) & (train['dropoff_latitude'] < dropoff_latitude_max)]
train = train.loc[train['passenger_count'] <= 8]
train.describe()


## === cell 6
def haversine_np(a):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = a
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def cityblock(a,dist='cityblock'):
    return pdist(a.reshape(2,2),dist)[0]


## === cell 8
dist_types = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
dist_types = ['chebyshev','euclidean','canberra','sqeuclidean','braycurtis','minkowski','hamming','cityblock']
from multiprocessing import Pool
pool = Pool()
combine = [train, test]
for dataset in combine:    
    coords = dataset[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].values

    dataset['haversine'] = pool.map(haversine_np,coords)
        
    
    for dist_type in dist_types:
        print(dist_type)
        dataset[dist_type] = pool.starmap(cityblock,[(x,dist_type,) for x in coords],1000)
    
    dataset['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
pool.close()
pool.join()
del pool
train.head(3)


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3726202364.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     19[0m     [0mdataset[0m[0;34m[[0m[0;34m'hour_of_day'[0m[0;34m][0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mhour[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m     [0mdataset[0m[0;34m[[0m[0;34m'day'[0m[0;34m][0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mday[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 21[0;31m     [0mdataset[0m[0;34m[[0m[0;34m'week'[0m[0;34m][0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mweek[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m     [0mdataset[0m[0;34m[[0m[0;34m'month'[0m[0;34m][0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mmonth[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m     [0mdataset[0m[0;34m[[0m[0;34m'day_of_year'[0m[0;34m][0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mpickup_datetime[0m[0;34m.[0m[0mdt[0m[0;34m.[0m[0mdayofyear[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'DatetimeProperties' object has no attribute 'week'

## === cell 9
features = [ 'pickup_latitude',  'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
             'haversine', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean', 'braycurtis',
             'minkowski',  'hamming', 'cityblock']

n = StandardScaler()
train[features] = n.fit_transform(train[features])
test[features] = n.transform(test[features])
