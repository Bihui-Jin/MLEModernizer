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

3.9

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
seaborn==0.12.2
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
import seaborn as sns
import time


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_dt=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows=10_000_000)
train_dt.head()


## === cell 2
train_dt.shape


## === cell 3
train_dt.info()


## === cell 4
test_dt=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
test_dt.head()


## === cell 5
test_dt.info()


## === cell 6
train_dt.isna().sum()


## === cell 7
train_dt['Difference_longitude']=np.abs(np.asarray(train_dt['pickup_longitude']-train_dt['dropoff_longitude']))
train_dt['Difference_latitude']=np.abs(np.asarray(train_dt['pickup_latitude']-train_dt['dropoff_latitude']))


test_dt['Difference_longitude']=np.abs(np.asarray(test_dt['pickup_longitude']-test_dt['dropoff_longitude']))
test_dt['Difference_latitude']=np.abs(np.asarray(test_dt['pickup_latitude']-test_dt['dropoff_latitude']))


## === cell 8
print(f'Before Dropping null values: {len(train_dt)}')
train_dt.dropna(inplace=True)
print(f'After Dropping null values: {len(train_dt)}')


## === cell 9
plot = train_dt[:2000].plot.scatter('Difference_longitude', 'Difference_latitude')


## === cell 10
train_dt=train_dt[(train_dt['Difference_longitude']<5.0)&(train_dt['Difference_latitude']<5.0)]


## === cell 11
ls1=list(train_dt['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-7:]
train_dt['pickuptime']=ls1    



ls1=list(test_dt['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-7:]
test_dt['pickuptime']=ls1


## === cell 12
train_dt.head()


## === cell 13
ls1=list(train_dt['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][:-4:]
    ls1[i]=pd.Timestamp(ls1[i])
    ls1[i]=ls1[i].weekday()
train_dt['Weekday']=ls1


ls1=list(test_dt['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][:-4:]
    ls1[i]=pd.Timestamp(ls1[i])
    ls1[i]=ls1[i].weekday()
test_dt['Weekday']=ls1


## === cell 14
train_dt.head()


## === cell 15
test_dt.head()


## === cell 16
train_dt.drop('pickup_datetime',inplace=True,axis=1)
test_dt.drop('pickup_datetime',inplace=True,axis=1)


## === cell 17
train_dt['Weekday'].replace(to_replace=[i for i in range(0,7)],
                            value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                              inplace=True)
test_dt['Weekday'].replace(to_replace=[i for i in range(0,7)],
                              value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                              inplace=True)


## === cell 18
train_one_hot=pd.get_dummies(train_dt['Weekday'])
test_one_hot=pd.get_dummies(test_dt['Weekday'])
train_dt=pd.concat([train_dt,train_one_hot],axis=1)
test_dt=pd.concat([test_dt,test_one_hot],axis=1)


## === cell 19
train_dt.drop('Weekday',axis=1,inplace=True)
test_dt.drop('Weekday',axis=1,inplace=True)


## === cell 20
ls1=list(train_dt['pickuptime'])
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
train_dt['pickuptime']=ls1


ls1=list(test_dt['pickuptime'])
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
test_dt['pickuptime']=ls1


## === cell 21
train_dt.head()


## === cell 22
R = 6373.0
lat1 =np.asarray(np.radians(train_dt['pickup_latitude']))
lon1 = np.asarray(np.radians(train_dt['pickup_longitude']))
lat2 = np.asarray(np.radians(train_dt['dropoff_latitude']))
lon2 = np.asarray(np.radians(train_dt['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
ls1=[] 
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

    
train_dt['Distance']=np.asarray(distance)*0.621



lat1 =np.asarray(np.radians(test_dt['pickup_latitude']))
lon1 = np.asarray(np.radians(test_dt['pickup_longitude']))
lat2 = np.asarray(np.radians(test_dt['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_dt['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
 
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_dt['Distance']=np.asarray(distance)*0.621


## === cell 23
R = 6373.0
lat1 =np.asarray(np.radians(train_dt['pickup_latitude']))
lon1 = np.asarray(np.radians(train_dt['pickup_longitude']))
lat2 = np.asarray(np.radians(train_dt['dropoff_latitude']))
lon2 = np.asarray(np.radians(train_dt['dropoff_longitude']))

lat3=np.zeros(len(train_dt))+np.radians(40.6413111)
lon3=np.zeros(len(train_dt))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_dt['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_dt['Dropoff_Distance_airport']=np.asarray(distance2)*0.621



lat1 =np.asarray(np.radians(test_dt['pickup_latitude']))
lon1 = np.asarray(np.radians(test_dt['pickup_longitude']))
lat2 = np.asarray(np.radians(test_dt['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_dt['dropoff_longitude']))

lat3=np.zeros(len(test_dt))+np.radians(40.6413111)
lon3=np.zeros(len(test_dt))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_dt['Pickup_Distance_airport']=np.asarray(distance1)*0.621


a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
test_dt['Dropoff_Distance_airport']=np.asarray(distance2)*0.621


## === cell 24
train_dt['Distance']=np.round(train_dt['Distance'],2)
train_dt['Pickup_Distance_airport']=np.round(train_dt['Pickup_Distance_airport'],2)
train_dt['Dropoff_Distance_airport']=np.round(train_dt['Dropoff_Distance_airport'],2)
test_dt['Distance']=np.round(test_dt['Distance'],2)
test_dt['Pickup_Distance_airport']=np.round(test_dt['Pickup_Distance_airport'],2)
test_dt['Dropoff_Distance_airport']=np.round(test_dt['Dropoff_Distance_airport'],2)


## === cell 25
train_dt.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
test_dt.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)


## === cell 26
train_dt['Difference_longitude']=np.abs(train_dt['Difference_longitude']-np.mean(train_dt['Difference_longitude']))
train_dt['Difference_longitude']=train_dt['Difference_longitude']/np.var(train_dt['Difference_longitude'])


## === cell 27
train_dt['Difference_latitude']=np.abs(train_dt['Difference_latitude']-np.mean(train_dt['Difference_latitude']))
train_dt['Difference_latitude']=train_dt['Difference_latitude']/np.var(train_dt['Difference_latitude'])


## === cell 28
test_dt['Difference_longitude']=np.abs(test_dt['Difference_longitude']-np.mean(test_dt['Difference_longitude']))
test_dt['Difference_longitude']=test_dt['Difference_longitude']/np.var(test_dt['Difference_longitude'])

test_dt['Difference_latitude']=np.abs(test_dt['Difference_latitude']-np.mean(test_dt['Difference_latitude']))
test_dt['Difference_latitude']=test_dt['Difference_latitude']/np.var(test_dt['Difference_latitude'])


## === cell 29
train_dt.shape


## === cell 30
test_dt.shape


## === cell 31
from sklearn.model_selection import train_test_split
X=train_dt.drop(['key','fare_amount'],axis=1)
y=train_dt['fare_amount']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=80)


## === cell 32
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))


## --- ERROR in cell 32, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3439814487.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mlinear_model[0m [0;32mimport[0m [0mLinearRegression[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mlr[0m[0;34m=[0m[0mLinearRegression[0m[0;34m([0m[0mnormalize[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mlr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mprint[0m[0;34m([0m[0mlr[0m[0;34m.[0m[0mscore[0m[0;34m([0m[0mX_test[0m[0;34m,[0m[0my_test[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: LinearRegression.__init__() got an unexpected keyword argument 'normalize'

## === cell 33
pred=np.round(lr.predict(test_dt.drop('key',axis=1)),2)
