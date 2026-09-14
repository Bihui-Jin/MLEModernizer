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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
td=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows=10_000_000)
td.head()#td:train data # ted:test data


## === cell 2
td.shape


## === cell 3
td.info()


## === cell 4
ted=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
ted.head()


## === cell 5
ted.info()


## === cell 6
td.isna().sum()


## === cell 7
td['Difference_longitude']=np.abs(np.asarray(td['pickup_longitude']-td['dropoff_longitude']))
td['Difference_latitude']=np.abs(np.asarray(td['pickup_latitude']-td['dropoff_latitude']))


ted['Difference_longitude']=np.abs(np.asarray(ted['pickup_longitude']-ted['dropoff_longitude']))
ted['Difference_latitude']=np.abs(np.asarray(ted['pickup_latitude']-ted['dropoff_latitude']))


## === cell 8
print(f'Before Dropping null values: {len(td)}')
td.dropna(inplace=True)
print(f'After Dropping null values: {len(td)}')


## === cell 9
plot = td[:2000].plot.scatter('Difference_longitude', 'Difference_latitude')


## === cell 10
td=td[(td['Difference_longitude']<5.0)&(td['Difference_latitude']<5.0)]


## === cell 11
l1=list(td['pickup_datetime'])
for i in range(len(l1)):
    l1[i]=l1[i][11:-7:]
td['pickuptime']=l1    



l1=list(ted['pickup_datetime'])
for i in range(len(l1)):
    l1[i]=l1[i][11:-7:]
ted['pickuptime']=l1   


## === cell 12
td.head()


## === cell 13
l1=list(td['pickup_datetime'])
for i in range(len(l1)):
    l1[i]=l1[i][:-4:]
    l1[i]=pd.Timestamp(l1[i])
    l1[i]=l1[i].weekday()
td['Weekday']=l1


l1=list(ted['pickup_datetime'])
for i in range(len(l1)):
    l1[i]=l1[i][:-4:]
    l1[i]=pd.Timestamp(l1[i])
    l1[i]=l1[i].weekday()
ted['Weekday']=l1

td.head()


## === cell 14
ted.head()


## === cell 15
td.drop('pickup_datetime',inplace=True,axis=1)
ted.drop('pickup_datetime',inplace=True,axis=1)

td['Weekday'].replace(to_replace=[i for i in range(0,7)],
                            value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                              inplace=True)
ted['Weekday'].replace(to_replace=[i for i in range(0,7)],
                              value=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                              inplace=True)

th=pd.get_dummies(td['Weekday'])
teh=pd.get_dummies(ted['Weekday'])
td=pd.concat([td,th],axis=1)
ted=pd.concat([ted,teh],axis=1)

td.drop('Weekday',axis=1,inplace=True)
ted.drop('Weekday',axis=1,inplace=True)

l1=list(td['pickuptime'])
for i in range(len(l1)):
    z=l1[i].split(':')
    l1[i]=int(z[0])*100+int(z[1])
td['pickuptime']=l1


l1=list(ted['pickuptime'])
for i in range(len(l1)):
    z=l1[i].split(':')
    l1[i]=int(z[0])*100+int(z[1])
ted['pickuptime']=l1

td.head()


## === cell 16
R = 6373.0
lat1 =np.asarray(np.radians(td['pickup_latitude']))
lon1 = np.asarray(np.radians(td['pickup_longitude']))
lat2 = np.asarray(np.radians(td['dropoff_latitude']))
lon2 = np.asarray(np.radians(td['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
l1=[] 
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

    
td['Distance']=np.asarray(distance)*0.621



lat1 =np.asarray(np.radians(ted['pickup_latitude']))
lon1 = np.asarray(np.radians(ted['pickup_longitude']))
lat2 = np.asarray(np.radians(ted['dropoff_latitude']))
lon2 = np.asarray(np.radians(ted['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
 
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
ted['Distance']=np.asarray(distance)*0.621


## === cell 17
R = 6373.0
lat1 =np.asarray(np.radians(td['pickup_latitude']))
lon1 = np.asarray(np.radians(td['pickup_longitude']))
lat2 = np.asarray(np.radians(td['dropoff_latitude']))
lon2 = np.asarray(np.radians(td['dropoff_longitude']))

lat3=np.zeros(len(td))+np.radians(40.6413111)
lon3=np.zeros(len(td))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
td['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
td['Dropoff_Distance_airport']=np.asarray(distance2)*0.621



lat1 =np.asarray(np.radians(ted['pickup_latitude']))
lon1 = np.asarray(np.radians(ted['pickup_longitude']))
lat2 = np.asarray(np.radians(ted['dropoff_latitude']))
lon2 = np.asarray(np.radians(ted['dropoff_longitude']))

lat3=np.zeros(len(ted))+np.radians(40.6413111)
lon3=np.zeros(len(ted))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
ted['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
ted['Dropoff_Distance_airport']=np.asarray(distance2)*0.621

td['Distance']=np.round(td['Distance'],2)
td['Pickup_Distance_airport']=np.round(td['Pickup_Distance_airport'],2)
td['Dropoff_Distance_airport']=np.round(td['Dropoff_Distance_airport'],2)
ted['Distance']=np.round(ted['Distance'],2)
ted['Pickup_Distance_airport']=np.round(ted['Pickup_Distance_airport'],2)
ted['Dropoff_Distance_airport']=np.round(ted['Dropoff_Distance_airport'],2)

td.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
ted.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)

td['Difference_longitude']=np.abs(td['Difference_longitude']-np.mean(td['Difference_longitude']))
td['Difference_longitude']=td['Difference_longitude']/np.var(td['Difference_longitude'])

td['Difference_latitude']=np.abs(td['Difference_latitude']-np.mean(td['Difference_latitude']))
td['Difference_latitude']=td['Difference_latitude']/np.var(td['Difference_latitude'])

ted['Difference_longitude']=np.abs(ted['Difference_longitude']-np.mean(ted['Difference_longitude']))
ted['Difference_longitude']=ted['Difference_longitude']/np.var(ted['Difference_longitude'])

ted['Difference_latitude']=np.abs(ted['Difference_latitude']-np.mean(ted['Difference_latitude']))
ted['Difference_latitude']=ted['Difference_latitude']/np.var(ted['Difference_latitude'])

td.shape


## === cell 18
ted.shape


## === cell 19
from sklearn.model_selection import train_test_split
X=td.drop(['key','fare_amount'],axis=1)
y=td['fare_amount']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=80)

from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))


## --- ERROR in cell 19, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/367281664.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mlinear_model[0m [0;32mimport[0m [0mLinearRegression[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mlr[0m[0;34m=[0m[0mLinearRegression[0m[0;34m([0m[0mnormalize[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0mlr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0mprint[0m[0;34m([0m[0mlr[0m[0;34m.[0m[0mscore[0m[0;34m([0m[0mX_test[0m[0;34m,[0m[0my_test[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: LinearRegression.__init__() got an unexpected keyword argument 'normalize'

## === cell 20
pred=np.round(lr.predict(ted.drop('key',axis=1)),2)
pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv').head()
