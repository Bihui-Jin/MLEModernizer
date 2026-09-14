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
df=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows=10_000_000)
df.head(5)


## === cell 2
for col in df.columns:
    print(col,"..................")
    print(df[f'{col}'].value_counts(),'\n')


## === cell 3
df.isna().sum()


## === cell 4
df.dropna(inplace=True)


## === cell 5
df['manhattan_distance']=((df.dropoff_latitude - df.pickup_latitude)+ (df.dropoff_longitude - df.pickup_longitude)).abs()
df['Difference_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
df['Difference_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


## === cell 6
df.head(5)


## === cell 7
ls1=df['pickup_datetime'].tolist()
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-7:]
df['pickuptime']=ls1    

ls1=df['pickuptime'].tolist()
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
df['pickuptime']=ls1


## === cell 8
df.head(5)


## === cell 9
import numpy as np
ls1=df['pickup_datetime'].tolist()
for i in range(len(ls1)):
    ls1[i]=ls1[i][:-4]
    ls1[i]=pd.Timestamp(ls1[i])
    ls1[i]=ls1[i].weekday()
    
from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder()
encodedls1=encoder.fit_transform(np.array(ls1).reshape(len(ls1),1))


## === cell 10
encodedls1=encodedls1.toarray()
print(len(encodedls1[0]))
days=['mon','tue','wed','thr','fri','sat','sun']
for i in range(len(days)):
     df[days[i]]=encodedls1[:,i]


## === cell 11
df.head(5)


## === cell 12
R = 6373.0
lat1 =np.asarray(np.radians(df['pickup_latitude']))
lon1 = np.asarray(np.radians(df['pickup_longitude']))
lat2 = np.asarray(np.radians(df['dropoff_latitude']))
lon2 = np.asarray(np.radians(df['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
ls1=[] 
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

    
df['Distance']=np.asarray(distance)*0.621


## === cell 13
R = 6373.0
lat1 =np.asarray(np.radians(df['pickup_latitude']))
lon1 = np.asarray(np.radians(df['pickup_longitude']))
lat2 = np.asarray(np.radians(df['dropoff_latitude']))
lon2 = np.asarray(np.radians(df['dropoff_longitude']))

lat3=np.zeros(len(df))+np.radians(40.6413111)
lon3=np.zeros(len(df))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
df['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
df['Dropoff_Distance_airport']=np.asarray(distance2)*0.621


## === cell 14
df['Distance']=np.round(df['Distance'],2)
df['Pickup_Distance_airport']=np.round(df['Pickup_Distance_airport'],2)
df['Dropoff_Distance_airport']=np.round(df['Dropoff_Distance_airport'],2)


## === cell 15
df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)


## === cell 16
df['Difference_longitude']=np.abs(df['Difference_longitude']-np.mean(df['Difference_longitude']))
df['Difference_longitude']=df['Difference_longitude']/np.var(df['Difference_longitude'])


## === cell 17
df['Difference_latitude']=np.abs(df['Difference_latitude']-np.mean(df['Difference_latitude']))
df['Difference_latitude']=df['Difference_latitude']/np.var(df['Difference_latitude'])


## === cell 19
df.head(5)


## === cell 20
test_data=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
test_data.dropna(inplace=True)
test_data.head(5)


## === cell 21
test_data['manhattan_distance']=((test_data.dropoff_latitude - test_data.pickup_latitude)+ (test_data.dropoff_longitude - test_data.pickup_longitude)).abs()
test_data['Difference_longitude'] = (test_data.dropoff_longitude - test_data.pickup_longitude).abs()
test_data['Difference_latitude'] = (test_data.dropoff_latitude - test_data.pickup_latitude).abs()

ls1=test_data['pickup_datetime'].tolist()
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-7:]
test_data['pickuptime']=ls1  

ls1=test_data['pickuptime'].tolist()
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
test_data['pickuptime']=ls1

ls2=test_data['pickup_datetime'].tolist()
for i in range(len(ls1)):
    ls2[i]=ls2[i][:-4]
    ls2[i]=pd.Timestamp(ls1[i])
    ls2[i]=ls2[i].weekday()

encodedls2=encoder.transform(np.array(ls2).reshape(len(ls2),1))
encodedls2=encodedls2.toarray()
    
for i in range(len(days)):
     test_data[days[i]]=encodedls2[:,i]
        
lat1 =np.asarray(np.radians(test_data['pickup_latitude']))
lon1 = np.asarray(np.radians(test_data['pickup_longitude']))
lat2 = np.asarray(np.radians(test_data['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_data['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
 
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data['Distance']=np.asarray(distance)*0.621


## === cell 22
R = 6373.0

lat1 =np.asarray(np.radians(test_data['pickup_latitude']))
lon1 = np.asarray(np.radians(test_data['pickup_longitude']))
lat2 = np.asarray(np.radians(test_data['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_data['dropoff_longitude']))

lat3=np.zeros(len(test_data))+np.radians(40.6413111)
lon3=np.zeros(len(test_data))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
test_data['Dropoff_Distance_airport']=np.asarray(distance2)*0.621


## === cell 23
test_data['Distance']=np.round(test_data['Distance'],2)
test_data['Pickup_Distance_airport']=np.round(test_data['Pickup_Distance_airport'],2)
test_data['Dropoff_Distance_airport']=np.round(test_data['Dropoff_Distance_airport'],2)

test_data.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)

test_data['Difference_longitude']=np.abs(test_data['Difference_longitude']-np.mean(test_data['Difference_longitude']))
test_data['Difference_longitude']=test_data['Difference_longitude']/np.var(test_data['Difference_longitude'])

test_data['Difference_latitude']=np.abs(test_data['Difference_latitude']-np.mean(test_data['Difference_latitude']))
test_data['Difference_latitude']=test_data['Difference_latitude']/np.var(test_data['Difference_latitude'])


## === cell 24
test_data.head(5)


## === cell 25
df.head(5)


## === cell 26
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(df.drop(['key','fare_amount','pickup_datetime'],axis=1),df['fare_amount'],test_size=0.01,random_state=80)

df=[1,2,3,4,5]


## === cell 27
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3439814487.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mlinear_model[0m [0;32mimport[0m [0mLinearRegression[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mlr[0m[0;34m=[0m[0mLinearRegression[0m[0;34m([0m[0mnormalize[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mlr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mprint[0m[0;34m([0m[0mlr[0m[0;34m.[0m[0mscore[0m[0;34m([0m[0mX_test[0m[0;34m,[0m[0my_test[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: LinearRegression.__init__() got an unexpected keyword argument 'normalize'

## === cell 28
X_test.info()
