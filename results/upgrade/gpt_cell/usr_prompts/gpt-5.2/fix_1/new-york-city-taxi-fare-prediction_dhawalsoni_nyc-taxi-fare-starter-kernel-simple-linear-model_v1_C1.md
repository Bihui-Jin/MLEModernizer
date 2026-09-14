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
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

print(os.listdir('../input'))


## === cell 1
train_data =  pd.read_csv('../input/train.csv', nrows = 10_000_000)
train_data.dtypes


## === cell 2
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_data)


## === cell 3
print(train_data.isnull().sum())


## === cell 4
train_data = train_data.dropna(how = 'any', axis = 'rows')


## === cell 5
plot = train_data.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


## === cell 6
train_data = train_data[(train_data.abs_diff_longitude < 5.0) & (train_data.abs_diff_latitude < 5.0)]


## === cell 7
test_data = pd.read_csv('../input/test.csv')
test_data.dtypes


## === cell 8
add_travel_vector_features(test_data)


## === cell 9
test_data.dtypes


## === cell 10
train_data.dtypes


## === cell 11
train_data.head()


## === cell 12
ls1=list(train_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-7:]
train_data['pickup_time']=ls1

ls1=list(test_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][11:-7:]
test_data['pickup_time']=ls1


## === cell 13
ls1=list(train_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=pd.Timestamp(ls1[i][:-4:]).weekday()
train_data['weekday']=ls1

ls1=list(test_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=pd.Timestamp(ls1[i][:-4:]).weekday()
test_data['weekday']=ls1


## === cell 14
train_data.drop('pickup_datetime',inplace=True,axis=1)
test_data.drop('pickup_datetime',inplace=True,axis=1)


## === cell 15
train_data.head(10)


## === cell 16
train_data['weekday'].replace(to_replace=[i for i in range(0,7)],value=['Monday','Tuesday','Wednesday','Thrusday','Friday','Saturday','Sunday'],inplace=True)
test_data['weekday'].replace(to_replace=[i for i in range(0,7)],value=['Monday','Tuesday','Wednesday','Thrusday','Friday','Saturday','Sunday'],inplace=True)


## === cell 17
train_one_shot=pd.get_dummies(train_data['weekday'])
test_one_shot=pd.get_dummies(test_data['weekday'])
test_data=pd.concat([test_data,test_one_shot],axis=1)
train_data=pd.concat([train_data,train_one_shot],axis=1)


## === cell 18
train_data.drop('weekday',inplace=True,axis=1)
test_data.drop('weekday',inplace=True,axis=1)


## === cell 19
ls1=list(train_data['pickup_time'])
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
train_data['pickup_time']=ls1

ls1=list(test_data['pickup_time'])
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
test_data['pickup_time']=ls1


## === cell 20
R=6372.0
lat1=np.asarray(np.radians(train_data['pickup_latitude']))
lon1=np.asarray(np.radians(train_data['pickup_longitude']))
lat2=np.asarray(np.radians(train_data['dropoff_latitude']))
lon2=np.asarray(np.radians(train_data['dropoff_latitude']))

dlon=lon2-lon1
dlat=lat2-lat1
ls1=[]
a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
distance=R*c

train_data['Distance']=np.asarray(distance)*0.621


lat1=np.asarray(np.radians(test_data['pickup_latitude']))
lon1=np.asarray(np.radians(test_data['pickup_longitude']))
lat2=np.asarray(np.radians(test_data['dropoff_latitude']))
lon2=np.asarray(np.radians(test_data['dropoff_latitude']))

dlon=lon2-lon1
dlat=lat2-lat1
ls1=[]
a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
distance=R*c

test_data['Distance']=np.asarray(distance)*0.621


## === cell 21
R=6372.0
lat1=np.asarray(np.radians(train_data['pickup_latitude']))
lon1=np.asarray(np.radians(train_data['pickup_longitude']))
lat2=np.asarray(np.radians(train_data['dropoff_latitude']))
lon2=np.asarray(np.radians(train_data['dropoff_latitude']))

lat3=np.zeros(len(train_data))+np.radians(40.641)
lon3=np.zeros(len(train_data))+np.radians(-73.778)
dlon_pickup=lon3-lon1
dlat_pickup=lat3-lat1
dlon_dropoff=lon3-lon2
dlat_dropoff=lat3-lat2
a1=np.sin(dlat_pickup/2)**2+np.cos(lat1)*np.cos(lat3)*np.sin(dlon_pickup/2)**2
c1=2*np.arctan2(np.sqrt(a1),np.sqrt(1-a1))
distance1=R*c1
train_data['pickup_distance_a']=np.asarray(distance1)*0.621

a2=np.sin(dlat_dropoff/2)**2+np.cos(lat2)*np.cos(lat3)*np.sin(dlon_dropoff/2)**2
c2=2*np.arctan2(np.sqrt(a2),np.sqrt(1-a2))
distance2=R*c2

train_data['dropoff_distance_a']=np.asarray(distance2)*0.621

lat1=np.asarray(np.radians(test_data['pickup_latitude']))
lon1=np.asarray(np.radians(test_data['pickup_longitude']))
lat2=np.asarray(np.radians(test_data['dropoff_latitude']))
lon2=np.asarray(np.radians(test_data['dropoff_latitude']))

lat3=np.zeros(len(test_data))+np.radians(40.641)
lon3=np.zeros(len(test_data))+np.radians(-73.778)
dlon_pickup=lon3-lon1
dlat_pickup=lat3-lat1
dlon_dropoff=lon3-lon2
dlat_dropoff=lat3-lat2
a1=np.sin(dlat_pickup/2)**2+np.cos(lat1)*np.cos(lat3)*np.sin(dlon_pickup/2)**2
c1=2*np.arctan2(np.sqrt(a1),np.sqrt(1-a1))
distance1=R*c1
test_data['pickup_distance_a']=np.asarray(distance1)*0.621

a2=np.sin(dlat_dropoff/2)**2+np.cos(lat2)*np.cos(lat3)*np.sin(dlon_dropoff/2)**2
c2=2*np.arctan2(np.sqrt(a2),np.sqrt(1-a2))
distance2=R*c2

test_data['dropoff_distance_a']=np.asarray(distance2)*0.621


## === cell 22
train_data['Distance']=np.round(train_data['Distance'],2)
train_data['pickup_distance_a']=np.round(train_data['pickup_distance_a'],2)
train_data['dropoff_distance_a']=np.round(train_data['dropoff_distance_a'],2)

test_data['Distance']=np.round(test_data['Distance'],2)
test_data['pickup_distance_a']=np.round(test_data['pickup_distance_a'],2)
test_data['dropoff_distance_a']=np.round(test_data['dropoff_distance_a'],2)


## === cell 23
train_data.drop(['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'],inplace=True,axis=1)
test_data.drop(['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'],inplace=True,axis=1)


## === cell 24
train_data['abs_diff_longitude']=train_data['abs_diff_longitude']-np.mean(train_data['abs_diff_longitude'])
train_data['abs_diff_longitude']=train_data['abs_diff_longitude']/np.var(train_data['abs_diff_longitude'])

train_data['abs_diff_latitude']=train_data['abs_diff_latitude']-np.mean(train_data['abs_diff_latitude'])
train_data['abs_diff_latitude']=train_data['abs_diff_latitude']/np.var(train_data['abs_diff_latitude'])

test_data['abs_diff_longitude']=test_data['abs_diff_longitude']-np.mean(test_data['abs_diff_longitude'])
test_data['abs_diff_longitude']=test_data['abs_diff_longitude']/np.var(test_data['abs_diff_longitude'])

test_data['abs_diff_latitude']=test_data['abs_diff_latitude']-np.mean(test_data['abs_diff_latitude'])
test_data['abs_diff_latitude']=test_data['abs_diff_latitude']/np.var(test_data['abs_diff_latitude'])


## === cell 25
from sklearn.model_selection import train_test_split
X=train_data.drop(['key','fare_amount'],axis=1)
y=train_data['fare_amount']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.01,random_state=80)


## === cell 26
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(Xtrain,ytrain)
print(lr.score(Xtest,ytest))


## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3514382585.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mlinear_model[0m [0;32mimport[0m [0mLinearRegression[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mlr[0m[0;34m=[0m[0mLinearRegression[0m[0;34m([0m[0mnormalize[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mlr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mXtrain[0m[0;34m,[0m[0mytrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mprint[0m[0;34m([0m[0mlr[0m[0;34m.[0m[0mscore[0m[0;34m([0m[0mXtest[0m[0;34m,[0m[0mytest[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: LinearRegression.__init__() got an unexpected keyword argument 'normalize'

## === cell 27
pred=np.round(lr.predict(test_data.drop('key',axis=1)),2)
submission=pd.DataFrame(data=pred,columns=['fare_amount'])
submission['key']=test_data['key']
submission=submission[['key','fare_amount']]
submission.set_index('key',inplace=True)
submission.to_csv('Submission.csv')
