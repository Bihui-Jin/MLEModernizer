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

5.69253

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3514382585.py in <cell line: 0>()
      1 from sklearn.linear_model import LinearRegression
----> 2 lr=LinearRegression(normalize=True)
      3 lr.fit(Xtrain,ytrain)
      4 print(lr.score(Xtest,ytest))

TypeError: LinearRegression.__init__() got an unexpected keyword argument 'normalize'

## === cell 27
pred=np.round(lr.predict(test_data.drop('key',axis=1)),2)
submission=pd.DataFrame(data=pred,columns=['fare_amount'])
submission['key']=test_data['key']
submission=submission[['key','fare_amount']]
submission.set_index('key',inplace=True)
submission.to_csv('Submission.csv')


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1916866545.py in <cell line: 0>()
----> 1 pred=np.round(lr.predict(test_data.drop('key',axis=1)),2)
      2 submission=pd.DataFrame(data=pred,columns=['fare_amount'])
      3 submission['key']=test_data['key']
      4 submission=submission[['key','fare_amount']]
      5 submission.set_index('key',inplace=True)

NameError: name 'lr' is not defined
