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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline


## === cell 1
train_df = pd.read_csv('../input/train.csv', nrows = 10000)


## === cell 2
train_df.shape


## === cell 3
test_df = pd.read_csv('../input/test.csv')


## === cell 4
test_df.shape


## === cell 5
train_df.head(5)


## === cell 6
train_df.isnull().sum()


## === cell 7
train_df.dropna(inplace=True)


## === cell 8
train_df.describe()


## === cell 9
train_df = train_df[train_df['fare_amount']>0]


## === cell 10
train_df.shape


## === cell 11
def distance(lat1, lon1, lat2, lon2):
    a = 0.5 - np.cos((lat2 - lat1) *  0.017453292519943295)/2 + np.cos(lat1 * 0.017453292519943295) * np.cos(lat2 * 0.017453292519943295) * (1 - np.cos((lon2 - lon1) *  0.017453292519943295)) / 2
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res


## === cell 12
train_df['distance'] = distance(train_df.pickup_latitude, train_df.pickup_longitude, \
                                      train_df.dropoff_latitude,train_df.dropoff_longitude)


## === cell 13
test_df['distance'] = distance(test_df.pickup_latitude, test_df.pickup_longitude, \
                                      test_df.dropoff_latitude,test_df.dropoff_longitude)


## === cell 14
train_df = train_df[train_df['distance']<15]


## === cell 15
train_df.describe()


## === cell 16
train_df = train_df[(train_df['passenger_count']!=0) & (train_df['passenger_count']<10)]


## === cell 17
train_df['hour'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
train_df['year'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)

test_df['hour'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
test_df['year'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)


## === cell 18
feat_cols_s = ['distance','passenger_count','hour','year']

X = train_df[feat_cols_s]
y = train_df['fare_amount']


## === cell 19
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


## === cell 20
from sklearn.ensemble import RandomForestRegressor


## === cell 21
r_reg= RandomForestRegressor(n_estimators=500)


## === cell 22
r_reg.fit(X_train,y_train)


## === cell 23
y_pred_final = r_reg.predict(test_df[feat_cols_s])

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('Random Forest regression.csv', index = False)


## === cell 24
import xgboost as xgb


## === cell 25
def XGBoost(X_train,X_test,y_train,y_test,num_rounds=300):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=num_rounds, 
                    early_stopping_rounds=20,evals=[(dtest,'test')],)


## === cell 26
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dtest_full = xgb.DMatrix(test_df[feat_cols_s])
if hasattr(xgbm, "best_iteration") and xgbm.best_iteration is not None:
    xgbm_pred = xgbm.predict(dtest_full, iteration_range=(0, xgbm.best_iteration + 1))
else:
    xgbm_pred = xgbm.predict(dtest_full)


## === cell 27
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount':xgbm_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('XGboost regression.csv', index = False)


## === cell 28
import operator

importance = xgbm.get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=["feature", "score"])
plt.figure()

df.plot(kind="barh", x="feature", y="score", legend=False, figsize=(10, 25))
plt.title("Feature Importance")


## === cell 29
import os
import sys
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf
    from packaging.version import parse as _vparse

    _pb_ver = _vparse(getattr(google.protobuf, "__version__", "0"))
    if _pb_ver.major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        importlib.invalidate_caches()
        if "google.protobuf" in sys.modules:
            del sys.modules["google.protobuf"]
except Exception:
    pass

try:
    import tensorflow as tf
except Exception:
    tf = None


## === cell 30
from sklearn.model_selection import train_test_split


## === cell 31
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=101)


## === cell 32
from sklearn.preprocessing import MinMaxScaler


## === cell 33
scaler = MinMaxScaler()


## === cell 34
scaler.fit(X_train)


## === cell 35
X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)


## === cell 36
X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)


## === cell 37
passenger_count = tf.feature_column.numeric_column('passenger_count')
distance = tf.feature_column.numeric_column('distance')
hour = tf.feature_column.numeric_column('hour')
year = tf.feature_column.numeric_column('year')


## === cell 38
feat_cols = [ passenger_count,distance]


## === cell 39
def input_func():
    features = {
        col: tf.convert_to_tensor(X_train[col].values) for col in X_train.columns
    }
    labels = tf.convert_to_tensor(y_train.values)
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = (
        ds.shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
        .repeat(1000)
        .batch(10)
    )
    return ds


## === cell 40
model = tf.estimator.DNNRegressor(hidden_units=[100,1000,100],feature_columns=feat_cols)


## --- ERROR in cell 40, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1105980672.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mmodel[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mestimator[0m[0;34m.[0m[0mDNNRegressor[0m[0;34m([0m[0mhidden_units[0m[0;34m=[0m[0;34m[[0m[0;36m100[0m[0;34m,[0m[0;36m1000[0m[0;34m,[0m[0;36m100[0m[0;34m][0m[0;34m,[0m[0mfeature_columns[0m[0;34m=[0m[0mfeat_cols[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: module 'tensorflow' has no attribute 'estimator'

## === cell 41
model.train(input_fn=input_func,steps=25000)
