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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

3.70811

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3726202364.py in <cell line: 0>()
     19     dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
     20     dataset['day'] = dataset.pickup_datetime.dt.day
---> 21     dataset['week'] = dataset.pickup_datetime.dt.week
     22     dataset['month'] = dataset.pickup_datetime.dt.month
     23     dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear

AttributeError: 'DatetimeProperties' object has no attribute 'week'

## === cell 9
features = [ 'pickup_latitude',  'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
             'haversine', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean', 'braycurtis',
             'minkowski',  'hamming', 'cityblock']

n = StandardScaler()
train[features] = n.fit_transform(train[features])
test[features] = n.transform(test[features])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1701697933.py in <cell line: 0>()
      5 n = StandardScaler()
      6 train[features] = n.fit_transform(train[features])
----> 7 test[features] = n.transform(test[features])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

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

KeyError: "['haversine', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean', 'braycurtis', 'minkowski', 'hamming', 'cityblock'] not in index"

## === cell 10
coords = ['pickup_latitude',  'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',]
concat = pd.concat([train[coords],test[coords]])
db = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True).fit(concat)
labels = db.labels_

train['cluster'] = labels[:train.shape[0]]
test['cluster'] = labels[train.shape[0]:]


## === cell 11
pca_features = [ 
             'haversine', 'chebyshev', 'euclidean', 'canberra',  'braycurtis',
             'minkowski',  'hamming', 'cityblock']

pca = PCA(n_components=3)
p_result = pca.fit_transform(train[pca_features])
train['pca0'] = p_result[:,0]
train['pca1'] = p_result[:,1]
train['pca2'] = p_result[:,2]

p_result = pca.transform(test[pca_features])
test['pca0'] = p_result[:,0]
test['pca1'] = p_result[:,1]
test['pca2'] = p_result[:,2]


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1713265058.py in <cell line: 0>()
      5 
      6 pca = PCA(n_components=3)
----> 7 p_result = pca.fit_transform(train[pca_features])
      8 train['pca0'] = p_result[:,0]
      9 train['pca1'] = p_result[:,1]

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in fit_transform(self, X, y)
    460         self._validate_params()
    461 
--> 462         U, S, Vt = self._fit(X)
    463         U = U[:, : self.n_components_]
    464 

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in _fit(self, X)
    483             )
    484 
--> 485         X = self._validate_data(
    486             X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
    487         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
PCA does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 12
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3114158282.py in <cell line: 0>()
      3 plt.figure(figsize=(20,20))
      4 plt.title('Pearson Correlation of Features', y=1.05, size=15)
----> 5 sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
      6             square=True, cmap=colormap, linecolor='white', annot=True)

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

ValueError: could not convert string to float: '2009-06-15 17:26:21.0000001'

## === cell 13
good_dist =  ['pca0','pca1','pca2','cluster','haversine','chebyshev','euclidean','canberra','braycurtis','minkowski','hamming','cityblock']
train_features_to_keep = ['fare_amount',] + ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'] + good_dist
train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)

test_features_to_keep = ['key',] + ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'] + good_dist
test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)

x_pred = test.drop('key', axis=1)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2088907711.py in <cell line: 0>()
      2 # Let's drop all the irrelevant features
      3 train_features_to_keep = ['fare_amount',] + ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'] + good_dist
----> 4 train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)
      5 
      6 test_features_to_keep = ['key',] + ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'] + good_dist

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given

## === cell 14
x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train['fare_amount'],random_state=123,test_size=0.2)


## === cell 15
def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse',
                            'eta':.3,
                            'max_depth':4,
                            'min_child_weight':3,
                           }
                    ,dtrain=matrix_train,num_boost_round=300,
                    early_stopping_rounds=10,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/657386467.py in <cell line: 0>()
     11     return model
     12 
---> 13 model=XGBmodel(x_train,x_test,y_train,y_test)

/tmp/ipykernel_11/657386467.py in XGBmodel(x_train, x_test, y_train, y_test)
      1 def XGBmodel(x_train,x_test,y_train,y_test):
----> 2     matrix_train = xgb.DMatrix(x_train,label=y_train)
      3     matrix_test = xgb.DMatrix(x_test,label=y_test)
      4     model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse',
      5                             'eta':.3,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1087         data = pd.DataFrame(data)
   1088     if _is_pandas_df(data):
-> 1089         return _from_pandas_df(
   1090             data, enable_categorical, missing, threads, feature_names, feature_types
   1091         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_pandas_df(data, enable_categorical, missing, nthread, feature_names, feature_types)
    520     feature_types: Optional[FeatureTypes],
    521 ) -> DispatchedDataBackendReturnType:
--> 522     data, feature_names, feature_types = _transform_pandas_df(
    523         data, enable_categorical, feature_names, feature_types
    524     )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:key: object, pickup_datetime: datetime64[ns, UTC]

## === cell 16
xgb.plot_importance(model)
plt.show()


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3635510548.py in <cell line: 0>()
----> 1 xgb.plot_importance(model)
      2 plt.show()

NameError: name 'model' is not defined

## === cell 17
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1597796262.py in <cell line: 0>()
----> 1 prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
      2 # Add to submission
      3 submission = pd.DataFrame({
      4         "key": test['key'],
      5         "fare_amount": prediction.round(2)

NameError: name 'model' is not defined

## === cell 18
submission


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351652756.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
