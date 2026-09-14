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
seaborn==0.12.2
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

8.39274

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
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
%matplotlib inline  

import os
print(os.listdir("../input"))



## === cell 1
taxi_ride_train= pd.read_csv("../input/train.csv", sep=",", index_col="key", header=0, parse_dates=["pickup_datetime"], nrows=99999)
taxi_ride_test= pd.read_csv("../input/test.csv", sep=",", index_col="key", header=0, parse_dates=["pickup_datetime"])
taxi_ride_train.head()


## === cell 2
print("The shape train data are {0}".format((taxi_ride_train.shape)))
print("The shape test data are {0}".format((taxi_ride_test.shape)))


## === cell 3
taxi_ride_train.info()


## === cell 4
taxi_ride_test.info()


## === cell 5
taxi_ride_train.dtypes.value_counts().reset_index()


## === cell 6
taxi_ride_train.isnull().sum().sum()


## === cell 7
taxi_ride_test.isnull().sum().sum()


## === cell 8
taxi_ride_train=taxi_ride_train.dropna(axis=0)
taxi_ride_test=taxi_ride_test.dropna(axis=0)
print(taxi_ride_train.isnull().sum().sum())
print(taxi_ride_test.isnull().sum().sum())


## === cell 9
def calculate_distance(row):
    R = 6373.0 # approximate radius of earth in km
    lat1 = radians(row[0])
    lon1 = radians(row[1])
    lat2 = radians(row[2])
    lon2 = radians(row[3])
    longitude_distance = lon2 - lon1
    latitude_distance = lat2 - lat1
    a = sin(latitude_distance / 2)**2 + cos(lat1) * cos(lat2) * sin(longitude_distance / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


## === cell 10
taxi_ride_train['ride_distance_km']=taxi_ride_train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(calculate_distance, axis=1)
taxi_ride_test['ride_distance_km']=taxi_ride_test[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].apply(calculate_distance, axis=1)


## === cell 11
taxi_ride_train['ride_distance_km'].describe()


## === cell 12
sns.boxplot(taxi_ride_train['ride_distance_km'])


## === cell 13
IQR = taxi_ride_train.ride_distance_km.quantile(0.75) - taxi_ride_train.ride_distance_km.quantile(0.25)
Lower_fence = taxi_ride_train.ride_distance_km.quantile(0.25) - (IQR * 3)
Upper_fence = taxi_ride_train.ride_distance_km.quantile(0.75) + (IQR * 3)
print('Distance outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


## === cell 14
distance_outlier_train=len(taxi_ride_train[taxi_ride_train['ride_distance_km']>=30])
distance_outlier_test=len(taxi_ride_test[taxi_ride_test['ride_distance_km']>=30])
print("There are {0} trains rows and {1} test rows that have distance value more than 30km".format(distance_outlier_train,distance_outlier_test))


## === cell 15
taxi_ride_train['ride_distance_km'] = np.where(taxi_ride_train['ride_distance_km'].astype("float64") <= 30.0, taxi_ride_train['ride_distance_km'], 30.0)
taxi_ride_train['ride_distance_km'] = np.where(taxi_ride_train['ride_distance_km'].astype("float64") >= 0.0 , taxi_ride_train['ride_distance_km'], 0.0)

taxi_ride_test['ride_distance_km'] = np.where(taxi_ride_test['ride_distance_km'].astype("float64") <= 30.0, taxi_ride_test['ride_distance_km'], 30.0)
taxi_ride_test['ride_distance_km'] = np.where(taxi_ride_test['ride_distance_km'].astype("float64") >= 0.0 , taxi_ride_test['ride_distance_km'], 0.0)


## === cell 16
sns.boxplot(taxi_ride_train['ride_distance_km'])


## === cell 17
sns.jointplot(x="ride_distance_km", y="fare_amount", data=taxi_ride_train);


## === cell 18
pick_up_date_train = taxi_ride_train.ix[:,'pickup_datetime']
pick_up_date_test = taxi_ride_test.ix[:,'pickup_datetime']

temp_df_train=pd.DataFrame({"year": pick_up_date_train.dt.year,
              "month": pick_up_date_train.dt.month,
              "day": pick_up_date_train.dt.day,
              "hour": pick_up_date_train.dt.hour,
              "dayofyear": pick_up_date_train.dt.dayofyear,
              "week": pick_up_date_train.dt.week,
              "weekday": pick_up_date_train.dt.weekday,
              "quarter": pick_up_date_train.dt.quarter,
             })

temp_df_test=pd.DataFrame({"year": pick_up_date_test.dt.year,
              "month": pick_up_date_test.dt.month,
              "day": pick_up_date_test.dt.day,
              "hour": pick_up_date_test.dt.hour,
              "dayofyear": pick_up_date_test.dt.dayofyear,
              "week": pick_up_date_test.dt.week,
              "weekday": pick_up_date_test.dt.weekday,
              "quarter": pick_up_date_test.dt.quarter,
             })

taxi_ride_train= pd.concat([taxi_ride_train, temp_df_train], axis=1)
taxi_ride_test= pd.concat([taxi_ride_test, temp_df_test], axis=1)
taxi_ride_train.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_test.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_train.head()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2150746073.py in <cell line: 0>()
----> 1 pick_up_date_train = taxi_ride_train.ix[:,'pickup_datetime']
      2 pick_up_date_test = taxi_ride_test.ix[:,'pickup_datetime']
      3 
      4 temp_df_train=pd.DataFrame({"year": pick_up_date_train.dt.year,
      5               "month": pick_up_date_train.dt.month,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'ix'

## === cell 19
taxi_ride_train.dtypes.value_counts().reset_index()


## === cell 20
print("The new dataset contains {0} null entries ".format(taxi_ride_train.isnull().sum().sum()))


## === cell 21
sns.distplot(taxi_ride_train['fare_amount'])


## === cell 22
taxi_ride_train['fare_amount'].describe()


## === cell 23
length_before=len(taxi_ride_train)
taxi_ride_train= taxi_ride_train[taxi_ride_train.fare_amount>=0.0]
length_after=len(taxi_ride_train)
print("No of rows removed {0}".format(length_before-length_after))


## === cell 24
print("Skweness before transformation {0}".format( taxi_ride_train.fare_amount.skew()))
sns.distplot(np.log(taxi_ride_train['fare_amount']+1))
taxi_ride_train['fare_amount']=np.log(taxi_ride_train['fare_amount']+1)
print("Skweness after transformation {0}".format( taxi_ride_train.fare_amount.skew()))


## === cell 25
Y_train=taxi_ride_train.fare_amount
X_train=taxi_ride_train.drop("fare_amount", axis=1)
X_test=taxi_ride_test
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
print("Shape of training set is {0}".format(X_train.shape))
print("Shape of Validation set is {0}".format(X_valid.shape))
print("Shape of testing set is {0}".format(X_test.shape))


## === cell 26
discrete_col_list=[]
continous_col_list=[]
for col in X_train.columns.tolist():
    if(taxi_ride_train[col].value_counts().count()/len(taxi_ride_train)) < 0.1:
        discrete_col_list.append(col)
    else:
        continous_col_list.append(col)
print("The descrete column in our data are {0}".format(discrete_col_list))
print("The continous column in our data are {0}".format(continous_col_list))


## === cell 27
for var in continous_col_list:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = taxi_ride_train.boxplot(column=var)
    fig.set_title('')
    
    plt.subplot(1, 2, 2)
    fig = taxi_ride_train[var].hist(bins=20)
    fig.set_xlabel(var)
 
    plt.show()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/352889074.py in <cell line: 0>()
      3     plt.figure(figsize=(15,6))
      4     plt.subplot(1, 2, 1)
----> 5     fig = taxi_ride_train.boxplot(column=var)
      6     fig.set_title('')
      7 

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py in boxplot_frame(self, column, by, ax, fontsize, rot, grid, figsize, layout, return_type, backend, **kwargs)
    531 ):
    532     plot_backend = _get_plot_backend(backend)
--> 533     return plot_backend.boxplot_frame(
    534         self,
    535         column=column,

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/boxplot.py in boxplot_frame(self, column, by, ax, fontsize, rot, grid, figsize, layout, return_type, **kwds)
    490     import matplotlib.pyplot as plt
    491 
--> 492     ax = boxplot(
    493         self,
    494         column=column,

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/boxplot.py in boxplot(data, column, by, ax, fontsize, rot, grid, figsize, layout, return_type, **kwds)
    467             columns = data.columns
    468         else:
--> 469             data = data[columns]
    470 
    471         result = plot_group(columns, data.values.T, ax, **kwds)

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
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['pickup_datetime'], dtype='object')] are in the [columns]"

## === cell 28
latitude_upper_range=90.0
latitude_lower_range=-90.0
for var in ['pickup_latitude','dropoff_latitude']:
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") <= latitude_upper_range, taxi_ride_train[var], latitude_upper_range)
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") >= latitude_lower_range , taxi_ride_train[var], latitude_lower_range)
    
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") <= latitude_upper_range, taxi_ride_test[var], latitude_upper_range)
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") >= latitude_lower_range , taxi_ride_test[var], latitude_lower_range)
    
longitude_upper_range=180.0
longitude_lower_range=-180.0
for var in ['pickup_latitude','dropoff_latitude']:
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") <= longitude_upper_range, taxi_ride_train[var], longitude_upper_range)
    taxi_ride_train[var] = np.where(taxi_ride_train[var].astype("float64") >= longitude_lower_range , taxi_ride_train[var], longitude_lower_range)
    
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") <= longitude_upper_range, taxi_ride_test[var], longitude_upper_range)
    taxi_ride_test[var] = np.where(taxi_ride_test[var].astype("float64") >= longitude_lower_range , taxi_ride_test[var], longitude_lower_range)


## === cell 29
for var in continous_col_list:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = taxi_ride_train.boxplot(column=var)
    fig.set_title('')
    
    plt.subplot(1, 2, 2)
    fig = taxi_ride_train[var].hist(bins=20)
    fig.set_xlabel(var)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3942822589.py in <cell line: 0>()
      2     plt.figure(figsize=(15,6))
      3     plt.subplot(1, 2, 1)
----> 4     fig = taxi_ride_train.boxplot(column=var)
      5     fig.set_title('')
      6 

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py in boxplot_frame(self, column, by, ax, fontsize, rot, grid, figsize, layout, return_type, backend, **kwargs)
    531 ):
    532     plot_backend = _get_plot_backend(backend)
--> 533     return plot_backend.boxplot_frame(
    534         self,
    535         column=column,

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/boxplot.py in boxplot_frame(self, column, by, ax, fontsize, rot, grid, figsize, layout, return_type, **kwds)
    490     import matplotlib.pyplot as plt
    491 
--> 492     ax = boxplot(
    493         self,
    494         column=column,

/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/boxplot.py in boxplot(data, column, by, ax, fontsize, rot, grid, figsize, layout, return_type, **kwds)
    467             columns = data.columns
    468         else:
--> 469             data = data[columns]
    470 
    471         result = plot_group(columns, data.values.T, ax, **kwds)

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
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['pickup_datetime'], dtype='object')] are in the [columns]"

## === cell 30
sns.distplot(np.sqrt(taxi_ride_train["ride_distance_km"]))
taxi_ride_train["ride_distance_km"]=np.sqrt(taxi_ride_train["ride_distance_km"])
taxi_ride_test["ride_distance_km"]=np.sqrt(taxi_ride_test["ride_distance_km"])


## === cell 31
for i,var in enumerate(discrete_col_list):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    sns.countplot(taxi_ride_train[var], ax=ax)


## === cell 32
sns.pairplot(taxi_ride_train, x_vars=continous_col_list, y_vars='fare_amount', size=15, aspect=0.7, kind='reg')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1403791915.py in <cell line: 0>()
      1 #sns.pairplot(X_train[continous_col_list])
----> 2 sns.pairplot(taxi_ride_train, x_vars=continous_col_list, y_vars='fare_amount', size=15, aspect=0.7, kind='reg')

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in pairplot(data, hue, hue_order, palette, vars, x_vars, y_vars, kind, diag_kind, markers, height, aspect, corner, dropna, plot_kws, diag_kws, grid_kws, size)
   2159     elif kind == "reg":
   2160         from .regression import regplot  # Avoid circular import
-> 2161         plotter(regplot, **plot_kws)
   2162     elif kind == "kde":
   2163         from .distributions import kdeplot  # Avoid circular import

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in map_offdiag(self, func, **kwargs)
   1426                     if x_var != y_var:
   1427                         indices.append((i, j))
-> 1428             self._map_bivariate(func, indices, **kwargs)
   1429         return self
   1430 

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in _map_bivariate(self, func, indices, **kwargs)
   1566             if ax is None:  # i.e. we are in corner mode
   1567                 continue
-> 1568             self._plot_bivariate(x_var, y_var, ax, func, **kws)
   1569         self._add_axis_labels()
   1570 

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in _plot_bivariate(self, x_var, y_var, ax, func, **kwargs)
   1575         """Draw a bivariate plot on the specified axes."""
   1576         if "hue" not in signature(func).parameters:
-> 1577             self._plot_bivariate_iter_hue(x_var, y_var, ax, func, **kwargs)
   1578             return
   1579 

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs)
   1649 
   1650             if str(func.__module__).startswith("seaborn"):
-> 1651                 func(x=x, y=y, **kws)
   1652             else:
   1653                 func(x, y, **kws)

/usr/local/lib/python3.11/dist-packages/seaborn/regression.py in regplot(data, x, y, x_estimator, x_bins, x_ci, scatter, fit_reg, ci, n_boot, units, seed, order, logistic, lowess, robust, logx, x_partial, y_partial, truncate, dropna, x_jitter, y_jitter, label, color, marker, scatter_kws, line_kws, ax)
    757     scatter_kws["marker"] = marker
    758     line_kws = {} if line_kws is None else copy.copy(line_kws)
--> 759     plotter.plot(ax, scatter_kws, line_kws)
    760     return ax
    761 

/usr/local/lib/python3.11/dist-packages/seaborn/regression.py in plot(self, ax, scatter_kws, line_kws)
    366 
    367         if self.fit_reg:
--> 368             self.lineplot(ax, line_kws)
    369 
    370         # Label the axes

/usr/local/lib/python3.11/dist-packages/seaborn/regression.py in lineplot(self, ax, kws)
    411         """Draw the model."""
    412         # Fit the regression model
--> 413         grid, yhat, err_bands = self.fit_regression(ax)
    414         edges = grid[0], grid[-1]
    415 

/usr/local/lib/python3.11/dist-packages/seaborn/regression.py in fit_regression(self, ax, x_range, grid)
    197                 else:
    198                     x_min, x_max = ax.get_xlim()
--> 199             grid = np.linspace(x_min, x_max, 100)
    200         ci = self.ci
    201 

/usr/local/lib/python3.11/dist-packages/numpy/core/function_base.py in linspace(start, stop, num, endpoint, retstep, dtype, axis)
    127     # Convert float/complex array scalars to float, gh-3504
    128     # and make sure one can use variables that have an __array_interface__, gh-6634
--> 129     start = asanyarray(start) * 1.0
    130     stop  = asanyarray(stop)  * 1.0
    131 

TypeError: unsupported operand type(s) for *: 'Timestamp' and 'float'

## === cell 33
sns.heatmap(X_train.corr())


## === cell 34
taxi_ride_train.groupby("hour")['fare_amount'].sum().plot()


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/125975845.py in <cell line: 0>()
----> 1 taxi_ride_train.groupby("hour")['fare_amount'].sum().plot()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'hour'

## === cell 35
taxi_ride_train.groupby("weekday")['fare_amount'].sum().plot()


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/39793302.py in <cell line: 0>()
----> 1 taxi_ride_train.groupby("weekday")['fare_amount'].sum().plot()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'weekday'

## === cell 36
taxi_ride_train.groupby("passenger_count")['fare_amount'].sum().plot()


## === cell 37
taxi_ride_train.groupby("month")['fare_amount'].sum().plot()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2628464822.py in <cell line: 0>()
----> 1 taxi_ride_train.groupby("month")['fare_amount'].sum().plot()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'month'

## === cell 38
taxi_ride_train.groupby("year")['fare_amount'].sum().plot()


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3768683738.py in <cell line: 0>()
----> 1 taxi_ride_train.groupby("year")['fare_amount'].sum().plot()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'year'

## === cell 39
pd.crosstab(taxi_ride_train.quarter, len(taxi_ride_train.fare_amount), margins=True) # create a crosstab


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2260080391.py in <cell line: 0>()
----> 1 pd.crosstab(taxi_ride_train.quarter, len(taxi_ride_train.fare_amount), margins=True) # create a crosstab

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'quarter'

## === cell 40
constant_features = [
    feat for feat in taxi_ride_train.columns if taxi_ride_train[feat].std() == 0
]
print(constant_features)


## === cell 41
sel_ = SelectFromModel(RandomForestRegressor(n_estimators=100))
sel_.fit(X_train, Y_train)
selected_feat = X_train.columns[(sel_.get_support())]
print("So the feature that holds highest importance are {0}".format(list(selected_feat)))


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/108569635.py in <cell line: 0>()
      1 sel_ = SelectFromModel(RandomForestRegressor(n_estimators=100))
----> 2 sel_.fit(X_train, Y_train)
      3 selected_feat = X_train.columns[(sel_.get_support())]
      4 print("So the feature that holds highest importance are {0}".format(list(selected_feat)))

/usr/local/lib/python3.11/dist-packages/sklearn/feature_selection/_from_model.py in fit(self, X, y, **fit_params)
    355         else:
    356             self.estimator_ = clone(self.estimator)
--> 357             self.estimator_.fit(X, y, **fit_params)
    358 
    359         if hasattr(self.estimator_, "feature_names_in_"):

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

TypeError: float() argument must be a string or a real number, not 'Timestamp'

## === cell 42
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
print("The features that are corelated with each other are {0}".format(corr_features))
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_valid.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)


## === cell 43
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train) #  fit  the scaler to the train set and then transform it
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test) # transform (scale) the test set


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3838460949.py in <cell line: 0>()
      1 scaler = RobustScaler()
----> 2 X_train_scaled = scaler.fit_transform(X_train) #  fit  the scaler to the train set and then transform it
      3 X_valid_scaled = scaler.transform(X_valid)
      4 X_test_scaled = scaler.transform(X_test) # transform (scale) the test set

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
   1514         # at fit, convert sparse matrices to csc for optimized computation of
   1515         # the quantiles
-> 1516         X = self._validate_data(
   1517             X,
   1518             accept_sparse="csc",

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

TypeError: float() argument must be a string or a real number, not 'Timestamp'

## === cell 44
regr = linear_model.LinearRegression()
regr.fit(X_train_scaled, Y_train)
Y_valid_pred = regr.predict(X_valid_scaled)
Y_test_pred = regr.predict(X_test_scaled)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_valid_pred))
print('Variance score: %.2f' % r2_score(Y_valid, Y_valid_pred))


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637454491.py in <cell line: 0>()
      1 regr = linear_model.LinearRegression()
----> 2 regr.fit(X_train_scaled, Y_train)
      3 Y_valid_pred = regr.predict(X_valid_scaled)
      4 Y_test_pred = regr.predict(X_test_scaled)
      5 print('Coefficients: \n', regr.coef_)

NameError: name 'X_train_scaled' is not defined

## === cell 45
def generate_residual_plot(label, prediction, type):
    plt.scatter(prediction, np.subtract(label, prediction))  # scatter plot
    title = 'Residual plot for predicting ' + type
    plt.title(title)  # set title
    plt.xlabel("Fitted Value")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.hlines(y=0, xmin=min(prediction), xmax=max(prediction), colors='orange', linewidth=3)  # plot ref line


## === cell 46
def generate_actual_vs_predicted_plot(label, prediction, type):
    plt.scatter(prediction, label, s=30, c='r', marker='+', zorder=10)  # scatter plot
    title = 'Actual vs Predicted values for ' + type
    plt.title(title)  # set title
    plt.xlabel("Predicted Values from model")  # set the xlabel
    plt.ylabel("Actual Values")  # set the ylabel
    plt.tight_layout()


## === cell 47
generate_residual_plot(Y_valid, Y_valid_pred,
                       "Taxi fares")


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/27475403.py in <cell line: 0>()
----> 1 generate_residual_plot(Y_valid, Y_valid_pred,
      2                        "Taxi fares")

NameError: name 'Y_valid_pred' is not defined

## === cell 48
generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred,
                       "Taxi fares")


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/393070165.py in <cell line: 0>()
----> 1 generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred,
      2                        "Taxi fares")

NameError: name 'Y_valid_pred' is not defined

## === cell 49
params = {'n_estimators': 700, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train_scaled, Y_train)
mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
print("MSE: %.4f" % mse)
print('Variance score: %.2f' % r2_score(Y_valid, clf.predict(X_valid_scaled)))


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1101869398.py in <cell line: 0>()
      3 clf = ensemble.GradientBoostingRegressor(**params)
      4 
----> 5 clf.fit(X_train_scaled, Y_train)
      6 mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
      7 print("MSE: %.4f" % mse)

NameError: name 'X_train_scaled' is not defined

## === cell 50
test_pred=pd.DataFrame(clf.predict(X_test_scaled), index=X_test.index)
test_pred.columns=["fare_amount"]
test_pred['fare_amount']= np.exp(test_pred.fare_amount)
test_pred.to_csv("my_submission.csv")


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1168410605.py in <cell line: 0>()
----> 1 test_pred=pd.DataFrame(clf.predict(X_test_scaled), index=X_test.index)
      2 test_pred.columns=["fare_amount"]
      3 test_pred['fare_amount']= np.exp(test_pred.fare_amount)
      4 test_pred.to_csv("my_submission.csv")

NameError: name 'X_test_scaled' is not defined

## === cell 51
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3957820875.py in <cell line: 0>()
      1 # Plot feature importance
----> 2 feature_importance = clf.feature_importances_
      3 # make importances relative to max importance
      4 feature_importance = 100.0 * (feature_importance / feature_importance.max())
      5 sorted_idx = np.argsort(feature_importance)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in feature_importances_(self)
    742             array of zeros.
    743         """
--> 744         self._check_initialized()
    745 
    746         relevant_trees = [

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _check_initialized(self)
    380     def _check_initialized(self):
    381         """Check that the estimator is initialized, raising an error if not."""
--> 382         check_is_fitted(self)
    383 
    384     def fit(self, X, y, sample_weight=None, monitor=None):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 52
generate_actual_vs_predicted_plot(Y_valid, clf.predict(X_valid_scaled),
                       "Taxi fares")


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3491330394.py in <cell line: 0>()
----> 1 generate_actual_vs_predicted_plot(Y_valid, clf.predict(X_valid_scaled),
      2                        "Taxi fares")

NameError: name 'X_valid_scaled' is not defined

## === cell 53
generate_residual_plot(Y_valid, clf.predict(X_valid_scaled),
                       "Taxi fares")


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4139521057.py in <cell line: 0>()
----> 1 generate_residual_plot(Y_valid, clf.predict(X_valid_scaled),
      2                        "Taxi fares")

NameError: name 'X_valid_scaled' is not defined
