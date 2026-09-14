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

3.87285

# 6. Current score

6.89829

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 1
train_df =  pd.read_csv('../input/train.csv', nrows = 1_000_000)


## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


## === cell 3
train_df['distance'] = haversine_np(train_df['pickup_longitude'], train_df['pickup_latitude'], 
                                    train_df['dropoff_longitude'], train_df['dropoff_latitude'])


## === cell 4
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime']) 


## === cell 5
train_df['year'] = train_df['pickup_datetime'].dt.year
train_df['month'] = train_df['pickup_datetime'].dt.month
train_df['day'] = train_df['pickup_datetime'].dt.day
train_df['hour'] = train_df['pickup_datetime'].dt.hour
train_df['minute'] = train_df['pickup_datetime'].dt.minute


## === cell 6
print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


## === cell 7
new_york_lat = 40
new_york_long = -74
train_df.describe()


## === cell 8
cond = True
for col in {'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'}:
    cond &= abs(train_df[col] - train_df[col].mean()) < 5


## === cell 9
print('Old size: %d' % len(train_df))
train_df = train_df[cond]
print('New size: %d' % len(train_df))


## === cell 10
train_df.describe()


## === cell 11
X = train_df[['distance','year','month','day','hour']].values
Y = train_df['fare_amount'].values


## === cell 12
from sklearn.ensemble import RandomForestRegressor


## === cell 13
kwargs = {'bootstrap': True,
 'max_depth': None,
 'max_features': 3,
 'min_samples_leaf': 9,
 'min_samples_split': 2}
rand_regr = RandomForestRegressor(n_estimators=20, **kwargs)


## === cell 14
rand_regr.fit(X, Y)


## === cell 15
y_pred = rand_regr.predict(X)
print('chi squared  rand forest with date %s' % (np.sum((Y-y_pred)**2.)/len(Y))**0.5)


## === cell 16
rand_regr.score(X,Y)


## === cell 27
test_df =  pd.read_csv('../input/test.csv')


## === cell 28
test_df['distance'] = haversine_np(test_df['pickup_longitude'], test_df['pickup_latitude'], 
                                    test_df['dropoff_longitude'], test_df['dropoff_latitude'])


## === cell 29
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime']) 


## === cell 30
test_df['year'] = test_df['pickup_datetime'].dt.year
test_df['month'] = test_df['pickup_datetime'].dt.month
test_df['day'] = test_df['pickup_datetime'].dt.day
test_df['hour'] = test_df['pickup_datetime'].dt.hour
test_df['minute'] = test_df['pickup_datetime'].dt.minute


## === cell 31
X_to_pred = test_df[['distance','year','month','day','hour']].values
y_pred = rand_regr.predict(X_to_pred)


## === cell 32
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)


## === cell 34
with sns.axes_style("white"):
    sns.jointplot(x=X.flatten(), y=Y, kind="hex", color="k", bins='log');


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3004148828.py in <cell line: 0>()
      1 with sns.axes_style("white"):
----> 2     sns.jointplot(x=X.flatten(), y=Y, kind="hex", color="k", bins='log');

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in jointplot(data, x, y, hue, kind, height, ratio, space, dropna, xlim, ylim, color, palette, hue_order, hue_norm, marginal_ticks, joint_kws, marginal_kws, **kwargs)
   2239 
   2240     # Initialize the JointGrid object
-> 2241     grid = JointGrid(
   2242         data=data, x=x, y=y, hue=hue,
   2243         palette=palette, hue_order=hue_order, hue_norm=hue_norm,

/usr/local/lib/python3.11/dist-packages/seaborn/axisgrid.py in __init__(self, data, x, y, hue, height, ratio, space, palette, hue_order, hue_norm, dropna, xlim, ylim, marginal_ticks)
   1720 
   1721         # Process the input variables
-> 1722         p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
   1723         plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]
   1724 

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
    960         # Construct a tidy plot DataFrame. This will convert a number of
    961         # types automatically, aligning on index in case of pandas objects
--> 962         plot_data = pd.DataFrame(plot_data)
    963 
    964         # Reduce the variables dictionary to fields with valid data

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 35
X = X.flatten()
mask = (X <50) & (Y <100)

with sns.axes_style("white"):
    p = sns.jointplot(x=X[mask], y=Y[mask], kind="hex", color="k", bins='log');

x=np.arange(0,50)
y=regr.predict(x.reshape(-1,1))
p.ax_joint.plot(x,y)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/718517359.py in <cell line: 0>()
      1 X = X.flatten()
----> 2 mask = (X <50) & (Y <100)
      3 # mask
      4 # fig, ax = plt.subplots()
      5 

ValueError: operands could not be broadcast together with shapes (4897470,) (979494,)
