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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

3.5071

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%load_ext autoreload
%autoreload 2


## === cell 1
%matplotlib inline

from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2010211772.py in <cell line: 0>()
      2 
      3 from fastai.imports import *
----> 4 from fastai.structured import *
      5 from sklearn.ensemble import RandomForestRegressor
      6 from IPython.display import display

ModuleNotFoundError: No module named 'fastai.structured'

## === cell 2
set_plot_sizes(12,14,16)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3851686703.py in <cell line: 0>()
      1 # Set the plot sizes
----> 2 set_plot_sizes(12,14,16)

NameError: name 'set_plot_sizes' is not defined

## === cell 3
PATH = "../input/"

df_raw = pd.read_csv(f'{PATH}train.csv', nrows=100000, parse_dates=['pickup_datetime'], dtype={'passenger_count': 'int8', 'fare_amount': 'float16'}, 
                     usecols=['fare_amount', 'pickup_datetime','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count'])
df_raw_test = pd.read_csv(f'{PATH}test.csv', parse_dates=['pickup_datetime'], dtype={'passenger_count': 'int8'})


## === cell 4
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


## === cell 5
display_all(df_raw.tail().T)


## === cell 6
display_all(df_raw.describe(include='all').T)


## === cell 7
display_all(df_raw_test.describe(include='all').T)


## === cell 8
df_raw[df_raw['fare_amount'] < 0]


## === cell 9
df_raw[df_raw['pickup_longitude'] < -75]


## === cell 10
df_raw[df_raw['pickup_longitude'] > -73]


## === cell 11
df_raw[df_raw['pickup_latitude'] < 40]


## === cell 12
df_raw[df_raw['pickup_latitude'] > 42]


## === cell 13
df_raw.shape


## === cell 14
df_raw = df_raw[df_raw['pickup_longitude'] > -76]
df_raw = df_raw[df_raw['pickup_longitude'] < -73]
df_raw = df_raw[df_raw['pickup_latitude'] > 40]
df_raw = df_raw[df_raw['pickup_latitude'] < 44]


## === cell 15
df_raw.shape


## === cell 16
train_cats(df_raw)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1914754276.py in <cell line: 0>()
      1 # Converts all strings to categorical features
----> 2 train_cats(df_raw)

NameError: name 'train_cats' is not defined

## === cell 17
add_datepart(df_raw, 'pickup_datetime')


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4089705538.py in <cell line: 0>()
      1 # Splits dates into subcomponenets
----> 2 add_datepart(df_raw, 'pickup_datetime')

NameError: name 'add_datepart' is not defined

## === cell 18
df_raw.info()


## === cell 19
df, y, nas = proc_df(df_raw, 'fare_amount')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4095622633.py in <cell line: 0>()
      1 # Splits the data into independent and dependent features and keeps a column 'nas' that keeps track of features that had missing values and had to be imputed
----> 2 df, y, nas = proc_df(df_raw, 'fare_amount')

NameError: name 'proc_df' is not defined

## === cell 20
m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, oob_score=True, n_jobs=-1)
m.fit(df, y)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1175187180.py in <cell line: 0>()
----> 1 m = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, oob_score=True, n_jobs=-1)
      2 m.fit(df, y)

NameError: name 'RandomForestRegressor' is not defined

## === cell 21
fi = rf_feat_importance(m, df); fi[:10]


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1727231847.py in <cell line: 0>()
      1 # Calculate Feature Importance
----> 2 fi = rf_feat_importance(m, df); fi[:10]

NameError: name 'rf_feat_importance' is not defined

## === cell 22
fi.plot('cols', 'imp', figsize=(10,6), legend=False)
plt.title('Feature Importance by Feature');


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/812307312.py in <cell line: 0>()
      1 # Shows the table above visually
----> 2 fi.plot('cols', 'imp', figsize=(10,6), legend=False)
      3 plt.title('Feature Importance by Feature');

NameError: name 'fi' is not defined

## === cell 23
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


## === cell 24
plot_fi(fi[:30])
plt.title('Feature Importance by Feature');


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1042938549.py in <cell line: 0>()
      1 # Same visualisation but easier to see which feature contributes how much
----> 2 plot_fi(fi[:30])
      3 plt.title('Feature Importance by Feature');

NameError: name 'fi' is not defined

## === cell 25
from scipy.cluster import hierarchy as hc


## === cell 26
corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
plt.title('Feature Similarities')
plt.show()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3685217817.py in <cell line: 0>()
----> 1 corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
      2 corr_condensed = hc.distance.squareform(1-corr)
      3 z = hc.linkage(corr_condensed, method='average')
      4 fig = plt.figure(figsize=(16,10))
      5 dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)

NameError: name 'df' is not defined

## === cell 27
train_cats(df_raw_test)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2237697315.py in <cell line: 0>()
----> 1 train_cats(df_raw_test)

NameError: name 'train_cats' is not defined

## === cell 28
add_datepart(df_raw_test, 'pickup_datetime')


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/311486364.py in <cell line: 0>()
----> 1 add_datepart(df_raw_test, 'pickup_datetime')

NameError: name 'add_datepart' is not defined

## === cell 29
df_test = proc_df(df_raw_test)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2063015931.py in <cell line: 0>()
----> 1 df_test = proc_df(df_raw_test)

NameError: name 'proc_df' is not defined

## === cell 30
y_pred = m.predict(df_raw_test.drop('key', axis=1))


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1097555550.py in <cell line: 0>()
----> 1 y_pred = m.predict(df_raw_test.drop('key', axis=1))

NameError: name 'm' is not defined

## === cell 31
my_submission = pd.DataFrame({'key': df_raw_test.key, 'fare_amount': y_pred})
my_submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1991936778.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({'key': df_raw_test.key, 'fare_amount': y_pred})
      2 my_submission.to_csv('submission.csv', index=False)

NameError: name 'y_pred' is not defined
