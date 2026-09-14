# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
from fastai.imports import *

from fastai.tabular.all import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display


## === cell 1
PATH = '../input'
df_raw = pd.read_csv(f'{PATH}/train.csv', nrows=10000000)


## === cell 2
def display_all(df):
    with pd.option_context('display.max_rows',1000):
        with pd.option_context('display.max_columns',1000):
            display(df)


## === cell 3
display_all(df_raw.head(5))


## === cell 4
add_datepart(df_raw,'pickup_datetime',drop=True, time=True)


## === cell 5
display_all(df_raw.head(5))


## === cell 6
def distance(data):
    data['longitutde_traversed'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['latitude_traversed'] = (data.dropoff_latitude - data.pickup_latitude).abs()


## === cell 7
distance(df_raw)


## === cell 8
display_all(df_raw.head(2).T)


## === cell 9
df_raw.isnull().sum()


## === cell 10
df_raw.dropna(axis=0, how='any', inplace=True)


## === cell 11
df_raw.shape


## === cell 12
key = df_raw.key
df_raw.drop('key', axis=1, inplace = True)


## === cell 13
df_raw.passenger_count.value_counts()


## === cell 14
df_raw = df_raw[(df_raw.passenger_count>0)&(df_raw.passenger_count<10)]


## === cell 15
len(df_raw)


## === cell 16
df_raw.reset_index(drop=True, inplace=True)


## === cell 17
outliers = []
for feature in df_raw.keys():
    Q1 = np.percentile(df_raw[feature],25,axis=0)
    Q3 = np.percentile(df_raw[feature],75,axis=0)
    step = 2*(Q3-Q1)
    feature_outlier = df_raw[~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))]
    outliers += feature_outlier.index.tolist()


## === cell 18
len(outliers)/len(df_raw)


## === cell 19
outliers = []
for feature in ['longitutde_traversed','latitude_traversed']:
    Q1 = np.percentile(df_raw[feature],25,axis=0)
    Q3 = np.percentile(df_raw[feature],75,axis=0)
    step = 10*(Q3-Q1)
    feature_outlier = df_raw[~((df_raw[feature] >= Q1 - step) & (df_raw[feature] <= Q3 + step))]
    outliers += feature_outlier.index.tolist()


## === cell 20
len(outliers)/len(df_raw)


## === cell 21
df = df_raw.drop(df_raw.index[outliers]).reset_index(drop = True)


## === cell 22
len(df)


## === cell 23
y = df_raw.fare_amount
df_raw.drop('fare_amount', axis=1, inplace = True)


## === cell 24
X_train, X_valid, y_train, y_valid = train_test_split(df_raw, y, test_size = 10000)


## === cell 25
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print(res)


## === cell 26
set_rf_samples(10000)


## === cell 27
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)


## === cell 28
test_set = pd.read_csv(f'{PATH}/test.csv')


## === cell 29
test_key = test_set.key
test_set.drop('key', axis = 1, inplace = True)


## === cell 30
add_datepart(test_set,'pickup_datetime',drop=True, time=True)
distance(test_set)


## === cell 31
test_predictions = m.predict(test_set)


## === cell 32
submission = pd.DataFrame({'key': test_key, 
                           'fare_amount': test_predictions})
submission.to_csv('submissions.csv', index=False)
