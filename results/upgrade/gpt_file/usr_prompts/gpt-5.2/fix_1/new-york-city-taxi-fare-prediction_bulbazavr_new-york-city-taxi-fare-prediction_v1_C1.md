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

3.12

# 3. Installed packages



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

5.67556

# 6. Current score

6.88544

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


## === cell 2
df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=25000)
df_test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')
df.head()


## === cell 3
df.shape


## === cell 4
df.info()


## === cell 5
df.describe()


## === cell 6
df.isnull().sum()


## === cell 7
df_test.isnull().sum()


## === cell 8
df.nunique()


## === cell 9
df.duplicated().sum()


## === cell 10
df.dropna(axis=0, inplace=True)
np.sum(pd.isnull(df))


## === cell 11
df['fare_amount'][df['fare_amount']<0] = 0.1
df[df['fare_amount']<0]


## === cell 12
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
df_test['pickup_datetime'] = pd.to_datetime(df_test.pickup_datetime)


## === cell 13
df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.day_name()
df.loc[:, 'pickup_date'] = df['pickup_datetime'].dt.day
df.loc[:, 'pickup_month'] = df['pickup_datetime'].dt.month
df.loc[:, 'pickup_day'] = df['pickup_datetime'].dt.dayofweek
df_test.loc[:, 'pickup_hour'] = df_test['pickup_datetime'].dt.hour
df_test.loc[:, 'pickup_weekday'] = df_test['pickup_datetime'].dt.day_name()
df_test.loc[:, 'pickup_date'] = df_test['pickup_datetime'].dt.day
df_test.loc[:, 'pickup_month'] = df_test['pickup_datetime'].dt.month
df_test.loc[:, 'pickup_day'] = df_test['pickup_datetime'].dt.dayofweek


## === cell 14
def baseFare(x):
    if x in range(16,20):
        base_fare = 3.50
    elif x in range(20,24):
        base_fare = 3
    else:
        base_fare = 2.50
    return base_fare

df['base_fare'] = df['pickup_hour'].apply(baseFare)
df_test['base_fare'] = df_test['pickup_hour'].apply(baseFare)
df['base_fare'], df['pickup_hour']


## === cell 15
df['fare'] = df['fare_amount'] - df['base_fare']


## === cell 16
from geopy.distance import great_circle
coordA=(df['pickup_latitude'][0], df['pickup_longitude'][0])
coordB=(df['dropoff_latitude'][0], df['dropoff_longitude'][0])
print (int(great_circle(coordA, coordB).kilometers))


## === cell 17
from math import radians, cos, sin, asin, sqrt
def haversineDistanceInKM(latA, lonA, latB, lonB):
    lonA, latA, lonB, latB = map(radians, [lonA, latA, lonB, latB])
    return int(12734 * asin(sqrt(
      sin((latB-latA)/2)**2+cos(latA)*cos(latB)*sin((lonB-lonA)/2)**2)))


latA = df['pickup_latitude'][0]
lonA = df['pickup_longitude'][0]
latB = df['dropoff_latitude'][0]
lonB = df['dropoff_longitude'][0]
print(haversineDistanceInKM(latA, lonA, latB, lonB))


## === cell 18
def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

df['haversine_distance'] = haversine_distance(df['pickup_latitude'].values, 
                                                     df['pickup_longitude'].values, 
                                                     df['dropoff_latitude'].values, 
                                                     df['dropoff_longitude'].values)
df_test['haversine_distance'] = haversine_distance(df_test['pickup_latitude'].values, 
                                                     df_test['pickup_longitude'].values, 
                                                     df_test['dropoff_latitude'].values, 
                                                     df_test['dropoff_longitude'].values)


## === cell 19
df['haversine_distance'].median(), df['haversine_distance'].mean(), 


## === cell 20
df.head()


## === cell 21
import sklearn.neighbors
dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
dist_miles = (dist.pairwise
    (np.radians(df[['pickup_latitude', 'pickup_longitude']]),
     np.radians(df[['dropoff_latitude','dropoff_longitude']]))*3959)
dist_km = (dist.pairwise
    (np.radians(df[['pickup_latitude', 'pickup_longitude']]),
     np.radians(df[['dropoff_latitude','dropoff_longitude']]))*6371)
df_dist_km = pd.DataFrame(dist_km)
df_dist_km.head()


## === cell 22
from sklearn.metrics.pairwise import haversine_distances
pickup_in_radians = np.radians(df[['pickup_latitude', 'pickup_longitude']])
dropoff_in_radians = np.radians(df[['dropoff_latitude','dropoff_longitude']])
result = pd.DataFrame(haversine_distances(pickup_in_radians, dropoff_in_radians)*6371)
result.head()


## === cell 23
mydiagonal = np.matrix.diagonal(np.array(result))
distance = pd.DataFrame(mydiagonal, index = df.index, columns = ['distance'])
distance.head()


## === cell 24
plt.figure(figsize=(22, 6))

plt.subplot(221)
sb.countplot(df['pickup_hour'])
plt.xlabel('Hour of Day')
plt.ylabel('Total number of pickups')
plt.title('Hourly Variation of Total number of pickups')

plt.subplot(223)
sb.countplot(df['pickup_date'])
plt.xlabel('Date')
plt.ylabel('Total number of pickups')
plt.title('Daily Variation of Total number of pickups')

plt.subplot(222)
sb.countplot(df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Week Day')
plt.ylabel('Total Number of pickups')
plt.title('Weekly Variation of Total number of pickups')

plt.subplot(224)
sb.countplot(df['pickup_month'])
plt.xlabel('Month')
plt.ylabel('Total number of pickups')
plt.title('Monthly Variation of Total number of pickups');


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4074725406.py in <cell line: 0>()
     18 # Day of week
     19 plt.subplot(222)
---> 20 sb.countplot(df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
     21                                            'Thursday', 'Friday', 'Saturday', 'Sunday'])
     22 plt.xlabel('Week Day')

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    479                 if order is not None:
    480                     error = "Input data must be a pandas object to reorder"
--> 481                     raise ValueError(error)
    482 
    483                 # The input data is an array

ValueError: Input data must be a pandas object to reorder

## === cell 25
plt.figure(figsize=(22, 6))

plt.subplot(121)
sb.countplot(df['passenger_count'])
plt.xlabel('Passenger Count')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Passenger Count')

plt.subplot(122)
sb.boxplot(df['passenger_count'], color = 'cyan', showmeans=True, 
           meanprops={"marker":"o", "markerfacecolor":"Red", 
                      "markeredgecolor":"black","markersize":"10"}
)
plt.xlabel('Passenger Count')
plt.title('Box plot of Passenger count');


## === cell 26
plt.figure(figsize=(22, 6))
sb.boxplot(x = df['passenger_count'],y = df['fare_amount'], color = 'cyan', showmeans=True, 
            meanprops={"marker":"o", "markerfacecolor":"Red", "markeredgecolor":"black","markersize":"10"}
)
plt.xlabel('Passenger Count')
plt.title ("Fare amount vs No. of passengers");


## === cell 27
plt.figure(figsize=(15, 6))
sb.boxplot(x = df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'],y = df['passenger_count'], color = 'cyan', showmeans=True, 
            meanprops={"marker":"o", "markerfacecolor":"Red", "markeredgecolor":"black","markersize":"10"}
)
plt.xlabel('Passenger Count')
plt.title ("No. of passengers vs Days of week");


## === cell 28
plt.figure(figsize=(22, 8))

plt.subplot(221)
sb.barplot(df['pickup_hour'], y = df['passenger_count'], palette = 'hsv')
plt.xlabel('Hour of Day')
plt.ylabel('Passenger count')
plt.title ("Passenger count vs Hour of Day")

plt.subplot(222)
sb.barplot(df['pickup_month'], y = df['passenger_count'],palette = 'hsv')
plt.xlabel('Month')
plt.ylabel('Passenger count')
plt.title ("Passenger count vs Month")

plt.subplot(223)
sb.barplot(x = df['pickup_date'], y = df['passenger_count'], palette = 'hsv')
plt.xlabel('Date')
plt.ylabel('Passenger count')
plt.title ("Passenger count vs Date")

plt.subplot(224)
sb.barplot(x = df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'],
           y = df['passenger_count'], palette = 'hsv')
plt.xlabel('Days of week')
plt.ylabel('Passenger count')
plt.title ("Passenger count vs Days of week")
plt.tight_layout();


## === cell 29
plt.figure(figsize=(15, 6))
sb.boxplot(x = df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'],
           y = df['fare_amount'], palette = 'rainbow', showmeans=True, 
            meanprops={"marker":"o", "markerfacecolor":"Red", "markeredgecolor":"black",
                       "markersize":"10"}
)
plt.xlabel('Fare amount')
plt.title ("Fare amount vs Days of week");


## === cell 30
sb.distplot(df['haversine_distance'], bins = 20);


## === cell 31
df['haversine_distance'].describe(), print("Median       ", df['haversine_distance'].median())


## === cell 32
df['haversine_distance'].quantile(0.25), df['haversine_distance'].quantile(0.75)


## === cell 33
IQR = df['haversine_distance'].quantile(0.75) - df['haversine_distance'].quantile(0.25)
IQR


## === cell 34
Q1 = df['haversine_distance'].quantile(0.25)
Q3 = df['haversine_distance'].quantile(0.75)
whisker_1 = Q1 - (1.5*IQR)
whisker_2 = Q3 + (1.5*IQR)

whisker_1, whisker_2


## === cell 35
df = df.loc[(df['haversine_distance']!=0) & (df['haversine_distance']<8)]
df.shape


## === cell 36
sb.distplot(df['haversine_distance'], bins = 20)
plt.show()


## === cell 37
from scipy import stats
x = df['haversine_distance']
y = df['fare_amount']
slope, intercept, r_value, p_value, std_err = stats.linregress(df['haversine_distance'],df['fare_amount'])
ax = sb.regplot(x, y, line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept), 
                                "color": "red"},scatter_kws={"color": "cyan"})
ax.legend();


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2460898254.py in <cell line: 0>()
      3 y = df['fare_amount']
      4 slope, intercept, r_value, p_value, std_err = stats.linregress(df['haversine_distance'],df['fare_amount'])
----> 5 ax = sb.regplot(x, y, line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept), 
      6                                 "color": "red"},scatter_kws={"color": "cyan"})
      7 ax.legend();

TypeError: regplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 2 keyword-only arguments) were given

## === cell 38
sb.relplot(x="haversine_distance", y="fare_amount", data=df, kind="scatter");


## === cell 39
plt.figure(figsize=(22, 8))

plt.subplot(221)
sb.barplot(df['pickup_hour'], y = df['haversine_distance'], palette = 'tab20')
plt.xlabel('Hour of Day')
plt.ylabel('Distance in Km')
plt.title ("Distance in Km vs Hour of Day")

plt.subplot(222)
sb.barplot(df['pickup_month'], y = df['haversine_distance'],palette = 'tab20',estimator = np.mean)
plt.xlabel('Month')
plt.ylabel('Distance in Km')
plt.title ("Distance in Km vs Month")

plt.subplot(223)
sb.barplot(x = df['pickup_date'], y = df['haversine_distance'], palette = 'tab20')
plt.xlabel('Date')
plt.ylabel('Distance in Km')
plt.title ("Distance in Km vs Date")

plt.subplot(224)
sb.barplot(x = df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'],
           y = df['haversine_distance'], palette = 'tab10')
plt.xlabel('Days of week')
plt.ylabel('Distance in Km')
plt.title ("Distance in Km vs Days of week")
plt.tight_layout();


## === cell 40
plt.figure(figsize=(22, 8))

plt.subplot(221)
sb.barplot(df['pickup_hour'], y = df['fare_amount'], palette = 'tab20')
plt.xlabel('Hour of Day')
plt.ylabel('Fare amount')
plt.title ("Fare amount vs Hour of Day")

plt.subplot(222)
sb.barplot(df['pickup_month'], y = df['fare_amount'],palette = 'tab20')
plt.xlabel('Month')
plt.ylabel('Fare amount')
plt.title ("Fare amount vs Month")

plt.subplot(223)
sb.barplot(x = df['pickup_date'], y = df['fare_amount'], palette = 'tab20')
plt.xlabel('Date')
plt.ylabel('Fare amount')
plt.title ("Fare amount vs Date")

plt.subplot(224)
sb.barplot(x = df['pickup_weekday'], order = ['Monday', 'Tuesday', 'Wednesday', 
                                           'Thursday', 'Friday', 'Saturday', 'Sunday'],
           y = df['fare_amount'], palette = 'tab10')
plt.xlabel('Days of week')
plt.ylabel('Fare amount')
plt.title ("Fare amount vs Days of week")
plt.tight_layout();


## === cell 41
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)
sb.despine(left=True)
sb.distplot(df['pickup_latitude'].values, label = 'pickup_latitude',color="b",bins = 100, ax=axes[0,0])
sb.distplot(df['pickup_longitude'].values, label = 'pickup_longitude',color="r",bins =100, ax=axes[1,0])
sb.distplot(df['dropoff_latitude'].values, label = 'dropoff_latitude',color="b",bins =100, ax=axes[0,1])
sb.distplot(df['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins =100, ax=axes[1,1])
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


## === cell 42
df = df.loc[(df.pickup_latitude > 40.6) & (df.pickup_latitude < 40.9)]
df = df.loc[(df.dropoff_latitude>40.6) & (df.dropoff_latitude < 40.9)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]
df_data_new = df.copy()
sb.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)#
sb.despine(left=True)
sb.distplot(df_data_new['pickup_latitude'].values, label = 'pickup_latitude',color="b",bins = 100, ax=axes[0,0])
sb.distplot(df_data_new['pickup_longitude'].values, label = 'pickup_longitude',color="r",bins =100, ax=axes[0,1])
sb.distplot(df_data_new['dropoff_latitude'].values, label = 'dropoff_latitude',color="b",bins =100, ax=axes[1, 0])
sb.distplot(df_data_new['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()

plt.show()


## === cell 43
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
hv.Distribution(df['fare_amount']).opts(title="Fare Amount Distribution", color="red",
                                                        xlabel="Fare Amount", ylabel="Density")\
.opts(opts.Distribution(width=700, height=300,tools=['hover'],show_grid=True))


## === cell 44
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
X =df.drop(['key', 'pickup_datetime','pickup_weekday', 'fare_amount', 'base_fare', 'fare'], axis = 1)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  
print(vif_data)


## === cell 45
plt.figure(figsize = (12,6))
sb.heatmap(df.drop(['key', 'pickup_datetime','pickup_weekday'], axis = 1).corr(), 
           cmap ='BuGn', annot = True);


## === cell 46
df[['pickup_latitude', 'pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']].corr()


## === cell 47
df.info()


## === cell 48
X = df.drop(['key', 'pickup_datetime','pickup_weekday', 'fare_amount', 'fare', 'base_fare', 'dropoff_latitude', 'dropoff_longitude'], 
            axis = 1)
y = df['fare_amount']


## === cell 49
from sklearn import preprocessing
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


## === cell 50
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.ndim)
print(y_train.ndim)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


## === cell 51
from sklearn.metrics import mean_squared_error
from math import sqrt
mean_pred = np.repeat(y_train.mean(),len(y_test))
sqrt(mean_squared_error(y_test, mean_pred))


## === cell 52
pip install -q --upgrade linear-tree


## === cell 53
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
np.mean(cross_val_score(lr, X_train, y_train, cv=5))


## === cell 54
from sklearn.model_selection import cross_validate
cv_results = cross_validate(lr, X_train, y_train, cv=5, return_estimator=True)

Coefficient = []
for model in cv_results['estimator']:
    Coefficient.append(model.coef_)
df_train = df.drop(['key', 'pickup_datetime','pickup_weekday', 'fare_amount', 'fare', 'base_fare', 
                    'dropoff_latitude', 'dropoff_longitude'], axis = 1)
coefficient = pd.DataFrame(Coefficient, columns = df_train.columns)
abs(coefficient.mean(axis =0)).sort_values(ascending = False)


## === cell 55
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(cv=5).fit(X_train, y_train)
ridge.score(X_train, y_train)


## === cell 56
from sklearn.model_selection import cross_validate
cv_results = cross_validate(ridge, X_train, y_train, cv=5, return_estimator=True)

Coefficient = []
for model in cv_results['estimator']:
    Coefficient.append(model.coef_)

coefficient = pd.DataFrame(Coefficient, columns = df_train.columns)
abs(coefficient.mean(axis =0)).sort_values(ascending = False)


## === cell 57
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X_train, y_train)
lasso.score(X_train, y_train)


## === cell 58
from sklearn.model_selection import cross_validate
cv_results = cross_validate(lasso, X_train, y_train, cv=5, return_estimator=True)

Coefficient = []
for model in cv_results['estimator']:
    Coefficient.append(model.coef_)
coefficient = pd.DataFrame(Coefficient, columns = df_train.columns)
abs(coefficient.mean(axis =0)).sort_values(ascending = False)


## === cell 59
from sklearn.linear_model import ElasticNetCV
elastic = ElasticNetCV(cv=5).fit(X_train, y_train)
elastic.score(X_train, y_train)


## === cell 60
from sklearn.model_selection import cross_validate
cv_results = cross_validate(elastic, X_train, y_train, cv=5, return_estimator=True)

Coefficient = []
for model in cv_results['estimator']:
    Coefficient.append(model.coef_)
coefficient = pd.DataFrame(Coefficient, columns = df_train.columns)
abs(coefficient.mean(axis =0)).sort_values(ascending = False)


## === cell 61
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
cv_score=[]
for i in range(1,4):
    poly_reg = PolynomialFeatures(degree = i)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg = LinearRegression()
    cv_score.append(np.mean(cross_val_score(poly_reg,X_poly,y_train,cv=5)))
x = range(1,4)
plt.scatter(x,cv_score)
plt.xticks(ticks=[1,2,3], labels=['Degree_1', 'Degree_2', 'Degree_3']);


## === cell 62
from sklearn.model_selection import cross_validate
cv_results = cross_validate(poly_reg, X_train, y_train, cv=5, return_estimator=True)

Coefficient = []
for model in cv_results['estimator']:
    Coefficient.append(model.coef_)
coefficient = pd.DataFrame(Coefficient, columns = df_train.columns)
abs(coefficient.mean(axis =0)).sort_values(ascending = False)


## === cell 63
from sklearn.neighbors import KNeighborsRegressor
cv_score=[]
for i in range(1,10):
 knn = KNeighborsRegressor(n_neighbors= i)
 cv_score.append(np.mean(cross_val_score(knn,X_train, y_train,cv=5)))
x = range(1,10)
plt.scatter(x,cv_score);


## === cell 64
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()
R_Squared = np.mean(cross_val_score(DT, X_train, y_train, cv=5))
Standard_deviation = np.std(cross_val_score(DT, X_train, y_train, cv=5))
print('R2 of Decision Tree Regression model is:',R_Squared)
print('Standard deviation of R2 of Decision Tree Regression model is:',Standard_deviation)


## === cell 65
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
R_Squared = np.mean(cross_val_score(rf, X_train, y_train, cv=5))
Standard_deviation = np.std(cross_val_score(rf, X_train, y_train, cv=5))
print('R2 of Random Forest Regression model is:',R_Squared)
print('Standard deviation of R2 of Random Forest Regression model is:',Standard_deviation)


## === cell 66
from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor()
np.mean(cross_val_score(GB, X_train, y_train, cv=5))


## === cell 67
df['haversine_distance_log'] = np.log(df['haversine_distance'].values + 1)
df['haversine_distance_sqrt'] = np.sqrt(df['haversine_distance'].values)
df['haversine_distance_sq'] = df['haversine_distance'].values**2
df_test['haversine_distance_log'] = np.log(df_test['haversine_distance'].values + 1)
df_test['haversine_distance_sqrt'] = np.sqrt(df_test['haversine_distance'].values)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)#
sb.despine(left=True)
sb.distplot(df['haversine_distance'], label = 'haversine_distance',color="b",bins = 100, ax=axes[0,0])
axes[0,0].set_title('Histogram of distance')
sb.distplot(df['haversine_distance_log'], label = 'haversine_distance_log',color="yellow",bins =100, ax=axes[0,1])
axes[0,1].set_title('Histogram of log of distance')
sb.distplot(df['haversine_distance_sqrt'], label = 'haversine_distance_sqrt',color="magenta",bins =100, ax=axes[1, 0])
axes[1,0].set_title('Histogram of sqrt of distance')
sb.distplot(df['haversine_distance_sq'], label = 'haversine_distance_sq',color="green",bins =100, ax=axes[1, 1])
axes[1,1].set_title('Histogram of square of distance')
plt.setp(axes, yticks=[])
plt.tight_layout()

plt.show()


## === cell 68
df_train = df.drop(['key', 'pickup_datetime','pickup_weekday', 'fare', 'fare_amount', 'base_fare',
                    'haversine_distance', 'haversine_distance_sq', 'haversine_distance_sqrt'], axis = 1)
df_test_copy = df_test.drop(['key', 'base_fare', 'pickup_datetime','pickup_weekday', 'base_fare','haversine_distance', 
                             'haversine_distance_sqrt'], axis = 1)
X = df_train.copy()
y = df['fare_amount']
df_train.columns, df_test_copy.columns


## === cell 69
scaler = preprocessing.StandardScaler()
X= scaler.fit(X).transform(X)
test_X= scaler.transform(df_test_copy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.model_selection import cross_validate
cv_results = cross_validate(ridge, X_train, y_train, cv=5, return_estimator=True)

Coefficient = []
for model in cv_results['estimator']:
    Coefficient.append(model.coef_)

coefficient = pd.DataFrame(Coefficient, columns = df_train.columns)
abs(coefficient.mean(axis =0)).sort_values(ascending = False)


## === cell 70
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(cv=5).fit(X_train, y_train)
ridge.score(X_train, y_train)


## === cell 71
df_test.head()


## === cell 72
def model_train_evaluation(y, ypred, model_name): 
       
    from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score, mean_absolute_percentage_error
    print("\n \n Model Evaluation Report: ")
    print('Mean Absolute Error(MAE) of', model_name,':', mean_absolute_error(y, ypred))
    print('Mean Squared Error(MSE) of', model_name,':', mean_squared_error(y, ypred))
    print('Root Mean Squared Error (RMSE) of', model_name,':', mean_squared_error(y, ypred, squared = False))
    print('Mean absolute percentage error (MAPE) of', model_name,':', mean_absolute_percentage_error(y, ypred))
    print('Explained Variance Score (EVS) of', model_name,':', explained_variance_score(y, ypred))
    print('R2 of', model_name,':', (r2_score(y, ypred)).round(2))
    print('\n \n')
    
    f, ax = plt.subplots(figsize=(12,6),dpi=100);
    plt.scatter(y, ypred, label="Actual vs Predicted")
    plt.xlabel('Fare amount')
    plt.ylabel('Fare amount')
    plt.title('Expection vs Prediction')
    plt.plot(y,y,'r', label="Perfect Expected Prediction")
    plt.legend()
    f.text(0.95, 0.06, 'AUTHOR: RINI CHRISTY',
         fontsize=12, color='green',
         ha='left', va='bottom', alpha=0.5);
    plt.show()


## === cell 73
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit (X_train, y_train)
Yhat_lr = lr.predict(X_test)
model_train_evaluation(y_test, Yhat_lr, 'Linear regression Model')


## === cell 74
test_pred = lr.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 75
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit (X_train, y_train)
Yhat_ridge = ridge.predict(X_test)
model_train_evaluation(y_test, Yhat_ridge, 'Ridge regression Model')


## === cell 76
test_pred = ridge.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 77
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit (X_train, y_train)
Yhat_rf = rf.predict(X_test)
model_train_evaluation(y_test, Yhat_rf, 'Random Forest Regression Model')


## === cell 78
test_pred = rf.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 79
from catboost import CatBoostRegressor
Cat = CatBoostRegressor(loss_function='RMSE', learning_rate = 0.1, 
                        max_depth = 5,  n_estimators = 100, silent = True)
Cat.fit (X_train, y_train)
Cat.fit (X_train, y_train)
Yhat_Cat = Cat.predict(X_test)
model_train_evaluation(y_test, Yhat_Cat, 'Ridge regression Model')


## === cell 80
test_pred = Cat.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 81
from sklearn.linear_model import SGDRegressor
SGD = SGDRegressor()
SGD.fit (X_train, y_train)
Yhat_SGD = SGD.predict(X_test)
model_train_evaluation(y_test, Yhat_SGD, 'SGD Regression Model')


## === cell 82
test_pred = SGD.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 83
from lightgbm import LGBMRegressor
LGBM = LGBMRegressor (boosting_type = 'gbdt', num_leaves = 31,  learning_rate = 0.1, 
                       max_depth = 5, n_estimators = 100, silent = True)
LGBM.fit (X_train, y_train)
Yhat_LGBM = LGBM.predict(X_test)
model_train_evaluation(y_test, Yhat_LGBM, 'LGBM Regression Model')


## === cell 84
test_pred = LGBM.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 85
from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor()
GB.fit (X_train, y_train)
Yhat_GB = GB.predict(X_test)
model_train_evaluation(y_test, Yhat_GB, 'Gradient Boosting Regression Model')


## === cell 86
test_pred = GB.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare_amount'])
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()


## === cell 87
Submission.to_csv('Submission.csv', index = False)


## === cell 88
df_test['haversine_distance_log'] = np.log(df_test['haversine_distance'].values + 1)
df_test['haversine_distance_sqrt'] = np.sqrt(df_test['haversine_distance'].values)
df_train = df.drop(['key', 'pickup_datetime','pickup_weekday', 'fare', 'fare_amount', 'base_fare', 'haversine_distance_sq', 
                    'pickup_hour'], axis = 1)
df_test_copy = df_test.drop(['key', 'base_fare', 'pickup_datetime','pickup_weekday', 'pickup_hour'], axis = 1)
X = df_train.copy()
y = df['fare']
scaler = preprocessing.StandardScaler()
X= scaler.fit(X).transform(X)
test_X= scaler.transform(df_test_copy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


## === cell 89
from lightgbm import LGBMRegressor
LGBM = LGBMRegressor (boosting_type = 'gbdt', num_leaves = 31,  learning_rate = 0.1, 
                       max_depth = 5, n_estimators = 100, silent = True)
LGBM.fit (X_train, y_train)
Yhat_LGBM = LGBM.predict(X_test)
model_train_evaluation(y_test, Yhat_LGBM, 'LGBM Regression Model')


## === cell 90
test_pred = LGBM.predict(test_X)
Submission = pd.DataFrame(test_pred, columns = ['fare'])
Submission['fare_amount'] = df_test['base_fare'] + Submission['fare'] 
Submission['key'] = df_test['key']
Submission = Submission[['key', 'fare_amount']]
Submission.head()
