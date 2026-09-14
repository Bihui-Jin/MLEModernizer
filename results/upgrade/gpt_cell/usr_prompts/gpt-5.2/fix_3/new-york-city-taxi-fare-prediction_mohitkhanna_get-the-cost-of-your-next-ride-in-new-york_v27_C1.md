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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
pick_up_date_train = taxi_ride_train.loc[:, "pickup_datetime"]
pick_up_date_test = taxi_ride_test.loc[:, "pickup_datetime"]

temp_df_train = pd.DataFrame(
    {
        "year": pick_up_date_train.dt.year,
        "month": pick_up_date_train.dt.month,
        "day": pick_up_date_train.dt.day,
        "hour": pick_up_date_train.dt.hour,
        "dayofyear": pick_up_date_train.dt.dayofyear,
        "week": pick_up_date_train.dt.isocalendar().week.astype(int),
        "weekday": pick_up_date_train.dt.weekday,
        "quarter": pick_up_date_train.dt.quarter,
    }
)

temp_df_test = pd.DataFrame(
    {
        "year": pick_up_date_test.dt.year,
        "month": pick_up_date_test.dt.month,
        "day": pick_up_date_test.dt.day,
        "hour": pick_up_date_test.dt.hour,
        "dayofyear": pick_up_date_test.dt.dayofyear,
        "week": pick_up_date_test.dt.isocalendar().week.astype(int),
        "weekday": pick_up_date_test.dt.weekday,
        "quarter": pick_up_date_test.dt.quarter,
    }
)

taxi_ride_train = pd.concat([taxi_ride_train, temp_df_train], axis=1)
taxi_ride_test = pd.concat([taxi_ride_test, temp_df_test], axis=1)
taxi_ride_train.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_test.drop("pickup_datetime", inplace=True, axis=1)
taxi_ride_train.head()


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


## === cell 33
sns.heatmap(X_train.corr())


## === cell 34
taxi_ride_train.groupby("hour")['fare_amount'].sum().plot()


## === cell 35
taxi_ride_train.groupby("weekday")['fare_amount'].sum().plot()


## === cell 36
taxi_ride_train.groupby("passenger_count")['fare_amount'].sum().plot()


## === cell 37
taxi_ride_train.groupby("month")['fare_amount'].sum().plot()


## === cell 38
taxi_ride_train.groupby("year")['fare_amount'].sum().plot()


## === cell 39
pd.crosstab(taxi_ride_train.quarter, len(taxi_ride_train.fare_amount), margins=True) # create a crosstab


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


## === cell 44
regr = linear_model.LinearRegression()
regr.fit(X_train_scaled, Y_train)
Y_valid_pred = regr.predict(X_valid_scaled)
Y_test_pred = regr.predict(X_test_scaled)
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_valid, Y_valid_pred))
print('Variance score: %.2f' % r2_score(Y_valid, Y_valid_pred))


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


## === cell 48
generate_actual_vs_predicted_plot(Y_valid, Y_valid_pred,
                       "Taxi fares")


## === cell 49
params = {'n_estimators': 700, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train_scaled, Y_train)
mse = mean_squared_error(Y_valid, clf.predict(X_valid_scaled))
print("MSE: %.4f" % mse)
print('Variance score: %.2f' % r2_score(Y_valid, clf.predict(X_valid_scaled)))


## --- ERROR in cell 49, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidParameterError[0m                     Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1101869398.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0mclf[0m [0;34m=[0m [0mensemble[0m[0;34m.[0m[0mGradientBoostingRegressor[0m[0;34m([0m[0;34m**[0m[0mparams[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m [0mclf[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train_scaled[0m[0;34m,[0m [0mY_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0mmse[0m [0;34m=[0m [0mmean_squared_error[0m[0;34m([0m[0mY_valid[0m[0;34m,[0m [0mclf[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX_valid_scaled[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0mprint[0m[0;34m([0m[0;34m"MSE: %.4f"[0m [0;34m%[0m [0mmse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, monitor)[0m
[1;32m    418[0m             [0mFitted[0m [0mestimator[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    419[0m         """
[0;32m--> 420[0;31m         [0mself[0m[0;34m.[0m[0m_validate_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    421[0m [0;34m[0m[0m
[1;32m    422[0m         [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mwarm_start[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_params[0;34m(self)[0m
[1;32m    598[0m         [0maccepted[0m [0mconstraints[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    599[0m         """
[0;32m--> 600[0;31m         validate_parameter_constraints(
[0m[1;32m    601[0m             [0mself[0m[0;34m.[0m[0m_parameter_constraints[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    602[0m             [0mself[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0mdeep[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py[0m in [0;36mvalidate_parameter_constraints[0;34m(parameter_constraints, params, caller_name)[0m
[1;32m     95[0m                 )
[1;32m     96[0m [0;34m[0m[0m
[0;32m---> 97[0;31m             raise InvalidParameterError(
[0m[1;32m     98[0m                 [0;34mf"The {param_name!r} parameter of {caller_name} must be"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m                 [0;34mf" {constraints_str}. Got {param_val!r} instead."[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidParameterError[0m: The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'squared_error', 'huber', 'absolute_error'}. Got 'ls' instead.

## === cell 50
test_pred=pd.DataFrame(clf.predict(X_test_scaled), index=X_test.index)
test_pred.columns=["fare_amount"]
test_pred['fare_amount']= np.exp(test_pred.fare_amount)
test_pred.to_csv("my_submission.csv")
