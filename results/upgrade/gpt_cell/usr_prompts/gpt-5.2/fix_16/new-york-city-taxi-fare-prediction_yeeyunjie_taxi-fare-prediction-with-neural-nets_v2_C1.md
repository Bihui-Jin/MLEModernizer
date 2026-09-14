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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
statsmodels==0.14.5
tf_keras==2.18.0

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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Sequential = None
Dense = LSTM = TimeDistributed = Flatten = MaxPooling1D = Conv1D = Dropout = None

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    PassiveAggressiveRegressor,
    SGDRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from math import radians, cos, sin, asin, sqrt

pd.set_option("display.float_format", lambda x: "%.3f" % x)

warnings.filterwarnings("ignore")
get_ipython().run_line_magic("matplotlib", "inline")


## === cell 2
df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',nrows = 1000000)

test_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')

sample = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv')


## === cell 3
print(f'Number of records: {df.shape[0]}')
print(f'Number of columns: {df.shape[1]}')


## === cell 4
df.info()


## === cell 5
df.describe()


## === cell 6
df.head()


## === cell 7
df[df.isnull().any(axis=1)]


## === cell 8
df.columns[df.isnull().any()]


## === cell 9
df1 = df[~df.isnull().any(axis=1)]


## === cell 10
incorrect_location = df1[((df1['dropoff_latitude'] < 0) | (df1['pickup_latitude'] < 0)) & ((df1['dropoff_longitude'] > 0) | (df1['pickup_longitude'] > 0))]


## === cell 11
incorrect_location.columns = ['key','fare_amount',"pickup_datetime","pickup_latitude","pickup_longitude",
                              "dropoff_latitude","dropoff_longitude","passenger_count"]


## === cell 12
df1.loc[df1.index.isin(incorrect_location.index),["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]] = incorrect_location[["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]]


## === cell 13
df1[((df1['dropoff_latitude'] < 0) | (df1['pickup_latitude'] < 0)) & ((df1['dropoff_longitude'] > 0) | (df1['pickup_longitude'] > 0))]


## === cell 14
todrop = df1[((df1['dropoff_latitude'] < 0) | (df1['pickup_latitude'] < 0)) & ((df1['dropoff_longitude'] > 0) | (df1['pickup_longitude'] > 0))]
df1 = df1[~df1.index.isin(todrop.index)]


## === cell 15
df1 = df1.drop(df1[(df1['dropoff_latitude'] == 0) & (df1['dropoff_longitude'] == 0) & (df1['pickup_latitude'] == 0) & (df1['pickup_longitude'] ==0)].index)


## === cell 16
df1.head()


## === cell 17
df1[((df1["pickup_latitude"] < 24) & (df1["pickup_latitude"] > 50)) | (df1["dropoff_latitude"]) < 24 & (df1["dropoff_latitude"] > 50)]


## === cell 18
df1.loc[((df1["pickup_longitude"] < -125) & (df1["pickup_longitude"]  > -67)) | (df1["dropoff_longitude"])  < -125 & (df1["dropoff_longitude"] > -67)]


## === cell 19
todrop = df1.loc[((df1["pickup_longitude"] < -125) & (df1["pickup_longitude"]  > -67)) | (df1["dropoff_longitude"])  < -125 & (df1["dropoff_longitude"] > -67)]
df1 = df1[~df1.index.isin(todrop.index)]


## === cell 20
fig,ax = plt.subplots(2,figsize = (12,8))
sns.boxplot(test_df['pickup_latitude'],ax = ax[0])
sns.boxplot(test_df['pickup_longitude'],ax = ax[1])


## === cell 21
test_df.describe()


## === cell 22
df1 = df1[((df1['pickup_longitude'] > -75) & (df1['pickup_longitude'] < -72)) & ((df1['pickup_latitude'] > 40) & (df1['pickup_latitude'] < 42)) & ((df1['dropoff_longitude'] > -75) & (df1['dropoff_longitude'] < -72)) & ((df1['dropoff_latitude'] > 40) & (df1['dropoff_latitude'] < 42))]


## === cell 23
fig,ax = plt.subplots(2,figsize = (12,8))
sns.boxplot(df1['pickup_latitude'],ax = ax[0])
sns.boxplot(df1['pickup_longitude'],ax = ax[1])


## === cell 24
df1 = df1.drop(df1[df1['fare_amount'] <= 0].index)


## === cell 25
fig,ax = plt.subplots(figsize = (12,8))
sns.boxplot(df1['passenger_count'])


## === cell 26
df1[df1['passenger_count'] >50]


## === cell 27
df1 = df1.drop(df1[df1['passenger_count'] > 50].index)


## === cell 28
fig,ax = plt.subplots(figsize = (12,8))
sns.boxplot(df1['fare_amount'])


## === cell 29
df1 = df1.drop(df1[df1['fare_amount'] > 200].index)


## === cell 30
pd.to_datetime(pd.to_datetime(df1.head()['pickup_datetime']).dt.strftime("%Y-%m-%d %H:%M"))


## === cell 31
df1['pickup_datetime'] = pd.to_datetime(pd.to_datetime(df1['pickup_datetime']).dt.strftime("%Y-%m-%d %H:%M"))
test_df['pickup_datetime'] = pd.to_datetime(pd.to_datetime(test_df['pickup_datetime']).dt.strftime("%Y-%m-%d %H:%M"))


## === cell 32
df1['year'] = df1['pickup_datetime'].dt.year
df1['month'] = df1['pickup_datetime'].dt.month
df1['day'] = df1['pickup_datetime'].dt.day
df1['weekday'] = df1['pickup_datetime'].dt.weekday
df1['hour'] = df1['pickup_datetime'].dt.hour
df1['min'] = df1['pickup_datetime'].dt.minute

test_df['year'] = test_df['pickup_datetime'].dt.year
test_df['month'] = test_df['pickup_datetime'].dt.month
test_df['day'] = test_df['pickup_datetime'].dt.day
test_df['weekday'] = test_df['pickup_datetime'].dt.weekday
test_df['hour'] = test_df['pickup_datetime'].dt.hour
test_df['min'] = test_df['pickup_datetime'].dt.minute


## === cell 33
def haversine(df2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1 = df2['pickup_longitude']
    lon2 = df2['dropoff_longitude']
    lat1 = df2['pickup_latitude']
    lat2 = df2['dropoff_latitude']
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return c * r


## === cell 34
df1['distance'] = df1.apply(haversine,axis = 1)


## === cell 35
df2 = df1.copy()
df2 = df2.drop(columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'])


## === cell 36
test_df['distance'] = test_df.apply(haversine,axis = 1)

test_df = test_df.drop(columns = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude'])


## === cell 37
corr_mat = df2.select_dtypes(include=[np.number]).corr()

mask = np.triu(np.ones_like(corr_mat, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_mat, ax=ax, annot=True, cmap=cmap, mask=mask)


## === cell 38
fig,ax = plt.subplots(2,figsize = (12,8))
sns.violinplot(y = df2['fare_amount'],x = df2['year'],ax = ax[0])
sns.violinplot(y = df2['fare_amount'],x = df2['month'],ax = ax[1])


## === cell 39
fig,ax = plt.subplots(2,figsize = (12,8))
sns.barplot(y = df2['fare_amount'],x = df2['year'],ax = ax[0],palette = 'Set2')
sns.barplot(y = df2['fare_amount'],x = df2['month'],ax = ax[1],palette = 'Set2')


## === cell 42
X = df2.drop(columns = ['fare_amount','key','pickup_datetime'])
y = df2['fare_amount']


## === cell 43
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42)


## === cell 44
print(f'train: {X_train.shape}')
print(f'test: {y_train.shape}')
print(f'val train: {X_test.shape}')
print(f'val test: {y_test.shape}')


## === cell 45
X_train.head()


## === cell 46
ss = StandardScaler()
ss.fit(X_train)
X_train_ss = ss.transform(X_train)
X_test_ss = ss.transform(X_test)


## === cell 47
def get_models(models=dict()):
    models['lr'] = LinearRegression()
    models['lasso'] = Lasso()
    models['ridge'] = Ridge()
    models['en'] = ElasticNet()
    models['huber'] = HuberRegressor()
    models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
   
    return models

def get_models_nl(models=dict()):
    models['svr'] = SVR()
    n_trees = 100
    models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
    models['bag'] = BaggingRegressor(n_estimators=n_trees)
    models['rf'] = RandomForestRegressor(n_estimators=n_trees)
    models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
    models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
    return models

def evaluate_models(models, X_train_ss,y_train,X_test_ss,y_test):
    for name, model in models.items():
        model_fit = model.fit(X_train_ss,y_train)
        train_preds = model_fit.predict(X_train_ss)
        test_preds = model_fit.predict(X_test_ss)
        train_mse = mean_squared_error(y_train,train_preds)
        test_mse = mean_squared_error(y_test,test_preds)
        print(f'{name}:')
        print(f'----')
        print(f'Train MAE: {round(train_mse,2)}')
        print(f'Test MAE: {round(test_mse,2)}')
        print(f'\n')
        
def pipeline(model):
    pipe = Pipeline([(model, model_dict[model])])
    return pipe

def params(model):
    

    if model == 'lasso':
        return {"alpha":[0.01,0.1,1,2,5,10],
               }
    
    
    elif model == 'ridge':
        return {
            "alpha":[0.01,0.1,1,2,5,10],
            }
    
    elif model == 'en':
        return {
            'alpha':[0.01,0.1,1,10],
            'l1_ratio':[0.2,0.3,0.4,0.5,0.6]
            }
    elif model == 'knn':
        return {
            'n_neighbors':[4,5,6,7]}

    elif model == 'dt':
        return {
            'max_depth':[3,4,5],
            'min_samples_split':[2,3,4],
            'min_samples_leaf':[2,3,4]
        }
    elif model == 'bag':
        return {
            'max_features':[100, 150]
        }
        
    elif model == 'rf':
        return {
            'n_estimators':[100,150],
            'max_depth':[4],
            'min_samples_leaf':[2,3,4]
        }
    elif model == 'et':
        return {
            'n_estimators':[50,100,150,200],
            'max_depth':[1000,2000,3000],
            'min_samples_leaf':[10000,20000,30000],
        }
    elif model == 'abc':
        return {
            'n_estimators':[50,100,150,200],
            'learning_rate':[0.3,0.6,1]
        }
    elif model == 'gbc':
        return {
            'learning_rate':[0.2],
            'max_depth':[1000,2000,3000],
            'min_samples_split':[10000,20000,30000]
            
        }
    elif model == 'xgb':
        return {
            'eval_metric' : ['auc'],
            'subsample' : [0.8], 
            'colsample_bytree' : [0.5], 
            'learning_rate' : [0.1],
            'max_depth' : [5], 
            'scale_pos_weight': [5], 
            'n_estimators' : [100,200],
            'reg_alpha' : [0, 0.05],
            'reg_lambda' : [2,3],
            'gamma' : [0.01]
                             
        }
    elif model == 'svr':
        return {
            'kernel': ['rbf', 'linear','poly'], 
            'C': [1,20,50,100],
            'gamma':['scale','auto'],
            'epsilon':[0.1,1,10]
        }
    elif model == 'ada':
        return {
            'n_estimators':[50,100,150],
            'learning_rate':[0.01,0.1,1],
            
        }
    elif model == 'bag':
        return {
            'n_estimators':[20,50,100,150],
            'max_features':[2,4,6],
            'max_samples':[0.1,0.2,0.3,0.5,0.7],
            'bootstrap':[True]
            
        }
    elif model == 'rf':
        return {
             'bootstrap': [True],
             'max_depth': [5,10,15],
             'max_features': ["auto", "sqrt", "log2"],
             'min_samples_leaf': [10000,20000,30000],
             'min_samples_split': [10000,20000,30000],
             'n_estimators': [50,200,300,400],
             'random_state': 42,
             }
    elif model == 'et':
        return {
             'bootstrap': [True],
             'max_depth': [5,10,15],
             'max_features': ["auto", "sqrt", "log2"],
             'min_samples_leaf': [10000,20000,30000],
             'min_samples_split': [10000,20000,30000],
             'n_estimators': [50,200,300,400],
             'random_state': 42,
        }
            
    elif model == 'gbm':
        return {
            'learning_rate' : [0.1,0.3,0.6,1], 
            'min_samples_split':[10000,20000,30000],
            'min_samples_leaf': [10000,20000,30000],
            'max_depth' : [8,10,20]
       }



def grid_search_rs(model,models,X_train = X_train_ss,y_train = y_train,X_test = X_test_ss,y_test=y_test):
    pipe_params = params(model)
    model = models[model]
    gs = RandomizedSearchCV(model,param_distributions = pipe_params,cv = 5,scoring = 'neg_mean_squared_error', verbose=True, n_jobs=8)
    gs.fit(X_train_ss,y_train)
    train_score = gs.score(X_train_ss,y_train)
    test_score = gs.score(X_test_ss,y_test)
    
    print(f'Results from: {model}')
    print(f'-----------------------------------')
    print(f'Best Hyperparameters: {gs.best_params_}')
    print(f'Mean MSE: {-round(gs.best_score_,4)}')
    print(f'Train Score: {-round(train_score,4)}')
    print(f'Test Score: {-round(test_score,4)}')
    print(' ')


## === cell 48
models = get_models()
evaluate_models(models,X_train_ss,y_train,X_test_ss,y_test)


## === cell 49
%time grid_search_rs("ridge",models)


## === cell 50
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(
    Dense(
        64,
        activation="relu",
        kernel_initializer="normal",
        input_dim=X_train_ss.shape[1],
    )
)
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam", metrics="mae")
history_model = model.fit(
    X_train_ss,
    y_train,
    epochs=100,
    batch_size=50000,
    validation_data=(X_test_ss, y_test),
    verbose=2,
)


## --- ERROR in cell 50, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 51
model1 = Sequential()
model1.add(Dense(128,activation = 'relu',kernel_initializer = 'normal',input_dim = X_train_ss.shape[1]))
model1.add(Dropout(0.3))
model1.add(Dense(64,activation = 'relu'))
model1.add(Dense(1))
model1.compile(loss = 'mse',optimizer = 'adam',metrics = 'mean_squared_error')
history_model1 = model1.fit(X_train_ss,y_train, epochs = 30, batch_size = 50000, validation_data = (X_test_ss,y_test),verbose = 2)
