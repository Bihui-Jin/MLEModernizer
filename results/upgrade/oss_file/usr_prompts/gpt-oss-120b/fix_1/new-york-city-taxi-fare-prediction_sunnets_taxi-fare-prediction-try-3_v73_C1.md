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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

4.67002

# 6. Current score

15.12557

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
def clean(df):
    
    print(' Old size: %d' % len(df))
    df = df.dropna(how = 'any', axis = 'rows')
    print(' New size after dropna: %d' % len(df))  
    
    df = df[(df['dropoff_longitude'] != df['pickup_longitude']) & (df['dropoff_latitude'] != df['pickup_latitude'])]
    print(' New size after removing same long lat: %d' % len(df))                 
    
    df = df[(df['dropoff_longitude'] != 0) & (df['pickup_longitude'] != 0) & (df['dropoff_latitude'] != 0) & (df['pickup_latitude'] != 0)] 
    print(' New size after removing 0 long lat: %d' % len(df))                          
    MinMax = (-74.5, -72.8, 40.5, 41.8)

    df = df[(MinMax[0] <= df['pickup_longitude']) & (df['pickup_longitude'] <= MinMax[1])]
    df = df[(MinMax[0] <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= MinMax[1])]
    df = df[(MinMax[2] <= df['pickup_latitude']) & (df['pickup_latitude'] <= MinMax[3])]
    df = df[(MinMax[2] <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= MinMax[3])]
       

    print(' New size after NYC lang lot: %d' % len(df))         
    
    df = df[(df['pickup_latitude'] != 0)]
    df = df[(df['pickup_latitude'] != 0)]
    df = df[(df['dropoff_longitude']!= 0)]
    df = df[(df['dropoff_latitude'] != 0)]
    
    print(' New size after lang lot > 0: %d' % len(df))         

    df = df[((df['pickup_latitude'] - df['dropoff_latitude']).abs() > 0.001)]
    df = df[((df['pickup_longitude'] - df['dropoff_longitude']).abs() > 0.001)]
    
    print(' New size after lang - lot > 0.001: %d' % len(df))         
    
    print(' New size after only NYC: %d' % len(df)) 
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 50)]
    
    print(' New size after removing outliers: %d' % len(df)) 
    
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
    print(' New size after removing 6=>passenger_count > 0 : %d' % len(df)) 
    
    
    
    
    nyc_coord = (40.7141667,-74.0063889) 
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    
             

    df = df[(nyc_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != nyc_coord[0])]
    df = df[(nyc_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != nyc_coord[0])]
    
    print(' New size after NY airport: %d' % len(df))
    
    df = df[(fk_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != fk_coord[0])]
    df = df[(fk_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != fk_coord[0])]
    
    print(' New size after jfk airport: %d' % len(df))
    
    df = df[(ewr_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != ewr_coord[0])]
    df = df[(ewr_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != ewr_coord[0])]
    
    print(' New size after ewr airport: %d' % len(df))
    df = df[(lga_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != lga_coord[0])]
    df = df[(lga_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != lga_coord[0])]
    

    print(' New size after lgr airport: %d' % len(df))
             
    df = df[(sol_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != sol_coord[0])]
    df = df[(sol_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != sol_coord[0])]
    

    print(' New size after sol removed: %d' % len(df))             
    
            
    print('Old size: %d' % len(df))
    df = remove_datapoints_from_water(df)
    print('New size: %d' % len(df))
    
        
    print(' New size: %d' % len(df))
    
    return df

def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
               (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    BB = (-74.5, -72.8, 40.5, 41.8)
    
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    
    return df[idx]
    
def late_night (row):
     if (row['hour'] <= 3) or (row['hour'] >= 0):
        return 1
     else:
        return 0


def night (row):
    if ((row['hour'] > 20) and (row['hour'] > 0)) and (row['weekday'] < 5):
        return 1
    else:
        return 0
    
def rush_hour (row):
    if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):
        return 1
    else:
        return 0   
    

       
    
    
def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['pickup_datetime'] =  df['pickup_datetime'].apply(lambda x: str(x))
    df['night'] = df.apply (lambda x: night(x), axis=1)
    df['late_night'] = df.apply (lambda x: late_night(x), axis=1)
    df['rush_hour'] = df.apply (lambda x: rush_hour(x), axis=1)
    
    return df


def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['latdiff'] = (lat1 - lat2).abs()
    df['londiff'] = (lon1 - lon2).abs()
    
    
    return df


def add_distances_features(df):
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    df['distance'] = np.sqrt(np.abs(df['pickup_longitude']-df['dropoff_longitude'])**2 + np.abs(df['pickup_latitude']-df['dropoff_latitude'])**2)
    
  
    return df

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
    
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('Output complete')
    
    
def plot_loss_accuracy_rmse(history):
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('Model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    


## === cell 1
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint


TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
SUBMISSION_NAME = 'submissiontry_water.csv'



BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1,2,3,4,5,6,7])
testKaggle = pd.read_csv(TEST_PATH)    


## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]


## === cell 5
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('test_df Size %d' % len(test_df))


## === cell 6
train_df.describe()


## === cell 7
test_df.describe()


## === cell 9

print('train_df clean')
train_df = clean(train_df)
test_df = clean(test_df)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/232042588.py in <cell line: 0>()
      2 
      3 print('train_df clean')
----> 4 train_df = clean(train_df)
      5 test_df = clean(test_df)

/tmp/ipykernel_11/3222618572.py in clean(df)
     97 
     98     print('Old size: %d' % len(df))
---> 99     df = remove_datapoints_from_water(df)
    100     print('New size: %d' % len(df))
    101 

/tmp/ipykernel_11/3222618572.py in remove_datapoints_from_water(df)
    115     # read nyc mask and turn into boolean map with
    116     # land = True, water = False
--> 117     nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    118 
    119     # calculate for each lon,lat coordinate the xy coordinate in the mask map

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in imread(fname, format)
   2193 @_copy_docstring_and_deprecators(matplotlib.image.imread)
   2194 def imread(fname, format=None):
-> 2195     return matplotlib.image.imread(fname, format)
   2196 
   2197 

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1556     if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
   1557         # Pillow doesn't handle URLs directly.
-> 1558         raise ValueError(
   1559             "Please open the URL for reading and pass the "
   1560             "result to Pillow, e.g. with "

ValueError: Please open the URL for reading and pass the result to Pillow, e.g. with ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.

## === cell 10
train_df.describe()


## === cell 11
print('train_df add_time_features')
train_df = add_time_features(train_df)
print('test_df add_time_features')
test_df = add_time_features(test_df)
print('testKaggle add_time_features')
testKaggle = add_time_features(testKaggle)


   


## === cell 12
train_df.describe()


## === cell 13
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 14
train_df.describe()


## === cell 15
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 16
train_df.describe()


## === cell 17
plot = train_df.iloc[:2000].plot.scatter('latdiff', 'londiff')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'passenger_count')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'year')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'month')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'day')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'hour')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'weekday')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'night')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'late_night')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'rush_hour')


## === cell 19
dropped_columns = ['pickup_datetime'] #'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', ]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


## === cell 20
train_df.shape


## === cell 21
train_df.describe()


## === cell 22
test_df.describe()


## === cell 23
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)


## === cell 24
train_df_main = train_df
validation_df_main = validation_df


## === cell 25
validation_df.describe()


## === cell 26
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 27
test_labels


## === cell 30
train_df.describe()


## === cell 31
validation_df.describe()


## === cell 32
test_df.describe()


## === cell 33
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 34
test_scaled


## === cell 35
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## === cell 36
checkpoint = ModelCheckpoint(filepath='my_model.h5', verbose=1, save_best_only=True)
model = Sequential()
model.add(Dense(256, activation='linear', input_dim=train_df_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae', 'accuracy', rmse, 'mse'])

print('Dataset size: %s' % DATASET_SIZE)
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: %s' % train_df_scaled.shape[1])
print('Features used: %s' % train_df.columns)
model.summary()

history = model.fit(x=train_df_scaled, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    verbose=1, callbacks=[checkpoint], validation_data=(validation_df_scaled, validation_labels), 
                    shuffle=True)
		    


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2261682520.py in <cell line: 0>()
     15 model.add(Dense(1))
     16 
---> 17 adam = optimizers.adam(lr=LEARNING_RATE)
     18 model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae', 'accuracy', rmse, 'mse'])
     19 

AttributeError: module 'keras.api.optimizers' has no attribute 'adam'

## === cell 37
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3016807036.py in <cell line: 0>()
      1 from IPython.display import SVG
----> 2 from keras.utils.vis_utils import model_to_dot
      3 SVG(model_to_dot(model).create(prog='dot', format='svg'))

ModuleNotFoundError: No module named 'keras.utils.vis_utils'

## === cell 38
plot_loss_accuracy_rmse(history)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/599541864.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)

NameError: name 'history' is not defined

## === cell 39
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print('train mean_squared_error:', score[0])
print('train mae:', score[1])
print('train accuracy:', score[2])
print('train rmse:', score[3])
print('train mse:', score[4])


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2549053234.py in <cell line: 0>()
----> 1 score = model.evaluate(train_df_scaled, train_labels, verbose=1)
      2 print(score)
      3 print('train mean_squared_error:', score[0])
      4 print('train mae:', score[1])
      5 print('train accuracy:', score[2])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 40
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print('Validation mean_squared_error:', score[0])
print('Validation mae:', score[1])
print('Validation accuracy:', score[2])
print('Validation rmse:', score[3])
print('Validation mse:', score[4])


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2537874319.py in <cell line: 0>()
----> 1 score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
      2 print(score)
      3 print('Validation mean_squared_error:', score[0])
      4 print('Validation mae:', score[1])
      5 print('Validation accuracy:', score[2])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 41
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print('Test mean_squared_error:', score[0])
print('Test mae:', score[1])
print('Test accuracy:', score[2])
print('Test rmse:', score[3])
print('Test mse:', score[4])


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2425544776.py in <cell line: 0>()
----> 1 score = model.evaluate(test_scaled, test_labels, verbose=1)
      2 print(score)
      3 print('Test mean_squared_error:', score[0])
      4 print('Test mae:', score[1])
      5 print('Test accuracy:', score[2])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py in _assert_compile_called(self, method_name)
   1047             else:
   1048                 msg += f"calling `{method_name}()`."
-> 1049             raise ValueError(msg)
   1050 
   1051     def _symbolic_build(self, iterator=None, data_batch=None):

ValueError: You must call `compile()` before using the model.

## === cell 43
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([validation_predictions.min(), validation_predictions.max()], [validation_predictions.min(), validation_predictions.max()], 'k--', lw=4)


## === cell 44
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([test_predictions.min(), test_predictions.max()], [test_predictions.min(), test_predictions.max()], 'k--', lw=4)


## === cell 46
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]


## === cell 47
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]


## === cell 48
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


## === cell 49
plt.figure(figsize=(20,10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title('Prediction vs Actual')
plt.ylabel('Fare Amount')
plt.xlabel('Transaction')
plt.legend(['Actual', 'prediction'], loc='upper right')
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]


## === cell 50
plt.figure(figsize=(20,10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title('Prediction vs Actual')
plt.ylabel('Fare Amount')
plt.xlabel('Transaction')
plt.legend(['Actual', 'prediction'], loc='upper right')
plt.show()



## === cell 51
error = validation_predictions - validation_labels
plt.hist(error, bins = 100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## === cell 52
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## === cell 53
print(len(error))
print(len(errorGreaterZero))
errorGreaterZero =     error[ (np.logical_or(error<=-1, error>=1))]
    

plt.hist(errorGreaterZero, bins = 100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1776489703.py in <cell line: 0>()
      1 print(len(error))
----> 2 print(len(errorGreaterZero))
      3 errorGreaterZero =     error[ (np.logical_or(error<=-1, error>=1))]
      4 
      5 

NameError: name 'errorGreaterZero' is not defined

## === cell 55
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 56

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)
