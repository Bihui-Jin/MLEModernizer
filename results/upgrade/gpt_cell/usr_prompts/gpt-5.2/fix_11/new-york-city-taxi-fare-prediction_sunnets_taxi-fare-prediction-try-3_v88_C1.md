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
    
    df = df[(df['passenger_count'] > 0)]
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
    
    return df


def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    
    
    return df


def add_distances_features(df):
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    
  
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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401

    try:
        from google.protobuf import __version__ as _pb_ver
    except Exception:
        _pb_ver = "0"
    try:
        _major = int(str(_pb_ver).split(".", 1)[0])
    except Exception:
        _major = 0

    if _major >= 4:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )

        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras import optimizers, regularizers


TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"


BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 800000


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


## === cell 4
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('test_df Size %d' % len(test_df))


## === cell 5
train_df.describe()


## === cell 6
test_df.describe()


## === cell 7
def clean(df):

    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))
    MinMax = (-74.5, -72.8, 40.5, 41.8)

    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]

    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]

    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]

    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]

    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]

    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]

    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]

    print(" New size after ewr airport: %d" % len(df))
    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]

    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]

    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))

    return df

    return df


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    with urllib.request.urlopen(url) as _resp:
        _img = Image.open(_resp)
        nyc_mask = np.array(_img)[:, :, 0] > 0.9

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude,
        df.pickup_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude,
        df.dropoff_latitude,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]

    return df[idx]


def late_night(row):
    if (row["hour"] <= 3) or (row["hour"] >= 0):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] > 20) and (row["hour"] > 0)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S %Z"
    )
    df["year"] = df["pickup_datetime"].apply(lambda x: x.year)
    df["month"] = df["pickup_datetime"].apply(lambda x: x.month)
    df["day"] = df["pickup_datetime"].apply(lambda x: x.day)
    df["hour"] = df["pickup_datetime"].apply(lambda x: x.hour)
    df["weekday"] = df["pickup_datetime"].apply(lambda x: x.weekday())

    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    return df


def add_distances_features(df):

    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete")


def plot_loss_accuracy_rmse(history):

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["rmse"])
    plt.plot(history.history["val_rmse"])
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()


## === cell 8
print('train_df add_time_features')
train_df = add_time_features(train_df)
print('test_df add_time_features')
test_df = add_time_features(test_df)
print('testKaggle add_time_features')
testKaggle = add_time_features(testKaggle)


   


## === cell 9
train_df.describe()


## === cell 10
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 11
train_df.describe()


## === cell 12
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 13
train_df.describe()


## === cell 14
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'passenger_count')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'year')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'month')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'day')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'hour')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'weekday')


## === cell 15
dropped_columns = ['pickup_datetime']#, 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count' ]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


## === cell 16
train_df.shape


## === cell 17
train_df.describe()


## === cell 18
test_df.describe()


## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)


## === cell 20
train_df_main = train_df
validation_df_main = validation_df


## === cell 21
validation_df.describe()


## === cell 22
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 23
test_labels


## === cell 26
train_df.describe()


## === cell 27
validation_df.describe()


## === cell 28
test_df.describe()


## === cell 29
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 30
test_scaled


## === cell 31
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## === cell 32
import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.math.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


checkpoint = ModelCheckpoint(filepath="my_model.h5", verbose=1, save_best_only=True)
model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse, "mse"]
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % train_df.columns)
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)


## === cell 33
from keras.models import load_model


## === cell 34
from IPython.display import SVG

from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog="dot", format="svg"))


## === cell 35
plot_loss_accuracy_rmse(history)


## === cell 36
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print('train mean_squared_error:', score[0])
print('train mae:', score[1])
print('train accuracy:', score[2])
print('train rmse:', score[3])
print('train mse:', score[4])


## === cell 37
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print('Validation mean_squared_error:', score[0])
print('Validation mae:', score[1])
print('Validation accuracy:', score[2])
print('Validation rmse:', score[3])
print('Validation mse:', score[4])


## === cell 38
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print('Test mean_squared_error:', score[0])
print('Test mae:', score[1])
print('Test accuracy:', score[2])
print('Test rmse:', score[3])
print('Test mse:', score[4])


## === cell 40
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([validation_predictions.min(), validation_predictions.max()], [validation_predictions.min(), validation_predictions.max()], 'k--', lw=4)


## === cell 41
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([test_predictions.min(), test_predictions.max()], [test_predictions.min(), test_predictions.max()], 'k--', lw=4)


## === cell 43
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]


## === cell 44
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]


## === cell 45
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


## === cell 46
plt.figure(figsize=(20,10))
plt.plot(validation_labels[:1000])
plt.plot(validation_predictions[:1000])
plt.title('Prediction vs Actual')
plt.ylabel('Fare Amount')
plt.xlabel('Transaction')
plt.legend(['Actual', 'prediction'], loc='upper right')
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]


## === cell 47
plt.figure(figsize=(20,10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title('Prediction vs Actual')
plt.ylabel('Fare Amount')
plt.xlabel('Transaction')
plt.legend(['Actual', 'prediction'], loc='upper right')
plt.show()



## === cell 48
error = validation_predictions - validation_labels
error = np.asarray(error).ravel()
error = error[np.isfinite(error)]

if error.size == 0:
    print(
        "Skipping validation error histogram: residuals contain no finite values (all NaN/Inf)."
    )
else:
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")


## === cell 49
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## --- ERROR in cell 49, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1247160261.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0merror[0m [0;34m=[0m [0mtest_predictions[0m [0;34m-[0m [0mtest_labels[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mplt[0m[0;34m.[0m[0mhist[0m[0;34m([0m[0merror[0m[0;34m,[0m [0mbins[0m [0;34m=[0m [0;36m50[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mplt[0m[0;34m.[0m[0mxlabel[0m[0;34m([0m[0;34m"Prediction Error"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0m_[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0mylabel[0m[0;34m([0m[0;34m"Count"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py[0m in [0;36mhist[0;34m(x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, data, **kwargs)[0m
[1;32m   2643[0m         [0morientation[0m[0;34m=[0m[0;34m'vertical'[0m[0;34m,[0m [0mrwidth[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mlog[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mcolor[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2644[0m         label=None, stacked=False, *, data=None, **kwargs):
[0;32m-> 2645[0;31m     return gca().hist(
[0m[1;32m   2646[0m         [0mx[0m[0;34m,[0m [0mbins[0m[0;34m=[0m[0mbins[0m[0;34m,[0m [0mrange[0m[0;34m=[0m[0mrange[0m[0;34m,[0m [0mdensity[0m[0;34m=[0m[0mdensity[0m[0;34m,[0m [0mweights[0m[0;34m=[0m[0mweights[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2647[0m         [0mcumulative[0m[0;34m=[0m[0mcumulative[0m[0;34m,[0m [0mbottom[0m[0;34m=[0m[0mbottom[0m[0;34m,[0m [0mhisttype[0m[0;34m=[0m[0mhisttype[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py[0m in [0;36minner[0;34m(ax, data, *args, **kwargs)[0m
[1;32m   1444[0m     [0;32mdef[0m [0minner[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mdata[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1445[0m         [0;32mif[0m [0mdata[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1446[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0mmap[0m[0;34m([0m[0msanitize_sequence[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1447[0m [0;34m[0m[0m
[1;32m   1448[0m         [0mbound[0m [0;34m=[0m [0mnew_sig[0m[0;34m.[0m[0mbind[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py[0m in [0;36mhist[0;34m(self, x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, **kwargs)[0m
[1;32m   6789[0m             [0;31m# this will automatically overwrite bins,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6790[0m             [0;31m# so that each histogram uses the same bins[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6791[0;31m             [0mm[0m[0;34m,[0m [0mbins[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mhistogram[0m[0;34m([0m[0mx[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0mbins[0m[0;34m,[0m [0mweights[0m[0;34m=[0m[0mw[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m,[0m [0;34m**[0m[0mhist_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6792[0m             [0mtops[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mm[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6793[0m         [0mtops[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mtops[0m[0;34m,[0m [0mfloat[0m[0;34m)[0m  [0;31m# causes problems later if it's an int[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/lib/histograms.py[0m in [0;36mhistogram[0;34m(a, bins, range, density, weights)[0m
[1;32m    778[0m     [0ma[0m[0;34m,[0m [0mweights[0m [0;34m=[0m [0m_ravel_and_check_weights[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mweights[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    779[0m [0;34m[0m[0m
[0;32m--> 780[0;31m     [0mbin_edges[0m[0;34m,[0m [0muniform_bins[0m [0;34m=[0m [0m_get_bin_edges[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mbins[0m[0;34m,[0m [0mrange[0m[0;34m,[0m [0mweights[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    781[0m [0;34m[0m[0m
[1;32m    782[0m     [0;31m# Histogram is an integer or a float array depending on the weights.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/lib/histograms.py[0m in [0;36m_get_bin_edges[0;34m(a, bins, range, weights)[0m
[1;32m    424[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m'`bins` must be positive, when an integer'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    425[0m [0;34m[0m[0m
[0;32m--> 426[0;31m         [0mfirst_edge[0m[0;34m,[0m [0mlast_edge[0m [0;34m=[0m [0m_get_outer_edges[0m[0;34m([0m[0ma[0m[0;34m,[0m [0mrange[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    427[0m [0;34m[0m[0m
[1;32m    428[0m     [0;32melif[0m [0mnp[0m[0;34m.[0m[0mndim[0m[0;34m([0m[0mbins[0m[0;34m)[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/lib/histograms.py[0m in [0;36m_get_outer_edges[0;34m(a, range)[0m
[1;32m    321[0m         [0mfirst_edge[0m[0;34m,[0m [0mlast_edge[0m [0;34m=[0m [0ma[0m[0;34m.[0m[0mmin[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0ma[0m[0;34m.[0m[0mmax[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    322[0m         [0;32mif[0m [0;32mnot[0m [0;34m([0m[0mnp[0m[0;34m.[0m[0misfinite[0m[0;34m([0m[0mfirst_edge[0m[0;34m)[0m [0;32mand[0m [0mnp[0m[0;34m.[0m[0misfinite[0m[0;34m([0m[0mlast_edge[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 323[0;31m             raise ValueError(
[0m[1;32m    324[0m                 "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))
[1;32m    325[0m [0;34m[0m[0m

[0;31mValueError[0m: autodetected range of [nan, nan] is not finite

## === cell 50
print(len(error))

errorGreaterZero = error[ (np.logical_or(error<=-1, error>=1))]
print(len(errorGreaterZero))    

plt.hist(errorGreaterZero, bins = 100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
