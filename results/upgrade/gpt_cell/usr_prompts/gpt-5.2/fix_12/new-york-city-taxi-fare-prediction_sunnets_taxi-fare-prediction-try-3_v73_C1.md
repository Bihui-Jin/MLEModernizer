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
import os

os.environ.setdefault("KERAS_BACKEND", "numpy")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers, regularizers


TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"


BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000


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


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    from urllib.request import urlopen
    from PIL import Image

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    nyc_mask_img = np.array(Image.open(urlopen(url)))
    nyc_mask = nyc_mask_img[:, :, 0] > 0.9

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


print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)


## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mHTTPError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3957030905.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     37[0m [0;34m[0m[0m
[1;32m     38[0m [0mprint[0m[0;34m([0m[0;34m"train_df clean"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 39[0;31m [0mtrain_df[0m [0;34m=[0m [0mclean[0m[0;34m([0m[0mtrain_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     40[0m [0mtest_df[0m [0;34m=[0m [0mclean[0m[0;34m([0m[0mtest_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3222618572.py[0m in [0;36mclean[0;34m(df)[0m
[1;32m     97[0m [0;34m[0m[0m
[1;32m     98[0m     [0mprint[0m[0;34m([0m[0;34m'Old size: %d'[0m [0;34m%[0m [0mlen[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 99[0;31m     [0mdf[0m [0;34m=[0m [0mremove_datapoints_from_water[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    100[0m     [0mprint[0m[0;34m([0m[0;34m'New size: %d'[0m [0;34m%[0m [0mlen[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    101[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3957030905.py[0m in [0;36mremove_datapoints_from_water[0;34m(df)[0m
[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m     [0murl[0m [0;34m=[0m [0;34m"https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 16[0;31m     [0mnyc_mask_img[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0murlopen[0m[0;34m([0m[0murl[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     17[0m     [0mnyc_mask[0m [0;34m=[0m [0mnyc_mask_img[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0;36m0[0m[0;34m][0m [0;34m>[0m [0;36m0.9[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36murlopen[0;34m(url, data, timeout, cafile, capath, cadefault, context)[0m
[1;32m    214[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    215[0m         [0mopener[0m [0;34m=[0m [0m_opener[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 216[0;31m     [0;32mreturn[0m [0mopener[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0murl[0m[0;34m,[0m [0mdata[0m[0;34m,[0m [0mtimeout[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    217[0m [0;34m[0m[0m
[1;32m    218[0m [0;32mdef[0m [0minstall_opener[0m[0;34m([0m[0mopener[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36mopen[0;34m(self, fullurl, data, timeout)[0m
[1;32m    523[0m         [0;32mfor[0m [0mprocessor[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mprocess_response[0m[0;34m.[0m[0mget[0m[0;34m([0m[0mprotocol[0m[0;34m,[0m [0;34m[[0m[0;34m][0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    524[0m             [0mmeth[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mprocessor[0m[0;34m,[0m [0mmeth_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 525[0;31m             [0mresponse[0m [0;34m=[0m [0mmeth[0m[0;34m([0m[0mreq[0m[0;34m,[0m [0mresponse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    526[0m [0;34m[0m[0m
[1;32m    527[0m         [0;32mreturn[0m [0mresponse[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36mhttp_response[0;34m(self, request, response)[0m
[1;32m    632[0m         [0;31m# request was successfully received, understood, and accepted.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    633[0m         [0;32mif[0m [0;32mnot[0m [0;34m([0m[0;36m200[0m [0;34m<=[0m [0mcode[0m [0;34m<[0m [0;36m300[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 634[0;31m             response = self.parent.error(
[0m[1;32m    635[0m                 'http', request, response, code, msg, hdrs)
[1;32m    636[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36merror[0;34m(self, proto, *args)[0m
[1;32m    561[0m         [0;32mif[0m [0mhttp_err[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    562[0m             [0margs[0m [0;34m=[0m [0;34m([0m[0mdict[0m[0;34m,[0m [0;34m'default'[0m[0;34m,[0m [0;34m'http_error_default'[0m[0;34m)[0m [0;34m+[0m [0morig_args[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 563[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_chain[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    564[0m [0;34m[0m[0m
[1;32m    565[0m [0;31m# XXX probably also want an abstract factory that knows when it makes[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36m_call_chain[0;34m(self, chain, kind, meth_name, *args)[0m
[1;32m    494[0m         [0;32mfor[0m [0mhandler[0m [0;32min[0m [0mhandlers[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    495[0m             [0mfunc[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mhandler[0m[0;34m,[0m [0mmeth_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 496[0;31m             [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    497[0m             [0;32mif[0m [0mresult[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    498[0m                 [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36mhttp_error_default[0;34m(self, req, fp, code, msg, hdrs)[0m
[1;32m    641[0m [0;32mclass[0m [0mHTTPDefaultErrorHandler[0m[0;34m([0m[0mBaseHandler[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    642[0m     [0;32mdef[0m [0mhttp_error_default[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mreq[0m[0;34m,[0m [0mfp[0m[0;34m,[0m [0mcode[0m[0;34m,[0m [0mmsg[0m[0;34m,[0m [0mhdrs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 643[0;31m         [0;32mraise[0m [0mHTTPError[0m[0;34m([0m[0mreq[0m[0;34m.[0m[0mfull_url[0m[0;34m,[0m [0mcode[0m[0;34m,[0m [0mmsg[0m[0;34m,[0m [0mhdrs[0m[0;34m,[0m [0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    644[0m [0;34m[0m[0m
[1;32m    645[0m [0;32mclass[0m [0mHTTPRedirectHandler[0m[0;34m([0m[0mBaseHandler[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mHTTPError[0m: HTTP Error 404: Not Found

## === cell 10
train_df.describe()
