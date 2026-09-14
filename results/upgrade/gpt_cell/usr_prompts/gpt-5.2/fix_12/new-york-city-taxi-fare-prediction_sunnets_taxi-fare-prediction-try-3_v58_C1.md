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
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 80000


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/771424182.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     15[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mmodel_selection[0m [0;32mimport[0m [0mtrain_test_split[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mSequential[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mlayers[0m [0;32mimport[0m [0mDense[0m[0;34m,[0m [0mDropout[0m[0;34m,[0m [0mBatchNormalization[0m[0;34m,[0m [0mLSTM[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mcallbacks[0m [0;32mimport[0m [0mEarlyStopping[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# DO NOT EDIT. Generated by api_gen.sh[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFloatDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFunction[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mInitializer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m """
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mdeserialize[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mget[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mserialize[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mconstraints[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mdatasets[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mimport[0m [0mtypes[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mcelu[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0melu[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mexponential[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mdtypes[0m [0;32mimport[0m [0mresult_type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mkeras_tensor[0m [0;32mimport[0m [0mKerasTensor[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mkeras_tensor[0m [0;32mimport[0m [0many_symbolic_tensors[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m [0;32mimport[0m [0mbackend_utils[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mdtypes[0m [0;32mimport[0m [0mresult_type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mAutocastScope[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mVariable[0m [0;32mas[0m [0mKerasVariable[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mget_autocast_scope[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mconfig[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mvariables[0m [0;32mimport[0m [0mstandardize_dtype[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0mBOOL_TYPES[0m [0;34m=[0m [0;34m([0m[0;34m"bool"[0m[0;34m,[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py[0m in [0;36m<module>[0;34m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mstateless_scope[0m [0;32mimport[0m [0mget_stateless_scope[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mcommon[0m[0;34m.[0m[0mstateless_scope[0m [0;32mimport[0m [0min_stateless_scope[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mnaming[0m [0;32mimport[0m [0mauto_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0maudio_dataset_utils[0m [0;32mimport[0m [0maudio_dataset_from_directory[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mdataset_utils[0m [0;32mimport[0m [0msplit_dataset[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mfile_utils[0m [0;32mimport[0m [0mget_file[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mimage_dataset_utils[0m [0;32mimport[0m [0mimage_dataset_from_directory[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mimage_utils[0m [0;32mimport[0m [0marray_to_img[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mdataset_utils[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmodule_utils[0m [0;32mimport[0m [0mtensorflow_io[0m [0;32mas[0m [0mtfio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py[0m in [0;36m<module>[0;34m[0m
[1;32m      7[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m [0;34m[0m[0m
[0;32m----> 9[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mtree[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mio_utils[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0massert_same_paths[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0massert_same_structure[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mflatten[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mflatten_with_path[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m[0;34m.[0m[0mtree_api[0m [0;32mimport[0m [0mis_nested[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;32mif[0m [0moptree[0m[0;34m.[0m[0mavailable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m     [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m [0;32mimport[0m [0moptree_impl[0m [0;32mas[0m [0mtree_impl[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32melif[0m [0mdmtree[0m[0;34m.[0m[0mavailable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m     [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mtree[0m [0;32mimport[0m [0mdmtree_impl[0m [0;32mas[0m [0mtree_impl[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py[0m in [0;36m<module>[0;34m[0m
[1;32m     11[0m [0;31m# Register backend-specific node classes[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;32mif[0m [0mbackend[0m[0;34m([0m[0;34m)[0m [0;34m==[0m [0;34m"tensorflow"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mtrackable[0m[0;34m.[0m[0mdata_structures[0m [0;32mimport[0m [0mListWrapper[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m     [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mtrackable[0m[0;34m.[0m[0mdata_structures[0m [0;32mimport[0m [0m_DictWrapper[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     47[0m [0m_tf2[0m[0;34m.[0m[0menable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m
[0;32m---> 49[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__operators__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0maudio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mautograph[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdecorator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdispatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mag_ctx[0m [0;32mimport[0m [0mcontrol_status_ctx[0m [0;31m# line: 34[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mimpl[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mtf_convert[0m [0;31m# line: 493[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m [0;32mimport[0m [0mthreading[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mag_logging[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mutil[0m[0;34m.[0m[0mtf_export[0m [0;32mimport[0m [0mtf_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;34m"""Utility module that contains APIs usable in the generated code."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mcontext_managers[0m [0;32mimport[0m [0mcontrol_dependency_on_returns[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0malias_tensors[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtensor_list[0m [0;32mimport[0m [0mdynamic_list_append[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mcontextlib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mtensor_array_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36m<module>[0;34m[0m
[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mattr_value_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfull_type_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfunction_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;31m# source: tensorflow/core/framework/attr_value.proto[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m"""Generated protocol buffer code."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mbuilder[0m [0;32mas[0m [0m_builder[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0m_descriptor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor_pool[0m [0;32mas[0m [0m_descriptor_pool[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36m<module>[0;34m[0m
[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0menum_type_wrapper[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mpython_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mreflection[0m [0;32mas[0m [0m_reflection[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py[0m in [0;36m<module>[0;34m[0m
[1;32m     36[0m [0;32mimport[0m [0mweakref[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;34m[0m[0m
[0;32m---> 38[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0mdescriptor_mod[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0mmessage_mod[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mtext_format[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py[0m in [0;36m<module>[0;34m[0m
[1;32m     27[0m   [0;31m# TODO: Remove this import after fix api_implementation[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m   [0;32mif[0m [0m_message[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m     [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0mpyext[0m [0;32mimport[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m   [0m_USE_C_DESCRIPTORS[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m [0;34m[0m[0m

[0;31mImportError[0m: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

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
