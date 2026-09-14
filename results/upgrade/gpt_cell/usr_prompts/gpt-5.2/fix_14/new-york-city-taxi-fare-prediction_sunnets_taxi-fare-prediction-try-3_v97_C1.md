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
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
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
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras.callbacks import ModelCheckpoint


TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"


BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000




## === cell 1
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1, 2, 3, 4, 5, 6, 7]
)
testKaggle = pd.read_csv(TEST_PATH)




## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]




## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))




## === cell 4
train_df.describe()




## === cell 5
test_df.describe()




## === cell 6


def remove_datapoints_from_water(df):
    from urllib.request import urlopen
    from PIL import Image

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        with urlopen(url) as resp:
            nyc_mask_img = np.array(Image.open(resp))
        nyc_mask = nyc_mask_img[:, :, 0] > 0.9
    except Exception as e:
        print(f"Warning: could not load NYC water mask ({e}). Skipping water filter.")
        return df

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
train_df = remove_datapoints_from_water(train_df)
test_df = remove_datapoints_from_water(test_df)

print("testKaggle clean (to match train/test preprocessing and reduce RMSE)")
testKaggle = remove_datapoints_from_water(testKaggle)


## === cell 7
def add_time_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_year"] = dt.dt.year.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_day"] = dt.dt.day.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_minute"] = dt.dt.minute.astype("float32")
    df["pickup_second"] = dt.dt.second.astype("float32")
    return df


print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)


## === cell 8
train_df.describe()




## === cell 9
def add_coordinate_features(df):
    dlon = (df["pickup_longitude"] - df["dropoff_longitude"]).astype("float32")
    dlat = (df["pickup_latitude"] - df["dropoff_latitude"]).astype("float32")

    df["abs_lon_diff"] = np.abs(dlon).astype("float32")
    df["abs_lat_diff"] = np.abs(dlat).astype("float32")
    df["euclidean_distance"] = np.sqrt(dlon * dlon + dlat * dlat).astype("float32")
    return df


print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)


## === cell 10
train_df.describe()




## === cell 11
def add_distances_features(df):
    df = df.copy()
    if "euclidean_distance" not in df.columns:
        dlon = (df["pickup_longitude"] - df["dropoff_longitude"]).astype("float32")
        dlat = (df["pickup_latitude"] - df["dropoff_latitude"]).astype("float32")
        df["euclidean_distance"] = np.sqrt(dlon * dlon + dlat * dlat).astype("float32")
    return df


print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)


print("Done with Adding features")


## === cell 12
train_df.describe()




## === cell 13
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")




## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3804[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3805[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_engine[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mcasted_key[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3806[0m         [0;32mexcept[0m [0mKeyError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;31mKeyError[0m: 'year'

The above exception was the direct cause of the following exception:

[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3352054352.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mplot[0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;36m2000[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m"fare_amount"[0m[0;34m,[0m [0;34m"passenger_count"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mplot[0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;36m2000[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m"fare_amount"[0m[0;34m,[0m [0;34m"year"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mplot[0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;36m2000[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m"fare_amount"[0m[0;34m,[0m [0;34m"month"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mplot[0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;36m2000[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m"fare_amount"[0m[0;34m,[0m [0;34m"day"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mplot[0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;36m2000[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m"fare_amount"[0m[0;34m,[0m [0;34m"hour"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py[0m in [0;36mscatter[0;34m(self, x, y, s, c, **kwargs)[0m
[1;32m   1746[0m             ...                       colormap='viridis')
[1;32m   1747[0m         """
[0;32m-> 1748[0;31m         [0;32mreturn[0m [0mself[0m[0;34m([0m[0mkind[0m[0;34m=[0m[0;34m"scatter"[0m[0;34m,[0m [0mx[0m[0;34m=[0m[0mx[0m[0;34m,[0m [0my[0m[0;34m=[0m[0my[0m[0;34m,[0m [0ms[0m[0;34m=[0m[0ms[0m[0;34m,[0m [0mc[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1749[0m [0;34m[0m[0m
[1;32m   1750[0m     def hexbin(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    973[0m         [0;32mif[0m [0mkind[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_dataframe_kinds[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    974[0m             [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mABCDataFrame[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 975[0;31m                 [0;32mreturn[0m [0mplot_backend[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mx[0m[0;34m=[0m[0mx[0m[0;34m,[0m [0my[0m[0;34m=[0m[0my[0m[0;34m,[0m [0mkind[0m[0;34m=[0m[0mkind[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    976[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    977[0m                 [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34mf"plot kind {kind} can only be used for data frames"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/__init__.py[0m in [0;36mplot[0;34m(data, kind, **kwargs)[0m
[1;32m     69[0m             [0mkwargs[0m[0;34m[[0m[0;34m"ax"[0m[0;34m][0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m"left_ax"[0m[0;34m,[0m [0max[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     70[0m     [0mplot_obj[0m [0;34m=[0m [0mPLOT_CLASSES[0m[0;34m[[0m[0mkind[0m[0;34m][0m[0;34m([0m[0mdata[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 71[0;31m     [0mplot_obj[0m[0;34m.[0m[0mgenerate[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     72[0m     [0mplot_obj[0m[0;34m.[0m[0mdraw[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     73[0m     [0;32mreturn[0m [0mplot_obj[0m[0;34m.[0m[0mresult[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py[0m in [0;36mgenerate[0;34m(self)[0m
[1;32m    499[0m         [0mself[0m[0;34m.[0m[0m_compute_plot_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    500[0m         [0mfig[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfig[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 501[0;31m         [0mself[0m[0;34m.[0m[0m_make_plot[0m[0;34m([0m[0mfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    502[0m         [0mself[0m[0;34m.[0m[0m_add_table[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    503[0m         [0mself[0m[0;34m.[0m[0m_make_legend[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py[0m in [0;36m_make_plot[0;34m(self, fig)[0m
[1;32m   1351[0m         scatter = ax.scatter(
[1;32m   1352[0m             [0mdata[0m[0;34m[[0m[0mx[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1353[0;31m             [0mdata[0m[0;34m[[0m[0my[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1354[0m             [0mc[0m[0;34m=[0m[0mc_values[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1355[0m             [0mlabel[0m[0;34m=[0m[0mlabel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   4100[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mnlevels[0m [0;34m>[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4101[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_multilevel[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4102[0;31m             [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4103[0m             [0;32mif[0m [0mis_integer[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4104[0m                 [0mindexer[0m [0;34m=[0m [0;34m[[0m[0mindexer[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3810[0m             ):
[1;32m   3811[0m                 [0;32mraise[0m [0mInvalidIndexError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3812[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3813[0m         [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3814[0m             [0;31m# If we have a listlike key, _check_indexing_error will raise[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'year'

## === cell 14
dropped_columns = [
    "pickup_datetime"
]  # , 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count' ]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")
