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
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "No compatible protobuf MessageFactory method found for GetPrototype."
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers
from tf_keras import regularizers

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
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

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=train_usecols
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype=datatypes,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)




## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)




## === cell 3
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))




## === cell 5
train_df.describe()




## === cell 6
validation_df.describe()




## === cell 7
test_df.describe()




## === cell 8
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

    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
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


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    nyc_mask = (
        plt.imread("https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png")[
            :, :, 0
        ]
        > 0.9
    )

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
    if (row["hour"] <= 3) or (row["hour"] >= 23):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] > 20) or (row["hour"] < 6)) and (row["weekday"] < 5):
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
        df["pickup_datetime"], errors="coerce", utc=False
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)

    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2

    return df


def add_distances_features(df):

    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )

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




## === cell 9
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    try:
        from urllib.request import urlopen
        from PIL import Image

        url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        with urlopen(url) as f:
            nyc_mask_img = np.array(Image.open(f))
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
    except Exception:
        return df


print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("testKaggle clean")
testKaggle = clean(testKaggle)




## === cell 10
train_df.describe()




## === cell 11
validation_df.describe()




## === cell 12
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)




## === cell 13
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
add_coordinate_features(validation_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)




## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")




## === cell 15
train_df.describe()




## === cell 16
validation_df.describe()




## === cell 17
dropped_columns = [
    "passenger_count",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")




## === cell 18
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")




## === cell 19
train_df.shape




## === cell 20
test_df.shape




## === cell 21
validation_df.shape




## === cell 22
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## --- ERROR in cell 22, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3284815564.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mscaler[0m [0;34m=[0m [0mpreprocessing[0m[0;34m.[0m[0mMinMaxScaler[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mtrain_df_scaled[0m [0;34m=[0m [0mscaler[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mtrain_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mvalidation_df_scaled[0m [0;34m=[0m [0mscaler[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mvalidation_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mtest_scaled[0m [0;34m=[0m [0mscaler[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mtest_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mtestKaggle_scaled[0m [0;34m=[0m [0mscaler[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mtestKaggle_clean[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py[0m in [0;36mwrapped[0;34m(self, X, *args, **kwargs)[0m
[1;32m    138[0m     [0;34m@[0m[0mwraps[0m[0;34m([0m[0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m     [0;32mdef[0m [0mwrapped[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         [0mdata_to_wrap[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata_to_wrap[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m             [0;31m# only wrap the first output for cross decomposition[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36mfit_transform[0;34m(self, X, y, **fit_params)[0m
[1;32m    876[0m         [0;32mif[0m [0my[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    877[0m             [0;31m# fit method of arity 1 (unsupervised transformation)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 878[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    879[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    880[0m             [0;31m# fit method of arity 2 (supervised transformation)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py[0m in [0;36mfit[0;34m(self, X, y)[0m
[1;32m    425[0m         [0;31m# Reset internal state before fitting[0m[0;34m[0m[0;34m[0m[0m
[1;32m    426[0m         [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 427[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mpartial_fit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    428[0m [0;34m[0m[0m
[1;32m    429[0m     [0;32mdef[0m [0mpartial_fit[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py[0m in [0;36mpartial_fit[0;34m(self, X, y)[0m
[1;32m    464[0m [0;34m[0m[0m
[1;32m    465[0m         [0mfirst_pass[0m [0;34m=[0m [0;32mnot[0m [0mhasattr[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m"n_samples_seen_"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 466[0;31m         X = self._validate_data(
[0m[1;32m    467[0m             [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    468[0m             [0mreset[0m[0;34m=[0m[0mfirst_pass[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_data[0;34m(self, X, y, reset, validate_separately, **check_params)[0m
[1;32m    563[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Validation should be done on X, y or both."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    564[0m         [0;32melif[0m [0;32mnot[0m [0mno_val_X[0m [0;32mand[0m [0mno_val_y[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 565[0;31m             [0mX[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0mX[0m[0;34m,[0m [0minput_name[0m[0;34m=[0m[0;34m"X"[0m[0;34m,[0m [0;34m**[0m[0mcheck_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    566[0m             [0mout[0m [0;34m=[0m [0mX[0m[0;34m[0m[0;34m[0m[0m
[1;32m    567[0m         [0;32melif[0m [0mno_val_X[0m [0;32mand[0m [0;32mnot[0m [0mno_val_y[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    877[0m                     [0marray[0m [0;34m=[0m [0mxp[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0marray[0m[0;34m,[0m [0mdtype[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    878[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 879[0;31m                     [0marray[0m [0;34m=[0m [0m_asarray_with_order[0m[0;34m([0m[0marray[0m[0;34m,[0m [0morder[0m[0;34m=[0m[0morder[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0mxp[0m[0;34m=[0m[0mxp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    880[0m             [0;32mexcept[0m [0mComplexWarning[0m [0;32mas[0m [0mcomplex_warning[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    881[0m                 raise ValueError(

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py[0m in [0;36m_asarray_with_order[0;34m(array, dtype, order, copy, xp)[0m
[1;32m    183[0m     [0;32mif[0m [0mxp[0m[0;34m.[0m[0m__name__[0m [0;32min[0m [0;34m{[0m[0;34m"numpy"[0m[0;34m,[0m [0;34m"numpy.array_api"[0m[0;34m}[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    184[0m         [0;31m# Use NumPy API to support order[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 185[0;31m         [0marray[0m [0;34m=[0m [0mnumpy[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0marray[0m[0;34m,[0m [0morder[0m[0;34m=[0m[0morder[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    186[0m         [0;32mreturn[0m [0mxp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0marray[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0mcopy[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    187[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m__array__[0;34m(self, dtype, copy)[0m
[1;32m   2151[0m     ) -> np.ndarray:
[1;32m   2152[0m         [0mvalues[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_values[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2153[0;31m         [0marr[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2154[0m         if (
[1;32m   2155[0m             [0mastype_is_view[0m[0;34m([0m[0mvalues[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0marr[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: could not convert string to float: '2014-03-31 19:44:00.000000231'

## === cell 23
from tf_keras import backend as backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
