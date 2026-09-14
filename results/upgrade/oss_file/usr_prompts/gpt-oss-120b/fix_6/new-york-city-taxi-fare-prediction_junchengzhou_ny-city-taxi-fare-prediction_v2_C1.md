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

19.42538

# 6. Current score

15.25186

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.25177) has done: 'The fix addresses three key failures:  
1. The scaler was trained on a dataset that included the `key` column but later transformed a dataset without it, causing a feature‑name mismatch. We now drop `key` from both train and test before scaling.  
2. TensorFlow import raised a protobuf error; we switch to the standalone `keras` package already installed.  
3. With the above corrections, `test_features` and the prediction array are correctly defined, allowing the submission file to be created.'
- What this solution (achieved 15.2526) has done: 'I slightly simplify the neural network (reduce hidden size from 10 to 5) and train for fewer epochs (3 instead of 10). These minimal changes modestly degrade performance, raising the RMSE from the current ~15.25 toward the target range around 19.4 while keeping the original pipeline and output format intact.'
- What this solution (achieved 10.12647) has done: 'I fixed the Keras import and metric definition to work with TensorFlow 2, reduced the network size (k = 3) and trained for only 1 epoch so the validation RMSE moves closer to the target range (around 18‑19). All other pipeline steps remain unchanged, and the script now writes a proper `submission.csv` file.'
- What this solution (achieved 15.25186) has done: 'I updated the script to fix the TensorFlow import error by using the standalone `keras` package, corrected the custom RMSE metric to work with Keras, and made the neural network slightly weaker (smaller hidden size and higher dropout) so the validation RMSE moves toward the target range while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import warnings

warnings.filterwarnings("ignore")



## === cell 2
train = pd.read_csv("../input/train.csv", nrows=10_000_000)



## === cell 3
train.head()



## === cell 4
test = pd.read_csv("../input/test.csv")



## === cell 5
test.head()



## === cell 6
train.dtypes




## === cell 7
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great‑circle distance between two points on the Earth
    (specified in decimal degrees).
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r




## === cell 8
def add_travel_distance_vector_features(df):
    df["distance"] = haversine(
        df["dropoff_longitude"],
        df["dropoff_latitude"],
        df["pickup_longitude"],
        df["pickup_latitude"],
    )


add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)



## === cell 9
train.dtypes



## === cell 10
train.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)



## === cell 11
train.head()



## === cell 12
train.isnull().sum()



## === cell 13
train.dropna(how="any", axis="rows", inplace=True)



## === cell 14
train.describe().astype("float16")



## === cell 15
sns.kdeplot(train.distance, shade=True)



## === cell 16
train.key = pd.to_datetime(train.key).values.astype(np.int64)
test.key = pd.to_datetime(test.key).values.astype(np.int64)



## === cell 17
train.head()



## === cell 18
train.distance.describe().astype("float16")



## === cell 19
y = train.pop("fare_amount")
X = train.copy()  # retain other columns for preprocessing



## === cell 20
from sklearn.preprocessing import MinMaxScaler

X = X.drop(columns=["key"]).values
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

test_features = scaler.transform(test.drop(columns=["key"]))



## === cell 21
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)



## === cell 22
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import backend as K




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 23
def rmse(y_true, y_pred):
    """Root Mean Squared Error metric compatible with Keras."""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




## === cell 24
def nn(n_feature, k=2, dropout_rate=0.5):
    """
    Build a small feed‑forward network.
    k: base hidden size (reduced to increase error).
    dropout_rate: higher dropout to regularise further.
    """
    model_in = Input(shape=(n_feature,))
    x = Dense(k, activation="relu")(model_in)
    x = Dropout(dropout_rate)(x)
    x = Dense(k * 4, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(k, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation="linear")(x)

    model = Model(inputs=model_in, outputs=x)
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])
    return model




## === cell 25
model = nn(X_train.shape[1], k=2, dropout_rate=0.5)



## === cell 26
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=1000,
    epochs=1,  # keep epochs low to stay near target RMSE
    verbose=1,
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1613486456.py in <cell line: 0>()
----> 1 history = model.fit(
      2     X_train,
      3     y_train,
      4     validation_data=(X_val, y_val),
      5     batch_size=1000,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/2977814755.py in rmse(y_true, y_pred)
      1 def rmse(y_true, y_pred):
      2     """Root Mean Squared Error metric compatible with Keras."""
----> 3     return K.sqrt(K.mean(K.square(y_pred - y_true)))
      4 
      5 

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 27
plt.plot(history.history["rmse"], label="train")
if "val_rmse" in history.history:
    plt.plot(history.history["val_rmse"], label="val")
plt.title("Model RMSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1441786449.py in <cell line: 0>()
----> 1 plt.plot(history.history["rmse"], label="train")
      2 if "val_rmse" in history.history:
      3     plt.plot(history.history["val_rmse"], label="val")
      4 plt.title("Model RMSE")
      5 plt.ylabel("RMSE")

NameError: name 'history' is not defined

## === cell 28
pres = model.predict(test_features)



## === cell 29
test_orig = pd.read_csv("../input/test.csv")



## === cell 30
submission = pd.DataFrame(
    {"key": test_orig.key, "fare_amount": pres.reshape(-1)},
    columns=["key", "fare_amount"],
)



## === cell 31
submission.to_csv("submission.csv", index=False)



## === cell 32
print(os.listdir("."))
