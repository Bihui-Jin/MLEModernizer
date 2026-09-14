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
ipywidgets==8.1.5
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
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
tqdm==4.67.1

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

3.97307

# 6. Current score

5.66763

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.66763) has done: 'The fixes address three critical problems: the removed `weekofyear` attribute (replaced with `isocalendar().week`), incompatibilities between the standalone Keras package and TensorFlow 2 (switching to `tensorflow.keras` imports), and missing variable definitions caused by earlier errors. After these adjustments the script runs end‑to‑end, trains a simple model on a sampled subset, predicts the fares for the test set, and writes a correctly‑formatted `submission.csv` ready for Kaggle.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ipywidgets import widgets
import tensorflow as tf
from tensorflow.keras import layers as lyr
from tensorflow.keras import activations as act
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

plt.rcParams["figure.figsize"] = (16, 9)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def read_csv_sampled(path, frac, chunksize=10**5, random_state=None):
    samples = []
    for df_chunk in tqdm(pd.read_csv(path, chunksize=chunksize)):
        samples.append(df_chunk.sample(frac=frac, random_state=random_state))
    df = pd.concat(samples, ignore_index=True)
    df["pickup_datetime"] = df["pickup_datetime"].apply(pd.Timestamp)
    return df




## === cell 2
df_train_sample = read_csv_sampled("../input/train.csv", 0.01, random_state=1989)



## === cell 3
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 4
df_train_sample.dropna(inplace=True)



## === cell 5
is_weird = df_train_sample["fare_amount"] < 0
is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)
is_weird |= df_train_sample["passenger_count"] == 0
df_train_sample = df_train_sample[~is_weird]




## === cell 6
def prep_data(df, shuffle=False):
    X_cat = np.vstack(
        [
            df["pickup_datetime"].dt.hour,  # 0‑23
            df["pickup_datetime"].dt.weekday + 24,  # 24‑30
            df["pickup_datetime"].dt.dayofyear + 30,  # 31‑396
            df["pickup_datetime"].dt.isocalendar().week + 396,  # 397‑449
            df["pickup_datetime"].dt.year - 2009 + 450,  # 450‑456
            df["passenger_count"] + 456,  # 457‑463
        ]
    ).T.astype(np.int32)

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values.astype(np.float32)
    X_deg /= 180.0

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype(np.float32)

    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_cat = X_cat[rnd_ind]
        X_deg = X_deg[rnd_ind]
        if "fare_amount" in df.columns:
            y = y[rnd_ind]

    if "fare_amount" in df.columns:
        return (X_cat, X_deg), y
    return (X_cat, X_deg)




## === cell 7
np.random.seed(1989)
(X_cat, X_deg), y = prep_data(df_train_sample, shuffle=True)




## === cell 8
def K_haversine_bearing(x):
    R = 6371e3
    x_rad = x * np.pi / 180.0
    lat1, lng1, lat2, lng2 = x_rad[:, 0], x_rad[:, 1], x_rad[:, 2], x_rad[:, 3]
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = K.sin(dlat / 2) * K.sin(dlat / 2) + K.cos(lat1) * K.cos(lat2) * K.sin(
        dlng / 2
    ) * K.sin(dlng / 2)
    c = 2 * tf.atan2(K.sqrt(a), K.sqrt(1 - a))
    d = K.log((R * c) + 1)
    x = K.sin(dlng) * K.cos(lat2)
    y = (K.cos(lat1) * K.sin(lat2)) - (K.sin(lat1) * K.cos(lat2) * K.cos(dlng))
    b = tf.atan2(x, y) / np.pi
    return K.concatenate([K.reshape(d, (-1, 1)), K.reshape(b, (-1, 1))])




## === cell 9
lyr_latlng_input = lyr.Input(shape=(X_deg.shape[1],), dtype="float32")
lyr_haversine_bearing = lyr.Lambda(K_haversine_bearing, name="haversine")(
    lyr_latlng_input
)

lyr_cat_input = lyr.Input(shape=(X_cat.shape[1],), dtype="int32")
lyr_cat_embeddings = lyr.Embedding(input_dim=int(X_cat.max()) + 1, output_dim=16)(
    lyr_cat_input
)
lyr_cat_flatten = lyr.Flatten()(lyr_cat_embeddings)

lyr_concat = lyr.concatenate([lyr_latlng_input, lyr_haversine_bearing, lyr_cat_flatten])

lyr_dense1 = lyr.Dense(256, activation=act.selu)(lyr_concat)
lyr_dense2 = lyr.Dense(256, activation=act.selu)(lyr_dense1)
lyr_dense3 = lyr.Dense(256, activation=act.selu)(lyr_dense2)
lyr_out = lyr.Dense(1, activation=act.selu)(lyr_dense3)

model = Model(inputs=[lyr_latlng_input, lyr_cat_input], outputs=lyr_out)
model.summary()



## === cell 10
model.compile(loss="mse", optimizer="nadam")



## === cell 11
model.fit([X_deg, X_cat], y, batch_size=512, epochs=50, validation_split=0.3, verbose=2)



## === cell 12
X_cat_test, X_deg_test = prep_data(df_test)
y_test_pred = np.clip(
    model.predict([X_deg_test, X_cat_test], batch_size=512, verbose=0)[:, 0], 0, None
)



## === cell 13
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test_pred})
df_sub.to_csv("submission.csv", index=False)
