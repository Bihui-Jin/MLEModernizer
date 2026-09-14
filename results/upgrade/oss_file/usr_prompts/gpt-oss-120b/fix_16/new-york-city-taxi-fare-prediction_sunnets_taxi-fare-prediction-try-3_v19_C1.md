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

4.59075

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.98902) has done: 'I drop the non‑numeric identifier column `key` (which contains datetime strings) before scaling, ensuring all feature data are numeric. This fixes the MinMaxScaler error, allows the model to train, and produces a valid submission CSV.'
- What this solution (achieved 8.85187) has done: 'I keep the overall pipeline unchanged but add a few lightweight features that are known to improve fare prediction: hour‑of‑day, day‑of‑week and month extracted from `pickup_datetime`, and a simple haversine distance between pickup and drop‑off points. I also retain the useful `passenger_count` column instead of dropping it. Finally, I slightly increase the GradientBoostingRegressor capacity (more trees and a deeper depth) to let the model exploit the extra features. These minimal changes should lower the RMSE toward the target 4.59075 without altering the core logic.'
- What this solution (achieved 7.5938) has done: 'I add a simple Manhattan distance feature (which complements the haversine distance) and slightly strengthen the GradientBoostingRegressor by increasing the number of trees and depth. I also compute and print the validation RMSE so we can see the improvement before writing the submission. These lightweight changes keep the original pipeline intact while moving the RMSE closer to the target.'
- What this solution (achieved 7.97065) has done: 'I keep the overall pipeline unchanged but stop feeding the tree‑based GradientBoostingRegressor with Min‑Max‑scaled data, because scaling does not benefit decision‑tree models and can actually hurt their performance. By using the original (unscaled) numeric features for the GBR branch while still preserving the earlier scaling step (in case you later switch to the TensorFlow model), the validation RMSE should move closer to the target 4.59075 without altering any core logic.'
- What this solution (achieved 7.85845) has done: 'I added a simple log‑transformation of the distance features (which often helps linear‑tree models) and strengthened the GradientBoostingRegressor by using more trees, a deeper depth and a smaller learning rate. These tweaks keep the original pipeline intact while giving the model more expressive power, which should lower the RMSE toward the target value.'

# 9. Code solution

## === cell 0
train_dtypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "key": "str",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=train_dtypes,
    usecols=list(train_dtypes.keys()),
)
testKaggle = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    usecols=list(test_dtypes.keys()),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/940424829.py in <cell line: 0>()
     18     "passenger_count": "uint8",
     19 }
---> 20 trainKaggle = pd.read_csv(
     21     TRAIN_PATH,
     22     nrows=DATASET_SIZE,

NameError: name 'pd' is not defined

## === cell 1
train_df, validation_df = model_selection.train_test_split(
    trainKaggle, test_size=0.10, random_state=1
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/4100284050.py in <cell line: 0>()
----> 1 train_df, validation_df = model_selection.train_test_split(
      2     trainKaggle, test_size=0.10, random_state=1
      3 )
      4 

NameError: name 'model_selection' is not defined

## === cell 2
_nyc_mask = None
_nyc_bb = (-74.5, -72.8, 40.5, 41.8)


def _load_nyc_mask():
    """Download the NYC water mask once with a short timeout and cache it."""
    global _nyc_mask
    if _nyc_mask is not None:
        return _nyc_mask
    try:
        import urllib.request
        from PIL import Image
        import io

        url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        with urllib.request.urlopen(url, timeout=5) as resp:
            img = Image.open(io.BytesIO(resp.read()))
        _nyc_mask = np.array(img)[:, :, 0] > 0.9
    except Exception:
        _nyc_mask = None
    return _nyc_mask


def remove_datapoints_from_water(df):
    """Filter out points that lie on water using the cached NYC mask."""
    nyc_mask = _load_nyc_mask()
    if nyc_mask is None:
        return df  # mask unavailable – keep original data

    BB = _nyc_bb

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (
            (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"),
            (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype("int"),
        )

    pickup_x, pickup_y = lonlat_to_xy(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    return df[idx]


def clean(df):
    print(f" Old size: {len(df)}")
    df = df.dropna(how="any", axis="rows")
    print(f" New size after dropna: {len(df)}")

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(f" New size after removing same long lat: {len(df)}")

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(f" New size after removing 0 long lat: {len(df)}")

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
    print(f" New size after only NYC: {len(df)}")

    df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(f" New size after removing outliers: {len(df)}")

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(f" New size after passenger count filter: {len(df)}")

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    for lon, lat in [nyc_coord, fk_coord, ewr_coord, lga_coord, sol_coord]:
        df = df[(lon != df["pickup_longitude"]) & (lat != df["pickup_latitude"])]
        df = df[(lon != df["dropoff_longitude"]) & (lat != df["dropoff_latitude"])]

    print(f" Old size before water filter: {len(df)}")
    df = remove_datapoints_from_water(df)
    print(f" New size after water filter: {len(df)}")
    return df




## === cell 3
print("Cleaning train split")
train_df = clean(train_df)
print("Cleaning validation split")
validation_df = clean(validation_df)
print("Skipping cleaning for test set (no fare_amount column).")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/134786897.py in <cell line: 0>()
      1 print("Cleaning train split")
----> 2 train_df = clean(train_df)
      3 print("Cleaning validation split")
      4 validation_df = clean(validation_df)
      5 print("Skipping cleaning for test set (no fare_amount column).")

NameError: name 'train_df' is not defined

## === cell 4
def add_time_features(df):
    """Extract hour, day of week and month from pickup_datetime."""
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_hour"] = dt.dt.hour.astype("int8")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("int8")
    df["pickup_month"] = dt.dt.month.astype("int8")
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype("int8")
    return df


def add_coordinate_features(df):
    """Placeholder – keep existing lat/lon columns unchanged."""
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometers."""
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def add_distances_features(df):
    """Add haversine distance between pickup and dropoff points."""
    df = df.copy()
    df["haversine_dist"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["haversine_dist_sq"] = df["haversine_dist"] ** 2
    return df


def add_manhattan_features(df):
    """Add approximated Manhattan distance (km) between pickup and dropoff."""
    df = df.copy()
    lat_mean = (df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0
    km_per_deg_lon = 111.320 * np.cos(np.radians(lat_mean))
    lat_km = 111.0 * np.abs(df["dropoff_latitude"] - df["pickup_latitude"])
    lon_km = km_per_deg_lon * np.abs(df["dropoff_longitude"] - df["pickup_longitude"])
    df["manhattan_dist"] = lat_km + lon_km
    return df


print("Adding time features")
train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3276473951.py in <cell line: 0>()
     54 
     55 print("Adding time features")
---> 56 train_df = add_time_features(train_df)
     57 validation_df = add_time_features(validation_df)
     58 testKaggle = add_time_features(testKaggle)

NameError: name 'train_df' is not defined

## === cell 5
print("Adding coordinate features")
train_df = add_coordinate_features(train_df)
validation_df = add_coordinate_features(validation_df)
testKaggle = add_coordinate_features(testKaggle)

print("Adding distance features")
train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
testKaggle = add_distances_features(testKaggle)

print("Adding Manhattan distance feature")
train_df = add_manhattan_features(train_df)
validation_df = add_manhattan_features(validation_df)
testKaggle = add_manhattan_features(testKaggle)

for df in (train_df, validation_df, testKaggle):
    df["log_haversine"] = np.log1p(df["haversine_dist"])
    df["log_manhattan"] = np.log1p(df["manhattan_dist"])
    df["log_haversine_sq"] = np.log1p(df["haversine_dist_sq"])

print("Feature engineering complete")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/4077930786.py in <cell line: 0>()
      1 print("Adding coordinate features")
----> 2 train_df = add_coordinate_features(train_df)
      3 validation_df = add_coordinate_features(validation_df)
      4 testKaggle = add_coordinate_features(testKaggle)
      5 

NameError: name 'train_df' is not defined

## === cell 6
dropped_columns = ["pickup_datetime", "key"]
train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)
print("Dropped unnecessary columns (kept passenger_count)")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2374406708.py in <cell line: 0>()
      1 dropped_columns = ["pickup_datetime", "key"]
----> 2 train_df = train_df.drop(dropped_columns, axis=1)
      3 validation_df = validation_df.drop(dropped_columns, axis=1)
      4 testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)
      5 print("Dropped unnecessary columns (kept passenger_count)")

NameError: name 'train_df' is not defined

## === cell 7
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Prepared labels and feature matrices")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2157898675.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values
      2 validation_labels = validation_df["fare_amount"].values
      3 
      4 train_df = train_df.drop(["fare_amount"], axis=1)
      5 validation_df = validation_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 8
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

print("Feature scaling completed")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1250521557.py in <cell line: 0>()
----> 1 scaler = preprocessing.MinMaxScaler()
      2 train_df_scaled = scaler.fit_transform(train_df)
      3 validation_df_scaled = scaler.transform(validation_df)
      4 testKaggle_scaled = scaler.transform(testKaggle_clean)
      5 

NameError: name 'preprocessing' is not defined

## === cell 9
use_tf = False



## === cell 10
if use_tf:
    from tensorflow.keras import backend, regularizers, optimizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization

    def rmse(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    model = Sequential()
    model.add(
        Dense(
            512,
            activation="relu",
            input_dim=train_df_scaled.shape[1],
            activity_regularizer=regularizers.l1(0.01),
        )
    )
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(8, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(1))

    adam = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse])

    print("Starting TensorFlow training")
    model.fit(
        x=train_df_scaled,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(validation_df_scaled, validation_labels),
        shuffle=True,
    )
    predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=0)
    val_pred = model.predict(validation_df_scaled, batch_size=128, verbose=0).ravel()
    val_rmse = np.sqrt(metrics.mean_squared_error(validation_labels, val_pred))
    print(f"Validation RMSE (TF model): {val_rmse:.5f}")
else:
    gbr = ensemble.GradientBoostingRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        max_features=0.8,
        random_state=42,
    )
    print("Training scikit‑learn GradientBoostingRegressor on raw features")
    gbr.fit(train_df.values, train_labels)
    predictionKaggle = gbr.predict(testKaggle_clean.values).reshape(-1, 1)

    val_pred = gbr.predict(validation_df.values)
    val_rmse = np.sqrt(metrics.mean_squared_error(validation_labels, val_pred))
    print(f"Validation RMSE (GBR): {val_rmse:.5f}")




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/249293920.py in <cell line: 0>()
     50 else:
     51     # Strengthened GradientBoostingRegressor (removed unsupported n_jobs)
---> 52     gbr = ensemble.GradientBoostingRegressor(
     53         n_estimators=2000,
     54         learning_rate=0.01,

NameError: name 'ensemble' is not defined

## === cell 11
def output_submission(test_df, predictions, id_column, prediction_column, file_name):
    """Write Kaggle submission CSV with required columns."""
    preds = predictions.ravel()
    submission = pd.DataFrame({id_column: test_df[id_column], prediction_column: preds})
    submission.to_csv(file_name, index=False)
    print(f"Submission saved to {file_name}")


output_submission(
    testKaggle,
    predictionKaggle,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3004953104.py in <cell line: 0>()
      8 
      9 output_submission(
---> 10     testKaggle,
     11     predictionKaggle,
     12     id_column="key",

NameError: name 'testKaggle' is not defined

## === cell 12
print("Sample predictions (first 5):")
print(predictionKaggle[:5])

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2389976649.py in <cell line: 0>()
      1 print("Sample predictions (first 5):")
----> 2 print(predictionKaggle[:5])

NameError: name 'predictionKaggle' is not defined
