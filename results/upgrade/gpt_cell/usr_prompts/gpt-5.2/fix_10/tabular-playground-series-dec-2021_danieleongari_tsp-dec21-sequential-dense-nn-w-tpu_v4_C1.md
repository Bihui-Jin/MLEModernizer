# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import gc
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"

_train_head = pd.read_csv(TRAIN_PATH, nrows=0)
_test_head = pd.read_csv(TEST_PATH, nrows=0)


def _make_dtype_map(cols, is_train: bool):
    dtype = {}
    for c in cols:
        if c == "Id" or (is_train and c == "Cover_Type"):
            dtype[c] = np.int32
        elif c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
            dtype[c] = np.int8
        else:
            dtype[c] = np.float32
    return dtype


dtype_train = _make_dtype_map(_train_head.columns, is_train=True)
dtype_test = _make_dtype_map(_test_head.columns, is_train=False)

DROP_SOIL = ("Soil_Type7", "Soil_Type15")



## === cell 2
train_cols = list(_train_head.columns)
test_cols = list(_test_head.columns)

for c in DROP_SOIL:
    if c in train_cols:
        train_cols.remove(c)
    if c in test_cols:
        test_cols.remove(c)

drop_cols_train = {"Id", "Cover_Type"}
drop_cols_test = {"Id"}

feat_cols_train = [c for c in train_cols if c not in drop_cols_train]
feat_cols_test = [c for c in test_cols if c not in drop_cols_test]

assert feat_cols_train == feat_cols_test, "Train/test feature columns mismatch."
feat_cols = feat_cols_train

aspect_idx = feat_cols.index("Aspect")
base_cols_no_aspect = feat_cols[:aspect_idx] + feat_cols[aspect_idx + 1 :]

hs_cols = ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
hs_idx_in_base = []
for col in hs_cols:
    try:
        hs_idx_in_base.append(base_cols_no_aspect.index(col))
    except ValueError:
        pass

n_base = len(base_cols_no_aspect)
nfeatures = n_base + 4

final_cols = base_cols_no_aspect + [
    "Aspect_cos",
    "Aspect_sin",
    "Sum_Hydrology",
    "Sub_Hydrology",
]




## === cell 3
def _count_train_rows_excluding_5(path: str, chunksize: int = 250_000) -> int:
    n = 0
    usecols = ["Cover_Type"]
    for ch in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"Cover_Type": np.int32},
        chunksize=chunksize,
        engine="c",
    ):
        n += int((ch["Cover_Type"].to_numpy(copy=False) != 5).sum())
    return n


def _count_rows(path: str, chunksize: int = 250_000) -> int:
    n = 0
    usecols = ["Id"]
    for ch in pd.read_csv(
        path, usecols=usecols, dtype={"Id": np.int32}, chunksize=chunksize, engine="c"
    ):
        n += len(ch)
    return n


def _write_features_and_targets_train(
    path: str,
    X_path: str,
    y_path: str,
    ids_path: str,
    chunksize: int = 250_000,
):
    n_keep = _count_train_rows_excluding_5(path, chunksize=chunksize)
    X_mm = np.memmap(X_path, mode="w+", dtype=np.float32, shape=(n_keep, nfeatures))
    y_mm = np.memmap(y_path, mode="w+", dtype=np.int32, shape=(n_keep,))
    ids_mm = np.memmap(ids_path, mode="w+", dtype=np.int32, shape=(n_keep,))

    usecols = ["Id", "Cover_Type"] + feat_cols
    usecols = [c for c in usecols if c not in DROP_SOIL]

    write_pos = 0
    for ch in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_train,
        chunksize=chunksize,
        engine="c",
        low_memory=False,
    ):
        m = ch["Cover_Type"].to_numpy(copy=False) != 5
        if not np.any(m):
            continue
        ch = ch.loc[m]

        ids = ch["Id"].to_numpy(copy=False).astype(np.int32, copy=False)
        y = ch["Cover_Type"].to_numpy(copy=False).astype(np.int32, copy=False)

        X_base = ch[base_cols_no_aspect].to_numpy(dtype=np.float32, copy=False)

        for j in hs_idx_in_base:
            np.clip(X_base[:, j], 0.0, 255.0, out=X_base[:, j])

        aspect = ch["Aspect"].to_numpy(dtype=np.float32, copy=False)
        aspect_rad = np.deg2rad(aspect)
        aspect_cos = np.cos(aspect_rad).astype(np.float32, copy=False)
        aspect_sin = np.sin(aspect_rad).astype(np.float32, copy=False)

        h = ch["Horizontal_Distance_To_Hydrology"].to_numpy(
            dtype=np.float32, copy=False
        )
        v = ch["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
        ah = np.abs(h)
        av = np.abs(v)
        sum_h = ah + av
        sub_h = ah - av

        n = X_base.shape[0]
        sl = slice(write_pos, write_pos + n)
        X_mm[sl, :n_base] = X_base
        X_mm[sl, n_base + 0] = aspect_cos
        X_mm[sl, n_base + 1] = aspect_sin
        X_mm[sl, n_base + 2] = sum_h
        X_mm[sl, n_base + 3] = sub_h

        y_mm[sl] = y
        ids_mm[sl] = ids
        write_pos += n

    assert write_pos == n_keep
    X_mm.flush()
    y_mm.flush()
    ids_mm.flush()
    return n_keep


def _write_features_test(
    path: str,
    X_path: str,
    ids_path: str,
    chunksize: int = 250_000,
):
    n = _count_rows(path, chunksize=chunksize)
    X_mm = np.memmap(X_path, mode="w+", dtype=np.float32, shape=(n, nfeatures))
    ids_mm = np.memmap(ids_path, mode="w+", dtype=np.int32, shape=(n,))

    usecols = ["Id"] + feat_cols
    usecols = [c for c in usecols if c not in DROP_SOIL]

    write_pos = 0
    for ch in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_test,
        chunksize=chunksize,
        engine="c",
        low_memory=False,
    ):
        ids = ch["Id"].to_numpy(copy=False).astype(np.int32, copy=False)
        X_base = ch[base_cols_no_aspect].to_numpy(dtype=np.float32, copy=False)
        for j in hs_idx_in_base:
            np.clip(X_base[:, j], 0.0, 255.0, out=X_base[:, j])

        aspect = ch["Aspect"].to_numpy(dtype=np.float32, copy=False)
        aspect_rad = np.deg2rad(aspect)
        aspect_cos = np.cos(aspect_rad).astype(np.float32, copy=False)
        aspect_sin = np.sin(aspect_rad).astype(np.float32, copy=False)

        h = ch["Horizontal_Distance_To_Hydrology"].to_numpy(
            dtype=np.float32, copy=False
        )
        v = ch["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
        ah = np.abs(h)
        av = np.abs(v)
        sum_h = ah + av
        sub_h = ah - av

        n_ch = X_base.shape[0]
        sl = slice(write_pos, write_pos + n_ch)
        X_mm[sl, :n_base] = X_base
        X_mm[sl, n_base + 0] = aspect_cos
        X_mm[sl, n_base + 1] = aspect_sin
        X_mm[sl, n_base + 2] = sum_h
        X_mm[sl, n_base + 3] = sub_h

        ids_mm[sl] = ids
        write_pos += n_ch

    assert write_pos == n
    X_mm.flush()
    ids_mm.flush()
    return n


X_train_path = "X_train_raw.dat"
y_train_path = "y_train.dat"
train_ids_path = "train_ids.dat"
X_test_path = "X_test_raw.dat"
test_ids_path = "test_ids.dat"

n_train = _write_features_and_targets_train(
    TRAIN_PATH, X_train_path, y_train_path, train_ids_path
)
n_test = _write_features_test(TEST_PATH, X_test_path, test_ids_path)

X_train_raw = np.memmap(
    X_train_path, mode="r+", dtype=np.float32, shape=(n_train, nfeatures)
)
X_test_raw = np.memmap(
    X_test_path, mode="r+", dtype=np.float32, shape=(n_test, nfeatures)
)
y_raw = np.memmap(y_train_path, mode="r", dtype=np.int32, shape=(n_train,))
train_ids = np.memmap(train_ids_path, mode="r", dtype=np.int32, shape=(n_train,))
test_ids = np.memmap(test_ids_path, mode="r", dtype=np.int32, shape=(n_test,))

gc.collect()




## === cell 4
def _exact_quantiles_1d(x: np.ndarray, qs=(0.25, 0.5, 0.75)):
    n = x.shape[0]
    return np.quantile(x.astype(np.float64, copy=False), qs, method="linear").astype(
        np.float32
    )


center_ = np.empty((nfeatures,), dtype=np.float32)
scale_ = np.empty((nfeatures,), dtype=np.float32)

for j in range(nfeatures):
    q25, q50, q75 = _exact_quantiles_1d(X_train_raw[:, j], qs=(0.25, 0.5, 0.75))
    center_[j] = q50
    iqr = q75 - q25
    scale_[j] = iqr if iqr != 0.0 else 1.0

X_train_raw -= center_
X_train_raw /= scale_
X_test_raw -= center_
X_test_raw /= scale_
X_train_raw.flush()
X_test_raw.flush()

train = X_train_raw
test = X_test_raw

gc.collect()



## === cell 5
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
target_int = le.fit_transform(np.asarray(y_raw))
target = to_categorical(target_int)

del target_int
gc.collect()



## === cell 6
gc.collect()



## === cell 7
shapes = {
    "nsamples": train.shape[0],
    "nfeatures": train.shape[1],
    "ncategories": target.shape[1],
}
print(shapes)



## === cell 8
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Number of replicas:", strategy.num_replicas_in_sync)
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("TPU not available; using strategy:", type(strategy).__name__)



## === cell 9
CURRENT_MODEL = 2



## === cell 10
if CURRENT_MODEL == 1:
    with strategy.scope():  # necessary for using the TPU
        model1 = keras.models.Sequential(
            [
                keras.layers.Input((shapes["nfeatures"],)),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(shapes["ncategories"], activation="softmax"),
            ]
        )
        model1.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )



## === cell 11
if CURRENT_MODEL == 1:
    n = train.shape[0]
    val_size = int(0.2 * n)
    x_val, y_val = train[:val_size], target[:val_size]
    x_tr, y_tr = train[val_size:], target[val_size:]

    batch_size = 128 * getattr(strategy, "num_replicas_in_sync", 1)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model1.fit(
        x=train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=[earlystop],
        verbose=2,
    )



## === cell 12
pass



## === cell 13
if CURRENT_MODEL == 2:
    dropout_rate = 0.1
    with strategy.scope():  # necessary for using the TPU
        model2 = keras.models.Sequential(
            [
                keras.layers.Input((shapes["nfeatures"],)),
                keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dropout(rate=dropout_rate),
                keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dropout(rate=dropout_rate),
                keras.layers.Dense(200, kernel_initializer="he_normal", use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Dense(shapes["ncategories"], activation="softmax"),
            ]
        )
        model2.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )



## === cell 14
if CURRENT_MODEL == 2:
    n = train.shape[0]
    val_size = int(0.05 * n)
    x_val, y_val = train[:val_size], target[:val_size]
    x_tr, y_tr = train[val_size:], target[val_size:]

    batch_size = 128 * getattr(strategy, "num_replicas_in_sync", 1)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    earlystop = EarlyStopping(patience=3, restore_best_weights=True)
    model2.fit(
        x=train_ds,
        epochs=20,
        validation_data=val_ds,
        verbose=2,
    )



## === cell 15
pass



## === cell 16
if CURRENT_MODEL == 1:
    test_pred = model1.predict(test, verbose=0, batch_size=4096)
elif CURRENT_MODEL == 2:
    test_pred = model2.predict(test, verbose=0, batch_size=4096)



## === cell 17
sub = pd.DataFrame(
    {
        "Id": np.asarray(test_ids),
        "Cover_Type": le.inverse_transform(np.argmax(test_pred, axis=1)),
    }
)
sub.to_csv("submission.csv", index=False)
display(sub.head())
print("Saved submission.csv with shape:", sub.shape)
