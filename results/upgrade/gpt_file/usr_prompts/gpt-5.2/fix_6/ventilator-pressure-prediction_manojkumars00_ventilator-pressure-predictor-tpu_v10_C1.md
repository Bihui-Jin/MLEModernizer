# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler

print("TensorFlow:", tf.__version__)

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA
except Exception:
    pass

plt.ioff()

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
sc = StandardScaler()
rc = RobustScaler()




## === cell 2
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 3
def dropCols(df, cols):
    df = df.copy(deep=False)
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 4
def add_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)

    new_breath = np.empty(len(df), dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_in_lag1[new_breath] = 0.0

    df["u_in_lag1"] = u_in_lag1
    df["diff_u_in1"] = u_in - u_in_lag1

    cs = np.cumsum(u_in)
    start_idx = np.flatnonzero(new_breath)
    base = np.empty_like(u_in)
    base[start_idx] = cs[start_idx] - u_in[start_idx]
    base = np.maximum.accumulate(base)  # forward-fill within each breath
    df["u_in_cumsum"] = cs - base

    return df




## === cell 5
train_data = pd.read_csv(train_path)
train_data = add_features_fast(train_data)




## === cell 6
cols_2_drop = ["id", "breath_id", "time_step"]




## === cell 7
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")




## === cell 8
train_df.shape




## === cell 9
train_df.isna().sum()




## === cell 10
rc.fit(train_df)
train_df = rc.transform(train_df).astype(np.float32, copy=False)




## === cell 11
train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.values.reshape(-1, 80, 1).astype(np.float32, copy=False)




## === cell 12
train_df.shape, Y.shape




## === cell 13
def build_model():
    model = tf.keras.Sequential()

    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True),
            input_shape=[80, train_df.shape[-1]],
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(180, return_sequences=True)))
    model.add(
        layers.Bidirectional(layers.LSTM(280, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(360, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(500, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(layers.LSTM(1000, dropout=0.2, return_sequences=True))
    )

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(1)))

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 14
_ = None




## === cell 15
callback1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=10,
    verbose=1,
)




## === cell 16
def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print("Running on TPU")
        return tf.distribute.TPUStrategy(tpu)
    except Exception:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print("Running on GPU")
            return tf.distribute.MirroredStrategy()
        print("Running on CPU")
        return tf.distribute.get_strategy()


strategy = get_strategy()




## === cell 17
EPOCH = 500
BATCH_SIZE = 1024

trained_models = []


def make_ds_tf(X, y=None, training=False):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(
            buffer_size=int(X.shape[0]), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def fold_weights_path(fold_idx_1based: int) -> str:
    return f"AdamPressurePreModel{fold_idx_1based}.weights.h5"


all_weights_exist = all(os.path.exists(fold_weights_path(i)) for i in range(1, 6))
if all_weights_exist:
    print(
        "Found all 5 fold weight files; skipping training and loading weights for inference."
    )
    with strategy.scope():
        for fold in range(5):
            model = build_model()
            model.load_weights(fold_weights_path(fold + 1))
            trained_models.append(model)
else:
    with strategy.scope():
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, Y)):
            print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
            X_train, X_valid = train_df[train_idx], train_df[valid_idx]
            y_train, y_valid = Y[train_idx], Y[valid_idx]

            model = build_model()

            callback_best_in_memory = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EPOCH,  # never triggers early stop; only used for restore_best_weights behavior
                restore_best_weights=True,
                verbose=0,
            )

            train_ds = make_ds_tf(X_train, y_train, training=True)
            valid_ds = make_ds_tf(X_valid, y_valid, training=False)

            his = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=EPOCH,
                callbacks=[callback1, callback_best_in_memory],
                verbose=2,
            )

            weights_path = fold_weights_path(fold + 1)
            model.save_weights(weights_path)

            trained_models.append(model)

            print("\n\n")




## === cell 18
models = trained_models
assert len(models) > 0, "No models were trained; cannot proceed to create a submission."




## === cell 19
test_data = pd.read_csv(test_path)
test_data = add_features_fast(test_data)

test_data = dropCols(test_data, cols_2_drop)
test_data = rc.transform(test_data).astype(np.float32, copy=False)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])




## === cell 20
test_data.shape




## === cell 21
test_ds = make_ds_tf(test_data, y=None, training=False)

preds = []
for model in models:
    preds.append(
        model.predict(test_ds, verbose=0).reshape(-1).astype(np.float32, copy=False)
    )
preds = np.stack(preds, axis=1)  # (n, n_models)
median_pre = np.median(preds, axis=1).astype(np.float32, copy=False).reshape(-1, 1)

median_pre.shape




## === cell 22
submission_file = pd.read_csv(sample_sub)
if len(submission_file) != len(median_pre):
    raise ValueError(
        f"Prediction length mismatch: submission has {len(submission_file)} rows but predictions have {len(median_pre)}"
    )

submission_file["pressure"] = median_pre.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote: submission.csv")
