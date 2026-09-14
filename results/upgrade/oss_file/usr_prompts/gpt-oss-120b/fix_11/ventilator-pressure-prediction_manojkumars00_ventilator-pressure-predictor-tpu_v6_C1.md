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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

mixed_policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(mixed_policy)




## === cell 1
BASE_PATH = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")




## === cell 2
def dropCols(df, cols):
    """Return a copy of *df* with the specified columns removed."""
    df = df.copy()
    df.drop(cols, axis=1, inplace=True)
    return df




## === cell 3
dtypes = {
    "C": np.int8,
    "R": np.int8,
    "breath_id": np.int32,
    "id": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
train_data = pd.read_csv(train_path, dtype=dtypes)

train_data["diff_u_in1"] = train_data["u_in"] - train_data.groupby(
    "breath_id", sort=False
)["u_in"].shift(1).fillna(0)
train_data["u_in_cumsum"] = (
    train_data["u_in"].groupby(train_data["breath_id"], sort=False).cumsum()
)




## === cell 4
cols_2_drop = ["id", "breath_id"]




## === cell 5
train_df = dropCols(train_data, cols_2_drop)
Y = train_df.pop("pressure")  # target




## === cell 6
rb = RobustScaler()
rb.fit(train_df)
train_df = rb.transform(train_df).astype(np.float16)
Y = Y.values.astype(np.float16)




## === cell 7
train_df = train_df.reshape(-1, 80, train_df.shape[-1])
Y = Y.reshape(-1, 80, 1)

print("train shape:", train_df.shape, "label shape:", Y.shape)




## === cell 8
def build_model():
    """Build the LSTM‑based regression model."""
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(300, return_sequences=True),
            input_shape=[80, train_df.shape[-1]],
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                150,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(80, return_sequences=True, kernel_initializer="random_normal")
        )
    )
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 9
def scheduler(epoch, lr):
    """Learning‑rate scheduler used during training."""
    if epoch > 200 and epoch % 10 == 0:
        return lr * tf.math.exp(-0.01)
    return lr


callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)




## === cell 10
EPOCH = 100
BATCH_SIZE = 4096  # larger batch → fewer steps per epoch, same total data passes

try:
    strategy = tf.distribute.MirroredStrategy()
except Exception:
    strategy = tf.distribute.get_strategy()

models = []  # will hold the 5‑fold models

with strategy.scope():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
        print("-" * 15, f"Fold {fold+1}", "-" * 15)
        X_train, X_valid = train_df[train_idx], train_df[valid_idx]
        y_train, y_valid = Y[train_idx], Y[valid_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .cache()
            .shuffle(1024, seed=42)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        valid_ds = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .cache()
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        model = build_model()
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            f"AdamPressureFold{fold+1}.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            callbacks=[callback_lr, checkpoint_cb],
            verbose=0,
        )

        model.load_weights(f"AdamPressureFold{fold+1}.weights.h5")
        models.append(model)

        tf.keras.backend.clear_session()




## === cell 11
test_data = pd.read_csv(test_path, dtype=dtypes)

test_data["diff_u_in1"] = test_data["u_in"] - test_data.groupby(
    "breath_id", sort=False
)["u_in"].shift(1).fillna(0)
test_data["u_in_cumsum"] = (
    test_data["u_in"].groupby(test_data["breath_id"], sort=False).cumsum()
)

test_data = dropCols(test_data, cols_2_drop)
test_data = rb.transform(test_data).astype(np.float16)
test_data = test_data.reshape(-1, 80, test_data.shape[-1])




## === cell 12
preds = []
for model in models:
    pred = model.predict(test_data, batch_size=BATCH_SIZE).reshape(-1, 1)
    preds.append(pred)

if not preds:
    raise RuntimeError("Model list is empty – training may have failed.")

preds = np.mean(
    np.stack(preds, axis=0).astype(np.float32), axis=0
)  # shape (num_rows, 1)




## === cell 13
submission = pd.read_csv(sample_sub_path)
submission["pressure"] = preds.reshape(-1)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
