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

# 5. Target score

0.1663

# 6. Current score

17.69475

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 17.69475) has done: 'I fix the environment-breaking imports by removing notebook-only magics and forcing protobuf to use the pure-Python implementation to avoid the `MessageFactory.GetPrototype` crash seen with protobuf 6.x. Then I make TPU usage optional (fallback to GPU/CPU with `MirroredStrategy`/default strategy) so the script runs in Kaggle even when no TPU is attached, which unblocks training and model checkpoint saving. I also fix model saving/loading to use a Keras-native `.keras` file (more stable with TF/Keras 2.18 than legacy `.h5`) and correct the submission creation so it always writes a valid `submission.csv` with `id,pressure`. These changes preserve the same model/feature logic while ensuring the pipeline runs end-to-end and produces a submission.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau,
)

from sklearn.model_selection import KFold, GroupKFold, train_test_split

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

rb = RobustScaler()
sc = StandardScaler()



## === cell 2
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")



## === cell 3
train[train < 0].count()



## === cell 4
train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0)
train["u_in_lag2"] = train.groupby("breath_id")["u_in"].shift(2).fillna(0)
train["u_in_diff1"] = train["u_in"] - train["u_in_lag1"]
train["u_in_diff2"] = train["u_in"] - train["u_in_lag2"]
train["u_in_cumsum"] = train["u_in"].groupby(train["breath_id"]).cumsum()

test["u_in_lag1"] = test.groupby("breath_id")["u_in"].shift(1).fillna(0)
test["u_in_lag2"] = test.groupby("breath_id")["u_in"].shift(2).fillna(0)
test["u_in_diff1"] = test["u_in"] - test["u_in_lag1"]
test["u_in_diff2"] = test["u_in"] - test["u_in_lag2"]
test["u_in_cumsum"] = test["u_in"].groupby(test["breath_id"]).cumsum()



## === cell 5
train



## === cell 6
targets = train["pressure"].to_numpy().reshape(-1, 80, 1)
test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure"], inplace=True)
test.drop(columns=["id", "breath_id"], inplace=True)



## === cell 7
rb.fit(train)

train_new = rb.transform(train)
test_new = rb.transform(test)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])



## === cell 8
"""
rb.fit(train)

train_new = rb.transform(train)
test_new = rb.transform(test)

train_re = train_new.reshape(-1, 80, train_new.shape[-1])
test_re = test_new.reshape(-1, 80, test_new.shape[-1])
"""




## === cell 9
def build_model0(input_shape, name):
    model = tf.keras.Sequential(name=name)

    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                360,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(layers.LSTM(260, dropout=0.25, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                180,
                dropout=0.2,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(100, return_sequences=True)))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))

    return model


def build_model1(input_shape, name):
    model = tf.keras.Sequential(name=name)

    model.add(
        layers.Bidirectional(
            layers.LSTM(520, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(layers.LSTM(400, dropout=0.2, return_sequences=True))
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                320,
                dropout=0.3,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(240, return_sequences=True)))
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                160,
                dropout=0.25,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(layers.Bidirectional(layers.LSTM(80, return_sequences=True)))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.TimeDistributed(layers.Dense(1, name=name)))

    return model


def build_model2(input_shape, name):
    model = tf.keras.Sequential(name=name)

    model.add(
        layers.Bidirectional(
            layers.LSTM(440, return_sequences=True, input_shape=input_shape)
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(360, return_sequences=True, kernel_initializer="random_normal")
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                260,
                dropout=0.3,
                return_sequences=True,
                kernel_initializer="random_normal",
            )
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(180, return_sequences=True, kernel_initializer="random_normal")
        )
    )
    model.add(
        layers.Bidirectional(
            layers.LSTM(120, return_sequences=True, kernel_initializer="random_normal")
        )
    )

    model.add(layers.Dense(64, activation="linear"))
    model.add(layers.Dense(1, activation="linear", name=name))

    return model


def build_ensembleModel(input_shape, name):
    model = tf.keras.Sequential(name=name)
    model.add(layers.Dense(16, activation="relu", input_shape=input_shape))
    model.add(layers.Dense(1, name=name))
    return model




## === cell 10
def build_model(input_size=[80, train_re.shape[-1]]):
    model0 = build_model0(input_size, name="model0")
    model1 = build_model1(input_size, name="model1")
    model2 = build_model1(input_size, name="model2")
    dupmodel2 = build_model2(input_size, name="dupmodel2")

    ensemblemodel = build_ensembleModel([80, 5], name="ensemble")

    Input = tf.keras.layers.Input(input_size)

    model0_out = model0(Input)
    model1_out = model1(Input)
    model2_out = model2(Input)
    dupmodel_2 = dupmodel2(Input)

    mean_out = layers.Add()([model0_out, model1_out, model2_out, dupmodel_2])
    mean_out = layers.Lambda(lambda x: x / 4, name="mean")(mean_out)

    concatenateModelsout = layers.Concatenate()(
        [model0_out, model1_out, model2_out, dupmodel_2, mean_out]
    )
    ensemble_out = ensemblemodel(concatenateModelsout)

    model = tf.keras.Model(
        inputs=Input,
        outputs=[
            model0_out,
            model1_out,
            model2_out,
            dupmodel_2,
            mean_out,
            ensemble_out,
        ],
    )

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError(), metrics=["mae"]
    )
    return model




## === cell 11
EPOCH = 400
BATCH_SIZE = 768

try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # auto-detect
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.TPUStrategy(tpu_resolver)
    print("Using TPU strategy")
except Exception as e:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print("Using MirroredStrategy on multiple GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        print("Using default strategy (CPU/GPU). TPU not available:", repr(e))



## === cell 12
checkpoint_path = "LModel.keras"

with strategy.scope():
    model = build_model()

    X_train, X_test, y_train, y_test = train_test_split(
        train_re, targets, test_size=0.2, shuffle=True, random_state=42
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, factor=0.9, patience=10
    )
    save_call = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, monitor="val_ensemble_mae", save_best_only=True
    )

    his = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        callbacks=[save_call, reduce_lr],
        verbose=2,
    )

stats = pd.DataFrame(his.history)
ax = stats.plot(figsize=(10, 6))
ax.grid(True)
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2787481646.py in <cell line: 0>()
     16     )
     17 
---> 18     his = model.fit(
     19         X_train,
     20         y_train,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in _build_nested(self, y_true, y_pred, loss, output_names, current_path)
    551                 continue
    552             if not key_check_fn(key, (y_true, y_pred)):
--> 553                 raise KeyError(
    554                     f"The path: {current_path + (key,)} in "
    555                     "the `loss` argument, can't be found in "

KeyError: "The path: (0,) in the `loss` argument, can't be found in either the model's output (`y_pred`) or in the labels (`y_true`)."

## === cell 13
if os.path.exists(checkpoint_path):
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
else:
    print("Warning: checkpoint not found; using current in-memory model.")



## === cell 14
test_pred = model.predict(test_re, verbose=1)



## === cell 15
"""
test_preds = []

for model in models:
    test_preds.append(model.predict(test_re).reshape(-1, 1))
test_pred = sum(test_preds)/3
"""



## === cell 16
pred_pressure = test_pred[-2].reshape(
    -1,
)



## === cell 17
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission_file["pressure"] = pred_pressure
submission_file.to_csv("submission.csv", index=False)
print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
