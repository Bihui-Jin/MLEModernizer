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

# 5. Target score

0.1487378239146059

# 6. Current score

17.65244

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 17.65244) has done: 'I fix the TensorFlow import crash by switching to the built-in `tf.keras` only path and disabling mixed-precision toggles that are incompatible with the current protobuf/TensorFlow stack in this environment. I fix the Transformer block call signature so Keras can trace the layer without requiring an explicit `training` argument, which is what currently stops model construction and prevents any predictions from being generated. I also fix the u_out weighting/masking logic so it correctly applies weights to inspiratory (u_out==0) timesteps after scaling, which should substantially reduce MAE toward your target without changing the model architecture. Finally, I make submission creation robust so it always writes a valid `.csv` even when running only one fold, using mean/median logic safely.'

# 9. Code solution

## === cell 0
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER = 81
FIRST_FOLD_ONLY = True
TRAIN_MODEL = False



## === cell 1
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

SEED = 42
keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("float32")
    print("Mixed precision disabled (float32 policy).")
except Exception as e:
    print("Mixed precision policy set skipped:", repr(e))



## === cell 3
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

print(train.shape, test.shape, submission.shape)




## === cell 4
def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    print("Step-1...Completed")

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
    df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
    df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
    df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
    df = df.fillna(0)
    print("Step-2...Completed")

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )
    print("Step-3...Completed")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
    print("Step-4...Completed")

    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["breath_id__u_in_lag"] = df["breath_id__u_in_lag"] * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["breath_id__u_in_lag2"] = df["breath_id__u_in_lag2"] * df["breath_id_lag2same"]
    print("Step-5...Completed")

    df["time_step_diff"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["ewm_u_in_mean"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=9)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(
            {
                "15_in_sum": "sum",
                "15_in_min": "min",
                "15_in_max": "max",
                "15_in_mean": "mean",
            }
        )
        .reset_index(level=0, drop=True)
    )
    print("Step-6...Completed")

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    print("Step-7...Completed")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")

    return df


train = add_features(train)
test = add_features(test)



## === cell 5
print("Train shape is now:", train.shape)
train.head()



## === cell 6
missing_in_test = [c for c in train.columns if c not in test.columns]
for c in missing_in_test:
    test[c] = 0
extra_in_test = [c for c in test.columns if c not in train.columns]
if extra_in_test:
    test = test.drop(columns=extra_in_test)
test = test[train.columns]
print(
    "Aligned columns. Missing added:",
    len(missing_in_test),
    "Extra dropped:",
    len(extra_in_test),
)



## === cell 7
train["pressure_diff"] = train.groupby("breath_id").pressure.diff().fillna(0)
train["pressure_integral"] = train.groupby("breath_id").pressure.cumsum() / 200
targets = (
    train[["pressure", "pressure_diff", "pressure_integral"]]
    .to_numpy()
    .reshape(-1, 80, 3)
)

train.drop(
    [
        "pressure",
        "pressure_diff",
        "pressure_integral",
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)

test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)



## === cell 8
print("Targets shape is", targets.shape)



## === cell 9
assert targets.shape[1] == 80, "Expected 80 timesteps per breath."



## === cell 10
COL_ORDER = (
    list(train.columns[:3]) + list(train.columns[-15:]) + list(train.columns[3:-15])
)
train = train[COL_ORDER]
test = test[COL_ORDER]

print("Num features:", train.shape[1])



## === cell 11
U_OUT_COLNAME = "u_out"
assert (
    U_OUT_COLNAME in train.columns
), "u_out column not found after feature engineering."
u_out_train_raw = train[U_OUT_COLNAME].to_numpy().reshape(-1, 80).astype(np.int8)
u_out_test_raw = test[U_OUT_COLNAME].to_numpy().reshape(-1, 80).astype(np.int8)



## === cell 12
RS = RobustScaler()
train = RS.fit_transform(train.astype("float32"))
test = RS.transform(test.astype("float32"))

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])



## === cell 13
print("Train reshaped:", train.shape, "Test reshaped:", test.shape)



## === cell 14
y_weight = np.ones((targets.shape[0], targets.shape[1], 1), dtype=np.float32)
y_weight[u_out_train_raw == 1] = 0.0



## === cell 15
train.shape, targets.shape, y_weight.shape



## === cell 16
if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") == 0:
    gpu_strategy = tf.distribute.get_strategy()
    print("single strategy")
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    print("multiple strategy")



## === cell 17
pass



## === cell 18
print("Using float32 training for stability.")



## === cell 19
pass




## === cell 20
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




## === cell 21
feat_dim = train.shape[-1] + 32
embed_dim = 64
num_heads = 8
ff_dim = 128
dropout_rate = 0.0
num_blocks = 12


def build_model():
    inputs = layers.Input(shape=train.shape[-2:])

    x = layers.Dense(feat_dim)(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    for k in range(num_blocks):
        x_old = x
        transformer_block = TransformerBlock(
            embed_dim, feat_dim, num_heads, ff_dim, dropout_rate
        )
        x = transformer_block(x)
        x = 0.7 * x + 0.3 * x_old  # SKIP CONNECTION

    x = layers.Dense(128, activation="selu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(3, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model




## === cell 22
pass



## === cell 23
import math
import matplotlib.pyplot as plt

LR_START = 1e-6
LR_MAX = 6e-4
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 420
STEPS = [60, 120, 240]


def lrfn(epoch):
    if epoch < STEPS[0]:
        epoch2 = epoch
        EPOCHS2 = STEPS[0]
    elif epoch < STEPS[0] + STEPS[1]:
        epoch2 = epoch - STEPS[0]
        EPOCHS2 = STEPS[1]
    else:
        epoch2 = epoch - STEPS[0] - STEPS[1]
        EPOCHS2 = STEPS[2]

    if epoch2 < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch2 + LR_START
    elif epoch2 < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr


rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, "-o")
print(
    "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
        lr_y[0], max(lr_y), lr_y[-1]
    )
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
plt.xlabel("Epoch", size=14)
plt.ylabel("Learning Rate", size=14)
plt.show()



## === cell 24
pass



## === cell 25
EPOCH = EPOCHS
BATCH_SIZE = 64
NUM_FOLDS = 11
VERBOSE = 1

with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    test_preds = []
    oof_preds = []
    oof_true = []
    all_mask = []
    test_folds = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        X_train, X_valid = train[train_idx], train[valid_idx]
        y_train, y_valid = targets[train_idx], targets[valid_idx]
        test_folds.append(valid_idx)

        checkpoint_filepath = f"folds{fold}_{VER}.hdf5"

        model = build_model()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="mae", sample_weight_mode="temporal")

        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        if TRAIN_MODEL:
            history = model.fit(
                X_train,
                y_train,
                verbose=VERBOSE,
                validation_data=(X_valid, y_valid, y_weight[valid_idx, :, :1]),
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                callbacks=[lr_callback, sv],
                sample_weight=y_weight[train_idx, :, :1],
            )
        else:
            wpath = f"../input/vent-tranformer/folds{fold}_{VER}.hdf5"
            if not os.path.exists(wpath):
                raise FileNotFoundError(
                    f"Pretrained weights not found at {wpath}. "
                    f"Either set TRAIN_MODEL=True or provide the weights dataset."
                )
            model.load_weights(wpath)

        print("Predicting Test...")
        pred_test = model.predict(test, batch_size=BATCH_SIZE, verbose=VERBOSE)[:, :, 0]
        test_preds.append(pred_test.reshape(-1))

        print("Predicting OOF...")
        pred_oof = model.predict(X_valid, batch_size=BATCH_SIZE, verbose=VERBOSE)[
            :, :, 0
        ]
        oof_preds.append(pred_oof.reshape(-1, 1))
        oof_true.append(y_valid[:, :, 0].reshape(-1, 1))

        score_all = mean_absolute_error(oof_true[-1], oof_preds[-1])
        print(f"Fold-{fold+1} | OOF all timesteps MAE: {score_all}")

        valid_u_out_raw = u_out_train_raw[valid_idx].reshape(-1)
        mask = np.where(valid_u_out_raw == 0)[0]
        mask_score = mean_absolute_error(oof_true[-1][mask], oof_preds[-1][mask])
        print(f"Fold-{fold+1} | OOF u_out=0 MAE: {mask_score}")
        all_mask.append(mask)

        np.save(
            f"oof_v{VER}_trans.npy",
            np.array(oof_preds, dtype=object),
            allow_pickle=True,
        )

        if FIRST_FOLD_ONLY:
            break



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3587968541.py in <cell line: 0>()
     21         checkpoint_filepath = f"folds{fold}_{VER}.hdf5"
     22 
---> 23         model = build_model()
     24         opt = tf.keras.optimizers.Adam(learning_rate=0.001)
     25         model.compile(optimizer=opt, loss="mae", sample_weight_mode="temporal")

/tmp/ipykernel_55/367684547.py in build_model()
     18             embed_dim, feat_dim, num_heads, ff_dim, dropout_rate
     19         )
---> 20         x = transformer_block(x)
     21         x = 0.7 * x + 0.3 * x_old  # SKIP CONNECTION
     22 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/3939224316.py in call(self, inputs, training)
     14     # The original signature required 'training' positional -> build crash.
     15     def call(self, inputs, training=None):
---> 16         attn_output = self.att(inputs, inputs)
     17         attn_output = self.dropout1(attn_output, training=training)
     18         out1 = self.layernorm1(inputs + attn_output)

TypeError: Exception encountered when calling TransformerBlock.call().

Could not automatically infer the output shape / dtype of 'transformer_block_1' (of type TransformerBlock). Either the `TransformerBlock.call()` method is incorrect, or you need to implement the `TransformerBlock.compute_output_spec() / compute_output_shape()` method. Error encountered:

Exception encountered when calling EinsumDense.call().

Failed to convert elements of SparseTensor(indices=Tensor("Placeholder_1:0", shape=(None, 3), dtype=int64), values=Tensor("Placeholder:0", shape=(None,), dtype=float32), dense_shape=Tensor("PlaceholderWithDefault:0", shape=(3,), dtype=int64)) to Tensor. Consider casting elements to a supported type. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.

Arguments received by EinsumDense.call():
  • inputs=tf.Tensor(shape=(None, 80, 96), dtype=float32)
  • training=None

Arguments received by TransformerBlock.call():
  • args=('<KerasTensor shape=(None, 80, 96), dtype=float32, sparse=True, name=keras_tensor_10>',)
  • kwargs=<class 'inspect._empty'>

## === cell 26
if FIRST_FOLD_ONLY:
    NUM_FOLDS = 1
print(
    "Folds run:",
    NUM_FOLDS,
    "Test preds:",
    len(test_preds),
    "OOF preds:",
    len(oof_preds),
)



## === cell 27
pass



## === cell 28
pass



## === cell 29
t = 0.0
for k in range(NUM_FOLDS):
    mask = all_mask[k]
    mae = np.mean(np.abs(oof_preds[k].flatten()[mask] - oof_true[k].flatten()[mask]))
    t += mae
    print("Fold", k, "has u_out=0 MAE =", mae)
print("Overall CV u_out=0 MAE =", t / NUM_FOLDS)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3107705551.py in <cell line: 0>()
      1 t = 0.0
      2 for k in range(NUM_FOLDS):
----> 3     mask = all_mask[k]
      4     mae = np.mean(np.abs(oof_preds[k].flatten()[mask] - oof_true[k].flatten()[mask]))
      5     t += mae

IndexError: list index out of range

## === cell 30
pass



## === cell 31
t = 0.0
for k in range(NUM_FOLDS):
    oof = oof_preds[k].copy()
    oof2 = (
        np.round((oof + 1.895744294564641) / 0.07030214545121005) * 0.07030214545121005
        - 1.895744294564641
    )
    mask = all_mask[k]
    mae = np.mean(np.abs(oof2.flatten()[mask] - oof_true[k].flatten()[mask]))
    t += mae
    print("Fold", k, "has u_out=0 MAE with PP =", mae)
print("Overall CV u_out=0 MAE with PP =", t / NUM_FOLDS)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3375993067.py in <cell line: 0>()
      1 t = 0.0
      2 for k in range(NUM_FOLDS):
----> 3     oof = oof_preds[k].copy()
      4     oof2 = (
      5         np.round((oof + 1.895744294564641) / 0.07030214545121005) * 0.07030214545121005

IndexError: list index out of range

## === cell 32
pass



## === cell 33
train_raw = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

folds = test_folds.copy()
for k in range(len(folds)):
    folds[k] = np.ones_like(folds[k]) * k
folds = np.hstack(folds)
folds = np.repeat(folds, 80)

valid_rows = np.hstack(test_folds)
valid_rows = 80 * np.repeat(valid_rows, 80)
shifter = np.tile(np.arange(80), len(valid_rows) // 80)
valid_rows = valid_rows + shifter

train_oof = train_raw.loc[valid_rows].copy()

oof_stack = np.vstack(oof_preds).reshape(-1)
train_oof["oof"] = oof_stack.astype("float32")
train_oof["fold"] = folds.astype("int16")

train_oof.head()



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/374346876.py in <cell line: 0>()
     15 train_oof = train_raw.loc[valid_rows].copy()
     16 
---> 17 oof_stack = np.vstack(oof_preds).reshape(-1)
     18 train_oof["oof"] = oof_stack.astype("float32")
     19 train_oof["fold"] = folds.astype("int16")

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 34
train_oof.id = train_oof.id.astype("int32")
train_oof.oof = train_oof.oof.astype("float32")
train_oof.fold = train_oof.fold.astype("int8")
train_oof[["id", "oof", "fold"]].to_csv(f"oof_v{VER}.csv", index=False)
train_oof[["id", "oof", "fold"]].head()



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3185401390.py in <cell line: 0>()
      1 train_oof.id = train_oof.id.astype("int32")
----> 2 train_oof.oof = train_oof.oof.astype("float32")
      3 train_oof.fold = train_oof.fold.astype("int8")
      4 train_oof[["id", "oof", "fold"]].to_csv(f"oof_v{VER}.csv", index=False)
      5 train_oof[["id", "oof", "fold"]].head()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'oof'

## === cell 35
pass



## === cell 36
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
pred_mean = np.mean(np.vstack(test_preds), axis=0)
submission["pressure"] = pred_mean
submission.to_csv(f"submission_mean_{VER}.csv", index=False)
print("Wrote:", f"submission_mean_{VER}.csv", "shape:", submission.shape)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1835603434.py in <cell line: 0>()
      3 )
      4 # Robust mean over folds (works for NUM_FOLDS=1 as well)
----> 5 pred_mean = np.mean(np.vstack(test_preds), axis=0)
      6 submission["pressure"] = pred_mean
      7 submission.to_csv(f"submission_mean_{VER}.csv", index=False)

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 37
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
pred_median = np.median(np.vstack(test_preds), axis=0)
submission["pressure"] = pred_median
submission.to_csv(f"submission_median_{VER}.csv", index=False)
print("Wrote:", f"submission_median_{VER}.csv", "shape:", submission.shape)



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2197434595.py in <cell line: 0>()
      3     "../input/ventilator-pressure-prediction/sample_submission.csv"
      4 )
----> 5 pred_median = np.median(np.vstack(test_preds), axis=0)
      6 submission["pressure"] = pred_median
      7 submission.to_csv(f"submission_median_{VER}.csv", index=False)

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 38
submission.head()



## === cell 39
submission.pressure = (
    np.round((submission.pressure + 1.895744294564641) / 0.07030214545121005)
    * 0.07030214545121005
    - 1.895744294564641
)
submission.to_csv(f"submission_median_snap_{VER}.csv", index=False)
print("Wrote:", f"submission_median_snap_{VER}.csv")



## === cell 40
submission.head()
