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

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation (common Kaggle issue) before importing TF. Then I fix the Keras layer call signature so `training` is optional, which unblocks model building and prediction. Next, I correct the `u_out` masking/sample-weight logic (it currently inverts weights and makes evaluation nonsensical) and keep the rest of the training/inference pipeline unchanged. Finally, I make the later OOF/median-submission cells robust to `FIRST_FOLD_ONLY` so they don’t index past available folds, and ensure at least one valid `submission_*.csv` is always written.'
- What this solution (achieved 17.65244) has done: 'I fix the TensorFlow/protobuf crash by enforcing a protobuf version/implementation combo that works reliably in Kaggle’s TF runtime, and I make model inputs dense to resolve the `sparse=True` KerasTensor issue that breaks `MultiHeadAttention/EinsumDense` shape inference. Next, I correct the sample-weight masking bug (it currently zeros the wrong axis) so training/validation loss matches the competition metric (only inspiratory phase `u_out=0` is scored), which should sharply reduce the MAE toward your target. Finally, I make the pipeline always produce at least one valid `submission_*.csv` even if pretrained weights aren’t available, by training when needed and by guarding the OOF/ensemble post-processing against empty fold lists.'
- What this solution (achieved 17.65244) has done: 'I fix the TensorFlow/protobuf crash by removing the fragile runtime protobuf downgrade logic and instead enforcing the pure-Python protobuf implementation (which is the intended workaround for this Kaggle environment). Then I fix the Transformer build crash by making the Keras `Input` explicitly dense (`sparse=False`) and by ensuring the attention layer is called in a way that Keras can trace shapes/dtypes reliably. Finally, I keep your training/inference logic intact (including the u_out=0 masking via sample weights) but add small guards so the notebook always writes at least one valid `submission_*.csv` even when `FIRST_FOLD_ONLY=True` and downstream fold-aggregation cells would otherwise error.'
- What this solution (achieved 17.65244) has done: 'I fix the TensorFlow/protobuf import crash by safely forcing the pure-Python protobuf implementation early and, if needed, falling back to a compatible protobuf version shipped in the Kaggle image (without changing any model/training semantics). Then I fix the Transformer build error by ensuring the tensors passed into `MultiHeadAttention` are dense (Keras is currently marking them as sparse, which breaks `EinsumDense`), while keeping your architecture and loops unchanged. Finally, I add small guards so that even with `FIRST_FOLD_ONLY=True` the notebook still completes fold aggregation cleanly and always writes at least one valid `submission_*.csv` file. These fixes are expected to dramatically reduce MAE from the current broken/uncalibrated state toward your target because the model actually train/predict correctly and the inspiratory-phase masking be applied as intended.'
- What this solution (achieved 17.65244) has done: 'I (1) make TensorFlow import reliable in this Kaggle environment by forcing the pure-Python protobuf implementation and removing the runtime pip-install fallback that still crashes. Then (2) I fix the TransformerBlock to avoid converting symbolic KerasTensors to eager tensors (which is what triggers the sparse/shape inference failure) by conditionally densifying only real SparseTensors and otherwise leaving the KerasTensor untouched. Finally, (3) I keep your training/inference logic the same but add small guards so that even with `FIRST_FOLD_ONLY=True` the code always completes, aggregates predictions safely, and writes at least one valid `submission_*.csv` file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

VER = 81
FIRST_FOLD_ONLY = True
TRAIN_MODEL = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import random

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)
tf.random.set_seed(SEED_GLOBAL)



## === cell 2
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)




## === cell 3
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



## === cell 4
print("Train shape is now:", train.shape)
train.head()



## === cell 5
missing_in_test = [c for c in train.columns if c not in test.columns]
for c in missing_in_test:
    test[c] = 0
extra_in_test = [c for c in test.columns if c not in train.columns]
if extra_in_test:
    test = test.drop(columns=extra_in_test)
test = test[train.columns]



## === cell 6
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



## === cell 7
print("Targets shape is", targets.shape)



## === cell 8
print("Train features:", train.shape, "Test features:", test.shape)



## === cell 9
COL_ORDER = (
    list(train.columns[:3]) + list(train.columns[-15:]) + list(train.columns[3:-15])
)
train = train[COL_ORDER]
test = test[COL_ORDER]

print("Train columns:")
np.array(COL_ORDER)



## === cell 10
assert train.shape[1] == test.shape[1]



## === cell 11
RS = RobustScaler()
train = RS.fit_transform(train.astype("float32"))
test = RS.transform(test.astype("float32"))

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

train = np.asarray(train, dtype=np.float32)
test = np.asarray(test, dtype=np.float32)



## === cell 12
print("Reshaped train/test:", train.shape, test.shape)



## === cell 13
U_OUT_IDX = 2  # keep for mask computation on X_valid as in original logic

train_raw = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", usecols=["breath_id", "u_out"]
)
u_out_raw_seq = train_raw["u_out"].to_numpy().reshape(-1, 80).astype(np.float32)

y_weight = (u_out_raw_seq == 0).astype(np.float32)[..., None]  # (n_breaths, 80, 1)



## === cell 14
train.shape, targets.shape, y_weight.shape



## === cell 15
print(
    "Weight stats:", y_weight.min(), y_weight.max(), "fraction kept:", y_weight.mean()
)



## === cell 16
if os.environ["CUDA_VISIBLE_DEVICES"].count(",") == 0:
    gpu_strategy = tf.distribute.get_strategy()
    print("single strategy")
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    print("multiple strategy")



## === cell 17
try:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled (experimental)")
except Exception as e:
    print("Mixed precision not enabled:", repr(e))



## === cell 18
tf.keras.backend.set_floatx("float32")




## === cell 19
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dense(feat_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        if isinstance(inputs, tf.SparseTensor):
            x = tf.sparse.to_dense(inputs)
        else:
            x = inputs

        attn_output = self.att(query=x, value=x, key=x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




## === cell 20
feat_dim = train.shape[-1] + 32
embed_dim = 64  # Embedding size for attention
num_heads = 8  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
dropout_rate = 0.0
num_blocks = 12


def build_model():
    inputs = layers.Input(shape=train.shape[-2:], dtype="float32", sparse=False)

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
    outputs = layers.Dense(3, activation="linear", dtype="float32")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model




## === cell 21
with gpu_strategy.scope():
    _m = build_model()
_m.summary()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2257314499.py in <cell line: 0>()
      1 with gpu_strategy.scope():
----> 2     _m = build_model()
      3 _m.summary()
      4 

/tmp/ipykernel_55/2886864301.py in build_model()
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

/tmp/ipykernel_55/2310643137.py in call(self, inputs, training)
     22             x = inputs
     23 
---> 24         attn_output = self.att(query=x, value=x, key=x, training=training)
     25         attn_output = self.dropout1(attn_output, training=training)
     26         out1 = self.layernorm1(x + attn_output)

ValueError: Exception encountered when calling TransformerBlock.call().

Could not automatically infer the output shape / dtype of 'transformer_block_1' (of type TransformerBlock). Either the `TransformerBlock.call()` method is incorrect, or you need to implement the `TransformerBlock.compute_output_spec() / compute_output_shape()` method. Error encountered:

Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 8, 64) for variable path='transformer_block_1/multi_head_attention_1/query/kernel'

Arguments received by TransformerBlock.call():
  • args=('<KerasTensor shape=(None, 80, 96), dtype=float32, sparse=True, name=keras_tensor_10>',)
  • kwargs=<class 'inspect._empty'>

## === cell 22
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
    elif epoch < STEPS[0] + STEPS[1] + STEPS[2]:
        epoch2 = epoch - STEPS[0] - STEPS[1]
        EPOCHS2 = STEPS[2]
    else:
        epoch2 = epoch - (STEPS[0] + STEPS[1] + STEPS[2])
        EPOCHS2 = max(1, STEPS[-1])

    if epoch2 < LR_RAMPUP_EPOCHS and LR_RAMPUP_EPOCHS > 0:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch2 + LR_START
    elif epoch2 < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = max(1, EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1)
        decay_epoch_index = epoch2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return float(lr)


rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, "-o")
print(
    "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
        lr_y[0], max(lr_y), lr_y[-1]
    )
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
plt.xlabel("Epoch", size=14)
plt.ylabel("Learning Rate", size=14)
plt.show()



## === cell 23
import glob


def _find_weight_file(fold, ver):
    p = f"../input/vent-tranformer/folds{fold}_{ver}.hdf5"
    if os.path.exists(p):
        return p
    return None




## === cell 24
EPOCH = EPOCHS
BATCH_SIZE = 64
NUM_FOLDS = 11
SEED = 42
VERBOSE = 1

_pretrained_any = _find_weight_file(0, VER) is not None
if not TRAIN_MODEL and not _pretrained_any:
    print(
        "Pretrained weights not found; switching TRAIN_MODEL=True to ensure a valid submission is produced."
    )
    TRAIN_MODEL = True

with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    test_preds = []
    oof_preds = []
    oof_true = []
    all_mask = []
    test_folds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        test_folds.append(test_idx)

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
            options=None,
        )
        if TRAIN_MODEL:
            history = model.fit(
                X_train,
                y_train,
                verbose=VERBOSE,
                validation_data=(X_valid, y_valid, y_weight[test_idx, :, :1]),
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                callbacks=[lr_callback, sv],
                sample_weight=y_weight[train_idx, :, :1],
            )
            if os.path.exists(checkpoint_filepath):
                model.load_weights(checkpoint_filepath)
        else:
            wpath = _find_weight_file(fold, VER)
            if wpath is None:
                raise FileNotFoundError(
                    f"Pretrained weights not found for fold {fold}, VER {VER}. "
                    f"Expected at ../input/vent-tranformer/folds{fold}_{VER}.hdf5"
                )
            model.load_weights(wpath)

        print("Predicting Test...")
        test_preds.append(
            model.predict(test, batch_size=BATCH_SIZE, verbose=VERBOSE)[
                :, :, 0
            ].reshape(-1)
        )

        print("Predicting OOF...")
        oof_preds.append(
            model.predict(X_valid, batch_size=BATCH_SIZE, verbose=VERBOSE)[
                :, :, 0
            ].reshape(-1, 1)
        )
        oof_true.append(y_valid[:, :, 0].reshape(-1, 1))

        u_out_valid_raw = u_out_raw_seq[test_idx].reshape(-1)
        mask = np.where(u_out_valid_raw == 0)[0]
        all_mask.append(mask)

        score_all = mean_absolute_error(oof_true[-1], oof_preds[-1])
        score_insp = mean_absolute_error(oof_true[-1][mask], oof_preds[-1][mask])
        print(f"Fold-{fold+1} | OOF all timesteps MAE: {score_all}")
        print(f"Fold-{fold+1} | OOF inspiratory (u_out=0) MAE: {score_insp}")

        np.save(f"oof_v{VER}_trans", oof_preds)

        if FIRST_FOLD_ONLY:
            break



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/464698671.py in <cell line: 0>()
     29         checkpoint_filepath = f"folds{fold}_{VER}.hdf5"
     30 
---> 31         model = build_model()
     32         opt = tf.keras.optimizers.Adam(learning_rate=0.001)
     33         model.compile(optimizer=opt, loss="mae", sample_weight_mode="temporal")

/tmp/ipykernel_55/2886864301.py in build_model()
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

/tmp/ipykernel_55/2310643137.py in call(self, inputs, training)
     22             x = inputs
     23 
---> 24         attn_output = self.att(query=x, value=x, key=x, training=training)
     25         attn_output = self.dropout1(attn_output, training=training)
     26         out1 = self.layernorm1(x + attn_output)

ValueError: Exception encountered when calling TransformerBlock.call().

Could not automatically infer the output shape / dtype of 'transformer_block_3' (of type TransformerBlock). Either the `TransformerBlock.call()` method is incorrect, or you need to implement the `TransformerBlock.compute_output_spec() / compute_output_shape()` method. Error encountered:

Shapes used to initialize variables must be fully-defined (no `None` dimensions). Received: shape=(None, 8, 64) for variable path='transformer_block_3/multi_head_attention_3/query/kernel'

Arguments received by TransformerBlock.call():
  • args=('<KerasTensor shape=(None, 80, 96), dtype=float32, sparse=True, name=keras_tensor_21>',)
  • kwargs=<class 'inspect._empty'>

## === cell 25
num_done_folds = len(oof_preds)
print("Folds completed:", num_done_folds)



## === cell 26
NUM_FOLDS_USED = num_done_folds



## === cell 27
assert NUM_FOLDS_USED > 0, "No folds were run; cannot create submission."



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/548568136.py in <cell line: 0>()
----> 1 assert NUM_FOLDS_USED > 0, "No folds were run; cannot create submission."
      2 

AssertionError: No folds were run; cannot create submission.

## === cell 28
t = 0.0
for k in range(NUM_FOLDS_USED):
    mask = all_mask[k]
    mae = np.mean(np.abs(oof_preds[k].flatten()[mask] - oof_true[k].flatten()[mask]))
    t += mae
    print("Fold", k, "has inspiratory MAE =", mae)
print("Overall CV inspiratory MAE =", t / NUM_FOLDS_USED)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_55/1304105873.py in <cell line: 0>()
      5     t += mae
      6     print("Fold", k, "has inspiratory MAE =", mae)
----> 7 print("Overall CV inspiratory MAE =", t / NUM_FOLDS_USED)
      8 

ZeroDivisionError: float division by zero

## === cell 29
t = 0.0
for k in range(NUM_FOLDS_USED):
    mask_k = all_mask[k]
    oof = oof_preds[k].copy()
    oof2 = (
        np.round((oof + 1.895744294564641) / 0.07030214545121005) * 0.07030214545121005
        - 1.895744294564641
    )
    mae = np.mean(np.abs(oof2.flatten()[mask_k] - oof_true[k].flatten()[mask_k]))
    t += mae
    print("Fold", k, "has inspiratory MAE with PP =", mae)
print("Overall CV inspiratory MAE with PP =", t / NUM_FOLDS_USED)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_55/1539111001.py in <cell line: 0>()
     10     t += mae
     11     print("Fold", k, "has inspiratory MAE with PP =", mae)
---> 12 print("Overall CV inspiratory MAE with PP =", t / NUM_FOLDS_USED)
     13 

ZeroDivisionError: float division by zero

## === cell 30
print("Num test_preds:", len(test_preds), "each length:", test_preds[0].shape[0])



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3095920215.py in <cell line: 0>()
----> 1 print("Num test_preds:", len(test_preds), "each length:", test_preds[0].shape[0])
      2 

IndexError: list index out of range

## === cell 31
train_df = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

folds = test_folds.copy()
for k in range(len(folds)):
    folds[k] = np.ones_like(folds[k]) * k
folds = np.hstack(folds)
folds = np.repeat(folds, 80)

test_folds_flat = np.hstack(test_folds)
test_folds_flat = 80 * np.repeat(test_folds_flat, 80)
shifter = np.tile(np.arange(80), len(test_folds_flat) // 80)
test_folds_flat += shifter

train_df = train_df.loc[test_folds_flat].copy()

oof_stack = np.vstack(oof_preds)  # (N,1)
train_df["oof"] = oof_stack.squeeze()
train_df["fold"] = folds[: len(train_df)]

train_df.head()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1029142627.py in <cell line: 0>()
     15 train_df = train_df.loc[test_folds_flat].copy()
     16 
---> 17 oof_stack = np.vstack(oof_preds)  # (N,1)
     18 train_df["oof"] = oof_stack.squeeze()
     19 train_df["fold"] = folds[: len(train_df)]

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 32
train_df["id"] = train_df["id"].astype("int32")
train_df["oof"] = train_df["oof"].astype("float32")
train_df["fold"] = train_df["fold"].astype("int8")
train_df[["id", "oof", "fold"]].to_csv(f"oof_v{VER}.csv", index=False)
train_df[["id", "oof", "fold"]].head()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'oof'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2572667822.py in <cell line: 0>()
      1 train_df["id"] = train_df["id"].astype("int32")
----> 2 train_df["oof"] = train_df["oof"].astype("float32")
      3 train_df["fold"] = train_df["fold"].astype("int8")
      4 train_df[["id", "oof", "fold"]].to_csv(f"oof_v{VER}.csv", index=False)
      5 train_df[["id", "oof", "fold"]].head()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'oof'

## === cell 33
test_pred_mean = np.mean(np.vstack(test_preds), axis=0)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3597664195.py in <cell line: 0>()
----> 1 test_pred_mean = np.mean(np.vstack(test_preds), axis=0)
      2 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 34
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = test_pred_mean
submission.to_csv(f"submission_mean_{VER}.csv", index=False)
print("Wrote:", f"submission_mean_{VER}.csv", "shape:", submission.shape)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/758312024.py in <cell line: 0>()
      2     "../input/ventilator-pressure-prediction/sample_submission.csv"
      3 )
----> 4 submission["pressure"] = test_pred_mean
      5 submission.to_csv(f"submission_mean_{VER}.csv", index=False)
      6 print("Wrote:", f"submission_mean_{VER}.csv", "shape:", submission.shape)

NameError: name 'test_pred_mean' is not defined

## === cell 35
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = np.median(np.vstack(test_preds), axis=0)
submission.to_csv(f"submission_median_{VER}.csv", index=False)
print("Wrote:", f"submission_median_{VER}.csv", "shape:", submission.shape)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1181717753.py in <cell line: 0>()
      2     "../input/ventilator-pressure-prediction/sample_submission.csv"
      3 )
----> 4 submission["pressure"] = np.median(np.vstack(test_preds), axis=0)
      5 submission.to_csv(f"submission_median_{VER}.csv", index=False)
      6 print("Wrote:", f"submission_median_{VER}.csv", "shape:", submission.shape)

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 36
submission.head()



## === cell 37
submission["pressure"] = (
    np.round((submission["pressure"] + 1.895744294564641) / 0.07030214545121005)
    * 0.07030214545121005
    - 1.895744294564641
)
submission.to_csv(f"submission_median_snap_{VER}.csv", index=False)
print("Wrote:", f"submission_median_snap_{VER}.csv", "shape:", submission.shape)



## === cell 38
submission.head()
