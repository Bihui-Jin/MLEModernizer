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

0.1811602724232202

# 6. Current score

109.66626

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 109.66626) has done: 'I fix the TensorFlow import/runtime crash by switching away from `tensorflow.python.keras` to the public `tf.keras` API, which avoids the `MessageFactory.GetPrototype` protobuf incompatibility. I also remove the hard requirement for TPU (since your environment doesn’t provide one) and automatically fall back to CPU/GPU while keeping the exact same model architecture. Next, I replace the missing `more-feature-testdata/*.npy` dependency by reconstructing the required test features directly from the provided `test.csv` (and similarly build the `rc_input`), preserving the semantics of having an `(N,80,features)` tensor plus an `(N,15)` RC vector. Finally, I make submission generation use the real `id` column from `test.csv` (not `1..N`) and ensure `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input"
WORKING_DIR = "/kaggle/working"

shown = 0
for dirname, _, filenames in os.walk(INPUT_DIR):
    for filename in filenames:
        if shown < 50:
            print(os.path.join(dirname, filename))
            shown += 1

COMP_DIR = os.path.join(INPUT_DIR, "ventilator-pressure-prediction")
if not os.path.exists(os.path.join(COMP_DIR, "test.csv")):
    COMP_DIR = INPUT_DIR

test_path = os.path.join(COMP_DIR, "test.csv")
sample_sub_path = os.path.join(COMP_DIR, "sample_submission.csv")

assert os.path.exists(test_path), f"Missing test.csv at: {test_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at: {sample_sub_path}"

test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)
print("test_df shape:", test_df.shape)
print("sample_sub shape:", sample_sub.shape)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    losses,
    optimizers,
    activations,
    initializers,
    backend,
    metrics,
)

tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2

steps_per_breath = test_df.groupby("breath_id")["id"].size().unique()
print(
    "Unique steps per breath:",
    steps_per_breath[:10],
    " ... total unique:",
    len(steps_per_breath),
)
assert (
    len(steps_per_breath) == 1 and steps_per_breath[0] == 80
), "Unexpected steps per breath; expected 80."

test_df = test_df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

breath_meta = test_df.groupby("breath_id", sort=False)[["R", "C"]].first().reset_index()
R = breath_meta["R"].astype(np.int32).values
C = breath_meta["C"].astype(np.int32).values

rc_input = np.stack(
    [
        R,
        C,
        R * C,
        R + C,
        R - C,
        C - R,
        R * R,
        C * C,
        (R == 5).astype(np.int32),
        (R == 20).astype(np.int32),
        (R == 50).astype(np.int32),
        (C == 10).astype(np.int32),
        (C == 20).astype(np.int32),
        (C == 50).astype(np.int32),
        np.ones_like(R, dtype=np.int32),
    ],
    axis=1,
).astype(np.int32)

u_in = test_df["u_in"].astype(np.float32).values
u_out = test_df["u_out"].astype(np.float32).values
time_step = test_df["time_step"].astype(np.float32).values

test_df["_u_in_cumsum"] = (
    test_df.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)
test_df["_u_in_lag1"] = (
    test_df.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0)
    .astype(np.float32)
)
test_df["_u_out_lag1"] = (
    test_df.groupby("breath_id", sort=False)["u_out"]
    .shift(1)
    .fillna(0)
    .astype(np.float32)
)
test_df["_u_in_diff1"] = (test_df["u_in"] - test_df["_u_in_lag1"]).astype(np.float32)

feature_cols = [
    "time_step",
    "u_in",
    "u_out",
    "_u_in_cumsum",
    "_u_in_lag1",
    "_u_out_lag1",
    "_u_in_diff1",
]
other_flat = test_df[feature_cols].astype(np.float32).values

num_breaths = breath_meta.shape[0]
other_x_input = other_flat.reshape(num_breaths, 80, other_flat.shape[-1]).astype(
    np.float32
)

print("rc_input shape:", rc_input.shape, rc_input.dtype)
print("other_x_input shape:", other_x_input.shape, other_x_input.dtype)




## === cell 3
class ExpandTileLayer(layers.Layer):
    def __init__(self):
        super(ExpandTileLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return backend.tile(backend.expand_dims(inputs, axis=-2), (1, 80, 1))




## === cell 4
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("Using TPU strategy")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (CPU/GPU). TPU not available:", repr(e))

with strategy.scope():
    rc_input_layer = keras.Input(shape=(15,), dtype="int32", name="rc_input_layer")
    other_x_input_layer = keras.Input(
        shape=(80, other_x_input.shape[-1]), dtype="float32", name="other_x_input"
    )

    rc_embedding_layer = layers.Dense(
        units=10, use_bias=False, activation=activations.selu, name="rc_embed_layer"
    )
    rc_embedding = rc_embedding_layer(rc_input_layer)
    rc_embedding = ExpandTileLayer()(rc_embedding)

    x_input = layers.Concatenate(axis=-1)([other_x_input_layer, rc_embedding])

    conv1d_1 = layers.Conv1D(filters=256, kernel_size=5, padding="same")(x_input)
    conv1d_1 = layers.Activation(activations.relu)(conv1d_1)
    conv1d_1 = layers.BatchNormalization()(conv1d_1)

    conv1d_2 = layers.Conv1D(filters=192, kernel_size=5, padding="same")(conv1d_1)
    conv1d_output = layers.Activation(activations.relu)(conv1d_2)
    conv1d_output = layers.BatchNormalization()(conv1d_output)

    the_feature = layers.Concatenate()([conv1d_output, x_input])

    ox_1 = layers.Bidirectional(
        layers.LSTM(
            units=672,
            return_sequences=True,
            kernel_initializer=initializers.GlorotUniform(),
        ),
        merge_mode="concat",
    )(the_feature)

    ox_2 = layers.Bidirectional(
        layers.LSTM(
            units=512,
            return_sequences=True,
            kernel_initializer=initializers.GlorotUniform(),
        ),
        merge_mode="concat",
    )(ox_1)

    ox = layers.Bidirectional(
        layers.LSTM(
            units=384,
            return_sequences=True,
            kernel_initializer=initializers.GlorotUniform(),
        ),
        merge_mode="concat",
    )(ox_2)

    ox = layers.Bidirectional(layers.GRU(units=192, return_sequences=True))(ox)

    output = layers.Concatenate(axis=-1)([ox, conv1d_output])
    output = layers.Dense(
        units=128,
        activation=activations.selu,
        kernel_initializer=initializers.GlorotUniform(),
    )(output)
    output = layers.Dense(units=1, kernel_initializer=initializers.GlorotUniform())(
        output
    )

    my_model = models.Model(
        inputs=[rc_input_layer, other_x_input_layer], outputs=[output]
    )

print(my_model.summary())



## === cell 5
weights_path = os.path.join(INPUT_DIR, "model-weights", "model_final_6.h5")
if not os.path.exists(weights_path):
    weights_path_alt = os.path.join(
        INPUT_DIR, "ventilator-pressure-prediction", "model-weights", "model_final_6.h5"
    )
    if os.path.exists(weights_path_alt):
        weights_path = weights_path_alt

print("Looking for weights at:", weights_path)
if os.path.exists(weights_path):
    my_model.load_weights(weights_path)
    print("Loaded weights.")
else:
    print(
        "WARNING: Weights file not found; proceeding with randomly initialized model."
    )



## === cell 6
pre_pressure = []
pre_y_ = my_model.predict(x=[rc_input, other_x_input], batch_size=4096, verbose=1)
pre_y = np.asarray(pre_y_).reshape(-1)  # (num_breaths*80,)
pre_pressure.append(pre_y)

print("Pred vector shape:", pre_y.shape)



## === cell 7
pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
p_max = 64.82099173863328  # unused but kept

sub_medclip = np.median(np.vstack(pre_pressure), axis=0)
sub_medclip = np.round((sub_medclip - p_min) / pressure_step) * pressure_step + p_min

ids = test_df["id"].values.astype(np.int32)
assert (
    ids.shape[0] == sub_medclip.shape[0]
), "Mismatch between predicted length and test ids."

submission = pd.DataFrame({"id": ids, "pressure": sub_medclip.astype(np.float32)})
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print(submission.tail())
print("submission rows:", len(submission))
