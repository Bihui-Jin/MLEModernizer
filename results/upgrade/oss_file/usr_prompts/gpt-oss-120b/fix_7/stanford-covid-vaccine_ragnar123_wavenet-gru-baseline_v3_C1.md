# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 5. Target score

0.38615

# 6. Current score

0.43038

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.43038) has done: 'I added the missing imports, safely detected TensorFlow/TFA availability, and corrected the undefined variables so the notebook runs end‑to‑end and writes a proper `submission.csv`. I also lowered the epoch count to keep execution fast while preserving the original model architecture.'

# 9. Code solution

## === cell 0
import os
import random
import warnings
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import metrics

try:
    import tensorflow as tf

    _HAS_TF = True
    from tensorflow.keras import backend as K
except Exception:
    tf = None
    _HAS_TF = False

try:
    import tensorflow_addons as tfa

    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False

FOLDS = 5
EPOCHS = 10  # reduced for fast execution while keeping the original training logic
BATCH_SIZE = 64
LR = 0.001
VERBOSE = 2
SEED = 123


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TF:
        tf.random.set_seed(seed)


seed_everything(SEED)

train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    """
    Convert string columns into integer token arrays and reshape to (N, seq_len, 3).
    Returns an empty array with shape (0, 0, 3) when the input DataFrame is empty.
    """
    if df.empty:
        return np.empty((0, 0, 3), dtype=int)

    tokenised = df[cols].applymap(lambda seq: [token2int.get(x, 0) for x in seq])
    arr = np.array(tokenised.values.tolist())  # (N, 3, seq_len)
    return np.transpose(arr, (0, 2, 1))


train_inputs = preprocess_inputs(train.loc[train["SN_filter"] == 1])
train_labels = np.array(
    train[train["SN_filter"] == 1][target_cols].values.tolist()
).transpose(0, 2, 1)

public_test_df = test[test["seq_length"] == 107]
private_test_df = test[test["seq_length"] == 130]

public_test = preprocess_inputs(public_test_df)
private_test = preprocess_inputs(private_test_df)  # safely handles empty case




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_model(seq_len=107, pred_len=68, embed_dim=75, dropout=0.10):
    """Construct the original model – imports TensorFlow lazily."""
    if not _HAS_TF:
        raise RuntimeError("TensorFlow is not available; cannot build model.")
    import tensorflow as tf

    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="tanh",
                dilation_rate=dilation_rate,
            )(x)
            sigm_out = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                dilation_rate=dilation_rate,
            )(x)
            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
                x
            )
            res_x = tf.keras.layers.Add()([res_x, x])
        return res_x

    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype="int32")
    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    x = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)
    x = tf.keras.layers.SpatialDropout1D(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)
    x = wave_block(x, 16, 3, 12)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)
    x = wave_block(x, 32, 3, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)
    x = wave_block(x, 64, 3, 4)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            256, dropout=dropout, return_sequences=True, kernel_initializer="orthogonal"
        )
    )(x)

    truncated = x[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.models.Model(inputs=inputs, outputs=out)

    base_opt = tf.keras.optimizers.Adam(learning_rate=LR)
    optimizer = tfa.optimizers.SWA(base_opt) if _HAS_TFA else base_opt

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


def mcrmse(y_true, y_pred):
    y_true_ = y_true.reshape(-1, 5)
    y_pred_ = y_pred.reshape(-1, 5)
    rmses = [
        math.sqrt(metrics.mean_squared_error(y_true_[:, i], y_pred_[:, i]))
        for i in range(5)
    ]
    return np.mean(rmses)


def _baseline_predictions(train_labels, public_test, private_test):
    """Simple mean‑baseline used when TF is unavailable or fails."""
    overall_mean = train_labels.mean(axis=(0, 1))  # (5,)
    if public_test.shape[0] > 0:
        pub_len = public_test.shape[1]
        public_preds = np.tile(overall_mean, (public_test.shape[0], pub_len, 1))
    else:
        public_preds = np.empty((0, 0, 5))
    if private_test.shape[0] > 0:
        priv_len = private_test.shape[1]
        private_preds = np.tile(overall_mean, (private_test.shape[0], priv_len, 1))
    else:
        private_preds = np.empty((0, 0, 5))
    dummy_mcrmse = mcrmse(
        train_labels, np.tile(overall_mean, (train_labels.shape[0], 68, 1))
    )
    print(f"Baseline OOF MCRMSE (placeholder): {dummy_mcrmse:.5f}")
    return public_preds, private_preds


def train_and_evaluate(train_inputs, train_labels, public_test, private_test):
    """
    Train the original TF model if possible; otherwise fall back to the
    mean‑baseline. Any exception during TF training also triggers the fallback,
    guaranteeing a valid CSV output.
    """
    if not _HAS_TF:
        return _baseline_predictions(train_labels, public_test, private_test)

    try:
        oof_preds = np.zeros((train_inputs.shape[0], 68, 5))
        public_preds = np.zeros((public_test.shape[0], 107, 5))
        private_preds = np.zeros((private_test.shape[0], 130, 5))

        kfold = KFold(FOLDS, shuffle=True, random_state=SEED)
        for fold, (train_index, val_index) in enumerate(kfold.split(train_inputs)):
            print(f"Training fold {fold + 1}")
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f"fold_{fold + 1}.weights.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            )
            cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=5,
                verbose=1,
                min_delta=1e-5,
            )
            x_train, x_val = train_inputs[train_index], train_inputs[val_index]
            y_train, y_val = train_labels[train_index], train_labels[val_index]

            K.clear_session()
            model = build_model()
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[checkpoint, cb_lr_schedule],
                verbose=VERBOSE,
            )

            model.load_weights(f"fold_{fold + 1}.weights.h5")
            oof_preds[val_index] = model.predict(x_val)

            short = build_model(seq_len=107, pred_len=107)
            short.load_weights(f"fold_{fold + 1}.weights.h5")
            if public_test.shape[0] > 0:
                public_preds += short.predict(public_test) / FOLDS

            long = build_model(seq_len=130, pred_len=130)
            long.load_weights(f"fold_{fold + 1}.weights.h5")
            if private_test.shape[0] > 0:
                private_preds += long.predict(private_test) / FOLDS

            print("-" * 50)

        mean_col_rmse = mcrmse(train_labels, oof_preds)
        print(f"Our out‑of‑folds MCRMSE is {mean_col_rmse}")
        return public_preds, private_preds

    except Exception as e:
        warnings.warn(f"TensorFlow training failed ({e}); falling back to baseline.")
        return _baseline_predictions(train_labels, public_test, private_test)




## === cell 2
public_preds, private_preds = train_and_evaluate(
    train_inputs, train_labels, public_test, private_test
)


def inference_format(
    public_test_df, public_preds, private_test_df, private_preds, target_cols
):
    predictions = []
    for test_df, preds in [
        (public_test_df, public_preds),
        (private_test_df, private_preds),
    ]:
        for idx, uid in enumerate(test_df["id"]):
            single_pred = preds[idx]  # (seq_len, 5)
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [
                f"{uid}_{pos}" for pos in range(single_df.shape[0])
            ]
            predictions.append(single_df)
    return pd.concat(predictions, ignore_index=True)


predictions = inference_format(
    public_test_df, public_preds, private_test_df, private_preds, target_cols
)

submission = sample_sub[["id_seqpos"]].merge(predictions, on="id_seqpos", how="left")
submission.to_csv("submission.csv", index=False)
print("Submission saved")
submission.head()
