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
plotly==5.24.1
plotly-express==0.4.1
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

0.47792

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

import warnings

warnings.filterwarnings("ignore")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
seed = 42



## === cell 2
DEVICE = "TPU"
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except Exception as e:
            print("failed to initialize TPU:", e)
            strategy = tf.distribute.get_strategy()
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
print("Number of devices: {}".format(strategy.num_replicas_in_sync))



## === cell 3
dropout_model = 0.36
hidden_dim_first = 128
hidden_dim_second = 256
hidden_dim_third = 128



## === cell 4
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")
train = train[train.signal_to_noise > 1]



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 6
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}




## === cell 7
def preprocess_inputs(df, cols=["sequence", "predicted_loop_type", "structure"]):
    """
    Convert string columns to integer token arrays.
    Returns shape (n_samples, seq_len, n_features). Handles empty dataframes safely.
    """
    if df.empty:
        return np.empty((0, 0, len(cols)), dtype=np.int32)

    seq_len = len(df.iloc[0][cols[0]])

    tokenized = df[cols].applymap(lambda seq: [token2int.get(x, 0) for x in seq])

    arr = np.array(tokenized.values.tolist())

    base_features = np.transpose(arr, (0, 2, 1)).astype(np.int32)
    return base_features


train_inputs_all = preprocess_inputs(train)
train_labels_all = np.array(train[target_cols].values.tolist()).transpose((0, 2, 1))




## === cell 8
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


def build_model(
    model_type=1,
    seq_len=107,
    pred_len=68,
    embed_dim=32,
    dropout=dropout_model,
    hidden_dim_first=hidden_dim_first,
    hidden_dim_second=hidden_dim_second,
    hidden_dim_third=hidden_dim_third,
):

    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )
    reshaped = tf.reshape(embed, shape=(-1, seq_len, embed_dim * 3))

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    hidden = EncoderLayer(96, 8, 256)(reshaped)
    hidden = EncoderLayer(96, 8, 256)(hidden)

    truncated = hidden[:, :pred_len]

    out = tf.keras.layers.Dense(len(target_cols), activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=adam, loss=MCRMSE, metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    return model




## === cell 9
tf.keras.backend.clear_session()
from tqdm.keras import TqdmCallback

lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", restore_best_weights=True, min_delta=0.001, patience=10
)




## === cell 10
def train_and_predict(
    n_folds=5,
    model_name="model",
    model_type=0,
    epochs=100,
    debug=True,
    dropout_model=dropout_model,
    hidden_dim_first=hidden_dim_first,
    hidden_dim_second=hidden_dim_second,
    hidden_dim_third=hidden_dim_third,
    seed=seed,
):

    print("Model:", model_name)

    ensemble_preds = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    val_losses = []
    historys = []

    for i, (train_idx, val_idx) in enumerate(
        skf.split(train_inputs_all, train["SN_filter"])
    ):
        print("Fold:", i + 1)
        with strategy.scope():
            model_train = build_model(
                model_type=model_type,
                dropout=dropout_model,
                hidden_dim_first=hidden_dim_first,
                hidden_dim_second=hidden_dim_second,
                hidden_dim_third=hidden_dim_third,
            )
            model_short = build_model(
                model_type=model_type,
                seq_len=107,
                pred_len=107,
                dropout=dropout_model,
                hidden_dim_first=hidden_dim_first,
                hidden_dim_second=hidden_dim_second,
                hidden_dim_third=hidden_dim_third,
            )
            model_long = build_model(
                model_type=model_type,
                seq_len=130,
                pred_len=130,
                dropout=dropout_model,
                hidden_dim_first=hidden_dim_first,
                hidden_dim_second=hidden_dim_second,
                hidden_dim_third=hidden_dim_third,
            )

        train_inputs, train_labels = (
            train_inputs_all[train_idx],
            train_labels_all[train_idx],
        )
        val_inputs, val_labels = train_inputs_all[val_idx], train_labels_all[val_idx]

        checkpoint_path = f"{model_name}_Fold_{i+1}.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True
        )

        history = model_train.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            batch_size=64,
            epochs=epochs,
            callbacks=[
                checkpoint,
                lr_callback,
                TqdmCallback(),
                tf.keras.callbacks.TerminateOnNaN(),
                es_callback,
            ],
            verbose=0,
        )

        print(
            f"{model_name} Min training loss={min(history.history['loss'])}, "
            f"min validation loss={min(history.history['val_loss'])}"
        )

        val_losses.append(min(history.history["val_loss"]))
        historys.append(history)

        model_short.load_weights(checkpoint_path)
        model_long.load_weights(checkpoint_path)

        public_preds = model_short.predict(public_inputs, verbose=0)
        private_preds = model_long.predict(private_inputs, verbose=0)

        preds_model = []
        for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
            if preds.shape[0] == 0:
                continue
            for uid, single_pred in zip(df.id, preds):
                single_df = pd.DataFrame(single_pred, columns=target_cols)
                single_df["id_seqpos"] = [
                    f"{uid}_{x}" for x in range(single_df.shape[0])
                ]
                preds_model.append(single_df)

        preds_model_df = pd.concat(preds_model, ignore_index=True)
        ensemble_preds[target_cols] += preds_model_df[target_cols].values / n_folds

        if debug:
            print("Intermediate ensemble result (first rows):")
            print(ensemble_preds[target_cols].head())

    ensemble_preds["id_seqpos"] = preds_model_df["id_seqpos"].values
    ensemble_preds = pd.merge(
        sample_sub[["id_seqpos"]], ensemble_preds, on="id_seqpos", how="left"
    )

    print("Mean Validation loss:", np.mean(val_losses))

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        for i, h in enumerate(historys):
            ax[0].plot(h.history["loss"], label=f"Fold{i+1}")
            ax[0].plot(h.history["val_loss"])
            ax[0].set_title("Loss")
            ax[1].plot(h.history["root_mean_squared_error"])
            ax[1].plot(h.history["val_root_mean_squared_error"])
            ax[1].set_title("RMSE")
            ax[2].plot(h.history.get("lr", []))
            ax[2].set_title("Learning Rate")
        plt.show()

    return ensemble_preds


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

ensembles = []
for i in range(1):
    model_name = f"model_{i+1}"
    ensemble = train_and_predict(
        n_folds=5,
        model_name=model_name,
        model_type=i,
        epochs=40,
        dropout_model=dropout_model,
        hidden_dim_first=hidden_dim_first,
        hidden_dim_second=hidden_dim_second,
        hidden_dim_third=hidden_dim_third,
        seed=seed,
    )
    ensembles.append(ensemble)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/2844024514.py in <cell line: 0>()
    144 for i in range(1):
    145     model_name = f"model_{i+1}"
--> 146     ensemble = train_and_predict(
    147         n_folds=5,
    148         model_name=model_name,

/tmp/ipykernel_54/2844024514.py in train_and_predict(n_folds, model_name, model_type, epochs, debug, dropout_model, hidden_dim_first, hidden_dim_second, hidden_dim_third, seed)
     24         print("Fold:", i + 1)
     25         with strategy.scope():
---> 26             model_train = build_model(
     27                 model_type=model_type,
     28                 dropout=dropout_model,

/tmp/ipykernel_54/2440158656.py in build_model(model_type, seq_len, pred_len, embed_dim, dropout, hidden_dim_first, hidden_dim_second, hidden_dim_third)
     21         inputs
     22     )
---> 23     reshaped = tf.reshape(embed, shape=(-1, seq_len, embed_dim * 3))
     24 
     25     reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 11
ensemble_final = ensembles[0].copy()
ensemble_final[target_cols] = 0.0
for ens in ensembles:
    ensemble_final[target_cols] += ens[target_cols].values / len(ensembles)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_54/1028792012.py in <cell line: 0>()
      1 # Simple averaging of the ensemble predictions (only one model in this case)
----> 2 ensemble_final = ensembles[0].copy()
      3 ensemble_final[target_cols] = 0.0
      4 for ens in ensembles:
      5     ensemble_final[target_cols] += ens[target_cols].values / len(ensembles)

IndexError: list index out of range

## === cell 12
submission = pd.DataFrame()
submission["id_seqpos"] = ensemble_final["id_seqpos"]
for col in target_cols:
    submission[col] = ensemble_final[col]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/287402649.py in <cell line: 0>()
      1 # Build the final submission dataframe
      2 submission = pd.DataFrame()
----> 3 submission["id_seqpos"] = ensemble_final["id_seqpos"]
      4 for col in target_cols:
      5     submission[col] = ensemble_final[col]

NameError: name 'ensemble_final' is not defined
