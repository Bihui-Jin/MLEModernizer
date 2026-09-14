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

0.42264

# 6. Current score

0.30893

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.30001) has done: 'I fix the environment crash by removing the `plotly` import (it triggers a protobuf incompatibility here) since it’s only used for optional plotting. I fix the model build error by replacing the raw `tf.reshape` on a KerasTensor with a Keras `Reshape` layer, keeping the same tensor shape/logic. I also fix preprocessing to return an empty, correctly-shaped array for empty dataframes (your `private_df` is empty because this competition’s test set is all length 107), and simplify inference to predict on the full test set and generate a submission aligned exactly to `sample_submission.csv`. Finally, I ensure weights saving/loading uses a TF2.18-compatible filename (`.weights.h5`) so training → inference runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.31456) has done: 'I fix the protobuf-related crash by setting the pure-Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` error in this environment). Then I ensure the checkpoint callback doesn’t monitor a missing validation metric by explicitly setting `mode="min"` and keeping the validation split, which prevents silent issues during training/weight saving. Finally, I keep the model/training logic unchanged and only make the pipeline robust so it always trains, loads weights, predicts, and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.29637) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf / TensorFlow combo by forcing protobuf 3 “python” and “cpp” implementations in a safe order before importing TensorFlow, with a fallback that pins `protobuf<4` behavior where possible. I also make the preprocessing compatible with pandas 2.2 by avoiding deprecated `applymap` on DataFrames and ensuring consistent int32 arrays. These changes are execution/stability-focused and keep your model architecture/training/inference logic the same so the score should remain in the same neighborhood (and at least a valid `submission.csv` be produced). Finally, I add a small environment/versions print to help confirm the fix in Kaggle logs without affecting results.'
- What this solution (achieved 0.30819) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation to pure-Python *before any TensorFlow import* and removing the try/except fallback that can’t recover once the wrong protobuf objects are loaded. I keep your model, preprocessing, training, and submission-building logic the same, only making the import sequence deterministic and compatible with this Kaggle environment. This should restore end-to-end execution and produce a valid `submission.csv` without changing the modeling semantics (so score should stay in the same neighborhood and remain better than the target). I also add a small safety check to ensure the output columns/order match `sample_submission.csv`.'
- What this solution (achieved 0.29775) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf to the pure-Python implementation *and* pre-importing `google.protobuf` before importing TensorFlow, which prevents the `MessageFactory.GetPrototype` AttributeError in this environment. I also add a safe fallback to set `TF_USE_LEGACY_KERAS=1` (only if needed) to avoid TF/Keras/protobuf edge incompatibilities without changing your model or training logic. The rest of the pipeline (preprocessing, model architecture, training loop, inference, and submission alignment to `sample_submission.csv`) remain the same so your score should stay in the same neighborhood (already better than the target) while becoming stable end-to-end. Finally, I keep the output filename as `submission.csv` and enforce the exact column order/shape.'
- What this solution (achieved 0.30357) has done: 'I fix the protobuf/TensorFlow import crash by switching to TF2.18’s default Keras (disable legacy Keras) and forcing the pure-Python protobuf implementation before importing anything from TensorFlow, without changing your model/training logic. I also remove the explicit `import google.protobuf` that can lock in an incompatible protobuf state and trigger `MessageFactory.GetPrototype`. These changes are execution/stability-focused and should keep your score in the same neighborhood (already better than the target, since lower is better). The rest of the pipeline (preprocessing, model, training loop, inference, and submission formatting aligned to `sample_submission.csv`) remains unchanged.'
- What this solution (achieved 0.29685) has done: 'I fix the immediate TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation and pre-importing protobuf internals *before* importing TensorFlow (this avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle image). I keep your model architecture, preprocessing, training loop, and submission-building logic unchanged, only adjusting the import/ENV order to make execution stable. I also add a small guard so the submission columns/order exactly match the sample, ensuring Kaggle accepts the file. These changes should be score-neutral (your current score is already better than the target band for a lower-is-better metric) and primarily restore end-to-end execution.'
- What this solution (achieved 0.29897) has done: 'I fix the crash in the TensorFlow import caused by an incompatible pre-import of protobuf internals by removing the `google.protobuf` / `message_factory` pre-import and only forcing the pure-Python protobuf implementation before importing TensorFlow. This is an execution/stability fix and should be score-neutral (your current score is already better than the target for a lower-is-better metric). I keep the model architecture, training loop, preprocessing, and submission-building logic the same, only adjusting the import order so the notebook runs end-to-end. I also keep the submission column order aligned to `sample_submission.csv` and ensure `submission.csv` is always written.'
- What this solution (achieved 0.30893) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf implementation to pure-Python and pre-importing `google.protobuf` *in a safe, minimal way* before importing TensorFlow, which is a known workaround for this Kaggle image. I also remove `TF_USE_LEGACY_KERAS=0` (leave it at default) because forcing the new Keras path can exacerbate protobuf/TF incompatibilities here. The rest of your pipeline (preprocessing, model architecture, training loop, inference, and submission building aligned to `sample_submission.csv`) remain unchanged to keep score behavior stable (already better than the target for a lower-is-better metric). Finally, I keep writing `submission.csv` with the exact required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import sys
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

import google.protobuf  # noqa: F401




## === cell 1
import tensorflow as tf
import tensorflow.keras.layers as L

tf.random.set_seed(SEED)

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("tensorflow:", tf.__version__)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    if df is None or len(df) == 0:
        return np.zeros((0, 107, 3), dtype=np.int32)

    n = len(df)
    seq_len = int(df["seq_length"].iloc[0]) if "seq_length" in df.columns else 107

    x = np.zeros((n, seq_len, 3), dtype=np.int32)
    for i, (_, row) in enumerate(df[cols].iterrows()):
        for k, col in enumerate(cols):
            s = row[col]
            x[i, :, k] = [token2int[ch] for ch in s]
    return x




## === cell 4
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))


def build_model(seq_len=107, pred_len=68, dropout=0.4, embed_dim=100, hidden_dim=128):
    inputs = L.Input(shape=(seq_len, 3), dtype="int32")

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = L.Reshape((seq_len, 3 * embed_dim))(embed)

    hidden = gru_layer(hidden_dim, dropout)(reshaped)
    hidden = L.Conv1D(
        filters=256, kernel_size=32, strides=1, activation=None, padding="same"
    )(hidden)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 5
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print("train:", train.shape, "test:", test.shape, "sample:", sample_df.shape)




## === cell 6
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist(), dtype=np.float32).transpose(
    (0, 2, 1)
)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## === cell 7
model = build_model(seq_len=107, pred_len=68)
model.summary()




## === cell 8
ckpt_path = "model.weights.h5"

history = model.fit(
    train_inputs,
    train_labels,
    batch_size=64,
    epochs=125,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ],
    validation_split=0.05,
    verbose=2,
)




## === cell 9
print(
    "Final loss:",
    history.history["loss"][-1],
    "Final val_loss:",
    history.history["val_loss"][-1],
)




## === cell 10
test_inputs = preprocess_inputs(test)
print("test_inputs:", test_inputs.shape, test_inputs.dtype)

infer_model = build_model(seq_len=107, pred_len=107)
if os.path.exists(ckpt_path):
    infer_model.load_weights(ckpt_path)
else:
    infer_model.set_weights(model.get_weights())

test_preds = infer_model.predict(test_inputs, batch_size=64, verbose=1)
print("test_preds:", test_preds.shape, test_preds.dtype)  # (n_test, 107, 5)




## === cell 11
preds_map = {c: [] for c in pred_cols}
id_seqpos_all = []

for i, uid in enumerate(test["id"].values):
    single_pred = test_preds[i]  # (107, 5)
    for pos in range(single_pred.shape[0]):
        id_seqpos_all.append(f"{uid}_{pos}")
        for j, c in enumerate(pred_cols):
            preds_map[c].append(float(single_pred[pos, j]))

preds_df = pd.DataFrame({"id_seqpos": id_seqpos_all, **preds_map})

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    if submission[c].isna().any():
        submission[c] = submission[c].fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]

print("submission shape:", submission.shape)
print(submission.head())




## === cell 12
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Columns:", list(submission.columns))
print("Rows:", len(submission))
