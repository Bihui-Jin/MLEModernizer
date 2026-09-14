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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.54902

# 6. Current score

0.29694

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.29703) has done: 'I remove the `plotly` import that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, since it isn’t required to train or create the submission. I fix the input tensor shape bug by transposing the output of `preprocess_inputs` so the model receives `(n, 107, 3)` instead of `(n, 3, 107)`, which unblocks training and produces the checkpoint file needed for inference. I also add small guards so downstream cells don’t crash if training fails, and ensure we always write a correctly ordered `submission.csv` matching `sample_submission.csv` (all required columns, correct row count). These changes are execution/stability fixes and should allow a valid submission to be generated end-to-end.'
- What this solution (achieved 0.29715) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by removing the unnecessary `plotly`/protobuf-import side effects and forcing TensorFlow to import cleanly (this is the runtime blocker). Then I make sure the training/inference path always runs end-to-end by adding small guards around checkpoint loading and by keeping the input tensor shape `(n, 107, 3)` consistent (to avoid silent shape mismatches). Finally, I keep the model/core training logic unchanged but ensure the submission is strictly aligned to `sample_submission.csv` ordering and has all required columns with the correct row count, so Kaggle accepts it and scores properly.'
- What this solution (achieved 0.29694) has done: 'I fix the protobuf/TensorFlow import crash by ensuring TensorFlow is imported before any package that can trigger protobuf incompatibilities, and by setting a safe protobuf implementation environment variable up-front. I also remove unused imports (like `MinMaxScaler` usage that never affects the model) to reduce surface area for runtime issues, while keeping the model/training/prediction logic unchanged. Finally, I keep the existing submission-building logic but add a strict sanity check that the produced file matches `sample_submission.csv` row order/shape so Kaggle accepts it reliably.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L

tf.keras.utils.set_random_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))


def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=32, hidden_dim=128):
    """
    Output:
      - (batch, pred_len, 5)
    """
    ids = L.Input(
        shape=(seq_len, 3), dtype=tf.int32
    )  # 3 categorical tracks per position

    seq_emb = L.Embedding(input_dim=4, output_dim=embed_dim, name="seq_emb")(
        ids[:, :, 0]
    )
    str_emb = L.Embedding(input_dim=3, output_dim=embed_dim, name="str_emb")(
        ids[:, :, 1]
    )
    loop_emb = L.Embedding(input_dim=7, output_dim=embed_dim, name="loop_emb")(
        ids[:, :, 2]
    )

    x = L.Concatenate(axis=-1)([seq_emb, str_emb, loop_emb])
    x = L.SpatialDropout1D(dropout)(x)

    x = L.AveragePooling1D(pool_size=1)(x)
    x = gru_layer(hidden_dim=hidden_dim, dropout=dropout)(x)

    truncated = x[:, :pred_len, :]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 3
vocab = {
    "sequence": {x: i for i, x in enumerate("A C G U".split())},
    "structure": {x: i for i, x in enumerate("( . )".split())},
    "predicted_loop_type": {x: i for i, x in enumerate("B E H I M S X".split())},
}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    def f(row):
        seq = row[cols[0]]
        struct = row[cols[1]]
        loop = row[cols[2]]
        return (
            [vocab["sequence"][ch] for ch in seq],
            [vocab["structure"][ch] for ch in struct],
            [vocab["predicted_loop_type"][ch] for ch in loop],
        )

    arr = df[cols].apply(f, axis=1).tolist()
    x = np.array(arr, dtype=np.int32)  # (n, 3, 107)
    x = np.transpose(x, (0, 2, 1))  # (n, 107, 3)  <-- required by model input
    return x




## === cell 4
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 5
print(pd.Series(list(train["structure"].iloc[0])).value_counts())
print(pd.Series(list(train["sequence"].iloc[0])).value_counts())
print(pd.Series(list(train["predicted_loop_type"].iloc[0])).value_counts())



## === cell 6
train_inputs = preprocess_inputs(train)

train_labels = (
    np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1)).astype(np.float32)
)



## === cell 7
train_inputs.shape, train_labels.shape



## === cell 8
for df in [train, test]:
    df["Paired"] = [
        sum((ch == "(") or (ch == ")") for ch in s) for s in df["structure"]
    ]
    df["Unpaired"] = [sum(ch == "." for ch in s) for s in df["structure"]]
    for col in ["E", "S", "H", "I", "G", "A", "U"]:
        if col in ["E", "S", "H", "I"]:
            df[col] = [
                sum(ch == col for ch in s) / len(s) for s in df["predicted_loop_type"]
            ]
        else:
            df[col] = [sum(ch == col for ch in s) / len(s) for s in df["sequence"]]


def safe_mean_pos(s, token):
    idx = [i for i, ch in enumerate(s) if ch == token]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(idx))


for a in ["G", "A", "C", "U"]:
    train[a + "_position"] = [safe_mean_pos(s, a) for s in train["sequence"]]
    test[a + "_position"] = [safe_mean_pos(s, a) for s in test["sequence"]]

for a in ["E", "S", "H"]:
    train[a + "_position"] = [safe_mean_pos(s, a) for s in train["predicted_loop_type"]]
    test[a + "_position"] = [safe_mean_pos(s, a) for s in test["predicted_loop_type"]]



## === cell 9
target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
target_columns.extend(["SN_filter", "signal_to_noise"])
target_columns.extend(
    [
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "reactivity_error",
        "deg_error_Mg_pH10",
    ]
)
train.drop(target_columns, axis=1, inplace=True)



## === cell 10
train_measurements = pd.concat(
    (train.select_dtypes("float64"), train.select_dtypes("int64")), axis=1
).to_numpy()



## === cell 11
train_measurements.shape



## === cell 12
np.min(train_measurements), np.max(train_measurements)



## === cell 13
pd.DataFrame(train_measurements).describe().T.head()



## === cell 14
model = build_model(seq_len=107, pred_len=68)
model.summary()



## === cell 15
train_inputs.shape, train_labels.shape



## === cell 16
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.ModelCheckpoint(
        "model.weights.h5", save_weights_only=True, save_best_only=True
    ),
]

history = model.fit(
    train_inputs,
    train_labels,
    batch_size=64,
    epochs=100,
    validation_split=0.05,
    callbacks=callbacks,
    verbose=2,
)



## === cell 17
pass



## === cell 18
pass



## === cell 19
test_measurements = pd.concat(
    (test.select_dtypes("float64"), test.select_dtypes("int64")), axis=1
).to_numpy()
test_inputs = preprocess_inputs(test)



## === cell 20
model_short = build_model(seq_len=107, pred_len=68)

if os.path.exists("model.weights.h5"):
    model_short.load_weights("model.weights.h5")
else:
    try:
        model_short.set_weights(model.get_weights())
    except Exception:
        pass

test_preds_68 = model_short.predict(test_inputs, batch_size=64, verbose=0)  # (n,68,5)

n = test_preds_68.shape[0]
pad_len = 107 - 68
test_preds = np.concatenate(
    [test_preds_68, np.zeros((n, pad_len, 5), dtype=np.float32)], axis=1
)  # (n,107,5)



## === cell 21
print("Pred shape:", test_preds.shape)



## === cell 22
preds_ls = []
for i, uid in enumerate(test["id"].values):
    single_pred = test_preds[i]  # (107,5)
    single_df = pd.DataFrame(single_pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)



## === cell 23
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission[pred_cols] = submission[pred_cols].fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]

assert submission.shape[0] == sample_df.shape[0], (submission.shape, sample_df.shape)
assert (submission["id_seqpos"].values == sample_df["id_seqpos"].values).all()

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission.shape)
print(submission.head())
