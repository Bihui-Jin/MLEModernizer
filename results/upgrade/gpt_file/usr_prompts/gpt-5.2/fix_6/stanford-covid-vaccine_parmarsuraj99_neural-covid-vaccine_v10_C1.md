# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.42119

# 6. Current score

0.29963

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28057) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by pinning `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow. Then I remove the broken “public/private 107 vs 130” branching (this dataset is 107 only), which is what triggers the `axes don't match array` error and cascades into missing variables. Finally, I make the feature column selection deterministic and correct (explicit `*_n` columns instead of the fragile `"e_n"` substring), and ensure predictions are generated for all test ids and merged onto `sample_submission.csv` to write a valid `submission.csv`.'
- What this solution (achieved 0.29656) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend earlier and more completely (including the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` and clearing already-imported protobuf modules before importing TF). Then I make the label encoding robust by fitting on both train and test vocabularies to avoid “unknown label” errors at inference while keeping the same feature/architecture/training loop. Finally, I keep the submission generation logic the same but add a safety check that every `id_seqpos` is produced and joined correctly before writing `submission.csv`. These changes are score-neutral or slightly stabilizing; they should keep you near your current (already better-than-target) score while restoring end-to-end execution.'
- What this solution (achieved 0.30026) has done: 'The crash happens before training because TensorFlow 2.18 is incompatible with the very new protobuf runtime in this environment (protobuf 6.x), causing the `MessageFactory.GetPrototype` attribute error at import time. The minimal, robust fix is to force TensorFlow to use the pure-Python protobuf backend and (when needed) downgrade protobuf to a TF-compatible version inside the notebook before importing TensorFlow; this is execution-only and should be score-neutral. I also add a small “fast-fail” check to ensure TensorFlow really imported after the fix, and keep the rest of your pipeline (encoding, model, training, prediction, submission merge) unchanged to preserve score behavior (already better than target). The script still write a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.30085) has done: 'I fix the TensorFlow/protobuf import crash by making the protobuf “pure-Python backend” enforcement happen before any TensorFlow-related import and by automatically retrying with a TF-compatible protobuf version if needed. This is execution-only and should be score-neutral (it doesn’t change your model, features, training loop, or inference). I also add a small safety check to ensure the environment variables are truly applied early, and keep all data loading, encoding, model definition, training, prediction, and submission merge logic unchanged so your score behavior stays aligned with the current (already better-than-target) performance. Finally, I keep writing `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.29963) has done: 'The run currently fails immediately when importing TensorFlow due to an incompatibility between TF 2.18 and the installed protobuf 6.x runtime. I make the TensorFlow import fix more robust by (1) forcing the pure-Python protobuf backend and (2) proactively installing a TF-compatible protobuf version before importing TensorFlow (and reloading modules), which resolves the `MessageFactory.GetPrototype` crash reliably in Kaggle. These changes are execution-only and do not alter your model, features, training loop, or inference, so they should keep the score behavior essentially unchanged (already better than the 0.42119 target). I also keep the submission writing logic the same and add one small guard to ensure the written file has the exact required column order and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess
import gc
import numpy as np
import pandas as pd


def _import_tf_with_protobuf_fix():
    """
    TF 2.18 can crash with protobuf 6.x in some Kaggle images:
    AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

    Minimal robust fix:
    - force pure-python protobuf backend via env vars (above)
    - ensure a TF-compatible protobuf is installed (protobuf 4.25.3)
    - clear already-imported protobuf/tensorflow modules and import TF fresh
    """
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf==4.25.3",
            ]
        )
    except Exception:
        pass

    for m in list(sys.modules):
        if m.startswith("google.protobuf") or m.startswith("tensorflow"):
            del sys.modules[m]
    gc.collect()

    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        msg = repr(e)
        if (
            ("GetPrototype" in msg)
            or ("MessageFactory" in msg)
            or ("google.protobuf" in msg)
        ):
            for m in list(sys.modules):
                if m.startswith("google.protobuf") or m.startswith("tensorflow"):
                    del sys.modules[m]
            gc.collect()
            import tensorflow as tf  # noqa: F401

            return tf
        raise


tf = _import_tf_with_protobuf_fix()

from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)



## === cell 1
train_df = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test_df = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
train_df.head()



## === cell 3
sample_df.head()



## === cell 4
train_df.columns



## === cell 5
train_df["sequence"].str.split("").apply(lambda x: (np.unique(x), len(x))).head()



## === cell 6
np.unique(train_df["seq_length"].values)



## === cell 7
feature_columns = ["sequence", "structure", "predicted_loop_type"]
target_columns = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 8
train_df.head(3)



## === cell 9
label_encoders = dict()
for column in feature_columns:
    encoder = LabelEncoder()
    vocab = list(
        set(train_df[column].apply(list).sum() + test_df[column].apply(list).sum())
    )
    encoder.fit(vocab)
    label_encoders[column] = encoder
    del encoder
    gc.collect()




## === cell 10
def transform_(df: pd.DataFrame, label_encoders: dict):
    for column in feature_columns:
        df[column + "_n"] = df[column].apply(
            lambda seq: label_encoders[column].transform(list(seq))
        )
    return df




## === cell 11
train_df = transform_(train_df, label_encoders)
test_df = transform_(test_df, label_encoders)

feature_columns_n = [c + "_n" for c in feature_columns]
feature_columns_n

train_x = np.array(train_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))
train_y = np.array(train_df[target_columns].values.tolist()).transpose((0, 2, 1))

train_x.shape, train_y.shape




## === cell 12
def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    inputs = L.Input(shape=(seq_len, 3))

    inputs_as = L.Lambda(lambda x: tf.split(x, inputs.shape[-1], axis=-1))(inputs)

    embeddings = []
    for inp_a, col in zip(inputs_as, feature_columns):
        embedding = L.Embedding(
            input_dim=len(label_encoders[col].classes_), output_dim=embed_dim
        )(inp_a)
        embedding = L.Reshape((-1, int(embedding.shape[2]) * int(embedding.shape[3])))(
            embedding
        )
        embeddings.append(embedding)

    embed_cat = L.Concatenate()(embeddings)

    cat = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(
        embed_cat
    )

    cat = cat[:, :pred_len]

    cat = L.Dense(64, activation="relu")(cat)
    dense = L.Dense(5, activation="linear")(cat)

    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss="mse", optimizer="adam")
    return model




## === cell 13
model = build_model()
model.summary()



## === cell 14
try:
    tf.keras.utils.plot_model(model, show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 15
model.fit(
    train_x,
    train_y,
    batch_size=64,
    epochs=80,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],
    validation_split=0.01,
    verbose=2,
)



## === cell 16
test_x = np.array(test_df[feature_columns_n].values.tolist()).transpose((0, 2, 1))

preds = model.predict(test_x, verbose=1)
preds.shape



## === cell 17
preds_ls = []
for i, uid in enumerate(test_df.id.values):
    single_pred = preds[i]  # (pred_len=68, 5)
    single_df = pd.DataFrame(single_pred, columns=target_columns)

    seq_length = int(test_df.loc[test_df.index[i], "seq_length"])
    if single_df.shape[0] < seq_length:
        pad_n = seq_length - single_df.shape[0]
        pad_block = np.repeat(single_pred[-1:, :], repeats=pad_n, axis=0)
        pad_df = pd.DataFrame(pad_block, columns=target_columns)
        single_df = pd.concat([single_df, pad_df], axis=0, ignore_index=True)

    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)
preds_df.head(), preds_df.shape



## === cell 18
submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

missing = submission[target_columns].isna().any(axis=1).sum()
if missing:
    print(f"Warning: {missing} rows missing predictions after merge; filling with 0.0")

for c in target_columns:
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

submission = submission[["id_seqpos"] + target_columns]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 19
assert (
    submission.shape[0] == sample_df.shape[0]
), "Row count mismatch vs sample_submission"
assert (
    list(submission.columns) == ["id_seqpos"] + target_columns
), "Column order mismatch"
submission.describe(include="all")



## === cell 20
tmp = submission["id_seqpos"].str.rsplit("_", n=1, expand=True)
tmp.columns = ["id", "seqpos"]
counts = tmp["id"].value_counts().head()
print("Top id row-counts (should be 107):")
print(counts)
