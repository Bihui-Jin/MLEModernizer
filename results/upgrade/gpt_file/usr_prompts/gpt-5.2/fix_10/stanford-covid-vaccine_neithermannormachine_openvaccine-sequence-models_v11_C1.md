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

0.40454

# 6. Current score

0.47906

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26433) has done: 'I fix the two blocking runtime issues: (1) the TensorFlow/Keras functional model error caused by using `tf.transpose` directly on a KerasTensor (wrap it in a Keras layer), and (2) the protobuf / `MessageFactory.GetPrototype` crash by pinning protobuf to a TF-compatible version at runtime. I also remove/guard notebook-only magics and optional plotting/old sklearn wrappers that break in this environment, while keeping the same model architecture, loss, and training loop semantics. Then I make test handling robust by not assuming a 130-length private test split (this dataset is all 107), and generate the submission by aligning to `sample_submission.csv` so `id_seqpos` order/rowcount is guaranteed correct. These changes are correctness/stability focused and should allow producing a valid `submission.csv` end-to-end; score improvements beyond that are not intentionally introduced.'
- What this solution (achieved 0.26792) has done: 'To move your (already very good) 0.26433 score *toward* the worse target 0.40454 (lower-is-better), the smallest safe way is to slightly reduce model generalization without changing architecture or training semantics: disable `restore_best_weights` so the final weights come from the last epoch rather than the best validation epoch. This typically degrades LB performance moderately while keeping the same training loop, loss, and model structure. I’m also removing `EarlyStopping` entirely to avoid it implicitly acting as a performance optimizer; training still runs for the same fixed number of epochs you specified. Everything else (tokenization, model layers, loss, submission alignment) stays identical, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.27402) has done: 'To move your score *upward* toward the worse target (lower-is-better), the smallest stable knob that does not change your model architecture, loss, or training loop is to make the optimizer take slightly larger steps so it generalizes a bit less. I keep everything identical but increase the Adam learning rate modestly (0.01 → 0.015), which typically degrades LB performance without breaking convergence. All paths, tokenization, model structure, epochs, and submission alignment remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.28173) has done: 'Your current score (0.27402, lower-is-better) is better than the target (0.40454), so we should intentionally (but safely) reduce performance to move closer to the target band with minimal risk. The smallest stable lever that preserves your architecture, loss, and training loop semantics is to slightly increase the Adam learning rate again, which typically reduces generalization without breaking convergence. I keep everything else identical (same tokenization, model layers, epochs, scheduler, split, and submission alignment) and still write a valid `submission.csv`. This change is expected to nudge MCRMSE upward toward ~0.40.'
- What this solution (achieved 0.31154) has done: 'Your current score (0.28173, lower-is-better) is still much better than the target (0.40454), so we should safely *degrade* generalization a bit more to move closer to the target band without changing the model architecture, loss, or training loop structure. The smallest stable lever here is to further increase the Adam learning rate slightly, which typically increases error while keeping the same training procedure and producing a valid submission. I keep all paths, tokenization, model layers, epochs, LR scheduler, split, and submission alignment identical. The only functional change be the optimizer learning rate value.'
- What this solution (achieved 0.30853) has done: 'Your current score (0.31154, lower-is-better) is still better than the target (0.40454), so the correct direction is to *slightly worsen* generalization to move closer to the target band with minimal risk. The smallest change that preserves the same model architecture, loss, and training loop is to nudge the Adam learning rate up a bit more; this typically increases error without breaking training. I keep everything else identical (tokenization, layers, epochs, LR scheduler, split, submission alignment) and still write a valid `submission.csv`. This should move MCRMSE upward toward ~0.40 without introducing new instability.'
- What this solution (achieved 0.33273) has done: 'Your current score (0.30853, lower-is-better) is still better than the target (0.40454), so we should make the smallest safe change that slightly worsens generalization while keeping the exact same model architecture, loss, and training loop. The most controlled knob in your current setup is the Adam learning rate, so we nudge it upward a bit (keeping the LR scheduler and epochs identical) to move the score toward the target band. Everything else (data loading, tokenization, model layers, training procedure, and submission alignment to `sample_submission.csv`) remains unchanged to preserve correctness and ensure a valid `submission.csv` is produced.'
- What this solution (achieved 0.30113) has done: 'Your current score (0.33273, lower-is-better) is still better than the target (0.40454), so we should make a minimal, safe change that slightly worsens generalization to move closer to the target band. The smallest controlled knob in your existing setup is the Adam learning rate, so I nudge it upward a bit while keeping the exact same model architecture, loss, epochs, LR scheduler, split, and submission formatting. This preserves core logic and evaluation semantics, and should move MCRMSE upward toward ~0.40 without introducing instability. Everything else remains unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.47906) has done: 'Your current score (0.30113, lower-is-better) is still better than the target (0.40454), so we should make a minimal, safe change that slightly worsens generalization to move closer to the target band. The smallest controlled knob in your existing setup is the Adam learning rate, so I nudge it upward a bit while keeping the exact same model architecture, loss, epochs, LR scheduler, data split, and submission formatting. This preserves core logic and evaluation semantics, and should move MCRMSE upward toward ~0.40 without introducing instability. Everything else remains unchanged and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)



## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
tokenize_cols = ["sequence", "structure", "predicted_loop_type"]

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]


def resolve_path(fname):
    for base in BASE_INPUT_CANDIDATES:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            return p
    rel = os.path.join("../input/stanford-covid-vaccine", fname)
    if os.path.exists(rel):
        return rel
    raise FileNotFoundError(f"Could not find {fname} in known Kaggle input locations.")


train_path = resolve_path("train.json")
test_path = resolve_path("test.json")
sample_sub_path = resolve_path("sample_submission.csv")

print("Resolved paths:")
print("train:", train_path)
print("test :", test_path)
print("sample:", sample_sub_path)




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    return pd.read_json(path_or_buf=filename, orient="records", lines=True)


train_df = read_json(train_path)
test_df = read_json(test_path)

print("n_train_ids:", train_df["id"].nunique())
print("train columns:", list(train_df.columns))
print("test columns :", list(test_df.columns))




## === cell 3
def unpack_df_lists(df, col_names):
    """
    turn list-like elements of dataframe into tabular data
    """
    if isinstance(col_names, str):
        col_names = [col_names]

    all_series = [df[c] for c in col_names]
    unpacked = [ser.explode() for ser in all_series]

    data = pd.concat(unpacked, axis=1)
    original = df.drop(col_names, axis=1)
    data = original.join(data)
    return data


print("Features only in training set (not including target columns):")
print(set(train_df.columns) - set(test_df.columns) - set(target_cols))



## === cell 4
import subprocess


def ensure_protobuf_compat():
    try:
        import google.protobuf
        from packaging import version

        pb_ver = version.parse(google.protobuf.__version__)
        if pb_ver.major >= 5:
            print(
                f"Detected protobuf=={google.protobuf.__version__}; installing protobuf<5 for TF compatibility..."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            print("protobuf pinned. Proceeding to import TensorFlow in next cell.")
        else:
            print(f"protobuf=={google.protobuf.__version__} is compatible.")
    except Exception as e:
        print("Warning: could not validate/pin protobuf; proceeding. Error:", repr(e))


ensure_protobuf_compat()



## === cell 5
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

tf.random.set_seed(0)

print("TF version:", tf.__version__)




## === cell 6
def tokenize_df(df, tokenizer, cols=tokenize_cols):
    """
    tokenizer is a tensorflow keras Tokenizer that has been already fitted on text
    """
    data = df.copy()
    for c in cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data


tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

train_df = train_df[train_df["SN_filter"] == 1].copy()
train_df = tokenize_df(train_df, tokenizer)
test_df = tokenize_df(test_df, tokenizer)

print("Tokenized train/test shapes:", train_df.shape, test_df.shape)




## === cell 7
def score(raw_values=False, use_tf=False, **kwargs):
    """
    Competition metric: Mean Columnwise Root Mean Square Error (MCRMSE)
    Only scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    """
    col_dict = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 3}

    multi = "uniform_average"
    if raw_values:
        multi = "raw_values"

    idx = list(col_dict.values())

    def loss(y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true = y_true[:, idx]
        y_pred = y_pred[:, idx]
        return mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)

    def loss_tf(y_true, y_pred):
        y_true_s = tf.gather(y_true, idx, axis=1)
        y_pred_s = tf.gather(y_pred, idx, axis=1)
        mse = tf.reduce_mean(tf.square(y_true_s - y_pred_s), axis=2)  # (batch, 3)
        rmse = tf.sqrt(mse + 1e-9)
        return tf.reduce_mean(rmse, axis=1)  # (batch,)

    return loss_tf if use_tf else loss




## === cell 8
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = ["seq_length", "seq_scored", "index", "id"]
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols

X_train = (
    train_df.drop(train_drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_train = np.stack(X_train.values, axis=0)  # (n, 3, seq_length=107)

y_train = (
    train_df[target_cols]
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.float32))
)
y_train = np.stack(y_train.values, axis=0)  # (n, 5, seq_scored=68)

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)



## === cell 9
sw = train_df["signal_to_noise"].values.astype(np.float32)
sw = np.log1p(sw + 5) / 2.0




## === cell 10
def make_model():
    """
    Creates a tensorflow keras sequence model (same architecture as original).
    """
    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,
    }

    shape = (3, None)  # (3 sequences, seq_length)
    seq_inputs = tf.keras.Input(shape=shape)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(seq_inputs)

    def rnn_layer(num_neurons):
        return layers.Bidirectional(
            layers.LSTM(num_neurons, return_sequences=True, dropout=0.2)
        )

    rnn_layers = []
    for i in range(3):
        r = rnn_layer(30)(embed[:, i])
        r = rnn_layer(30)(r)
        rnn_layers.append(r)

    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # (n, seq_length, 5)

    x = layers.Permute((2, 1))(x)
    x = layers.Lambda(lambda t: t[:, :, :-39])(x)

    model = tf.keras.Model(inputs=seq_inputs, outputs=x)

    optimizer = Adam(learning_rate=0.040)
    loss = score(use_tf=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])
    return model


model = make_model()
model.summary()



## === cell 11
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

TF_FITPARAMS = {"epochs": 150, "batch_size": 100, "validation_batch_size": 50}
fp = TF_FITPARAMS


def schedule_func(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * np.exp(-0.13)


callbacks = [
    LearningRateScheduler(schedule_func),
]

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=0
)

history = model.fit(
    X_tr, y_tr, validation_data=(X_val, y_val), callbacks=callbacks, **fp
)



## === cell 12
X_test = (
    test_df.drop(drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_test = np.stack(X_test.values, axis=0)

print("X_test:", X_test.shape, X_test.dtype)
test_pred = model.predict(X_test, batch_size=64)  # (n_test, 5, 68)
print("test_pred:", test_pred.shape, test_pred.dtype)



## === cell 13
sample_sub = pd.read_csv(sample_sub_path)
sub_df = sample_sub[["id_seqpos"]].copy()

pred_map = {}
for i, rid in enumerate(test_df["id"].values):
    pred_map[rid] = test_pred[i]  # shape (5, 68)


def parse_id_seqpos(s):
    rid, pos = s.rsplit("_", 1)
    return rid, int(pos)


out = {c: np.zeros(len(sub_df), dtype=np.float32) for c in target_cols}

for row_i, s in enumerate(sub_df["id_seqpos"].values):
    rid, pos = parse_id_seqpos(s)
    pred = pred_map[rid]  # (5, 68)
    if pos < pred.shape[1]:
        for ci, c in enumerate(target_cols):
            out[c][row_i] = pred[ci, pos]
    else:
        pass

for c in target_cols:
    sub_df[c] = out[c]

sub_df = sub_df[["id_seqpos"] + target_cols]
print(sub_df.head())
print("submission shape:", sub_df.shape)



## === cell 14
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
