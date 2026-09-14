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

0.71003

# 6. Current score

0.29808

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27388) has done: 'Diagnosis: Cell 20 crashes while building the Functional Keras model because `make_model()` applies raw TensorFlow ops (`tf.transpose` and tensor slicing) directly to a `KerasTensor`. In TF/Keras 2.18, symbolic `KerasTensor` objects can only be transformed via Keras layers/ops, not arbitrary `tf.*` functions or Python slicing, so `tf.transpose(x, ...)` raises `ValueError: A KerasTensor cannot be used as input to a TensorFlow function`.

Patch summary: Update only cell 20 to monkey-patch a compatible `make_model()` that preserves the exact architecture and semantics, replacing `tf.transpose` with a `layers.Permute` layer and replacing `x[:, :, :68]` with a Keras `layers.Lambda` slice. Then proceed to build, summarize, and fit the model exactly as before, keeping the same variables (`model`, `history`) for downstream cells.

Updated cells: Only cell 20 is changed.

Compatibility notes for cell k+1: Cell 21 only imports sklearn utilities; it does not depend on internal model construction details. This patch keeps `model` and `history` defined exactly as expected after cell 20 completes.

Assumptions: `make_model` from cell 19 must remain unchanged; we only override it in cell 20 to unblock execution under KerasTensor rules without changing model logic/output shape.'
- What this solution (achieved 0.26917) has done: 'Diagnosis: Cell 29 fails at `sub_private['seqpos'] = sub_private.apply(..., axis=1)` because in pandas 2.2.x `DataFrame.apply(..., axis=1)` can return a DataFrame (multiple columns) instead of a 1D Series when the return value is list-like, which then cannot be assigned to a single column. Here we only need a per-row Python list of positions, so we should force a 1D result deterministically.  
Patch summary: In cell 29, replace the row-wise `apply` that builds `seqpos` with a `Series.map` over the `seq_length` column, which always produces a 1D Series of Python lists and is equivalent in semantics. No other logic (targets filled with zeros, unrolling, id_seqpos creation, column ordering) is changed.  
Updated cells: Only cell 29 is modified.  
Compatibility notes for cell k+1: `sub_private` remains a DataFrame with the same columns (`id_seqpos` + `target_cols`) and row count, so `pd.concat([sub_df, sub_private])` in cell 30 continues to work unchanged.  
Assumptions: `seq_length` is an integer-like column for all rows in `test_private` (as in the provided schema), and we still want `seqpos` to span `range(seq_length)` (unchanged behavior).'
- What this solution (achieved 0.27241) has done: 'Diagnosis: Cell 29 fails because `sub_private.apply(lambda row: [0] * row["seq_length"], axis=1)` is being interpreted by pandas as producing a multi-column DataFrame (instead of a 1D Series of list objects), so assigning it to a single column (e.g., `"reactivity"`) raises “Cannot set a DataFrame with multiple columns…”. This can happen with pandas 2.x when `apply` output is inferred/expanded unexpectedly. The intent is to create one list per row; we should force a scalar (object) output per row to prevent expansion.  

Patch summary: In cell 29 only, replace the per-column `apply` assignment with an explicit list-of-lists construction using the `seq_length` Series, ensuring each new target column is a 1D object-dtype Series (one list per row). This preserves identical semantics (all zeros, length `seq_length`) while making the assignment deterministic under pandas 2.2.3.  

Updated cells:  

Compatibility notes for cell k+1: Output `sub_private` keeps the same columns and format (`id_seqpos` + `target_cols`) and can still be concatenated with `sub_df` in cell 30 without changes.  

Assumptions: `seq_length` is numeric (or convertible to int) for all rows where `seq_length == 130`.'
- What this solution (achieved 0.27211) has done: 'The crash happens because `sub_private.apply(..., axis=1)` is returning a multi-column `DataFrame` instead of a 1D `Series`, so Pandas refuses to assign it into a single column (`id_seqpos`). This can occur when there are duplicate column names (often introduced by `explode()`/joins) or when the lambda returns a non-scalar. The minimal fix is to build `id_seqpos` deterministically via vectorized string operations on the `id` and `seqpos` columns, which always produces a 1D Series and avoids the DataFrame assignment path. No model/training logic is touched, and the output columns/order remain the same for downstream concatenation in cell 30. Assumption: after `unpack_df_lists`, `sub_private` still has accessible `id` and `seqpos` columns (as intended by the original code).'
- What this solution (achieved 0.29808) has done: 'We need to move your score downward (lower is better) from 0.27211 toward the target 0.71003, so we should deliberately reduce performance with minimal, safe changes while keeping the same model and training loop intact. The smallest legitimate lever that preserves core logic is to restrict training to the noisier/unfiltered samples by removing the `SN_filter == 1` filter; this typically worsens generalization and should increase MCRMSE toward the target band. I also add a deterministic seed to keep the (now intentionally worse) result stable run-to-run, and keep submission generation unchanged. No architecture/loss/loop changes are made.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt

"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""


## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    file = open(filename)
    df = pd.read_json(path_or_buf=file, orient="records", lines=True)
    return df




## === cell 3
train_df = read_json("../input/stanford-covid-vaccine/train.json")

print(train_df["id"].nunique())
print(train_df.columns)
train_df


## === cell 4
test_df = read_json("../input/stanford-covid-vaccine/test.json")


print("Features only in training set (not including target columns):")
set(train_df.columns) - set(test_df.columns) - set(target_cols)


## === cell 5
test_df


## === cell 6
"""! ls
#! ls draw_rna
! ls forna

! python forna/forna_server.py -s -d"""


## === cell 7
"""seq = train_df.loc[0, 'sequence']
struct = train_df.loc[0, 'structure']

seq, struct"""




## === cell 8
def unpack_df_lists(df, col_names):
    """
    turn list-like elements of dataframe into tabular data

    works great
    """
    if isinstance(
        col_names, str
    ):  # if string is passed in, convert to list for convenience
        col_names = [col_names]

    all_series = [df[c] for c in col_names]  # select relevant columns
    unpacked = [
        ser.explode() for ser in all_series
    ]  # unpack lists for each feature series

    data = pd.concat(unpacked, axis=1)  # concat unpacked columns together

    original = df.drop(col_names, axis=1)  # drop columns with list elements
    data = original.join(data)  # then join unpacked data to original df

    return data




## === cell 9
def feature_engineer(df, train=True, **kwargs):

    unpack_cols = [
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]  # only need to unpack things in training set

    if train:
        data = unpack_df_lists(df, unpack_cols)
    else:  # if test data, need to add rows manually
        data = df.copy()
        data["temp"] = data.apply(
            lambda row: [0] * row["seq_length"], axis=1
        )  # adds temp column with list-like elements, of len(seq_scored) for that row
        data = unpack_df_lists(
            data, "temp"
        )  # unpack to right length using this function
        del data["temp"]  # delete the temp column

    data["seqpos"] = 1
    data["seqpos"] = data.groupby("id").cumsum()["seqpos"] - 1

    seq_temp = pd.concat([data["sequence"], data["seqpos"]], axis=1)
    data["nucleotide"] = seq_temp.apply(
        lambda row: row["sequence"][row["seqpos"]], axis=1
    )  # get base at seqpos in sequence string

    loop_temp = pd.concat([data["predicted_loop_type"], data["seqpos"]], axis=1)
    data["pred_loop_seqpos"] = loop_temp.apply(
        lambda row: row["predicted_loop_type"][row["seqpos"]], axis=1
    )  # get type at seqpos in predicted_loop_type string

    data = pd.get_dummies(
        data, columns=["nucleotide", "pred_loop_seqpos"]
    )  # do one-hot encoding on predicted_loop_type & nucleotide column

    return data




## === cell 10
sequences = train_df["sequence"].values

print(sequences)

sequences.shape


## === cell 11
tokenize_cols = ["sequence", "structure", "predicted_loop_type"]


def tokenize_df(df, tokenizer, cols=tokenize_cols):
    """
    tokenizer is a tensorflow keras Tokenizer that has been already fitted on text
    """
    data = df.copy()
    for c in tokenize_cols:
        data[c] = tokenizer.texts_to_sequences(data[c])
    return data




## === cell 12
import os
import sys
import site
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _pb_major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _pb_ver is None or (_pb_major(_pb_ver) is not None and _pb_major(_pb_ver) >= 5):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    sp = site.getsitepackages()
except Exception:
    sp = []
if sp:
    tf_pkg_path = os.path.join(sp[0], "tensorflow")
    if os.path.isdir(tf_pkg_path) and tf_pkg_path not in sys.path:
        sys.path.insert(0, tf_pkg_path)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tf.keras.utils.set_random_seed(123)

tokenizer = Tokenizer(filters=None, lower=False, char_level=True)
tokenizer.fit_on_texts("().ACGUBEHIMSX")

temp = train_df.copy()
temp = tokenize_df(temp, tokenizer)

train_df = temp


## === cell 13
import matplotlib.pyplot as plt
import seaborn as sns

_base_drop = [
    "index",
    "id",
    "sequence",
    "structure",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
]
_numeric_df = train_df.drop(_base_drop, axis=1, errors="ignore").select_dtypes(
    include=[np.number]
)

corr_data = _numeric_df.corr()

corr_data


## === cell 14
mask = np.ma.masked_inside(
    corr_data.values, -0.15, 0.15
).mask  # get most powerful features

plt.figure(figsize=(13, 13))
sns.heatmap(corr_data, annot=True, mask=mask)


## === cell 15
"""
For tensorflow compatibility, metrics should have signature f(y_true, y_pred)
For sklearn compatibility, metrics should have signature f(y_true, y_pred, **kwargs)
"""


def score(raw_values=False, use_tf=False, **kwargs):
    """
    This is competition metric: Mean Columnwise Root Mean Square Error (MCRMSE)
    Averages RMSE loss over all scored columns (only 3 are scored)

    Parameters:
    For now, kwargs is ignored
    tf -- True if using in tensorflow, false if not
    col_dict is a dictionary that maps column number index to column name
        keys 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'
        values are numeric index of that column in y_pred
    raw_values determines if losses for each column are returned or just the average
        if True, losses for each of columns are returned
        if False, only average is returned

    Returns a loss function that computes MCRMSE for scored columns
    """

    col_dict = {"reactivity": 0, "deg_Mg_pH10": 1, "deg_Mg_50C": 3}

    unscored = set([0, 1, 2, 3, 4]) - set(col_dict.values())

    multi = "uniform_average"
    if raw_values:
        multi = "raw_values"

    def loss(y_true, y_pred):
        """
        y_true & y_pred may have more columns than needed for scoring
        select only necessary ones for scoring

        y_true & y_pred have shapes (n, 5), where n is # of id_seqpos combos
        """
        from sklearn.metrics import mean_squared_error

        y_true = np.array(y_true)  # convert to np for convenience
        y_pred = np.array(y_pred)

        y_true = y_true[:, list(col_dict.values())]
        y_pred = y_pred[:, list(col_dict.values())]

        metric = mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)
        return metric

    def loss_tf(y_true, y_pred):
        import tensorflow as tf
        import tensorflow.keras.backend as tf_kb

        for c in unscored:
            y_pred[:, c] = y_true[:, c]

        colwise_mse = tf_kb.mean(
            tf_kb.square(y_true - y_pred)
        )  # first take average squared difference over row examples (shape 1x3)
        return tf_kb.mean(
            tf_kb.sqrt(colwise_mse)
        )  # then sqrt and take average over columns (shape 1x1 / scalar)

    if not use_tf:
        return loss
    else:
        return loss_tf




## === cell 16
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]  # features only in train set
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = [
    "seq_length",
    "seq_scored",  # don't actually use seq_length and seq_scored for training - just metadata
    "index",
    "id",
]  # also not actually useful for training
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols


X_train = (
    train_df.drop(train_drop_cols, axis=1)  # selects only relevant columns
    .apply(
        lambda row: [e for e in row], axis=1
    )  # concatenates each element in row to list
    .apply(lambda e: np.array(e))  # creates 2d numpy array from those elements
)
X_train = np.stack(
    X_train.values, axis=0
)  # might not work for test data, since there is two different shapes

y_train = (
    train_df[target_cols]
    .apply(
        lambda row: [e for e in row], axis=1
    )  # concatenates each element in row to list
    .apply(lambda e: np.array(e))  # creates 2d numpy array from those elements
)
y_train = np.stack(y_train.values, axis=0)

"""
#maybe can use this as example weights -- higher signal_to_noise means higher weight?
#probably gotta make sure to cap the weight though, otherwise training dominated by top signal_to_noise
signal_to_noise = train_df[signal_cols]  #not necessary anymore
"""


np.array(train_df.drop(train_drop_cols, axis=1).iloc[0].explode()).reshape(
    3, -1
).astype("int")

"""
l = []
for arr in X_train.values[0]:
    l.append(arr)
    
np.array(l)"""


X_train


## === cell 17
y_train


## === cell 18
import tensorflow.keras.layers as layers

try:
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor  # noqa: F401
except ModuleNotFoundError:
    KerasRegressor = None


def make_model():

    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,  # play around with this
    }

    shape = (3, None)  # 3 sequences of unknown length

    inputs = tf.keras.Input(shape=shape)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(inputs)

    rnn_layer = layers.Bidirectional(
        layers.LSTM(30, return_sequences=True)
    )  # define the rnn type to use

    rnn_layers = []
    for i in range(embed.shape[1]):
        rnn_layers.append(rnn_layer(embed[:, i]))

    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # need output layer of 5 x seq_scored

    x = tf.transpose(
        x, (0, 2, 1)
    )  # reshape output so it makes more sense -- 5 layers of seq_scored length
    print(x.shape)
    x = x[:, :, :68]  # compact sequence of 107 into 68

    model = tf.keras.Model(inputs=inputs, outputs=x)

    optimizer = "adam"
    loss = "mse"

    model.compile(optimizer=optimizer, loss=loss, metrics=[])

    return model




## === cell 19
TF_FITPARAMS = {"epochs": 100, "batch_size": 100}
fp = TF_FITPARAMS


def make_model():
    import tensorflow as tf
    import tensorflow.keras.layers as layers

    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,
        "output_dim": 100,  # play around with this
    }

    shape = (3, None)  # 3 sequences of unknown length

    inputs = tf.keras.Input(shape=shape)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(inputs)

    rnn_layer = layers.Bidirectional(
        layers.LSTM(30, return_sequences=True)
    )  # define the rnn type to use

    rnn_layers = []
    for i in range(embed.shape[1]):
        rnn_layers.append(rnn_layer(embed[:, i]))

    x = layers.Concatenate()(rnn_layers)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # need output layer of 5 x seq_scored

    x = layers.Permute((2, 1))(x)

    print(x.shape)

    x = layers.Lambda(lambda t: t[:, :, :68])(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    optimizer = "adam"
    loss = "mse"

    model.compile(optimizer=optimizer, loss=loss, metrics=[])

    return model


model = make_model()

if any("KerasTensor cannot be used" in str(w) for w in []):
    pass  # no-op; kept to avoid altering external logic

model.summary()

print(X_train.shape, y_train.shape)

history = model.fit(X_train, y_train, **fp)


## === cell 20
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



## === cell 21
test_df = read_json("../input/stanford-covid-vaccine/test.json")

test_df


## === cell 22
temp_test = test_df[test_df["seq_length"] == 107]
temp_test = tokenize_df(temp_test, tokenizer)

temp_test


## === cell 23
X_test = (
    temp_test.drop(drop_cols, axis=1)  # selects only relevant columns
    .apply(
        lambda row: [e for e in row], axis=1
    )  # concatenates each element in row to list
    .apply(lambda e: np.array(e))  # creates 2d numpy array from those elements
)
X_test = np.stack(
    X_test.values, axis=0
)  # might not work for test data, since there is two different shapes

X_test


## === cell 24
test_pred = model.predict(X_test)
test_pred


## === cell 25
sub_df = temp_test.drop(
    tokenize_cols + ["index", "seq_scored"], axis=1
)  # first drop feature and irrelevant columns
sub_df["seqpos"] = sub_df.apply(lambda row: list(range(row["seq_length"])), axis=1)
sub_df = unpack_df_lists(sub_df, "seqpos")  # now unroll
sub_df["id_seqpos"] = sub_df.apply(
    lambda row: row["id"] + "_" + str(row["seqpos"]), axis=1
)


def pad_pred(p, i):
    """
    i is final length
    fills unscored rows with 0s
    """
    start = list(p)  # first elements
    padding = [0] * (i - len(start))  # add padding to make up difference
    return start + padding


pred_df = pd.DataFrame(
    [[pad_pred(l, 107) for l in e] for e in test_pred], columns=target_cols
)  # here, each row holds a list in each column
pred_df = unpack_df_lists(pred_df, target_cols)  # then unroll those lists
pred_df.index = sub_df.index
pred_df = pred_df.reset_index()

sub_df = sub_df.reset_index()
sub_df = sub_df[["id_seqpos"]]


print(sub_df.shape, pred_df.shape)
sub_df = sub_df.join(pred_df).set_index("index")


sub_df


## === cell 26
test_private = test_df[test_df["seq_length"] == 130]

sub_private = test_private[["id", "seq_length"]].copy()

sub_private["seqpos"] = sub_private["seq_length"].map(
    lambda n: list(range(int(n)))
)  # create seqpos column

seq_lens = sub_private["seq_length"].astype(int).tolist()
for c in target_cols:  # fill in targets with 0s
    sub_private[c] = pd.Series(
        ([0] * n for n in seq_lens), index=sub_private.index, dtype="object"
    )

sub_private = unpack_df_lists(sub_private, ["seqpos"] + target_cols)  # unroll lists

sub_private["id_seqpos"] = (
    sub_private["id"].astype(str) + "_" + sub_private["seqpos"].astype(str)
)

sub_private = sub_private.drop(["id", "seq_length", "seqpos"], axis=1)
sub_private = sub_private[["id_seqpos"] + target_cols]  # rearrange column order

sub_private


## === cell 27
sub_df = pd.concat([sub_df, sub_private])

sub_df


## === cell 28
sns.pairplot(pd.DataFrame(y_train.reshape(-1, 5), columns=target_cols))


## === cell 29
sns.pairplot(sub_df)


## === cell 30
sub_df.to_csv("submission.csv", index=False)
