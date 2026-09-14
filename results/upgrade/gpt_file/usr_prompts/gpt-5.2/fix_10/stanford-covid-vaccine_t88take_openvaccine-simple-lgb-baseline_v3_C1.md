# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

# 5. Code solution

## === cell 0
import gc
import os
import random
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

sns.set(style="darkgrid")
SEEDS = 42

random.seed(SEEDS)
np.random.seed(SEEDS)




## === cell 1
def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred)) ** 0.5


def mcrmse(df_true, df_pred, cols):
    rmses = []
    for c in cols:
        rmses.append(rmse(df_true[c].values, df_pred[c].values))
    return float(np.mean(rmses))




## === cell 2
class TreeModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.tr_data = None
        self.vl_data = None
        self.model = None

    def train(
        self,
        params,
        train_x,
        train_y,
        valid_x=None,
        valid_y=None,
        num_round=None,
        early_stopping=None,
        verbose=None,
    ):
        if self.model_type == "lgb":
            train_x_arr = (
                train_x.values
                if isinstance(train_x, (pd.DataFrame, pd.Series))
                else train_x
            )
            train_y_arr = (
                train_y.values
                if isinstance(train_y, (pd.Series, pd.DataFrame))
                else train_y
            )
            self.tr_data = lgb.Dataset(
                train_x_arr, label=train_y_arr, free_raw_data=False
            )

            valid_sets = [self.tr_data]
            valid_names = ["train"]

            if valid_x is not None and valid_y is not None:
                valid_x_arr = (
                    valid_x.values
                    if isinstance(valid_x, (pd.DataFrame, pd.Series))
                    else valid_x
                )
                valid_y_arr = (
                    valid_y.values
                    if isinstance(valid_y, (pd.Series, pd.DataFrame))
                    else valid_y
                )
                self.vl_data = lgb.Dataset(
                    valid_x_arr, label=valid_y_arr, free_raw_data=False
                )
                valid_sets.append(self.vl_data)
                valid_names.append("valid")

            callbacks = []
            if early_stopping is not None and self.vl_data is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))
            if verbose is not None:
                callbacks.append(lgb.log_evaluation(period=verbose))

            self.model = lgb.train(
                params,
                self.tr_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                num_boost_round=num_round,
                callbacks=callbacks,
            )

    def predict(self, X):
        if self.model_type == "lgb":
            X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
            best_iter = getattr(self.model, "best_iteration", None)
            if best_iter is None or best_iter == 0:
                return self.model.predict(X_arr)
            return self.model.predict(X_arr, num_iteration=best_iter)

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")




## === cell 3
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
submission = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

if "SN_filter" in train.columns:
    train = train.loc[train["SN_filter"] == 1].reset_index(drop=True)




## === cell 4
def _encode_char_matrix(strings, encmap, L):
    """
    strings: iterable of length N, each a python str of length L
    returns int16 array of shape (N, L), with -1 for unknowns
    """
    N = len(strings)
    out = np.full((N, L), -1, dtype=np.int16)
    for i, s in enumerate(strings):
        ss = s[:L]
        out[i, : len(ss)] = np.fromiter(
            (encmap.get(ch, -1) for ch in ss), count=len(ss), dtype=np.int16
        )
    return out


def _shift_features(mat, shifts=(1, 2, 3, 4, 5), prefix_a="a", prefix_b="b"):
    """
    mat: (N, L) int16
    returns dict of shifted matrices keyed by names like 'b1_sequence', 'a3_structure'
    shift logic matches original: for position p, bK is value at p-K if in range else -1,
    aK is value at p+K if in range else -1.
    """
    N, L = mat.shape
    out = {}
    for k in shifts:
        b = np.full((N, L), -1, dtype=np.int16)
        if k < L:
            b[:, k:] = mat[:, : L - k]
        out[f"{prefix_b}{k}"] = b

        a = np.full((N, L), -1, dtype=np.int16)
        if k < L:
            a[:, : L - k] = mat[:, k:]
        out[f"{prefix_a}{k}"] = a
    return out


L_FULL = 107
L_SCORED = int(train["seq_scored"].iloc[0])  # 68 for train

sequence_encmap = {"A": 0, "G": 1, "C": 2, "U": 3}
structure_encmap = {".": 0, "(": 1, ")": 2}
looptype_encmap = {"S": 0, "E": 1, "H": 2, "I": 3, "X": 4, "M": 5, "B": 6}

seq_train = _encode_char_matrix(train["sequence"].tolist(), sequence_encmap, L_FULL)
str_train = _encode_char_matrix(train["structure"].tolist(), structure_encmap, L_FULL)
loop_train = _encode_char_matrix(
    train["predicted_loop_type"].tolist(), looptype_encmap, L_FULL
)

N_tr = len(train)
ids_tr = train["id"].astype(str).values
pos_scored = np.arange(L_SCORED, dtype=np.int16)

id_rep_tr = np.repeat(ids_tr, L_SCORED)
pos_rep_tr = np.tile(pos_scored, N_tr).astype(np.int16)
id_seqpos_tr = np.char.add(
    np.char.add(id_rep_tr.astype(str), "_"), pos_rep_tr.astype(str)
)

sequence_flat = seq_train[:, :L_SCORED].reshape(-1)
structure_flat = str_train[:, :L_SCORED].reshape(-1)
loop_flat = loop_train[:, :L_SCORED].reshape(-1)

train_data = pd.DataFrame(
    {
        "id": id_rep_tr,
        "id_seqpos": id_seqpos_tr,
        "sequence": sequence_flat,
        "structure": structure_flat,
        "predicted_loop_type": loop_flat,
    }
)

target_cols = [
    "reactivity",
    "reactivity_error",
    "deg_Mg_pH10",
    "deg_error_Mg_pH10",
    "deg_pH10",
    "deg_error_pH10",
    "deg_Mg_50C",
    "deg_error_Mg_50C",
    "deg_50C",
    "deg_error_50C",
]
for c in target_cols:
    train_data[c] = np.asarray(train[c].tolist(), dtype=np.float64).reshape(-1)

shifts = (1, 2, 3, 4, 5)
for name, mat in (
    ("sequence", seq_train),
    ("structure", str_train),
    ("predicted_loop_type", loop_train),
):
    shifted = _shift_features(mat, shifts=shifts, prefix_a="a", prefix_b="b")
    for k in shifts:
        train_data[f"b{k}_{name}"] = shifted[f"b{k}"][:, :L_SCORED].reshape(-1)
        train_data[f"a{k}_{name}"] = shifted[f"a{k}"][:, :L_SCORED].reshape(-1)

print(train_data.head())



## === cell 5
sub_parts = submission[["id_seqpos"]].copy()
sub_parts[["id", "seqpos"]] = sub_parts["id_seqpos"].str.rsplit("_", n=1, expand=True)
sub_parts["seqpos"] = sub_parts["seqpos"].astype(int)

seq_test = _encode_char_matrix(test["sequence"].tolist(), sequence_encmap, L_FULL)
str_test = _encode_char_matrix(test["structure"].tolist(), structure_encmap, L_FULL)
loop_test = _encode_char_matrix(
    test["predicted_loop_type"].tolist(), looptype_encmap, L_FULL
)

N_te = len(test)
ids_te = test["id"].astype(str).values
pos_full = np.arange(L_FULL, dtype=np.int16)

id_rep_te = np.repeat(ids_te, L_FULL)
pos_rep_te = np.tile(pos_full, N_te).astype(np.int16)
id_seqpos_te = np.char.add(
    np.char.add(id_rep_te.astype(str), "_"), pos_rep_te.astype(str)
)

test_data_full = pd.DataFrame(
    {
        "id": id_rep_te,
        "id_seqpos": id_seqpos_te,
        "sequence": seq_test.reshape(-1),
        "structure": str_test.reshape(-1),
        "predicted_loop_type": loop_test.reshape(-1),
    }
)

for name, mat in (
    ("sequence", seq_test),
    ("structure", str_test),
    ("predicted_loop_type", loop_test),
):
    shifted = _shift_features(mat, shifts=shifts, prefix_a="a", prefix_b="b")
    for k in shifts:
        test_data_full[f"b{k}_{name}"] = shifted[f"b{k}"].reshape(-1)
        test_data_full[f"a{k}_{name}"] = shifted[f"a{k}"].reshape(-1)

test_data = sub_parts[["id_seqpos"]].merge(test_data_full, on="id_seqpos", how="left")
test_data["id"] = test_data["id"].fillna(
    test_data["id_seqpos"].str.rsplit("_", n=1, expand=True)[0]
)

print(test_data.head())
print("test_data rows:", len(test_data), "submission rows:", len(submission))



## === cell 6
enc_targets = ["sequence", "structure", "predicted_loop_type"]

for df in (train_data, test_data):
    for c in [c for c in df.columns if any(t in c for t in enc_targets)]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int16)



## === cell 7
not_use_cols = ["id", "id_seqpos"]
features = [c for c in test_data.columns if c not in not_use_cols]
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]



## === cell 8
FOLD_N = 5
gkf = GroupKFold(n_splits=FOLD_N)



## === cell 9
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "learning_rate": 0.1,
    "seed": SEEDS,
    "feature_fraction_seed": SEEDS,
    "bagging_seed": SEEDS,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": max(1, os.cpu_count() or 1),
    "verbosity": -1,
}



## === cell 10
folds = list(
    gkf.split(train_data[features], train_data["reactivity"], train_data["id"])
)

X_all = train_data[features].values
X_test = test_data[features].values

feature_importances_list = []
result = {}
oof_df = pd.DataFrame({"id_seqpos": train_data["id_seqpos"].values})
test_pred_df = pd.DataFrame({"id_seqpos": test_data["id_seqpos"].values})

NUM_ROUND = 1200
EARLY_STOPPING = 100

for target in targets:
    oof_pred = np.empty(len(train_data), dtype=np.float64)
    preds = np.zeros(len(test_data), dtype=np.float64)
    scores = 0.0

    y_all = train_data[target].values

    for n, (tr_idx, vl_idx) in enumerate(folds):
        tr_x = X_all[tr_idx]
        tr_y = y_all[tr_idx]
        vl_x = X_all[vl_idx]
        vl_y = y_all[vl_idx]
        vl_id = train_data["id_seqpos"].iloc[vl_idx].values

        model = TreeModel(model_type="lgb")
        model.train(
            params,
            tr_x,
            tr_y,
            vl_x,
            vl_y,
            num_round=NUM_ROUND,
            early_stopping=EARLY_STOPPING,
            verbose=1000,
        )

        feature_importances_list.append(
            pd.DataFrame(
                {
                    "feature": model.feature_names_,
                    "importance": model.feature_importances_,
                    "fold": n,
                    "target": target,
                }
            )
        )

        vl_pred = model.predict(vl_x)
        oof_pred[vl_idx] = vl_pred

        score = rmse(vl_y, vl_pred)
        scores += score / FOLD_N
        print(f"target={target} fold={n} rmse={score}")

        pred = model.predict(X_test)
        preds += pred / FOLD_N

        gc.collect()

    oof_df[target] = oof_pred
    test_pred_df[target] = preds

    print(f"{target}_cv_rmse : {scores}")
    result[target] = scores

feature_importances = (
    pd.concat(feature_importances_list, axis=0, ignore_index=True)
    if feature_importances_list
    else pd.DataFrame()
)



## === cell 11
print(result)
print(f"total_rmse_mean_5targets : {np.mean(list(result.values()))}")

oof_true = train_data[["id_seqpos"] + scored_targets].copy()
oof_pred = oof_df[["id_seqpos"] + scored_targets].copy()
oof_join = oof_true.merge(oof_pred, on="id_seqpos", suffixes=("_true", "_pred"))
mcrmse_score = mcrmse(
    oof_join[[f"{c}_true" for c in scored_targets]].rename(
        columns={f"{c}_true": c for c in scored_targets}
    ),
    oof_join[[f"{c}_pred" for c in scored_targets]].rename(
        columns={f"{c}_pred": c for c in scored_targets}
    ),
    scored_targets,
)
print(f"oof_mcrmse_scored_3targets : {mcrmse_score}")



## === cell 12
if len(feature_importances) > 0 and "target" in feature_importances.columns:
    for target in targets:
        tmp = feature_importances[feature_importances["target"] == target]
        if len(tmp) == 0:
            continue
        order = list(
            tmp.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)["feature"]
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(x="importance", y="feature", data=tmp, order=order)
        plt.title(target)
        plt.tight_layout()



## === cell 13
print(oof_df.head())



## === cell 14
submission_aligned = submission[["id_seqpos"]].merge(
    test_pred_df, on="id_seqpos", how="left"
)

for c in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    if c not in submission_aligned.columns:
        submission_aligned[c] = 0.0
    submission_aligned[c] = pd.to_numeric(
        submission_aligned[c], errors="coerce"
    ).astype(float)
    submission_aligned[c] = (
        submission_aligned[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

for c in targets:
    assert (
        submission_aligned[c].isna().sum() == 0
    ), f"Found NaNs in submission column {c}"

submission_aligned = submission_aligned[
    ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
]
print(submission_aligned.head())



## === cell 15
print(oof_df.shape)
print(submission_aligned.shape)

assert (
    submission_aligned.shape[0] == submission.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission_aligned.columns) == list(
    submission.columns
), "Column order mismatch vs sample_submission"



## === cell 16
oof_df.to_csv("oof_df.csv", index=False)
submission_aligned.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
