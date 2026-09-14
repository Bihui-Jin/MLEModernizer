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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.56141

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from copy import deepcopy
import json
import os

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder




## === cell 1
def mcrmse_loss(y_true, y_pred, N=3):
    """
    Calculates competition eval metric
    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=0) / n)) / N




## === cell 2
DATA_DIR = "../input/stanford-covid-vaccine"

train_path = os.path.join(DATA_DIR, "train.json")
test_path = os.path.join(DATA_DIR, "test.json")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

data_train = [json.loads(line) for line in open(train_path, "r")]
data_test = [json.loads(line) for line in open(test_path, "r")]
test_set = pd.read_csv(sample_path)



## === cell 3
for jason in data_train:
    jason["step"] = list(range(jason["seq_scored"]))
    jason["sequence"] = list(jason["sequence"])
    jason["structure"] = list(jason["structure"])
    jason["predicted_loop_type"] = list(jason["predicted_loop_type"])



## === cell 4
for jason in data_test:
    jason["step"] = list(range(jason["seq_scored"]))
    jason["sequence"] = list(jason["sequence"])
    jason["structure"] = list(jason["structure"])
    jason["predicted_loop_type"] = list(jason["predicted_loop_type"])



## === cell 5
train = pd.json_normalize(
    data=data_train,
    record_path="reactivity",
    meta=["id", "signal_to_noise", "SN_filter", "seq_length", "seq_scored"],
)
train.rename(columns={0: "reactivity"}, inplace=True)
train["step"] = pd.json_normalize(data=data_train, record_path="step")
train["sequence"] = pd.json_normalize(data=data_train, record_path="sequence")
train["structure"] = pd.json_normalize(data=data_train, record_path="structure")
train["predicted_loop_type"] = pd.json_normalize(
    data=data_train, record_path="predicted_loop_type"
)
train["reactivity_error"] = pd.json_normalize(
    data=data_train, record_path="reactivity_error"
)
train["deg_Mg_pH10"] = pd.json_normalize(data=data_train, record_path="deg_Mg_pH10")
train["deg_error_Mg_pH10"] = pd.json_normalize(
    data=data_train, record_path="deg_error_Mg_pH10"
)
train["deg_pH10"] = pd.json_normalize(data=data_train, record_path="deg_pH10")
train["deg_error_pH10"] = pd.json_normalize(
    data=data_train, record_path="deg_error_pH10"
)
train["deg_Mg_50C"] = pd.json_normalize(data=data_train, record_path="deg_Mg_50C")
train["deg_error_Mg_50C"] = pd.json_normalize(
    data=data_train, record_path="deg_error_Mg_50C"
)
train["deg_50C"] = pd.json_normalize(data=data_train, record_path="deg_50C")
train["deg_error_50C"] = pd.json_normalize(data=data_train, record_path="deg_error_50C")

train.set_index(["id", "step"], inplace=True)



## === cell 6
test = pd.json_normalize(
    data=data_test, record_path="sequence", meta=["id", "seq_length", "seq_scored"]
)
test.rename(columns={0: "sequence"}, inplace=True)
test["step"] = pd.json_normalize(data=data_test, record_path="step")
test["sequence"] = pd.json_normalize(data=data_test, record_path="sequence")
test["structure"] = pd.json_normalize(data=data_test, record_path="structure")
test["predicted_loop_type"] = pd.json_normalize(
    data=data_test, record_path="predicted_loop_type"
)
test.set_index(["id", "step"], inplace=True)



## === cell 7
train



## === cell 8
test



## === cell 9
np.random.seed(2020)



## === cell 10
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

X_train_enc = deepcopy(train)
X_test_enc = deepcopy(test)

missing_in_test = [c for c in X_train_enc.columns if c not in X_test_enc.columns]
for c in missing_in_test:
    X_test_enc[c] = np.nan

missing_in_train = [c for c in X_test_enc.columns if c not in X_train_enc.columns]
for c in missing_in_train:
    X_train_enc[c] = np.nan

X_test_enc = X_test_enc[X_train_enc.columns]

category_cols = [c for c in X_train_enc.columns if X_train_enc[c].dtype == "object"]
print(category_cols)

for c in category_cols:
    le = LabelEncoder()
    all_vals = pd.concat(
        [X_train_enc[c].astype(str), X_test_enc[c].astype(str)], axis=0
    )
    le.fit(all_vals.values)
    X_train_enc[c] = le.transform(X_train_enc[c].astype(str).values)
    X_test_enc[c] = le.transform(X_test_enc[c].astype(str).values)

X_train_enc = X_train_enc.apply(pd.to_numeric, errors="coerce")
X_test_enc = X_test_enc.apply(pd.to_numeric, errors="coerce")



## === cell 11
X = X_train_enc.drop(target_cols, axis=1)
y = X_train_enc.loc[:, target_cols]



## === cell 12
hgbr = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_iter=1750,
        max_depth=15,
        early_stopping=True,
        n_iter_no_change=10,
        learning_rate=0.0025,
        tol=1e-6,
        validation_fraction=0.2,
        verbose=2,
        max_leaf_nodes=64,
    ),
    n_jobs=4,
)

gbr = MultiOutputRegressor(
    GradientBoostingRegressor(
        loss="huber",
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.0025,
        tol=1e-7,
        validation_fraction=0.2,
        n_iter_no_change=15,
        verbose=2,
    )
)



## === cell 13
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=2020
)



## === cell 14
X_train



## === cell 15
hgbr.fit(X_train, y_train)



## === cell 16
y_pred = hgbr.predict(X_valid)



## === cell 17
print(mean_squared_error(y_valid, y_pred, squared=False))



## === cell 18
hgbr.fit(X_train_enc[X_test_enc.columns], y)



## === cell 19
y_pred_2 = hgbr.predict(X_test_enc)



## === cell 20
y_pred_2.shape



## === cell 21
baseline_means = y[target_cols].mean(axis=0).values  # shape (5,)
baseline_pred = np.tile(baseline_means, (y_pred_2.shape[0], 1))

alpha = 0.18  # preserves original intent from provided code
y_pred_2 = (1.0 - alpha) * y_pred_2 + alpha * baseline_pred



## === cell 22
test_meta = pd.DataFrame(
    [
        {"id": d["id"], "seq_scored": d["seq_scored"], "seq_length": d["seq_length"]}
        for d in data_test
    ]
).set_index("id")

pred_df = pd.DataFrame(
    y_pred_2, index=X_test_enc.index, columns=target_cols
)  # index: (id, step) for step<seq_scored

rows = []
for rid, grp in pred_df.groupby(level=0, sort=False):
    grp_sorted = grp.droplevel(0).sort_index()
    seq_scored = int(test_meta.loc[rid, "seq_scored"])
    seq_length = int(test_meta.loc[rid, "seq_length"])

    grp_sorted = grp_sorted.reindex(range(seq_scored)).ffill().bfill()

    if seq_length > seq_scored:
        last_row = grp_sorted.iloc[[-1]].to_numpy()
        ext = np.repeat(last_row, repeats=(seq_length - seq_scored), axis=0)
        ext_df = pd.DataFrame(
            ext, index=range(seq_scored, seq_length), columns=target_cols
        )
        full = pd.concat([grp_sorted, ext_df], axis=0)
    else:
        full = grp_sorted.iloc[:seq_length]

    full.insert(0, "id", rid)
    full.insert(1, "step", full.index.astype(int))
    rows.append(full.reset_index(drop=True))

full_pred = pd.concat(rows, axis=0, ignore_index=True)
full_pred["id_seqpos"] = (
    full_pred["id"].astype(str) + "_" + full_pred["step"].astype(str)
)

submission = test_set[["id_seqpos"]].merge(
    full_pred[["id_seqpos"] + target_cols], on="id_seqpos", how="left"
)

for i, c in enumerate(target_cols):
    submission[c] = submission[c].astype(float).fillna(float(baseline_means[i]))

submission = submission[test_set.columns]
print(submission.head(10))
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/674102650.py in <cell line: 0>()
     20     # Ensure we have predictions for 0..seq_scored-1
     21     # If anything is missing, forward-fill then back-fill as a fallback.
---> 22     grp_sorted = grp_sorted.reindex(range(seq_scored)).ffill().bfill()
     23 
     24     # Extend to seq_length by repeating last available prediction

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5376         tolerance=None,
   5377     ) -> DataFrame:
-> 5378         return super().reindex(
   5379             labels=labels,
   5380             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5608 
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy
   5612         ).__finalize__(self, method="reindex")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5631 
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method
   5635             )

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4427                 elif not self.is_unique:
   4428                     # GH#42568
-> 4429                     raise ValueError("cannot reindex on an axis with duplicate labels")
   4430                 else:
   4431                     indexer, _ = self.get_indexer_non_unique(target)

ValueError: cannot reindex on an axis with duplicate labels
