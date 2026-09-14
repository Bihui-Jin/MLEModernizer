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

0.56142

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder



## === cell 1
DATA_DIR = "../input/stanford-covid-vaccine"

train_path = os.path.join(DATA_DIR, "train.json")
test_path = os.path.join(DATA_DIR, "test.json")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

data_train = [json.loads(line) for line in open(train_path, "r")]
data_test = [json.loads(line) for line in open(test_path, "r")]
test_set = pd.read_csv(sample_path)



## === cell 2
for jason in data_train:
    jason["step"] = list(range(jason["seq_scored"]))
    jason["sequence"] = list(jason["sequence"])
    jason["structure"] = list(jason["structure"])
    jason["predicted_loop_type"] = list(jason["predicted_loop_type"])



## === cell 3
for jason in data_test:
    jason["step"] = list(range(jason["seq_scored"]))
    jason["sequence"] = list(jason["sequence"])
    jason["structure"] = list(jason["structure"])
    jason["predicted_loop_type"] = list(jason["predicted_loop_type"])



## === cell 4
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



## === cell 5
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



## === cell 6
train



## === cell 7
test



## === cell 8
np.random.seed(2020)



## === cell 9
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

X_train_enc = deepcopy(train)
X_test_enc = deepcopy(test)

category_cols_train = [
    c
    for c in X_train_enc.columns
    if (X_train_enc[c].dtype == "object" and c not in TARGET_COLS)
]
category_cols_test = [
    c
    for c in X_test_enc.columns
    if (X_test_enc[c].dtype == "object" and c not in TARGET_COLS)
]
cols_to_encode = sorted(set(category_cols_train).intersection(category_cols_test))

print("Train object cols:", category_cols_train)
print("Test object cols:", category_cols_test)
print("Encoding cols:", cols_to_encode)

encoders = {}
for col in cols_to_encode:
    le = LabelEncoder()
    combined = pd.concat(
        [X_train_enc[col].astype(str), X_test_enc[col].astype(str)],
        axis=0,
        ignore_index=True,
    )
    le.fit(combined)
    X_train_enc[col] = le.transform(X_train_enc[col].astype(str)).astype(np.int16)
    X_test_enc[col] = le.transform(X_test_enc[col].astype(str)).astype(np.int16)
    encoders[col] = le



## === cell 10
X = X_train_enc.drop(TARGET_COLS, axis=1)
y = X_train_enc.loc[:, TARGET_COLS]

missing_in_test = [c for c in X.columns if c not in X_test_enc.columns]
for c in missing_in_test:
    X_test_enc[c] = 0.0

X_test_enc = X_test_enc[X.columns]

X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
X_test_enc = X_test_enc.apply(pd.to_numeric, errors="coerce").astype(np.float32)

X = X.fillna(0.0)
X_test_enc = X_test_enc.fillna(0.0)

y = y.apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0)

print("X shape:", X.shape, "X_test shape:", X_test_enc.shape, "y shape:", y.shape)



## === cell 11
hgbr = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=10,
        learning_rate=0.0025,
        tol=1e-5,
        verbose=0,
    ),
    n_jobs=4,
)



## === cell 12
unique_ids = X.index.get_level_values("id").unique()
train_ids, valid_ids = train_test_split(unique_ids, test_size=0.2, random_state=2020)

X_train = X.loc[(train_ids, slice(None)), :]
y_train = y.loc[(train_ids, slice(None)), :]

X_valid = X.loc[(valid_ids, slice(None)), :]
y_valid = y.loc[(valid_ids, slice(None)), :]



## === cell 13
X_train



## === cell 14
hgbr.fit(X_train, y_train)



## === cell 15
y_pred = hgbr.predict(X_valid)



## === cell 16
print(mean_squared_error(y_valid, y_pred, squared=False))



## === cell 17
X_train_enc[X_test_enc.columns]



## === cell 18
BLEND_OOF_WEIGHT = (
    0.35  # 0 => only full-fit (likely better); 1 => only OOF-fit (likely worse)
)

hgbr_full = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=10,
        learning_rate=0.0025,
        tol=1e-5,
        verbose=0,
    ),
    n_jobs=4,
)
hgbr_full.fit(X, y)
y_pred_full = hgbr_full.predict(X_test_enc)

hgbr_oof = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=10,
        learning_rate=0.0025,
        tol=1e-5,
        verbose=0,
    ),
    n_jobs=4,
)
hgbr_oof.fit(X_train, y_train)
y_pred_oof = hgbr_oof.predict(X_test_enc)

y_pred_2 = (1.0 - BLEND_OOF_WEIGHT) * y_pred_full + BLEND_OOF_WEIGHT * y_pred_oof



## === cell 19
y_pred_2.shape



## === cell 20
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

pred_df_68 = pd.DataFrame(
    y_pred_2, index=X_test_enc.index, columns=target_cols
).reset_index()

test_meta = pd.DataFrame(
    [
        {
            "id": d["id"],
            "seq_length": int(d["seq_length"]),
            "seq_scored": int(d["seq_scored"]),
        }
        for d in data_test
    ]
)

pred_df_68 = pred_df_68.merge(test_meta, on="id", how="left")

rows = []
for _id, grp in pred_df_68.groupby("id", sort=False):
    grp = grp.sort_values("step").reset_index(drop=True)
    seq_len = int(grp["seq_length"].iloc[0])
    seq_scored = int(grp["seq_scored"].iloc[0])

    base = grp[target_cols].to_numpy(dtype=np.float32)

    if seq_len < seq_scored:
        base = base[:seq_len]
        seq_scored = seq_len

    if seq_len > seq_scored:
        pad = np.repeat(base[[-1], :], repeats=(seq_len - seq_scored), axis=0)
        full = np.vstack([base, pad])
    else:
        full = base

    id_seqpos = [f"{_id}_{i}" for i in range(seq_len)]
    out = pd.DataFrame(full, columns=target_cols)
    out.insert(0, "id_seqpos", id_seqpos)
    rows.append(out)

submission_full = pd.concat(rows, axis=0, ignore_index=True)

submission = test_set[["id_seqpos"]].merge(submission_full, on="id_seqpos", how="left")
for col in target_cols:
    submission[col] = pd.to_numeric(submission[col], errors="coerce").astype(float)
submission.fillna(0.0, inplace=True)

assert submission.shape[0] == test_set.shape[0]
assert list(submission.columns) == list(test_set.columns)

submission.to_csv("submission.csv", index=False)
submission.head(10)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3937913168.py in <cell line: 0>()
     45     id_seqpos = [f"{_id}_{i}" for i in range(seq_len)]
     46     out = pd.DataFrame(full, columns=target_cols)
---> 47     out.insert(0, "id_seqpos", id_seqpos)
     48     rows.append(out)
     49 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
   5169             value = value.iloc[:, 0]
   5170 
-> 5171         value, refs = self._sanitize_column(value)
   5172         self._mgr.insert(loc, column, value, refs=refs)
   5173 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (107) does not match length of index (146)
