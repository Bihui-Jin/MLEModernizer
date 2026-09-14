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

# 5. Target score

0.41876

# 6. Current score

0.26966

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The crash happens because `oof_df` is referenced in cell 16 even when the training/OOF-generation cell that creates it was never run, so the variable doesn’t exist. The minimal safe fix is to guard the `to_csv` call with the same `globals()` presence check already used in earlier cells, without changing any modeling logic or data. This keeps output behavior identical when `oof_df` exists, and prevents a hard failure when it doesn’t. `submission` is already defined earlier (cell 3), so we leave its `to_csv` untouched.'
- What this solution (achieved 0.63824) has done: 'Your current score (0.63824, lower-is-better) is worse than the target (0.41876), so we need a small, low-risk improvement without changing the model approach. The biggest easy win here is to ensure training only uses high-quality rows (`SN_filter==1`) as recommended by the competition notes; this typically reduces noise and improves MCRMSE while keeping the same feature engineering and LightGBM training logic. I also make the categorical encoding robust so that any unexpected symbols map safely instead of becoming object/NaN (which can silently hurt LightGBM). Finally, I keep the submission writing unchanged and retain the existing guards so the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 0.26966) has done: 'I make two minimal, score-relevant fixes while keeping your LightGBM setup, folds, features, and loss unchanged. First, I stop the model from being penalized on the 39 unscored positions by training only on the first `seq_scored` positions (instead of all 107 positions created by the 68-loop plus boundary logic), which better matches the MCRMSE evaluation. Second, I ensure your test predictions are not left as zeros for any `id_seqpos` that didn’t appear in `test_data` (a common silent alignment issue) by explicitly filling missing rows with the per-target mean prediction, which is typically safer than zero for this competition. The script still run end-to-end and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import gc
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

sns.set(style="darkgrid")
SEEDS = 42

random.seed(SEEDS)
np.random.seed(SEEDS)
os.environ["PYTHONHASHSEED"] = str(SEEDS)




## === cell 1
def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred)) ** 0.5




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
        train_weight=None,
        valid_weight=None,
    ):
        if self.model_type == "lgb":
            self.tr_data = lgb.Dataset(train_x, label=train_y, weight=train_weight)
            self.vl_data = lgb.Dataset(valid_x, label=valid_y, weight=valid_weight)

            callbacks = []
            if early_stopping is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))
            if verbose is not None:
                callbacks.append(lgb.log_evaluation(period=verbose))

            self.model = lgb.train(
                params,
                self.tr_data,
                valid_sets=[self.tr_data, self.vl_data],
                num_boost_round=num_round,
                callbacks=callbacks,
            )

        return self.model

    def predict(self, X):
        if self.model_type == "lgb":
            return self.model.predict(X, num_iteration=self.model.best_iteration)

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")




## === cell 3
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
submission = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

if "SN_filter" in train.columns:
    train = train.loc[train["SN_filter"] == 1].reset_index(drop=True)



## === cell 4
train_data = []
for mol_id in train["id"].unique():
    sample_data = train.loc[train["id"] == mol_id]
    sample_seq_length = sample_data.seq_length.values[0]
    sample_seq_scored = int(sample_data.seq_scored.values[0])

    for i in range(sample_seq_scored):
        sample_dict = {
            "id": sample_data["id"].values[0],
            "id_seqpos": sample_data["id"].values[0] + "_" + str(i),
            "sequence": sample_data["sequence"].values[0][i],
            "structure": sample_data["structure"].values[0][i],
            "predicted_loop_type": sample_data["predicted_loop_type"].values[0][i],
            "reactivity": sample_data["reactivity"].values[0][i],
            "reactivity_error": sample_data["reactivity_error"].values[0][i],
            "deg_Mg_pH10": sample_data["deg_Mg_pH10"].values[0][i],
            "deg_error_Mg_pH10": sample_data["deg_error_Mg_pH10"].values[0][i],
            "deg_pH10": sample_data["deg_pH10"].values[0][i],
            "deg_error_pH10": sample_data["deg_error_pH10"].values[0][i],
            "deg_Mg_50C": sample_data["deg_Mg_50C"].values[0][i],
            "deg_error_Mg_50C": sample_data["deg_error_Mg_50C"].values[0][i],
            "deg_50C": sample_data["deg_50C"].values[0][i],
            "deg_error_50C": sample_data["deg_error_50C"].values[0][i],
        }

        shifts = [1, 2, 3, 4, 5]
        shift_cols = ["sequence", "structure", "predicted_loop_type"]
        for shift, col in itertools.product(shifts, shift_cols):
            if i - shift >= 0:
                sample_dict["b" + str(shift) + "_" + col] = sample_data[col].values[0][
                    i - shift
                ]
            else:
                sample_dict["b" + str(shift) + "_" + col] = -1

            if i + shift <= sample_seq_length - 1:
                sample_dict["a" + str(shift) + "_" + col] = sample_data[col].values[0][
                    i + shift
                ]
            else:
                sample_dict["a" + str(shift) + "_" + col] = -1

        train_data.append(sample_dict)
train_data = pd.DataFrame(train_data)
train_data.head()



## === cell 5
test_data = []
for mol_id in test["id"].unique():
    sample_data = test.loc[test["id"] == mol_id]
    sample_seq_length = sample_data.seq_length.values[0]
    for i in range(sample_seq_length):
        sample_dict = {
            "id": sample_data["id"].values[0],
            "id_seqpos": sample_data["id"].values[0] + "_" + str(i),
            "sequence": sample_data["sequence"].values[0][i],
            "structure": sample_data["structure"].values[0][i],
            "predicted_loop_type": sample_data["predicted_loop_type"].values[0][i],
        }

        shifts = [1, 2, 3, 4, 5]
        shift_cols = ["sequence", "structure", "predicted_loop_type"]
        for shift, col in itertools.product(shifts, shift_cols):
            if i - shift >= 0:
                sample_dict["b" + str(shift) + "_" + col] = sample_data[col].values[0][
                    i - shift
                ]
            else:
                sample_dict["b" + str(shift) + "_" + col] = -1

            if i + shift <= sample_seq_length - 1:
                sample_dict["a" + str(shift) + "_" + col] = sample_data[col].values[0][
                    i + shift
                ]
            else:
                sample_dict["a" + str(shift) + "_" + col] = -1

        test_data.append(sample_dict)
test_data = pd.DataFrame(test_data)
test_data.head()



## === cell 6
sequence_encmap = {"A": 0, "G": 1, "C": 2, "U": 3}
structure_encmap = {".": 0, "(": 1, ")": 2}
looptype_encmap = {"S": 0, "E": 1, "H": 2, "I": 3, "X": 4, "M": 5, "B": 6}

enc_targets = ["sequence", "structure", "predicted_loop_type"]
enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]

for t, m in zip(enc_targets, enc_maps):
    cols = [c for c in train_data.columns if t in c]
    for c in cols:
        train_data[c] = train_data[c].map(m).fillna(-1).astype(np.int16)
        test_data[c] = test_data[c].map(m).fillna(-1).astype(np.int16)



## === cell 7
not_use_cols = ["id", "id_seqpos"]
features = [c for c in test_data.columns if c not in not_use_cols]

scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
all_targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
targets = scored_targets



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
    "data_random_seed": SEEDS,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
}



## === cell 10
error_cols_for_filter = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_Mg_50C",
]
for c in error_cols_for_filter:
    if c in train_data.columns:
        train_data = train_data.loc[
            train_data[c].notna() & (train_data[c] > 0)
        ].reset_index(drop=True)

oof_df = train_data[["id_seqpos"]].copy()
for t in targets:
    oof_df[t] = np.nan

test_pred = np.zeros((len(test_data), len(targets)), dtype=np.float32)
result = {}
feature_importances = pd.DataFrame(columns=["target", "feature", "importance"])

groups = train_data["id"].values
X = train_data[features]
X_test = test_data[features]

target_to_errcol = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

for ti, target in enumerate(targets):
    y = train_data[target].values.astype(np.float32)
    oof = np.zeros(len(train_data), dtype=np.float32)
    fold_rmse = []

    err_col = target_to_errcol.get(target, None)
    if err_col is not None and err_col in train_data.columns:
        err = train_data[err_col].values.astype(np.float32)
        w_all = 1.0 / (np.maximum(err, 1e-3) ** 2)
        w_all = np.clip(w_all, 1e-2, 1e2).astype(np.float32)
    else:
        w_all = None

    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, y, groups=groups)):
        tr_x, tr_y = X.iloc[tr_idx], y[tr_idx]
        vl_x, vl_y = X.iloc[vl_idx], y[vl_idx]

        tr_w = w_all[tr_idx] if w_all is not None else None
        vl_w = w_all[vl_idx] if w_all is not None else None

        model = TreeModel("lgb")
        model.train(
            params=params,
            train_x=tr_x,
            train_y=tr_y,
            valid_x=vl_x,
            valid_y=vl_y,
            num_round=2000,
            early_stopping=100,
            verbose=None,
            train_weight=tr_w,
            valid_weight=vl_w,
        )

        vl_pred = model.predict(vl_x).astype(np.float32)
        oof[vl_idx] = vl_pred
        fold_rmse.append(rmse(vl_y, vl_pred))

        test_pred[:, ti] += model.predict(X_test).astype(np.float32) / FOLD_N

        fi = pd.DataFrame(
            {
                "target": target,
                "feature": model.feature_names_,
                "importance": model.feature_importances_,
            }
        )
        feature_importances = pd.concat(
            [feature_importances, fi], axis=0, ignore_index=True
        )

        del model, tr_x, tr_y, vl_x, vl_y, vl_pred, tr_w, vl_w
        gc.collect()

    oof_df[target] = oof
    result[target] = float(np.mean(fold_rmse))

pred_df = pd.DataFrame(test_pred, columns=targets)
pred_df["id_seqpos"] = test_data["id_seqpos"].values

submission_base = submission[["id_seqpos"]].copy()
submission = submission_base.merge(
    pred_df, on="id_seqpos", how="left", validate="one_to_one"
)
assert (
    submission["id_seqpos"].values.tolist()
    == submission_base["id_seqpos"].values.tolist()
)

if "deg_pH10" not in submission.columns:
    submission["deg_pH10"] = np.nan
if "deg_50C" not in submission.columns:
    submission["deg_50C"] = np.nan

submission["deg_pH10"] = submission["deg_pH10"].fillna(submission["deg_Mg_pH10"])
submission["deg_50C"] = submission["deg_50C"].fillna(submission["deg_Mg_50C"])

for t in targets:
    if t in submission.columns:
        mean_pred = float(np.nanmean(submission[t].values.astype(np.float32)))
        if not np.isfinite(mean_pred):
            mean_pred = 0.0
        submission[t] = submission[t].fillna(mean_pred)

for t in all_targets:
    if t not in submission.columns:
        submission[t] = 0.0
    submission[t] = submission[t].fillna(0.0).astype(np.float32)



## === cell 11
if "result" in globals():
    display(result)
    display(f"total : {np.mean(list(result.values()))}")
else:
    display("result is not defined (previous evaluation cell may not have been run).")



## === cell 12
if "feature_importances" not in globals():
    feature_importances = pd.DataFrame(columns=["target", "feature", "importance"])

for target in targets:
    tmp = feature_importances[feature_importances.target == target]
    if tmp.empty:
        continue

    order = list(
        tmp.groupby("feature")
        .mean(numeric_only=True)
        .sort_values("importance", ascending=False)
        .index
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=tmp, order=order)
    plt.title(target)
    plt.tight_layout()



## === cell 13
if "oof_df" in globals():
    display(oof_df.head())
else:
    display(
        "oof_df is not defined (OOF generation/training cell may not have been run)."
    )



## === cell 14
submission.head()



## === cell 15
if "oof_df" in globals():
    display(oof_df.shape)
else:
    display(
        "oof_df is not defined (OOF generation/training cell may not have been run)."
    )

display(submission.shape)



## === cell 16
if "oof_df" in globals():
    oof_df.to_csv("oof_df.csv", index=False)

submission.to_csv("submission.csv", index=False)
