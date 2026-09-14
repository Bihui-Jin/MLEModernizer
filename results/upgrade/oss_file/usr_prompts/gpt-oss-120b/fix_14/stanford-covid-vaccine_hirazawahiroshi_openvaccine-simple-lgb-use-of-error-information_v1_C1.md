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
import os
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import concurrent.futures

SEEDS = 42
np.random.seed(SEEDS)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return (mean_squared_error(y_true, y_pred)) ** 0.5




## === cell 1
BASE_PATH = "/kaggle/input/stanford-covid-vaccine"
train_path = os.path.join(BASE_PATH, "train.json")
test_path = os.path.join(BASE_PATH, "test.json")
sample_submission_path = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
submission_template = pd.read_csv(sample_submission_path)




## === cell 2
train_data_rows = []
for mol_id in train["id"].unique():
    sample_data = train.loc[train["id"] == mol_id]
    sample_seq_length = sample_data["seq_length"].values[0]
    for i in range(68):  # only first 68 positions have ground‑truth
        sample_dict = {
            "id": sample_data["id"].values[0],
            "id_seqpos": f"{sample_data['id'].values[0]}_{i}",
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
                sample_dict[f"b{shift}_{col}"] = sample_data[col].values[0][i - shift]
            else:
                sample_dict[f"b{shift}_{col}"] = -1
            if i + shift <= sample_seq_length - 1:
                sample_dict[f"a{shift}_{col}"] = sample_data[col].values[0][i + shift]
            else:
                sample_dict[f"a{shift}_{col}"] = -1
        train_data_rows.append(sample_dict)

train_data = pd.DataFrame(train_data_rows)




## === cell 3
test_data_rows = []
for mol_id in test["id"].unique():
    sample_data = test.loc[test["id"] == mol_id]
    sample_seq_length = sample_data["seq_length"].values[0]
    for i in range(sample_seq_length):
        sample_dict = {
            "id": sample_data["id"].values[0],
            "id_seqpos": f"{sample_data['id'].values[0]}_{i}",
            "sequence": sample_data["sequence"].values[0][i],
            "structure": sample_data["structure"].values[0][i],
            "predicted_loop_type": sample_data["predicted_loop_type"].values[0][i],
        }
        shifts = [1, 2, 3, 4, 5]
        shift_cols = ["sequence", "structure", "predicted_loop_type"]
        for shift, col in itertools.product(shifts, shift_cols):
            if i - shift >= 0:
                sample_dict[f"b{shift}_{col}"] = sample_data[col].values[0][i - shift]
            else:
                sample_dict[f"b{shift}_{col}"] = -1
            if i + shift <= sample_seq_length - 1:
                sample_dict[f"a{shift}_{col}"] = sample_data[col].values[0][i + shift]
            else:
                sample_dict[f"a{shift}_{col}"] = -1
        test_data_rows.append(sample_dict)

test_data = pd.DataFrame(test_data_rows)




## === cell 4
sequence_encmap = {"A": 0, "G": 1, "C": 2, "U": 3}
structure_encmap = {".": 0, "(": 1, ")": 2}
looptype_encmap = {"S": 0, "E": 1, "H": 2, "I": 3, "X": 4, "M": 5, "B": 6}
enc_targets = ["sequence", "structure", "predicted_loop_type"]
enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]

for t, m in zip(enc_targets, enc_maps):
    for col in [c for c in train_data.columns if t in c]:
        train_data[col] = train_data[col].replace(m)
        test_data[col] = test_data[col].replace(m)

cat_features = [
    col
    for col in train_data.columns
    if any(key in col for key in ["sequence", "structure", "predicted_loop_type"])
    and col not in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
]
train_data[cat_features] = train_data[cat_features].astype(int)
test_data[cat_features] = test_data[cat_features].astype(int)

not_use_cols = ["id", "id_seqpos"]
features = [c for c in test_data.columns if c not in not_use_cols]
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

X_train_np = train_data[features].astype(np.float32).values
X_test_np = test_data[features].astype(np.float32).values

cat_feature_indices = [i for i, col in enumerate(features) if col in cat_features]

FOLD_N = 5
gkf = GroupKFold(n_splits=FOLD_N)
folds = list(
    gkf.split(X_train_np, train_data[targets[0]].values, groups=train_data["id"])
)

params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "learning_rate": 0.005,
    "num_leaves": 511,
    "min_child_samples": 30,
    "max_bin": 255,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": SEEDS,
    "verbosity": -1,
    "bagging_seed": SEEDS,
    "feature_fraction_seed": SEEDS,
    "num_threads": 4,  # limit threads to avoid oversubscription
}

feature_importance_list = []
result = {}
oof_df = pd.DataFrame({"id_seqpos": train_data["id_seqpos"]})
true_vals_series = train_data.set_index("id_seqpos")[targets]




## === cell 5
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
        early_stopping_rounds=None,
        verbose=None,
    ):
        if self.model_type == "lgb":
            self.tr_data = lgb.Dataset(
                train_x,
                label=train_y,
                categorical_feature=cat_feature_indices,
                free_raw_data=False,
            )
            valid_sets = [self.tr_data]
            callbacks = []
            if valid_x is not None and valid_y is not None:
                self.vl_data = lgb.Dataset(
                    valid_x,
                    label=valid_y,
                    categorical_feature=cat_feature_indices,
                    free_raw_data=False,
                )
                valid_sets.append(self.vl_data)
                if early_stopping_rounds is not None:
                    callbacks.append(
                        lgb.early_stopping(
                            stopping_rounds=early_stopping_rounds, verbose=False
                        )
                    )
            if verbose is not None:
                callbacks.append(lgb.log_evaluation(period=verbose))
            self.model = lgb.train(
                params,
                self.tr_data,
                num_boost_round=num_round,
                valid_sets=valid_sets,
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


def train_target(target):
    """Train all folds for a single target and return OOF, predictions and importances."""
    oof_target = pd.DataFrame()
    preds = np.zeros(len(test_data))
    local_fi = []

    for fold, (tr_idx, vl_idx) in enumerate(folds):
        tr_x = X_train_np[tr_idx]
        tr_y = train_data[target].values[tr_idx]
        vl_x = X_train_np[vl_idx]
        vl_y = train_data[target].values[vl_idx]
        vl_id = train_data["id_seqpos"].iloc[vl_idx]

        model = TreeModel(model_type="lgb")
        model.train(
            params,
            tr_x,
            tr_y,
            vl_x,
            vl_y,
            num_round=12000,
            early_stopping_rounds=800,
            verbose=200,
        )

        fi_tmp = pd.DataFrame(
            {
                "feature": model.feature_names_,
                "importance": model.feature_importances_,
                "fold": fold,
                "target": target,
            }
        )
        local_fi.append(fi_tmp)

        vl_pred = model.predict(vl_x)
        oof_target = pd.concat(
            [oof_target, pd.DataFrame({"id_seqpos": vl_id, target: vl_pred})],
            ignore_index=True,
        )
        preds += model.predict(X_test_np) / FOLD_N

    true_vals = true_vals_series.loc[oof_target["id_seqpos"], target]
    oof_score = rmse(true_vals, oof_target[target])
    return (target, oof_target, preds, local_fi, oof_score)


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(train_target, t): t for t in targets}
    for future in concurrent.futures.as_completed(futures):
        target, oof_target, preds, local_fi, oof_score = future.result()
        feature_importance_list.extend(local_fi)
        oof_df = oof_df.merge(oof_target, on="id_seqpos", how="inner")
        submission_template[target] = pd.Series(
            preds, index=submission_template["id_seqpos"]
        )
        result[target] = oof_score
        print(f"{target}_rmse (OOF): {oof_score:.5f}")

feature_importances = pd.concat(feature_importance_list, ignore_index=True)




## === cell 6
print("Result per target (OOF RMSE):")
for k, v in result.items():
    print(f"{k}: {v:.5f}")

scored_targets = ["reactivity", "deg_Mg_pH10", "deg_50C"]
score_vals = [result[t] for t in scored_targets]
print(f"Mean RMSE (scored columns): {np.mean(score_vals):.5f}")




## === cell 7
for target in targets:
    tmp = feature_importances[feature_importances["target"] == target]
    order = (
        tmp.groupby("feature")["importance"].mean().sort_values(ascending=False).index
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=tmp, order=order)
    plt.title(f"Feature importance - {target}")
    plt.tight_layout()
    plt.show()




## === cell 8
oof_df.head()




## === cell 9
submission_template.head()




## === cell 10
oof_df.to_csv("oof_df.csv", index=False)
submission_template.to_csv("submission.csv", index=False)
print("Files saved: oof_df.csv, submission.csv")
