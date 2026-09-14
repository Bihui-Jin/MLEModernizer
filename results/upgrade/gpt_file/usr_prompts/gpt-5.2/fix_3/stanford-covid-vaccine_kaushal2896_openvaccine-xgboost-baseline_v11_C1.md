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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

0.55663

# 6. Current score

0.47709

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47383) has done: 'Your current score (0.47376, lower-is-better) is substantially better than the target (0.55663), so to move toward the target we should slightly *reduce* performance with minimal, stable changes rather than improve it. The smallest safe lever here is to reduce model capacity/fit by narrowing the GridSearch space to simpler XGBoost settings (fewer trees, shallower depth, lower learning rate), while keeping the same feature engineering, model family, and training approach. I’m also fixing a bug in your MCRMSE checks where `y_true` and `y_pred` were swapped, so your local metric prints correctly (this doesn’t affect the Kaggle submission file). The submission writing and row alignment logic are preserved.'
- What this solution (achieved 0.47709) has done: 'Your current score (0.47383, lower-is-better) is substantially better than the target (0.55663), so to move toward the target we should make a small, stable change that slightly *reduces* performance without changing the modeling approach. The safest lever is to increase regularization and make boosting more conservative (fewer trees, shallower depth, higher min_child_weight/gamma), while keeping the same features, MultiOutputRegressor, and GridSearchCV workflow. I also keep the submission-building logic identical but add a strict row-count assertion against `sample_submission.csv` to avoid accidental misalignment (doesn’t change predictions). These changes should nudge the leaderboard score upward (worse) toward ~0.556 without risking invalid submissions.'

# 9. Code solution

## === cell 0
import json

import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor



## === cell 1
sample_sub_df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
train_df = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test_df = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)



## === cell 2
print(train_df.shape)
print(test_df.shape)
print(sample_sub_df.shape)



## === cell 3
train_df.head(3)



## === cell 4
test_df.head(3)



## === cell 5
sample_sub_df.head(3)



## === cell 6
train_df["reactivity"] = train_df["reactivity"].apply(lambda x: np.mean(x))
train_df["deg_Mg_pH10"] = train_df["deg_Mg_pH10"].apply(lambda x: np.mean(x))
train_df["deg_pH10"] = train_df["deg_pH10"].apply(lambda x: np.mean(x))
train_df["deg_Mg_50C"] = train_df["deg_Mg_50C"].apply(lambda x: np.mean(x))
train_df["deg_50C"] = train_df["deg_50C"].apply(lambda x: np.mean(x))



## === cell 7
train_df.head()



## === cell 8
train_df = train_df.drop(
    [
        "id",
        "index",
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
        "SN_filter",
        "signal_to_noise",
        "deg_pH10",
        "deg_50C",
    ],
    axis=1,
)
train_df.head()



## === cell 9
X_train = train_df.drop(["reactivity", "deg_Mg_pH10", "deg_Mg_50C"], axis=1)
Y_train = train_df[["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]]



## === cell 10
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train, Y_train, test_size=0.15, random_state=28
)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape




## === cell 11
def featurize(df):
    df["total_A_count"] = df["sequence"].apply(lambda s: s.count("A"))
    df["total_G_count"] = df["sequence"].apply(lambda s: s.count("G"))
    df["total_U_count"] = df["sequence"].apply(lambda s: s.count("U"))
    df["total_C_count"] = df["sequence"].apply(lambda s: s.count("C"))

    df["total_dot_count"] = df["structure"].apply(lambda s: s.count("."))
    df["total_ob_count"] = df["structure"].apply(lambda s: s.count("("))
    df["total_cb_count"] = df["structure"].apply(lambda s: s.count(")"))

    return df




## === cell 12
X_train = featurize(X_train)
X_test = featurize(X_test)



## === cell 13
X_train = X_train.drop(["sequence", "structure", "predicted_loop_type"], axis=1)
X_test = X_test.drop(["sequence", "structure", "predicted_loop_type"], axis=1)



## === cell 14
X_train.head()



## === cell 15
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




## === cell 16
def mcrmse_loss(y_true, y_pred, N=3):
    """
    Calculates competition eval metric (MCRMSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=0) / n)) / N


custom_scorer = make_scorer(mcrmse_loss, greater_is_better=False)



## === cell 17
xgb = XGBRegressor(
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=10.0,
    reg_alpha=5.0,
    min_child_weight=10.0,
    gamma=0.5,
    random_state=28,
)

params = {
    "estimator__n_estimators": [50, 100, 150],
    "estimator__learning_rate": [0.03, 0.05],
    "estimator__max_depth": [1, 2],
}

reg = MultiOutputRegressor(xgb)

gs = GridSearchCV(
    reg,
    param_grid=params,
    cv=5,
    return_train_score=True,
    n_jobs=-1,
    scoring=custom_scorer,
)
gs.fit(X_train, Y_train)



## === cell 18
gs.best_params_



## === cell 19
xgb = XGBRegressor(
    max_depth=gs.best_params_["estimator__max_depth"],
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=10.0,
    reg_alpha=5.0,
    min_child_weight=10.0,
    gamma=0.5,
    n_estimators=gs.best_params_["estimator__n_estimators"],
    learning_rate=gs.best_params_["estimator__learning_rate"],
    random_state=28,
)

reg = MultiOutputRegressor(xgb)
reg.fit(X_train, Y_train)



## === cell 20
mcrmse_loss(np.array(Y_train), reg.predict(X_train))



## === cell 21
mcrmse_loss(np.array(Y_test), reg.predict(X_test))



## === cell 22
test = featurize(test_df.drop(["index", "id"], axis=1))
test = test.drop(["sequence", "structure", "predicted_loop_type"], axis=1)
test = scaler.transform(test)



## === cell 23
preds = pd.DataFrame(reg.predict(test))



## === cell 24
submission_df = preds.loc[preds.index.repeat(list(test_df["seq_length"]))].reset_index(
    drop=True
)
submission_df = submission_df.rename(
    columns={0: "reactivity", 1: "deg_Mg_pH10", 2: "deg_Mg_50C"}
)
submission_df["id_seqpos"] = sample_sub_df["id_seqpos"]
submission_df["deg_pH10"] = 0.0
submission_df["deg_50C"] = 0.0
submission_df = submission_df[
    ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
]

assert submission_df.shape[0] == sample_sub_df.shape[0], (
    submission_df.shape,
    sample_sub_df.shape,
)
submission_df.head()



## === cell 25
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
