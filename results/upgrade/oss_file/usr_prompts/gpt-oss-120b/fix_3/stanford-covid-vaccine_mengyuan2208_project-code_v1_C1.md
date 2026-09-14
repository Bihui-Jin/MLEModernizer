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

3.10

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.4904

# 6. Current score

0.42774

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42774) has done: 'Implemented fixes for the runtime errors:

- **Cell 14**: Corrected random sample selection to use the actual DataFrame length, preventing out‑of‑bounds indexing.
- **Cell 28**: Replaced the incorrect `str.rsplit` usage with `str.rsplit(..., expand=True)` to properly split `id_seqpos` into `id` and `seqpos` columns and cast `seqpos` to integer.
- Added brief comments describing each fix.

The rest of the pipeline remains unchanged, preserving the original modelling logic and ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.42774) has done: 'Implemented fixes to resolve import errors and runtime crashes:
- Removed unavailable `forgi` imports and guarded the optional RNA‑structure plot call.
- Adjusted cell numbering to start at 1 as required.
- Kept all core modeling logic unchanged to preserve the current low MCRMSE score (0.42774), which already outperforms the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sns.set(style="darkgrid")




## === cell 1
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")




## === cell 2
print("train data shape: ", train.shape)
print("test data shape: ", test.shape)
print("sample submission shape: ", sample_sub.shape)




## === cell 3
train.head()




## === cell 4
test.head()




## === cell 5
sample_sub.head()




## === cell 6
train["seq_length"].value_counts()




## === cell 7
test["seq_length"].value_counts()




## === cell 8
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.boxplot(data=train, x="signal_to_noise", ax=ax[0])
ax[0].set_title("Signal/Noise")
sns.countplot(data=train, y="SN_filter", ax=ax[1])
ax[1].set_title("SN_filter")
plt.show()




## === cell 9
avg_reactivity = np.array(list(map(np.array, train.reactivity))).mean(axis=0)
avg_deg_50C = np.array(list(map(np.array, train.deg_50C))).mean(axis=0)
avg_deg_pH10 = np.array(list(map(np.array, train.deg_pH10))).mean(axis=0)
avg_deg_Mg_50C = np.array(list(map(np.array, train.deg_Mg_50C))).mean(axis=0)
avg_deg_Mg_pH10 = np.array(list(map(np.array, train.deg_Mg_pH10))).mean(axis=0)




## === cell 10
plt.figure(figsize=(20, 10))
sns.lineplot(x=range(68), y=avg_reactivity, label="avg_reactivity")
sns.lineplot(x=range(68), y=avg_deg_50C, label="avg_deg_50C")
sns.lineplot(x=range(68), y=avg_deg_pH10, label="avg_deg_ph10")
sns.lineplot(x=range(68), y=avg_deg_Mg_50C, label="avg_deg_Mg_50C")
sns.lineplot(x=range(68), y=avg_deg_Mg_pH10, label="avg_deg_Mg_pH10")
plt.xlabel("Positions on the RNA sequence")
plt.xticks(range(0, 68))
plt.ylabel("Values")
plt.title("Average Target Values vs Positions")
plt.show()




## === cell 11
np.corrcoef(
    np.vstack(
        (avg_reactivity, avg_deg_50C, avg_deg_pH10, avg_deg_Mg_50C, avg_deg_Mg_pH10)
    )
)




## === cell 12
def plot_sample(sample):
    """
    Reference: https://www.kaggle.com/erelin6613/openvaccine-rna-visualization
    Visualize RNA using viennarna
    Arguments:
    sample: pandas.series, a sample of RNA, must contain 'id', structure' and 'sequence'
    """
    pass




## === cell 13
sample = train.sample(1).iloc[0]




## === cell 14
mask = train["SN_filter"] == 1
train = train[mask]




## === cell 15
train = train.drop(
    [
        "signal_to_noise",
        "SN_filter",
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
    ],
    axis=1,
)
train.shape




## === cell 16
train_data = []
for ID in train["id"].unique():
    entry = train.loc[train["id"] == ID]
    for i in range(entry["seq_scored"].values[0]):
        sample_dict = {
            "id": entry["id"].values[0],
            "id_seqpos": f"{entry['id'].values[0]}_{i}",
            "sequence": entry["sequence"].values[0][i],
            "structure": entry["structure"].values[0][i],
            "predicted_loop_type": entry["predicted_loop_type"].values[0][i],
            "reactivity": entry["reactivity"].values[0][i],
            "deg_Mg_pH10": entry["deg_Mg_pH10"].values[0][i],
            "deg_pH10": entry["deg_pH10"].values[0][i],
            "deg_Mg_50C": entry["deg_Mg_50C"].values[0][i],
            "deg_50C": entry["deg_50C"].values[0][i],
        }
        train_data.append(sample_dict)
train_data = pd.DataFrame(train_data)
train_data.head()




## === cell 17
test_data = []
for ID in test["id"].unique():
    entry = test.loc[test["id"] == ID]
    for i in range(entry["seq_length"].values[0]):
        sample_dict = {
            "id": entry["id"].values[0],
            "id_seqpos": f"{entry['id'].values[0]}_{i}",
            "sequence": entry["sequence"].values[0][i],
            "structure": entry["structure"].values[0][i],
            "predicted_loop_type": entry["predicted_loop_type"].values[0][i],
        }
        test_data.append(sample_dict)
test_data = pd.DataFrame(test_data)
test_data.head()




## === cell 18
dict_sequence = {"A": 0, "G": 1, "U": 2, "C": 3}
dict_structure = {"(": 0, ")": 1, ".": 2}
dict_looptype = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
train_data["sequence"] = train_data["sequence"].replace(dict_sequence)
train_data["structure"] = train_data["structure"].replace(dict_structure)
train_data["predicted_loop_type"] = train_data["predicted_loop_type"].replace(
    dict_looptype
)
test_data["sequence"] = test_data["sequence"].replace(dict_sequence)
test_data["structure"] = test_data["structure"].replace(dict_structure)
test_data["predicted_loop_type"] = test_data["predicted_loop_type"].replace(
    dict_looptype
)
train_data.head()




## === cell 19
X_train = train_data.drop(
    ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"], axis=1
)
Y_train = train_data[["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]]




## === cell 20
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)
X_train.shape, X_val.shape, Y_train.shape, Y_val.shape




## === cell 21
def mcrmse_loss(y_true, y_pred, N=5):
    """
    Calculates competition eval metric
    """
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=0) / n)) / N




## === cell 22
xgb = XGBRegressor(
    n_estimators=800,
    eval_metric="rmse",
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
)




## === cell 23
features = ["sequence", "structure", "predicted_loop_type"]
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
sub = pd.DataFrame(test_data["id_seqpos"])
feature_importances = pd.DataFrame(index=features)
tr_X = X_train[features]
vl_X = X_val[features]
ts_X = test_data[features]
tr_X.shape, vl_X.shape, ts_X.shape




## === cell 24
for i in range(5):
    tr_Y, vl_Y = Y_train[targets[i]], Y_val[targets[i]]
    xgb.fit(tr_X, tr_Y)
    feature_importances.insert(i, targets[i], xgb.feature_importances_)
    vl_pred = xgb.predict(vl_X)
    loss = mcrmse_loss(vl_Y, vl_pred)
    print(f"{targets[i]} loss : {loss}")
    sub[targets[i]] = xgb.predict(ts_X)




## === cell 25
fig, ax = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle("Feature Importances Visualization")
for i in range(5):
    sns.barplot(
        x=features, y=targets[i], data=feature_importances, ax=ax[i // 2][i % 2]
    )
plt.tight_layout()
plt.show()




## === cell 26
sub.to_csv("submission.csv", index=False)
sub.shape, sub.head()




## === cell 27
sub[["id", "seqpos"]] = sub["id_seqpos"].str.rsplit("_", n=1, expand=True)
sub["seqpos"] = sub["seqpos"].astype(int)
sub = sub.sort_values(by=["id", "seqpos"]).reset_index(drop=True)
reac0 = sub.groupby("id")["reactivity"].apply(list).iloc[0]
reac1 = (
    sub.groupby("id")["reactivity"].apply(list).iloc[1000]
    if sub["id"].nunique() > 1000
    else reac0
)
reac2 = (
    sub.groupby("id")["reactivity"].apply(list).iloc[2000]
    if sub["id"].nunique() > 2000
    else reac0
)
reac3 = (
    sub.groupby("id")["reactivity"].apply(list).iloc[3000]
    if sub["id"].nunique() > 3000
    else reac0
)




## === cell 28
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
fig.suptitle("Predicted Reactivity vs Position")
sns.lineplot(data=reac0, ax=ax[0])
sns.lineplot(data=reac1, ax=ax[1])
sns.lineplot(data=reac2, ax=ax[2])
sns.lineplot(data=reac3, ax=ax[3])
plt.show()
