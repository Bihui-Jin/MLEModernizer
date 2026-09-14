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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.3568481764028675

# 6. Current score

0.40892

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I replace the missing‑file load with a robust search for the provided `sample_submission.csv`, read it, apply the same scaling that the original script intended, and then write a proper `submission.csv`. This fixes the file‑not‑found and undefined‑variable errors while preserving the original logic.'
- What this solution (achieved 0.63824) has done: 'We replace the placeholder scaling with a lightweight, per‑base regression model: we flatten the training JSON into rows per scored position, encode simple sequence features (nucleotide, loop type, structure character, position index) and fit a small Ridge regression for each of the five targets. The same features are built for the test set (including all 107 positions) and predictions are written to a properly‑formatted `submission.csv`. This adds predictive power while keeping the workflow simple and self‑contained, moving the MCRMSE from 0.638 toward the target 0.357.'
- What this solution (achieved 0.4228) has done: 'I fixed the JSON loading to handle line‑delimited files, expanded the feature engineering (added quadratic position term and one‑hot encodings for nucleotide, loop type, and structure) and switched the Ridge regularization to a smaller α (0.01) for a slightly more flexible model. These changes resolve the runtime errors and give the model a modest boost toward the target score while keeping the overall workflow unchanged.'
- What this solution (achieved 0.41704) has done: 'I add two lightweight features (sine and cosine of the normalized position) and standard‑scale the feature matrix before training. This keeps the same Ridge models but lets them use better‑conditioned inputs, and I lower the regularisation (α = 0.001) to give the model a bit more flexibility. These minimal changes are expected to lower the MCRMSE toward the target while preserving the original workflow.'
- What this solution (achieved 0.41704) has done: 'I add a couple of lightweight positional features (raw position normalized by the scored length and its interaction with the overall sequence length) and slightly reduce the Ridge regularisation (α = 0.0005). These changes keep the overall pipeline unchanged while giving the model a bit more expressive power, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41704) has done: 'I slightly increase the Ridge regularisation strength (α = 0.01 instead of 0.0005). A modestly larger α typically reduces over‑fitting on the training rows and improves generalisation on the scored bases, moving the MCRMSE closer to the target score while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.41704) has done: 'I add a quick validation split and loop over a few Ridge regularisation strengths to pick the α that gives the lowest internal MCRMSE (computed on the three scored targets). The chosen α is then used to train the final models on the full training data, which should modestly lower the leaderboard score while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.40892) has done: 'Implemented a finer α search and added second‑degree polynomial features (via `PolynomialFeatures`) before scaling. This keeps the Ridge‑based modeling unchanged while giving the linear models richer interactions, which should lower the validation MCRMSE and move the score nearer the target.'
- What this solution (achieved 0.40892) has done: 'The update fixes the validation split so that features and targets stay aligned, enabling a reliable search for the best Ridge regularisation strength. By splitting indices once and using them for all targets, the chosen α is more appropriate, which should lower the MCRMSE toward the target while preserving the original modeling pipeline and output format. No other logic is changed.'

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


def find_path(rel_paths):
    for p in rel_paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Required file not found.")


train_path = find_path(
    [
        "data/train.json",
        "input/train.json",
        "train.json",
        "../input/train.json",
        "../input/stanford-covid-vaccine/train.json",
    ]
)

test_path = find_path(
    [
        "data/test.json",
        "input/test.json",
        "test.json",
        "../input/test.json",
        "../input/stanford-covid-vaccine/test.json",
    ]
)

sample_path = find_path(
    [
        "data/sample_submission.csv",
        "input/sample_submission.csv",
        "sample_submission.csv",
        "../input/sample_submission.csv",
        "../input/stanford-covid-vaccine/sample_submission.csv",
    ]
)


def load_json(path):
    """Load a JSON file that may be a single array or line‑delimited objects."""
    with open(path, "r") as f:
        raw = f.read().strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            objs = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    objs.append(json.loads(line))
            return objs


train_data = load_json(train_path)
test_data = load_json(test_path)

nuc_to_int = {"A": 0, "C": 1, "G": 2, "U": 3}
loop_to_int = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
struct_to_int = {"(": 0, ")": 1, ".": 2}


def encode_features(sample, pos):
    """Return an enriched feature vector for a given sample and position."""
    seq_len = sample["seq_length"]
    pos_norm = pos / seq_len
    sin_pos = np.sin(2 * np.pi * pos_norm)
    cos_pos = np.cos(2 * np.pi * pos_norm)

    nuc_idx = nuc_to_int.get(sample["sequence"][pos], -1)
    nuc_onehot = [0, 0, 0, 0]
    if 0 <= nuc_idx < 4:
        nuc_onehot[nuc_idx] = 1

    loop_idx = loop_to_int.get(sample["predicted_loop_type"][pos], -1)
    loop_onehot = [0] * 7
    if 0 <= loop_idx < 7:
        loop_onehot[loop_idx] = 1

    struct_idx = struct_to_int.get(sample["structure"][pos], -1)
    struct_onehot = [0, 0, 0]
    if 0 <= struct_idx < 3:
        struct_onehot[struct_idx] = 1

    scored_norm = sample["seq_scored"] / seq_len
    pos_sq = pos_norm**2
    pos_over_scored = pos / sample["seq_scored"] if sample["seq_scored"] > 0 else 0.0
    pos_scored_inter = pos_norm * scored_norm

    return (
        [
            pos_norm,
            pos_sq,
            scored_norm,
            sin_pos,
            cos_pos,
            pos_over_scored,
            pos_scored_inter,
        ]
        + nuc_onehot
        + loop_onehot
        + struct_onehot
    )


feature_list = []
target_list = {
    "reactivity": [],
    "deg_Mg_pH10": [],
    "deg_pH10": [],
    "deg_Mg_50C": [],
    "deg_50C": [],
}
id_seqpos_list = []

for sample in train_data:
    scored = sample["seq_scored"]
    for pos in range(scored):
        feats = encode_features(sample, pos)
        feature_list.append(feats)
        id_seqpos_list.append(f"{sample['id']}_{pos}")
        for key in target_list:
            target_list[key].append(sample[key][pos])

X_train_raw = np.array(feature_list, dtype=np.float32)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_raw)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_poly)


def mcrmse_score(y_true_dict, y_pred_dict):
    """Mean columnwise RMSE over the three scored targets."""
    rmses = []
    for key in ["reactivity", "deg_Mg_pH10", "deg_pH10"]:
        diff = y_true_dict[key] - y_pred_dict[key]
        rmses.append(np.sqrt(np.mean(diff**2)))
    return np.mean(rmses)


alphas = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
best_alpha = alphas[0]
best_score = np.inf

indices = np.arange(X_train.shape[0])
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
X_tr = X_train[train_idx]
X_val = X_train[val_idx]

y_arrays = {k: np.array(v, dtype=np.float32) for k, v in target_list.items()}

for alpha in alphas:
    models_tmp = {}
    for key in target_list:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_tr, y_arrays[key][train_idx])
        models_tmp[key] = model

    preds_val = {k: m.predict(X_val) for k, m in models_tmp.items()}

    score = mcrmse_score(
        {
            "reactivity": y_arrays["reactivity"][val_idx],
            "deg_Mg_pH10": y_arrays["deg_Mg_pH10"][val_idx],
            "deg_pH10": y_arrays["deg_pH10"][val_idx],
        },
        {
            "reactivity": preds_val["reactivity"],
            "deg_Mg_pH10": preds_val["deg_Mg_pH10"],
            "deg_pH10": preds_val["deg_pH10"],
        },
    )
    if score < best_score:
        best_score = score
        best_alpha = alpha

models = {}
for target, y in target_list.items():
    y_arr = np.array(y, dtype=np.float32)
    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_train, y_arr)
    models[target] = model




## === cell 1
test_features = []
test_id_seqpos = []

for sample in test_data:
    seq_len = sample["seq_length"]
    for pos in range(seq_len):
        feats = encode_features(sample, pos)
        test_features.append(feats)
        test_id_seqpos.append(f"{sample['id']}_{pos}")

X_test_raw = np.array(test_features, dtype=np.float32)
X_test_poly = poly.transform(X_test_raw)  # same polynomial expansion
X_test = scaler.transform(X_test_poly)  # same scaling as training

preds = {}
for target, model in models.items():
    preds[target] = model.predict(X_test)




## === cell 2
sub_df = pd.read_csv(sample_path)  # ensures correct column order & ids
sub_df["reactivity"] = preds["reactivity"]
sub_df["deg_Mg_pH10"] = preds["deg_Mg_pH10"]
sub_df["deg_pH10"] = preds["deg_pH10"]
sub_df["deg_Mg_50C"] = preds["deg_Mg_50C"]
sub_df["deg_50C"] = preds["deg_50C"]




## === cell 3
output_path = "submission.csv"
sub_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
