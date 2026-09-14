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
tqdm==4.67.1

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

0.37958

# 6. Current score

0.42262

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42231) has done: 'The fix corrects how training labels are assembled: the original code created an array with shape (n, 5, 68) causing a transposed prediction matrix and a DataFrame shape mismatch. By stacking each target column’s per‑position vectors and transposing to (n, 68, 5), the downstream logic now matches the expected dimensions, allowing the submission file to be generated without errors. No other logic is altered, preserving the original model approach.'
- What this solution (achieved 0.4223) has done: 'I keep the original preprocessing and data handling but add a tiny validation‑based calibration step. After splitting the filtered training set, I compute per‑target linear scaling factors (slope + intercept) that best map the simple mean‑baseline predictions to the true values on the validation split. These factors are then applied to the test‑set mean predictions, giving a modest but systematic improvement toward the target score while leaving the core “mean‑baseline” logic unchanged.'
- What this solution (achieved 0.42278) has done: 'I keep the overall mean‑baseline approach but make the calibration more granular: instead of a single slope + intercept per target, I fit a separate linear scaling for each of the 68 positions. This adds per‑position a and b coefficients (still a simple linear adjustment), which is a minimal change that often reduces the validation error and thus moves the public score closer to the target. The rest of the pipeline—including preprocessing, padding, and CSV generation—remains unchanged.'
- What this solution (achieved 0.42278) has done: 'I keep the original mean‑baseline and per‑position linear calibration, but add a lightweight blending step. Using the validation split, I compute the uncalibrated and calibrated predictions, then search a small set of blend weights w ∈ [0,1] to find the one that gives the lowest MCRMSE on validation. The same optimal w is then applied to the test predictions, creating a modest but targeted improvement toward the target score without altering the core model logic.'
- What this solution (achieved 0.42278) has done: 'This update adds a tiny ridge regularisation when fitting the per‑position linear calibrations (reducing over‑fitting), refines the blending weight with a finer search around the best coarse value, and applies a simple bias‑correction derived from the validation set to the final test predictions. These minimal adjustments keep the original mean‑baseline approach intact while nudging the validation MCRMSE closer to the target score.'
- What this solution (achieved 0.42267) has done: 'I strengthen regularisation on the per‑position linear calibrations (increase ridge α) and then shrink the fitted slopes (a) toward 1 and intercepts (b) toward 0. This reduces over‑fitting on the validation split, which should lower the MCRMSE on the leaderboard and move the score closer to the target while keeping the original mean‑baseline pipeline unchanged.'
- What this solution (achieved 0.42262) has done: 'I replace the simple un‑weighted average used for the baseline predictions with a signal‑to‑noise weighted mean. Using the provided “signal_to_noise” column as a weight should give a baseline that is closer to the true distribution, which modestly reduces the validation MCRMSE and moves the score toward the target while keeping all other logic unchanged.'

# 9. Code solution

## === cell 0
import warnings, os, sys

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, json, gc, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train_path = "/kaggle/input/stanford-covid-vaccine/train.json"
test_path = "/kaggle/input/stanford-covid-vaccine/test.json"
sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = [
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]  # correct target columns

token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}




## === cell 1
def preprocess_inputs(df):
    """
    Convert the three string columns into integer arrays of shape (n_samples, seq_len, 3).
    """
    seq = df["sequence"].apply(lambda s: [token2int[ch] for ch in s]).tolist()
    struct = df["structure"].apply(lambda s: [token2int[ch] for ch in s]).tolist()
    loop = (
        df["predicted_loop_type"].apply(lambda s: [token2int[ch] for ch in s]).tolist()
    )
    seq_arr = np.array(seq, dtype=np.int32)
    struct_arr = np.array(struct, dtype=np.int32)
    loop_arr = np.array(loop, dtype=np.int32)
    return np.stack([seq_arr, struct_arr, loop_arr], axis=2)  # (n, seq_len, 3)




## === cell 2
train_clean = train[train.signal_to_noise > 1].reset_index(drop=True)

train_split, val_split = train_test_split(
    train_clean, test_size=0.2, random_state=42, shuffle=True
)


def make_labels(df):
    label_arrays = [np.vstack(df[col].values) for col in target_cols]  # each (n, 68)
    return np.stack(label_arrays, axis=2)  # (n, 68, 5)


train_labels = make_labels(train_split)
val_labels = make_labels(val_split)

train_inputs = preprocess_inputs(train_split)




## === cell 3
weights = train_split["signal_to_noise"].values.astype(np.float32)  # (n,)
weights_sum = np.maximum(weights.sum(), 1e-8)  # avoid division by zero
weighted_sum = np.tensordot(weights, train_labels, axes=([0], [0]))  # (68, 5)
mean_preds_train = weighted_sum / weights_sum  # (68, 5)

a_factors = np.zeros((len(target_cols), 68))
b_factors = np.zeros((len(target_cols), 68))

preds_val_flat = np.tile(
    mean_preds_train[None, :, :], (len(val_split), 1, 1)
)  # (n_val,68,5)

ridge_alpha = 1e-2

for t_idx, col in enumerate(target_cols):
    for pos in range(68):
        X = preds_val_flat[:, pos, t_idx][:, None]  # (n_val, 1)
        y = val_labels[:, pos, t_idx]  # (n_val,)
        A = np.hstack([X, np.ones_like(X)])  # (n_val, 2)

        AtA = A.T @ A + ridge_alpha * np.eye(2)
        coeffs = np.linalg.solve(AtA, A.T @ y)
        a_factors[t_idx, pos], b_factors[t_idx, pos] = coeffs

shrink = 0.9
a_factors = 1 + (a_factors - 1) * shrink
b_factors = b_factors * shrink




## === cell 4
test_inputs = preprocess_inputs(test)




## === cell 5
val_preds_cal = preds_val_flat * a_factors.T[None, :, :] + b_factors.T[None, :, :]


def mcrmse(y_true, y_pred):
    diff = y_true - y_pred
    rmse_per_target = np.sqrt(np.mean(diff**2, axis=(0, 1)))
    return rmse_per_target.mean()


best_w = 0.0
best_score = mcrmse(val_labels, preds_val_flat)

for w in np.arange(0.0, 1.01, 0.05):
    blended = w * val_preds_cal + (1 - w) * preds_val_flat
    score = mcrmse(val_labels, blended)
    if score < best_score:
        best_score = score
        best_w = w

fine_range = np.arange(max(0.0, best_w - 0.05), min(1.0, best_w + 0.05) + 1e-6, 0.01)
for w in fine_range:
    blended = w * val_preds_cal + (1 - w) * preds_val_flat
    score = mcrmse(val_labels, blended)
    if score < best_score:
        best_score = score
        best_w = w

blended_best = best_w * val_preds_cal + (1 - best_w) * preds_val_flat
bias = (val_labels - blended_best).mean(axis=(0, 1))  # shape (5,)

test_preds_uncal = np.tile(
    mean_preds_train[None, :, :], (len(test), 1, 1)
)  # (n_test, 68, 5)

a = a_factors.T[None, :, :]  # (1, 68, 5)
b = b_factors.T[None, :, :]  # (1, 68, 5)

test_preds_cal = test_preds_uncal * a + b
test_preds = best_w * test_preds_cal + (1 - best_w) * test_preds_uncal

test_preds += bias[None, None, :]




## === cell 6
rows = []
for idx, row in test.iterrows():
    uid = row["id"]
    preds = test_preds[idx]  # (68, 5)
    full_len = row["seq_length"]
    if preds.shape[0] < full_len:
        pad_len = full_len - preds.shape[0]
        pad = np.tile(preds[-1:], (pad_len, 1))
        preds_full = np.vstack([preds, pad])
    else:
        preds_full = preds[:full_len]

    df_pred = pd.DataFrame(preds_full, columns=target_cols)
    df_pred["id_seqpos"] = [f"{uid}_{i}" for i in range(full_len)]
    rows.append(df_pred)

submission_df = pd.concat(rows, ignore_index=True)

submission = sample_sub[["id_seqpos"]].merge(submission_df, on="id_seqpos", how="left")
submission[target_cols] = submission[target_cols].fillna(0)




## === cell 7
submission.to_csv("submission.csv", index=False)
