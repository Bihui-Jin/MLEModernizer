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

0.35405

# 6. Current score

0.4544

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42492) has done: 'The current notebook fails immediately because it tries to read a non-existent `../input/worst-submission/ensemble52.csv`, so no `df` is created and all later cells crash. To make it run end-to-end in this environment and still keep the intended “baseline + small calibration” core idea, I switch the input to the provided `sample_submission.csv` and fill it with constant predictions derived from the training-set means (a legitimate, simple baseline). I also keep your existing calibration step (`/= 1.03`) but apply it safely only to numeric target columns. Finally, I write a valid `submission.csv` with the exact required columns and row order from the sample submission.'
- What this solution (achieved 0.4241) has done: 'Your current solution is a per-position mean baseline with a fixed global scaling (`/ 1.03`), which is likely slightly miscalibrated relative to the leaderboard optimum for this baseline. To move the score down toward the 0.35405 target (lower is better) while preserving the same core approach, I only change the single calibration factor and make it learnable from the training set via a tiny holdout split (still the same mean-by-position predictions, just scaled). Concretely: build the same position-mean predictor on a training subset, evaluate MCRMSE on the remaining subset over the 3 scored targets, grid-search one scalar multiplier `alpha` applied to all targets, then retrain position means on full training and apply the chosen `alpha` for test predictions. This keeps the model logic identical (position means + one scalar calibration) but should reduce your error versus a hardcoded divisor.'
- What this solution (achieved 0.42165) has done: 'Your current approach is already a position-wise mean baseline with a single scalar calibration; the easiest way to move the score down toward the 0.35405 target (lower is better) without changing the core logic is to (1) compute the position means using only the high-quality `SN_filter==1` training rows (still the same estimator, just less noisy data), and (2) pick the scalar `alpha` using a validation split that is also restricted to `SN_filter==1` so the calibration matches the scored distribution better. I keep the same prediction construction (position means, pad to 107, broadcast by `seqpos`) and the same metric computation (MCRMSE over the 3 scored targets). This should typically improve the baseline score while staying within Kaggle constraints and preserving evaluation semantics. The script still write a valid `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.42166) has done: 'I keep your exact “position-wise mean baseline + single scalar calibration” core logic, but make the calibration match the evaluation more closely by selecting `alpha` using only the 3 scored targets (as you already do) **and only over the scored positions**, while ensuring the holdout split is deterministic and respects the `seq_scored` length from the data rather than assuming consistency. I also slightly tighten the alpha search to a narrower band around 1.0 with finer resolution, which is a minimal change that often yields a small but reliable MCRMSE improvement for this baseline without changing the model. Finally, I keep submission formatting identical and still write `submission.csv`.'
- What this solution (achieved 0.42166) has done: 'To move your score down toward the 0.35405 target (lower is better) while keeping the same “position-wise mean baseline + single scalar calibration” core logic, I only make the calibration step more robust and better matched to the evaluation distribution. Concretely, I pick `alpha` via deterministic 5-fold CV (still a single global scalar) on `SN_filter==1` rows and only over the scored 68 positions and 3 scored targets, then refit position means on all HQ rows and apply the CV-chosen `alpha` to test. This reduces variance from a single holdout split (a common reason baselines plateau around ~0.42) without changing the model family or adding complexity that would alter semantics. Submission formatting and row alignment remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.4544) has done: 'Your current position-wise mean baseline is already stable; to move the MCRMSE down toward the 0.35405 target (lower is better) with minimal logic change, I keep the same estimator but reduce avoidable noise and mismatch in calibration. Specifically, I (1) compute position means as a **signal-to-noise–weighted mean** using the provided `*_error_*` arrays (still a per-position mean, just weighted), and (2) keep the same single-scalar calibration `alpha` but pick it via the same deterministic 5-fold CV using the weighted means. This typically improves this baseline without changing the “mean-by-position + global scaling” core idea, and still writes an identical-format `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
from pathlib import Path



## === cell 1
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/stanford-covid-vaccine"),
    Path("/kaggle/data/stanford-covid-vaccine"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]

data_dir = None
for d in DATA_DIR_CANDIDATES:
    if (
        (d / "sample_submission.csv").exists()
        and (d / "train.json").exists()
        and (d / "test.json").exists()
    ):
        data_dir = d
        break

if data_dir is None:
    raise FileNotFoundError(
        "Could not locate dataset files. Expected sample_submission.csv, train.json, test.json under /kaggle/input or /kaggle/data."
    )

sample_path = data_dir / "sample_submission.csv"
train_path = data_dir / "train.json"
test_path = data_dir / "test.json"

sample_sub = pd.read_csv(sample_path)
train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)

(sample_sub.shape, train.shape, test.shape)



## === cell 2
TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

ERROR_COL = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def mcrmse(y_true, y_pred, eps=1e-12):
    mse = np.mean((y_true - y_pred) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse + eps)
    return float(np.mean(rmse))


def build_pos_means(train_df, targets, use_weighted=True, weight_clip=(1e-3, 1e3)):
    """
    Position-wise mean baseline.
    If use_weighted, compute inverse-variance weighted mean using the provided error arrays:
        w = 1 / (err^2)
    This preserves the same model family (per-position constant predictor), just denoises.
    """
    means = {}
    for t in targets:
        y = np.vstack(train_df[t].values).astype(np.float64)

        if use_weighted and (ERROR_COL.get(t) in train_df.columns):
            err = np.vstack(train_df[ERROR_COL[t]].values).astype(np.float64)
            w = 1.0 / (err * err + 1e-12)
            if weight_clip is not None:
                w = np.clip(w, weight_clip[0], weight_clip[1])
            num = np.sum(w * y, axis=0)
            den = np.sum(w, axis=0) + 1e-12
            means[t] = num / den
        else:
            means[t] = y.mean(axis=0)

    return means


def make_pred_from_posmeans(pos_means, seq_length, targets):
    full_means = {}
    for t in targets:
        m68 = np.asarray(pos_means[t], dtype=np.float64)
        if len(m68) == seq_length:
            full_means[t] = m68
        else:
            pad_val = float(m68[-1])
            full_means[t] = np.concatenate(
                [m68, np.full(seq_length - len(m68), pad_val, dtype=np.float64)]
            )
    return full_means


train_hq = train.loc[train["SN_filter"] == 1].reset_index(drop=True)
if len(train_hq) < 50:
    train_hq = train.reset_index(drop=True)

seq_scored = int(train_hq["seq_scored"].mode().iloc[0])

rng = np.random.RandomState(0)
idx = np.arange(len(train_hq), dtype=np.int64)
rng.shuffle(idx)

K = 5
fold_sizes = np.full(K, len(idx) // K, dtype=np.int64)
fold_sizes[: len(idx) % K] += 1

fold_starts = np.concatenate([[0], np.cumsum(fold_sizes)[:-1]])
fold_ends = fold_starts + fold_sizes

alphas = np.linspace(0.90, 1.10, 401)
cv_scores = np.zeros_like(alphas, dtype=np.float64)

for k in range(K):
    va_idx = idx[fold_starts[k] : fold_ends[k]]
    tr_idx = np.concatenate([idx[: fold_starts[k]], idx[fold_ends[k] :]])

    train_tr = train_hq.iloc[tr_idx].reset_index(drop=True)
    train_va = train_hq.iloc[va_idx].reset_index(drop=True)

    pos_means_tr = build_pos_means(train_tr, TARGETS, use_weighted=True)

    pred_va = np.stack(
        [
            np.tile(pos_means_tr[t][None, :seq_scored], (len(train_va), 1))
            for t in SCORED_TARGETS
        ],
        axis=-1,
    )
    true_va = np.stack(
        [np.vstack(train_va[t].values)[:, :seq_scored] for t in SCORED_TARGETS], axis=-1
    )

    for i, a in enumerate(alphas):
        cv_scores[i] += mcrmse(true_va, pred_va * a)

cv_scores /= K
best_i = int(np.argmin(cv_scores))
best_alpha = float(alphas[best_i])

best_alpha, float(cv_scores[best_i])



## === cell 3
pos_means_full = build_pos_means(train_hq, TARGETS, use_weighted=True)

seq_length = int(test["seq_length"].mode().iloc[0])
full_means = make_pred_from_posmeans(pos_means_full, seq_length, TARGETS)

df = sample_sub.copy()
seqpos = df["id_seqpos"].str.split("_").str[-1].astype(int).values

for t in TARGETS:
    df[t] = full_means[t][seqpos]

df.head()



## === cell 4
df[TARGETS] = (df[TARGETS] * best_alpha).astype(np.float32)

assert df["id_seqpos"].isna().sum() == 0
assert df[TARGETS].isna().sum().sum() == 0



## === cell 5
required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
if list(df.columns) != required_cols:
    df = df[required_cols]

assert df.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
df.head()



## === cell 6
df.to_csv("submission.csv", index=False)
print("HQ train rows used (SN_filter==1, with fallback):", len(train_hq))
print("seq_scored used:", seq_scored)
print("Chosen calibration alpha (5-fold CV):", best_alpha)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
