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

0.3518779703353691

# 6. Current score

0.48104

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'I remove the dependency on the missing `../input/worst-submission/ensemble52.csv` and instead build a valid submission directly from the provided `sample_submission.csv` so the notebook runs end-to-end. To preserve the intent of “use a baseline submission then patch a specific id”, I keep your existing patch line but apply it to the sample submission template. I also keep reading `test.json` (though it isn’t needed for the submission) to minimize changes and to ensure paths are correct in this environment. Finally, I write a proper `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.42418) has done: 'Your current submission is essentially the sample submission with one id’s `reactivity` forced to 0, which is why the score is far from the target. To move the score closer to the target with minimal changes and without changing the “no model” core approach, I replace the single-id patch with a simple, train-derived per-position mean baseline (computed from `train.json`) for all 5 targets, and then repeat those means for every test `id_seqpos`. This is still a lightweight baseline (no ML, no new dependencies) but substantially reduce MCRMSE versus zeros/sample defaults, moving you toward the 0.3519 target. I keep the same I/O paths and still generate `submission.csv` with the exact required schema and row count.'
- What this solution (achieved 0.42173) has done: 'We keep your “no model” baseline but make one small change that typically improves MCRMSE: compute position-wise means only on high-quality training rows (`SN_filter==1`) and with a light cap on extreme values by clipping to the 1st–99th percentile per position, which reduces noise/outlier impact while preserving the same mean-per-position core logic. We also keep the same handling for unscored positions (global mean), and ensure the submission schema and row alignment remain identical. These changes are directly aimed at reducing error toward your target (lower is better) without altering the overall approach or adding dependencies. The script still run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.4437) has done: 'Your current baseline is already in the right family (position-wise means), but it’s still a bit noisy because it uses a simple mean after light clipping. To move the MCRMSE down toward the 0.3519 target with minimal semantic change, I keep the exact same “train-derived per-position constant prediction” core logic, but replace the mean with a per-position median (more robust than mean) while keeping the same SN_filter==1 selection and 1–99% clipping. I also compute the “unscored positions” fallback as the median of the per-position values (instead of mean) for consistency. Everything else (paths, column order, row alignment, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.42877) has done: 'We keep your exact “train-derived per-position constant prediction” approach, but make it slightly closer to the competition metric by optimizing only the scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) and then using simple linear mappings to derive the two unscored targets from those predictions. This is a minimal semantic change (still no ML model; still per-position constants), but it typically reduces noise for the scored columns by not over-regularizing them via the unscored columns, helping move MCRMSE down toward your 0.3519 target. We also switch the robust statistic from median to a per-position trimmed mean (after the same 1–99% clipping) which often performs better than a strict median on this dataset while staying in the same “robust central tendency” family. Submission format/paths stay the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.42173) has done: 'We keep your exact “per-position constant prediction + linear mapping for unscored targets” core logic, but make it better aligned with the MCRMSE objective by using an RMSE-optimal center: per-position mean (still after the same 1–99% clipping) instead of a trimmed mean. We also compute the linear mapping (deg_pH10 from deg_Mg_pH10, deg_50C from deg_Mg_50C) with a more stable ridge-style variance floor, which reduces noisy extreme slopes without changing the overall linear-per-position mapping approach. Finally, we ensure numerical stability by filling any remaining NaNs with the corresponding global fallback per column (instead of always 0.0), which avoids unnecessary error inflation. These are minimal, local changes and should move the score downward (better) from 0.42877 toward your 0.35188 target.'
- What this solution (achieved 0.4825) has done: 'To move your MCRMSE down toward the 0.3519 target (lower is better) with minimal semantic change, I keep the exact same “train-derived per-position constant prediction + per-position linear mapping for unscored columns” approach. The only modeling change is to compute those per-position constants using the RMSE-optimal *weighted mean* (weights = 1 / error² from the provided `*_error_*` columns) on `SN_filter==1` rows, which directly aligns with the metric and typically reduces noise. I also apply the same clipping you already use, but do it on both values and weights (by clipping values only) to keep robustness while preserving the core logic. Submission schema, row alignment, paths, and runtime remain the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.48104) has done: 'Your current weighted-mean baseline likely regressed because (a) extreme/invalid error values can dominate weights and (b) the linear mapping uses only Y-errors, not accounting for X-noise, which can amplify variance. I keep the exact same core approach (per-position constant predictions for scored targets + per-position linear mapping for unscored targets) but make two minimal, metric-aligned stability fixes: clip/guard the error-derived weights and use a slightly stronger (but still tiny) variance floor in the per-position regression. These changes are directly aimed at reducing MCRMSE (lower is better) back toward your target without changing the modeling family or adding dependencies, and the script still write a valid `submission.csv` with the correct schema/row count.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)

df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

expected_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"sample_submission.csv is missing columns: {missing}")
df = df[expected_cols]



## === cell 2
sequences = list(
    set(df_test[df_test.seq_length != 130].id)
)  # sequences from private set



## === cell 3
sequences[-10:]



## === cell 4
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
unscored_cols = ["deg_pH10", "deg_50C"]
all_target_cols = scored_cols + unscored_cols

seq_scored = int(df_train["seq_scored"].iloc[0])

if "SN_filter" in df_train.columns:
    df_train_use = df_train[df_train["SN_filter"] == 1].copy()
    if len(df_train_use) == 0:
        df_train_use = df_train
else:
    df_train_use = df_train

target_to_err = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def _safe_weights_from_err(err_2d: np.ndarray, eps=1e-6, w_max=1e4):
    e = err_2d.astype(np.float64)
    e = np.where(np.isfinite(e), e, np.nan)
    e = np.where(e > 0.0, e, np.nan)
    w = 1.0 / (e * e + eps)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, w_max)
    return w


def _weighted_pos_center(
    values_2d: np.ndarray, err_2d: np.ndarray, q_lo=0.01, q_hi=0.99, eps=1e-6
):
    """Compute per-position weighted mean after value clipping; weights = 1/(err^2+eps)."""
    v = values_2d.astype(np.float64)

    lo = np.quantile(v, q_lo, axis=0)
    hi = np.quantile(v, q_hi, axis=0)
    vc = np.clip(v, lo, hi)

    w = _safe_weights_from_err(err_2d, eps=eps, w_max=1e4)
    wsum = np.sum(w, axis=0)
    wsum = np.where(wsum <= 0.0, 1.0, wsum)
    return np.sum(w * vc, axis=0) / wsum, float(np.sum(w * vc) / max(np.sum(w), 1.0))


pos_stats = {}
global_stats = {}

for c in scored_cols:
    arr = np.vstack(df_train_use[c].values).astype(np.float64)  # (n_train, seq_scored)
    err_col = target_to_err.get(c, None)
    if err_col is not None and err_col in df_train_use.columns:
        err = np.vstack(df_train_use[err_col].values).astype(np.float64)
        pos_stats[c], global_stats[c] = _weighted_pos_center(
            arr, err, q_lo=0.01, q_hi=0.99, eps=1e-6
        )
    else:
        lo = np.quantile(arr, 0.01, axis=0)
        hi = np.quantile(arr, 0.99, axis=0)
        arr = np.clip(arr, lo, hi)
        pos_stats[c] = np.mean(arr, axis=0).astype(np.float64)
        global_stats[c] = float(np.mean(pos_stats[c]))

lin_map = {}  # unscored_col -> (source_scored_col, a_vec, b_vec, global_fallback)

var_floor = 5e-4  # was 1e-4

for uc, sc in [("deg_pH10", "deg_Mg_pH10"), ("deg_50C", "deg_Mg_50C")]:
    X = np.vstack(df_train_use[sc].values).astype(np.float64)
    Y = np.vstack(df_train_use[uc].values).astype(np.float64)

    X_lo = np.quantile(X, 0.01, axis=0)
    X_hi = np.quantile(X, 0.99, axis=0)
    Y_lo = np.quantile(Y, 0.01, axis=0)
    Y_hi = np.quantile(Y, 0.99, axis=0)
    Xc = np.clip(X, X_lo, X_hi)
    Yc = np.clip(Y, Y_lo, Y_hi)

    err_col_y = target_to_err.get(uc, None)
    if err_col_y is not None and err_col_y in df_train_use.columns:
        Ey = np.vstack(df_train_use[err_col_y].values).astype(np.float64)
        Wy = _safe_weights_from_err(Ey, eps=1e-6, w_max=1e4)
        zero_w_cols = np.sum(Wy, axis=0) <= 0.0
        if np.any(zero_w_cols):
            Wy[:, zero_w_cols] = 1.0
    else:
        Wy = np.ones_like(Yc, dtype=np.float64)

    a = np.empty(seq_scored, dtype=np.float64)
    b = np.empty(seq_scored, dtype=np.float64)

    for j in range(seq_scored):
        x = Xc[:, j]
        y = Yc[:, j]
        w = Wy[:, j]

        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0.0:
            w = np.ones_like(y, dtype=np.float64)
            wsum = float(len(y))

        mx = float(np.sum(w * x) / wsum)
        my = float(np.sum(w * y) / wsum)

        vx = float(np.sum(w * (x - mx) ** 2) / wsum)
        cov = float(np.sum(w * (x - mx) * (y - my)) / wsum)

        a[j] = cov / (vx + var_floor)
        b[j] = my - a[j] * mx

    glob_denom = float(np.sum(Wy))
    glob = float(np.sum(Wy * Yc) / (glob_denom if glob_denom > 0.0 else 1.0))
    lin_map[uc] = (sc, a, b, glob)

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
mask_scored = seqpos < seq_scored

for c in scored_cols:
    preds = np.empty(len(df), dtype=np.float64)
    preds[mask_scored] = pos_stats[c][seqpos[mask_scored]]
    preds[~mask_scored] = global_stats[c]
    df[c] = preds

for uc in unscored_cols:
    sc, a_vec, b_vec, glob = lin_map[uc]
    preds = np.empty(len(df), dtype=np.float64)
    sp = seqpos[mask_scored]
    preds[mask_scored] = (
        a_vec[sp] * df.loc[mask_scored, sc].to_numpy(dtype=np.float64) + b_vec[sp]
    )
    preds[~mask_scored] = glob
    df[uc] = preds



## === cell 5
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
df[pred_cols] = df[pred_cols].astype(float)

fill_values = {
    "reactivity": global_stats["reactivity"],
    "deg_Mg_pH10": global_stats["deg_Mg_pH10"],
    "deg_Mg_50C": global_stats["deg_Mg_50C"],
    "deg_pH10": lin_map["deg_pH10"][3],
    "deg_50C": lin_map["deg_50C"][3],
}
df[pred_cols] = df[pred_cols].fillna(value=fill_values)

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
