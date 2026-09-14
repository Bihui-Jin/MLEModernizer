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

0.3690968699324042

# 6. Current score

0.48389

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to read a non-existent file (`../input/worst-submission/ensemble52.csv`), so `df` is never created and all later cells crash. I replace that dependency by generating a valid baseline submission directly from the provided `sample_submission.csv`, keeping your existing scaling step (`/= 1.16`) intact to preserve the intended “calibration” logic. This run end-to-end in the given environment and always write a properly formatted `submission.csv` with the required columns. Since you currently have no valid score, this aims to yield a valid submission (and a reasonable baseline score) rather than tuning further.'
- What this solution (achieved 0.43136) has done: 'Your current submission is effectively a constant-zero baseline (with an extra `/= 1.16` scaling that does nothing), which explains the large gap to the target MCRMSE. To move the score down toward the target with minimal changes and without changing your “read sample_submission → write submission” core flow, I replace the constant predictions with per-position means computed from `train.json` (a standard, fast, leakage-free baseline). This keeps the same output schema/paths, still writes `submission.csv`, and should substantially reduce MCRMSE while staying simple and deterministic. I keep your numeric sanitization and column order checks intact.'
- What this solution (achieved 0.49673) has done: 'Your current baseline uses simple per-position means from all training rows, but it ignores the competition’s provided quality filter (`SN_filter`) and the different noise levels across samples; both can be incorporated without changing the overall “compute position statistics → fill sample_submission → write CSV” logic. I keep the same per-position mean approach, but compute *weighted* per-position means using inverse-variance weights derived from the provided `*_error_*` arrays (a standard way to reduce RMSE under heteroskedastic noise) and restrict to `SN_filter==1` rows to better match the test distribution. This is a minimal, fast change (pure numpy) that should reduce MCRMSE from 0.43136 toward your 0.3691 target without altering submission format or paths. The rest of your pipeline (including the `/= 1.16` calibration and numeric sanitization) is left intact for stability.'
- What this solution (achieved 0.48057) has done: 'We need to reduce your MCRMSE from 0.49673 toward the target 0.3691 (lower is better), but keep the same “per-position aggregated statistics → fill sample_submission → write submission.csv” core logic. The smallest safe improvement is to change the weighting scheme to better reflect the evaluation metric: compute a *weighted per-position mean across sequences using the provided errors*, and then optionally apply a light shrinkage toward the (unweighted) mean to reduce sensitivity to noisy weight extremes. I also make the “tail (unscored) positions” fill less arbitrary by using the per-target global mean rather than “last scored position”, which can otherwise inject positional bias into unscored rows. Everything else (paths, column order, `/= 1.16` calibration, numeric sanitization, output file) stays intact.'
- What this solution (achieved 0.4415) has done: 'Your current logic is already a fast “per-position aggregation → fill sample_submission → write submission.csv” baseline, but it’s underperforming likely because (a) the inverse-variance weights can be too peaky at some positions and (b) the fixed `/= 1.16` scaling may now be miscalibrated for this improved baseline. To move MCRMSE down toward the 0.3691 target with minimal risk and without changing the overall approach, I keep the same weighted-per-position mean core, but add a tiny, deterministic stabilizer: a per-position weight floor based on a small fraction of total weight (reduces domination by a few sequences) and a very small additional shrinkage toward the global mean at each position. I also replace the fixed 1.16 scaling with a data-derived global scalar computed from training (closed-form least-squares fit), which aligns prediction scale to the metric without changing the prediction family. Submission format, paths, and runtime remain the same and it still always writes a valid `submission.csv`.'
- What this solution (achieved 0.44081) has done: 'Your current gap to the target is 0.4415 − 0.3691 ≈ 0.0724 (lower is better), so we should improve modestly without changing the “per-position aggregated statistics → fill sample_submission → write CSV” core flow. The biggest likely issue is that the global scale fitting is currently applied with the wrong direction (you compute `scale` for `y ≈ scale * yhat` but then divide by `1/scale`, effectively multiplying by `scale`), which can miscalibrate predictions and worsen MCRMSE. I fix the scaling application to use `df *= scale` (or equivalently `df /= 1/scale`) consistent with the fitted objective, and I compute the scale only on the three scored targets to better align with the metric while keeping the same closed-form global-scalar approach. Everything else (paths, weighted per-position means, shrinkage, sanitization, and submission writing) stays the same.'
- What this solution (achieved 0.44081) has done: 'We need to move your (lower-is-better) MCRMSE from 0.44081 down toward the 0.36910 target, so we should improve the baseline slightly without changing the overall “per-position aggregated statistics → fill sample_submission → global scalar calibration → write submission.csv” flow. The smallest high-impact fix is to align the aggregation and the calibration to the metric: compute per-position means and the global scale using only the three scored targets, while still outputting all five required columns by filling the two unscored targets with the same stable per-position mean approach (no extra calibration). Additionally, we should compute the fitted scale against the *same* pre-scaled predictions that go into `df` (after shrinkage), ensuring the scale fit and application are fully consistent. These are minimal edits (no new model, no new features) and should reduce the gap toward the target.'
- What this solution (achieved 0.48508) has done: 'Your current baseline is already structurally sound (per-position aggregation → fill sample_submission → global scalar calibration → write CSV), but it’s likely underperforming because the scale fit is being driven by many noisy training points that shouldn’t count equally under the competition’s heteroskedastic measurement errors. I keep the exact same prediction family and pipeline, but change the *scale fitting step only* to a weighted least-squares global scalar using the provided per-position error arrays for the three scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C). This is minimal, deterministic, and directly aligned with the metric (reduces error where labels are more reliable) without changing the model/feature logic. Everything else (SN_filter usage, per-position means with shrinkage, filling unscored positions, submission schema/path) stays intact.'
- What this solution (achieved 0.48389) has done: 'We need to move your MCRMSE down (lower is better) from 0.48508 toward 0.36910, so we should make a small, low-risk improvement without changing the overall “per-position aggregated statistics → fill sample_submission → global scalar calibration → write CSV” pipeline. The most likely regression here is that the global scale is being fitted with ultra-peaky inverse-variance weights, which can overfit the scale to a few positions/samples with tiny reported errors and hurt generalization. I keep your exact weighted least-squares scalar approach but (1) compute the scale weights from errors with the same clipping/flooring you already use in aggregation, and (2) additionally cap the scale-fit weights to an upper quantile to prevent a small fraction of points dominating. Everything else (SN_filter usage, shrinkage, tail fill, sanitization, schema, and writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
SAMPLE_PATH_CANDIDATES = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
]
TRAIN_PATH_CANDIDATES = [
    "/kaggle/input/train.json",
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
]

sample_path = None
for p in SAMPLE_PATH_CANDIDATES:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations. "
        f"Tried: {SAMPLE_PATH_CANDIDATES}"
    )

train_path = None
for p in TRAIN_PATH_CANDIDATES:
    if os.path.exists(p):
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError(
        "Could not find train.json in expected locations. "
        f"Tried: {TRAIN_PATH_CANDIDATES}"
    )

df = pd.read_csv(sample_path)

required_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"sample_submission is missing required columns: {missing}")

target_cols = required_cols[1:]
train_targets = target_cols

train_df = pd.read_json(train_path, lines=True)

if "SN_filter" in train_df.columns:
    train_df_f = train_df.loc[train_df["SN_filter"] == 1].reset_index(drop=True)
    if len(train_df_f) == 0:
        train_df_f = train_df
else:
    train_df_f = train_df

seq_scored = int(train_df_f["seq_scored"].iloc[0])

err_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

eps = 1e-6
w_clip_hi = 1e4  # prevents a few ultra-small errors from dominating a position
w_floor_frac = 0.02  # small stabilizer against weight spikes (keeps same core logic)

shrink_alpha = 0.15
global_shrink = 0.03  # very small

pos_means = {}
global_means = {}

train_pred_for_scale = {}
train_y_for_scale = {}
train_w_for_scale = {}

scored_targets = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

for col in train_targets:
    y = np.vstack(train_df_f[col].values).astype(np.float32)  # (n_samples, seq_scored)
    mu_simple = y.mean(axis=0)
    global_mu = float(mu_simple.mean())
    global_means[col] = global_mu

    err_col = err_map.get(col, None)
    if err_col is not None and err_col in train_df_f.columns:
        e = np.vstack(train_df_f[err_col].values).astype(np.float32)
        w = 1.0 / np.square(np.maximum(e, eps))
        w = np.minimum(w, w_clip_hi).astype(np.float32)

        w_mean_pos = w.mean(axis=0, keepdims=True)
        w_floor = (w_floor_frac * w_mean_pos).astype(np.float32)
        w = np.maximum(w, w_floor)

        mu_w = (w * y).sum(axis=0) / (w.sum(axis=0) + eps)
        m = (1.0 - shrink_alpha) * mu_w + shrink_alpha * mu_simple
    else:
        w = None
        m = mu_simple

    m = (1.0 - global_shrink) * m + global_shrink * np.float32(global_mu)
    pos_means[col] = m.astype(np.float32)

    if col in scored_targets:
        train_pred_for_scale[col] = np.tile(pos_means[col][None, :], (y.shape[0], 1))
        train_y_for_scale[col] = y
        if w is None:
            train_w_for_scale[col] = np.ones_like(y, dtype=np.float32)
        else:
            train_w_for_scale[col] = w.astype(np.float32)

seqpos = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).values
for col in train_targets:
    m = pos_means[col]
    preds = np.zeros(len(df), dtype=np.float32)
    in_scored = seqpos < len(m)
    preds[in_scored] = m[seqpos[in_scored]]
    preds[~in_scored] = np.float32(global_means[col])
    df[col] = preds

df.head()



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:]



## === cell 3
scale_weight_qcap = 0.995  # tiny robustness; should improve generalization and move MCRMSE down toward target

num = 0.0
den = 0.0
for col in scored_targets:
    yp = train_pred_for_scale[col].ravel().astype(np.float64)
    yt = train_y_for_scale[col].ravel().astype(np.float64)

    ww = train_w_for_scale[col].ravel().astype(np.float64)

    if scale_weight_qcap is not None and 0.0 < scale_weight_qcap < 1.0:
        cap = float(np.quantile(ww, scale_weight_qcap))
        if cap > 0:
            ww = np.minimum(ww, cap)

    num += float(np.dot(ww * yp, yt))
    den += float(np.dot(ww * yp, yp))

scale = num / (den + 1e-12)  # y_hat_scaled = scale * y_hat

for col in scored_targets:
    df[col] = (df[col].astype(np.float32) * np.float32(scale)).astype(np.float32)

print(
    "Fitted global scale (weighted, robust-capped) =",
    scale,
    " -> applying to scored targets only",
)



## === cell 4
for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0.0, inplace=True)

df.describe()



## === cell 5
df.to_csv("submission.csv", index=False)

assert df.shape[1] == 6, "Submission must have 6 columns"
assert (
    df.columns.tolist() == required_cols
), f"Column order mismatch: {df.columns.tolist()}"
print("Wrote submission.csv", df.shape)
print(df.head())
