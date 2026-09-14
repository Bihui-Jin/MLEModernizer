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

0.3519866169312086

# 6. Current score

0.43808

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails because it tries to read a non-existent `../input/worst-submission/ensemble52.csv`, so `df` is never created and all later cells crash. To make the pipeline run end-to-end and produce a valid submission, I replace that missing input with the competition-provided `sample_submission.csv` from the available dataset path. I also keep the existing downstream logic intact (including the row override for a specific `id_...` prefix) and add a couple of lightweight checks to guarantee the submission has the required columns and no NaNs before writing `submission.csv`.'
- What this solution (achieved 0.63824) has done: 'Your current code is essentially submitting the sample submission (plus one small override), which explains the weak score; to move toward the target (lower is better), we need real predictions while keeping the overall pipeline simple and fast. The minimal, metric-aligned improvement is to train a basic per-position regressor from the provided `train.json` using only lightweight, local features from `sequence`, `structure`, `predicted_loop_type`, and position—then predict for every test `id_seqpos` and write the same submission schema. This preserves the general “read base submission -> fill predictions -> write submission.csv” flow, but replaces constant/zero values with learned estimates. We also keep your existing special-case override intact to avoid unintended changes in behavior.'
- What this solution (achieved 0.63824) has done: 'I fix the runtime error by avoiding the Ridge solver path that calls SciPy’s `cg(tol=...)`, which is incompatible with the environment’s SciPy; setting `solver="svd"` keeps the same Ridge model but makes it robust. I also make the file existence checks compatible with newer pandas (using `os.path.exists`) so the train/test JSON discovery doesn’t fail silently. The rest of the pipeline stays the same: explode to per-position rows, fit a multi-output linear regressor on one-hot features, merge predictions into the sample submission, apply your special-case override, and write `submission.csv` with the required columns. These changes are execution-stability fixes and should also allow the learned model (vs. sample submission) to improve the score toward your lower-is-better target.'
- What this solution (achieved 0.63824) has done: 'I fix the crash by making Ridge compatible with the sparse one-hot output (the `svd` solver can’t fit an intercept on sparse matrices). The smallest stable change is to set `fit_intercept=False` while keeping `solver="svd"` so the rest of your pipeline (features, MultiOutputRegressor, training loop) stays the same and runs end-to-end. I also keep your existing post-merge fill/override behavior intact and add a tiny safety check that predictions merged correctly for all rows. This should both produce a valid `submission.csv` and improve score versus the near-sample-submission baseline, moving toward your lower-is-better target.'
- What this solution (achieved 0.42872) has done: 'I fix the runtime error by switching Ridge from the incompatible `solver="svd"` (which cannot handle the sparse matrix produced by `OneHotEncoder`) to a sparse-compatible solver, keeping the same model and training flow. To avoid solver/API issues in this environment, I use `solver="lsqr"` and remove the unsupported `random_state` argument for Ridge. Everything else (feature creation, per-position exploding, MultiOutputRegressor training, merge into sample submission, and the existing special-case override) stays the same, ensuring the pipeline runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.42647) has done: 'Your current model is learning from all training rows, including low-quality experiments that the competition notes as noisy; the smallest score-improving change (while keeping the same Ridge+one-hot core logic) is to train only on `SN_filter==1` rows to better match the test distribution. I also add a tiny, metric-aligned clipping of predictions to the standard competition range (non-negative), which typically reduces RMSE without changing the modeling approach. Finally, I remove the post-merge `fillna(0.0)` fallback by ensuring every `id_seqpos` is predicted (and assert this), so we don’t silently insert harmful zeros. These are minimal, stability-safe changes that should move your lower-is-better score toward the target.'
- What this solution (achieved 0.4622) has done: 'We keep your Ridge+one-hot per-position pipeline intact and only make small, metric-aligned adjustments to reduce the MCRMSE gap toward the lower-is-better target. The main change is to reweight training examples using the provided per-position measurement errors (via `sample_weight`), so noisier targets contribute less without changing the model class or training loop. We also avoid training on extreme/outlier target values by applying the competition’s standard `-0.5` floor filter at the per-position level (consistent with dataset notes), and we keep your existing SN_filter==1 restriction and special-case override. Finally, we tighten the post-processing clip to the common StanVac range [-0.5, +inf) (instead of [0, +inf)) to avoid biasing negatives upward, which typically improves RMSE.'
- What this solution (achieved 0.47049) has done: 'We keep your exact Ridge+one-hot per-position pipeline, but fix one post-processing issue that can hurt MCRMSE: the current hard `fillna(0.0)` can silently inject bad values if anything ever goes missing; instead we enforce full coverage and only fall back to a safer per-target mean computed from the training data (which is much less damaging than zeros). We also make the sample weighting more metric-aligned by using the *scored* targets only (the leaderboard only scores 3/5), while still training/predicting all 5 columns exactly as required. Finally, we keep your existing SN_filter==1 restriction, per-position -0.5 floor filter, clipping, and the special-case override unchanged.'
- What this solution (achieved 0.63824) has done: 'We keep your exact Ridge + one-hot per-position pipeline, but make two small, metric-aligned fixes that typically reduce MCRMSE without changing the modeling approach. First, we add a tiny “local context” feature set (prev/next base/structure/loop) while still using the same OneHotEncoder+Ridge model; this is a minimal extension of the existing feature extraction and helps capture nearest-neighbor effects in RNA. Second, we make sample weighting consistent with what Ridge/MultiOutputRegressor actually supports by passing weights correctly to each underlying estimator (so the intended error-based downweighting reliably takes effect). Everything else (SN_filter==1, -0.5 floor filter, clipping, merge logic, override, and submission writing) stays the same.'
- What this solution (achieved 0.44042) has done: 'I fix the crash in model fitting by passing `sample_weight` in the way `Pipeline` + `MultiOutputRegressor` actually supports (as `reg__sample_weight`), instead of the invalid `reg__estimator__sample_weight` that currently reaches `Ridge.fit()` as an unexpected kwarg. This keeps your exact Ridge+OneHotEncoder+MultiOutputRegressor pipeline and preserves the intended error-based downweighting behavior so training uses the weights (which should help move the lower-is-better score toward the target). I also add a small guard to ensure the computed `sample_weight` length matches `X_train` rows and is finite, so the run is stable and always produces a valid `submission.csv`. No other modeling logic, features, or post-processing be changed.'
- What this solution (achieved 0.44042) has done: 'We keep your exact Ridge + one-hot (with local context) pipeline and only make metric-aligned, minimal adjustments to reduce MCRMSE from 0.44042 toward the lower-is-better target 0.35199. The main change is to stop training on all five targets equally: instead, we train a second model only for the three scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C) using the same features/estimator, then use those predictions for the scored columns while keeping your existing 5-target model for the two unscored columns. Additionally, we replace the global clipping floor (-0.5) with per-target floors based on the observed training minimums (capped at -0.5), which is a small calibration tweak that typically reduces RMSE without changing core logic. Everything else (SN_filter==1, per-position explode, sample weighting, merge/fallback, and your special-case override) remains intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.43878) has done: 'We keep your exact Ridge+OneHotEncoder per-position setup and only make two small, metric-aligned adjustments to move MCRMSE down toward the 0.35199 target. First, we cap extreme sample weights (coming from very small reported errors) so a few positions don’t dominate the fit and hurt generalization; this preserves the same weighting idea but stabilizes it. Second, we apply a light shrinkage of predictions toward the training means for the three scored targets only, which often improves RMSE by reducing overconfident deviations without changing the model or features. Everything else (SN_filter filtering, explode logic, local context features, two-model combination, clipping, merge, override, and submission writing) remains intact.'
- What this solution (achieved 0.43808) has done: 'We keep your exact Ridge+OneHotEncoder + local-context per-position pipeline, but make two small, metric-aligned adjustments aimed at lowering MCRMSE from 0.43878 toward the 0.35199 target. First, we tune the amount of mean-shrinkage (a pure post-processing calibration) by selecting the best shrink value via a lightweight internal train/validation split on the already-exploded per-position rows, without changing the model or features. Second, we slightly adjust the sample-weight cap quantile (still the same weighting idea) by choosing between a couple of nearby caps using the same internal validation, which stabilizes generalization without altering core logic. Everything else (SN_filter filtering, per-position explode, two-model combination for scored columns, clipping, merge/fallback, your special-case override, and writing `submission.csv`) remains intact.'
- What this solution (achieved 0.43808) has done: 'Your score (0.43808) is still worse than the target (0.35199), so we should make a small, low-risk improvement that keeps your Ridge+OneHot per-position approach intact. The most likely issue is that the internal validation split is currently done per-position row, which leaks information across positions of the same `id` between train/val and tends to select suboptimal calibration/weight caps; we switch that split to be **grouped by `id`** (sequence-level split) while keeping the same model and tuning grids. Additionally, we apply the same mean-shrinkage calibration to the **validation score after clipping** (already done) and keep everything else unchanged, so the only functional difference is better hyperparameter selection. This should move MCRMSE downward toward the target without changing core modeling logic or runtime materially.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
SUB_PATH_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

sub_path = None
df = None
for p in SUB_PATH_CANDIDATES:
    if os.path.exists(p):
        df = pd.read_csv(p)
        sub_path = p
        break

if sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle paths. "
        f"Tried: {SUB_PATH_CANDIDATES}"
    )

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
    raise ValueError(
        f"Loaded submission is missing required columns: {missing}. Got columns: {list(df.columns)}"
    )
df = df[required_cols].copy()



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:]



## === cell 3
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

TRAIN_PATH_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "/kaggle/data/train.json",
]
TEST_PATH_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine/test.json",
    "/kaggle/data/stanford-covid-vaccine/test.json",
    "/kaggle/input/test.json",
    "/kaggle/data/test.json",
]

train_path = next((p for p in TRAIN_PATH_CANDIDATES if os.path.exists(p)), None)
test_path = next((p for p in TEST_PATH_CANDIDATES if os.path.exists(p)), None)
if train_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not find train/test json. train_path={train_path}, test_path={test_path}"
    )

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

if "SN_filter" in train.columns:
    train = train.loc[train["SN_filter"].astype(int) == 1].reset_index(drop=True)

scored_len = int(train["seq_scored"].iloc[0])  # expected 68

error_cols_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}


def _explode_to_positions(meta_df: pd.DataFrame, has_targets: bool) -> pd.DataFrame:
    """Create one row per seqpos (0..seq_length-1) with char-level categorical features.
    If has_targets, include 5 targets for positions < seq_scored (else NaN) and per-target errors.
    """
    records = []
    for row in meta_df.itertuples(index=False):
        seq = row.sequence
        struct = row.structure
        loop = row.predicted_loop_type
        L = int(row.seq_length)
        scored = int(row.seq_scored)
        if has_targets:
            y_arrays = {c: getattr(row, c) for c in target_cols}
            e_arrays = {}
            for c in target_cols:
                ec = error_cols_map.get(c, None)
                e_arrays[c] = (
                    getattr(row, ec) if (ec is not None and hasattr(row, ec)) else None
                )

        for pos in range(L):
            prev_pos = pos - 1
            next_pos = pos + 1

            rec = {
                "id": row.id,
                "seqpos": pos,
                "base": seq[pos],
                "struct": struct[pos],
                "loop": loop[pos],
                "pos": pos,
                "base_prev": seq[prev_pos] if prev_pos >= 0 else "<START>",
                "struct_prev": struct[prev_pos] if prev_pos >= 0 else "<START>",
                "loop_prev": loop[prev_pos] if prev_pos >= 0 else "<START>",
                "base_next": seq[next_pos] if next_pos < L else "<END>",
                "struct_next": struct[next_pos] if next_pos < L else "<END>",
                "loop_next": loop[next_pos] if next_pos < L else "<END>",
            }
            if has_targets:
                if pos < scored:
                    for c in target_cols:
                        rec[c] = float(y_arrays[c][pos])
                        if e_arrays[c] is not None:
                            rec[f"{c}__err"] = float(e_arrays[c][pos])
                        else:
                            rec[f"{c}__err"] = np.nan
                else:
                    for c in target_cols:
                        rec[c] = np.nan
                        rec[f"{c}__err"] = np.nan
            records.append(rec)
    return pd.DataFrame.from_records(records)


train_pos = _explode_to_positions(train, has_targets=True)
train_pos = train_pos.dropna(subset=target_cols).reset_index(drop=True)

min_across_targets = train_pos[target_cols].min(axis=1)
train_pos = train_pos.loc[min_across_targets > -0.5].reset_index(drop=True)

fallback_means = train_pos[target_cols].mean(axis=0).to_dict()

train_target_mins = train_pos[target_cols].min(axis=0).to_dict()
clip_floors = {c: max(-0.5, float(train_target_mins[c])) for c in target_cols}

test_pos = _explode_to_positions(test, has_targets=False)

feature_cols = [
    "base",
    "struct",
    "loop",
    "pos",
    "base_prev",
    "struct_prev",
    "loop_prev",
    "base_next",
    "struct_next",
    "loop_next",
]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            [
                "base",
                "struct",
                "loop",
                "base_prev",
                "struct_prev",
                "loop_prev",
                "base_next",
                "struct_next",
                "loop_next",
            ],
        ),
        ("num", "passthrough", ["pos"]),
    ],
    remainder="drop",
)


def _make_model(n_outputs: int):
    if n_outputs > 1:
        return Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "reg",
                    MultiOutputRegressor(
                        Ridge(alpha=1.0, solver="lsqr", fit_intercept=False)
                    ),
                ),
            ]
        )
    else:
        return Pipeline(
            steps=[
                ("prep", preprocess),
                ("reg", Ridge(alpha=1.0, solver="lsqr", fit_intercept=False)),
            ]
        )


def _mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmses = []
    for j in range(y_true.shape[1]):
        rmses.append(np.sqrt(np.mean((y_true[:, j] - y_pred[:, j]) ** 2)))
    return float(np.mean(rmses))


X_train = train_pos[feature_cols]

rng = np.random.RandomState(0)
val_frac = 0.12
unique_ids = train_pos["id"].unique()
val_ids = set(unique_ids[rng.rand(len(unique_ids)) < val_frac])
val_mask = train_pos["id"].isin(val_ids).to_numpy()

X_tr = X_train.loc[~val_mask].reset_index(drop=True)
X_va = X_train.loc[val_mask].reset_index(drop=True)

Y_tr_all = train_pos.loc[~val_mask, target_cols].values
Y_va_scored = train_pos.loc[val_mask, scored_target_cols].values

Y_tr_scored = train_pos.loc[~val_mask, scored_target_cols].values

err_cols_scored = [f"{c}__err" for c in scored_target_cols]


def _build_sample_weight(df_pos: pd.DataFrame, cap_q: float):
    if not all(c in df_pos.columns for c in err_cols_scored):
        return None
    errs = df_pos[err_cols_scored].to_numpy(dtype=np.float32)
    errs = np.where(np.isfinite(errs), errs, np.nan)
    row_err = np.nanmean(errs, axis=1)
    row_err = np.where(np.isfinite(row_err), row_err, np.nanmedian(row_err))
    row_err = np.clip(row_err, 1e-3, np.inf)
    w = 1.0 / (row_err**2)
    w = w / np.mean(w)

    wcap = np.quantile(w, cap_q)
    if np.isfinite(wcap) and wcap > 0:
        w = np.clip(w, 0.0, wcap)
        w = w / np.mean(w)
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all():
        raise ValueError("sample_weight contains non-finite values after processing.")
    return w


cap_grid = [0.985, 0.99]  # very small change; chooses the better stabilizer
shrink_grid = [0.04, 0.06, 0.08]  # small post-processing calibration only

best = {"score": np.inf, "cap_q": 0.99, "shrink": 0.06}

for cap_q in cap_grid:
    w_all = _build_sample_weight(train_pos, cap_q)
    w_tr = w_all[~val_mask] if w_all is not None else None

    model_all5 = _make_model(n_outputs=5)
    model_scored3 = _make_model(n_outputs=3)

    if w_tr is not None:
        if w_tr.shape[0] != X_tr.shape[0]:
            raise ValueError("Internal weight length mismatch (train split).")
        model_all5.fit(X_tr, Y_tr_all, reg__sample_weight=w_tr)
        model_scored3.fit(X_tr, Y_tr_scored, reg__sample_weight=w_tr)
    else:
        model_all5.fit(X_tr, Y_tr_all)
        model_scored3.fit(X_tr, Y_tr_scored)

    pred_all5_va = np.asarray(model_all5.predict(X_va), dtype=np.float32)
    pred_scored3_va = np.asarray(model_scored3.predict(X_va), dtype=np.float32)

    pred_comb_va = pred_all5_va.copy()
    scored_idx = [target_cols.index(c) for c in scored_target_cols]
    for j, idx in enumerate(scored_idx):
        pred_comb_va[:, idx] = pred_scored3_va[:, j]

    for j, c in enumerate(target_cols):
        pred_comb_va[:, j] = np.clip(pred_comb_va[:, j], clip_floors[c], np.inf)

    for shrink in shrink_grid:
        pred_eval = pred_comb_va[:, scored_idx].copy()
        for jj, c in enumerate(scored_target_cols):
            mu = float(fallback_means[c])
            pred_eval[:, jj] = (1.0 - shrink) * pred_eval[:, jj] + shrink * mu

        score = _mcrmse(Y_va_scored, pred_eval)
        if score < best["score"]:
            best = {"score": score, "cap_q": cap_q, "shrink": shrink}

best_cap_q = best["cap_q"]
best_shrink = best["shrink"]

sample_weight = _build_sample_weight(train_pos, best_cap_q)

Y_train_all = train_pos[target_cols].values
Y_train_scored = train_pos[scored_target_cols].values

model_all5 = _make_model(n_outputs=5)
model_scored3 = _make_model(n_outputs=3)

if sample_weight is not None:
    if sample_weight.shape[0] != X_train.shape[0]:
        raise ValueError(
            f"sample_weight length mismatch: {sample_weight.shape[0]} vs X_train rows {X_train.shape[0]}"
        )
    model_all5.fit(X_train, Y_train_all, reg__sample_weight=sample_weight)
    model_scored3.fit(X_train, Y_train_scored, reg__sample_weight=sample_weight)
else:
    model_all5.fit(X_train, Y_train_all)
    model_scored3.fit(X_train, Y_train_scored)

pred_all5 = np.asarray(model_all5.predict(test_pos[feature_cols]), dtype=np.float32)
pred_scored3 = np.asarray(
    model_scored3.predict(test_pos[feature_cols]), dtype=np.float32
)

pred_combined = pred_all5.copy()
scored_idx = [target_cols.index(c) for c in scored_target_cols]
for j, idx in enumerate(scored_idx):
    pred_combined[:, idx] = pred_scored3[:, j]

for c in scored_target_cols:
    j = target_cols.index(c)
    mu = float(fallback_means[c])
    pred_combined[:, j] = (1.0 - best_shrink) * pred_combined[:, j] + best_shrink * mu

for j, c in enumerate(target_cols):
    pred_combined[:, j] = np.clip(pred_combined[:, j], clip_floors[c], np.inf)

test_pos["id_seqpos"] = (
    test_pos["id"].astype(str) + "_" + test_pos["seqpos"].astype(str)
)
pred_df = pd.DataFrame(pred_combined, columns=target_cols)
pred_df.insert(0, "id_seqpos", test_pos["id_seqpos"].values)

df = df.drop(columns=target_cols).merge(pred_df, on="id_seqpos", how="left")

missing_after_merge = df[target_cols].isna().any(axis=1).sum()
if missing_after_merge:
    for c in target_cols:
        df[c] = df[c].fillna(fallback_means[c])



## === cell 4
df.loc[df.id_seqpos.str.startswith("id_366486252"), "reactivity"] = 0



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
for c in target_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if df[target_cols].isna().any().any():
    for c in target_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(fallback_means.get(c, 0.0))

df = df[["id_seqpos"] + target_cols].copy()



## === cell 6
df.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv with shape={df.shape} (loaded base from {sub_path})")
print(
    f"Selected calibration: weight_cap_q={best_cap_q}, shrink={best_shrink}, val_mcrmse(scored)={best['score']:.6f}"
)
print(df.head())
