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

0.47144

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The current notebook fails because it tries to load a non-existent CSV from `../input/worst-submission/ensemble52.csv`, so `df` is never created and all following cells crash. To make this run end-to-end and output a valid `submission.csv`, I instead load the provided `sample_submission.csv` from the competition dataset path that exists in your environment. I keep the original downstream logic intact (deriving `sequences`, applying the single id-specific overwrite, and writing the CSV). This yield a valid submission file (though its score be baseline/sample-level since no model inference is present in the provided code).'
- What this solution (achieved 0.42418) has done: 'Your current score is much worse than the target (lower is better), and the main reason is that you’re essentially submitting the sample_submission baseline (with one small overwrite), i.e., no learning from train.json at all. To move the score substantially toward the target while keeping changes minimal and within your available packages, I replace the “all sample zeros” behavior with a simple per-position mean regressor learned from train.json (predict the mean value for each of the 5 targets at each sequence position). This preserves the overall “no complex model” spirit, adds only lightweight training logic, and produces correctly-shaped predictions for every `id_seqpos`. I also keep your existing id-specific overwrite intact (it may slightly help/hurt, but it preserves your notebook’s intended semantics).'
- What this solution (achieved 0.42166) has done: 'Your current score (0.42418, lower-is-better) is worse than the target (0.35188), so we should legitimately improve predictions while keeping your “per-position mean from train.json” core logic intact. The smallest meaningful gain without changing the modeling approach is to (1) train means only on high-quality rows (SN_filter==1), which matches common practice for this dataset and usually reduces noise, and (2) score-align the predictions by clipping to the training distribution per target/position to avoid extreme values that hurt RMSE. I also keep your existing id-specific overwrite exactly as-is to preserve your notebook’s intended semantics. The script still run end-to-end and write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.48245) has done: 'Your current score (0.42166, lower-is-better) is still meaningfully worse than the target (0.35188), so we should improve predictions while keeping your “per-position mean from train.json” approach intact. The smallest high-impact change is to compute the per-position means using **error-weighted averaging** (inverse-variance weights from the provided `*_error_*` arrays), which better matches the noise structure of this dataset without changing the model class. I keep your SN_filter==1 training subset and your per-position clipping, but make the clipping consistent with the weighted fit by computing quantiles on the same filtered data. The rest of the pipeline (submission alignment, id-specific overwrite, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.48245) has done: 'You’re currently worse than the target (lower is better), so we make a small, metric-aligned improvement without changing the core “per-position mean regressor” approach. The main fix is to compute the per-position clipping bounds on the same noise-aware basis as the weighted mean (i.e., use **weighted quantiles** instead of unweighted 1%/99%), which reduces the impact of noisy/low-quality measurements that can distort clipping and hurt RMSE. We keep your SN_filter==1 subset, inverse-variance weighting, tail fill behavior, and the id-specific overwrite exactly as-is. The result still runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.48245) has done: 'We keep your core “per-position regressor” intact (SN_filter==1, inverse-variance weighted mean, and clipping), but fix a key mismatch with the competition metric: only the first `seq_scored` positions are scored, so we should not let the noisy tail (positions 68–106) influence the scale/constraints. Concretely, we compute the mean and clipping bounds from the *scored* region only, and then for unscored positions we fill with a stable global constant (the last scored-position estimate) and use the same bounds—this usually reduces harmful variance without changing the model class. We also add a tiny safety fix to build `sequences` deterministically (stable ordering) without changing predictions. The submission format and your id-specific overwrite are preserved exactly.'
- What this solution (achieved 0.48244) has done: 'Your current score is worse than the target (lower is better), so we keep your exact “per-position weighted mean + clipping” core logic but make two minimal, metric-aligned fixes that typically improve MCRMSE. First, we compute the per-position means and clipping bounds using only the scored targets’ most reliable observations by additionally filtering out extreme-noise rows via a simple threshold on `signal_to_noise` (still no model change; just cleaner training statistics). Second, we ensure robustness to any NaNs/inf in the stacked arrays by masking them consistently in both the weighted mean and weighted quantile computations, preventing silent contamination that can noticeably hurt RMSE. The submission writing, schema, tail-fill behavior, and your id-specific overwrite remain unchanged.'
- What this solution (achieved 0.48217) has done: 'We keep your exact “per-position inverse-variance weighted mean + per-position clipping” approach, but tighten the training-data selection to better match the competition’s public/private test distribution without changing the model class. Concretely, we (1) use a slightly stricter `signal_to_noise` cutoff for the statistics (still deterministic, no extra training loops), and (2) make the clipping a bit less aggressive by moving from 1%/99% to 0.5%/99.5% weighted quantiles, which typically reduces bias from over-clipping while still protecting against outliers. We also ensure the weighted quantile uses a proper interpolation step (rather than step-function selection) to reduce discretization artifacts at small n, which can hurt RMSE. The submission format, tail-fill behavior, and your id-specific overwrite remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.4713) has done: 'We keep your exact “per-position inverse-variance weighted mean + weighted-quantile clipping + tail fill” approach, but fix two small issues that can materially hurt MCRMSE without changing the model class. First, your weighting currently uses all error values equally; we cap (winsorize) the per-position errors to avoid extremely tiny error values dominating the weighted mean/quantiles (a common pathology in this dataset). Second, we replace the single hard-coded id-specific overwrite (which is very likely harmful) with a no-op unless that id actually exists in the current test set, preserving intent while preventing accidental degradation. These are minimal, deterministic changes and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.4713) has done: 'We keep your exact “per-position inverse-variance weighted mean + weighted-quantile clipping + tail fill” core logic, but make two small, score-aligned fixes that commonly improve MCRMSE for this competition. First, we compute the weighting/clipping statistics only on the same high-quality subset you already select, but additionally drop rows with any non-finite target/error values in the scored region so they can’t silently skew weighted means/quantiles. Second, we remove the id-specific overwrite (reactivity=0) because it is almost certainly harmful on the test distribution; to preserve intent while avoiding score regression, we only apply it if that id exists **and** the overwrite value is within the learned per-position clipping band for those positions. The output remains a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.47144) has done: 'We keep your exact “per-position inverse-variance weighted mean + weighted-quantile clipping + tail fill” approach, but make the smallest metric-aligned improvement: compute the final per-position prediction as a **shrinkage blend** between the per-position mean and a global mean (same target), which typically reduces variance and improves RMSE on this dataset without changing the model class. We estimate a single deterministic shrinkage strength from the data using the effective sample size from your existing weights, then apply the same blending to the clipping bands (so clipping stays consistent with the prediction). Everything else (SN_filter/SNR filtering, error winsorization, scored-only fitting, submission alignment, and the guarded id-specific overwrite) remains unchanged, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.47144) has done: 'We keep your exact “per-position inverse-variance weighted mean + shrinkage blend + weighted-quantile clipping + tail fill” core logic, but fix a key mismatch: your **global** (shrinkage anchor) mean/quantiles are currently computed over only the scored region values *plus a large block of implicit zeros from positions 68–106* due to `y_flat = y.reshape(-1)` while `w_flat = w.reshape(-1)` includes weights for the unscored tail. That silently biases global stats toward 0 and harms all positions via shrinkage and blended clipping. The minimal, metric-aligned correction is to compute `w_flat` from the same scored-region weights (`w`) and to mask with the same finite mask, so global stats reflect only scored, valid observations. This should move the score down (better) toward your target without changing the modeling approach.'
- What this solution (achieved 0.47144) has done: 'We keep your exact per-position inverse-variance weighted mean + shrinkage + weighted-quantile clipping + tail-fill approach, but fix one remaining mismatch between how global shrinkage anchors are computed vs. how per-position stats are computed. Concretely, we compute the global mean/quantiles from the same *finite-masked* vectors used for weighting (i.e., only positions/entries with positive finite weights and finite targets), rather than letting any residual non-finite/zero-weight entries influence the anchors through inconsistent masking. This is a minimal, deterministic change that typically reduces bias in the shrinkage anchors and therefore improves scored-position RMSE without changing the model class. Everything else (data filters, winsorization, quantile levels, overwrite guard, and submission formatting) remains unchanged and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import os

CANDIDATE_SAMPLE_SUB_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/stanford-covid-vaccine/sample_submission.csv",
    "../input/sample_submission.csv",
    "/kaggle/data/stanford-covid-vaccine/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]

sample_path = next((p for p in CANDIDATE_SAMPLE_SUB_PATHS if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv in any of: {CANDIDATE_SAMPLE_SUB_PATHS}"
    )

sub = pd.read_csv(sample_path)
sub.head()



## === cell 2
CANDIDATE_TRAIN_PATHS = [
    "/kaggle/input/stanford-covid-vaccine/train.json",
    "/kaggle/input/train.json",
    "../input/stanford-covid-vaccine/train.json",
    "../input/train.json",
    "/kaggle/data/stanford-covid-vaccine/train.json",
    "/kaggle/data/train.json",
]

train_path = next((p for p in CANDIDATE_TRAIN_PATHS if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.json in any of: {CANDIDATE_TRAIN_PATHS}"
    )

train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
seq_scored = int(train["seq_scored"].iloc[0])  # expected 68
seq_length = int(train["seq_length"].iloc[0])  # expected 107

if "SN_filter" in train.columns:
    train_fit = train[train["SN_filter"].astype(int) == 1].copy()
    if len(train_fit) == 0:
        train_fit = train
else:
    train_fit = train

SNR_THRESHOLD = 2.0
if "signal_to_noise" in train_fit.columns:
    train_fit = train_fit[
        train_fit["signal_to_noise"].astype(float) >= SNR_THRESHOLD
    ].copy()
    if len(train_fit) == 0:
        train_fit = (
            train[train.get("SN_filter", 1).astype(int) == 1].copy()
            if "SN_filter" in train.columns
            else train
        )

error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

pos_means = {}
pos_q_low = {}
pos_q_high = {}

global_means = {}
global_q_low = {}
global_q_high = {}

EPS = 1e-6  # numerical stability for weights


def _weighted_quantile_1d(values, weights, q):
    """Compute weighted quantile for 1D arrays with linear interpolation. Assumes q in [0,1]."""
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.float32(np.nan)
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return np.float32(np.nan)

    cw = cw / total
    idx = int(np.searchsorted(cw, q, side="left"))
    if idx <= 0:
        return v[0]
    if idx >= len(v):
        return v[-1]
    cw0, cw1 = cw[idx - 1], cw[idx]
    v0, v1 = v[idx - 1], v[idx]
    if cw1 <= cw0:
        return v1
    t = (q - cw0) / (cw1 - cw0)
    return (1.0 - t) * v0 + t * v1


def _weighted_quantiles_per_position(y2d, w2d, q):
    """Vector of weighted quantiles across rows for each position (column)."""
    y2d = np.asarray(y2d, dtype=np.float32)
    w2d = np.asarray(w2d, dtype=np.float32)
    out = np.empty(y2d.shape[1], dtype=np.float32)
    for j in range(y2d.shape[1]):
        out[j] = _weighted_quantile_1d(y2d[:, j], w2d[:, j], q)
    return out


Q_LOW = 0.005
Q_HIGH = 0.995

ERR_Q_LOW = 0.05
ERR_Q_HIGH = 0.95

valid_rows = np.ones(len(train_fit), dtype=bool)
for c in target_cols:
    y_tmp = np.stack(train_fit[c].values).astype(np.float32)[:, :seq_scored]
    valid_rows &= np.all(np.isfinite(y_tmp), axis=1)

    err_col = error_map.get(c, None)
    if err_col is not None and err_col in train_fit.columns:
        e_tmp = np.stack(train_fit[err_col].values).astype(np.float32)[:, :seq_scored]
        valid_rows &= np.all(np.isfinite(e_tmp) & (e_tmp > 0), axis=1)

train_fit_clean = train_fit.iloc[valid_rows].copy()
if len(train_fit_clean) == 0:
    train_fit_clean = train_fit  # safety fallback to preserve end-to-end execution

SHRINKAGE_K = np.float32(200.0)

for c in target_cols:
    y_full = np.stack(train_fit_clean[c].values).astype(np.float32)
    y = y_full[:, :seq_scored]

    err_col = error_map.get(c, None)
    if err_col is not None and err_col in train_fit_clean.columns:
        e_full = np.stack(train_fit_clean[err_col].values).astype(np.float32)
        e = e_full[:, :seq_scored]

        finite_e = np.isfinite(e) & (e > 0)
        if np.any(finite_e):
            e_low = np.nanquantile(
                np.where(finite_e, e, np.nan), ERR_Q_LOW, axis=0
            ).astype(np.float32)
            e_high = np.nanquantile(
                np.where(finite_e, e, np.nan), ERR_Q_HIGH, axis=0
            ).astype(np.float32)
            e_low = np.where(np.isfinite(e_low) & (e_low > 0), e_low, np.float32(0.0))
            e_high = np.where(
                np.isfinite(e_high) & (e_high > 0), e_high, np.float32(np.inf)
            )

            e_cap = e.copy()
            for j in range(e_cap.shape[1]):
                lo = e_low[j]
                hi = e_high[j]
                if lo > 0:
                    e_cap[:, j] = np.maximum(e_cap[:, j], lo)
                if np.isfinite(hi):
                    e_cap[:, j] = np.minimum(e_cap[:, j], hi)
        else:
            e_cap = e

        finite = np.isfinite(y) & np.isfinite(e_cap) & (e_cap > 0)
        w = np.zeros_like(y, dtype=np.float32)
        w[finite] = 1.0 / (np.square(e_cap[finite]) + EPS)

        y_safe = np.where(np.isfinite(y), y, 0.0).astype(np.float32)
        wy = w * y_safe
        denom = np.sum(w, axis=0).astype(np.float32)
        denom_safe = np.where(denom <= 0, 1.0, denom).astype(np.float32)

        raw_pos_mean = (np.sum(wy, axis=0) / denom_safe).astype(np.float32)

        y_flat = y.reshape(-1).astype(np.float32)
        w_flat = w.reshape(-1).astype(np.float32)
        flat_mask = np.isfinite(y_flat) & np.isfinite(w_flat) & (w_flat > 0)
        if np.any(flat_mask):
            y_f = y_flat[flat_mask]
            w_f = w_flat[flat_mask]
            global_means[c] = float(np.sum(w_f * y_f) / np.maximum(np.sum(w_f), 1.0))
            global_q_low[c] = float(_weighted_quantile_1d(y_f, w_f, Q_LOW))
            global_q_high[c] = float(_weighted_quantile_1d(y_f, w_f, Q_HIGH))
        else:
            global_means[c] = float(np.nanmean(y_flat))
            global_q_low[c] = float(np.nanquantile(y_flat, Q_LOW))
            global_q_high[c] = float(np.nanquantile(y_flat, Q_HIGH))

        alpha = (denom / (denom + SHRINKAGE_K)).astype(np.float32)
        alpha = np.where(np.isfinite(alpha), alpha, np.float32(0.0))

        pos_means[c] = (
            alpha * raw_pos_mean + (1.0 - alpha) * np.float32(global_means[c])
        ).astype(np.float32)

        raw_q_low = _weighted_quantiles_per_position(y, w, Q_LOW).astype(np.float32)
        raw_q_high = _weighted_quantiles_per_position(y, w, Q_HIGH).astype(np.float32)

        pos_q_low[c] = (
            alpha * raw_q_low + (1.0 - alpha) * np.float32(global_q_low[c])
        ).astype(np.float32)
        pos_q_high[c] = (
            alpha * raw_q_high + (1.0 - alpha) * np.float32(global_q_high[c])
        ).astype(np.float32)
    else:
        raw_pos_mean = np.nanmean(y, axis=0).astype(np.float32)
        raw_q_low = np.nanquantile(y, Q_LOW, axis=0).astype(np.float32)
        raw_q_high = np.nanquantile(y, Q_HIGH, axis=0).astype(np.float32)

        y_flat = y.reshape(-1)
        global_means[c] = float(np.nanmean(y_flat))
        global_q_low[c] = float(np.nanquantile(y_flat, Q_LOW))
        global_q_high[c] = float(np.nanquantile(y_flat, Q_HIGH))

        n_eff = np.float32(y.shape[0])
        alpha = (n_eff / (n_eff + SHRINKAGE_K)).astype(np.float32)

        pos_means[c] = (
            alpha * raw_pos_mean + (1.0 - alpha) * np.float32(global_means[c])
        ).astype(np.float32)
        pos_q_low[c] = (
            alpha * raw_q_low + (1.0 - alpha) * np.float32(global_q_low[c])
        ).astype(np.float32)
        pos_q_high[c] = (
            alpha * raw_q_high + (1.0 - alpha) * np.float32(global_q_high[c])
        ).astype(np.float32)

tail_fill = {c: float(pos_means[c][seq_scored - 1]) for c in target_cols}
tail_q_low = {c: float(pos_q_low[c][seq_scored - 1]) for c in target_cols}
tail_q_high = {c: float(pos_q_high[c][seq_scored - 1]) for c in target_cols}

pos_means["__seq_scored__"] = seq_scored
pos_means["__seq_length__"] = seq_length

pos_means



## === cell 3
df = sub.copy()

seqpos = df["id_seqpos"].str.split("_").str[-1].astype(int).values

for c in target_cols:
    pred = np.empty(len(df), dtype=np.float32)
    scored_mask = seqpos < pos_means["__seq_scored__"]
    pred[scored_mask] = pos_means[c][seqpos[scored_mask]]
    pred[~scored_mask] = tail_fill[c]

    clip_low = np.empty(len(df), dtype=np.float32)
    clip_high = np.empty(len(df), dtype=np.float32)
    clip_low[scored_mask] = pos_q_low[c][seqpos[scored_mask]]
    clip_high[scored_mask] = pos_q_high[c][seqpos[scored_mask]]
    clip_low[~scored_mask] = tail_q_low[c]
    clip_high[~scored_mask] = tail_q_high[c]
    pred = np.clip(pred, clip_low, clip_high)

    df[c] = pred

df.head()



## === cell 4
sequences = sorted(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:]



## === cell 5
mask_b06 = df.id_seqpos.str.startswith("id_b06e7e027_")
if bool(mask_b06.any()):
    sp = df.loc[mask_b06, "id_seqpos"].str.split("_").str[-1].astype(int).values
    scored_mask_local = sp < pos_means["__seq_scored__"]

    low = np.empty_like(sp, dtype=np.float32)
    high = np.empty_like(sp, dtype=np.float32)
    low[scored_mask_local] = pos_q_low["reactivity"][sp[scored_mask_local]]
    high[scored_mask_local] = pos_q_high["reactivity"][sp[scored_mask_local]]
    low[~scored_mask_local] = np.float32(tail_q_low["reactivity"])
    high[~scored_mask_local] = np.float32(tail_q_high["reactivity"])

    ok = (0.0 >= low) & (0.0 <= high)
    idx = df.index[mask_b06][ok]
    if len(idx) > 0:
        df.loc[idx, "reactivity"] = 0.0



## === cell 6
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
