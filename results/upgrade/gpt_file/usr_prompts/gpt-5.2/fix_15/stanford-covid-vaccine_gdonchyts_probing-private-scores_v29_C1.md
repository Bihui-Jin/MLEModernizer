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

0.47871

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The main failure is that the script tries to read a non-existent `../input/worst-submission/ensemble52.csv`, so `df` is never created and all later cells crash. To make it run end-to-end and produce a valid submission, I replace that missing file dependency with loading the provided `sample_submission.csv` and filling it with safe constant predictions (score be poor but valid, and avoids inventing a new model given the strict “preserve core logic” constraint). I also fix the cell numbering to start at 1 and ensure all referenced variables exist. Finally, I keep your existing “set one sequence’s reactivity to 0” line but guard it so it won’t error if the id isn’t present.'
- What this solution (achieved 0.47889) has done: 'We keep your current “fill sample_submission with constants” core approach (since you’re not training a model here), but make two minimal tweaks that usually improve MCRMSE for this competition: (1) fill non-scored positions (seqpos >= seq_scored) with 0.0 while keeping scored positions at a calibrated constant, and (2) set separate constants per target based on the train mean over valid (SN_filter==1) samples, which is a small, legitimate calibration step that typically reduces RMSE versus all-zeros. This preserves the same overall logic (no new model/training loop), but uses information already available in train.json to choose better constant predictions and avoids wasting error mass on unscored positions. It still write a valid `submission.csv` with correct columns and row order. The existing special-case “one id reactivity = 0” line is kept and guarded.'
- What this solution (achieved 0.47889) has done: 'We keep your “constant-fill submission” core approach, but make two small calibration changes that typically reduce MCRMSE: (1) compute per-target means using only the first `seq_scored` positions (the only positions with labels) instead of averaging across variable-length arrays, and (2) clip constant predictions to a reasonable range learned from training (e.g., 1st–99th percentile over scored positions) to reduce RMSE impact from outliers. We also compute means/quantiles from a slightly cleaner subset (`SN_filter==1` and finite values) while keeping all file paths and the submission writing logic unchanged. These are minimal, metric-aligned adjustments and should move the score down toward your target without changing the overall solution style.'
- What this solution (achieved 0.55557) has done: 'I keep your constant-fill submission approach, but calibrate the constants to better match the evaluation: MCRMSE only scores 3 targets, so we compute and use per-target means specifically for those scored targets (and still output all 5 columns). To reduce RMSE further without changing the approach, we compute the constant means using inverse-variance weights derived from the provided per-position measurement errors in `train.json` (a minimal, metric-aligned calibration step). We also keep your existing “non-scored positions = 0.0” handling and the guarded special-case id override, and continue writing a valid `submission.csv` with the same paths. This should move the score down (better) toward your target while preserving the core logic (no model/training).'
- What this solution (achieved 0.50664) has done: 'I keep your constant-fill submission strategy intact, but calibrate the constants in a way that better matches the MCRMSE objective: for each target, use the per-position median over clean training samples (SN_filter==1) instead of a global inverse-variance weighted mean, since medians are more robust to heavy-tailed noise and outliers that can inflate RMSE. I also compute the clip bounds (1%–99%) from the same scored-position pool and clip the chosen constant into that range (still minimal and metric-aligned). Finally, I ensure we only use the first `seq_scored` positions everywhere (as you already do) and keep the “non-scored positions = 0.0” handling and the guarded special-case id override unchanged.'
- What this solution (achieved 0.47901) has done: 'You’re currently well above the target (0.50664 vs 0.35188; lower is better), so we should make a minimal, metric-aligned improvement without changing the “constant-fill submission” core approach. The biggest low-risk gain here is to calibrate the constant per target to directly minimize RMSE: for a constant predictor, the RMSE-optimal constant is the mean (not the median), so we switch from median to mean computed over scored positions and clean (SN_filter==1) training samples. To keep robustness similar to what you had, we compute the mean after mild winsorization (clip training values to the 1%–99% range) and then clip the final constant into that same range. Everything else (non-scored positions set to 0.0, column schema, guarded id override, and writing submission.csv) remains unchanged.'
- What this solution (achieved 0.42173) has done: 'We keep your “constant-fill submission” core logic, but make a minimal metric-aligned improvement: instead of predicting one constant per target for all scored positions, we predict a separate constant per position (seqpos) per target computed from the training set. This is still the same approach (no model/training loop), just a better-calibrated constant baseline that usually reduces MCRMSE because each nucleotide position has different typical values. We compute these per-seqpos means on clean samples (SN_filter==1) over the first `seq_scored` positions, apply light winsorization per position to reduce outlier impact, and keep your existing behavior of setting non-scored positions to 0.0 plus the guarded special-case id override. The output file path/name and submission schema stay unchanged and it still write a valid `submission.csv`.'
- What this solution (achieved 0.42173) has done: 'We keep your per-seqpos constant baseline intact, but make a small metric-aligned calibration: compute the per-position means using only the *scored* targets/positions and then set the two unscored targets (`deg_pH10`, `deg_50C`) to be a simple linear blend of the related scored targets (`deg_Mg_pH10` and `deg_Mg_50C`) per position. This often reduces error versus learning those two targets independently with noisy labels, while preserving the same “no model/training, just calibrated constants” core approach. We also compute per-position winsorization quantiles only on SN_filter==1 and enforce finite-only values as you already do, keeping non-scored positions at 0.0 and preserving your special-case id override. The output schema/path remains unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.48263) has done: 'Your current per-seqpos mean baseline is already a reasonable “no-model” approach, but it’s leaving score on the table because it treats every training sample equally even though the dataset provides per-position measurement errors. To move the score down toward the 0.3519 target while keeping the same core logic (still just calibrated constants, no model/training loop), I compute **inverse-variance weighted per-position means** for the 3 scored targets using their corresponding `*_error_*` arrays (and keep your existing winsorization to limit outlier impact). For the 2 unscored targets, I keep your existing linear-blend approach unchanged. This is a minimal, metric-aligned calibration change that should reduce RMSE on scored columns without altering file paths or submission formatting.'
- What this solution (achieved 0.48236) has done: 'To move your (lower-is-better) score down toward the 0.3519 target while preserving the same “per-seqpos calibrated constants” core logic, I make two minimal, metric-aligned fixes: (1) apply the inverse-variance weighting consistently by winsorizing values using **unweighted** quantile bounds but computing the **weighted mean on the raw values** (so the weighting isn’t distorted by clipping), and (2) fill missing/invalid per-position estimates by falling back to the unweighted per-position mean (instead of defaulting to 0.0). Both changes keep the same overall approach (no model/training), keep paths/format identical, and should reduce RMSE on the 3 scored targets without impacting submission validity. Everything else (non-scored positions set to 0.0, unscored-target blending, special-case id override, writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.47366) has done: 'Your current score (0.48236, lower is better) is still far from the target (0.35188), so we should make a small but metric-aligned improvement while keeping your “per-seqpos calibrated constants (with error-weighting)” core approach unchanged. The main low-risk gain is to make the inverse-variance weighting more robust by adding a tiny error-floor (so extremely small reported errors don’t dominate) and by winsorizing the *weights* as well (cap extreme weights per position). Additionally, we should compute the weighted mean using the same clipped-in-range samples you already select, but with stabilized weights; this often improves MCRMSE without changing the overall baseline nature. Everything else (per-position means, unscored positions set to 0.0, unscored-target linear blends, special-case id override, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.50947) has done: 'Your current score (0.47366; lower is better) is still well above the target (0.35188), so we should make a minimal, metric-aligned improvement while keeping your “per-seqpos calibrated constants with error-weighting” approach intact. The smallest likely gain is to compute **positionwise weighted means** using a more robust estimator: use the same inverse-variance idea but estimate the constant via a **weighted median** (more resistant to heavy-tailed noise/outliers that inflate RMSE) and keep your existing unweighted winsor bounds, error floor, and weight capping. Everything else (non-scored positions set to 0.0, linear blends for unscored targets, guarded id override, file paths, and submission writing) remains unchanged to preserve evaluation semantics and runtime.'
- What this solution (achieved 0.47366) has done: 'Your current score (0.50947, lower is better) is still far above the target (0.35188), so we should make a small, metric-aligned improvement while keeping the same “per-seqpos calibrated constants using error arrays” core approach. The main issue is that switching to a weighted median can move predictions away from the squared-loss optimum; for MCRMSE (RMSE-based), the constant that minimizes squared error is the (weighted) mean. I therefore revert the aggregation for scored targets back to a stabilized inverse-variance weighted mean (keeping your existing unweighted winsor bounds, error floor, and weight capping), which should reduce RMSE and move the score downward toward the target. Everything else (non-scored positions set to 0.0, linear blends for unscored targets, special-case id override, file paths, and submission writing) remains unchanged.'
- What this solution (achieved 0.47871) has done: 'We keep your existing “per-seqpos calibrated constants with inverse-variance weighting” approach, but fix a key statistical mismatch: the winsor bounds are currently computed from the *unweighted values* only, which can be too wide when low-error (high-weight) measurements have a tighter distribution; we instead compute winsor bounds using a lightweight **weighted quantile** per position and then take a stabilized weighted mean within those bounds. This is still the same core logic (per-position constants + error-weighting), just a more metric-aligned calibration that typically reduces RMSE for the 3 scored targets. We also make the error-floor and weight-cap scale adapt per-position using robust summaries of the error distribution (so positions with universally larger errors aren’t over-regularized), which is a minimal change and preserves runtime. Submission formatting, non-scored positions set to 0.0, and the unscored-target linear blends remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
sequences_public = set(df_train.id)



## === cell 3
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences = [s for s in sequences if s not in sequences_public]
sequences[-10:]



## === cell 4
len(sequences), len(sequences_public)



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_target_cols = [
    "reactivity",
    "deg_Mg_pH10",
    "deg_Mg_50C",
]  # only these are scored

for c in target_cols:
    if c not in df.columns:
        df[c] = 0.0

df_train_f = df_train[df_train["SN_filter"] == 1].copy()
seq_scored_train = (
    int(df_train_f["seq_scored"].iloc[0]) if "seq_scored" in df_train_f.columns else 68
)

scored_target_to_error = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
}


def _per_pos_winsorized_means(values_list, n_scored, qlo=0.01, qhi=0.99):
    per_pos = [[] for _ in range(n_scored)]
    for a in values_list:
        if not isinstance(a, list):
            continue
        aa = a[:n_scored]
        if len(aa) < n_scored:
            continue
        for i, v in enumerate(aa):
            if v is None:
                continue
            if not np.isfinite(v):
                continue
            per_pos[i].append(float(v))

    means = np.zeros(n_scored, dtype=np.float32)
    clip_ranges = [(-1.0, 1.0)] * n_scored

    for i in range(n_scored):
        vals = np.asarray(per_pos[i], dtype=np.float32)
        if vals.size == 0:
            means[i] = 0.0
            clip_ranges[i] = (-1.0, 1.0)
            continue
        lo = float(np.quantile(vals, qlo))
        hi = float(np.quantile(vals, qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(vals)), float(np.max(vals))
        clipped = np.clip(vals, lo, hi)
        means[i] = float(np.mean(clipped))
        clip_ranges[i] = (lo, hi)

    return means, clip_ranges


def _weighted_quantile(values, weights, q):
    """
    Minimal metric-aligned addition:
    - For weighted constant prediction, robust bounds should reflect the *weighted* distribution.
    - This computes a standard weighted quantile (no interpolation), sufficient for winsor bounds.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[m]
    w = w[m]
    if v.size == 0:
        return np.nan
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = q * cw[-1]
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def _per_pos_weighted_means_with_weighted_winsor_bounds(
    values_list,
    errors_list,
    n_scored,
    qlo=0.01,
    qhi=0.99,
    fallback_means=None,
    fallback_clip_ranges=None,
    base_err_floor=0.02,
    w_qhi=0.995,
):
    """
    Minimal improvement toward lower MCRMSE while preserving the same core logic:
    - Still per-position constants.
    - Still inverse-variance weighting via error arrays.
    - Change: compute winsor bounds from the *weighted* distribution (weighted quantiles),
      then compute a stabilized weighted mean within those bounds.
    - Also make the error-floor adaptive per position using a robust error quantile, so
      positions with generally larger measurement errors aren't over-penalized.
    """
    per_pos_vals = [[] for _ in range(n_scored)]
    per_pos_errs = [[] for _ in range(n_scored)]

    for a, e in zip(values_list, errors_list):
        if not (isinstance(a, list) and isinstance(e, list)):
            continue
        aa = a[:n_scored]
        ee = e[:n_scored]
        if len(aa) < n_scored or len(ee) < n_scored:
            continue
        for i, (v, er) in enumerate(zip(aa, ee)):
            if v is None or er is None:
                continue
            if not (np.isfinite(v) and np.isfinite(er)):
                continue
            er = float(er)
            if er <= 0.0:
                continue
            per_pos_vals[i].append(float(v))
            per_pos_errs[i].append(er)

    means = np.zeros(n_scored, dtype=np.float32)
    clip_ranges = [(-1.0, 1.0)] * n_scored

    for i in range(n_scored):
        vals = np.asarray(per_pos_vals[i], dtype=np.float64)
        errs = np.asarray(per_pos_errs[i], dtype=np.float64)

        if vals.size == 0:
            if fallback_means is not None:
                means[i] = float(fallback_means[i])
            else:
                means[i] = 0.0
            if fallback_clip_ranges is not None:
                lo, hi = fallback_clip_ranges[i]
                clip_ranges[i] = (float(lo), float(hi))
            else:
                clip_ranges[i] = (-1.0, 1.0)
            continue

        er_q = float(np.quantile(errs, 0.10)) if errs.size > 0 else base_err_floor
        err_floor = float(max(base_err_floor, er_q * 0.5))
        es = np.maximum(errs, err_floor)
        w = 1.0 / (es**2)

        if w.size >= 10:
            cap = float(np.quantile(w, w_qhi))
            if np.isfinite(cap) and cap > 0:
                w = np.minimum(w, cap)

        lo = _weighted_quantile(vals, w, qlo)
        hi = _weighted_quantile(vals, w, qhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(np.quantile(vals, qlo))
            hi = float(np.quantile(vals, qhi))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(np.min(vals)), float(np.max(vals))

        clip_ranges[i] = (float(lo), float(hi))

        m = (vals >= lo) & (vals <= hi)
        vsel = vals[m]
        wsel = w[m]

        if vsel.size == 0:
            means[i] = (
                float(fallback_means[i])
                if fallback_means is not None
                else float(np.mean(vals))
            )
            continue

        denom = float(np.sum(wsel))
        if not np.isfinite(denom) or denom <= 0.0:
            means[i] = float(np.mean(vsel))
        else:
            means[i] = float(np.sum(wsel * vsel) / denom)

    return means.astype(np.float32), clip_ranges


target_pos_means = {}
target_pos_clip = {}

unweighted_means = {}
unweighted_clips = {}
for c in target_cols:
    means, clip_ranges = _per_pos_winsorized_means(
        df_train_f[c].tolist(), seq_scored_train
    )
    unweighted_means[c] = means
    unweighted_clips[c] = clip_ranges

for c in target_cols:
    if c in scored_target_to_error:
        err_col = scored_target_to_error[c]

        means_w, clip_ranges_w = _per_pos_weighted_means_with_weighted_winsor_bounds(
            df_train_f[c].tolist(),
            df_train_f[err_col].tolist(),
            seq_scored_train,
            qlo=0.01,
            qhi=0.99,
            fallback_means=unweighted_means[c],
            fallback_clip_ranges=unweighted_clips[c],
            base_err_floor=0.02,
            w_qhi=0.995,
        )

        target_pos_means[c] = means_w
        target_pos_clip[c] = clip_ranges_w
    else:
        target_pos_means[c] = unweighted_means[c]
        target_pos_clip[c] = unweighted_clips[c]


def _fit_linear_blend(y_list, x_list, n_scored):
    xs = []
    ys = []
    for y, x in zip(y_list, x_list):
        if not (isinstance(y, list) and isinstance(x, list)):
            continue
        yy = y[:n_scored]
        xx = x[:n_scored]
        if len(yy) < n_scored or len(xx) < n_scored:
            continue
        for yi, xi in zip(yy, xx):
            if yi is None or xi is None:
                continue
            if not (np.isfinite(yi) and np.isfinite(xi)):
                continue
            ys.append(float(yi))
            xs.append(float(xi))
    if len(xs) < 10:
        return 1.0, 0.0
    X = np.asarray(xs, dtype=np.float64)
    Y = np.asarray(ys, dtype=np.float64)
    xm = float(X.mean())
    ym = float(Y.mean())
    xv = float(((X - xm) ** 2).mean())
    if xv <= 1e-12:
        return 1.0, 0.0
    cov = float(((X - xm) * (Y - ym)).mean())
    a = cov / xv
    b = ym - a * xm
    a = float(np.clip(a, 0.0, 2.0))
    b = float(np.clip(b, -0.5, 0.5))
    return a, b


a_pH10, b_pH10 = _fit_linear_blend(
    df_train_f["deg_pH10"].tolist(),
    df_train_f["deg_Mg_pH10"].tolist(),
    seq_scored_train,
)
a_50C, b_50C = _fit_linear_blend(
    df_train_f["deg_50C"].tolist(), df_train_f["deg_Mg_50C"].tolist(), seq_scored_train
)

deg_pH10_blend = (a_pH10 * target_pos_means["deg_Mg_pH10"] + b_pH10).astype(np.float32)
deg_50C_blend = (a_50C * target_pos_means["deg_Mg_50C"] + b_50C).astype(np.float32)

deg_pH10_clip = np.array(
    [cr[0] for cr in target_pos_clip["deg_pH10"]], dtype=np.float32
), np.array([cr[1] for cr in target_pos_clip["deg_pH10"]], dtype=np.float32)
deg_50C_clip = np.array(
    [cr[0] for cr in target_pos_clip["deg_50C"]], dtype=np.float32
), np.array([cr[1] for cr in target_pos_clip["deg_50C"]], dtype=np.float32)

target_pos_means["deg_pH10"] = np.clip(
    deg_pH10_blend, deg_pH10_clip[0], deg_pH10_clip[1]
).astype(np.float32)
target_pos_means["deg_50C"] = np.clip(
    deg_50C_blend, deg_50C_clip[0], deg_50C_clip[1]
).astype(np.float32)

tmp = df["id_seqpos"].astype(str).str.rsplit("_", n=1, expand=True)
df["_id"] = tmp[0]
df["_seqpos"] = tmp[1].astype(int)

test_meta = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)[
    ["id", "seq_scored"]
]
df = df.merge(test_meta, how="left", left_on="_id", right_on="id")
df["seq_scored"] = df["seq_scored"].fillna(seq_scored_train).astype(int)

scored_mask = df["_seqpos"] < df["seq_scored"]

for c in target_cols:
    means = target_pos_means[c]
    pos = df["_seqpos"].to_numpy()
    pos_clipped = np.clip(pos, 0, seq_scored_train - 1)
    pred = means[pos_clipped].astype(np.float32)
    df.loc[scored_mask, c] = pred[scored_mask.to_numpy()]
    df.loc[~scored_mask, c] = 0.0

df[target_cols] = df[target_cols].astype(float).fillna(0.0)



## === cell 6
mask = df.id_seqpos.astype(str).str.startswith("id_79819a72b")
if mask.any():
    df.loc[mask, "reactivity"] = 0.0



## === cell 7
df = df.drop(
    columns=[c for c in ["_id", "_seqpos", "id", "seq_scored"] if c in df.columns]
)
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)

summary = {}
for k in target_cols:
    summary[k] = [float(x) for x in target_pos_means[k][:5]]
print("Per-position means (first 5 seqpos) =", summary)
print("Linear blends: deg_pH10 = a*deg_Mg_pH10 + b with a,b =", a_pH10, b_pH10)
print("Linear blends: deg_50C  = a*deg_Mg_50C  + b with a,b =", a_50C, b_50C)
