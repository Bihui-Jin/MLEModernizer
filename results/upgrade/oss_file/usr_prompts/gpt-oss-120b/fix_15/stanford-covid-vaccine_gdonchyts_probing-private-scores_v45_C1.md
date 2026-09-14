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

0.40861

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'The fix loads the official sample submission instead of a missing file, reads the training data to compute average values for the three scored targets, fills the whole submission with those averages (a simple but valid baseline), and finally writes a proper `submission.csv`. This resolves the FileNotFound and NameError issues and ensures a correctly‑formatted CSV is produced, moving the score toward the target range.'
- What this solution (achieved 0.42418) has done: 'I replace the global‑mean baseline with a simple position‑specific baseline: compute the mean for each target at each of the 68 scored positions and use those values for the corresponding rows in the submission. For positions beyond 68 (which are unscored) I keep the overall mean as a fallback. This small refinement preserves the original logic while giving more accurate predictions, moving the validation score closer to the target.'
- What this solution (achieved 0.63824) has done: 'I added a lightweight nucleotide‑aware baseline: for each of the three scored targets it now uses the mean value conditioned on both position (0‑67) *and* the base (A,C,G,U) observed at that position in the training set. The script merges the test sequences to the submission, extracts the correct base for each row, and falls back to the simple position‑wise mean when a base‑specific mean is unavailable (or for positions ≥ 68). This modest enrichment keeps the original logic intact while giving more accurate predictions, moving the validation score closer to the target.'
- What this solution (achieved 0.41796) has done: 'The update fixes the column‑name typo (`deg_p10` → `deg_pH10`), corrects the string split used to extract the sample id, improves handling of missing base‑specific statistics, and adds a light smoothing that blends base‑specific and position‑wise means. These changes resolve the runtime errors and give a modest but meaningful boost in predictive accuracy, moving the validation score closer to the target while preserving the original baseline logic.'
- What this solution (achieved 0.4112) has done: 'I added a lightweight “loop‑type” baseline that mirrors the existing base‑specific and position‑wise statistics. For each scored column we now compute the mean value per position per loop‑type (S,M,I,B,H,E,X) from the training set, falling back to the position mean when a loop‑type has no data. In the prediction function the final estimate is a blend of the base‑specific mean (70 %), the loop‑type mean (20 %), and the plain position mean (10 %). This modest extra signal should lower the validation MCRMSE, moving the score closer to the target without altering the core modelling logic.'
- What this solution (achieved 0.4115) has done: 'I fixed the runtime errors by removing the broken validation‑weight search (which caused the `val` variable to become a float and the missing `best_weights` name) and replaced it with a fixed, sensible blending weight tuple. The script now consistently defines `best_weights`, builds the prediction functions, merges test data, fills all target columns using the learned position/base/loop statistics, and writes a correctly formatted `submission.csv` without any NameError or AttributeError.'
- What this solution (achieved 0.40302) has done: 'I keep the original baseline logic but replace the fixed blending weights with data‑driven weights that are learned on the held‑out validation split. By fitting a simple non‑negative least‑squares model on the validation part for each scored target and then averaging the resulting weights, the blended predictor gains a modest but effective calibration, moving the MCRMSE closer to the target score while preserving the core implementation.'
- What this solution (achieved 0.40288) has done: 'I keep the overall baseline logic but improve the blending step by using column‑specific weights learned on the validation split instead of a single averaged weight vector. This small change better captures the differing importance of base‑, loop‑, and position‑means for each target, and should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41134) has done: 'I compute the overall and position‑wise means using the full training set (instead of only the 80 % split) and replace the learned blending weights with the stable fallback weights [0.7, 0.2, 0.1] for each scored target. This keeps the core baseline unchanged while using more data for the statistics and applying a proven‑effective weighting, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.4087) has done: 'I compute validation‑based blending weights for each scored target instead of using the fixed [0.7, 0.2, 0.1] tuple. By expanding the held‑out validation set to per‑position rows, comparing the three component predictions (base‑specific, loop‑specific, position‑mean) against the true values, and assigning weights inversely proportional to each component’s MSE (then normalising), the model’s calibration improves and the MCRMSE moves closer to the target score while keeping the original baseline logic unchanged.'
- What this solution (achieved 0.40875) has done: 'The patch keeps the original three‑component baseline but improves the statistics by aggregating base‑ and loop‑specific means over the full training set (instead of only the 80 % split) and smooths the validation‑derived blending weights with a modest equal‑weight fallback. These adjustments are expected to reduce the validation MCRMSE, moving the score closer to the target while preserving the core logic.'
- What this solution (achieved 0.4087) has done: 'I keep the overall baseline logic unchanged but improve the blending weights that combine base‑specific, loop‑specific and position‑wise predictions.  
Instead of learning the weights on only the 20 % validation split, I compute them on the full training set (which gives more stable estimates) and blend the inverse‑MSE weights more aggressively (70 % inverse‑MSE + 30 % equal weight). This modest adjustment is expected to lower the MCRMSE toward the target score while preserving the core architecture and output format.'
- What this solution (achieved 0.40861) has done: 'I keep the overall baseline logic unchanged but make the blending rely purely on the inverse‑MSE‑derived weights (set `blend_factor` to 1.0). This removes the equal‑weight fallback, allowing the more informative base‑specific and loop‑specific components to have a larger impact, which should reduce the MCRMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_50C", "deg_pH10", "deg_Mg_50C"]
scored_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10"]

rng = np.random.default_rng(42)
perm = rng.permutation(len(train))
split_idx = int(0.8 * len(train))
train_stats = train.iloc[perm[:split_idx]].reset_index(drop=True)
val_stats = train.iloc[perm[split_idx:]].reset_index(drop=True)

overall_means = {}
pos_means = {}
for col in target_cols:
    all_vals = np.concatenate(train[col].values)
    overall_means[col] = all_vals.mean()
    stacked = np.vstack(train[col].values)  # (n_samples, 68)
    pos_means[col] = stacked.mean(axis=0)  # (68,)

base_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3}
n_bases = 4
pos_base_sums = {col: np.zeros((68, n_bases)) for col in scored_cols}
pos_base_counts = {col: np.zeros((68, n_bases)) for col in scored_cols}

loop_chars = ["S", "M", "I", "B", "H", "E", "X"]
loop_to_idx = {c: i for i, c in enumerate(loop_chars)}
n_loops = len(loop_chars)
pos_loop_sums = {col: np.zeros((68, n_loops)) for col in scored_cols}
pos_loop_counts = {col: np.zeros((68, n_loops)) for col in scored_cols}

for _, row in train.iterrows():
    seq = row["sequence"][:68]
    loops = row["predicted_loop_type"][:68]
    for col in scored_cols:
        vals = row[col]  # list of 68 floats
        for p, (base, lt, val) in enumerate(zip(seq, loops, vals)):
            b_idx = base_to_idx.get(base)
            if b_idx is not None:
                pos_base_sums[col][p, b_idx] += val
                pos_base_counts[col][p, b_idx] += 1
            l_idx = loop_to_idx.get(lt)
            if l_idx is not None:
                pos_loop_sums[col][p, l_idx] += val
                pos_loop_counts[col][p, l_idx] += 1

pos_base_means = {}
for col in scored_cols:
    with np.errstate(divide="ignore", invalid="ignore"):
        means = pos_base_sums[col] / pos_base_counts[col]
    means = np.where(np.isnan(means), pos_means[col][:, np.newaxis], means)
    pos_base_means[col] = means  # (68, 4)

pos_loop_means = {}
for col in scored_cols:
    with np.errstate(divide="ignore", invalid="ignore"):
        means = pos_loop_sums[col] / pos_loop_counts[col]
    means = np.where(np.isnan(means), pos_means[col][:, np.newaxis], means)
    pos_loop_means[col] = means  # (68, 7)


def _components(row, col):
    """Return (base_specific, loop_specific, pos_mean) for a given row & column."""
    p = row["pos"]
    if p >= 68:
        return (np.nan, np.nan, overall_means[col])

    base = row["sequence"][p]
    loop = row["predicted_loop_type"][p]

    b_idx = base_to_idx.get(base)
    base_specific = pos_base_means[col][p, b_idx] if b_idx is not None else np.nan
    l_idx = loop_to_idx.get(loop)
    loop_specific = pos_loop_means[col][p, l_idx] if l_idx is not None else np.nan
    pos_mean = pos_means[col][p]

    if np.isnan(base_specific):
        base_specific = pos_mean
    if np.isnan(loop_specific):
        loop_specific = pos_mean
    return (base_specific, loop_specific, pos_mean)


train_rows = []
for _, row in train.iterrows():
    seq = row["sequence"]
    loops = row["predicted_loop_type"]
    for p in range(row["seq_scored"]):
        entry = {
            "id": row["id"],
            "pos": p,
            "sequence": seq,
            "predicted_loop_type": loops,
        }
        for col in scored_cols:
            entry[col] = row[col][p]
        train_rows.append(entry)

train_df = pd.DataFrame(train_rows)

for col in scored_cols:
    comps = train_df.apply(lambda r: _components(r, col), axis=1)
    train_df[f"{col}_base"] = comps.map(lambda t: t[0])
    train_df[f"{col}_loop"] = comps.map(lambda t: t[1])
    train_df[f"{col}_pos"] = comps.map(lambda t: t[2])

epsilon = 1e-12
col_weights = {}
blend_factor = 1.0  # use pure inverse‑MSE weighting
for col in scored_cols:
    y_true = train_df[col].values.astype(float)

    pred_base = train_df[f"{col}_base"].values
    pred_loop = train_df[f"{col}_loop"].values
    pred_pos = train_df[f"{col}_pos"].values

    mse_base = np.nanmean((y_true - pred_base) ** 2)
    mse_loop = np.nanmean((y_true - pred_loop) ** 2)
    mse_pos = np.nanmean((y_true - pred_pos) ** 2)

    inv = np.array(
        [
            1.0 / (mse_base + epsilon),
            1.0 / (mse_loop + epsilon),
            1.0 / (mse_pos + epsilon),
        ]
    )
    inv_norm = inv / inv.sum()
    equal = np.full(3, 1.0 / 3.0)

    weights = blend_factor * inv_norm + (1 - blend_factor) * equal
    col_weights[col] = weights


def blended_predict(row, col, w_base, w_loop, w_pos):
    p = row["pos"]
    if p >= 68:
        return overall_means[col]

    base = row["sequence"][p]
    loop = row["predicted_loop_type"][p]

    b_idx = base_to_idx.get(base)
    base_specific = pos_base_means[col][p, b_idx] if b_idx is not None else np.nan
    l_idx = loop_to_idx.get(loop)
    loop_specific = pos_loop_means[col][p, l_idx] if l_idx is not None else np.nan
    pos_mean = pos_means[col][p]

    if np.isnan(base_specific):
        base_specific = pos_mean
    if np.isnan(loop_specific):
        loop_specific = pos_mean

    return w_base * base_specific + w_loop * loop_specific + w_pos * pos_mean




## === cell 1
df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
submission = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

submission["pos"] = submission["id_seqpos"].str.extract(r"_(\d+)$").astype(int)
submission["id"] = submission["id_seqpos"].str.rsplit("_", n=1).str[0]

submission = submission.merge(
    df_test[["id", "sequence", "predicted_loop_type"]], on="id", how="left"
)


def final_predict(row, col):
    w_base, w_loop, w_pos = col_weights[col]
    return blended_predict(row, col, w_base, w_loop, w_pos)


for col in scored_cols:
    submission[col] = submission.apply(lambda r: final_predict(r, col), axis=1)

for col in set(target_cols) - set(scored_cols):
    mask = submission["pos"] < 68
    submission.loc[mask, col] = submission.loc[mask, "pos"].map(
        lambda p: pos_means[col][p]
    )
    submission.loc[~mask, col] = overall_means[col]

submission.drop(columns=["pos", "id", "sequence", "predicted_loop_type"], inplace=True)
submission.to_csv("submission.csv", index=False)
