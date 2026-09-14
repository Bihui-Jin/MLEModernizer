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

0.3558588776558584

# 6. Current score

0.42564

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'Your notebook fails because it tries to read a non-existent external submission file (`../input/worst-submission/ensemble52.csv`), so `df` is never created and all later cells error. To make it run end-to-end in this environment, I instead load the provided `sample_submission.csv` and use it as the base `df`, ensuring the required columns and row count exist. The rest of your logic (extracting sequence ids and scaling predictions by 1.05) is preserved; it now produce a valid `submission.csv` with the correct format. This won’t be a good score, but it yield a valid submission file; once you provide a real model/prediction source, we can then tune toward the target score.'
- What this solution (achieved 0.4256) has done: 'Your current submission is effectively the all-zeros baseline (sample_submission values are 0, and you further divide by 1.05), which explains the weak 0.63824 score. To move toward the much better target (lower MCRMSE), the smallest legitimate improvement without changing the “no-model” core logic is to replace the constant-zero predictions with per-position mean targets learned from `train.json` (a simple prior), while keeping the same submission formatting pipeline. This uses only provided competition data, produces the correct 25680-row file, and should substantially reduce error versus zeros, moving the score closer to ~0.356. I also keep non-scored columns filled consistently (use their train means too) to avoid any NaNs and preserve valid output.'
- What this solution (achieved 0.42166) has done: 'Your current score (0.4256, lower is better) is still worse than the target (0.3559), so we should improve predictions slightly without changing the overall “per-position prior from train.json” core logic. The smallest high-impact fix is to stop applying the global `/= 1.05` shrink, because it systematically biases all targets downward and typically worsens RMSE when targets are centered near their true mean. Additionally, we can make the per-position mean more robust by applying the competition’s known quality filter (`SN_filter == 1`) when computing those means, which usually better matches the (filtered) test distribution while keeping the same exact feature-free approach. These two minimal adjustments should move MCRMSE down toward the target band while preserving the same submission formatting and semantics.'
- What this solution (achieved 0.42166) has done: 'We keep the same “per-position mean prior from train.json” core logic, but make it slightly closer to the test distribution by computing means only over the high-quality training rows (SN_filter==1) *and* clipping extreme/negative target values before averaging (a common minimal robustness step for this competition’s noisy tails). We also avoid using the last scored-position mean as a proxy for unscored positions; instead we fill unscored positions with the global (across all scored positions) mean per target to reduce position-edge bias. These changes don’t alter the modeling approach (still a feature-free prior) but typically reduce MCRMSE a bit, moving your 0.42166 closer to the 0.35586 target while keeping output format identical. The script still writes a valid `submission.csv` with the required 25680 rows and 6 columns.'
- What this solution (achieved 0.42178) has done: 'We’re currently worse than the target (0.42166 vs 0.35586, lower is better), so we should make a small, legitimate improvement while keeping the same “no-feature per-position prior” core logic. The minimal high-impact adjustment is to compute the per-position prior using a **signal-to-noise weighted mean** (still only using train targets, no model), which better matches the competition’s measurement reliability and typically reduces RMSE. To avoid destabilizing changes, we keep your SN_filter==1 selection, keep clipping, and keep the same submission formatting; we only change how the means are aggregated. This should move the score closer to the target without altering evaluation semantics or runtime materially.'
- What this solution (achieved 0.42178) has done: 'Your current score (0.42178, lower is better) is still worse than the target (0.35586), so we should improve predictions slightly while keeping the same “per-position prior from train.json” approach. The smallest likely win is to compute the prior using only training rows that better match the test distribution: keep `SN_filter==1` and additionally filter to `signal_to_noise >= 1.0` (the competition’s described test filtering), which typically reduces MCRMSE without changing the overall logic. To avoid over-shrinking or clipping away useful signal, we keep clipping but relax the upper clip bound a bit (still robust, but less bias), and we keep the same submission formatting and filling strategy. These changes are minimal, fast, and should move the score downward toward the target band.'
- What this solution (achieved 0.42475) has done: 'We keep your exact “per-position prior from train.json” approach, but adjust the aggregation to better match the test distribution without introducing a new model. Specifically, we (1) use the provided per-measurement error arrays to compute a simple inverse-variance weight per sample (more reliable samples influence the mean more than noisy ones), and (2) blend the position-wise mean with a small amount of global mean (a light shrinkage) to reduce overfitting to position noise—both are minimal changes that usually lower RMSE. We keep your SN_filter and signal_to_noise filtering, clipping, unscored-position filling, and submission formatting identical. This should move the score down (better) toward the 0.3559 target while remaining fast and deterministic.'
- What this solution (achieved 0.4611) has done: 'We’re currently worse than the target (0.42475 vs 0.35586, lower is better), so we should make a small improvement without changing the core “per-position prior from train.json” approach. The most likely minimal win is to fix the inverse-variance weighting: right now you collapse the per-position errors into a single per-sample scalar, but the data provides per-position errors, so we should weight each position by its own reliability (still the same prior, just correctly applied). To avoid overly aggressive weights on tiny errors, we clamp the per-position variance with a small floor and keep your existing SN_filter + signal_to_noise filters, clipping, shrinkage, and unscored-position filling. This keeps runtime low and deterministic, and should move MCRMSE down toward the target band.'
- What this solution (achieved 0.4319) has done: 'To move your MCRMSE down toward the 0.3559 target (lower is better) while keeping the exact same “per-position prior from train.json” core logic, I make the weighting less brittle and more consistent with the competition’s scoring distribution. Specifically, I (1) increase the variance floor used in inverse-variance weighting to prevent a few tiny reported errors from dominating the mean (which can hurt generalization), and (2) slightly increase the global-mean shrinkage so position noise is damped a bit more (still the same prior, just steadier). I also ensure we only use the first `seq_scored` positions from any arrays (safety against any edge cases) without changing semantics. Everything else (filters, clipping, filling unscored, submission format) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 0.42573) has done: 'To move your MCRMSE down toward the 0.3559 target (lower is better) while preserving the exact same “per-position prior from train.json” core logic, I’m making the weighting/shrinkage less biased and closer to the test scoring distribution. Concretely, I (1) restrict the training aggregation to the same scored positions the metric uses (first 68) while still outputting predictions for all 107 positions, (2) reduce the shrinkage strength slightly (your recent increase to 0.20 likely over-flattens position signal and can worsen RMSE), and (3) use a milder inverse-variance weighting by using a slightly larger variance floor (stabilizes) but also removing extra signal_to_noise multiplication to avoid double-counting reliability (which can overfit to training noise). These are minimal, fast changes that keep the same “no-model, weighted per-position mean prior” semantics and should improve the score directionally from 0.4319 toward the target band.'
- What this solution (achieved 0.4234) has done: 'To move your MCRMSE down from 0.42573 toward the 0.35586 target (lower is better) while keeping the exact same “weighted per-position mean prior from train.json” core logic, I make two minimal calibration/stability adjustments. First, I slightly reduce the inverse-variance weight aggressiveness by raising the variance floor a bit, which prevents a few tiny-reported errors from dominating and often improves generalization to test. Second, I slightly reduce the shrinkage toward the global mean so we preserve more genuine position signal (your current 0.12 can still be a bit over-flattening). Everything else—filters, clipping, using only scored positions for aggregation, filling unscored positions, and exact submission formatting—stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.42376) has done: 'Your current score (0.4234, lower is better) is still worse than the target (0.3559), so we should make a small improvement without changing the core “weighted per-position mean prior from train.json” approach. The most likely minimal win is to use a **winsorized weighted mean** per position (cap extreme values only for the averaging step) so outliers don’t skew the prior, while still clipping final values exactly as before. To prevent overly-aggressive inverse-variance weights from tiny error values, we slightly raise the variance floor (a stability/calibration tweak, not a new model). Everything else—filters, using only scored positions for aggregation, shrinkage approach, unscored filling, and submission formatting—remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.42559) has done: 'We’re still worse than the target (0.42376 vs 0.35586, lower is better), so we make a very small, low-risk calibration improvement without changing the core “winsorized inverse-variance weighted per-position mean prior from train.json” approach. The main tweak is to reduce over-smoothing by slightly lowering the shrinkage toward the global mean, which should preserve more true position signal and typically improves RMSE for this competition. We also make the winsorization a bit less aggressive (10–90% instead of 5–95%) so the mean isn’t pulled toward the center too much, while keeping the same final clipping and all formatting identical. Everything still runs fast, deterministically, and writes a valid `submission.csv` with the required columns and 25680 rows.'
- What this solution (achieved 0.42564) has done: 'We’re currently worse than the target (0.42559 vs 0.35586, lower is better), so we make one small, low-risk calibration improvement while keeping the exact same “winsorized inverse-variance weighted per-position mean prior” core logic. The main change is to compute the shrinkage blend using the **weighted global mean** (with the same inverse-variance weights) instead of the unweighted mean of `m_pos`, which better matches how `m_pos` itself is estimated and typically reduces bias. Everything else (filters, per-position weighting, winsorization quantiles, variance floor, clipping, scored/unscored filling, and submission formatting) remains the same. This is minimal, deterministic, fast, and should move MCRMSE downward toward the target band.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
from pathlib import Path



## === cell 1
base_dir_candidates = [
    Path("/kaggle/input/stanford-covid-vaccine"),
    Path("/kaggle/data/stanford-covid-vaccine"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]
sample_path = None
for base in base_dir_candidates:
    p = base / "sample_submission.csv"
    if p.exists():
        sample_path = p
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input/data paths."
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
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

for c in required_cols[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

df.head()



## === cell 2
sequences = list(set(["_".join(v.split("_")[:-1]) for v in df.id_seqpos.values]))
sequences[-10:]



## === cell 3
train_path = None
for base in base_dir_candidates:
    p = base / "train.json"
    if p.exists():
        train_path = p
        break
if train_path is None:
    raise FileNotFoundError(
        "Could not locate train.json in expected Kaggle input/data paths."
    )

train = pd.read_json(train_path, lines=True)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
for c in target_cols:
    if c not in train.columns:
        raise ValueError(f"train.json missing expected target column: {c}")

train_used = train
if "SN_filter" in train_used.columns:
    train_used = train_used[train_used["SN_filter"] == 1].copy()
if "signal_to_noise" in train_used.columns:
    train_used = train_used[train_used["signal_to_noise"] >= 1.0].copy()
if len(train_used) == 0:
    train_used = (
        train[train["SN_filter"] == 1].copy()
        if "SN_filter" in train.columns
        else train.copy()
    )
    if len(train_used) == 0:
        train_used = train.copy()

seq_scored = int(train_used["seq_scored"].iloc[0])

clip_lo, clip_hi = -0.5, 20.0

eps = np.float32(1e-6)

var_floor = np.float32(8e-2)  # keep: stability against tiny reported errors dominating

err_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}

pos_means = {}
global_means = {}

shrink_alpha = np.float32(0.05)  # keep: small shrinkage toward global mean

seqpos = df["id_seqpos"].str.rsplit("_", n=1, expand=True)[1].astype(np.int16).values
in_scored = seqpos < seq_scored

for c in target_cols:
    arr = np.vstack(train_used[c].values).astype(np.float32)[:, :seq_scored]  # (n, 68)
    arr = np.clip(arr, clip_lo, clip_hi)

    w_base = np.ones(len(train_used), dtype=np.float32)

    err_col = err_map.get(c)
    if err_col in train_used.columns:
        err = np.vstack(train_used[err_col].values).astype(np.float32)[:, :seq_scored]
        err = np.where(np.isfinite(err), err, np.nan).astype(np.float32)

        var = np.square(err).astype(np.float32)
        var = np.where(np.isfinite(var), var, np.nan).astype(np.float32)
        var = np.where(var > var_floor, var, var_floor).astype(np.float32)

        w = (w_base[:, None] / (var + eps)).astype(np.float32)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0).astype(np.float32)

        wsum_pos = np.sum(w, axis=0).astype(np.float32)
        wsum_pos = np.where(wsum_pos > 0, wsum_pos, 1.0).astype(np.float32)

        arr_w = np.empty_like(arr, dtype=np.float32)
        for j in range(seq_scored):
            wj = w[:, j]
            aj = arr[:, j]
            mask = (wj > 0) & np.isfinite(aj)
            if np.any(mask):
                wv = wj[mask].astype(np.float64)
                av = aj[mask].astype(np.float64)
                order = np.argsort(av)
                avs = av[order]
                wvs = wv[order]
                cdf = np.cumsum(wvs)
                total = cdf[-1] if cdf[-1] > 0 else 1.0

                q10 = float(avs[np.searchsorted(cdf, 0.10 * total, side="left")])
                q90 = float(avs[np.searchsorted(cdf, 0.90 * total, side="left")])
                arr_w[:, j] = np.clip(aj, q10, q90).astype(np.float32)
            else:
                arr_w[:, j] = aj.astype(np.float32)

        m_pos = (np.sum(w * arr_w, axis=0) / wsum_pos).astype(np.float32)

        wsum_all = float(np.sum(w))
        if not np.isfinite(wsum_all) or wsum_all <= 0:
            m_global = np.float32(np.mean(m_pos))
        else:
            m_global = np.float32(np.sum(w * arr_w) / np.float32(wsum_all))

    else:
        w = np.ones(len(train_used), dtype=np.float32)
        w_sum = float(np.sum(w)) if float(np.sum(w)) > 0 else 1.0
        m_pos = (w[:, None] * arr).sum(axis=0) / w_sum
        m_pos = m_pos.astype(np.float32)
        m_global = np.float32(np.mean(m_pos))

    m_pos = np.where(np.isfinite(m_pos), m_pos, 0.0).astype(np.float32)
    if not np.isfinite(float(m_global)):
        m_global = np.float32(np.mean(m_pos))

    m_pos = (np.float32(1.0) - shrink_alpha) * m_pos + shrink_alpha * m_global

    pos_means[c] = m_pos
    global_means[c] = float(np.mean(m_pos))

    fill_unscored = np.float32(global_means[c])

    preds = np.empty(len(df), dtype=np.float32)
    preds[in_scored] = m_pos[seqpos[in_scored]]
    preds[~in_scored] = fill_unscored
    df[c] = preds



## === cell 4
df[required_cols[1:]] = (
    df[required_cols[1:]]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype("float32")
)
df = df[required_cols]



## === cell 5
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
