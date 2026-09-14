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

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.3717

# 6. Current score

0.50941

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'The script now skips the broken custom model code and instead builds a simple baseline that predicts the average value of each target at every scored position (0‑67). For positions beyond the scored region (68‑106) it predicts zero. This eliminates the import and data‑loading errors, correctly creates the required `id_seqpos` identifiers, and writes a properly formatted `submission.csv` file.'
- What this solution (achieved 0.41641) has done: 'I add a lightweight per‑position, per‑nucleotide average model.  
First we compute overall position‑wise means (already done).  
Then we augment them with means for each base (A,C,G,U) at every scored position, falling back to the overall means when a specific base‑level estimate is missing.  
Predictions for the test set now use these more informative averages for the first 68 positions, keeping zeros for the unscored tail. This small change should lower the MCRMSE from 0.42418 toward the target 0.3717 while preserving the original pipeline and output format.'
- What this solution (achieved 0.46822) has done: 'I replace the simple un‑weighted averages with inverse‑variance weighted averages (using the provided error columns) for both the overall per‑position means and the per‑base‑per‑position means.  This keeps the same overall pipeline but gives more reliable estimates, which should reduce the MCRMSE and move the score closer to the target.'
- What this solution (achieved 0.46822) has done: 'I keep the overall pipeline and weighting logic but make the predictions a small blend of the base‑specific weighted means and the overall weighted means, using the total weight as a confidence indicator (beta = 1.0). This reduces over‑reliance on noisy base‑specific estimates, which should lower the MCRMSE toward the target while preserving the original structure and format.'
- What this solution (achieved 0.46822) has done: 'I keep the overall pipeline but adjust the blending to rely more on the robust overall position‑wise averages, which were hurting performance when weighted too heavily by noisy base‑specific estimates. I increase the blending factor `beta` (giving the overall mean more influence) and apply a tiny smoothing window to the overall means to reduce variance across neighboring positions. These minimal changes are expected to lower the MCRMSE toward the target while preserving the original logic and output format.'
- What this solution (achieved 0.4164) has done: 'I replace the variance‑weighted averaging with simple unweighted averages for both the global per‑position means and the per‑base per‑position means, then lower the blending factor `beta` to 1.0 so the predictions rely more on the base‑specific statistics. These minimal adjustments keep the overall pipeline intact while reducing noise, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41641) has done: 'I replace the blended prediction with a simple hierarchy: when a base‑specific mean exists for the position I use it directly, otherwise I fall back to the smoothed global mean. This removes the unnecessary β blending and lets the more informative nucleotide‑level statistics dominate, which should lower the MCRMSE toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.41958) has done: 'I blend the base‑specific means with the smoothed overall position means (using an α of 0.6) so predictions rely on base information but are regularized toward the more stable overall averages. This small adjustment keeps the core logic unchanged while aiming to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41641) has done: 'I increase the blend factor `alpha` to 1.0 so that predictions use the base‑specific mean whenever it is available and fall back to the smoothed overall mean only when a base‑specific value is missing. This small change removes the averaging with the overall mean that was inflating the error, moving the MCRMSE closer to the target lower value.'
- What this solution (achieved 0.50941) has done: 'The patch upgrades the averaging step to use inverse‑variance weighting (leveraging the provided error columns), applies a slightly wider smoothing kernel (size 5) to both the global and base‑specific means, and blends the base‑specific prediction with the smoothed global mean using a modest blend factor (α = 0.9). These minimal statistical refinements keep the original workflow intact while targeting a lower MCRMSE, moving the score closer to the desired 0.3717.'

# 9. Code solution

## === cell 0
import json, os
import numpy as np, pandas as pd
from pathlib import Path



## === cell 1
base_path = Path("/kaggle/input/stanford-covid-vaccine")
train_path = base_path / "train.json"
test_path = base_path / "test.json"
sample_sub_path = base_path / "sample_submission.csv"

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
sample_sub = pd.read_csv(sample_sub_path)



## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}
eps = 1e-8

weighted_means = {}
for col in target_cols:
    vals = np.stack(train[col].values)  # (n_samples, 68)
    errs = np.stack(train[error_map[col]].values)  # (n_samples, 68)
    weights = 1.0 / (errs**2 + eps)
    weighted_sum = np.sum(vals * weights, axis=0)
    weight_total = np.sum(weights, axis=0) + eps
    weighted_means[col] = weighted_sum / weight_total  # (68,)

kernel = np.ones(5) / 5.0
smoothed_means = {}
for col in target_cols:
    smoothed = np.convolve(weighted_means[col], kernel, mode="same")
    smoothed_means[col] = smoothed



## === cell 3
scored_len = 68
seq_len = 107
base2idx = {"A": 0, "C": 1, "G": 2, "U": 3}
alpha = 0.9  # blend factor: favour base‑specific mean but keep some global smoothing

base_weighted_sums = {
    col: np.zeros((scored_len, 4), dtype=float) for col in target_cols
}
base_weights = {col: np.zeros((scored_len, 4), dtype=float) for col in target_cols}

for _, row in train.iterrows():
    seq = row["sequence"]
    for pos in range(scored_len):
        base = seq[pos]
        if base not in base2idx:
            continue
        bidx = base2idx[base]
        for col in target_cols:
            val = row[col][pos]
            err = row[error_map[col]][pos]
            w = 1.0 / (err**2 + eps)
            base_weighted_sums[col][pos, bidx] += val * w
            base_weights[col][pos, bidx] += w

base_means = {}
for col in target_cols:
    avg = np.full((scored_len, 4), np.nan, dtype=float)
    mask = base_weights[col] > 0
    avg[mask] = base_weighted_sums[col][mask] / base_weights[col][mask]
    smoothed = np.empty_like(avg)
    for b in range(4):
        series = avg[:, b]
        nan_mask = np.isnan(series)
        series_filled = np.where(nan_mask, 0.0, series)
        conv = np.convolve(series_filled, kernel, mode="same")
        conv[nan_mask] = np.nan
        smoothed[:, b] = conv
    base_means[col] = smoothed



## === cell 4
rows = []
for _, row in test.iterrows():
    mol_id = row["id"]
    seq = row["sequence"]
    for pos in range(seq_len):
        id_seqpos = f"{mol_id}_{pos}"
        if pos < scored_len:
            base = seq[pos]
            bidx = base2idx.get(base, None)
            if bidx is not None and not np.isnan(base_means[target_cols[0]][pos, bidx]):
                row_vals = [
                    (
                        alpha * base_means[col][pos, bidx]
                        + (1 - alpha) * smoothed_means[col][pos]
                        if not np.isnan(base_means[col][pos, bidx])
                        else smoothed_means[col][pos]
                    )
                    for col in target_cols
                ]
            else:
                row_vals = [smoothed_means[col][pos] for col in target_cols]
        else:
            row_vals = [0.0] * len(target_cols)
        rows.append([id_seqpos] + row_vals)

sub_df = pd.DataFrame(rows, columns=["id_seqpos"] + target_cols)



## === cell 5
sub_df = sub_df[sample_sub.columns]

output_path = Path("/kaggle/working/submission.csv")
sub_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {sub_df.shape}")
