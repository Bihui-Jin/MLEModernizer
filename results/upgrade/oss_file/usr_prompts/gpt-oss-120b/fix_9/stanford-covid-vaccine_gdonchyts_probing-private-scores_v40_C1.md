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

0.41638

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I replace the missing baseline CSV load with the provided sample submission template, compute simple mean‑based predictions from the training data, fill the submission DataFrame with those values, and finally write a proper `submission.csv`. This resolves the file‑not‑found and undefined‑variable errors while keeping the core logic unchanged.'
- What this solution (achieved 0.63824) has done: 'I replace the single global‑mean predictions with position‑specific means for the first 68 scored bases. By computing a mean per nucleotide position from the training data and using the position extracted from each `id_seqpos`, the predictions become better calibrated to the true variation across positions, which should lower the MCRMSE toward the target score while keeping the original workflow unchanged. Unscored positions (≥68) still use the overall global mean to retain a valid submission format.'
- What this solution (achieved 0.42418) has done: 'I replace the `np.where` calls with explicit masked assignments so that position‑specific means are only accessed for positions < 68, preventing the out‑of‑bounds indexing error. This fixes the runtime crash and allows the model to use the intended per‑position averages, which should lower the MCRMSE toward the target score. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.41641) has done: 'I add per‑position, per‑nucleotide mean predictions instead of only per‑position means. By computing a 68 × 4 matrix (position × base) for each target from the training set and using the actual base at each scored position in the test set, the predictions become better calibrated, which should lower the MCRMSE toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.41638) has done: 'I add a small smoothing (shrinkage) factor when combining the per‑position‑per‑base means with the overall position means. This reduces variance for positions with few observations and is expected to lower the MCRMSE toward the target while keeping the overall averaging logic unchanged.'
- What this solution (achieved 0.4164) has done: 'I lower the smoothing constant from 5.0 to 2.0 so the per‑position‑per‑base averages rely more on observed data, and I correctly handle bases that are missing or unknown (“N”) by falling back to the pure position‑wise mean instead of mistakenly using the last base column. These minimal adjustments keep the original workflow intact while expectedly reducing the MCRMSE toward the target.'
- What this solution (achieved 0.4164) has done: 'I lower the shrinkage constant `alpha` from 2.0 to 1.0 so the per‑position‑per‑base averages rely more on the observed data, which typically reduces the MCRMSE a bit and moves the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.41638) has done: 'I increase the shrinkage constant `alpha` from 1.0 to 5.0 so that the per‑position‑per‑base means are pulled more toward the robust position‑wise averages, which should reduce variance and lower the MCRMSE toward the target while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)


def column_mean(col_name):
    values = np.concatenate(train[col_name].values)
    return values.mean()


def column_position_mean(col_name):
    stacked = np.vstack(train[col_name].values)
    return stacked.mean(axis=0)  # length 68


mean_reactivity = column_mean("reactivity")
mean_deg_Mg_pH10 = column_mean("deg_Mg_pH10")
mean_deg_pH10 = column_mean("deg_pH10")
mean_deg_Mg_50C = column_mean("deg_Mg_50C")
mean_deg_50C = column_mean("deg_50C")

pos_mean_reactivity = column_position_mean("reactivity")
pos_mean_deg_Mg_pH10 = column_position_mean("deg_Mg_pH10")
pos_mean_deg_pH10 = column_position_mean("deg_pH10")
pos_mean_deg_Mg_50C = column_position_mean("deg_Mg_50C")
pos_mean_deg_50C = column_position_mean("deg_50C")

base_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3}
n_pos = 68
n_base = 4

sum_reactivity = np.zeros((n_pos, n_base))
cnt_reactivity = np.zeros((n_pos, n_base))
sum_deg_Mg_pH10 = np.zeros((n_pos, n_base))
cnt_deg_Mg_pH10 = np.zeros((n_pos, n_base))
sum_deg_pH10 = np.zeros((n_pos, n_base))
cnt_deg_pH10 = np.zeros((n_pos, n_base))
sum_deg_Mg_50C = np.zeros((n_pos, n_base))
cnt_deg_Mg_50C = np.zeros((n_pos, n_base))
sum_deg_50C = np.zeros((n_pos, n_base))
cnt_deg_50C = np.zeros((n_pos, n_base))

for seq, reac, dmgMgpH, dmgpH, dmgMg50, dmg50 in zip(
    train["sequence"],
    train["reactivity"],
    train["deg_Mg_pH10"],
    train["deg_pH10"],
    train["deg_Mg_50C"],
    train["deg_50C"],
):
    for pos in range(n_pos):
        base = seq[pos]
        idx = base_to_idx.get(base)
        if idx is None:
            continue
        sum_reactivity[pos, idx] += reac[pos]
        cnt_reactivity[pos, idx] += 1
        sum_deg_Mg_pH10[pos, idx] += dmgMgpH[pos]
        cnt_deg_Mg_pH10[pos, idx] += 1
        sum_deg_pH10[pos, idx] += dmgpH[pos]
        cnt_deg_pH10[pos, idx] += 1
        sum_deg_Mg_50C[pos, idx] += dmgMg50[pos]
        cnt_deg_Mg_50C[pos, idx] += 1
        sum_deg_50C[pos, idx] += dmg50[pos]
        cnt_deg_50C[pos, idx] += 1

alpha = 5.0

pos_base_mean_reactivity = (sum_reactivity + alpha * pos_mean_reactivity[:, None]) / (
    cnt_reactivity + alpha
)
pos_base_mean_deg_Mg_pH10 = (
    sum_deg_Mg_pH10 + alpha * pos_mean_deg_Mg_pH10[:, None]
) / (cnt_deg_Mg_pH10 + alpha)
pos_base_mean_deg_pH10 = (sum_deg_pH10 + alpha * pos_mean_deg_pH10[:, None]) / (
    cnt_deg_pH10 + alpha
)
pos_base_mean_deg_Mg_50C = (sum_deg_Mg_50C + alpha * pos_mean_deg_Mg_50C[:, None]) / (
    cnt_deg_Mg_50C + alpha
)
pos_base_mean_deg_50C = (sum_deg_50C + alpha * pos_mean_deg_50C[:, None]) / (
    cnt_deg_50C + alpha
)




## === cell 1
df = sample_sub.copy()

positions = df["id_seqpos"].str.rsplit("_", n=1).str[-1].astype(int).values
ids = df["id_seqpos"].str.rsplit("_", n=1).str[0].values

mask_scored = positions < 68

id_to_seq = dict(zip(df_test["id"], df_test["sequence"]))

bases = np.empty(len(df), dtype=str)
for i, (pid, pos) in enumerate(zip(ids, positions)):
    seq = id_to_seq.get(pid, "")
    bases[i] = seq[pos] if pos < len(seq) else "N"

base_idx = np.array([base_to_idx.get(b, -1) for b in bases])

df["reactivity"] = mean_reactivity
df["deg_Mg_pH10"] = mean_deg_Mg_pH10
df["deg_pH10"] = mean_deg_pH10
df["deg_Mg_50C"] = mean_deg_Mg_50C
df["deg_50C"] = mean_deg_50C

df.loc[mask_scored, "reactivity"] = pos_mean_reactivity[positions[mask_scored]]
df.loc[mask_scored, "deg_Mg_pH10"] = pos_mean_deg_Mg_pH10[positions[mask_scored]]
df.loc[mask_scored, "deg_pH10"] = pos_mean_deg_pH10[positions[mask_scored]]
df.loc[mask_scored, "deg_Mg_50C"] = pos_mean_deg_Mg_50C[positions[mask_scored]]
df.loc[mask_scored, "deg_50C"] = pos_mean_deg_50C[positions[mask_scored]]

mask_valid_base = mask_scored & (base_idx >= 0)
scored_pos = positions[mask_valid_base]
scored_idx = base_idx[mask_valid_base]

df.loc[mask_valid_base, "reactivity"] = pos_base_mean_reactivity[scored_pos, scored_idx]
df.loc[mask_valid_base, "deg_Mg_pH10"] = pos_base_mean_deg_Mg_pH10[
    scored_pos, scored_idx
]
df.loc[mask_valid_base, "deg_pH10"] = pos_base_mean_deg_pH10[scored_pos, scored_idx]
df.loc[mask_valid_base, "deg_Mg_50C"] = pos_base_mean_deg_Mg_50C[scored_pos, scored_idx]
df.loc[mask_valid_base, "deg_50C"] = pos_base_mean_deg_50C[scored_pos, scored_idx]




## === cell 2
df.to_csv("submission.csv", index=False)
