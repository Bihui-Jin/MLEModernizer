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

0.43513

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I fix the missing CSV path, ensure all variables are defined, and add a simple baseline prediction (using global means from the training data) so the script runs end‑to‑end and writes a valid `submission.csv`. This resolves the FileNotFoundError and NameError issues while keeping the core logic unchanged.'
- What this solution (achieved 0.42418) has done: 'I replace the global‑mean baseline with a simple per‑position mean baseline: for each of the three scored targets I compute the average value at every sequence position (0‑67) across the training set and use those position‑wise averages as predictions. This keeps the original workflow (loading data, handling the sample submission, applying a manual tweak for a specific id) while providing more informative predictions, which should lower the MCRMSE toward the target score. The script now also extracts the position from the `id_seqpos` column and maps it to the appropriate per‑position mean.'
- What this solution (achieved 0.41641) has done: 'The fix adds a safe fallback for positions beyond the 68‑base scored region, preventing the IndexError, and ensures a prediction is always returned by using the overall mean when a position is out of bounds. The `get_pred` function is updated accordingly, and the script now reliably creates the `submission` DataFrame and writes `submission.csv`.'
- What this solution (achieved 0.41633) has done: 'I adjust the prediction function to blend the exact (position, nucleotide) mean with the overall positional mean, which smooths noisy per‑nucleotide estimates and should modestly lower the MCRMSE. I also remove the hard‑coded override for a single id, as that artificial spike can hurt overall error. These minimal edits keep the original workflow intact while moving the score closer to the target.'
- What this solution (achieved 0.42067) has done: 'I smooth the positional mean predictions with a short rolling window and reduce the blending weight so the model relies more on the less‑noisy position‑wise averages. This keeps the original per‑nucleotide maps but makes the baseline predictions slightly more accurate, moving the MCRMSE closer to the target score while preserving the core workflow.'
- What this solution (achieved 0.43218) has done: 'I increase the smoothing window for the positional means to better capture trends and lower the blending weight so predictions rely more on the stable smoothed positional averages (instead of the noisier per‑nucleotide means). These modest adjustments keep the original workflow intact while nudging the MCRMSE closer to the target score.'
- What this solution (achieved 0.42336) has done: 'I lower the smoothing window to 5 (so positional means keep more detail) and increase the blending weight on the per‑nucleotide mean from 0.3 to 0.5. These minimal tweaks keep the original workflow while giving the model a bit more specific signal, which should reduce the MCRMSE toward the target value.'
- What this solution (achieved 0.43517) has done: 'I reduce the influence of the noisy per‑nucleotide means by lowering the blending weight (favoring the smoother positional averages) and increase the smoothing window slightly so the positional means capture broader trends. These minimal tweaks keep the overall workflow unchanged while aiming to lower the MCRMSE toward the target value.'
- What this solution (achieved 0.42667) has done: 'I restore a slightly stronger per‑nucleotide influence and use a shorter smoothing window, which should give more detailed position predictions and lower the MCRMSE toward the target. Specifically, the smoothing window is reduced from 7 to 5 and the blending weight defaults from 0.2 to 0.4 so the nucleotide‑wise mean contributes more to each prediction.'
- What this solution (achieved 0.43517) has done: 'I lower the blend weight for the per‑nucleotide means (from 0.4 to 0.2) so the smoother positional averages dominate, and I increase the smoothing window to 7 for a slightly more stable positional baseline. These minimal tweaks keep the overall workflow unchanged while aiming to reduce the MCRMSE toward the target.'
- What this solution (achieved 0.42856) has done: 'I slightly reduce the smoothing window to capture more positional detail and increase the per‑nucleotide blending weight so predictions rely more on the informative nucleotide‑wise averages. These modest tweaks keep the overall workflow unchanged while expectedly lowering the MCRMSE toward the target.'
- What this solution (achieved 0.42284) has done: 'I reduce the smoothing window to capture more positional detail and increase the blending weight so the nucleotide‑specific mean has a stronger influence. These small adjustments should lower the MCRMSE toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.43513) has done: 'I will lower the blending weight for nucleotide‑specific means (to rely more on the smoother positional averages) and increase the smoothing window from 3 to 5 so the position‑wise baseline is less noisy. These minimal tweaks keep the original workflow intact while expectedly reducing the MCRMSE toward the target score.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

reactivity_pos_mean = np.mean(np.stack(df_train["reactivity"].values), axis=0)
deg_Mg_pH10_pos_mean = np.mean(np.stack(df_train["deg_Mg_pH10"].values), axis=0)
deg_pH10_pos_mean = np.mean(np.stack(df_train["deg_pH10"].values), axis=0)
deg_Mg_50C_pos_mean = np.mean(np.stack(df_train["deg_Mg_50C"].values), axis=0)
deg_50C_pos_mean = np.mean(np.stack(df_train["deg_50C"].values), axis=0)


def _smooth(arr, window=5):
    """Smooth array with a centered rolling mean.
    A larger window (5) yields a more stable positional baseline."""
    ser = pd.Series(arr)
    return ser.rolling(window, min_periods=1, center=True).mean().values


reactivity_pos_mean_s = _smooth(reactivity_pos_mean)
deg_Mg_pH10_pos_mean_s = _smooth(deg_Mg_pH10_pos_mean)
deg_pH10_pos_mean_s = _smooth(deg_pH10_pos_mean)
deg_Mg_50C_pos_mean_s = _smooth(deg_Mg_50C_pos_mean)
deg_50C_pos_mean_s = _smooth(deg_50C_pos_mean)

records = []
for row in df_train.itertuples():
    seq = row.sequence
    L = row.seq_scored
    for i in range(L):
        nuc = seq[i]
        records.append(
            {
                "pos": i,
                "nuc": nuc,
                "reactivity": row.reactivity[i],
                "deg_Mg_pH10": row.deg_Mg_pH10[i],
                "deg_pH10": row.deg_pH10[i],
                "deg_Mg_50C": row.deg_Mg_50C[i],
                "deg_50C": row.deg_50C[i],
            }
        )
df_expanded = pd.DataFrame(records)

grouped = df_expanded.groupby(["pos", "nuc"]).mean().reset_index()
reactivity_pos_nuc_map = {
    (row.pos, row.nuc): row.reactivity for row in grouped.itertuples()
}
deg_Mg_pH10_pos_nuc_map = {
    (row.pos, row.nuc): row.deg_Mg_pH10 for row in grouped.itertuples()
}
deg_pH10_pos_nuc_map = {
    (row.pos, row.nuc): row.deg_pH10 for row in grouped.itertuples()
}
deg_Mg_50C_pos_nuc_map = {
    (row.pos, row.nuc): row.deg_Mg_50C for row in grouped.itertuples()
}
deg_50C_pos_nuc_map = {(row.pos, row.nuc): row.deg_50C for row in grouped.itertuples()}

try:
    df_sub = pd.read_csv("../input/sample_submission.csv")
except FileNotFoundError:
    df_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")

try:
    df_test = pd.read_json("../input/test.json", lines=True)
except FileNotFoundError:
    df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)

df_sub["pos"] = df_sub["id_seqpos"].str.rsplit("_", n=1, expand=True)[1].astype(int)
df_sub["id"] = df_sub["id_seqpos"].str.rsplit("_", n=1, expand=True)[0]

df_merge = df_sub.merge(df_test[["id", "sequence"]], on="id", how="left")
df_merge["nuc"] = df_merge.apply(
    lambda r: r.sequence[r.pos] if pd.notnull(r.sequence) else np.nan, axis=1
)


def get_pred(pos, nuc, map_dict, pos_mean, weight_map=0.2):
    """
    Blended prediction:
    - Uses a lower per‑nucleotide weight (0.2) so the smoother positional mean dominates.
    - Falls back to the smoothed positional mean when nucleotide info is missing.
    - Returns the overall mean for positions beyond the training range.
    """
    if pos >= len(pos_mean):
        return float(pos_mean.mean())
    key = (pos, nuc)
    if pd.isna(nuc) or key not in map_dict:
        return float(pos_mean[pos])
    return float(weight_map * map_dict[key] + (1 - weight_map) * pos_mean[pos])


df_merge["reactivity"] = df_merge.apply(
    lambda r: get_pred(r.pos, r.nuc, reactivity_pos_nuc_map, reactivity_pos_mean_s),
    axis=1,
)
df_merge["deg_Mg_pH10"] = df_merge.apply(
    lambda r: get_pred(r.pos, r.nuc, deg_Mg_pH10_pos_nuc_map, deg_Mg_pH10_pos_mean_s),
    axis=1,
)
df_merge["deg_pH10"] = df_merge.apply(
    lambda r: get_pred(r.pos, r.nuc, deg_pH10_pos_nuc_map, deg_pH10_pos_mean_s), axis=1
)
df_merge["deg_Mg_50C"] = df_merge.apply(
    lambda r: get_pred(r.pos, r.nuc, deg_Mg_50C_pos_nuc_map, deg_Mg_50C_pos_mean_s),
    axis=1,
)
df_merge["deg_50C"] = df_merge.apply(
    lambda r: get_pred(r.pos, r.nuc, deg_50C_pos_nuc_map, deg_50C_pos_mean_s), axis=1
)

submission = df_merge[
    ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
]



## === cell 1
submission.to_csv("submission.csv", index=False)
