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

0.42178

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I replace the missing baseline file load with a simple, fully self‑contained pipeline: read the training JSON, compute the overall mean for each of the five scored targets, then generate a submission for every position of every test sample using those means (the required length = `seq_length`). This removes the `FileNotFoundError`, defines `df` correctly, and guarantees a correctly formatted CSV (`id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C`). The approach keeps the original logic (no model change) while providing a valid submission that score on the leaderboard.'
- What this solution (achieved 0.44603) has done: 'I compute per‑position averages for each of the three scored targets (reactivity, deg_Mg_pH10, deg_pH10) and use those values for the positions that are actually scored (pos < seq_scored). For positions beyond the scored region I fall back to the overall column mean, keeping the original simple baseline. This small calibration change should lower the MCRMSE toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.44977) has done: 'I add a simple shrinkage of the per‑position means toward the overall column mean (using α≈0.7). This keeps the same overall logic but gives slightly more robust predictions for the scored positions, which should lower the MCRMSE and move the score closer to the target. The rest of the pipeline and the submission format remain unchanged.'
- What this solution (achieved 0.42518) has done: 'I increase the reliance on the per‑position averages by raising the shrinkage factor α to 0.9 (more weight on position‑specific means) and extend the same shrinkage to the remaining two targets (`deg_Mg_50C`, `deg_50C`). This small calibration change keeps the original baseline logic while expectedly lowering the MCRMSE toward the target score.'
- What this solution (achieved 0.42147) has done: 'I filter the training data to keep only high‑quality samples (SN_filter = 1) and compute both overall and per‑position means on this cleaned set. Then I increase the shrinkage factor α from 0.9 to 0.97 so the predictions rely more on the position‑specific statistics, which should reduce the MCRMSE and move the score closer to the target while preserving the original pipeline logic.'
- What this solution (achieved 0.42196) has done: 'The change lowers the shrinkage factor α from 0.97 to 0.85, giving more weight to the overall column means and less to the noisy per‑position averages. This simple regularisation is expected to reduce over‑fitting on the training positions, thereby lowering the MCRMSE and moving the score closer to the target while keeping the original pipeline untouched.'
- What this solution (achieved 0.42936) has done: 'I lowered the shrinkage factor `alpha` from 0.85 to 0.60 so the predictions rely more on the overall column means and less on the noisy per‑position averages. This simple regularisation reduces over‑fitting on the training positions and is expected to lower the MCRMSE, moving the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.43649) has done: 'I smooth the per‑position averages with a small moving‑average filter to reduce noise, then increase the shrinkage weight (α) so the predictions rely more on these cleaned position‑specific statistics. This keeps the original baseline structure while providing more reliable values for the scored positions, which should lower the MCRMSE toward the target.'
- What this solution (achieved 0.42166) has done: 'I increase reliance on the position‑specific statistics by removing the shrinkage (set α = 1.0) and disabling the smoothing (use a window of 1, which leaves the per‑position means unchanged). This keeps the original pipeline unchanged apart from these calibrated parameters, and should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.44546) has done: 'I add per‑position variance‑based weighting so that noisy positions are shrunk toward the overall column mean while reliable positions keep more of their position‑specific average. This small calibration keeps the original baseline logic but should reduce over‑fitting and lower the MCRMSE toward the target.'
- What this solution (achieved 0.45443) has done: 'I replace the per‑position means with per‑position medians, which are less sensitive to outliers, and then keep the same shrink‑toward‑overall‑mean weighting. This small statistical tweak should lower the MCRMSE toward the target while leaving the overall pipeline unchanged.'
- What this solution (achieved 0.44546) has done: 'The update switches from per‑position medians to per‑position means while keeping the same shrink‑toward‑overall‑mean logic, which better captures the central tendency of the training data and is expected to lower the MCRMSE toward the target. No core modeling logic is altered, and the submission CSV is still written exactly as before.'
- What this solution (achieved 0.42166) has done: 'I remove the shrink‑toward‑overall‑mean step, using the raw per‑position averages directly (the three scored targets benefit most from this). This reduces the regularisation that was lowering the score, moving the metric closer to the target while keeping the rest of the pipeline unchanged. The script is renumbered to start at cell 1 as required.'
- What this solution (achieved 0.42178) has done: 'I replace the unweighted per‑position averages with signal‑to‑noise‑weighted averages while keeping the overall structure unchanged. Weighted position means give more influence to high‑quality samples, which should lower the MCRMSE and move the score closer to the target. The rest of the pipeline (fallback to overall column means for positions beyond `seq_scored` and CSV writing) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

df_test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
df_train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

df_train_clean = df_train[df_train["SN_filter"] == 1].reset_index(drop=True)




## === cell 1
def column_mean(arr_series):
    """Overall mean across all values in a series of lists."""
    return np.mean(np.concatenate(arr_series.values))


def weighted_position_mean(arr_series, weights):
    """
    arr_series: pd.Series of list‑like arrays (shape: n_samples × n_positions)
    weights: 1‑D array of shape (n_samples,)
    Returns a 1‑D array of length n_positions with weighted averages.
    """
    stacked = np.stack(arr_series.values)  # (n_samples, n_positions)
    w = weights[:, None]  # (n_samples, 1)
    weighted_sum = np.sum(stacked * w, axis=0)  # (n_positions,)
    weight_total = np.sum(w, axis=0)  # scalar
    return weighted_sum / weight_total


mean_reactivity = column_mean(df_train_clean["reactivity"])
mean_deg_Mg_pH10 = column_mean(df_train_clean["deg_Mg_pH10"])
mean_deg_pH10 = column_mean(df_train_clean["deg_pH10"])
mean_deg_Mg_50C = column_mean(df_train_clean["deg_Mg_50C"])
mean_deg_50C = column_mean(df_train_clean["deg_50C"])

sn_weights = df_train_clean["signal_to_noise"].values

pos_mean_reactivity = weighted_position_mean(df_train_clean["reactivity"], sn_weights)
pos_mean_deg_Mg_pH10 = weighted_position_mean(df_train_clean["deg_Mg_pH10"], sn_weights)
pos_mean_deg_pH10 = weighted_position_mean(df_train_clean["deg_pH10"], sn_weights)
pos_mean_deg_Mg_50C = weighted_position_mean(df_train_clean["deg_Mg_50C"], sn_weights)
pos_mean_deg_50C = weighted_position_mean(df_train_clean["deg_50C"], sn_weights)

shrunk_reactivity = pos_mean_reactivity
shrunk_deg_Mg_pH10 = pos_mean_deg_Mg_pH10
shrunk_deg_pH10 = pos_mean_deg_pH10
shrunk_deg_Mg_50C = pos_mean_deg_Mg_50C
shrunk_deg_50C = pos_mean_deg_50C



## === cell 2
rows = []
for _, row in df_test.iterrows():
    sample_id = row["id"]
    seq_len = int(row["seq_length"])  # total length (107)
    seq_scored = int(row["seq_scored"])  # positions that are scored (usually 68)

    for pos in range(seq_len):
        if pos < seq_scored:
            reactivity_val = shrunk_reactivity[pos]
            deg_Mg_pH10_val = shrunk_deg_Mg_pH10[pos]
            deg_pH10_val = shrunk_deg_pH10[pos]
            deg_Mg_50C_val = shrunk_deg_Mg_50C[pos]
            deg_50C_val = shrunk_deg_50C[pos]
        else:
            reactivity_val = mean_reactivity
            deg_Mg_pH10_val = mean_deg_Mg_pH10
            deg_pH10_val = mean_deg_pH10
            deg_Mg_50C_val = mean_deg_Mg_50C
            deg_50C_val = mean_deg_50C

        rows.append(
            {
                "id_seqpos": f"{sample_id}_{pos}",
                "reactivity": reactivity_val,
                "deg_Mg_pH10": deg_Mg_pH10_val,
                "deg_pH10": deg_pH10_val,
                "deg_Mg_50C": deg_Mg_50C_val,
                "deg_50C": deg_50C_val,
            }
        )

df_submission = pd.DataFrame(
    rows,
    columns=[
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ],
)

df_submission.to_csv("submission.csv", index=False)
