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

No external packages required in the script and installed.

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

0.3756351841458719

# 6. Current score

0.48351

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I fixed the JSON loading by enabling `lines=True` for the line‑delimited files, which resolves the “Trailing data” error. I also reorganized the notebook into a clean sequence of cells (starting at 1) so that each variable is defined before it’s used, and the final submission CSV is written correctly. No changes are made to the modeling logic; the script now runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 0.47904) has done: 'I replace the simple un‑weighted per‑position averages with a weighted average that uses the provided error columns (inverse‑variance weighting). This keeps the overall structure of the model unchanged but gives more reliable estimates for each position, which should lower the MCRMSE and move the score closer to the target.'
- What this solution (achieved 0.48016) has done: 'I boost the weighted‑average baseline by incorporating each sample’s `signal_to_noise` (higher‑quality samples get more influence). This keeps the overall per‑position averaging logic untouched while giving more weight to reliable measurements, which should lower the MCRMSE and move the score closer to the target.'
- What this solution (achieved 0.49358) has done: 'I add per‑position minimum/maximum tracking and a short smoothing step to the baseline weighted‑average predictions. By clipping predictions to the observed training range and smoothing them across nearby positions, we reduce extreme or noisy estimates, which should lower the MCRMSE and move the score closer to the target while keeping the overall modeling approach unchanged.'
- What this solution (achieved 0.48016) has done: 'I remove the unnecessary smoothing step, using the raw weighted means (clipped to observed ranges) for each position. Smoothing can blur genuine signal and hurts the MCRMSE, so dropping it should lower the error toward the target while keeping all other logic unchanged.'
- What this solution (achieved 0.4817) has done: 'I keep the overall weighted‑average baseline but add a small global‑mean blending step. After computing the per‑position weighted means I also compute a single weighted overall mean for each target column and then blend the two predictions (95 % position‑specific, 5 % global). This modest regularisation often reduces over‑fitting to noisy positions and should lower the MCRMSE, moving the score closer to the target while leaving the core logic unchanged.'
- What this solution (achieved 0.49436) has done: 'I add a light smoothing of the per‑position weighted means (a small 3‑point rolling average) to reduce noisy predictions, and lower the blending factor α from 0.95 to 0.85 so the global mean contributes more. These minimal changes keep the core weighted‑average logic intact while expectedly lowering the MCRMSE toward the target.'
- What this solution (achieved 0.4817) has done: 'I remove the unnecessary 3‑point smoothing (using the raw clipped per‑position means) and increase the blending factor α from 0.85 to 0.95 so that the predictions rely more on the position‑specific weighted averages and less on the global mean. These minimal tweaks keep the overall weighted‑average logic intact while expectedly lowering the MCRMSE toward the target score.'
- What this solution (achieved 0.48075) has done: 'I increase the blending factor `alpha` from 0.95 to 0.98 so the predictions rely more on the per‑position weighted averages (which already use error‑based and signal‑to‑noise weighting) and less on the global mean. This small tweak keeps the overall modeling approach unchanged while expectedly lowering the MCRMSE toward the target score.'
- What this solution (achieved 0.48442) has done: 'I incorporate the binary `SN_filter` flag into the sample weighting (so filtered‑out samples contribute no weight) and slightly reduce the blending factor `alpha` from 0.98 to 0.96. These tiny adjustments keep the overall weighted‑average baseline unchanged while giving a bit more regularisation, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.49987) has done: 'I add a light 3‑point smoothing to the per‑position weighted means and increase the regularisation by lowering the blending factor `alpha` from 0.96 to 0.90. Smoothing reduces noisy position‑specific estimates, and a smaller `alpha` lets the global mean stabilize predictions, both of which should lower the MCRMSE toward the target while leaving the overall modelling approach unchanged.'
- What this solution (achieved 0.48351) has done: 'I increase the blending factor so predictions rely more on the position‑specific weighted averages (alpha = 0.99) and remove the extra 3‑point smoothing step, keeping the original weighted‑average logic otherwise unchanged. This should reduce the MCRMSE toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = "/kaggle/input/stanford-covid-vaccine/"
DATA_DIR = BASE_DIR  # same as BASE_DIR
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")

train = pd.read_json(TRAIN_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
error_map = {
    "reactivity": "reactivity_error",
    "deg_Mg_pH10": "deg_error_Mg_pH10",
    "deg_pH10": "deg_error_pH10",
    "deg_Mg_50C": "deg_error_Mg_50C",
    "deg_50C": "deg_error_50C",
}



## === cell 1
max_seq_scored = train["seq_scored"].max()
position_means = {}
global_means = {}
position_mins = {}
position_maxs = {}

for col in target_cols:
    sums = np.zeros(max_seq_scored, dtype=np.float64)
    weights = np.zeros(max_seq_scored, dtype=np.float64)
    mins = np.full(max_seq_scored, np.inf, dtype=np.float64)
    maxs = np.full(max_seq_scored, -np.inf, dtype=np.float64)

    err_col = error_map[col]
    for _, row in train.iterrows():
        vals = np.array(row[col], dtype=np.float64)
        err_vals = np.array(row[err_col], dtype=np.float64)
        err_vals = np.where(err_vals <= 0, 1e-6, err_vals)
        w = 1.0 / (err_vals**2)

        sn = row.get("signal_to_noise", 1.0)
        sn = sn if sn > 0 else 1e-3
        w = w * sn

        sn_filter = row.get("SN_filter", 1)
        w = w * (sn_filter if sn_filter > 0 else 0)

        l = len(vals)
        sums[:l] += vals * w
        weights[:l] += w
        mins[:l] = np.minimum(mins[:l], vals)
        maxs[:l] = np.maximum(maxs[:l], vals)

    means = np.divide(sums, weights, out=np.zeros_like(sums), where=weights != 0)
    clipped_means = np.clip(means, mins, maxs)

    position_means[col] = clipped_means
    position_mins[col] = mins
    position_maxs[col] = maxs

    total_weight = np.sum(weights)
    if total_weight > 0:
        global_means[col] = np.sum(sums) / total_weight
    else:
        global_means[col] = 0.0



## === cell 2
alpha = 0.99

pred_rows = []
for _, row in test.iterrows():
    uid = row["id"]
    seq_len = row["seq_length"]
    seq_scored = row["seq_scored"]
    for pos in range(seq_len):
        pred_dict = {"id_seqpos": f"{uid}_{pos}"}
        for col in target_cols:
            if pos < seq_scored:
                pos_val = position_means[col][pos]
                blended = alpha * pos_val + (1 - alpha) * global_means[col]
                pred_dict[col] = blended
            else:
                pred_dict[col] = 0.0
        pred_rows.append(pred_dict)

pred_df = pd.DataFrame(pred_rows)



## === cell 3
submission = sample_sub[["id_seqpos"]].merge(pred_df, on="id_seqpos", how="left")
submission[target_cols] = submission[target_cols].fillna(0.0)



## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with shape {submission.shape}")
