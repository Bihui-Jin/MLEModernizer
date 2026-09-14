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

0.3783348710000906

# 6. Current score

0.50502

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42418) has done: 'I fix the JSON loading (the files are line‑delimited JSON, not a single array), ensure the train and test data are correctly defined, and then keep the original mean‑based prediction logic unchanged. The script now run end‑to‑end and write a valid `submission.csv` file in `/kaggle/working`.'
- What this solution (achieved 0.44716) has done: 'I replace the simple per‑position averaging with a per‑position median, which often gives a more robust central tendency for noisy experimental data. This change is limited to the statistics computation (cell 2) and keeps the rest of the pipeline unchanged, so the submission format remains valid while moving the MCRMSE closer to the target lower score.'
- What this solution (achieved 0.42418) has done: 'I revert the central‑tendency calculation from median back to the arithmetic mean, because the median increased the validation error (higher MCRMSE). By assigning the mean values to the same variable names (`median_*`), the rest of the pipeline stays unchanged while the predictions become closer to the target score.'
- What this solution (achieved 0.42166) has done: 'I filter out low‑quality training samples before computing the per‑position averages, using the provided `SN_filter` flag and `signal_to_noise` value. Keeping only high‑quality records should give a cleaner mean estimate and therefore lower the MCRMSE, moving the score closer to the target while preserving the original prediction pipeline.'
- What this solution (achieved 0.42507) has done: 'I replace the simple per‑position arithmetic mean with a trimmed mean (discarding the lowest 5 % and highest 5 % of values at each position). This reduces the impact of outlier measurements while keeping the overall averaging approach unchanged, so it should lower the MCRMSE and move the score toward the target. The rest of the pipeline and submission format remain the same.'
- What this solution (achieved 0.42178) has done: 'The fix adds missing imports, defines the correct input file paths, and updates the cell numbering so the script runs from start to finish, builds the submission DataFrame, and writes a valid `submission.csv` file. No core modeling logic is changed, preserving the original weighted‑mean predictions while ensuring a proper end‑to‑end execution.'
- What this solution (achieved 0.43636) has done: 'I broaden the training data to use **all samples** (instead of discarding those with low SN_filter or low signal‑to‑noise) and keep the signal‑to‑noise as a weight (default 1 for missing values). After computing the weighted per‑position means, I apply a tiny smoothing (3‑point moving average) – a minimal, safe tweak that often reduces per‑position noise and should lower the MCRMSE toward the target without altering the core averaging logic.'
- What this solution (achieved 0.43967) has done: 'We filter out low‑quality training samples using the `SN_filter` flag and a signal‑to‑noise threshold, then compute a trimmed weighted mean (discarding the lowest 5 % and highest 5 % of values per position) before applying the same light smoothing. This modest change reduces noisy extremes while keeping the original averaging approach, expected to lower the MCRMSE toward the target score.'
- What this solution (achieved 0.4313) has done: 'I simplify the pipeline by using all training samples (removing the strict SN_filter / signal‑to‑noise filter) and compute a slightly more robust trimmed weighted mean (trim = 0.10). I also drop the extra smoothing step, keeping the raw per‑position estimates, which typically yields a lower MCRMSE and moves the score toward the target.'
- What this solution (achieved 0.42178) has done: 'I add a simple quality filter (keeping only samples with `SN_filter==1` and `signal_to_noise>=1.0`) and remove the trimming step by calling the weighted‑mean helper with `trim=0.0`. This keeps the core averaging logic but uses cleaner data and a pure weighted mean, which should lower the MCRMSE toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.45853) has done: 'We fix the typo that prevented the last target mean from being computed, correctly define `mean_deg_50C`, and apply a tiny 3‑point moving‑average smoothing to all per‑position means. This keeps the original weighted‑mean logic while reducing per‑position noise, which should lower the MCRMSE toward the target. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.50502) has done: 'I tighten the weighting by using inverse‑variance (error²) instead of plain error, which gives more influence to high‑confidence measurements, and increase the smoothing window from 3 to 5 to reduce per‑position noise. These minimal tweaks keep the overall averaging pipeline unchanged while aiming to lower the MCRMSE toward the target score.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
from pathlib import Path

train_path = Path("/kaggle/input/stanford-covid-vaccine/train.json")
test_path = Path("/kaggle/input/stanford-covid-vaccine/test.json")


def load_json_lines(path):
    """Load a line‑delimited JSON file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


train_data = load_json_lines(train_path)
test_data = load_json_lines(test_path)

SCORABLE = 68

filtered = [
    s
    for s in train_data
    if s.get("SN_filter", 1) == 1
    and s.get("signal_to_noise", 0) >= 1.0
    and len(s["reactivity"]) >= SCORABLE
]

reactivity_vals = []
reactivity_errs = []
deg_Mg_p2_vals = []
deg_Mg_p2_errs = []
deg_p2_vals = []
deg_p2_errs = []
deg_Mg_50C_vals = []
deg_Mg_50C_errs = []
deg_50C_vals = []
deg_50C_errs = []
sample_weights = []  # base weight = signal‑to‑noise

for sample in filtered:
    sample_weights.append(sample.get("signal_to_noise", 1.0))
    reactivity_vals.append(sample["reactivity"][:SCORABLE])
    reactivity_errs.append(sample["reactivity_error"][:SCORABLE])
    deg_Mg_p2_vals.append(sample["deg_Mg_pH10"][:SCORABLE])
    deg_Mg_p2_errs.append(sample["deg_error_Mg_pH10"][:SCORABLE])
    deg_p2_vals.append(sample["deg_pH10"][:SCORABLE])
    deg_p2_errs.append(sample["deg_error_pH10"][:SCORABLE])
    deg_Mg_50C_vals.append(sample["deg_Mg_50C"][:SCORABLE])
    deg_Mg_50C_errs.append(sample["deg_error_Mg_50C"][:SCORABLE])
    deg_50C_vals.append(sample["deg_50C"][:SCORABLE])
    deg_50C_errs.append(sample["deg_error_50C"][:SCORABLE])

if not reactivity_vals:
    raise ValueError("No training samples passed the filters.")

reactivity_arr = np.array(reactivity_vals, dtype=float)
reactivity_err_arr = np.array(reactivity_errs, dtype=float)
deg_Mg_p2_arr = np.array(deg_Mg_p2_vals, dtype=float)
deg_Mg_p2_err_arr = np.array(deg_Mg_p2_errs, dtype=float)
deg_p2_arr = np.array(deg_p2_vals, dtype=float)
deg_p2_err_arr = np.array(deg_p2_errs, dtype=float)
deg_Mg_50C_arr = np.array(deg_Mg_50C_vals, dtype=float)
deg_Mg_50C_err_arr = np.array(deg_Mg_50C_errs, dtype=float)
deg_50C_arr = np.array(deg_50C_vals, dtype=float)
deg_50C_err_arr = np.array(deg_50C_errs, dtype=float)
base_weights = np.array(sample_weights, dtype=float)


def weighted_mean_with_errors(values, base_w, errors, eps=1e-6):
    """
    Per‑position weighted mean using inverse‑variance weighting.
    Weight ∝ base_w / (error² + eps) to give higher influence to low‑error samples.
    """
    w = base_w[:, None] / (errors**2 + eps)
    weighted_sum = np.sum(values * w, axis=0)
    weight_sum = np.sum(w, axis=0)
    return weighted_sum / weight_sum


mean_reactivity = weighted_mean_with_errors(
    reactivity_arr, base_weights, reactivity_err_arr
)
mean_deg_Mg_pH10 = weighted_mean_with_errors(
    deg_Mg_p2_arr, base_weights, deg_Mg_p2_err_arr
)
mean_deg_pH10 = weighted_mean_with_errors(deg_p2_arr, base_weights, deg_p2_err_arr)
mean_deg_Mg_50C = weighted_mean_with_errors(
    deg_Mg_50C_arr, base_weights, deg_Mg_50C_err_arr
)
mean_deg_50C = weighted_mean_with_errors(deg_50C_arr, base_weights, deg_50C_err_arr)


def smooth_series(arr, window=5):
    """Simple moving‑average smoothing with a slightly larger window for better noise reduction."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


mean_reactivity = smooth_series(mean_reactivity)
mean_deg_Mg_pH10 = smooth_series(mean_deg_Mg_pH10)
mean_deg_pH10 = smooth_series(mean_deg_pH10)
mean_deg_Mg_50C = smooth_series(mean_deg_Mg_50C)
mean_deg_50C = smooth_series(mean_deg_50C)




## === cell 1
rows = []
for sample in test_data:
    seq_id = sample["id"]
    seq_len = sample["seq_length"]  # total length (107)
    for pos in range(seq_len):
        id_seqpos = f"{seq_id}_{pos}"
        if pos < SCORABLE:
            rows.append(
                {
                    "id_seqpos": id_seqpos,
                    "reactivity": mean_reactivity[pos],
                    "deg_Mg_pH10": mean_deg_Mg_pH10[pos],
                    "deg_pH10": mean_deg_pH10[pos],
                    "deg_Mg_50C": mean_deg_Mg_50C[pos],
                    "deg_50C": mean_deg_50C[pos],
                }
            )
        else:
            rows.append(
                {
                    "id_seqpos": id_seqpos,
                    "reactivity": mean_reactivity[-1],
                    "deg_Mg_pH10": mean_deg_Mg_pH10[-1],
                    "deg_pH10": mean_deg_pH10[-1],
                    "deg_Mg_50C": mean_deg_Mg_50C[-1],
                    "deg_50C": mean_deg_50C[-1],
                }
            )

submission = pd.DataFrame(
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

print("Submission shape:", submission.shape)
print(submission.head())




## === cell 2
output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False)
print(f"Saved submission to {output_path}")
