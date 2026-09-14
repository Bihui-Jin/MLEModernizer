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

0.3578912193294472

# 6. Current score

0.43487

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'I fix the JSON loading error by reading the files line‑by‑line (they are newline‑delimited JSON), then compute the global means and generate the submission file. This resolves the runtime crashes and ensures a valid `submission.csv` is produced.'
- What this solution (achieved 0.42166) has done: 'We replace the simple global‑mean baseline with per‑position means (computed only from high‑quality training samples) and fall back to the overall mean for positions beyond the scored region. This adds a modest amount of signal while keeping the original modeling approach unchanged, and should move the validation error closer to the target lower score.'
- What this solution (achieved 0.42205) has done: 'I replace the simple unweighted per‑position averages with weighted averages that use each sample’s `signal_to_noise` as a reliability weight (and include all samples, not only those with `SN_filter==1`). Weighted means are less affected by noisy measurements, so the predictions should become more accurate and the validation MCRMSE should move closer to the target lower score.'
- What this solution (achieved 0.42205) has done: 'The fix adds the missing imports, a helper to read newline‑delimited JSON, and loads the training data before computing the weighted per‑position means. Cells are renumbered starting at 1, and the final cell writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.42178) has done: 'The update narrows training data to high‑quality samples (`SN_filter == 1`) and reduces the smoothing term when blending per‑position means with the global mean. These minimal changes keep the overall baseline approach while giving the model cleaner statistics, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.42178) has done: 'I increase the smoothing factor from 0.5 to 1.0 so that per‑position predictions are blended a bit more with the global mean, which reduces variance from noisy positions while keeping the same weighted‑mean logic. This tiny regularisation is expected to lower the validation MCRMSE toward the target without altering the core baseline approach. I also renumber the cells to start at 1 so the script follows the required format.'
- What this solution (achieved 0.42205) has done: 'I lower the smoothing blend factor (from 1.0 to 0.5) so per‑position means contribute more directly, and I drop the strict `SN_filter == 1` restriction to use all training samples (still weighted by their `signal_to_noise`). These minimal adjustments keep the core baseline unchanged while giving the predictions a stronger, better‑estimated signal, which should reduce the MCRMSE toward the target score.'
- What this solution (achieved 0.42178) has done: 'I filter the training data to keep only high‑quality samples (`SN_filter == 1`), keep the weight as the original `signal_to_noise`, and remove the smoothing blend (set it to 0) so that per‑position weighted means are used directly. This small change preserves the baseline logic while giving the predictions a stronger, less‑biased signal, which should move the MCRMSE closer to the target.'
- What this solution (achieved 0.42463) has done: 'I keep the baseline weighted‑mean approach but include **all** training samples (removing the `SN_filter==1` restriction) and add a modest blend with the global mean (smoothing = 0.2). This uses more data while regularising noisy positions, which should lower the MCRMSE toward the target without changing the core modeling logic.'
- What this solution (achieved 0.43487) has done: 'I filter the training data to keep only high‑quality samples (`SN_filter == 1`), and increase the smoothing factor from 0.2 to 0.5 so predictions blend more with the global mean, which reduces variance from noisy positions and should lower the MCRMSE toward the target. The cell order is renumbered to start at 1 while preserving the original logic.'
- What this solution (achieved 0.42463) has done: 'The changes remove the strict `SN_filter == 1` restriction so all training samples contribute (still weighted by their `signal_to_noise`), and lower the smoothing factor to 0.2 so per‑position means dominate the predictions. This modest adjustment keeps the original baseline logic while improving calibration, which should reduce the validation MCRMSE toward the target score.'
- What this solution (achieved 0.43487) has done: 'I filter the training data to keep only high‑quality samples (`SN_filter == 1`), which reduces noise, and increase the smoothing factor from 0.2 to 0.5 so predictions rely more on the stable global mean. These minimal adjustments keep the original weighted‑mean baseline while aiming to lower the MCRMSE toward the target score.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_json_lines(path: Path):
    """Read a newline‑delimited JSON file and return a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


train_path_candidates = [
    Path("../input/train.json"),
    Path("../input/stanford-covid-vaccine/train.json"),
    Path("input/train.json"),
    Path("train.json"),
]
train_path = next((p for p in train_path_candidates if p.exists()), None)
if train_path is None:
    raise FileNotFoundError("train.json not found in expected locations.")
train_data = load_json_lines(train_path)




## === cell 1
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
seq_len_total = 107

pos_weighted_sums = {t: np.zeros(seq_len_total, dtype=np.float64) for t in targets}
pos_weights = {t: np.zeros(seq_len_total, dtype=np.float64) for t in targets}
overall_weighted_sums = {t: 0.0 for t in targets}
overall_weights = {t: 0.0 for t in targets}

for entry in train_data:
    if entry.get("SN_filter", 1) != 1:
        continue
    weight = entry.get("signal_to_noise", 1.0)
    if not isinstance(weight, (int, float)) or weight <= 0:
        weight = 1.0
    seq_scored = entry["seq_scored"]  # typically 68
    for t in targets:
        vals = entry[t]  # list of length = seq_scored
        for i, v in enumerate(vals):
            pos_weighted_sums[t][i] += v * weight
            pos_weights[t][i] += weight
        overall_weighted_sums[t] += sum(v * weight for v in vals)
        overall_weights[t] += weight * len(vals)

smoothing = 0.5

per_pos_means = {}
global_fallback_means = {}

for t in targets:
    overall_mean = (
        overall_weighted_sums[t] / overall_weights[t] if overall_weights[t] > 0 else 0.0
    )
    global_fallback_means[t] = overall_mean
    means = np.empty(seq_len_total, dtype=np.float64)
    for i in range(seq_len_total):
        if pos_weights[t][i] > 0:
            pos_mean = pos_weighted_sums[t][i] / pos_weights[t][i]
        else:
            pos_mean = overall_mean
        means[i] = (1.0 - smoothing) * pos_mean + smoothing * overall_mean
    per_pos_means[t] = means

test_path_candidates = [
    Path("../input/test.json"),
    Path("../input/stanford-covid-vaccine/test.json"),
    Path("input/test.json"),
    Path("test.json"),
]
test_path = next((p for p in test_path_candidates if p.exists()), None)
if test_path is None:
    raise FileNotFoundError("test.json not found in expected locations.")
test_data = load_json_lines(test_path)

rows = []
for entry in test_data:
    seq_id = entry["id"]
    seq_len = entry["seq_length"]  # always 107
    seq_scored = entry["seq_scored"]
    for pos in range(seq_len):
        rows.append(
            {
                "id_seqpos": f"{seq_id}_{pos}",
                "reactivity": (
                    per_pos_means["reactivity"][pos]
                    if pos < seq_scored
                    else global_fallback_means["reactivity"]
                ),
                "deg_Mg_pH10": (
                    per_pos_means["deg_Mg_pH10"][pos]
                    if pos < seq_scored
                    else global_fallback_means["deg_Mg_pH10"]
                ),
                "deg_pH10": (
                    per_pos_means["deg_pH10"][pos]
                    if pos < seq_scored
                    else global_fallback_means["deg_pH10"]
                ),
                "deg_Mg_50C": (
                    per_pos_means["deg_Mg_50C"][pos]
                    if pos < seq_scored
                    else global_fallback_means["deg_Mg_50C"]
                ),
                "deg_50C": (
                    per_pos_means["deg_50C"][pos]
                    if pos < seq_scored
                    else global_fallback_means["deg_50C"]
                ),
            }
        )

submission_df = pd.DataFrame(
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




## === cell 2
submission_df.to_csv("submission.csv", index=False)
