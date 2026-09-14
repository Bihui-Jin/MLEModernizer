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

0.47502

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4702) has done: 'We fix the JSON loading (the files are JSON‑lines, not a single JSON object) so `train_df` and `test_df` are created correctly, then compute the column‑wise averages and build the submission. These changes unblock the script and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.47276) has done: 'I replace the simple unweighted per‑position means with a signal‑to‑noise weighted average (using the `signal_to_noise` column that is already present in the training data). This keeps the overall structure of the script unchanged while giving more influence to higher‑quality samples, which should lower the MCRMSE and move the score from 0.4702 toward the target 0.3519. No other logic is altered, and the script still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.49012) has done: 'The update enhances the weighted averaging by incorporating each target’s per‑position error values, giving higher influence to measurements with lower experimental error (inverse‑error weighting) while still using the overall signal‑to‑noise as a base weight. This fine‑grained weighting should reduce prediction variance and move the MCRMSE closer to the target without altering any core model logic or I/O handling.'
- What this solution (achieved 0.48983) has done: 'I filter the training data to keep only high‑quality samples (where `SN_filter` == 1) and raise the signal‑to‑noise weight to a stronger exponent (2.0). This gives cleaner, higher‑quality averages and should lower the MCRMSE toward the target while keeping the original averaging logic unchanged.'
- What this solution (achieved 0.48941) has done: 'I tighten the training data to only the higher‑quality samples (keep rows where `SN_filter == 1` and `signal_to_noise` is at least the median) and increase the weighting exponent from 2.0 to 2.5 so that the best samples dominate the averaged predictions. These minimal adjustments keep the original averaging logic while giving more influence to reliable measurements, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.48166) has done: 'I revert the aggressive data filtering and reduce the weighting influence so that more training samples contribute uniformly. By using all rows (no SN_filter or median SNR cut) and setting the base‑weight exponent to 0 (effectively neutralising the signal‑to‑noise weighting) while keeping the error‑based inverse weighting, the predictions become a more stable average, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.48277) has done: 'I strengthen the weighting used for the per‑position averages by (a) multiplying the signal‑to‑noise weight with the `SN_filter` flag (so high‑quality samples get extra influence) and (b) increasing the exponent to 1.0 for a more pronounced effect. These tiny adjustments keep the original averaging logic untouched while expected to lower the MCRMSE toward the target. The rest of the pipeline remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.47629) has done: 'I filter the training set to keep only the high‑quality samples (`SN_filter == 1`), strengthen the signal‑to‑noise weighting by using a larger exponent (2.0), and compute a simple bias correction from the training data so that the averaged predictions are shifted toward the true values. These minimal adjustments keep the original averaging logic but should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.47502) has done: 'I replace the single global bias with a per‑position bias for each target, computed from the high‑quality training rows. This keeps the existing weighting logic but adjusts each position’s prediction by the average residual observed in training, which should lower the MCRMSE toward the target. The rest of the pipeline (loading, weighting, submission creation) stays unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path


def find_file(name: str) -> Path:
    base = Path("/kaggle/input")
    matches = list(base.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return matches[0]


train_path = find_file("train.json")
test_path = find_file("test.json")

train_df = pd.read_json(train_path, lines=True)
test_df = pd.read_json(test_path, lines=True)




## === cell 1
def weighted_mean_with_error(col, base_weights, error_col, exponent=0):
    """
    Compute a weighted mean per position.
    - base_weights are raised to ``exponent``; exponent=0 makes them uniform.
    - Each sample weight is divided by (error + eps) to favour low‑error measurements.
    """
    eps = 1e-6
    arr = np.vstack(col.values)  # (samples, positions)
    err = np.vstack(error_col.values)  # (samples, positions)

    base_pow = np.power(base_weights, exponent)
    w = base_pow[:, None] / (err + eps)  # (samples, positions)

    weighted_sum = (arr * w).sum(axis=0)
    weight_sum = w.sum(axis=0)
    weighted_mean = weighted_sum / weight_sum
    return weighted_mean


high_quality = train_df[train_df["SN_filter"] == 1].copy()
if high_quality.empty:
    high_quality = train_df.copy()  # fallback if filter removes everything

signal_weights = (
    high_quality["signal_to_noise"]
    .fillna(high_quality["signal_to_noise"].median())
    .astype(float)
    .values
)
sn_flag = high_quality["SN_filter"].fillna(0).astype(float).values
base_weights = signal_weights * (sn_flag + 0.5)

exp = 2.0

avg_reactivity = weighted_mean_with_error(
    high_quality["reactivity"],
    base_weights,
    high_quality["reactivity_error"],
    exponent=exp,
)
avg_deg_Mg_pH10 = weighted_mean_with_error(
    high_quality["deg_Mg_pH10"],
    base_weights,
    high_quality["deg_error_Mg_pH10"],
    exponent=exp,
)
avg_deg_pH10 = weighted_mean_with_error(
    high_quality["deg_pH10"], base_weights, high_quality["deg_error_pH10"], exponent=exp
)
avg_deg_Mg_50C = weighted_mean_with_error(
    high_quality["deg_Mg_50C"],
    base_weights,
    high_quality["deg_error_Mg_50C"],
    exponent=exp,
)
avg_deg_50C = weighted_mean_with_error(
    high_quality["deg_50C"], base_weights, high_quality["deg_error_50C"], exponent=exp
)

max_scored = int(high_quality["seq_scored"].max())  # typically 68
bias_sum = {
    "reactivity": np.zeros(max_scored),
    "deg_Mg_pH10": np.zeros(max_scored),
    "deg_pH10": np.zeros(max_scored),
    "deg_Mg_50C": np.zeros(max_scored),
    "deg_50C": np.zeros(max_scored),
}
bias_cnt = np.zeros(max_scored)  # count per position

for _, row in high_quality.iterrows():
    seq_scored = int(row["seq_scored"])
    true_reac = np.array(row["reactivity"])
    true_degMg10 = np.array(row["deg_Mg_pH10"])
    true_deg10 = np.array(row["deg_pH10"])
    true_degMg50 = np.array(row["deg_Mg_50C"])
    true_deg50 = np.array(row["deg_50C"])

    pred_reac = avg_reactivity[:seq_scored]
    pred_degMg10 = avg_deg_Mg_pH10[:seq_scored]
    pred_deg10 = avg_deg_pH10[:seq_scored]
    pred_degMg50 = avg_deg_Mg_50C[:seq_scored]
    pred_deg50 = avg_deg_50C[:seq_scored]

    bias_sum["reactivity"][:seq_scored] += true_reac - pred_reac
    bias_sum["deg_Mg_pH10"][:seq_scored] += true_degMg10 - pred_degMg10
    bias_sum["deg_pH10"][:seq_scored] += true_deg10 - pred_deg10
    bias_sum["deg_Mg_50C"][:seq_scored] += true_degMg50 - pred_degMg50
    bias_sum["deg_50C"][:seq_scored] += true_deg50 - pred_deg50

    bias_cnt[:seq_scored] += 1

bias = {k: (v / np.where(bias_cnt > 0, bias_cnt, 1)) for k, v in bias_sum.items()}




## === cell 2
rows = []
for _, row in test_df.iterrows():
    seq_id = row["id"]
    seq_len = int(row["seq_length"])
    seq_scored = int(row["seq_scored"])
    for pos in range(seq_len):
        id_seqpos = f"{seq_id}_{pos}"
        if pos < seq_scored:
            reactivity = avg_reactivity[pos] + bias["reactivity"][pos]
            deg_Mg_pH10 = avg_deg_Mg_pH10[pos] + bias["deg_Mg_pH10"][pos]
            deg_pH10 = avg_deg_pH10[pos] + bias["deg_pH10"][pos]
            deg_Mg_50C = avg_deg_Mg_50C[pos] + bias["deg_Mg_50C"][pos]
            deg_50C = avg_deg_50C[pos] + bias["deg_50C"][pos]
        else:
            reactivity = deg_Mg_pH10 = deg_pH10 = deg_Mg_50C = deg_50C = 0.0
        rows.append(
            {
                "id_seqpos": id_seqpos,
                "reactivity": reactivity,
                "deg_Mg_pH10": deg_Mg_pH10,
                "deg_pH10": deg_pH10,
                "deg_Mg_50C": deg_Mg_50C,
                "deg_50C": deg_50C,
            }
        )

submission = pd.DataFrame(rows).sort_values("id_seqpos").reset_index(drop=True)




## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} ({submission.shape[0]} rows).")
