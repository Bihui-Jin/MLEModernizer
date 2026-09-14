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

0.3807426429593401

# 6. Current score

0.44035

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The script now avoids the earlier import conflict, uses the correct data path for the sample submission, and skips the broken model‑training sections. It simply creates a valid submission file by filling all target columns with zeros, ensuring the notebook runs from start to finish and produces `submission.csv` in the required format.'
- What this solution (achieved 0.63824) has done: 'The fix removes the problematic TensorFlow import, loads the training data, and computes per‑position average values for each target (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C). These averages replace the all‑zero predictions, giving a reasonable baseline that lowers the MCRMSE toward the target while keeping the original workflow and submission format unchanged. The script now writes a valid `submission.csv` file with the required columns.'
- What this solution (achieved 0.42198) has done: 'The fix changes the train‑data loading to handle line‑delimited JSON (preventing the JSONDecodeError) and computes a weighted per‑position mean using the “signal_to_noise” value, which gives a slightly better baseline than a simple average. The rest of the workflow (validation of the sample submission, filling predictions, and writing the CSV) remains unchanged, ensuring a valid `submission.csv` is produced and the score moves closer to the target.'
- What this solution (achieved 0.43017) has done: 'The changes add a light smoothing step to the per‑position weighted averages, which makes the predictions a bit more realistic across neighboring bases. By convolving each target’s mean vector with a small kernel, we keep the core logic unchanged while likely reducing the MCRMSE, moving the score closer to the target. The rest of the pipeline (validation, loading, and CSV output) remains identical.'
- What this solution (achieved 0.41985) has done: 'The update adds a nucleotide‑aware baseline: for each position it computes weighted averages of the targets per base (A, C, G, U) from the training data, smooths these vectors, and then uses the test sequence’s base at the required position to select the appropriate value (falling back to the general smoothed mean when needed). This keeps the original workflow intact while providing more specific predictions, which should lower the MCRMSE and move the score closer to the target.'
- What this solution (achieved 0.42034) has done: 'I keep the original workflow but add a small shrinkage step that blends the nucleotide‑specific predictions with the overall smoothed per‑position means. This reduces variance from noisy nucleotide‑specific averages and should lower the MCRMSE, moving the score closer to the target while preserving all core logic and output format.'
- What this solution (achieved 0.4237) has done: 'I lower the blend factor for the nucleotide‑specific predictions (alpha) from 0.8 to 0.4 so the more stable, globally‑smoothed per‑position means dominate the final prediction. This simple tweak keeps the core logic untouched while likely reducing variance and moving the MCRMSE closer to the target lower score.'
- What this solution (achieved 0.44191) has done: 'I lower the blend factor so the stable globally‑smoothed predictions dominate (alpha = 0.2) and replace the 3‑point smoothing kernel with a slightly broader 5‑point average. These tiny adjustments keep the original workflow unchanged while reducing prediction variance, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.45114) has done: 'I lower the blending factor to 0 so the predictions rely solely on the globally smoothed per‑position means (removing the noisy nucleotide‑specific component) and I smooth those means with a slightly wider 7‑point moving average kernel. This keeps the overall workflow unchanged while likely reducing variance and moving the MCRMSE closer to the target lower score.'
- What this solution (achieved 0.44731) has done: 'I re‑introduce a modest amount of nucleotide‑specific information by setting the blending factor `alpha` to 0.3 (instead of 0). This keeps the globally smoothed means as the dominant baseline while allowing the per‑nucleotide conditioned predictions to contribute, which should lower the MCRMSE toward the target without altering the overall workflow.'
- What this solution (achieved 0.45647) has done: 'I lower the nucleotide‑specific blend factor to 0.1 (so the stable global smoothed mean dominates) and use a slightly wider 9‑position moving‑average kernel for both the global and nucleotide‑specific smoothing. These minimal tweaks keep the original workflow intact while reducing prediction variance, which should move the MCRMSE closer to the target lower score.'
- What this solution (achieved 0.44035) has done: 'We increase the nucleotide‑specific blend factor (α) to give the model more useful per‑base information and shrink the smoothing kernel from 9 to 5 positions, which prior experiments showed lowers the MCRMSE. These are the only changes, preserving the overall workflow and output format.'

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd
import numpy as np




## === cell 1
data_dir = "/kaggle/input/stanford-covid-vaccine/"
sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

expected_cols = [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
missing = set(expected_cols) - set(sample_sub.columns)
if missing:
    raise ValueError(f"Sample submission is missing columns: {missing}")




## === cell 2
train_path = os.path.join(data_dir, "train.json")
train_entries = []
with open(train_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            train_entries.append(json.loads(line))

test_path = os.path.join(data_dir, "test.json")
test_entries = {}
with open(test_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            entry = json.loads(line)
            test_entries[entry["id"]] = entry

target_names = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
nuc2idx = {"A": 0, "C": 1, "G": 2, "U": 3}

weights = np.array([e.get("signal_to_noise", 1.0) for e in train_entries], dtype=float)

target_arrays = {t: [] for t in target_names}
for entry in train_entries:
    for t in target_names:
        arr = entry.get(t, [])
        if isinstance(arr, list) and len(arr) > 0:
            target_arrays[t].append(arr)

mean_per_pos = {}
for t in target_names:
    arr_np = np.array(target_arrays[t], dtype=float)  # (n_samples, 68)
    if arr_np.ndim == 2 and arr_np.shape[0] == len(weights):
        mean_per_pos[t] = np.average(arr_np, axis=0, weights=weights)
    else:
        mean_per_pos[t] = np.mean(arr_np, axis=0)

kernel = np.ones(5) / 5.0
smoothed_means = {
    t: np.convolve(vec, kernel, mode="same") for t, vec in mean_per_pos.items()
}

cond_sums = {t: np.zeros((68, 4), dtype=float) for t in target_names}
cond_weights = {t: np.zeros((68, 4), dtype=float) for t in target_names}

for entry, w in zip(train_entries, weights):
    seq = entry["sequence"]
    for t in target_names:
        arr = entry[t]  # list of length 68
        for pos in range(min(len(arr), len(seq), 68)):
            nuc = seq[pos]
            idx = nuc2idx.get(nuc)
            if idx is not None:
                cond_sums[t][pos, idx] += w * arr[pos]
                cond_weights[t][pos, idx] += w

cond_means = {}
for t in target_names:
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = cond_sums[t] / cond_weights[t]
    for pos in range(68):
        for idx in range(4):
            if np.isnan(avg[pos, idx]):
                avg[pos, idx] = smoothed_means[t][pos]
    cond_means[t] = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=avg
    )

blended_means = {}
alpha = 0.3  # blend factor: 30 % nucleotide‑specific, 70 % global smoothed mean
for t in target_names:
    global_broadcast = smoothed_means[t][:, None]
    blended_means[t] = alpha * cond_means[t] + (1 - alpha) * global_broadcast




## === cell 3
submission = sample_sub.copy()


def parse_id_seqpos(id_seqpos):
    """Return (sample_id, position)."""
    sample_id, pos_str = id_seqpos.rsplit("_", 1)
    return sample_id, int(pos_str)


for idx, row in submission.iterrows():
    sample_id, pos = parse_id_seqpos(row["id_seqpos"])
    entry = test_entries.get(sample_id, {})
    seq_scored = entry.get("seq_scored", 68)
    if pos < seq_scored:
        seq = entry.get("sequence", "")
        nuc = seq[pos] if pos < len(seq) else None
        nuc_idx = nuc2idx.get(nuc)
        for t in target_names:
            if nuc_idx is not None:
                pred = blended_means[t][pos, nuc_idx]
            else:
                pred = smoothed_means[t][pos]
            submission.at[idx, t] = pred
    else:
        for t in target_names:
            submission.at[idx, t] = 0.0




## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
