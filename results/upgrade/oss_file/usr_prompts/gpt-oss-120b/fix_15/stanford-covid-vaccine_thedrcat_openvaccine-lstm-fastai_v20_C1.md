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

0.37567

# 6. Current score

0.42046

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47906) has done: 'The script was failing because it tried to reinstall incompatible package versions, missed essential imports, and relied on a complex model pipeline that was never built. All of those steps are removed and replaced with a simple, fully‑functional baseline: we load the data, compute the overall mean of each target in the training set, and fill every test position with those means. This guarantees a valid `submission.csv` with the correct columns and format, allowing the notebook to run end‑to‑end.'
- What this solution (achieved 0.42418) has done: 'The update switches from a single global mean per target to a per‑position mean (using the training data) and falls back to the overall mean for positions beyond the scored region. This more closely matches the true distribution of each base, so the predictions become better calibrated and the MCRMSE should move closer to the target score while keeping the overall simple baseline unchanged.'
- What this solution (achieved 0.41641) has done: 'I add a lightweight per‑position, per‑nucleotide mean lookup and use it when predicting each target. This keeps the original simple baseline while giving the model a small amount of sequence‑aware calibration, which should lower the MCRMSE toward the target score without altering the overall pipeline.'
- What this solution (achieved 0.41641) has done: 'I keep the overall simple baseline but adjust the fallback prediction when a nucleotide‑specific mean is unavailable. Instead of using the raw per‑position mean (which can be noisy), I blend it with the overall mean (70 % position mean + 30 % overall mean). This small regularisation should lower the RMS error for the scored positions, moving the score closer to the target while preserving the existing pipeline and output format.'
- What this solution (achieved 0.41641) has done: 'I smooth the per‑position means to reduce noise and give them equal weight with the overall mean in the fallback (POS_WEIGHT = OVERALL_WEIGHT = 0.5). This small change keeps the original per‑nucleotide lookup while making the predictions a bit more stable, which should lower the MCRMSE toward the target score.'
- What this solution (achieved 0.41641) has done: 'I increase the reliance on the per‑position means by changing the fallback blend from an even 50/50 mix to a stronger 80 % position‑mean + 20 % overall‑mean weighting. This small tweak keeps the overall pipeline unchanged while giving more weight to the more informative position‑specific signal, which should lower the MCRMSE and move the score closer to the target.'
- What this solution (achieved 0.41641) has done: 'The update reduces the reliance on potentially noisy position‑specific predictions by lowering the POS_WEIGHT from 0.8 to 0.6 and raising the OVERALL_WEIGHT from 0.2 to 0.4. This adds more regularisation from the overall mean while keeping the existing per‑nucleotide and smoothed position logic unchanged, which should lower the MCRMSE toward the target without altering the core pipeline.'
- What this solution (achieved 0.4245) has done: 'The update adds a lightweight blending step when a nucleotide‑specific mean is available: the prediction now combines the nucleotide‑specific mean with the existing position‑smoothed/overall fallback (using a 50 % / 50 % mix). This regularises noisy nucleotide‑only estimates and is expected to lower the MCRMSE, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.4189) has done: 'I slightly adjust the blending weights that combine the position‑smoothed mean, the overall mean, and the nucleotide‑specific mean. Increasing the reliance on the position‑specific signal (POS_WEIGHT = 0.7, OVERALL_WEIGHT = 0.3) and giving more weight to the nucleotide‑specific estimate (NUC_BLEND_WEIGHT = 0.7) should reduce the MCRMSE, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.41775) has done: 'I keep the overall baseline architecture but make two small, targeted tweaks that should lower the MCRMSE: (1) smooth the per‑position means with a slightly wider window (size 5) to reduce noise, and (2) give a bit more weight to the position‑specific and nucleotide‑specific signals (POS_WEIGHT = 0.8, OVERALL_WEIGHT = 0.2, NUC_BLEND_WEIGHT = 0.8). These changes preserve the original logic while making the predictions more calibrated, moving the score closer to the target.'
- What this solution (achieved 0.42883) has done: 'I slightly reduce the reliance on the noisy per‑position and nucleotide‑specific means and increase the contribution of the overall global mean, while also smoothing the position means with a slightly larger window. This modest regularisation is expected to lower the MCRMSE toward the target score without altering the core baseline logic.'
- What this solution (achieved 0.41775) has done: 'I tighten the model’s calibration by giving more weight to the informative per‑position and nucleotide‑specific means and using a narrower smoothing window (size 5). This should reduce the MCRMSE and move the score from 0.42883 down toward the target 0.37567 while preserving the overall baseline logic.'
- What this solution (achieved 0.41679) has done: 'I keep the overall baseline unchanged but smooth the position‑wise means with a slightly larger window (7 instead of 5) and give a bit more emphasis to the smoothed position signal and the nucleotide‑specific estimate (POS_WEIGHT = 0.9, OVERALL_WEIGHT = 0.1, NUC_BLEND_WEIGHT = 0.9). These minor tweaks should reduce noise and improve calibration, moving the MCRMSE closer to the target without altering the core pipeline.'
- What this solution (achieved 0.42046) has done: 'I slightly regularise the predictions by decreasing the reliance on the noisy position‑wise and nucleotide‑specific means and increasing the contribution of the overall global mean. This small adjustment (POS_WEIGHT = 0.7, OVERALL_WEIGHT = 0.3, NUC_BLEND_WEIGHT = 0.7) is expected to lower the MCRMSE enough to bring the score within the target tolerance while keeping the original baseline logic unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os

DATA_PATH = "/kaggle/input/stanford-covid-vaccine"
TRAIN_PATH = os.path.join(DATA_PATH, "train.json")
TEST_PATH = os.path.join(DATA_PATH, "test.json")
SAMPLE_SUB_PATH = os.path.join(DATA_PATH, "sample_submission.csv")

train = pd.read_json(TRAIN_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)  # keep column order

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

pos_means = {}
overall_means = {}
for t in TARGETS:
    vals = np.vstack(train[t].values)  # (n_samples, 68)
    pos_means[t] = vals.mean(axis=0)  # mean for each scored position
    overall_means[t] = pos_means[t].mean()  # fallback overall mean

SMOOTH_WINDOW = 7
pos_means_smoothed = {}
for t in TARGETS:
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    smoothed = np.convolve(pos_means[t], kernel, mode="same")
    pos_means_smoothed[t] = smoothed

nucs = ["A", "C", "G", "U"]
seq_array = np.array([list(s) for s in train["sequence"]])  # (n_samples, 107)
cond_means = {t: [] for t in TARGETS}
for pos in range(107):
    for t in TARGETS:
        target_vals = np.full(train.shape[0], np.nan, dtype=float)
        if pos < len(train.iloc[0][t]):  # only first 68 positions have true values
            target_vals = np.array(
                [row[pos] if pos < len(row) else np.nan for row in train[t]]
            )
        nuc_dict = {}
        for n in nucs:
            mask = seq_array[:, pos] == n
            if mask.any():
                mean_val = np.nanmean(target_vals[mask])
                if not np.isnan(mean_val):
                    nuc_dict[n] = mean_val
        cond_means[t].append(nuc_dict)  # index == position, value == dict nuc->mean




## === cell 1
rows = []
POS_WEIGHT = 0.7  # weight for smoothed position mean
OVERALL_WEIGHT = 0.3  # weight for global overall mean
NUC_BLEND_WEIGHT = 0.7  # weight for nucleotide‑specific mean

for _, row in test.iterrows():
    id_base = row["id"]
    seq_scored = int(row["seq_scored"])  # number of positions that are scored (68)
    seq = row["sequence"]
    for pos in range(107):
        nuc = seq[pos]
        preds = []
        for t in TARGETS:
            fallback = (
                POS_WEIGHT * pos_means_smoothed[t][pos]
                + OVERALL_WEIGHT * overall_means[t]
                if pos < seq_scored
                else overall_means[t]
            )
            nuc_dict = cond_means[t][pos]
            if nuc in nuc_dict:
                pred = (
                    NUC_BLEND_WEIGHT * nuc_dict[nuc] + (1 - NUC_BLEND_WEIGHT) * fallback
                )
            else:
                pred = fallback
            preds.append(pred)
        rows.append([f"{id_base}_{pos}"] + preds)

submission = pd.DataFrame(rows, columns=["id_seqpos"] + TARGETS)




## === cell 2
submission = submission[sample_sub.columns]  # ensure exact column order

output_path = "submission.csv"  # Kaggle will pick this up from the working directory
submission.to_csv(output_path, index=False)

print(f"Submission file written to {output_path} with shape {submission.shape}")
