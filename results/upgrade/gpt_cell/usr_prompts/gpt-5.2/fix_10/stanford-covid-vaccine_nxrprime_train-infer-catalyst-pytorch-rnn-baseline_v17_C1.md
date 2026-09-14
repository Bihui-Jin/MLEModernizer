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

0.4772

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The crash happens immediately because the notebook imports `catalyst`, which is not installed in this environment. Since the remaining cells shown don’t use `dl`/`utils` yet, the minimal safe fix is to make those imports optional and provide a clear fallback (set them to `None`) so execution can continue deterministically. This keeps all existing logic intact while unblocking the rest of the notebook. No other cells are modified.'
- What this solution (achieved 0.63824) has done: 'Diagnosis: Cell 6 crashes because `dl` is `None` (the `catalyst` import failed in cell 0), so `dl.Runner` does not exist. Since we must not change earlier cells, the fix must make cell 6 work whether Catalyst is installed or not. We can conditionally define `CustomRunner` using Catalyst when available, and otherwise provide a minimal compatible fallback class that preserves the same `_handle_batch` and `predict_batch` core logic and attributes used later.

Patch summary: In cell 6, wrap the `CustomRunner` definition in a conditional on `dl` and `utils`. If Catalyst is missing, define a small runner class that stores `model`, `optimizer`, and `device`, provides `batch_metrics` dict, and runs the same forward/loss/backprop steps. Also ensure `predict_batch` uses the same tensor transforms.

Updated cells: Only cell 6 is modified.

Compatibility notes for cell k+1: Cell 7 calls `utils.get_device()`. This patch does not change `utils`; it only avoids referencing `dl.Runner` when `dl` is `None`. If Catalyst is missing, later code that relies on `utils.get_device()` may still fail, but the reported crash in cell 6 is fixed without altering cell 7.

Assumptions: The immediate goal is to unblock execution at cell 6 without changing model/training semantics; any subsequent Catalyst-specific API usage outside this snippet is unchanged and may require Catalyst to be installed.'
- What this solution (achieved 0.63824) has done: 'The crash happens because `catalyst` isn’t installed, so `utils` was set to `None` in cell 0; calling `utils.get_device()` in cell 7 then raises an `AttributeError`. The minimal fix is to make cell 7 robust: if `utils` is unavailable (or doesn’t provide `get_device`), fall back to choosing `"cuda"` when available else `"cpu"`. This preserves the intended semantics of selecting the active device while avoiding any dependency on Catalyst. No other cells or modeling logic are changed, and the variable name `device` remains identical for downstream compatibility.'
- What this solution (achieved 0.63824) has done: 'Diagnosis: Cell 9 assumes Catalyst is installed and that `CustomRunner` inherits `dl.Runner` (which provides `.train()`), but Catalyst isn’t available in this environment so the fallback `CustomRunner` is used. The fallback currently implements only `predict_batch` and `_handle_batch`, so calling `runner.train(...)` raises `AttributeError: 'CustomRunner' object has no attribute 'train'`. The minimal fix is to add a `train()` method to the fallback `CustomRunner` that runs the same core training loop semantics (forward, loss, backward, optimizer step, scheduler step) and writes a Catalyst-compatible `metrics.csv` into `logdir` so cell 10’s `utils.plot_metrics(...)` remains compatible. No model architecture, loss, or data logic is changed—only the missing training API is implemented.

Patch summary: In cell 9, define a deterministic fallback `train()` method on `CustomRunner` when Catalyst is missing. The method iterates over provided loaders, uses the existing `_handle_batch()` for loss/metric computation and optimization, steps the scheduler if provided, and logs epoch metrics to `../working/metrics.csv` in the format expected by Catalyst plotting utilities. This resolves the crash and preserves interfaces used by cell 10.'
- What this solution (achieved 0.63824) has done: 'Diagnosis: Cell 10 crashes because `catalyst` is not installed, so in cell 0 `utils` is set to `None`. Cell 10 unconditionally calls `utils.plot_metrics(...)`, causing `AttributeError: 'NoneType' object has no attribute 'plot_metrics'`. The training loop already writes `../working/metrics.csv` in the fallback runner, so we can safely plot from that file when Catalyst utils are unavailable.

Patch summary: Update only cell 10 to (1) use `utils.plot_metrics` when Catalyst is available, otherwise (2) read `../working/metrics.csv` produced by the fallback trainer and plot the requested metrics with matplotlib. This preserves all training/evaluation semantics and only fixes the plotting crash.

Updated cells: Only cell 10 is changed.

Compatibility notes for cell k+1: No variables used by cell 11 are modified; `runner`, `test_dataloader`, and `sub` behavior remain unchanged. This patch only affects metric visualization.

Assumptions: Matplotlib is available in the environment (typical for notebook runtimes); if not, the fallback print a small message instead of crashing.'

# 9. Code solution

## === cell 0
import json
import os
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    import catalyst.dl as dl
    import catalyst.dl.utils as utils
except ModuleNotFoundError:
    dl = None
    utils = None

import pandas as pd
import numpy as np




## === cell 1
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 2
def one_hot(categories, string):
    encoding = np.zeros((len(string), len(categories)))
    for idx, char in enumerate(string):
        encoding[idx, categories.index(char)] = 1
    return encoding


def featurize(entity):
    sequence = one_hot(list("ACGU"), entity["sequence"])
    structure = one_hot(list(".()"), entity["structure"])
    loop_type = one_hot(list("BEHIMSX"), entity["predicted_loop_type"])
    features = np.hstack([sequence, structure, loop_type])
    return features


def char_encode(index, features, feature_size):
    half_size = (feature_size - 1) // 2

    if index - half_size < 0:
        char_features = features[: index + half_size + 1]
        padding = np.zeros((int(half_size - index), char_features.shape[1]))
        char_features = np.vstack([padding, char_features])
    elif index + half_size + 1 > len(features):
        char_features = features[index - half_size :]
        padding = np.zeros(
            (int(half_size - (len(features) - index)) + 1, char_features.shape[1])
        )
        char_features = np.vstack([char_features, padding])
    else:
        char_features = features[index - half_size : index + half_size + 1]

    return char_features




## === cell 3
class VaxDataset(Dataset):
    def __init__(self, path, test=False, ids_keep=None, sn_filter_only=False):
        """
        Minimal, score-relevant change:
        - sn_filter_only=True keeps only SN_filter==1 training records (common competition practice)
        - ids_keep optionally restricts which record ids to include (for a light validation split)
        """
        self.path = path
        self.test = test
        self.ids_keep = set(ids_keep) if ids_keep is not None else None
        self.sn_filter_only = sn_filter_only
        self.features = []
        self.targets = []
        self.ids = []
        self.load_data()

    def load_data(self):
        with open(self.path, "r") as text:
            for line in text:
                records = json.loads(line)

                if (not self.test) and self.sn_filter_only:
                    if int(records.get("SN_filter", 0)) != 1:
                        continue

                if self.ids_keep is not None:
                    if records["id"] not in self.ids_keep:
                        continue

                features = featurize(records)

                for char_i in range(records["seq_scored"]):
                    char_features = char_encode(char_i, features, 21)
                    self.features.append(char_features)
                    self.ids.append("%s_%d" % (records["id"], char_i))

                if not self.test:
                    targets = np.stack(
                        [
                            records["reactivity"],
                            records["deg_Mg_pH10"],
                            records["deg_Mg_50C"],
                        ],
                        axis=1,
                    )
                    self.targets.extend(
                        [targets[char_i] for char_i in range(records["seq_scored"])]
                    )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index], dtype=torch.float32)
        if self.test:
            return x, self.ids[index]
        else:
            y = torch.tensor(self.targets[index], dtype=torch.float32)
            return x, y, self.ids[index]




## === cell 4
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class VaxModel(nn.Module):
    def __init__(self):
        super(VaxModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(14, 32, 1, 1),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=2, mode="linear"),
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, 1, 1),
        )
        self.layers2 = nn.Sequential(
            nn.GRU(42, 32),
        )
        self.final = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )

    def forward(self, features):
        features = self.layers(features)
        features = features.permute(1, 0, 2)
        features = self.layers2(features)
        final = self.final(features[0])
        return final[0, :, :]




## === cell 5
if utils is not None and hasattr(utils, "get_device"):
    device = utils.get_device()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 6
model = VaxModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

criterion = nn.MSELoss()




## === cell 7
def mcrmse_loss(y_true, y_pred, N=3):
    """
    Calculates competition eval metric
    """
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=0) / n)) / N




## === cell 8
train_path = "../input/stanford-covid-vaccine/train.json"

all_train_ids = []
with open(train_path, "r") as f:
    for line in f:
        r = json.loads(line)
        if int(r.get("SN_filter", 0)) == 1:
            all_train_ids.append(r["id"])

all_train_ids = np.array(sorted(set(all_train_ids)))
rng = np.random.RandomState(42)
rng.shuffle(all_train_ids)
n_val = max(1, int(0.1 * len(all_train_ids)))
val_ids = set(all_train_ids[:n_val])
tr_ids = set(all_train_ids[n_val:])

train_dataset = VaxDataset(train_path, sn_filter_only=True, ids_keep=tr_ids)
val_dataset = VaxDataset(train_path, sn_filter_only=True, ids_keep=val_ids)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_dataset = VaxDataset("../input/stanford-covid-vaccine/test.json", test=True)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=2000
)



## === cell 9
if dl is not None and hasattr(dl, "Runner"):

    class CustomRunner(dl.Runner):
        def predict_batch(self, batch):
            return (
                self.model(batch[0].to(self.device).permute(0, 2, 1).float()),
                batch[1],
            )

        def _handle_batch(self, batch):
            x, y = batch[0], batch[1]
            x = x.to(self.device).permute(0, 2, 1).float()
            y = y.to(self.device).float()
            y_hat = self.model(x)

            loss = criterion(y_hat, y)
            score = mcrmse_loss(y_hat, y)
            self.batch_metrics.update({"loss": loss, "metric": score})

            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

else:

    class CustomRunner:
        """Minimal fallback to avoid dependency on Catalyst when it's not installed."""

        def __init__(self, model=None, optimizer=None, device=None):
            self.model = model
            self.optimizer = optimizer
            self.device = (
                device
                if device is not None
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.batch_metrics = {}
            self.is_train_loader = False

        def predict_batch(self, batch):
            x = batch[0].to(self.device).permute(0, 2, 1).float()
            return self.model(x), batch[1]

        def predict_loader(self, loader):
            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    preds, ids = self.predict_batch(batch)
                    yield preds, ids

        def _handle_batch(self, batch):
            x, y = batch[0], batch[1]
            x = x.to(self.device).permute(0, 2, 1).float()
            y = y.to(self.device).float()
            y_hat = self.model(x)

            loss = criterion(y_hat, y)
            score = mcrmse_loss(y_hat, y)
            self.batch_metrics.update({"loss": loss, "metric": score})

            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        def train(
            self,
            model,
            optimizer,
            loaders,
            logdir,
            num_epochs,
            scheduler=None,
            verbose=False,
            load_best_on_end=True,
            **kwargs,
        ):
            self.model = model.to(self.device)
            self.optimizer = optimizer

            os.makedirs(logdir, exist_ok=True)
            metrics_path = os.path.join(logdir, "metrics.csv")
            with open(metrics_path, "w") as f:
                f.write("epoch,loader,loss,metric\n")

            for epoch in range(num_epochs):
                for loader_name, loader in loaders.items():
                    is_train = loader_name == "train"
                    self.is_train_loader = is_train
                    self.model.train() if is_train else self.model.eval()

                    sum_loss = 0.0
                    sum_metric = 0.0
                    n_batches = 0

                    for batch in loader:
                        if is_train:
                            self._handle_batch(batch)
                            if scheduler is not None:
                                scheduler.step()
                        else:
                            with torch.no_grad():
                                self._handle_batch(batch)

                        loss_val = float(
                            self.batch_metrics["loss"].detach().cpu().item()
                        )
                        metric_val = float(self.batch_metrics["metric"])
                        sum_loss += loss_val
                        sum_metric += metric_val
                        n_batches += 1

                    avg_loss = sum_loss / max(n_batches, 1)
                    avg_metric = sum_metric / max(n_batches, 1)
                    with open(metrics_path, "a") as f:
                        f.write(f"{epoch},{loader_name},{avg_loss},{avg_metric}\n")
            return self




## === cell 10
loaders = {"train": train_dataloader, "valid": val_dataloader}
runner = CustomRunner(device=device)

runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    logdir="../working",
    num_epochs=5,
    scheduler=scheduler,
    verbose=False,
    load_best_on_end=True,
)



## === cell 11
if utils is not None and hasattr(utils, "plot_metrics"):
    utils.plot_metrics(logdir="../working", metrics=["loss", "metric"])
else:
    metrics_path = os.path.join("../working", "metrics.csv")
    try:
        import matplotlib.pyplot as plt

        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, m in zip(axes, ["loss", "metric"]):
                if m in df_metrics.columns:
                    for loader_name in sorted(df_metrics["loader"].unique()):
                        dfl = df_metrics[df_metrics["loader"] == loader_name]
                        ax.plot(dfl["epoch"].values, dfl[m].values, label=loader_name)
                    ax.set_title(m)
                    ax.set_xlabel("epoch")
                    ax.set_ylabel(m)
                    ax.legend()
                else:
                    ax.set_visible(False)
            plt.tight_layout()
            plt.show()
        else:
            print(f"metrics file not found: {metrics_path}")
    except Exception as e:
        print(f"Could not plot metrics (fallback). Reason: {e}")



## === cell 12
model.eval()
val_preds = []
val_trues = []
with torch.no_grad():
    for xb, yb, _ids in val_dataloader:
        xb = xb.to(device).permute(0, 2, 1).float()
        yb = yb.to(device).float()
        pred = model(xb)
        val_preds.append(pred)
        val_trues.append(yb)

val_preds = torch.cat(val_preds, dim=0)
val_trues = torch.cat(val_trues, dim=0)
print("Local validation MCRMSE (approx):", mcrmse_loss(val_trues, val_preds))



## === cell 13
sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
sub_targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

pred_map = {}

PRED_MIN, PRED_MAX = -0.5, 2.0

model.eval()
with torch.no_grad():
    if hasattr(runner, "predict_loader"):
        pred_iter = runner.predict_loader(test_dataloader)
        for preds, ids in pred_iter:
            preds = torch.clamp(preds, PRED_MIN, PRED_MAX)
            preds_np = preds.detach().cpu().numpy()
            ids_list = list(ids)
            for i, id_seqpos in enumerate(ids_list):
                pred_map[id_seqpos] = preds_np[i]
    else:
        for xb, ids in test_dataloader:
            xb = xb.to(device).permute(0, 2, 1).float()
            preds = model(xb)
            preds = torch.clamp(preds, PRED_MIN, PRED_MAX).detach().cpu().numpy()
            for i, id_seqpos in enumerate(list(ids)):
                pred_map[id_seqpos] = preds[i]

for idx, row in sub.iterrows():
    key = row["id_seqpos"]
    if key in pred_map:
        sub.at[idx, "reactivity"] = float(pred_map[key][0])
        sub.at[idx, "deg_Mg_pH10"] = float(pred_map[key][1])
        sub.at[idx, "deg_Mg_50C"] = float(pred_map[key][2])

sub = sub[["id_seqpos"] + sub_targets]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
