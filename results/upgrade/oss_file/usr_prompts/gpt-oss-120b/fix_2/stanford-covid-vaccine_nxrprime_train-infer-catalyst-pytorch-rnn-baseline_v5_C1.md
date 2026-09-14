# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.47986

# 6. Current score

0.63824

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.63824) has done: 'I fixed the missing imports, removed the unavailable catalyst dependency, replaced the broken custom runner with a simple PyTorch training loop, and corrected the model architecture so that the input dimensions match the flattened feature window. These changes let the script load the data, train a lightweight MLP, generate predictions for the test set, and write a proper submission.csv file.'

# 9. Code solution

## === cell 0
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




## === cell 1
def one_hot(categories, string):
    encoding = np.zeros((len(string), len(categories)), dtype=np.float32)
    for idx, char in enumerate(string):
        encoding[idx, categories.index(char)] = 1.0
    return encoding


def featurize(entity):
    sequence = one_hot(list("ACGU"), entity["sequence"])
    structure = one_hot(list(".()"), entity["structure"])
    loop_type = one_hot(list("BEHIMSX"), entity["predicted_loop_type"])
    features = np.hstack([sequence, structure, loop_type])  # shape (seq_len, 14)
    return features


def char_encode(index, features, window_size):
    half = (window_size - 1) // 2
    start = max(0, index - half)
    end = min(len(features), index + half + 1)
    char_feat = features[start:end]
    if start > index - half:  # pad left
        pad_len = (index - half) - start
        padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
        char_feat = np.vstack([padding, char_feat])
    if end < index + half + 1:  # pad right
        pad_len = (index + half + 1) - end
        padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
        char_feat = np.vstack([char_feat, padding])
    return char_feat




## === cell 2
class VaxDataset(Dataset):
    def __init__(self, path, test=False):
        self.test = test
        self.features = []
        self.targets = []
        self.ids = []
        self._load(path)

    def _load(self, path):
        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)
                feats = featurize(rec)  # (seq_len, 14)
                for i in range(rec["seq_scored"]):
                    win = char_encode(i, feats, 21)  # (21, 14)
                    self.features.append(torch.from_numpy(win))
                    self.ids.append(f"{rec['id']}_{i}")
                    if not self.test:
                        tgt = np.stack(
                            [rec["reactivity"], rec["deg_Mg_pH10"], rec["deg_Mg_50C"]],
                            axis=1,
                        )  # (68,3)
                        self.targets.append(torch.from_numpy(tgt[i].astype(np.float32)))
        self.features = [f.float() for f in self.features]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.test:
            return self.features[idx], self.ids[idx]
        else:
            return self.features[idx], self.targets[idx], self.ids[idx]




## === cell 3
class SimpleMlp(nn.Module):
    def __init__(self, window_len=21, feat_dim=14, hidden=128):
        super().__init__()
        input_dim = window_len * feat_dim  # 21*14 = 294
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMlp().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.MSELoss()



## === cell 5
train_dataset = VaxDataset("../input/stanford-covid-vaccine/train.json")
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/705083480.py in <cell line: 0>()
----> 1 train_dataset = VaxDataset("../input/stanford-covid-vaccine/train.json")
      2 train_loader = DataLoader(
      3     train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
      4 )
      5 

/tmp/ipykernel_55/1845910896.py in __init__(self, path, test)
      5         self.targets = []
      6         self.ids = []
----> 7         self._load(path)
      8 
      9     def _load(self, path):

/tmp/ipykernel_55/1845910896.py in _load(self, path)
     13                 feats = featurize(rec)  # (seq_len, 14)
     14                 for i in range(rec["seq_scored"]):
---> 15                     win = char_encode(i, feats, 21)  # (21, 14)
     16                     self.features.append(torch.from_numpy(win))
     17                     self.ids.append(f"{rec['id']}_{i}")

/tmp/ipykernel_55/2707082864.py in char_encode(index, features, window_size)
     21     if start > index - half:  # pad left
     22         pad_len = (index - half) - start
---> 23         padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
     24         char_feat = np.vstack([padding, char_feat])
     25     if end < index + half + 1:  # pad right

ValueError: negative dimensions are not allowed

## === cell 6
def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    for feats, targets, _ in loader:
        feats = feats.to(device)  # (B, 21, 14)
        targets = targets.to(device)  # (B, 3)
        optimizer.zero_grad()
        preds = model(feats)  # (B, 3)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)




## === cell 7
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    loss = train_one_epoch(train_loader)
    print(f"Epoch {epoch}/{num_epochs} - Train loss: {loss:.6f}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2650223295.py in <cell line: 0>()
      1 num_epochs = 5
      2 for epoch in range(1, num_epochs + 1):
----> 3     loss = train_one_epoch(train_loader)
      4     print(f"Epoch {epoch}/{num_epochs} - Train loss: {loss:.6f}")
      5 

NameError: name 'train_loader' is not defined

## === cell 8
test_dataset = VaxDataset("../input/stanford-covid-vaccine/test.json", test=True)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
preds_list = []
ids_list = []
with torch.no_grad():
    for feats, ids in test_loader:
        feats = feats.to(device)
        out = model(feats)  # (B,3)
        preds_list.append(out.cpu())
        ids_list.extend(ids)

all_preds = torch.cat(preds_list, dim=0).numpy()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4275053354.py in <cell line: 0>()
----> 1 test_dataset = VaxDataset("../input/stanford-covid-vaccine/test.json", test=True)
      2 test_loader = DataLoader(
      3     test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
      4 )
      5 

/tmp/ipykernel_55/1845910896.py in __init__(self, path, test)
      5         self.targets = []
      6         self.ids = []
----> 7         self._load(path)
      8 
      9     def _load(self, path):

/tmp/ipykernel_55/1845910896.py in _load(self, path)
     13                 feats = featurize(rec)  # (seq_len, 14)
     14                 for i in range(rec["seq_scored"]):
---> 15                     win = char_encode(i, feats, 21)  # (21, 14)
     16                     self.features.append(torch.from_numpy(win))
     17                     self.ids.append(f"{rec['id']}_{i}")

/tmp/ipykernel_55/2707082864.py in char_encode(index, features, window_size)
     21     if start > index - half:  # pad left
     22         pad_len = (index - half) - start
---> 23         padding = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
     24         char_feat = np.vstack([padding, char_feat])
     25     if end < index + half + 1:  # pad right

ValueError: negative dimensions are not allowed

## === cell 9
sub = pd.read_csv(
    "../input/stanford-covid-vaccine/sample_submission.csv", index_col="id_seqpos"
)
sub.loc[ids_list, ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]] = all_preds



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1651050053.py in <cell line: 0>()
      3 )
      4 # Ensure the order of ids matches submission index
----> 5 sub.loc[ids_list, ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]] = all_preds
      6 

NameError: name 'all_preds' is not defined

## === cell 10
sub.to_csv("submission.csv")
