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

0.4772

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 1
def one_hot(categories, string):
    encoding = np.zeros((len(string), len(categories)), dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    for idx, char in enumerate(string):
        encoding[idx, cat_to_idx[char]] = 1.0
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
        padding = np.zeros(
            (int(half_size - index), char_features.shape[1]), dtype=np.float32
        )
        char_features = np.vstack([padding, char_features])
    elif index + half_size + 1 > len(features):
        char_features = features[index - half_size :]
        padding_len = int(half_size - (len(features) - index - 1))
        padding = np.zeros((padding_len, char_features.shape[1]), dtype=np.float32)
        char_features = np.vstack([char_features, padding])
    else:
        char_features = features[index - half_size : index + half_size + 1]
    return char_features




## === cell 2
class VaxDataset(Dataset):
    def __init__(self, path, test=False):
        self.path = path
        self.test = test
        self.features = []
        self.targets = []
        self.ids = []
        self.load_data()

    def load_data(self):
        with open(self.path, "r") as f:
            data = json.load(f)  # file contains a list of records
        for records in data:
            features = featurize(records)

            for char_i in range(records["seq_scored"]):
                char_features = char_encode(char_i, features, 21)
                self.features.append(char_features)
                self.ids.append(f"{records['id']}_{char_i}")

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

    def __getitem__(self, idx):
        if self.test:
            return self.features[idx], self.ids[idx]
        else:
            return self.features[idx], self.targets[idx], self.ids[idx]




## === cell 3
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class VaxModel(nn.Module):
    def __init__(self):
        super(VaxModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(14, 32, kernel_size=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, kernel_size=1, stride=1),
        )
        self.layers2 = nn.GRU(42, 32, batch_first=True)
        self.final = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )

    def forward(self, features):
        x = self.layers(features)  # (B, 1, L')
        x = x.permute(0, 2, 1)  # (B, L', 1)
        padding = torch.zeros(x.size(0), x.size(1), 41, device=x.device)
        x = torch.cat([x, padding], dim=2)  # (B, L', 42)
        _, hn = self.layers2(x)  # hn: (1, B, 32)
        hn = hn.squeeze(0)  # (B, 32)
        out = self.final(hn)  # (B, 3)
        return out




## === cell 4
model = VaxModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=2000, mode="triangular"
)



## === cell 5
train_dataset = VaxDataset("../input/stanford-covid-vaccine/train.json")
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)

test_dataset = VaxDataset("../input/stanford-covid-vaccine/test.json", test=True)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
JSONDecodeError                           Traceback (most recent call last)
/tmp/ipykernel_55/2652078127.py in <cell line: 0>()
----> 1 train_dataset = VaxDataset("../input/stanford-covid-vaccine/train.json")
      2 train_loader = DataLoader(
      3     train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
      4 )
      5 

/tmp/ipykernel_55/3220969996.py in __init__(self, path, test)
      6         self.targets = []
      7         self.ids = []
----> 8         self.load_data()
      9 
     10     def load_data(self):

/tmp/ipykernel_55/3220969996.py in load_data(self)
     10     def load_data(self):
     11         with open(self.path, "r") as f:
---> 12             data = json.load(f)  # file contains a list of records
     13         for records in data:
     14             features = featurize(records)

/usr/lib/python3.11/json/__init__.py in load(fp, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
    291     kwarg; otherwise ``JSONDecoder`` is used.
    292     """
--> 293     return loads(fp.read(),
    294         cls=cls, object_hook=object_hook,
    295         parse_float=parse_float, parse_int=parse_int,

/usr/lib/python3.11/json/__init__.py in loads(s, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)
    344             parse_int is None and parse_float is None and
    345             parse_constant is None and object_pairs_hook is None and not kw):
--> 346         return _default_decoder.decode(s)
    347     if cls is None:
    348         cls = JSONDecoder

/usr/lib/python3.11/json/decoder.py in decode(self, s, _w)
    338         end = _w(s, end).end()
    339         if end != len(s):
--> 340             raise JSONDecodeError("Extra data", s, end)
    341         return obj
    342 

JSONDecodeError: Extra data: line 2 column 1 (char 5340)

## === cell 6
def mcrmse_loss(y_true, y_pred, N=3):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=0) / n)) / N


num_epochs = 5
model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    epoch_metrics = []
    for batch in train_loader:
        x, y, _ = batch
        x = x.to(device).permute(0, 2, 1).float()  # (B, C, L)
        y = y.to(device).float()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_losses.append(loss.item())
        epoch_metrics.append(mcrmse_loss(y, y_hat))
    avg_loss = np.mean(epoch_losses)
    avg_metric = np.mean(epoch_metrics)
    print(
        f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.5f} - mcrmse: {avg_metric:.5f}"
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/721571812.py in <cell line: 0>()
     11     epoch_losses = []
     12     epoch_metrics = []
---> 13     for batch in train_loader:
     14         x, y, _ = batch
     15         x = x.to(device).permute(0, 2, 1).float()  # (B, C, L)

NameError: name 'train_loader' is not defined

## === cell 7
model.eval()
preds = []
ids = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, id_batch = batch
        x_batch = x_batch.to(device).permute(0, 2, 1).float()
        out = model(x_batch)  # (B, 3)
        preds.append(out.cpu().numpy())
        ids.extend(id_batch)

preds = np.concatenate(preds, axis=0)

sub_path = "../input/stanford-covid-vaccine/sample_submission.csv"
sub = pd.read_csv(sub_path, index_col="id_seqpos")
pred_df = pd.DataFrame(
    preds, index=ids, columns=["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
)
sub.update(pred_df)
sub.to_csv("submission.csv")
print("Submission file written to submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3987317104.py in <cell line: 0>()
      3 ids = []
      4 with torch.no_grad():
----> 5     for batch in test_loader:
      6         x_batch, id_batch = batch
      7         x_batch = x_batch.to(device).permute(0, 2, 1).float()

NameError: name 'test_loader' is not defined
