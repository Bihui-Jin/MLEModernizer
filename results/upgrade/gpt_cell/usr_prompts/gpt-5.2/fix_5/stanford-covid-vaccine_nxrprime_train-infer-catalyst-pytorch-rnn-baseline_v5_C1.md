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

0.47986

# 6. Current score

0.38205

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63824) has done: 'The crash happens immediately because the notebook imports `catalyst` (`catalyst.dl` and `catalyst.dl.utils`) but that package is not installed in the environment. To keep the rest of the notebook intact, I replace those imports with a tiny local stub that provides `utils.get_device()` (the only referenced API in the next cell). The stub deterministically return a `torch.device` based on CUDA availability, matching typical Catalyst behavior, and preserve the `dl` and `utils` names expected by later cells. No model/training logic is changed—this only unblocks execution.'
- What this solution (achieved 0.63824) has done: 'Diagnosis: Cell 7 subclasses `dl.Runner`, but `catalyst` is not installed in this environment, so earlier code in cell 0 creates a `_CatalystDLStub` without a `Runner` attribute. When Python evaluates `class CustomRunner(dl.Runner)`, it raises `AttributeError: '_CatalystDLStub' object has no attribute 'Runner'`.  
Patch summary: In cell 7 only, define `CustomRunner` to inherit from `dl.Runner` when it exists, otherwise inherit from `object` so the class can be created without Catalyst. Keep the same `predict_batch` and `_handle_batch` logic unchanged to preserve training/inference semantics if Catalyst is available.  
Updated cells: Only cell 7 is modified.  
Compatibility notes for cell k+1: Cell 8 only uses `utils.get_device()`; no interface/variable changes affect it.  
Assumptions: Later cells may instantiate/use `CustomRunner` only when Catalyst is installed; in this environment the fallback simply prevents the immediate class-definition crash.'
- What this solution (achieved 0.63824) has done: 'Diagnosis: `catalyst` is not installed, so `dl.Runner` is missing and `RunnerBase` falls back to `object`. As a result, `CustomRunner` does not inherit an `__init__` that accepts `device=...`, causing `TypeError: CustomRunner() takes no arguments` when instantiated in cell 9. The fix is to provide a minimal Runner-compatible stub inside cell 9 only, implementing `__init__`, `train`, and `predict_loader` methods used later, while keeping the existing training loop semantics and `CustomRunner`’s batch logic intact.

Patch summary: In cell 9, create a lightweight `RunnerBase` implementation when Catalyst is unavailable; it stores `device`, runs the same epoch/batch loop calling `self._handle_batch`, and provides `predict_loader` that calls `self.predict_batch`. No model/loss/optimizer/scheduler logic is changed; only the missing runner interface is supplied so `CustomRunner(device=device)` works and cell 10 can call `runner.predict_loader(...)`.

Updated cells: cell 9 only (buggy cell).

Compatibility notes for cell k+1: The patched runner exposes `predict_loader(loader=...)` yielding `(predictions, ids)` exactly as cell 10 expects; `predictions` is a torch Tensor on the same device and `ids` is the list of id strings from the dataset. `runner.train(...)` accepts the same arguments used in cell 9 and trains the same `model` with the same `optimizer`, `criterion`, and `scheduler`.

Assumptions: Catalyst is not available (as indicated by the existing stub), so implementing the minimal runner API is sufficient; CUDA may or may not be available, but existing code already uses `.cuda()` and `utils.get_device()`.'
- What this solution (achieved 0.38205) has done: 'The crash happens because `catalyst` is not installed, so `dl` is a stub without `Runner`, and `CustomRunner` is defined earlier inheriting from `object` (which has no `__init__` accepting `device=`). In cell 9 you later define a fallback `RunnerBase` with an `__init__`, but `CustomRunner` is already created and doesn’t inherit from that fallback class. The minimal fix is to redefine `CustomRunner` inside cell 9 to inherit from the correct `RunnerBase` (either Catalyst’s `dl.Runner` if available, or the local fallback), preserving the same `predict_batch` and `_handle_batch` logic.'

# 9. Code solution

## === cell 0
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.nn.functional as F

try:
    import catalyst.dl as dl  # type: ignore
    import catalyst.dl.utils as utils  # type: ignore
except ModuleNotFoundError:

    class _CatalystUtilsStub:
        @staticmethod
        def get_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _CatalystDLStub:
        pass

    dl = _CatalystDLStub()
    utils = _CatalystUtilsStub()

import pandas as pd
import numpy as np


## === cell 1
utils.get_device()


## === cell 2
def one_hot(categories, string):
    encoding = np.zeros((len(string), len(categories)))
    for idx, char in enumerate(string):
        encoding[idx, categories.index(char)] = 1
    return encoding

def featurize(entity):
    sequence = one_hot(list('ACGU'), entity['sequence'])
    structure = one_hot(list('.()'), entity['structure'])
    loop_type = one_hot(list('BEHIMSX'), entity['predicted_loop_type'])
    features = np.hstack([sequence, structure, loop_type])
    return features 

def char_encode(index, features, feature_size):
    half_size = (feature_size - 1) // 2
    
    if index - half_size < 0:
        char_features = features[:index+half_size+1]
        padding = np.zeros((int(half_size - index), char_features.shape[1]))
        char_features = np.vstack([padding, char_features])
    elif index + half_size + 1 > len(features):
        char_features = features[index-half_size:]
        padding = np.zeros((int(half_size - (len(features) - index))+1, char_features.shape[1]))
        char_features = np.vstack([char_features, padding])
    else:
        char_features = features[index-half_size:index+half_size+1]
    
    return char_features


## === cell 3
class VaxDataset(Dataset):
    def __init__(self, path, test=False):
        self.path = path
        self.test = test
        self.features = []
        self.targets = []
        self.ids = []
        self.load_data()
    
    def load_data(self):
        with open(self.path, 'r') as text:
            for line in text:
                records = json.loads(line)
                features = featurize(records)
                
                for char_i in range(records['seq_scored']):
                    char_features = char_encode(char_i, features, 21)
                    self.features.append(char_features)
                    self.ids.append('%s_%d' % (records['id'], char_i))
                        
                if not self.test:
                    targets = np.stack([records['reactivity'], records['deg_Mg_pH10'], records['deg_Mg_50C']], axis=1)
                    self.targets.extend([targets[char_i] for char_i in range(records['seq_scored'])])
                    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        if self.test:
            return self.features[index], self.ids[index]
        else:
            return self.features[index], self.targets[index], self.ids[index]


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
            nn.Dropout(0.2),
            nn.Conv1d(32, 1, 1, 1),
            nn.PReLU(),
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(21, 32),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )
    
    def forward(self, features):
        return self.layers(features)


## === cell 5
model = VaxModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.005, momentum=0.9)
criterion = nn.MSELoss()


## === cell 6
train_dataset = VaxDataset('../input/stanford-covid-vaccine/train.json')
train_dataloader = DataLoader(train_dataset, 16, shuffle=True, num_workers=4, pin_memory=True)


## === cell 7
RunnerBase = getattr(dl, "Runner", object)


class CustomRunner(RunnerBase):

    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device).permute(0, 2, 1).float()), batch[1]

    def _handle_batch(self, batch):
        x, y = batch[0], batch[1]
        x = x.cuda().permute(0, 2, 1).float()
        y = y.cuda().float()
        y_hat = self.model(x)

        loss = criterion(y_hat, y)
        self.batch_metrics.update({"loss": loss})

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


## === cell 8
device = utils.get_device()


## === cell 9
test_dataset = VaxDataset("../input/stanford-covid-vaccine/test.json", test=True)
test_dataloader = DataLoader(
    test_dataset, 16, num_workers=4, drop_last=False, pin_memory=True
)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=2000
)

loaders = {"train": train_dataloader}

if not hasattr(dl, "Runner"):

    class RunnerBase:
        def __init__(self, device=None):
            self.device = device if device is not None else utils.get_device()
            self.model = None
            self.optimizer = None
            self.scheduler = None
            self.is_train_loader = False
            self.batch_metrics = {}

        def train(
            self,
            model,
            optimizer,
            loaders,
            logdir=None,
            num_epochs=1,
            scheduler=None,
            verbose=False,
            load_best_on_end=False,
        ):
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler

            train_loader = loaders.get("train")
            if train_loader is None:
                raise ValueError("Expected loaders to contain a 'train' DataLoader.")

            self.model.train()
            for _epoch in range(num_epochs):
                self.is_train_loader = True
                for batch in train_loader:
                    self.batch_metrics = {}
                    self._handle_batch(batch)
                    if self.scheduler is not None:
                        self.scheduler.step()
            self.is_train_loader = False
            return None

        def predict_loader(self, loader):
            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    yield self.predict_batch(batch)

else:
    RunnerBase = dl.Runner


class CustomRunner(RunnerBase):
    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device).permute(0, 2, 1).float()), batch[1]

    def _handle_batch(self, batch):
        x, y = batch[0], batch[1]
        x = x.cuda().permute(0, 2, 1).float()
        y = y.cuda().float()
        y_hat = self.model(x)

        loss = criterion(y_hat, y)
        self.batch_metrics.update({"loss": loss})

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


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


## === cell 10
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv', index_col='id_seqpos')

for predictions, ids in runner.predict_loader(loader=test_dataloader):
    sub.loc[ids, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = predictions.detach().cpu().numpy()


## === cell 11
sub.to_csv('submission.csv')
