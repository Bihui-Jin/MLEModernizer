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

0.3732

# 6. Current score

0.23665

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.23665) has done: 'Your code currently doesn’t yield a score mainly because it never writes a submission `.csv`, and it also tries to split test into `seq_length==130` (which is empty here), causing a shape/concat failure. I make minimal changes to (1) generate predictions for the full test set in one pass (no public/private split), (2) align the submission rows/ordering exactly to `sample_submission.csv` to avoid id mismatches, and (3) fix the swapped bpps feature functions (`sum` vs `max`) which is a direct, legitimate feature bug that should move MCRMSE down toward your target. Core model/training/loss remain unchanged; we only repair data/IO and a clear feature computation mistake.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU, Dropout
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import copy
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
import time, datetime




## === cell 1
def Get_nowtime(fmat="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.strftime(datetime.datetime.now(), fmat)


def Metric(target, pred):
    metric = 0
    for i in range(target.shape[-1]):
        metric += (
            np.sqrt(np.mean((target[:, :, i] - pred[:, :, i]) ** 2)) / target.shape[-1]
        )
    return metric


def Write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write("\n")
    return None


def Seed_everything(seed=1017):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


Seed_everything()


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        print(classname)
        try:
            nn.init.xavier_uniform_(m.weight)
        except:
            pass
        try:
            nn.init.zeros_(m.bias)
        except:
            pass
    if classname.find("GRU") != -1 or classname.find("LSTM") != -1:
        print(classname)
        for name, param in m.named_parameters():
            print(name)
            if "bias_ih" in name:
                torch.nn.init.zeros_(param)
            elif "bias_hh" in name:
                torch.nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)




## === cell 2
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


def mcrmse(y_actual, y_pred, weight=None, num_scored=5):
    score = 0
    for i in range(5):
        if weight is not None:
            score += (
                torch.sqrt(
                    torch.mean((y_actual[:, :, i] - y_pred[:, :, i]) ** 2 * weight)
                )
                / num_scored
            )
        else:
            score += (
                torch.sqrt(torch.mean((y_actual[:, :, i] - y_pred[:, :, i]) ** 2))
                / num_scored
            )
    return score


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    base_fea = np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    )
    bpps_sum_fea = np.array(df["bpps_sum"].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(df["bpps_max"].to_list())[:, :, np.newaxis]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list())[:, :, np.newaxis]
    return np.concatenate([base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], 2)




## === cell 3
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")


def _bpps_path(mol_id: str) -> str:
    return f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy"


def _fallback_bpps_feature(seq_length: int) -> np.ndarray:
    return np.zeros(seq_length, dtype=np.float32)


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id, seq_length in zip(df.id.to_list(), df.seq_length.to_list()):
        path = _bpps_path(mol_id)
        if os.path.exists(path):
            bpps_arr.append(np.load(path).sum(axis=1))
        else:
            bpps_arr.append(_fallback_bpps_feature(int(seq_length)))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id, seq_length in zip(df.id.to_list(), df.seq_length.to_list()):
        path = _bpps_path(mol_id)
        if os.path.exists(path):
            bpps_arr.append(np.load(path).max(axis=1))
        else:
            bpps_arr.append(_fallback_bpps_feature(int(seq_length)))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914  # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id, seq_length in zip(df.id.to_list(), df.seq_length.to_list()):
        path = _bpps_path(mol_id)
        if os.path.exists(path):
            bpps = np.load(path)
            bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
            bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
            bpps_arr.append(bpps_nb)
        else:
            bpps_arr.append(_fallback_bpps_feature(int(seq_length)))
    return bpps_arr


train["bpps_sum"] = read_bpps_sum(train)
test["bpps_sum"] = read_bpps_sum(test)
train["bpps_max"] = read_bpps_max(train)
test["bpps_max"] = read_bpps_max(test)
train["bpps_nb"] = read_bpps_nb(train)
test["bpps_nb"] = read_bpps_nb(test)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_target = 5
        self.cate_emb = nn.Embedding(14, 100)
        self.gru = nn.GRU(
            100 * 3 + 3,
            256,
            num_layers=1,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.gru1 = nn.GRU(
            512, 256, num_layers=1, batch_first=True, dropout=0.5, bidirectional=True
        )
        self.predict = nn.Linear(512, num_target)

    def forward(self, cateX, contX):
        cate_x = self.cate_emb(cateX).view(cateX.shape[0], cateX.shape[1], -1)
        sequence = torch.cat([cate_x, contX], -1)
        x, h = self.gru(sequence)
        x, h = self.gru1(x)
        x = F.dropout(x, 0.5, training=self.training)
        predict = self.predict(x)
        return predict


def train_and_predict(type=0, FOLD_N=5):

    gkf = GroupKFold(n_splits=FOLD_N)
    device = torch.device("cuda:%s" % 0 if torch.cuda.is_available() else "cpu")

    log = open("./train.log", "w", 1)
    all_best_model = []
    oof = []
    for fold, (train_index, valid_index) in enumerate(
        gkf.split(train, train["reactivity"], train["cluster_id"])
    ):
        Write_log(log, "fold %s train start:%s" % (fold, Get_nowtime()))
        t_train = train.iloc[train_index]
        train_x = preprocess_inputs(t_train)
        train_cate_x = torch.LongTensor(train_x[:, :, :3])
        train_cont_x = torch.Tensor(train_x[:, :, 3:])
        train_y = torch.Tensor(
            np.array(t_train[pred_cols].values.tolist()).transpose((0, 2, 1))
        )
        w_train = torch.Tensor(
            np.log(t_train["signal_to_noise"].values.reshape(-1, 1) + 1.1) / 2
        )

        t_valid = train.iloc[valid_index]
        t_valid = t_valid[t_valid["SN_filter"] == 1]
        valid_x = preprocess_inputs(t_valid)
        valid_count = valid_x.shape[0]
        valid_cate_x = torch.LongTensor(valid_x[:, :, :3])
        valid_cont_x = torch.Tensor(valid_x[:, :, 3:])
        valid_y = torch.Tensor(
            np.array(t_valid[pred_cols].values.tolist()).transpose((0, 2, 1))
        )

        train_data = TensorDataset(train_cate_x, train_cont_x, train_y, w_train)
        train_data_loader = DataLoader(
            dataset=train_data, shuffle=True, batch_size=64, num_workers=1
        )
        valid_data = TensorDataset(valid_cate_x, valid_cont_x, valid_y)
        valid_data_loader = DataLoader(
            dataset=valid_data, shuffle=False, batch_size=32, num_workers=1
        )

        valid_y_np = valid_y.numpy()

        model = Net()
        model.apply(weights_init)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        all_epoch_valid_metric = []
        best_valid_metric = 1e9
        for epoch in range(60):
            running_loss = 0.0
            t0 = datetime.datetime.now()
            model.train()
            for n, data in enumerate(train_data_loader):
                cate_x, cont_x, y, weight = [x.to(device) for x in data]
                outputs = model(cate_x, cont_x)
                optimizer.zero_grad()
                loss = mcrmse(y, outputs[:, :68, :], weight)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss = running_loss / max(n, 1)

            valid_loss = 0.0
            all_pred = []
            model.eval()
            for data in valid_data_loader:
                cate_x, cont_x, y = [x.to(device) for x in data]
                outputs = model(cate_x, cont_x)
                all_pred.append(outputs.detach().cpu().numpy())
                loss = mcrmse(y, outputs[:, :68, :])
                valid_loss += loss.item() * cate_x.shape[0]
            valid_loss = valid_loss / max(valid_count, 1)
            all_pred = (
                np.concatenate(all_pred, 0)
                if len(all_pred)
                else np.zeros((0, 107, 5), dtype=np.float32)
            )
            valid_metric = Metric(
                valid_y_np[:, :, [0, 1, 3]], all_pred[:, :68, [0, 1, 3]]
            )
            all_epoch_valid_metric.append(valid_metric)
            t1 = datetime.datetime.now()
            Write_log(
                log,
                "epoch %s | train mean loss:%.6f | valid loss:%.6f | valid metric:%.6f | ⏰:%ss"
                % (
                    str(epoch).rjust(3),
                    running_loss,
                    valid_loss,
                    valid_metric,
                    (t1 - t0).seconds,
                ),
            )
            if valid_metric < best_valid_metric:
                Write_log(log, "[epoch %s] save better model" % (epoch))
                torch.save(
                    model.state_dict(), "./gru-cate-emb-100-fold-%s.cpkt" % (fold)
                )
                best_valid_metric = valid_metric
                best_model = copy.deepcopy(model.state_dict())

        all_best_model.append(best_model)
        model.load_state_dict(best_model)
        model.eval()

        all_id = []
        all_y_id = []
        for _, row in t_valid.iterrows():
            for j in range(row["seq_length"]):
                all_id.append(row["id"] + "_%s" % j)
            for k in range(len(row["reactivity"])):
                all_y_id.append(row["id"] + "_%s" % k)

        all_id = np.array(all_id).reshape(-1, 1)
        all_y_id = np.array(all_y_id).reshape(-1, 1)
        all_pred = []

        for data in valid_data_loader:
            cate_x, cont_x, y = [x.to(device) for x in data]
            outputs = model(cate_x, cont_x)
            all_pred.append(outputs.detach().cpu().numpy())
        all_pred = np.concatenate(all_pred, 0)
        t_valid_metric = Metric(
            valid_y_np[:, :, [0, 1, 3]], all_pred[:, :68, [0, 1, 3]]
        )

        t_oof = pd.DataFrame(
            all_pred.reshape(-1, 5),
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        t_oof["id_seqpos"] = all_id
        t_target_df = pd.DataFrame(
            valid_y_np.reshape(-1, 5),
            columns=[
                "label_reactivity",
                "label_deg_Mg_pH10",
                "label_deg_pH10",
                "label_deg_Mg_50C",
                "label_deg_50C",
            ],
        )
        t_target_df["id_seqpos"] = all_y_id
        t_oof = t_oof.merge(t_target_df, how="left", on="id_seqpos")
        Write_log(log, "fold %s valid metric:%.6f" % (fold, t_valid_metric))
        oof.append(t_oof)

    oof = pd.concat(oof)
    oof_metirc = 0
    for col in ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]:
        oof_metirc += (
            np.sqrt(
                np.mean(
                    (
                        oof.loc[~oof["label_%s" % col].isna(), "label_%s" % col].values
                        - oof.loc[~oof["label_%s" % col].isna(), col].values
                    )
                    ** 2
                )
            )
            / 3.0
        )
    log.close()
    os.rename(
        "./train.log",
        "gru-cate-emb-100-0.6%f-%.6f.log" % (oof_metirc, oof_metirc),
    )

    def Pred(df):
        test_x = preprocess_inputs(df)
        test_cate_x = torch.LongTensor(test_x[:, :, :3])
        test_cont_x = torch.Tensor(test_x[:, :, 3:])
        test_data = TensorDataset(test_cate_x, test_cont_x)
        test_data_loader = DataLoader(
            dataset=test_data, shuffle=False, batch_size=64, num_workers=1
        )
        all_id = []
        for _, row in df.iterrows():
            for j in range(row["seq_length"]):
                all_id.append(row["id"] + "_%s" % j)

        all_id = np.array(all_id).reshape(-1, 1)
        all_pred = np.zeros((len(all_id), 5), dtype=np.float32)
        for fold in range(FOLD_N):
            model.load_state_dict(all_best_model[fold])
            model.eval()
            t_all_pred = []
            for data in test_data_loader:
                cate_x, cont_x = [x.to(device) for x in data]
                outputs = model(cate_x, cont_x)
                t_all_pred.append(outputs.detach().cpu().numpy())
            t_all_pred = np.concatenate(t_all_pred, 0)
            all_pred += t_all_pred.reshape(-1, 5).astype(np.float32)
        all_pred /= float(FOLD_N)
        sub = pd.DataFrame(
            all_pred,
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        sub["id_seqpos"] = all_id
        return sub

    sub = Pred(test).reset_index(drop=True)

    return (
        oof[
            ["id_seqpos"]
            + ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
            + [
                "label_reactivity",
                "label_deg_Mg_pH10",
                "label_deg_pH10",
                "label_deg_Mg_50C",
                "label_deg_50C",
            ]
        ],
        sub[
            ["id_seqpos"]
            + ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        ],
    )




## === cell 4
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=200, random_state=110).fit(
    preprocess_inputs(train)[:, :, 0]
)
train["cluster_id"] = kmeans_model.labels_



## === cell 5
oof_df, pred_df = train_and_predict(type=0, FOLD_N=5)

submission = sample_sub[["id_seqpos"]].merge(pred_df, on="id_seqpos", how="left")

for c in ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]:
    if c not in submission.columns:
        submission[c] = 0.0
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

submission = submission[
    ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
