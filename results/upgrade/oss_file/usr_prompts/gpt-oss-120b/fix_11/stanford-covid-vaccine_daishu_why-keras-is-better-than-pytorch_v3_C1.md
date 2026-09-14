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

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
import copy
import datetime


def preprocess_inputs(df):
    nuc_map = {"A": 0, "C": 1, "G": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "N": 1, "M": 2, "I": 3, "B": 4, "H": 5, "E": 6, "X": 7}
    seq_len = df.iloc[0]["seq_length"]
    arr = np.zeros((len(df), seq_len, 3), dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        seq = row.sequence
        struct = row.structure
        loop = row.predicted_loop_type
        seq = seq[:seq_len].ljust(seq_len, "A")
        struct = struct[:seq_len].ljust(seq_len, ".")
        loop = loop[:seq_len].ljust(seq_len, "N")
        arr[i, :, 0] = [nuc_map.get(ch, 0) for ch in seq]
        arr[i, :, 1] = [struct_map.get(ch, 2) for ch in struct]
        arr[i, :, 2] = [loop_map.get(ch, 1) for ch in loop]
    return arr


def Write_log(log_file, text):
    log_file.write(text + "\n")
    print(text)


def Get_nowtime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def mcrmse(y_true, y_pred, weight=None):
    diff = (y_true - y_pred) ** 2
    rmse_per_target = torch.sqrt(diff.mean(dim=1))  # (batch, n_targets)
    mean_rmse = rmse_per_target.mean(dim=1)  # (batch,)
    if weight is not None:
        loss = (mean_rmse * weight).mean()
    else:
        loss = mean_rmse.mean()
    return loss


def Metric(y_true, y_pred):
    diff = (y_true - y_pred) ** 2
    rmse_per_target = np.sqrt(diff.mean(axis=1))  # (samples, 3)
    return rmse_per_target.mean()


train_pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
submission_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 1
data_dir = "/kaggle/input/stanford-covid-vaccine"
if not os.path.isdir(data_dir):
    data_dir = "data/stanford-covid-vaccine"

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)

seq_features = preprocess_inputs(train)[:, :, 0]  # tokenised sequence only
kmeans_model = KMeans(n_clusters=200, random_state=110, n_init="auto").fit(
    seq_features.reshape(len(train), -1)
)
train["cluster_id"] = kmeans_model.labels_




## === cell 2
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
        x, _ = self.gru(sequence)
        x, _ = self.gru1(x)
        x = F.dropout(x, 0.5, training=self.training)
        predict = self.predict(x)
        return predict




## === cell 3
def train_and_predict(FOLD_N=5):
    SCORE_LEN = 68  # number of positions used for scoring
    gkf = GroupKFold(n_splits=FOLD_N)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log = open("./train.log", "w", 1)

    all_best_model = []
    oof_frames = []

    for fold, (train_idx, valid_idx) in enumerate(
        gkf.split(train, train["reactivity"], train["cluster_id"])
    ):
        Write_log(log, f"fold {fold} train start: {Get_nowtime()}")
        t_train = train.iloc[train_idx].reset_index(drop=True)
        t_valid = train.iloc[valid_idx].reset_index(drop=True)

        train_x = preprocess_inputs(t_train)
        valid_x = preprocess_inputs(t_valid)

        train_cate_x = torch.LongTensor(train_x[:, :, :3])
        valid_cate_x = torch.LongTensor(valid_x[:, :, :3])

        dummy_shape_train = (train_x.shape[0], train_x.shape[1], 3)
        dummy_shape_valid = (valid_x.shape[0], valid_x.shape[1], 3)
        train_cont_x = torch.zeros(dummy_shape_train, dtype=torch.float)
        valid_cont_x = torch.zeros(dummy_shape_valid, dtype=torch.float)

        train_y = torch.Tensor(
            np.array(t_train[train_pred_cols].values.tolist()).transpose((0, 2, 1))
        )
        w_train = torch.Tensor(
            np.log(t_train["signal_to_noise"].values.reshape(-1, 1) + 1.1) / 2
        )
        valid_y = torch.Tensor(
            np.array(t_valid[train_pred_cols].values.tolist()).transpose((0, 2, 1))
        )

        train_dataset = TensorDataset(train_cate_x, train_cont_x, train_y, w_train)
        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=64, num_workers=0
        )

        valid_dataset = TensorDataset(valid_cate_x, valid_cont_x, valid_y)
        valid_loader = DataLoader(
            valid_dataset, shuffle=False, batch_size=32, num_workers=0
        )

        model = Net()
        model.apply(weights_init)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_valid_metric = 1e9
        best_state = None

        for epoch in range(30):
            model.train()
            running_loss = 0.0
            for i, (cate_x, cont_x, y, weight) in enumerate(train_loader):
                cate_x, cont_x, y, weight = (
                    cate_x.to(device),
                    cont_x.to(device),
                    y.to(device),
                    weight.to(device).squeeze(),
                )
                optimizer.zero_grad()
                outputs = model(cate_x, cont_x)[:, :SCORE_LEN, :]
                loss = mcrmse(y, outputs, weight)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss /= max(1, i + 1)

            model.eval()
            valid_loss = 0.0
            all_pred = []
            with torch.no_grad():
                for cate_x, cont_x, y in valid_loader:
                    cate_x, cont_x, y = (
                        cate_x.to(device),
                        cont_x.to(device),
                        y.to(device),
                    )
                    outputs = model(cate_x, cont_x)[:, :SCORE_LEN, :]
                    all_pred.append(outputs.cpu().numpy())
                    loss = mcrmse(y, outputs)
                    valid_loss += loss.item() * cate_x.shape[0]
            valid_loss /= len(valid_x)
            all_pred = np.concatenate(all_pred, 0)

            valid_metric = Metric(
                valid_y.numpy()[:, :, [0, 1, 2]],
                all_pred[:, :, [0, 1, 2]],
            )
            Write_log(
                log,
                f"epoch {epoch:3d} | train loss:{running_loss:.6f} | valid loss:{valid_loss:.6f} | metric:{valid_metric:.6f}",
            )

            if valid_metric < best_valid_metric:
                best_valid_metric = valid_metric
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, f"./gru-cate-emb-100-fold-{fold}.cpkt")
                Write_log(log, f"[epoch {epoch}] new best model saved")

        all_best_model.append(best_state)
        model.load_state_dict(best_state)
        model.eval()

        ids = []
        for _, row in t_valid.iterrows():
            for j in range(SCORE_LEN):
                ids.append(f"{row['id']}_{j}")

        oof_pred = all_pred[:, :SCORE_LEN, :].reshape(-1, 5)
        oof_df = pd.DataFrame(oof_pred, columns=submission_cols)
        oof_df["id_seqpos"] = np.array(ids).reshape(-1)  # 1‑D array

        target_array = np.array(
            t_valid[train_pred_cols].values.tolist()
        )  # (samples,5,68)
        target_reshaped = target_array.transpose((0, 2, 1)).reshape(
            -1, 5
        )  # (samples*68,5)

        target_df = pd.DataFrame(
            target_reshaped,
            columns=[f"label_{c}" for c in train_pred_cols],
        )
        target_df["id_seqpos"] = np.array(ids).reshape(-1)

        oof_df = oof_df.merge(target_df, on="id_seqpos", how="left")
        oof_frames.append(oof_df)

    oof = pd.concat(oof_frames, ignore_index=True)

    def Predict(df):
        test_x = preprocess_inputs(df)
        test_cate_x = torch.LongTensor(test_x[:, :, :3])
        dummy_shape = (test_x.shape[0], test_x.shape[1], 3)
        test_cont_x = torch.zeros(dummy_shape, dtype=torch.float)
        test_dataset = TensorDataset(test_cate_x, test_cont_x)
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=64, num_workers=0
        )

        ids = []
        for _, row in df.iterrows():
            for j in range(row["seq_length"]):
                ids.append(f"{row['id']}_{j}")
        ids = np.array(ids).reshape(-1)  # flatten to 1‑D

        agg_pred = np.zeros((len(ids), 5), dtype=np.float32)
        with torch.no_grad():
            for fold in range(FOLD_N):
                model.load_state_dict(all_best_model[fold])
                model.eval()
                fold_preds = []
                for cate_x, cont_x in test_loader:
                    cate_x, cont_x = cate_x.to(device), cont_x.to(device)
                    out = model(cate_x, cont_x)
                    fold_preds.append(out.cpu().numpy())
                fold_preds = np.concatenate(fold_preds, 0)
                agg_pred += fold_preds.reshape(-1, 5)
        agg_pred /= FOLD_N

        sub = pd.DataFrame(agg_pred, columns=submission_cols)
        sub["id_seqpos"] = ids
        sub = sub[["id_seqpos"] + submission_cols]
        return sub

    public_sub = Predict(test)  # test contains all rows
    sub = public_sub

    log.close()
    return oof, sub




## === cell 4
oof, sub = train_and_predict()
oof.to_csv("./oof.csv", index=False)
sub.to_csv("./submission.csv", index=False)
