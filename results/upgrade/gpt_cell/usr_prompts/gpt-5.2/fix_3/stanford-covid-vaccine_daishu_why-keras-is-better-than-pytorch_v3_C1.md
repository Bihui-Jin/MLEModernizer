# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

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
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler
import time,datetime


## === cell 1
def Get_nowtime(fmat='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.strftime(datetime.datetime.now(),fmat)

def Metric(target,pred):
    metric = 0
    for i in range(target.shape[-1]):
        metric += (np.sqrt(np.mean((target[:,:,i]-pred[:,:,i])**2))/target.shape[-1])
    return metric

def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None

def Seed_everything(seed=1017):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything()

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        print(classname)
        try:
            nn.init.xavier_uniform_(m.weight)
        except:
            pass
        try:
            nn.init.zeros_(m.bias)
        except:
            pass
    if classname.find('GRU') != -1 or classname.find('LSTM') != -1:
        print(classname)
        for name, param in m.named_parameters():
            print(name)
            if 'bias_ih' in name:
                 torch.nn.init.zeros_(param)
            elif 'bias_hh' in name:
                torch.nn.init.zeros_(param)
            elif 'weight_ih' in name:
                 nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)


## === cell 2
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

def mcrmse(y_actual, y_pred, weight=None, num_scored=5):
    score = 0
    for i in range(5):
        if weight is not None:
            score += torch.sqrt(torch.mean((y_actual[:,:,i]-y_pred[:,:,i])**2*weight)) / num_scored
        else:
            score += torch.sqrt(torch.mean((y_actual[:,:,i]-y_pred[:,:,i])**2)) / num_scored
    return score

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea], 2)


## === cell 3
def _resolve_sc_dataset_root():
    candidates = [
        "../input/stanford-covid-vaccine",
        "/kaggle/input/stanford-covid-vaccine",
        "/kaggle/data/stanford-covid-vaccine",
        "/kaggle/working/stanford-covid-vaccine",
        "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine",
        "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine",
        "/kaggle/working/stanford-covid-vaccine/stanford-covid-vaccine",
    ]

    def has_required_files(root):
        return (
            os.path.exists(os.path.join(root, "train.json"))
            and os.path.exists(os.path.join(root, "test.json"))
            and os.path.isdir(os.path.join(root, "bpps"))
        )

    for p in candidates:
        if has_required_files(p):
            return p
        nested = os.path.join(p, "stanford-covid-vaccine")
        if has_required_files(nested):
            return nested

    raise FileNotFoundError(
        "Could not locate stanford-covid-vaccine dataset root with train.json/test.json and bpps/. "
        "Tried: " + ", ".join(candidates)
    )


SC_DATA_ROOT = _resolve_sc_dataset_root()

train = pd.read_json(os.path.join(SC_DATA_ROOT, "train.json"), lines=True)
test = pd.read_json(os.path.join(SC_DATA_ROOT, "test.json"), lines=True)


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(
            np.load(os.path.join(SC_DATA_ROOT, "bpps", f"{mol_id}.npy")).max(axis=1)
        )
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(
            np.load(os.path.join(SC_DATA_ROOT, "bpps", f"{mol_id}.npy")).sum(axis=1)
        )
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914  # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(os.path.join(SC_DATA_ROOT, "bpps", f"{mol_id}.npy"))
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
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

        valid_y = valid_y.numpy()

        model = Net()
        model.apply(weights_init)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        all_valid_metric = []
        all_epoch_valid_metric = []
        not_improve_epochs = 0
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
            running_loss = running_loss / n

            valid_loss = 0.0
            all_pred = []
            model.eval()
            for data in valid_data_loader:
                cate_x, cont_x, y = [x.to(device) for x in data]
                outputs = model(cate_x, cont_x)
                all_pred.append(outputs.detach().cpu().numpy())
                loss = mcrmse(y, outputs[:, :68, :])
                valid_loss += loss.item() * cate_x.shape[0]
            valid_loss = valid_loss / valid_count
            all_pred = np.concatenate(all_pred, 0)
            valid_metric = Metric(valid_y[:, :, [0, 1, 3]], all_pred[:, :68, [0, 1, 3]])
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
                not_improve_epochs = 0
            else:
                not_improve_epochs += 1
                Write_log(log, "Not improve epoch +1 ---> %s" % not_improve_epochs)

        all_best_model.append(best_model)
        model.load_state_dict(best_model)
        model.eval()
        all_id = []
        all_y_id = []
        for i, row in t_valid.iterrows():
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
        t_valid_metric = Metric(valid_y[:, :, [0, 1, 3]], all_pred[:, :68, [0, 1, 3]])
        t_oof = pd.DataFrame(
            all_pred.reshape(-1, 5),
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        t_oof["id_seqpos"] = all_id
        t_target_df = pd.DataFrame(
            valid_y.reshape(-1, 5),
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
        all_valid_metric.append(t_valid_metric)
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
        "gru-cate-emb-100-0.6%f-%.6f.log" % (np.mean(all_valid_metric), oof_metirc),
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
        for i, row in df.iterrows():
            for j in range(row["seq_length"]):
                all_id.append(row["id"] + "_%s" % j)

        all_id = np.array(all_id).reshape(-1, 1)
        all_pred = np.zeros(len(all_id) * 5).reshape(len(all_id), 5)
        for fold in range(FOLD_N):
            model.load_state_dict(all_best_model[fold])
            model.eval()
            t_all_pred = []
            for data in test_data_loader:
                cate_x, cont_x = [x.to(device) for x in data]
                outputs = model(cate_x, cont_x)
                t_all_pred.append(outputs.detach().cpu().numpy())
            t_all_pred = np.concatenate(t_all_pred, 0)
            all_pred += t_all_pred.reshape(-1, 5)
        all_pred /= FOLD_N
        sub = pd.DataFrame(
            all_pred,
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        sub["id_seqpos"] = all_id
        return sub

    public_sub = Pred(test.loc[test["seq_length"] == 107])
    private_sub = Pred(test.loc[test["seq_length"] == 130])
    sub = pd.concat([public_sub, private_sub]).reset_index(drop=True)
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


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/547237554.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     33[0m [0;34m[0m[0m
[1;32m     34[0m [0;34m[0m[0m
[0;32m---> 35[0;31m [0mSC_DATA_ROOT[0m [0;34m=[0m [0m_resolve_sc_dataset_root[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     36[0m [0;34m[0m[0m
[1;32m     37[0m [0mtrain[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_json[0m[0;34m([0m[0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mSC_DATA_ROOT[0m[0;34m,[0m [0;34m"train.json"[0m[0;34m)[0m[0;34m,[0m [0mlines[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/547237554.py[0m in [0;36m_resolve_sc_dataset_root[0;34m()[0m
[1;32m     27[0m             [0;32mreturn[0m [0mnested[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m [0;34m[0m[0m
[0;32m---> 29[0;31m     raise FileNotFoundError(
[0m[1;32m     30[0m         [0;34m"Could not locate stanford-covid-vaccine dataset root with train.json/test.json and bpps/. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m         [0;34m"Tried: "[0m [0;34m+[0m [0;34m", "[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mcandidates[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: Could not locate stanford-covid-vaccine dataset root with train.json/test.json and bpps/. Tried: ../input/stanford-covid-vaccine, /kaggle/input/stanford-covid-vaccine, /kaggle/data/stanford-covid-vaccine, /kaggle/working/stanford-covid-vaccine, /kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine, /kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine, /kaggle/working/stanford-covid-vaccine/stanford-covid-vaccine

## === cell 4
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train)[:,:,0])
train['cluster_id'] = kmeans_model.labels_
