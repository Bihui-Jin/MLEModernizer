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
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,  GroupKFold
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

import warnings

warnings.simplefilter('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## === cell 1
def make_X(dt, dense_cols, cat_feats):
    X = {"dense": dt[dense_cols].to_numpy()}
    for i, v in enumerate(cat_feats):
        X[v] = dt[[v]].to_numpy()
    return X

class Loader:

    def __init__(self, X, y, shuffle=True, batch_size=64, cat_cols=[]):

        self.X_cont = X["dense"]
        self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)
        self.y = y

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_conts = self.X_cont.shape[1]
        self.len = self.X_cont.shape[0]
        n_batches, remainder = divmod(self.len, self.batch_size)

        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.remainder = remainder  # for debugging

        self.idxes = np.array([i for i in range(self.len)])

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes
            np.random.shuffle(ridxes)
            self.X_cat = self.X_cat[[ridxes]]
            self.X_cont = self.X_cont[[ridxes]]
            if self.y is not None:
                self.y = self.y[[ridxes]]

        return self

    def __next__(self):
        if self.i >= self.len:
            raise StopIteration

        if self.y is not None:
            y = torch.FloatTensor(self.y[self.i:self.i + self.batch_size].astype(np.float32))

        else:
            y = None

        xcont = torch.FloatTensor(self.X_cont[self.i:self.i + self.batch_size])
        xcat = torch.LongTensor(self.X_cat[self.i:self.i + self.batch_size])

        batch = (xcont, xcat, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


## === cell 2

class model_nn(nn.Module):

    def __init__(self, hidden_dim, output_dim, emb_dims, n_cont):
        super().__init__()

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])

        self.n_embs = n_embs  # + t_embs
        self.n_cont = n_cont

        inp_dim = n_embs + n_cont
        self.inp_dim = inp_dim

        self.fc0 = nn.Linear(inp_dim, hidden_dim)
        self.relu0 = nn.ReLU(True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU(True)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def encode_and_combine_data(self, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        return xcat

    def forward(self, cont_data, cat_data):
        cont_data = cont_data.to(device)
        cat_data = cat_data.to(device)

        cat_data = self.encode_and_combine_data(cat_data)

        x = torch.cat([cont_data, cat_data], dim=1)

        hz = self.fc0(x)
        hz = self.relu0(hz)
        hz = self.fc1(hz)
        hz = self.relu1(hz)

        out = self.fc2(hz)
        return out


## === cell 3
def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

def metric(outputs, target):

    confidence = np.abs(outputs[:, 2] - outputs[:, 0])
    clip = np.where(confidence > 70, confidence, 70)
    delta = np.abs(outputs[:, 1] - target)
    delta = np.where(delta > 1000, 1000, delta)

    metrics = (delta*np.sqrt(2)/clip) + np.log(clip*np.sqrt(2))

    return np.mean(metrics)


## === cell 4

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def model_training(model, train_loader, val_loader, epochs,
                   batch_size=64, lr=0.001, patience=10,
                   model_path='model.pth'):
    if os.path.isfile(model_path):

        model = torch.load(model_path)

        return model

    else:

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2,
                                      factor=0.4, verbose=True)

        train_losses = []
        val_losses = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for epoch in tqdm(range(epochs)):
            train_loss, val_loss = 0, 0
            model.train()
            bar = tqdm(train_loader)

            for i, (X_cont, X_cat, y) in enumerate(bar):
                preds = model(X_cont, X_cat)
                loss = quantile_loss(preds, y.to(device), [0.2, 0.5, 0.8])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss += loss.item() / len(train_loader)

            val_preds = []
            true_y = []
            model.eval()
            with torch.no_grad():
                for phase in ["valid"]:
                    if phase == "train":
                        loader = train_loader
                    else:
                        loader = val_loader

                    for i, (X_cont, X_cat, y) in enumerate(loader):
                        preds = model(X_cont, X_cat)

                        val_preds.append(preds)
                        true_y.append(y)

                        loss = quantile_loss(preds, y.to(device), [0.2, 0.5, 0.8])
                        val_loss += loss.item() / len(loader)

                val_preds = torch.cat(val_preds, dim=0).detach().cpu().numpy()
                true_y = torch.cat(true_y, dim=0).detach().cpu().numpy()
                score = metric(val_preds, true_y)

            print(f"[{phase}] Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val score: {score:.4f}")

            early_stopping(score, model, path=model_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

        model = torch.load(model_path)

        return model


## === cell 5
ROOT = "../input/osic-pulmonary-fibrosis-progression"
Model_Root = 'models'  #this root for the prediction phase 


## === cell 6

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")

tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([tr, chunk, sub], axis=0, ignore_index=True)

data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC", "Percent"]].copy()
base.columns = ["Patient", "min_FVC", "min_percent"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)

data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base

COLS = ["Sex", "SmokingStatus"]  # ,'Age'
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["min_percent_norm"] = (data["min_percent"] - data["min_percent"].min()) / (
    data["min_percent"].max() - data["min_percent"].min()
)

cat_feat = ["Sex", "SmokingStatus"]

uniques = []
for i, v in enumerate(cat_feat):
    data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]])
    uniques.append(len(data[v].unique()))

tr = data.loc[data.WHERE == "train"]
chunk = data.loc[data.WHERE == "val"]
sub = data.loc[data.WHERE == "test"]

FE += ["age", "week", "BASE"]

print(list(tr))
print(list(sub))
print(FE)
print(cat_feat)
print(uniques)

dims = [1, 2]
emb_dims = [(x, y) for x, y in zip(uniques, dims)]
n_cont = len(FE)


## === cell 7

hidden_dim = 128
out_dim = 3

kfold = 5
skf = KFold(n_splits=kfold, shuffle=True, random_state=42)

quantiles = [0.2, 0.5, 0.8]
avg_preds = np.zeros((len(sub), len(quantiles)))


def _fix_cat_shape(loader):
    for X_cont, X_cat, y in loader:
        if X_cat is not None:
            if X_cat.dim() > 2:
                X_cat = X_cat.view(X_cat.size(0), -1)
            elif X_cat.dim() == 1:
                X_cat = X_cat.view(-1, 1)
        yield X_cont, X_cat, y


for i, (train_index, test_index) in enumerate(skf.split(tr, tr["FVC"].values)):
    print("[Fold %d/%d]" % (i + 1, kfold))

    model_path = f"{Model_Root}/nn_model_%s.pth" % i

    if os.path.isfile(model_path):

        X_test = make_X(sub, FE, cat_feat)
        test_loader = Loader(
            X_test, None, cat_cols=cat_feat, batch_size=256, shuffle=False
        )

        preds = []
        model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)
        model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            model.eval()
            for i, (X_cont, X_cat, y) in enumerate(tqdm(_fix_cat_shape(test_loader))):
                out = model(X_cont, X_cat)
                preds.append(out)

            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
            avg_preds += preds

    else:

        X_train, X_valid = tr.iloc[train_index], tr.iloc[test_index]
        y_train, y_valid = (
            tr.iloc[train_index]["FVC"].values,
            tr.iloc[test_index]["FVC"].values,
        )

        X_train = make_X(X_train.reset_index(), FE, cat_feat)
        X_valid = make_X(X_valid.reset_index(), FE, cat_feat)

        train_loader = Loader(
            X_train, y_train, cat_cols=cat_feat, batch_size=16, shuffle=True
        )
        val_loader = Loader(
            X_valid, y_valid, cat_cols=cat_feat, batch_size=64, shuffle=True
        )

        model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)

        final_model = model_training(
            model,
            _fix_cat_shape(train_loader),
            _fix_cat_shape(val_loader),
            epochs=1000,
            batch_size=64,
            lr=0.01,
            patience=20,
            model_path="nn_model_%s.pth" % i,
        )

        X_test = make_X(sub, FE, cat_feat)
        test_loader = Loader(
            X_test, None, cat_cols=cat_feat, batch_size=256, shuffle=False
        )

        preds = []
        model = model_nn(hidden_dim, out_dim, emb_dims, n_cont).to(device)
        model.load_state_dict(torch.load("nn_model_%s.pth" % i))
        with torch.no_grad():
            model.eval()
            for i, (X_cont, X_cat, y) in enumerate(tqdm(_fix_cat_shape(test_loader))):
                out = model(X_cont, X_cat)
                preds.append(out)

            preds = torch.cat(preds, dim=0).detach().cpu().numpy()
            avg_preds += preds


avg_preds = avg_preds / kfold
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["FVC"] = avg_preds[:, 1]
sub["Confidence"] = np.abs(avg_preds[:, 2] - avg_preds[:, 0])

sub.to_csv("submission.csv", index=False)


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4093334505.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     66[0m         [0mmodel[0m [0;34m=[0m [0mmodel_nn[0m[0;34m([0m[0mhidden_dim[0m[0;34m,[0m [0mout_dim[0m[0;34m,[0m [0memb_dims[0m[0;34m,[0m [0mn_cont[0m[0;34m)[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     67[0m [0;34m[0m[0m
[0;32m---> 68[0;31m         final_model = model_training(
[0m[1;32m     69[0m             [0mmodel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     70[0m             [0m_fix_cat_shape[0m[0;34m([0m[0mtrain_loader[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3455412088.py[0m in [0;36mmodel_training[0;34m(model, train_loader, val_loader, epochs, batch_size, lr, patience, model_path)[0m
[1;32m     65[0m [0;34m[0m[0m
[1;32m     66[0m             [0;32mfor[0m [0mi[0m[0;34m,[0m [0;34m([0m[0mX_cont[0m[0;34m,[0m [0mX_cat[0m[0;34m,[0m [0my[0m[0;34m)[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mbar[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 67[0;31m                 [0mpreds[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX_cont[0m[0;34m,[0m [0mX_cat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     68[0m                 [0mloss[0m [0;34m=[0m [0mquantile_loss[0m[0;34m([0m[0mpreds[0m[0;34m,[0m [0my[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m,[0m [0;34m[[0m[0;36m0.2[0m[0;34m,[0m [0;36m0.5[0m[0;34m,[0m [0;36m0.8[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     69[0m                 [0moptimizer[0m[0;34m.[0m[0mzero_grad[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/347977094.py[0m in [0;36mforward[0;34m(self, cont_data, cat_data)[0m
[1;32m     33[0m         [0mcat_data[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mencode_and_combine_data[0m[0;34m([0m[0mcat_data[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m [0;34m[0m[0m
[0;32m---> 35[0;31m         [0mx[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mcat[0m[0;34m([0m[0;34m[[0m[0mcont_data[0m[0;34m,[0m [0mcat_data[0m[0;34m][0m[0;34m,[0m [0mdim[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     36[0m [0;34m[0m[0m
[1;32m     37[0m         [0mhz[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfc0[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: Tensors must have same number of dimensions: got 3 and 2
