# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

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
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.6766

# 6. Current score

23.63444

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 23.63444) has done: 'I fix the preprocessing so train/test have identical feature columns (removing the accidental `pressure` leakage into test) and ensure both are converted to NumPy arrays before scaling/reshaping, which resolves the `RobustScaler` feature-name error and the `DataFrame.reshape`/`torch.from_numpy` errors. I also fix the fastai training call: you accidentally invoked `fit_one_cycle` on the model due to attribute forwarding; explicitly calling `learn.fit_one_cycle(...)` after creating the `Learner` fixes this runtime error. Finally, I restore the test DataLoader creation after preprocessing so inference runs and writes a valid `submission.csv` with `id,pressure` and correct row alignment.'
- What this solution (achieved 23.63444) has done: 'I fix the fastai training call that’s currently being attribute-forwarded to the model (causing `VentilatorModel` to look for `fit_one_cycle`) by ensuring we invoke `fit_one_cycle` on the actual `Learner` instance and not via a name collision. I also remove the incompatible `ReduceLROnPlateau` callback usage (fastai expects its own `ReduceLROnPlateau`/`TrackerCallback` semantics; the current one triggers the forwarding path) and keep training otherwise identical. Finally, I keep inference/submission writing unchanged but add a small safety check to guarantee the submission length matches `sample_submission.csv` and is written with the required columns.'
- What this solution (achieved 23.63444) has done: 'The runtime error is caused by fastai’s attribute-forwarding: your `Learner` instance is being treated like its default component (the model), so `fit_one_cycle` gets looked up on `VentilatorModel`. I fix this by explicitly importing and using fastai’s training extension (`from fastai.callback.schedule import fit_one_cycle`) and calling it as a function on the `Learner`, which bypasses the forwarding issue while keeping the same training loop semantics. I also make inference use `learn.model` (the trained model) rather than the original `model` variable to avoid any possibility of using an untrained reference. These changes are minimal, unblock end-to-end execution, and should substantially reduce the MAE versus the current broken/ineffective training behavior, moving the score toward the target.'
- What this solution (achieved 23.63444) has done: 'I fix the `fit_one_cycle` crash by removing the incorrect import/usage that results in a `NoneType` callable and instead call the bound method `learn.fit_one_cycle(...)`, which is the stable fastai API in your installed version. I also replace the classification `accuracy` metric (which is not aligned with the MAE evaluation and can confuse training feedback) with only the discrete MAE metric while keeping the same model, loss, preprocessing, and training loop semantics. Finally, I keep inference and submission writing unchanged but add a small safety check for `n_classes`/target mapping consistency to ensure end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved 23.63444) has done: 'I fix the fastai `Learner` attribute-forwarding issue that causes `fit_one_cycle` to be looked up on the model by explicitly importing and calling fastai’s training function `fit_one_cycle(learn, ...)` (this preserves the same training loop semantics). I also ensure the learner has `model` as its default component to avoid any accidental forwarding issues in this environment/version combination. These changes are score-improving (your current score indicates training effectively didn’t run) but keep the same model, loss, data, and preprocessing. Everything else (feature engineering, scaling, reshaping, inference, and submission writing) remains unchanged so the pipeline runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 23.63444) has done: 'I fix the training crash by removing the incorrect `fit_one_cycle` function call (it’s being imported as `None` in this environment/version combination) and instead use the stable fastai API `learn.fit_one_cycle(...)`, which preserves the same training loop semantics. I also ensure the submission `id` column is taken from the test file and predictions are written in the same order, so row alignment is guaranteed. These are minimal runtime/stability fixes and should also improve score substantially because the model actually train before inference. Everything else (features, scaling, model, loss, discrete pressure mapping, and inference argmax) is kept unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler

from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Listing /kaggle/input:")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith((".csv", ".md")):
            print(os.path.join(dirname, filename))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
df = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")
df_test = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")

print(df.shape, df_test.shape)
print(df.columns)



## === cell 2
target_dic = {v: i for i, v in enumerate(sorted(df["pressure"].unique().tolist()))}
target_dic_inv = {v: k for k, v in target_dic.items()}

n_classes = len(target_dic)
print(
    "n_classes:",
    n_classes,
    "min/max pressure:",
    min(target_dic.keys()),
    max(target_dic.keys()),
)

assert set(target_dic.values()) == set(range(n_classes))



## === cell 3
pass



## === cell 4
pass




## === cell 5
def add_features(df_):
    df = df_.copy()

    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )

    df["u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["u_in_lag"] = df["u_in_lag"] * df["breath_id_lagsame"]
    df["u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["u_in_lag2"] = df["u_in_lag2"] * df["breath_id_lag2same"]

    df["u_out_lag2"] = df["u_out"].shift(2).fillna(0)
    df["u_out_lag2"] = df["u_out_lag2"] * df["breath_id_lag2same"]

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["RC"] = df["R"] + df["C"]

    df = pd.get_dummies(df)
    return df


train = add_features(df)
test = add_features(df_test)
del df
gc.collect()

train_features = train.drop(columns=["pressure"])
train_target_col = train["pressure"].copy()

train_features, test = train_features.align(test, join="left", axis=1, fill_value=0)
train = train_features.copy()
train["pressure"] = train_target_col.values

print("Aligned shapes:", train.shape, test.shape)



## === cell 6
train["pressure"] = train["pressure"].map(target_dic)



## === cell 7
train["pressure"].head()



## === cell 8
train.shape



## === cell 9
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
    "u_out_lag2",
]
train.drop(drop_cols, axis=1, inplace=True)

test = test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
        "u_out_lag2",
    ],
    axis=1,
)

print(train.shape, test.shape, targets.shape)



## === cell 10
train.head()



## === cell 11
RS = RobustScaler()
train_np = RS.fit_transform(train.to_numpy(dtype=np.float32))
test_np = RS.transform(test.to_numpy(dtype=np.float32))



## === cell 12
train = train_np.reshape(-1, 80, train_np.shape[-1])
test = test_np.reshape(-1, 80, train_np.shape[-1])

print("Reshaped:", train.shape, test.shape)



## === cell 13
idx = list(range(len(train)))
len(idx)



## === cell 14
pass



## === cell 15
train.shape[-2:]




## === cell 16
class VentilatorDataset(Dataset):
    def __init__(self, data, target, label_dic=None):
        self.data = torch.from_numpy(data).float()
        self.label_dic = label_dic
        if target is not None:
            self.targets = torch.from_numpy(target).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, "targets"):
            x = self.data[idx]
            y = self.targets[idx]
            return x, y
        else:
            return self.data[idx]




## === cell 17
class config:
    EXP_NAME = "exp080_conti_rc"

    INPUT = "/kaggle/input/ventilator-pressure-prediction"
    OUTPUT = "/kaggle/working"
    N_FOLD = 5
    SEED = 0

    LR = 5e-3
    N_EPOCHS = 50
    EMBED_SIZE = 64
    HIDDEN_SIZE = 256
    BS = 512
    WEIGHT_DECAY = 1e-3

    USE_LAG = 4
    CONT_FEATURES = (
        ["u_in", "u_out", "time_step"]
        + ["u_in_cumsum", "u_in_cummean", "area", "cross", "cross2"]
        + ["R_cate", "C_cate"]
    )
    LAG_FEATURES = ["breath_time"]
    LAG_FEATURES += [f"u_in_lag_{i}" for i in range(1, USE_LAG + 1)]
    LAG_FEATURES += [f"u_in_time{i}" for i in range(1, USE_LAG + 1)]
    LAG_FEATURES += [f"u_out_lag_{i}" for i in range(1, USE_LAG + 1)]
    ALL_FEATURES = CONT_FEATURES + LAG_FEATURES

    NOT_WATCH_PARAM = ["INPUT"]




## === cell 18
class VentilatorModel(nn.Module):
    def __init__(self, input_size=25):
        hidden = [400, 300, 200, 100]
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size, hidden[0], batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            2 * hidden[0], hidden[1], batch_first=True, bidirectional=True
        )
        self.lstm3 = nn.LSTM(
            2 * hidden[1], hidden[2], batch_first=True, bidirectional=True
        )
        self.lstm4 = nn.LSTM(
            2 * hidden[2], hidden[3], batch_first=True, bidirectional=True
        )
        self.fc1 = nn.Linear(2 * hidden[3], 4 * hidden[3])
        self.selu = nn.SELU()
        self.fc2 = nn.Linear(4 * hidden[3], n_classes)
        self._reinitialize()

    def _reinitialize(self):
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)
            elif "fc" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "bias" in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)
        return x




## === cell 19
pass



## === cell 20
batch_size = 512

submission = pd.read_csv(
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
)
test_ids = df_test["id"].to_numpy(copy=True)

test_dataset = VentilatorDataset(test, None)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)



## === cell 21
pass



## === cell 22
train_index = list(range(int(0.95 * len(train))))
valid_index = list(range(int(0.95 * len(train)), len(train)))



## === cell 23
train_input, valid_input = train[train_index], train[valid_index]
train_targets, valid_targets = targets[train_index], targets[valid_index]

train_dataset = VentilatorDataset(train_input, train_targets)
valid_dataset = VentilatorDataset(valid_input, valid_targets)



## === cell 24
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
dls = DataLoaders(train_loader, valid_loader)



## === cell 25
model = VentilatorModel(input_size=train.shape[-1]).to(device)




## === cell 26
def mae_loss_discrete(y_pred, y):
    y_clone = y.clone().detach().to(dtype=torch.double).cpu()
    y_pred_clone = y_pred.clone().detach().to(dtype=torch.double).cpu()

    y_clone.apply_(lambda element: float(target_dic_inv[int(element)]))
    y_pred_idx = torch.max(y_pred_clone, axis=2)[1].to(dtype=torch.long)
    y_pred_idx = y_pred_idx.to(dtype=torch.double)
    y_pred_idx.apply_(lambda element: float(target_dic_inv[int(element)]))

    return nn.L1Loss()(y_pred_idx, y_clone)




## === cell 27
learn = Learner(
    dls, model, loss_func=CrossEntropyLossFlat(), metrics=[mae_loss_discrete]
)
learn._default = "model"



## === cell 28
try:
    _ = learn.lr_find(show_plot=False)
except Exception as e:
    print("lr_find skipped due to:", repr(e))



## === cell 29
learn.fit_one_cycle(10, lr_max=1e-3)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1417165151.py in <cell line: 0>()
      1 # Bugfix: fastai's `fit_one_cycle` import can resolve to None in some setups; use the bound method.
      2 # This preserves the same training loop semantics and ensures training actually runs.
----> 3 learn.fit_one_cycle(10, lr_max=1e-3)
      4 

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'VentilatorModel' object has no attribute 'fit_one_cycle'

## === cell 30
del (
    train_dataset,
    valid_dataset,
    train_loader,
    valid_loader,
    train_index,
    valid_index,
    train_input,
    valid_input,
    train_targets,
    valid_targets,
    dls,
    train,
)
gc.collect()



## === cell 31
gc.collect()




## === cell 32
def get_pred(batch):
    m = learn.model
    m.eval()
    with torch.no_grad():
        pred = m(batch.to(device))
        pred_idx = torch.max(pred, axis=2)[1].to(dtype=torch.float).cpu()
        pred_idx.apply_(lambda element: float(target_dic_inv[int(element)]))
        pred_flat = pred_idx.squeeze(-1).flatten().numpy()
    return pred_flat




## === cell 33
with torch.no_grad():
    preds = [get_pred(batch) for batch in test_loader]
preds = np.concatenate(preds)
print("preds shape:", preds.shape)



## === cell 34
del test_loader
gc.collect()



## === cell 35
if len(preds) != len(test_ids):
    raise ValueError(f"Prediction length {len(preds)} != test length {len(test_ids)}")

submission_out = pd.DataFrame({"id": test_ids, "pressure": preds.astype(np.float32)})

if len(submission_out) != len(submission):
    raise ValueError(
        f"Output length {len(submission_out)} != sample_submission length {len(submission)}"
    )

submission_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())
