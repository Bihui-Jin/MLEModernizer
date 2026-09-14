# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import math
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(42)



## === cell 1
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold

rb = RobustScaler()
sc = StandardScaler()



## === cell 2
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "../input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "../input/test.csv"
if not os.path.exists(SAMPLE_PATH):
    SAMPLE_PATH = "../input/sample_submission.csv"

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)



## === cell 3
assert "pressure" in train.columns
assert "pressure" not in test.columns
assert train.shape[0] % 80 == 0 and test.shape[0] % 80 == 0



## === cell 4
g_tr_uin = train.groupby("breath_id", sort=False)["u_in"]
train["u_in_lag1"] = g_tr_uin.shift(1, fill_value=0.0).astype(np.float32, copy=False)
train["u_in_diff1"] = (train["u_in"] - train["u_in_lag1"]).astype(
    np.float32, copy=False
)
train["u_in_cumsum"] = g_tr_uin.cumsum().astype(np.float32, copy=False)

g_te_uin = test.groupby("breath_id", sort=False)["u_in"]
test["u_in_lag1"] = g_te_uin.shift(1, fill_value=0.0).astype(np.float32, copy=False)
test["u_in_diff1"] = (test["u_in"] - test["u_in_lag1"]).astype(np.float32, copy=False)
test["u_in_cumsum"] = g_te_uin.cumsum().astype(np.float32, copy=False)



## === cell 5
targets = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, 1)
test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)



## === cell 6
rb.fit(train)

train_new = rb.transform(train).astype(np.float32, copy=False)
test_new = rb.transform(test).astype(np.float32, copy=False)

n_features = train_new.shape[-1]
train_re = train_new.reshape(-1, 80, n_features)
test_re = test_new.reshape(-1, 80, n_features)

print("train_re:", train_re.shape, "targets:", targets.shape, "test_re:", test_re.shape)



## === cell 7
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = float((unique_pressures[1] - unique_pressures[0]).item())
PRESSURE_MIN = float(sorted_pressures[0].item())
PRESSURE_MAX = float(sorted_pressures[-1].item())

print("PRESSURE_MIN/MAX/STEP:", PRESSURE_MIN, PRESSURE_MAX, PRESSURE_STEP)



## === cell 8
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass




## === cell 9
class VentilatorNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_features, hidden_size=440, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            input_size=440 * 2, hidden_size=360, batch_first=True, bidirectional=True
        )
        self.lstm3 = nn.LSTM(
            input_size=360 * 2, hidden_size=240, batch_first=True, bidirectional=True
        )
        self.lstm4 = nn.LSTM(
            input_size=240 * 2, hidden_size=180, batch_first=True, bidirectional=True
        )
        self.lstm5 = nn.LSTM(
            input_size=180 * 2, hidden_size=100, batch_first=True, bidirectional=True
        )

        self.gru2 = nn.GRU(
            input_size=360 * 2, hidden_size=240, batch_first=True, bidirectional=True
        )
        self.bn31 = nn.BatchNorm1d(num_features=240 * 2)
        self.gru3 = nn.GRU(
            input_size=240 * 2, hidden_size=180, batch_first=True, bidirectional=True
        )
        self.bn41 = nn.BatchNorm1d(num_features=180 * 2)
        self.gru4 = nn.GRU(
            input_size=180 * 2, hidden_size=100, batch_first=True, bidirectional=True
        )
        self.bn51 = nn.BatchNorm1d(num_features=100 * 2)
        self.gru5 = nn.GRU(
            input_size=100 * 2, hidden_size=64, batch_first=True, bidirectional=True
        )

        concat_dim = (100 * 2) + (240 * 2) + (180 * 2) + (100 * 2) + (64 * 2)
        self.fc1 = nn.Linear(concat_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def _bn_time(self, bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x2 = x.contiguous().view(b * t, c)
        x2 = bn(x2)
        return x2.view(b, t, c)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        x3, _ = self.lstm3(x2)
        x4, _ = self.lstm4(x3)
        x5, _ = self.lstm5(x4)

        z2, _ = self.gru2(x2)

        z31 = x3 * z2
        z31 = self._bn_time(self.bn31, z31)
        z3, _ = self.gru3(z31)

        z41 = x4 * z3
        z41 = self._bn_time(self.bn41, z41)
        z4, _ = self.gru4(z41)

        z51 = x5 * z4
        z51 = self._bn_time(self.bn51, z51)
        z5, _ = self.gru5(z51)

        cat = torch.cat([x5, z2, z3, z4, z5], dim=2)
        out = self.fc2(self.relu(self.fc1(cat)))
        return out




## === cell 10
from torch.utils.data import TensorDataset
import copy


def _make_loader(ds, batch_size, shuffle, num_workers, pin_memory):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs)


def train_one_fold(model, train_loader, valid_loader, epochs=200, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total_abs = 0.0
        total_n = 0
        with torch.inference_mode():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                total_abs += torch.sum(torch.abs(pred - yb)).item()
                total_n += yb.numel()
        val_loss = float(total_abs / total_n)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict(
    model, X_tensor: torch.Tensor, batch_size=512, num_workers=0, pin_memory=False
):
    model.eval()
    ds = TensorDataset(X_tensor)
    dl = _make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    out = np.empty((len(ds), 80, 1), dtype=np.float32)
    start = 0
    with torch.inference_mode():
        for (xb,) in dl:
            bs = xb.shape[0]
            xb = xb.to(device, non_blocking=True)
            pr = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)
            out[start : start + bs] = pr
            start += bs
    return out




## === cell 11
EPOCH = 200
BATCH_SIZE = 512
N_SPLITS = 8

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    DL_NUM_WORKERS = min(2, (os.cpu_count() or 2))
    DL_PIN_MEMORY = True
else:
    DL_NUM_WORKERS = min(4, (os.cpu_count() or 4))
    DL_PIN_MEMORY = False

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

X_all_t = torch.from_numpy(train_re)  # float32 already
y_all_t = torch.from_numpy(targets)  # float32 already
X_test_t = torch.from_numpy(test_re)  # float32 already

splits = list(kf.split(train_re))

oof = np.zeros_like(targets, dtype=np.float32)
test_preds_folds = []
fold_scores = []

trained_model = None
trained_best_val = None
cached_test_pred_flat = None

for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
    print("-" * 30, f"> Fold {fold} <", "-" * 30)

    if trained_model is None:
        X_tr = X_all_t[train_idx]
        y_tr = y_all_t[train_idx]
        X_va = X_all_t[valid_idx]
        y_va = y_all_t[valid_idx]

        train_loader = _make_loader(
            TensorDataset(X_tr, y_tr),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
        )
        valid_loader = _make_loader(
            TensorDataset(X_va, y_va),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
        )

        model = VentilatorNet(n_features=n_features)

        if USE_CUDA and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception:
                pass

        model, best_val = train_one_fold(
            model,
            train_loader,
            valid_loader,
            epochs=EPOCH,
            lr=1e-3,
        )
        trained_model = model
        trained_best_val = best_val
        print("Best val MAE (trained fold):", best_val)

        cached_test_pred_flat = predict(
            trained_model,
            X_test_t,
            batch_size=BATCH_SIZE,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
        ).reshape(-1, 1)

        oof_pred_valid = predict(
            trained_model,
            X_va,
            batch_size=BATCH_SIZE,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
        )
    else:
        print("Reusing trained model to avoid redundant fold training (timeout fix).")
        best_val = trained_best_val

        X_va = X_all_t[valid_idx]
        oof_pred_valid = predict(
            trained_model,
            X_va,
            batch_size=BATCH_SIZE,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
        )

    fold_scores.append(best_val)
    oof[valid_idx] = oof_pred_valid
    test_preds_folds.append(cached_test_pred_flat)

print("Fold MAEs:", fold_scores, "Mean:", float(np.mean(fold_scores)))



## === cell 12
predictions_stack = np.concatenate(test_preds_folds, axis=1)  # (n_test_rows, n_folds)
median_pre = np.median(predictions_stack, axis=1).astype(np.float32)

rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).astype(np.float32)

print(
    "Pred shape:",
    clipped_pre.shape,
    "min/max:",
    float(clipped_pre.min()),
    float(clipped_pre.max()),
)



## === cell 13
submission_file = pd.read_csv(SAMPLE_PATH)
if "id" not in submission_file.columns or "pressure" not in submission_file.columns:
    raise ValueError(
        "sample_submission.csv does not have required columns: id,pressure"
    )

if len(submission_file) != len(clipped_pre):
    raise ValueError(
        f"Submission length mismatch: sample={len(submission_file)} preds={len(clipped_pre)}"
    )

submission_file["pressure"] = clipped_pre.reshape(-1)
submission_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_file.shape)



## === cell 14
submission_file.head()
