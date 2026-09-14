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

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)



## === cell 3
assert "pressure" in train.columns
assert "pressure" not in test.columns
assert train.shape[0] % 80 == 0 and test.shape[0] % 80 == 0



## === cell 4
train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0)
train["u_in_diff1"] = train["u_in"] - train["u_in_lag1"]
train["u_in_cumsum"] = train["u_in"].groupby(train["breath_id"]).cumsum()

test["u_in_lag1"] = test.groupby("breath_id")["u_in"].shift(1).fillna(0)
test["u_in_diff1"] = test["u_in"] - test["u_in_lag1"]
test["u_in_cumsum"] = test["u_in"].groupby(test["breath_id"]).cumsum()



## === cell 5
targets = train["pressure"].to_numpy().reshape(-1, 80, 1).astype(np.float32)
test_id = test["id"].copy()

train.drop(columns=["id", "breath_id", "pressure", "time_step"], inplace=True)
test.drop(columns=["id", "breath_id", "time_step"], inplace=True)



## === cell 6
rb.fit(train)

train_new = rb.transform(train).astype(np.float32)
test_new = rb.transform(test).astype(np.float32)

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
class VentilatorDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


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
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict(model, X, batch_size=512):
    model.eval()
    ds = VentilatorDataset(X, None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            pr = model(xb).detach().cpu().numpy()
            preds.append(pr)
    return np.concatenate(preds, axis=0)




## === cell 11
EPOCH = 200
BATCH_SIZE = 512
N_SPLITS = 8

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof = np.zeros_like(targets, dtype=np.float32)
test_preds_folds = []

fold_scores = []
for fold, (train_idx, valid_idx) in enumerate(kf.split(train_re, targets), start=1):
    print("-" * 30, f"> Fold {fold} <", "-" * 30)
    X_train, y_train = train_re[train_idx], targets[train_idx]
    X_valid, y_valid = train_re[valid_idx], targets[valid_idx]

    train_loader = DataLoader(
        VentilatorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        VentilatorDataset(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = VentilatorNet(n_features=n_features)
    model, best_val = train_one_fold(
        model, train_loader, valid_loader, epochs=EPOCH, lr=1e-3
    )
    fold_scores.append(best_val)
    print("Best val MAE:", best_val)

    oof_pred = predict(model, X_valid, batch_size=BATCH_SIZE)
    oof[valid_idx] = oof_pred

    test_pred = predict(model, test_re, batch_size=BATCH_SIZE).reshape(-1, 1)
    test_preds_folds.append(test_pred)

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
