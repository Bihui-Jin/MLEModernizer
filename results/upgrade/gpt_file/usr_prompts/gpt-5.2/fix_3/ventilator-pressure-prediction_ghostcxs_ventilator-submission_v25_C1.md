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
import random
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input"

print("Using DATA_DIR:", DATA_DIR)
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename in ("train.csv", "test.csv", "sample_submission.csv"):
            print(os.path.join(dirname, filename))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import torch
import torch.nn as nn

print("Torch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 2
def build_features(
    df: pd.DataFrame, steps_per_breath: int = 80, is_train: bool = False
):
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    n_rows = len(df)
    if n_rows % steps_per_breath != 0:
        raise ValueError(
            f"Unexpected rows ({n_rows}) not divisible by {steps_per_breath}."
        )
    n_breaths = n_rows // steps_per_breath

    R_vals = df.groupby("breath_id")["R"].first().values
    C_vals = df.groupby("breath_id")["C"].first().values
    R_map = {5: 0, 20: 1, 50: 2}
    C_map = {10: 0, 20: 1, 50: 2}
    R_idx = np.vectorize(lambda x: R_map.get(int(x), 0))(R_vals).astype(np.int64)
    C_idx = np.vectorize(lambda x: C_map.get(int(x), 0))(C_vals).astype(np.int64)

    rc_base = np.stack(
        [
            R_idx,
            C_idx,
            R_idx * 3 + C_idx,  # combined code 0..8
            R_idx + 10,  # offset code
            C_idx + 20,  # offset code
        ],
        axis=1,
    ).astype(np.int64)

    test_x_rc = np.tile(rc_base, (1, 3)).astype(np.int64)  # (n_breaths, 15)

    u_in = df["u_in"].values.astype(np.float32).reshape(n_breaths, steps_per_breath)
    u_out = df["u_out"].values.astype(np.float32).reshape(n_breaths, steps_per_breath)
    t = df["time_step"].values.astype(np.float32).reshape(n_breaths, steps_per_breath)

    u_in_cum = np.cumsum(u_in, axis=1)
    u_in_lag1 = np.concatenate(
        [np.zeros((n_breaths, 1), np.float32), u_in[:, :-1]], axis=1
    )
    u_in_diff1 = u_in - u_in_lag1

    test_x_other = np.stack(
        [t, u_in, u_out, u_in_cum, u_in_lag1, u_in_diff1], axis=-1
    ).astype(np.float32)
    ids = df["id"].values.astype(np.int64)

    if is_train:
        y = (
            df["pressure"]
            .values.astype(np.float32)
            .reshape(n_breaths, steps_per_breath)
        )
        return test_x_rc, test_x_other, y, ids, df
    return test_x_rc, test_x_other, ids, df


test_path = os.path.join(DATA_DIR, "test.csv")
test_df = pd.read_csv(test_path)
test_x_rc, test_x_other, test_ids, test_df_sorted = build_features(
    test_df, is_train=False
)

print("test_x_rc:", test_x_rc.shape, test_x_rc.dtype)
print("test_x_other:", test_x_other.shape, test_x_other.dtype)
print("test_ids:", test_ids.shape, test_ids[:5], " ...", test_ids[-5:])




## === cell 3
class VentilatorNet(nn.Module):
    """
    Matches the core architecture semantics:
    - rc "embedding" via Dense(10, selu, no bias)
    - tile to sequence length 80 and concatenate to other inputs
    - Conv1D(256, k=5) + ReLU + BN
    - Conv1D(192, k=5) + ReLU + BN
    - Concatenate([conv_out, x_input])
    - BiLSTM 672 -> BiLSTM 512 -> BiLSTM 384 -> BiGRU 192
    - Concatenate([ox, conv_out]) then Dense 128 selu then Dense 1
    """

    def __init__(self, other_dim: int, steps: int = 80):
        super().__init__()
        self.steps = steps
        self.other_dim = other_dim

        self.rc_dense = nn.Linear(15, 10, bias=False)
        self.selu = nn.SELU()

        in_ch = other_dim + 10  # after concat

        self.conv1 = nn.Conv1d(
            in_channels=in_ch, out_channels=256, kernel_size=5, padding=2
        )
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(
            in_channels=256, out_channels=192, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(192)
        self.relu = nn.ReLU()

        self.bilstm1 = nn.LSTM(
            input_size=192 + in_ch,
            hidden_size=672,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=672 * 2, hidden_size=512, batch_first=True, bidirectional=True
        )
        self.bilstm3 = nn.LSTM(
            input_size=512 * 2, hidden_size=384, batch_first=True, bidirectional=True
        )
        self.bigru = nn.GRU(
            input_size=384 * 2, hidden_size=192, batch_first=True, bidirectional=True
        )

        self.fc1 = nn.Linear(192 * 2 + 192, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, rc_x, other_x):
        rc = rc_x.float()
        rc = self.selu(self.rc_dense(rc))  # (B, 10)
        rc = rc.unsqueeze(1).expand(-1, self.steps, -1)  # (B, 80, 10)

        x = torch.cat([other_x, rc], dim=-1)  # (B, 80, other_dim+10)

        x_c = x.transpose(1, 2)
        c1 = self.relu(self.conv1(x_c))
        c1 = self.bn1(c1)
        c2 = self.relu(self.conv2(c1))
        c2 = self.bn2(c2)  # (B, 192, T)
        conv_out = c2.transpose(1, 2)  # (B, 80, 192)

        feat = torch.cat([conv_out, x], dim=-1)  # (B, 80, 192+in_ch)

        o1, _ = self.bilstm1(feat)
        o2, _ = self.bilstm2(o1)
        o3, _ = self.bilstm3(o2)
        og, _ = self.bigru(o3)  # (B, 80, 384)

        out = torch.cat([og, conv_out], dim=-1)  # (B, 80, 384+192)
        out = self.selu(self.fc1(out))
        out = self.fc2(out)  # (B, 80, 1)
        return out




## === cell 4
train_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_path)

train_x_rc, train_x_other, train_y, train_ids, train_df_sorted = build_features(
    train_df, is_train=True
)
print("train shapes:", train_x_rc.shape, train_x_other.shape, train_y.shape)

Xrc = torch.from_numpy(train_x_rc)
Xo = torch.from_numpy(train_x_other)
Y = torch.from_numpy(train_y).unsqueeze(-1)  # (B, 80, 1)

n_breaths = Xrc.shape[0]
val_size = max(1, int(0.02 * n_breaths))
idx = np.arange(n_breaths)
np.random.shuffle(idx)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

Xrc_tr, Xo_tr, Y_tr = Xrc[tr_idx], Xo[tr_idx], Y[tr_idx]
Xrc_va, Xo_va, Y_va = Xrc[val_idx], Xo[val_idx], Y[val_idx]

model = VentilatorNet(other_dim=train_x_other.shape[-1], steps=80).to(device)

u_out_tr = torch.from_numpy(
    train_df_sorted["u_out"].values.astype(np.float32).reshape(n_breaths, 80)
)[tr_idx].to(device)
u_out_va = torch.from_numpy(
    train_df_sorted["u_out"].values.astype(np.float32).reshape(n_breaths, 80)
)[val_idx].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
mae = nn.L1Loss(reduction="none")


def run_epoch(train: bool, batch_size: int = 256):
    model.train(train)
    Xrc_ = Xrc_tr if train else Xrc_va
    Xo_ = Xo_tr if train else Xo_va
    Y_ = Y_tr if train else Y_va
    u_out_ = u_out_tr if train else u_out_va

    n = Xrc_.shape[0]
    order = torch.randperm(n) if train else torch.arange(n)
    total_loss = 0.0
    total_count = 0.0

    for i in range(0, n, batch_size):
        b = order[i : i + batch_size]
        rc_b = Xrc_[b].to(device)
        xo_b = Xo_[b].to(device)
        y_b = Y_[b].to(device)
        uo_b = u_out_[b]  # (B, 80)

        pred = model(rc_b, xo_b)  # (B,80,1)
        loss_raw = mae(pred, y_b).squeeze(-1)  # (B,80)
        mask = (uo_b < 0.5).float()  # inspiratory only
        loss = (loss_raw * mask).sum() / (mask.sum() + 1e-6)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * float(mask.sum().item())
        total_count += float(mask.sum().item())

    return total_loss / max(1.0, total_count)


epochs = 4
for ep in range(1, epochs + 1):
    tr_loss = run_epoch(train=True, batch_size=256)
    va_loss = run_epoch(train=False, batch_size=256)
    print(
        f"epoch {ep}/{epochs} - train_mae(insp): {tr_loss:.5f} - val_mae(insp): {va_loss:.5f}"
    )



## === cell 5
model.eval()
with torch.no_grad():
    Xrc_te = torch.from_numpy(test_x_rc).to(device)
    Xo_te = torch.from_numpy(test_x_other).to(device)

    preds = []
    bs = 512
    for i in range(0, Xrc_te.shape[0], bs):
        p = (
            model(Xrc_te[i : i + bs], Xo_te[i : i + bs]).squeeze(-1).cpu().numpy()
        )  # (B,80)
        preds.append(p)
    pred_breath = np.concatenate(preds, axis=0)  # (n_breaths,80)

pre_y = pred_breath.reshape(-1)

pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
p_max = 64.82099173863328

sub_medclip = np.clip(pre_y, p_min, p_max)
sub_medclip = np.round((sub_medclip - p_min) / pressure_step) * pressure_step + p_min

test_ids_sorted = test_df_sorted["id"].values.astype(np.int64)
if sub_medclip.shape[0] != test_ids_sorted.shape[0]:
    raise ValueError(
        f"Prediction length {sub_medclip.shape[0]} != number of test rows {test_ids_sorted.shape[0]}"
    )

submission = pd.DataFrame(
    {"id": test_ids_sorted, "pressure": sub_medclip.astype(np.float32)}
)
submission.to_csv("./submission.csv", index=False)
print(submission.head())
print("Wrote ./submission.csv with shape:", submission.shape, "path: ./submission.csv")
