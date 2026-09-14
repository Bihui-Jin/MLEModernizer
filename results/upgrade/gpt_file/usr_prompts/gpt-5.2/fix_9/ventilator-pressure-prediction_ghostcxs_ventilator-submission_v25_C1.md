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
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
import torch
import torch.nn as nn

print("Torch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

cpu_cnt = os.cpu_count() or 2
torch.set_num_threads(min(8, max(1, cpu_cnt // 2)))
torch.set_num_interop_threads(1)




## === cell 2
def build_features(
    df: pd.DataFrame, steps_per_breath: int = 80, is_train: bool = False
):
    n_rows = len(df)
    if n_rows % steps_per_breath != 0:
        raise ValueError(
            f"Unexpected rows ({n_rows}) not divisible by {steps_per_breath}."
        )
    n_breaths = n_rows // steps_per_breath

    breath_ids = df["breath_id"].to_numpy(dtype=np.int32, copy=False)
    bid0 = breath_ids[::steps_per_breath]
    ok = True
    if n_breaths > 1:
        ok = np.all(breath_ids[steps_per_breath:] != breath_ids[:-steps_per_breath])
        ok = ok and (bid0.shape[0] == n_breaths)
    if not ok:
        df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
            drop=True
        )
        breath_ids = df["breath_id"].to_numpy(dtype=np.int32, copy=False)
        bid0 = breath_ids[::steps_per_breath]
        if n_breaths > 1:
            ok = np.all(breath_ids[steps_per_breath:] != breath_ids[:-steps_per_breath])
        if not ok:
            raise ValueError(
                "Input is not organized in contiguous 80-row breath blocks; cannot safely reshape."
            )

    R_vals = (
        df["R"]
        .to_numpy(dtype=np.int16, copy=False)[::steps_per_breath]
        .astype(np.int64, copy=False)
    )
    C_vals = (
        df["C"]
        .to_numpy(dtype=np.int16, copy=False)[::steps_per_breath]
        .astype(np.int64, copy=False)
    )

    R_idx = (R_vals == 20).astype(np.int64) + (R_vals == 50).astype(np.int64) * 2
    C_idx = (C_vals == 20).astype(np.int64) + (C_vals == 50).astype(np.int64) * 2

    rc_base = np.stack(
        [
            R_idx,
            C_idx,
            R_idx * 3 + C_idx,  # combined code 0..8
            R_idx + 10,  # offset code
            C_idx + 20,  # offset code
        ],
        axis=1,
    ).astype(np.int64, copy=False)

    test_x_rc = np.tile(rc_base, (1, 3)).astype(np.int64, copy=False)  # (n_breaths, 15)

    u_in = (
        df["u_in"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, steps_per_breath)
    )
    u_out = (
        df["u_out"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, steps_per_breath)
    )
    t = (
        df["time_step"]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, steps_per_breath)
    )

    u_in_cum = np.cumsum(u_in, axis=1, dtype=np.float32)
    u_in_lag1 = np.empty_like(u_in, dtype=np.float32)
    u_in_lag1[:, 0] = 0.0
    u_in_lag1[:, 1:] = u_in[:, :-1]
    u_in_diff1 = u_in - u_in_lag1

    test_x_other = np.stack(
        [t, u_in, u_out, u_in_cum, u_in_lag1, u_in_diff1], axis=-1
    ).astype(np.float32, copy=False)
    ids = df["id"].to_numpy(dtype=np.int64, copy=False)

    if is_train:
        y = (
            df["pressure"]
            .to_numpy(dtype=np.float32, copy=False)
            .reshape(n_breaths, steps_per_breath)
        )
        return test_x_rc, test_x_other, y, u_out, ids, df
    return test_x_rc, test_x_other, ids, df


test_path = os.path.join(DATA_DIR, "test.csv")
test_dtypes = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
}
test_df = pd.read_csv(test_path, dtype=test_dtypes)
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
train_dtypes = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
train_df = pd.read_csv(train_path, dtype=train_dtypes)

train_x_rc, train_x_other, train_y, train_u_out, train_ids, train_df_sorted = (
    build_features(train_df, is_train=True)
)
print("train shapes:", train_x_rc.shape, train_x_other.shape, train_y.shape)

n_breaths = train_x_rc.shape[0]
val_size = max(1, int(0.02 * n_breaths))
idx = np.arange(n_breaths)
np.random.shuffle(idx)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

import torch.utils.data as tud

Xrc_all_cpu = torch.as_tensor(train_x_rc)  # int64 CPU
Xo_all_cpu = torch.as_tensor(train_x_other)  # float32 CPU
Y_all_cpu = torch.as_tensor(train_y).view(n_breaths, 80, 1)  # float32 CPU (B,80,1)
mask_all_cpu = torch.as_tensor(
    (train_u_out < 0.5).astype(np.float32, copy=False)
)  # (B,80)

full_ds = tud.TensorDataset(Xrc_all_cpu, Xo_all_cpu, Y_all_cpu, mask_all_cpu)

batch_size = 256
pin = torch.cuda.is_available()

num_workers = 0
if pin:
    num_workers = min(4, max(2, cpu_cnt // 2))

loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin,
    drop_last=False,
)
if num_workers > 0:
    loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

train_sampler = tud.SubsetRandomSampler(tr_idx)

val_subset = tud.Subset(
    full_ds, val_idx.tolist() if isinstance(val_idx, np.ndarray) else list(val_idx)
)
train_subset = tud.Subset(
    full_ds, tr_idx.tolist() if isinstance(tr_idx, np.ndarray) else list(tr_idx)
)

train_loader = tud.DataLoader(train_subset, shuffle=True, **loader_kwargs)
val_loader = tud.DataLoader(val_subset, shuffle=False, **loader_kwargs)

model = VentilatorNet(other_dim=train_x_other.shape[-1], steps=80).to(device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile: enabled")
    except Exception as e:
        print("torch.compile: unavailable, continuing without it. Reason:", repr(e))

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
mae = nn.L1Loss(reduction="none")


def run_epoch_from_loader(train: bool, loader):
    model.train(train)
    total_loss = 0.0
    total_count = 0.0

    for rc_b, xo_b, y_b, m_b in loader:
        rc_b = rc_b.to(device, non_blocking=True)
        xo_b = xo_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        m_b = m_b.to(device, non_blocking=True)

        pred = model(rc_b, xo_b)  # (B,80,1)
        loss_raw = mae(pred, y_b).squeeze(-1)  # (B,80)
        loss = (loss_raw * m_b).sum() / (m_b.sum() + 1e-6)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        msum = float(m_b.sum().item())
        total_loss += loss.item() * msum
        total_count += msum

    return total_loss / max(1.0, total_count)


epochs = 4
for ep in range(1, epochs + 1):
    tr_loss = run_epoch_from_loader(train=True, loader=train_loader)
    va_loss = run_epoch_from_loader(train=False, loader=val_loader)
    print(
        f"epoch {ep}/{epochs} - train_mae(insp): {tr_loss:.5f} - val_mae(insp): {va_loss:.5f}"
    )



## === cell 5
model.eval()
with torch.inference_mode():
    Xrc_te = torch.as_tensor(test_x_rc)
    Xo_te = torch.as_tensor(test_x_other)

    import torch.utils.data as tud

    test_ds = tud.TensorDataset(Xrc_te, Xo_te)
    pin = torch.cuda.is_available()

    num_workers = 0
    if pin:
        num_workers = min(4, max(2, cpu_cnt // 2))

    test_loader_kwargs = dict(
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    if num_workers > 0:
        test_loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    test_loader = tud.DataLoader(test_ds, **test_loader_kwargs)

    n_te_breaths = test_x_rc.shape[0]
    pred_breath = np.empty((n_te_breaths, 80), dtype=np.float32)
    off = 0
    for rc_b, xo_b in test_loader:
        bs = rc_b.shape[0]
        rc_b = rc_b.to(device, non_blocking=True)
        xo_b = xo_b.to(device, non_blocking=True)
        p = model(rc_b, xo_b).squeeze(-1).detach().cpu().numpy()  # (B,80)
        pred_breath[off : off + bs] = p
        off += bs

pre_y = pred_breath.reshape(-1)

pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
p_max = 64.82099173863328

sub_medclip = np.clip(pre_y, p_min, p_max)
sub_medclip = np.round((sub_medclip - p_min) / pressure_step) * pressure_step + p_min

test_ids_sorted = test_df_sorted["id"].to_numpy(dtype=np.int64, copy=False)
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
