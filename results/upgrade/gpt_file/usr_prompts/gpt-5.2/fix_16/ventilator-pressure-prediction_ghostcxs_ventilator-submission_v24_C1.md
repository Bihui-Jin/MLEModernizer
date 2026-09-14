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
import gc
import math
import random
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input"
COMP_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_sub_path = os.path.join(COMP_DIR, "sample_submission.csv")
if not os.path.exists(train_path):
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("train_path:", train_path)
print("test_path:", test_path)
print("sample_sub_path:", sample_sub_path)

WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

SEQ_LEN = 80
RC_DIM = 15  # as required by the architecture

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch:", torch.__version__, "device:", device)



## === cell 2
train = pd.read_csv(
    train_path,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)
sample_sub = pd.read_csv(sample_sub_path)

print(train.shape, test.shape, sample_sub.shape)
print(train.columns)


def _assert_breath_grouping_ok(df: pd.DataFrame, seq_len: int, name: str):
    n = len(df)
    assert n % seq_len == 0, f"{name}: rows must be multiple of {seq_len}"
    b = df["breath_id"].to_numpy(np.int32, copy=False).reshape(-1, seq_len)
    assert np.all(
        b == b[:, :1]
    ), f"{name}: breath_id not constant within sequences; sorting would be required"


_assert_breath_grouping_ok(train, SEQ_LEN, "train")
_assert_breath_grouping_ok(test, SEQ_LEN, "test")



## === cell 3
rc_keys = train["R"].astype(np.int32).to_numpy(copy=False) * 1000 + train["C"].astype(
    np.int32
).to_numpy(copy=False)
uniq_keys = np.unique(rc_keys)
RC_COMBOS = sorted([(int(k // 1000), int(k % 1000)) for k in uniq_keys.tolist()])
rc_to_idx = {r * 1000 + c: i for i, (r, c) in enumerate(RC_COMBOS)}

_sorted_keys = np.array(sorted(rc_to_idx.keys()), dtype=np.int32)
_sorted_vals = np.array([rc_to_idx[int(k)] for k in _sorted_keys], dtype=np.int32)


def build_rc_onehot(df: pd.DataFrame) -> np.ndarray:
    R_first = (
        df["R"]
        .to_numpy(np.int16, copy=False)
        .reshape(-1, SEQ_LEN)[:, 0]
        .astype(np.int32, copy=False)
    )
    C_first = (
        df["C"]
        .to_numpy(np.int16, copy=False)
        .reshape(-1, SEQ_LEN)[:, 0]
        .astype(np.int32, copy=False)
    )
    key = (R_first * 1000 + C_first).astype(np.int32, copy=False)

    pos = np.searchsorted(_sorted_keys, key)
    idx = _sorted_vals[pos]

    onehot = np.zeros((idx.shape[0], RC_DIM), dtype=np.float32)
    onehot[np.arange(idx.shape[0]), idx] = 1.0
    return onehot


def build_other_features_memmap(df: pd.DataFrame, path: str) -> np.memmap:
    n_seq = len(df) // SEQ_LEN
    mm = np.memmap(path, mode="w+", dtype=np.float32, shape=(n_seq, SEQ_LEN, 3))
    mm[..., 0] = (
        df["time_step"].to_numpy(np.float32, copy=False).reshape(n_seq, SEQ_LEN)
    )
    mm[..., 1] = df["u_in"].to_numpy(np.float32, copy=False).reshape(n_seq, SEQ_LEN)
    mm[..., 2] = df["u_out"].to_numpy(np.float32, copy=False).reshape(n_seq, SEQ_LEN)
    return mm


def build_targets_memmap(df: pd.DataFrame, path: str) -> np.memmap:
    n_seq = len(df) // SEQ_LEN
    mm = np.memmap(path, mode="w+", dtype=np.float32, shape=(n_seq, SEQ_LEN, 1))
    mm[..., 0] = df["pressure"].to_numpy(np.float32, copy=False).reshape(n_seq, SEQ_LEN)
    return mm


def build_uout_mask_memmap(df: pd.DataFrame, path: str) -> np.memmap:
    n_seq = len(df) // SEQ_LEN
    mm = np.memmap(path, mode="w+", dtype=np.float32, shape=(n_seq, SEQ_LEN, 1))
    u_out = df["u_out"].to_numpy(np.int8, copy=False).reshape(n_seq, SEQ_LEN)
    mm[..., 0] = (u_out == 0).astype(np.float32, copy=False)
    return mm


train_x_rc = build_rc_onehot(train)
test_x_rc = build_rc_onehot(test)

train_x_other = build_other_features_memmap(
    train, os.path.join(WORK_DIR, "train_x_other.f32.mm")
)
test_x_other = build_other_features_memmap(
    test, os.path.join(WORK_DIR, "test_x_other.f32.mm")
)
train_y = build_targets_memmap(train, os.path.join(WORK_DIR, "train_y.f32.mm"))
train_w = build_uout_mask_memmap(train, os.path.join(WORK_DIR, "train_w.f32.mm"))

print("train_x_rc:", train_x_rc.shape, train_x_rc.dtype)
print("train_x_other:", train_x_other.shape, train_x_other.dtype, type(train_x_other))
print("train_y:", train_y.shape, train_y.dtype, type(train_y))
print("train_w:", train_w.shape, train_w.dtype, type(train_w))
print("test_x_rc:", test_x_rc.shape, test_x_rc.dtype)
print("test_x_other:", test_x_other.shape, test_x_other.dtype, type(test_x_other))

test_ids = test["id"].to_numpy(np.int32, copy=False)
del train, test, sample_sub, rc_keys, uniq_keys
gc.collect()



## === cell 4
means = np.array(
    [
        float(train_x_other[..., 0].mean()),
        float(train_x_other[..., 1].mean()),
        float(train_x_other[..., 2].mean()),
    ],
    dtype=np.float32,
)
stds = (
    np.array(
        [
            float(train_x_other[..., 0].std()),
            float(train_x_other[..., 1].std()),
            float(train_x_other[..., 2].std()),
        ],
        dtype=np.float32,
    )
    + 1e-6
)

scale_idx = np.array([0, 1], dtype=np.int32)


def apply_scaling_inplace(x: np.ndarray):
    x[..., scale_idx] -= means[scale_idx]
    x[..., scale_idx] /= stds[scale_idx]
    return x


train_x_other = apply_scaling_inplace(train_x_other)
test_x_other = apply_scaling_inplace(test_x_other)

try:
    train_x_other.flush()
    test_x_other.flush()
except Exception:
    pass

print("means:", means, "stds:", stds)




## === cell 5
class VentilatorModel(nn.Module):
    def __init__(self, other_dim=3, rc_dim=15):
        super().__init__()
        self.rc_embed = nn.Linear(rc_dim, 10, bias=False)

        in_ch = other_dim + 10

        self.conv1 = nn.Conv1d(
            in_channels=in_ch, out_channels=256, kernel_size=5, padding=2
        )
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(
            in_channels=256, out_channels=192, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm1d(192)

        feat_dim = 192 + in_ch  # concat(conv2_out, x_input)

        self.lstm1 = nn.LSTM(
            input_size=feat_dim, hidden_size=672, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            input_size=672 * 2, hidden_size=512, batch_first=True, bidirectional=True
        )
        self.lstm3 = nn.LSTM(
            input_size=512 * 2, hidden_size=384, batch_first=True, bidirectional=True
        )
        self.gru = nn.GRU(
            input_size=384 * 2, hidden_size=192, batch_first=True, bidirectional=True
        )

        out_dim = 192 * 2 + 192  # concat(ox, conv1d_output)
        self.fc1 = nn.Linear(out_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, rc_input, other_x):
        rc_emb = self.rc_embed(rc_input)  # (B, 10)
        rc_emb = F.selu(rc_emb)
        rc_emb = rc_emb.unsqueeze(1).expand(-1, other_x.size(1), -1)  # (B, 80, 10)

        x_input = torch.cat([other_x, rc_emb], dim=-1)  # (B, 80, 13)

        x_c = x_input.permute(0, 2, 1)  # (B, 13, 80)

        c1 = self.conv1(x_c)
        c1 = F.relu(c1)
        c1 = self.bn1(c1)

        c2 = self.conv2(c1)
        c2 = F.relu(c2)
        c2 = self.bn2(c2)  # (B, 192, 80)

        conv_out = c2.permute(0, 2, 1)  # (B, 80, 192)

        the_feature = torch.cat([conv_out, x_input], dim=-1)  # (B, 80, 205)

        ox, _ = self.lstm1(the_feature)
        ox, _ = self.lstm2(ox)
        ox, _ = self.lstm3(ox)
        ox, _ = self.gru(ox)  # (B, 80, 384)

        out = torch.cat([ox, conv_out], dim=-1)  # (B, 80, 576)
        out = F.selu(self.fc1(out))
        out = self.fc2(out)  # (B, 80, 1)
        return out


model = VentilatorModel(other_dim=3, rc_dim=RC_DIM).to(device)
print(model)



## === cell 6
n_breaths = train_x_other.shape[0]
idx = np.arange(n_breaths, dtype=np.int32)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_size = int(0.05 * n_breaths)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

BATCH_SIZE = 256
EPOCHS = 3
LR = 1e-3


class VentDataset(Dataset):
    def __init__(self, indices, x_rc, x_other_mm, y_mm=None, w_mm=None):
        self.indices = indices.astype(np.int32, copy=False)
        self.x_rc = x_rc
        self.x_other_mm = x_other_mm
        self.y_mm = y_mm
        self.w_mm = w_mm

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        j = int(self.indices[i])
        rc = self.x_rc[j]  # (15,) float32
        oth = self.x_other_mm[j]  # (80,3) float32 view on memmap
        if self.y_mm is None:
            return rc, oth
        y = self.y_mm[j]  # (80,1) float32 view on memmap
        w = self.w_mm[j]  # (80,1) float32 view on memmap
        return rc, oth, y, w


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


num_workers = min(4, (os.cpu_count() or 2))
g = torch.Generator()
g.manual_seed(SEED)

ds_tr = VentDataset(tr_idx, train_x_rc, train_x_other, train_y, train_w)
ds_va = VentDataset(val_idx, train_x_rc, train_x_other, train_y, train_w)

dl_tr = DataLoader(
    ds_tr,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)
dl_va = DataLoader(
    ds_va,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def masked_mae(pred, target, weight):
    abs_err = (pred - target).abs() * weight
    denom = weight.sum().clamp_min(1.0)
    return abs_err.sum() / denom


for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    tr_den = 0
    for rc, oth, y, w in dl_tr:
        rc = rc.to(device, non_blocking=True)
        oth = oth.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(rc, oth)
        loss = masked_mae(pred, y, w)
        loss.backward()
        optimizer.step()

        bs = rc.size(0)
        tr_loss += loss.item() * bs
        tr_den += bs

    model.eval()
    va_loss = 0.0
    va_den = 0
    with torch.no_grad():
        for rc, oth, y, w in dl_va:
            rc = rc.to(device, non_blocking=True)
            oth = oth.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            pred = model(rc, oth)
            loss = masked_mae(pred, y, w)
            bs = rc.size(0)
            va_loss += loss.item() * bs
            va_den += bs

    print(
        f"epoch {epoch}/{EPOCHS} - train_loss: {tr_loss/tr_den:.6f} - val_loss: {va_loss/va_den:.6f}"
    )



## === cell 7
PRED_BS = 512
n_test = test_x_other.shape[0]
te_idx = np.arange(n_test, dtype=np.int32)

ds_te = VentDataset(te_idx, test_x_rc, test_x_other, y_mm=None, w_mm=None)
dl_te = DataLoader(
    ds_te,
    batch_size=PRED_BS,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)

model.eval()
preds = np.empty((n_test, SEQ_LEN, 1), dtype=np.float32)

offset = 0
with torch.no_grad():
    for rc, oth in dl_te:
        rc = rc.to(device, non_blocking=True)
        oth = oth.to(device, non_blocking=True)
        out = model(rc, oth).detach().cpu().numpy().astype(np.float32, copy=False)
        bs = out.shape[0]
        preds[offset : offset + bs] = out
        offset += bs

pred = preds.reshape(-1)

pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
pred = np.round((pred - p_min) / pressure_step) * pressure_step + p_min
pred = pred.astype(np.float32)

sub = pd.DataFrame({"id": test_ids, "pressure": pred})
sub = sub.sort_values("id").reset_index(drop=True)

assert sub.shape[0] == test_ids.shape[0]
assert sub["id"].is_unique
assert sub["id"].min() == int(test_ids.min()) and sub["id"].max() == int(test_ids.max())

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))
print(sub.head())

del preds
gc.collect()
