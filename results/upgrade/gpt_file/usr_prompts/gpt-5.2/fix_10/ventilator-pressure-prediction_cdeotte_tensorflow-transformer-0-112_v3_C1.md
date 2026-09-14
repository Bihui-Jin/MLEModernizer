# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.1487378239146059

# 6. Current score

0.54428

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the TensorFlow import crash by switching to the built-in `tf.keras` only path and disabling mixed-precision toggles that are incompatible with the current protobuf/TensorFlow stack in this environment. I fix the Transformer block call signature so Keras can trace the layer without requiring an explicit `training` argument, which is what currently stops model construction and prevents any predictions from being generated. I also fix the u_out weighting/masking logic so it correctly applies weights to inspiratory (u_out==0) timesteps after scaling, which should substantially reduce MAE toward your target without changing the model architecture. Finally, I make submission creation robust so it always writes a valid `.csv` even when running only one fold, using mean/median logic safely.'
- What this solution (achieved 17.65244) has done: 'The timeout is dominated by feature engineering on the full 5.4M-row training set: repeated `groupby().shift()` calls, `groupby().rolling().agg()` with dict-aggregation, and `pd.get_dummies()` on a huge frame. I keep the exact same features and model logic, but compute groupby shifts/diffs in a single grouped pass, replace the rolling dict-agg with four equivalent rolling reductions, and avoid expensive one-hot expansion by using fixed integer encodings for `R`, `C`, and `R__C` (same information content, much faster and deterministic). I also reduce DataLoader overhead during inference (pin memory + larger batch + workers) and remove redundant disk reads for OOF export by reusing already-loaded raw columns; these changes do not affect predictions. Training is already disabled; if it triggers due to missing weights, runtime still be large, but the main win is making the default inference pipeline fit under 600s.'
- What this solution (achieved 17.65244) has done: 'The timeout is dominated by slow pandas feature engineering on 5.4M rows and extra inference overhead from repeated DataLoader construction and unnecessary CPU/GPU syncs. I keep the exact same features/model/loss, but make feature engineering faster by using `groupby(...).shift()` in one pass, using `groupby.cumcount()` instead of creating temporary columns, and avoiding repeated `Series.astype()`/`fillna()` work. I also reduce inference overhead by reusing a single `valid_loader_x` (instead of rebuilding it) and by using a faster multi-worker setup tuned to CPU count; predictions remain identical. Finally, I prevent accidental training (which would never finish) by raising immediately if pretrained weights are missing when `TRAIN_MODEL=False` (core logic unchanged; this just avoids a guaranteed timeout path).'
- What this solution (achieved 0.54428) has done: 'I fix the immediate runtime error by ensuring KFold is never constructed with `n_splits=1` (scikit-learn disallows it) and instead run a single deterministic holdout split when `FIRST_FOLD_ONLY=True`. I also make the fold loop always define `test_preds/oof_preds/...` so downstream cells don’t crash, and consolidate submission writing so a valid `.csv` is produced even if weights are missing (falling back to training only when required by your current logic). Finally, I keep your existing model/feature/loss logic intact and only touch the split + control-flow bugs and robust submission creation.'

# 9. Code solution

## === cell 0
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER = 81
FIRST_FOLD_ONLY = True
TRAIN_MODEL = False  # will be auto-overridden to True if pretrained weights are missing



## === cell 1
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    else "cpu"
)
print("Device:", DEVICE)



## === cell 2
print("Using float32 training for stability.")



## === cell 3
DTYPES_TRAIN = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
DTYPES_TEST = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

TRAIN_COLS = list(DTYPES_TRAIN.keys())
TEST_COLS = list(DTYPES_TEST.keys())

train_raw = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype=DTYPES_TRAIN,
    usecols=TRAIN_COLS,
)
test_raw = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    dtype=DTYPES_TEST,
    usecols=TEST_COLS,
)
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
print(train_raw.shape, test_raw.shape, submission.shape)




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    breath_id = df["breath_id"]
    g = df.groupby("breath_id", sort=False)

    u_in = df["u_in"]
    u_out = df["u_out"]
    time_step = df["time_step"]

    df["cross"] = (u_in * u_out).astype("float32")
    df["cross2"] = (time_step * u_out).astype("float32")

    df["area"] = (
        (time_step * u_in).groupby(breath_id, sort=False).cumsum().astype("float32")
    )
    df["time_step_cumsum"] = g["time_step"].cumsum().astype("float32")
    df["u_in_cumsum"] = g["u_in"].cumsum().astype("float32")

    u_in_g = g["u_in"]
    u_out_g = g["u_out"]
    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = u_in_g.shift(k)
        df[f"u_out_lag{k}"] = u_out_g.shift(k)
        df[f"u_in_lag_back{k}"] = u_in_g.shift(-k)
        df[f"u_out_lag_back{k}"] = u_out_g.shift(-k)

    df.fillna(0, inplace=True)

    u_in_max = u_in_g.transform("max").astype("float32")
    u_in_mean = u_in_g.transform("mean").astype("float32")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = (u_in_max - u_in).astype("float32")
    df["breath_id__u_in__diffmean"] = (u_in_mean - u_in).astype("float32")

    for k in (1, 2, 3, 4):
        df[f"u_in_diff{k}"] = (u_in - df[f"u_in_lag{k}"]).astype("float32")
        df[f"u_out_diff{k}"] = (u_out - df[f"u_out_lag{k}"]).astype("float32")

    df["count"] = (g.cumcount() + 1).astype("int16")
    df["u_in_cummean"] = (df["u_in_cumsum"] / df["count"]).astype("float32")

    df["breath_id_lag"] = breath_id.shift(1).fillna(0).astype("int32")
    df["breath_id_lag2"] = breath_id.shift(2).fillna(0).astype("int32")
    df["breath_id_lagsame"] = (df["breath_id_lag"] == breath_id).astype("int8")
    df["breath_id_lag2same"] = (df["breath_id_lag2"] == breath_id).astype("int8")

    df["breath_id__u_in_lag"] = (
        u_in.shift(1).fillna(0) * df["breath_id_lagsame"]
    ).astype("float32")
    df["breath_id__u_in_lag2"] = (
        u_in.shift(2).fillna(0) * df["breath_id_lag2same"]
    ).astype("float32")

    df["time_step_diff"] = g["time_step"].diff().fillna(0).astype("float32")

    ewm_mean = u_in_g.ewm(halflife=9).mean()
    df["ewm_u_in_mean"] = ewm_mean.to_numpy(dtype=np.float32, copy=False)

    ru = u_in_g.rolling(window=15, min_periods=1)
    df["15_in_sum"] = ru.sum().to_numpy(dtype=np.float32, copy=False)
    df["15_in_min"] = ru.min().to_numpy(dtype=np.float32, copy=False)
    df["15_in_max"] = ru.max().to_numpy(dtype=np.float32, copy=False)
    df["15_in_mean"] = ru.mean().to_numpy(dtype=np.float32, copy=False)

    df["u_in_lagback_diff1"] = (u_in - df["u_in_lag_back1"]).astype("float32")
    df["u_out_lagback_diff1"] = (u_out - df["u_out_lag_back1"]).astype("float32")
    df["u_in_lagback_diff2"] = (u_in - df["u_in_lag_back2"]).astype("float32")
    df["u_out_lagback_diff2"] = (u_out - df["u_out_lag_back2"]).astype("float32")

    R_map = {5: 0, 20: 1, 50: 2}
    C_map = {10: 0, 20: 1, 50: 2}
    df["R_enc"] = df["R"].map(R_map).astype("int8")
    df["C_enc"] = df["C"].map(C_map).astype("int8")
    df["R__C_enc"] = (
        df["R_enc"].astype("int16") * 3 + df["C_enc"].astype("int16")
    ).astype("int8")

    df["one"] = 1
    return df


train = add_features(train_raw)
test = add_features(test_raw)



## === cell 5
print("Train shape is now:", train.shape)
train.head()



## === cell 6
train_cols = train.columns
missing_in_test = train_cols.difference(test.columns)
for c in missing_in_test:
    test[c] = 0
extra_in_test = test.columns.difference(train_cols)
if len(extra_in_test) > 0:
    test = test.drop(columns=list(extra_in_test))
test = test[train_cols]
print(
    "Aligned columns. Missing added:",
    len(missing_in_test),
    "Extra dropped:",
    len(extra_in_test),
)



## === cell 7
train["pressure_diff"] = (
    train.groupby("breath_id", sort=False).pressure.diff().fillna(0).astype("float32")
)
train["pressure_integral"] = (
    train.groupby("breath_id", sort=False).pressure.cumsum() / 200
).astype("float32")
targets = (
    train[["pressure", "pressure_diff", "pressure_integral"]]
    .to_numpy()
    .reshape(-1, 80, 3)
)

train.drop(
    [
        "pressure",
        "pressure_diff",
        "pressure_integral",
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)

test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)



## === cell 8
print("Targets shape is", targets.shape)



## === cell 9
assert targets.shape[1] == 80, "Expected 80 timesteps per breath."



## === cell 10
COL_ORDER = (
    list(train.columns[:3]) + list(train.columns[-15:]) + list(train.columns[3:-15])
)
train = train[COL_ORDER]
test = test[COL_ORDER]
print("Num features:", train.shape[1])



## === cell 11
U_OUT_COLNAME = "u_out"
assert (
    U_OUT_COLNAME in train.columns
), "u_out column not found after feature engineering."
u_out_train_raw = train[U_OUT_COLNAME].to_numpy().reshape(-1, 80).astype(np.int8)
u_out_test_raw = test[U_OUT_COLNAME].to_numpy().reshape(-1, 80).astype(np.int8)




## === cell 12
def robust_scale_fit_transform(X_train_2d: np.ndarray, X_test_2d: np.ndarray):
    X_train_2d = np.asarray(X_train_2d, dtype=np.float32, order="C")
    X_test_2d = np.asarray(X_test_2d, dtype=np.float32, order="C")

    q25 = np.quantile(X_train_2d, 0.25, axis=0, method="linear").astype(
        np.float32, copy=False
    )
    q50 = np.quantile(X_train_2d, 0.50, axis=0, method="linear").astype(
        np.float32, copy=False
    )
    q75 = np.quantile(X_train_2d, 0.75, axis=0, method="linear").astype(
        np.float32, copy=False
    )
    iqr = (q75 - q25).astype(np.float32, copy=False)

    iqr_safe = iqr.copy()
    iqr_safe[iqr_safe == 0] = 1.0

    X_train_scaled = (X_train_2d - q50) / iqr_safe
    X_test_scaled = (X_test_2d - q50) / iqr_safe
    return X_train_scaled.astype(np.float32, copy=False), X_test_scaled.astype(
        np.float32, copy=False
    )


train_2d = train.to_numpy(dtype=np.float32, copy=False)
test_2d = test.to_numpy(dtype=np.float32, copy=False)
train_scaled, test_scaled = robust_scale_fit_transform(train_2d, test_2d)

train = train_scaled.reshape(-1, 80, train_scaled.shape[-1]).astype(
    "float32", copy=False
)
test = test_scaled.reshape(-1, 80, train_scaled.shape[-1]).astype("float32", copy=False)



## === cell 13
print("Train reshaped:", train.shape, "Test reshaped:", test.shape)



## === cell 14
y_weight = np.ones((targets.shape[0], targets.shape[1], 1), dtype=np.float32)
y_weight[u_out_train_raw == 1] = 0.0



## === cell 15
train.shape, targets.shape, y_weight.shape



## === cell 16
base_feat_dim = train.shape[-1] + 32
embed_dim = 64
num_heads = 8
ff_dim = 128
dropout_rate = 0.0
num_blocks = 12

feat_dim = int(math.ceil(base_feat_dim / num_heads) * num_heads)
print(
    "base_feat_dim:",
    base_feat_dim,
    "-> adjusted feat_dim:",
    feat_dim,
    "num_heads:",
    num_heads,
)


class TransformerBlockPT(nn.Module):
    def __init__(self, feat_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(feat_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, feat_dim),
        )
        self.ln2 = nn.LayerNorm(feat_dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.drop(ffn_out))
        return x


class VentTransformerPT(nn.Module):
    def __init__(self, in_dim, feat_dim, num_heads, ff_dim, num_blocks, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, feat_dim)
        self.ln0 = nn.LayerNorm(feat_dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockPT(feat_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.ln0(self.proj(x))
        for blk in self.blocks:
            x_old = x
            x = blk(x)
            x = 0.7 * x + 0.3 * x_old  # same skip mixing
        return self.head(x)




## === cell 17
LR_START = 1e-6
LR_MAX = 6e-4
LR_MIN = 1e-6
EPOCHS = 420
STEPS = [60, 120, 240]


def lrfn(epoch):
    if epoch < STEPS[0]:
        epoch2 = epoch
        EPOCHS2 = STEPS[0]
    elif epoch < STEPS[0] + STEPS[1]:
        epoch2 = epoch - STEPS[0]
        EPOCHS2 = STEPS[1]
    else:
        epoch2 = epoch - STEPS[0] - STEPS[1]
        EPOCHS2 = STEPS[2]

    decay_total_epochs = EPOCHS2 - 1
    decay_epoch_index = epoch2
    phase = math.pi * decay_epoch_index / max(1, decay_total_epochs)
    cosine_decay = 0.5 * (1 + math.cos(phase))
    lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr


print(
    "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(
        lrfn(0), max(lrfn(x) for x in range(EPOCHS)), lrfn(EPOCHS - 1)
    )
)




## === cell 18
class VentDataset(Dataset):
    def __init__(self, X, y=None, w=None):
        X = np.asarray(X, dtype=np.float32)
        self.X = torch.from_numpy(X)
        self.y = (
            None if y is None else torch.from_numpy(np.asarray(y, dtype=np.float32))
        )
        self.w = (
            None if w is None else torch.from_numpy(np.asarray(w, dtype=np.float32))
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx], self.w[idx]


def masked_mae_loss(pred, target, weight):
    abs_err = torch.abs(pred - target)
    w = weight
    abs_err = abs_err * w
    denom = torch.clamp(w.sum() * pred.shape[-1], min=1.0)
    return abs_err.sum() / denom




## === cell 19
EPOCH = EPOCHS
BATCH_SIZE = 512
NUM_FOLDS = 11
VERBOSE = 1

USE_SINGLE_HOLDOUT = bool(FIRST_FOLD_ONLY)

if not TRAIN_MODEL:
    wpath_probe = f"../input/vent-transformer/folds0_{VER}.pt"
    wpath_probe2 = f"../input/vent-tranformer/folds0_{VER}.pt"
    if (not os.path.exists(wpath_probe)) and (not os.path.exists(wpath_probe2)):
        print("Pretrained weights not found; enabling TRAIN_MODEL=True fallback.")
        TRAIN_MODEL = True

if TRAIN_MODEL:
    EPOCH = 35  # keep existing fallback behavior

DL_NUM_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))
DL_PIN_MEMORY = DEVICE.type == "cuda"
DL_PREFETCH = 4

test_ds = VentDataset(test)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DL_NUM_WORKERS,
    pin_memory=DL_PIN_MEMORY,
    prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
    persistent_workers=(DL_NUM_WORKERS > 0),
    drop_last=False,
)

test_preds = []
oof_preds = []
oof_true = []
all_mask = []
test_folds = []

n_breaths = train.shape[0]
if USE_SINGLE_HOLDOUT:
    rng = np.random.RandomState(SEED)
    idx = np.arange(n_breaths)
    rng.shuffle(idx)
    n_valid = max(1, int(round(0.1 * n_breaths)))
    valid_idx = np.sort(idx[:n_valid])
    train_idx = np.sort(idx[n_valid:])
    splits = [(train_idx, valid_idx)]
    print(f"Using single holdout split: train={len(train_idx)} valid={len(valid_idx)}")
else:
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = list(kf.split(train, targets))

for fold, (train_idx, valid_idx) in enumerate(splits):
    print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = targets[train_idx], targets[valid_idx]
    w_train, w_valid = y_weight[train_idx], y_weight[valid_idx]
    test_folds.append(valid_idx)

    model = VentTransformerPT(
        in_dim=train.shape[-1],
        feat_dim=feat_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_blocks=num_blocks,
        dropout=dropout_rate,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val = float("inf")
    best_state = None

    if TRAIN_MODEL:
        train_loader = DataLoader(
            VentDataset(X_train, y_train, w_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
            prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
            persistent_workers=(DL_NUM_WORKERS > 0),
            drop_last=False,
        )
        valid_loader = DataLoader(
            VentDataset(X_valid, y_valid, w_valid),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DL_NUM_WORKERS,
            pin_memory=DL_PIN_MEMORY,
            prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
            persistent_workers=(DL_NUM_WORKERS > 0),
            drop_last=False,
        )

        for epoch in range(EPOCH):
            lr = lrfn(epoch)
            for pg in opt.param_groups:
                pg["lr"] = lr

            model.train()
            tr_loss = 0.0
            ntr = 0
            for xb, yb, wb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                wb = wb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = masked_mae_loss(pred, yb, wb)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * xb.size(0)
                ntr += xb.size(0)

            model.eval()
            va_loss = 0.0
            nva = 0
            with torch.no_grad():
                for xb, yb, wb in valid_loader:
                    xb = xb.to(DEVICE, non_blocking=True)
                    yb = yb.to(DEVICE, non_blocking=True)
                    wb = wb.to(DEVICE, non_blocking=True)
                    pred = model(xb)
                    loss = masked_mae_loss(pred, yb, wb)
                    va_loss += loss.item() * xb.size(0)
                    nva += xb.size(0)

            tr_loss /= max(1, ntr)
            va_loss /= max(1, nva)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCH} lr={lr:.2e} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}"
                )

            if va_loss < best_val:
                best_val = va_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), f"folds{fold}_{VER}.pt")
    else:
        wpath1 = f"../input/vent-transformer/folds{fold}_{VER}.pt"
        wpath2 = f"../input/vent-tranformer/folds{fold}_{VER}.pt"
        wpath = wpath1 if os.path.exists(wpath1) else wpath2
        model.load_state_dict(torch.load(wpath, map_location="cpu"))
        model.to(DEVICE)

    compiled_model = model
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            compiled_model = model

    print("Predicting Test...")
    compiled_model.eval()
    preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            pb = compiled_model(xb)[:, :, 0].detach().cpu().numpy()
            preds.append(pb)
    pred_test = np.concatenate(preds, axis=0)  # (Nbreaths, 80)
    test_preds.append(pred_test.reshape(-1))

    print("Predicting OOF...")
    valid_loader_x = DataLoader(
        VentDataset(X_valid),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=DL_PIN_MEMORY,
        prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
        persistent_workers=(DL_NUM_WORKERS > 0),
        drop_last=False,
    )
    preds = []
    with torch.no_grad():
        for xb in valid_loader_x:
            xb = xb.to(DEVICE, non_blocking=True)
            pb = compiled_model(xb)[:, :, 0].detach().cpu().numpy()
            preds.append(pb)
    pred_oof = np.concatenate(preds, axis=0)
    oof_preds.append(pred_oof.reshape(-1, 1))
    oof_true.append(y_valid[:, :, 0].reshape(-1, 1))

    score_all = mean_absolute_error(oof_true[-1], oof_preds[-1])
    print(f"Fold-{fold+1} | OOF all timesteps MAE: {score_all}")

    valid_u_out_raw = u_out_train_raw[valid_idx].reshape(-1)
    mask = np.where(valid_u_out_raw == 0)[0]
    mask_score = mean_absolute_error(oof_true[-1][mask], oof_preds[-1][mask])
    print(f"Fold-{fold+1} | OOF u_out=0 MAE: {mask_score}")
    all_mask.append(mask)

    np.save(
        f"oof_v{VER}_trans.npy", np.array(oof_preds, dtype=object), allow_pickle=True
    )

    if FIRST_FOLD_ONLY:
        break



## === cell 20
if FIRST_FOLD_ONLY:
    NUM_FOLDS_RUN = 1
else:
    NUM_FOLDS_RUN = NUM_FOLDS
print(
    "Folds run:",
    NUM_FOLDS_RUN,
    "Test preds:",
    len(test_preds),
    "OOF preds:",
    len(oof_preds),
)



## === cell 21
t = 0.0
K = len(oof_preds)
for k in range(K):
    mask = all_mask[k]
    mae = np.mean(np.abs(oof_preds[k].flatten()[mask] - oof_true[k].flatten()[mask]))
    t += mae
    print("Fold", k, "has u_out=0 MAE =", mae)
print("Overall CV u_out=0 MAE =", t / max(1, K))



## === cell 22
t = 0.0
K = len(oof_preds)
for k in range(K):
    oof = oof_preds[k].copy()
    oof2 = (
        np.round((oof + 1.895744294564641) / 0.07030214545121005) * 0.07030214545121005
        - 1.895744294564641
    )
    mask = all_mask[k]
    mae = np.mean(np.abs(oof2.flatten()[mask] - oof_true[k].flatten()[mask]))
    t += mae
    print("Fold", k, "has u_out=0 MAE with PP =", mae)
print("Overall CV u_out=0 MAE with PP =", t / max(1, K))



## === cell 23
if len(test_folds) > 0 and len(oof_preds) > 0:
    folds = test_folds.copy()
    for k in range(len(folds)):
        folds[k] = np.ones_like(folds[k]) * k
    folds = np.hstack(folds)
    folds = np.repeat(folds, 80)

    valid_rows = np.hstack(test_folds)
    valid_rows = 80 * np.repeat(valid_rows, 80)
    shifter = np.tile(np.arange(80), len(valid_rows) // 80)
    valid_rows = valid_rows + shifter

    train_oof = train_raw.loc[valid_rows, ["id"]].copy()

    oof_stack = np.vstack(oof_preds).reshape(-1)
    train_oof["oof"] = oof_stack.astype("float32")
    train_oof["fold"] = folds.astype("int16")

    train_oof["id"] = train_oof["id"].astype("int32")
    train_oof["oof"] = train_oof["oof"].astype("float32")
    train_oof["fold"] = train_oof["fold"].astype("int8")
    train_oof[["id", "oof", "fold"]].to_csv(f"oof_v{VER}.csv", index=False)
    print("Wrote:", f"oof_v{VER}.csv", "shape:", train_oof.shape)
else:
    print("Skipping OOF export: no OOF predictions were generated.")



## === cell 24
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

if len(test_preds) == 0:
    print("WARNING: No test predictions; writing zeros submission.")
    submission["pressure"] = 0.0
else:
    pred_mean = np.mean(np.vstack(test_preds), axis=0).astype("float32")
    submission["pressure"] = pred_mean

submission.to_csv(f"submission_mean_{VER}.csv", index=False)
print("Wrote:", f"submission_mean_{VER}.csv", "shape:", submission.shape)



## === cell 25
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
if len(test_preds) == 0:
    submission["pressure"] = 0.0
else:
    pred_median = np.median(np.vstack(test_preds), axis=0).astype("float32")
    submission["pressure"] = pred_median
submission.to_csv(f"submission_median_{VER}.csv", index=False)
print("Wrote:", f"submission_median_{VER}.csv", "shape:", submission.shape)



## === cell 26
submission = pd.read_csv(f"submission_median_{VER}.csv")
submission["pressure"] = (
    np.round((submission["pressure"] + 1.895744294564641) / 0.07030214545121005)
    * 0.07030214545121005
    - 1.895744294564641
).astype("float32")
submission.to_csv(f"submission_median_snap_{VER}.csv", index=False)
print("Wrote:", f"submission_median_snap_{VER}.csv")



## === cell 27
submission.head()
