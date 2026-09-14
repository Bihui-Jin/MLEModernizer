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
sentence-transformers==4.1.0
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
tqdm==4.67.1
transformers==4.53.3
wandb==0.21.0

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

0.1580959161450108

# 6. Current score

8.43315

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'I fix the runtime ImportError by replacing the deprecated `transformers.AdamW` import with `torch.optim.AdamW` (same optimizer semantics) and keep the scheduler import intact. I also fix the `RobustScaler` NameError by ensuring it is imported before `norm_scale` runs, and make device selection safe (use CUDA if available, else CPU) so the script runs in any Kaggle environment. Next, I ensure `test_loader` is always defined by making the data-prep cell complete and not interrupted by earlier errors, and I add a small safety fallback if pretrained models aren’t found so a valid `submission.csv` is still produced. These changes are execution/stability fixes and do not alter the model architecture or inference logic when the pretrained models are present.'
- What this solution (achieved 8.43315) has done: 'I make the notebook run end-to-end even when the pretrained fold models are missing by replacing the hard failure with a safe fallback that still produces a valid `submission.csv`. I also fix the `IndexError` in `test_loop_pred` by handling the edge case where `models` is empty (which is what happens when no weights are found), so `outs[-1]` is never accessed on an empty list. These are execution/stability fixes; they don’t change the model architecture or inference logic when pretrained models are present. With pretrained weights present, predictions are unchanged; without them, the fallback produces a reasonable baseline submission (still likely far from target, but valid).'
- What this solution (achieved 8.43315) has done: 'Your current score (8.43315, lower is better) is far worse than the target (0.1581), and the biggest reason is that you’re not actually using the intended pretrained fold models (so the pipeline falls back to a weak global-mean baseline). I make the smallest change that materially improves score: ensure we correctly locate and load the fold weight files from the competition’s common Kaggle Dataset structure (including searching inside the provided `ventilator-pressure-prediction` folder), and keep everything else (features, model, argmax decoding, prior log-proba fusion) identical. I also make submission alignment robust by forcing predictions to match the test `id` order (no behavioral change when the order is already correct, but prevents silent misalignment). These changes should move the score sharply toward the target band by restoring the original intended inference behavior.'
- What this solution (achieved 8.43315) has done: 'The current gap to the target is very large (8.43 vs 0.158, lower is better), and the most likely cause is that you’re still not actually loading any pretrained fold models—so you fall back to a weak constant baseline. I make minimal changes to (1) robustly find and load the fold weights by checking multiple common Kaggle input locations and filtering to exactly the expected fold files, and (2) enforce a stable fold-ensemble order (f0..f4) so inference matches training. I also make the submission alignment stricter by building the prediction dataframe directly in test row order and then assigning into `df_submission` (avoids any accidental merge mismatch if `id` dtype/order differs). Core model/feature code and decoding stay identical.'

# 9. Code solution

## === cell 0
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"



## === cell 1
import gc
import random
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW  # noqa: F401
from transformers import get_cosine_schedule_with_warmup  # noqa: F401

from sklearn.preprocessing import RobustScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
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




## === cell 3
class VentilatorDataset(Dataset):

    def __init__(self, df, label_dic=None):
        self.dfs = [_df for _, _df in df.groupby("breath_id")]
        self.label_dic = label_dic

    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, item):
        df = self.dfs[item]
        X = df[config.ALL_FEATURES].values
        y = df["pressure"].values
        if self.label_dic is None:
            label = [-1]
        else:
            label = [self.label_dic[i] for i in y]

        d = {
            "X": torch.tensor(X).float(),
            "y": torch.tensor(label).long(),
        }
        return d




## === cell 4
class VentilatorModel(nn.Module):

    def __init__(self):
        super(VentilatorModel, self).__init__()
        self.seq_emb = nn.Sequential(
            nn.Linear(
                len(config.CONT_FEATURES) + len(config.LAG_FEATURES), config.EMBED_SIZE
            ),
            nn.LayerNorm(config.EMBED_SIZE),
        )

        self.lstm = nn.LSTM(
            config.EMBED_SIZE,
            config.HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
            num_layers=4,
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE * 2),
            nn.LayerNorm(config.HIDDEN_SIZE * 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE * 2, 950),
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, X, y=None):
        seq_x = X
        emb_x = self.seq_emb(seq_x)

        out, _ = self.lstm(emb_x, None)
        logits = self.head(out)

        if y is None:
            loss = None
        else:
            loss = self.loss_fn(logits, y)

        return logits, loss

    def loss_fn(self, y_pred, y_true):
        loss = nn.CrossEntropyLoss()(y_pred.reshape(-1, 950), y_true.reshape(-1))
        return loss


model = VentilatorModel()




## === cell 5
def test_loop(model, loader, target_dic_inv):
    predicts = []
    model.eval()
    for d in loader:
        with torch.no_grad():
            out, _ = model(d["X"].to(device))
        out = torch.tensor(
            [[target_dic_inv[j.item()] for j in i] for i in out.argmax(2)]
        )
        predicts.append(out.cpu())

    return torch.vstack(predicts).numpy().reshape(-1)




## === cell 6
def add_feature(df):
    df["time_delta"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["delta"] = df["time_delta"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["delta"].cumsum()

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df = df.drop(["count", "one"], axis=1)
    return df


def add_lag_feature(df):
    for lag in range(1, config.USE_LAG + 1):
        df[f"breath_id_lag{lag}"] = df["breath_id"].shift(lag).fillna(0)
        df[f"breath_id_lag{lag}same"] = np.select(
            [df[f"breath_id_lag{lag}"] == df["breath_id"]], [1], 0
        )

        df[f"u_in_lag_{lag}"] = (
            df["u_in"].shift(lag).fillna(0) * df[f"breath_id_lag{lag}same"]
        )
        df[f"u_in_time{lag}"] = df["u_in"] - df[f"u_in_lag_{lag}"]
        df[f"u_out_lag_{lag}"] = (
            df["u_out"].shift(lag).fillna(0) * df[f"breath_id_lag{lag}same"]
        )

    df["time_step_lag"] = (
        df["time_step"].shift(1).fillna(0) * df[f"breath_id_lag{lag}same"]
    )
    df["breath_time"] = df["time_step"] - df["time_step_lag"]

    drop_columns = ["time_step_lag"]
    drop_columns += [f"breath_id_lag{i}" for i in range(1, config.USE_LAG + 1)]
    drop_columns += [f"breath_id_lag{i}same" for i in range(1, config.USE_LAG + 1)]
    df = df.drop(drop_columns, axis=1)

    df = df.fillna(0)
    return df


c_dic = {10: 0, 20: 1, 50: 2}
r_dic = {5: 0, 20: 1, 50: 2}
rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}


def add_category_features(df):
    df["C_cate"] = df["C"].map(c_dic)
    df["R_cate"] = df["R"].map(r_dic)
    df["RC_sum"] = (df["R"] + df["C"]).map(rc_sum_dic)
    df["RC_dot"] = (df["R"] * df["C"]).map(rc_dot_dic)
    return df


norm_features = config.CONT_FEATURES + config.LAG_FEATURES


def norm_scale(train_df, test_df):
    scaler = RobustScaler()
    all_u_in = np.vstack(
        [train_df[norm_features].values, test_df[norm_features].values]
    )
    scaler.fit(all_u_in)
    train_df[norm_features] = scaler.transform(train_df[norm_features].values)
    test_df[norm_features] = scaler.transform(test_df[norm_features].values)
    return train_df, test_df




## === cell 7
train_df = pd.read_csv(f"{config.INPUT}/train.csv")
test_df = pd.read_csv(f"{config.INPUT}/test.csv")
sub_df = pd.read_csv(f"{config.INPUT}/sample_submission.csv")
oof = np.zeros(len(train_df))
test_preds_lst = []

target_dic = {
    v: i for i, v in enumerate(sorted(train_df["pressure"].unique().tolist()))
}
target_dic_inv = {v: k for k, v in target_dic.items()}

gkf = GroupKFold(n_splits=config.N_FOLD).split(
    train_df, train_df.pressure, groups=train_df.breath_id
)
for fold, (_, valid_idx) in enumerate(gkf):
    train_df.loc[valid_idx, "fold"] = fold

train_df = add_feature(train_df)
test_df = add_feature(test_df)
train_df = add_lag_feature(train_df)
test_df = add_lag_feature(test_df)
train_df = add_category_features(train_df)
test_df = add_category_features(test_df)
train_df, test_df = norm_scale(train_df, test_df)

test_df["pressure"] = -1
test_dset = VentilatorDataset(test_df)

safe_num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
test_loader = DataLoader(
    test_dset,
    batch_size=config.BS,
    pin_memory=torch.cuda.is_available(),
    shuffle=False,
    drop_last=False,
    num_workers=safe_num_workers,
)



## === cell 8
unique_pressures = train_df["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)




## === cell 9
def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )




## === cell 10
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(config.SEED)




## === cell 11
def load_state_dict_safely(model, model_path: str):
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model




## === cell 12
from glob import glob
import re

candidate_globs = [
    "/kaggle/input/**/exp080_conti_rc/ventilator_f*_best_model.bin",
    "/kaggle/input/**/ventilator_f*_best_model.bin",
]
all_found = []
for g in candidate_globs:
    all_found.extend(glob(g, recursive=True))
all_found = sorted(set(all_found))

fold_pat = re.compile(r"ventilator_f(\d+)_best_model\.bin$")
fold_to_path = {}
for p in all_found:
    m = fold_pat.search(p)
    if m:
        f = int(m.group(1))
        if 0 <= f < config.N_FOLD:
            if (f not in fold_to_path) or (f"/{config.EXP_NAME}/" in p):
                fold_to_path[f] = p

model_paths = [fold_to_path[f] for f in range(config.N_FOLD) if f in fold_to_path]

models = []
if len(model_paths) == 0:
    print(
        "WARNING: No pretrained model files found under /kaggle/input. "
        "Will write a baseline submission using a global mean pressure fallback."
    )
else:
    print("Found pretrained model files (fold-ordered):")
    for p in model_paths:
        print(" -", p)
    for model_path in model_paths:
        m = VentilatorModel()
        m = load_state_dict_safely(m, model_path)
        models.append(m)



## === cell 13
classes, class_freq = np.unique(
    train_df["pressure"].map(target_dic), return_counts=True
)



## === cell 14
class_proba = class_freq / np.sum(class_freq)



## === cell 15
class_proba_t = torch.tensor(class_proba, dtype=torch.float32, device=device)




## === cell 16
def test_loop_pred(models, loader, target_dic_inv, class_proba=None):
    if models is None or len(models) == 0:
        raise ValueError("test_loop_pred called with empty models list.")

    predicts = []
    for model in models:
        model.eval()
    if class_proba is not None:
        class_proba = torch.unsqueeze(
            torch.unsqueeze(torch.log(class_proba), dim=0), dim=0
        )  # (1, 1, 950)
    for d in tqdm(loader):
        outs = []
        with torch.no_grad():
            for model in models:
                out, _ = model(d["X"].to(device))
                out = torch.log(torch.nn.functional.softmax(out, dim=2))  # B, S, 950
                outs.append(out)
        if class_proba is not None:
            outs.append(class_proba.expand(outs[-1].shape))
        out = torch.stack(outs, dim=2)  # B, S, n_folds + 1, 950
        out = torch.sum(out, dim=2)
        out = torch.tensor(
            [[target_dic_inv[j.item()] for j in i] for i in out.argmax(2)]
        )
        predicts.append(out.cpu())

    return torch.vstack(predicts).numpy().reshape(-1)




## === cell 17
df_submission = pd.read_csv(f"{config.INPUT}/sample_submission.csv")

if len(models) > 0:
    preds = test_loop_pred(
        models, test_loader, target_dic_inv, class_proba=class_proba_t
    )
    preds = np.array([find_nearest(p) for p in preds], dtype=np.float64)

    if len(preds) != len(test_df):
        raise RuntimeError(f"Pred length {len(preds)} != test rows {len(test_df)}")

    df_submission = df_submission.copy()
    df_submission["pressure"] = preds
else:
    fallback_pressure = float(train_df["pressure"].mean())
    df_submission["pressure"] = find_nearest(fallback_pressure)

df_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_submission.shape)
print(df_submission.head())
print("Saved to:", os.path.abspath("submission.csv"))
