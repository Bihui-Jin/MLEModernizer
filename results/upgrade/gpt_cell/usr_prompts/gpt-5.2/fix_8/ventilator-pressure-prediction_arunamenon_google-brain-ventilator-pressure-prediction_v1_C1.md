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
ipywidgets==8.1.5
joblib==1.5.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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
xgboost==2.0.3

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

3.3629

# 6. Current score

1.63832

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.98253) has done: 'Diagnosis: The crash happens in cell 1 during imports because TensorFlow pulls in `protobuf`, and the installed `protobuf==6.33.0` is incompatible with the TensorFlow/TFDS stack in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known breakage with newer protobuf versions where TensorFlow expects older APIs. Since we cannot change installed packages, the safe runtime fix is to force protobuf to use its pure-Python implementation (which preserves the expected API surface) before importing TensorFlow.

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` via `os.environ` before `import tensorflow as tf`. This is a minimal, localized change that avoids the protobuf C++ API mismatch and allows the rest of the notebook to run unchanged.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: No variables, imports, or interfaces used by cell 2 are changed; `tf` import successfully and the subsequent CSV reads remain identical.

Assumptions: The environment allows setting environment variables at runtime before importing TensorFlow, and the pure-Python protobuf implementation is available (it is bundled with the `protobuf` package).'
- What this solution (achieved 6.00081) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 1, raising `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between TensorFlow (and its protobuf-generated code paths) and `protobuf==6.x`, where TensorFlow expects older protobuf APIs. The current environment has `protobuf==6.33.0`, so importing TensorFlow fails immediately. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation and to ensure the environment variable is set before TensorFlow is imported.

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and also set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` before importing TensorFlow. This avoids the incompatible C++/upb protobuf path that triggers the missing `GetPrototype` attribute. No model/training logic is changed; only import-time environment configuration is adjusted.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: Cell 2 expects `pd` to be imported and functional; this remains unchanged. TensorFlow import is unblocked so any later cells that use `tf` continue to work as intended.

Assumptions: TensorFlow is required later in the notebook, so fixing its import is necessary to proceed. The environment allows setting `os.environ` prior to importing TensorFlow (standard behavior in notebooks/scripts).'
- What this solution (achieved 5.75824) has done: 'Diagnosis: The crash happens in cell 1 during imports because the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` forces the pure-Python protobuf runtime, which is incompatible with the installed protobuf 6.x and triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when TensorFlow (and related deps) load protobuf. This is an environment/config issue, not a model/code logic issue. The minimal safe fix is to stop forcing the pure-Python protobuf implementation (and version), letting protobuf use its default (C++/upb) backend that is compatible with protobuf 6.x and TensorFlow 2.18. No other logic needs to change.

Patch summary: Remove/disable the two `os.environ.setdefault(...)` lines that force the Python protobuf implementation, keeping everything else identical so later cells can run unchanged.

Updated cells: Only cell 1 is modified, with a short inline comment explaining the fix.

Compatibility notes for cell k+1: All imports and variable definitions (`SEED`, modules, etc.) remain the same; cell 2 continues to read the same CSV paths and work once imports succeed.

Assumptions: The runtime supports the default protobuf backend (upb/C++) shipped with the installed `protobuf==6.33.0`, and no other part of the notebook relies on forcing the pure-Python protobuf implementation.'
- What this solution (achieved 6.01484) has done: 'The crash happens before any model/data code runs: importing `tensorflow` triggers an incompatibility between the installed `protobuf==6.33.0` and TensorFlow’s internal protobuf usage, leading to `MessageFactory.GetPrototype` missing. The smallest deterministic fix is to force TensorFlow to use the pure-Python protobuf implementation via environment variables **before** importing TensorFlow. This avoids the C++ protobuf API mismatch without changing any later logic. The patch is localized to cell 1 and preserves all downstream variables and imports.'
- What this solution (achieved 6.16772) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` because the environment has `protobuf==6.33.0`, whose API removed `MessageFactory.GetPrototype`, while TensorFlow 2.18 still expects it in some code paths. The two `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*` environment variables are not sufficient to avoid this incompatibility in this runtime. The minimal deterministic fix is to monkey‑patch `google.protobuf.message_factory.MessageFactory.GetPrototype` to alias `GetMessageClass` (or a safe fallback) before importing TensorFlow.

Patch summary: In cell 1 only, add a small compatibility patch right after setting the protobuf environment variables and before importing TensorFlow. This restores the missing attribute expected by TensorFlow under protobuf 6.x without changing any model/training logic.

Updated cells:'
- What this solution (achieved 1.63832) has done: 'The timeout is dominated by per-`__getitem__` pandas filtering/sorting/scaling inside `DataRetriever_LSTM`, which forces millions of slow DataFrame operations across epochs and inference. I keep the same LSTM, loss, epochs, and semantics, but precompute the per-breath 80×6 scaled arrays once using vectorized numpy, then have the Dataset serve tensors by index with zero pandas work. I also remove/disable notebook-only interactive/plot-heavy cells (they don’t affect training/inference outputs) and replace validation/test inverse-scaling done via DataFrames with equivalent vectorized `scaler.inverse_transform` on numpy arrays. Finally, I speed up DataLoader and GPU transfer with pinned memory and persistent workers (when workers>0) while keeping determinism.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
        if hasattr(_message_factory.MessageFactory, "GetMessageClass"):
            _message_factory.MessageFactory.GetPrototype = (
                _message_factory.MessageFactory.GetMessageClass
            )
        else:

            def _get_prototype_fallback(self, descriptor):
                return self.GetMessages([descriptor])[descriptor.full_name]

            _message_factory.MessageFactory.GetPrototype = _get_prototype_fallback
except Exception:
    pass

import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
from torch import nn
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
import warnings
import torch.optim as optim
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection as sk_model_selection
import random

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
train_data = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
sample_submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
test_data = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")




## === cell 2
print(
    "train:",
    train_data.shape,
    "test:",
    test_data.shape,
    "sample_sub:",
    sample_submission.shape,
)




## === cell 3
pass




## === cell 4
pass




## === cell 5
pass




## === cell 6
pass




## === cell 7
pass




## === cell 8
pass




## === cell 9
pass




## === cell 10
pass




## === cell 11
pass




## === cell 12
pass




## === cell 13
pass




## === cell 14
pass




## === cell 15
pass




## === cell 16
breath_id_list = train_data["breath_id"].unique().tolist()
df_train_ids, df_valid_ids = sk_model_selection.train_test_split(
    breath_id_list, test_size=0.2, random_state=SEED
)

df_train = train_data[train_data["breath_id"].isin(df_train_ids)].reset_index(drop=True)
df_valid = train_data[train_data["breath_id"].isin(df_valid_ids)].reset_index(drop=True)




## === cell 17
scaler = MinMaxScaler()
scaler.fit(df_train[["R", "C", "time_step", "u_in", "u_out", "pressure"]])




## === cell 18
def build_breath_arrays(df: pd.DataFrame, scaler: MinMaxScaler, is_train: bool):
    cols = ["breath_id", "time_step", "R", "C", "u_in", "u_out"]
    if is_train:
        cols = cols + ["pressure"]
    else:
        df = df.copy()
        df["pressure"] = 0.0
        cols = cols + ["pressure"]

    df_sorted = (
        df[cols]
        .sort_values(["breath_id", "time_step"], kind="mergesort")
        .reset_index(drop=True)
    )

    mat = df_sorted[["R", "C", "time_step", "u_in", "u_out", "pressure"]].to_numpy(
        dtype=np.float32, copy=False
    )
    mat_scaled = scaler.transform(mat).astype(np.float32, copy=False)

    breath_ids = df_sorted["breath_id"].to_numpy()
    unique_ids, starts, counts = np.unique(
        breath_ids, return_index=True, return_counts=True
    )

    if not np.all(counts == 80):
        bad = unique_ids[counts != 80][:10]
        raise ValueError(
            f"Expected 80 rows per breath, found mismatches for breath_id(s) like: {bad}"
        )

    breath_data = mat_scaled.reshape(len(unique_ids), 80, 6)
    return unique_ids.astype(np.int64, copy=False), breath_data


train_breath_ids_all, train_breath_data_all = build_breath_arrays(
    train_data, scaler, is_train=True
)
test_breath_ids_all, test_breath_data_all = build_breath_arrays(
    test_data, scaler, is_train=False
)

_train_id2idx = {bid: i for i, bid in enumerate(train_breath_ids_all)}
_test_id2idx = {bid: i for i, bid in enumerate(test_breath_ids_all)}

print(
    "Precomputed arrays:",
    "train breaths:",
    train_breath_data_all.shape,
    "test breaths:",
    test_breath_data_all.shape,
)




## === cell 19
class DataRetriever_LSTM(torch_data.Dataset):
    def __init__(self, breath_id_list, train_flag):
        self.breath_id_list = list(breath_id_list)
        self.train_flag = train_flag

        if self.train_flag:
            self._id2idx = _train_id2idx
            self._data = train_breath_data_all
        else:
            self._id2idx = _test_id2idx
            self._data = test_breath_data_all

        self._indices = np.fromiter(
            (self._id2idx[bid] for bid in self.breath_id_list),
            dtype=np.int64,
            count=len(self.breath_id_list),
        )

    def __len__(self):
        return len(self.breath_id_list)

    def __getitem__(self, index):
        idx = int(self._indices[index])
        arr = self._data[idx]  # (80, 6): [R,C,time_step,u_in,u_out,pressure] scaled
        X = torch.from_numpy(arr[:, [2, 0, 1, 3, 4]]).float()
        if self.train_flag:
            y = torch.from_numpy(arr[:, 5]).float()
            return {"X": X, "y": y}
        else:
            return {"X": X, "id": int(self.breath_id_list[index])}




## === cell 20
pass




## === cell 21
pass




## === cell 22
pass




## === cell 23
pass




## === cell 24
pass




## === cell 25
pass




## === cell 26
pass




## === cell 27
pass




## === cell 28
pass




## === cell 29
pass




## === cell 30
class LSTMModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device
    ):
        super(LSTMModel, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0.to(self.device)

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0.to(self.device)

        out, (hn, cn) = self.lstm(
            x, (h0.detach().to(self.device), c0.detach().to(self.device))
        )

        out = self.fc(out).squeeze(-1)
        return out




## === cell 31
class Trainer:
    def __init__(self, model, device, optimizer, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        train_loss_list = []
        val_loss_list = []
        train_mae_list = []
        val_mae_list = []

        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_mae, train_mse, train_time = self.train_epoch(
                train_loader
            )
            valid_loss, valid_mae, valid_mse, valid_time = self.valid_epoch(
                valid_loader
            )

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, mae: {:.2f}, time: {:.2f} s            ",
                n_epoch,
                train_loss,
                train_mae,
                train_time,
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, mae: {:.2f}, time: {:.2f} s",
                n_epoch,
                valid_loss,
                valid_mae,
                valid_time,
            )

            if self.best_valid_score > valid_loss:
                self.info_message(
                    "Validation loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score,
                    valid_loss,
                    self.lastmodel,
                )
                self.best_valid_score = valid_loss
                self.save_model(n_epoch, save_path, valid_loss)
                self.n_patience = 0
            else:
                self.n_patience += 1

            train_loss_list.append(train_loss)
            val_loss_list.append(valid_loss)
            train_mae_list.append(train_mae)
            val_mae_list.append(valid_mae)

            if self.n_patience >= patience:
                self.info_message(
                    "\nValidation loss didn't improve last {} epochs.", patience
                )
                break

        return {
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_mae": train_mae_list,
            "val_mae": val_mae_list,
            "n_epoch": n_epoch,
        }

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0
        runnning_mae = 0
        runnning_mse = 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device, non_blocking=True)
            targets = batch["y"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            outputs = self.model(X)
            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            self.optimizer.step()

            error = (
                (torch.abs(outputs - targets).sum(axis=1) / outputs.shape[1]).sum()
                / outputs.shape[0]
            ).data
            squared_error = (
                (
                    (((outputs - targets) * (outputs - targets)).sum(axis=1))
                    / outputs.shape[1]
                ).sum()
                / (outputs.shape[0])
            ).data
            runnning_mae += error
            runnning_mse += squared_error

            message = "Train Step {}/{}, train_loss: {:.4f}, train_mae: {:.2f}"
            self.info_message(
                message,
                step,
                len(train_loader),
                sum_loss / step,
                runnning_mae / step,
                end="\r",
            )

        return (
            sum_loss / len(train_loader),
            runnning_mae / len(train_loader),
            runnning_mse / len(train_loader),
            int(time.time() - t),
        )

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        runnning_mae = 0
        runnning_mse = 0

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device, non_blocking=True)
                targets = batch["y"].to(self.device, non_blocking=True)

                outputs = self.model(X)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()

                error = (
                    (torch.abs(outputs - targets).sum(axis=1) / outputs.shape[1]).sum()
                    / outputs.shape[0]
                ).data
                squared_error = (
                    (
                        (((outputs - targets) * (outputs - targets)).sum(axis=1))
                        / outputs.shape[1]
                    ).sum()
                    / (outputs.shape[0])
                ).data
                runnning_mae += error
                runnning_mse += squared_error

            message = "Valid Step {}/{}, valid_loss: {:.4f}, valid_mae: {:.2f}"
            self.info_message(
                message,
                step,
                len(valid_loader),
                sum_loss / step,
                runnning_mae / step,
                end="\r",
            )

        return (
            sum_loss / len(valid_loader),
            runnning_mae / len(valid_loader),
            runnning_mse / len(valid_loader),
            int(time.time() - t),
        )

    def save_model(self, n_epoch, save_path, loss):
        self.lastmodel = f"{save_path}"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)




## === cell 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 5
output_dim = 80
hidden_dim = 64
layer_dim = 3
batch_size = 8
dropout = 0.2
n_epochs = 2
patient_epochs = 10
learning_rate = 1e-3
weight_decay = 1e-6

model_params = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "layer_dim": layer_dim,
    "output_dim": output_dim,
    "dropout_prob": dropout,
    "device": device,
}

model = LSTMModel(**model_params)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss(reduction="mean")

train_data_retriever = DataRetriever_LSTM(
    df_train["breath_id"].unique().tolist(), train_flag=True
)

_num_workers = min(2, os.cpu_count() or 2)
train_loader = torch_data.DataLoader(
    train_data_retriever,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)

valid_data_retriever = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)

valid_loader = torch_data.DataLoader(
    valid_data_retriever,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)

trainer = Trainer(model, device, optimizer, criterion)

history = trainer.fit(
    n_epochs,
    train_loader,
    valid_loader,
    f"lstm_model.pth",
    patient_epochs,
)




## === cell 33
pass




## === cell 34
pass




## === cell 35
model_params = {
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "layer_dim": layer_dim,
    "output_dim": output_dim,
    "dropout_prob": dropout,
    "device": device,
}

model = LSTMModel(**model_params)

checkpoint = torch.load(f"./lstm_model.pth", map_location=device)
print(checkpoint["best_valid_score"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)




## === cell 36
y_pred = []
y_true = []

valid_data_retriever = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)

valid_loader = torch_data.DataLoader(
    valid_data_retriever,
    batch_size=batch_size * 5,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)

for e, batch in enumerate(valid_loader):
    print(f"{e}/{len(valid_loader)}", end="\r")
    with torch.no_grad():
        X = batch["X"].to(device, non_blocking=True)  # (B,80,5) scaled
        pred_scaled = model(X).detach().cpu().numpy()  # (B,80)
        y_scaled = batch["y"].cpu().numpy()  # (B,80)

        X_cpu = batch["X"].cpu().numpy()  # (B,80,5) in order [time_step,R,C,u_in,u_out]
        B, T, _ = X_cpu.shape
        dummy = np.empty((B * T, 6), dtype=np.float32)
        dummy[:, 2] = X_cpu[:, :, 0].reshape(-1)  # time_step
        dummy[:, 0] = X_cpu[:, :, 1].reshape(-1)  # R
        dummy[:, 1] = X_cpu[:, :, 2].reshape(-1)  # C
        dummy[:, 3] = X_cpu[:, :, 3].reshape(-1)  # u_in
        dummy[:, 4] = X_cpu[:, :, 4].reshape(-1)  # u_out

        dummy[:, 5] = pred_scaled.reshape(-1)
        pred_inv = scaler.inverse_transform(dummy)[:, 5].reshape(B, T)

        dummy[:, 5] = y_scaled.reshape(-1)
        true_inv = scaler.inverse_transform(dummy)[:, 5].reshape(B, T)

        y_pred.append(pred_inv.reshape(-1).tolist())
        y_true.append(true_inv.reshape(-1).tolist())




## === cell 37
pass




## === cell 38
import gc

y_pred = []
ids = []

test_data_retriever = DataRetriever_LSTM(
    test_data["breath_id"].unique().tolist(), train_flag=False
)

test_loader = torch_data.DataLoader(
    test_data_retriever,
    batch_size=batch_size
    * 10,  # larger batch reduces overhead; does not change predictions.
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)

for e, batch in enumerate(test_loader):
    print(f"{e}/{len(test_loader)}", end="\r")
    with torch.no_grad():
        X = batch["X"].to(device, non_blocking=True)  # (B,80,5) scaled
        pred_scaled = model(X).detach().cpu().numpy()  # (B,80)

        X_cpu = batch["X"].cpu().numpy()
        B, T, _ = X_cpu.shape
        dummy = np.empty((B * T, 6), dtype=np.float32)
        dummy[:, 2] = X_cpu[:, :, 0].reshape(-1)  # time_step
        dummy[:, 0] = X_cpu[:, :, 1].reshape(-1)  # R
        dummy[:, 1] = X_cpu[:, :, 2].reshape(-1)  # C
        dummy[:, 3] = X_cpu[:, :, 3].reshape(-1)  # u_in
        dummy[:, 4] = X_cpu[:, :, 4].reshape(-1)  # u_out
        dummy[:, 5] = pred_scaled.reshape(-1)

        pred_inv = scaler.inverse_transform(dummy)[:, 5].reshape(B, T)
        y_pred.append(pred_inv.reshape(-1))
        ids.append(batch["id"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




## === cell 39
final_output = sample_submission[["id"]].sort_values(by=["id"], ascending=True)
final_output["pressure"] = np.concatenate(y_pred, axis=0)
final_output.to_csv("submission.csv", index=False)




## === cell 40
final_output["breath_id"] = np.concatenate([np.repeat(x.numpy(), 80) for x in ids])




## === cell 41
print(final_output.shape)
print(final_output.head())




## === cell 42
print("Saved submission.csv with", len(final_output), "rows")




## === cell 43
pass
