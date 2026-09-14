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

1.6145

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.6145) has done: 'The timeout is dominated by repeated per-breath pandas filtering/sorting inside `__getitem__` (millions of row scans across epochs) and by the per-sample pandas+`inverse_transform` loops during validation/test prediction. I keep the exact LSTM, loss, optimizer, and training loop semantics, but refactor the dataset to use precomputed contiguous numpy tensors per breath (one-time conversion) so each batch fetch is O(1) without pandas. I also vectorize the inverse-scaling step using the scaler’s `min_`/`scale_` constants directly (exactly equivalent to `MinMaxScaler.inverse_transform`) and remove the expensive per-sample DataFrame construction. Finally, I enable efficient DataLoader settings (`pin_memory`, `persistent_workers` where applicable) without changing data or training behavior.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils import data as torch_data
import torch.optim as optim

from sklearn import model_selection as sk_model_selection
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
train_data = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
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
sample_submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    dtype={"id": "int32", "pressure": "float32"},
)
test_data = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
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



## === cell 2
print(train_data.shape)
train_data.head()



## === cell 3
print("# Breath IDs in train data:", train_data["breath_id"].nunique())
train_data[train_data["breath_id"] == 1]



## === cell 4
print(train_data["breath_id"].nunique())
train_data[train_data["breath_id"] == 2]



## === cell 5
print(train_data["breath_id"].nunique())
train_data[train_data["breath_id"] == 3]



## === cell 6
temp = (
    train_data.groupby(["breath_id"])
    .agg({"R": "nunique", "C": "nunique"})
    .reset_index()
)
print(
    "# Breath ids with >1 R or >1 C:", temp[(temp["R"] > 1) | (temp["C"] > 1)].shape[0]
)
temp.head()



## === cell 7
temp = (
    train_data.groupby(["breath_id"])
    .size()
    .reset_index()
    .rename(columns={0: "# Entries"})
)
print(temp["# Entries"].unique())
temp.head()



## === cell 8
print(train_data["id"].nunique(), train_data.shape)
train_data[train_data["id"] == 1].head()



## === cell 9
train_data.describe()



## === cell 10
print(sample_submission.shape)
sample_submission.head()



## === cell 11
print(test_data.shape)
print("# Breath IDs in test data:", test_data["breath_id"].nunique())
test_data.head()



## === cell 12
temp = (
    test_data.groupby(["breath_id"]).agg({"R": "nunique", "C": "nunique"}).reset_index()
)
print(
    "# Breath ids with >1 R or >1 C:", temp[(temp["R"] > 1) | (temp["C"] > 1)].shape[0]
)
temp.head()



## === cell 13
temp = (
    test_data.groupby(["breath_id"])
    .size()
    .reset_index()
    .rename(columns={0: "# Entries"})
)
print(temp["# Entries"].unique())
temp.head()



## === cell 14
breath_id_list = train_data["breath_id"].unique().tolist()
df_train_ids, df_valid_ids = sk_model_selection.train_test_split(
    breath_id_list, test_size=0.2, random_state=SEED
)

df_train = train_data[train_data["breath_id"].isin(df_train_ids)].reset_index(drop=True)
df_valid = train_data[train_data["breath_id"].isin(df_valid_ids)].reset_index(drop=True)



## === cell 15
scaler = MinMaxScaler()
scaler.fit(df_train[["R", "C", "time_step", "u_in", "u_out", "pressure"]])




## === cell 16
def _prepare_breath_arrays(df: pd.DataFrame, has_pressure: bool):
    df2 = df.sort_values(["breath_id", "time_step"], ascending=True, kind="mergesort")
    breath_ids = df2["breath_id"].to_numpy()
    unique_breath_ids, counts = np.unique(breath_ids, return_counts=True)
    if not np.all(counts == 80):
        raise ValueError("Unexpected breath length; expected 80 for all breaths.")
    n_breaths = unique_breath_ids.shape[0]

    cols = ["R", "C", "time_step", "u_in", "u_out"]
    if has_pressure:
        cols = cols + ["pressure"]
    arr = (
        df2[cols]
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_breaths, 80, len(cols))
    )

    idx_map = {int(bid): i for i, bid in enumerate(unique_breath_ids.tolist())}
    return unique_breath_ids.astype(np.int32, copy=False), idx_map, arr


train_breath_ids_all, train_breath_idx_map, train_arr_raw = _prepare_breath_arrays(
    train_data, has_pressure=True
)
test_breath_ids_all, test_breath_idx_map, test_arr_raw = _prepare_breath_arrays(
    test_data.assign(pressure=0.0), has_pressure=True
)

train_arr_scaled = (
    scaler.transform(train_arr_raw.reshape(-1, 6))
    .astype(np.float32, copy=False)
    .reshape(train_arr_raw.shape)
)
test_arr_scaled = (
    scaler.transform(test_arr_raw.reshape(-1, 6))
    .astype(np.float32, copy=False)
    .reshape(test_arr_raw.shape)
)

gc.collect()




## === cell 17
class DataRetriever(torch_data.Dataset):
    def __init__(self, breath_id_list, train_flag):
        self.breath_id_list = breath_id_list
        self.train_flag = train_flag

    def __len__(self):
        return len(self.breath_id_list)

    def __getitem__(self, index):
        breath_id = self.breath_id_list[index]

        formatted_train_data = pd.DataFrame(data=None)

        if self.train_flag:
            formatted_data = (
                train_data[train_data["breath_id"] == breath_id][["breath_id"]]
                .iloc[0:1, :]
                .reset_index(drop=True)
            )
        else:
            formatted_data = (
                test_data[test_data["breath_id"] == breath_id][["breath_id"]]
                .iloc[0:1, :]
                .reset_index(drop=True)
            )
            formatted_data["pressure"] = 0

        for i in range(0, 80):
            base = train_data if self.train_flag else test_data
            cols = ["R", "C", "time_step", "u_in", "u_out"] + (
                ["pressure"] if self.train_flag else []
            )
            temp = (
                base[base["breath_id"] == breath_id][cols]
                .iloc[i : i + 1, :]
                .reset_index(drop=True)
            )
            if not self.train_flag:
                temp["pressure"] = 0
            temp = temp.sort_values(by=["time_step"], ascending=True)
            temp = temp[["R", "C", "time_step", "u_in", "u_out", "pressure"]]
            temp.columns = [
                temp.columns[j] + "_" + str(i + 1) for j in range(0, len(temp.columns))
            ]
            formatted_data = pd.concat(
                [formatted_data.reset_index(drop=True), temp.reset_index(drop=True)],
                axis=1,
            ).reset_index(drop=True)

        formatted_train_data = pd.concat(
            [formatted_train_data, formatted_data], axis=0
        ).reset_index(drop=True)

        X = torch.tensor(
            np.stack(
                [
                    formatted_train_data[
                        [x for x in formatted_train_data.columns if "time_step_" in x]
                    ].iloc[0],
                    formatted_train_data[
                        [x for x in formatted_train_data.columns if "R_" in x]
                    ].iloc[0],
                    formatted_train_data[
                        [x for x in formatted_train_data.columns if "C_" in x]
                    ].iloc[0],
                    formatted_train_data[
                        [x for x in formatted_train_data.columns if "u_in_" in x]
                    ].iloc[0],
                    formatted_train_data[
                        [x for x in formatted_train_data.columns if "u_out_" in x]
                    ].iloc[0],
                ],
                axis=1,
            )
        ).float()

        if self.train_flag:
            return {
                "X": X,
                "y": torch.tensor(
                    formatted_train_data[
                        [x for x in formatted_train_data.columns if "pressure" in x]
                    ].iloc[0]
                ).float(),
            }
        else:
            return {"X": X, "id": breath_id}




## === cell 18
class DataRetriever_LSTM(torch_data.Dataset):
    def __init__(self, breath_id_list, train_flag):
        self.breath_id_list = [int(b) for b in breath_id_list]
        self.train_flag = train_flag

    def __len__(self):
        return len(self.breath_id_list)

    def __getitem__(self, index):
        breath_id = self.breath_id_list[index]

        if self.train_flag:
            i = train_breath_idx_map[breath_id]
            data_scaled = train_arr_scaled[
                i
            ]  # [80, 6] (R,C,time_step,u_in,u_out,pressure)
        else:
            i = test_breath_idx_map[breath_id]
            data_scaled = test_arr_scaled[
                i
            ]  # pressure is 0 but kept for identical scaler usage

        X_np = np.stack(
            [
                data_scaled[:, 2],
                data_scaled[:, 0],
                data_scaled[:, 1],
                data_scaled[:, 3],
                data_scaled[:, 4],
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        X = torch.from_numpy(X_np)

        if self.train_flag:
            y = torch.from_numpy(data_scaled[:, 5].astype(np.float32, copy=False))
            return {"X": X, "y": y}
        else:
            return {"X": X, "id": torch.tensor(breath_id, dtype=torch.int32)}




## === cell 19
formatted_train_data = pd.DataFrame(data=None)
breath_id_list_small = train_data["breath_id"].unique().tolist()[:10]

for breath_id in tqdm(breath_id_list_small):
    formatted_data = (
        train_data[train_data["breath_id"] == breath_id][["breath_id"]]
        .iloc[0:1, :]
        .reset_index(drop=True)
    )
    for i in range(0, 80):
        temp = (
            train_data[train_data["breath_id"] == breath_id][
                ["R", "C", "time_step", "u_in", "u_out", "pressure"]
            ]
            .iloc[i : i + 1, :]
            .reset_index(drop=True)
        )
        temp.columns = [
            temp.columns[j] + "_" + str(i + 1) for j in range(0, len(temp.columns))
        ]
        formatted_data = pd.concat(
            [formatted_data.reset_index(drop=True), temp.reset_index(drop=True)], axis=1
        ).reset_index(drop=True)
    formatted_train_data = pd.concat(
        [formatted_train_data, formatted_data], axis=0
    ).reset_index(drop=True)



## === cell 20
formatted_train_data = formatted_train_data[
    [x for x in formatted_train_data.columns if "pressure" not in x]
    + [x for x in formatted_train_data.columns if "pressure" in x]
]
formatted_train_data.head()




## === cell 21
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
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))  # out: [B, 80, H]
        out = self.fc(out).squeeze(-1)  # [B, 80]
        return out




## === cell 22
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
                "[Epoch Train: {}] loss: {:.4f}, mae: {:.4f}, time: {:.2f} s            ",
                n_epoch,
                train_loss,
                float(train_mae),
                train_time,
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, mae: {:.4f}, time: {:.2f} s",
                n_epoch,
                valid_loss,
                float(valid_mae),
                valid_time,
            )

            if self.best_valid_score > valid_loss:
                self.info_message(
                    "Validation loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score,
                    valid_loss,
                    save_path,
                )
                self.best_valid_score = valid_loss
                self.save_model(n_epoch, save_path, valid_loss)
                self.n_patience = 0
            else:
                self.n_patience += 1

            train_loss_list.append(train_loss)
            val_loss_list.append(valid_loss)
            train_mae_list.append(train_mae.detach().cpu())
            val_mae_list.append(valid_mae.detach().cpu())

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
        sum_loss = 0.0
        runnning_mae = 0.0
        runnning_mse = 0.0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device, non_blocking=True)
            targets = batch["y"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            outputs = self.model(X)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            sum_loss += loss.detach().item()

            error = (torch.abs(outputs - targets).mean()).detach()
            squared_error = (((outputs - targets) ** 2).mean()).detach()
            runnning_mae += error
            runnning_mse += squared_error

            message = "Train Step {}/{}, train_loss: {:.4f}, train_mae: {:.4f}"
            self.info_message(
                message,
                step,
                len(train_loader),
                sum_loss / step,
                float(runnning_mae / step),
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
        sum_loss = 0.0
        runnning_mae = 0.0
        runnning_mse = 0.0

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device, non_blocking=True)
                targets = batch["y"].to(self.device, non_blocking=True)

                outputs = self.model(X)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()

                error = (torch.abs(outputs - targets).mean()).detach()
                squared_error = (((outputs - targets) ** 2).mean()).detach()
                runnning_mae += error
                runnning_mse += squared_error

            message = "Valid Step {}/{}, valid_loss: {:.4f}, valid_mae: {:.4f}"
            self.info_message(
                message,
                step,
                len(valid_loader),
                sum_loss / step,
                float(runnning_mae / step),
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




## === cell 23
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

model = LSTMModel(**model_params).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss(reduction="mean")

train_data_retriever = DataRetriever_LSTM(
    df_train["breath_id"].unique().tolist(), train_flag=True
)

train_loader = torch_data.DataLoader(
    train_data_retriever,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

valid_data_retriever = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)
valid_loader = torch_data.DataLoader(
    valid_data_retriever,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

trainer = Trainer(model, device, optimizer, criterion)

history = trainer.fit(
    n_epochs,
    train_loader,
    valid_loader,
    "lstm_model.pth",
    patient_epochs,
)



## === cell 24
model = LSTMModel(**model_params).to(device)

checkpoint = torch.load("./lstm_model.pth", map_location=device)
print(checkpoint["best_valid_score"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



## === cell 25
scale_ = scaler.scale_.astype(np.float32, copy=False)
min_ = scaler.min_.astype(np.float32, copy=False)

y_pred = []
y_true = []

valid_data_retriever = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)
valid_loader = torch_data.DataLoader(
    valid_data_retriever,
    batch_size=batch_size * 5,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

for e, batch in enumerate(valid_loader):
    print(f"{e}/{len(valid_loader)}", end="\r")
    with torch.no_grad():
        X = batch["X"].to(device, non_blocking=True)  # [B,80,5] scaled
        tmp_res = (
            model(X).detach().cpu().numpy().astype(np.float32, copy=False)
        )  # [B,80] scaled pressure
        tmp_y = (
            batch["y"].cpu().numpy().astype(np.float32, copy=False)
        )  # [B,80] scaled pressure

        X_np = (
            batch["X"].cpu().numpy().astype(np.float32, copy=False)
        )  # [B,80,5] in order [time_step,R,C,u_in,u_out]
        B = X_np.shape[0]

        base6 = np.empty((B, 80, 6), dtype=np.float32)
        base6[:, :, 0] = X_np[:, :, 1]  # R
        base6[:, :, 1] = X_np[:, :, 2]  # C
        base6[:, :, 2] = X_np[:, :, 0]  # time_step
        base6[:, :, 3] = X_np[:, :, 3]  # u_in
        base6[:, :, 4] = X_np[:, :, 4]  # u_out

        base6_pred = base6.copy()
        base6_true = base6.copy()
        base6_pred[:, :, 5] = tmp_res
        base6_true[:, :, 5] = tmp_y

        inv_pred = ((base6_pred.reshape(-1, 6) - min_) / scale_)[:, 5].reshape(B, 80)
        inv_true = ((base6_true.reshape(-1, 6) - min_) / scale_)[:, 5].reshape(B, 80)

        y_pred.extend(inv_pred.tolist())
        y_true.extend(inv_true.tolist())



## === cell 26
y_pred = []
ids = []

test_breath_ids = test_data["breath_id"].unique().tolist()
test_data_retriever = DataRetriever_LSTM(test_breath_ids, train_flag=False)
test_loader = torch_data.DataLoader(
    test_data_retriever,
    batch_size=batch_size * 5,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

for e, batch in enumerate(test_loader):
    print(f"{e}/{len(test_loader)}", end="\r")
    with torch.no_grad():
        X = batch["X"].to(device, non_blocking=True)  # [B,80,5] scaled
        tmp_res = (
            model(X).detach().cpu().numpy().astype(np.float32, copy=False)
        )  # [B,80] scaled pressure

        X_np = batch["X"].cpu().numpy().astype(np.float32, copy=False)
        B = X_np.shape[0]

        base6 = np.empty((B, 80, 6), dtype=np.float32)
        base6[:, :, 0] = X_np[:, :, 1]  # R
        base6[:, :, 1] = X_np[:, :, 2]  # C
        base6[:, :, 2] = X_np[:, :, 0]  # time_step
        base6[:, :, 3] = X_np[:, :, 3]  # u_in
        base6[:, :, 4] = X_np[:, :, 4]  # u_out
        base6[:, :, 5] = tmp_res  # predicted pressure (scaled)

        inv_pred = ((base6.reshape(-1, 6) - min_) / scale_)[:, 5].astype(
            np.float32, copy=False
        )
        y_pred.append(inv_pred)

        ids.append(batch["id"].cpu().numpy())

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



## === cell 27
pred_flat = np.concatenate(y_pred, axis=0)
final_output = sample_submission[["id"]].copy()
final_output["pressure"] = pred_flat[: len(final_output)]
final_output.to_csv("submission.csv", index=False)
print(final_output.shape)
final_output.head()



## === cell 28
final_output["breath_id"] = np.concatenate(
    [np.concatenate([[i] * 80 for i in x]) for x in ids]
)[: len(final_output)]



## === cell 29
print(final_output.shape)
final_output.head()



## === cell 30
print("Test data pressure values\n")
print(final_output["pressure"].describe())
print("\nTrain data pressure values\n")
print(train_data["pressure"].describe())
