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

# 5. Code solution

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

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import plotly.offline as pyo
import plotly.graph_objs as go

pyo.init_notebook_mode()
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn import model_selection as sk_model_selection
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score
from sklearn import metrics
import optuna
from sklearn.metrics import log_loss
from optuna.samplers import TPESampler
import functools
from functools import partial
import xgboost as xgb
import joblib
import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import warnings

warnings.filterwarnings("ignore")
import torch.optim as optim
import time
from sklearn.preprocessing import MinMaxScaler

SEED = 42

import random

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 1
train_data = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
sample_submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
test_data = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")



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
temp



## === cell 8
print(train_data["id"].nunique(), train_data.shape)
train_data[train_data["id"] == 1]



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
temp




## === cell 14
def interactive_line_chart(Breath_ID):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=train_data[train_data["breath_id"] == Breath_ID]["pressure"],
            x=train_data[train_data["breath_id"] == Breath_ID]["time_step"],
            mode="lines",
            name="pressure",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=train_data[train_data["breath_id"] == Breath_ID]["u_in"],
            x=train_data[train_data["breath_id"] == Breath_ID]["time_step"],
            mode="lines",
            name="u_in",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=train_data[train_data["breath_id"] == Breath_ID]["u_out"],
            x=train_data[train_data["breath_id"] == Breath_ID]["time_step"],
            mode="lines",
            name="u_out",
        )
    )

    fig.update_layout(
        title="Variation by time step", xaxis_title="Time step", yaxis_title="Value"
    )
    fig.show()


w = widgets.interactive(
    interactive_line_chart, Breath_ID=train_data["breath_id"].unique().tolist()
)
display(w)



## === cell 15
fig = px.histogram(
    train_data.groupby(["breath_id"]).agg({"pressure": "mean"}).reset_index(),
    x="pressure",
    nbins=20,
)
fig.show()



## === cell 16
breath_id_list = train_data["breath_id"].unique().tolist()
df_train, df_valid = sk_model_selection.train_test_split(
    breath_id_list, test_size=0.2, random_state=SEED
)

df_train = train_data[train_data["breath_id"].isin(df_train)].reset_index(drop=True)
df_valid = train_data[train_data["breath_id"].isin(df_valid)].reset_index(drop=True)



## === cell 17
scaler = MinMaxScaler()
scaler.fit(df_train[["R", "C", "time_step", "u_in", "u_out", "pressure"]])




## === cell 18
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
            temp = (
                formatted_data[formatted_data["breath_id"] == breath_id][
                    ["R", "C", "time_step", "u_in", "u_out", "pressure"]
                ]
                .iloc[i : i + 1, :]
                .reset_index(drop=True)
            )
            temp = temp.sort_values(by=["time_step"], ascending=True)
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




## === cell 19
class DataRetriever_LSTM(torch_data.Dataset):
    def __init__(self, breath_id_list, train_flag):
        self.breath_id_list = breath_id_list
        self.train_flag = train_flag

    def __len__(self):
        return len(self.breath_id_list)

    def __getitem__(self, index):
        breath_id = self.breath_id_list[index]

        if self.train_flag:
            formatted_data = (
                train_data[train_data["breath_id"] == breath_id]
                .sort_values(by=["time_step"], ascending=True)[
                    ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
                ]
                .reset_index(drop=True)
            )
        else:
            formatted_data = (
                test_data[test_data["breath_id"] == breath_id]
                .sort_values(by=["time_step"], ascending=True)[
                    ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
                ]
                .reset_index(drop=True)
            )
            formatted_data["pressure"] = 0

        formatted_data = pd.DataFrame(
            scaler.transform(
                formatted_data[["R", "C", "time_step", "u_in", "u_out", "pressure"]]
            )
        ).reset_index(drop=True)
        formatted_data.columns = ["R", "C", "time_step", "u_in", "u_out", "pressure"]

        X = torch.tensor(
            np.stack(
                [
                    formatted_data["time_step"],
                    formatted_data["R"],
                    formatted_data["C"],
                    formatted_data["u_in"],
                    formatted_data["u_out"],
                ],
                axis=1,
            )
        ).float()

        if self.train_flag:
            return {"X": X, "y": torch.tensor(formatted_data["pressure"]).float()}
        else:
            return {"X": X, "id": breath_id}




## === cell 20
formatted_train_data = pd.DataFrame(data=None)
breath_id_list = train_data["breath_id"].unique().tolist()[:10]

for breath_id in tqdm(breath_id_list):
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



## === cell 21
formatted_train_data = formatted_train_data[
    [x for x in formatted_train_data.columns if "pressure" not in x]
    + [x for x in formatted_train_data.columns if "pressure" in x]
]
formatted_train_data.head()



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
pass




## === cell 31
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




## === cell 32
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
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
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
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

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




## === cell 33
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

train_loader = torch_data.DataLoader(
    train_data_retriever,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

valid_data_retriever = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)

valid_loader = torch_data.DataLoader(
    valid_data_retriever,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

trainer = Trainer(model, device, optimizer, criterion)

history = trainer.fit(
    n_epochs,
    train_loader,
    valid_loader,
    f"lstm_model.pth",
    patient_epochs,
)



## === cell 34
temp = pd.DataFrame(
    data={"Train loss": history["train_loss"], "Validation loss": history["val_loss"]},
    columns=["Train loss", "Validation loss"],
)
temp["epoch"] = temp.index + 1

fig = go.Figure()
fig.add_trace(
    go.Scatter(y=temp["Train loss"], x=temp["epoch"], mode="lines", name="Train loss")
)
fig.add_trace(
    go.Scatter(
        y=temp["Validation loss"], x=temp["epoch"], mode="lines", name="Validation loss"
    )
)
fig.update_layout(title="Model loss", xaxis_title="Epoch", yaxis_title="Loss")
fig.show()



## === cell 35
temp = pd.DataFrame(
    data={
        "Train MAE": [x.cpu().numpy().item() for x in history["train_mae"]],
        "Validation MAE": [x.cpu().numpy().item() for x in history["val_mae"]],
    },
    columns=["Train MAE", "Validation MAE"],
)
temp["epoch"] = temp.index + 1

fig = go.Figure()
fig.add_trace(
    go.Scatter(y=temp["Train MAE"], x=temp["epoch"], mode="lines", name="Train MAE")
)
fig.add_trace(
    go.Scatter(
        y=temp["Validation MAE"], x=temp["epoch"], mode="lines", name="Validation MAE"
    )
)
fig.update_layout(title="Model accuracy", xaxis_title="Epoch", yaxis_title="MAE")
fig.show()



## === cell 36
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



## === cell 37
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
)

for e, batch in enumerate(valid_loader):
    print(f"{e}/{len(valid_loader)}", end="\r")
    with torch.no_grad():
        tmp_res = (model(batch["X"].to(device))).cpu().numpy()
        temp = pd.DataFrame(np.vstack(batch["X"].cpu().numpy()))
        temp["predicted pressure"] = np.concatenate(tmp_res)
        temp["actual pressure"] = np.concatenate(batch["y"].numpy().tolist())
        temp.columns = [
            "time_step",
            "R",
            "C",
            "u_in",
            "u_out",
            "predicted pressure",
            "actual pressure",
        ]
        y_pred.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "predicted pressure"]]
                )
            )
            .iloc[:, 5]
            .tolist()
        )
        y_true.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "actual pressure"]]
                )
            )
            .iloc[:, 5]
            .tolist()
        )




## === cell 38
def interactive_line_chart_for_validation(Breath_ID):
    data_retriever = DataRetriever_LSTM([Breath_ID], train_flag=True)

    with torch.no_grad():
        batch = np.expand_dims(data_retriever[0]["X"], axis=0)
        tmp_res = (model(torch.tensor(batch).float().to(device))).cpu().numpy()
        temp = pd.DataFrame(np.vstack(batch))
        temp["predicted pressure"] = np.concatenate(tmp_res)
        temp["actual pressure"] = data_retriever[0]["y"].numpy().tolist()
        temp.columns = [
            "time_step",
            "R",
            "C",
            "u_in",
            "u_out",
            "predicted pressure",
            "actual pressure",
        ]
        y_pred.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "predicted pressure"]]
                )
            )
            .iloc[:, 5]
            .tolist()
        )
        y_true.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "actual pressure"]]
                )
            )
            .iloc[:, 5]
            .tolist()
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=temp["actual pressure"],
            x=temp["time_step"],
            mode="lines",
            name="actual pressure",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=temp["predicted pressure"],
            x=temp["time_step"],
            mode="lines",
            name="predicted pressure",
        )
    )

    fig.update_layout(
        title="Variation by time step", xaxis_title="Time step", yaxis_title="Value"
    )
    fig.show()


w = widgets.interactive(
    interactive_line_chart_for_validation,
    Breath_ID=train_data["breath_id"].unique().tolist(),
)
display(w)



## === cell 39
import gc

y_pred = []
ids = []

test_data_retriever = DataRetriever_LSTM(
    test_data["breath_id"].unique().tolist(), train_flag=False
)

test_loader = torch_data.DataLoader(
    test_data_retriever,
    batch_size=batch_size * 5,
    shuffle=False,
    num_workers=0,
)

for e, batch in enumerate(test_loader):
    print(f"{e}/{len(test_loader)}", end="\r")
    with torch.no_grad():
        tmp_res = (model(batch["X"].to(device))).cpu().numpy()
        temp = pd.DataFrame(np.vstack(batch["X"].cpu().numpy()))
        temp["predicted pressure"] = np.concatenate(tmp_res)
        temp.columns = ["time_step", "R", "C", "u_in", "u_out", "predicted pressure"]
        y_pred.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "predicted pressure"]]
                )
            )
            .iloc[:, 5]
            .tolist()
        )
        ids.append(batch["id"])
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



## === cell 40
final_output = sample_submission[["id"]].sort_values(by=["id"], ascending=True)
final_output["pressure"] = np.concatenate(y_pred)
final_output.to_csv("submission.csv", index=False)



## === cell 41
final_output["breath_id"] = np.concatenate(
    [np.concatenate([[i] * 80 for i in x.numpy()]) for x in ids]
)



## === cell 42
print(final_output.shape)
final_output.head()



## === cell 43
print("Test data pressure values\n")
display(final_output["pressure"].describe())
print("\nTrain data pressure values\n")
display(train_data["pressure"].describe())



## === cell 44
fig = px.histogram(
    final_output.groupby(["breath_id"]).agg({"pressure": "mean"}).reset_index(),
    x="pressure",
    nbins=20,
)
fig.show()
