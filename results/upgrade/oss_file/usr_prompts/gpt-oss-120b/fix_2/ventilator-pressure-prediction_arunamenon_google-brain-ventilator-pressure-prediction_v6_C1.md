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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import plotly.offline as pyo
import plotly.graph_objs as go

pyo.init_notebook_mode()  # Set notebook mode to work in offline
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import sklearn.model_selection as sk_model_selection
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score
from sklearn import metrics
import optuna
from optuna.samplers import TPESampler
import functools
from functools import partial
import xgboost as xgb
import joblib
import torch
from torch import nn
from torch.utils import data as torch_data
import warnings

warnings.filterwarnings("ignore")
import torch.optim as optim
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, normalize
import gc

SEED = 42



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



## === cell 5
breath_id_list = train_data["breath_id"].unique().tolist()
df_train, df_valid = sk_model_selection.train_test_split(
    breath_id_list, test_size=0.2, random_state=SEED
)

df_train = train_data[train_data["breath_id"].isin(df_train)].reset_index(drop=True)
df_valid = train_data[train_data["breath_id"].isin(df_valid)].reset_index(drop=True)



## === cell 6
scaler = MinMaxScaler()
scaler.fit(df_train[["R", "C", "time_step", "u_in", "u_out", "pressure"]])



## === cell 8
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




## === cell 9
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

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out




## === cell 10
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
        for n_epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                X = batch["X"].to(self.device)
                y = batch["y"].to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.mean(train_losses)

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in valid_loader:
                    X = batch["X"].to(self.device)
                    y = batch["y"].to(self.device)
                    preds = self.model(X)
                    loss = self.criterion(preds, y)
                    val_losses.append(loss.item())
            valid_loss = np.mean(val_losses)

            print(
                f"Epoch {n_epoch}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}"
            )

            if valid_loss < self.best_valid_score:
                self.best_valid_score = valid_loss
                self.n_patience = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_valid_score": self.best_valid_score,
                        "epoch": n_epoch,
                    },
                    save_path,
                )
                self.lastmodel = save_path
            else:
                self.n_patience += 1
                if self.n_patience >= patience:
                    print(f"Early stopping at epoch {n_epoch}")
                    break
        checkpoint = torch.load(self.lastmodel, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return self.model




## === cell 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 5
output_dim = 80
hidden_dim = 128
layer_dim = 3
batch_size = 128
dropout = 0.02
n_epochs = 10  # modest number for quick training
patience = 3
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
criterion = nn.L1Loss(reduction="mean")



## === cell 12
train_dataset = DataRetriever_LSTM(
    df_train["breath_id"].unique().tolist(), train_flag=True
)
valid_dataset = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)

train_loader = torch_data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
)
valid_loader = torch_data.DataLoader(
    valid_dataset, batch_size=batch_size * 5, shuffle=False, num_workers=8
)



## === cell 13
trainer = Trainer(model, device, optimizer, criterion)
model = trainer.fit(
    epochs=n_epochs,
    train_loader=train_loader,
    valid_loader=valid_loader,
    save_path="lstm_model.pth",
    patience=patience,
)



## === cell 14
y_pred = []
y_true = []

valid_data_retriever = DataRetriever_LSTM(
    df_valid["breath_id"].unique().tolist(), train_flag=True
)

valid_loader = torch_data.DataLoader(
    valid_data_retriever,
    batch_size=batch_size * 5,
    shuffle=False,
    num_workers=8,
)

for e, batch in enumerate(valid_loader):
    with torch.no_grad():
        tmp_res = (model(batch["X"].to(device))).cpu().numpy()
        temp = pd.DataFrame(np.vstack(batch["X"].cpu().numpy()))
        temp["predicted pressure"] = np.concatenate(tmp_res)
        temp["actual pressure"] = np.concatenate(batch["y"].numpy())
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
            .values
        )
        y_true.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "actual pressure"]]
                )
            )
            .iloc[:, 5]
            .values
        )

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)



## === cell 15
print("MAE for validation data:", np.mean(np.abs(y_pred - y_true)))



## === cell 16
y_pred_test = []
ids = []

test_data_retriever = DataRetriever_LSTM(
    test_data["breath_id"].unique().tolist(), train_flag=False
)

test_loader = torch_data.DataLoader(
    test_data_retriever,
    batch_size=batch_size * 5,
    shuffle=False,
    num_workers=8,
)

for e, batch in enumerate(test_loader):
    with torch.no_grad():
        tmp_res = (model(batch["X"].to(device))).cpu().numpy()
        temp = pd.DataFrame(np.vstack(batch["X"].cpu().numpy()))
        temp["predicted pressure"] = np.concatenate(tmp_res)
        temp.columns = ["time_step", "R", "C", "u_in", "u_out", "predicted pressure"]
        y_pred_test.append(
            pd.DataFrame(
                scaler.inverse_transform(
                    temp[["R", "C", "time_step", "u_in", "u_out", "predicted pressure"]]
                )
            )
            .iloc[:, 5]
            .values
        )
        ids.append(batch["id"])
        gc.collect()
        torch.cuda.empty_cache()

final_output = sample_submission[["id"]].sort_values(by=["id"], ascending=True)
final_output["pressure"] = np.concatenate(y_pred_test)

final_output["pressure"] = (
    np.round((final_output.pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
final_output.pressure = np.clip(final_output.pressure, PRESSURE_MIN, PRESSURE_MAX)

final_output.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' written with shape:", final_output.shape)
